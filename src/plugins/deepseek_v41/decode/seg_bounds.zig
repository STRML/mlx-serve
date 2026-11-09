//! Test-only ceilings for the kernel map's FFN segment: `nopre` runs the MoE on the previous step's held inputs, `oracle`
//! also runs the FFN's pre stages and discards them. Both hold the MoE's launch behind the summed attention output.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const dsv41 = @import("../deepseek_v41.zig");
const moe_fp4 = @import("moe_fp4.zig");
const iou = @import("../../../sdk/util/io_util.zig");

const A = mlx.mlx_array;
const none: A = .{ .ctx = null };

pub var build_us: i64 = 0;

pub fn nowUs() i64 {
    return iou.nowUsMonotonic(std.Io.Threaded.global_single_threaded.io());
}

pub const Arm = enum { base, nopre, oracle };

var arm: Arm = .base;

pub fn current() Arm {
    return arm;
}

pub fn set(a: Arm) void {
    arm = a;
}

const UP_DEP_INS = moe_fp4.UP_INS ++ [_][*:0]const u8{"dep"};

var up_kernel: ?mlx.mlx_fast_metal_kernel = null;

fn upKernel() !mlx.mlx_fast_metal_kernel {
    if (up_kernel) |k| return k;
    up_kernel = try dsv41.buildKernel("dsv41_seg_bounds_up", &UP_DEP_INS, &.{"out"}, moe_fp4.UP_BW_SOURCE, moe_fp4.HEADER_BW);
    return up_kernel.?;
}

const max_layers = 64;
const Held = struct { a: [6]A = @splat(none), in: [5]A = @splat(none), verify: bool = false };
var held: [max_layers]Held = @splat(.{});

pub var cache_on = false;

fn hold(slot: *A, v: A) void {
    if (slot.ctx != null) _ = mlx.mlx_array_free(slot.*);
    slot.* = mlx.mlx_array_new();
    _ = mlx.mlx_array_set(slot, v);
}

pub fn release() void {
    for (&held) |*h| for (&[_][]A{ &h.a, &h.in }) |slots| for (slots) |*slot| {
        if (slot.ctx != null) _ = mlx.mlx_array_free(slot.*);
        slot.* = none;
    };
}

pub fn rememberIn(li: usize, x: A, ao: A, sa: *const dsv41.Split, verify: bool) void {
    if (!cache_on) return;
    for (&held[li].in, [_]A{ x, ao, sa.pre, sa.post, sa.comb }) |*slot, v| hold(slot, v);
    held[li].verify = verify;
}

pub fn verifyOf(li: usize) bool {
    return held[li].verify;
}

pub fn inOf(li: usize) ?*const [5]A {
    if (held[li].in[0].ctx == null) return null;
    return &held[li].in;
}

pub fn remember(li: usize, p: []const A) void {
    if (!cache_on) return;
    const pick = [_]usize{ 4, 5, 6, 1, 2, 3 };
    for (&held[li].a, pick) |*slot, i| hold(slot, p[i]);
}

pub fn heldOf(li: usize) ?*const [6]A {
    if (held[li].a[0].ctx == null) return null;
    return &held[li].a;
}

fn ci(v: usize) c_int {
    return @intCast(v);
}

fn dims(a: A, axis: usize) usize {
    return @intCast(mlx.getShape(a)[axis]);
}

fn up(sc: *dsv41.Scope, lim: A, ly: *const dsv41.Layer, h: A, idx: A, wts: A, dep: A) !A {
    const gu = if (ly.experts.views) |v| v[0] else ly.experts.gu.?;
    const sgu = ly.shared_gu.?;
    const rows = dims(h, 0);
    const n = dims(gu.w, 1) / 2;
    const topk = dims(idx, 1);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ ci(rows), ci(topk + 1), ci(n) }, 3, .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, ci(32 * n * (topk * rows + 1)), 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 64, 1, 1));
    inline for (.{ .{ "R", rows }, .{ "K", dims(h, 1) }, .{ "N", n }, .{ "TOPK", topk } }) |t| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, t[0], ci(t[1])));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", .bfloat16));
    const ins = [_]A{ h, gu.w, gu.s, sgu.w, sgu.s, idx, wts, lim, dep };
    const iv = mlx.mlx_vector_array_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, try upKernel(), iv, cfg, sc.s));
    var o = mlx.mlx_array_new();
    return sc.res(mlx.mlx_vector_array_get(&o, ov, 0), &o);
}

pub fn tpFfn(sc: *dsv41.Scope, m: *dsv41.Dsv41Model, li: usize, x: A, ao: A, sa: *const dsv41.Split, pre: *A) !?A {
    const a = current();
    if (a == .base) {
        rememberIn(li, x, ao, sa, m.verifying);
        return null;
    }
    const c = heldOf(li) orelse return null;
    const kf = m.dec.moe_fp4 orelse return null;
    const ly = &m.layers[li];
    const R = dims(x, 0);
    const act = try up(sc, kf.lim, ly, c[0], c[1], c[2], ao);
    const one = &m.tp_units[R - 1];
    const y = (try moe_fp4.moeDown(sc, kf, ly, act, c[1], one.pre, one)) orelse return null;
    const moe = try dsv41.tpSum(sc, m, try sc.reshape(y, &.{ ci(R), ci(dims(x, 2)) }));
    if (a == .nopre) {
        pre.* = c[3];
        return try dsv41.hcExpand(sc, x, moe, &.{ .pre = c[3], .post = c[4], .comb = c[5] });
    }
    var p: [7]A = undefined;
    try dsv41.runRegion(sc, m, li, .ffn_pre, &.{ x, ao, sa.pre, sa.post, sa.comb }, &p);
    pre.* = p[1];
    return try dsv41.hcExpand(sc, p[0], moe, &.{ .pre = p[1], .post = p[2], .comb = p[3] });
}
