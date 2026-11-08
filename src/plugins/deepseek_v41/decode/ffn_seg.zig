const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const kernelSource = @import("../../../mlx/kernels.zig").source;
const dsv41 = @import("../deepseek_v41.zig");
const moe_fp4 = @import("moe_fp4.zig");
const iou = @import("../../../sdk/util/io_util.zig");

pub var build_us: i64 = 0;

pub fn nowUs() i64 {
    return iou.nowUsMonotonic(std.Io.Threaded.global_single_threaded.io());
}

const A = mlx.mlx_array;
const none: A = .{ .ctx = null };

pub const Arm = enum { base, split, nopre, rsu, fin, finsplit, oracle };

pub var arm: Arm = .base;
var arm_read = false;

pub fn current() Arm {
    if (!arm_read) {
        arm_read = true;
        if (std.c.getenv("DSV41_FFN_SEG")) |raw| arm = std.meta.stringToEnum(Arm, std.mem.span(raw)) orelse .split;
    }
    return arm;
}

pub fn set(a: Arm) void {
    arm_read = true;
    arm = a;
}

const UP_SOURCE = kernelSource("plugins/deepseek_v41/kernels/dec_moe_fp4_up_part.metal");
const DOWN_SOURCE = kernelSource("plugins/deepseek_v41/kernels/dec_moe_fp4_down_sx.metal");
const UP_INS = [_][*:0]const u8{ "x", "w", "scales", "sw", "sscales", "inds", "wts", "lim", "dep" };
const DOWN_INS = [_][*:0]const u8{ "x", "w", "scales", "sw", "sscales", "inds", "res", "post", "comb", "sx" };

const ffn_in = @import("ffn_in.zig");
const RSU_SOURCE = kernelSource("plugins/deepseek_v41/kernels/dec_route_su.metal");
const FIN_SOURCE = kernelSource("plugins/deepseek_v41/kernels/dec_ffn_in_tp.metal");
const RSU_INS = [_][*:0]const u8{ "logits", "bias", "scal", "x", "sw", "sscales", "lim" };
const FIN_INS = [_][*:0]const u8{ "y", "x", "spre", "spost", "scomb" };

const Kern = struct { up: mlx.mlx_fast_metal_kernel, down: mlx.mlx_fast_metal_kernel, rsu: mlx.mlx_fast_metal_kernel, fin: mlx.mlx_fast_metal_kernel, di: A, dw: A };
var kern: ?Kern = null;

fn kernels() !Kern {
    if (kern) |k| return k;
    const ku = try dsv41.buildKernel("dsv41_dec_moe_fp4_up_part", &UP_INS, &.{"out"}, UP_SOURCE, moe_fp4.HEADER_BW);
    const kd = try dsv41.buildKernel("dsv41_dec_moe_fp4_down_sx", &DOWN_INS, &.{"out"}, DOWN_SOURCE, moe_fp4.HEADER_BW);
    const di = mlx.mlx_array_new_data(&@as([16]u32, @splat(0)), &[_]c_int{16}, 1, .uint32);
    const dw = mlx.mlx_array_new_data(&@as([16]f32, @splat(0)), &[_]c_int{16}, 1, .float32);
    try mlx.check(mlx.mlx_array_eval(di));
    try mlx.check(mlx.mlx_array_eval(dw));
    const kr = try dsv41.buildKernel("dsv41_dec_route_su", &RSU_INS, &.{ "idx", "wts", "sact" }, RSU_SOURCE, moe_fp4.HEADER_BW);
    const kfin = try dsv41.buildKernel("dsv41_dec_ffn_in_tp", &FIN_INS, &.{ "x1", "hv", "ctr", "parts", "logits" }, FIN_SOURCE, "");
    kern = .{ .up = ku, .down = kd, .rsu = kr, .fin = kfin, .di = di, .dw = dw };
    return kern.?;
}

pub const max_layers = 64;
const Held = struct { a: [6]A = @splat(none), in: [5]A = @splat(none) };
var held: [max_layers]Held = @splat(.{});
pub var cache_on = false;

fn hold(slot: *A, v: A) void {
    if (slot.ctx != null) _ = mlx.mlx_array_free(slot.*);
    slot.* = mlx.mlx_array_new();
    _ = mlx.mlx_array_set(slot, v);
}

pub fn rememberIn(li: usize, x: A, ao: A, sa: *const dsv41.Split) void {
    if (!cache_on) return;
    for (&held[li].in, [_]A{ x, ao, sa.pre, sa.post, sa.comb }) |*slot, v| hold(slot, v);
}

pub fn inOf(li: usize) ?*const [5]A {
    if (held[li].in[0].ctx == null) return null;
    return &held[li].in;
}

pub fn remember(li: usize, p: []const A) void {
    if (!cache_on) return;
    const h = &held[li];
    const pick = [_]usize{ 4, 5, 6, 1, 2, 3 };
    for (&h.a, pick) |*slot, i| {
        if (slot.ctx != null) _ = mlx.mlx_array_free(slot.*);
        slot.* = mlx.mlx_array_new();
        _ = mlx.mlx_array_set(slot, p[i]);
    }
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

fn apply(sc: *dsv41.Scope, k: mlx.mlx_fast_metal_kernel, cfg: mlx.mlx_fast_metal_kernel_config, ins: []const A) !A {
    const iv = mlx.mlx_vector_array_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, k, iv, cfg, sc.s));
    var o = mlx.mlx_array_new();
    return sc.res(mlx.mlx_vector_array_get(&o, ov, 0), &o);
}

fn up(sc: *dsv41.Scope, k: Kern, lim: A, part: usize, ly: *const dsv41.Layer, h: A, idx: A, wts: A, dep: A) !A {
    const gu = if (ly.experts.views) |v| v[0] else ly.experts.gu.?;
    const sgu = ly.shared_gu.?;
    const rows = dims(h, 0);
    const kk = dims(h, 1);
    const topk = dims(idx, 1);
    const n = dims(gu.w, 1) / 2;
    const slots: usize = switch (part) {
        1 => 1,
        2 => topk * rows,
        else => topk * rows + 1,
    };
    const out_slots: usize = switch (part) {
        1 => 1,
        2 => topk,
        else => topk + 1,
    };
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ ci(rows), ci(out_slots), ci(n) }, 3, .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, ci(32 * n * slots), 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 64, 1, 1));
    inline for (.{ .{ "R", rows }, .{ "K", kk }, .{ "N", n }, .{ "TOPK", topk }, .{ "PART", part } }) |t| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, t[0], ci(t[1])));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", .bfloat16));
    const ii = if (part == 1) k.di else idx;
    const ww = if (part == 1) k.dw else wts;
    return apply(sc, k.up, cfg, &.{ h, gu.w, gu.s, sgu.w, sgu.s, ii, ww, lim, dep });
}

fn down(sc: *dsv41.Scope, k: Kern, ly: *const dsv41.Layer, act: A, sx: A, idx: A, one: *const dsv41.Split) !A {
    const dn = if (ly.experts.views) |v| &v[1] else &ly.experts.down;
    const sh = &ly.shared[2];
    const R = dims(act, 0);
    const K = dims(act, 2);
    const N = dims(one.pre, 2);
    const hc: usize = 1;
    const topk = dims(idx, 1);
    const nc = moe_fp4.downCols(K, N, hc) orelse return error.FfnSegShape;
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ ci(R), ci(hc), ci(N) }, 3, .bfloat16));
    const tg = 32 * (topk + 2);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, ci(N / (2 * nc) * R * tg), 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, ci(tg), 1, 1));
    inline for (.{ .{ "K", K }, .{ "N", N }, .{ "TOPK", topk }, .{ "HC", hc }, .{ "NC", nc }, .{ "R", R } }) |t| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, t[0], ci(t[1])));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", .bfloat16));
    return apply(sc, k.down, cfg, &.{ act, dn.w, dn.s, sh.w, sh.s, idx, one.pre, one.post, one.comb, sx });
}

pub fn moeSplit(sc: *dsv41.Scope, m: *const dsv41.Dsv41Model, ly: *const dsv41.Layer, h: A, idx: A, wts: A) !?A {
    const kf = m.dec.moe_fp4 orelse return null;
    if (!kf.bw) return null;
    const k = try kernels();
    const R = dims(h, 0);
    const su = try up(sc, k, kf.lim, 1, ly, h, idx, wts, h);
    const ru = try up(sc, k, kf.lim, 2, ly, h, idx, wts, h);
    const y = try down(sc, k, ly, ru, su, idx, &m.tp_units[R - 1]);
    return try sc.reshape(y, &.{ ci(R), ci(dims(h, 1)) });
}

fn applyN(comptime n: usize, sc: *dsv41.Scope, k: mlx.mlx_fast_metal_kernel, cfg: mlx.mlx_fast_metal_kernel_config, ins: []const A) ![n]A {
    const iv = mlx.mlx_vector_array_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, k, iv, cfg, sc.s));
    var outs: [n]A = undefined;
    for (&outs, 0..) |*o, i| {
        var t = mlx.mlx_array_new();
        o.* = try sc.res(mlx.mlx_vector_array_get(&t, ov, i), &t);
    }
    return outs;
}

fn routeSu(sc: *dsv41.Scope, m: *const dsv41.Dsv41Model, k: Kern, lim: A, ly: *const dsv41.Layer, h: A) ![3]A {
    const logits = try dsv41.gateLogits(sc, m, ly, h);
    const sgu = ly.shared_gu.?;
    const R = dims(h, 0);
    const KD = dims(h, 1);
    const E = dims(ly.gate_w, 0);
    const K = ly.top_k;
    const n = dims(sgu.w, 0) / 2;
    const tg = (E + 31) / 32 * 32;
    const sgs = tg / 32;
    const scal = try sc.keep(mlx.mlx_array_new_data(&[_]f32{m.cfg.router_scaling_factor}, &[_]c_int{1}, 1, .float32));
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ ci(R), ci(K) }, 2, .uint32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ ci(R), ci(K) }, 2, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ ci(R), 1, ci(n) }, 3, .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, ci((R + (n + sgs - 1) / sgs) * tg), 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, ci(tg), 1, 1));
    inline for (.{ .{ "R", R }, .{ "E", E }, .{ "TOPK", K }, .{ "KD", KD }, .{ "N", n } }) |t| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, t[0], ci(t[1])));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_bool(cfg, "NORM", m.cfg.moe_route_norm and K > 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", .bfloat16));
    return applyN(3, sc, k.rsu, cfg, &.{ logits, ly.gate_bias, scal, h, sgu.w, sgu.s, lim });
}

fn finEntry(sc: *dsv41.Scope, m: *const dsv41.Dsv41Model, k: Kern, ly: *const dsv41.Layer, x: A, ao: A, sa: *const dsv41.Split) !?ffn_in.Out {
    const ks = m.dec.ffn_in orelse return null;
    const c = &m.cfg;
    const p: ffn_in.Dims = .{ .rows = dims(x, 0), .d = c.hidden_size, .k = 1024, .e = dims(ly.gate_w, 0), .top_k = ly.top_k, .iters = c.dsv4_hc_sinkhorn_iters, .norm = c.moe_route_norm and ly.top_k > 1 };
    if (!ffn_in.servesB(p) or mlx.mlx_array_dtype(ly.gate_w) != .bfloat16) return null;
    const R = p.rows;
    const D = p.d;
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ ci(R), 4, ci(D) }, 3, .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ ci(R), ci(D) }, 2, .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{16}, 1, .uint32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ci(ffn_in.partsLen(p))}, 1, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ ci(R), ci(p.e) }, 2, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, ci(D), ci(R), 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 256, 1, 1));
    inline for (.{ .{ "D", D }, .{ "HC", 4 } }) |t| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, t[0], ci(t[1])));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", .bfloat16));
    const a = try applyN(5, sc, k.fin, cfg, &.{ ao, x, sa.pre, sa.post, sa.comb });
    const scal = try sc.keep(mlx.mlx_array_new_data(&[_]f32{c.router_scaling_factor}, &[_]c_int{1}, 1, .float32));
    const b = try ffn_in.launchB(sc, ks, p, ffn_in.weightsOf(ly), .{ .x1 = a[0], .hv = a[1], .ctr = a[2], .parts = a[3], .logits = a[4] }, try dsv41.hcEps(sc, m), scal);
    return .{ .x1 = a[0], .h2 = b.h2, .sf = b.sf, .idx = b.idx, .wts = b.wts };
}

pub var last_idx: [2]A = .{ none, none };

pub fn segment(sc: *dsv41.Scope, m: *dsv41.Dsv41Model, li: usize, x: A, ao: A, sa: *const dsv41.Split, pre: *A) !?A {
    if (current() != .base) return tpFfn(sc, m, li, x, ao, sa, pre);
    var p: [7]A = undefined;
    try dsv41.runRegion(sc, m, li, .ffn_pre, &.{ x, ao, sa.pre, sa.post, sa.comb }, &p);
    pre.* = p[1];
    last_idx[0] = p[5];
    const part = (try dsv41.moeFused(sc, m, &m.layers[li], p[4], p[5], p[6])) orelse return null;
    return try dsv41.hcExpand(sc, p[0], try dsv41.tpSum(sc, m, part), &.{ .pre = p[1], .post = p[2], .comb = p[3] });
}

pub fn tpFfn(sc: *dsv41.Scope, m: *dsv41.Dsv41Model, li: usize, x: A, ao: A, sa: *const dsv41.Split, pre: *A) !?A {
    const a = current();
    if (a == .base) {
        rememberIn(li, x, ao, sa);
        return null;
    }
    const ly = &m.layers[li];
    if (a == .nopre) {
        const c = heldOf(li) orelse return null;
        const kf = m.dec.moe_fp4 orelse return null;
        const k = try kernels();
        const R = dims(x, 0);
        const act = try up(sc, k, kf.lim, 0, ly, c[0], c[1], c[2], ao);
        const one = &m.tp_units[R - 1];
        const y = (try moe_fp4.moeDown(sc, kf, ly, act, c[1], one.pre, one)) orelse return null;
        pre.* = c[3];
        return try dsv41.hcExpand(sc, x, try dsv41.tpSum(sc, m, try sc.reshape(y, &.{ ci(R), ci(dims(x, 2)) })), &.{ .pre = c[3], .post = c[4], .comb = c[5] });
    }
    if (a == .oracle) {
        const c = heldOf(li) orelse return null;
        const kf = m.dec.moe_fp4 orelse return null;
        const k = try kernels();
        const R = dims(x, 0);
        var p: [7]A = undefined;
        try dsv41.runRegion(sc, m, li, .ffn_pre, &.{ x, ao, sa.pre, sa.post, sa.comb }, &p);
        pre.* = p[1];
        const act = try up(sc, k, kf.lim, 0, ly, c[0], c[1], c[2], ao);
        const one = &m.tp_units[R - 1];
        const y = (try moe_fp4.moeDown(sc, kf, ly, act, c[1], one.pre, one)) orelse return null;
        return try dsv41.hcExpand(sc, p[0], try dsv41.tpSum(sc, m, try sc.reshape(y, &.{ ci(R), ci(dims(x, 2)) })), &.{ .pre = p[1], .post = p[2], .comb = p[3] });
    }
    if (a == .rsu) {
        const kf = m.dec.moe_fp4 orelse return null;
        const k = try kernels();
        const st = try dsv41.hcExpandPreNorm(sc, m, x, ao, sa, sa.pre, &ly.hc_ffn, ly.ffn_norm);
        const r = try routeSu(sc, m, k, kf.lim, ly, st.h);
        last_idx[1] = r[0];
        const R = dims(x, 0);
        const ru = try up(sc, k, kf.lim, 2, ly, st.h, r[0], r[1], st.h);
        const y = try down(sc, k, ly, ru, r[2], r[0], &m.tp_units[R - 1]);
        pre.* = st.sp.pre;
        return try dsv41.hcExpand(sc, st.x1, try dsv41.tpSum(sc, m, try sc.reshape(y, &.{ ci(R), ci(dims(x, 2)) })), &st.sp);
    }
    if (a == .fin or a == .finsplit) {
        const k = try kernels();
        const f = (try finEntry(sc, m, k, ly, x, ao, sa)) orelse return null;
        last_idx[1] = f.idx;
        const part = (if (a == .fin) try dsv41.moeFused(sc, m, ly, f.h2, f.idx, f.wts) else try moeSplit(sc, m, ly, f.h2, f.idx, f.wts)) orelse return null;
        pre.* = f.sf.pre;
        return try dsv41.hcExpand(sc, f.x1, try dsv41.tpSum(sc, m, part), &f.sf);
    }
    var p: [7]A = undefined;
    try dsv41.runRegion(sc, m, li, .ffn_pre, &.{ x, ao, sa.pre, sa.post, sa.comb }, &p);
    pre.* = p[1];
    last_idx[1] = p[5];
    const part = (try moeSplit(sc, m, ly, p[4], p[5], p[6])) orelse return null;
    return try dsv41.hcExpand(sc, p[0], try dsv41.tpSum(sc, m, part), &.{ .pre = p[1], .post = p[2], .comb = p[3] });
}
