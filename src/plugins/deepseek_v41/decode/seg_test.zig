//! The fused decode segments' switch, coverage rules and exactness at every level (`DSV41_FUSED_SEG`).

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const device = @import("../../../mlx/device.zig");
const env = @import("../../../env.zig");
const kernelSource = @import("../../../mlx/kernels.zig").source;
const dsv41 = @import("../deepseek_v41.zig");
const decode = @import("decode.zig");
const ffn_in = @import("ffn_in.zig");
const fixture = @import("fixture.zig");
const kernel_map = @import("kernel_map.zig");
const seg = @import("seg.zig");
const seg_ffn_in = @import("seg_ffn_in.zig");
const seg_ffn = @import("seg_ffn.zig");
const iou = @import("../../../sdk/util/io_util.zig");
const seg_bounds = @import("seg_bounds.zig");

const A = mlx.mlx_array;
const none: A = .{ .ctx = null };
const Fixture = fixture.Fixture;

fn gpuTest() bool {
    return mlx.metalKernelsAvailable() and !mlx.noGpuBackend() and !mlx.virtualGpu();
}

const SumCall = struct { n: u8, dims: [4]c_int, dtype: mlx.mlx_dtype };

const Sums = struct {
    var calls: std.ArrayList(SumCall) = .empty;

    fn sum(x: A) anyerror!A {
        const shape = mlx.getShape(x);
        var c: SumCall = .{ .n = @intCast(shape.len), .dims = @splat(0), .dtype = mlx.mlx_array_dtype(x) };
        for (shape, 0..) |d, i| c.dims[i] = d;
        try calls.append(std.testing.allocator, c);
        var out = mlx.mlx_array_new();
        _ = mlx.mlx_array_set(&out, x);
        return out;
    }

    fn install() void {
        dsv41.clusterHooks().tp_sum = &sum;
    }

    fn remove() void {
        dsv41.clusterHooks().tp_sum = null;
        calls.clearAndFree(std.testing.allocator);
    }
};

const tp4: fixture.Opts = .{ .lo = 20, .hi = 23, .experts = 8, .ctx = 48, .fused = true, .experts_fp4 = true, .tp = .{ .rank = 0, .size = 4 } };

fn runSteps(gpa: std.mem.Allocator, fx: *Fixture, rows_list: []const usize) ![]u8 {
    var all: std.ArrayList(u8) = .empty;
    errdefer all.deinit(gpa);
    var p0 = fx.state().n;
    for (rows_list, 0..) |rows, k| {
        const x = try fx.stream(rows);
        defer _ = mlx.mlx_array_free(x);
        const pre = try fx.cpre(rows);
        defer _ = mlx.mlx_array_free(pre);
        const out = try dsv41.workerStep(fx.m, gpa, x, pre, p0, false, k + 1 == rows_list.len and rows > 1);
        defer for (out) |o| {
            _ = mlx.mlx_array_free(o);
        };
        for (out) |o| {
            const host = try dsv41.hostF32(o);
            defer gpa.free(host);
            try all.appendSlice(gpa, std.mem.sliceAsBytes(host));
        }
        p0 += rows;
    }
    return all.toOwnedSlice(gpa);
}

test "dsv41 fused seg: level off is stock bit for bit, no seg state, no seg dispatch" {
    if (!gpuTest()) return error.SkipZigTest;
    const gpa = std.testing.allocator;
    Sums.install();
    defer Sums.remove();
    seg.dispatches = 0;
    const steps = [_]usize{ 1, 1, 1, 1, 1, 1, 1, 1, decode.max_rows };
    var ref_opts = tp4;
    ref_opts.seg = .off;
    const ref = try Fixture.init(gpa, ref_opts);
    defer ref.deinit();
    const ref_bytes = try runSteps(gpa, ref, &steps);
    defer gpa.free(ref_bytes);
    const fx = try Fixture.init(gpa, tp4);
    defer fx.deinit();
    const got_bytes = try runSteps(gpa, fx, &steps);
    defer gpa.free(got_bytes);
    try std.testing.expect(fx.m.seg == null);
    try std.testing.expectEqual(seg.Level.off, fx.m.fused_seg);
    try std.testing.expectEqual(@as(u64, 0), seg.dispatches);
    try std.testing.expectEqualSlices(u8, ref_bytes, got_bytes);
}

test "dsv41 fused seg: DSV41_FUSED_SEG reads 0 to 5 and refuses anything else" {
    try std.testing.expectEqual(seg.Level.off, seg.levelOf(null));
    try std.testing.expectEqual(seg.Level.off, seg.levelOf("0"));
    for ([_]seg.Level{ .ffn_entry, .ffn_seg, .attn_seg, .sum_overlap, .lossy }, 1..) |level, n| {
        var buf: [1]u8 = .{'0' + @as(u8, @intCast(n))};
        try std.testing.expectEqual(level, seg.levelOf(&buf));
    }
    for ([_][]const u8{ "6", "-1", "", "on", "1 ", "255", "99999" }) |bad| try std.testing.expectEqual(seg.Level.off, seg.levelOf(bad));
}

test "dsv41 fused seg: serves declines non-NAX, TP != 4, streamed, rows > 1, affine experts, f32" {
    if (!gpuTest()) return error.SkipZigTest;
    const gpa = std.testing.allocator;
    Sums.install();
    defer Sums.remove();
    var small = tp4;
    small.ctx = 0;
    const fx = try Fixture.init(gpa, small);
    defer fx.deinit();
    const nax = device.naxAvailable();
    if (nax) {
        try std.testing.expect(seg.serves(fx.m, 1) == null);
        try std.testing.expectEqualStrings("rows > 1", seg.serves(fx.m, 2).?);
        try std.testing.expectEqualStrings("no rows", seg.serves(fx.m, 0).?);
    }
    var stub = fx.m.*;
    stub.tp = .{ .rank = 0, .size = 2 };
    try std.testing.expectEqualStrings("tp size 2", seg.serves(&stub, 1).?);
    stub.tp = .{ .rank = 0, .size = 8 };
    try std.testing.expectEqualStrings("tp size 8", seg.serves(&stub, 1).?);
    stub.tp = null;
    try std.testing.expectEqualStrings("no tensor split", seg.serves(&stub, 1).?);
    stub = fx.m.*;
    stub.attn_gather = null;
    try std.testing.expectEqualStrings("no NAX", seg.serves(&stub, 1).?);
    stub = fx.m.*;
    stub.streamed_experts = true;
    if (nax) try std.testing.expectEqualStrings("streamed experts", seg.serves(&stub, 1).?);
    stub = fx.m.*;
    stub.act = .float32;
    if (nax) try std.testing.expectEqualStrings("activations not bf16", seg.serves(&stub, 1).?);
    stub = fx.m.*;
    stub.dec.moe_fp4 = null;
    try std.testing.expectEqualStrings("fused decode kernels off", seg.serves(&stub, 1).?);

    var affine = small;
    affine.experts_fp4 = false;
    const fa = try Fixture.init(gpa, affine);
    defer fa.deinit();
    if (nax) try std.testing.expectEqualStrings("affine experts", seg.serves(fa.m, 1).?);
    var plain = small;
    plain.fused = false;
    const fp = try Fixture.init(gpa, plain);
    defer fp.deinit();
    try std.testing.expectEqualStrings("fused decode kernels off", seg.serves(fp.m, 1).?);
}

test "dsv41 fused seg: covers pins the attention and FFN layer sets" {
    const cfg = try fixture.releaseConfig();
    const attn = [_]usize{ 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 21, 22, 23, 25, 26, 27, 29, 30, 31, 33, 34, 35, 37, 38, 39 };
    try std.testing.expectEqual(@as(usize, 30), attn.len);
    for (0..cfg.num_hidden_layers + 3) |li| {
        try std.testing.expectEqual(std.mem.indexOfScalar(usize, &attn, li) != null, seg.coversIn(&cfg, li, .attn));
        try std.testing.expectEqual(li < 40, seg.coversIn(&cfg, li, .ffn));
    }
    var engram = cfg;
    engram.dsv41_n_engram_layers = 1;
    engram.dsv41_engram_layers[0] = 4;
    try std.testing.expect(!seg.coversIn(&engram, 4, .attn));
    try std.testing.expect(seg.coversIn(&engram, 4, .ffn));
    if (!gpuTest()) return;
    Sums.install();
    defer Sums.remove();
    var small = tp4;
    small.ctx = 0;
    const fx = try Fixture.init(std.testing.allocator, small);
    defer fx.deinit();
    for (small.lo..small.hi) |li| for ([_]seg.Which{ .attn, .ffn }) |which| try std.testing.expectEqual(seg.coversIn(&fx.m.cfg, li, which), seg.covers(fx.m, li, which));
    try std.testing.expect(!seg.covers(fx.m, 20, .attn) and seg.covers(fx.m, 21, .attn));
}

test "dsv41 fused seg: a batched step of 2..8 rows runs stock bit for bit at every level" {
    if (!gpuTest()) return error.SkipZigTest;
    const gpa = std.testing.allocator;
    Sums.install();
    defer Sums.remove();
    seg.dispatches = 0;
    const steps = [_]usize{ 2, 3, 4, 5, 6, 7, 8 };
    var ref_bytes: []u8 = &.{};
    defer gpa.free(ref_bytes);
    for (0..6) |n| {
        var o = tp4;
        o.seg = @enumFromInt(n);
        const fx = try Fixture.init(gpa, o);
        defer fx.deinit();
        try std.testing.expectEqual(o.seg, fx.m.fused_seg);
        const bytes = try runSteps(gpa, fx, &steps);
        if (n == 0) {
            ref_bytes = bytes;
            continue;
        }
        defer gpa.free(bytes);
        try std.testing.expectEqualSlices(u8, ref_bytes, bytes);
    }
    try std.testing.expectEqual(@as(u64, 0), seg.dispatches);
}

test "dsv41 fused seg: every level makes the same sums in the same order with the same shapes" {
    if (!gpuTest()) return error.SkipZigTest;
    const gpa = std.testing.allocator;
    Sums.install();
    defer Sums.remove();
    const fx = try Fixture.init(gpa, .{ .lo = 0, .hi = 40, .experts = 8, .ctx = 0, .fused = true, .experts_fp4 = true, .tp = .{ .rank = 0, .size = 4 } });
    defer fx.deinit();
    for ([_]usize{ 1, decode.max_rows }) |rows| {
        var level0: []SumCall = &.{};
        defer gpa.free(level0);
        for (0..6) |n| {
            fx.m.fused_seg = @enumFromInt(n);
            Sums.calls.clearRetainingCapacity();
            const x = try fx.stream(rows);
            defer _ = mlx.mlx_array_free(x);
            const pre = try fx.cpre(rows);
            defer _ = mlx.mlx_array_free(pre);
            const out = try dsv41.workerStep(fx.m, gpa, x, pre, 0, false, false);
            for (out) |o| _ = mlx.mlx_array_free(o);
            if (n == 0) {
                level0 = try gpa.dupe(SumCall, Sums.calls.items);
                try std.testing.expectEqual(@as(usize, 80), level0.len);
                continue;
            }
            try std.testing.expectEqualSlices(SumCall, level0, Sums.calls.items);
        }
    }
}

test "dsv41 fused seg: kmap: segCheck fails the broken arm and passes seg:N" {
    if (!gpuTest()) return error.SkipZigTest;
    const gpa = std.testing.allocator;
    Sums.install();
    defer Sums.remove();
    var o = tp4;
    o.experts = 16;
    const fx = try Fixture.init(gpa, o);
    defer fx.deinit();
    seg_bounds.cache_on = true;
    defer {
        seg_bounds.cache_on = false;
        seg_bounds.release();
        seg_bounds.set(.base);
        seg.broken = .none;
    }
    const nax = device.naxAvailable();
    gpa.free(try runSteps(gpa, fx, &.{ 2, 2 }));
    try kernel_map.segCheck(gpa, fx, o.lo, o.hi, .{ .level = .off });
    for (1..6) |n| try std.testing.expectError(error.SegCheckSkipped, kernel_map.segCheck(gpa, fx, o.lo, o.hi, .{ .level = @enumFromInt(n) }));
    try std.testing.expectError(error.SegCheckFailed, kernel_map.segCheck(gpa, fx, o.lo, o.hi, .broken));
    try std.testing.expectError(error.SegCheckFailed, kernel_map.segCheck(gpa, fx, o.lo, o.hi, .nan));
    for (o.lo..o.hi) |li| try std.testing.expectEqual(@as(c_int, 2), mlx.getShape(seg_bounds.inOf(li).?[0])[0]);
    gpa.free(try runSteps(gpa, fx, &.{ 1, 1 }));
    for (1..6) |n| {
        const level: seg.Level = @enumFromInt(n);
        if (nax) try kernel_map.segCheck(gpa, fx, o.lo, o.hi, .{ .level = level }) else try std.testing.expectError(error.SegCheckSkipped, kernel_map.segCheck(gpa, fx, o.lo, o.hi, .{ .level = level }));
    }
    try std.testing.expectError(error.SegCheckFailed, kernel_map.segCheck(gpa, fx, o.lo, o.hi, .broken));
    try std.testing.expectError(error.SegCheckFailed, kernel_map.segCheck(gpa, fx, o.lo, o.hi, .nan));
    try std.testing.expectEqual(seg.Break.none, seg.broken);
    try std.testing.expectEqual(seg.Level.off, fx.m.fused_seg);
}

test "dsv41 fused seg: kmap: the route check compares each row's picks in order and its weights" {
    const topk = 2;
    const same_idx = [_]f32{ 1, 2, 2, 1 };
    const w = [_]f32{ 0.6, 0.4, 0.7, 0.3 };
    try std.testing.expectEqual(@as(usize, 0), kernel_map.routeRowsBad(&same_idx, &same_idx, &w, &w, 2, topk));
    const swapped_rows = [_]f32{ 2, 1, 1, 2 };
    try std.testing.expectEqual(@as(usize, 2), kernel_map.routeRowsBad(&same_idx, &swapped_rows, &w, &w, 2, topk));
    const tie_flipped = [_]f32{ 2, 1, 2, 1 };
    try std.testing.expectEqual(@as(usize, 1), kernel_map.routeRowsBad(&same_idx, &tie_flipped, &w, &w, 2, topk));
    const moved = [_]f32{ 0.6, 0.4, 0.7, 0.3001 };
    try std.testing.expectEqual(@as(usize, 1), kernel_map.routeRowsBad(&same_idx, &same_idx, &w, &moved, 2, topk));
    const nan_w = [_]f32{ 0.6, 0.4, std.math.nan(f32), 0.3 };
    try std.testing.expectEqual(@as(usize, 1), kernel_map.routeRowsBad(&same_idx, &same_idx, &w, &nan_w, 2, topk));
}

test "dsv41 fused seg: init reads DSV41_FUSED_SEG into the model" {
    if (!gpuTest()) return error.SkipZigTest;
    const gpa = std.testing.allocator;
    Sums.install();
    defer Sums.remove();
    var o = tp4;
    o.ctx = 0;
    o.from_env = true;
    const expected = [_]struct { raw: ?[*:0]const u8, level: seg.Level }{
        .{ .raw = "3", .level = .attn_seg },
        .{ .raw = "0", .level = .off },
        .{ .raw = "9", .level = .off },
        .{ .raw = null, .level = .off },
    };
    for (expected) |e| {
        const saved = env.override(.DSV41_FUSED_SEG, e.raw);
        defer saved.restore();
        try std.testing.expectEqual(e.level, seg.levelFromEnv());
        const fx = try Fixture.init(gpa, o);
        defer fx.deinit();
        try std.testing.expectEqual(e.level, fx.m.fused_seg);
    }
}

fn hostsOf(comptime n: usize, srcs: [n]A) ![n][]f32 {
    var out: [n][]f32 = undefined;
    var made: usize = 0;
    errdefer for (out[0..made]) |h| std.testing.allocator.free(h);
    for (srcs, &out) |a, *h| {
        h.* = try dsv41.hostF32(a);
        made += 1;
    }
    return out;
}

const Inputs = struct { x: A, y: A, cpre: A, sa: dsv41.Split };

fn inputsOf(sc: *dsv41.Scope, fx: *Fixture, ly: *const dsv41.Layer, rows: usize) !Inputs {
    const x = try sc.keep(try fx.stream(rows));
    const cpre = try sc.keep(try fx.cpre(rows));
    const y = try sc.keep(try fx.normal(&.{ @intCast(rows), @intCast(fx.m.cfg.hidden_size) }, .bfloat16, 0, 1));
    const sa = (try dsv41.hcStage(sc, fx.m, x, null, cpre, &ly.hc_attn, ly.attn_norm)).sp;
    return .{ .x = x, .y = y, .cpre = cpre, .sa = sa };
}

fn chainOf(sc: *dsv41.Scope, m: *const dsv41.Dsv41Model, ly: *const dsv41.Layer, in: Inputs) !ffn_in.Out {
    const pf = try dsv41.hcExpandPreNorm(sc, m, in.x, in.y, &in.sa, in.sa.pre, &ly.hc_ffn, ly.ffn_norm);
    const rt = try dsv41.route(sc, m, ly, pf.h, none);
    return .{ .x1 = pf.x1, .h2 = pf.h, .sf = pf.sp, .idx = rt.idx, .wts = rt.wts };
}

fn evalOut(o: ffn_in.Out) !void {
    try dsv41.evalAll(&.{ o.x1, o.h2, o.sf.pre, o.sf.post, o.sf.comb, o.idx, o.wts });
}

test "dsv41 fused seg ffn_in: R = 1 at 384 experts no further from float64 than today's chain" {
    if (!gpuTest()) return error.SkipZigTest;
    const gpa = std.testing.allocator;
    const fx = try Fixture.init(gpa, .{ .ctx = 0, .fused = true });
    defer fx.deinit();
    _ = try seg.ensure(fx.m);
    const w = try ffn_in.randomWeights(77);
    defer ffn_in.freeWeights(w);
    const ly = ffn_in.layerWith(fx.layer(21), w);
    const c = &fx.m.cfg;
    var worst_wts: f64 = 0;
    for ([_]usize{ 1, 6 }) |rows| {
        var sc = dsv41.Scope.init(gpa, mlx.gpuStream());
        defer sc.deinit();
        const in = try inputsOf(&sc, fx, &ly, rows);
        const chain = try chainOf(&sc, fx.m, &ly, in);
        const before = seg.dispatches;
        const out = (try seg_ffn_in.launch(&sc, fx.m, 21, &ly, in.x, in.y, &in.sa)) orelse return error.Declined;
        try std.testing.expectEqual(before + 1, seg.dispatches);
        try evalOut(chain);
        try evalOut(out);
        const p: ffn_in.Dims = .{ .rows = rows, .d = c.hidden_size, .e = 384, .top_k = ly.top_k, .iters = c.dsv4_hc_sinkhorn_iters, .norm = c.moe_route_norm and ly.top_k > 1 };
        const h = try hostsOf(11, .{ in.x, in.sa.pre, in.sa.post, in.sa.comb, w.fn_w, w.fn_scale, w.fn_base, w.norm_w, w.gate_w, w.gate_bias, in.y });
        defer for (h) |a| gpa.free(a);
        var truth = try ffn_in.truthOf(gpa, .{ .p = p, .wo = &.{}, .oa = &.{}, .x = h[0], .pre = h[1], .post = h[2], .comb = h[3], .fnw = h[4], .scale = h[5], .base = h[6], .norm_w = h[7], .gate_w = h[8], .bias = h[9], .eps0 = c.rms_norm_eps, .eps1 = c.dsv4_hc_eps, .y = h[10] });
        defer truth.deinit();
        const cmp = try ffn_in.compareToChain(gpa, fx.m, p, &truth, chain, out);
        try std.testing.expectEqual(@as(usize, 0), cmp.differ);
        worst_wts = @max(worst_wts, cmp.wts_rel);
    }
    std.debug.print("[seg] ffn_in route weights vs the chain: largest relative difference {e:.2}\n", .{worst_wts});
    try std.testing.expect(worst_wts < kernel_map.wts_rel_tol);
}

fn expectSameBytes(gpa: std.mem.Allocator, label: []const u8, a: A, b: A) !void {
    const u = try dsv41.hostF32(a);
    defer gpa.free(u);
    const v = try dsv41.hostF32(b);
    defer gpa.free(v);
    std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(u), std.mem.sliceAsBytes(v)) catch |e| {
        std.debug.print("{s} differs\n", .{label});
        return e;
    };
}

const tp_source = kernelSource("plugins/deepseek_v41/kernels/dec_ffn_in_tp.metal");
const spike_ins = [_][*:0]const u8{ "y", "x", "spre", "spost", "scomb" };
const spike_outs = [_][*:0]const u8{ "x1", "hv", "ctr", "parts", "logits" };

fn spikeChain(sc: *dsv41.Scope, m: *const dsv41.Dsv41Model, ly: *const dsv41.Layer, in: Inputs) !ffn_in.Out {
    const rows: usize = @intCast(mlx.getShape(in.x)[0]);
    const c = &m.cfg;
    const p: ffn_in.Dims = .{ .rows = rows, .d = c.hidden_size, .e = @intCast(mlx.getShape(ly.gate_w)[0]), .top_k = ly.top_k, .iters = c.dsv4_hc_sinkhorn_iters, .norm = c.moe_route_norm and ly.top_k > 1 };
    const kern = try dsv41.buildKernel("dsv41_dec_ffn_in_tp", &spike_ins, &spike_outs, tp_source, "");
    defer _ = mlx.mlx_fast_metal_kernel_free(kern);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    const D: c_int = @intCast(p.d);
    const R: c_int = @intCast(rows);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ R, 4, D }, 3, .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ R, D }, 2, .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{16}, 1, .uint32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{@intCast(ffn_in.partsLen(p))}, 1, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ R, @intCast(p.e) }, 2, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, D, R, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 256, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "D", D));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "HC", 4));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", .bfloat16));
    const ins = [_]A{ in.y, in.x, in.sa.pre, in.sa.post, in.sa.comb };
    const iv = mlx.mlx_vector_array_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, kern, iv, cfg, sc.s));
    var a: [5]A = undefined;
    for (&a, 0..) |*o, i| {
        o.* = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(o, ov, i));
        _ = try sc.keep(o.*);
    }
    const scal = try sc.keep(mlx.mlx_array_new_data(&[_]f32{c.router_scaling_factor}, &[_]c_int{1}, 1, .float32));
    const ks = m.dec.ffn_in orelse return error.Declined;
    const b = try ffn_in.launchB(sc, ks, p, ffn_in.weightsOf(ly), .{ .x1 = a[0], .hv = a[1], .ctr = a[2], .parts = a[3], .logits = a[4] }, try dsv41.hcEps(sc, m), scal);
    return .{ .x1 = a[0], .h2 = b.h2, .sf = b.sf, .idx = b.idx, .wts = b.wts };
}

test "dsv41 fused seg ffn_in: the one launch writes the spike's two launches' bytes" {
    if (!gpuTest()) return error.SkipZigTest;
    const gpa = std.testing.allocator;
    const fx = try Fixture.init(gpa, .{ .ctx = 0, .fused = true });
    defer fx.deinit();
    _ = try seg.ensure(fx.m);
    const w = try ffn_in.randomWeights(78);
    defer ffn_in.freeWeights(w);
    const ly = ffn_in.layerWith(fx.layer(22), w);
    for ([_]usize{ 1, 2, 3, 6 }) |rows| {
        var sc = dsv41.Scope.init(gpa, mlx.gpuStream());
        defer sc.deinit();
        const in = try inputsOf(&sc, fx, &ly, rows);
        const want = try spikeChain(&sc, fx.m, &ly, in);
        const got = (try seg_ffn_in.launch(&sc, fx.m, 22, &ly, in.x, in.y, &in.sa)) orelse return error.Declined;
        try evalOut(want);
        try evalOut(got);
        const names = [_][]const u8{ "x1", "h2", "pre", "post", "comb", "idx", "wts" };
        const a = [_]A{ want.x1, want.h2, want.sf.pre, want.sf.post, want.sf.comb, want.idx, want.wts };
        const b = [_]A{ got.x1, got.h2, got.sf.pre, got.sf.post, got.sf.comb, got.idx, got.wts };
        for (names, a, b) |name, u, v| try expectSameBytes(gpa, name, u, v);
    }
}

fn tieLayer(w: ffn_in.Weights, base: *const dsv41.Layer) !dsv41.Layer {
    const host = try dsv41.hostF32(w.gate_w);
    defer std.testing.allocator.free(host);
    const d = host.len / 384;
    @memcpy(host[300 * d ..][0..d], host[7 * d ..][0..d]);
    const gate = mlx.mlx_array_new_data(host.ptr, &[_]c_int{ 384, @intCast(d) }, 2, .float32);
    var gate_bf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gate);
    try mlx.check(mlx.mlx_astype(&gate_bf, gate, .bfloat16, mlx.gpuStream()));
    try mlx.check(mlx.mlx_array_eval(gate_bf));
    var bias: [384]f32 = @splat(0);
    for (0..5) |e| bias[e] = 100;
    bias[7] = 50;
    bias[300] = 50;
    const bias_a = mlx.mlx_array_new_data(&bias, &[_]c_int{ 1, 384 }, 2, .float32);
    var l = ffn_in.layerWith(base, w);
    l.gate_w = gate_bf;
    l.gate_bias = bias_a;
    return l;
}

test "dsv41 fused seg: four ranks route bit for bit over 1000 launches, an exact tie to the lower index" {
    if (!gpuTest()) return error.SkipZigTest;
    const gpa = std.testing.allocator;
    var ranks: [4]*Fixture = undefined;
    var made: usize = 0;
    defer for (ranks[0..made]) |r| r.deinit();
    for (&ranks, 0..) |*r, i| {
        var o = tp4;
        o.lo = 21;
        o.hi = 22;
        o.ctx = 0;
        o.tp = .{ .rank = @intCast(i), .size = 4 };
        r.* = try Fixture.init(gpa, o);
        made += 1;
        _ = try seg.ensure(r.*.m);
    }
    const w = try ffn_in.randomWeights(79);
    defer ffn_in.freeWeights(w);
    const ly = try tieLayer(w, ranks[0].layer(21));
    defer {
        _ = mlx.mlx_array_free(ly.gate_w);
        _ = mlx.mlx_array_free(ly.gate_bias);
    }
    var sc0 = dsv41.Scope.init(gpa, mlx.gpuStream());
    defer sc0.deinit();
    const in = try inputsOf(&sc0, ranks[0], &ly, 1);
    var first: ?[2][]f32 = null;
    defer if (first) |f| for (f) |a| gpa.free(a);
    const before = seg.dispatches;
    for (0..250) |_| for (ranks) |r| {
        var sc = dsv41.Scope.init(gpa, mlx.gpuStream());
        defer sc.deinit();
        const out = (try seg_ffn_in.launch(&sc, r.m, 21, &ly, in.x, in.y, &in.sa)) orelse return error.Declined;
        try dsv41.evalAll(&.{ out.idx, out.wts });
        const got = [2][]f32{ try dsv41.hostF32(out.idx), try dsv41.hostF32(out.wts) };
        if (first) |f| {
            defer for (got) |a| gpa.free(a);
            for (f, got) |u, v| try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(u), std.mem.sliceAsBytes(v));
            continue;
        }
        first = got;
        for (got[0][0..5]) |e| try std.testing.expect(e < 5);
        try std.testing.expectEqual(@as(f32, 7), got[0][5]);
        for (got[0]) |e| try std.testing.expect(e != 300);
    };
    try std.testing.expectEqual(before + 1000, seg.dispatches);
}

test "dsv41 fused seg: four ranks at level 1 meet at one stream" {
    if (!gpuTest() or !device.naxAvailable()) return error.SkipZigTest;
    const gpa = std.testing.allocator;
    var o = tp4;
    o.ctx = 0;
    o.seg = .ffn_entry;
    var probe = try Fixture.init(gpa, o);
    defer probe.deinit();
    const in = try fixture.Input.of(probe, 1);
    defer in.deinit();
    const before = seg.dispatches;
    const got = try fixture.tpStreamN(gpa, o, in, 4);
    defer gpa.free(got);
    try std.testing.expect(seg.dispatches > before);
}

test "dsv41 fused seg: level 1 decodes layers 20..23 no further from the f32 model than stock" {
    if (!gpuTest() or !device.naxAvailable()) return error.SkipZigTest;
    Sums.install();
    defer Sums.remove();
    var o = tp4;
    o.ctx = 601;
    o.seg = .ffn_entry;
    const before = seg.dispatches;
    try decode.layersAgreeOn(std.testing.allocator, o);
    try std.testing.expect(seg.dispatches > before);
}

const fp4_tp4: fixture.Opts = .{ .lo = 20, .hi = 23, .experts = 16, .ctx = 0, .fused = true, .experts_fp4 = true, .tp = .{ .rank = 0, .size = 4 } };

fn hostPair(a: A, b: A) ![2][]f32 {
    return .{ try dsv41.hostF32(a), try dsv41.hostF32(b) };
}

test "dsv41 fused seg ffn: the segment's MoE rows equal moeFused's on the same h2 and routes, bit for bit" {
    if (!gpuTest() or !device.naxAvailable()) return error.SkipZigTest;
    const gpa = std.testing.allocator;
    var o = fp4_tp4;
    o.seg = .ffn_seg;
    const fx = try Fixture.init(gpa, o);
    defer fx.deinit();
    _ = try seg.ensure(fx.m);
    try std.testing.expectEqual(seg_ffn.ReadMode.staged, seg_ffn.read_mode);
    var printed = false;
    for (o.lo..o.hi) |li| for (0..4) |_| {
        const ly = fx.layer(li);
        var sc = dsv41.Scope.init(gpa, mlx.gpuStream());
        defer sc.deinit();
        const in = try inputsOf(&sc, fx, ly, 1);
        const before = seg.dispatches;
        const l1 = (try seg_ffn_in.launch(&sc, fx.m, li, ly, in.x, in.y, &in.sa)) orelse return error.Declined;
        const l2 = (try seg_ffn.launch(&sc, fx.m, ly, li, in.x, in.y, &in.sa)) orelse return error.Declined;
        try std.testing.expectEqual(before + 2, seg.dispatches);
        const h2_l2 = try l2.h2(&sc);
        const idx_l2 = try l2.idx(&sc);
        const wts_l2 = try l2.wts(&sc);
        const want = (try dsv41.moeFused(&sc, fx.m, ly, h2_l2, idx_l2, wts_l2)) orelse return error.Declined;
        const chain = try chainOf(&sc, fx.m, ly, in);
        try dsv41.evalAll(&.{ l2.x1, h2_l2, l2.sf.pre, l2.sf.post, l2.sf.comb, idx_l2, wts_l2, l2.part, want, l1.x1, l1.h2, l1.sf.pre, l1.sf.post, l1.sf.comb, l1.idx, l1.wts });
        try evalOut(chain);
        try std.testing.expectEqualSlices(c_int, &.{ 1, @intCast(fx.m.cfg.hidden_size) }, mlx.getShape(l2.part));
        try std.testing.expectEqual(mlx.mlx_array_dtype(want), mlx.mlx_array_dtype(l2.part));
        try expectSameBytes(gpa, "part", want, l2.part);
        const names = [_][]const u8{ "x1", "h2", "pre", "post", "comb", "idx", "wts" };
        const a = [_]A{ l1.x1, l1.h2, l1.sf.pre, l1.sf.post, l1.sf.comb, l1.idx, l1.wts };
        const b = [_]A{ l2.x1, h2_l2, l2.sf.pre, l2.sf.post, l2.sf.comb, idx_l2, wts_l2 };
        for (names, a, b) |name, u, v| try expectSameBytes(gpa, name, u, v);
        const ri = try hostPair(chain.idx, idx_l2);
        const rw = try hostPair(chain.wts, wts_l2);
        defer for (ri) |h| gpa.free(h);
        defer for (rw) |h| gpa.free(h);
        try std.testing.expectEqual(@as(usize, 0), kernel_map.routeRowsBad(ri[0], ri[1], rw[0], rw[1], 1, ly.top_k));
        if (!printed) {
            printed = true;
            const p: ffn_in.Dims = .{ .rows = 1, .d = fx.m.cfg.hidden_size, .e = 16, .top_k = ly.top_k, .iters = fx.m.cfg.dsv4_hc_sinkhorn_iters };
            const g = seg_ffn.geometryOf(p, 576, p.d, 8);
            std.debug.print("[seg] ffn level 2 host grid: {d} threadgroups of {d} threads = {d} entry + {d} shared + {d} routed + {d} down\n", .{ g.items, ffn_in.b_threads, g.nin, g.ncb, p.top_k * g.ncb, g.items - g.nin - g.ncb - p.top_k * g.ncb });
        }
    };
}

test "dsv41 fused seg: level 2 decodes layers 20..23 no further from the f32 model than stock" {
    if (!gpuTest() or !device.naxAvailable()) return error.SkipZigTest;
    Sums.install();
    defer Sums.remove();
    var o = tp4;
    o.ctx = 601;
    o.seg = .ffn_seg;
    const before = seg.dispatches;
    try decode.layersAgreeOn(std.testing.allocator, o);
    try std.testing.expect(seg.dispatches > before);
}

test "dsv41 fused seg: four ranks route bit for bit over 1000 launches at level 2, an exact tie to the lower index" {
    if (!gpuTest() or !device.naxAvailable()) return error.SkipZigTest;
    const gpa = std.testing.allocator;
    var ranks: [4]*Fixture = undefined;
    var made: usize = 0;
    defer for (ranks[0..made]) |r| r.deinit();
    for (&ranks, 0..) |*r, i| {
        var o = tp4;
        o.lo = 21;
        o.hi = 23;
        o.experts = 384;
        o.ctx = 0;
        o.seg = .ffn_seg;
        o.tp = .{ .rank = @intCast(i), .size = 4 };
        r.* = try Fixture.init(gpa, o);
        made += 1;
        _ = try seg.ensure(r.*.m);
    }
    const w = try ffn_in.randomWeights(79);
    defer ffn_in.freeWeights(w);
    const ly = try tieLayer(w, ranks[0].layer(21));
    defer {
        _ = mlx.mlx_array_free(ly.gate_w);
        _ = mlx.mlx_array_free(ly.gate_bias);
    }
    var sc0 = dsv41.Scope.init(gpa, mlx.gpuStream());
    defer sc0.deinit();
    const in = try inputsOf(&sc0, ranks[0], &ly, 1);
    var first: ?[3][]f32 = null;
    defer if (first) |f| for (f) |a| gpa.free(a);
    const before = seg.dispatches;
    for (0..250) |n| for (ranks) |r| {
        var sc = dsv41.Scope.init(gpa, mlx.gpuStream());
        defer sc.deinit();
        const li: usize = 21 + (n % 2);
        const lyi = if (li == 21) ly else ffn_in.layerWith(r.layer(22), w);
        const out = (try seg_ffn.launch(&sc, r.m, &lyi, li, in.x, in.y, &in.sa)) orelse return error.Declined;
        const idx_o = try out.idx(&sc);
        const wts_o = try out.wts(&sc);
        try dsv41.evalAll(&.{ idx_o, wts_o, out.part });
        const got = [3][]f32{ try dsv41.hostF32(idx_o), try dsv41.hostF32(wts_o), try dsv41.hostF32(out.part) };
        if (li != 21) {
            for (got) |a| gpa.free(a);
            continue;
        }
        if (first) |f| {
            defer for (got) |a| gpa.free(a);
            for (f, got) |u, v| try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(u), std.mem.sliceAsBytes(v));
            continue;
        }
        first = got;
        for (got[0][0..5]) |e| try std.testing.expect(e < 5);
        try std.testing.expectEqual(@as(f32, 7), got[0][5]);
        for (got[0]) |e| try std.testing.expect(e != 300);
    };
    try std.testing.expectEqual(before + 1000, seg.dispatches);
}

fn freshState(fx: *Fixture) !void {
    if (fx.m.dec_state) |*st| st.deinit();
    fx.m.dec_state = null;
    fx.m.dec_state = try dsv41.initState(fx.m, fx.gpa);
}

fn stepOnce(fx: *Fixture, in: fixture.Input) ![]f32 {
    try freshState(fx);
    return fixture.streamOf(fx.gpa, fx, in);
}

test "dsv41 fused seg: an injected abort after phase k leaves the next step equal to a clean run" {
    if (!gpuTest() or !device.naxAvailable()) return error.SkipZigTest;
    const gpa = std.testing.allocator;
    Sums.install();
    defer Sums.remove();
    var o = tp4;
    o.ctx = 0;
    o.seg = .ffn_seg;
    const clean = try Fixture.init(gpa, o);
    defer clean.deinit();
    const victim = try Fixture.init(gpa, o);
    defer victim.deinit();
    const in = try fixture.Input.of(clean, 1);
    defer in.deinit();
    defer seg_ffn.hook = .{};
    const want = try stepOnce(clean, in);
    defer gpa.free(want);
    const before = seg.dispatches;
    for ([_]u32{ 1, 2, 3, 4 }) |k| {
        seg_ffn.hook = .{ .abort_at = k };
        try freshState(victim);
        try std.testing.expectError(error.Dsv41FusedSegTimeout, fixture.streamOf(gpa, victim, in));
        try std.testing.expectEqual(seg.Level.off, victim.m.fused_seg);
        seg_ffn.hook = .{};
        victim.m.fused_seg = .ffn_seg;
        const got = try stepOnce(victim, in);
        defer gpa.free(got);
        try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(want), std.mem.sliceAsBytes(got));
    }
    try std.testing.expect(seg.dispatches > before);
}

fn nowUs() i64 {
    return iou.nowUsMonotonic(std.Io.Threaded.global_single_threaded.io());
}

test "dsv41 fused seg: a spin timeout fails the step by name and latches stock" {
    if (!gpuTest() or !device.naxAvailable()) return error.SkipZigTest;
    const gpa = std.testing.allocator;
    Sums.install();
    defer Sums.remove();
    var o = tp4;
    o.ctx = 0;
    o.seg = .ffn_seg;
    const fx = try Fixture.init(gpa, o);
    defer fx.deinit();
    const in = try fixture.Input.of(fx, 1);
    defer in.deinit();
    defer seg_ffn.hook = .{};
    for ([_]u32{ 2, 3, 4 }) |k| {
        seg_ffn.hook = .{ .skew_at = k };
        try freshState(fx);
        fx.m.fused_seg = .ffn_seg;
        const t0 = nowUs();
        try std.testing.expectError(error.Dsv41FusedSegTimeout, fixture.streamOf(gpa, fx, in));
        std.debug.print("[seg] phase {d} wait target +1: timed out, step failed after {d:.1} ms\n", .{ k, @as(f64, @floatFromInt(nowUs() - t0)) / 1e3 });
        try std.testing.expectEqual(seg.Level.off, fx.m.fused_seg);
    }
}

test "dsv41 fused seg: a launch whose grid is not the kernel's item count fails by name and latches stock" {
    if (!gpuTest() or !device.naxAvailable()) return error.SkipZigTest;
    const gpa = std.testing.allocator;
    Sums.install();
    defer Sums.remove();
    var o = tp4;
    o.ctx = 0;
    o.seg = .ffn_seg;
    const fx = try Fixture.init(gpa, o);
    defer fx.deinit();
    const in = try fixture.Input.of(fx, 1);
    defer in.deinit();
    seg_ffn.grid_extra = 1;
    defer seg_ffn.grid_extra = 0;
    try std.testing.expectError(error.Dsv41FusedSegTimeout, fixture.streamOf(gpa, fx, in));
    try std.testing.expectEqual(seg.Level.off, fx.m.fused_seg);
}
