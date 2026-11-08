//! The fused decode segments' switch, coverage rules and exactness at every level (`DSV41_FUSED_SEG`).

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const device = @import("../../../mlx/device.zig");
const dsv41 = @import("../deepseek_v41.zig");
const decode = @import("decode.zig");
const fixture = @import("fixture.zig");
const kernel_map = @import("kernel_map.zig");
const seg = @import("seg.zig");
const seg_bounds = @import("seg_bounds.zig");

const A = mlx.mlx_array;
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
        seg.broken = false;
    }
    gpa.free(try runSteps(gpa, fx, &.{ 2, 2 }));
    for (0..6) |n| try kernel_map.segCheck(gpa, fx, o.lo, o.hi, .{ .level = @enumFromInt(n) });
    try std.testing.expectError(error.SegCheckFailed, kernel_map.segCheck(gpa, fx, o.lo, o.hi, .broken));
    for (o.lo..o.hi) |li| try std.testing.expectEqual(@as(c_int, 2), mlx.getShape(seg_bounds.inOf(li).?[0])[0]);
    try std.testing.expect(!seg.broken);
    try std.testing.expectEqual(seg.Level.off, fx.m.fused_seg);
}
