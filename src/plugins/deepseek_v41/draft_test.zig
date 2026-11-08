//! The DSpark draft over a release-shaped fixture of random weights (`decode/fixture.zig` `initDraft`): the compiled
//! stages and tail draft what the op chains draft, and the ubench times both (`DSV41_DEC_UBENCH`).

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const iou = @import("../../sdk/util/io_util.zig");
const decode = @import("decode/decode.zig");
const fixture = @import("decode/fixture.zig");
const dsv41 = @import("deepseek_v41.zig");

fn gpuUnavailable() bool {
    return !mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu();
}

fn draftAll(fx: *fixture.Fixture, states: []const *dsv41.State, t1s: []const u32, keep_logits: bool, out: []dsv41.DsparkDraft) !void {
    try dsv41.dsparkDraftBatched(fx.m, fx.gpa, states, t1s, keep_logits, out);
}

fn expectSameDrafts(gpa: std.mem.Allocator, a: []dsv41.DsparkDraft, b: []dsv41.DsparkDraft) !void {
    for (a, b) |*x, *y| {
        defer x.deinit(gpa);
        defer y.deinit(gpa);
        try std.testing.expectEqualSlices(u32, x.ids, y.ids);
        try std.testing.expectEqual(x.len, y.len);
        try std.testing.expectEqualSlices(f32, x.confidence, y.confidence);
        try std.testing.expectEqualSlices(f32, x.logits, y.logits);
    }
}

fn compareCompiledToEager(fx: *fixture.Fixture, states: []const *dsv41.State, t1s: []const u32, keep_logits: bool) !void {
    const gpa = fx.gpa;
    var eager: [4]dsv41.DsparkDraft = undefined;
    var compiled: [4]dsv41.DsparkDraft = undefined;
    const n = states.len;
    fx.m.draft_compiled = false;
    try draftAll(fx, states, t1s, keep_logits, eager[0..n]);
    fx.m.draft_compiled = true;
    try draftAll(fx, states, t1s, keep_logits, compiled[0..n]);
    try expectSameDrafts(gpa, eager[0..n], compiled[0..n]);
}

test "dsv41 draft: compiled stages and tail draft the ids, confidences and logits the op chains draft, one state and a batch, across the window's fill" {
    if (gpuUnavailable()) return error.SkipZigTest;
    const gpa = std.testing.allocator;
    const fx = try fixture.Fixture.initDraft(gpa, .{ .experts = 16, .vocab = 4096, .ctx = 120, .act = .bfloat16 });
    defer fx.deinit();
    fx.m.ds_want_conf = true;
    var second = try dsv41.initState(fx.m, gpa);
    defer second.deinit();
    for (second.ds) |*ring| {
        _ = mlx.mlx_array_free(ring.buf);
        ring.buf = try fx.normal(&.{ @intCast(ring.cap), @intCast(ring.width) }, fx.m.act, 0, 1);
    }
    second.n = 40;
    for (0..12) |step| {
        const t1s = [_]u32{ @intCast(7 + 31 * step), @intCast(1000 + 13 * step) };
        const states = [_]*dsv41.State{ fx.state(), &second };
        try compareCompiledToEager(fx, states[0..1], t1s[0..1], step % 2 == 0);
        try compareCompiledToEager(fx, &states, &t1s, step % 2 == 1);
        fx.state().n += 1;
        second.n += 1;
    }
}

test "dsv41 draft: without the confidence head the compiled tail drafts the same ids" {
    if (gpuUnavailable()) return error.SkipZigTest;
    const gpa = std.testing.allocator;
    const fx = try fixture.Fixture.initDraft(gpa, .{ .experts = 16, .vocab = 4096, .ctx = 200, .act = .bfloat16 });
    defer fx.deinit();
    for (0..4) |step| {
        const t1s = [_]u32{@intCast(11 + step)};
        try compareCompiledToEager(fx, &.{fx.state()}, &t1s, false);
    }
}

const Measure = struct { total_us: f64, build_us: f64, eval_us: f64, ops: u64 };

fn nowUs() i64 {
    return iou.nowUsMonotonic(std.Io.Threaded.global_single_threaded.io());
}

fn median(xs: []f64) f64 {
    std.mem.sort(f64, xs, {}, std.sort.asc(f64));
    return xs[xs.len / 2];
}

fn measureDraft(fx: *fixture.Fixture, t1: u32) !Measure {
    const gpa = fx.gpa;
    const reps = 40;
    var total: [reps]f64 = undefined;
    for (0..reps + 5) |i| {
        const t0 = nowUs();
        var d = try dsv41.dsparkDraft(fx.m, gpa, fx.state(), t1);
        d.deinit(gpa);
        if (i >= 5) total[i - 5] = @floatFromInt(nowUs() - t0);
    }
    var build: [reps]f64 = undefined;
    var eval: [reps]f64 = undefined;
    var ops: u64 = 0;
    for (0..reps + 5) |i| {
        var sc = dsv41.Scope.init(gpa, fx.m.s);
        defer sc.deinit();
        const ops0 = mlx.op_count.load(.monotonic);
        const t0 = nowUs();
        const g = try dsv41.draftGraph(&sc, fx.m, gpa, &.{fx.state()}, &.{t1}, false);
        const t1_us = nowUs();
        try mlx.check(mlx.mlx_array_eval(g.ids));
        const t2_us = nowUs();
        if (i < 5) continue;
        ops = mlx.op_count.load(.monotonic) - ops0;
        build[i - 5] = @floatFromInt(t1_us - t0);
        eval[i - 5] = @floatFromInt(t2_us - t1_us);
    }
    return .{ .total_us = median(&total), .build_us = median(&build), .eval_us = median(&eval), .ops = ops };
}

fn dependentAddUs(fx: *fixture.Fixture) !f64 {
    const n = 400;
    var times: [7]f64 = undefined;
    for (0..2 + times.len) |rep| {
        var sc = dsv41.Scope.init(fx.gpa, fx.m.s);
        defer sc.deinit();
        var x = try sc.keep(try fx.normal(&.{ 5, 5120 }, .bfloat16, 0, 1));
        const one = try sc.f(1.0);
        const t0 = nowUs();
        for (0..n) |_| x = try sc.add(x, one);
        try mlx.check(mlx.mlx_array_eval(x));
        if (rep >= 2) times[rep - 2] = @as(f64, @floatFromInt(nowUs() - t0)) / n;
    }
    return median(&times);
}

test "dsv41 draft ubench: one draft of the release's DSpark stages, op chains against compiled regions (DSV41_DEC_UBENCH)" {
    if (!decode.ubenchOn() or gpuUnavailable()) return error.SkipZigTest;
    const gpa = std.testing.allocator;
    const fx = try fixture.Fixture.initDraft(gpa, .{});
    defer fx.deinit();
    std.debug.print("[draft-ubench] dependent_add_us={d:.2}\n", .{try dependentAddUs(fx)});
    const stages = fx.m.n_mtp;
    for ([_]usize{ stages, 1, stages, 1 }) |n_stages| {
        fx.m.n_mtp = n_stages;
        for ([_]bool{ false, true, false, true }) |compiled| {
            fx.m.draft_compiled = compiled;
            const r = try measureDraft(fx, 42);
            std.debug.print("[draft-ubench] stages={d} compiled={} ms_per_draft={d:.3} build_ms={d:.3} eval_ms={d:.3} ops_issued={d}\n", .{ n_stages, compiled, r.total_us / 1e3, r.build_us / 1e3, r.eval_us / 1e3, r.ops });
        }
    }
}
