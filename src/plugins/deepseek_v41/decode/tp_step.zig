//! One tensor-split rank's decode step over the fixture's layers, its sums an identity (no hand-off): the GPU
//! dispatches the step encodes and its time, `DSV41_TP_FUSE` on and off.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const iou = @import("../../../sdk/util/io_util.zig");
const dsv41 = @import("../deepseek_v41.zig");
const decode = @import("decode.zig");
const fixture = @import("fixture.zig");

const A = mlx.mlx_array;

fn nowUs() i64 {
    return iou.nowUsMonotonic(std.Io.Threaded.global_single_threaded.io());
}

fn identitySum(x: A) anyerror!A {
    var out = mlx.mlx_array_new();
    _ = mlx.mlx_array_set(&out, x);
    return out;
}

/// One step's graph over the fixture's layers at `p0`, unevaluated: the stream, the mix, then every layer's state.
fn build(sc: *dsv41.Scope, gpa: std.mem.Allocator, fx: *fixture.Fixture, x0: A, pre0: A, p0: usize, outs: *std.ArrayList(A)) !void {
    const m = fx.m;
    const p = m.pair.?;
    const st = &m.dec_state.?;
    var x = x0;
    var pre = pre0;
    var sh: dsv41.Shared = .{ .carry = true };
    for (p.lo..p.hi) |li| x = try dsv41.block(sc, m, st, li, x, &pre, p0, &sh);
    x = try dsv41.settle(sc, x, &sh);
    st.n = p0 + @as(usize, @intCast(mlx.getShape(x0)[0]));
    try outs.appendSlice(gpa, &.{ x, pre });
    for (st.layers[p.lo..p.hi]) |*ls| try dsv41.appendLayerState(gpa, outs, ls);
}

extern "c" fn open_memstream(buf: *?[*]u8, len: *usize) ?*std.c.FILE;
extern "c" fn free(p: ?*anyopaque) void;

/// Primitives that make no launch of their own: views, and arrays already made.
const views = [_][]const u8{ "Reshape", "Broadcast", "Slice", "ExpandDims", "Squeeze", "Transpose", "StopGradient", "AsStrided", "Depends", "NumberOfElements" };

/// The launches `outs`' graph makes: every primitive the printed tape lists, less the views.
fn launches(outs: []const A) !usize {
    var buf: ?[*]u8 = null;
    var len: usize = 0;
    const f = open_memstream(&buf, &len) orelse return error.NoMemstream;
    const vec = mlx.mlx_vector_array_new_data(outs.ptr, outs.len);
    defer _ = mlx.mlx_vector_array_free(vec);
    const namer = mlx.mlx_node_namer_new();
    defer _ = mlx.mlx_node_namer_free(namer);
    _ = mlx.mlx_print_graph(f, namer, vec);
    _ = std.c.fclose(f);
    defer free(buf);
    const text = buf.?[0..len];
    var c: usize = 0;
    var it = std.mem.tokenizeScalar(u8, text, '\n');
    while (it.next()) |line| {
        const name = line[0 .. std.mem.indexOfAny(u8, line, " (:") orelse line.len];
        if (name.len == 0 or !std.ascii.isUpper(name[0]) or std.mem.startsWith(u8, line, "Inputs:") or std.mem.startsWith(u8, line, "Outputs:")) continue;
        const view = for (views) |v| {
            if (std.mem.eql(u8, name, v)) break true;
        } else false;
        if (!view) c += 1;
    }
    return c;
}

fn rankOpts(lo: usize, hi: usize, ctx: usize, experts: u32) fixture.Opts {
    return .{ .lo = lo, .hi = hi, .experts = experts, .ctx = ctx, .fused = true, .experts_fp4 = true, .tp = .{ .rank = 0, .size = 4 } };
}

fn withIdentitySums() void {
    dsv41.clusterHooks().tp_sum = &identitySum;
}

test "dsv41 tp step: a TP4 rank's decode step encodes one launch fewer a layer past two rows with the hc expand inside the next layer's input" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const gpa = std.testing.allocator;
    withIdentitySums();
    defer dsv41.clusterHooks().tp_sum = null;
    defer dsv41.tp_fuse = null;
    // An arm a fixture, each fed the same steps: a step that grows a state's buffers (`Full`, `SliceUpdate`) grows both.
    var arms: [2]*fixture.Fixture = undefined;
    var made: usize = 0;
    defer for (arms[0..made]) |fx| fx.deinit();
    for (&arms) |*fx| {
        fx.* = try fixture.Fixture.init(gpa, rankOpts(20, 24, 64, 8));
        made += 1;
    }
    const layers = arms[0].m.pair.?.hi - arms[0].m.pair.?.lo;
    for ([_]usize{ 1, 2, 3, decode.max_rows }) |rows| {
        const x = try arms[0].stream(rows);
        defer _ = mlx.mlx_array_free(x);
        const pre = try arms[0].cpre(rows);
        defer _ = mlx.mlx_array_free(pre);
        var got: [2]usize = undefined;
        for (arms, &got, [_]bool{ false, true }) |fx, *g, on| {
            dsv41.tp_fuse = on;
            var sc = dsv41.Scope.init(gpa, fx.m.s);
            defer sc.deinit();
            var outs: std.ArrayList(A) = .empty;
            defer outs.deinit(gpa);
            try build(&sc, gpa, fx, x, pre, 64, &outs);
            g.* = try launches(outs.items);
            try dsv41.evalAll(outs.items);
        }
        std.debug.print("[tp-step] TP4 rank 0, {d} layers, R={d}: {d} launches a step fuse off, {d} on ({d:.2} -> {d:.2} a layer)\n", .{
            layers, rows, got[0], got[1], @as(f64, @floatFromInt(got[0])) / @as(f64, @floatFromInt(layers)), @as(f64, @floatFromInt(got[1])) / @as(f64, @floatFromInt(layers)),
        });
        // At every layer boundary the step crosses the next layer's stage kernel makes the stream: one launch fewer where
        // the stage runs anyway (past two rows), the same count where it takes the expand's place. The last layer's stays.
        try std.testing.expectEqual(if (rows > 2) got[0] - (layers - 1) else got[0], got[1]);
    }
}

/// A step's times: the graph built first and then evaluated (`eval`, the device's time with the encoding under it), and
/// the serving loop's way (`served`: `workerStep`, a layer handed to the GPU as it is built).
const Times = struct { eval: f64, served: f64 };

fn median(v: []f64) f64 {
    std.mem.sort(f64, v, {}, std.sort.asc(f64));
    return v[v.len / 2];
}

/// `chain` steps built then evaluated, each fed the last one's stream as a zero-weighted dependency: microseconds a step.
fn evalUs(gpa: std.mem.Allocator, fx: *fixture.Fixture, x0: A, pre0: A, p0: usize) !f64 {
    const chain = 4;
    var sc = dsv41.Scope.init(gpa, fx.m.s);
    defer sc.deinit();
    var outs: std.ArrayList(A) = .empty;
    defer outs.deinit(gpa);
    var x = x0;
    var pre = pre0;
    for (0..chain) |k| {
        outs.clearRetainingCapacity();
        try build(&sc, gpa, fx, x, pre, p0 + k % 2, &outs);
        const first = try sc.astype(try sc.slice(try sc.reshape(outs.items[0], &.{-1}), 0, 0, 1), .float32);
        x = try sc.astype(try sc.add(x0, try sc.mul(first, try sc.f(0))), mlx.mlx_array_dtype(x0));
        pre = outs.items[1];
    }
    try outs.append(gpa, x);
    const t0 = nowUs();
    try dsv41.evalAll(outs.items);
    return @as(f64, @floatFromInt(nowUs() - t0)) / chain;
}

fn times(gpa: std.mem.Allocator, fx: *fixture.Fixture, rows: usize) !Times {
    const x = try fx.stream(rows);
    defer _ = mlx.mlx_array_free(x);
    const pre = try fx.cpre(rows);
    defer _ = mlx.mlx_array_free(pre);
    var e: [9]f64 = undefined;
    for (0..3) |_| _ = try evalUs(gpa, fx, x, pre, fx.m.dec_state.?.n);
    for (&e) |*v| v.* = try evalUs(gpa, fx, x, pre, fx.m.dec_state.?.n);
    var s: [9]f64 = undefined;
    for (0..3 + s.len) |rep| {
        const t0 = nowUs();
        const out = try dsv41.workerStep(fx.m, gpa, x, pre, fx.m.dec_state.?.n, false, false);
        for (out) |o| _ = mlx.mlx_array_free(o);
        if (rep >= 3) s[rep - 3] = @floatFromInt(nowUs() - t0);
    }
    return .{ .eval = median(&e), .served = median(&s) };
}

test "dsv41 tp step ubench: a TP4 rank's decode step through the trunk, 1, 2 and 6 rows, DSV41_TP_FUSE off and on (DSV41_DEC_UBENCH)" {
    if (!decode.ubenchOn() or !mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const gpa = std.testing.allocator;
    withIdentitySums();
    defer dsv41.clusterHooks().tp_sum = null;
    defer dsv41.tp_fuse = null;
    const ctx = 2048;
    const rounds = 7;
    // The trunk's two halves, each its own fixture over the release's 384 experts cut to a quarter; the arms alternate
    // within a fixture, each round's times a median, the rounds' median kept.
    var total: [3][2]Times = std.mem.zeroes([3][2]Times);
    for ([_][2]usize{ .{ 0, 20 }, .{ 20, 40 } }) |half| {
        const fx = try fixture.Fixture.init(gpa, rankOpts(half[0], half[1], ctx, 384));
        defer fx.deinit();
        for ([_]usize{ 1, 2, decode.max_rows }, 0..) |rows, ri| {
            var ev: [2][rounds]f64 = undefined;
            var sv: [2][rounds]f64 = undefined;
            for (0..rounds) |k| for ([_]bool{ false, true }, 0..) |on, fi| {
                dsv41.tp_fuse = on;
                const t = try times(gpa, fx, rows);
                ev[fi][k] = t.eval;
                sv[fi][k] = t.served;
            };
            for (0..2) |fi| {
                const t: Times = .{ .eval = median(&ev[fi]), .served = median(&sv[fi]) };
                total[ri][fi].eval += t.eval;
                total[ri][fi].served += t.served;
                std.debug.print("[tp-step] layers {d}-{d} R={d} fuse={}: eval {d:.0} us, served {d:.0} us (rounds' range eval {d:.0}..{d:.0})\n", .{ half[0], half[1], rows, fi == 1, t.eval, t.served, ev[fi][0], ev[fi][rounds - 1] });
            }
        }
    }
    for ([_]usize{ 1, 2, decode.max_rows }, 0..) |rows, ri| std.debug.print("[tp-step] TP4 rank 0, 40 layers, R={d}: eval {d:.2} -> {d:.2} ms, served {d:.2} -> {d:.2} ms (fuse off -> on)\n", .{
        rows, total[ri][0].eval / 1e3, total[ri][1].eval / 1e3, total[ri][0].served / 1e3, total[ri][1].served / 1e3,
    });
}
