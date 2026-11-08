//! `mlx-serve dist-probe --sweep`: all_sum over the group at the message sizes a tensor-parallel decode and its prefill
//! chunks move, in bf16 and f32, each measured with the input ready and fed by a GPU op as a tensor-parallel sum is.
const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const io_util = @import("../../../sdk/util/io_util.zig");
const log = @import("../../../sdk/util/log.zig");
const dist = @import("distributed.zig");
const timeline = @import("sum_timeline.zig");

pub const message_bytes = [_]u32{ 4, 10240, 16384, 20480, 32768, 61440, 65536, 122880, 262144 };
pub const samples_per_row = 200;
const warmup = 20;

pub const Dtype = enum {
    bf16,
    f32,

    fn itemBytes(d: Dtype) u32 {
        return if (d == .bf16) 2 else 4;
    }

    fn mlxDtype(d: Dtype) mlx.mlx_dtype {
        return if (d == .bf16) .bfloat16 else .float32;
    }
};

pub const Arm = enum {
    ready,
    gpu_fed,
};

/// JACCL's mesh all_reduce: past two ranks a message over 32 KiB is a reduce-scatter then an all-gather of its
/// per-rank share, else every rank sends its whole vector to every peer. The frame is the buffer one transfer
/// uses (macOS 26.3+); a message takes `chunks` of them.
pub const Path = struct { frame: u32, chunks: u32, scatter_gather: bool };

const frame_bytes: u32 = 4096;
const frame_sizes = 8;
const scatter_gather_over: u32 = 32 * 1024;

pub fn pathOf(bytes: u32, ranks: u32) Path {
    const scatter_gather = ranks > 2 and bytes > scatter_gather_over;
    const moved = if (scatter_gather) bytes / ranks else bytes;
    const frame = frameFor(moved);
    return .{ .frame = frame, .chunks = (moved + frame - 1) / frame, .scatter_gather = scatter_gather };
}

fn frameFor(moved: u32) u32 {
    var k: u5 = frame_sizes - 1;
    while (k > 0) : (k -= 1) {
        if (moved >= frame_bytes << k) return frame_bytes << k;
    }
    return frame_bytes;
}

pub const Row = struct { p50: u64, p90: u64, min: u64 };

pub fn rowOf(samples: []u64) Row {
    std.mem.sort(u64, samples, {}, std.sort.asc(u64));
    return .{ .p50 = timeline.nearestRank(samples, 50), .p90 = timeline.nearestRank(samples, 90), .min = samples[0] };
}

pub fn formatRow(w: *std.Io.Writer, ranks: u32, rank: u32, dt: Dtype, bytes: u32, arm: Arm, row: Row) !void {
    const path = pathOf(bytes, ranks);
    const us = 1000.0;
    try w.print("[dist-probe] sweep ranks={d} rank={d} {s} {d} B {s} {d}x{d} B {s}: p50 {d:.1} p90 {d:.1} min {d:.1} us", .{
        ranks,                                        rank,                                  @tagName(dt),                          bytes,
        if (path.scatter_gather) "rs+ag" else "full", path.chunks,                           path.frame,                            @tagName(arm),
        @as(f64, @floatFromInt(row.p50)) / us,        @as(f64, @floatFromInt(row.p90)) / us, @as(f64, @floatFromInt(row.min)) / us,
    });
}

fn evalOne(a: mlx.mlx_array) !void {
    try mlx.check(mlx.mlx_array_eval(a));
}

fn add(a: mlx.mlx_array, b: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_add(&out, a, b, s));
    return out;
}

fn once(link: *const dist.Link, s: mlx.mlx_stream, x: mlx.mlx_array, zero: mlx.mlx_array, arm: Arm) !void {
    const input = if (arm == .gpu_fed) try add(x, zero, s) else x;
    defer if (arm == .gpu_fed) {
        _ = mlx.mlx_array_free(input);
    };
    const summed = try link.allSum(input);
    defer _ = mlx.mlx_array_free(summed);
    if (arm == .ready) return evalOne(summed);
    const consumed = try add(summed, zero, s);
    defer _ = mlx.mlx_array_free(consumed);
    try evalOne(consumed);
}

fn measure(io: std.Io, link: *const dist.Link, s: mlx.mlx_stream, dt: Dtype, bytes: u32, arm: Arm) !Row {
    var x = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x);
    var zero = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(zero);
    try mlx.check(mlx.mlx_ones(&x, &[_]c_int{@intCast(bytes / dt.itemBytes())}, 1, dt.mlxDtype(), s));
    try mlx.check(mlx.mlx_zeros(&zero, &[_]c_int{1}, 1, dt.mlxDtype(), s));
    try evalOne(x);
    try evalOne(zero);
    for (0..warmup) |_| try once(link, s, x, zero, arm);
    timeline.discard();
    var samples: [samples_per_row]u64 = undefined;
    for (&samples) |*sample| {
        var sw = io_util.Stopwatch.init(io);
        try once(link, s, x, zero, arm);
        sample.* = sw.read();
    }
    return rowOf(&samples);
}

pub fn run(io: std.Io, link: *const dist.Link, s: mlx.mlx_stream) !void {
    timeline.arm(link.rank);
    defer timeline.disarm();
    for (std.enums.values(Dtype)) |dt| for (message_bytes) |bytes| for (std.enums.values(Arm)) |arm| {
        const row = try measure(io, link, s, dt, bytes, arm);
        var buf: [320]u8 = undefined;
        var w: std.Io.Writer = .fixed(&buf);
        try formatRow(&w, link.size, link.rank, dt, bytes, arm, row);
        log.info("{s}\n", .{w.buffered()});
        timeline.report();
    };
    log.info("[dist-probe] sweep done\n", .{});
}

test "dist sum sweep: the sizes the tensor sums move, smallest first" {
    try std.testing.expect(std.sort.isSorted(u32, &message_bytes, {}, std.sort.asc(u32)));
    for ([_]u32{ 4, 10240, 16384, 20480, 32768, 61440, 65536, 122880, 262144 }) |b| {
        try std.testing.expect(std.mem.indexOfScalar(u32, &message_bytes, b) != null);
    }
    for (message_bytes) |b| try std.testing.expect(b % 4 == 0);
}

test "dist sum sweep: JACCL's path and frames at four ranks, switching to reduce-scatter past 32 KiB" {
    try std.testing.expectEqual(Path{ .frame = 4096, .chunks = 1, .scatter_gather = false }, pathOf(4, 4));
    try std.testing.expectEqual(Path{ .frame = 8192, .chunks = 2, .scatter_gather = false }, pathOf(10240, 4));
    try std.testing.expectEqual(Path{ .frame = 16384, .chunks = 2, .scatter_gather = false }, pathOf(20480, 4));
    try std.testing.expectEqual(Path{ .frame = 32768, .chunks = 1, .scatter_gather = false }, pathOf(32768, 4));
    try std.testing.expectEqual(Path{ .frame = 8192, .chunks = 2, .scatter_gather = true }, pathOf(61440, 4));
    try std.testing.expectEqual(Path{ .frame = 65536, .chunks = 1, .scatter_gather = true }, pathOf(262144, 4));
    try std.testing.expect(!pathOf(262144, 2).scatter_gather);
}

test "dist sum sweep: a row is the sorted samples' median, 90th percentile and minimum" {
    var samples = [_]u64{ 30_000, 10_000, 20_000, 50_000, 40_000, 60_000, 70_000, 80_000, 90_000, 100_000 };
    try std.testing.expectEqual(Row{ .p50 = 50_000, .p90 = 90_000, .min = 10_000 }, rowOf(&samples));
}

test "dist sum sweep: the row line names the path, the dtype and the arm" {
    var buf: [320]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    try formatRow(&w, 4, 1, .bf16, 61440, .gpu_fed, .{ .p50 = 41_500, .p90 = 52_000, .min = 38_000 });
    try std.testing.expectEqualStrings("[dist-probe] sweep ranks=4 rank=1 bf16 61440 B rs+ag 2x8192 B gpu_fed: p50 41.5 p90 52.0 min 38.0 us", w.buffered());
}
