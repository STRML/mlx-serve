//! Where a tensor-parallel sum's time goes, armed by diag `tp-sum-timeline`: a patched MLX (patches/mlx-tp-sum-timeline.patch,
//! patches/jaccl-tp-sum-cqe.patch) calls `onPhase` and `onPeer` on the CPU stream's thread; a stock MLX only makes `arm` say so.
const std = @import("std");
const builtin = @import("builtin");
const env = @import("../../../env.zig");
const log = @import("../../../sdk/util/log.zig");
const io_util = @import("../../../sdk/util/io_util.zig");
const mlx = @import("../../../mlx/ffi.zig");
const Shared = @import("../../../sdk/util/shared.zig").Shared;

pub const max_peers = 8;
pub const ring_len = 32768;
pub const steps_per_report = 256;
pub const max_sizes = 6;

pub const Phase = enum(u32) {
    wait_begin,
    /// a: the fence is seen.
    fence_seen,
    /// b: all_sum is entered; the argument is the sum's bytes.
    sum_enter,
    /// c: all_sum returns.
    sum_return,
    /// d: the CPU's store that releases the GPU's fence wait.
    release,
};

pub const Sample = struct {
    bytes: u32 = 0,
    wait_begin: u64 = 0,
    fence_seen: u64 = 0,
    enter: u64 = 0,
    leave: u64 = 0,
    release: u64 = 0,
    peers: [max_peers]u32 = @splat(0),
};

pub const Recorder = struct {
    const Stage = enum { idle, waiting, seen, in_sum, summed };

    cur: Sample = .{},
    stage: Stage = .idle,

    pub fn phase(self: *Recorder, ring: *Ring, p: Phase, arg: u64, ns: u64) void {
        switch (p) {
            .wait_begin => {
                self.cur = .{ .wait_begin = ns };
                self.stage = .waiting;
            },
            .fence_seen => if (self.stage == .waiting) {
                self.cur.fence_seen = ns;
                self.stage = .seen;
            },
            .sum_enter => {
                if (self.stage != .waiting and self.stage != .seen) self.cur = .{};
                self.cur.bytes = @intCast(@min(arg, std.math.maxInt(u32)));
                self.cur.enter = ns;
                self.stage = .in_sum;
            },
            .sum_return => if (self.stage == .in_sum) {
                self.cur.leave = ns;
                self.stage = .summed;
            },
            .release => {
                if (self.stage == .summed) {
                    self.cur.release = ns;
                    ring.push(self.cur);
                }
                self.stage = .idle;
            },
        }
    }

    /// Single-wire sums only (under 512 KB): a multi-wire sum polls on JACCL's pool threads.
    pub fn peerDone(self: *Recorder, peer: u32, ns: u64) void {
        if (self.stage != .in_sum or peer >= max_peers) return;
        self.cur.peers[peer] = @intCast(@max(1, @min(ns -| self.cur.enter, std.math.maxInt(u32))));
    }
};

pub const Ring = struct {
    slots: []Sample = &.{},
    head: std.atomic.Value(u64) = .init(0),
    tail: u64 = 0,

    pub const Taken = struct { n: usize, dropped: u64 };

    pub fn push(r: *Ring, s: Sample) void {
        const head = r.head.load(.monotonic);
        r.slots[head & (r.slots.len - 1)] = s;
        r.head.store(head + 1, .release);
    }

    pub fn take(r: *Ring, out: []Sample) Taken {
        const cap: u64 = r.slots.len;
        const head = r.head.load(.acquire);
        var dropped: u64 = 0;
        if (head - r.tail > cap) {
            dropped = head - r.tail - cap;
            r.tail = head - cap;
        }
        const first = r.tail;
        const n: usize = @intCast(head - first);
        for (out[0..n], 0..) |*o, k| o.* = r.slots[(first + k) & (cap - 1)];
        const lost: usize = @intCast(@min(n, overwritten(first, r.head.fetchAdd(0, .acq_rel), cap)));
        std.mem.copyForwards(Sample, out[0 .. n - lost], out[lost..n]);
        r.tail = head;
        return .{ .n = n - lost, .dropped = dropped + lost };
    }
};

/// How many of the samples from `first` the producer may have begun overwriting once `head` are published: it writes
/// slot `head` before it publishes it.
pub fn overwritten(first: u64, head: u64, cap: u64) u64 {
    return if (head + 1 > first + cap) head + 1 - first - cap else 0;
}

pub const Metric = enum {
    spin,
    lead,
    comm,
    release,
    cqe_first,
    cqe_last,
};
pub const metric_count = std.enums.values(Metric).len;

pub fn value(s: *const Sample, m: Metric) ?u64 {
    const seen = s.fence_seen != 0;
    return switch (m) {
        .spin => if (seen) s.fence_seen -| s.wait_begin else null,
        .lead => if (seen) s.enter -| s.fence_seen else null,
        .comm => s.leave -| s.enter,
        .release => s.release -| s.leave,
        .cqe_first => peerExtreme(s, .min),
        .cqe_last => peerExtreme(s, .max),
    };
}

fn peerExtreme(s: *const Sample, comptime which: enum { min, max }) ?u64 {
    var out: ?u64 = null;
    for (s.peers) |p| {
        if (p == 0) continue;
        out = if (out) |o| (if (which == .min) @min(o, p) else @max(o, p)) else p;
    }
    return out;
}

pub const Quantiles = struct { p50: u64, p90: u64, max: u64 };

pub fn quantiles(sorted: []const u64) Quantiles {
    return .{ .p50 = nearestRank(sorted, 50), .p90 = nearestRank(sorted, 90), .max = sorted[sorted.len - 1] };
}

pub fn nearestRank(sorted: []const u64, pct: u64) u64 {
    const rank = (sorted.len * pct + 99) / 100;
    return sorted[@max(rank, 1) - 1];
}

pub const Group = struct {
    bytes: u32 = 0,
    n: u32 = 0,
    q: [metric_count]?Quantiles = @splat(null),
};

pub fn summarize(samples: []const Sample, scratch: []u64, out: *[max_sizes + 1]Group) []const Group {
    var used: usize = 0;
    var counts: [max_sizes + 1]u32 = @splat(0);
    for (samples) |s| counts[slot(out, &used, s.bytes)] += 1;
    for (out[0..used], counts[0..used]) |*g, c| g.n = c;
    std.mem.sort(Group, out[0..used], {}, struct {
        fn more(_: void, a: Group, b: Group) bool {
            return a.n > b.n;
        }
    }.more);
    for (out[0..used]) |*g| {
        inline for (0..metric_count) |mi| {
            var k: usize = 0;
            for (samples) |*s| if (slotOf(out[0..used], s.bytes) == g.bytes) {
                if (value(s, @fromBackingInt(@intCast(mi)))) |v| {
                    scratch[k] = v;
                    k += 1;
                }
            };
            if (k > 0) {
                std.mem.sort(u64, scratch[0..k], {}, std.sort.asc(u64));
                g.q[mi] = quantiles(scratch[0..k]);
            }
        }
    }
    return out[0..used];
}

fn slot(groups: *[max_sizes + 1]Group, used: *usize, bytes: u32) usize {
    for (groups[0..used.*], 0..) |g, i| if (g.bytes == bytes) return i;
    if (used.* < max_sizes) {
        groups[used.*] = .{ .bytes = bytes };
        used.* += 1;
        return used.* - 1;
    }
    if (used.* == max_sizes) {
        groups[used.*] = .{ .bytes = 0 };
        used.* += 1;
    }
    return used.* - 1;
}

fn slotOf(groups: []const Group, bytes: u32) u32 {
    for (groups) |g| if (g.bytes == bytes) return g.bytes;
    return 0;
}

fn writeTriple(w: *std.Io.Writer, name: []const u8, q: ?Quantiles) !void {
    try w.print(" {s} ", .{name});
    const t = q orelse return w.writeAll("-");
    const us = 1000.0;
    try w.print("{d:.1}/{d:.1}/{d:.1}", .{ @as(f64, @floatFromInt(t.p50)) / us, @as(f64, @floatFromInt(t.p90)) / us, @as(f64, @floatFromInt(t.max)) / us });
}

/// `a` the fence seen, `b` all_sum entered, `c` it returned, `d` the release store; spin runs from the wait task's start
/// to `a`, and cqe-first / cqe-last are the earliest and latest peer's last receive completion after `b`. Microseconds,
/// p50/p90/max.
pub fn formatLine(w: *std.Io.Writer, rank: u32, g: Group, dropped: u64) !void {
    try w.print("[tp-sum] rank {d} {d} B n={d} us", .{ rank, g.bytes, g.n });
    const names = [metric_count][]const u8{ "spin", "a>b", "b>c", "c>d", "cqe-first", "cqe-last" };
    inline for (names, 0..) |name, i| try writeTriple(w, name, g.q[i]);
    try w.print(" dropped {d}", .{dropped});
}

pub const Timeline = struct {
    live: std.atomic.Value(bool) = .init(false),
    rank: u32 = 0,
    steps: u32 = 0,
    rec: Recorder = .{},
    ring: Ring = .{},
    taken: []Sample = &.{},
    scratch: []u64 = &.{},
};

pub const timeline_shared = Shared(Timeline, "tp_sum_timeline", .{});

fn timeline() *Timeline {
    return timeline_shared.get();
}

fn onPhase(p: u32, arg: u64, ns: u64) callconv(.c) void {
    const t = timeline();
    if (!t.live.load(.acquire)) return;
    t.rec.phase(&t.ring, std.enums.fromInt(Phase, p) orelse return, arg, ns);
}

fn onPeer(peer: u32, ns: u64) callconv(.c) void {
    const t = timeline();
    if (!t.live.load(.acquire)) return;
    t.rec.peerDone(peer, ns);
}

const PhaseHook = *const fn (u32, u64, u64) callconv(.c) void;
const PeerHook = *const fn (u32, u64) callconv(.c) void;
const SetPhaseHook = *const fn (?PhaseHook) callconv(.c) void;
const SetPeerHook = *const fn (?PeerHook) callconv(.c) void;

fn lookup(comptime T: type, name: [*:0]const u8) ?T {
    if (comptime builtin.os.tag != .macos) return null;
    const any_image: ?*anyopaque = @ptrFromInt(@as(usize, @bitCast(@as(isize, -2))));
    return @ptrCast(@alignCast(std.c.dlsym(any_image, name) orelse return null));
}

pub fn wanted() bool {
    return env.on(.TP_SUM_TIMELINE);
}

pub fn arm(rank: u32) void {
    if (!wanted()) return;
    const t = timeline();
    if (t.live.load(.acquire)) return;
    const set_phase = lookup(SetPhaseHook, "mlx_set_tp_sum_hook") orelse {
        log.warn("[tp-sum] this MLX has no sum hook: apply patches/mlx-tp-sum-timeline.patch to lib/mlx-src and rebuild it\n", .{});
        return;
    };
    const alloc = std.heap.page_allocator;
    t.ring = .{ .slots = alloc.alloc(Sample, ring_len) catch return };
    t.taken = alloc.alloc(Sample, ring_len) catch return freeBuffers(t);
    t.scratch = alloc.alloc(u64, ring_len) catch return freeBuffers(t);
    t.rank = rank;
    t.steps = 0;
    t.rec = .{};
    t.live.store(true, .release);
    set_phase(&onPhase);
    const set_peer = lookup(SetPeerHook, "jaccl_set_cqe_hook");
    if (set_peer) |f| f(&onPeer);
    log.info("[tp-sum] rank {d}: a = fence seen, b = all_sum entry, c = all_sum return, d = release store; spin = wait task start to a, cqe = a peer's last receive completion after b; peers {s}\n", .{ rank, if (set_peer == null) "unstamped (no patches/jaccl-tp-sum-cqe.patch)" else "stamped" });
}

/// Prints what is left and frees the buffers. The GPU is idle (a model unloads), so no sum is in flight.
pub fn disarm() void {
    const t = timeline();
    if (!t.live.load(.acquire)) return;
    report();
    t.live.store(false, .release);
    if (lookup(SetPhaseHook, "mlx_set_tp_sum_hook")) |f| f(null);
    if (lookup(SetPeerHook, "jaccl_set_cqe_hook")) |f| f(null);
    freeBuffers(t);
}

fn freeBuffers(t: *Timeline) void {
    const alloc = std.heap.page_allocator;
    if (t.ring.slots.len > 0) alloc.free(t.ring.slots);
    if (t.taken.len > 0) alloc.free(t.taken);
    if (t.scratch.len > 0) alloc.free(t.scratch);
    t.ring = .{};
    t.taken = &.{};
    t.scratch = &.{};
}

pub fn discard() void {
    const t = timeline();
    if (t.live.load(.acquire)) _ = t.ring.take(t.taken);
}

pub fn step() void {
    const t = timeline();
    if (!t.live.load(.acquire)) return;
    t.steps += 1;
    if (t.steps >= steps_per_report) report();
}

pub fn report() void {
    const t = timeline();
    if (!t.live.load(.acquire)) return;
    t.steps = 0;
    const taken = t.ring.take(t.taken);
    var groups: [max_sizes + 1]Group = undefined;
    for (summarize(t.taken[0..taken.n], t.scratch, &groups)) |g| printGroup(t.rank, g, taken.dropped);
    if (taken.n == 0 and taken.dropped > 0) log.warn("[tp-sum] rank {d}: {d} sums dropped\n", .{ t.rank, taken.dropped });
}

fn printGroup(rank: u32, g: Group, dropped: u64) void {
    var buf: [512]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    formatLine(&w, rank, g, dropped) catch return;
    log.info("{s}\n", .{w.buffered()});
}

// ── Tests ───────────────────────────────────────────────────────────────

fn testRing(slots: []Sample) Ring {
    return .{ .slots = slots };
}

test "dist tp-sum timeline: a sum's five calls become one sample with each phase's gap" {
    var slots: [4]Sample = undefined;
    var ring = testRing(&slots);
    var rec: Recorder = .{};
    rec.phase(&ring, .wait_begin, 0, 1_000);
    rec.phase(&ring, .fence_seen, 0, 13_000);
    rec.phase(&ring, .sum_enter, 10240, 13_500);
    rec.peerDone(2, 20_500);
    rec.peerDone(1, 31_500);
    rec.phase(&ring, .sum_return, 0, 35_000);
    rec.phase(&ring, .release, 0, 35_300);
    var out: [4]Sample = undefined;
    const taken = ring.take(&out);
    try std.testing.expectEqual(@as(usize, 1), taken.n);
    const s = out[0];
    try std.testing.expectEqual(@as(u32, 10240), s.bytes);
    try std.testing.expectEqual(@as(?u64, 12_000), value(&s, .spin));
    try std.testing.expectEqual(@as(?u64, 500), value(&s, .lead));
    try std.testing.expectEqual(@as(?u64, 21_500), value(&s, .comm));
    try std.testing.expectEqual(@as(?u64, 300), value(&s, .release));
    try std.testing.expectEqual(@as(?u64, 7_000), value(&s, .cqe_first));
    try std.testing.expectEqual(@as(?u64, 18_000), value(&s, .cqe_last));
}

test "dist tp-sum timeline: a release with no sum before it, and a sum with no wait, are told apart" {
    var slots: [4]Sample = undefined;
    var ring = testRing(&slots);
    var rec: Recorder = .{};
    rec.phase(&ring, .wait_begin, 0, 10);
    rec.phase(&ring, .fence_seen, 0, 20);
    rec.phase(&ring, .release, 0, 30);
    rec.phase(&ring, .sum_enter, 64, 100);
    rec.phase(&ring, .sum_return, 0, 150);
    rec.phase(&ring, .release, 0, 160);
    var out: [4]Sample = undefined;
    const taken = ring.take(&out);
    try std.testing.expectEqual(@as(usize, 1), taken.n);
    try std.testing.expectEqual(@as(?u64, null), value(&out[0], .spin));
    try std.testing.expectEqual(@as(?u64, null), value(&out[0], .lead));
    try std.testing.expectEqual(@as(?u64, 50), value(&out[0], .comm));
    try std.testing.expectEqual(@as(?u64, null), value(&out[0], .cqe_last));
}

test "dist tp-sum timeline: a peer's completion outside a sum is ignored" {
    var slots: [4]Sample = undefined;
    var ring = testRing(&slots);
    var rec: Recorder = .{};
    rec.peerDone(1, 5);
    rec.phase(&ring, .sum_enter, 8, 10);
    rec.phase(&ring, .sum_return, 0, 20);
    rec.peerDone(1, 25);
    rec.phase(&ring, .release, 0, 30);
    var out: [4]Sample = undefined;
    _ = ring.take(&out);
    try std.testing.expectEqual(@as(?u64, null), value(&out[0], .cqe_first));
}

test "dist tp-sum timeline: the ring hands samples over once, in order, and counts the ones it lapped" {
    var slots: [4]Sample = undefined;
    var ring = testRing(&slots);
    var out: [4]Sample = undefined;
    for (0..3) |i| ring.push(.{ .bytes = @intCast(i) });
    var taken = ring.take(&out);
    try std.testing.expectEqual(@as(usize, 3), taken.n);
    for (out[0..3], 0..) |s, i| try std.testing.expectEqual(@as(u32, @intCast(i)), s.bytes);
    try std.testing.expectEqual(@as(usize, 0), ring.take(&out).n);
    for (3..12) |i| ring.push(.{ .bytes = @intCast(i) });
    taken = ring.take(&out);
    try std.testing.expectEqual(@as(usize, 3), taken.n);
    try std.testing.expectEqual(@as(u64, 6), taken.dropped);
    for (out[0..3], 9..) |s, i| try std.testing.expectEqual(@as(u32, @intCast(i)), s.bytes);
}

test "dist tp-sum timeline: overwritten counts the samples a producer a lap ahead may have begun to write over" {
    try std.testing.expectEqual(@as(u64, 0), overwritten(10, 13, 4));
    try std.testing.expectEqual(@as(u64, 1), overwritten(10, 14, 4));
    try std.testing.expectEqual(@as(u64, 3), overwritten(10, 16, 4));
}

test "dist tp-sum timeline: a producer thread lapping a small ring never hands over a torn or out-of-order sample" {
    var slots: [64]Sample = undefined;
    var ring = testRing(&slots);
    const total: u64 = 200_000;
    const producer = try std.Thread.spawn(.{}, struct {
        fn run(r: *Ring) void {
            for (0..total) |i| r.push(.{ .bytes = @truncate(i), .wait_begin = i, .release = i });
        }
    }.run, .{&ring});
    var out: [64]Sample = undefined;
    var seen: u64 = 0;
    var last: ?u64 = null;
    while (seen < total) {
        const taken = ring.take(&out);
        seen += taken.n + taken.dropped;
        for (out[0..taken.n]) |s| {
            try std.testing.expectEqual(s.wait_begin, s.release);
            try std.testing.expectEqual(@as(u32, @truncate(s.wait_begin)), s.bytes);
            if (last) |l| try std.testing.expect(s.wait_begin > l);
            last = s.wait_begin;
        }
    }
    producer.join();
    try std.testing.expectEqual(total, seen);
}

test "dist tp-sum timeline: nearest-rank quantiles" {
    const v = [_]u64{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    try std.testing.expectEqual(Quantiles{ .p50 = 5, .p90 = 9, .max = 10 }, quantiles(&v));
    try std.testing.expectEqual(Quantiles{ .p50 = 7, .p90 = 7, .max = 7 }, quantiles(v[6..7]));
}

test "dist tp-sum timeline: sizes group most sums first, the ones past the cap share a bucket, and absent metrics stay null" {
    var samples: [14]Sample = undefined;
    for (&samples, 0..) |*s, i| {
        s.* = .{ .bytes = if (i < 6) 10240 else @intCast(100 + i), .enter = 10, .leave = 10 + 1000 * (@as(u64, i) + 1), .release = 10 + 1000 * (@as(u64, i) + 1) + 50 };
    }
    var scratch: [14]u64 = undefined;
    var groups: [max_sizes + 1]Group = undefined;
    const got = summarize(&samples, &scratch, &groups);
    try std.testing.expectEqual(@as(usize, max_sizes + 1), got.len);
    try std.testing.expectEqual(@as(u32, 10240), got[0].bytes);
    try std.testing.expectEqual(@as(u32, 6), got[0].n);
    try std.testing.expectEqual(Quantiles{ .p50 = 3000, .p90 = 6000, .max = 6000 }, got[0].q[@backingInt(Metric.comm)].?);
    try std.testing.expectEqual(@as(?Quantiles, null), got[0].q[@backingInt(Metric.spin)]);
    var other_n: u32 = 0;
    for (got) |g| {
        if (g.bytes == 0) other_n = g.n;
    }
    try std.testing.expectEqual(@as(u32, 3), other_n);
}

test "dist tp-sum timeline: the line shows each metric as p50/p90/max in microseconds" {
    var g: Group = .{ .bytes = 10240, .n = 20480 };
    g.q[@backingInt(Metric.comm)] = .{ .p50 = 21_000, .p90 = 30_500, .max = 301_000 };
    g.q[@backingInt(Metric.lead)] = .{ .p50 = 400, .p90 = 800, .max = 11_000 };
    var buf: [512]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    try formatLine(&w, 2, g, 0);
    try std.testing.expectEqualStrings("[tp-sum] rank 2 10240 B n=20480 us spin - a>b 0.4/0.8/11.0 b>c 21.0/30.5/301.0 c>d - cqe-first - cqe-last - dropped 0", w.buffered());
}

test "dist tp-sum timeline: the diag token arms it and a name no MLX exports is not found" {
    const saved = env.override(.MLX_SERVE_DIAG, "tp-sum-timeline");
    defer saved.restore();
    try std.testing.expect(wanted());
    try std.testing.expect(lookup(SetPhaseHook, "mlx_serve_no_such_hook") == null);
}

const counted = struct {
    var phases: [256]u32 = undefined;
    var n: usize = 0;

    fn hook(p: u32, _: u64, _: u64) callconv(.c) void {
        if (n < phases.len) phases[n] = p;
        n += 1;
    }
};

fn handOffChain(links: usize) !void {
    const gpu = mlx.gpuStream();
    const cpu = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(cpu);
    var zero = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(zero);
    var cur = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_zeros(&zero, &[_]c_int{1}, 1, .bfloat16, gpu));
    try mlx.check(mlx.mlx_ones(&cur, &[_]c_int{ 1, 5120 }, 2, .bfloat16, gpu));
    try mlx.check(mlx.mlx_array_eval(zero));
    for (0..links) |_| {
        var on_gpu = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(on_gpu);
        var on_cpu = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_add(&on_gpu, cur, zero, gpu));
        try mlx.check(mlx.mlx_add(&on_cpu, on_gpu, zero, cpu));
        _ = mlx.mlx_array_free(cur);
        cur = on_cpu;
    }
    defer _ = mlx.mlx_array_free(cur);
    try mlx.check(mlx.mlx_array_eval(cur));
}

fn linkUs(io: std.Io, links: usize) !f64 {
    var sw = io_util.Stopwatch.init(io);
    try handOffChain(links);
    return @as(f64, @floatFromInt(sw.read())) / 1000.0 / @as(f64, @floatFromInt(links));
}

test "dist tp-sum timeline: on a patched MLX a GPU-fed CPU op fires the wait, fence and release calls in order" {
    const set_phase = lookup(SetPhaseHook, "mlx_set_tp_sum_hook") orelse return error.SkipZigTest;
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const fast = env.override(.MLX_METAL_FAST_SYNCH, "1");
    defer fast.restore();
    const links = 20;
    try handOffChain(links);
    counted.n = 0;
    set_phase(&counted.hook);
    defer set_phase(null);
    try handOffChain(links);
    set_phase(null);
    try std.testing.expectEqual(@as(usize, 3 * links - 1), counted.n);
    const want = [_]Phase{ .wait_begin, .fence_seen, .release };
    for (counted.phases[0..counted.n], 0..) |got, k| try std.testing.expectEqual(@backingInt(want[k % 3]), got);
}

test "dist tp-sum timeline: the hooks' cost on a GPU-CPU-GPU hand-off, armed and not (patched MLX)" {
    const set_phase = lookup(SetPhaseHook, "mlx_set_tp_sum_hook") orelse return error.SkipZigTest;
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const fast = env.override(.MLX_METAL_FAST_SYNCH, "1");
    defer fast.restore();
    const diag = env.override(.MLX_SERVE_DIAG, "tp-sum-timeline");
    defer diag.restore();
    arm(0);
    defer disarm();
    try std.testing.expect(timeline().live.load(.acquire));
    const io = std.testing.io;
    var off: f64 = std.math.inf(f64);
    var on: f64 = std.math.inf(f64);
    for (0..8) |_| {
        set_phase(null);
        off = @min(off, try linkUs(io, 200));
        set_phase(&onPhase);
        on = @min(on, try linkUs(io, 200));
    }
    std.debug.print("[tp-sum] hand-off link: {d:.2} us unarmed, {d:.2} us armed (+{d:.2} us over the three calls)\n", .{ off, on, on - off });
}
