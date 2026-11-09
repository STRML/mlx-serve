//! The fused decode segments' phase boundary on the device: generation-stamped arrival counters, the bounded wait, and
//! three ways to join a phase to the next inside one launch (last arriver, ordered tickets, resident spin). The probe
//! kernel here runs a six-phase chain of known sums under each, meters a boundary's cost, and finds the grid a
//! spinning launch keeps co-resident.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const env = @import("../../../env.zig");
const log = @import("../../../sdk/util/log.zig");
const kernelSource = @import("../../../mlx/kernels.zig").source;
const dsv41 = @import("../deepseek_v41.zig");
const seg = @import("seg.zig");
const seg_bounds = @import("seg_bounds.zig");

const A = mlx.mlx_array;

const HEADER = kernelSource("plugins/deepseek_v41/kernels/seg_proto_header.metal");
const PROBE = kernelSource("plugins/deepseek_v41/kernels/seg_proto_probe.metal");
const INS = [_][*:0]const u8{ "cnt", "buf", "status", "gen", "arg", "dep" };
const OUTS = [_][*:0]const u8{"sink"};

const salt: usize = 2654435;
const step: usize = 1000003;
const resident_launches = 50;
const max_probe_groups = 8192;

/// How a probe launch joins its phases.
pub const Arm = enum(usize) { ticket = 0, spin = 1, last = 2, dispatch = 3, resident = 4, idle = 5 };

/// How the probe's phases move data across groups: relaxed atomics; ordinary loads and stores; `coherent(device)` loads
/// and stores; ordinary stores read back by atomic loads; ordinary stores read back by `seg_read_peer_output`;
/// `seg_write_peer_output` read back by `seg_read_peer_output`, the pair a fused kernel uses.
pub const Access = enum(usize) { atomic = 0, plain = 1, coherent = 2, atomic_load = 3, peer_read = 4, peer = 5 };

/// The arms with an ordinary store or load on the cross-group path: on the two-die M5 Ultra a plain load reads a stale
/// line, and a plain store that fills part of a line is not visible to a coherent load before the launch ends.
fn reproducer(access: Access) bool {
    return access == .plain or access == .atomic_load or access == .peer_read;
}

pub const Launch = struct {
    arm: Arm,
    groups: u32,
    threads: u32 = seg.spin_threads,
    phases: u32 = 6,
    tgmem: u32 = seg.spin_tgmem,
    polls: u32 = 0,
    serial_items: u32 = 0,
    access: Access = .atomic,
    layout: Layout = .grouped,

    fn launched(self: Launch) u32 {
        return if (self.arm == .ticket) self.phases * self.groups else self.groups;
    }

    fn words(self: Launch) usize {
        return self.phases * self.groups * self.threads;
    }

    fn spot(self: Launch, p: usize, g: usize, t: usize) usize {
        const n = self.groups * self.threads;
        return p * n + switch (self.layout) {
            .grouped => g * self.threads + t,
            .interleaved => t * self.groups + g,
        };
    }
};

/// Where a phase's words sit: a group's contiguous, so each simdgroup stores whole 128-byte lines; or interleaved across
/// groups, so a store fills one word of a line, as a kernel's per-group partials do.
pub const Layout = enum(usize) { grouped = 0, interleaved = 1 };

pub const Result = struct { us: i64, status: u32 };

const alloc = std.heap.c_allocator;

fn u32Array(words: []const u32) !A {
    const a = mlx.mlx_array_new_data(words.ptr, &[_]c_int{@intCast(words.len)}, 1, .uint32);
    if (a.ctx == null) return error.MlxArrayNew;
    return a;
}

const Rig = struct {
    kernel: mlx.mlx_fast_metal_kernel,
    stream: mlx.mlx_stream,
    state: seg.State,
    buf: A,

    fn init(slots: usize, buf_words: usize) !Rig {
        const kernel = try dsv41.buildKernel("dsv41_seg_proto_probe", &INS, &OUTS, PROBE, HEADER);
        errdefer _ = mlx.mlx_fast_metal_kernel_free(kernel);
        var state = try seg.State.init(slots, .ticket);
        errdefer state.deinit();
        const zeros = try alloc.alloc(u32, buf_words);
        defer alloc.free(zeros);
        @memset(zeros, 0);
        return .{ .kernel = kernel, .stream = mlx.gpuStream(), .state = state, .buf = try u32Array(zeros) };
    }

    fn deinit(self: *Rig) void {
        _ = mlx.mlx_array_free(self.buf);
        self.state.deinit();
        _ = mlx.mlx_fast_metal_kernel_free(self.kernel);
        _ = mlx.mlx_stream_free(self.stream);
    }

    fn scope(self: *Rig) dsv41.Scope {
        return dsv41.Scope.init(alloc, self.stream);
    }

    fn words(self: *const Rig) []const u32 {
        return mlx.mlx_array_data_uint32(self.buf).?[0..mlx.mlx_array_size(self.buf)];
    }

    fn launch(self: *Rig, sc: *dsv41.Scope, l: Launch, gen: A, phase: u32, dep: A) !A {
        const cfg = mlx.mlx_fast_metal_kernel_config_new();
        defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{2}, 1, .uint32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, @intCast(l.launched() * l.threads), 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, @intCast(l.threads), 1, 1));
        const targs = [_]struct { name: [*:0]const u8, v: usize }{
            .{ .name = "PROTO", .v = @intFromEnum(l.arm) },
            .{ .name = "GROUPS", .v = l.groups },
            .{ .name = "THREADS", .v = l.threads },
            .{ .name = "PHASES", .v = l.phases },
            .{ .name = "POLLS", .v = l.polls },
            .{ .name = "ITEMS", .v = l.serial_items },
            .{ .name = "ACCESS", .v = @intFromEnum(l.access) },
            .{ .name = "LAYOUT", .v = @intFromEnum(l.layout) },
            .{ .name = "TGMEM", .v = l.tgmem },
            .{ .name = "SALT", .v = salt },
            .{ .name = "STEP", .v = step },
        };
        for (targs) |t| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, t.name, @intCast(t.v)));
        const arg = try sc.keep(try u32Array(&.{phase}));
        const ins = [_]A{ self.state.counters.buf, self.buf, self.state.status, gen, arg, dep };
        const iv = mlx.mlx_vector_array_new_data(&ins, ins.len);
        defer _ = mlx.mlx_vector_array_free(iv);
        var ov = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(ov);
        try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, self.kernel, iv, cfg, sc.s));
        var sink = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(&sink, ov, 0));
        return sc.keep(sink);
    }

    fn once(self: *Rig, l: Launch) !Result {
        var sc = self.scope();
        defer sc.deinit();
        const gen = try self.state.counters.next(&sc);
        const dep = try sc.keep(try u32Array(&.{0}));
        const t0 = seg_bounds.nowUs();
        var sink = try self.launch(&sc, l, gen, 0, dep);
        if (l.arm == .dispatch) for (1..l.phases) |p| {
            sink = try self.launch(&sc, l, gen, @intCast(p), sink);
        };
        try mlx.check(mlx.mlx_array_eval(sink));
        return .{ .us = seg_bounds.nowUs() - t0, .status = self.state.code() };
    }

    fn chainUs(self: *Rig, l: Launch, count: u32) !Result {
        var sc = self.scope();
        defer sc.deinit();
        var sink = try sc.keep(try u32Array(&.{0}));
        for (0..count) |k| {
            const gen = try self.state.counters.next(&sc);
            sink = try self.launch(&sc, l, gen, @intCast(k % l.phases), sink);
        }
        const t0 = seg_bounds.nowUs();
        try mlx.check(mlx.mlx_array_eval(sink));
        return .{ .us = seg_bounds.nowUs() - t0, .status = self.state.code() };
    }
};

fn expectedBuffer(gpa: std.mem.Allocator, l: Launch, gen: u32) ![]u32 {
    const n = l.groups * l.threads;
    const out = try gpa.alloc(u32, l.phases * n);
    const g_salt = gen *% @as(u32, salt);
    for (out[0..n], 0..) |*v, gid| v.* = @as(u32, @intCast(gid)) +% g_salt;
    for (1..l.phases) |p| for (0..l.groups) |g| for (0..l.threads) |t| {
        const from = out[(p - 1) * n + ((g + p) % l.groups) * l.threads + t];
        out[p * n + g * l.threads + t] = from +% @as(u32, @intCast(p * step)) +% g_salt;
    };
    return out;
}

fn debugDump(gpa: std.mem.Allocator, rig: *const Rig, l: Launch) void {
    const want = expectedBuffer(gpa, l, rig.state.counters.gen) catch return;
    defer gpa.free(want);
    const n = l.groups * l.threads;
    var shown: usize = 0;
    const all = rig.words();
    for (want, 0..) |w, i| {
        const got = all[l.spot(i / n, i % n / l.threads, i % l.threads)];
        if (w == got or shown >= 4) continue;
        shown += 1;
        std.debug.print("  phase {d} g {d} t {d}: got {d} want {d}\n", .{ i / n, i % n / l.threads, i % l.threads, got, w });
    }
    std.debug.print("  counters {any} status {d} gen {d}\n", .{ mlx.mlx_array_data_uint32(rig.state.counters.buf).?[0..8], rig.state.code(), rig.state.counters.gen });
}

fn bufferMismatches(gpa: std.mem.Allocator, rig: *const Rig, l: Launch, gen: u32) !usize {
    const want = try expectedBuffer(gpa, l, gen);
    defer gpa.free(want);
    const n = l.groups * l.threads;
    const all = rig.words();
    var bad: usize = 0;
    for (want, 0..) |w, i| bad += @intFromBool(w != all[l.spot(i / n, i % n / l.threads, i % l.threads)]);
    return bad;
}

fn median(vals: []f64) f64 {
    std.mem.sort(f64, vals, {}, std.sort.asc(f64));
    return vals[vals.len / 2];
}

/// The largest grid of `tg_threads`-thread groups holding `tgmem_bytes` of threadgroup memory that stays co-resident
/// across 50 launches, found by growing and then bisecting; 0 when even two groups do not. Every wait is bounded, so a
/// grid past residency ends in a status, not a hang.
pub fn probeResidency(tg_threads: u32, tgmem_bytes: u32) !u32 {
    var rig = try Rig.init(1, seg.slot_words);
    defer rig.deinit();
    const at = Launch{ .arm = .resident, .groups = 2, .threads = tg_threads, .tgmem = tgmem_bytes };
    var good: u32 = 0;
    var bad: u32 = 2;
    while (bad <= max_probe_groups) : (bad *= 2) {
        if (!try residentAt(&rig, at, bad)) break;
        good = bad;
    }
    if (good == 0) return 0;
    while (bad - good > 1) {
        const mid = good + (bad - good) / 2;
        if (try residentAt(&rig, at, mid)) good = mid else bad = mid;
    }
    return good;
}

fn residentAt(rig: *Rig, base: Launch, groups: u32) !bool {
    var l = base;
    l.groups = groups;
    for (0..resident_launches) |_| {
        if ((try rig.once(l)).status == 0) continue;
        try rig.state.clear();
        return false;
    }
    return true;
}

fn gpuTest() bool {
    return mlx.metalKernelsAvailable() and !mlx.noGpuBackend() and !mlx.virtualGpu();
}

const launches_per_arm = 1000;

const Outcome = enum { ok, timed_out, bad_status, wrong_words, wrong_winners };

const ChainResult = struct { outcome: Outcome = .ok, launch: usize = 0, count: usize = 0, total: usize = 0, status: u32 = 0 };

fn expectedWinners(arm: Arm, boundary: usize) u32 {
    return switch (arm) {
        .dispatch => 0,
        .last => @intFromBool(boundary == 0),
        else => 1,
    };
}

fn runChain(gpa: std.mem.Allocator, arm: Arm, access: Access, groups: u32, layout: Layout) !ChainResult {
    const l = Launch{ .arm = arm, .groups = groups, .access = access, .layout = layout };
    var rig = try Rig.init(1, l.words());
    defer rig.deinit();
    const total = l.phases * l.groups * l.threads;
    const winners = @constCast(mlx.mlx_array_data_uint32(rig.state.status).?)[1..mlx.mlx_array_size(rig.state.status)];
    for (0..launches_per_arm) |k| {
        @memset(winners, 0);
        const r = try rig.once(l);
        if (r.status != 0) return .{ .outcome = if (arm == .spin) .timed_out else .bad_status, .launch = k, .status = r.status };
        const bad = try bufferMismatches(gpa, &rig, l, rig.state.counters.gen);
        if (bad != 0) {
            debugDump(gpa, &rig, l);
            return .{ .outcome = .wrong_words, .launch = k, .count = bad, .total = total };
        }
        for (0..l.phases - 1) |boundary| if (winners[boundary] != expectedWinners(arm, boundary)) return .{ .outcome = .wrong_winners, .launch = k, .count = winners[boundary] };
    }
    return .{};
}

const spin_attempts = 4;

const SpinResult = struct { chain: ChainResult, retries: u32 };

fn runSpinChain(gpa: std.mem.Allocator, access: Access, layout: Layout, probed: u32) !SpinResult {
    var floor = probed;
    var result: SpinResult = .{ .chain = .{}, .retries = 0 };
    for (0..spin_attempts) |retries| {
        result = .{ .chain = try runChain(gpa, .spin, access, seg.spinGrid(floor), layout), .retries = @intCast(retries) };
        if (result.chain.outcome != .timed_out) return result;
        floor = @min(floor, try probeResidency(seg.spin_threads, seg.spin_tgmem));
        if (floor < 2) return result;
    }
    return result;
}

fn reportChain(arm: Arm, access: Access, layout: Layout, r: ChainResult, retries: u32) void {
    const asserted = if (reproducer(access)) "diagnostic, expected to fail on a multi-die GPU" else "asserted";
    const a = @tagName(arm);
    const c = @tagName(access);
    const l = @tagName(layout);
    switch (r.outcome) {
        .ok => std.debug.print("[seg] {s} access={s} layout={s}: ok over {d} launches ({s}, spin retries {d})\n", .{ a, c, l, launches_per_arm, asserted, retries }),
        .wrong_words => std.debug.print("[seg] {s} access={s} layout={s}: FAIL launch {d}: {d} of {d} words differ ({s})\n", .{ a, c, l, r.launch, r.count, r.total, asserted }),
        .wrong_winners => std.debug.print("[seg] {s} access={s} layout={s}: FAIL launch {d}: {d} arrival winners at a boundary ({s})\n", .{ a, c, l, r.launch, r.count, asserted }),
        .timed_out, .bad_status => std.debug.print("[seg] {s} access={s} layout={s}: FAIL launch {d}: status {d} ({s}, spin retries {d})\n", .{ a, c, l, r.launch, r.status, asserted, retries }),
    }
}

test "dsv41 fused seg proto: last-arriver, ordered tickets and resident spin give every phase all of the previous phase's writes, through atomic, coherent and peer accesses, with a group's words contiguous or interleaved across groups (plain stores or loads are the multi-die reproducers, reported not asserted)" {
    if (!gpuTest()) return error.SkipZigTest;
    const gpa = std.testing.allocator;
    const resident = @min(try probeResidency(seg.spin_threads, seg.spin_tgmem), try probeResidency(seg.spin_threads, seg.spin_tgmem));
    if (resident < 2) return error.SkipZigTest;
    std.debug.print("[seg] resident tg={d} tgmem={d} groups={d} (least of two probes)\n", .{ seg.spin_threads, seg.spin_tgmem, resident });
    var failures: u32 = 0;
    const base = try runChain(gpa, .dispatch, .atomic, resident, .grouped);
    reportChain(.dispatch, .atomic, .grouped, base, 0);
    failures += @intFromBool(base.outcome != .ok);
    for ([_]Layout{ .grouped, .interleaved }) |layout| for (std.enums.values(Access)) |access| for ([_]Arm{ .last, .ticket, .spin }) |arm| {
        const run: SpinResult = if (arm == .spin) try runSpinChain(gpa, access, layout, resident) else .{ .chain = try runChain(gpa, arm, access, resident, layout), .retries = 0 };
        reportChain(arm, access, layout, run.chain, run.retries);
        if (reproducer(access)) continue;
        failures += @intFromBool(run.chain.outcome != .ok or run.retries > 1);
    };
    std.debug.print("[seg] {d} asserted arms failed\n", .{failures});
    try std.testing.expectEqual(@as(u32, 0), failures);
}

const quiet_timeout_ms = [2]f64{ 3, 20 };
const load_bound_us = 50_000;

const Load = struct {
    stop: std.atomic.Value(bool) = .init(false),
    reads: std.atomic.Value(u32) = .init(0),
    failed: std.atomic.Value(bool) = .init(false),

    fn run(self: *Load, words: usize) void {
        self.stream(words) catch self.failed.store(true, .release);
    }

    fn stream(self: *Load, words: usize) !void {
        const dev = mlx.mlx_device_new_type(.gpu, 0);
        defer _ = mlx.mlx_device_free(dev);
        const s = mlx.mlx_stream_new_device(dev);
        defer _ = mlx.mlx_stream_free(s);
        var big = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(big);
        try mlx.check(mlx.mlx_ones(&big, &[_]c_int{@intCast(words)}, 1, .float32, s));
        try mlx.check(mlx.mlx_array_eval(big));
        while (!self.stop.load(.acquire)) {
            var total = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(total);
            try mlx.check(mlx.mlx_sum(&total, big, false, s));
            try mlx.check(mlx.mlx_array_eval(total));
            _ = self.reads.fetchAdd(1, .release);
        }
    }
};

test "dsv41 fused seg proto: a grid twice the probed residency under a concurrent GPU load ends with every status ok or spin_timeout within the bound (needs a quiet GPU)" {
    if (!gpuTest()) return error.SkipZigTest;
    const resident = try probeResidency(seg.spin_threads, seg.spin_tgmem);
    if (resident < 2) return error.SkipZigTest;
    const poll_ms = try timeoutMs(0);
    if (poll_ms < quiet_timeout_ms[0] or poll_ms > quiet_timeout_ms[1]) {
        std.debug.print("[seg] load test skipped: the default timeout measures {d:.2} ms, outside {d}-{d} ms, so the GPU is not quiet\n", .{ poll_ms, quiet_timeout_ms[0], quiet_timeout_ms[1] });
        return error.SkipZigTest;
    }
    const l = Launch{ .arm = .spin, .groups = 2 * resident };
    var rig = try Rig.init(1, l.phases * l.groups * l.threads);
    defer rig.deinit();
    _ = try spinTimes(&rig, l, 1);
    const idle = try spinTimes(&rig, l, 20);
    var load: Load = .{};
    const reader = try std.Thread.spawn(.{}, Load.run, .{ &load, @as(usize, 512) * 1024 * 1024 });
    while (load.reads.load(.acquire) == 0 and !load.failed.load(.acquire)) std.Thread.yield() catch {};
    const busy = try spinTimes(&rig, l, 40);
    load.stop.store(true, .release);
    reader.join();
    try std.testing.expect(!load.failed.load(.acquire));
    std.debug.print("[seg] load groups={d} (2x {d}): idle worst={d}us timeouts={d}/20, under load worst={d}us timeouts={d}/40 ({d} background reads of 2 GB)\n", .{ l.groups, resident, idle.worst, idle.timeouts, busy.worst, busy.timeouts, load.reads.load(.acquire) });
    if (busy.worst >= load_bound_us) {
        std.debug.print("[seg] a timeout under load took {d} us against the {d} us bound; the bound is a poll count (default timeout {d:.2} ms idle) and needs a quiet GPU\n", .{ busy.worst, load_bound_us, poll_ms });
        return error.TestUnexpectedResult;
    }
}

const Times = struct { worst: i64, timeouts: u32 };

fn spinTimes(rig: *Rig, l: Launch, launches: u32) !Times {
    var t: Times = .{ .worst = 0, .timeouts = 0 };
    for (0..launches) |_| {
        const r = try rig.once(l);
        t.worst = @max(t.worst, r.us);
        try std.testing.expect(r.status == 0 or r.status == @intFromEnum(seg.Status.spin_timeout));
        if (r.status == 0) continue;
        t.timeouts += 1;
        try rig.state.clear();
    }
    return t;
}

fn dirtyBoundaries(rig: *Rig, l: Launch, gen: u32, dirty_gen: u32) void {
    const words = @constCast(mlx.mlx_array_data_uint32(rig.state.counters.buf).?);
    words[0] = (gen - 1) *% l.phases *% l.groups;
    for (words[1..l.phases]) |*w| w.* = (dirty_gen << 16) | (l.groups / 2);
}

const DirtyCase = struct { gen: u32, dirty_gen: u32 };

test "dsv41 fused seg proto: generation stamps make a dirty counter unreachable, an older or a newer one, up to generation 0xffff" {
    if (!gpuTest()) return error.SkipZigTest;
    const gpa = std.testing.allocator;
    const groups: u32 = 24;
    const cases = [_]DirtyCase{ .{ .gen = 5, .dirty_gen = 4 }, .{ .gen = 5, .dirty_gen = 9 }, .{ .gen = seg.max_gen, .dirty_gen = seg.max_gen - 1 } };
    for (cases) |c| for ([_]Arm{ .last, .ticket, .spin }) |arm| {
        const l = Launch{ .arm = arm, .groups = groups };
        var rig = try Rig.init(1, l.phases * l.groups * l.threads);
        defer rig.deinit();
        rig.state.counters.gen = c.gen - 1;
        dirtyBoundaries(&rig, l, c.gen, c.dirty_gen);
        const r = try rig.once(l);
        try std.testing.expectEqual(c.gen, rig.state.counters.gen);
        try std.testing.expectEqual(@as(u32, 0), r.status);
        try std.testing.expectEqual(@as(usize, 0), try bufferMismatches(gpa, &rig, l, c.gen));
        try rig.state.counters.reset();
        try std.testing.expectEqual(@as(u32, 0), rig.state.counters.gen);
        for (mlx.mlx_array_data_uint32(rig.state.counters.buf).?[0..seg.slot_words]) |w| try std.testing.expectEqual(@as(u32, 0), w);
        const again = try rig.once(l);
        try std.testing.expectEqual(@as(u32, 1), rig.state.counters.gen);
        try std.testing.expectEqual(@as(u32, 0), again.status);
        try std.testing.expectEqual(@as(usize, 0), try bufferMismatches(gpa, &rig, l, 1));
    };
}

test "dsv41 fused seg proto: a dirty ticket word is refused with a status and the generation wraps to a cleared slot" {
    if (!gpuTest()) return error.SkipZigTest;
    const l = Launch{ .arm = .ticket, .groups = 8 };
    var rig = try Rig.init(1, l.phases * l.groups * l.threads);
    defer rig.deinit();
    @constCast(mlx.mlx_array_data_uint32(rig.state.counters.buf).?)[0] = 3;
    rig.state.counters.gen = 0;
    const refused = try rig.once(l);
    try std.testing.expectEqual(@as(u32, @intFromEnum(seg.Status.spin_timeout)), refused.status);
    try rig.state.clear();
    rig.state.counters.gen = seg.max_gen;
    var sc = rig.scope();
    defer sc.deinit();
    _ = try rig.state.counters.next(&sc);
    try std.testing.expectEqual(@as(u32, 1), rig.state.counters.gen);
}

test "dsv41 fused seg proto: afterStep is a no-op without state or with an ok status and latches the segments off on any other" {
    if (!gpuTest()) return error.SkipZigTest;
    const gpa = std.testing.allocator;
    const m = try gpa.create(dsv41.Dsv41Model);
    defer gpa.destroy(m);
    m.seg = null;
    m.fused_seg = .ffn_seg;
    try seg.afterStep(m);
    try std.testing.expectEqual(seg.Level.ffn_seg, m.fused_seg);
    var state = try seg.State.init(4, .ticket);
    defer state.deinit();
    m.seg = &state;
    try seg.afterStep(m);
    try std.testing.expectEqual(seg.Level.ffn_seg, m.fused_seg);
    const sinks = log.sinks_shared.get();
    const stderr_was = sinks.stderr_enabled;
    sinks.stderr_enabled = false;
    defer sinks.stderr_enabled = stderr_was;
    for ([_]seg.Status{ .spin_timeout, .sum_timeout, .injected_abort }) |bad| {
        m.fused_seg = .ffn_seg;
        @constCast(mlx.mlx_array_data_uint32(state.status).?)[0] = @intFromEnum(bad);
        state.counters.gen = 7;
        @constCast(mlx.mlx_array_data_uint32(state.counters.buf).?)[3] = 99;
        try std.testing.expectError(error.Dsv41FusedSegTimeout, seg.afterStep(m));
        try std.testing.expectEqual(seg.Level.off, m.fused_seg);
        try std.testing.expectEqual(@as(u32, 0), state.code());
        try std.testing.expectEqual(@as(u32, 0), state.counters.gen);
        try std.testing.expectEqual(@as(u32, 0), mlx.mlx_array_data_uint32(state.counters.buf).?[3]);
        try seg.afterStep(m);
    }
}

fn probeOn() bool {
    return env.get(.DSV41_SEG_PROBE) != null and gpuTest();
}

const sweep_steps = 8;
const sweep_slots = sweep_steps + 4;

fn sweepGroups(out: *[sweep_slots]u32, resident: u32) []const u32 {
    var n: usize = 0;
    if (env.span(.DSV41_SEG_PROBE_GROUPS)) |range| {
        const dash = std.mem.indexOfScalar(u8, range, '-') orelse return out[0..0];
        const lo = std.fmt.parseInt(u32, range[0..dash], 10) catch return out[0..0];
        const hi = std.fmt.parseInt(u32, range[dash + 1 ..], 10) catch return out[0..0];
        for (0..sweep_steps) |i| {
            out[n] = lo + (hi - lo) * @as(u32, @intCast(i)) / (sweep_steps - 1);
            n += 1;
        }
        return out[0..n];
    }
    for ([_]u32{ 1, 8, resident / 4, resident / 2, resident * 3 / 4, resident, resident * 5 / 4, resident * 3 / 2, resident * 2 }) |g| {
        if (g == 0 or (n > 0 and out[n - 1] == g)) continue;
        out[n] = g;
        n += 1;
    }
    return out[0..n];
}

const Boundary = struct { us: f64, timeouts: u32 };

fn boundaryUs(arm: Arm, access: Access, groups: u32) !Boundary {
    const per_trial: u32 = 200;
    const hi_phases: u32 = if (arm == .last) 2 else 11;
    const reps: u32 = switch (arm) {
        .last => per_trial,
        .dispatch => per_trial + 1,
        else => per_trial / (hi_phases - 1),
    };
    const hi = Launch{ .arm = arm, .groups = groups, .phases = if (arm == .dispatch) 2 else hi_phases, .serial_items = 1, .access = access };
    var lo = hi;
    lo.serial_items = 1;
    lo.phases = if (arm == .dispatch) 2 else 1;
    const lo_reps: u32 = if (arm == .dispatch) 1 else reps;
    var rig = try Rig.init(1, hi.phases * hi.groups * hi.threads);
    defer rig.deinit();
    var trials: [7]f64 = undefined;
    var timeouts: u32 = 0;
    _ = try rig.chainUs(hi, reps);
    for (&trials) |*t| {
        try rig.state.clear();
        const base = try rig.chainUs(lo, lo_reps);
        try rig.state.clear();
        const full = try rig.chainUs(hi, reps);
        if (base.status != 0 or full.status != 0) timeouts += 1;
        t.* = @as(f64, @floatFromInt(full.us - base.us)) / per_trial;
    }
    return .{ .us = median(&trials), .timeouts = timeouts };
}

fn timeoutMs(polls: u32) !f64 {
    const l = Launch{ .arm = .idle, .groups = 1, .threads = 32, .tgmem = 0, .polls = polls };
    var rig = try Rig.init(1, seg.slot_words);
    defer rig.deinit();
    var ms: [7]f64 = undefined;
    for (&ms) |*t| {
        try rig.state.clear();
        const r = try rig.once(l);
        t.* = @as(f64, @floatFromInt(r.us)) / 1000;
    }
    return median(&ms);
}

test "dsv41 fused seg proto ubench: phase boundary cost by protocol, grid and die span (DSV41_SEG_PROBE)" {
    if (!probeOn()) return error.SkipZigTest;
    const resident = try probeResidency(seg.spin_threads, seg.spin_tgmem);
    std.debug.print("[seg] resident tg={d} tgmem={d} groups={d}\n", .{ seg.spin_threads, seg.spin_tgmem, resident });
    for ([_]u32{ 10_000, 20_000, 40_000, 80_000 }) |polls| std.debug.print("[seg] timeout polls={d} ms={d:.2}\n", .{ polls, try timeoutMs(polls) });
    std.debug.print("[seg] timeout default ms={d:.2}\n", .{try timeoutMs(0)});
    var groups: [sweep_slots]u32 = undefined;
    for (sweepGroups(&groups, resident)) |g| for ([_]Arm{ .dispatch, .last, .ticket, .spin }) |arm| for (std.enums.values(Access)) |access| {
        if (arm == .dispatch and access != .atomic) continue;
        const b = try boundaryUs(arm, access, g);
        std.debug.print("[seg] boundary {s}{s}{s} groups={d} us={d:.3} timeouts={d}/7\n", .{ @tagName(arm), if (access == .atomic) "" else "-", if (access == .atomic) "" else @tagName(access), g, b.us, b.timeouts });
    };
}
