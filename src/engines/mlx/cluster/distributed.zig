//! Multi-Mac inference over MLX's distributed group (JACCL = RDMA over
//! Thunderbolt 5, macOS 26.2+). Rank 0 is the server; every other rank is a
//! worker that executes what rank 0 tells it, in lockstep.
const std = @import("std");
const env = @import("../../../env.zig");
const builtin = @import("builtin");
const mlx = @import("../../../mlx/ffi.zig");
const crt = @import("../../../sdk/util/crt.zig");
const io_util = @import("../../../sdk/util/io_util.zig");
const log = @import("../../../sdk/util/log.zig");
const model_mod = @import("../../../sdk/config/model.zig");
const status = @import("../../../host/status.zig");
const Qwen4Trunk = @import("../../../plugins/qwen4_exp/trunk.zig").Trunk;
const ForwardCtx = @import("../forward_ctx.zig").ForwardCtx;
const KVCache = @import("../kv/cache.zig").KVCache;
const SSMCacheEntry = @import("../ssm/entry.zig").SSMCacheEntry;
const ssmFreeSpecCapture = @import("../ssm/entry.zig").ssmFreeSpecCapture;
const ssmFreeQsaState = @import("../../../plugins/qwen4_exp/qsa/history.zig").ssmFreeQsaState;
const h3_split = @import("../../../plugins/minimax_h3/cluster.zig");
const wire = @import("../../../sdk/util/wire.zig");
const lan = @import("../../../host/lan/lan.zig");
const dsv41_mod = @import("../../../plugins/deepseek_v41/deepseek_v41.zig");
const dist_watchdog = @import("watchdog.zig");

pub const Mode = enum {
    /// Each rank owns a contiguous range of layers; the hidden stream hops at the seams.
    pipeline,
    /// Every rank runs every layer on its slice of each MoE expert; one all_sum per MoE layer.
    tensor,
    /// A video DiT whole on every rank, each step's rows split (`h3_split`); the group's
    /// model decides it, never the flag.
    rows,

    pub fn parse(s: []const u8) ?Mode {
        return std.meta.stringToEnum(Mode, s);
    }
};

/// The most ranks a group holds (JACCL's mesh: a cable from every Mac to every other).
pub const max_ranks = 8;
pub const TeardownError = error{ DistTeardownWrongThread, DistTeardownBusy };
pub const ReleaseError = TeardownError || error{ReleaseSendFailed};

/// How this process joins the group. Each rank names only its OWN RDMA devices: in a pair the
/// one `rdma_enX` whose port faces the peer, past a pair a comma list with the port facing each
/// rank in rank order (its own entry `-` or empty). JACCL reads only its own row of the device
/// matrix but validates the whole square, so the other rows repeat names.
pub const Config = struct {
    rank: u32,
    size: u32 = 2,
    /// `ip:port` rank 0 listens on; workers connect out (no inbound firewall prompt there). Past a pair
    /// each worker dials rank 0's address on its own cable, so rank 0 listens on every address (`jacclCoordinator`).
    coordinator: []const u8,
    device: []const u8,
    /// CLOCK_MONOTONIC ms (`crt.monotonicMs`) by which this rank's part of the group must form, bells included;
    /// 0 gives `default_formation_ms` from the start of `Link.open`.
    formation_deadline_ms: u64 = 0,
    /// Completes the READY/GO handshake before returning the link.
    ready_in_open: bool = false,
    open_control: ?*OpenControl = null,
};

var jaccl_deadline_warned: std.atomic.Value(bool) = .init(false);

/// A formation's budget when the caller sets no deadline: a worker started a little before rank 0 still joins.
const default_formation_ms: u64 = 10_000;

fn formationDeadline(cfg: Config) u64 {
    if (cfg.formation_deadline_ms != 0) return cfg.formation_deadline_ms;
    return @as(u64, @intCast(crt.monotonicMs())) + default_formation_ms;
}

/// A follower whose formation fails returns 80-100 ms after JACCL throws, most of it freeing the RDMA objects it had
/// made (G2b run 200: rank 0, which fails before it makes any, returns within 33 ms). JACCL's deadline comes that much
/// earlier, so a failed open still returns by the formation deadline.
const jaccl_teardown_ms: u64 = 150;

fn jacclDeadline(deadline_ms: u64) u64 {
    return @max(deadline_ms -| jaccl_teardown_ms, 1);
}

test "JACCL's deadline leaves the teardown margin before the formation deadline" {
    try std.testing.expectEqual(@as(u64, 9_850), jacclDeadline(10_000));
    try std.testing.expectEqual(@as(u64, 1), jacclDeadline(100));
}

fn msLeft(deadline_ms: u64) u32 {
    const now: u64 = @intCast(crt.monotonicMs());
    return @intCast(@min(deadline_ms -| now, std.math.maxInt(u32)));
}

/// `cfg.device` as one entry per rank: a single device (a pair) faces the one peer.
fn deviceList(cfg: Config, out: *[max_ranks][]const u8) ![]const []const u8 {
    if (cfg.size < 2 or cfg.size > max_ranks or cfg.rank >= cfg.size) return error.DistDeviceList;
    if (std.mem.indexOfScalar(u8, cfg.device, ',') == null) {
        // One port faces one peer only.
        if (cfg.size != 2 or cfg.device.len == 0) return error.DistDeviceList;
        for (out[0..2]) |*d| d.* = cfg.device;
        out[cfg.rank] = "-";
        return out[0..2];
    }
    var it = std.mem.splitScalar(u8, cfg.device, ',');
    var n: usize = 0;
    while (it.next()) |raw| : (n += 1) {
        if (n == cfg.size) return error.DistDeviceList;
        const d = std.mem.trim(u8, raw, " ");
        const own = d.len == 0 or std.mem.eql(u8, d, "-");
        if (own != (n == cfg.rank)) return error.DistDeviceList;
        out[n] = d;
    }
    if (n != cfg.size) return error.DistDeviceList;
    return out[0..n];
}

/// Writes the JACCL device matrix for `cfg` into `out`: this rank's row names the port facing each peer; row
/// `r` of another rank repeats those names, with the port facing `r` in this rank's column.
pub fn deviceMatrixJson(cfg: Config, out: []u8) ![]const u8 {
    var lb: [max_ranks][]const u8 = undefined;
    const list = try deviceList(cfg, &lb);
    var w: std.Io.Writer = .fixed(out);
    try w.writeByte('[');
    for (0..cfg.size) |r| {
        if (r != 0) try w.writeByte(',');
        try w.writeByte('[');
        for (0..cfg.size) |c| {
            if (c != 0) try w.writeByte(',');
            if (r == c) try w.writeAll("null") else try w.print("\"{s}\"", .{list[if (c == cfg.rank) r else c]});
        }
        try w.writeByte(']');
    }
    try w.writeByte(']');
    return w.buffered();
}

/// The coordinator address JACCL uses on this rank: past a pair rank 0 listens on every address, since each
/// worker reaches it over its own cable and subnet; a worker, and a pair's rank 0, use the address given.
pub fn jacclCoordinator(cfg: Config, out: []u8) ![:0]const u8 {
    const c = try bellAddress(cfg);
    return std.fmt.bufPrintSentinel(out, "{d}.{d}.{d}.{d}:{d}", .{ c.ip4[0], c.ip4[1], c.ip4[2], c.ip4[3], c.port }, 0);
}

/// Where the bell listens (rank 0) or dials (a worker), on the coordinator's port + 1: as `jacclCoordinator`.
fn bellAddress(cfg: Config) !Coordinator {
    var c = try parseCoordinator(cfg.coordinator);
    if (cfg.rank == 0 and cfg.size > 2) c.ip4 = .{ 0, 0, 0, 0 };
    return c;
}

pub const LayerSpan = @import("../../../sdk/weights/shard.zig").LayerSpan;

pub const Shard = @import("../../../sdk/weights/shard.zig").Shard;

pub const headOnly = @import("../../../sdk/weights/shard.zig").headOnly;

const LinkState = struct {
    n: u64 = 0,
    rank: u32 = 0,
    bells_seen: std.atomic.Value(u32) = .init(0),
    closing: std.atomic.Value(bool) = .init(false),
    releasing: std.atomic.Value(bool) = .init(false),
    aborting: std.atomic.Value(bool) = .init(false),
    abort_timer: ?std.Thread = null,
    teardown_timer: ?std.Thread = null,
    teardown_done: std.atomic.Value(bool) = .init(false),
    test_teardown_pause: if (builtin.is_test) ?*TestTeardownPause else void = if (builtin.is_test) null else {},
    exit_mu: std.atomic.Mutex = .unlocked,
    watcher: ?std.Thread = null,
    control: ?*OpenControl = null,
    switch_open: bool = false,
    go_committed: bool = false,
    test_exits: if (builtin.is_test) ?*TestDistExit else void = if (builtin.is_test) null else {},
    /// The worker whose bell closed (0: none).
    lost_rank: std.atomic.Value(u32) = .init(0),
    /// The worker's compute time for its last step (µs), sent over the bell after each one.
    worker_us: std.atomic.Value(u32) = .init(0),
    /// Zero before READY, else the wire code + 1; publishes the other frame fields.
    ready_code: [max_ranks]std.atomic.Value(u32) = @splat(.init(0)),
    ready_resident_mib: [max_ranks]std.atomic.Value(u32) = @splat(.init(0)),
    ready_calib_us: [max_ranks]std.atomic.Value(u32) = @splat(.init(0)),
    ready_load_ms: [max_ranks]std.atomic.Value(u32) = @splat(.init(0)),
    /// A worker bell closed; receives from that worker cannot complete.
    worker_gone: std.atomic.Value(bool) = .init(false),
    /// Receives from the worker in progress on the inference thread (`queuedRecv`).
    recvs: std.atomic.Value(u32) = .init(0),
    loop_tick_ms: std.atomic.Value(u64) = .init(0),
    watchdog: dist_watchdog.State = .{},
    watchdog_started: std.atomic.Value(bool) = .init(false),
    watchdog_stop: std.atomic.Value(bool) = .init(false),
    /// The watchdog and (rank 0) heartbeat threads of the open link; `Link.deinit` joins them.
    watchdog_threads: [2]?std.Thread = .{ null, null },
    heartbeat_active: std.atomic.Value(bool) = .init(false),
    follower_pending: std.atomic.Value(bool) = .init(false),
    /// Armed by the first leader byte after READY/GO.
    follower_ready: std.atomic.Value(bool) = .init(false),
    bell_mu: std.atomic.Mutex = .unlocked,
    ready_to_go_ms: u32 = 0,

    fn exit(self: *LinkState, detected: dist_watchdog.ExitCause, rank: ?u32, waited: ?u64) void {
        lock(&self.exit_mu);
        defer self.exit_mu.unlock();
        if (detected == .teardown_timeout and self.teardown_done.load(.acquire)) return;
        if (detected != .teardown_timeout and (self.closing.load(.acquire) or self.releasing.load(.acquire))) return;
        if ((detected == .bell_eof or detected == .bell_timeout) and self.aborting.load(.acquire)) return;
        const cancelled = if (self.control) |control| control.cancelled.load(.acquire) else false;
        const cause: dist_watchdog.ExitCause = if (cancelled and self.go_committed)
            (if (self.rank == 0) .partial_go else .cancelled_after_go)
        else
            detected;
        if (comptime builtin.is_test) if (self.test_exits) |counter| {
            _ = counter.causes.fetchOr(@as(u32, 1) << @backingInt(cause), .acq_rel);
            _ = counter.count.fetchAdd(1, .acq_rel);
            return;
        };
        log.err("[dist] attempt={d} cause={s}\n", .{ self.n, @tagName(cause) });
        distExit(cause, rank, waited);
    }
};

fn lock(mu: *std.atomic.Mutex) void {
    while (!mu.tryLock()) std.atomic.spinLoopHint();
}

pub const OpenControl = struct {
    failure_rank: std.atomic.Value(u32) = .init(std.math.maxInt(u32)),

    n: u64,
    cancelled: std.atomic.Value(bool) = .init(false),
    mu: std.atomic.Mutex = .unlocked,
    fds: [max_ranks + 2]std.c.fd_t = @splat(Bell.none),

    pub fn failedRank(self: *const OpenControl) ?u32 {
        const rank = self.failure_rank.load(.acquire);
        return if (rank == std.math.maxInt(u32)) null else rank;
    }

    pub fn publish(self: *OpenControl, fd: std.c.fd_t) !void {
        if (fd == Bell.none) return;
        lock(&self.mu);
        defer self.mu.unlock();
        for (&self.fds) |*slot| {
            if (slot.* != Bell.none) continue;
            slot.* = fd;
            if (!self.cancelled.load(.acquire)) return;
            _ = std.c.shutdown(fd, std.c.SHUT.RDWR);
            return error.DistAborted;
        }
        return error.DistTooManyBells;
    }

    pub fn cancel(self: *OpenControl) void {
        lock(&self.mu);
        defer self.mu.unlock();
        self.cancelled.store(true, .release);
        for (self.fds) |fd| if (fd != Bell.none) {
            _ = std.c.shutdown(fd, std.c.SHUT.RDWR);
        };
    }

    fn close(self: *OpenControl, fd: std.c.fd_t) void {
        lock(&self.mu);
        defer self.mu.unlock();
        for (&self.fds) |*slot| if (slot.* == fd) {
            slot.* = Bell.none;
        };
        _ = std.c.close(fd);
    }
};

fn closeBell(control: ?*OpenControl, fd: std.c.fd_t) void {
    if (fd == Bell.none) return;
    if (control) |c| c.close(fd) else _ = std.c.close(fd);
}

fn newLink(cfg: Config) !Link {
    const c = cluster();
    lock(&c.state_mu);
    defer c.state_mu.unlock();
    if (c.backends_not_cleared) return error.DistBackendsNotCleared;
    if (c.active_state != null) return error.DistLinkAlreadyOpen;
    const state = try std.heap.page_allocator.create(LinkState);
    state.* = .{ .rank = cfg.rank, .control = cfg.open_control, .n = if (cfg.open_control) |ctrl| ctrl.n else 0, .switch_open = cfg.ready_in_open };
    c.last_liveness = .{};
    c.active_state = state;
    return .{ .group = .{}, .rank = cfg.rank, .size = cfg.size, .state = state };
}

var test_clear_result: if (builtin.is_test) ?c_int else void = if (builtin.is_test) null else {};

var test_env_writes: if (builtin.is_test) usize else void = if (builtin.is_test) 0 else {};

fn setForeignIfDifferent(name: [*:0]const u8, value: [:0]const u8) void {
    if (std.c.getenv(name)) |old| if (std.mem.eql(u8, std.mem.span(old), value)) return;
    _ = env.setForeign(name, value);
    if (comptime builtin.is_test) test_env_writes += 1;
}

fn setDistEnv(cfg: Config, path: [:0]const u8) !void {
    var rank_buf: [16]u8 = undefined;
    var coord_buf: [128]u8 = undefined;
    setForeignIfDifferent("MLX_RANK", try std.fmt.bufPrintSentinel(&rank_buf, "{d}", .{cfg.rank}, 0));
    setForeignIfDifferent("MLX_JACCL_COORDINATOR", try jacclCoordinator(cfg, &coord_buf));
    setForeignIfDifferent("MLX_IBV_DEVICES", path);
}

fn envMatches(name: [*:0]const u8, value: []const u8) bool {
    const old = std.c.getenv(name) orelse return false;
    return std.mem.eql(u8, std.mem.span(old), value);
}

/// Main thread at startup, before background environment readers.
pub fn primeEnv(cfg: Config) !void {
    if (cfg.size < 2 or cfg.size > max_ranks or cfg.rank >= cfg.size) return error.InvalidArgs;
    var matrix: [1024]u8 = undefined;
    _ = deviceMatrixJson(cfg, &matrix) catch |err| {
        log.err("[dist] {s}: --dist-device needs {d} entries in rank order with this box's own `-` at rank {d}: got \"{s}\"\n", .{ @errorName(err), cfg.size, cfg.rank, cfg.device });
        return err;
    };
    var buf: [128]u8 = undefined;
    const path = try std.fmt.bufPrintSentinel(&buf, "/tmp/mlx-serve-jaccl-{d}.json", .{std.c.getpid()}, 0);
    try setDistEnv(cfg, path);
    // A switch forms its group long after this process's first fence; MLX reads the switch only there.
    preferFastSynch();
}

pub fn envPrimed(cfg: Config) bool {
    var rank: [16]u8 = undefined;
    var coord: [128]u8 = undefined;
    var path: [128]u8 = undefined;
    return envMatches("MLX_RANK", std.fmt.bufPrint(&rank, "{d}", .{cfg.rank}) catch return false) and
        envMatches("MLX_JACCL_COORDINATOR", jacclCoordinator(cfg, &coord) catch return false) and
        envMatches("MLX_IBV_DEVICES", std.fmt.bufPrint(&path, "/tmp/mlx-serve-jaccl-{d}.json", .{std.c.getpid()}) catch return false);
}

pub const Link = struct {
    state: ?*LinkState = null,
    group: mlx.mlx_distributed_group,
    ready_to_go_ms: ?u32 = null,
    rank: u32,
    size: u32,
    /// Rank 0: the bell to each worker, by rank (entry 0 unused). A worker: its bell to rank 0, entry 0.
    bells: [max_ranks]Bell = @splat(.{}),

    /// Joins the group. Blocks until every rank has connected to the coordinator.
    pub fn open(io: std.Io, cfg: Config) !Link {
        if (cfg.ready_in_open and cfg.formation_deadline_ms == 0) return error.InvalidArgs;
        if (cfg.ready_in_open and !envPrimed(cfg)) return error.DistEnvNotPrimed;
        var link = try newLink(cfg);
        errdefer link.deinit();
        errdefer link.captureFailureRank();
        const started_ms: u64 = @intCast(crt.monotonicMs());
        if (comptime builtin.os.tag != .macos) return error.RdmaUnavailable;
        if (!mlx.mlx_distributed_is_available("jaccl")) {
            log.err("[dist] JACCL is unavailable: RDMA over Thunderbolt needs macOS 26.2+ and `rdma_ctl enable` (Recovery)\n", .{});
            return error.RdmaUnavailable;
        }
        var path_buf: [128]u8 = undefined;
        const path = try std.fmt.bufPrintSentinel(&path_buf, "/tmp/mlx-serve-jaccl-{d}.json", .{std.c.getpid()}, 0);
        var json_buf: [1024]u8 = undefined;
        const json = deviceMatrixJson(cfg, &json_buf) catch |err| {
            log.err("[dist] the device needs one RDMA port per rank in rank order, this rank's own `-` ({d} ranks, rank {d}): got \"{s}\"\n", .{ cfg.size, cfg.rank, cfg.device });
            return err;
        };
        {
            const f = try std.Io.Dir.createFileAbsolute(io, path, .{});
            defer f.close(io);
            var wb: [256]u8 = undefined;
            var fw = f.writer(io, &wb);
            try fw.interface.writeAll(json);
            try fw.interface.flush();
        }
        if (!cfg.ready_in_open) try setDistEnv(cfg, path);
        log.info("[dist] joining as rank {d}/{d} via {s}, coordinator {s}\n", .{ cfg.rank, cfg.size, cfg.device, cfg.coordinator });
        // Rank 0 files each worker's bell, by the rank it names, while the group forms. A worker dials its bell
        // before JACCL, so a rank with no bell has not started: the join names it.
        const coord = try bellAddress(cfg);
        const deadline = formationDeadline(cfg);
        var rv: Rendezvous = .{ .deadline_ms = deadline, .listener = if (cfg.rank == 0) Bell.listen(coord, cfg.size - 1) else Bell.none, .size = cfg.size, .state = link.state, .control = cfg.open_control };
        defer rv.close();
        if (cfg.open_control) |ctrl| try ctrl.publish(rv.listener);
        var acceptor: ?std.Thread = if (rv.listener != Bell.none) std.Thread.spawn(.{ .stack_size = 1 << 16 }, Rendezvous.run, .{&rv}) catch null else null;
        defer if (acceptor) |t| {
            rv.stop.store(true, .release);
            t.join();
        };
        // A worker never runs bell-less: without the bell it has no heartbeat and no EOF to leave on when rank 0 dies.
        var to_host: Bell = if (cfg.rank == 0) .{} else dialBellControlled(coord, cfg.rank, deadline, cfg.open_control);
        errdefer if (to_host.open()) {
            closeBell(cfg.open_control, to_host.fd);
        };
        if (cfg.rank != 0 and !to_host.open()) {
            log.err("[dist] rank {d}: no bell connection to rank 0 at {s} (port + 1) by the formation deadline\n", .{ cfg.rank, cfg.coordinator });
            link.captureMissingRank();
            return error.DistNoBell;
        }
        if (!mlx.jacclFormationDeadlineAvailable() and !jaccl_deadline_warned.swap(true, .acq_rel))
            log.err("[dist] JACCL formation deadline unavailable: this libjaccl lacks jaccl_set_formation_deadline_ms (a stock MLX), so a missing peer blocks formation until the process dies\n", .{});
        mlx.setJacclFormationDeadline(jacclDeadline(deadline));
        defer mlx.setJacclFormationDeadline(0);
        var g = mlx.mlx_distributed_group_new();
        defer _ = mlx.mlx_distributed_group_free(g);
        mlx.check(mlx.mlx_distributed_init(&g, true, "jaccl")) catch {
            var eb: [512]u8 = undefined;
            const past: i64 = crt.monotonicMs() - @as(i64, @intCast(jacclDeadline(deadline)));
            log.err("[dist] JACCL init failed ({d} ms past its deadline): {s}\n", .{ past, mlx.takeError(&eb) orelse "unknown" });
            if (cfg.rank == 0) link.captureMissingRank();
            return error.RdmaInitFailed;
        };
        link.group = g;
        g = .{};
        link.rank = @intCast(mlx.mlx_distributed_group_rank(link.group));
        link.size = @intCast(mlx.mlx_distributed_group_size(link.group));
        if (link.rank != cfg.rank or link.size != cfg.size) return error.RdmaGroupMismatch;
        if (link.rank != 0) {
            link.bells[0] = to_host;
            to_host = .{};
        } else if (acceptor) |t| {
            // Every worker dialed its bell before it joined JACCL; one still missing gets what is left of the deadline.
            _ = rv.waitAll(msLeft(deadline));
            rv.stop.store(true, .release);
            t.join();
            acceptor = null;
            rv.handOver(&link);
        }
        if (link.rank == 0) {
            var mb: [64]u8 = undefined;
            const missing = missingRanks(link.bellMask(), link.size, &mb);
            if (missing.len > 0) {
                log.err("[dist] no bell connection from rank {s} by the formation deadline: is that worker an older mlx-serve?\n", .{missing});
                link.captureMissingRank();
                return error.DistNoBell;
            }
        }
        try link.finishOpen(cfg, started_ms, deadline);
        log.info("[dist] joined: rank {d} of {d}\n", .{ link.rank, link.size });
        return link;
    }

    fn finishOpen(self: *Link, cfg: Config, started_ms: u64, deadline: u64) !void {
        self.attachBells();
        if (cfg.ready_in_open) {
            if (self.rank == 0) {
                leaderAwaitReadyThenGo(&self.bells, self.size, deadline) catch |err| {
                    if (err == error.DistPartialGo) self.state.?.exit(.partial_go, null, null);
                    return err;
                };
                self.ready_to_go_ms = self.state.?.ready_to_go_ms;
            } else {
                var active: usize = 0;
                _ = mlx.mlx_get_active_memory(&active);
                try workerReadyAwaitGo(self.bells[0], deadline, .{ .resident_mib = @intCast(active >> 20), .load_ms = loadElapsedMs(started_ms) });
                self.state.?.follower_pending.store(true, .release);
            }
        }
        self.state.?.go_committed = cfg.ready_in_open;
        startWatchdog(self);
        if (cfg.ready_in_open or cfg.rank != 0) startWatchers(self) catch |err| {
            log.err("[dist] cannot watch peer: {s}\n", .{@errorName(err)});
            if (cfg.ready_in_open) self.state.?.exit(.watchdog_thread, null, null);
        };
    }

    fn attachBells(self: *Link) void {
        for (&self.bells) |*bell| bell.state = self.state;
    }

    fn captureFailureRank(self: *const Link) void {
        const state = self.state.?;
        const control = state.control orelse return;
        const lost = state.lost_rank.load(.acquire);
        if (lost != 0) {
            control.failure_rank.store(lost, .release);
            return;
        }
    }

    fn captureMissingRank(self: *const Link) void {
        const state = self.state.?;
        const control = state.control orelse return;
        if (self.rank != 0) {
            control.failure_rank.store(0, .release);
            return;
        }
        const seen = state.bells_seen.load(.acquire);
        for (1..self.size) |rank| {
            if (seen & (@as(u32, 1) << @intCast(rank)) != 0) continue;
            control.failure_rank.store(@intCast(rank), .release);
            return;
        }
    }

    pub fn activate(self: *Link) error{DistTeardownWrongThread}!void {
        if (cluster().inference_thread != std.Thread.getCurrentId()) return error.DistTeardownWrongThread;
        const c = cluster();
        lock(&c.state_mu);
        defer c.state_mu.unlock();
        if (c.switch_group) |g| if (g.link == self) {
            c.tp_link = self;
        };
        if (self.rank == 0 and !self.state.?.heartbeat_active.load(.acquire)) activateHeartbeat(self.state.?);
    }

    pub fn cancelledAfterGo(self: *Link) void {
        const state = self.state orelse return;
        if (!state.go_committed) return;
        state.exit(if (self.rank == 0) .partial_go else .cancelled_after_go, null, null);
    }

    pub fn checkTeardownThread(self: *const Link) TeardownError!void {
        if (cluster().inference_thread != std.Thread.getCurrentId()) return error.DistTeardownWrongThread;
        const state = self.state orelse return error.DistTeardownBusy;
        if (state.recvs.load(.acquire) != 0 or state.watchdog.in_flight.load(.acquire)) return error.DistTeardownBusy;
    }

    pub fn drain(self: *const Link) TeardownError!void {
        try self.checkTeardownThread();
        if (comptime builtin.is_test) if (self.state.?.test_teardown_pause) |pause| pause.visit(.drain);
        const cpu = mlx.mlx_default_cpu_stream_new();
        defer _ = mlx.mlx_stream_free(cpu);
        _ = mlx.mlx_synchronize(cpu);
        if (!mlx.noGpuBackend()) {
            const gpu = mlx.mlx_default_gpu_stream_new();
            defer _ = mlx.mlx_stream_free(gpu);
            _ = mlx.mlx_synchronize(gpu);
        }
    }

    pub fn disarm(self: *Link) void {
        const state = self.state.?;
        const deadline = @as(u64, @intCast(crt.monotonicMs())) + env.intValue(u32, env.get(.MLX_SERVE_DIST_STALL_MS), 30_000);
        state.teardown_timer = std.Thread.spawn(.{ .stack_size = 1 << 20 }, teardownDeadline, .{ state, deadline }) catch {
            state.exit(.watchdog_thread, null, null);
            return;
        };
        lock(&state.exit_mu);
        state.releasing.store(true, .release);
        state.exit_mu.unlock();
        if (state.watcher) |t| t.join();
        state.watcher = null;
        stopWatchdog(state);
    }

    pub fn release(self: *Link, rank: *u32) ReleaseError!void {
        try self.checkTeardownThread();
        if (self.rank != 0) return;
        const deadline = @as(u64, @intCast(crt.monotonicMs())) + 2000;
        for (1..self.size) |k| {
            const bell = self.bells[k];
            if (!bell.open() or std.c.send(bell.fd, &[_]u8{Bell.release_byte}, 1, std.c.MSG.DONTWAIT) != 1) {
                rank.* = @intCast(k);
                return error.ReleaseSendFailed;
            }
        }
        self.awaitFollowerEofs(deadline);
    }

    fn awaitFollowerEofs(self: *const Link, deadline: u64) void {
        var fds: [max_ranks]std.posix.pollfd = undefined;
        var pending: u32 = self.size - 1;
        for (1..self.size) |k| fds[k] = .{ .fd = self.bells[k].fd, .events = std.posix.POLL.IN, .revents = 0 };
        while (pending > 0 and msLeft(deadline) > 0) {
            _ = std.posix.poll(fds[1..self.size], @intCast(msLeft(deadline))) catch continue;
            for (fds[1..self.size]) |*fd| {
                if (fd.fd == Bell.none or fd.revents == 0) continue;
                var buf: [64]u8 = undefined;
                const n = std.c.recv(fd.fd, &buf, buf.len, std.c.MSG.DONTWAIT);
                if (n > 0) continue;
                if (n < 0 and (std.c._errno().* == @backingInt(std.c.E.AGAIN) or std.c._errno().* == @backingInt(std.c.E.INTR))) continue;
                fd.fd = Bell.none;
                pending -= 1;
            }
        }
    }

    pub fn abortFormed(self: *Link) void {
        const state = self.state.?;
        lock(&state.exit_mu);
        defer state.exit_mu.unlock();
        if (state.closing.load(.acquire) or state.releasing.load(.acquire) or state.aborting.swap(true, .acq_rel)) return;
        state.abort_timer = std.Thread.spawn(.{ .stack_size = 1 << 16 }, abortFallback, .{self.*}) catch {
            _ = std.c.shutdown(self.bells[0].fd, std.c.SHUT.RDWR);
            return;
        };
    }

    pub fn deinit(self: *Link) void {
        const state = self.state orelse return;
        lock(&state.exit_mu);
        state.closing.store(true, .release);
        state.exit_mu.unlock();
        for (self.bells) |bell| if (bell.open()) {
            _ = std.c.shutdown(bell.fd, std.c.SHUT.RDWR);
        };
        if (state.watcher) |t| t.join();
        if (state.abort_timer) |t| t.join();
        stopWatchdog(state);
        for (&self.bells) |*bell| if (bell.open()) {
            closeBell(state.control, bell.fd);
            bell.* = .{};
        };
        if (comptime builtin.is_test) if (state.test_teardown_pause) |pause| pause.visit(.free);
        _ = mlx.mlx_distributed_group_free(self.group);
        self.group = .{};
        const clear_result = if (builtin.is_test) test_clear_result orelse mlx.mlx_distributed_clear_backends() else mlx.mlx_distributed_clear_backends();
        const c = cluster();
        lock(&c.state_mu);
        if (clear_result != 0) c.backends_not_cleared = true;
        if (c.switch_group) |g| if (g.link == self) {
            c.switch_group = null;
        };
        if (c.tp_link) |tp| if (tp.state == state) {
            c.tp_link = null;
            c.tp_live = null;
            c.tp_spec = null;
        };
        if (c.active_state == state) {
            c.last_liveness = readLiveness(state);
            c.active_state = null;
        }
        c.state_mu.unlock();
        state.teardown_done.store(true, .release);
        if (state.teardown_timer) |t| t.join();
        std.heap.page_allocator.destroy(state);
        self.state = null;
    }

    /// Bit `k` set for each worker `k` rank 0 has a bell to.
    fn bellMask(self: *const Link) u32 {
        var m: u32 = 0;
        for (1..self.size) |k| if (self.bells[k].open()) {
            m |= @as(u32, 1) << @intCast(k);
        };
        return m;
    }

    // The collectives have CPU kernels only, so every op runs on this thread's CPU stream
    // (streams are thread-local); MLX fences the GPU work they read.

    /// Lazy send of `x` to `dst`; the returned array must be evaluated (or depended on).
    pub fn send(self: *const Link, x: mlx.mlx_array, dst: u32) !mlx.mlx_array {
        const cs = mlx.mlx_default_cpu_stream_new();
        defer _ = mlx.mlx_stream_free(cs);
        var r = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_distributed_send(&r, x, @intCast(dst), self.group, cs));
        return r;
    }

    pub fn recv(self: *const Link, shape: []const c_int, dtype: mlx.mlx_dtype, src: u32) !mlx.mlx_array {
        const cs = mlx.mlx_default_cpu_stream_new();
        defer _ = mlx.mlx_stream_free(cs);
        var r = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_distributed_recv(&r, shape.ptr, shape.len, dtype, @intCast(src), self.group, cs));
        return r;
    }

    pub fn allSum(self: *const Link, x: mlx.mlx_array) !mlx.mlx_array {
        const cs = mlx.mlx_default_cpu_stream_new();
        defer _ = mlx.mlx_stream_free(cs);
        var r = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_distributed_all_sum(&r, x, self.group, cs));
        return r;
    }

    /// Every rank's `x` stacked on axis 0 in rank order; the shapes must match.
    pub fn allGather(self: *const Link, x: mlx.mlx_array) !mlx.mlx_array {
        const cs = mlx.mlx_default_cpu_stream_new();
        defer _ = mlx.mlx_stream_free(cs);
        var r = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_distributed_all_gather(&r, x, self.group, cs));
        return r;
    }
};

/// A plain TCP connection beside the RDMA link, on the coordinator's port + 1, one per worker. Rank 0
/// writes one byte before each control message to a worker and the worker sleeps on it before receiving
/// one: JACCL waits by polling the link in a loop, which held a core at 100% while a worker idled. A worker
/// names its rank in the first byte it writes. A `pipe_byte` alone is a DeepSeek-V4.1 tensor pipe step: no message
/// follows it (`OP_DSV41_TP_PIPE`).
const Bell = struct {
    fd: std.c.fd_t = none,
    follower: bool = false,
    state: ?*LinkState = null,

    const msg_byte: u8 = 1;
    const pipe_byte: u8 = 2;
    const draft_byte: u8 = 3;
    const verify_byte: u8 = 4;
    const heartbeat_byte: u8 = 5;
    const go_byte: u8 = 6;
    const abort_byte: u8 = 7;
    const release_byte: u8 = 8;

    /// No connection. `fd_t` is a HANDLE on Windows, where a cluster never forms.
    const none: std.c.fd_t = if (builtin.os.tag == .windows) std.os.windows.INVALID_HANDLE_VALUE else -1;

    fn open(b: Bell) bool {
        return b.fd != none;
    }

    fn listen(coord: Coordinator, backlog: u32) std.c.fd_t {
        const fd = std.c.socket(std.posix.AF.INET, std.posix.SOCK.STREAM, 0);
        if (fd < 0) return none;
        const one: c_int = 1;
        _ = std.c.setsockopt(fd, std.posix.SOL.SOCKET, std.posix.SO.REUSEADDR, &one, @sizeOf(c_int));
        var sa: std.posix.sockaddr.in = .{ .port = std.mem.nativeToBig(u16, coord.port +% 1), .addr = @bitCast(coord.ip4) };
        if (std.c.bind(fd, @ptrCast(&sa), @sizeOf(std.posix.sockaddr.in)) != 0 or std.c.listen(fd, @intCast(backlog)) != 0) {
            _ = std.c.close(fd);
            return none;
        }
        return fd;
    }

    /// One worker's connection and the rank it names, or null when none came within `ms`.
    fn acceptRank(listener: std.c.fd_t, ms: i32, control: ?*OpenControl, deadline_ms: u64) ?struct { rank: u32, bell: Bell } {
        if (listener == none) return null;
        var fds = [_]std.posix.pollfd{.{ .fd = listener, .events = std.posix.POLL.IN, .revents = 0 }};
        if ((std.posix.poll(&fds, ms) catch return null) == 0) return null;
        const fd = std.c.accept(listener, null, null);
        if (fd < 0) return null;
        const b: Bell = .{ .fd = noDelay(fd) };
        if (control) |ctrl| ctrl.publish(fd) catch {
            closeBell(control, fd);
            return null;
        };
        var rb = [_]std.posix.pollfd{.{ .fd = fd, .events = std.posix.POLL.IN, .revents = 0 }};
        var rank: [1]u8 = undefined;
        if ((std.posix.poll(&rb, @intCast(@min(200, msLeft(deadline_ms)))) catch 0) == 0 or (if (b.readAll(&rank)) |_| false else |_| true)) {
            closeBell(control, fd);
            return null;
        }
        return .{ .rank = rank[0], .bell = b };
    }

    /// A worker's bell: connected, then its rank.
    fn connect(coord: Coordinator, rank: u32, control: ?*OpenControl, deadline_ms: u64) Bell {
        const fd = std.c.socket(std.posix.AF.INET, std.posix.SOCK.STREAM, 0);
        if (fd < 0) return .{};
        if (control) |ctrl| ctrl.publish(fd) catch {
            closeBell(control, fd);
            return .{};
        };
        const flags = std.c.fcntl(fd, std.c.F.GETFL);
        const nonblock: c_int = @bitCast(std.c.O{ .NONBLOCK = true });
        if (flags < 0 or std.c.fcntl(fd, std.c.F.SETFL, flags | nonblock) != 0) {
            closeBell(control, fd);
            return .{};
        }
        const connected = beginConnect(fd, coord, control) catch {
            closeBell(control, fd);
            return .{};
        };
        if (!connected) awaitConnect(fd, deadline_ms, control) catch {
            closeBell(control, fd);
            return .{};
        };
        if (control) |ctrl| if (ctrl.cancelled.load(.acquire)) {
            closeBell(control, fd);
            return .{};
        };
        if (std.c.fcntl(fd, std.c.F.SETFL, flags) != 0) {
            closeBell(control, fd);
            return .{};
        }
        const b = makeFollower(fd);
        b.writeAll(&[_]u8{@intCast(rank)}) catch {
            closeBell(control, fd);
            return .{};
        };
        return b;
    }

    fn beginConnect(fd: std.c.fd_t, coord: Coordinator, control: ?*OpenControl) !bool {
        if (control) |ctrl| lock(&ctrl.mu);
        defer if (control) |ctrl| ctrl.mu.unlock();
        if (control) |ctrl| if (ctrl.cancelled.load(.acquire)) return error.DistAborted;
        var addr: std.posix.sockaddr.in = .{ .port = std.mem.nativeToBig(u16, coord.port +% 1), .addr = @bitCast(coord.ip4) };
        if (std.c.connect(fd, @ptrCast(&addr), @sizeOf(std.posix.sockaddr.in)) == 0) return true;
        if (std.c._errno().* != @backingInt(std.c.E.INPROGRESS)) return error.DistPeerGone;
        return false;
    }

    fn awaitConnect(fd: std.c.fd_t, deadline_ms: u64, control: ?*OpenControl) !void {
        const end = @min(deadline_ms, @as(u64, @intCast(crt.monotonicMs())) + 3000);
        var fds = [_]std.posix.pollfd{.{ .fd = fd, .events = std.posix.POLL.OUT, .revents = 0 }};
        while (msLeft(end) > 0) {
            if (control) |ctrl| if (ctrl.cancelled.load(.acquire)) return error.DistAborted;
            const wait_ms = @min(msLeft(end), if (control != null) @as(u32, 20) else 3000);
            if (try std.posix.poll(&fds, @intCast(wait_ms)) == 0) continue;
            if (control) |ctrl| if (ctrl.cancelled.load(.acquire)) return error.DistAborted;
            var err: c_int = 0;
            var len: std.posix.socklen_t = @sizeOf(c_int);
            if (std.c.getsockopt(fd, std.posix.SOL.SOCKET, std.posix.SO.ERROR, &err, &len) != 0 or err != 0) return error.DistPeerGone;
            return;
        }
        return error.DistNoBell;
    }

    fn makeFollower(fd: std.c.fd_t) Bell {
        return .{ .fd = noDelay(fd), .follower = true };
    }

    /// Also close-on-exec: a child that inherited the bell would hold it open after this process exits, and the peer
    /// would never see the EOF that tells it to leave.
    fn noDelay(fd: std.c.fd_t) std.c.fd_t {
        const one: c_int = 1;
        if (comptime builtin.os.tag == .macos) _ = std.c.setsockopt(fd, std.posix.SOL.SOCKET, std.posix.SO.NOSIGPIPE, &one, @sizeOf(c_int));
        _ = std.c.setsockopt(fd, std.posix.IPPROTO.TCP, std.posix.TCP.NODELAY, &one, @sizeOf(c_int));
        _ = std.c.fcntl(fd, std.posix.F.SETFD, @as(c_int, std.posix.FD_CLOEXEC));
        return fd;
    }

    fn writeAll(b: Bell, bytes: []const u8) !void {
        if (b.state) |state| lock(&state.bell_mu);
        defer if (b.state) |state| state.bell_mu.unlock();
        var off: usize = 0;
        while (off < bytes.len) {
            const n = std.c.write(b.fd, bytes.ptr + off, bytes.len - off);
            if (n <= 0) {
                if (b.follower) if (b.state) |state| {
                    if (state.follower_ready.load(.acquire)) followerExit(state, .eof);
                    if (state.aborting.load(.acquire)) return error.FallRequested;
                };
                return error.DistPeerGone;
            }
            off += @intCast(n);
        }
    }

    fn readAll(b: Bell, out: []u8) !void {
        var off: usize = 0;
        while (off < out.len) {
            const n = std.c.read(b.fd, out.ptr + off, out.len - off);
            const failure = dist_watchdog.classifyBellFailure(n, std.c._errno().*, @backingInt(std.c.E.INTR), @backingInt(std.c.E.AGAIN));
            if (failure) |f| {
                if (b.follower) if (b.state) |state| {
                    if (state.follower_ready.load(.acquire)) followerExit(state, f);
                    if (state.aborting.load(.acquire)) return error.FallRequested;
                };
                return error.DistPeerGone;
            }
            if (n < 0) continue;
            off += @intCast(n);
        }
    }

    fn ring(b: Bell) !void {
        return b.ringWith(msg_byte);
    }

    fn pipe(b: Bell) !void {
        return b.ringWith(pipe_byte);
    }

    fn ringWith(b: Bell, byte: u8) !void {
        if (!b.open()) return;
        try b.writeAll(&.{byte});
    }

    fn say(b: Bell, bytes: []const u8) !void {
        if (!b.open()) return error.DistPeerGone;
        return b.writeAll(bytes);
    }

    fn msgOf(b: Bell, byte: u8) !?Msg {
        return switch (byte) {
            release_byte => .{ OP_RELEASE, 0, 0, 0, 0, 0, 0, 0 },
            pipe_byte => .{ OP_DSV41_TP_PIPE, 0, 0, 0, 0, 0, 0, 0 },
            draft_byte => try b.draftMsg(),
            verify_byte => try b.verifyMsg(),
            else => null,
        };
    }

    fn draftMsg(b: Bell) !Msg {
        var p: [5]u8 = undefined;
        try b.readAll(&p);
        return .{ OP_DSV41_TP_DRAFT_PIPE, p[0], @bitCast(std.mem.readInt(u32, p[1..5], .little)), 0, 0, 0, 0, 0 };
    }

    fn verifyMsg(b: Bell) !Msg {
        var p: [2]u8 = undefined;
        try b.readAll(&p);
        return .{ OP_DSV41_TP_VERIFY_PIPE, p[0], p[1], 0, 0, 0, 0, 0 };
    }

    /// The byte rank 0 rang (`msg_byte` without a bell: a message follows on the link).
    fn wait(b: Bell) !u8 {
        if (!b.open()) return msg_byte;
        // Steps come a decode tick apart: poll that long before sleeping, so a step does not wait for the kernel to
        // wake this thread.
        if (comptime builtin.os.tag == .macos) {
            var fds = [_]std.posix.pollfd{.{ .fd = b.fd, .events = std.posix.POLL.IN, .revents = 0 }};
            for (0..spin_polls) |_| if ((std.posix.poll(&fds, 0) catch break) > 0) break;
        }
        var x: [1]u8 = undefined;
        while (true) {
            const n = std.c.read(b.fd, &x, 1);
            if (n == 1) {
                if (b.follower and b.state != null and b.state.?.follower_pending.swap(false, .acq_rel)) {
                    dist_watchdog.setBellTimeout(b.fd, idleTimeoutMs());
                    b.state.?.follower_ready.store(true, .release);
                }
                if (x[0] == heartbeat_byte) continue;
                return x[0];
            }
            const failure = dist_watchdog.classifyBellFailure(n, std.c._errno().*, @backingInt(std.c.E.INTR), @backingInt(std.c.E.AGAIN));
            if (failure) |f| {
                if (b.follower) if (b.state) |state| {
                    if (state.follower_ready.load(.acquire)) followerExit(state, f);
                    if (state.aborting.load(.acquire)) return error.FallRequested;
                };
                return error.DistPeerGone;
            }
            if (n < 0) continue;
        }
    }
};

const READY_TAG: u32 = 0xFFFFFF00;

const ReadyFrame = struct {
    code: u8 = 0,
    resident_mib: u32 = 0,
    calib_us: u32 = 0,
    load_ms: u32 = 0,

    fn encode(frame: ReadyFrame) [16]u8 {
        var out: [16]u8 = undefined;
        std.mem.writeInt(u32, out[0..4], READY_TAG | frame.code, .little);
        std.mem.writeInt(u32, out[4..8], frame.resident_mib, .little);
        std.mem.writeInt(u32, out[8..12], frame.calib_us, .little);
        std.mem.writeInt(u32, out[12..16], frame.load_ms, .little);
        return out;
    }

    fn decode(bytes: *const [16]u8) ReadyFrame {
        return .{
            .code = @truncate(std.mem.readInt(u32, bytes[0..4], .little)),
            .resident_mib = std.mem.readInt(u32, bytes[4..8], .little),
            .calib_us = std.mem.readInt(u32, bytes[8..12], .little),
            .load_ms = std.mem.readInt(u32, bytes[12..16], .little),
        };
    }
};

const BellWord = union(enum) { step: u32, ready: ReadyFrame };

const BellDecoder = struct {
    bytes: [16]u8 = undefined,
    used: usize = 0,
    need: usize = 4,

    fn read(self: *BellDecoder, bell: Bell) !?BellWord {
        while (true) {
            const n = std.c.recv(bell.fd, self.bytes[self.used..self.need].ptr, self.need - self.used, std.c.MSG.DONTWAIT);
            if (n == 0) return error.DistPeerGone;
            if (n < 0) switch (std.c._errno().*) {
                @backingInt(std.c.E.INTR) => continue,
                @backingInt(std.c.E.AGAIN) => return null,
                else => return error.DistPeerGone,
            };
            self.used += @intCast(n);
            if (self.used < self.need) continue;
            const word = std.mem.readInt(u32, self.bytes[0..4], .little);
            if (self.need == 4 and word >= READY_TAG) {
                self.need = 16;
                continue;
            }
            self.used = 0;
            self.need = 4;
            return if (word >= READY_TAG) .{ .ready = ReadyFrame.decode(&self.bytes) } else .{ .step = word };
        }
    }
};

fn sendReadyFrame(bell: Bell, frame: ReadyFrame) !void {
    return bell.say(&frame.encode());
}

fn sendStepTime(bell: Bell, us: i64) !void {
    var bytes: [4]u8 = undefined;
    std.mem.writeInt(u32, &bytes, @intCast(std.math.clamp(us, 0, READY_TAG - 1)), .little);
    try bell.say(&bytes);
}

fn loadElapsedMs(start_ms: u64) u32 {
    return @intCast(@min(@as(u64, @intCast(crt.monotonicMs())) -| start_ms, std.math.maxInt(u32)));
}

fn readyCodeName(code: u8) []const u8 {
    return switch (code) {
        0 => "ready",
        LOAD_FAILED => "LOAD_FAILED",
        NOT_FOUND => "NOT_FOUND",
        MISMATCH => "MISMATCH",
        else => "unknown",
    };
}

const GoProgress = struct {
    sent: u32 = 0,

    fn check(self: GoProgress, now_ms: u64, deadline_ms: u64) !void {
        if (now_ms < deadline_ms -| 200) return;
        return self.writeFailed();
    }

    fn writeFailed(self: GoProgress) error{ DistPartialGo, DistNoGo } {
        return if (self.sent > 0) error.DistPartialGo else error.DistNoGo;
    }
};

fn abortBells(bells: []const Bell, size: u32) void {
    for (bells[1..size]) |bell| _ = std.c.send(bell.fd, &[_]u8{Bell.abort_byte}, 1, std.c.MSG.DONTWAIT);
}

fn leaderAwaitReadyThenGo(bells: []const Bell, size: u32, deadline_ms: u64) !void {
    var go: GoProgress = .{};
    errdefer if (go.sent == 0) abortBells(bells, size);
    var decoders: [max_ranks]BellDecoder = @splat(.{});
    var fds: [max_ranks]std.posix.pollfd = undefined;
    var ready: u32 = 0;
    var ready_ms: u64 = 0;
    const all = ((@as(u32, 1) << @intCast(size)) - 1) & ~@as(u32, 1);
    for (1..size) |k| fds[k] = .{ .fd = bells[k].fd, .events = std.posix.POLL.IN, .revents = 0 };
    while (ready != all) {
        const left = msLeft(deadline_ms);
        if (left == 0) {
            var buf: [64]u8 = undefined;
            log.err("[dist] no READY from rank {s} by the formation deadline\n", .{missingRanks(ready, size, &buf)});
            for (1..size) |k| {
                if (ready & (@as(u32, 1) << @intCast(k)) != 0) continue;
                if (bells[k].state) |state| state.lost_rank.store(@intCast(k), .release);
                break;
            }
            return error.DistNoReady;
        }
        _ = try std.posix.poll(fds[1..size], @intCast(@min(left, std.math.maxInt(i32))));
        for (1..size) |k| {
            if (fds[k].revents == 0) continue;
            const message = decoders[k].read(bells[k]) catch |err| {
                if (bells[k].state) |state| state.lost_rank.store(@intCast(k), .release);
                return err;
            } orelse continue;
            const frame = switch (message) {
                .ready => |frame| frame,
                .step => return error.DistBadMessage,
            };
            if (frame.code != 0) {
                if (bells[k].state) |state| state.lost_rank.store(@intCast(k), .release);
                log.err("[dist] rank {d} not ready: {s}\n", .{ k, readyCodeName(frame.code) });
                return error.DistWorkerLoadFailed;
            }
            ready_ms = @intCast(crt.monotonicMs());
            log.info("[dist] worker rank {d} READY: {d} MiB, load {d} ms\n", .{ k, frame.resident_mib, frame.load_ms });
            ready |= @as(u32, 1) << @intCast(k);
        }
    }
    for (1..size) |k| {
        go.check(@intCast(crt.monotonicMs()), deadline_ms) catch |err| {
            logGoFailure(go.sent, size);
            return err;
        };
        if (std.c.send(bells[k].fd, &[_]u8{Bell.go_byte}, 1, std.c.MSG.DONTWAIT) != 1) {
            logGoFailure(go.sent, size);
            return go.writeFailed();
        }
        go.sent += 1;
    }
    if (bells[1].state) |state| state.ready_to_go_ms = loadElapsedMs(ready_ms);
}

fn logGoFailure(sent: u32, size: u32) void {
    if (sent == 0) return;
    const mask = ((@as(u32, 1) << @intCast(size)) - 1) & ~@as(u32, 1);
    const formed = ((@as(u32, 1) << @intCast(sent + 1)) - 1) & ~@as(u32, 1);
    var fb: [64]u8 = undefined;
    var nb: [64]u8 = undefined;
    log.err("[dist] partial GO: ranks {s} formed, {s} not\n", .{ missingRanks(mask & ~formed, size, &fb), missingRanks(formed, size, &nb) });
}

fn workerReadyAwaitGo(bell: Bell, deadline_ms: u64, frame: ReadyFrame) !void {
    const unformed: Bell = .{ .fd = bell.fd };
    sendReadyFrame(unformed, frame) catch return error.DistAborted;
    var fds = [_]std.posix.pollfd{.{ .fd = bell.fd, .events = std.posix.POLL.IN, .revents = 0 }};
    while (true) {
        const left = msLeft(deadline_ms);
        if (left == 0) return error.DistNoGo;
        if (try std.posix.poll(&fds, @intCast(@min(left, std.math.maxInt(i32)))) == 0) return error.DistNoGo;
        var byte: [1]u8 = undefined;
        unformed.readAll(&byte) catch return error.DistAborted;
        return switch (byte[0]) {
            Bell.go_byte => {},
            Bell.abort_byte => error.DistAborted,
            else => error.DistBadMessage,
        };
    }
}

fn publishReady(state: *LinkState, rank: usize, frame: ReadyFrame) void {
    state.ready_resident_mib[rank].store(frame.resident_mib, .monotonic);
    state.ready_calib_us[rank].store(frame.calib_us, .monotonic);
    state.ready_load_ms[rank].store(frame.load_ms, .monotonic);
    state.ready_code[rank].store(@as(u32, frame.code) + 1, .release);
}

fn awaitColdReady(state: *LinkState, size: u32, deadline_ms: u64) !void {
    while (true) {
        const gone = state.worker_gone.load(.acquire);
        var all = true;
        for (1..size) |k| {
            const code = state.ready_code[k].load(.acquire);
            if (code > 1) return error.DistWorkerLoadFailed;
            if (code == 0) all = false;
        }
        if (gone) return error.DistWorkerLeft;
        if (all) return;
        if (msLeft(deadline_ms) == 0) return error.DistReadyTimeout;
        crt.sleepMs(10);
    }
}

fn idleTimeoutMs() u32 {
    return @max(1, env.intValue(u32, env.get(.MLX_SERVE_DIST_IDLE_MS), 3_000));
}

fn followerExit(state: *LinkState, reason: dist_watchdog.BellFailure) void {
    if (state.aborting.load(.acquire)) return;
    if (state.closing.load(.acquire) or state.releasing.load(.acquire)) return;
    log.err("[dist-watchdog] leader gone: bell {s} (idle timeout {d} ms); this worker exits\n", .{ @tagName(reason), idleTimeoutMs() });
    state.exit(if (reason == .eof) .bell_eof else .bell_timeout, 0, if (reason == .timeout) idleTimeoutMs() else null);
}

/// Writes the `[dist-exit]` record (fsynced), without exiting: for a path that leaves through `main`.
fn noteDistExit(cause: dist_watchdog.ExitCause, peer_rank: ?u32, waited_ms: ?u64) void {
    var buf: [160]u8 = undefined;
    log.err("{s}", .{dist_watchdog.exitRecord(&buf, crt.wallMs(), cause, peer_rank, liveness().step, waited_ms)});
    log.syncFile();
}

pub fn exitReleaseFailed(rank: u32) noreturn {
    distExit(.release_failed, rank, null);
}

pub fn exitSwitchAbort(n: u64) noreturn {
    log.err("[dist-exit] epoch_ms={d} cause=switch_abort n={d} peer=- step={d} waited_ms=-\n", .{ crt.wallMs(), n, liveness().step });
    log.syncFile();
    std.c._exit(1);
}

pub fn exitWorker(err: ?anyerror) noreturn {
    if (err) |e| {
        noteDistExit(.worker_error, 0, null);
        log.err("[dist] worker: {s}\n", .{@errorName(e)});
    }
    std.process.exit(if (err != null) 1 else 0);
}

pub fn switchStallNote(buf: []u8) []const u8 {
    const c = cluster();
    lock(&c.state_mu);
    defer c.state_mu.unlock();
    const state = c.active_state orelse return "";
    if (!state.watchdog.in_flight.load(.acquire)) return "";
    const started = state.watchdog.started_ms.load(.monotonic);
    return std.fmt.bufPrint(buf, "step={d} waited_ms={d} peer={d}", .{ state.watchdog.step.load(.monotonic), @as(u64, @intCast(crt.monotonicMs())) -| started, state.lost_rank.load(.acquire) }) catch "";
}

fn distExit(cause: dist_watchdog.ExitCause, peer_rank: ?u32, waited_ms: ?u64) noreturn {
    noteDistExit(cause, peer_rank, waited_ms);
    std.c._exit(1);
}

/// Rank 0's side of the bells while the group forms: files each connection under the rank it names, a
/// restarted worker's replacing the one before it, until stopped.
const Rendezvous = struct {
    deadline_ms: u64 = std.math.maxInt(u64),
    control: ?*OpenControl = null,
    state: ?*LinkState = null,
    listener: std.c.fd_t,
    size: u32,
    fds: [max_ranks]std.c.fd_t = @splat(Bell.none),
    mu: std.atomic.Mutex = .unlocked,
    stop: std.atomic.Value(bool) = .init(false),

    fn lock(self: *Rendezvous) void {
        while (!self.mu.tryLock()) std.atomic.spinLoopHint();
    }

    fn run(self: *Rendezvous) void {
        while (!self.stop.load(.acquire) and msLeft(self.deadline_ms) > 0) {
            if (self.control) |control| if (control.cancelled.load(.acquire)) return;
            const got = Bell.acceptRank(self.listener, @intCast(@min(100, msLeft(self.deadline_ms))), self.control, self.deadline_ms) orelse continue;
            if (got.rank == 0 or got.rank >= self.size) {
                log.warn("[dist] a bell named rank {d} of a {d}-Mac group; closed\n", .{ got.rank, self.size });
                closeBell(self.control, got.bell.fd);
                continue;
            }
            self.lock();
            defer self.mu.unlock();
            if (self.fds[got.rank] != Bell.none) closeBell(self.control, self.fds[got.rank]);
            self.fds[got.rank] = got.bell.fd;
            if (self.state) |state| state.bells_seen.store(self.seenLocked(), .release);
        }
    }

    fn seenLocked(self: *const Rendezvous) u32 {
        var m: u32 = 0;
        for (1..self.size) |k| if (self.fds[k] != Bell.none) {
            m |= @as(u32, 1) << @intCast(k);
        };
        return m;
    }

    /// Bit `k` set for each worker `k` whose bell has come.
    fn seen(self: *Rendezvous) u32 {
        self.lock();
        defer self.mu.unlock();
        return self.seenLocked();
    }

    /// Whether every worker's bell came within `ms`.
    fn waitAll(self: *Rendezvous, ms: u32) bool {
        const all: u32 = ((@as(u32, 1) << @intCast(self.size)) - 1) & ~@as(u32, 1);
        var waited: u32 = 0;
        while (self.seen() != all) : (waited += 20) {
            if (waited >= ms or self.stop.load(.acquire)) return false;
            if (self.control) |ctrl| if (ctrl.cancelled.load(.acquire)) return false;
            crt.sleepMs(20);
        }
        return true;
    }

    /// The bells, now `link`'s.
    fn handOver(self: *Rendezvous, link: *Link) void {
        self.lock();
        defer self.mu.unlock();
        for (1..self.size) |k| {
            link.bells[k] = .{ .fd = self.fds[k] };
            self.fds[k] = Bell.none;
        }
    }

    fn close(self: *Rendezvous) void {
        for (&self.fds) |*fd| if (fd.* != Bell.none) {
            closeBell(self.control, fd.*);
            fd.* = Bell.none;
        };
        if (self.listener != Bell.none) closeBell(self.control, self.listener);
        self.listener = Bell.none;
    }
};

/// The workers of a `size`-Mac group missing from `mask` (bit `k` = rank `k`), as "2, 3".
fn missingRanks(mask: u32, size: u32, out: []u8) []const u8 {
    var w: std.Io.Writer = .fixed(out);
    for (1..size) |k| {
        if (mask & (@as(u32, 1) << @intCast(k)) != 0) continue;
        if (w.end > 0) w.writeAll(", ") catch break;
        w.print("{d}", .{k}) catch break;
    }
    return w.buffered();
}

/// Zero-timeout polls `Bell.wait` makes before it blocks: some 50 ms.
const spin_polls = 50_000;

const Coordinator = struct { ip4: [4]u8, port: u16 };

fn parseCoordinator(text: []const u8) !Coordinator {
    const colon = std.mem.lastIndexOfScalar(u8, text, ':') orelse return error.DistBadCoordinator;
    var out: Coordinator = .{ .ip4 = undefined, .port = std.fmt.parseInt(u16, text[colon + 1 ..], 10) catch return error.DistBadCoordinator };
    var it = std.mem.splitScalar(u8, text[0..colon], '.');
    for (&out.ip4) |*b| b.* = std.fmt.parseInt(u8, it.next() orelse return error.DistBadCoordinator, 10) catch return error.DistBadCoordinator;
    if (it.next() != null) return error.DistBadCoordinator;
    return out;
}

// ── Wire protocol: fixed int32[8] control messages, then any payload tensor ──

pub const MAGIC = wire.MAGIC;
/// 12: incremental state from a leader-only TP prefill chunk.
const VERSION: i32 = 12;
pub const Msg = wire.Msg;
const OP_FORWARD: i32 = 1;
pub const OP_STOP = wire.OP_STOP;
const OP_MOE: i32 = 3;
pub const OP_H3_JOB = wire.OP_H3_JOB;
pub const OP_H3_STEP = wire.OP_H3_STEP;
pub const OP_H3_END = wire.OP_H3_END;
const OP_DSV41: i32 = 7;
const OP_DSV41_TP: i32 = 9;
/// Tensor parallel: one batched step over several states, then the states' positions and ids.
const OP_DSV41_TP_BATCH: i32 = 10;
/// Tensor parallel: rank 0 dropped a state; the others drop theirs.
const OP_DSV41_TP_FREE: i32 = 11;
/// Tensor parallel: what rank 0 did with a prefix-cache snapshot (`dsv41.SnapOp`), for the others' copies.
const OP_DSV41_TP_SNAP: i32 = 12;
/// Tensor parallel: a snapshot rank 0 read from disk (`dsv41.tpLoadPlan`'s tensors), the worker's copy of its cache.
const OP_DSV41_TP_LOAD: i32 = 13;
/// Tensor parallel, a sharded DSpark draft: its states' gens, positions and first tokens (`dsv41.tpWorkerDraft`).
const OP_DSV41_TP_DRAFT: i32 = 14;
/// Never sent: what a worker's `recvMsg` returns for a `Bell.pipe_byte` (`dsv41.tpWorkerPipe`).
const OP_DSV41_TP_PIPE: i32 = 15;
const OP_DSV41_TP_DRAFT_PIPE: i32 = 16;
const OP_DSV41_TP_VERIFY_PIPE: i32 = 17;
const OP_DSV41_TP_PREFILL: i32 = 18;
pub const OP_RELEASE = wire.OP_RELEASE;

/// `link` as a plugin's split job talks over it (`sdk/util/wire.zig`).
pub fn wireOf(link: *const Link) wire.Link {
    return .{ .ctx = link, .vt = &WireOps.vt };
}

const WireOps = struct {
    const vt: wire.Link.VTable = .{ .send_op = sendOp_, .recv_op = recvOp_, .send_now = sendNow_, .recv_now = recvNow_, .all_gather = allGather_ };

    fn of(ctx: *const anyopaque) *const Link {
        return @ptrCast(@alignCast(ctx));
    }
    fn sendOp_(ctx: *const anyopaque, m: Msg) anyerror!void {
        return sendOp(of(ctx), m);
    }
    fn recvOp_(ctx: *const anyopaque) anyerror!Msg {
        return recvOp(of(ctx));
    }
    fn sendNow_(ctx: *const anyopaque, x: mlx.mlx_array, dst: u32) anyerror!void {
        return sendNow(of(ctx), x, dst);
    }
    fn recvNow_(ctx: *const anyopaque, shape: []const c_int, dtype: mlx.mlx_dtype, src: u32) anyerror!mlx.mlx_array {
        return recvNow(of(ctx), shape, dtype, src);
    }
    fn allGather_(ctx: *const anyopaque, x: mlx.mlx_array) anyerror!mlx.mlx_array {
        return of(ctx).allGather(x);
    }
};

/// A control message to the other rank of a two-Mac group (the rows and pipeline wires pair two Macs only).
pub fn sendOp(link: *const Link, m: Msg) !void {
    return sendMsg(link, m, 1 - link.rank);
}

pub fn recvOp(link: *const Link) !Msg {
    return recvMsg(link, 1 - link.rank);
}

/// `x` sent and complete before this returns.
pub fn sendNow(link: *const Link, x: mlx.mlx_array, dst: u32) !void {
    const r = try link.send(x, dst);
    defer _ = mlx.mlx_array_free(r);
    try evalOne(r);
}

pub fn recvNow(link: *const Link, shape: []const c_int, dtype: mlx.mlx_dtype, src: u32) !mlx.mlx_array {
    return queuedRecv(link, shape, dtype, src);
}

fn sendMsg(link: *const Link, m: Msg, dst: u32) !void {
    if (link.rank == 0) {
        cluster().tp_live = null;
        cluster().tp_spec = null;
        try link.bells[dst].ring();
    }
    const shape = [_]c_int{8};
    const a = mlx.mlx_array_new_data(&m, &shape, 1, .int32);
    defer _ = mlx.mlx_array_free(a);
    const r = try link.send(a, dst);
    defer _ = mlx.mlx_array_free(r);
    try evalOne(r);
}

fn recvMsg(link: *const Link, src: u32) !Msg {
    if (link.rank != 0) if (try link.bells[0].msgOf(try link.bells[0].wait())) |m| return m;
    const r = try link.recv(&.{8}, .int32, src);
    defer _ = mlx.mlx_array_free(r);
    try evalOne(r);
    const d = mlx.mlx_array_data_int32(r) orelse return error.DistBadMessage;
    var m: Msg = undefined;
    @memcpy(&m, d[0..8]);
    return m;
}

fn mib(b: u64) i32 {
    return @intCast(@min(b >> 20, std.math.maxInt(i32)));
}

/// What a worker reports on joining: its memory, then its host name (`name_len` bytes).
const Hello = struct {
    avail_mib: i32,
    total_mib: i32,
    name_len: i32 = 0,

    fn here(name_len: usize) Hello {
        return .{ .avail_mib = mib(status.getAvailableMemBytes()), .total_mib = mib(status.getTotalMemBytes()), .name_len = @intCast(name_len) };
    }
    fn msg(h: Hello) Msg {
        return .{ MAGIC, VERSION, h.avail_mib, h.total_mib, h.name_len, 0, 0, 0 };
    }
    fn parse(m: Msg) !Hello {
        if (m[0] != MAGIC or m[1] != VERSION) return error.DistProtocolMismatch;
        if (m[4] < 0 or m[4] > 64) return error.DistBadMessage;
        return .{ .avail_mib = m[2], .total_mib = m[3], .name_len = m[4] };
    }
};

/// What rank 0 asks a worker to load: the model by folder name, then its `modelIdentity`,
/// checked against the worker's own copy, and its part of it.
const Plan = struct {
    mode: Mode,
    span: LayerSpan,
    hidden: i32,
    experts: i32,
    /// DeepSeek-V4.1 tensor: every rank holds its slice of the DSpark stages and drafts (`Shard.stages`).
    stages: bool = false,

    fn msg(p: Plan, name_len: usize) Msg {
        return .{ MAGIC, @backingInt(p.mode), @intCast(p.span.start), @intCast(p.span.end), @intCast(p.span.total), @intCast(name_len), p.hidden, p.experts };
    }
    fn parse(m: Msg) !Plan {
        if (m[0] != MAGIC) return error.DistProtocolMismatch;
        return .{ .mode = @fromBackingInt(@intCast(m[1])), .span = .{ .start = @intCast(m[2]), .end = @intCast(m[3]), .total = @intCast(m[4]) }, .hidden = m[6], .experts = m[7] };
    }
    fn matches(p: Plan, cfg: *const model_mod.ModelConfig) bool {
        return cfg.num_hidden_layers == p.span.total and cfg.hidden_size == p.hidden and cfg.num_experts == p.experts;
    }
    /// Sends the plan for `model_dir` to every worker, each its whole plan before the next; returns that copy's
    /// `modelIdentity`. A worker takes its part from its own rank.
    fn send(p: Plan, link: *const Link, model_dir: []const u8) !u64 {
        const g = cluster().group.?;
        const id = try modelIdentity(g.io, g.allocator, model_dir);
        const name = std.fs.path.basename(model_dir);
        for (1..link.size) |k| link.state.?.ready_code[k].store(0, .release);
        for (1..link.size) |k| {
            const dst: u32 = @intCast(k);
            try sendMsg(link, p.msg(name.len), dst);
            try sendBytes(link, name, dst);
            try sendMsg(link, .{ MAGIC, @bitCast(@as(u32, @truncate(id))), @bitCast(@as(u32, @truncate(id >> 32))), @intFromBool(p.stages), 0, 0, 0, 0 }, dst);
        }
        return id;
    }
};

/// One copy of a model: its `config.json` and the path and size of every weight file under
/// it, so two folders of one name in another quantization or revision never pair.
pub fn modelIdentity(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) !u64 {
    if (model_dir.len == 0) return error.BadModelDir;
    var dir = try std.Io.Dir.cwd().openDir(io, model_dir, .{ .iterate = true });
    defer dir.close(io);
    const cfg = try dir.readFileAlloc(io, "config.json", allocator, .limited(16 * 1024 * 1024));
    defer allocator.free(cfg);
    var id = std.hash.Wyhash.hash(0, cfg);
    var walker = try dir.walk(allocator);
    defer walker.deinit();
    while (try walker.next(io)) |e| {
        if ((e.kind != .file and e.kind != .sym_link) or !(std.mem.endsWith(u8, e.basename, ".safetensors") or std.mem.endsWith(u8, e.basename, ".bin"))) continue;
        const st = try e.dir.statFile(io, e.basename, .{});
        // Summed, so the walk's order does not matter.
        id +%= std.hash.Wyhash.hash(st.size, e.path);
    }
    return id;
}

/// Worker failure codes in its READY message (0 = ready).
const NOT_FOUND: i32 = 2;
const MISMATCH: i32 = 3;
const LOAD_FAILED: i32 = 1;

/// The seam that splits `layers` by free memory, on a full-attention group boundary.
pub fn chooseSplit(layers: u32, interval: u32, avail0: u64, avail1: u64) u32 {
    const g = @max(interval, 1);
    const share = @as(f64, @floatFromInt(avail0)) / @as(f64, @floatFromInt(@max(avail0 + avail1, 1)));
    const raw: u32 = @intFromFloat(@round(@as(f64, @floatFromInt(layers)) * share / @as(f64, @floatFromInt(g))));
    return std.math.clamp(raw, 1, layers / g - 1) * g;
}

/// One worker as rank 0 knows it: its hello, its host name and what its load left resident (its READY message).
const Remote = struct {
    hello: Hello = .{ .avail_mib = 0, .total_mib = 0 },
    /// The worker's host name, as its LAN sharing names it.
    name_buf: [64]u8 = undefined,
    name_len: usize = 0,
    resident_mib: u32 = 0,

    fn name(r: *const Remote) []const u8 {
        return r.name_buf[0..r.name_len];
    }
};

/// Rank 0's handle on the workers: the group, the split, and how far the worker's caches have advanced.
pub const Peer = struct {
    link: Link,
    mode: Mode,
    /// The model the group serves, by folder name (`--dist-model`, else `--model`).
    model: []const u8,
    split: ?u32,
    /// Each worker by rank (entry 0 unused). A pair's worker is rank 1, which the pipeline and rows wires talk to.
    remotes: [max_ranks]Remote = @splat(.{}),
    /// Set by the first load of `model` (`planFor`), reused by every reload.
    span: LayerSpan = .{ .start = 0, .end = 0, .total = 0 },
    shard: ?Shard = null,
    planned: bool = false,
    /// The `modelIdentity` of the copy the plan was made for.
    plan_id: u64 = 0,
    /// DeepSeek-V4.1: the worker's layers in the middle or tail of the trunk.
    remote: ?LayerSpan = null,
    ready: bool = false,
    /// A worker refused or failed its load: it has left, and nothing waits on it again.
    failed: bool = false,
    /// Exchanges with the worker since the group formed (`/props` cluster).
    exchanges: u64 = 0,
    /// Tokens the worker's caches hold. Every forward must continue them or start over at 0.
    peer_pos: usize = 0,
    /// Rows mode: each Mac's time for one calibration pass, and the link's measured rate.
    host_calib_us: u64 = 0,
    worker_calib_us: u64 = 0,
    link_bytes_per_us: f64 = 0,
    /// `link` as the rows job talks over it (`rowsPair`).
    split_link: wire.Link = undefined,

    /// A pair's worker (rank 1).
    fn worker(self: *const Peer) *const Remote {
        return &self.remotes[1];
    }

    pub fn workerName(self: *const Peer) []const u8 {
        return self.worker().name();
    }

    /// Hands the stream to the worker and returns its output stream. Lazy: the sends are
    /// queued (async) ahead of the receive so the comm stream runs them in that order.
    pub fn hop(self: *Peer, h: mlx.mlx_array, offset: usize) !mlx.mlx_array {
        if (!self.ready) try self.awaitReady();
        const sh = mlx.getShape(h);
        log.debug("[dist] hop at {d}: {d}x{d} rows (worker holds {d})\n", .{ offset, sh[0], sh[1], self.peer_pos });
        if (offset != 0 and offset != self.peer_pos) {
            log.err("[dist] forward at {d} but the worker holds {d} tokens: its caches cannot follow this reuse\n", .{ offset, self.peer_pos });
            return error.DistPeerDiverged;
        }
        try self.sendOrdered(.{ OP_FORWARD, sh[0], sh[1], @intCast(offset & 0x7fffffff), @intCast(offset >> 31), sh[2], @backingInt(mlx.mlx_array_dtype(h)), 0 }, &.{h});
        self.peer_pos = offset + @as(usize, @intCast(sh[1]));
        self.exchanges += 1;
        return queuedRecv(&self.link, sh, mlx.mlx_array_dtype(h), 1);
    }

    /// Tensor mode: one MoE layer's input and router logits to the worker.
    pub fn sendMoe(self: *Peer, x: mlx.mlx_array, logits: mlx.mlx_array, layer: u32) !void {
        if (!self.ready) try self.awaitReady();
        const xs = mlx.getShape(x);
        const ls = mlx.getShape(logits);
        self.exchanges += 1;
        try self.sendOrdered(.{ OP_MOE, xs[0], xs[1], xs[2], ls[ls.len - 1], @backingInt(mlx.mlx_array_dtype(x)), @intCast(layer), @backingInt(mlx.mlx_array_dtype(logits)) }, &.{ x, logits });
    }

    /// The worker's partial for the layer input `x`: its shape and dtype.
    pub fn recvLike(self: *Peer, x: mlx.mlx_array) !mlx.mlx_array {
        return queuedRecv(&self.link, mlx.getShape(x), mlx.mlx_array_dtype(x), 1);
    }

    /// Header, then each payload, chained so the comm stream runs them in order, and
    /// queued now (async) so a receive built after them cannot overtake them.
    fn sendOrdered(self: *Peer, hdr_host: Msg, payload: []const mlx.mlx_array) !void {
        const shape = [_]c_int{8};
        const hdr = mlx.mlx_array_new_data(&hdr_host, &shape, 1, .int32);
        defer _ = mlx.mlx_array_free(hdr);
        try self.link.bells[1].ring();
        var last = try self.link.send(hdr, 1);
        defer _ = mlx.mlx_array_free(last);
        for (payload) |p| {
            const ordered = try dependsOn(p, last);
            defer _ = mlx.mlx_array_free(ordered);
            const sent = try self.link.send(ordered, 1);
            _ = mlx.mlx_array_free(last);
            last = sent;
        }
        const v = mlx.mlx_vector_array_new_data(&[_]mlx.mlx_array{last}, 1);
        defer _ = mlx.mlx_vector_array_free(v);
        try mlx.check(mlx.mlx_async_eval(v));
    }

    /// The worker holds its part of the copy the plan was made for: another folder of the same
    /// name (another quantization or revision) would pair this Mac's part with the wrong weights.
    fn samePlannedCopy(self: *const Peer, model_dir: []const u8) !void {
        const g = cluster().group.?;
        if (try modelIdentity(g.io, g.allocator, model_dir) == self.plan_id) return;
        log.err("[dist] {s} is not the copy the worker loaded; restart the server to pair with it\n", .{model_dir});
        return error.DistModelMismatch;
    }

    /// Every worker's READY, bounded by the load deadline.
    fn awaitReady(self: *Peer) !void {
        const seconds = env.intValue(u32, env.get(.MLX_SERVE_DIST_LOAD_DEADLINE_S), 900);
        return self.awaitReadyUntil(self.link.state.?, @as(u64, @intCast(crt.monotonicMs())) + @as(u64, seconds) * 1000);
    }

    fn awaitReadyUntil(self: *Peer, state: *LinkState, deadline_ms: u64) !void {
        if (self.failed) return error.DistWorkerLoadFailed;
        awaitColdReady(state, self.link.size, deadline_ms) catch |err| {
            self.failed = true;
            for (1..self.link.size) |k| {
                const code = state.ready_code[k].load(.acquire);
                if (code <= 1) continue;
                switch (code - 1) {
                    NOT_FOUND => log.err("[dist] worker rank {d} has no model named {s} in its model folders\n", .{ k, self.model }),
                    MISMATCH => log.err("[dist] worker rank {d}'s copy of {s} is not the same model (quantization, revision or geometry)\n", .{ k, self.model }),
                    else => log.err("[dist] worker rank {d} failed to load its part of {s}\n", .{ k, self.model }),
                }
            }
            return err;
        };
        for (1..self.link.size) |k| {
            const resident = state.ready_resident_mib[k].load(.monotonic);
            self.remotes[k].resident_mib = resident;
            if (self.mode == .rows) self.worker_calib_us = state.ready_calib_us[k].load(.monotonic);
        }
        activateHeartbeat(state);
        self.ready = true;
    }
};

/// A receive completed on the host before anything reads it. A lazy one lets a GPU command
/// buffer wait on it, and MLX signals a buffer's events only when the whole buffer ends: the
/// buffer that computes the payload can then also hold the wait for the reply to it, which
/// deadlocks until Metal's watchdog kills it. Done here, the receive also pairs with the
/// peer's send even when nobody reads it (a warmup that drops its output).
fn queuedRecv(link: *const Link, shape: []const c_int, dtype: mlx.mlx_dtype, src: u32) !mlx.mlx_array {
    _ = link.state.?.recvs.fetchAdd(1, .acq_rel);
    defer _ = link.state.?.recvs.fetchSub(1, .acq_rel);
    const r = try link.recv(shape, dtype, src);
    errdefer _ = mlx.mlx_array_free(r);
    try evalOne(r);
    return r;
}

/// `x` that evaluates only after `dep`.
fn dependsOn(x: mlx.mlx_array, dep: mlx.mlx_array) !mlx.mlx_array {
    const ins = mlx.mlx_vector_array_new_data(&[_]mlx.mlx_array{x}, 1);
    defer _ = mlx.mlx_vector_array_free(ins);
    const deps = mlx.mlx_vector_array_new_data(&[_]mlx.mlx_array{dep}, 1);
    defer _ = mlx.mlx_vector_array_free(deps);
    var out = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(out);
    try mlx.check(mlx.mlx_depends(&out, ins, deps));
    var r = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_vector_array_get(&r, out, 0));
    return r;
}

const Liveness = struct { worker_gone: bool = false, lost_rank: u32 = 0, bells_seen: u32 = 0, step: u64 = 0 };

fn readLiveness(state: *LinkState) Liveness {
    return .{ .worker_gone = state.worker_gone.load(.acquire), .lost_rank = state.lost_rank.load(.acquire), .bells_seen = state.bells_seen.load(.acquire), .step = state.watchdog.step.load(.monotonic) };
}

const SwitchGroup = struct { model: []const u8, link: *Link };

/// Process-wide under `-Dunits` (sdk/util/shared.zig): one copy in the host.
const ClusterState = struct {
    switch_group: ?SwitchGroup = null,
    last_liveness: Liveness = .{},
    inference_thread: ?std.Thread.Id = null,
    group: ?Group = null,
    backends_not_cleared: bool = false,
    joined: std.atomic.Value(bool) = .init(false),
    joined_link: Link = undefined,
    last_job_buf: [160]u8 = undefined,
    last_job_len: usize = 0,
    /// The inference thread writes the note while `/props` reads it.
    last_job_mu: std.atomic.Mutex = .unlocked,
    last_split: [2]u32 = .{ 0, 0 },
    peer_slot: ?Peer = null,
    /// Round trips since the last timing line, and their totals (µs): the whole exchange and the
    /// worker's share of it.
    timing: struct { n: u32 = 0, round_us: u64 = 0, worker_us: u64 = 0 } = .{},
    /// DeepSeek-V4.1 tensor: the link the forward's sums run over (either rank), and the sums built since the last step
    /// began, which every rank must match step by step.
    tp_link: ?*const Link = null,
    tp_sums: u32 = 0,
    /// Values summed this step: the ranks agree on it as on the count, or their sums pair up wrong.
    tp_values: usize = 0,
    tp_last: struct { p0: usize = 0, rows: usize = 0 } = .{},
    /// Rank 0: the state the workers last stepped, while no other message has reached them since (`sendMsg` clears it).
    tp_live: ?TpLive = null,
    tp_spec: ?TpSpec = null,
    /// What each worker (by rank) said about the group's model when it joined: 0 has it, else NOT_FOUND or
    /// MISMATCH (-1 = not asked: no bell connection).
    copies: [max_ranks]std.atomic.Value(i32) = @splat(.init(-1)),
    /// Each worker's free memory as it answered on joining (MiB), for the load bill.
    free_mib: [max_ranks]std.atomic.Value(u32) = @splat(.init(0)),
    active_state: ?*LinkState = null,
    state_mu: std.atomic.Mutex = .unlocked,
};
pub const cluster_shared = @import("../../../sdk/util/shared.zig").Shared(ClusterState, "cluster_state", .{});
inline fn cluster() *ClusterState {
    return cluster_shared.get();
}

pub fn bindInferenceThread() void {
    cluster().inference_thread = std.Thread.getCurrentId();
}

pub fn setSwitchGroup(link: *Link, model: []const u8) void {
    const c = cluster();
    lock(&c.state_mu);
    defer c.state_mu.unlock();
    std.debug.assert(c.switch_group == null);
    c.switch_group = .{ .model = model, .link = link };
}

fn switchShardFor(model: []const u8) ?Shard {
    const c = cluster();
    lock(&c.state_mu);
    defer c.state_mu.unlock();
    const g = c.switch_group orelse return null;
    if (!samePath(g.model, model)) return null;
    return .{ .rank = g.link.rank, .size = g.link.size };
}

pub fn switchHolds(model: []const u8) bool {
    return switchShardFor(model) != null;
}

fn beginDistStep() void {
    const c = cluster();
    lock(&c.state_mu);
    defer c.state_mu.unlock();
    const state = c.active_state orelse return;
    state.watchdog.begin(@intCast(crt.monotonicMs()));
}

/// Advances liveness at the top of an inference iteration.
pub fn tickInferenceLoop() void {
    const c = cluster();
    lock(&c.state_mu);
    defer c.state_mu.unlock();
    const state = c.active_state orelse return;
    const now: u64 = @intCast(crt.monotonicMs());
    state.loop_tick_ms.store(now, .release);
    if (c.tp_link == null) return;
    if (state.watchdog.in_flight.load(.acquire)) state.watchdog.progress(now) else state.watchdog.begin(now);
}

/// Ends the watched iteration before the inference thread waits.
pub fn parkInferenceLoop() void {
    const c = cluster();
    lock(&c.state_mu);
    defer c.state_mu.unlock();
    const state = c.active_state orelse return;
    state.watchdog.done();
    state.loop_tick_ms.store(@intCast(crt.monotonicMs()), .release);
}

fn activateHeartbeat(state: *LinkState) void {
    state.loop_tick_ms.store(0, .release);
    state.heartbeat_active.store(true, .release);
}

const Heartbeat = struct { state: *LinkState, bells: [max_ranks]Bell, size: u32, interval_ms: u32 };

/// Sleeps `ms` in short steps; true once the link's watchdog is told to stop, so `Link.deinit` joins within ~50 ms.
fn sleepUnlessStopped(c: *LinkState, ms: u32) bool {
    var left = ms;
    while (left > 0) {
        if (c.closing.load(.acquire) or c.releasing.load(.acquire) or c.watchdog_stop.load(.acquire)) return true;
        const step = @min(left, 50);
        crt.sleepMs(step);
        left -= step;
    }
    return c.closing.load(.acquire) or c.releasing.load(.acquire) or c.watchdog_stop.load(.acquire);
}

fn teardownDeadline(state: *LinkState, deadline: u64) void {
    while (!state.teardown_done.load(.acquire)) {
        if (msLeft(deadline) == 0) return state.exit(.teardown_timeout, null, null);
        crt.sleepMs(@min(msLeft(deadline), 25));
    }
}

fn abortFallback(link: Link) void {
    const state = link.state.?;
    if (sleepUnlessStopped(state, 2000)) return;
    lock(&state.exit_mu);
    defer state.exit_mu.unlock();
    if (state.closing.load(.acquire) or state.releasing.load(.acquire)) return;
    _ = std.c.shutdown(link.bells[0].fd, std.c.SHUT.RDWR);
}

fn heartbeatLoop(h: Heartbeat) void {
    while (true) {
        if (sleepUnlessStopped(h.state, h.interval_ms)) return;
        if (!h.state.heartbeat_active.load(.acquire)) continue;
        const now_ms: u64 = @intCast(crt.monotonicMs());
        if (!dist_watchdog.loopAlive(now_ms, h.state.loop_tick_ms.load(.acquire), h.state.watchdog.in_flight.load(.acquire), 1_000)) continue;
        for (1..h.size) |rank| if (h.bells[rank].open()) {
            h.bells[rank].writeAll(&.{Bell.heartbeat_byte}) catch {};
        };
    }
}

fn watchdogLoop(c: *LinkState, deadline_ms: u64, leader: bool) void {
    while (true) {
        if (sleepUnlessStopped(c, 250)) return;
        const now_ms: u64 = @intCast(crt.monotonicMs());
        const last_tick_ms = c.loop_tick_ms.load(.acquire);
        const in_flight = c.watchdog.in_flight.load(.acquire);
        const ready = c.heartbeat_active.load(.acquire);
        if (leader and ready and !dist_watchdog.loopAlive(now_ms, last_tick_ms, in_flight, @as(u64, idleTimeoutMs()))) {
            log.err("[dist-watchdog] leader loop stalled: waited_ms={d}; this rank exits\n", .{now_ms -| last_tick_ms});
            return c.exit(.self_check, null, now_ms -| last_tick_ms);
        }
        const waited = c.watchdog.stalledMs(now_ms, deadline_ms) orelse continue;
        log.err("[dist-watchdog] stall: step={d} waited_ms={d}; this rank exits\n", .{ c.watchdog.step.load(.monotonic), waited });
        return c.exit(.stall, null, waited);
    }
}

/// Starts the open link's watchdog (and, on rank 0, its heartbeat). Once per link: `Link.deinit` stops and joins
/// them, so a group formed again in the same process gets its own.
fn startWatchdog(link: *const Link) void {
    const c = link.state.?;
    if (c.watchdog_started.swap(true, .acq_rel)) return;
    c.watchdog_stop.store(false, .release);
    const deadline: u64 = env.intValue(u32, env.get(.MLX_SERVE_DIST_STALL_MS), 30_000);
    c.watchdog_threads[0] = std.Thread.spawn(.{ .stack_size = 1 << 16 }, watchdogLoop, .{ c, deadline, link.rank == 0 and link.size > 1 }) catch {
        log.err("[dist-watchdog] cannot start watchdog thread\n", .{});
        return c.exit(.watchdog_thread, null, null);
    };
    if (link.rank == 0) {
        const interval = @max(1, idleTimeoutMs() / 3);
        c.watchdog_threads[1] = std.Thread.spawn(.{ .stack_size = 1 << 16 }, heartbeatLoop, .{Heartbeat{ .state = c, .bells = link.bells, .size = link.size, .interval_ms = interval }}) catch {
            log.err("[dist-watchdog] cannot start heartbeat thread\n", .{});
            return c.exit(.heartbeat_thread, null, null);
        };
    }
}

fn stopWatchdog(c: *LinkState) void {
    c.watchdog_stop.store(true, .release);
    for (&c.watchdog_threads) |*slot| if (slot.*) |t| {
        t.join();
        slot.* = null;
    };
    c.watchdog_started.store(false, .release);
}

fn startWatchers(link: *const Link) !void {
    const state = link.state.?;
    if (state.watcher != null) return;
    if (link.rank != 0 and !state.switch_open and link.size <= 2) return;
    state.watcher = if (link.rank == 0)
        try std.Thread.spawn(.{ .stack_size = 1 << 16 }, watchWorker, .{link.*})
    else
        try std.Thread.spawn(.{ .stack_size = 1 << 16 }, watchHost, .{link.*});
}

/// The group this server leads (`--dist`): set at startup, while a thread waits
/// for the worker to join; the handshake runs when the group's model loads.
const Group = struct {
    io: std.Io,
    allocator: std.mem.Allocator,
    mode: Mode,
    model: []const u8,
    split: ?u32,
    /// Macs in the group, rank 0 included.
    size: u32 = 2,
    /// How long after startup a load of the group model that finds a worker missing reads as the group still forming
    /// (0: missing at once, a pair's way). Either way the load fails at once: the inference thread never waits.
    join_timeout_s: u32 = 0,
    /// When the server started waiting for its workers (µs, monotonic).
    started_us: i64 = 0,
};

pub const JoinState = enum { joined, forming, late };

/// Whether the group has joined, is still forming `waited_ms` after startup, or is past its join timeout.
fn joinState(joined_now: bool, waited_ms: u64, timeout_s: u32) JoinState {
    if (joined_now) return .joined;
    return if (waited_ms < @as(u64, timeout_s) * 1000) .forming else .late;
}
/// Each worker's answer about the group's model, by rank.
fn loadCopies() [max_ranks]i32 {
    var out: [max_ranks]i32 = undefined;
    for (&out, &cluster().copies) |*o, *c| o.* = c.load(.acquire);
    return out;
}

/// The first worker that lacks the group's model, by rank, and why (NOT_FOUND or MISMATCH).
fn firstRefusal(copies: []const i32, size: u32) ?struct { rank: u32, code: i32 } {
    for (1..size) |k| if (copies[k] == NOT_FOUND or copies[k] == MISMATCH) return .{ .rank = @intCast(k), .code = copies[k] };
    return null;
}

/// The `/props` "cluster" object (with its leading comma), empty when this server is
/// not rank 0 of a group: what the app draws the split from.
fn liveness() Liveness {
    const c = cluster();
    lock(&c.state_mu);
    defer c.state_mu.unlock();
    return if (c.active_state) |state| readLiveness(state) else c.last_liveness;
}

fn hasJoined() bool {
    return cluster().joined.load(.acquire) and !liveness().worker_gone;
}

pub fn propsJson(allocator: std.mem.Allocator) ![]u8 {
    if (peer()) |p| return clusterJson(allocator, p);
    const g = cluster().group orelse return allocator.dupe(u8, "");
    var mb: [64]u8 = undefined;
    return std.fmt.allocPrint(allocator, ",\"cluster\":{{\"mode\":\"{s}\",\"model\":{f},\"joined\":{},\"planned\":false,\"worker_ready\":false,\"worker_model\":\"{s}\",\"worker_gone\":{},\"size\":{d},\"missing_ranks\":\"{s}\",\"lost_rank\":{d},\"jaccl_deadline\":{}}}", .{
        @tagName(g.mode),                                 std.json.fmt(std.fs.path.basename(g.model), .{}), hasJoined(),                           workerCopyName(), liveness().worker_gone, g.size,
        missingRanks(liveness().bells_seen, g.size, &mb), liveness().lost_rank,                             mlx.jacclFormationDeadlineAvailable(),
    });
}

/// The `/v1/models` "machine" fields of a Thunderbolt pair this server leads (with a leading
/// comma), empty when it leads none: its model and whether the other Mac has joined, so that
/// Mac's app can offer to join it.
pub fn pairFields(allocator: std.mem.Allocator) ![]u8 {
    const g = cluster().group orelse return allocator.dupe(u8, "");
    return pairFieldsFor(allocator, std.fs.path.basename(g.model), hasJoined());
}

fn pairFieldsFor(allocator: std.mem.Allocator, model: []const u8, has_joined: bool) ![]u8 {
    return std.fmt.allocPrint(allocator, ",\"leads\":{f},\"joined\":{}", .{ std.json.fmt(model, .{}), has_joined });
}

fn clusterJson(allocator: std.mem.Allocator, p: *const Peer) ![]u8 {
    const sp = p.span;
    while (!cluster().last_job_mu.tryLock()) std.atomic.spinLoopHint();
    defer cluster().last_job_mu.unlock();
    // Rows mode: the worker's speed relative to this Mac and the link, once measured.
    const speed: f64 = if (p.worker_calib_us > 0 and p.host_calib_us > 0) @as(f64, @floatFromInt(p.host_calib_us)) / @as(f64, @floatFromInt(p.worker_calib_us)) else 0;
    // Every worker by rank; the pair's fields above them stay rank 1's.
    var workers: std.Io.Writer.Allocating = .init(allocator);
    defer workers.deinit();
    const copies = loadCopies();
    for (1..p.link.size) |k| {
        const r = &p.remotes[k];
        try workers.writer.print("{s}{{\"rank\":{d},\"name\":{f},\"free_mib\":{d},\"total_mib\":{d},\"resident_mib\":{d},\"model\":\"{s}\"}}", .{ if (k > 1) "," else "", k, std.json.fmt(r.name(), .{}), r.hello.avail_mib, r.hello.total_mib, r.resident_mib, copyName(copies[k]) });
    }
    const w = p.worker();
    return std.fmt.allocPrint(allocator, ",\"cluster\":{{\"mode\":\"{s}\",\"model\":{f},\"joined\":{},\"planned\":{},\"worker_ready\":{},\"layers\":{d},\"here\":[{d},{d}],\"worker\":[{d},{d}],\"expert_shards\":{d},\"worker_free_mib\":{d},\"worker_total_mib\":{d},\"exchanges\":{d},\"worker_name\":{f},\"worker_speed\":{d:.2},\"link_gbps\":{d:.1},\"last_job\":{f},\"split_rows\":[{d},{d}],\"worker_model\":\"{s}\",\"worker_gone\":{},\"size\":{d},\"lost_rank\":{d},\"jaccl_deadline\":{},\"workers\":[{s}]}}", .{
        @tagName(p.mode),                       std.json.fmt(std.fs.path.basename(p.model), .{}),                     hasJoined(),                      p.planned,               p.ready,           sp.total,               sp.start,                          sp.end,
        if (p.mode == .pipeline) sp.end else 0, sp.total,                                                             if (p.shard) |sh| sh.size else 0, w.hello.avail_mib,       w.hello.total_mib, p.exchanges,            std.json.fmt(p.workerName(), .{}), speed,
        p.link_bytes_per_us / 1000,             std.json.fmt(cluster().last_job_buf[0..cluster().last_job_len], .{}), cluster().last_split[0],          cluster().last_split[1], workerCopyName(),  liveness().worker_gone, p.link.size,                       liveness().lost_rank,
        mlx.jacclFormationDeadlineAvailable(),  workers.written(),
    });
}

/// `/props` cluster "worker_model": whether the workers have the group's model (the first that lacks it decides).
fn workerCopyName() []const u8 {
    const copies = loadCopies();
    const size = if (cluster().group) |g| g.size else 2;
    if (firstRefusal(&copies, size)) |r| return copyName(r.code);
    return copyName(copies[1]);
}

fn copyName(code: i32) []const u8 {
    return switch (code) {
        0 => "ok",
        NOT_FOUND => "missing",
        MISMATCH => "different",
        else => "unknown",
    };
}

/// What the last video job did with the worker (`/props` cluster "last_job", and its rows
/// on each Mac, zeros when it ran here alone).
fn noteJob(split: [2]u32, line: []const u8) void {
    while (!cluster().last_job_mu.tryLock()) std.atomic.spinLoopHint();
    defer cluster().last_job_mu.unlock();
    cluster().last_split = split;
    const n = if (line.len <= cluster().last_job_buf.len) line.len else 0;
    @memcpy(cluster().last_job_buf[0..n], line[0..n]);
    cluster().last_job_len = n;
}

/// Rank 0 at shutdown: releases the worker. A peer that vanishes mid-protocol crashes the
/// other side inside Apple's RDMA library, so the group always ends with this message.
pub fn shutdown() void {
    const p = peer() orelse return;
    if (!p.ready) return;
    // The stopped workers post no more receives: a later message (the model's unload) would wait forever.
    p.ready = false;
    // Every worker gets its stop, even past one it did not reach.
    var stopped: u32 = 0;
    for (1..p.link.size) |k| {
        sendMsg(&p.link, .{ OP_STOP, 0, 0, 0, 0, 0, 0, 0 }, @intCast(k)) catch |err| {
            var eb: [512]u8 = undefined;
            log.warn("[dist] the stop did not reach worker rank {d} ({s}: {s})\n", .{ k, @errorName(err), mlx.takeError(&eb) orelse "" });
            continue;
        };
        stopped += 1;
    }
    if (stopped > 0) log.info("[dist] {d} of {d} workers released\n", .{ stopped, p.link.size - 1 });
}

/// Rank 0, on the inference thread, when the group's model unloads here: every worker frees its
/// part too and waits for the next plan, which the model's next load sends.
pub fn release() void {
    const p = peer() orelse return;
    if (!p.ready) return;
    for (1..p.link.size) |k| sendMsg(&p.link, .{ OP_RELEASE, 0, 0, 0, 0, 0, 0, 0 }, @intCast(k)) catch |err| {
        log.warn("[dist] the release did not reach worker rank {d} ({s})\n", .{ k, @errorName(err) });
        return;
    };
    p.planned = false;
    p.ready = false;
    p.remote = null;
    p.shard = null;
    p.peer_pos = 0;
    log.info("[dist] {s} unloaded: the worker released its part\n", .{std.fs.path.basename(p.model)});
}

/// Rank 0's peer, once `connect` has run.
pub fn peer() ?*Peer {
    return if (cluster().peer_slot) |*p| p else null;
}

/// What this rank holds of `model_dir`'s model when it is the group's model: on the first
/// load the split is chosen and the worker told to load its part; reloads reuse it.
pub const Part = union(enum) { span: LayerSpan, shard: Shard, remote: LayerSpan };

pub fn planFor(model_dir: []const u8, config: *const model_mod.ModelConfig) !?Part {
    if (switchShardFor(model_dir)) |shard| {
        try groupFits(.tensor, config, shard.size);
        var sh = shard;
        sh.stages = dsv41DraftShard(config, sh.size);
        return .{ .shard = sh };
    }
    if (!serves(model_dir)) return null;
    const p = try join();
    if (p.planned) try p.samePlannedCopy(model_dir) else try plan(p, config, model_dir);
    if (p.remote) |r| return .{ .remote = r };
    return if (p.shard) |sh| .{ .shard = sh } else .{ .span = p.span };
}

/// What this Mac holds of `whole` resident bytes of `model_dir` when its pair serves it: its run
/// of layers, chosen as the plan chooses it (or the plan itself once made); `whole` otherwise.
pub fn hostShare(model_dir: []const u8, config: *const model_mod.ModelConfig, whole: u64) u64 {
    if (switchShardFor(model_dir)) |sh| return whole / sh.size + whole / 10;
    if (!holds(model_dir)) return whole;
    const g = cluster().group.?;
    const n: u64 = config.num_hidden_layers;
    if (n == 0) return whole;
    if (peer()) |p| if (p.planned) return if (p.mode == .tensor) whole / g.size + whole / 10 else whole * p.span.len() / n;
    return shareFor(g, config, whole, status.getAvailableMemBytes() >> 20, cluster().free_mib[1].load(.acquire));
}

fn shareFor(g: Group, config: *const model_mod.ModelConfig, whole: u64, mine: u64, theirs: u64) u64 {
    const n: u64 = config.num_hidden_layers;
    if (theirs == 0) return whole;
    if (config.isDsv41()) {
        if (g.mode == .tensor) return whole / g.size + whole / 10;
        const r = dsv41_mod.pairRange(config, mine, theirs) orelse return whole;
        return whole * (n - (r[1] - r[0])) / n;
    }
    if (!config.isQwen4()) return whole;
    return switch (g.mode) {
        .tensor => whole / g.size + whole / 10,
        .rows => whole,
        .pipeline => whole * (g.split orelse chooseSplit(@intCast(n), config.full_attention_interval, mine, theirs)) / n,
    };
}

/// Whether a Thunderbolt pair holds `model_dir`: the worker joined, so no other load unloads it
/// to make room (an explicit unload still does, on both Macs: `release`).
pub fn holds(model_dir: []const u8) bool {
    return switchHolds(model_dir) or (hasJoined() and serves(model_dir));
}

/// Whether `model_dir` is the model this server's group serves.
pub fn serves(model_dir: []const u8) bool {
    if (switchHolds(model_dir)) return true;
    if (comptime builtin.os.tag != .macos) return false; // a cluster exists only on macOS
    const g = cluster().group orelse return false;
    return samePath(g.model, model_dir);
}

/// One folder however it is spelled (symlinks, `..`, a trailing slash); a path that does not
/// resolve matches only itself. A folder elsewhere with the same name is another copy.
fn samePath(a: []const u8, b: []const u8) bool {
    if (std.mem.eql(u8, a, b)) return true;
    var za: [std.fs.max_path_bytes]u8 = undefined;
    var zb: [std.fs.max_path_bytes]u8 = undefined;
    var ra: [std.fs.max_path_bytes]u8 = undefined;
    var rb: [std.fs.max_path_bytes]u8 = undefined;
    if (a.len >= za.len or b.len >= zb.len) return false;
    @memcpy(za[0..a.len], a);
    za[a.len] = 0;
    @memcpy(zb[0..b.len], b);
    zb[b.len] = 0;
    const pa = std.c.realpath(@ptrCast(&za), &ra) orelse return false;
    const pb = std.c.realpath(@ptrCast(&zb), &rb) orelse return false;
    return std.mem.eql(u8, std.mem.span(pa), std.mem.span(pb));
}

/// `planRows` as a split job reads it (`h3_split.Pair`).
fn rowsPair(model_dir: []const u8, hidden: i32, blocks: i32) anyerror!h3_split.Pair {
    const p = try planRows(model_dir, hidden, blocks);
    p.split_link = wireOf(&p.link);
    return .{
        .link = &p.split_link,
        .host_calib_us = &p.host_calib_us,
        .link_bytes_per_us = &p.link_bytes_per_us,
        .exchanges = &p.exchanges,
        .worker_calib_us = p.worker_calib_us,
        .worker_free_mib = @as(u64, @intCast(@max(p.worker().hello.avail_mib, 0))) -| p.worker().resident_mib,
    };
}

/// Rows mode: the worker told to load the group's video model whole, and ready.
fn planRows(model_dir: []const u8, hidden: i32, blocks: i32) !*Peer {
    if (cluster().group.?.size > 2) return error.DistNeedsPair; // the rows wire talks to rank 1 alone
    const p = try join();
    if (!p.planned) {
        p.mode = .rows;
        const pl: Plan = .{ .mode = .rows, .span = .{ .start = 0, .end = @intCast(blocks), .total = @intCast(blocks) }, .hidden = hidden, .experts = 0 };
        p.plan_id = try pl.send(&p.link, model_dir);
        p.planned = true;
    } else try p.samePlannedCopy(model_dir);
    if (!p.ready) try p.awaitReady();
    return p;
}

/// Every worker's hello, read once, in rank order.
fn join() !*Peer {
    if (cluster().peer_slot == null) {
        const g = cluster().group.?;
        try checkJoined(g);
        if (firstRefusal(&loadCopies(), g.size)) |r| {
            log.err("[dist] worker rank {d} {s} {s}: download the same model there, then join again\n", .{ r.rank, if (r.code == MISMATCH) "has another copy of" else "does not have", std.fs.path.basename(g.model) });
            return error.DistWorkerMissingModel;
        }
        const link = &cluster().joined_link;
        var p: Peer = .{ .link = link.*, .mode = g.mode, .model = g.model, .split = g.split };
        for (1..link.size) |k| {
            // A worker sends its hello only once rung: JACCL's send polls until this recv is posted.
            try link.bells[k].ring();
            const r = &p.remotes[k];
            r.hello = try Hello.parse(try recvMsg(link, @intCast(k)));
            if (r.hello.name_len > 0) try recvBytes(link, r.name_buf[0..@intCast(r.hello.name_len)], @intCast(k));
            r.name_len = @intCast(r.hello.name_len);
            log.info("[dist] worker rank {d} {s}: {d} MiB free of {d}\n", .{ k, r.name(), r.hello.avail_mib, r.hello.total_mib });
        }
        cluster().peer_slot = p;
    }
    return &cluster().peer_slot.?;
}

/// Fails the load at once while a worker is missing (`DistWorkerNotJoined`, a 503): as the group forming until the
/// join timeout, then naming the ranks with no bell. The inference thread serves every other model meanwhile.
fn checkJoined(g: Group) !void {
    const waited: u64 = @intCast(@max(@divTrunc(io_util.nowUsMonotonic(g.io) - g.started_us, 1000), 0));
    const state = joinState(hasJoined(), waited, g.join_timeout_s);
    if (state == .joined) return;
    const seen = liveness().bells_seen;
    if (g.size == 2) {
        log.err("[dist] {s} is served with a worker: start `mlx-serve worker` on the other Mac first\n", .{std.fs.path.basename(g.model)});
    } else if (state == .forming) {
        log.info("[dist] {s}: cluster forming ({d} of {d} Macs)\n", .{ std.fs.path.basename(g.model), @popCount(seen) + 1, g.size });
    } else {
        var mb: [64]u8 = undefined;
        const missing = missingRanks(seen, g.size, &mb);
        log.err("[dist] {s} is served by {d} Macs and rank {s} has not joined {d} s after startup: start `mlx-serve worker --rank k --size {d}` there\n", .{ std.fs.path.basename(g.model), g.size, if (missing.len > 0) missing else "(none; the group is still forming)", g.join_timeout_s, g.size });
    }
    return error.DistWorkerNotJoined;
}

/// Whether `size` Macs can serve `config` in `mode`: past a pair only DeepSeek-V4.1's tensor split (every other wire
/// talks to rank 1 alone), and a tensor split only of the dimensions `size` divides. Checked before the plan goes out.
fn groupFits(mode: Mode, config: *const model_mod.ModelConfig, size: u32) !void {
    if (size > 2 and !(mode == .tensor and config.isDsv41())) {
        log.err("[dist] {d} Macs serve only DeepSeek-V4.1 with --dist tensor; {s} with --dist {s} pairs two\n", .{ size, config.model_type, @tagName(mode) });
        return error.DistNeedsPair;
    }
    if (mode != .tensor) return;
    if (config.shardUneven(size)) |dim| {
        log.err("[dist] {d} Macs cannot split {s} of {s} evenly\n", .{ size, dim, config.model_type });
        return error.DistShardUneven;
    }
}

fn plan(self: *Peer, config: *const model_mod.ModelConfig, model_dir: []const u8) !void {
    try groupFits(self.mode, config, cluster().group.?.size);
    if (config.isDsv41()) return planDsv41(self, config, model_dir);
    if (!config.isQwen4()) {
        log.err("[dist] {s}: this architecture ({s}) has no multi-Mac path yet\n", .{ self.model, config.model_type });
        return error.DistArchUnsupported;
    }
    const n = config.num_hidden_layers;
    var pl: Plan = .{ .mode = self.mode, .span = .{ .start = 0, .end = n, .total = n }, .hidden = @intCast(config.hidden_size), .experts = @intCast(config.num_experts) };
    switch (self.mode) {
        .tensor => {
            if (config.num_experts == 0) return error.DistTensorNeedsMoe;
            self.shard = .{ .rank = 0, .size = self.link.size };
            log.info("[dist] tensor: each expert of {s} split across this Mac and the worker\n", .{std.fs.path.basename(self.model)});
        },
        .rows => return error.DistArchUnsupported,
        .pipeline => {
            const mine: u64 = status.getAvailableMemBytes() >> 20;
            const k = self.split orelse chooseSplit(n, config.full_attention_interval, mine, @intCast(self.worker().hello.avail_mib));
            if (k == 0 or k >= n or (config.full_attention_interval != 0 and k % config.full_attention_interval != 0)) return error.DistSplitOffInterval;
            self.span = .{ .start = 0, .end = k, .total = n };
            pl.span = .{ .start = k, .end = n, .total = n };
            log.info("[dist] pipeline: layers 0..{d} of {s} here ({d} MiB free), {d}..{d} on the worker ({d} MiB free of {d})\n", .{ k, std.fs.path.basename(self.model), mine, k, n, self.worker().hello.avail_mib, self.worker().hello.total_mib });
        },
    }
    self.plan_id = try pl.send(&self.link, model_dir);
    self.planned = true;
}

/// DeepSeek-V4.1: the worker takes a run of layers its share of the memory holds, cut where
/// no layer reads another Mac's state (`deepseek_v41.pairRange`); the stream and mix cross twice.
fn planDsv41(self: *Peer, config: *const model_mod.ModelConfig, model_dir: []const u8) !void {
    const n = config.num_hidden_layers;
    switch (self.mode) {
        .pipeline => {},
        .tensor => {
            const stages = dsv41DraftShard(config, self.link.size);
            self.shard = .{ .rank = 0, .size = self.link.size, .stages = stages };
            log.info("[dist] tensor: every expert and attention head of {s} split across {d} Macs; DSpark drafts on {s}\n", .{ std.fs.path.basename(self.model), self.link.size, if (stages) "every rank" else "rank 0 alone" });
            const pl: Plan = .{ .mode = .tensor, .span = .{ .start = 0, .end = n, .total = n }, .hidden = @intCast(config.hidden_size), .experts = @intCast(config.num_experts), .stages = stages };
            self.plan_id = try pl.send(&self.link, model_dir);
            self.planned = true;
            installDsv41Tp(&self.link);
            log.info("[dist] tensor: decode steps {s}\n", .{if (dsv41_mod.tpPipeOn()) "keep their token on the GPU, each rung on the bells alone (DSV41_TP_PIPE=0: an id header per step)" else "each send their id in a header"});
            log.info("[dist] tensor: DSpark drafts and verifies {s}\n", .{if (dsparkPipeWanted(&self.link)) "of the live state rung on the bells, their values after the byte (DSV41_DSPARK_PIPE)" else "each send a header"});
            return;
        },
        .rows => {
            log.err("[dist] {s}: DeepSeek-V4.1 splits by layers or experts (--dist pipeline|tensor)\n", .{self.model});
            return error.DistArchUnsupported;
        },
    }
    const mine: u64 = status.getAvailableMemBytes() >> 20;
    const r = dsv41_mod.pairRange(config, mine, @intCast(self.worker().hello.avail_mib)) orelse return error.DistSplitOffInterval;
    self.remote = .{ .start = @intCast(r[0]), .end = @intCast(r[1]), .total = n };
    self.span = .{ .start = 0, .end = @intCast(n - (r[1] - r[0])), .total = n };
    log.info("[dist] pipeline: layers {d}..{d} of {s} on the worker ({d} MiB free of {d}), the other {d} here ({d} MiB free)\n", .{ r[0], r[1], std.fs.path.basename(self.model), self.worker().hello.avail_mib, self.worker().hello.total_mib, n - (r[1] - r[0]), mine });
    const pl: Plan = .{ .mode = .pipeline, .span = self.remote.?, .hidden = @intCast(config.hidden_size), .experts = @intCast(config.num_experts) };
    self.plan_id = try pl.send(&self.link, model_dir);
    self.planned = true;
    dsv41_mod.clusterHooks().pair_exchange = &dsv41Exchange;
}

pub fn installDsv41Tp(link: *const Link) void {
    cluster().tp_link = link;
    cluster().tp_live = null;
    cluster().tp_spec = null;
    cluster().tp_sums = 0;
    cluster().tp_values = 0;
    cluster().tp_last = .{};
    dsv41_mod.clusterHooks().* = .{ .tp_sum = &dsv41TpSum, .tp_gather = &dsv41TpGather };
    if (link.rank != 0) return;
    const hooks = dsv41_mod.clusterHooks();
    hooks.tp_step = &dsv41TpStep;
    hooks.tp_pipe = if (tpPipeWanted(link)) &dsv41TpPipe else null;
    hooks.tp_batch = &dsv41TpBatch;
    hooks.tp_draft = &dsv41TpDraft;
    const ds_pipe = dsparkPipeWanted(link);
    hooks.tp_draft_pipe = if (ds_pipe) &dsv41TpDraftPipe else null;
    hooks.tp_verify_pipe = if (ds_pipe) &dsv41TpVerifyPipe else null;
    hooks.tp_chained = &dsv41TpChained;
    hooks.tp_free = &dsv41TpFree;
    hooks.tp_snap = &dsv41TpSnap;
    hooks.tp_load = &dsv41TpLoad;
    hooks.tp_prefill = &dsv41TpPrefill;
}

pub fn clearDsv41Tp() void {
    dsv41_mod.clusterHooks().* = .{};
    cluster().tp_link = null;
    cluster().tp_live = null;
    cluster().tp_spec = null;
}

/// DeepSeek-V4.1 tensor, every rank: `x` summed over the group, lazily on this thread's CPU stream.
fn dsv41TpSum(x: mlx.mlx_array) anyerror!mlx.mlx_array {
    const link = cluster().tp_link orelse return error.DistWorkerNotJoined;
    cluster().tp_sums += 1;
    cluster().tp_values += mlx.mlx_array_size(x);
    return link.allSum(x);
}

/// DeepSeek-V4.1 tensor, every rank: every rank's `x` stacked on axis 0 in rank order, lazily on this thread's CPU
/// stream (each rank's slice of the logits, `deepseek_v41.headLogits`).
fn dsv41TpGather(x: mlx.mlx_array) anyerror!mlx.mlx_array {
    const link = cluster().tp_link orelse return error.DistWorkerNotJoined;
    return link.allGather(x);
}

/// The sums the step before this one built, logged where a prompt step ends or the count is not a decode step's.
fn noteTpStep(p0: usize, rows: usize) void {
    const c = cluster();
    c.tp_link.?.state.?.watchdog.progress(@intCast(crt.monotonicMs()));
    if (env.on(.DIST_STEPS)) log.info("[dist-step] epoch_ms={d} rows={d} step={d}\n", .{ crt.wallMs(), rows, c.tp_link.?.state.?.watchdog.step.load(.monotonic) });
    if (c.tp_last.rows > 0 and (c.tp_last.rows > 1 or c.tp_sums != 80)) log.info("[dist] tensor step at {d}: {d} rows, {d} sums of {d} values\n", .{ c.tp_last.p0, c.tp_last.rows, c.tp_sums, c.tp_values });
    c.tp_sums = 0;
    c.tp_values = 0;
    c.tp_last = .{ .p0 = p0, .rows = rows };
}

fn tpLeader(wait_ready: bool) !*const Link {
    const link = cluster().tp_link orelse return error.DistWorkerNotJoined;
    if (link.rank != 0) return error.DistWorkerNotJoined;
    if (peer()) |p| if (&p.link == link and !p.ready) {
        if (!wait_ready) return error.DistWorkerNotJoined;
        try p.awaitReady();
    };
    return link;
}

fn noteTpExchange(link: *const Link) void {
    if (peer()) |p| if (&p.link == link) {
        p.exchanges += 1;
    };
}

/// Host, DeepSeek-V4.1 tensor: one step's ids to the worker, which runs the same forward on its halves.
fn dsv41TpStep(ids: []const u32, p0: usize, snap: bool, all: bool, vo: dsv41_mod.VerifyOpts, gen: u64, images: ?mlx.mlx_array) anyerror!void {
    const verify = vo.open;
    const link = try tpLeader(true);
    noteTpStep(p0, ids.len);
    // Ids fit an int32 (the vocabulary is checked before any forward).
    const a = mlx.mlx_array_new_data(ids.ptr, &[_]c_int{@intCast(ids.len)}, 1, .int32);
    defer _ = mlx.mlx_array_free(a);
    noteTpExchange(link);
    const flags = @as(i32, @intFromBool(snap)) | @as(i32, @intFromBool(all)) << 1 | @as(i32, @intFromBool(verify)) << 2 | @as(i32, @intFromBool(images != null)) << 3 | @as(i32, vo.bits()) << 4;
    // Sent now, on the CPU stream alone: `sendOrdered` chains its sends through the GPU stream, and a GPU wait on a send
    // that blocks until the worker's receive deadlocked MLX's fast fences. Each worker gets its whole step before the
    // next, and all of them before this rank builds the step's first sum: a worker blocks only in a sum, which needs
    // every rank, so no send waits on a rank that waits on another.
    for (1..link.size) |k| {
        const dst: u32 = @intCast(k);
        try sendMsg(link, tpHeader(ids, p0, flags, gen), dst);
        if (ids.len > 1) try sendNow(link, a, dst);
        if (images) |r| try sendNow(link, r, dst);
    }
    cluster().tp_live = TpLive.after(gen, p0 + ids.len, verify);
    cluster().tp_spec = TpSpec.afterStep(gen, p0, ids.len, verify);
}

const TpSpec = union(enum) {
    verify: struct { gen: u64, n0: usize, end: usize },
    drafted: struct { gen: u64, n: usize },

    fn afterStep(gen: u64, p0: usize, rows: usize, verify: bool) ?TpSpec {
        return if (verify) .{ .verify = .{ .gen = gen, .n0 = p0, .end = p0 + rows } } else null;
    }

    fn keep(spec: ?TpSpec, gen: u64, p0: usize) ?u8 {
        const s = spec orelse return null;
        if (s != .verify) return null;
        const v = s.verify;
        if (v.gen != gen or p0 <= v.n0 or p0 > v.end) return null;
        return std.math.cast(u8, p0 - v.n0);
    }

    fn follows(spec: ?TpSpec, gen: u64, n: usize, rows: usize) bool {
        const s = spec orelse return false;
        if (s != .drafted or rows == 0 or rows > std.math.maxInt(u8)) return false;
        return s.drafted.gen == gen and s.drafted.n == n;
    }
};

fn dsv41TpDraftPipe(gen: u64, p0: usize, t1: u32) anyerror!bool {
    const link = try tpLeader(true);
    const c = cluster();
    const keep = TpSpec.keep(c.tp_spec, gen, p0) orelse return false;
    noteTpStep(p0, 0);
    var bytes: [6]u8 = .{ Bell.draft_byte, keep, 0, 0, 0, 0 };
    std.mem.writeInt(u32, bytes[2..6], t1, .little);
    for (1..link.size) |k| try link.bells[k].say(&bytes);
    c.tp_spec = .{ .drafted = .{ .gen = gen, .n = p0 } };
    return true;
}

fn dsv41TpVerifyPipe(gen: u64, n: usize, rows: usize, flags: u8) anyerror!bool {
    const link = try tpLeader(true);
    const c = cluster();
    if (!TpSpec.follows(c.tp_spec, gen, n, rows)) return false;
    noteTpStep(n, rows);
    noteTpExchange(link);
    const bytes = [_]u8{ Bell.verify_byte, @intCast(rows), flags };
    for (1..link.size) |k| try link.bells[k].say(&bytes);
    c.tp_spec = TpSpec.afterStep(gen, n, rows, true);
    return true;
}

fn dsv41TpChained(gen: u64, n: usize) anyerror!void {
    const c = cluster();
    c.tp_live = null;
    c.tp_spec = .{ .drafted = .{ .gen = gen, .n = n } };
}

fn dsparkPipeWanted(link: *const Link) bool {
    if (!env.on(.DSV41_DSPARK_PIPE)) return false;
    var mb: [64]u8 = undefined;
    return missingRanks(link.bellMask(), link.size, &mb).len == 0;
}

/// What rank 0 last told the workers of a DeepSeek-V4.1 tensor state: the one they stepped and where it now stands.
/// A step that opens a verify leaves none: the workers follow it only with the next header (`followVerify`).
const TpLive = struct {
    gen: u64,
    n: usize,

    fn after(gen: u64, end: usize, verify: bool) ?TpLive {
        return if (verify) null else .{ .gen = gen, .n = end };
    }

    fn follows(live: ?TpLive, gen: u64, n: usize) bool {
        const l = live orelse return false;
        return l.gen == gen and l.n == n;
    }
};

/// Host, DeepSeek-V4.1 tensor: the next decode step on the state the workers last stepped, rung on each bell alone;
/// every rank takes the step's token from rank 0 in its graph (`dsv41.tpToken`). False when the workers did not last
/// step `gen` to `n`, or heard anything since: the step goes with its header.
fn dsv41TpPipe(gen: u64, n: usize) anyerror!bool {
    const link = try tpLeader(true);
    const c = cluster();
    if (!TpLive.follows(c.tp_live, gen, n)) return false;
    noteTpStep(n, 1);
    noteTpExchange(link);
    for (1..link.size) |k| try link.bells[k].pipe();
    c.tp_live = .{ .gen = gen, .n = n + 1 };
    c.tp_spec = null;
    return true;
}

/// Rank 0 hands plain decode steps over by bell alone (`DSV41_TP_PIPE`, on unless 0) when every worker has its bell.
/// Workers follow whatever their bell says, so their own environment never changes the collectives.
fn tpPipeWanted(link: *const Link) bool {
    if (env.get(.DSV41_TP_PIPE)) |v| if (v[0] == '0') return false;
    var mb: [64]u8 = undefined;
    return missingRanks(link.bellMask(), link.size, &mb).len == 0;
}

/// Host: one batched step over the states `gens`, each at its position `p0s[k]` with its ids `chunks[k]`.
fn dsv41TpBatch(gens: []const u64, p0s: []const usize, chunks: []const []const u32, verify: bool) anyerror!void {
    const link = try tpLeader(true);
    var rows: usize = 0;
    for (chunks) |ch| rows += ch.len;
    noteTpStep(p0s[0], rows);
    const body = try std.heap.page_allocator.alloc(i32, tpBatchLen(gens.len, rows));
    defer std.heap.page_allocator.free(body);
    tpBatchPack(body, gens, p0s, chunks);
    const a = mlx.mlx_array_new_data(body.ptr, &[_]c_int{@intCast(body.len)}, 1, .int32);
    defer _ = mlx.mlx_array_free(a);
    noteTpExchange(link);
    for (1..link.size) |k| {
        try sendMsg(link, .{ OP_DSV41_TP_BATCH, @intCast(gens.len), @intCast(rows), @intFromBool(verify), 0, 0, 0, 0 }, @intCast(k));
        try sendNow(link, a, @intCast(k));
    }
}

/// Whether rank 0's DSpark stages load and split across the group (the plugin loads them as `dsparkWanted` says).
fn dsv41DraftShard(config: *const model_mod.ModelConfig, size: u32) bool {
    const binding = config.plugin orelse return false;
    const cfg = @import("../../../plugins/deepseek_v41/config.zig").of(&binding.parsed);
    return dsv41_mod.draftShardWanted(size) and dsv41_mod.dsparkWanted(cfg, null, config.mtp_override);
}

/// Host: a sharded DSpark draft over the states `gens`, each at `p0s[k]` drafting from `t1s[k]` (a batch body with one
/// id a state). Its sums are not a step's: the tally restarts with no rows, so the next step logs only its own.
fn dsv41TpDraft(gens: []const u64, p0s: []const usize, t1s: []const u32) anyerror!void {
    const link = try tpLeader(true);
    noteTpStep(p0s[0], 0);
    var chunks: [dsv41_mod.region_rows][]const u32 = undefined;
    for (chunks[0..t1s.len], t1s) |*ch, *t| ch.* = t[0..1];
    var body: [tpBatchLen(dsv41_mod.region_rows, dsv41_mod.region_rows)]i32 = undefined;
    const len = tpBatchLen(gens.len, gens.len);
    tpBatchPack(body[0..len], gens, p0s, chunks[0..gens.len]);
    const a = mlx.mlx_array_new_data(&body, &[_]c_int{@intCast(len)}, 1, .int32);
    defer _ = mlx.mlx_array_free(a);
    for (1..link.size) |k| {
        try sendMsg(link, .{ OP_DSV41_TP_DRAFT, @intCast(gens.len), 0, 0, 0, 0, 0, 0 }, @intCast(k));
        try sendNow(link, a, @intCast(k));
    }
    if (gens.len == 1) cluster().tp_spec = .{ .drafted = .{ .gen = gens[0], .n = p0s[0] } };
}

/// Host: the state `gen` is gone; the workers drop their copies.
fn dsv41TpFree(gen: u64) void {
    const link = tpLeader(false) catch return;
    sendAll(link, .{ OP_DSV41_TP_FREE, @intCast(gen & 0x7fffffff), @intCast(gen >> 31), 0, 0, 0, 0, 0 });
}

/// A control message with no payload to every worker. Best effort, like the pair's: a worker it misses has left,
/// and the bell's watcher ends the group.
fn sendAll(link: *const Link, m: Msg) void {
    for (1..link.size) |k| sendMsg(link, m, @intCast(k)) catch |err| log.warn("[dist] op {d} did not reach worker rank {d} ({s})\n", .{ m[0], k, @errorName(err) });
}

/// Host: the snapshot `id` it read from disk, as its ids and then `dsv41.tpLoadPlan`'s tensors in order.
fn dsv41TpLoad(id: u64, ids: []const u32, tensors: []const mlx.mlx_array) anyerror!void {
    const link = try tpLeader(true);
    cluster().tp_link.?.state.?.watchdog.progress(@intCast(crt.monotonicMs()));
    const a = mlx.mlx_array_new_data(ids.ptr, &[_]c_int{@intCast(ids.len)}, 1, .int32);
    defer _ = mlx.mlx_array_free(a);
    const io = if (cluster().group) |g| g.io else std.Io.Threaded.global_single_threaded.io();
    const t0 = io_util.nowUsMonotonic(io);
    var bytes: usize = 0;
    for (tensors) |t| bytes += mlx.mlx_array_size(t) * mlx.mlx_array_itemsize(t);
    // One worker after another, each its whole snapshot.
    for (1..link.size) |k| {
        const dst: u32 = @intCast(k);
        try sendMsg(link, .{ OP_DSV41_TP_LOAD, @intCast(id & 0x7fffffff), @intCast(id >> 31), @intCast(ids.len & 0x7fffffff), @intCast(ids.len >> 31), @intCast(tensors.len), 0, 0 }, dst);
        try sendNow(link, a, dst);
        for (tensors) |t| try sendNow(link, t, dst);
    }
    if (link.size >= 2) log.info("[dist] snapshot {d}: {d} MiB to {d} workers in {d} ms\n", .{ id, bytes >> 20, link.size - 1, @divTrunc(io_util.nowUsMonotonic(io) - t0, 1000) });
}

fn dsv41TpPrefill(gen: u64, p0: usize, n: usize, capture: bool, tensors: []const mlx.mlx_array, stages: usize) anyerror!void {
    if (n <= p0 or stages > 128) return error.DistBadMessage;
    const link = try tpLeader(true);
    link.state.?.watchdog.progress(@intCast(crt.monotonicMs()));
    const flags: i32 = @as(i32, @intCast(stages)) | @as(i32, @intFromBool(capture)) << 16;
    const header: Msg = .{ OP_DSV41_TP_PREFILL, @intCast(gen & 0x7fffffff), @intCast(gen >> 31), @intCast(p0 & 0x7fffffff), @intCast(p0 >> 31), @intCast(n & 0x7fffffff), @intCast(n >> 31), flags };
    for (1..link.size) |k| {
        const dst: u32 = @intCast(k);
        try sendMsg(link, header, dst);
        for (tensors) |x| try sendNow(link, x, dst);
    }
}

/// Host: what it did with a snapshot (`dsv41.SnapOp`); the worker does the same with its copy.
fn dsv41TpSnap(op: dsv41_mod.SnapOp, a: u64, b: u64) void {
    const link = tpLeader(false) catch return;
    sendAll(link, .{ OP_DSV41_TP_SNAP, @backingInt(op), @intCast(a & 0x7fffffff), @intCast(a >> 31), @intCast(b & 0x7fffffff), @intCast(b >> 31), 0, 0 });
}

/// A tensor step's header: its rows, position, flags, the state it advances and, for one row, the id itself.
fn tpHeader(ids: []const u32, p0: usize, flags: i32, gen: u64) Msg {
    return .{ OP_DSV41_TP, @intCast(ids.len), @intCast(p0 & 0x7fffffff), @intCast(p0 >> 31), flags, if (ids.len == 1) @intCast(ids[0]) else 0, @intCast(gen & 0x7fffffff), @intCast(gen >> 31) };
}

fn joined(lo: i32, hi: i32) u64 {
    return @as(u64, @intCast(lo)) | (@as(u64, @intCast(hi)) << 31);
}

/// A batched step's body: per state its gen and position (two halves each) and its row count, then every row's id.
fn tpBatchLen(states: usize, rows: usize) usize {
    return 5 * states + rows;
}

fn tpBatchPack(out: []i32, gens: []const u64, p0s: []const usize, chunks: []const []const u32) void {
    var at = 5 * gens.len;
    for (gens, p0s, chunks, 0..) |g, p, ch, k| {
        out[5 * k ..][0..5].* = .{ @intCast(g & 0x7fffffff), @intCast(g >> 31), @intCast(p & 0x7fffffff), @intCast(p >> 31), @intCast(ch.len) };
        for (ch, out[at..][0..ch.len]) |t, *o| o.* = @intCast(t);
        at += ch.len;
    }
}

/// State `k` of a batched step's body: its gen, position and ids (pointing into `body`).
fn tpBatchState(body: []const i32, states: usize, k: usize) struct { gen: u64, p0: usize, ids: []const u32 } {
    var at = 5 * states;
    for (0..k) |j| at += @intCast(body[5 * j + 4]);
    const e = body[5 * k ..][0..5];
    const ids: [*]const u32 = @ptrCast(body.ptr + at);
    return .{ .gen = joined(e[0], e[1]), .p0 = @intCast(joined(e[2], e[3])), .ids = ids[0..@intCast(e[4])] };
}

test "tensor step header: one id and the state's gen ride in it" {
    const m = tpHeader(&.{129279}, (1 << 33) + 5, 4, (1 << 40) + 3);
    try std.testing.expectEqual(@as(i32, 129279), m[5]);
    try std.testing.expectEqual(@as(i32, 4), m[4]);
    try std.testing.expectEqual(@as(u64, (1 << 33) + 5), joined(m[2], m[3]));
    try std.testing.expectEqual(@as(u64, (1 << 40) + 3), joined(m[6], m[7]));
    try std.testing.expectEqual(@as(i32, 0), tpHeader(&.{ 1, 2 }, 0, 0, 1)[5]);
}

test "dist N-rank: rank 0 pipes a decode step only on the state the workers last stepped, where they left it" {
    const live = TpLive.after(7, 40, false);
    try std.testing.expect(TpLive.follows(live, 7, 40));
    try std.testing.expect(!TpLive.follows(live, 7, 39));
    try std.testing.expect(!TpLive.follows(live, 8, 40));
    try std.testing.expect(TpLive.after(7, 40, true) == null);
    try std.testing.expect(!TpLive.follows(null, 7, 40));
}

test "dsv41 dspark pipe: rank 0 rings a draft only after the verify the workers opened, and a verify only of the draft they made" {
    const v = TpSpec.afterStep(7, 40, 6, true);
    try std.testing.expect(TpSpec.afterStep(7, 40, 6, false) == null);
    try std.testing.expectEqual(@as(?u8, 1), TpSpec.keep(v, 7, 41));
    try std.testing.expectEqual(@as(?u8, 6), TpSpec.keep(v, 7, 46));
    try std.testing.expect(TpSpec.keep(v, 7, 40) == null);
    try std.testing.expect(TpSpec.keep(v, 7, 47) == null);
    try std.testing.expect(TpSpec.keep(v, 8, 41) == null);
    try std.testing.expect(TpSpec.keep(null, 7, 41) == null);
    try std.testing.expect(!TpSpec.follows(v, 7, 40, 3));
    const d: ?TpSpec = .{ .drafted = .{ .gen = 7, .n = 44 } };
    try std.testing.expect(TpSpec.keep(d, 7, 44) == null);
    try std.testing.expect(TpSpec.follows(d, 7, 44, 6));
    try std.testing.expect(!TpSpec.follows(d, 7, 44, 0));
    try std.testing.expect(!TpSpec.follows(d, 7, 45, 6));
    try std.testing.expect(!TpSpec.follows(d, 9, 44, 6));
}

test "dsv41 dspark pipe: a worker reads a draft's kept count and first token, and a verify's rows, off its bell" {
    if (comptime builtin.os.tag != .macos) return error.SkipZigTest;
    var pair: [2]std.c.fd_t = undefined;
    try std.testing.expectEqual(@as(c_int, 0), std.c.socketpair(std.posix.AF.UNIX, std.posix.SOCK.STREAM, 0, &pair));
    defer for (pair) |fd| {
        _ = std.c.close(fd);
    };
    const host: Bell = .{ .fd = pair[0] };
    var link: Link = .{ .group = undefined, .rank = 1, .size = 4 };
    link.bells[0] = .{ .fd = pair[1] };
    var draft: [6]u8 = .{ Bell.draft_byte, 3, 0, 0, 0, 0 };
    std.mem.writeInt(u32, draft[2..6], 129279, .little);
    try host.say(&draft);
    try host.say(&.{ Bell.verify_byte, 6, 3 });
    try host.pipe();
    const a = try recvMsg(&link, 0);
    try std.testing.expectEqual(OP_DSV41_TP_DRAFT_PIPE, a[0]);
    try std.testing.expectEqual(@as(i32, 3), a[1]);
    try std.testing.expectEqual(@as(u32, 129279), @as(u32, @bitCast(a[2])));
    const b = try recvMsg(&link, 0);
    try std.testing.expectEqual(OP_DSV41_TP_VERIFY_PIPE, b[0]);
    try std.testing.expectEqual(@as(i32, 6), b[1]);
    try std.testing.expectEqual(@as(i32, 3), b[2]);
    try std.testing.expectEqual(OP_DSV41_TP_PIPE, (try recvMsg(&link, 0))[0]);
}

test "tensor batched step body: every state's gen, position and ids come back in order" {
    const gens = [_]u64{ 7, (1 << 35) + 1 };
    const p0s = [_]usize{ 40, (1 << 32) + 9 };
    const chunks = [_][]const u32{ &.{ 5, 6, 7 }, &.{129279} };
    var body: [tpBatchLen(2, 4)]i32 = undefined;
    tpBatchPack(&body, &gens, &p0s, &chunks);
    for (0..2) |k| {
        const e = tpBatchState(&body, 2, k);
        try std.testing.expectEqual(gens[k], e.gen);
        try std.testing.expectEqual(p0s[k], e.p0);
        try std.testing.expectEqualSlices(u32, chunks[k], e.ids);
    }
}

/// Host, DeepSeek-V4.1: one step's stream and mix through the worker's layers and back.
fn dsv41Exchange(x: mlx.mlx_array, pre: mlx.mlx_array, p0: usize, snap: bool, verify: bool) anyerror![2]mlx.mlx_array {
    const self = peer() orelse return error.DistWorkerNotJoined;
    if (!self.ready) try self.awaitReady();
    const xs = mlx.getShape(x);
    const t0 = io_util.nowUsMonotonic(cluster().group.?.io);
    defer noteRoundTrip(io_util.nowUsMonotonic(cluster().group.?.io) - t0);
    try self.sendOrdered(.{ OP_DSV41, xs[0], @intCast(p0 & 0x7fffffff), @intCast(p0 >> 31), @as(i32, @intFromBool(snap)) | @as(i32, @intFromBool(verify)) << 1, xs[1], @backingInt(mlx.mlx_array_dtype(x)), @backingInt(mlx.mlx_array_dtype(pre)) }, &.{ x, pre });
    self.exchanges += 1;
    const xo = try queuedRecv(&self.link, xs, mlx.mlx_array_dtype(x), 1);
    errdefer _ = mlx.mlx_array_free(xo);
    return .{ xo, try queuedRecv(&self.link, mlx.getShape(pre), mlx.mlx_array_dtype(pre), 1) };
}

/// Every 64 round trips: what one took, and how much of it the worker computed (the rest is the
/// link, both sends and the hand-off). The worker's time arrives over the bell, a step late.
fn noteRoundTrip(us: i64) void {
    cluster().timing.n += 1;
    cluster().timing.round_us += @intCast(@max(us, 0));
    cluster().timing.worker_us += cluster().peer_slot.?.link.state.?.worker_us.load(.acquire);
    if (cluster().timing.n < 64) return;
    const n: f64 = @floatFromInt(cluster().timing.n);
    const round = @as(f64, @floatFromInt(cluster().timing.round_us)) / n / 1000;
    const work = @as(f64, @floatFromInt(cluster().timing.worker_us)) / n / 1000;
    log.info("[dist] timing over {d} steps: round trip {d:.2} ms = worker {d:.2} ms + link {d:.2} ms\n", .{ cluster().timing.n, round, work, round - work });
    cluster().timing = .{};
}

/// Worker, DeepSeek-V4.1: its run of layers (`pl.span`) and one step per OP_DSV41, or the whole model on its half of
/// every expert and attention head and one step per OP_DSV41_TP, until rank 0 stops it.
fn serveDsv41(io: std.Io, allocator: std.mem.Allocator, link: *const Link, dir: []const u8, config: *model_mod.ModelConfig, pl: Plan, ready: *WorkerReady) !bool {
    switch (pl.mode) {
        .tensor => config.applyShard(.{ .rank = link.rank, .size = link.size, .stages = pl.stages }) catch |err| {
            log.err("[dist] worker load failed: {s}\n", .{@errorName(err)});
            return err;
        },
        else => config.dist_dsv41 = .{ .lo = pl.span.start, .hi = pl.span.end, .worker = true },
    }
    var w = model_mod.loadModelWeights(io, allocator, dir, config, false) catch |err| {
        log.err("[dist] worker load failed: {s}\n", .{@errorName(err)});
        return err;
    };
    defer w.deinit();
    const binding = config.plugin orelse return error.NotPluginServed;
    const m = dsv41_mod.init(allocator, @import("../../../plugins/deepseek_v41/config.zig").of(&binding.parsed), &w, mlx.gpuStream(), .{ .pair = config.dist_dsv41, .tp = config.dist_shard, .dspark = pl.stages, .draft_shard = pl.stages, .head_8bit = dsv41_mod.head8Wanted(), .pack_dir = if (pl.mode == .tensor) dir else null }) catch |err| {
        log.err("[dist] worker load failed: {s}\n", .{@errorName(err)});
        return err;
    };
    defer m.deinit();
    // Rank 0 drafts with this rank's stages: without them its first draft would wait on this rank forever.
    if (pl.stages and !m.draft_shard) {
        log.err("[dist] worker load failed: rank 0 drafts on every rank but this copy loaded no DSpark stages\n", .{});
        return error.Dsv41DraftShardNoStages;
    }
    installDsv41Tp(link);
    defer clearDsv41Tp();
    var active: usize = 0;
    _ = mlx.mlx_get_active_memory(&active);
    try ready.send(0, @intCast(active >> 20), 0);
    log.info("[dist] worker ready ({d} MiB resident)\n", .{active >> 20});
    return serveDsv41Steps(io, allocator, link, m);
}

pub fn serveDsv41Steps(io: std.Io, allocator: std.mem.Allocator, link: *const Link, m: *dsv41_mod.Dsv41Model) !bool {
    return dsv41StepLoop(io, allocator, link, m, .{ .ctx = link, .vt = &Dsv41WorkerWire.vt });
}

const Dsv41WorkerWire = struct {
    const vt = blk: {
        var v = WireOps.vt;
        v.recv_op = receive;
        break :blk v;
    };

    fn receive(ctx: *const anyopaque) !Msg {
        return recvMsg(WireOps.of(ctx), 0);
    }
};

fn dsv41StepLoop(io: std.Io, allocator: std.mem.Allocator, link: *const Link, m: *dsv41_mod.Dsv41Model, transport: wire.Link) !bool {
    // Rank 0's other live states (its parked slots and idle state), by gen; the module holds the one it advances.
    var parked: dsv41_mod.Parked = .init(allocator);
    defer dsv41_mod.dropParked(&parked);
    var snaps: dsv41_mod.Snaps = .init(allocator);
    defer dsv41_mod.dropSnaps(&snaps, allocator);
    const d: c_int = @intCast(m.cfg.hidden_size);
    var steps: u64 = 0;
    while (true) {
        const msg = try transport.recvOp();
        const tracked = msg[0] != OP_DSV41_TP_LOAD;
        if (tracked) beginDistStep();
        defer if (tracked) link.state.?.watchdog.done();
        switch (msg[0]) {
            OP_STOP => break,
            OP_RELEASE => return true,
            OP_DSV41 => {
                const p0: usize = @as(usize, @intCast(msg[2])) | (@as(usize, @intCast(msg[3])) << 31);
                const x = try transport.recvNow(&.{ msg[1], msg[5], d }, @fromBackingInt(@intCast(msg[6])), 0);
                defer _ = mlx.mlx_array_free(x);
                const pre = try transport.recvNow(&.{ msg[1], msg[5] }, @fromBackingInt(@intCast(msg[7])), 0);
                defer _ = mlx.mlx_array_free(pre);
                const t0 = io_util.nowUsMonotonic(io);
                const out = try dsv41_mod.workerStep(m, allocator, x, pre, p0, msg[4] & 1 != 0, msg[4] & 2 != 0);
                defer for (out) |o| {
                    _ = mlx.mlx_array_free(o);
                };
                sendStepTime(link.bells[0], io_util.nowUsMonotonic(io) - t0) catch {};
                try transport.sendNow(out[0], 0);
                try transport.sendNow(out[1], 0);
                steps += 1;
                if (msg[1] > 1 or steps % 256 == 0) log.info("[dist] worker: step {d}: {d} rows at {d}\n", .{ steps, msg[1], p0 });
            },
            OP_DSV41_TP => {
                const p0: usize = @as(usize, @intCast(msg[2])) | (@as(usize, @intCast(msg[3])) << 31);
                const L: usize = @intCast(msg[1]);
                var one = [_]u32{@intCast(msg[5])};
                var sent: mlx.mlx_array = .{ .ctx = null };
                defer if (sent.ctx != null) {
                    _ = mlx.mlx_array_free(sent);
                };
                const ids = if (L == 1) one[0..] else blk: {
                    sent = try transport.recvNow(&.{msg[1]}, .int32, 0);
                    const data = mlx.mlx_array_data_int32(sent) orelse return error.NoData;
                    break :blk @as([*]const u32, @ptrCast(data))[0..L];
                };
                // Bit 3: the rows of the chunk's image positions follow, one per position.
                var images: mlx.mlx_array = .{ .ctx = null };
                defer if (images.ctx != null) {
                    _ = mlx.mlx_array_free(images);
                };
                if (msg[4] & 8 != 0) images = try transport.recvNow(&.{ @intCast(m.imageRows(ids)), @intCast(m.cfg.hidden_size) }, m.act, 0);
                noteTpStep(p0, L);
                try dsv41_mod.tpWorkerStep(m, allocator, &parked, joined(msg[6], msg[7]), ids, p0, msg[4] & 1 != 0, if (msg[4] & 2 != 0) .all else .last, dsv41_mod.VerifyOpts.fromBits(msg[4] & 4 != 0, @intCast((msg[4] >> 4) & 0xff)), if (images.ctx != null) images else null);
                steps += 1;
            },
            OP_DSV41_TP_PIPE => {
                noteTpStep(if (m.dec_state) |st| st.n else 0, 1);
                try dsv41_mod.tpWorkerPipe(m, allocator);
                steps += 1;
            },
            OP_DSV41_TP_DRAFT_PIPE => {
                noteTpStep(if (m.dec_state) |st| st.n else 0, 0);
                try dsv41_mod.tpWorkerDraftPipe(m, allocator, &parked, @intCast(msg[1]), @bitCast(msg[2]));
            },
            OP_DSV41_TP_VERIFY_PIPE => {
                const rows: usize = @intCast(msg[1]);
                noteTpStep(if (m.dec_state) |st| st.n else 0, rows);
                try dsv41_mod.tpWorkerVerifyPipe(m, allocator, &parked, rows, @intCast(msg[2] & 0xff));
                steps += 1;
            },
            OP_DSV41_TP_BATCH => {
                const n: usize = @intCast(msg[1]);
                const rows: usize = @intCast(msg[2]);
                const sent = try transport.recvNow(&.{@intCast(tpBatchLen(n, rows))}, .int32, 0);
                defer _ = mlx.mlx_array_free(sent);
                const body = (mlx.mlx_array_data_int32(sent) orelse return error.NoData)[0..tpBatchLen(n, rows)];
                var gens: [dsv41_mod.region_rows]u64 = undefined;
                var p0s: [dsv41_mod.region_rows]usize = undefined;
                var chunks: [dsv41_mod.region_rows][]const u32 = undefined;
                if (n == 0 or n > gens.len) return error.DistBadMessage;
                for (0..n) |k| {
                    const e = tpBatchState(body, n, k);
                    gens[k] = e.gen;
                    p0s[k] = e.p0;
                    chunks[k] = e.ids;
                }
                noteTpStep(p0s[0], rows);
                try dsv41_mod.tpWorkerBatch(m, allocator, &parked, gens[0..n], p0s[0..n], chunks[0..n], msg[3] != 0);
                steps += 1;
            },
            OP_DSV41_TP_DRAFT => {
                const n: usize = @intCast(msg[1]);
                if (n == 0 or n > dsv41_mod.region_rows) return error.DistBadMessage;
                const sent = try transport.recvNow(&.{@intCast(tpBatchLen(n, n))}, .int32, 0);
                defer _ = mlx.mlx_array_free(sent);
                const body = (mlx.mlx_array_data_int32(sent) orelse return error.NoData)[0..tpBatchLen(n, n)];
                var gens: [dsv41_mod.region_rows]u64 = undefined;
                var p0s: [dsv41_mod.region_rows]usize = undefined;
                var t1s: [dsv41_mod.region_rows]u32 = undefined;
                for (0..n) |k| {
                    const e = tpBatchState(body, n, k);
                    if (e.ids.len != 1) return error.DistBadMessage;
                    gens[k] = e.gen;
                    p0s[k] = e.p0;
                    t1s[k] = e.ids[0];
                }
                noteTpStep(p0s[0], 0);
                try dsv41_mod.tpWorkerDraft(m, allocator, &parked, gens[0..n], p0s[0..n], t1s[0..n]);
            },
            OP_DSV41_TP_FREE => dsv41_mod.tpWorkerFree(m, &parked, joined(msg[1], msg[2])),
            OP_DSV41_TP_SNAP => {
                const op = std.enums.fromInt(dsv41_mod.SnapOp, msg[1]) orelse return error.DistBadMessage;
                try dsv41_mod.tpWorkerSnap(m, allocator, &parked, &snaps, op, joined(msg[2], msg[3]), joined(msg[4], msg[5]));
            },
            OP_DSV41_TP_LOAD => {
                const n: usize = @intCast(joined(msg[3], msg[4]));
                const sent = try transport.recvNow(&.{@intCast(n)}, .int32, 0);
                defer _ = mlx.mlx_array_free(sent);
                const ids = @as([*]const u32, @ptrCast(mlx.mlx_array_data_int32(sent) orelse return error.NoData))[0..n];
                const p = try dsv41_mod.tpLoadPlan(m, n);
                if (p.len != @as(usize, @intCast(msg[5]))) return error.DistPeerDiverged;
                const xs = try allocator.alloc(mlx.mlx_array, p.len);
                var got: usize = 0;
                defer {
                    for (xs[0..got]) |x| _ = mlx.mlx_array_free(x);
                    allocator.free(xs);
                }
                for (p.specs(), xs) |spec, *x| {
                    const shape = [_]c_int{ @intCast(spec.rows), @intCast(spec.width) };
                    x.* = try transport.recvNow(shape[0..spec.rank()], spec.dtype, 0);
                    got += 1;
                }
                try dsv41_mod.tpWorkerLoad(m, allocator, &snaps, joined(msg[1], msg[2]), ids, xs);
            },
            OP_DSV41_TP_PREFILL => {
                for (msg[1..]) |word| if (word < 0) return error.DistBadMessage;
                if (msg[7] & ~@as(i32, 0x1ffff) != 0) return error.DistBadMessage;
                const p0: usize = @intCast(joined(msg[3], msg[4]));
                const n: usize = @intCast(joined(msg[5], msg[6]));
                const stages: usize = @intCast(msg[7] & 0xffff);
                const p = try dsv41_mod.prefillPlan(m, p0, n, stages);
                const xs = try allocator.alloc(mlx.mlx_array, p.len);
                var got: usize = 0;
                defer {
                    for (xs[0..got]) |x| _ = mlx.mlx_array_free(x);
                    allocator.free(xs);
                }
                for (p.specs(), xs) |spec, *x| {
                    const shape = [_]c_int{ @intCast(spec.rows), @intCast(spec.width) };
                    x.* = try transport.recvNow(shape[0..spec.rank()], spec.dtype, 0);
                    got += 1;
                }
                try dsv41_mod.tpWorkerPrefill(m, allocator, &parked, joined(msg[1], msg[2]), p0, n, msg[7] & (1 << 16) != 0, xs, stages);
            },
            else => return error.DistBadMessage,
        }
    }
    log.info("[dist] worker: stopped by rank 0 after {d} steps\n", .{steps});
    return false;
}

fn sendBytes(link: *const Link, bytes: []const u8, dst: u32) !void {
    const a = mlx.mlx_array_new_data(bytes.ptr, &[_]c_int{@intCast(bytes.len)}, 1, .uint8);
    defer _ = mlx.mlx_array_free(a);
    const r = try link.send(a, dst);
    defer _ = mlx.mlx_array_free(r);
    try evalOne(r);
}

fn recvBytes(link: *const Link, out: []u8, src: u32) !void {
    const r = try link.recv(&[_]c_int{@intCast(out.len)}, .uint8, src);
    defer _ = mlx.mlx_array_free(r);
    try evalOne(r);
    @memcpy(out, (mlx.mlx_array_data_uint8(r) orelse return error.DistBadMessage)[0..out.len]);
}

/// Rank 0 at startup: waits for the worker on a thread of its own, so the server
/// serves meanwhile. Only the RDMA link opens there; every message runs on the
/// inference thread, when `model` loads (`planFor`).
pub fn connect(io: std.Io, allocator: std.mem.Allocator, cfg: Config, mode: Mode, model: []const u8, split: ?u32, join_timeout_s: ?u32) !void {
    if (comptime builtin.os.tag != .macos) return error.RdmaUnavailable; // a cluster exists only on macOS
    try checkHost(cfg, mode);
    try primeEnv(cfg);
    cluster().group = .{ .io = io, .allocator = allocator, .mode = mode, .model = try allocator.dupe(u8, model), .split = split, .size = cfg.size, .join_timeout_s = join_timeout_s orelse defaultJoinTimeout(cfg.size), .started_us = io_util.nowUsMonotonic(io) };
    h3_split.hooks().* = .{ .serves = &serves, .plan_rows = &rowsPair, .note_job = &noteJob };
    const t = try std.Thread.spawn(.{ .stack_size = 1 << 20 }, waitForWorker, .{ io, cfg });
    t.detach();
}

/// A pair's worker is missing at once when absent (the app restarts it); a larger group reads as forming for ten
/// minutes before the log names the missing ranks.
fn defaultJoinTimeout(size: u32) u32 {
    return if (size > 2) 600 else 0;
}

/// The host's flags, refused at startup rather than on the link's thread: the size, the mode past a pair, the device list.
fn checkHost(cfg: Config, mode: Mode) !void {
    if (cfg.size < 2 or cfg.size > max_ranks) {
        log.err("[dist] --dist-size takes 2..{d} Macs, got {d}\n", .{ max_ranks, cfg.size });
        return error.InvalidArgs;
    }
    if (cfg.size > 2 and mode != .tensor) {
        log.err("[dist] --dist-size {d} needs --dist tensor (DeepSeek-V4.1): --dist {s} pairs two Macs\n", .{ cfg.size, @tagName(mode) });
        return error.DistNeedsPair;
    }
    var buf: [1024]u8 = undefined;
    _ = deviceMatrixJson(cfg, &buf) catch |err| {
        log.err("[dist] --dist-device needs the port facing each of the {d} ranks in rank order, this Mac's own `-`: got \"{s}\"\n", .{ cfg.size, cfg.device });
        return err;
    };
}

/// The collectives run on the CPU stream between GPU work, and MLX's default fence waits on a command buffer at each
/// hand-off (~0.3 ms, two a tensor layer); its shared-memory fences take microseconds. MLX reads the switch once, at its
/// first fence, so it is set before anything runs; an explicit value wins.
fn preferFastSynch() void {
    if (env.get(.MLX_METAL_FAST_SYNCH) == null) env.set(.MLX_METAL_FAST_SYNCH, "1");
}

fn waitForWorker(io: std.Io, cfg: Config) void {
    cluster().joined_link = Link.open(io, cfg) catch |err| {
        log.err("[dist] the cluster link did not open: {s}\n", .{@errorName(err)});
        return;
    };
    // Asked before `joined`: the inference thread's first ring must not land inside the question.
    const link = &cluster().joined_link;
    for (1..link.size) |k| askForModel(link, @intCast(k)) catch |err| {
        log.err("[dist] worker rank {d} did not answer about the model ({s}); is it an older mlx-serve?\n", .{ k, @errorName(err) });
        cluster().joined_link.deinit();
        return;
    };
    cluster().joined.store(true, .release);
    log.info("[dist] {d} workers joined; they load their parts with {s}\n", .{ link.size - 1, std.fs.path.basename(cluster().group.?.model) });
    startWatchers(link) catch |err| {
        log.err("[dist] cannot watch workers: {s}\n", .{@errorName(err)});
        distExit(.watchdog_thread, null, null);
    };
}

/// Reads worker READY frames and step times until a bell closes, returning its rank.
fn watchBells(link: *const Link, state: *LinkState) u32 {
    var fds: [max_ranks]std.posix.pollfd = undefined;
    var ranks: [max_ranks]u32 = undefined;
    var decoders: [max_ranks]BellDecoder = @splat(.{});
    var n: usize = 0;
    for (1..link.size) |k| if (link.bells[k].open()) {
        fds[n] = .{ .fd = link.bells[k].fd, .events = std.posix.POLL.IN, .revents = 0 };
        ranks[n] = @intCast(k);
        n += 1;
    };
    if (n == 0) return 0;
    const lost = outer: while (!state.closing.load(.acquire) and !state.releasing.load(.acquire)) {
        _ = std.posix.poll(fds[0..n], 50) catch break :outer ranks[0];
        for (fds[0..n], ranks[0..n]) |*f, k| {
            if (f.revents == 0) continue;
            const word = decoders[k].read(.{ .fd = f.fd }) catch break :outer k;
            if (word) |message| switch (message) {
                .step => |us| state.worker_us.store(us, .release),
                .ready => |frame| {
                    if (frame.code != 0) {
                        log.err("[dist] rank {d} not ready: {s}\n", .{ k, readyCodeName(frame.code) });
                    } else log.info("[dist] worker rank {d} READY: {d} MiB, load {d} ms\n", .{ k, frame.resident_mib, frame.load_ms });
                    publishReady(state, k, frame);
                },
            };
        }
    } else 0;
    if (state.closing.load(.acquire) or state.releasing.load(.acquire)) return 0;
    state.lost_rank.store(lost, .release);
    state.worker_gone.store(true, .release);
    return lost;
}

/// Dials rank 0's bell until it answers or `deadline_ms` (CLOCK_MONOTONIC) passes, waiting 20 ms and doubling to 1 s
/// between tries as JACCL's connect does; a closed bell when the deadline passes first. A worker that starts a few ms
/// before rank 0 listens joins instead of going on without a bell.
fn dialBellControlled(coord: Coordinator, rank: u32, deadline_ms: u64, control: ?*OpenControl) Bell {
    var wait: u32 = 20;
    while (msLeft(deadline_ms) > 0) {
        if (control) |ctrl| if (ctrl.cancelled.load(.acquire)) return .{};
        const b = Bell.connect(coord, rank, control, deadline_ms);
        if (b.open()) return b;
        const left = msLeft(deadline_ms);
        if (left <= wait) return .{};
        var slept: u32 = 0;
        while (slept < wait) : (slept += 10) {
            if (control) |ctrl| if (ctrl.cancelled.load(.acquire)) return .{};
            crt.sleepMs(@min(wait - slept, 10));
        }
        wait = @min(wait * 2, 1000);
    }
    return .{};
}

/// A loopback coordinator whose bell port (port + 1) nothing listens on: an ephemeral port the kernel just handed out and
/// released, so no other test or process holds it.
fn closedBellCoordinator() !Coordinator {
    const fd = std.c.socket(std.posix.AF.INET, std.posix.SOCK.STREAM, 0);
    if (fd < 0) return error.SkipZigTest;
    defer _ = std.c.close(fd);
    var addr: std.posix.sockaddr.in = .{ .port = 0, .addr = @bitCast([4]u8{ 127, 0, 0, 1 }) };
    if (std.c.bind(fd, @ptrCast(&addr), @sizeOf(std.posix.sockaddr.in)) != 0) return error.SkipZigTest;
    var len: std.posix.socklen_t = @sizeOf(std.posix.sockaddr.in);
    if (std.c.getsockname(fd, @ptrCast(&addr), &len) != 0) return error.SkipZigTest;
    return .{ .ip4 = .{ 127, 0, 0, 1 }, .port = std.mem.bigToNative(u16, addr.port) -% 1 };
}

test "a worker whose bell is refused past the formation deadline gets no bell, near the deadline" {
    const coord = try closedBellCoordinator();
    const t0 = crt.monotonicMs();
    const b = dialBellControlled(coord, 1, @as(u64, @intCast(t0)) + 300, null);
    const took = crt.monotonicMs() - t0;
    try std.testing.expect(!b.open());
    try std.testing.expect(took >= 150 and took < 1500);
}

test "a worker dialing an address that drops its SYN still returns by the formation deadline" {
    const coord: Coordinator = .{ .ip4 = .{ 192, 0, 2, 1 }, .port = 46185 };
    const t0 = crt.monotonicMs();
    const b = dialBellControlled(coord, 1, @as(u64, @intCast(t0)) + 400, null);
    const took = crt.monotonicMs() - t0;
    try std.testing.expect(!b.open());
    try std.testing.expect(took < 400 + 100);
}

test "a worker that dials before rank 0 listens joins once the listener opens" {
    const coord: Coordinator = .{ .ip4 = .{ 127, 0, 0, 1 }, .port = 46183 };
    const Late = struct {
        fn listenAfter(c: Coordinator, fd: *std.c.fd_t) void {
            crt.sleepMs(120);
            fd.* = Bell.listen(c, 1);
        }
    };
    var listener: std.c.fd_t = Bell.none;
    const t = try std.Thread.spawn(.{}, Late.listenAfter, .{ coord, &listener });
    const b = dialBellControlled(coord, 1, @as(u64, @intCast(crt.monotonicMs())) + 3000, null);
    t.join();
    defer if (listener != Bell.none) {
        _ = std.c.close(listener);
    };
    try std.testing.expect(b.open());
    _ = std.c.close(b.fd);
}

const TestBells = struct {
    leader: Link = .{ .group = .{}, .rank = 0, .size = 2 },
    worker: Bell,

    fn open(port: ?u16) !TestBells {
        var coord: Coordinator = .{ .ip4 = .{ 127, 0, 0, 1 }, .port = port orelse std.math.maxInt(u16) };
        const listener = Bell.listen(coord, 1);
        if (listener == Bell.none) return error.TestListenFailed;
        defer _ = std.c.close(listener);
        if (port == null) {
            var addr: std.posix.sockaddr.in = undefined;
            var len: std.posix.socklen_t = @sizeOf(std.posix.sockaddr.in);
            if (std.c.getsockname(listener, @ptrCast(&addr), &len) != 0) return error.TestSocketNameFailed;
            coord.port = std.mem.bigToNative(u16, addr.port) -% 1;
        }
        const worker = Bell.connect(coord, 1, null, testDeadline(3000));
        if (!worker.open()) return error.TestConnectFailed;
        errdefer _ = std.c.close(worker.fd);
        const got = Bell.acceptRank(listener, 1000, null, testDeadline(2000)) orelse return error.TestAcceptFailed;
        var out: TestBells = .{ .worker = worker };
        out.leader.bells[1] = got.bell;
        return out;
    }

    fn close(self: *TestBells) void {
        if (self.worker.open()) _ = std.c.close(self.worker.fd);
        for (self.leader.bells) |b| if (b.open()) {
            _ = std.c.close(b.fd);
        };
    }
};

fn testDeadline(ms: u64) u64 {
    return @as(u64, @intCast(crt.monotonicMs())) + ms;
}

fn testReadBellWord(decoder: *BellDecoder, bell: Bell) !BellWord {
    const deadline = testDeadline(1000);
    var fds = [_]std.posix.pollfd{.{ .fd = bell.fd, .events = std.posix.POLL.IN, .revents = 0 }};
    while (msLeft(deadline) > 0) {
        _ = try std.posix.poll(&fds, @intCast(msLeft(deadline)));
        if (try decoder.read(bell)) |word| return word;
    }
    return error.TestBellTimeout;
}

const TestReadyWorker = struct {
    bell: Bell,
    deadline: u64,
    frame: ReadyFrame = .{},
    result: ?anyerror = null,

    fn run(self: *@This()) void {
        workerReadyAwaitGo(self.bell, self.deadline, self.frame) catch |err| {
            self.result = err;
        };
    }
};

test "dist switch matrix 13: cold READY waits beyond idle until the first leader byte" {
    if (comptime builtin.os.tag != .macos) return error.SkipZigTest;
    const old_idle = if (env.get(.MLX_SERVE_DIST_IDLE_MS)) |v| blk: {
        const saved = try std.testing.allocator.allocSentinel(u8, std.mem.len(v), 0);
        @memcpy(saved, std.mem.span(v));
        break :blk saved;
    } else null;
    var state: LinkState = .{};
    defer {
        if (old_idle) |v| {
            env.set(.MLX_SERVE_DIST_IDLE_MS, v);
            std.testing.allocator.free(v);
        } else env.unset(.MLX_SERVE_DIST_IDLE_MS);
        state.follower_pending.store(false, .release);
        state.follower_ready.store(false, .release);
    }
    env.set(.MLX_SERVE_DIST_IDLE_MS, "40");
    var pair = try TestBells.open(46129);
    defer pair.close();
    var worker_link: Link = .{ .group = .{}, .rank = 1, .size = 2 };
    worker_link.state = &state;
    worker_link.bells[0] = pair.worker;
    worker_link.attachBells();
    pair.worker.state = &state;
    var ready: WorkerReady = .{ .link = &worker_link, .started_ms = testDeadline(0) };
    try ready.send(0, 123, 456);
    var decoder: BellDecoder = .{};
    const frame = (try testReadBellWord(&decoder, pair.leader.bells[1])).ready;
    try std.testing.expectEqual(@as(u32, 456), frame.calib_us);
    var probe = struct {
        bell: Bell,
        result: ?anyerror = null,
        returned: std.atomic.Value(bool) = .init(false),
        fn run(self: *@This()) void {
            _ = self.bell.wait() catch |err| {
                self.result = err;
                self.returned.store(true, .release);
                return;
            };
            self.returned.store(true, .release);
        }
    }{ .bell = .{ .fd = pair.worker.fd } };
    const thread = try std.Thread.spawn(.{}, @TypeOf(probe).run, .{&probe});
    crt.sleepMs(160);
    const returned_early = probe.returned.load(.acquire);
    const armed_early = state.follower_ready.load(.acquire);
    pair.leader.bells[1].ring() catch {};
    thread.join();
    try std.testing.expect(!returned_early);
    try std.testing.expect(!armed_early);
    try std.testing.expectEqual(@as(?anyerror, null), probe.result);
    try pair.leader.bells[1].writeAll(&.{ Bell.heartbeat_byte, Bell.pipe_byte });
    try std.testing.expectEqual(Bell.pipe_byte, try pair.worker.wait());
    try std.testing.expect(state.follower_ready.load(.acquire));
    try std.testing.expectError(error.DistPeerGone, (Bell{ .fd = pair.worker.fd }).wait());
}

test "dist switch heartbeat activation clears a stale inference tick" {
    var state: LinkState = .{};
    state.loop_tick_ms.store(1, .release);
    activateHeartbeat(&state);
    try std.testing.expect(state.heartbeat_active.load(.acquire));
    try std.testing.expect(dist_watchdog.loopAlive(100_000, state.loop_tick_ms.load(.acquire), false, 1_000));
}

test "dist open refuses another live link without changing its state" {
    var link = try newLink(.{ .rank = 1, .size = 2, .coordinator = "127.0.0.1:46131", .device = "loopback" });
    defer link.deinit();
    startWatchdog(&link);
    link.state.?.follower_ready.store(true, .release);
    const io = std.Io.Threaded.global_single_threaded.io();
    try std.testing.expectError(error.DistLinkAlreadyOpen, Link.open(io, .{ .rank = 0, .size = 0, .device = "", .coordinator = "127.0.0.1:46131" }));
    try std.testing.expect(link.state.?.watchdog_started.load(.acquire));
    try std.testing.expect(!link.state.?.watchdog_stop.load(.acquire));
    try std.testing.expect(link.state.?.watchdog_threads[0] != null);
    try std.testing.expect(link.state.?.follower_ready.load(.acquire));
}

test "dist ready-in-open requires an explicit shared deadline before formation" {
    const io = std.Io.Threaded.global_single_threaded.io();
    for ([_]u32{ 0, 1 }) |rank| {
        try std.testing.expectError(error.InvalidArgs, Link.open(io, .{ .rank = rank, .size = 0, .device = "", .coordinator = "127.0.0.1:46131", .ready_in_open = true }));
    }
}

test "dist bell READY matrix 1: every worker receives GO" {
    if (comptime builtin.os.tag != .macos) return error.SkipZigTest;
    var pair = try TestBells.open(46101);
    defer pair.close();
    var other = try TestBells.open(46103);
    defer other.close();
    pair.leader.size = 3;
    pair.leader.bells[2] = other.leader.bells[1];
    other.leader.bells[1] = .{};
    const deadline = testDeadline(2000);
    var a: TestReadyWorker = .{ .bell = pair.worker, .deadline = deadline };
    var b: TestReadyWorker = .{ .bell = other.worker, .deadline = deadline };
    const ta = try std.Thread.spawn(.{}, TestReadyWorker.run, .{&a});
    const tb = try std.Thread.spawn(.{}, TestReadyWorker.run, .{&b});
    const result = leaderAwaitReadyThenGo(&pair.leader.bells, 3, deadline);
    ta.join();
    tb.join();
    try result;
    try std.testing.expectEqual(@as(?anyerror, null), a.result);
    try std.testing.expectEqual(@as(?anyerror, null), b.result);
}

test "dist bell READY matrix 3: missing READY aborts every worker" {
    if (comptime builtin.os.tag != .macos) return error.SkipZigTest;
    var pair = try TestBells.open(46105);
    defer pair.close();
    var silent = try TestBells.open(46127);
    defer silent.close();
    pair.leader.size = 3;
    pair.leader.bells[2] = silent.leader.bells[1];
    silent.leader.bells[1] = .{};
    var worker: TestReadyWorker = .{ .bell = pair.worker, .deadline = testDeadline(1500) };
    const t = try std.Thread.spawn(.{}, TestReadyWorker.run, .{&worker});
    const result = leaderAwaitReadyThenGo(&pair.leader.bells, 3, testDeadline(80));
    t.join();
    try std.testing.expectError(error.DistNoReady, result);
    try std.testing.expectEqual(error.DistAborted, worker.result.?);
    var byte: [1]u8 = undefined;
    try silent.worker.readAll(&byte);
    try std.testing.expectEqual(@as(u8, 7), byte[0]);
}

test "dist bell READY matrix 4: failure code aborts open on both ranks" {
    if (comptime builtin.os.tag != .macos) return error.SkipZigTest;
    for ([_]u8{ 1, 2, 3 }) |code| {
        var pair = try TestBells.open(46107);
        defer pair.close();
        const deadline = testDeadline(1500);
        var worker: TestReadyWorker = .{ .bell = pair.worker, .deadline = deadline, .frame = .{ .code = code } };
        const t = try std.Thread.spawn(.{}, TestReadyWorker.run, .{&worker});
        try std.testing.expectError(error.DistWorkerLoadFailed, leaderAwaitReadyThenGo(&pair.leader.bells, 2, deadline));
        t.join();
        try std.testing.expectEqual(error.DistAborted, worker.result.?);
    }
}

test "dist bell READY matrix 5: leader EOF before GO is a clean abort" {
    if (comptime builtin.os.tag != .macos) return error.SkipZigTest;
    var pair = try TestBells.open(46109);
    defer pair.close();
    var worker: TestReadyWorker = .{ .bell = pair.worker, .deadline = testDeadline(1000) };
    const t = try std.Thread.spawn(.{}, TestReadyWorker.run, .{&worker});
    var frame: [16]u8 = undefined;
    try pair.leader.bells[1].readAll(&frame);
    _ = std.c.close(pair.leader.bells[1].fd);
    pair.leader.bells[1] = .{};
    t.join();
    try std.testing.expectEqual(error.DistAborted, worker.result.?);
}

test "dist bell READY matrix 6: partial GO uses the fatal exit cause" {
    var go: GoProgress = .{};
    try go.check(799, 1000);
    go.sent = 1;
    try std.testing.expectError(error.DistPartialGo, go.check(800, 1000));
    try std.testing.expectEqual(error.DistPartialGo, go.writeFailed());
    go.sent = 0;
    try std.testing.expectError(error.DistNoGo, go.check(800, 1000));
    try std.testing.expectEqual(error.DistNoGo, go.writeFailed());
    var buf: [160]u8 = undefined;
    try std.testing.expect(std.mem.indexOf(u8, dist_watchdog.exitRecord(&buf, 1, .partial_go, null, 0, null), "cause=partial_go") != null);
}

test "dist bell READY matrix 7: no GO reaches the worker deadline" {
    if (comptime builtin.os.tag != .macos) return error.SkipZigTest;
    var pair = try TestBells.open(46111);
    defer pair.close();
    try std.testing.expectError(error.DistNoGo, workerReadyAwaitGo(pair.worker, testDeadline(80), .{}));
}

test "dist bell READY matrix 8: unexpected GO byte is rejected" {
    if (comptime builtin.os.tag != .macos) return error.SkipZigTest;
    var pair = try TestBells.open(46113);
    defer pair.close();
    try pair.leader.bells[1].writeAll(&.{5});
    try std.testing.expectError(error.DistBadMessage, workerReadyAwaitGo(pair.worker, testDeadline(1000), .{}));
}

test "dist bell READY matrix 9: split frame assembles little endian words" {
    if (comptime builtin.os.tag != .macos) return error.SkipZigTest;
    var pair = try TestBells.open(46115);
    defer pair.close();
    const bytes = [_]u8{ 0, 255, 255, 255, 0x34, 0x12, 0, 0, 0, 0, 0, 0, 0x78, 0x56, 0, 0 };
    var decoder: BellDecoder = .{};
    try pair.worker.writeAll(bytes[0..3]);
    try std.testing.expectEqual(@as(?BellWord, null), try decoder.read(pair.leader.bells[1]));
    try pair.worker.writeAll(bytes[3..9]);
    try std.testing.expectEqual(@as(?BellWord, null), try decoder.read(pair.leader.bells[1]));
    try pair.worker.writeAll(bytes[9..]);
    const got = (try testReadBellWord(&decoder, pair.leader.bells[1])).ready;
    try std.testing.expectEqual(@as(u32, 0x1234), got.resident_mib);
    try std.testing.expectEqual(@as(u32, 0x5678), got.load_ms);
    try std.testing.expectEqualSlices(u8, &bytes, &got.encode());
}

test "dist bell READY matrix 10: oversized step times stay below READY tag" {
    if (comptime builtin.os.tag != .macos) return error.SkipZigTest;
    var pair = try TestBells.open(46117);
    defer pair.close();
    for ([_]i64{ -1, 0, 123, 0xFFFFFF00, std.math.maxInt(i64) }) |us| {
        try sendStepTime(pair.worker, us);
        var decoder: BellDecoder = .{};
        const got = (try testReadBellWord(&decoder, pair.leader.bells[1])).step;
        try std.testing.expectEqual(@as(u32, @intCast(std.math.clamp(us, 0, 0xFFFFFEFF))), got);
    }
}

const TestBellWatcher = struct {
    fn run(link: *const Link, state: *LinkState) void {
        _ = watchBells(link, state);
    }
};

fn testColdPeer(link: Link) Peer {
    return .{ .link = link, .mode = .rows, .model = "test", .split = null };
}

test "dist bell READY matrix 11: cold READY reaches awaitReady through watcher" {
    if (comptime builtin.os.tag != .macos) return error.SkipZigTest;
    var pair = try TestBells.open(46119);
    defer pair.close();
    var state: LinkState = .{};
    var peer_state = testColdPeer(pair.leader);
    const t = try std.Thread.spawn(.{}, TestBellWatcher.run, .{ &pair.leader, &state });
    defer {
        _ = std.c.shutdown(pair.worker.fd, std.c.SHUT.RDWR);
        t.join();
    }
    try sendReadyFrame(pair.worker, .{ .resident_mib = 321, .load_ms = 789 });
    try peer_state.awaitReadyUntil(&state, testDeadline(1000));
    try std.testing.expect(peer_state.ready);
    try std.testing.expectEqual(@as(u64, 321), peer_state.remotes[1].resident_mib);
    try std.testing.expectEqual(@as(u32, 789), state.ready_load_ms[1].load(.acquire));
    _ = std.c.close(pair.worker.fd);
    pair.worker = .{};
}

test "dist bell READY matrix 12: cold load failure marks peer failed" {
    if (comptime builtin.os.tag != .macos) return error.SkipZigTest;
    for ([_]u8{ 1, 2, 3 }) |code| {
        var pair = try TestBells.open(46121);
        defer pair.close();
        var state: LinkState = .{};
        var peer_state = testColdPeer(pair.leader);
        try sendReadyFrame(pair.worker, .{ .code = code });
        _ = std.c.close(pair.worker.fd);
        pair.worker = .{};
        const t = try std.Thread.spawn(.{}, TestBellWatcher.run, .{ &pair.leader, &state });
        t.join();
        try std.testing.expectError(error.DistWorkerLoadFailed, peer_state.awaitReadyUntil(&state, testDeadline(1000)));
        try std.testing.expect(peer_state.failed);
    }
}

test "dist bell READY matrix 13: cold load deadline marks peer failed" {
    var state: LinkState = .{};
    var peer_state = testColdPeer(.{ .group = .{}, .rank = 0, .size = 2 });
    try std.testing.expectError(error.DistReadyTimeout, peer_state.awaitReadyUntil(&state, testDeadline(20)));
    try std.testing.expect(peer_state.failed);
}

test "dist bell READY matrix 14: bell EOF interrupts cold load promptly" {
    if (comptime builtin.os.tag != .macos) return error.SkipZigTest;
    var pair = try TestBells.open(46123);
    defer pair.close();
    var state: LinkState = .{};
    var peer_state = testColdPeer(pair.leader);
    const t = try std.Thread.spawn(.{}, TestBellWatcher.run, .{ &pair.leader, &state });
    defer {
        _ = std.c.shutdown(pair.worker.fd, std.c.SHUT.RDWR);
        t.join();
    }
    _ = std.c.close(pair.worker.fd);
    pair.worker = .{};
    const start = crt.monotonicMs();
    try std.testing.expectError(error.DistWorkerLeft, peer_state.awaitReadyUntil(&state, testDeadline(5000)));
    try std.testing.expect(crt.monotonicMs() - start < 1000);
    try std.testing.expect(peer_state.failed);
}

test "dist bell READY matrix 15: workers without the current protocol are refused at hello" {
    for ([_]i32{ 10, 11 }) |old| try std.testing.expectError(error.DistProtocolMismatch, Hello.parse(.{ MAGIC, old, 0, 0, 0, 0, 0, 0 }));
    _ = try Hello.parse(.{ MAGIC, VERSION, 0, 0, 0, 0, 0, 0 });
}

test "dist bell READY matrix 16: rows calibration reaches peer through frame" {
    if (comptime builtin.os.tag != .macos) return error.SkipZigTest;
    var pair = try TestBells.open(46125);
    defer pair.close();
    var state: LinkState = .{};
    var peer_state = testColdPeer(pair.leader);
    const t = try std.Thread.spawn(.{}, TestBellWatcher.run, .{ &pair.leader, &state });
    defer {
        _ = std.c.shutdown(pair.worker.fd, std.c.SHUT.RDWR);
        t.join();
    }
    try sendReadyFrame(pair.worker, .{ .resident_mib = 555, .calib_us = 23456, .load_ms = 123 });
    try peer_state.awaitReadyUntil(&state, testDeadline(1000));
    try std.testing.expectEqual(@as(u64, 23456), peer_state.worker_calib_us);
    _ = std.c.close(pair.worker.fd);
    pair.worker = .{};
}

test "dist bell a rows worker's READY leaves its idle exit unarmed" {
    if (comptime builtin.os.tag != .macos) return error.SkipZigTest;
    var pair = try TestBells.open(46133);
    defer pair.close();
    var state: LinkState = .{};
    defer state.follower_pending.store(false, .release);
    state.follower_pending.store(false, .release);
    var worker_link: Link = .{ .group = .{}, .rank = 1, .size = 2 };
    worker_link.state = &state;
    worker_link.bells[0] = pair.worker;
    worker_link.attachBells();
    pair.worker.state = &state;
    var ready: WorkerReady = .{ .link = &worker_link, .started_ms = testDeadline(0) };
    try WorkerReady.rows(&ready, 1, 2);
    var decoder: BellDecoder = .{};
    try std.testing.expectEqual(@as(u32, 2), (try testReadBellWord(&decoder, pair.leader.bells[1])).ready.calib_us);
    try std.testing.expect(ready.sent);
    try std.testing.expect(!state.follower_pending.load(.acquire));
}

/// Waits on the bells for a worker to leave: a read returns only at its end.
fn watchWorker(link: Link) void {
    const state = link.state.?;
    const lost = watchBells(&link, state);
    if (lost == 0 or state.closing.load(.acquire)) return;
    log.warn("[dist] attempt={d} worker rank {d} left\n", .{ state.n, lost });
    if (!state.switch_open and link.size == 2) {
        while (state.recvs.load(.acquire) == 0) if (sleepUnlessStopped(state, 100)) return;
    }
    if (state.closing.load(.acquire) or state.releasing.load(.acquire)) return;
    if (@import("../../../host/http/server.zig").launch().global_scheduler) |sch| @import("../scheduler.zig").failInFlight(sch, "DistWorkerLeft");
    if (sleepUnlessStopped(state, 2000)) return;
    state.exit(.worker_left, lost, null);
}

/// Right after the worker joins, over the bell connection (no MLX off the inference thread):
/// the group's model by name and `modelIdentity`, answered with whether the worker has it.
fn askForModel(link: *const Link, rank: u32) !void {
    const bell = link.bells[rank];
    if (!bell.open()) return;
    const g = cluster().group.?;
    const name = std.fs.path.basename(g.model);
    const id = try modelIdentity(g.io, g.allocator, g.model);
    var head: [10]u8 = undefined;
    std.mem.writeInt(u16, head[0..2], @intCast(name.len), .little);
    std.mem.writeInt(u64, head[2..10], id, .little);
    try bell.writeAll(&head);
    try bell.writeAll(name);
    var fds = [_]std.posix.pollfd{.{ .fd = bell.fd, .events = std.posix.POLL.IN, .revents = 0 }};
    if (try std.posix.poll(&fds, 30_000) == 0) return error.DistNoAnswer;
    var reply: [5]u8 = undefined;
    try bell.readAll(&reply);
    cluster().copies[rank].store(reply[0], .release);
    cluster().free_mib[rank].store(std.mem.readInt(u32, reply[1..5], .little), .release);
    if (reply[0] != 0) log.err("[dist] worker rank {d} does not have {s} ({s}): download the same model there, then join again\n", .{ rank, name, copyName(reply[0]) });
}

/// The worker's side of `askForModel`: looks the model up in its folders and answers with
/// that and its free memory.
fn answerModel(io: std.Io, allocator: std.mem.Allocator, link: *const Link, roots: []const []const u8) !bool {
    if (!link.bells[0].open()) return true;
    var head: [10]u8 = undefined;
    try link.bells[0].readAll(&head);
    var name_buf: [256]u8 = undefined;
    const len = std.mem.readInt(u16, head[0..2], .little);
    if (len == 0 or len > name_buf.len) return error.DistBadMessage;
    const name = name_buf[0..len];
    try link.bells[0].readAll(name);
    var reply: [5]u8 = undefined;
    std.mem.writeInt(u32, reply[1..5], @intCast(@min(status.getAvailableMemBytes() >> 20, std.math.maxInt(u32))), .little);
    const found = findModel(io, allocator, roots, name, std.mem.readInt(u64, head[2..10], .little)) catch |err| switch (err) {
        error.DistModelMismatch => {
            log.err("[dist] worker: {s} here is not the host's copy (quantization or revision)\n", .{name});
            reply[0] = @intCast(MISMATCH);
            try link.bells[0].writeAll(&reply);
            return false;
        },
        else => return err,
    };
    if (found) |dir| {
        allocator.free(dir);
        reply[0] = 0;
        try link.bells[0].writeAll(&reply);
        return true;
    }
    log.err("[dist] worker: no model named {s} in {s}\n", .{ name, roots[0] });
    reply[0] = @intCast(NOT_FOUND);
    try link.bells[0].writeAll(&reply);
    return false;
}

/// The folder named `name` in `roots` (`<root>/<name>` or `<root>/<org>/<name>`) whose
/// `modelIdentity` is `id`; `error.DistModelMismatch` when only other copies of that name exist.
fn findModel(io: std.Io, allocator: std.mem.Allocator, roots: []const []const u8, name: []const u8, id: u64) !?[]u8 {
    var seen = false;
    for (roots) |root| {
        if (try sameModel(io, allocator, &.{ root, name }, id, &seen)) |p| return p;
        var dir = std.Io.Dir.openDirAbsolute(io, root, .{ .iterate = true }) catch continue;
        defer dir.close(io);
        var it = dir.iterate();
        while (it.next(io) catch null) |e| {
            if (e.kind != .directory) continue;
            if (try sameModel(io, allocator, &.{ root, e.name, name }, id, &seen)) |p| return p;
        }
    }
    return if (seen) error.DistModelMismatch else null;
}

fn sameModel(io: std.Io, allocator: std.mem.Allocator, parts: []const []const u8, id: u64, seen: *bool) !?[]u8 {
    const p = try std.fs.path.join(allocator, parts);
    errdefer allocator.free(p);
    const here = modelIdentity(io, allocator, p) catch {
        allocator.free(p);
        return null;
    };
    seen.* = true;
    if (here == id) return p;
    allocator.free(p);
    return null;
}

/// `mlx-serve worker --coordinator IP:PORT --device rdma_enX [--model-dir DIR]...`: joins
/// rank 0's group, loads the part of the model rank 0 names (found by folder name in its own
/// model folders, `~/.mlx-serve/models` by default), then runs forwards until rank 0 stops it.
pub fn cmdWorker(io: std.Io, allocator: std.mem.Allocator, args: []const []const u8) !void {
    bindInferenceThread();
    errdefer noteDistExit(.worker_error, 0, null);
    // JACCL (RDMA over Thunderbolt) exists only on macOS.
    if (comptime builtin.os.tag != .macos) return error.RdmaUnavailable;
    preferFastSynch();
    var roots: std.ArrayList([]const u8) = .empty;
    defer roots.deinit(allocator);
    var rest: std.ArrayList([]const u8) = .empty;
    defer rest.deinit(allocator);
    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--model-dir") and i + 1 < args.len) {
            try roots.append(allocator, args[i + 1]);
            i += 1;
        } else try rest.append(allocator, args[i]);
    }
    var home_root: ?[]u8 = null;
    defer if (home_root) |h| allocator.free(h);
    if (roots.items.len == 0) {
        home_root = try @import("../../../host/cli/cli.zig").modelsRootPath(allocator, std.mem.span(env.get(.HOME) orelse "/tmp"));
        try roots.append(allocator, home_root.?);
    }
    const with_rank = try allocator.alloc([]const u8, rest.items.len + 2);
    defer allocator.free(with_rank);
    const cfg = try parseArgs(workerArgs(rest.items, with_rank));

    // A failed JACCL init leaks its socket, so a retry is a fresh process: the app restarts
    // the worker a few seconds later (headless: run it under launchd or in a loop).
    var link = Link.open(io, cfg) catch |err| {
        if (err == error.RdmaInitFailed) log.info("[dist] worker: the host at {s} is not listening yet; retrying\n", .{cfg.coordinator});
        return err;
    };
    defer link.deinit();
    var host_buf: [std.posix.HOST_NAME_MAX]u8 = undefined;
    const host = std.posix.gethostname(&host_buf) catch "mac";
    const short = host[0..@min(host.len, 64)];
    if (!try answerModel(io, allocator, &link, roots.items)) {
        // The host refuses to use this worker; sleep until it leaves (unpair, restart).
        while (true) _ = try link.bells[0].wait();
    }
    log.info("[dist] worker: joined, waiting for the host to load a model\n", .{});
    // The host rings when it reads the hello, which it does when it first loads the model.
    _ = try link.bells[0].wait();
    try sendMsg(&link, Hello.here(short.len).msg(), 0);
    if (short.len > 0) try sendBytes(&link, short, 0);
    while (true) {
        const pmsg = try recvMsg(&link, 0);
        if (pmsg[0] == OP_STOP or !try servePlan(io, allocator, &link, roots.items, pmsg)) break;
        log.info("[dist] worker: released its part, waiting for the host to load a model\n", .{});
    }
}

/// A worker past a pair: exits when rank 0's bell closes, even while the inference thread waits inside a sum. It only
/// peeks, so the rings stay for `Bell.wait`.
fn watchHost(link: Link) void {
    const state = link.state.?;
    const b = link.bells[0];
    var fds = [_]std.posix.pollfd{.{ .fd = b.fd, .events = std.posix.POLL.IN, .revents = 0 }};
    while (!state.closing.load(.acquire) and !state.releasing.load(.acquire)) {
        if ((std.posix.poll(&fds, 50) catch return) == 0) continue;
        if (state.closing.load(.acquire) or state.releasing.load(.acquire)) return;
        // The worker closed the bell itself (a clean stop): nothing to watch.
        if (fds[0].revents & std.posix.POLL.NVAL != 0) return;
        var x: [1]u8 = undefined;
        const n = std.c.recv(b.fd, &x, 1, std.c.MSG.PEEK | std.c.MSG.DONTWAIT);
        if (n < 0) switch (std.c._errno().*) {
            @backingInt(std.c.E.INTR), @backingInt(std.c.E.AGAIN) => continue,
            @backingInt(std.c.E.BADF), @backingInt(std.c.E.NOTSOCK) => return,
            else => break,
        };
        if (n == 0 or dist_watchdog.tcpPeerGone(b.fd)) break;
        crt.sleepMs(20); // a ring the inference thread has yet to read
    }
    followerExit(state, .eof);
}

const WorkerReady = struct {
    link: *const Link,
    started_ms: u64,
    sent: bool = false,

    fn send(self: *WorkerReady, code: u8, resident_mib: u32, calib_us: u32) !void {
        try sendReadyFrame(self.link.bells[0], .{ .code = code, .resident_mib = resident_mib, .calib_us = calib_us, .load_ms = loadElapsedMs(self.started_ms) });
        self.sent = true;
        if (code != 0) return;
        self.link.state.?.follower_pending.store(true, .release);
    }

    /// A rows worker reports READY but never arms the idle exit: rank 0 runs a video job on its inference thread, which
    /// stops the loop ticks that gate its heartbeats, so a long job would end an armed worker.
    fn rows(ctx: *anyopaque, resident_mib: u32, calib_us: u32) anyerror!void {
        const self: *WorkerReady = @ptrCast(@alignCast(ctx));
        try sendReadyFrame(self.link.bells[0], .{ .code = 0, .resident_mib = resident_mib, .calib_us = calib_us, .load_ms = loadElapsedMs(self.started_ms) });
        self.sent = true;
    }
};

/// One plan: finds and loads this Mac's part, then serves until rank 0 stops it (false) or
/// releases it (true: the host unloaded the model and may plan again).
fn servePlan(io: std.Io, allocator: std.mem.Allocator, link: *const Link, roots: []const []const u8, pmsg: Msg) !bool {
    var ready: WorkerReady = .{ .link = link, .started_ms = @intCast(crt.monotonicMs()) };
    return servePlanReady(io, allocator, link, roots, pmsg, &ready) catch |err| {
        if (!ready.sent) ready.send(switch (err) {
            error.DistModelMismatch => MISMATCH,
            error.DistModelNotFound => NOT_FOUND,
            else => LOAD_FAILED,
        }, 0, 0) catch {};
        return err;
    };
}

fn servePlanReady(io: std.Io, allocator: std.mem.Allocator, link: *const Link, roots: []const []const u8, pmsg: Msg, ready: *WorkerReady) !bool {
    var pl = try Plan.parse(pmsg);
    var name_buf: [256]u8 = undefined;
    const name_len: usize = @intCast(pmsg[5]);
    if (name_len == 0 or name_len > name_buf.len) return error.DistBadMessage;
    try recvBytes(link, name_buf[0..name_len], 0);
    const name = name_buf[0..name_len];
    const idm = try recvMsg(link, 0);
    if (idm[0] != MAGIC) return error.DistProtocolMismatch;
    const id = @as(u64, @as(u32, @bitCast(idm[1]))) | @as(u64, @as(u32, @bitCast(idm[2]))) << 32;
    pl.stages = idm[3] != 0;
    switch (pl.mode) {
        .pipeline => log.info("[dist] worker: {s} layers {d}..{d} of {d} (pipeline)\n", .{ name, pl.span.start, pl.span.end, pl.span.total }),
        .tensor => log.info("[dist] worker rank {d}: {s}, 1/{d} of every expert (tensor)\n", .{ link.rank, name, link.size }),
        .rows => log.info("[dist] worker: {s}, the whole DiT (video steps)\n", .{name}),
    }

    const found = findModel(io, allocator, roots, name, id) catch |err| {
        if (err == error.DistModelMismatch) log.err("[dist] worker: {s} here is not the host's copy (quantization or revision)\n", .{name});
        return err;
    };
    const dir = found orelse {
        log.err("[dist] worker: no model named {s} in {s}\n", .{ name, roots[0] });
        return error.DistModelNotFound;
    };
    defer allocator.free(dir);
    if (pl.mode == .rows) {
        const wl = wireOf(link);
        const released = h3_split.serveWorker(allocator, io, &wl, dir, pl.hidden, @intCast(pl.span.total), .{ .ctx = ready, .send = &WorkerReady.rows }) catch |err| {
            log.err("[dist] worker load failed: {s}\n", .{@errorName(err)});
            return err;
        };
        if (!released) log.info("[dist] worker: stopped by rank 0\n", .{});
        return released;
    }
    var config = try model_mod.parseConfig(io, allocator, dir);
    defer config.deinit(allocator);
    if (!pl.matches(&config)) {
        log.err("[dist] worker: {s} here has a different geometry than the host's\n", .{dir});
        return error.DistModelMismatch;
    }
    if (config.isDsv41()) return serveDsv41(io, allocator, link, dir, &config, pl, ready) catch |err| {
        log.err("[dist] worker: {s}\n", .{@errorName(err)});
        return err;
    };
    var xfm = load(io, allocator, dir, &config, pl.mode, pl.span, .{ .rank = link.rank, .size = link.size }) catch |err| {
        log.err("[dist] worker load failed: {s}\n", .{@errorName(err)});
        return err;
    };
    defer xfm.deinit(allocator);
    var active: usize = 0;
    _ = mlx.mlx_get_active_memory(&active);
    try ready.send(0, @intCast(active >> 20), 0);
    log.info("[dist] worker ready ({d} MiB resident)\n", .{active >> 20});

    const t = &xfm.t;
    var ctx: ForwardCtx = .{ .cache = &xfm.cache, .moe_seq_offset = &xfm.off, .ssm_entries = xfm.entries, .capture_hidden = null, .vision_embeddings = null };
    var forwards: u64 = 0;
    while (true) {
        const m = try recvMsg(link, 0);
        switch (m[0]) {
            OP_STOP => break,
            OP_RELEASE => return true,
            OP_FORWARD => {
                const offset: usize = @as(usize, @intCast(m[3])) | (@as(usize, @intCast(m[4])) << 31);
                if (offset == 0) {
                    try xfm.reset();
                } else if (offset != xfm.off) {
                    log.err("[dist] forward at {d} but this worker holds {d} tokens\n", .{ offset, xfm.off });
                    return error.DistPeerDiverged;
                }
                const shape = [_]c_int{ m[1], m[2], m[5] };
                const h = try queuedRecv(link, &shape, @fromBackingInt(@intCast(m[6])), 0);
                defer _ = mlx.mlx_array_free(h);
                const raw = try t.forwardQwen4Stream(&ctx, h);
                defer _ = mlx.mlx_array_free(raw);
                // Rank 0 receives exactly the bytes it sent: JACCL sizes its buffers from the byte count.
                if (!std.mem.eql(c_int, mlx.getShape(raw), mlx.getShape(h))) return error.DistBadMessage;
                var out = raw;
                if (mlx.mlx_array_dtype(raw) != mlx.mlx_array_dtype(h)) {
                    out = mlx.mlx_array_new();
                    try mlx.check(mlx.mlx_astype(&out, raw, mlx.mlx_array_dtype(h), t.s));
                }
                defer if (out.ctx != raw.ctx) {
                    _ = mlx.mlx_array_free(out);
                };
                const sent = try link.send(out, 0);
                defer _ = mlx.mlx_array_free(sent);
                try evalOne(sent);
                forwards += 1;
                if (m[1] * m[2] > 1 or forwards % 256 == 0) log.info("[dist] worker: forward {d}: {d}x{d} rows, now at {d} tokens\n", .{ forwards, m[1], m[2], xfm.off });
            },
            OP_MOE => {
                const x = try queuedRecv(link, &.{ m[1], m[2], m[3] }, @fromBackingInt(@intCast(m[5])), 0);
                defer _ = mlx.mlx_array_free(x);
                const logits = try queuedRecv(link, &.{ m[1], m[2], m[4] }, @fromBackingInt(@intCast(m[7])), 0);
                defer _ = mlx.mlx_array_free(logits);
                const raw = try t.moeExpertsRouted(x, @intCast(m[6]), logits);
                defer _ = mlx.mlx_array_free(raw);
                if (!std.mem.eql(c_int, mlx.getShape(raw), mlx.getShape(x))) return error.DistBadMessage;
                var out = raw;
                if (mlx.mlx_array_dtype(raw) != mlx.mlx_array_dtype(x)) {
                    out = mlx.mlx_array_new();
                    try mlx.check(mlx.mlx_astype(&out, raw, mlx.mlx_array_dtype(x), t.s));
                }
                defer if (out.ctx != raw.ctx) {
                    _ = mlx.mlx_array_free(out);
                };
                const sent = try link.send(out, 0);
                defer _ = mlx.mlx_array_free(sent);
                try evalOne(sent);
                forwards += 1;
            },
            else => return error.DistBadMessage,
        }
    }
    log.info("[dist] worker: stopped by rank 0 after {d} forwards\n", .{forwards});
    return false;
}

/// The worker's part of the pack: its weights, the trunk module over them and the one request's caches.
const Loaded = struct {
    w: model_mod.Weights,
    t: Qwen4Trunk,
    cache: KVCache,
    entries: []SSMCacheEntry,
    off: usize = 0,

    /// A new request: no keys, no recurrent state.
    fn reset(self: *Loaded) !void {
        try self.cache.reinit(self.t.config.num_hidden_layers, self.cache.config);
        for (self.entries) |*e| {
            freeEntry(e);
            e.* = .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false };
        }
        self.off = 0;
    }

    fn deinit(self: *Loaded, a: std.mem.Allocator) void {
        for (self.entries) |*e| freeEntry(e);
        a.free(self.entries);
        self.cache.deinit();
        self.t.deinit();
        self.w.deinit();
    }

    fn freeEntry(e: *SSMCacheEntry) void {
        ssmFreeSpecCapture(e);
        _ = mlx.mlx_array_free(e.conv_state);
        _ = mlx.mlx_array_free(e.ssm_state);
        ssmFreeQsaState(e);
    }
};

fn load(io: std.Io, allocator: std.mem.Allocator, dir: []const u8, config: *model_mod.ModelConfig, mode: Mode, span: LayerSpan, sh: Shard) !Loaded {
    if (!config.isQwen4()) return error.DistArchUnsupported;
    switch (mode) {
        .pipeline => try config.applyLayerSpan(span),
        .tensor => try config.applyShard(sh),
        .rows => return error.DistArchUnsupported,
    }
    var w = try model_mod.loadModelWeights(io, allocator, dir, config, false);
    errdefer w.deinit();
    var t = try Qwen4Trunk.init(io, allocator, config.*, &w);
    errdefer t.deinit();
    _ = mlx.applyWiredPolicy();
    t.compileMoeRouting();
    if (config.linear_num_key_heads > 0) t.compileGdnGate();
    var cache = try KVCache.init(allocator, config.num_hidden_layers);
    errdefer cache.deinit();
    const entries = try allocator.alloc(SSMCacheEntry, config.num_hidden_layers);
    for (entries) |*e| e.* = .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false };
    return .{ .w = w, .t = t, .cache = cache, .entries = entries };
}

fn evalOne(a: mlx.mlx_array) !void {
    try mlx.check(mlx.mlx_array_eval(a));
}

/// `--chain`: a decode-shaped tensor-parallel token, `layers` x (a one-row matmul on the GPU, then an all_sum),
/// built as one graph and evaluated once, against the same token evaluated layer by layer, and without the all_sums.
fn chainProbe(io: std.Io, link: *const Link, s: mlx.mlx_stream) !void {
    const d: c_int = 5120;
    const layers = 40;
    var key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_random_key(&key, 7));
    var w = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w);
    try mlx.check(mlx.mlx_random_normal(&w, &[_]c_int{ d, d }, 2, .bfloat16, 0, 1.0 / @sqrt(@as(f32, @floatFromInt(d))), key, s));
    var x0 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x0);
    try mlx.check(mlx.mlx_ones(&x0, &[_]c_int{ 1, d }, 2, .bfloat16, s));
    try evalOne(w);
    try evalOne(x0);
    const half = mlx.mlx_array_new_float(0.5);
    defer _ = mlx.mlx_array_free(half);
    const Chain = enum { one_graph, per_layer, no_all_sum, all_sum_only, cpu_only };
    for (std.enums.values(Chain)) |mode| {
        var total: u64 = 0;
        const iters = 30;
        for (0..iters + 3) |it| {
            var sw = io_util.Stopwatch.init(io);
            var x = mlx.mlx_array_new();
            _ = mlx.mlx_array_set(&x, x0);
            const cs = mlx.mlx_default_cpu_stream_new();
            defer _ = mlx.mlx_stream_free(cs);
            for (0..layers) |_| {
                var y = mlx.mlx_array_new();
                if (mode == .all_sum_only or mode == .cpu_only) _ = mlx.mlx_array_set(&y, x) else try mlx.check(mlx.mlx_matmul(&y, x, w, s));
                if (mode != .no_all_sum) {
                    const r = try link.allSum(y);
                    _ = mlx.mlx_array_free(y);
                    y = r;
                }
                var z = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_multiply(&z, y, half, if (mode == .cpu_only) cs else s));
                _ = mlx.mlx_array_free(y);
                _ = mlx.mlx_array_free(x);
                x = z;
                if (mode == .per_layer) try evalOne(x);
            }
            try evalOne(x);
            _ = mlx.mlx_array_free(x);
            if (it >= 3) total += sw.read();
        }
        const per_token_us = @as(f64, @floatFromInt(total)) / iters / 1000.0;
        log.info("[dist-probe] chain {s}: {d} layers a token, {d:.1} us a token, {d:.1} us a layer\n", .{ @tagName(mode), layers, per_token_us, per_token_us / layers });
    }
}

/// `mlx-serve dist-probe --rank R --coordinator IP:PORT --device rdma_enX`:
/// joins the group and measures round trips at the sizes inference moves
/// (one decode row of a hyper-connection stream, a prefill chunk) and all_sum.
pub fn cmdProbe(io: std.Io, args: []const []const u8) !void {
    log.info("[dist-probe] jaccl formation deadline: {s}\n", .{if (mlx.jacclFormationDeadlineAvailable()) "available" else "missing"});
    // `--soak N`: N header + one-row-stream round trips, the shape of a pipeline decode hop.
    var soak: u64 = 0;
    var chain = false;
    var rest: [16][]const u8 = undefined;
    var n_rest: usize = 0;
    var ai: usize = 0;
    while (ai < args.len) : (ai += 1) {
        if (std.mem.eql(u8, args[ai], "--soak") and ai + 1 < args.len) {
            soak = std.fmt.parseInt(u64, args[ai + 1], 10) catch return error.InvalidArgs;
            ai += 1;
        } else if (std.mem.eql(u8, args[ai], "--chain")) {
            chain = true;
        } else if (n_rest < rest.len) {
            rest[n_rest] = args[ai];
            n_rest += 1;
        }
    }
    const cfg = try parseArgs(rest[0..n_rest]);
    var link = try Link.open(io, cfg);
    defer link.deinit();
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const other: u32 = if (link.rank == 0) 1 else 0;
    if (soak > 0) return soakLink(io, &link, other, soak, s);
    if (chain) return chainProbe(io, &link, s);

    const sizes = [_]c_int{ 1, 5120, 10240, 262144, 6553600 }; // f32 elements: 4 B .. 25 MB
    for (sizes) |n| {
        const shape = [_]c_int{n};
        var x = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_ones(&x, &shape, 1, .float32, s));
        defer _ = mlx.mlx_array_free(x);
        try evalOne(x);
        const iters: u32 = if (n > 1_000_000) 20 else 200;
        var best: u64 = std.math.maxInt(u64);
        var total: u64 = 0;
        for (0..iters + 5) |i| {
            var sw = io_util.Stopwatch.init(io);
            if (link.rank == 0) {
                const sent = try link.send(x, other);
                defer _ = mlx.mlx_array_free(sent);
                try evalOne(sent);
                const back = try link.recv(&shape, .float32, other);
                defer _ = mlx.mlx_array_free(back);
                try evalOne(back);
            } else {
                const got = try link.recv(&shape, .float32, other);
                defer _ = mlx.mlx_array_free(got);
                try evalOne(got);
                const sent = try link.send(got, other);
                defer _ = mlx.mlx_array_free(sent);
                try evalOne(sent);
            }
            const ns = sw.read();
            if (i >= 5) {
                total += ns;
                best = @min(best, ns);
            }
        }
        const bytes: f64 = @floatFromInt(@as(u64, @intCast(n)) * 4);
        const mean_us = @as(f64, @floatFromInt(total)) / @as(f64, @floatFromInt(iters)) / 1000.0;
        log.info("[dist-probe] round trip {d:>9} B: mean {d:>9.1} us, best {d:>9.1} us, {d:>6.2} GB/s each way\n", .{ n * 4, mean_us, @as(f64, @floatFromInt(best)) / 1000.0, 2.0 * bytes / (mean_us * 1000.0) });
    }
    for ([_]c_int{ 10240, 262144 }) |n| {
        const shape = [_]c_int{n};
        var x = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_ones(&x, &shape, 1, .bfloat16, s));
        defer _ = mlx.mlx_array_free(x);
        try evalOne(x);
        var total: u64 = 0;
        const iters: u32 = 200;
        var v: f32 = 0;
        for (0..iters + 5) |i| {
            var sw = io_util.Stopwatch.init(io);
            const r = try link.allSum(x);
            defer _ = mlx.mlx_array_free(r);
            try evalOne(r);
            if (i >= 5) total += sw.read();
            if (i == 0) {
                var first = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(first);
                var f32a = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(f32a);
                try mlx.check(mlx.mlx_astype(&f32a, r, .float32, s));
                try mlx.check(mlx.mlx_sum(&first, f32a, false, s));
                try mlx.check(mlx.mlx_array_item_float32(&v, first));
            }
        }
        log.info("[dist-probe] all_sum bf16 {d:>9} B: mean {d:>9.1} us (sum check {d} == {d})\n", .{ n * 2, @as(f64, @floatFromInt(total)) / @as(f64, @floatFromInt(iters)) / 1000.0, v, n * @as(c_int, @intCast(link.size)) });
    }
    log.info("[dist-probe] done\n", .{});
}

fn soakLink(io: std.Io, link: *const Link, other: u32, n: u64, s: mlx.mlx_stream) !void {
    // Both ranks draw the same sizes: 1..2400 rows of a 20 KB stream, the range real hops span.
    var rng = std.Random.DefaultPrng.init(0x5eed);
    var sw = io_util.Stopwatch.init(io);
    var bytes: u64 = 0;
    for (0..n) |k| {
        const rows: c_int = rng.random().intRangeAtMost(c_int, 1, 2400);
        const shape = [_]c_int{ 1, rows, 10240 };
        if (link.rank == 0) {
            var x = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_ones(&x, &shape, 3, .bfloat16, s));
            defer _ = mlx.mlx_array_free(x);
            try evalOne(x);
            try sendMsg(link, .{ OP_FORWARD, rows, @intCast(k & 0x7fffffff), 0, 0, 0, 0, 0 }, other);
            const sent = try link.send(x, other);
            defer _ = mlx.mlx_array_free(sent);
            try evalOne(sent);
            const back = try link.recv(&shape, .bfloat16, other);
            defer _ = mlx.mlx_array_free(back);
            try evalOne(back);
        } else {
            const m = try recvMsg(link, other);
            if (m[1] != rows) return error.DistBadMessage;
            const got = try link.recv(&shape, .bfloat16, other);
            defer _ = mlx.mlx_array_free(got);
            try evalOne(got);
            const sent = try link.send(got, other);
            defer _ = mlx.mlx_array_free(sent);
            try evalOne(sent);
        }
        bytes += @as(u64, @intCast(rows)) * 20480;
        if ((k + 1) % 1000 == 0) {
            log.info("[dist-probe] soak: {d} round trips, {d} MB each way in {d} ms\n", .{ k + 1, bytes >> 20, sw.read() / std.time.ns_per_ms });
            sw = io_util.Stopwatch.init(io);
            bytes = 0;
        }
    }
    log.info("[dist-probe] soak: {d} round trips clean\n", .{n});
}

pub fn parseArgs(args: []const []const u8) !Config {
    var rank: ?u32 = null;
    var size: u32 = 2;
    var coordinator: ?[]const u8 = null;
    var device: ?[]const u8 = null;
    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const a = args[i];
        if (i + 1 >= args.len) {
            log.err("[dist] {s} needs a value\n", .{a});
            return error.InvalidArgs;
        }
        const v = args[i + 1];
        if (std.mem.eql(u8, a, "--rank")) {
            rank = std.fmt.parseInt(u32, v, 10) catch return error.InvalidArgs;
        } else if (std.mem.eql(u8, a, "--size")) {
            size = std.fmt.parseInt(u32, v, 10) catch return error.InvalidArgs;
        } else if (std.mem.eql(u8, a, "--coordinator")) {
            coordinator = v;
        } else if (std.mem.eql(u8, a, "--device")) {
            device = v;
        } else {
            log.err("[dist] unknown option {s}\n", .{a});
            return error.InvalidArgs;
        }
        i += 1;
    }
    if (rank == null or coordinator == null or device == null) {
        log.err("usage: mlx-serve dist-probe --rank N [--size N] --coordinator IP:PORT --device rdma_enX[,rdma_enY...]\n", .{});
        return error.InvalidArgs;
    }
    if (size < 2 or size > max_ranks or rank.? >= size) {
        log.err("[dist] --size takes 2..{d} ranks and --rank one below it: got rank {d} of {d}\n", .{ max_ranks, rank.?, size });
        return error.InvalidArgs;
    }
    return .{ .rank = rank.?, .size = size, .coordinator = coordinator.?, .device = device.? };
}

/// A worker's arguments with `--rank 1` added unless it names its own (a pair's worker is rank 1).
fn workerArgs(args: []const []const u8, out: [][]const u8) []const []const u8 {
    @memcpy(out[0..args.len], args);
    for (args) |a| if (std.mem.eql(u8, a, "--rank")) return out[0..args.len];
    out[args.len] = "--rank";
    out[args.len + 1] = "1";
    return out[0 .. args.len + 2];
}

test "deviceMatrixJson: each row names the local device, the diagonal is null" {
    var buf: [256]u8 = undefined;
    const j = try deviceMatrixJson(.{ .rank = 1, .coordinator = "x:1", .device = "rdma_en1" }, &buf);
    try std.testing.expectEqualStrings("[[null,\"rdma_en1\"],[\"rdma_en1\",null]]", j);
}

test "dist N-rank: a device list names each peer's port in this rank's row, and every row is a valid mesh" {
    var buf: [512]u8 = undefined;
    // Rank 2 of 4: its own entry is `-`; the other rows repeat names so JACCL's whole-square check passes.
    const j = try deviceMatrixJson(.{ .rank = 2, .size = 4, .coordinator = "x:1", .device = "rdma_en2,rdma_en3,-,rdma_en5" }, &buf);
    try std.testing.expectEqualStrings("[[null,\"rdma_en3\",\"rdma_en2\",\"rdma_en5\"],[\"rdma_en2\",null,\"rdma_en3\",\"rdma_en5\"],[\"rdma_en2\",\"rdma_en3\",null,\"rdma_en5\"],[\"rdma_en2\",\"rdma_en3\",\"rdma_en5\",null]]", j);
    // An empty own entry reads as `-`.
    const e = try deviceMatrixJson(.{ .rank = 0, .size = 4, .coordinator = "x:1", .device = ",a,b,c" }, &buf);
    try std.testing.expectEqualStrings("[[null,\"a\",\"b\",\"c\"],[\"a\",null,\"b\",\"c\"],[\"b\",\"a\",null,\"c\"],[\"c\",\"a\",\"b\",null]]", e);
    // A pair's list writes the pair's matrix.
    const p = try deviceMatrixJson(.{ .rank = 1, .size = 2, .coordinator = "x:1", .device = "rdma_en1,-" }, &buf);
    try std.testing.expectEqualStrings("[[null,\"rdma_en1\"],[\"rdma_en1\",null]]", p);
    // Past a pair one port cannot face every peer; a list must have one entry per rank, own entry empty.
    try std.testing.expectError(error.DistDeviceList, deviceMatrixJson(.{ .rank = 0, .size = 4, .coordinator = "x:1", .device = "rdma_en1" }, &buf));
    try std.testing.expectError(error.DistDeviceList, deviceMatrixJson(.{ .rank = 0, .size = 4, .coordinator = "x:1", .device = "-,a,b" }, &buf));
    try std.testing.expectError(error.DistDeviceList, deviceMatrixJson(.{ .rank = 1, .size = 4, .coordinator = "x:1", .device = "-,a,b,c" }, &buf));
    try std.testing.expectError(error.DistDeviceList, deviceMatrixJson(.{ .rank = 0, .size = 4, .coordinator = "x:1", .device = "-,a,,c" }, &buf));
}

test "dist switch startup: rank one primes its environment and invalid device order cannot change it" {
    const cfg: Config = .{ .rank = 1, .size = 2, .coordinator = "127.0.0.1:46201", .device = "rdma_en1,-" };
    try primeEnv(cfg);
    try std.testing.expect(envMatches("MLX_RANK", "1"));
    try std.testing.expect(envPrimed(cfg));
    var bad = cfg;
    bad.rank = 0;
    try std.testing.expectError(error.DistDeviceList, primeEnv(bad));
    try std.testing.expect(envPrimed(cfg));
    bad = cfg;
    bad.rank = 2;
    try std.testing.expectError(error.InvalidArgs, primeEnv(bad));
    try std.testing.expect(envPrimed(cfg));
}

test "dist switch startup: a switch-primed process turns on MLX fast sync before its first fence, unless set explicitly" {
    env.unset(.MLX_METAL_FAST_SYNCH);
    try primeEnv(.{ .rank = 1, .size = 2, .coordinator = "127.0.0.1:46202", .device = "rdma_en1,-" });
    try std.testing.expect(envMatches("MLX_METAL_FAST_SYNCH", "1"));
    env.set(.MLX_METAL_FAST_SYNCH, "0");
    try primeEnv(.{ .rank = 1, .size = 2, .coordinator = "127.0.0.1:46202", .device = "rdma_en1,-" });
    try std.testing.expect(envMatches("MLX_METAL_FAST_SYNCH", "0"));
    env.unset(.MLX_METAL_FAST_SYNCH);
}

test "dist N-rank: past a pair rank 0 listens on every address, each worker dials its own" {
    var buf: [64]u8 = undefined;
    try std.testing.expectEqualStrings("10.254.77.1:32323", try jacclCoordinator(.{ .rank = 0, .size = 2, .coordinator = "10.254.77.1:32323", .device = "d" }, &buf));
    try std.testing.expectEqualStrings("0.0.0.0:32323", try jacclCoordinator(.{ .rank = 0, .size = 4, .coordinator = "10.254.12.1:32323", .device = "d" }, &buf));
    try std.testing.expectEqualStrings("10.254.13.1:32323", try jacclCoordinator(.{ .rank = 2, .size = 4, .coordinator = "10.254.13.1:32323", .device = "d" }, &buf));
    const pair = try bellAddress(.{ .rank = 0, .size = 2, .coordinator = "10.254.77.1:32323", .device = "d" });
    try std.testing.expectEqual([4]u8{ 10, 254, 77, 1 }, pair.ip4);
    const four = try bellAddress(.{ .rank = 0, .size = 4, .coordinator = "10.254.12.1:32323", .device = "d" });
    try std.testing.expectEqual([4]u8{ 0, 0, 0, 0 }, four.ip4);
    try std.testing.expectEqual(@as(u16, 32323), four.port);
    const worker = try bellAddress(.{ .rank = 3, .size = 4, .coordinator = "10.254.14.1:32323", .device = "d" });
    try std.testing.expectEqual([4]u8{ 10, 254, 14, 1 }, worker.ip4);
}

test "dist N-rank: --rank and --size name the worker's place; a pair is the default" {
    const pair = try parseArgs(&.{ "--rank", "1", "--coordinator", "10.0.0.1:5", "--device", "rdma_en1" });
    try std.testing.expectEqual(@as(u32, 2), pair.size);
    const four = try parseArgs(&.{ "--rank", "3", "--size", "4", "--coordinator", "10.0.0.1:5", "--device", "a,b,c,-" });
    try std.testing.expectEqual(@as(u32, 3), four.rank);
    try std.testing.expectEqual(@as(u32, 4), four.size);
    try std.testing.expectError(error.InvalidArgs, parseArgs(&.{ "--rank", "4", "--size", "4", "--coordinator", "10.0.0.1:5", "--device", "d" }));
    try std.testing.expectError(error.InvalidArgs, parseArgs(&.{ "--rank", "0", "--size", "1", "--coordinator", "10.0.0.1:5", "--device", "d" }));
    try std.testing.expectError(error.InvalidArgs, parseArgs(&.{ "--rank", "0", "--size", "99", "--coordinator", "10.0.0.1:5", "--device", "d" }));
    // The worker's own --rank wins over the default it adds.
    var out: [12][]const u8 = undefined;
    const w = workerArgs(&.{ "--rank", "2", "--size", "4", "--coordinator", "c:1", "--device", "d" }, &out);
    try std.testing.expectEqual(@as(u32, 2), (try parseArgs(w)).rank);
    const d = workerArgs(&.{ "--coordinator", "c:1", "--device", "d" }, &out);
    try std.testing.expectEqual(@as(u32, 1), (try parseArgs(d)).rank);
}

test "LayerSpan.remap: a later rank renumbers its trunk layers and drops the head-only towers" {
    var buf: [128]u8 = undefined;
    const tail: LayerSpan = .{ .start = 32, .end = 48, .total = 48 };
    try std.testing.expectEqualStrings("model.language_model.layers.0.mlp.gate.weight", tail.remap("model.language_model.layers.32.mlp.gate.weight", &buf).?);
    try std.testing.expectEqualStrings("language_model.model.layers.15.self_attn.q_proj.weight", tail.remap("language_model.model.layers.47.self_attn.q_proj.weight", &buf).?);
    try std.testing.expect(tail.remap("model.language_model.layers.31.mlp.gate.weight", &buf) == null);
    try std.testing.expect(tail.remap("mtp.layers.0.attn_hyper_connection.hc_norm.weight", &buf) == null);
    try std.testing.expect(tail.remap("model.visual.blocks.3.attn.qkv.weight", &buf) == null);
    try std.testing.expectEqualStrings("lm_head.weight", tail.remap("lm_head.weight", &buf).?);

    const head: LayerSpan = .{ .start = 0, .end = 32, .total = 48 };
    try std.testing.expectEqualStrings("model.language_model.layers.31.mlp.gate.weight", head.remap("model.language_model.layers.31.mlp.gate.weight", &buf).?);
    try std.testing.expect(head.remap("model.language_model.layers.32.mlp.gate.weight", &buf) == null);
    try std.testing.expectEqualStrings("mtp.layers.0.attn_hyper_connection.hc_norm.weight", head.remap("mtp.layers.0.attn_hyper_connection.hc_norm.weight", &buf).?);
}

test "chooseSplit: a 2:1 memory split puts two thirds of the layers on rank 0, on a group boundary" {
    try std.testing.expectEqual(@as(u32, 32), chooseSplit(48, 4, 240_000, 120_000));
    try std.testing.expectEqual(@as(u32, 24), chooseSplit(48, 4, 100, 100));
    try std.testing.expectEqual(@as(u32, 44), chooseSplit(48, 4, 1000, 1));
    try std.testing.expectEqual(@as(u32, 4), chooseSplit(48, 4, 1, 1000));
}

test "Shard: gate/up keep a band of rows per expert, down a band of columns" {
    try std.testing.expect(Shard.axis("language_model.model.layers.3.mlp.switch_mlp.gate_proj.weight").? == .rows);
    try std.testing.expect(Shard.axis("language_model.model.layers.3.mlp.shared_expert.down_proj.scales").? == .cols);
    try std.testing.expect(Shard.axis("language_model.model.layers.3.mlp.switch_mlp.gate_proj.expert_scale") == null);
    try std.testing.expect(Shard.axis("language_model.model.layers.3.mlp.shared_expert_gate.weight") == null);
    try std.testing.expect(Shard.axis("mtp.layers.0.mlp.switch_mlp.up_proj.weight") == null);
    // [2 experts, 4 rows, 2 bytes]: rank 1 of 2 keeps rows 2..3 of each expert.
    const buf = [_]u8{ 0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 17 };
    var out: [8]u8 = undefined;
    (Shard{ .rank = 1, .size = 2 }).slice(&out, &buf, 2, 4, 2);
    try std.testing.expectEqualSlices(u8, &.{ 4, 5, 6, 7, 14, 15, 16, 17 }, &out);
}

test "Shard: DeepSeek-V4.1 splits its experts and its attention by heads, not its indexer, compressor, routers or DSpark stages" {
    try std.testing.expect(Shard.axis("layers.3.ffn.experts.w1.weight").? == .rows);
    try std.testing.expect(Shard.axis("layers.3.ffn.experts.w3.scales").? == .rows);
    try std.testing.expect(Shard.axis("layers.3.ffn.experts.w2.weight").? == .cols);
    try std.testing.expect(Shard.axis("layers.3.ffn.shared_experts.w1.weight").? == .rows);
    try std.testing.expect(Shard.axis("layers.3.ffn.shared_experts.w2.scales").? == .cols);
    try std.testing.expect(Shard.axis("language_model.layers.3.attn.wq_b.scales").? == .rows);
    try std.testing.expect(Shard.axis("language_model.layers.3.attn.wo_a.weight").? == .rows);
    try std.testing.expect(Shard.axis("language_model.layers.3.attn.wo_b.weight").? == .cols);
    try std.testing.expect(Shard.axis("language_model.layers.3.attn.attn_sink").? == .vec);
    try std.testing.expect(Shard.axis("layers.3.attn.indexer.wq_b.weight") == null);
    try std.testing.expect(Shard.axis("layers.3.attn.wq_a.weight") == null);
    try std.testing.expect(Shard.axis("layers.3.attn.wkv.weight") == null);
    try std.testing.expect(Shard.axis("layers.3.attn.compressor.wkv.weight") == null);
    try std.testing.expect(Shard.axis("layers.3.ffn.gate.weight") == null);
    try std.testing.expect(Shard.axis("layers.3.ffn.gate.bias") == null);
    try std.testing.expect(Shard.axis("mtp.0.ffn.experts.w1.weight") == null);
    try std.testing.expect(Shard.axis("mtp.0.attn.wq_b.weight") == null);
}

test "clusterJson: valid JSON naming both ranks' layers and the worker" {
    const a = std.testing.allocator;
    var p: Peer = .{ .link = .{ .group = undefined, .rank = 0, .size = 2 }, .mode = .pipeline, .model = "/m/org/Flash-FP4", .split = null, .span = .{ .start = 0, .end = 16, .total = 48 }, .planned = true, .ready = true, .exchanges = 3 };
    p.remotes[1].hello = .{ .avail_mib = 100, .total_mib = 128 };
    const name = "Mac \"two\"";
    @memcpy(p.remotes[1].name_buf[0..name.len], name);
    p.remotes[1].name_len = name.len;
    const j = try clusterJson(a, &p);
    defer a.free(j);
    const doc = try std.fmt.allocPrint(a, "{{{s}}}", .{j[1..]});
    defer a.free(doc);
    const parsed = try std.json.parseFromSlice(std.json.Value, a, doc, .{});
    defer parsed.deinit();
    const c = parsed.value.object.get("cluster").?.object;
    try std.testing.expectEqualStrings("pipeline", c.get("mode").?.string);
    try std.testing.expectEqualStrings("Flash-FP4", c.get("model").?.string);
    try std.testing.expectEqual(@as(i64, 16), c.get("here").?.array.items[1].integer);
    try std.testing.expectEqualStrings(name, c.get("worker_name").?.string);
}

test "findModel pairs the host's copy of a name, not another quantization of it" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    for ([_][]const u8{ "org-a/Flash", "org-b/Flash" }, [_][]const u8{ "4bit", "8bit" }) |dir, q| {
        try tmp.dir.createDirPath(io, dir);
        const cfg = try std.fmt.allocPrint(allocator, "{s}/config.json", .{dir});
        defer allocator.free(cfg);
        try tmp.dir.writeFile(io, .{ .sub_path = cfg, .data = q });
        const w = try std.fmt.allocPrint(allocator, "{s}/model.safetensors", .{dir});
        defer allocator.free(w);
        try tmp.dir.writeFile(io, .{ .sub_path = w, .data = q });
    }
    var cwd_buf: [std.fs.max_path_bytes]u8 = undefined;
    const cwd = std.mem.span(@as([*:0]const u8, @ptrCast(std.c.getcwd(&cwd_buf, cwd_buf.len) orelse return error.NoCwd)));
    const root = try std.fmt.allocPrint(allocator, "{s}/.zig-cache/tmp/{s}", .{ cwd, tmp.sub_path });
    defer allocator.free(root);
    const b = try std.fmt.allocPrint(allocator, "{s}/org-b/Flash", .{root});
    defer allocator.free(b);

    const found = (try findModel(io, allocator, &.{root}, "Flash", try modelIdentity(io, allocator, b))).?;
    defer allocator.free(found);
    try std.testing.expectEqualStrings(b, found);
    try std.testing.expectError(error.DistModelMismatch, findModel(io, allocator, &.{root}, "Flash", 42));
    try std.testing.expect((try findModel(io, allocator, &.{root}, "Other", 42)) == null);
}

test "pairFields names the model a Thunderbolt pair leads, as valid JSON" {
    const a = std.testing.allocator;
    const f = try pairFieldsFor(a, "MiniMax \"H3\"", false);
    defer a.free(f);
    const j = try std.fmt.allocPrint(a, "{{\"os\":\"macos\"{s}}}", .{f});
    defer a.free(j);
    const parsed = try std.json.parseFromSlice(std.json.Value, a, j, .{});
    defer parsed.deinit();
    try std.testing.expectEqualStrings("MiniMax \"H3\"", parsed.value.object.get("leads").?.string);
    try std.testing.expect(!parsed.value.object.get("joined").?.bool);
}

test "a plan made for one copy refuses another folder of the same name" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    for ([_][]const u8{ "a/Flash", "b/Flash" }, [_][]const u8{ "4bit", "8bit" }) |dir, q| {
        try tmp.dir.createDirPath(io, dir);
        const cfg = try std.fmt.allocPrint(allocator, "{s}/config.json", .{dir});
        defer allocator.free(cfg);
        try tmp.dir.writeFile(io, .{ .sub_path = cfg, .data = q });
    }
    var cwd_buf: [std.fs.max_path_bytes]u8 = undefined;
    const cwd = std.mem.span(@as([*:0]const u8, @ptrCast(std.c.getcwd(&cwd_buf, cwd_buf.len) orelse return error.NoCwd)));
    const a = try std.fmt.allocPrint(allocator, "{s}/.zig-cache/tmp/{s}/a/Flash", .{ cwd, tmp.sub_path });
    defer allocator.free(a);
    const b = try std.fmt.allocPrint(allocator, "{s}/.zig-cache/tmp/{s}/b/Flash", .{ cwd, tmp.sub_path });
    defer allocator.free(b);

    cluster().group = .{ .io = io, .allocator = allocator, .mode = .pipeline, .model = a, .split = null };
    defer cluster().group = null;
    var p: Peer = undefined;
    p.plan_id = try modelIdentity(io, allocator, a);
    try p.samePlannedCopy(a);
    try std.testing.expectError(error.DistModelMismatch, p.samePlannedCopy(b));
}

test "the pair holds its own folder, not another one with the same name" {
    if (comptime builtin.os.tag != .macos) return error.SkipZigTest; // a cluster exists only on macOS
    const io = std.testing.io;
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.createDirPath(io, "a/Flash");
    try tmp.dir.createDirPath(io, "b/Flash");
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const root = buf[0..try tmp.dir.realPath(io, &buf)];
    const a = try std.fmt.allocPrint(allocator, "{s}/a/Flash", .{root});
    defer allocator.free(a);
    const a_spelled = try std.fmt.allocPrint(allocator, "{s}/b/../a/Flash/", .{root});
    defer allocator.free(a_spelled);
    const b = try std.fmt.allocPrint(allocator, "{s}/b/Flash", .{root});
    defer allocator.free(b);

    cluster().group = .{ .io = io, .allocator = allocator, .mode = .pipeline, .model = a, .split = null };
    defer cluster().group = null;
    cluster().joined.store(true, .release);
    defer cluster().joined.store(false, .release);
    try std.testing.expect(holds(a));
    try std.testing.expect(holds(a_spelled));
    try std.testing.expect(!holds(b));
    // The other folder loads, and is billed, whole.
    var c: model_mod.ModelConfig = .{ .model_type = "qwen4_exp", .legacy = .{ .caps = .{ .long_ctx = true } }, .num_hidden_layers = 8 };
    cluster().free_mib[1].store(40, .release);
    defer cluster().free_mib[1].store(0, .release);
    try std.testing.expectEqual(@as(u64, 800), hostShare(b, &c, 800));
}

test "parseCoordinator reads the host's address and port" {
    const c = try parseCoordinator("10.254.77.1:32323");
    try std.testing.expectEqual([4]u8{ 10, 254, 77, 1 }, c.ip4);
    try std.testing.expectEqual(@as(u16, 32323), c.port);
    try std.testing.expectError(error.DistBadCoordinator, parseCoordinator("10.254.77:32323"));
    try std.testing.expectError(error.DistBadCoordinator, parseCoordinator("10.254.77.1"));
}

test "the bell wakes rank 1 once per ring, says which ring, and reports a host that left" {
    if (comptime builtin.os.tag != .macos) return error.SkipZigTest;
    const coord: Coordinator = .{ .ip4 = .{ 127, 0, 0, 1 }, .port = 39411 };
    const listener = Bell.listen(coord, 1);
    try std.testing.expect(listener >= 0);
    defer _ = std.c.close(listener);
    const worker = Bell.connect(coord, 1, null, testDeadline(3000));
    try std.testing.expect(worker.open());
    const got = Bell.acceptRank(listener, 10_000, null, testDeadline(2000)).?;
    try std.testing.expectEqual(@as(u32, 1), got.rank);
    const host = got.bell;
    try host.ring();
    try host.pipe();
    try host.ring();
    try std.testing.expectEqual(Bell.msg_byte, try worker.wait());
    try std.testing.expectEqual(Bell.pipe_byte, try worker.wait());
    try std.testing.expectEqual(Bell.msg_byte, try worker.wait());
    _ = std.c.close(host.fd);
    try std.testing.expectError(error.DistPeerGone, worker.wait());
    _ = std.c.close(worker.fd);
}

test "the worker answers whether it has the host's copy of the model when it joins" {
    if (comptime builtin.os.tag != .macos) return error.SkipZigTest;
    const io = std.testing.io;
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    for ([_][]const u8{ "host/Flash", "worker/Flash", "other/Flash" }, [_][]const u8{ "4bit", "4bit", "8bit" }) |dir, q| {
        try tmp.dir.createDirPath(io, dir);
        const cfg = try std.fmt.allocPrint(allocator, "{s}/config.json", .{dir});
        defer allocator.free(cfg);
        try tmp.dir.writeFile(io, .{ .sub_path = cfg, .data = q });
    }
    var cwd_buf: [std.fs.max_path_bytes]u8 = undefined;
    const cwd = std.mem.span(@as([*:0]const u8, @ptrCast(std.c.getcwd(&cwd_buf, cwd_buf.len) orelse return error.NoCwd)));
    const base = try std.fmt.allocPrint(allocator, "{s}/.zig-cache/tmp/{s}", .{ cwd, tmp.sub_path });
    defer allocator.free(base);
    const host_model = try std.fmt.allocPrint(allocator, "{s}/host/Flash", .{base});
    defer allocator.free(host_model);
    cluster().group = .{ .io = io, .allocator = allocator, .mode = .pipeline, .model = host_model, .split = null };
    defer cluster().group = null;
    defer cluster().copies[1].store(-1, .release);

    const Case = struct { root: []const u8, want: i32 };
    for ([_]Case{ .{ .root = "worker", .want = 0 }, .{ .root = "other", .want = MISMATCH }, .{ .root = "nowhere", .want = NOT_FOUND } }) |c| {
        const coord: Coordinator = .{ .ip4 = .{ 127, 0, 0, 1 }, .port = 39421 };
        const listener = Bell.listen(coord, 1);
        try std.testing.expect(listener >= 0);
        defer _ = std.c.close(listener);
        var worker: Link = .{ .group = undefined, .rank = 1, .size = 2 };
        worker.bells[0] = Bell.connect(coord, 1, null, testDeadline(3000));
        defer _ = std.c.close(worker.bells[0].fd);
        var host: Link = .{ .group = undefined, .rank = 0, .size = 2 };
        host.bells[1] = Bell.acceptRank(listener, 10_000, null, testDeadline(2000)).?.bell;
        defer _ = std.c.close(host.bells[1].fd);
        const asker = try std.Thread.spawn(.{}, askForModel, .{ &host, 1 });
        const root = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ base, c.root });
        defer allocator.free(root);
        const has = try answerModel(io, allocator, &worker, &.{root});
        asker.join();
        try std.testing.expectEqual(c.want == 0, has);
        try std.testing.expectEqual(c.want, cluster().copies[1].load(.acquire));
    }
}

test "a paired model is never picked to unload while the worker is joined" {
    if (comptime builtin.os.tag != .macos) return error.SkipZigTest; // a cluster exists only on macOS
    const reg_mod = @import("../../../host/registry/registry.zig");
    var reg = try reg_mod.ModelRegistry.init(std.testing.allocator, std.Io.Threaded.global_single_threaded.io(), null, 3, 0, null);
    defer reg.deinit();
    const flash = try reg.registerStub("Flash", "/m/Flash", 100);
    const other = try reg.registerStub("other", "other", 100);
    reg.mutex.lockUncancelable(reg.io);
    reg.markReadyLocked(flash, 100);
    reg.markReadyLocked(other, 100);
    reg.mutex.unlock(reg.io);
    _ = try reg.ensureLoaded("other");
    reg.release(other); // `other` is now the newest; Flash the oldest

    cluster().group = .{ .io = std.testing.io, .allocator = std.testing.allocator, .mode = .pipeline, .model = "/m/Flash", .split = null };
    defer cluster().group = null;
    cluster().joined.store(true, .release);
    defer cluster().joined.store(false, .release);

    reg.mutex.lockUncancelable(reg.io);
    defer reg.mutex.unlock(reg.io);
    try std.testing.expectEqualStrings("other", reg.pickLruEvictable("", false).?.id);
    try std.testing.expect(!reg_mod.ModelRegistry.idleEvictable(flash, std.math.maxInt(i64), 0));
    cluster().joined.store(false, .release);
    try std.testing.expectEqualStrings("Flash", reg.pickLruEvictable("", false).?.id);
}

test "the leader bills only the layers it keeps of a model its pair splits" {
    var c: model_mod.ModelConfig = .{ .model_type = "deepseek_v41", .num_hidden_layers = 8 };
    c.dsv4_compress_ratios[0..8].* = .{ 0, 0, 2, 2, 2, 1, 1, 1 };
    c.dsv41_kv_sources = (1 << 2) | (1 << 5);
    c.dsv41_index_sources = (1 << 2) | (1 << 5) | (1 << 7);
    c.dsv41_candidate_source = 5;
    c.dsv41_engram_layers[0..2].* = .{ 1, 2 };
    c.dsv41_n_engram_layers = 2;
    const g: Group = .{ .io = std.testing.io, .allocator = std.testing.allocator, .mode = .pipeline, .model = "/m/V41", .split = null };
    // The worker's 40% holds layers 5..8: this Mac keeps 5 of 8.
    try std.testing.expectEqual(@as(u64, 500), shareFor(g, &c, 800, 60, 40));
    // A worker that has not said how much it has: the whole model.
    try std.testing.expectEqual(@as(u64, 800), shareFor(g, &c, 800, 60, 0));
}

test "dist N-rank: rank 0 files each worker's bell under the rank it names, and a restarted worker's replaces the old one" {
    if (comptime builtin.os.tag != .macos) return error.SkipZigTest;
    const coord: Coordinator = .{ .ip4 = .{ 127, 0, 0, 1 }, .port = 39431 };
    var rv: Rendezvous = .{ .listener = Bell.listen(coord, 3), .size = 4 };
    try std.testing.expect(rv.listener >= 0);
    defer rv.close();
    const t = try std.Thread.spawn(.{}, Rendezvous.run, .{&rv});
    const w3 = Bell.connect(coord, 3, null, testDeadline(3000));
    const w1 = Bell.connect(coord, 1, null, testDeadline(3000));
    try std.testing.expect(w3.open() and w1.open());
    defer _ = std.c.close(w3.fd);
    try std.testing.expect(rv.waitAll(3000) == false);
    var buf: [32]u8 = undefined;
    try std.testing.expectEqualStrings("2", missingRanks(rv.seen(), 4, &buf));
    // Rank 1 restarts: its new connection wins and the old one is closed.
    const w1b = Bell.connect(coord, 1, null, testDeadline(3000));
    defer _ = std.c.close(w1b.fd);
    _ = std.c.close(w1.fd);
    const w2 = Bell.connect(coord, 2, null, testDeadline(3000));
    defer _ = std.c.close(w2.fd);
    try std.testing.expect(rv.waitAll(3000));
    rv.stop.store(true, .release);
    t.join();
    try std.testing.expectEqualStrings("", missingRanks(rv.seen(), 4, &buf));
    var link: Link = .{ .group = undefined, .rank = 0, .size = 4 };
    rv.handOver(&link);
    for (1..4) |k| try std.testing.expect(link.bells[k].open());
    defer for (1..4) |k| {
        _ = std.c.close(link.bells[k].fd);
    };
    try link.bells[1].ring();
    try std.testing.expectEqual(Bell.msg_byte, try w1b.wait());
    try link.bells[3].pipe();
    try std.testing.expectEqual(Bell.pipe_byte, try w3.wait());
    try std.testing.expectEqual(@as(u32, 0b1110), link.bellMask());
}

test "dist N-rank: the watcher names the worker whose bell closed, after reading the others' step times" {
    if (comptime builtin.os.tag != .macos) return error.SkipZigTest;
    var pairs: [4][2]std.c.fd_t = undefined;
    var link: Link = .{ .group = undefined, .rank = 0, .size = 4 };
    for (1..4) |k| {
        try std.testing.expectEqual(@as(c_int, 0), std.c.socketpair(std.posix.AF.UNIX, std.posix.SOCK.STREAM, 0, &pairs[k]));
        link.bells[k] = .{ .fd = pairs[k][0] };
    }
    defer for (1..4) |k| {
        _ = std.c.close(pairs[k][0]);
        if (k != 2) _ = std.c.close(pairs[k][1]);
    };
    var took: [4]u8 = undefined;
    std.mem.writeInt(u32, &took, 1234, .little);
    try (Bell{ .fd = pairs[1][1] }).writeAll(&took);
    var state: LinkState = .{};
    // Rank 1's time is read first; then rank 2 leaves.
    const t = try std.Thread.spawn(.{}, struct {
        fn leave(fd: std.c.fd_t) void {
            crt.sleepMs(200);
            _ = std.c.close(fd);
        }
    }.leave, .{pairs[2][1]});
    defer t.join();
    try std.testing.expectEqual(@as(u32, 2), watchBells(&link, &state));
    try std.testing.expectEqual(@as(u32, 1234), state.worker_us.load(.acquire));
}

test "dist N-rank: a load before every worker joined fails at once, as forming until the join timeout and late after it" {
    try std.testing.expectEqual(JoinState.joined, joinState(true, 0, 600));
    try std.testing.expectEqual(JoinState.forming, joinState(false, 599_999, 600));
    try std.testing.expectEqual(JoinState.late, joinState(false, 600_000, 600));
    // A pair's timeout is 0: late at once, today's refusal.
    try std.testing.expectEqual(JoinState.late, joinState(false, 0, 0));
}

test "dist N-rank: the join names every rank that has not joined, and refuses a rank without the model by its number" {
    var buf: [64]u8 = undefined;
    try std.testing.expectEqualStrings("2, 3", missingRanks(0b0010, 4, &buf));
    try std.testing.expectEqualStrings("1, 2, 3", missingRanks(0, 4, &buf));
    try std.testing.expectEqualStrings("1", missingRanks(0, 2, &buf));
    try std.testing.expectEqualStrings("", missingRanks(0b1110, 4, &buf));
    try std.testing.expect(firstRefusal(&.{ 0, 0, 0, 0 }, 4) == null);
    const r = firstRefusal(&.{ 0, 0, MISMATCH, NOT_FOUND }, 4).?;
    try std.testing.expectEqual(@as(u32, 2), r.rank);
    try std.testing.expectEqual(MISMATCH, r.code);
    // A rank that was never asked (no bell) is no refusal.
    try std.testing.expect(firstRefusal(&.{ 0, -1, 0, 0 }, 4) == null);
}

test "dist N-rank: /props lists every worker of a four-Mac group" {
    const a = std.testing.allocator;
    var p: Peer = .{ .link = .{ .group = undefined, .rank = 0, .size = 4 }, .mode = .tensor, .model = "/m/V41", .split = null, .span = .{ .start = 0, .end = 43, .total = 43 }, .planned = true, .ready = true };
    for (1..4) |k| {
        p.remotes[k].hello = .{ .avail_mib = @intCast(100 * k), .total_mib = 256 };
        const name = try std.fmt.bufPrint(&p.remotes[k].name_buf, "studio{d}", .{k + 1});
        p.remotes[k].name_len = name.len;
        p.remotes[k].resident_mib = @intCast(10 * k);
    }
    const j = try clusterJson(a, &p);
    defer a.free(j);
    const doc = try std.fmt.allocPrint(a, "{{{s}}}", .{j[1..]});
    defer a.free(doc);
    const parsed = try std.json.parseFromSlice(std.json.Value, a, doc, .{});
    defer parsed.deinit();
    const c = parsed.value.object.get("cluster").?.object;
    try std.testing.expectEqual(@as(i64, 4), c.get("size").?.integer);
    // The pair's fields stay rank 1's.
    try std.testing.expectEqualStrings("studio2", c.get("worker_name").?.string);
    const ws = c.get("workers").?.array.items;
    try std.testing.expectEqual(@as(usize, 3), ws.len);
    try std.testing.expectEqual(@as(i64, 3), ws[2].object.get("rank").?.integer);
    try std.testing.expectEqualStrings("studio4", ws[2].object.get("name").?.string);
    try std.testing.expectEqual(@as(i64, 300), ws[2].object.get("free_mib").?.integer);
    try std.testing.expectEqual(@as(i64, 30), ws[2].object.get("resident_mib").?.integer);
}

test "dist N-rank: each of four Macs bills a quarter of a tensor split, plus the same tenth" {
    var c: model_mod.ModelConfig = .{ .model_type = "deepseek_v41", .num_hidden_layers = 8 };
    const g: Group = .{ .io = std.testing.io, .allocator = std.testing.allocator, .mode = .tensor, .model = "/m/V41", .split = null, .size = 4 };
    try std.testing.expectEqual(@as(u64, 200 + 80), shareFor(g, &c, 800, 60, 40));
    var pair = g;
    pair.size = 2;
    try std.testing.expectEqual(@as(u64, 400 + 80), shareFor(pair, &c, 800, 60, 40));
    var q: model_mod.ModelConfig = .{ .model_type = "qwen4_exp", .legacy = .{ .caps = .{ .long_ctx = true } }, .num_hidden_layers = 8 };
    try std.testing.expectEqual(@as(u64, 400 + 80), shareFor(pair, &q, 800, 60, 40));
}

test "dist N-rank: past a pair only DeepSeek-V4.1's tensor split forms a group, and only one its heads divide" {
    var c: model_mod.ModelConfig = .{ .model_type = "deepseek_v41", .num_hidden_layers = 8, .num_attention_heads = 64, .dsv4_o_groups = 8, .moe_intermediate_size = 2304 };
    try groupFits(.tensor, &c, 2);
    try groupFits(.tensor, &c, 4);
    try groupFits(.pipeline, &c, 2);
    try std.testing.expectError(error.DistShardUneven, groupFits(.tensor, &c, 3));
    try std.testing.expectError(error.DistNeedsPair, groupFits(.pipeline, &c, 4));
    try std.testing.expectError(error.DistNeedsPair, groupFits(.rows, &c, 4));
    var q: model_mod.ModelConfig = .{ .model_type = "qwen4_exp", .legacy = .{ .caps = .{ .long_ctx = true } }, .num_hidden_layers = 8, .num_experts = 4, .moe_intermediate_size = 512 };
    try groupFits(.tensor, &q, 2);
    try std.testing.expectError(error.DistNeedsPair, groupFits(.tensor, &q, 4));
}

test "after the stop message rank 0 sends the worker nothing more" {
    if (comptime builtin.os.tag != .macos) return error.SkipZigTest; // a cluster exists only on macOS
    // A read-only bell fails to ring, so no message reaches the (absent) RDMA link.
    const fd = std.c.open("/dev/null", .{ .ACCMODE = .RDONLY });
    defer _ = std.c.close(fd);
    cluster().peer_slot = .{ .link = .{ .group = undefined, .rank = 0, .size = 2 }, .mode = .pipeline, .model = "/m/H3", .split = null, .ready = true };
    cluster().peer_slot.?.link.bells[1] = .{ .fd = fd };
    defer cluster().peer_slot = null;
    shutdown();
    try std.testing.expect(!peer().?.ready); // the model's unload at teardown must not message a stopped worker
}
const switcher = @import("switch.zig");

const TestDistExit = struct {
    count: std.atomic.Value(u32) = .init(0),
    causes: std.atomic.Value(u32) = .init(0),
};

const TestSwitchOpen = struct {
    const Request = struct { n: u64, role: switcher.Role, deadline_epoch_ms: i64, cfg: Config };
    var entered: std.atomic.Value(bool) = .init(false);
    var finish: std.atomic.Value(bool) = .init(false);
    var returned: std.atomic.Value(bool) = .init(false);
    var peer_bell: Bell = .{};
    var extra_peer: Bell = .{};
    var failure: ?anyerror = null;
    var block = false;
    var ignore_cancel = false;
    var exits: TestDistExit = .{};
    var post_go = false;

    fn setup() void {
        bindInferenceThread();
        entered.store(false, .release);
        finish.store(false, .release);
        returned.store(false, .release);
        exits = .{};
        post_go = false;
        failure = null;
        block = false;
        ignore_cancel = false;
        switcher.testing.open = open;
    }

    fn cleanup() void {
        finish.store(true, .release);
        switcher.testing.reset();
        if (peer_bell.open()) _ = std.c.close(peer_bell.fd);
        peer_bell = .{};
        if (extra_peer.open()) _ = std.c.close(extra_peer.fd);
        extra_peer = .{};
    }

    fn open(_: std.Io, cfg: Config) anyerror!Link {
        var link = try newLink(cfg);
        errdefer link.deinit();
        errdefer link.captureFailureRank();
        link.state.?.test_exits = &exits;
        const pair = try TestBells.open(null);
        link.bells[if (cfg.rank == 0) @as(usize, 1) else 0] = if (cfg.rank == 0) pair.leader.bells[1] else pair.worker;
        peer_bell = if (cfg.rank == 0) pair.worker else pair.leader.bells[1];
        if (cfg.rank == 0 and cfg.size == 3) {
            const extra = try TestBells.open(null);
            link.bells[2] = extra.leader.bells[1];
            extra_peer = extra.worker;
            try cfg.open_control.?.publish(link.bells[2].fd);
        }
        link.attachBells();
        try cfg.open_control.?.publish(link.bells[if (cfg.rank == 0) @as(usize, 1) else 0].fd);
        link.state.?.go_committed = post_go;
        if (post_go and cfg.rank != 0) try startWatchers(&link);
        entered.store(true, .release);
        if (block) while (!finish.load(.acquire)) {
            if (!ignore_cancel and cfg.open_control.?.cancelled.load(.acquire)) {
                var byte: [1]u8 = undefined;
                try link.bells[if (cfg.rank == 0) @as(usize, 1) else 0].readAll(&byte);
                return error.TestExpectedShutdown;
            }
            crt.sleepMs(1);
        };
        if (failure) |err| {
            if (err == error.DistNoReady) try leaderAwaitReadyThenGo(&link.bells, link.size, cfg.formation_deadline_ms);
            return err;
        }
        if (!ignore_cancel) {
            const PeerReady = struct {
                fn run(bell: Bell, rank: u32, deadline: u64) void {
                    if (rank == 0) {
                        workerReadyAwaitGo(bell, deadline, .{}) catch {};
                        return;
                    }
                    var bells: [max_ranks]Bell = @splat(.{});
                    bells[1] = bell;
                    leaderAwaitReadyThenGo(&bells, 2, deadline) catch {};
                }
            };
            const t = try std.Thread.spawn(.{}, PeerReady.run, .{ peer_bell, cfg.rank, cfg.formation_deadline_ms });
            defer t.join();
            const extra = if (extra_peer.open()) try std.Thread.spawn(.{}, PeerReady.run, .{ extra_peer, cfg.rank, cfg.formation_deadline_ms }) else null;
            defer if (extra) |thread| thread.join();
            try link.finishOpen(cfg, testDeadline(0), cfg.formation_deadline_ms);
        } else startWatchdog(&link);
        returned.store(true, .release);
        return link;
    }

    fn openUnranked(_: std.Io, cfg: Config) anyerror!Link {
        var link = try newLink(cfg);
        errdefer link.deinit();
        errdefer link.captureFailureRank();
        const coord: Coordinator = .{ .ip4 = .{ 127, 0, 0, 1 }, .port = 46211 };
        const listener = Bell.listen(coord, 1);
        if (listener == Bell.none) return error.TestListenFailed;
        defer closeBell(cfg.open_control, listener);
        try cfg.open_control.?.publish(listener);
        peer_bell = .{ .fd = try lan.connectTimeout(coord.ip4, coord.port + 1, 500) };
        entered.store(true, .release);
        const got = Bell.acceptRank(listener, 100, cfg.open_control, cfg.formation_deadline_ms) orelse return error.DistAborted;
        closeBell(cfg.open_control, got.bell.fd);
        return error.TestExpectedShutdown;
    }

    fn start(req: Request) !void {
        try primeEnv(req.cfg);
        try switcher.start(.{ .cfg = req.cfg, .model_dir = "/m/switch" }, req.role, req.n, req.deadline_epoch_ms);
    }

    fn request(n: u64) Request {
        return .{ .n = n, .role = .leader, .deadline_epoch_ms = crt.wallMs() + 2000, .cfg = .{ .rank = 0, .size = 2, .coordinator = "127.0.0.1:46201", .device = "loopback" } };
    }

    fn awaitEntered() !void {
        const end = testDeadline(2000);
        while (!entered.load(.acquire)) {
            if (msLeft(end) == 0) return error.TestOpenNotEntered;
            crt.sleepMs(1);
        }
    }

    fn outcome(n: u64) !switcher.FormOutcome {
        const end = testDeadline(4000);
        while (msLeft(end) > 0) {
            if (switcher.poll(n)) |o| return o;
            crt.sleepMs(1);
        }
        return error.TestSwitchTimeout;
    }
};

test "dist switch matrix 1: READY and GO publish one borrowed heap Link" {
    TestSwitchOpen.setup();
    defer TestSwitchOpen.cleanup();
    try TestSwitchOpen.start(TestSwitchOpen.request(1));
    const first = (try TestSwitchOpen.outcome(1)).formed.link;
    try std.testing.expectEqual(first, switcher.poll(1).?.formed.link);
    try std.testing.expect(first.state.?.go_committed);
    try std.testing.expect(switcher.poll(0) == null);
}

test "dist switch matrix 2: forming and formed attempts refuse another start" {
    TestSwitchOpen.setup();
    defer TestSwitchOpen.cleanup();
    TestSwitchOpen.block = true;
    try TestSwitchOpen.start(TestSwitchOpen.request(2));
    try TestSwitchOpen.awaitEntered();
    try std.testing.expectError(error.Busy, TestSwitchOpen.start(TestSwitchOpen.request(3)));
    TestSwitchOpen.finish.store(true, .release);
    _ = (try TestSwitchOpen.outcome(2)).formed.link;
    try std.testing.expectError(error.Busy, TestSwitchOpen.start(TestSwitchOpen.request(3)));
}

test "dist switch matrix 3: bad config and expired deadline spawn nothing" {
    TestSwitchOpen.setup();
    defer TestSwitchOpen.cleanup();
    var req = TestSwitchOpen.request(3);
    req.deadline_epoch_ms = crt.wallMs() - 1;
    try std.testing.expectError(error.InvalidArgs, TestSwitchOpen.start(req));
    req = TestSwitchOpen.request(3);
    req.cfg.size = 1;
    try std.testing.expectError(error.InvalidArgs, TestSwitchOpen.start(req));
    req.cfg.size = 2;
    req.cfg.rank = 2;
    try std.testing.expectError(error.InvalidArgs, TestSwitchOpen.start(req));
    try std.testing.expect(!TestSwitchOpen.entered.load(.acquire));
    try std.testing.expect(switcher.poll(3) == null);
}

test "dist switch matrix 4: cancel shuts down a forming bell before publication" {
    TestSwitchOpen.setup();
    defer TestSwitchOpen.cleanup();
    TestSwitchOpen.block = true;
    try TestSwitchOpen.start(TestSwitchOpen.request(4));
    try TestSwitchOpen.awaitEntered();
    try std.testing.expectEqual(.unknown, switcher.cancel(99));
    try std.testing.expectEqual(.cancelled, switcher.cancel(4));
    try std.testing.expectEqual(.cancelled, try TestSwitchOpen.outcome(4));
    try std.testing.expect(!TestSwitchOpen.returned.load(.acquire));
    try std.testing.expectEqual(@as(u32, 0), TestSwitchOpen.exits.count.load(.acquire));

    TestSwitchOpen.cleanup();
    TestSwitchOpen.setup();
    switcher.testing.open = TestSwitchOpen.openUnranked;
    try TestSwitchOpen.start(TestSwitchOpen.request(4));
    try TestSwitchOpen.awaitEntered();
    crt.sleepMs(20);
    const began = crt.monotonicMs();
    try std.testing.expectEqual(.cancelled, switcher.cancel(4));
    try std.testing.expectEqual(.cancelled, try TestSwitchOpen.outcome(4));
    try std.testing.expect(crt.monotonicMs() - began < 500);

    const coord: Coordinator = .{ .ip4 = .{ 127, 0, 0, 1 }, .port = 46213 };
    const listener = Bell.listen(coord, 1);
    try std.testing.expect(listener != Bell.none);
    defer _ = std.c.close(listener);
    const fd = std.c.socket(std.posix.AF.INET, std.posix.SOCK.STREAM, 0);
    try std.testing.expect(fd >= 0);
    var control: OpenControl = .{ .n = 4 };
    defer closeBell(&control, fd);
    const flags = std.c.fcntl(fd, std.c.F.GETFL);
    const nonblock: c_int = @bitCast(std.c.O{ .NONBLOCK = true });
    try std.testing.expectEqual(@as(c_int, 0), std.c.fcntl(fd, std.c.F.SETFL, flags | nonblock));
    try control.publish(fd);
    control.cancel();
    try std.testing.expectError(error.DistAborted, Bell.beginConnect(fd, coord, &control));
}

test "dist switch matrix 5: cancel after publication leaves the formed Link live" {
    TestSwitchOpen.setup();
    defer TestSwitchOpen.cleanup();
    try TestSwitchOpen.start(TestSwitchOpen.request(5));
    const link = (try TestSwitchOpen.outcome(5)).formed.link;
    try std.testing.expectEqual(.already_formed, switcher.cancel(5));
    try std.testing.expectEqual(link, switcher.poll(5).?.formed.link);
    try std.testing.expect(!link.state.?.closing.load(.acquire));
}

test "dist switch matrix 6: cancel and formed CAS have exactly one winner" {
    for ([_]bool{ false, true }) |cancel_first| {
        TestSwitchOpen.setup();
        defer TestSwitchOpen.cleanup();
        TestSwitchOpen.block = true;
        TestSwitchOpen.ignore_cancel = true;
        try TestSwitchOpen.start(TestSwitchOpen.request(6));
        try TestSwitchOpen.awaitEntered();
        if (cancel_first) try std.testing.expectEqual(.cancelled, switcher.cancel(6));
        TestSwitchOpen.finish.store(true, .release);
        if (cancel_first) {
            try std.testing.expectEqual(.cancelled, try TestSwitchOpen.outcome(6));
            try std.testing.expectEqual(@as(u32, 0), TestSwitchOpen.exits.count.load(.acquire));
        } else {
            _ = (try TestSwitchOpen.outcome(6)).formed.link;
            try std.testing.expectEqual(.already_formed, switcher.cancel(6));
        }
    }

    for ([_]bool{ false, true }) |follower| {
        TestSwitchOpen.setup();
        defer TestSwitchOpen.cleanup();
        TestSwitchOpen.block = true;
        TestSwitchOpen.ignore_cancel = true;
        TestSwitchOpen.post_go = true;
        var req = TestSwitchOpen.request(6);
        if (follower) {
            req.role = .follower;
            req.cfg.rank = 1;
        }
        try TestSwitchOpen.start(req);
        try TestSwitchOpen.awaitEntered();
        try std.testing.expectEqual(.cancelled, switcher.cancel(6));
        if (follower) {
            const end = testDeadline(1000);
            while (TestSwitchOpen.exits.count.load(.acquire) == 0 and msLeft(end) > 0) crt.sleepMs(1);
        }
        TestSwitchOpen.finish.store(true, .release);
        try std.testing.expectEqual(.cancelled, try TestSwitchOpen.outcome(6));
        const cause: dist_watchdog.ExitCause = if (follower) .cancelled_after_go else .partial_go;
        try std.testing.expectEqual(@as(u32, 1) << @backingInt(cause), TestSwitchOpen.exits.causes.load(.acquire));
    }
}

test "dist switch matrix 7: open errors reach the caller by name" {
    for ([_]anyerror{ error.DistNoReady, error.DistAborted, error.RdmaInitFailed }) |err| {
        TestSwitchOpen.setup();
        defer TestSwitchOpen.cleanup();
        TestSwitchOpen.failure = err;
        try TestSwitchOpen.start(TestSwitchOpen.request(7));
        try std.testing.expectEqualStrings(@errorName(err), (try TestSwitchOpen.outcome(7)).failed.err);
        try std.testing.expectEqualStrings(@errorName(err), switcher.poll(7).?.failed.err);
    }
}

test "dist switch matrix 8: a late thread quarantines without joining or reuse" {
    TestSwitchOpen.setup();
    defer TestSwitchOpen.cleanup();
    TestSwitchOpen.block = true;
    TestSwitchOpen.ignore_cancel = true;
    var req = TestSwitchOpen.request(8);
    req.deadline_epoch_ms = crt.wallMs() + 100;
    try TestSwitchOpen.start(req);
    try TestSwitchOpen.awaitEntered();
    try std.testing.expectEqual(.quarantined, try TestSwitchOpen.outcome(8));
    try std.testing.expect(switcher.quarantined());
    try std.testing.expect(!TestSwitchOpen.returned.load(.acquire));
    try std.testing.expectError(error.Busy, TestSwitchOpen.start(TestSwitchOpen.request(9)));
    TestSwitchOpen.finish.store(true, .release);
    crt.sleepMs(100);
    try std.testing.expectEqual(.quarantined, switcher.poll(8).?);

    TestSwitchOpen.cleanup();
    TestSwitchOpen.setup();
    TestSwitchOpen.block = true;
    TestSwitchOpen.ignore_cancel = true;
    req = TestSwitchOpen.request(8);
    req.deadline_epoch_ms = crt.wallMs() + 100;
    try TestSwitchOpen.start(req);
    try TestSwitchOpen.awaitEntered();
    crt.sleepMs(1200);
    TestSwitchOpen.finish.store(true, .release);
    const end = testDeadline(1000);
    while (!TestSwitchOpen.returned.load(.acquire) and msLeft(end) > 0) crt.sleepMs(1);
    crt.sleepMs(50);
    try std.testing.expectEqual(.quarantined, switcher.poll(8).?);
}

test "dist switch matrix 9: teardown joins live watchers before closing fds" {
    var pair = try TestBells.open(46203);
    defer pair.close();
    var link = try newLink(.{ .rank = 0, .size = 2, .coordinator = "127.0.0.1:46203", .device = "loopback" });
    var exits: TestDistExit = .{};
    link.state.?.test_exits = &exits;
    link.bells[1] = pair.leader.bells[1];
    pair.leader.bells[1] = .{};
    link.attachBells();
    startWatchdog(&link);
    try startWatchers(&link);
    const fd = link.bells[1].fd;
    link.deinit();
    try std.testing.expectEqual(@as(u32, 0), exits.count.load(.acquire));
    try std.testing.expect(link.state == null);
    try std.testing.expectEqual(@as(c_int, -1), std.c.fcntl(fd, std.c.F.GETFD));
}

test "dist switch matrix 10: re-formed Link starts fresh watchdog and heartbeat" {
    const stall = env.override(.MLX_SERVE_DIST_STALL_MS, "40");
    defer stall.restore();
    for (0..2) |cycle| {
        var pair = try TestBells.open(46205);
        defer pair.close();
        var link = try newLink(.{ .rank = 0, .size = 2, .coordinator = "127.0.0.1:46205", .device = "loopback" });
        defer link.deinit();
        link.bells[1] = pair.leader.bells[1];
        pair.leader.bells[1] = .{};
        link.attachBells();
        const state = link.state.?;
        var exits: TestDistExit = .{};
        state.test_exits = &exits;
        try std.testing.expect(!state.worker_gone.load(.acquire));
        try std.testing.expectEqual(@as(u32, 0), state.lost_rank.load(.acquire));
        try std.testing.expect(!state.follower_ready.load(.acquire));
        try std.testing.expectEqual(@as(u64, 0), state.loop_tick_ms.load(.acquire));
        startWatchdog(&link);
        activateHeartbeat(state);
        var fds = [_]std.posix.pollfd{.{ .fd = pair.worker.fd, .events = std.posix.POLL.IN, .revents = 0 }};
        try std.testing.expect((try std.posix.poll(&fds, 2000)) > 0);
        var byte: [1]u8 = undefined;
        try pair.worker.readAll(&byte);
        try std.testing.expectEqual(Bell.heartbeat_byte, byte[0]);
        try std.testing.expect(state.watchdog_threads[0] != null);
        state.watchdog.begin(@as(u64, @intCast(crt.monotonicMs())) -| 100);
        const end = testDeadline(1000);
        while (exits.count.load(.acquire) == 0 and msLeft(end) > 0) crt.sleepMs(1);
        try std.testing.expectEqual(@as(u32, 1) << @backingInt(dist_watchdog.ExitCause.stall), exits.causes.load(.acquire));
        if (cycle == 0) {
            state.worker_gone.store(true, .release);
            state.lost_rank.store(1, .release);
            state.follower_ready.store(true, .release);
            state.loop_tick_ms.store(1, .release);
        }
    }
}

test "dist switch matrix 11: a switch follower watches leader EOF" {
    TestSwitchOpen.setup();
    defer TestSwitchOpen.cleanup();
    var req = TestSwitchOpen.request(11);
    req.role = .follower;
    req.cfg.rank = 1;
    try TestSwitchOpen.start(req);
    const link = (try TestSwitchOpen.outcome(11)).formed.link;
    try std.testing.expect(link.state.?.follower_pending.load(.acquire));
    try std.testing.expect(!link.state.?.follower_ready.load(.acquire));
    _ = std.c.shutdown(TestSwitchOpen.peer_bell.fd, std.c.SHUT.RDWR);
    const end = testDeadline(1000);
    while (TestSwitchOpen.exits.count.load(.acquire) == 0 and msLeft(end) > 0) crt.sleepMs(1);
    try std.testing.expectEqual(@as(u32, 1), TestSwitchOpen.exits.count.load(.acquire));
    try std.testing.expectEqual(@as(u32, 1) << @backingInt(dist_watchdog.ExitCause.bell_eof), TestSwitchOpen.exits.causes.load(.acquire));
}

test "dist switch matrix 12: identical environment values cause no setenv" {
    const names = [_][*:0]const u8{ "MLX_RANK", "MLX_JACCL_COORDINATOR", "MLX_IBV_DEVICES" };
    var saved: [3]?[:0]u8 = @splat(null);
    for (names, &saved) |name, *slot| {
        if (std.c.getenv(name)) |old| {
            const value = std.mem.span(old);
            const saved_value = try std.testing.allocator.allocSentinel(u8, value.len, 0);
            @memcpy(saved_value, value);
            slot.* = saved_value;
        }
    }
    defer for (names, saved) |name, old| {
        if (old) |value| {
            _ = env.setForeign(name, value);
            std.testing.allocator.free(value);
        } else env.unsetForeign(name);
    };
    const cfg: Config = .{ .rank = 0, .size = 2, .coordinator = "127.0.0.1:46209", .device = "loopback" };
    try setDistEnv(cfg, "/tmp/step4-env.json");
    const writes = test_env_writes;
    try setDistEnv(cfg, "/tmp/step4-env.json");
    try std.testing.expectEqual(writes, test_env_writes);
}

test "dist review 1: silent bell does not delay real ranks past formation" {
    const coord: Coordinator = .{ .ip4 = .{ 127, 0, 0, 1 }, .port = 46221 };
    var rv: Rendezvous = .{ .listener = Bell.listen(coord, 3), .size = 3 };
    try std.testing.expect(rv.listener != Bell.none);
    defer rv.close();
    const silent = try lan.connectTimeout(coord.ip4, coord.port + 1, 500);
    defer _ = std.c.close(silent);
    const t = try std.Thread.spawn(.{}, Rendezvous.run, .{&rv});
    defer {
        rv.stop.store(true, .release);
        t.join();
    }
    const one = Bell.connect(coord, 1, null, testDeadline(1000));
    defer _ = std.c.close(one.fd);
    const two = Bell.connect(coord, 2, null, testDeadline(1000));
    defer _ = std.c.close(two.fd);
    try std.testing.expect(rv.waitAll(900));
}

test "dist review 2: failed backend clear refuses the next open by name" {
    const cfg: Config = .{ .rank = 0, .size = 2, .coordinator = "127.0.0.1:46223", .device = "loopback", .formation_deadline_ms = testDeadline(20) };
    var link = try newLink(cfg);
    test_clear_result = 1;
    defer {
        test_clear_result = null;
        cluster().backends_not_cleared = false;
    }
    link.deinit();
    var reopened = newLink(cfg) catch |err| {
        try std.testing.expectEqual(error.DistBackendsNotCleared, err);
        return;
    };
    defer reopened.deinit();
    return error.TestExpectedBackendRefusal;
}

test "dist review 3: cold teardown retains the last step and formation bell mask" {
    var link = try newLink(.{ .rank = 0, .size = 3, .coordinator = "127.0.0.1:46225", .device = "-,one,two" });
    link.state.?.watchdog.step.store(41, .monotonic);
    link.state.?.bells_seen.store(2, .release);
    link.deinit();
    try std.testing.expectEqual(@as(u64, 41), liveness().step);
    try std.testing.expectEqual(@as(u32, 2), liveness().bells_seen);
    var failed = try newLink(.{ .rank = 0, .size = 3, .coordinator = "127.0.0.1:46225", .device = "-,one,two" });
    failed.state.?.bells_seen.store(4, .release);
    failed.deinit();
    var buf: [64]u8 = undefined;
    try std.testing.expectEqualStrings("1", missingRanks(liveness().bells_seen, 3, &buf));
}

test "dist review 4: three-rank partial GO uses the link exit seam" {
    var one = try TestBells.open(46227);
    defer one.close();
    var two = try TestBells.open(46229);
    defer two.close();
    const cfg: Config = .{ .rank = 0, .size = 3, .coordinator = "127.0.0.1:46227", .device = "-,one,two", .ready_in_open = true };
    var link = try newLink(cfg);
    defer link.deinit();
    link.bells[1] = one.leader.bells[1];
    one.leader.bells[1] = .{};
    link.bells[2] = two.leader.bells[1];
    two.leader.bells[1] = .{};
    var exits: TestDistExit = .{};
    link.state.?.test_exits = &exits;
    try sendReadyFrame(one.worker, .{});
    try sendReadyFrame(two.worker, .{});
    _ = std.c.shutdown(link.bells[2].fd, std.c.SHUT.WR);
    try std.testing.expectError(error.DistPartialGo, link.finishOpen(cfg, testDeadline(0), testDeadline(1000)));
    var byte: [1]u8 = undefined;
    try one.worker.readAll(&byte);
    try std.testing.expectEqual(Bell.go_byte, byte[0]);
    try std.testing.expectEqual(@as(u32, 1) << @backingInt(dist_watchdog.ExitCause.partial_go), exits.causes.load(.acquire));
}

test "dist review 5: cancel identifies an already finished failed attempt" {
    TestSwitchOpen.setup();
    defer TestSwitchOpen.cleanup();
    TestSwitchOpen.failure = error.DistNoReady;
    try TestSwitchOpen.start(TestSwitchOpen.request(75));
    try std.testing.expectEqualStrings("DistNoReady", (try TestSwitchOpen.outcome(75)).failed.err);
    try std.testing.expectEqualStrings("already_finished", @tagName(switcher.cancel(75)));
}

test "dist review 6: TP1 ticks do not arm a switch link before mode swap" {
    TestSwitchOpen.setup();
    defer TestSwitchOpen.cleanup();
    try TestSwitchOpen.start(TestSwitchOpen.request(76));
    const link = (try TestSwitchOpen.outcome(76)).formed.link;
    tickInferenceLoop();
    try std.testing.expect(!link.state.?.heartbeat_active.load(.acquire));
    parkInferenceLoop();
}

test "dist review 7: unprimed switch environment is refused before spawn" {
    TestSwitchOpen.setup();
    defer TestSwitchOpen.cleanup();
    const req = TestSwitchOpen.request(77);
    setForeignIfDifferent("MLX_RANK", "99");
    defer setForeignIfDifferent("MLX_RANK", "0");
    try std.testing.expectError(error.DistEnvNotPrimed, switcher.start(.{ .cfg = req.cfg, .model_dir = "/m/switch" }, req.role, req.n, req.deadline_epoch_ms));
    try std.testing.expect(!TestSwitchOpen.entered.load(.acquire));
}

test "dist switch slice3 interface: context start publishes measured formation" {
    TestSwitchOpen.setup();
    defer TestSwitchOpen.cleanup();
    const req = TestSwitchOpen.request(81);
    try primeEnv(req.cfg);
    TestSwitchOpen.block = true;
    const ctx: switcher.Context = .{ .cfg = req.cfg, .model_dir = "/m/switch" };
    try switcher.start(ctx, .leader, 81, req.deadline_epoch_ms);
    try TestSwitchOpen.awaitEntered();
    crt.sleepMs(25);
    TestSwitchOpen.finish.store(true, .release);
    const formed = (try TestSwitchOpen.outcome(81)).formed;
    try std.testing.expect(formed.open_ms >= 25);
    try std.testing.expect(formed.ready_to_go_ms != null);
    try std.testing.expect(formed.ready_to_go_ms.? < formed.open_ms);
    try std.testing.expectEqual(formed.link, switcher.poll(81).?.formed.link);
    try switcher.activate(81);
    try std.testing.expect(formed.link.state.?.heartbeat_active.load(.acquire));
}

test "dist switch slice3 failure: missing READY names the peer and elapsed time" {
    TestSwitchOpen.setup();
    defer TestSwitchOpen.cleanup();
    TestSwitchOpen.failure = error.DistNoReady;
    var req = TestSwitchOpen.request(82);
    req.deadline_epoch_ms = crt.wallMs() + 80;
    try TestSwitchOpen.start(req);
    const failed = (try TestSwitchOpen.outcome(82)).failed;
    try std.testing.expectEqualStrings("DistNoReady", failed.err);
    try std.testing.expectEqual(@as(?u32, 1), failed.rank);
    try std.testing.expect(failed.elapsed_ms >= 60);
}

const TestReleasePeer = struct {
    bell: Bell,
    delay_ms: u32,
    result: ?anyerror = null,
    early_eof: bool = false,
    sw: ?*switch_controller.Switch = null,
    before_eof: ?switch_controller.Snapshot = null,

    fn run(self: *TestReleasePeer) void {
        dist_watchdog.setBellTimeout(self.bell.fd, 1000);
        const link: Link = .{ .group = .{}, .rank = 1, .size = 3, .bells = blk: {
            var bells: [max_ranks]Bell = @splat(.{});
            bells[0] = self.bell;
            break :blk bells;
        } };
        const msg = recvMsg(&link, 0) catch |err| {
            self.result = err;
            return;
        };
        if (msg[0] != OP_RELEASE) {
            self.result = error.TestExpectedRelease;
            return;
        }
        if (self.sw) |sw| self.before_eof = sw.snapshot();
        crt.sleepMs(self.delay_ms);
        var byte: [1]u8 = undefined;
        self.early_eof = std.c.recv(self.bell.fd, &byte, 1, std.c.MSG.PEEK | std.c.MSG.DONTWAIT) == 0;
        _ = std.c.shutdown(self.bell.fd, std.c.SHUT.RDWR);
    }
};

test "dist switch slice3 release: leader waits for both follower EOFs" {
    TestSwitchOpen.setup();
    defer TestSwitchOpen.cleanup();
    var req = TestSwitchOpen.request(83);
    req.cfg.size = 3;
    req.cfg.device = "-,loopback,loopback";
    try TestSwitchOpen.start(req);
    const link = (try TestSwitchOpen.outcome(83)).formed.link;
    var one: TestReleasePeer = .{ .bell = TestSwitchOpen.peer_bell, .delay_ms = 70 };
    var two: TestReleasePeer = .{ .bell = TestSwitchOpen.extra_peer, .delay_ms = 150 };
    const t1 = try std.Thread.spawn(.{}, TestReleasePeer.run, .{&one});
    const t2 = try std.Thread.spawn(.{}, TestReleasePeer.run, .{&two});

    var rank: u32 = 99;
    const began = crt.monotonicMs();
    const result = switcher.release(83, link, &rank);
    t1.join();
    t2.join();
    try result;
    try std.testing.expect(crt.monotonicMs() - began >= 150);
    try std.testing.expect(one.result == null and two.result == null);
    try std.testing.expect(!one.early_eof and !two.early_eof);
    try std.testing.expect(switcher.poll(83) == null);
    try std.testing.expect(cluster().active_state == null);
}

test "dist switch slice3 release: missing EOF has one two-second cap" {
    TestSwitchOpen.setup();
    defer TestSwitchOpen.cleanup();
    try TestSwitchOpen.start(TestSwitchOpen.request(84));
    const link = (try TestSwitchOpen.outcome(84)).formed.link;
    var rank: u32 = 99;
    const began = crt.monotonicMs();
    try switcher.release(84, link, &rank);
    const elapsed = crt.monotonicMs() - began;
    try std.testing.expect(elapsed >= 1900 and elapsed < 2600);
    try std.testing.expect(switcher.poll(84) == null);
}

test "dist switch slice3 release: send failure returns the missing rank without exit" {
    TestSwitchOpen.setup();
    defer TestSwitchOpen.cleanup();
    var req = TestSwitchOpen.request(85);
    req.cfg.size = 3;
    req.cfg.device = "-,loopback,loopback";
    try TestSwitchOpen.start(req);
    const link = (try TestSwitchOpen.outcome(85)).formed.link;
    _ = std.c.shutdown(link.bells[2].fd, std.c.SHUT.WR);
    var rank: u32 = 99;
    try std.testing.expectError(error.ReleaseSendFailed, switcher.release(85, link, &rank));
    try std.testing.expectEqual(@as(u32, 2), rank);
    try std.testing.expectEqual(@as(u32, 0), TestSwitchOpen.exits.count.load(.acquire));
    try std.testing.expect(link.state != null);
}

test "dist switch slice3 release: follower only tears down its own link" {
    TestSwitchOpen.setup();
    defer TestSwitchOpen.cleanup();
    var req = TestSwitchOpen.request(86);
    req.role = .follower;
    req.cfg.rank = 1;
    try TestSwitchOpen.start(req);
    const formed = (try TestSwitchOpen.outcome(86)).formed;
    try std.testing.expect(formed.ready_to_go_ms == null);
    var rank: u32 = 99;
    try switcher.release(86, formed.link, &rank);
    var byte: [1]u8 = undefined;
    try std.testing.expectEqual(@as(isize, 0), std.c.read(TestSwitchOpen.peer_bell.fd, &byte, 1));
    try std.testing.expectEqual(@as(u32, 0), TestSwitchOpen.exits.count.load(.acquire));
}

test "dist switch slice3 fall: both roles close without RELEASE or EOF wait" {
    for ([_]bool{ false, true }) |follower| {
        TestSwitchOpen.setup();
        defer TestSwitchOpen.cleanup();
        var req = TestSwitchOpen.request(87);
        if (follower) {
            req.role = .follower;
            req.cfg.rank = 1;
        }
        try TestSwitchOpen.start(req);
        const link = (try TestSwitchOpen.outcome(87)).formed.link;
        cluster().tp_link = link;
        try switcher.fall(86, link);
        try std.testing.expect(switcher.poll(87) != null);
        const began = crt.monotonicMs();
        try switcher.fall(87, link);
        try std.testing.expect(crt.monotonicMs() - began < 500);
        var byte: [1]u8 = undefined;
        try std.testing.expectEqual(@as(isize, 0), std.c.read(TestSwitchOpen.peer_bell.fd, &byte, 1));
        try std.testing.expect(cluster().active_state == null and cluster().tp_link == null);
        try std.testing.expectEqual(@as(u32, 0), TestSwitchOpen.exits.count.load(.acquire));
    }
}

test "dist switch slice3 abort: RELEASE within two seconds cancels the fallback" {
    TestSwitchOpen.setup();
    defer TestSwitchOpen.cleanup();
    var req = TestSwitchOpen.request(88);
    req.role = .follower;
    req.cfg.rank = 1;
    try TestSwitchOpen.start(req);
    const link = (try TestSwitchOpen.outcome(88)).formed.link;
    crt.sleepMs(80);
    const began = crt.monotonicMs();
    switcher.abortFormed(88);
    try TestSwitchOpen.peer_bell.writeAll(&[_]u8{Bell.release_byte});
    try std.testing.expectEqual(OP_RELEASE, (try recvMsg(link, 0))[0]);
    var rank: u32 = 99;
    try switcher.release(88, link, &rank);
    try std.testing.expect(crt.monotonicMs() - began < 500);
    try std.testing.expectEqual(@as(u32, 0), TestSwitchOpen.exits.count.load(.acquire));
}

test "dist switch slice3 abort: timer EOF returns FallRequested without an exit" {
    TestSwitchOpen.setup();
    defer TestSwitchOpen.cleanup();
    var req = TestSwitchOpen.request(89);
    req.role = .follower;
    req.cfg.rank = 1;
    try TestSwitchOpen.start(req);
    const link = (try TestSwitchOpen.outcome(89)).formed.link;
    const began = crt.monotonicMs();
    switcher.abortFormed(89);
    crt.sleepMs(80);
    switcher.abortFormed(89);
    try std.testing.expectError(error.FallRequested, recvMsg(link, 0));
    const elapsed = crt.monotonicMs() - began;
    try std.testing.expect(elapsed >= 1900 and elapsed < 2600);
    var byte: [1]u8 = undefined;
    try std.testing.expectError(error.FallRequested, link.bells[0].readAll(&byte));
    try std.testing.expectError(error.FallRequested, sendStepTime(link.bells[0], 10));
    try switcher.fall(89, link);
    try std.testing.expectEqual(@as(u32, 0), TestSwitchOpen.exits.count.load(.acquire));
    try std.testing.expect(switcher.poll(89) == null);
}

test "dist switch slice3 abort: release ignores the same N and only fences a higher N" {
    TestSwitchOpen.setup();
    defer TestSwitchOpen.cleanup();
    try TestSwitchOpen.start(TestSwitchOpen.request(90));
    const link = (try TestSwitchOpen.outcome(90)).formed.link;
    const Abort = struct {
        fn run(l: *Link) void {
            while (!l.state.?.releasing.load(.acquire)) crt.sleepMs(1);
            switcher.abortFormed(90);
            switcher.abortFormed(100);
        }
    };
    const t = try std.Thread.spawn(.{}, Abort.run, .{link});
    var rank: u32 = 99;
    const result = switcher.release(90, link, &rank);
    t.join();
    try result;
    try std.testing.expectEqual(@as(u32, 0), TestSwitchOpen.exits.count.load(.acquire));
    try std.testing.expectError(error.InvalidArgs, TestSwitchOpen.start(TestSwitchOpen.request(91)));
}

test "dist switch slice3 group: attempt shards are visible and retire independently of startup" {
    for ([_]bool{ false, true }) |startup| {
        TestSwitchOpen.setup();
        defer TestSwitchOpen.cleanup();
        const old = cluster().group;
        defer cluster().group = old;
        cluster().group = if (startup) .{ .io = std.testing.io, .allocator = std.testing.allocator, .mode = .pipeline, .model = "/m/cold", .split = null } else null;
        var req = TestSwitchOpen.request(101);
        if (startup) {
            req.role = .follower;
            req.cfg.rank = 1;
        }
        try TestSwitchOpen.start(req);
        const link = (try TestSwitchOpen.outcome(101)).formed.link;
        var cfg: model_mod.ModelConfig = .{ .model_type = "deepseek_v41", .num_hidden_layers = 8, .num_attention_heads = 64, .dsv4_o_groups = 8, .moe_intermediate_size = 2304 };
        try std.testing.expect(serves("/m/switch") and holds("/m/switch"));
        const part = try planFor("/m/switch", &cfg);
        try std.testing.expect(part != null);
        try std.testing.expectEqual(req.cfg.rank, part.?.shard.rank);
        try std.testing.expectEqual(@as(u32, 2), part.?.shard.size);
        try std.testing.expectEqual(@as(u64, 600), hostShare("/m/switch", &cfg, 1000));
        if (startup) {
            var rank: u32 = 99;
            try switcher.release(101, link, &rank);
        } else try switcher.fall(101, link);
        try std.testing.expect(!serves("/m/switch") and !holds("/m/switch"));
        try std.testing.expect(try planFor("/m/switch", &cfg) == null);
        try std.testing.expectEqual(startup, serves("/m/cold"));
    }
}

test "dist switch slice3 unload: a formed attempt keeps its link and resident model" {
    TestSwitchOpen.setup();
    defer TestSwitchOpen.cleanup();
    const Registry = @import("../../../host/registry/registry.zig").ModelRegistry;
    var reg = try Registry.init(std.testing.allocator, std.testing.io, null, 3, 0, null);
    defer reg.deinit();
    const entry = try reg.registerStub("switch", "/m/switch", 1024);
    entry.bytes_resident = 1024;
    entry.state = .ready;
    try TestSwitchOpen.start(TestSwitchOpen.request(102));
    const link = (try TestSwitchOpen.outcome(102)).formed.link;
    const scheduler = @import("../scheduler.zig");
    var sch: scheduler.Scheduler = undefined;
    sch.io = std.testing.io;
    sch.registry = reg;
    try std.testing.expectError(error.DistSwitchActive, scheduler.unloadModel(&sch, "switch"));
    entry.state = .evicting;
    var req: @import("../../../host/gpu_thread/requests.zig").UnloadRequest = .{ .entry = entry };
    @import("../runtime/load_requests.zig").runUnloadRequest(&sch, &req);
    try std.testing.expect(req.done);
    try std.testing.expectEqualStrings("DistSwitchActive", req.error_name.?);
    try std.testing.expect(!entry.unloadResident());
    try std.testing.expectEqual(.ready, entry.state);
    try std.testing.expectEqual(@as(u64, 1024), entry.bytes_resident);
    try std.testing.expectEqual(link, switcher.poll(102).?.formed.link);
    try std.testing.expect(!link.state.?.closing.load(.acquire));
    try switcher.fall(102, link);
    try std.testing.expect(entry.unloadResident());
    try std.testing.expectEqual(.unloaded, entry.state);
}

test "dist fix2 finding 1: formed follower survives 3500ms before leader activation" {
    TestSwitchOpen.setup();
    defer TestSwitchOpen.cleanup();
    var req = TestSwitchOpen.request(201);
    req.role = .follower;
    req.cfg.rank = 1;
    try TestSwitchOpen.start(req);
    const link = (try TestSwitchOpen.outcome(201)).formed.link;
    var probe = struct {
        link: *Link,
        returned: std.atomic.Value(bool) = .init(false),
        result: ?anyerror = null,
        op: i32 = 0,
        fn run(self: *@This()) void {
            const msg = recvMsg(self.link, 0) catch |err| {
                self.result = err;
                self.returned.store(true, .release);
                return;
            };
            self.op = msg[0];
            self.returned.store(true, .release);
        }
    }{ .link = link };
    const t = try std.Thread.spawn(.{}, @TypeOf(probe).run, .{&probe});
    crt.sleepMs(3500);
    const early = probe.returned.load(.acquire);
    try TestSwitchOpen.peer_bell.writeAll(&.{Bell.release_byte});
    t.join();
    try std.testing.expect(!early);
    try std.testing.expectEqual(@as(u32, 0), TestSwitchOpen.exits.count.load(.acquire));
    try std.testing.expectEqual(@as(?anyerror, null), probe.result);
    try std.testing.expectEqual(OP_RELEASE, probe.op);
    try std.testing.expect(link.state.?.follower_ready.load(.acquire));
    try switcher.fall(201, link);
}

test "dist fix2 finding 2: aborting bell timeouts request fall without exit" {
    TestSwitchOpen.setup();
    defer TestSwitchOpen.cleanup();
    var req = TestSwitchOpen.request(202);
    req.role = .follower;
    req.cfg.rank = 1;
    try TestSwitchOpen.start(req);
    const link = (try TestSwitchOpen.outcome(202)).formed.link;
    try TestSwitchOpen.peer_bell.writeAll(&.{Bell.pipe_byte});
    try std.testing.expectEqual(Bell.pipe_byte, try link.bells[0].wait());
    switcher.abortFormed(202);
    dist_watchdog.setBellTimeout(link.bells[0].fd, 40);
    try std.testing.expectError(error.FallRequested, recvMsg(link, 0));
    var byte: [1]u8 = undefined;
    try std.testing.expectError(error.FallRequested, link.bells[0].readAll(&byte));
    try std.testing.expectEqual(@as(u32, 0), TestSwitchOpen.exits.count.load(.acquire));
    try switcher.fall(202, link);
}

test "dist fix2 finding 2: watcher EOF admitted before abort rechecks under exit mutex" {
    TestSwitchOpen.setup();
    defer TestSwitchOpen.cleanup();
    var req = TestSwitchOpen.request(203);
    req.role = .follower;
    req.cfg.rank = 1;
    try TestSwitchOpen.start(req);
    const link = (try TestSwitchOpen.outcome(203)).formed.link;
    const state = link.state.?;
    try TestSwitchOpen.peer_bell.writeAll(&.{Bell.pipe_byte});
    try std.testing.expectEqual(Bell.pipe_byte, try link.bells[0].wait());
    var probe = struct {
        bell: Bell,
        result: ?anyerror = null,
        fn run(self: *@This()) void {
            var byte: [1]u8 = undefined;
            self.bell.readAll(&byte) catch |err| {
                self.result = err;
            };
        }
    }{ .bell = link.bells[0] };
    lock(&state.exit_mu);
    _ = std.c.shutdown(TestSwitchOpen.peer_bell.fd, std.c.SHUT.RDWR);
    const reader = try std.Thread.spawn(.{}, @TypeOf(probe).run, .{&probe});
    crt.sleepMs(100);
    state.aborting.store(true, .release);
    state.exit_mu.unlock();
    reader.join();
    if (state.watcher) |t| t.join();
    state.watcher = null;
    try std.testing.expectEqual(@as(u32, 0), TestSwitchOpen.exits.count.load(.acquire));
    try std.testing.expectEqual(error.FallRequested, probe.result.?);
    try switcher.fall(203, link);
}

test "dist fix2 finding 3: eviction refuses a switch victim without reclaiming its bytes" {
    TestSwitchOpen.setup();
    defer TestSwitchOpen.cleanup();
    const Registry = @import("../../../host/registry/registry.zig").ModelRegistry;
    var reg = try Registry.init(std.testing.allocator, std.testing.io, null, 3, 0, null);
    defer reg.deinit();
    const victim = try reg.registerStub("switch", "/m/switch", 1024);
    reg.markReadyLocked(victim, 1024);
    victim.state = .evicting;
    const later = try reg.registerStub("later", "/m/later", 512);
    reg.markReadyLocked(later, 512);
    later.state = .evicting;
    const target = try reg.registerStub("target", "/m/target", 2048);
    target.state = .loading;
    try TestSwitchOpen.start(TestSwitchOpen.request(204));
    const link = (try TestSwitchOpen.outcome(204)).formed.link;
    const scheduler = @import("../scheduler.zig");
    var sch: scheduler.Scheduler = undefined;
    sch.io = std.testing.io;
    sch.registry = reg;
    sch.hot_prefix_cache = null;
    sch.allocator = std.testing.allocator;
    sch.warm_queue = .empty;
    var victims = [_]*@import("../../../host/registry/registry.zig").LoadedModel{ victim, later };
    var cfg: model_mod.ModelConfig = .{};
    var req: @import("../../../host/gpu_thread/requests.zig").LoadRequest = .{
        .entry = target,
        .config = &cfg,
        .tok = undefined,
        .chat_config = undefined,
        .model_dir = "/m/target",
        .allocator = std.testing.allocator,
        .evict_entries = &victims,
        .ane_declined = true,
    };
    defer if (req.error_name) |name| std.testing.allocator.free(name);
    @import("../runtime/load_requests.zig").runLoadRequest(&sch, &req);
    try std.testing.expect(req.done);
    try std.testing.expectEqualStrings("DistSwitchActive", req.error_name.?);
    try std.testing.expectEqual(error.DistSwitchActive, Registry.loadErrorFromName(req.error_name));
    try std.testing.expectEqual(.unloaded, target.state);
    try std.testing.expectEqual(.ready, victim.state);
    try std.testing.expectEqual(.ready, later.state);
    try std.testing.expectEqual(@as(u64, 1024), victim.bytes_resident);
    try std.testing.expectEqual(@as(u64, 1536), reg.current_resident_bytes);
    try std.testing.expectEqual(link, switcher.poll(204).?.formed.link);
    try switcher.fall(204, link);
}

test "dist fix2 finding 4: teardown deadline survives disarm and closing" {
    bindInferenceThread();
    const saved = if (env.span(.MLX_SERVE_DIST_STALL_MS)) |v| try std.testing.allocator.dupeSentinel(u8, v, 0) else null;
    defer if (saved) |v| {
        env.set(.MLX_SERVE_DIST_STALL_MS, v);
        std.testing.allocator.free(v);
    } else env.unset(.MLX_SERVE_DIST_STALL_MS);
    env.set(.MLX_SERVE_DIST_STALL_MS, "60");
    var link = try newLink(.{ .rank = 0, .size = 2, .coordinator = "127.0.0.1:46231", .device = "loopback" });
    defer link.deinit();
    var exits: TestDistExit = .{};
    link.state.?.test_exits = &exits;
    startWatchdog(&link);
    link.disarm();
    link.state.?.closing.store(true, .release);
    crt.sleepMs(200);
    try std.testing.expectEqual(@as(u32, 1), exits.count.load(.acquire));
    link.state.?.teardown_done.store(true, .release);
    link.state.?.exit(.teardown_timeout, null, null);
    try std.testing.expectEqual(@as(u32, 1), exits.count.load(.acquire));
}

test "dist fix2 finding 4: unsafe teardown and activation refuse without side effects" {
    TestSwitchOpen.setup();
    defer TestSwitchOpen.cleanup();
    try TestSwitchOpen.start(TestSwitchOpen.request(205));
    const link = (try TestSwitchOpen.outcome(205)).formed.link;
    var rank: u32 = 99;
    link.state.?.recvs.store(1, .release);
    try std.testing.expectError(error.DistTeardownBusy, switcher.release(205, link, &rank));
    try std.testing.expectError(error.DistTeardownBusy, switcher.fall(205, link));
    link.state.?.recvs.store(0, .release);
    link.state.?.watchdog.begin(testDeadline(0));
    try std.testing.expectError(error.DistTeardownBusy, switcher.release(205, link, &rank));
    try std.testing.expectError(error.DistTeardownBusy, switcher.fall(205, link));
    link.state.?.watchdog.done();
    var probe = struct {
        link: *Link,
        release_error: ?anyerror = null,
        fall_error: ?anyerror = null,
        activate_error: ?anyerror = null,
        fn run(self: *@This()) void {
            var peer_rank: u32 = 99;
            switcher.activate(205) catch |err| {
                self.activate_error = err;
            };
            switcher.release(205, self.link, &peer_rank) catch |err| {
                self.release_error = err;
            };
            switcher.fall(205, self.link) catch |err| {
                self.fall_error = err;
            };
        }
    }{ .link = link };
    const t = try std.Thread.spawn(.{}, @TypeOf(probe).run, .{&probe});
    t.join();
    try std.testing.expectEqual(error.DistTeardownWrongThread, probe.activate_error.?);
    try std.testing.expectEqual(error.DistTeardownWrongThread, probe.release_error.?);
    try std.testing.expectEqual(error.DistTeardownWrongThread, probe.fall_error.?);
    try std.testing.expectEqual(link, switcher.poll(205).?.formed.link);
    try std.testing.expect(!link.state.?.releasing.load(.acquire));
    try std.testing.expect(!link.state.?.closing.load(.acquire));
    try std.testing.expectEqual(@as(u32, 99), rank);
    try switcher.fall(205, link);
}

const TestTeardownPause = struct {
    stage: enum { drain, free },
    early_release: bool = false,
    thread_id: ?std.Thread.Id = null,

    fn visit(self: *@This(), stage: @TypeOf(self.stage)) void {
        if (stage != self.stage) return;
        self.thread_id = std.Thread.getCurrentId();
        if (stage == .drain) {
            var byte: [1]u8 = undefined;
            self.early_release = std.c.recv(TestSwitchOpen.peer_bell.fd, &byte, 1, std.c.MSG.PEEK | std.c.MSG.DONTWAIT) == 1;
        }
        crt.sleepMs(200);
    }
};

test "dist fix2 finding 4: drain precedes RELEASE and deadline covers drain and group free" {
    const saved = if (env.span(.MLX_SERVE_DIST_STALL_MS)) |v| try std.testing.allocator.dupeSentinel(u8, v, 0) else null;
    defer if (saved) |v| {
        env.set(.MLX_SERVE_DIST_STALL_MS, v);
        std.testing.allocator.free(v);
    } else env.unset(.MLX_SERVE_DIST_STALL_MS);
    env.set(.MLX_SERVE_DIST_STALL_MS, "60");
    for ([_]@FieldType(TestTeardownPause, "stage"){ .drain, .free }) |stage| {
        TestSwitchOpen.setup();
        defer TestSwitchOpen.cleanup();
        try TestSwitchOpen.start(TestSwitchOpen.request(206));
        const link = (try TestSwitchOpen.outcome(206)).formed.link;
        var pause: TestTeardownPause = .{ .stage = stage };
        link.state.?.test_teardown_pause = &pause;
        var peer_wait: TestReleasePeer = .{ .bell = TestSwitchOpen.peer_bell, .delay_ms = 0 };
        const t = if (stage == .free) try std.Thread.spawn(.{}, TestReleasePeer.run, .{&peer_wait}) else null;
        var rank: u32 = 99;
        const result = switcher.release(206, link, &rank);
        if (t) |thread| thread.join();
        try result;
        try std.testing.expect(peer_wait.result == null);
        try std.testing.expect(!pause.early_release);
        try std.testing.expectEqual(@as(u32, 1), TestSwitchOpen.exits.count.load(.acquire));
    }
}

test "dist fix2 finding 5: abort between formed CAS and return survives publication" {
    TestSwitchOpen.setup();
    defer TestSwitchOpen.cleanup();
    const Gate = struct {
        var entered: std.atomic.Value(bool) = .init(false);
        var proceed: std.atomic.Value(bool) = .init(false);
        fn pause() void {
            entered.store(true, .release);
            while (!proceed.load(.acquire)) crt.sleepMs(1);
        }
    };
    Gate.entered.store(false, .release);
    Gate.proceed.store(false, .release);
    defer Gate.proceed.store(true, .release);
    switcher.testing.after_formed = Gate.pause;
    var req = TestSwitchOpen.request(207);
    req.role = .follower;
    req.cfg.rank = 1;
    try TestSwitchOpen.start(req);
    const end = testDeadline(2000);
    while (!Gate.entered.load(.acquire) and msLeft(end) > 0) crt.sleepMs(1);
    const entered = Gate.entered.load(.acquire);
    const unpublished = switcher.poll(207) == null;
    switcher.abortFormed(207);
    Gate.proceed.store(true, .release);
    const link = (try TestSwitchOpen.outcome(207)).formed.link;
    try std.testing.expect(entered and unpublished);
    try std.testing.expect(link.state.?.aborting.load(.acquire));
    try TestSwitchOpen.peer_bell.writeAll(&.{Bell.release_byte});
    try std.testing.expectEqual(OP_RELEASE, (try recvMsg(link, 0))[0]);
    var rank: u32 = 99;
    try switcher.release(207, link, &rank);
    try std.testing.expectEqual(@as(u32, 0), TestSwitchOpen.exits.count.load(.acquire));
}

test "dist fix2 finding 7: quarantined late formation never publishes a group" {
    for ([_]bool{ false, true }) |poll_before_return| {
        TestSwitchOpen.setup();
        defer TestSwitchOpen.cleanup();
        const old = cluster().group;
        defer cluster().group = old;
        cluster().group = null;
        TestSwitchOpen.block = true;
        TestSwitchOpen.ignore_cancel = true;
        var req = TestSwitchOpen.request(208);
        req.deadline_epoch_ms = crt.wallMs() + 80;
        try TestSwitchOpen.start(req);
        try TestSwitchOpen.awaitEntered();
        crt.sleepMs(1200);
        if (poll_before_return) try std.testing.expect(switcher.quarantined());
        TestSwitchOpen.finish.store(true, .release);
        const end = testDeadline(2000);
        while (!TestSwitchOpen.returned.load(.acquire) and msLeft(end) > 0) crt.sleepMs(1);
        crt.sleepMs(50);
        try std.testing.expectEqual(.quarantined, switcher.poll(208).?);
        try std.testing.expect(!switchHolds("/m/switch"));
        try std.testing.expect(!holds("/m/switch") and !serves("/m/switch"));
        var cfg: model_mod.ModelConfig = .{ .model_type = "deepseek_v41", .num_hidden_layers = 8, .num_attention_heads = 64, .dsv4_o_groups = 8, .moe_intermediate_size = 2304 };
        try std.testing.expect(try planFor("/m/switch", &cfg) == null);
        try std.testing.expectEqual(@as(u64, 1000), hostShare("/m/switch", &cfg, 1000));
    }
}

test "dist fix2 public exit: release failure records rank and exits one" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var root_buf: [512]u8 = undefined;
    const root = root_buf[0..try tmp.dir.realPath(std.testing.io, &root_buf)];
    var path_buf: [600]u8 = undefined;
    const path = try std.fmt.bufPrint(&path_buf, "{s}/release-exit.log", .{root});
    const pid = std.c.fork();
    if (pid < 0) return error.TestForkFailed;
    if (pid == 0) {
        log.openFile(path, 0) catch std.c._exit(99);
        exitReleaseFailed(3);
    }
    var status_code: c_int = 0;
    try std.testing.expectEqual(pid, std.c.waitpid(pid, &status_code, 0));
    try std.testing.expectEqual(@as(c_int, 1 << 8), status_code);
    const record = try tmp.dir.readFileAlloc(std.testing.io, "release-exit.log", std.testing.allocator, .limited(4096));
    defer std.testing.allocator.free(record);
    try std.testing.expect(std.mem.startsWith(u8, record, "[dist-exit] epoch_ms="));
    try std.testing.expect(std.mem.indexOf(u8, record, " cause=release_failed peer=3 step=") != null);
    try std.testing.expect(std.mem.endsWith(u8, record, " waited_ms=-\n"));
}

const switch_adapter = @import("switch_adapter.zig");
const switch_controller = @import("../../../host/internal/switch.zig");

const TestAdapter = struct {
    sw: switch_controller.Switch,
    adapter: switch_adapter.Adapter = undefined,
    hook_thread: ?std.Thread.Id = null,
    hook_error: bool = false,
    hook_entered: std.atomic.Value(bool) = .init(false),
    hook_finish: std.atomic.Value(bool) = .init(true),
    var active: ?*@This() = null;

    fn setup(self: *@This(), follower: bool, n: u64) !void {
        TestSwitchOpen.setup();
        self.* = .{ .sw = switch_controller.Switch.init(std.testing.io) };
        self.sw.spawn_watcher = false;
        var req = TestSwitchOpen.request(n);
        req.cfg.rank = if (follower) 1 else 0;
        try primeEnv(req.cfg);
        self.adapter.init(&self.sw, .{ .cfg = req.cfg, .model_dir = "/m/switch" });
        self.sw.formation = self.adapter.formation();
        active = self;
        switch_adapter.setSwapHook(swap);
        switch_adapter.bindInferenceThread(null);
    }

    fn cleanup(self: *@This()) void {
        _ = self;
        switch_adapter.unbindInferenceThread();
        switch_adapter.setSwapHook(null);
        active = null;
        TestSwitchOpen.cleanup();
    }

    fn swap(_: u64, link: *Link, role: switcher.Role, to: @import("../../../abi/plugin.zig").ServingMode) !void {
        if (to == .tp1) return;
        const self = active.?;
        self.hook_thread = std.Thread.getCurrentId();
        try std.testing.expectEqual(role == .leader, link.rank == 0);
        try std.testing.expect(cluster().tp_link == null);
        self.hook_entered.store(true, .release);
        while (!self.hook_finish.load(.acquire)) crt.sleepMs(1);
        if (self.hook_error) return error.TestSwapFailed;
    }

    fn form(self: *@This(), follower: bool, n: u64) !*Link {
        try std.testing.expectEqual(@as(u16, 200), self.sw.command(.drain, n, 0).status);
        try std.testing.expectEqual(@as(u16, 200), self.sw.command(if (follower) .join else .tp4, n, crt.wallMs() + 2000).status);
        const formed = (try TestSwitchOpen.outcome(n)).formed;
        const out = self.sw.formation.?.poll(self.sw.formation.?.ctx, n).?.formed;
        try std.testing.expectEqual(@intFromPtr(formed.link), @intFromPtr(out.ptr));
        try std.testing.expectEqual(formed.open_ms, out.open_ms);
        try std.testing.expectEqual(formed.ready_to_go_ms orelse 0, out.ready_to_go_ms);
        return formed.link;
    }
};

test "dist switch adapter: controller forms activates and releases both roles inline" {
    for ([_]bool{ false, true }) |follower| {
        var t: TestAdapter = undefined;
        try t.setup(follower, 301);
        defer t.cleanup();
        try std.testing.expectEqual(@as(u8, 1), t.sw.snapshot().peers);
        const link = try t.form(follower, 301);
        _ = t.sw.pump();
        try std.testing.expectEqual(.formed, t.sw.snapshot().state);
        try std.testing.expectEqual(std.Thread.getCurrentId(), t.hook_thread.?);
        try std.testing.expectEqual(link, cluster().tp_link.?);
        var peer_close: TestReleasePeer = .{ .bell = TestSwitchOpen.peer_bell, .delay_ms = 15 };
        const peer_thread = if (!follower) try std.Thread.spawn(.{}, TestReleasePeer.run, .{&peer_close}) else null;
        defer if (peer_thread) |th| th.join();
        try std.testing.expectEqual(@as(u16, 200), t.sw.command(.tp1, 301, 0).status);
        try std.testing.expectEqual(.released, t.sw.snapshot().state);
        try std.testing.expect(switcher.poll(301) == null);
        try std.testing.expect(cluster().tp_link == null);
    }
}

test "dist switch adapter: abort fallback reports release_timeout after FallRequested" {
    var t: TestAdapter = undefined;
    try t.setup(true, 302);
    defer t.cleanup();
    const link = try t.form(true, 302);
    _ = t.sw.pump();
    _ = t.sw.command(.abort, 302, 0);
    try std.testing.expectEqual(.aborting, t.sw.snapshot().state);
    try std.testing.expectError(error.FallRequested, recvMsg(link, 0));
    switch_adapter.followerFell(302, link);
    try std.testing.expectEqual(.aborted, t.sw.snapshot().state);
    try std.testing.expectEqual(.release_timeout, t.sw.snapshot().failure);
    try std.testing.expect(t.sw.link == null);
    try std.testing.expect(switcher.poll(302) == null);
}

test "dist switch higher abort: forming cancels the running attempt and waits for confirmation" {
    for ([_]bool{ false, true }) |follower| {
        var t: TestAdapter = undefined;
        try t.setup(follower, 5);
        defer t.cleanup();
        TestSwitchOpen.block = true;
        TestSwitchOpen.ignore_cancel = true;
        try std.testing.expectEqual(@as(u16, 200), t.sw.command(.drain, 5, 0).status);
        try std.testing.expectEqual(@as(u16, 200), t.sw.command(if (follower) .join else .tp4, 5, 0).status);
        try TestSwitchOpen.awaitEntered();
        const reply = t.sw.command(.abort, 7, 0);
        try std.testing.expectEqual(@as(u16, 200), reply.status);
        try std.testing.expectEqual(.aborting, reply.snap.state);
        try std.testing.expectEqual(@as(u64, 5), reply.snap.n);
        try std.testing.expectEqual(@as(u64, 7), reply.snap.high_water);
        try std.testing.expect(!reply.snap.admitting);
        try std.testing.expect(switcher.poll(5) == null);
        TestSwitchOpen.finish.store(true, .release);
        try std.testing.expectEqual(.cancelled, try TestSwitchOpen.outcome(5));
        _ = t.sw.pump();
        const done = t.sw.snapshot();
        try std.testing.expectEqual(.aborted, done.state);
        try std.testing.expectEqual(.cancelled, done.failure);
        try std.testing.expect(done.admitting);
        try std.testing.expectEqualStrings("stale", t.sw.command(.join, 6, 0).reason);
        try std.testing.expectEqual(@as(u32, 0), TestSwitchOpen.exits.count.load(.acquire));
    }
}

test "dist switch adapter: failed or absent swap hook consumes each role once" {
    for ([_]bool{ false, true }) |follower| for ([_]bool{ false, true }) |missing| {
        var t: TestAdapter = undefined;
        try t.setup(follower, 303);
        defer t.cleanup();
        t.hook_error = true;
        if (missing) switch_adapter.setSwapHook(null);
        _ = try t.form(follower, 303);
        var peer_close: TestReleasePeer = .{ .bell = TestSwitchOpen.peer_bell, .delay_ms = 0 };
        const th = if (!follower) try std.Thread.spawn(.{}, TestReleasePeer.run, .{&peer_close}) else null;
        defer if (th) |thread| thread.join();
        const f = t.sw.formation.?;
        const link = f.poll(f.ctx, 303).?.formed.ptr;
        try std.testing.expectError(error.SwapFailed, f.activate(f.ctx, 303, link));
        try std.testing.expect(switcher.poll(303) == null);
        try std.testing.expect(cluster().tp_link == null);
        try std.testing.expectEqual(@as(u32, 0), TestSwitchOpen.exits.count.load(.acquire));
    };
}

test "dist switch adapter: controller records swap failure after adapter cleanup" {
    var t: TestAdapter = undefined;
    try t.setup(true, 304);
    defer t.cleanup();
    t.hook_error = true;
    _ = try t.form(true, 304);
    _ = t.sw.pump();
    try std.testing.expectEqual(.aborted, t.sw.snapshot().state);
    try std.testing.expectEqual(.other, t.sw.snapshot().failure);
    try std.testing.expect(switcher.poll(304) == null);
}

test "dist switch adapter: unprimed start refuses before opener and maps cancellation" {
    var t: TestAdapter = undefined;
    try t.setup(false, 305);
    defer t.cleanup();
    setForeignIfDifferent("MLX_RANK", "99");
    defer setForeignIfDifferent("MLX_RANK", "0");
    const f = t.sw.formation.?;
    try std.testing.expectError(error.DistEnvNotPrimed, f.start(f.ctx, .leader, 305, crt.wallMs() + 2000));
    try std.testing.expect(!TestSwitchOpen.entered.load(.acquire));
    try std.testing.expectEqual(.unknown, f.cancel(f.ctx, 305));
    try primeEnv(t.adapter.context.cfg);
    TestSwitchOpen.block = true;
    try f.start(f.ctx, .leader, 305, crt.wallMs() + 2000);
    try TestSwitchOpen.awaitEntered();
    try std.testing.expectEqual(.cancelled, f.cancel(f.ctx, 305));
    _ = try TestSwitchOpen.outcome(305);
    try std.testing.expectEqual(.cancelled, f.poll(f.ctx, 305).?);
}

test "dist switch adapter: release failure names only the failed rank" {
    var t: TestAdapter = undefined;
    try t.setup(false, 306);
    defer t.cleanup();
    const link = try t.form(false, 306);
    _ = std.c.shutdown(link.bells[1].fd, std.c.SHUT.WR);
    const f = t.sw.formation.?;
    var rank: ?u32 = 99;
    try std.testing.expectError(error.ReleaseSendFailed, f.release(f.ctx, 306, link, &rank));
    try std.testing.expectEqual(@as(?u32, 1), rank);
}

test "dist switch adapter: exit_process fsyncs switch_abort cause and attempt N" {
    var t: TestAdapter = undefined;
    try t.setup(false, 307);
    defer t.cleanup();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var root_buf: [512]u8 = undefined;
    const root = root_buf[0..try tmp.dir.realPath(std.testing.io, &root_buf)];
    var path_buf: [600]u8 = undefined;
    const path = try std.fmt.bufPrint(&path_buf, "{s}/adapter-exit.log", .{root});
    const pid = std.c.fork();
    if (pid < 0) return error.TestForkFailed;
    if (pid == 0) {
        log.openFile(path, 0) catch std.c._exit(99);
        const f = t.sw.formation.?;
        f.exit_process(f.ctx, 307);
        std.c._exit(98);
    }
    var status_code: c_int = 0;
    try std.testing.expectEqual(pid, std.c.waitpid(pid, &status_code, 0));
    try std.testing.expectEqual(@as(c_int, 1 << 8), status_code);
    const record = try tmp.dir.readFileAlloc(std.testing.io, "adapter-exit.log", std.testing.allocator, .limited(4096));
    defer std.testing.allocator.free(record);
    try std.testing.expect(std.mem.indexOf(u8, record, "cause=switch_abort") != null);
    try std.testing.expect(std.mem.indexOf(u8, record, "n=307") != null);
}

const TestAdapterRunner = struct {
    ready: std.atomic.Value(bool) = .init(false),
    stop: std.atomic.Value(bool) = .init(false),
    id: std.Thread.Id = undefined,
    scheduler: ?*@import("../scheduler.zig").Scheduler = null,

    fn run(self: *@This()) void {
        switch_adapter.bindInferenceThread(self.scheduler);
        self.id = std.Thread.getCurrentId();
        self.ready.store(true, .release);
        defer switch_adapter.unbindInferenceThread();
        while (!self.stop.load(.acquire)) {
            if (self.scheduler) |sch| {
                sch.queue_mu.lockUncancelable(sch.io);
                while (!@import("../runtime/loop.zig").hasWorkPendingLocked(sch) and !self.stop.load(.acquire))
                    sch.queue_cond.waitTimeout(sch.io, &sch.queue_mu, .{ .duration = .{ .raw = .fromMilliseconds(2000), .clock = .boot } }) catch {
                        sch.queue_mu.unlock(sch.io);
                        return;
                    };
                sch.queue_mu.unlock(sch.io);
            }
            tickInferenceLoop();
            _ = switch_adapter.drain();
            crt.sleepMs(1);
        }
    }
};

test "dist switch adapter: off-thread activate release and fall wait for inference completion" {
    for ([_]bool{ false, true }) |falling| {
        var t: TestAdapter = undefined;
        try t.setup(true, 308);
        defer t.cleanup();
        const link = try t.form(true, 308);
        switch_adapter.unbindInferenceThread();
        var sch: @import("../scheduler.zig").Scheduler = undefined;
        sch.io = std.testing.io;
        sch.queue_mu = .init;
        sch.queue_cond = .init;
        sch.pending = .empty;
        sch.decoding = .empty;
        sch.vision_queue = .empty;
        sch.embed_queue = .empty;
        sch.cleanup_queue = .empty;
        sch.load_queue = .empty;
        sch.gen_queue = .empty;
        sch.unload_queue = .empty;
        sch.cache_export_queue = .empty;
        sch.cache_import_queue = .empty;
        sch.warm_queue = .empty;
        var runner: TestAdapterRunner = .{ .scheduler = &sch };
        const worker = try std.Thread.spawn(.{}, TestAdapterRunner.run, .{&runner});
        defer {
            runner.stop.store(true, .release);
            sch.queue_mu.lockUncancelable(sch.io);
            sch.queue_cond.signal(sch.io);
            sch.queue_mu.unlock(sch.io);
            worker.join();
        }
        while (!runner.ready.load(.acquire)) crt.sleepMs(1);
        t.hook_finish.store(false, .release);
        var caller = struct {
            f: switch_controller.Formation,
            link: *Link,
            done: std.atomic.Value(bool) = .init(false),
            result: ?anyerror = null,
            fn run(self: *@This()) void {
                self.f.activate(self.f.ctx, 308, self.link) catch |err| {
                    self.result = err;
                };
                self.done.store(true, .release);
            }
        }{ .f = t.sw.formation.?, .link = link };
        const client = try std.Thread.spawn(.{}, @TypeOf(caller).run, .{&caller});
        var client_joined = false;
        defer {
            t.hook_finish.store(true, .release);
            if (!client_joined) client.join();
        }
        while (!t.hook_entered.load(.acquire)) crt.sleepMs(1);
        try std.testing.expect(!caller.done.load(.acquire));
        try std.testing.expectEqual(runner.id, t.hook_thread.?);
        t.hook_finish.store(true, .release);
        client.join();
        client_joined = true;
        try std.testing.expect(caller.result == null);
        var pause: TestTeardownPause = .{ .stage = .drain };
        link.state.?.test_teardown_pause = &pause;
        const f = t.sw.formation.?;
        const began = crt.monotonicMs();
        var rank: ?u32 = 99;
        if (falling) f.fall(f.ctx, 308, link) else try f.release(f.ctx, 308, link, &rank);
        try std.testing.expect(crt.monotonicMs() - began >= 200);
        try std.testing.expectEqual(runner.id, pause.thread_id.?);
        if (!falling) try std.testing.expectEqual(@as(?u32, null), rank);
        try std.testing.expect(switcher.poll(308) == null);
    }
}

test "dist switch adapter: unavailable inference cannot orphan an activation" {
    var t: TestAdapter = undefined;
    try t.setup(false, 309);
    defer t.cleanup();
    switch_adapter.unbindInferenceThread();
    const f = t.sw.formation.?;
    try std.testing.expectError(error.Unavailable, f.start(f.ctx, .leader, 309, crt.wallMs() + 2000));
    try std.testing.expect(!TestSwitchOpen.entered.load(.acquire));
    const pid = std.c.fork();
    if (pid < 0) return error.TestForkFailed;
    if (pid == 0) {
        var link: Link = undefined;
        f.activate(f.ctx, 309, &link) catch std.c._exit(98);
        std.c._exit(99);
    }
    var status_code: c_int = 0;
    try std.testing.expectEqual(pid, std.c.waitpid(pid, &status_code, 0));
    try std.testing.expectEqual(@as(c_int, 1 << 8), status_code);
}

test "dist switch adapter: follower RELEASE completes an abort before the fallback" {
    var t: TestAdapter = undefined;
    try t.setup(true, 310);
    defer t.cleanup();
    const link = try t.form(true, 310);
    _ = t.sw.pump();
    _ = t.sw.command(.abort, 310, 0);
    try TestSwitchOpen.peer_bell.writeAll(&[_]u8{Bell.release_byte});
    try std.testing.expectEqual(OP_RELEASE, (try recvMsg(link, 0))[0]);
    const f = t.sw.formation.?;
    var rank: ?u32 = 99;
    try f.release(f.ctx, 310, link, &rank);
    try std.testing.expectEqual(@as(?u32, null), rank);
    try std.testing.expectEqual(.released, t.sw.snapshot().state);
    try std.testing.expect(switcher.poll(310) == null);
}

test "dist switch adapter: failed formation preserves error name rank and timing" {
    var t: TestAdapter = undefined;
    try t.setup(false, 311);
    defer t.cleanup();
    TestSwitchOpen.failure = error.DistNoReady;
    const f = t.sw.formation.?;
    try f.start(f.ctx, .leader, 311, crt.wallMs() + 80);
    const failed = (try TestSwitchOpen.outcome(311)).failed;
    const out = f.poll(f.ctx, 311).?.failed;
    try std.testing.expectEqualStrings("DistNoReady", out.err);
    try std.testing.expectEqual(@as(?u32, 1), out.rank);
    try std.testing.expectEqual(failed.elapsed_ms, out.elapsed_ms);
    try std.testing.expect(out.elapsed_ms >= 60);
    try std.testing.expectEqual(.already_finished, f.cancel(f.ctx, 311));
}

test "dist switch adapter: stall note follows the active watchdog" {
    var t: TestAdapter = undefined;
    try t.setup(false, 312);
    defer t.cleanup();
    var buf: [256]u8 = undefined;
    try std.testing.expectEqualStrings("", t.sw.stallNote(&buf));
    _ = try t.form(false, 312);
    _ = t.sw.pump();
    const note = t.sw.stallNote(&buf);
    try std.testing.expect(std.mem.indexOf(u8, note, "step=") != null);
    try std.testing.expect(std.mem.indexOf(u8, note, "waited_ms=") != null);
    parkInferenceLoop();
    try std.testing.expectEqualStrings("", t.sw.stallNote(&buf));
}

test "dist switch adapter: invalid configuration is refused before installation" {
    var sw = switch_controller.Switch.init(std.testing.io);
    for ([_]struct { size: u32, rank: u32 }{
        .{ .size = 0, .rank = 0 },
        .{ .size = 1, .rank = 0 },
        .{ .size = 999, .rank = 0 },
        .{ .size = 2, .rank = 2 },
    }) |bad| {
        var backend: ?*@import("../../../host/internal/formation.zig").Backend = null;
        const name = switch_adapter.install(&sw, .{ .coordinator = "127.0.0.1:46201", .device = "loopback", .model_dir = "/m/switch", .size = bad.size, .rank = bad.rank }, &backend);
        try std.testing.expectEqualStrings("InvalidArgs", name orelse "success");
        try std.testing.expect(backend == null);
    }
}

const TestModeAdapter = struct {
    base: TestAdapter = undefined,
    sch: @import("../../../host/gpu_thread/scheduler.zig").Scheduler = undefined,
    reg: *@import("../../../host/registry/registry.zig").ModelRegistry = undefined,
    entry: *@import("../../../host/registry/registry.zig").LoadedModel = undefined,
    xfm: @import("../transformer.zig").Transformer = undefined,
    whole: dsv41_mod.Dsv41Model = undefined,
    tensor: dsv41_mod.Dsv41Model = undefined,
    model: @import("../../../plugins/deepseek_v41/plugin.zig").Model = undefined,
    table: @import("../../../abi/plugin.zig").ModelTable = undefined,
    bound: @import("../plugin/driver.zig").Bound = undefined,
    swaps: usize = 0,
    fail: bool = false,
    var active: *@This() = undefined;

    fn setup(self: *@This(), follower: bool, n: u64) !void {
        try self.base.setup(follower, n);
        self.swaps = 0;
        self.fail = false;
        self.whole.dec_state = null;
        self.tensor.dec_state = null;
        self.tensor.cfg = .{ .hidden_size = 4 };
        self.model.mdl = &self.whole;
        self.model.modes = .{ .tp1 = &self.whole, .tp = &self.tensor };
        self.model.parked = .empty;
        self.model.idle = null;
        self.model.owner = 0;
        self.model.tp_ids = .{};
        self.table = @import("../../../plugins/deepseek_v41/plugin.zig").table;
        self.table.swap_mode = observe;
        self.bound = .{ .table = &self.table, .model = &self.model };
        self.xfm.plugin = &self.bound;
        self.reg = try @import("../../../host/registry/registry.zig").ModelRegistry.init(std.testing.allocator, std.testing.io, null, 1, 0, null);
        self.entry = try self.reg.registerStub("switch", "/m/switch", 0);
        self.entry.state = .ready;
        self.entry.transformer = &self.xfm;
        self.sch.registry = self.reg;
        self.sch.io = std.testing.io;
        active = self;
        switch_adapter.installSwapHook();
        switch_adapter.bindInferenceThread(&self.sch);
    }

    fn cleanup(self: *@This()) void {
        self.entry.transformer = null;
        self.entry.state = .unloaded;
        self.reg.deinit();
        self.base.cleanup();
    }

    fn observe(mp: *anyopaque, to: @import("../../../abi/plugin.zig").ServingMode) callconv(.c) @import("../../../abi/plugin.zig").Verdict {
        const t = active;
        std.debug.assert(cluster().tp_link != null);
        std.debug.assert(dsv41_mod.clusterHooks().tp_sum != null);
        std.debug.assert(dsv41_mod.clusterHooks().tp_gather != null);
        std.debug.assert((to == .tp) == (t.model.modes.active == .tp1));
        if (cluster().tp_link.?.rank == 0) std.debug.assert(dsv41_mod.clusterHooks().tp_step != null);
        t.swaps += 1;
        if (t.fail and to == .tp) return .refuse(.failed, "TestSwapFailed");
        return @import("../../../plugins/deepseek_v41/plugin.zig").table.swap_mode.?(mp, to);
    }

    fn expectClear(self: *@This()) !void {
        try std.testing.expectEqual(.tp1, self.model.modes.active);
        try std.testing.expect(cluster().tp_link == null);
        try std.testing.expectEqualDeep(@as(@TypeOf(dsv41_mod.clusterHooks().*), .{}), dsv41_mod.clusterHooks().*);
    }
};

test "dist switch mode: installed hook orders leader activate release and clears failed swaps" {
    for ([_]bool{ false, true }) |fail| {
        var t: TestModeAdapter = undefined;
        try t.setup(false, 401);
        defer t.cleanup();
        t.fail = fail;
        _ = try t.base.form(false, 401);
        var closer: TestReleasePeer = .{ .bell = TestSwitchOpen.peer_bell, .delay_ms = 0 };
        const th = try std.Thread.spawn(.{}, TestReleasePeer.run, .{&closer});
        defer th.join();
        _ = t.base.sw.pump();
        if (fail) {
            try std.testing.expectEqual(.aborted, t.base.sw.snapshot().state);
            try std.testing.expectEqual(@as(usize, 1), t.swaps);
        } else {
            try std.testing.expectEqual(.tp, t.model.modes.active);
            try std.testing.expectEqual(@as(u16, 200), t.base.sw.command(.tp1, 401, 0).status);
            try std.testing.expectEqual(@as(usize, 2), t.swaps);
            try std.testing.expectEqual(.released, t.base.sw.snapshot().state);
        }
        try t.expectClear();
    }
}

test "dist switch higher abort: leader releases the running link and clears TP mode before publishing released" {
    var t: TestModeAdapter = undefined;
    try t.setup(false, 5);
    defer t.cleanup();
    _ = try t.base.form(false, 5);
    _ = t.base.sw.pump();
    try std.testing.expectEqual(.tp, t.model.modes.active);
    var closer: TestReleasePeer = .{ .bell = TestSwitchOpen.peer_bell, .delay_ms = 20, .sw = &t.base.sw };
    const th = try std.Thread.spawn(.{}, TestReleasePeer.run, .{&closer});
    const reply = t.base.sw.command(.abort, 7, 0);
    th.join();
    try std.testing.expectEqual(@as(u16, 200), reply.status);
    try std.testing.expectEqual(@as(?anyerror, null), closer.result);
    try std.testing.expectEqual(.releasing, closer.before_eof.?.state);
    try std.testing.expectEqual(@as(u64, 5), closer.before_eof.?.n);
    try std.testing.expectEqual(@as(u64, 7), closer.before_eof.?.high_water);
    try std.testing.expect(!closer.early_eof);
    try std.testing.expectEqual(@as(usize, 2), t.swaps);
    try std.testing.expectEqual(.released, reply.snap.state);
    try std.testing.expectEqual(@as(u64, 7), reply.snap.high_water);
    try std.testing.expect(t.base.sw.link == null);
    try std.testing.expect(switcher.poll(5) == null);
    try std.testing.expect(!switchHolds("/m/switch"));
    try t.expectClear();
    try std.testing.expectEqualStrings("stale", t.base.sw.command(.join, 6, 0).reason);
}

test "dist switch higher abort: follower arms the two second fall for the running attempt" {
    var t: TestModeAdapter = undefined;
    try t.setup(true, 5);
    defer t.cleanup();
    const link = try t.base.form(true, 5);
    _ = t.base.sw.pump();
    const began = crt.monotonicMs();
    const reply = t.base.sw.command(.abort, 7, 0);
    try std.testing.expectEqual(@as(u16, 200), reply.status);
    try std.testing.expect(link.state.?.aborting.load(.acquire));
    try std.testing.expectEqual(.aborting, reply.snap.state);
    try std.testing.expectEqual(@as(u64, 5), reply.snap.n);
    try std.testing.expectEqual(@as(u64, 7), reply.snap.high_water);
    try std.testing.expectEqual(.tp, t.model.modes.active);
    try std.testing.expect(switch_adapter.serveFollower());
    const elapsed = crt.monotonicMs() - began;
    try std.testing.expect(elapsed >= 1900 and elapsed < 4000);
    const done = t.base.sw.snapshot();
    try std.testing.expectEqual(.aborted, done.state);
    try std.testing.expectEqual(.release_timeout, done.failure);
    try std.testing.expect(!done.quarantined);
    try std.testing.expectEqual(@as(u64, 7), done.high_water);
    try std.testing.expectEqual(@as(usize, 2), t.swaps);
    try std.testing.expectEqual(@as(u32, 0), TestSwitchOpen.exits.count.load(.acquire));
    try std.testing.expect(t.base.sw.link == null);
    try std.testing.expect(switcher.poll(5) == null);
    try std.testing.expect(!switchHolds("/m/switch"));
    try t.expectClear();
}

test "dist switch mode: leader pipes through the formed link without a cold Peer" {
    var t: TestModeAdapter = undefined;
    try t.setup(false, 402);
    defer t.cleanup();
    _ = try t.base.form(false, 402);
    _ = t.base.sw.pump();
    try std.testing.expect(peer() == null);
    cluster().tp_live = .{ .gen = 7, .n = 2 };
    try std.testing.expect(try dsv41_mod.clusterHooks().tp_pipe.?(7, 2));
    const other: Link = .{ .group = .{}, .rank = 1, .size = 2, .bells = blk: {
        var bells: [max_ranks]Bell = @splat(.{});
        bells[0] = TestSwitchOpen.peer_bell;
        break :blk bells;
    } };
    try std.testing.expectEqual(OP_DSV41_TP_PIPE, (try recvMsg(&other, 0))[0]);
    var closer: TestReleasePeer = .{ .bell = TestSwitchOpen.peer_bell, .delay_ms = 0 };
    const th = try std.Thread.spawn(.{}, TestReleasePeer.run, .{&closer});
    defer th.join();
    _ = t.base.sw.command(.tp1, 402, 0);
    try t.expectClear();
}

test "dist switch mode: follower serves RELEASE and abort fallback through the shared loop" {
    for ([_]bool{ false, true }) |abort| {
        var t: TestModeAdapter = undefined;
        try t.setup(true, 403);
        defer t.cleanup();
        _ = try t.base.form(true, 403);
        _ = t.base.sw.pump();
        try std.testing.expectEqual(.tp, t.model.modes.active);
        if (abort) {
            _ = t.base.sw.command(.abort, 403, 0);
        } else {
            try TestSwitchOpen.peer_bell.say(&.{ Bell.heartbeat_byte, Bell.release_byte });
        }
        try std.testing.expect(switch_adapter.serveFollower());
        try std.testing.expectEqual(if (abort) @as(@TypeOf(t.base.sw.snapshot().failure), .release_timeout) else .none, t.base.sw.snapshot().failure);
        try std.testing.expectEqual(if (abort) @as(@TypeOf(t.base.sw.snapshot().state), .aborted) else .released, t.base.sw.snapshot().state);
        try std.testing.expectEqual(@as(usize, 2), t.swaps);
        try std.testing.expect(t.base.sw.link == null);
        try std.testing.expect(switcher.poll(403) == null);
        try t.expectClear();
    }
}

test "dist switch mode: stale fall cannot restore a newer active mode" {
    var t: TestModeAdapter = undefined;
    try t.setup(true, 404);
    defer t.cleanup();
    const link = try t.base.form(true, 404);
    _ = t.base.sw.pump();
    const f = t.base.sw.formation.?;
    f.fall(f.ctx, 403, link);
    try std.testing.expectEqual(.tp, t.model.modes.active);
    try std.testing.expectEqual(link, cluster().tp_link.?);
    try TestSwitchOpen.peer_bell.say(&.{Bell.release_byte});
    try std.testing.expect(switch_adapter.serveFollower());
    try t.expectClear();
}

const TestWorkerHeaders = struct {
    fn receive(ctx: *const anyopaque) !Msg {
        const link: *const Link = @ptrCast(@alignCast(ctx));
        var msg: Msg = undefined;
        try link.bells[0].readAll(std.mem.asBytes(&msg));
        return msg;
    }

    const vt = blk: {
        var v = WireOps.vt;
        v.recv_op = receive;
        break :blk v;
    };
};

const TestPrefillWire = struct {
    model: *dsv41_mod.Dsv41Model,
    capture: bool,
    messages: usize = 0,
    tensors: usize = 0,
    const gen: u64 = (@as(u64, 7) << 31) | 37;

    fn receive(ctx: *const anyopaque) !Msg {
        const self: *@This() = @ptrCast(@alignCast(@constCast(ctx)));
        const next = self.messages;
        self.messages += 1;
        if (next == 1 or next == 2) {
            const n: usize = if (next == 1) 25 else 50;
            try std.testing.expectEqual(gen, self.model.state_gen);
            const st = self.model.dec_state orelse return error.TestPrefillStateMissing;
            try std.testing.expectEqual(n, st.n);
            if (self.capture) try std.testing.expectEqual(n, (st.snap orelse return error.TestPrefillCaptureMissing).n);
        }
        return switch (next) {
            0, 1 => .{ 18, 37, 7, if (next == 0) 0 else 25, 0, if (next == 0) 25 else 50, 0, if (self.capture) 1 << 16 else 0 },
            2 => .{ OP_DSV41_TP_FREE, 37, 7, 0, 0, 0, 0, 0 },
            3 => blk: {
                try std.testing.expect(self.model.dec_state == null);
                break :blk .{ OP_RELEASE, 0, 0, 0, 0, 0, 0, 0 };
            },
            else => error.TestPrefillUnexpectedMessage,
        };
    }

    fn tensor(ctx: *const anyopaque, shape: []const c_int, dtype: mlx.mlx_dtype, src: u32) !mlx.mlx_array {
        const self: *@This() = @ptrCast(@alignCast(@constCast(ctx)));
        try std.testing.expectEqualSlices(c_int, &.{ 32, 64 }, shape);
        try std.testing.expectEqual(mlx.mlx_dtype.float32, dtype);
        try std.testing.expectEqual(@as(u32, 0), src);
        self.tensors += 1;
        var x = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(x);
        try mlx.check(mlx.mlx_zeros(&x, shape.ptr, shape.len, dtype, self.model.s));
        return x;
    }

    const vt = blk: {
        var v = WireOps.vt;
        v.recv_op = receive;
        v.recv_now = tensor;
        break :blk v;
    };
};

test "dist dsv41 leader prefill: the shared follower loop imports each chunk before FREE and RELEASE" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    for ([_]bool{ false, true }) |capture| {
        var m: dsv41_mod.Dsv41Model = undefined;
        m.gpa = std.testing.allocator;
        m.s = mlx.gpuStream();
        m.cfg = .{ .model_type = "deepseek_v41", .hidden_size = 4, .num_hidden_layers = 1, .head_dim = 64, .sliding_window = 16, .dsv4_index_head_dim = 32 };
        m.whole_cfg = m.cfg;
        m.n_layers = 1;
        m.n_mtp = 0;
        m.vocab = 128;
        m.act = .float32;
        m.tp = .{ .rank = 1, .size = 4 };
        m.head = .{ .w = .{ .ctx = null } };
        m.hash = null;
        m.granted_tg = null;
        m.topk_tg = 32;
        m.hc_tg = 32;
        m.ds_block = 0;
        m.span = 1;
        m.moe_rows = 1;
        m.ced_split = null;
        m.ced_min_skip = 0;
        m.qat_off = true;
        m.attn_f16 = false;
        m.dec_state = null;
        m.state_gen = 0;
        m.tp_drafted = null;
        m.ds_chain = null;
        defer if (m.dec_state) |*st| st.deinit();
        var state: LinkState = .{};
        const link: Link = .{ .state = &state, .group = .{ .ctx = null }, .rank = 1, .size = 4 };
        var supplied: TestPrefillWire = .{ .model = &m, .capture = capture };
        const transport: wire.Link = .{ .ctx = &supplied, .vt = &TestPrefillWire.vt };
        try std.testing.expect(try dsv41StepLoop(std.testing.io, std.testing.allocator, &link, &m, transport));
        try std.testing.expectEqual(@as(usize, 4), supplied.messages);
        try std.testing.expectEqual(@as(usize, 2), supplied.tensors);
    }
}

test "dist switch mode: follower executes FREE and RELEASE in the cold worker loop without weights" {
    var t: TestModeAdapter = undefined;
    try t.setup(true, 405);
    defer t.cleanup();
    const link = try t.base.form(true, 405);
    _ = t.base.sw.pump();
    t.tensor.dec_state = .{ .gpa = std.testing.allocator, .layers = &.{} };
    t.tensor.state_gen = 7;
    const free: Msg = .{ OP_DSV41_TP_FREE, 7, 0, 0, 0, 0, 0, 0 };
    const release_msg: Msg = .{ OP_RELEASE, 0, 0, 0, 0, 0, 0, 0 };
    try TestSwitchOpen.peer_bell.say(std.mem.asBytes(&free));
    try TestSwitchOpen.peer_bell.say(std.mem.asBytes(&release_msg));
    const transport: wire.Link = .{ .ctx = link, .vt = &TestWorkerHeaders.vt };
    try std.testing.expect(try dsv41StepLoop(std.testing.io, std.testing.allocator, link, &t.tensor, transport));
    try std.testing.expect(t.tensor.dec_state == null);
    const f = t.base.sw.formation.?;
    var rank: ?u32 = null;
    try f.release(f.ctx, 405, link, &rank);
    try std.testing.expectEqual(.released, t.base.sw.snapshot().state);
    try t.expectClear();
}

test "dist cold worker loop: STOP returns false and unknown messages keep their error" {
    for ([_]i32{ OP_STOP, -1 }) |op| {
        TestSwitchOpen.setup();
        defer TestSwitchOpen.cleanup();
        var req = TestSwitchOpen.request(406);
        req.role = .follower;
        req.cfg.rank = 1;
        try TestSwitchOpen.start(req);
        const link = (try TestSwitchOpen.outcome(406)).formed.link;
        var m: dsv41_mod.Dsv41Model = undefined;
        m.cfg = .{ .hidden_size = 4 };
        const msg: Msg = .{ op, 0, 0, 0, 0, 0, 0, 0 };
        try TestSwitchOpen.peer_bell.say(std.mem.asBytes(&msg));
        const transport: wire.Link = .{ .ctx = link, .vt = &TestWorkerHeaders.vt };
        const result = dsv41StepLoop(std.testing.io, std.testing.allocator, link, &m, transport);
        if (op == OP_STOP) try std.testing.expect(!try result) else try std.testing.expectError(error.DistBadMessage, result);
        try switcher.fall(406, link);
    }
}

test "dist switch mode: an abort before publication leaves the follower available for teardown" {
    var t: TestModeAdapter = undefined;
    try t.setup(true, 407);
    defer t.cleanup();
    const link = try t.base.form(true, 407);
    const f = t.base.sw.formation.?;
    try f.activate(f.ctx, 407, link);
    _ = t.base.sw.command(.abort, 407, 0);
    try std.testing.expectEqual(.tp1, t.base.sw.snapshot().mode);
    try TestSwitchOpen.peer_bell.say(&.{Bell.release_byte});
    try std.testing.expect(!switch_adapter.serveFollower());
    try std.testing.expectEqual(.tp, t.model.modes.active);
    f.fall(f.ctx, 407, link);
    try t.expectClear();
}

test "dist dsv41 worker: every TP4 follower receives control from rank zero" {
    for (1..4) |rank| {
        TestSwitchOpen.setup();
        defer TestSwitchOpen.cleanup();
        var req = TestSwitchOpen.request(408);
        req.role = .follower;
        req.cfg.rank = @intCast(rank);
        req.cfg.size = 4;
        req.cfg.device = switch (rank) {
            1 => "loopback,-,loopback,loopback",
            2 => "loopback,loopback,-,loopback",
            3 => "loopback,loopback,loopback,-",
            else => unreachable,
        };
        try TestSwitchOpen.start(req);
        const link = (try TestSwitchOpen.outcome(408)).formed.link;
        var m: dsv41_mod.Dsv41Model = undefined;
        m.cfg = .{ .hidden_size = 4 };
        try TestSwitchOpen.peer_bell.say(&.{Bell.release_byte});
        try std.testing.expect(try serveDsv41Steps(std.testing.io, std.testing.allocator, link, &m));
        try switcher.fall(408, link);
        try std.testing.expectEqual(@as(u32, 0), TestSwitchOpen.exits.count.load(.acquire));
    }
}
