//! One box's layout-switch controller (plan 61c items 5 and 11): the attempt machine behind a mutex, the public
//! admission count it gates, and the interface the formation code (mlx's step 4) plugs in behind.

const std = @import("std");
const attempt = @import("attempt.zig");
const io_util = @import("../../sdk/util/io_util.zig");
const log = @import("../../sdk/util/log.zig");
const status = @import("../status.zig");

/// The formation deadline, past `t1`, a router gives when it names none (plan 61c item 4: t1 + 1,900 ms).
pub const default_formation_ms: i64 = 1900;
/// A held request whose progress sum has not moved this long makes `/health` 503 (the router's watchdog, 300 s prefilling).
pub const stall_limit_ms: i64 = 120_000;
pub const prefill_stall_limit_ms: i64 = 300_000;
const watch_interval_ms: i64 = 5;

pub const StartError = error{ ListenerTimeout, Unavailable, Busy, InvalidArgs, DistEnvNotPrimed };
/// The box could not swap to its TP model on the formed group; the group is released.
pub const ActivateError = error{SwapFailed};
/// `release` also sets the rank it could not reach.
pub const ReleaseError = error{ReleaseSendFailed};
pub const CancelResult = enum { cancelled, already_formed, already_finished, unknown };

/// A formed group: the formation code's link (opaque here) and how long its open and its READY-to-last-GO took.
pub const Link = struct { ptr: *anyopaque, open_ms: u32 = 0, ready_to_go_ms: u32 = 0 };
/// `err` is the error's name; `rank` the peer a failure is attributed to, when known.
pub const Failed = struct { err: []const u8, rank: ?u32 = null, elapsed_ms: u32 = 0 };
pub const FormOutcome = union(enum) { formed: Link, failed: Failed, cancelled, quarantined };

/// What the group code behind `tp4 N`, `join N`, `tp1 N` and `abort N` provides (mlx's `cluster/switch.zig`, step 4,
/// behind an adapter). Every call runs on an HTTP thread or the attempt's watcher, never under the controller's lock.
/// `activate`, `release` and `fall` touch the group on the inference thread in step 4: the adapter marshals each there
/// and returns when it is done.
/// The outcome of a formation is POLLED (`poll`); `releasing` and `released` come back as events, from whatever thread
/// read OP_RELEASE or finished the swap to TP1.
pub const Formation = struct {
    ctx: *anyopaque,
    /// Begin forming the group as `role`. `deadline_epoch_ms` is the router's wall-clock deadline, shared by every rank
    /// (the formation code converts it to its own clock). A leader returns once its listener is open.
    start: *const fn (ctx: *anyopaque, role: attempt.Role, n: u64, deadline_epoch_ms: i64) StartError!void,
    /// Ask an in-progress formation to unwind. `already_formed` and `already_finished`: it ended first, and `poll` will say so.
    cancel: *const fn (ctx: *anyopaque, n: u64) CancelResult,
    /// This attempt's outcome once its thread has returned; null while it runs or for another N.
    poll: *const fn (ctx: *anyopaque, n: u64) ?FormOutcome,
    /// A formed group, before it is published: swap the box to its TP model, then tell the link it is live. Blocks until
    /// the inference thread has done both. On an error the adapter has released or fallen the group.
    activate: *const fn (ctx: *anyopaque, n: u64, link: *anyopaque) ActivateError!void,
    /// A follower is told to abort a formed group: it keeps serving until OP_RELEASE, and the formation code falls by
    /// itself after its own 2 s window, then reports through `formationFailed(n, .release_timeout)`.
    abort_formed: *const fn (ctx: *anyopaque, n: u64) void,
    /// Send OP_RELEASE as leader, or tear down after OP_RELEASE as follower. A send failure names its rank.
    release: *const fn (ctx: *anyopaque, n: u64, link: *anyopaque, rank: *?u32) ReleaseError!void,
    /// A follower leaves the group by itself (no OP_RELEASE came): bell shutdown, join the watchers, free the link.
    fall: *const fn (ctx: *anyopaque, n: u64, link: *anyopaque) void,
    /// A formation thread outlived its unwind limit: the process is poisoned.
    quarantined: *const fn (ctx: *anyopaque) bool,
    /// Log the cause and `_exit(1)` after a failed release send.
    exit_process: *const fn (ctx: *anyopaque, n: u64) void,
    peers: *const fn (ctx: *anyopaque) u8,
    /// The stalled op, the last completed collective and the peer waited on, for the abort's log line.
    stall_note: *const fn (ctx: *anyopaque, buf: []u8) []const u8,
};

fn failureOf(err: []const u8) attempt.Failure {
    const table = [_]struct { name: []const u8, why: attempt.Failure }{
        .{ .name = "DistWorkerNotJoined", .why = .join },
        .{ .name = "DistNoReady", .why = .ready },
        .{ .name = "DistReadyTimeout", .why = .ready },
        .{ .name = "DistNoGo", .why = .go },
        .{ .name = "DistAborted", .why = .go },
        .{ .name = "DistPartialGo", .why = .go },
    };
    for (table) |row| if (std.mem.eql(u8, err, row.name)) return row.why;
    return .other;
}

/// The sum the router's watchdog uses (generation + prefill, total + live) and whether a prefill is running.
pub const Progress = struct { sum: u64, prefilling: bool };
pub const ProgressFn = *const fn () Progress;

pub const Health = enum { ok, quarantined, aborting, stalled };

pub const Snapshot = struct {
    mode: attempt.Mode,
    n: u64,
    high_water: u64,
    role: attempt.Role,
    state: attempt.State,
    failure: attempt.Failure,
    failure_rank: ?u32,
    peers: u8,
    in_flight: u32,
    admitting: bool,
    quarantined: bool,
    progress: ?u64,
    stalled_ms: i64,
    stalled: bool,
};

pub const Reply = struct {
    status: u16,
    /// A static tag; empty on success.
    reason: []const u8 = "",
    snap: Snapshot,
};

const Stall = struct {
    sum: u64 = 0,
    since_ms: i64 = 0,
    tracking: bool = false,

    fn observe(self: *Stall, now_ms: i64, in_flight: u32, sum: u64) i64 {
        if (in_flight == 0) {
            self.tracking = false;
            return 0;
        }
        if (!self.tracking or sum != self.sum) {
            self.tracking = true;
            self.sum = sum;
            self.since_ms = now_ms;
            return 0;
        }
        return now_ms - self.since_ms;
    }
};

pub const Switch = struct {
    io: std.Io,
    mu: std.Io.Mutex = .init,
    m: attempt.Machine = .{},
    formation: ?Formation = null,
    progress: ?ProgressFn = null,
    now_fn: *const fn (std.Io) i64 = io_util.nowMsMonotonic,
    wall_fn: *const fn (std.Io) i64 = io_util.nowMs,
    in_flight: u32 = 0,
    stall: Stall = .{},
    /// The formed group's link, from `formed` until a release or a fall consumes it.
    link: ?*anyopaque = null,
    failure_rank: ?u32 = null,
    watching: bool = false,
    spawn_watcher: bool = true,

    pub fn init(io: std.Io) Switch {
        return .{ .io = io };
    }

    fn now(self: *Switch) i64 {
        return self.now_fn(self.io);
    }

    fn lock(self: *Switch) void {
        self.mu.lockUncancelable(self.io);
    }

    fn unlock(self: *Switch) void {
        self.mu.unlock(self.io);
    }

    fn snapshotLocked(self: *Switch, now_ms: i64) Snapshot {
        self.m.tick(now_ms);
        var progress: ?u64 = null;
        var stalled_ms: i64 = 0;
        var limit = stall_limit_ms;
        if (self.progress) |p| {
            const pr = p();
            progress = pr.sum;
            stalled_ms = self.stall.observe(now_ms, self.in_flight, pr.sum);
            if (pr.prefilling) limit = prefill_stall_limit_ms;
        }
        return .{
            .mode = self.m.mode(),
            .n = self.m.n,
            .high_water = self.m.highWater(),
            .role = self.m.role,
            .state = self.m.state,
            .failure = self.m.failure,
            .failure_rank = self.failure_rank,
            .peers = if (self.formation) |f| f.peers(f.ctx) else 0,
            .in_flight = self.in_flight,
            .admitting = self.m.admits(),
            .quarantined = self.m.quarantined,
            .progress = progress,
            .stalled_ms = stalled_ms,
            .stalled = stalled_ms >= limit,
        };
    }

    pub fn snapshot(self: *Switch) Snapshot {
        self.lock();
        defer self.unlock();
        return self.snapshotLocked(self.now());
    }

    pub fn health(self: *Switch) Health {
        const s = self.snapshot();
        if (s.quarantined) return .quarantined;
        if (s.state == .aborting) return .aborting;
        if (s.stalled) return .stalled;
        return .ok;
    }

    /// Admit one public inference request, or refuse it while an attempt moves the box or it follows a leader.
    pub fn enter(self: *Switch) bool {
        self.lock();
        defer self.unlock();
        self.m.tick(self.now());
        if (!self.m.admits()) return false;
        self.in_flight += 1;
        return true;
    }

    pub fn leave(self: *Switch) void {
        self.lock();
        defer self.unlock();
        self.in_flight -|= 1;
    }

    /// True once no admitted request remains while attempt `n` is still draining, waiting at most `wait_ms`. An attempt
    /// that was aborted, fenced or expired meanwhile is not drained, however empty the count.
    pub fn waitDrained(self: *Switch, n: u64, wait_ms: i64) bool {
        const deadline = io_util.nowMsMonotonic(self.io) + wait_ms;
        while (true) {
            self.lock();
            self.m.tick(self.now());
            const draining = self.m.n == n and self.m.state == .draining;
            const idle = self.in_flight == 0;
            self.unlock();
            if (!draining) return false;
            if (idle) return true;
            if (io_util.nowMsMonotonic(self.io) >= deadline) return false;
            std.Io.sleep(self.io, .fromMilliseconds(5), .real) catch {};
        }
    }

    /// Apply one router command. `deadline_epoch_ms` is the router's wall-clock formation deadline for `tp4` and
    /// `join`; 0 means `default_formation_ms` from now.
    pub fn command(self: *Switch, cmd: attempt.Cmd, n: u64, deadline_epoch_ms: i64) Reply {
        const now_ms = self.now();
        const needs_formation = cmd == .tp4 or cmd == .join or cmd == .tp1;
        self.lock();
        if (needs_formation and self.formation == null) {
            const snap = self.snapshotLocked(now_ms);
            self.unlock();
            return .{ .status = 501, .reason = "no_formation", .snap = snap };
        }
        const out = self.m.command(cmd, n, now_ms);
        if (out == .ok and out.ok == .start_formation) self.failure_rank = null;
        const role = self.m.role;
        const snap = self.snapshotLocked(now_ms);
        self.unlock();
        log.info("[switch] {s} n={d} -> {s} state={s} n_box={d}\n", .{ @tagName(cmd), n, outcomeName(out), @tagName(snap.state), snap.n });
        return switch (out) {
            .refused => |why| .{ .status = 409, .reason = @tagName(why), .snap = snap },
            .noop => .{ .status = 200, .snap = snap },
            .ok => |effect| self.perform(cmd, effect, role, snap.n, if (deadline_epoch_ms == 0) self.wall_fn(self.io) + default_formation_ms else deadline_epoch_ms, snap),
        };
    }

    fn perform(self: *Switch, cmd: attempt.Cmd, effect: attempt.Effect, role: attempt.Role, n: u64, deadline_epoch_ms: i64, snap: Snapshot) Reply {
        const f = self.formation orelse return .{ .status = 200, .snap = snap };
        if (cmd == .abort and effect != .none and effect != .reopen_admission) self.logAbort(f, n);
        switch (effect) {
            .none, .close_admission, .reopen_admission => {},
            .start_formation => {
                f.start(f.ctx, role, n, deadline_epoch_ms) catch |err| {
                    const why: attempt.Failure = if (err == error.ListenerTimeout) .listener else .other;
                    _ = self.formationFailed(n, why);
                    return .{ .status = 502, .reason = if (err == error.DistEnvNotPrimed) "dist_env_not_primed" else @tagName(why), .snap = self.snapshot() };
                };
                self.watch();
            },
            .cancel_formation => if (f.cancel(f.ctx, n) == .unknown) {
                _ = self.formationFailed(n, .cancelled);
            } else self.watch(),
            .await_release => {
                f.abort_formed(f.ctx, n);
                self.watch();
            },
            .begin_release => return self.beginRelease(f, n),
        }
        return .{ .status = 200, .snap = self.snapshot() };
    }

    fn beginRelease(self: *Switch, f: Formation, n: u64) Reply {
        self.lock();
        const link = self.link;
        self.link = null;
        self.unlock();
        var rank: ?u32 = null;
        const sent: ReleaseError!void = if (link) |l| f.release(f.ctx, n, l, &rank) else error.ReleaseSendFailed;
        sent catch {
            self.lock();
            _ = self.m.releaseFailed(n, self.now());
            self.failure_rank = rank;
            self.unlock();
            log.err("[switch] release n={d} failed to reach rank {d}; exiting\n", .{ n, rank orelse 0 });
            f.exit_process(f.ctx, n);
            return .{ .status = 500, .reason = "release_failed", .snap = self.snapshot() };
        };
        return .{ .status = 200, .snap = self.snapshot() };
    }

    fn logAbort(self: *Switch, f: Formation, n: u64) void {
        _ = self;
        var buf: [256]u8 = undefined;
        log.warn("[switch] abort n={d} stall: {s}\n", .{ n, f.stall_note(f.ctx, &buf) });
    }

    pub fn stallNote(self: *Switch, buf: []u8) []const u8 {
        const f = self.formation orelse return "";
        return f.stall_note(f.ctx, buf);
    }

    /// One look at the formation thread's result and at a follower's wait for OP_RELEASE. True while the attempt still
    /// needs watching. The watcher thread calls it every few ms; tests call it by hand.
    pub fn pump(self: *Switch) bool {
        const f = self.formation orelse return false;
        self.lock();
        self.m.tick(self.now());
        const n = self.m.n;
        const state = self.m.state;
        const polls = state == .forming or (state == .aborting and !self.m.awaiting_release and self.link == null);
        const gone = self.m.quarantined;
        self.unlock();
        if (gone) return false;
        if (state != .forming and state != .aborting) return false;
        if (f.quarantined(f.ctx)) {
            self.lock();
            self.m.quarantined = true;
            self.unlock();
            log.err("[switch] n={d} formation thread outlived its unwind limit; box quarantined\n", .{n});
            return false;
        }
        if (polls) if (f.poll(f.ctx, n)) |out| self.apply(f, n, out);
        return true;
    }

    fn apply(self: *Switch, f: Formation, n: u64, out: FormOutcome) void {
        switch (out) {
            .formed => |l| self.onFormed(f, n, l),
            .failed => |x| {
                self.lock();
                if (n == self.m.n) self.failure_rank = x.rank;
                self.unlock();
                log.warn("[switch] formation n={d} failed: {s} rank={d} after {d} ms\n", .{ n, x.err, x.rank orelse 0, x.elapsed_ms });
                _ = self.formationFailed(n, failureOf(x.err));
            },
            .cancelled => _ = self.formationFailed(n, .cancelled),
            .quarantined => {
                self.lock();
                self.m.quarantined = true;
                self.unlock();
            },
        }
    }

    fn onFormed(self: *Switch, f: Formation, n: u64, l: Link) void {
        self.lock();
        const live = n == self.m.n and self.m.state == .forming and !self.m.quarantined;
        self.unlock();
        var swap_failed = false;
        if (live) {
            f.activate(f.ctx, n, l.ptr) catch {
                log.err("[switch] n={d}: the swap to the TP model failed; releasing the group\n", .{n});
                swap_failed = true;
            };
        }
        self.lock();
        const live_before = self.m.state;
        const role = self.m.role;
        const same_n = n == self.m.n;
        const ok = !swap_failed and self.m.formed(n, self.now());
        if (ok) self.link = l.ptr;
        const quarantined = self.m.quarantined;
        self.unlock();
        log.info("[switch] formed n={d} published={} open_ms={d} ready_to_go_ms={d}\n", .{ n, ok, l.open_ms, l.ready_to_go_ms });
        if (swap_failed) {
            _ = self.formationFailed(n, .other);
            return;
        }
        if (ok or quarantined) return;
        const aborting = same_n and live_before == .aborting;
        if (role == .leader and aborting) {
            var rank: ?u32 = null;
            f.release(f.ctx, n, l.ptr, &rank) catch {
                log.err("[switch] release n={d} failed to reach rank {d}; exiting\n", .{ n, rank orelse 0 });
                f.exit_process(f.ctx, n);
            };
        } else f.fall(f.ctx, n, l.ptr);
        if (aborting) _ = self.formationFailed(n, .cancelled);
    }

    /// Start the attempt's watcher unless one runs. It stops once no attempt is forming or waiting out an abort.
    fn watch(self: *Switch) void {
        if (!self.spawn_watcher) return;
        self.lock();
        if (self.watching) {
            self.unlock();
            return;
        }
        self.watching = true;
        self.unlock();
        const th = std.Thread.spawn(.{}, watchLoop, .{self}) catch {
            self.lock();
            self.watching = false;
            self.unlock();
            log.err("[switch] cannot start the formation watcher; the unwind limit will quarantine the box\n", .{});
            return;
        };
        th.detach();
    }

    fn watchLoop(self: *Switch) void {
        while (true) {
            while (self.pump()) std.Io.sleep(self.io, .fromMilliseconds(watch_interval_ms), .real) catch {};
            self.lock();
            const more = !self.m.quarantined and (self.m.state == .forming or self.m.state == .aborting);
            if (!more) self.watching = false;
            self.unlock();
            if (!more) return;
        }
    }

    /// The group code's own report that the formation thread finished (an alternative to waiting for `pump`).
    pub fn formationFailed(self: *Switch, n: u64, why: attempt.Failure) bool {
        self.lock();
        defer self.unlock();
        const ok = self.m.formationFailed(n, self.now(), why);
        if (ok) self.link = null;
        log.warn("[switch] formation n={d} ended: {s} published={}\n", .{ n, @tagName(why), ok });
        return ok;
    }

    /// A follower read OP_RELEASE.
    pub fn releasing(self: *Switch, n: u64) bool {
        self.lock();
        defer self.unlock();
        return self.m.releasing(n, self.now());
    }

    /// The link is freed, MLX's cached group cleared and the TP1 model swap_failed back in.
    pub fn released(self: *Switch, n: u64) bool {
        self.lock();
        defer self.unlock();
        const ok = self.m.released(n, self.now());
        if (ok) self.link = null;
        log.info("[switch] released n={d} published={}\n", .{ n, ok });
        return ok;
    }
};

fn outcomeName(o: attempt.Outcome) []const u8 {
    return switch (o) {
        .ok => |e| @tagName(e),
        .noop => "noop",
        .refused => |r| @tagName(r),
    };
}

/// The poll body the router reads each second. OS readings are sampled here on the HTTP thread.
pub fn renderSnapshot(io: std.Io, buf: []u8, s: Snapshot) []const u8 {
    const room_bytes = status.getFreeInactiveMemBytes();
    return renderWith(buf, s, io_util.nowMs(io), if (room_bytes) |b| b >> 20 else null);
}

/// `room_mib` null means the OS query failed; the router then skips a step-up on memory.
fn renderWith(buf: []u8, s: Snapshot, clock_ms: i64, room_mib: ?u64) []const u8 {
    var room_buf: [24]u8 = undefined;
    const room = if (room_mib) |r| std.fmt.bufPrint(&room_buf, "{d}", .{r}) catch "null" else "null";
    var progress_buf: [24]u8 = undefined;
    const progress = if (s.progress) |p| std.fmt.bufPrint(&progress_buf, "{d}", .{p}) catch "null" else "null";
    var rank_buf: [16]u8 = undefined;
    const rank = if (s.failure_rank) |r| std.fmt.bufPrint(&rank_buf, "{d}", .{r}) catch "null" else "null";
    return std.fmt.bufPrint(buf, "{{\"mode\":\"{s}\",\"n\":{d},\"high_water\":{d},\"role\":\"{s}\",\"state\":\"{s}\",\"failure\":\"{s}\",\"failure_rank\":{s},\"peers\":{d}," ++
        "\"in_flight\":{d},\"admitting\":{},\"quarantined\":{},\"progress\":{s},\"stalled_ms\":{d},\"clock_ms\":{d},\"room_mib\":{s}}}", .{
        @tagName(s.mode), s.n,         s.high_water,  @tagName(s.role), @tagName(s.state), @tagName(s.failure), rank, s.peers,
        s.in_flight,      s.admitting, s.quarantined, progress,         s.stalled_ms,      clock_ms,            room,
    }) catch "{}";
}

const t = std.testing;

test "a failed memory query renders room_mib as null, never as a zero reading" {
    var buf: [512]u8 = undefined;
    const snap: Snapshot = .{ .mode = .tp1, .n = 0, .high_water = 0, .role = .none, .state = .idle, .failure = .none, .failure_rank = null, .peers = 0, .in_flight = 0, .admitting = true, .quarantined = false, .progress = null, .stalled_ms = 0, .stalled = false };
    const body = renderWith(&buf, snap, 1, null);
    try t.expect(std.mem.indexOf(u8, body, "\"room_mib\":null") != null);
    const ok = renderWith(&buf, snap, 1, 7);
    try t.expect(std.mem.indexOf(u8, ok, "\"room_mib\":7}") != null);
}

const Fake = struct {
    calls: std.ArrayList(u8) = .empty,
    start_err: ?StartError = null,
    release_err: ?ReleaseError = null,
    cancel_result: CancelResult = .cancelled,
    outcome: ?FormOutcome = null,
    poisoned: bool = false,
    started_role: attempt.Role = .none,
    started_deadline: i64 = 0,
    handed_link: ?*anyopaque = null,
    slot: u8 = 0,
    activate_err: ?ActivateError = null,
    sw: ?*Switch = null,
    state_at_activate: attempt.State = .idle,
    aborted_n: u64 = 0,

    fn deinit(self: *Fake) void {
        self.calls.deinit(t.allocator);
    }

    fn note(self: *Fake, c: u8) void {
        self.calls.append(t.allocator, c) catch unreachable;
    }

    fn formedLink(self: *Fake) FormOutcome {
        return .{ .formed = .{ .ptr = &self.slot, .open_ms = 140, .ready_to_go_ms = 12 } };
    }
};

var test_now: i64 = 0;
fn testNow(_: std.Io) i64 {
    return test_now;
}
const test_wall: i64 = 1_700_000_000_000;
fn testWall(_: std.Io) i64 {
    return test_wall;
}

fn fake(ctx: *anyopaque) *Fake {
    return @ptrCast(@alignCast(ctx));
}
fn fakeStart(ctx: *anyopaque, role: attempt.Role, _: u64, deadline_epoch_ms: i64) StartError!void {
    const f = fake(ctx);
    f.note('s');
    f.started_role = role;
    f.started_deadline = deadline_epoch_ms;
    if (f.start_err) |e| return e;
}
fn fakeCancel(ctx: *anyopaque, _: u64) CancelResult {
    fake(ctx).note('c');
    return fake(ctx).cancel_result;
}
fn fakePoll(ctx: *anyopaque, _: u64) ?FormOutcome {
    const f = fake(ctx);
    const out = f.outcome;
    f.outcome = null;
    return out;
}
fn fakeActivate(ctx: *anyopaque, n: u64, link: *anyopaque) ActivateError!void {
    const f = fake(ctx);
    f.note('a');
    f.handed_link = link;
    if (f.sw) |sw| f.state_at_activate = sw.snapshot().state;
    if (f.activate_err) |e| {
        if (f.started_role == .leader) {
            var rank: ?u32 = null;
            fakeRelease(ctx, n, link, &rank) catch fakeExit(ctx, n);
        } else fakeFall(ctx, n, link);
        return e;
    }
}
fn fakeAbortFormed(ctx: *anyopaque, n: u64) void {
    fake(ctx).note('b');
    fake(ctx).aborted_n = n;
}
fn fakeRelease(ctx: *anyopaque, _: u64, link: *anyopaque, rank: *?u32) ReleaseError!void {
    const f = fake(ctx);
    f.note('r');
    f.handed_link = link;
    if (f.release_err) |e| {
        rank.* = 2;
        return e;
    }
}
fn fakeFall(ctx: *anyopaque, _: u64, link: *anyopaque) void {
    fake(ctx).note('f');
    fake(ctx).handed_link = link;
}
fn fakeQuarantined(ctx: *anyopaque) bool {
    return fake(ctx).poisoned;
}
fn fakeExit(ctx: *anyopaque, _: u64) void {
    fake(ctx).note('x');
}
fn fakePeers(_: *anyopaque) u8 {
    return 3;
}
fn fakeNote(_: *anyopaque, buf: []u8) []const u8 {
    return std.fmt.bufPrint(buf, "op=all_sum peer=2", .{}) catch "";
}

fn fakeFormation(f: *Fake) Formation {
    return .{
        .ctx = f,
        .start = fakeStart,
        .cancel = fakeCancel,
        .poll = fakePoll,
        .activate = fakeActivate,
        .abort_formed = fakeAbortFormed,
        .release = fakeRelease,
        .fall = fakeFall,
        .quarantined = fakeQuarantined,
        .exit_process = fakeExit,
        .peers = fakePeers,
        .stall_note = fakeNote,
    };
}

fn newSwitch(f: ?*Fake) Switch {
    test_now = 10_000;
    var sw = Switch.init(t.io);
    sw.now_fn = testNow;
    sw.wall_fn = testWall;
    sw.spawn_watcher = false;
    if (f) |x| sw.formation = fakeFormation(x);
    return sw;
}

fn formedAs(sw: *Switch, f: *Fake, role: attempt.Role, n: u64) !void {
    _ = sw.command(.drain, n, 0);
    _ = sw.command(if (role == .leader) .tp4 else .join, n, 0);
    f.outcome = f.formedLink();
    try t.expect(sw.pump());
    try t.expectEqual(attempt.State.formed, sw.snapshot().state);
}

test "admission: counts entries while open, refuses while draining, reopens on abort" {
    var sw = newSwitch(null);
    try t.expect(sw.enter());
    try t.expect(sw.enter());
    try t.expectEqual(@as(u32, 2), sw.snapshot().in_flight);
    try t.expectEqual(@as(u16, 200), sw.command(.drain, 1, 0).status);
    try t.expect(!sw.enter());
    sw.leave();
    sw.leave();
    try t.expectEqual(@as(u32, 0), sw.snapshot().in_flight);
    try t.expectEqual(@as(u16, 200), sw.command(.abort, 1, 0).status);
    try t.expect(sw.enter());
    sw.leave();
}

test "drain waits for the local in-flight count to reach zero" {
    var sw = newSwitch(null);
    try t.expect(sw.enter());
    _ = sw.command(.drain, 1, 0);
    try t.expect(!sw.waitDrained(1, 20));
    sw.leave();
    try t.expect(sw.waitDrained(1, 20));
}

test "a drain whose attempt was aborted is not drained once the count reaches zero" {
    var sw = newSwitch(null);
    try t.expect(sw.enter());
    _ = sw.command(.drain, 1, 0);
    _ = sw.command(.abort, 1, 0);
    sw.leave();
    try t.expect(!sw.waitDrained(1, 20));
}

test "a drain is drained only for the attempt that is draining" {
    var sw = newSwitch(null);
    _ = sw.command(.drain, 1, 0);
    _ = sw.command(.abort, 1, 0);
    _ = sw.command(.drain, 2, 0);
    try t.expect(!sw.waitDrained(1, 20));
    try t.expect(sw.waitDrained(2, 20));
}

test "a drain aborted while it waits answers false without running out the wait" {
    var sw = newSwitch(null);
    try t.expect(sw.enter());
    _ = sw.command(.drain, 1, 0);
    const Aborter = struct {
        fn run(s: *Switch) void {
            std.Io.sleep(s.io, .fromMilliseconds(30), .real) catch {};
            _ = s.command(.abort, 1, 0);
            s.leave();
        }
    };
    const th = try std.Thread.spawn(.{}, Aborter.run, .{&sw});
    defer th.join();
    try t.expect(!sw.waitDrained(1, 2000));
}

test "tp4 and join need a formation backend and leave the machine untouched without one" {
    var sw = newSwitch(null);
    _ = sw.command(.drain, 1, 0);
    const r = sw.command(.tp4, 1, 0);
    try t.expectEqual(@as(u16, 501), r.status);
    try t.expectEqualStrings("no_formation", r.reason);
    try t.expectEqual(attempt.State.draining, sw.snapshot().state);
}

test "tp4 hands the formation the router's wall-clock deadline untouched; a polled formed outcome publishes" {
    var f: Fake = .{};
    defer f.deinit();
    var sw = newSwitch(&f);
    _ = sw.command(.drain, 4, 0);
    const r = sw.command(.tp4, 4, test_wall + 1_900);
    try t.expectEqual(@as(u16, 200), r.status);
    try t.expectEqual(attempt.Role.leader, f.started_role);
    try t.expectEqual(test_wall + 1_900, f.started_deadline);
    try t.expectEqual(attempt.State.forming, sw.snapshot().state);
    try t.expect(sw.pump());
    try t.expectEqual(attempt.State.forming, sw.snapshot().state);
    f.outcome = f.formedLink();
    try t.expect(sw.pump());
    try t.expectEqual(attempt.Mode.tp4, sw.snapshot().mode);
    try t.expect(sw.snapshot().admitting);
    try t.expect(!sw.pump());
}

test "join with no deadline gets the wall clock plus the 1,900 ms default" {
    var f: Fake = .{};
    defer f.deinit();
    var sw = newSwitch(&f);
    _ = sw.command(.drain, 4, 0);
    _ = sw.command(.join, 4, 0);
    try t.expectEqual(attempt.Role.follower, f.started_role);
    try t.expectEqual(test_wall + 1_900, f.started_deadline);
    try t.expect(!sw.snapshot().admitting);
}

test "a start that fails lands the box at aborted with the reason, TP1 admitting" {
    var f: Fake = .{ .start_err = error.ListenerTimeout };
    defer f.deinit();
    var sw = newSwitch(&f);
    _ = sw.command(.drain, 4, 0);
    const r = sw.command(.tp4, 4, 0);
    try t.expectEqual(@as(u16, 502), r.status);
    try t.expectEqualStrings("listener", r.reason);
    const s = sw.snapshot();
    try t.expectEqual(attempt.State.aborted, s.state);
    try t.expectEqual(attempt.Failure.listener, s.failure);
    try t.expect(s.admitting);
}

test "a failed outcome lands at aborted with the formation's reason and the rank it names" {
    var f: Fake = .{};
    defer f.deinit();
    var sw = newSwitch(&f);
    _ = sw.command(.drain, 4, 0);
    _ = sw.command(.tp4, 4, 0);
    f.outcome = .{ .failed = .{ .err = "DistNoReady", .rank = 3, .elapsed_ms = 2002 } };
    try t.expect(sw.pump());
    const s = sw.snapshot();
    try t.expectEqual(attempt.State.aborted, s.state);
    try t.expectEqual(attempt.Failure.ready, s.failure);
    try t.expectEqual(@as(?u32, 3), s.failure_rank);
    try t.expect(s.admitting);
    var buf: [512]u8 = undefined;
    try t.expect(std.mem.indexOf(u8, renderSnapshot(t.io, &buf, s), "\"failure_rank\":3") != null);
}

test "an unknown error name is `other`; a cancelled outcome is `cancelled`" {
    try t.expectEqual(attempt.Failure.other, failureOf("SomethingNew"));
    try t.expectEqual(attempt.Failure.go, failureOf("DistAborted"));
    try t.expectEqual(attempt.Failure.join, failureOf("DistWorkerNotJoined"));
}

test "abort while forming cancels the formation; the cancelled outcome completes it" {
    var f: Fake = .{};
    defer f.deinit();
    var sw = newSwitch(&f);
    _ = sw.command(.drain, 4, 0);
    _ = sw.command(.join, 4, 0);
    const r = sw.command(.abort, 4, 0);
    try t.expectEqual(@as(u16, 200), r.status);
    try t.expectEqualStrings("sc", f.calls.items);
    try t.expectEqual(attempt.State.aborting, sw.snapshot().state);
    f.outcome = .cancelled;
    try t.expect(sw.pump());
    try t.expectEqual(attempt.State.aborted, sw.snapshot().state);
    try t.expectEqual(attempt.Failure.cancelled, sw.snapshot().failure);
    try t.expect(!sw.pump());
}

test "a cancel that finds no formation to unwind lands at aborted at once" {
    var f: Fake = .{ .cancel_result = .unknown };
    defer f.deinit();
    var sw = newSwitch(&f);
    _ = sw.command(.drain, 4, 0);
    _ = sw.command(.join, 4, 0);
    _ = sw.command(.abort, 4, 0);
    try t.expectEqual(attempt.State.aborted, sw.snapshot().state);
}

test "a formation that finished before the cancel: the leader releases the group it never published, a follower falls" {
    var f: Fake = .{ .cancel_result = .already_formed };
    defer f.deinit();
    var sw = newSwitch(&f);
    _ = sw.command(.drain, 4, 0);
    _ = sw.command(.tp4, 4, 0);
    _ = sw.command(.abort, 4, 0);
    f.outcome = f.formedLink();
    try t.expect(sw.pump());
    try t.expectEqualStrings("scr", f.calls.items);
    try t.expectEqual(@intFromPtr(&f.slot), @intFromPtr(f.handed_link.?));
    try t.expectEqual(attempt.State.aborted, sw.snapshot().state);

    var g: Fake = .{ .cancel_result = .already_formed };
    defer g.deinit();
    var sw2 = newSwitch(&g);
    _ = sw2.command(.drain, 4, 0);
    _ = sw2.command(.join, 4, 0);
    _ = sw2.command(.abort, 4, 0);
    g.outcome = g.formedLink();
    try t.expect(sw2.pump());
    try t.expectEqualStrings("scf", g.calls.items);
    try t.expectEqual(attempt.State.aborted, sw2.snapshot().state);
}

test "tp1 hands the link to the backend's release; a failed release exits, names the rank and keeps the box closed" {
    var f: Fake = .{};
    defer f.deinit();
    var sw = newSwitch(&f);
    try formedAs(&sw, &f, .leader, 4);
    try t.expectEqual(@as(u16, 200), sw.command(.tp1, 4, 0).status);
    try t.expectEqual(@intFromPtr(&f.slot), @intFromPtr(f.handed_link.?));
    try t.expectEqual(attempt.State.releasing, sw.snapshot().state);
    try t.expect(sw.released(4));
    try t.expectEqual(attempt.Mode.tp1, sw.snapshot().mode);
    try t.expectEqual(@as(u16, 200), sw.command(.tp1, 4, 0).status);
    try t.expectEqualStrings("sar", f.calls.items);

    var g: Fake = .{ .release_err = error.ReleaseSendFailed };
    defer g.deinit();
    var sw2 = newSwitch(&g);
    try formedAs(&sw2, &g, .leader, 1);
    const r = sw2.command(.tp1, 1, 0);
    try t.expectEqual(@as(u16, 500), r.status);
    try t.expectEqualStrings("release_failed", r.reason);
    try t.expectEqualStrings("sarx", g.calls.items);
    try t.expectEqual(@as(?u32, 2), sw2.snapshot().failure_rank);
    try t.expect(!sw2.snapshot().admitting);
}

test "abort of a formed group at the leader is the planned down; a repeat of it changes nothing" {
    var f: Fake = .{};
    defer f.deinit();
    var sw = newSwitch(&f);
    try formedAs(&sw, &f, .leader, 4);
    try t.expectEqual(@as(u16, 200), sw.command(.abort, 4, 0).status);
    try t.expectEqualStrings("sar", f.calls.items);
    try t.expectEqual(attempt.State.releasing, sw.snapshot().state);
    try t.expectEqual(@as(u16, 200), sw.command(.abort, 4, 0).status);
    try t.expectEqualStrings("sar", f.calls.items);
    const fenced = sw.command(.abort, 6, 0);
    try t.expectEqual(@as(u16, 200), fenced.status);
    try t.expectEqual(attempt.State.releasing, fenced.snap.state);
    try t.expectEqual(@as(u64, 4), fenced.snap.n);
    try t.expectEqual(@as(u64, 6), fenced.snap.high_water);
}

test "abort of a formed group at a follower hands it to the formation code and serves it until OP_RELEASE, then releases" {
    var f: Fake = .{};
    defer f.deinit();
    var sw = newSwitch(&f);
    try formedAs(&sw, &f, .follower, 4);
    _ = sw.command(.abort, 4, 0);
    try t.expectEqualStrings("sab", f.calls.items);
    try t.expectEqual(@as(u64, 4), f.aborted_n);
    try t.expectEqual(attempt.Mode.tp4, sw.snapshot().mode);
    try t.expect(sw.pump());
    test_now += 1999;
    try t.expect(sw.pump());
    try t.expectEqualStrings("sab", f.calls.items);
    try t.expect(sw.releasing(4));
    try t.expect(sw.released(4));
    test_now += 5000;
    try t.expect(!sw.pump());
    try t.expectEqualStrings("sab", f.calls.items);
    try t.expectEqual(attempt.State.released, sw.snapshot().state);
}

test "a repeat of the follower's abort does not hand the group over twice" {
    var f: Fake = .{};
    defer f.deinit();
    var sw = newSwitch(&f);
    try formedAs(&sw, &f, .follower, 4);
    _ = sw.command(.abort, 4, 0);
    _ = sw.command(.abort, 4, 0);
    try t.expectEqualStrings("sab", f.calls.items);
}

test "the follower's 2 s fall belongs to the formation code: it reports it and the box lands at aborted in TP1" {
    var f: Fake = .{};
    defer f.deinit();
    var sw = newSwitch(&f);
    try formedAs(&sw, &f, .follower, 4);
    _ = sw.command(.abort, 4, 0);
    test_now += 2000;
    try t.expect(sw.pump());
    try t.expectEqualStrings("sab", f.calls.items);
    try t.expectEqual(attempt.State.aborting, sw.snapshot().state);
    try t.expect(sw.formationFailed(4, .release_timeout));
    const s = sw.snapshot();
    try t.expectEqual(attempt.State.aborted, s.state);
    try t.expectEqual(attempt.Failure.release_timeout, s.failure);
    try t.expectEqual(attempt.Mode.tp1, s.mode);
    try t.expect(s.admitting);
    try t.expect(!sw.pump());
}

test "a follower's abort the formation code never reports quarantines the box" {
    var f: Fake = .{};
    defer f.deinit();
    var sw = newSwitch(&f);
    try formedAs(&sw, &f, .follower, 4);
    _ = sw.command(.abort, 4, 0);
    test_now += 4500;
    try t.expectEqual(Health.quarantined, sw.health());
    try t.expect(!sw.pump());
    try t.expectEqualStrings("sab", f.calls.items);
}

test "a formed group is activated, with its link, before it is published" {
    var f: Fake = .{};
    defer f.deinit();
    var sw = newSwitch(&f);
    f.sw = &sw;
    try formedAs(&sw, &f, .leader, 4);
    try t.expectEqualStrings("sa", f.calls.items);
    try t.expectEqual(@intFromPtr(&f.slot), @intFromPtr(f.handed_link.?));
    try t.expectEqual(attempt.State.forming, f.state_at_activate);
    try t.expectEqual(attempt.Mode.tp4, sw.snapshot().mode);
}

test "a swap that fails releases the leader's group and lands at aborted in TP1" {
    var f: Fake = .{ .activate_err = error.SwapFailed };
    defer f.deinit();
    var sw = newSwitch(&f);
    _ = sw.command(.drain, 4, 0);
    _ = sw.command(.tp4, 4, 0);
    f.outcome = f.formedLink();
    try t.expect(sw.pump());
    try t.expectEqualStrings("sar", f.calls.items);
    const s = sw.snapshot();
    try t.expectEqual(attempt.State.aborted, s.state);
    try t.expectEqual(attempt.Failure.other, s.failure);
    try t.expectEqual(attempt.Mode.tp1, s.mode);
    try t.expect(s.admitting);
}

test "a swap that fails makes a follower fall" {
    var f: Fake = .{ .activate_err = error.SwapFailed };
    defer f.deinit();
    var sw = newSwitch(&f);
    _ = sw.command(.drain, 4, 0);
    _ = sw.command(.join, 4, 0);
    f.outcome = f.formedLink();
    try t.expect(sw.pump());
    try t.expectEqualStrings("saf", f.calls.items);
    try t.expectEqual(attempt.State.aborted, sw.snapshot().state);
}

test "a leader whose release fails after a failed swap exits" {
    var f: Fake = .{ .activate_err = error.SwapFailed, .release_err = error.ReleaseSendFailed };
    defer f.deinit();
    var sw = newSwitch(&f);
    _ = sw.command(.drain, 4, 0);
    _ = sw.command(.tp4, 4, 0);
    f.outcome = f.formedLink();
    _ = sw.pump();
    try t.expectEqualStrings("sarx", f.calls.items);
}

test "a start before the environment is primed is a 502 that names it and lands at aborted" {
    var f: Fake = .{ .start_err = error.DistEnvNotPrimed };
    defer f.deinit();
    var sw = newSwitch(&f);
    _ = sw.command(.drain, 4, 0);
    const r = sw.command(.tp4, 4, 0);
    try t.expectEqual(@as(u16, 502), r.status);
    try t.expectEqualStrings("dist_env_not_primed", r.reason);
    try t.expectEqual(attempt.State.aborted, r.snap.state);
    try t.expectEqual(attempt.Failure.other, r.snap.failure);
}

test "a cancel that finds the formation already finished lets the polled outcome end the attempt" {
    var f: Fake = .{ .cancel_result = .already_finished };
    defer f.deinit();
    var sw = newSwitch(&f);
    _ = sw.command(.drain, 4, 0);
    _ = sw.command(.join, 4, 0);
    _ = sw.command(.abort, 4, 0);
    try t.expectEqual(attempt.State.aborting, sw.snapshot().state);
    f.outcome = .{ .failed = .{ .err = "DistNoReady", .rank = 1, .elapsed_ms = 1900 } };
    try t.expect(sw.pump());
    try t.expectEqual(attempt.State.aborted, sw.snapshot().state);
    try t.expectEqualStrings("sc", f.calls.items);
}

test "a refusal is a 409 carrying the box's N so the router can adopt it" {
    var sw = newSwitch(null);
    _ = sw.command(.drain, 7, 0);
    const r = sw.command(.drain, 3, 0);
    try t.expectEqual(@as(u16, 409), r.status);
    try t.expectEqualStrings("stale", r.reason);
    try t.expectEqual(@as(u64, 7), r.snap.n);
}

test "a drain nobody follows expires into aborted on the next observation" {
    var sw = newSwitch(null);
    _ = sw.command(.drain, 2, 0);
    try t.expect(!sw.enter());
    test_now += 3500;
    try t.expect(sw.enter());
    sw.leave();
    try t.expectEqual(attempt.State.aborted, sw.snapshot().state);
}

test "quarantine: a formation that never unwinds closes admission and turns /health 503" {
    var f: Fake = .{};
    defer f.deinit();
    var sw = newSwitch(&f);
    _ = sw.command(.drain, 2, 0);
    _ = sw.command(.tp4, 2, 0);
    try t.expectEqual(Health.ok, sw.health());
    test_now += 2500;
    try t.expectEqual(Health.quarantined, sw.health());
    try t.expect(!sw.enter());
    try t.expectEqual(@as(u16, 409), sw.command(.abort, 2, 0).status);
    try t.expect(!sw.pump());
}

test "the formation code's own poison flag quarantines the box on the next pump" {
    var f: Fake = .{};
    defer f.deinit();
    var sw = newSwitch(&f);
    _ = sw.command(.drain, 2, 0);
    _ = sw.command(.tp4, 2, 0);
    f.poisoned = true;
    try t.expect(!sw.pump());
    try t.expectEqual(Health.quarantined, sw.health());
    f.outcome = f.formedLink();
    try t.expect(!sw.pump());
    try t.expectEqualStrings("s", f.calls.items);
}

test "a quarantined outcome from poll quarantines the box" {
    var f: Fake = .{};
    defer f.deinit();
    var sw = newSwitch(&f);
    _ = sw.command(.drain, 2, 0);
    _ = sw.command(.tp4, 2, 0);
    f.outcome = .quarantined;
    _ = sw.pump();
    try t.expectEqual(Health.quarantined, sw.health());
}

test "health is 503 while aborting" {
    var f: Fake = .{};
    defer f.deinit();
    var sw = newSwitch(&f);
    _ = sw.command(.drain, 2, 0);
    _ = sw.command(.tp4, 2, 0);
    _ = sw.command(.abort, 2, 0);
    try t.expectEqual(Health.aborting, sw.health());
    f.outcome = .cancelled;
    _ = sw.pump();
    try t.expectEqual(Health.ok, sw.health());
}

var test_progress: Progress = .{ .sum = 0, .prefilling = false };
fn testProgress() Progress {
    return test_progress;
}

test "health is 503 when a held request makes no progress past the watchdog, 300 s while prefilling" {
    var sw = newSwitch(null);
    sw.progress = testProgress;
    test_progress = .{ .sum = 100, .prefilling = false };
    try t.expect(sw.enter());
    try t.expectEqual(Health.ok, sw.health());
    test_now += 119_999;
    try t.expectEqual(Health.ok, sw.health());
    test_now += 1;
    try t.expectEqual(Health.stalled, sw.health());
    test_progress.sum = 101;
    try t.expectEqual(Health.ok, sw.health());
    test_progress.prefilling = true;
    test_now += 299_999;
    try t.expectEqual(Health.ok, sw.health());
    test_now += 1;
    try t.expectEqual(Health.stalled, sw.health());
    sw.leave();
    try t.expectEqual(Health.ok, sw.health());
}

test "the mode snapshot carries mode, N, role, peers and the stall note an abort logs" {
    var f: Fake = .{};
    defer f.deinit();
    var sw = newSwitch(&f);
    try formedAs(&sw, &f, .leader, 6);
    var buf: [512]u8 = undefined;
    const body = renderSnapshot(t.io, &buf, sw.snapshot());
    const parsed = try std.json.parseFromSlice(std.json.Value, t.allocator, body, .{});
    defer parsed.deinit();
    const o = parsed.value.object;
    try t.expectEqualStrings("tp4", o.get("mode").?.string);
    try t.expectEqual(@as(i64, 6), o.get("n").?.integer);
    try t.expectEqual(@as(i64, 6), o.get("high_water").?.integer);
    try t.expectEqualStrings("leader", o.get("role").?.string);
    try t.expectEqualStrings("formed", o.get("state").?.string);
    try t.expectEqual(@as(i64, 3), o.get("peers").?.integer);
    try t.expect(o.get("progress").? == .null);
    try t.expect(o.get("failure_rank").? == .null);
    var nb: [64]u8 = undefined;
    try t.expectEqualStrings("op=all_sum peer=2", sw.stallNote(&nb));
}

test "internal switch: stale release completion preserves the current group's link" {
    var f: Fake = .{};
    defer f.deinit();
    var sw = newSwitch(&f);
    try formedAs(&sw, &f, .leader, 4);
    const link = sw.link;
    try t.expect(!sw.released(3));
    try t.expectEqual(link, sw.link);
    try t.expectEqual(attempt.State.formed, sw.snapshot().state);
}
