//! One box's layout-switch controller (plan 61c items 5 and 11): the attempt machine behind a mutex, the public
//! admission count it gates, and the interface the formation code (mlx's step 4) plugs in behind.

const std = @import("std");
const attempt = @import("attempt.zig");
const io_util = @import("../../sdk/util/io_util.zig");
const log = @import("../../sdk/util/log.zig");

/// The formation deadline a router gives when it names none.
pub const default_formation_ms: i64 = 2000;
/// A held request whose progress sum has not moved this long makes `/health` 503 (the router's watchdog, 300 s prefilling).
pub const stall_limit_ms: i64 = 120_000;
pub const prefill_stall_limit_ms: i64 = 300_000;

pub const StartError = error{ ListenerTimeout, Unavailable };
pub const ReleaseError = error{PeerUnreachable};

/// What the group code behind `tp4 N`, `join N`, `tp1 N` and `abort N` provides. Every call runs on an HTTP thread and
/// never under the controller's lock; the group code reports back through `formed`, `formationFailed`, `releasing`
/// and `released`, from whatever thread finished the work.
pub const Formation = struct {
    ctx: *anyopaque,
    /// Begin forming the group as `role`, absolute CLOCK_MONOTONIC `deadline_ms`. A leader returns once the JACCL
    /// listener is open; a follower returns as soon as its switch thread is running.
    start: *const fn (ctx: *anyopaque, role: attempt.Role, n: u64, deadline_ms: i64) StartError!void,
    /// Cancel an in-progress formation; the switch thread unwinds and calls `formationFailed`.
    cancel: *const fn (ctx: *anyopaque, n: u64) void,
    /// Leader only: send OP_RELEASE and begin the down-switch; `released` reports its end.
    release: *const fn (ctx: *anyopaque, n: u64) ReleaseError!void,
    /// Log the cause and `_exit(1)`: an abort of a formed group, or a failed release send.
    exit_process: *const fn (ctx: *anyopaque, n: u64) void,
    peers: *const fn (ctx: *anyopaque) u8,
    /// The stalled op, the last completed collective and the peer waited on, for the abort's log line.
    stall_note: *const fn (ctx: *anyopaque, buf: []u8) []const u8,
};

/// The sum the router's watchdog uses (generation + prefill, total + live) and whether a prefill is running.
pub const Progress = struct { sum: u64, prefilling: bool };
pub const ProgressFn = *const fn () Progress;

pub const Health = enum { ok, quarantined, aborting, stalled };

pub const Snapshot = struct {
    mode: attempt.Mode,
    n: u64,
    role: attempt.Role,
    state: attempt.State,
    failure: attempt.Failure,
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
    in_flight: u32 = 0,
    stall: Stall = .{},

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
            .role = self.m.role,
            .state = self.m.state,
            .failure = self.m.failure,
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

    /// True once no admitted request remains, waiting at most `wait_ms`.
    pub fn waitDrained(self: *Switch, wait_ms: i64) bool {
        const deadline = io_util.nowMsMonotonic(self.io) + wait_ms;
        while (true) {
            self.lock();
            const idle = self.in_flight == 0;
            self.unlock();
            if (idle) return true;
            if (io_util.nowMsMonotonic(self.io) >= deadline) return false;
            std.Io.sleep(self.io, .fromMilliseconds(5), .real) catch {};
        }
    }

    /// Apply one router command. `deadline_ms` is an absolute CLOCK_MONOTONIC time for `tp4` and `join`; 0 means two seconds from now.
    pub fn command(self: *Switch, cmd: attempt.Cmd, n: u64, deadline_ms: i64) Reply {
        const now_ms = self.now();
        const needs_formation = cmd == .tp4 or cmd == .join or cmd == .tp1;
        self.lock();
        if (needs_formation and self.formation == null) {
            const snap = self.snapshotLocked(now_ms);
            self.unlock();
            return .{ .status = 501, .reason = "no_formation", .snap = snap };
        }
        const out = self.m.command(cmd, n, now_ms);
        const role = self.m.role;
        const snap = self.snapshotLocked(now_ms);
        self.unlock();
        log.info("[switch] {s} n={d} -> {s} state={s} n_box={d}\n", .{ @tagName(cmd), n, outcomeName(out), @tagName(snap.state), snap.n });
        return switch (out) {
            .refused => |why| .{ .status = 409, .reason = @tagName(why), .snap = snap },
            .noop => .{ .status = 200, .snap = snap },
            .ok => |effect| self.perform(effect, role, n, if (deadline_ms == 0) now_ms + default_formation_ms else deadline_ms, snap),
        };
    }

    fn perform(self: *Switch, effect: attempt.Effect, role: attempt.Role, n: u64, deadline_ms: i64, snap: Snapshot) Reply {
        const f = self.formation orelse return .{ .status = 200, .snap = snap };
        switch (effect) {
            .none, .close_admission, .reopen_admission => {},
            .start_formation => f.start(f.ctx, role, n, deadline_ms) catch |err| {
                const why: attempt.Failure = if (err == error.ListenerTimeout) .listener else .other;
                _ = self.formationFailed(n, why);
                return .{ .status = 502, .reason = @tagName(why), .snap = self.snapshot() };
            },
            .cancel_formation => {
                self.logAbort(f, n);
                f.cancel(f.ctx, n);
            },
            .exit_process => {
                self.logAbort(f, n);
                f.exit_process(f.ctx, n);
            },
            .begin_release => f.release(f.ctx, n) catch {
                self.lock();
                _ = self.m.releaseFailed(n, self.now());
                self.unlock();
                log.err("[switch] release n={d} failed to reach a follower; exiting\n", .{n});
                f.exit_process(f.ctx, n);
                return .{ .status = 500, .reason = "release_failed", .snap = self.snapshot() };
            },
        }
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

    /// The switch thread finished forming. False: this attempt is no longer live, so the thread publishes nothing.
    pub fn formed(self: *Switch, n: u64) bool {
        self.lock();
        defer self.unlock();
        const ok = self.m.formed(n, self.now());
        log.info("[switch] formed n={d} published={}\n", .{ n, ok });
        return ok;
    }

    pub fn formationFailed(self: *Switch, n: u64, why: attempt.Failure) bool {
        self.lock();
        defer self.unlock();
        const ok = self.m.formationFailed(n, self.now(), why);
        log.warn("[switch] formation n={d} ended: {s} published={}\n", .{ n, @tagName(why), ok });
        return ok;
    }

    /// A follower read OP_RELEASE.
    pub fn releasing(self: *Switch, n: u64) bool {
        self.lock();
        defer self.unlock();
        return self.m.releasing(n, self.now());
    }

    /// The link is freed, MLX's cached group cleared and the TP1 model swapped back in.
    pub fn released(self: *Switch, n: u64) bool {
        self.lock();
        defer self.unlock();
        const ok = self.m.released(n, self.now());
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

/// The poll body the router reads each second: `{mode, n, high_water, role, state, peers, ...}`.
pub fn renderSnapshot(buf: []u8, s: Snapshot) []const u8 {
    var progress_buf: [24]u8 = undefined;
    const progress = if (s.progress) |p| std.fmt.bufPrint(&progress_buf, "{d}", .{p}) catch "null" else "null";
    return std.fmt.bufPrint(buf, "{{\"mode\":\"{s}\",\"n\":{d},\"high_water\":{d},\"role\":\"{s}\",\"state\":\"{s}\",\"failure\":\"{s}\",\"peers\":{d}," ++
        "\"in_flight\":{d},\"admitting\":{},\"quarantined\":{},\"progress\":{s},\"stalled_ms\":{d}}}", .{
        @tagName(s.mode),        s.n,         s.n,           @tagName(s.role), @tagName(s.state), @tagName(s.failure), s.peers,
        s.in_flight,             s.admitting, s.quarantined, progress,         s.stalled_ms,
    }) catch "{}";
}

const t = std.testing;

const Fake = struct {
    calls: std.ArrayList(u8) = .empty,
    start_err: ?anyerror = null,
    release_err: ?anyerror = null,
    started_role: attempt.Role = .none,
    started_deadline: i64 = 0,
    sync_form: ?*Switch = null,

    fn deinit(self: *Fake) void {
        self.calls.deinit(t.allocator);
    }
};

var test_now: i64 = 0;
fn testNow(_: std.Io) i64 {
    return test_now;
}

fn fakeStart(ctx: *anyopaque, role: attempt.Role, n: u64, deadline_ms: i64) StartError!void {
    const f: *Fake = @ptrCast(@alignCast(ctx));
    f.calls.append(t.allocator, 's') catch unreachable;
    f.started_role = role;
    f.started_deadline = deadline_ms;
    if (f.start_err) |_| return error.ListenerTimeout;
    if (f.sync_form) |sw| _ = sw.formed(n);
}
fn fakeCancel(ctx: *anyopaque, _: u64) void {
    const f: *Fake = @ptrCast(@alignCast(ctx));
    f.calls.append(t.allocator, 'c') catch unreachable;
}
fn fakeRelease(ctx: *anyopaque, _: u64) ReleaseError!void {
    const f: *Fake = @ptrCast(@alignCast(ctx));
    f.calls.append(t.allocator, 'r') catch unreachable;
    if (f.release_err) |_| return error.PeerUnreachable;
}
fn fakeExit(ctx: *anyopaque, _: u64) void {
    const f: *Fake = @ptrCast(@alignCast(ctx));
    f.calls.append(t.allocator, 'x') catch unreachable;
}
fn fakePeers(_: *anyopaque) u8 {
    return 3;
}
fn fakeNote(_: *anyopaque, buf: []u8) []const u8 {
    return std.fmt.bufPrint(buf, "op=all_sum peer=2", .{}) catch "";
}

fn fakeFormation(f: *Fake) Formation {
    return .{ .ctx = f, .start = fakeStart, .cancel = fakeCancel, .release = fakeRelease, .exit_process = fakeExit, .peers = fakePeers, .stall_note = fakeNote };
}

fn newSwitch(f: ?*Fake) Switch {
    test_now = 10_000;
    var sw = Switch.init(t.io);
    sw.now_fn = testNow;
    if (f) |x| sw.formation = fakeFormation(x);
    return sw;
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
    try t.expect(!sw.waitDrained(20));
    sw.leave();
    try t.expect(sw.waitDrained(20));
}

test "tp4 and join need a formation backend and leave the machine untouched without one" {
    var sw = newSwitch(null);
    _ = sw.command(.drain, 1, 0);
    const r = sw.command(.tp4, 1, 0);
    try t.expectEqual(@as(u16, 501), r.status);
    try t.expectEqualStrings("no_formation", r.reason);
    try t.expectEqual(attempt.State.draining, sw.snapshot().state);
}

test "tp4 starts formation as leader with the router's deadline; the thread publishes through formed" {
    var f: Fake = .{};
    defer f.deinit();
    var sw = newSwitch(&f);
    _ = sw.command(.drain, 4, 0);
    const r = sw.command(.tp4, 4, 11_900);
    try t.expectEqual(@as(u16, 200), r.status);
    try t.expectEqual(attempt.Role.leader, f.started_role);
    try t.expectEqual(@as(i64, 11_900), f.started_deadline);
    try t.expectEqual(attempt.State.forming, sw.snapshot().state);
    try t.expect(sw.formed(4));
    try t.expectEqual(attempt.Mode.tp4, sw.snapshot().mode);
    try t.expect(sw.snapshot().admitting);
}

test "join with no deadline gets the default two seconds" {
    var f: Fake = .{};
    defer f.deinit();
    var sw = newSwitch(&f);
    _ = sw.command(.drain, 4, 0);
    _ = sw.command(.join, 4, 0);
    try t.expectEqual(attempt.Role.follower, f.started_role);
    try t.expectEqual(@as(i64, 12_000), f.started_deadline);
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

test "abort while forming cancels the formation; the thread's unwind completes it" {
    var f: Fake = .{};
    defer f.deinit();
    var sw = newSwitch(&f);
    _ = sw.command(.drain, 4, 0);
    _ = sw.command(.join, 4, 0);
    const r = sw.command(.abort, 4, 0);
    try t.expectEqual(@as(u16, 200), r.status);
    try t.expectEqualStrings("sc", f.calls.items);
    try t.expectEqual(attempt.State.aborting, sw.snapshot().state);
    try t.expect(!sw.formed(4));
    try t.expect(sw.formationFailed(4, .cancelled));
    try t.expectEqual(attempt.State.aborted, sw.snapshot().state);
}

test "abort of a formed group exits the process" {
    var f: Fake = .{};
    defer f.deinit();
    var sw = newSwitch(&f);
    _ = sw.command(.drain, 4, 0);
    _ = sw.command(.join, 4, 0);
    _ = sw.formed(4);
    _ = sw.command(.abort, 4, 0);
    try t.expectEqualStrings("sx", f.calls.items);
}

test "tp1 releases through the backend; a failed release exits and keeps the box closed" {
    var f: Fake = .{};
    defer f.deinit();
    var sw = newSwitch(&f);
    _ = sw.command(.drain, 4, 0);
    _ = sw.command(.tp4, 4, 0);
    _ = sw.formed(4);
    try t.expectEqual(@as(u16, 200), sw.command(.tp1, 4, 0).status);
    try t.expectEqual(attempt.State.releasing, sw.snapshot().state);
    try t.expect(sw.released(4));
    try t.expectEqual(attempt.Mode.tp1, sw.snapshot().mode);
    try t.expectEqual(@as(u16, 200), sw.command(.tp1, 4, 0).status);
    try t.expectEqualStrings("sr", f.calls.items);

    var g: Fake = .{ .release_err = error.PeerUnreachable };
    defer g.deinit();
    var sw2 = newSwitch(&g);
    _ = sw2.command(.drain, 1, 0);
    _ = sw2.command(.tp4, 1, 0);
    _ = sw2.formed(1);
    const r = sw2.command(.tp1, 1, 0);
    try t.expectEqual(@as(u16, 500), r.status);
    try t.expectEqualStrings("release_failed", r.reason);
    try t.expectEqualStrings("srx", g.calls.items);
    try t.expect(!sw2.snapshot().admitting);
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
}

test "health is 503 while aborting" {
    var f: Fake = .{};
    defer f.deinit();
    var sw = newSwitch(&f);
    _ = sw.command(.drain, 2, 0);
    _ = sw.command(.tp4, 2, 0);
    _ = sw.command(.abort, 2, 0);
    try t.expectEqual(Health.aborting, sw.health());
    _ = sw.formationFailed(2, .cancelled);
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
    _ = sw.command(.drain, 6, 0);
    _ = sw.command(.tp4, 6, 0);
    _ = sw.formed(6);
    var buf: [512]u8 = undefined;
    const body = renderSnapshot(&buf, sw.snapshot());
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
    var nb: [64]u8 = undefined;
    try t.expectEqualStrings("op=all_sum peer=2", sw.stallNote(&nb));
}
