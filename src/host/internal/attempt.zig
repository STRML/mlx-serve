//! The layout-switch attempt state table of one box (plan 61c item 5): pure, no clock, no I/O.
//! Every command in every state, repeats included, is a row of the table-driven test below.

const std = @import("std");

pub const State = enum { idle, draining, forming, formed, releasing, released, aborting, aborted };
pub const Role = enum { none, leader, follower };
pub const Mode = enum { tp1, tp4 };
pub const Cmd = enum { drain, tp4, join, tp1, abort };

/// Why a formation ended without a group; the router's switch log names it.
pub const Failure = enum { none, listener, join, ready, go, deadline, cancelled, drain_expired, release_timeout, other };

pub const Config = struct {
    /// A box that took "drain N" but saw no "tp4 N" or "join N" this long aborts N itself (1 s drain + 2 s formation + slack).
    drain_expiry_ms: i64 = 3500,
    /// A formation or an abort that has not unwound by now leaves the thread's resources untouched and quarantines the box.
    unwind_limit_ms: i64 = 2500,
    /// A follower told to abort a formed group serves it this long for the leader's OP_RELEASE, then falls by itself.
    release_wait_ms: i64 = 2000,
};

/// What the caller does after an accepted command.
pub const Effect = enum { none, close_admission, reopen_admission, start_formation, cancel_formation, begin_release, await_release, exit_process };

pub const Refusal = enum {
    /// Attempt 0, which no router issues.
    invalid,
    /// N is below the box's high-water: a delayed command of an attempt already fenced.
    stale,
    /// The command is not accepted in the box's state for this N.
    state,
    quarantined,
};

pub const Outcome = union(enum) {
    ok: Effect,
    /// An idempotent repeat: nothing changes.
    noop,
    refused: Refusal,
};

pub const Machine = struct {
    cfg: Config = .{},
    state: State = .idle,
    /// The latest attempt this box knows: its high-water, and the one every command is judged against.
    n: u64 = 0,
    role: Role = .none,
    entered_ms: i64 = 0,
    quarantined: bool = false,
    failure: Failure = .none,
    /// A follower aborting a formed group, still serving it until OP_RELEASE or `release_wait_ms`.
    awaiting_release: bool = false,
    fall_due: bool = false,
    /// Set from `formed` until the link is gone: a follower waiting out its abort still serves TP4.
    grouped: bool = false,

    pub fn command(self: *Machine, cmd: Cmd, n: u64, now_ms: i64) Outcome {
        self.tick(now_ms);
        if (self.quarantined) return .{ .refused = .quarantined };
        if (n == 0) return .{ .refused = .invalid };
        if (n < self.n) return .{ .refused = .stale };
        return switch (cmd) {
            .drain => self.drain(n, now_ms),
            .tp4 => self.form(.leader, n, now_ms),
            .join => self.form(.follower, n, now_ms),
            .tp1 => self.release(n, now_ms),
            .abort => self.abort(n, now_ms),
        };
    }

    fn atRest(self: *const Machine) bool {
        return switch (self.state) {
            .idle, .released, .aborted => true,
            else => false,
        };
    }

    fn enter(self: *Machine, state: State, now_ms: i64) void {
        self.state = state;
        self.entered_ms = now_ms;
    }

    fn drain(self: *Machine, n: u64, now_ms: i64) Outcome {
        if (n == self.n) return if (self.state == .draining) .noop else .{ .refused = .state };
        if (!self.atRest()) return .{ .refused = .state };
        self.n = n;
        self.role = .none;
        self.failure = .none;
        self.enter(.draining, now_ms);
        return .{ .ok = .close_admission };
    }

    fn form(self: *Machine, role: Role, n: u64, now_ms: i64) Outcome {
        if (n != self.n or self.state != .draining) return .{ .refused = .state };
        self.role = role;
        self.enter(.forming, now_ms);
        return .{ .ok = .start_formation };
    }

    fn release(self: *Machine, n: u64, now_ms: i64) Outcome {
        if (n != self.n) return .{ .refused = .state };
        return switch (self.state) {
            .formed => blk: {
                self.enter(.releasing, now_ms);
                break :blk .{ .ok = .begin_release };
            },
            .releasing, .released, .aborted => .noop,
            else => .{ .refused = .state },
        };
    }

    fn abort(self: *Machine, n: u64, now_ms: i64) Outcome {
        const out: Outcome = switch (self.state) {
            .draining => blk: {
                self.failure = .cancelled;
                self.enter(.aborted, now_ms);
                break :blk .{ .ok = .reopen_admission };
            },
            .forming => blk: {
                self.enter(.aborting, now_ms);
                break :blk .{ .ok = .cancel_formation };
            },
            .formed => if (self.role == .leader) blk: {
                self.enter(.releasing, now_ms);
                break :blk .{ .ok = .begin_release };
            } else blk: {
                self.awaiting_release = true;
                self.enter(.aborting, now_ms);
                break :blk .{ .ok = .await_release };
            },
            .releasing => blk: {
                self.enter(.aborting, now_ms);
                break :blk .{ .ok = .exit_process };
            },
            .aborting => .noop,
            .idle, .released, .aborted => if (n == self.n) .noop else blk: {
                self.enter(.aborted, now_ms);
                break :blk .{ .ok = .none };
            },
        };
        self.n = n;
        return out;
    }

    /// Time-driven transitions; every observation of the machine runs it first, so no timer thread exists.
    pub fn tick(self: *Machine, now_ms: i64) void {
        const waited = now_ms - self.entered_ms;
        switch (self.state) {
            .draining => if (waited >= self.cfg.drain_expiry_ms) {
                self.failure = .drain_expired;
                self.enter(.aborted, now_ms);
            },
            .aborting => if (self.awaiting_release) {
                if (waited >= self.cfg.release_wait_ms) {
                    self.awaiting_release = false;
                    self.fall_due = true;
                    self.enter(.aborting, now_ms);
                }
            } else if (waited >= self.cfg.unwind_limit_ms) {
                self.quarantined = true;
            },
            .forming => if (waited >= self.cfg.unwind_limit_ms) {
                self.quarantined = true;
            },
            else => {},
        }
    }

    /// The switch thread finished forming the group. False: this attempt is no longer live, publish nothing.
    pub fn formed(self: *Machine, n: u64, now_ms: i64) bool {
        if (self.quarantined or n != self.n or self.state != .forming) return false;
        self.grouped = true;
        self.enter(.formed, now_ms);
        return true;
    }

    /// The switch thread unwound without a group (a missed deadline, a missing peer, or the cancel of an abort).
    pub fn formationFailed(self: *Machine, n: u64, now_ms: i64, why: Failure) bool {
        if (self.quarantined or n != self.n) return false;
        if (self.state != .forming and self.state != .aborting) return false;
        self.failure = why;
        self.grouped = false;
        self.enter(.aborted, now_ms);
        return true;
    }

    /// A follower read OP_RELEASE, whether it was serving the group or waiting out an abort.
    pub fn releasing(self: *Machine, n: u64, now_ms: i64) bool {
        if (self.quarantined or n != self.n) return false;
        if (self.state != .formed and !(self.state == .aborting and self.awaiting_release)) return false;
        self.awaiting_release = false;
        self.enter(.releasing, now_ms);
        return true;
    }

    /// True once after a follower's wait for OP_RELEASE ran out: the caller falls, then reports `formationFailed(.release_timeout)`.
    pub fn takeFallDue(self: *Machine) bool {
        const due = self.fall_due;
        self.fall_due = false;
        return due;
    }

    /// The box freed its link, cleared MLX's cached group and swapped back to the TP1 model.
    pub fn released(self: *Machine, n: u64, now_ms: i64) bool {
        if (self.quarantined or n != self.n) return false;
        if (self.state != .formed and self.state != .releasing) return false;
        self.grouped = false;
        self.enter(.released, now_ms);
        return true;
    }

    /// `release()` failed to reach a follower: the leader exits, and until it does the box stays closed.
    pub fn releaseFailed(self: *Machine, n: u64, now_ms: i64) bool {
        if (self.quarantined or n != self.n or self.state != .releasing) return false;
        self.enter(.aborting, now_ms);
        return true;
    }

    /// Public inference is admitted in TP1 and by the TP4 leader; never while an attempt moves the box.
    pub fn admits(self: *const Machine) bool {
        if (self.quarantined) return false;
        return self.atRest() or (self.state == .formed and self.role == .leader);
    }

    pub fn mode(self: *const Machine) Mode {
        return switch (self.state) {
            .formed, .releasing => .tp4,
            .aborting => if (self.grouped) .tp4 else .tp1,
            else => .tp1,
        };
    }
};

const t = std.testing;

const Row = struct {
    /// Machine before: state, N, role.
    from: State,
    n: u64 = 5,
    role: Role = .none,
    cmd: Cmd,
    arg: u64,
    want: Outcome,
    /// State and N after (the high-water moves with N).
    to: State,
    to_n: u64 = 5,
};

const ok = struct {
    fn e(x: Effect) Outcome {
        return .{ .ok = x };
    }
};

const table = [_]Row{
    // drain
    .{ .from = .idle, .n = 0, .cmd = .drain, .arg = 1, .want = ok.e(.close_admission), .to = .draining, .to_n = 1 },
    .{ .from = .idle, .n = 0, .cmd = .drain, .arg = 0, .want = .{ .refused = .invalid }, .to = .idle, .to_n = 0 },
    .{ .from = .released, .cmd = .drain, .arg = 6, .want = ok.e(.close_admission), .to = .draining, .to_n = 6 },
    .{ .from = .aborted, .cmd = .drain, .arg = 6, .want = ok.e(.close_admission), .to = .draining, .to_n = 6 },
    .{ .from = .aborted, .cmd = .drain, .arg = 5, .want = .{ .refused = .state }, .to = .aborted },
    .{ .from = .aborted, .cmd = .drain, .arg = 4, .want = .{ .refused = .stale }, .to = .aborted },
    .{ .from = .draining, .cmd = .drain, .arg = 5, .want = .noop, .to = .draining },
    .{ .from = .draining, .cmd = .drain, .arg = 6, .want = .{ .refused = .state }, .to = .draining },
    .{ .from = .forming, .cmd = .drain, .arg = 5, .want = .{ .refused = .state }, .to = .forming },
    .{ .from = .forming, .cmd = .drain, .arg = 6, .want = .{ .refused = .state }, .to = .forming },
    .{ .from = .formed, .cmd = .drain, .arg = 6, .want = .{ .refused = .state }, .to = .formed },
    .{ .from = .releasing, .cmd = .drain, .arg = 6, .want = .{ .refused = .state }, .to = .releasing },
    .{ .from = .aborting, .cmd = .drain, .arg = 6, .want = .{ .refused = .state }, .to = .aborting },
    // tp4 (leader) and join (follower): only a draining box at the same N
    .{ .from = .draining, .cmd = .tp4, .arg = 5, .want = ok.e(.start_formation), .to = .forming },
    .{ .from = .draining, .cmd = .join, .arg = 5, .want = ok.e(.start_formation), .to = .forming },
    .{ .from = .draining, .cmd = .tp4, .arg = 4, .want = .{ .refused = .stale }, .to = .draining },
    .{ .from = .draining, .cmd = .join, .arg = 6, .want = .{ .refused = .state }, .to = .draining },
    .{ .from = .idle, .n = 0, .cmd = .join, .arg = 1, .want = .{ .refused = .state }, .to = .idle, .to_n = 0 },
    .{ .from = .forming, .cmd = .tp4, .arg = 5, .want = .{ .refused = .state }, .to = .forming },
    .{ .from = .forming, .cmd = .join, .arg = 5, .want = .{ .refused = .state }, .to = .forming },
    .{ .from = .formed, .cmd = .join, .arg = 5, .want = .{ .refused = .state }, .to = .formed },
    .{ .from = .aborted, .cmd = .join, .arg = 5, .want = .{ .refused = .state }, .to = .aborted },
    .{ .from = .aborting, .cmd = .tp4, .arg = 5, .want = .{ .refused = .state }, .to = .aborting },
    .{ .from = .released, .cmd = .join, .arg = 5, .want = .{ .refused = .state }, .to = .released },
    .{ .from = .releasing, .cmd = .tp4, .arg = 5, .want = .{ .refused = .state }, .to = .releasing },
    .{ .from = .idle, .n = 0, .cmd = .tp4, .arg = 1, .want = .{ .refused = .state }, .to = .idle, .to_n = 0 },
    .{ .from = .formed, .cmd = .tp4, .arg = 5, .want = .{ .refused = .state }, .to = .formed },
    .{ .from = .released, .cmd = .tp4, .arg = 5, .want = .{ .refused = .state }, .to = .released },
    .{ .from = .aborted, .cmd = .tp4, .arg = 5, .want = .{ .refused = .state }, .to = .aborted },
    .{ .from = .releasing, .cmd = .join, .arg = 5, .want = .{ .refused = .state }, .to = .releasing },
    .{ .from = .aborting, .cmd = .join, .arg = 5, .want = .{ .refused = .state }, .to = .aborting },
    // tp1
    .{ .from = .formed, .role = .leader, .cmd = .tp1, .arg = 5, .want = ok.e(.begin_release), .to = .releasing },
    .{ .from = .releasing, .cmd = .tp1, .arg = 5, .want = .noop, .to = .releasing },
    .{ .from = .released, .cmd = .tp1, .arg = 5, .want = .noop, .to = .released },
    .{ .from = .aborted, .cmd = .tp1, .arg = 5, .want = .noop, .to = .aborted },
    .{ .from = .formed, .cmd = .tp1, .arg = 4, .want = .{ .refused = .stale }, .to = .formed },
    .{ .from = .formed, .cmd = .tp1, .arg = 6, .want = .{ .refused = .state }, .to = .formed },
    .{ .from = .idle, .n = 0, .cmd = .tp1, .arg = 1, .want = .{ .refused = .state }, .to = .idle, .to_n = 0 },
    .{ .from = .draining, .cmd = .tp1, .arg = 5, .want = .{ .refused = .state }, .to = .draining },
    .{ .from = .forming, .cmd = .tp1, .arg = 5, .want = .{ .refused = .state }, .to = .forming },
    .{ .from = .aborting, .cmd = .tp1, .arg = 5, .want = .{ .refused = .state }, .to = .aborting },
    // abort
    .{ .from = .draining, .cmd = .abort, .arg = 5, .want = ok.e(.reopen_admission), .to = .aborted },
    .{ .from = .forming, .cmd = .abort, .arg = 5, .want = ok.e(.cancel_formation), .to = .aborting },
    .{ .from = .aborting, .cmd = .abort, .arg = 5, .want = .noop, .to = .aborting },
    .{ .from = .formed, .role = .leader, .cmd = .abort, .arg = 5, .want = ok.e(.begin_release), .to = .releasing },
    .{ .from = .formed, .role = .follower, .cmd = .abort, .arg = 5, .want = ok.e(.await_release), .to = .aborting },
    .{ .from = .releasing, .cmd = .abort, .arg = 5, .want = ok.e(.exit_process), .to = .aborting },
    .{ .from = .released, .cmd = .abort, .arg = 5, .want = .noop, .to = .released },
    .{ .from = .aborted, .cmd = .abort, .arg = 5, .want = .noop, .to = .aborted },
    .{ .from = .aborted, .cmd = .abort, .arg = 4, .want = .{ .refused = .stale }, .to = .aborted },
    .{ .from = .draining, .cmd = .abort, .arg = 4, .want = .{ .refused = .stale }, .to = .draining },
    // abort of an attempt the box never saw fences it
    .{ .from = .idle, .n = 0, .cmd = .abort, .arg = 7, .want = ok.e(.none), .to = .aborted, .to_n = 7 },
    .{ .from = .released, .cmd = .abort, .arg = 7, .want = ok.e(.none), .to = .aborted, .to_n = 7 },
    .{ .from = .aborted, .cmd = .abort, .arg = 7, .want = ok.e(.none), .to = .aborted, .to_n = 7 },
    .{ .from = .draining, .cmd = .abort, .arg = 7, .want = ok.e(.reopen_admission), .to = .aborted, .to_n = 7 },
    .{ .from = .forming, .cmd = .abort, .arg = 7, .want = ok.e(.cancel_formation), .to = .aborting, .to_n = 7 },
    .{ .from = .formed, .role = .leader, .cmd = .abort, .arg = 7, .want = ok.e(.begin_release), .to = .releasing, .to_n = 7 },
    .{ .from = .formed, .role = .follower, .cmd = .abort, .arg = 7, .want = ok.e(.await_release), .to = .aborting, .to_n = 7 },
};

fn machineIn(from: State, n: u64, role: Role) Machine {
    return .{ .state = from, .n = n, .role = role };
}

test "attempt table: every command in every state" {
    for (table, 0..) |row, i| {
        var m = machineIn(row.from, row.n, row.role);
        const got = m.command(row.cmd, row.arg, 1000);
        errdefer std.debug.print("row {d}: {s} {s} {d}\n", .{ i, @tagName(row.from), @tagName(row.cmd), row.arg });
        try t.expectEqualDeep(row.want, got);
        try t.expectEqual(row.to, m.state);
        try t.expectEqual(row.to_n, m.n);
    }
}

test "attempt table covers every command in every state" {
    for (std.enums.values(State)) |s| {
        for (std.enums.values(Cmd)) |c| {
            var seen = false;
            for (table) |row| {
                if (row.from == s and row.cmd == c) seen = true;
            }
            errdefer std.debug.print("no row: {s} {s}\n", .{ @tagName(s), @tagName(c) });
            try t.expect(seen);
        }
    }
}

test "a delayed join after an abort of the same N is refused by state" {
    var m: Machine = .{};
    _ = m.command(.drain, 3, 0);
    _ = m.command(.abort, 3, 10);
    try t.expectEqual(Outcome{ .refused = .state }, m.command(.join, 3, 20));
    try t.expectEqual(State.aborted, m.state);
}

test "a drain that arrives after its abort is fenced; drain first is expired by the box" {
    var late: Machine = .{};
    try t.expectEqual(Outcome{ .ok = .none }, late.command(.abort, 9, 0));
    try t.expectEqual(Outcome{ .refused = .stale }, late.command(.drain, 8, 5));
    try t.expectEqual(Outcome{ .refused = .state }, late.command(.drain, 9, 5));

    var m: Machine = .{};
    _ = m.command(.drain, 9, 0);
    m.tick(3499);
    try t.expectEqual(State.draining, m.state);
    try t.expect(!m.admits());
    m.tick(3500);
    try t.expectEqual(State.aborted, m.state);
    try t.expectEqual(@as(u64, 9), m.n);
    try t.expectEqual(Failure.drain_expired, m.failure);
    try t.expect(m.admits());
    try t.expectEqual(Outcome{ .refused = .state }, m.command(.join, 9, 3600));
}

test "formation: published only for the live attempt" {
    var m: Machine = .{};
    _ = m.command(.drain, 2, 0);
    _ = m.command(.tp4, 2, 10);
    try t.expect(!m.formed(1, 20));
    try t.expect(m.formed(2, 30));
    try t.expectEqual(State.formed, m.state);
    try t.expect(!m.formed(2, 40));
    try t.expectEqual(Mode.tp4, m.mode());
}

test "formation failure and cancel both land at aborted, TP1 admitting" {
    var m: Machine = .{};
    _ = m.command(.drain, 2, 0);
    _ = m.command(.join, 2, 10);
    try t.expect(!m.admits());
    try t.expect(m.formationFailed(2, 50, .deadline));
    try t.expectEqual(State.aborted, m.state);
    try t.expectEqual(Failure.deadline, m.failure);
    try t.expect(m.admits());
    try t.expectEqual(Mode.tp1, m.mode());

    var c: Machine = .{};
    _ = c.command(.drain, 3, 0);
    _ = c.command(.tp4, 3, 1);
    _ = c.command(.abort, 3, 2);
    try t.expectEqual(State.aborting, c.state);
    try t.expect(!c.admits());
    try t.expect(c.formationFailed(3, 9, .cancelled));
    try t.expectEqual(State.aborted, c.state);
}

test "a thread that does not unwind quarantines the box and freezes every command" {
    var m: Machine = .{};
    _ = m.command(.drain, 4, 0);
    _ = m.command(.tp4, 4, 1);
    m.tick(1 + 2499);
    try t.expect(!m.quarantined);
    m.tick(1 + 2500);
    try t.expect(m.quarantined);
    try t.expect(!m.admits());
    try t.expectEqual(Outcome{ .refused = .quarantined }, m.command(.abort, 4, 3000));
    try t.expect(!m.formationFailed(4, 3100, .deadline));
    try t.expectEqual(State.forming, m.state);

    var a: Machine = .{};
    _ = a.command(.drain, 4, 0);
    _ = a.command(.tp4, 4, 0);
    _ = a.command(.abort, 4, 2000);
    a.tick(2000 + 2500);
    try t.expect(a.quarantined);
}

test "a leader serves in TP4 only when formed; a follower never" {
    var l: Machine = .{};
    _ = l.command(.drain, 1, 0);
    _ = l.command(.tp4, 1, 0);
    try t.expect(l.formed(1, 5));
    try t.expect(l.admits());
    var f: Machine = .{};
    _ = f.command(.drain, 1, 0);
    _ = f.command(.join, 1, 0);
    try t.expect(f.formed(1, 5));
    try t.expect(!f.admits());
    try t.expectEqual(Mode.tp4, f.mode());
}

test "release: follower sees OP_RELEASE then reports released; both sides end at TP1 admitting" {
    var f: Machine = .{};
    _ = f.command(.drain, 1, 0);
    _ = f.command(.join, 1, 0);
    _ = f.formed(1, 5);
    try t.expect(f.releasing(1, 10));
    try t.expectEqual(State.releasing, f.state);
    try t.expect(!f.admits());
    try t.expect(f.released(1, 20));
    try t.expectEqual(State.released, f.state);
    try t.expect(f.admits());
    try t.expectEqual(Mode.tp1, f.mode());
    try t.expect(!f.released(1, 30));
    try t.expect(!f.releasing(2, 30));

    var l: Machine = .{};
    _ = l.command(.drain, 1, 0);
    _ = l.command(.tp4, 1, 0);
    _ = l.formed(1, 5);
    try t.expectEqual(Outcome{ .ok = .begin_release }, l.command(.tp1, 1, 10));
    try t.expect(l.released(1, 20));
    try t.expectEqual(State.released, l.state);
    try t.expectEqual(Outcome{ .ok = .close_admission }, l.command(.drain, 2, 30));
}

test "a failed release send leaves the box aborting, closed" {
    var l: Machine = .{};
    _ = l.command(.drain, 1, 0);
    _ = l.command(.tp4, 1, 0);
    _ = l.formed(1, 5);
    _ = l.command(.tp1, 1, 10);
    try t.expect(l.releaseFailed(1, 11));
    try t.expectEqual(State.aborting, l.state);
    try t.expect(!l.admits());
    try t.expect(!l.releaseFailed(1, 12));
}

test "an idle box admits TP1; a draining or forming box does not" {
    var m: Machine = .{};
    try t.expect(m.admits());
    try t.expectEqual(Mode.tp1, m.mode());
    _ = m.command(.drain, 1, 0);
    try t.expect(!m.admits());
}

fn formedFollower(n: u64) Machine {
    var m: Machine = .{};
    _ = m.command(.drain, n, 0);
    _ = m.command(.join, n, 0);
    _ = m.formed(n, 5);
    return m;
}

test "a follower's abort of a formed group waits for the leader's OP_RELEASE, then releases as for tp1" {
    var m = formedFollower(3);
    try t.expectEqual(Outcome{ .ok = .await_release }, m.command(.abort, 3, 100));
    try t.expect(!m.admits());
    try t.expectEqual(Mode.tp4, m.mode());
    try t.expect(m.releasing(3, 500));
    try t.expectEqual(State.releasing, m.state);
    try t.expect(m.released(3, 900));
    m.tick(100 + 2000);
    try t.expect(!m.takeFallDue());
    try t.expectEqual(State.released, m.state);
}

test "a follower with no OP_RELEASE for 2 s falls by itself, and the fall is due once" {
    var m = formedFollower(3);
    _ = m.command(.abort, 3, 100);
    m.tick(100 + 1999);
    try t.expect(!m.takeFallDue());
    m.tick(100 + 2000);
    try t.expect(m.takeFallDue());
    try t.expect(!m.takeFallDue());
    try t.expect(!m.releasing(3, 2200));
    try t.expect(m.formationFailed(3, 2300, .release_timeout));
    try t.expectEqual(State.aborted, m.state);
    try t.expectEqual(Failure.release_timeout, m.failure);
    try t.expectEqual(Mode.tp1, m.mode());
    try t.expect(m.admits());
}

test "a follower's wait does not count toward the unwind limit; its fall does" {
    var m = formedFollower(3);
    _ = m.command(.abort, 3, 0);
    m.tick(2000 + 2499);
    try t.expect(m.takeFallDue());
    try t.expect(!m.quarantined);
    m.tick(2000 + 2499 + 2499);
    try t.expect(!m.quarantined);
    m.tick(2000 + 2499 + 2500);
    try t.expect(m.quarantined);
}

test "the leader's abort of a formed group is the planned down: releasing, then released" {
    var l: Machine = .{};
    _ = l.command(.drain, 3, 0);
    _ = l.command(.tp4, 3, 0);
    _ = l.formed(3, 5);
    try t.expectEqual(Outcome{ .ok = .begin_release }, l.command(.abort, 3, 10));
    try t.expectEqual(State.releasing, l.state);
    try t.expect(l.released(3, 50));
    try t.expectEqual(Outcome.noop, l.command(.abort, 3, 60));
    try t.expectEqual(State.released, l.state);
}
