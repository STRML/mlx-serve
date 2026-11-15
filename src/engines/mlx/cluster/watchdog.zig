//! Distributed stall detection: a step with a collective in flight longer than the deadline means a peer died or froze.
const std = @import("std");

pub const State = struct {
    in_flight: std.atomic.Value(bool) = .init(false),
    started_ms: std.atomic.Value(u64) = .init(0),
    step: std.atomic.Value(u64) = .init(0),

    /// A collective step starts; a step already in flight keeps its start time.
    pub fn begin(self: *State, now_ms: u64) void {
        if (self.in_flight.load(.acquire)) return;
        _ = self.step.fetchAdd(1, .monotonic);
        self.started_ms.store(now_ms, .monotonic);
        self.in_flight.store(true, .release);
    }

    /// A forward inside the step started: the deadline counts from here, so a long prefill of many chunks never trips it.
    pub fn progress(self: *State, now_ms: u64) void {
        if (self.in_flight.load(.acquire)) self.started_ms.store(now_ms, .monotonic);
    }

    pub fn done(self: *State) void {
        self.in_flight.store(false, .release);
    }

    /// How long the step in flight has waited, once past `deadline_ms`; null while idle or within it.
    pub fn stalledMs(self: *const State, now_ms: u64, deadline_ms: u64) ?u64 {
        if (!self.in_flight.load(.acquire)) return null;
        const waited = now_ms -| self.started_ms.load(.monotonic);
        return if (waited >= deadline_ms) waited else null;
    }
};

pub fn loopAlive(now_ms: u64, last_tick_ms: u64, in_flight: bool, limit_ms: u64) bool {
    return in_flight or now_ms -| last_tick_ms <= limit_ms;
}

pub const BellFailure = enum { timeout, eof };

pub fn setBellTimeout(fd: std.c.fd_t, ms: u32) void {
    const timeout = std.c.timeval{ .sec = @intCast(ms / 1000), .usec = @intCast((ms % 1000) * 1000) };
    _ = std.c.setsockopt(fd, std.posix.SOL.SOCKET, std.posix.SO.RCVTIMEO, &timeout, @sizeOf(@TypeOf(timeout)));
}

/// A bell read's result: null to keep reading (data, or an interrupted read), else why the leader is gone.
pub fn classifyBellFailure(n: isize, errno: i32, interrupted: i32, would_block: i32) ?BellFailure {
    if (n == 0) return .eof;
    if (n > 0 or errno == interrupted) return null;
    return if (errno == would_block) .timeout else .eof;
}

test "a step in flight past the deadline is a stall; idle and finished steps are not" {
    var s: State = .{};
    try std.testing.expectEqual(@as(?u64, null), s.stalledMs(50_000, 10));
    s.begin(100);
    try std.testing.expectEqual(@as(?u64, null), s.stalledMs(105, 10));
    try std.testing.expectEqual(@as(?u64, 260), s.stalledMs(360, 10));
    s.begin(300);
    try std.testing.expectEqual(@as(?u64, 260), s.stalledMs(360, 10));
    s.done();
    try std.testing.expectEqual(@as(?u64, null), s.stalledMs(10_000, 10));
    s.begin(10_000);
    try std.testing.expectEqual(@as(u64, 2), s.step.load(.monotonic));
}

test "idle loop ticks keep the heartbeat alive, stale idle trips, and a step keeps it alive" {
    for (1..9) |tick| {
        const now_ms: u64 = tick * 500;
        try std.testing.expect(loopAlive(now_ms + 1_000, now_ms, false, 1_000));
    }
    try std.testing.expect(!loopAlive(7_001, 4_000, false, 3_000));
    try std.testing.expect(loopAlive(20_000, 4_000, true, 1_000));
}

test "a step of many forwards survives while each forward is within the deadline; a hung forward fires" {
    var s: State = .{};
    s.begin(0);
    var now: u64 = 0;
    for (0..20) |_| {
        now += 2_000;
        try std.testing.expectEqual(@as(?u64, null), s.stalledMs(now, 10_000));
        s.progress(now);
    }
    try std.testing.expectEqual(@as(?u64, 10_000), s.stalledMs(now + 10_000, 10_000));
}

test "a bell read: timeout and EOF end the follower, data and EINTR do not" {
    try std.testing.expectEqual(BellFailure.timeout, classifyBellFailure(-1, 11, 4, 11).?);
    try std.testing.expectEqual(BellFailure.eof, classifyBellFailure(0, 0, 4, 11).?);
    try std.testing.expectEqual(BellFailure.eof, classifyBellFailure(-1, 54, 4, 11).?);
    try std.testing.expectEqual(@as(?BellFailure, null), classifyBellFailure(-1, 4, 4, 11));
    try std.testing.expectEqual(@as(?BellFailure, null), classifyBellFailure(3, 0, 4, 11));
}

test "a follower bell waits through the idle timeout before ready" {
    var pair: [2]std.c.fd_t = undefined;
    try std.testing.expectEqual(@as(c_int, 0), std.c.socketpair(std.posix.AF.UNIX, std.posix.SOCK.STREAM, 0, &pair));
    defer for (pair) |fd| {
        _ = std.c.close(fd);
    };
    var probe = struct {
        fd: std.c.fd_t,
        result: isize = -1,
        returned: std.atomic.Value(bool) = .init(false),
        fn read(self: *@This()) void {
            var byte: u8 = undefined;
            self.result = std.c.read(self.fd, @ptrCast(&byte), 1);
            self.returned.store(true, .release);
        }
    }{ .fd = pair[0] };
    const thread = try std.Thread.spawn(.{}, @TypeOf(probe).read, .{&probe});
    var fds = [_]std.posix.pollfd{.{ .fd = pair[0], .events = std.posix.POLL.IN, .revents = 0 }};
    try std.testing.expectEqual(@as(usize, 0), try std.posix.poll(&fds, 50));
    try std.testing.expect(!probe.returned.load(.acquire));
    try std.testing.expectEqual(@as(isize, 1), std.c.write(pair[1], &[_]u8{1}, 1));
    thread.join();
    try std.testing.expectEqual(@as(isize, 1), probe.result);
    setBellTimeout(pair[0], 20);
    var byte: u8 = undefined;
    const n = std.c.read(pair[0], @ptrCast(&byte), 1);
    try std.testing.expectEqual(BellFailure.timeout, classifyBellFailure(n, std.c._errno().*, @backingInt(std.c.E.INTR), @backingInt(std.c.E.AGAIN)).?);
}
