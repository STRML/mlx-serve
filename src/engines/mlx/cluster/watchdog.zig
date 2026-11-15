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

pub const BellFailure = enum { timeout, eof };

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

test "a bell read: timeout and EOF end the follower, data and EINTR do not" {
    try std.testing.expectEqual(BellFailure.timeout, classifyBellFailure(-1, 11, 4, 11).?);
    try std.testing.expectEqual(BellFailure.eof, classifyBellFailure(0, 0, 4, 11).?);
    try std.testing.expectEqual(BellFailure.eof, classifyBellFailure(-1, 54, 4, 11).?);
    try std.testing.expectEqual(@as(?BellFailure, null), classifyBellFailure(-1, 4, 4, 11));
    try std.testing.expectEqual(@as(?BellFailure, null), classifyBellFailure(3, 0, 4, 11));
}
