//! Distributed progress tracking and follower exit paths.
const std = @import("std");

pub const State = struct {
    steps: std.atomic.Value(u64) = .init(0),
    in_flight: std.atomic.Value(bool) = .init(false),
    started_ms: std.atomic.Value(u64) = .init(0),
    step_id: std.atomic.Value(u64) = .init(0),
    attempt: std.atomic.Value(u32) = .init(0),

    pub fn begin(self: *State, id: u64, now_ms: u64) void {
        if (self.in_flight.load(.acquire)) return;
        self.step_id.store(id, .monotonic);
        self.started_ms.store(now_ms, .monotonic);
        self.in_flight.store(true, .release);
    }

    pub fn complete(self: *State) void {
        _ = self.steps.fetchAdd(1, .release);
        self.in_flight.store(false, .release);
    }

    pub fn completeIfInFlight(self: *State) bool {
        if (!self.in_flight.swap(false, .acq_rel)) return false;
        _ = self.steps.fetchAdd(1, .release);
        return true;
    }

    pub fn stalled(self: *State, now_ms: u64, deadline_ms: u64) ?Event {
        if (!self.in_flight.load(.acquire)) return null;
        const started = self.started_ms.load(.monotonic);
        if (now_ms -| started < deadline_ms) return null;
        return .{
            .step = self.step_id.load(.monotonic),
            .waited_ms = now_ms - started,
            .attempt = self.attempt.fetchAdd(1, .monotonic) + 1,
        };
    }
};

pub const Event = struct { step: u64, waited_ms: u64, attempt: u32 };
pub const ExitFn = *const fn (i32) void;
pub const StallFn = *const fn (?*anyopaque, Event) void;
pub const BellFailure = enum { timeout, eof };

pub fn classifyBellFailure(n: isize, errno: i32, interrupted: i32, would_block: i32) ?BellFailure {
    if (n == 0) return .eof;
    if (n > 0 or errno == interrupted) return null;
    return if (errno == would_block) .timeout else .eof;
}

pub fn exitProcess(code: i32) void {
    std.c._exit(code);
}

pub fn fireIfStalled(state: *State, now_ms: u64, deadline_ms: u64, ctx: ?*anyopaque, on_stall: StallFn) bool {
    const event = state.stalled(now_ms, deadline_ms) orelse return false;
    on_stall(ctx, event);
    return true;
}

pub fn followerFailure(_: BellFailure, exit_fn: ExitFn) void {
    exit_fn(1);
}

var test_exits: u32 = 0;
var test_event: ?Event = null;

fn testExit(_: i32) void {
    test_exits += 1;
}

fn testStall(_: ?*anyopaque, event: Event) void {
    test_event = event;
    testExit(1);
}

test "watchdog fires exit after a stalled in-flight step" {
    var state: State = .{};
    test_exits = 0;
    test_event = null;
    state.begin(7, 100);
    try std.testing.expect(fireIfStalled(&state, 360, 10, null, &testStall));
    try std.testing.expectEqual(@as(u32, 1), test_exits);
    try std.testing.expectEqual(@as(u64, 260), test_event.?.waited_ms);
}

test "watchdog ignores idle and completed progress" {
    var state: State = .{};
    try std.testing.expect(state.stalled(50_000, 10) == null);
    for (0..20) |step| {
        const now: u64 = @intCast(step * 500);
        state.begin(step + 1, now);
        try std.testing.expect(state.completeIfInFlight());
        try std.testing.expect(state.stalled(now + 11, 10) == null);
    }
    try std.testing.expectEqual(@as(u64, 20), state.steps.load(.monotonic));
}

test "follower timeout and EOF use the exit seam" {
    test_exits = 0;
    for ([_]struct { n: isize, errno: i32, failure: BellFailure }{
        .{ .n = -1, .errno = 11, .failure = .timeout },
        .{ .n = 0, .errno = 0, .failure = .eof },
    }) |read| {
        const failure = classifyBellFailure(read.n, read.errno, 4, 11).?;
        try std.testing.expectEqual(read.failure, failure);
        followerFailure(failure, &testExit);
    }
    try std.testing.expectEqual(@as(u32, 2), test_exits);
}
