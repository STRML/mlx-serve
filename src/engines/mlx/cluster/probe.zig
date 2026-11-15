const std = @import("std");

pub const ProgressDeadline = struct {
    last: u64 = 0,
    since_ms: u64 = 0,

    pub fn expired(self: *ProgressDeadline, progress: u64, now_ms: u64, timeout_ms: u64) bool {
        if (progress != self.last) {
            self.last = progress;
            self.since_ms = now_ms;
        }
        return now_ms -| self.since_ms >= timeout_ms;
    }
};

pub fn compact(rank: u32, size: u32, survivors: u32) !struct { rank: u32, size: u32 } {
    if (size < 2 or size > 8 or rank >= size or survivors >> @intCast(size) != 0) return error.InvalidArgs;
    if (survivors & 1 == 0) return error.DistCoordinatorGone;
    const bit = @as(u32, 1) << @intCast(rank);
    if (survivors & bit == 0) return error.DistPeerGone;
    const count: u32 = @popCount(survivors);
    return .{ .rank = @popCount(survivors & (bit - 1)), .size = count };
}

pub fn compactDevices(survivors: u32, ports: []const []const u8, out: []u8) ![]const u8 {
    var w: std.Io.Writer = .fixed(out);
    var first = true;
    for (ports, 0..) |port, rank| {
        if (survivors & (@as(u32, 1) << @intCast(rank)) == 0) continue;
        if (!first) try w.writeByte(',');
        try w.writeAll(port);
        first = false;
    }
    return w.buffered();
}

pub fn peerRanks(rank: u32, size: u32, out: []u8) ![]const u8 {
    var w: std.Io.Writer = .fixed(out);
    var first = true;
    for (0..size) |peer| {
        if (peer == rank) continue;
        if (!first) try w.writeByte(',');
        try w.print("{d}", .{peer});
        first = false;
    }
    return w.buffered();
}

test "probe watchdog resets its deadline only on completed progress" {
    var deadline: ProgressDeadline = .{ .last = 7, .since_ms = 100 };
    try std.testing.expect(!deadline.expired(7, 149, 50));
    try std.testing.expect(!deadline.expired(8, 150, 50));
    try std.testing.expect(!deadline.expired(8, 199, 50));
    try std.testing.expect(deadline.expired(8, 200, 50));
    try std.testing.expectEqual(@as(u64, 8), deadline.last);
    var buf: [32]u8 = undefined;
    try std.testing.expectEqualStrings("0,2", try peerRanks(1, 3, &buf));
}

test "probe recovery compacts surviving ranks and their original device ports" {
    const ports = [_][]const u8{ "rdma_en2", "rdma_en3", "-", "rdma_en5" };
    var buf: [256]u8 = undefined;
    const here = try compact(2, 4, 0b1101);
    try std.testing.expectEqual(@as(u32, 1), here.rank);
    try std.testing.expectEqual(@as(u32, 3), here.size);
    try std.testing.expectEqualStrings("rdma_en2,-,rdma_en5", try compactDevices(0b1101, &ports, &buf));
    try std.testing.expectError(error.DistPeerGone, compact(1, 4, 0b1101));
    const alone = try compact(0, 2, 0b01);
    try std.testing.expectEqual(@as(u32, 1), alone.size);
    try std.testing.expectError(error.DistCoordinatorGone, compact(1, 3, 0b110));
    try std.testing.expectError(error.InvalidArgs, compact(0, 3, 0b1101));
}
