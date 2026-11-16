//! The tip of each session, persisted by itself: a prefix entry is shared between sessions, so its record cannot
//! carry one session's name. An append-only file of fixed records; a damaged or half-written record is skipped.

const std = @import("std");

pub const Tip = struct { session: u64, id: u64, n: u64, variant: u64, seq: u64 };

pub const RECORD_BYTES: usize = 48;
pub const DEFAULT_CAP: usize = 65536;

pub const Index = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    path: []u8,
    cap: usize,

    pub fn init(allocator: std.mem.Allocator, io: std.Io, path: []const u8, cap: usize) !Index {
        return .{ .allocator = allocator, .io = io, .path = try allocator.dupe(u8, path), .cap = cap };
    }

    pub fn deinit(self: *Index) void {
        self.allocator.free(self.path);
    }

    /// The newest tip of each of the newest `cap` sessions, oldest first.
    pub fn load(self: *Index, out: *std.ArrayList(Tip)) !void {
        _ = .{ self, out };
    }

    pub fn append(self: *Index, tip: Tip) void {
        _ = .{ self, tip };
    }

    pub fn find(self: *Index, session: u64) ?Tip {
        _ = .{ self, session };
        return null;
    }
};

const testing = std.testing;

fn tmpPath(tmp: *testing.TmpDir, buf: []u8) ![]const u8 {
    const n = try tmp.dir.realPath(testing.io, buf);
    return std.fmt.bufPrint(buf[n..], "/session-tips.bin", .{}) catch error.NoSpaceLeft;
}

fn tipOf(session: u64, seq: u64) Tip {
    return .{ .session = session, .id = session * 100 + seq, .n = 512 + seq, .variant = 0, .seq = seq };
}

fn loaded(index: *Index) !std.ArrayList(Tip) {
    var out = std.ArrayList(Tip).empty;
    errdefer out.deinit(testing.allocator);
    try index.load(&out);
    return out;
}

test "a session index returns each session's newest tip, oldest first, after a reopen" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const path = try tmpPath(&tmp, &buf);
    {
        var index = try Index.init(testing.allocator, testing.io, path, 16);
        defer index.deinit();
        index.append(tipOf(5, 1));
        index.append(tipOf(6, 2));
        index.append(tipOf(5, 3));
    }
    var index = try Index.init(testing.allocator, testing.io, path, 16);
    defer index.deinit();
    var out = try loaded(&index);
    defer out.deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 2), out.items.len);
    try testing.expectEqual(tipOf(6, 2), out.items[0]);
    try testing.expectEqual(tipOf(5, 3), out.items[1]);
    try testing.expectEqual(tipOf(5, 3), index.find(5).?);
    try testing.expect(index.find(7) == null);
}

test "a session index skips a half-written tail and a damaged record, and appends after them" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const path = try tmpPath(&tmp, &buf);
    {
        var index = try Index.init(testing.allocator, testing.io, path, 16);
        defer index.deinit();
        index.append(tipOf(1, 1));
        index.append(tipOf(2, 2));
        index.append(tipOf(3, 3));
    }
    {
        const f = try std.Io.Dir.openFileAbsolute(testing.io, path, .{ .mode = .read_write });
        defer f.close(testing.io);
        try f.writePositionalAll(testing.io, "\xff", RECORD_BYTES + 9);
        try f.setLength(testing.io, 3 * RECORD_BYTES - 10);
    }
    var index = try Index.init(testing.allocator, testing.io, path, 16);
    defer index.deinit();
    {
        var out = try loaded(&index);
        defer out.deinit(testing.allocator);
        try testing.expectEqual(@as(usize, 1), out.items.len);
        try testing.expectEqual(tipOf(1, 1), out.items[0]);
    }
    index.append(tipOf(4, 4));
    var again = try Index.init(testing.allocator, testing.io, path, 16);
    defer again.deinit();
    var out = try loaded(&again);
    defer out.deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 2), out.items.len);
    try testing.expectEqual(tipOf(4, 4), out.items[1]);
}

test "a session index keeps the newest sessions at its cap, and find reads the file" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const path = try tmpPath(&tmp, &buf);
    var index = try Index.init(testing.allocator, testing.io, path, 3);
    defer index.deinit();
    for (1..15) |s| index.append(tipOf(s, s));
    try testing.expectEqual(tipOf(14, 14), index.find(14).?);
    var out = try loaded(&index);
    defer out.deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 3), out.items.len);
    try testing.expectEqual(tipOf(12, 12), out.items[0]);
    try testing.expectEqual(tipOf(14, 14), out.items[2]);
    try testing.expectEqual(tipOf(12, 12), index.find(12).?);
    try testing.expect(index.find(2) == null);
    const f = try std.Io.Dir.openFileAbsolute(testing.io, path, .{});
    defer f.close(testing.io);
    try testing.expectEqual(@as(u64, 3 * RECORD_BYTES), try f.length(testing.io));
}
