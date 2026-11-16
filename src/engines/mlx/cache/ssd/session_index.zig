//! The tip of each session, persisted by itself: a prefix entry is shared between sessions, so its record cannot
//! carry one session's name. An append-only file of fixed records; a damaged or half-written record is skipped.

const std = @import("std");
const log = @import("../../../../sdk/util/log.zig");

pub const Tip = struct { session: u64, id: u64, n: u64, variant: u64, seq: u64 };

pub const RECORD_BYTES: usize = 48;
pub const DEFAULT_CAP: usize = 65536;
/// The file is rewritten, newest `cap` sessions only, once it holds this many records per cap.
const COMPACT_FACTOR: usize = 4;
const MAX_READ_BYTES: usize = 256 * 1024 * 1024;
const CHECK_SEED: u64 = 0x7365_7373_7469_7073;

fn checksum(body: []const u8) u64 {
    return std.hash.Wyhash.hash(CHECK_SEED, body);
}

fn encode(tip: Tip, out: *[RECORD_BYTES]u8) void {
    const words = [5]u64{ tip.session, tip.id, tip.n, tip.variant, tip.seq };
    for (words, 0..) |w, i| std.mem.writeInt(u64, out[i * 8 ..][0..8], w, .little);
    std.mem.writeInt(u64, out[40..48], checksum(out[0..40]), .little);
}

fn decode(record: *const [RECORD_BYTES]u8) ?Tip {
    if (std.mem.readInt(u64, record[40..48], .little) != checksum(record[0..40])) return null;
    var words: [5]u64 = undefined;
    for (&words, 0..) |*w, i| w.* = std.mem.readInt(u64, record[i * 8 ..][0..8], .little);
    return .{ .session = words[0], .id = words[1], .n = words[2], .variant = words[3], .seq = words[4] };
}

fn bySeq(_: void, a: Tip, b: Tip) bool {
    return if (a.seq != b.seq) a.seq < b.seq else a.session < b.session;
}

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

    /// The newest tip of each of the newest `cap` sessions, oldest first. A file holding anything else (older tips of
    /// a session, sessions past the cap, a damaged record) is rewritten to just these.
    pub fn load(self: *Index, out: *std.ArrayList(Tip)) !void {
        const bytes = self.readAll() orelse return;
        defer self.allocator.free(bytes);
        var newest = std.AutoHashMap(u64, Tip).init(self.allocator);
        defer newest.deinit();
        var clean = bytes.len % RECORD_BYTES == 0;
        var records: usize = 0;
        var at: usize = 0;
        while (at + RECORD_BYTES <= bytes.len) : (at += RECORD_BYTES) {
            const tip = decode(bytes[at..][0..RECORD_BYTES]) orelse {
                clean = false;
                continue;
            };
            records += 1;
            const held = try newest.getOrPut(tip.session);
            if (!held.found_existing or held.value_ptr.seq <= tip.seq) held.value_ptr.* = tip;
        }
        const first = out.items.len;
        var it = newest.valueIterator();
        while (it.next()) |tip| try out.append(self.allocator, tip.*);
        std.mem.sort(Tip, out.items[first..], {}, bySeq);
        const kept = @min(out.items.len - first, self.cap);
        const drop = out.items.len - first - kept;
        if (drop > 0) std.mem.copyForwards(Tip, out.items[first..][0..kept], out.items[first + drop ..][0..kept]);
        out.shrinkRetainingCapacity(first + kept);
        if (!clean or records != kept) self.rewrite(out.items[first..]);
    }

    pub fn append(self: *Index, tip: Tip) void {
        self.appendRecord(tip) catch |err| log.warn("[session-index] append failed: {s}\n", .{@errorName(err)});
    }

    /// The newest tip the file holds for `session`.
    pub fn find(self: *Index, session: u64) ?Tip {
        const bytes = self.readAll() orelse return null;
        defer self.allocator.free(bytes);
        var best: ?Tip = null;
        var at: usize = 0;
        while (at + RECORD_BYTES <= bytes.len) : (at += RECORD_BYTES) {
            const tip = decode(bytes[at..][0..RECORD_BYTES]) orelse continue;
            if (tip.session != session) continue;
            if (best == null or best.?.seq <= tip.seq) best = tip;
        }
        return best;
    }

    fn appendRecord(self: *Index, tip: Tip) !void {
        if (self.recordCount() >= COMPACT_FACTOR * self.cap) self.compact();
        const f = try std.Io.Dir.createFileAbsolute(self.io, self.path, .{ .truncate = false });
        defer f.close(self.io);
        const len = try f.length(self.io);
        var record: [RECORD_BYTES]u8 = undefined;
        encode(tip, &record);
        try f.writePositionalAll(self.io, &record, len - len % RECORD_BYTES);
    }

    fn recordCount(self: *Index) usize {
        const f = std.Io.Dir.openFileAbsolute(self.io, self.path, .{}) catch return 0;
        defer f.close(self.io);
        return @intCast((f.length(self.io) catch return 0) / RECORD_BYTES);
    }

    fn compact(self: *Index) void {
        var tips = std.ArrayList(Tip).empty;
        defer tips.deinit(self.allocator);
        self.load(&tips) catch |err| log.warn("[session-index] compaction failed: {s}\n", .{@errorName(err)});
    }

    fn readAll(self: *Index) ?[]u8 {
        const f = std.Io.Dir.openFileAbsolute(self.io, self.path, .{}) catch return null;
        defer f.close(self.io);
        var buffer: [8192]u8 = undefined;
        var reader = f.reader(self.io, &buffer);
        return reader.interface.allocRemaining(self.allocator, .limited(MAX_READ_BYTES)) catch null;
    }

    fn rewrite(self: *Index, tips: []const Tip) void {
        self.rewriteFile(tips) catch |err| log.warn("[session-index] rewrite failed: {s}\n", .{@errorName(err)});
    }

    fn rewriteFile(self: *Index, tips: []const Tip) !void {
        const tmp_path = try std.fmt.allocPrint(self.allocator, "{s}.tmp", .{self.path});
        defer self.allocator.free(tmp_path);
        {
            const f = try std.Io.Dir.createFileAbsolute(self.io, tmp_path, .{});
            defer f.close(self.io);
            var buffer: [8192]u8 = undefined;
            var writer = f.writer(self.io, &buffer);
            var record: [RECORD_BYTES]u8 = undefined;
            for (tips) |tip| {
                encode(tip, &record);
                try writer.interface.writeAll(&record);
            }
            try writer.interface.flush();
        }
        try std.Io.Dir.renameAbsolute(tmp_path, self.path, self.io);
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
