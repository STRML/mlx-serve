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

    /// The newest tip of each of the newest `cap` sessions, oldest first. The file keeps each of those sessions' two
    /// newest tips, so a session whose newest record is gone has one to fall back to (`findBefore`); a file holding
    /// anything else (older tips, sessions past the cap, a damaged record) is rewritten to just those.
    pub fn load(self: *Index, out: *std.ArrayList(Tip)) !void {
        const bytes = self.readAll() orelse return;
        defer self.allocator.free(bytes);
        const Pair = struct { newest: Tip, previous: ?Tip = null };
        var pairs = std.AutoHashMap(u64, Pair).init(self.allocator);
        defer pairs.deinit();
        var clean = bytes.len % RECORD_BYTES == 0;
        var records: usize = 0;
        var at: usize = 0;
        while (at + RECORD_BYTES <= bytes.len) : (at += RECORD_BYTES) {
            const tip = decode(bytes[at..][0..RECORD_BYTES]) orelse {
                clean = false;
                continue;
            };
            records += 1;
            const held = try pairs.getOrPut(tip.session);
            if (!held.found_existing) {
                held.value_ptr.* = .{ .newest = tip };
            } else if (tip.seq >= held.value_ptr.newest.seq) {
                if (tip.seq > held.value_ptr.newest.seq) held.value_ptr.previous = held.value_ptr.newest;
                held.value_ptr.newest = tip;
            } else if (held.value_ptr.previous == null or tip.seq >= held.value_ptr.previous.?.seq) {
                held.value_ptr.previous = tip;
            }
        }
        var newest = std.ArrayList(Tip).empty;
        defer newest.deinit(self.allocator);
        var it = pairs.valueIterator();
        while (it.next()) |pair| try newest.append(self.allocator, pair.newest);
        std.mem.sort(Tip, newest.items, {}, bySeq);
        const kept = newest.items[newest.items.len -| self.cap..];
        var survivors = std.ArrayList(Tip).empty;
        defer survivors.deinit(self.allocator);
        for (kept) |tip| {
            try survivors.append(self.allocator, tip);
            if (pairs.get(tip.session).?.previous) |previous| try survivors.append(self.allocator, previous);
        }
        std.mem.sort(Tip, survivors.items, {}, bySeq);
        try out.appendSlice(self.allocator, kept);
        if (!clean or records != survivors.items.len) self.rewrite(survivors.items);
    }

    pub fn append(self: *Index, tip: Tip) void {
        self.appendRecord(tip) catch |err| log.warn("[session-index] append failed: {s}\n", .{@errorName(err)});
    }

    /// The newest tip the file holds for `session` older than `seq` and of another boundary than `id`: where a session
    /// falls back to when the record of its newest one is gone.
    pub fn findBefore(self: *Index, session: u64, id: u64, seq: u64) ?Tip {
        _ = .{ self, session, id, seq };
        return null;
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
    const tail = try std.fmt.bufPrint(buf[n..], "/session-tips.bin", .{});
    return buf[0 .. n + tail.len];
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
        try index.appendRecord(tipOf(5, 1));
        try index.appendRecord(tipOf(6, 2));
        try index.appendRecord(tipOf(5, 3));
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
    try testing.expectEqual(tipOf(5, 1), index.findBefore(5, tipOf(5, 3).id, 3).?);
    try testing.expect(index.findBefore(5, tipOf(5, 1).id, 1) == null);
}

test "a session index keeps a session's two newest tips through compaction" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const path = try tmpPath(&tmp, &buf);
    var index = try Index.init(testing.allocator, testing.io, path, 4);
    defer index.deinit();
    for (1..6) |seq| try index.appendRecord(tipOf(5, seq));
    var out = try loaded(&index);
    defer out.deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 1), out.items.len);
    try testing.expectEqual(tipOf(5, 5), out.items[0]);
    try testing.expectEqual(tipOf(5, 4), index.findBefore(5, tipOf(5, 5).id, 5).?);
    try testing.expect(index.findBefore(5, tipOf(5, 4).id, 4) == null);
    const f = try std.Io.Dir.openFileAbsolute(testing.io, path, .{});
    defer f.close(testing.io);
    try testing.expectEqual(@as(u64, 2 * RECORD_BYTES), try f.length(testing.io));
}

test "a session index skips a half-written tail and a damaged record, and appends after them" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const path = try tmpPath(&tmp, &buf);
    {
        var index = try Index.init(testing.allocator, testing.io, path, 16);
        defer index.deinit();
        try index.appendRecord(tipOf(1, 1));
        try index.appendRecord(tipOf(2, 2));
        try index.appendRecord(tipOf(3, 3));
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
    try index.appendRecord(tipOf(4, 4));
    var again = try Index.init(testing.allocator, testing.io, path, 16);
    defer again.deinit();
    var out = try loaded(&again);
    defer out.deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 2), out.items.len);
    try testing.expectEqual(tipOf(4, 4), out.items[1]);
}

test "a session index counts a tip of one boundary once, so the fallback is another boundary" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const path = try tmpPath(&tmp, &buf);
    var index = try Index.init(testing.allocator, testing.io, path, 8);
    defer index.deinit();
    const first: Tip = .{ .session = 5, .id = 100, .n = 512, .variant = 0, .seq = 1 };
    for ([_]u64{ 2, 3, 4 }) |seq| try index.appendRecord(.{ .session = 5, .id = 200, .n = 513, .variant = 0, .seq = seq });
    try index.appendRecord(first);
    var out = try loaded(&index);
    defer out.deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 1), out.items.len);
    try testing.expectEqual(@as(u64, 4), out.items[0].seq);
    try testing.expectEqual(first, index.findBefore(5, 200, 5).?);
    const f = try std.Io.Dir.openFileAbsolute(testing.io, path, .{});
    defer f.close(testing.io);
    try testing.expectEqual(@as(u64, 2 * RECORD_BYTES), try f.length(testing.io));
}

test "a session index keeps the newest sessions at its cap, and find reads the file" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const path = try tmpPath(&tmp, &buf);
    var index = try Index.init(testing.allocator, testing.io, path, 3);
    defer index.deinit();
    for (1..15) |s| try index.appendRecord(tipOf(s, s));
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
