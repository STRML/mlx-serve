//! Bounded host rows. Sources fill bytes only; the consumer owns device arrays.
const std = @import("std");

pub const Source = struct {
    ctx: *const anyopaque,
    row_bytes: usize,
    read: *const fn (*const anyopaque, u32, []u8) bool,
};

pub fn allowed(budget: usize, available: u64) usize {
    return @min(budget, available -| (12 << 30));
}

pub const ReadAhead = struct {
    gpa: std.mem.Allocator,
    stores: []Store,
    bytes: usize,
    threads: [workers]?std.Thread = @splat(null),
    stop: std.atomic.Value(bool) = .init(false),
    cursor: std.atomic.Value(usize) = .init(0),
    hits: std.atomic.Value(usize) = .init(0),
    misses: std.atomic.Value(usize) = .init(0),
    ready: std.atomic.Value(usize) = .init(0),

    const workers = 16;
    const stack_bytes = if (@import("builtin").os.tag == .linux) 1 << 20 else 128 << 10;
    const worker_bytes = workers * (stack_bytes + (64 << 10));
    const empty = std.math.maxInt(u32);
    const Entry = struct { id: u32 = 0, slot: u32 = empty };
    const Store = struct {
        source: Source,
        index: []Entry,
        ids: []u32,
        state: []std.atomic.Value(u32),
        data: []u8,
        count: usize = 0,

        fn find(self: *const Store, id: u32) usize {
            var pos: usize = @as(u32, id *% 0x9e3779b9) & (self.index.len - 1);
            while (self.index[pos].slot != empty and self.index[pos].id != id) pos = (pos + 1) & (self.index.len - 1);
            return pos;
        }
        fn deinit(self: Store, gpa: std.mem.Allocator) void {
            gpa.free(self.index);
            gpa.free(self.ids);
            gpa.free(self.state);
            gpa.free(self.data);
        }
    };

    fn indexLen(cap: usize) usize {
        return std.math.ceilPowerOfTwoAssert(usize, @max(2, cap * 2));
    }
    fn bill(sources: []const Source, cap: usize) usize {
        var bytes: usize = @sizeOf(ReadAhead) + worker_bytes + sources.len * @sizeOf(Store);
        for (sources) |src| bytes += indexLen(cap) * @sizeOf(Entry) + cap * (src.row_bytes + 8);
        return bytes;
    }

    pub fn create(gpa: std.mem.Allocator, sources: []const Source, budget: usize, max_rows: usize) !?*ReadAhead {
        if (sources.len == 0 or max_rows == 0 or bill(sources, 1) > budget) return null;
        var lo: usize = 1;
        var hi = @min(max_rows, budget / @max(1, sources[0].row_bytes));
        while (lo < hi) {
            const mid = lo + (hi - lo + 1) / 2;
            if (bill(sources, mid) <= budget) lo = mid else hi = mid - 1;
        }
        const self = try gpa.create(ReadAhead);
        errdefer gpa.destroy(self);
        const stores = try gpa.alloc(Store, sources.len);
        var made: usize = 0;
        errdefer {
            for (stores[0..made]) |s| s.deinit(gpa);
            gpa.free(stores);
        }
        for (sources, stores) |src, *s| {
            const index = try gpa.alloc(Entry, indexLen(lo));
            errdefer gpa.free(index);
            const ids = try gpa.alloc(u32, lo);
            errdefer gpa.free(ids);
            const state = try gpa.alloc(std.atomic.Value(u32), lo);
            errdefer gpa.free(state);
            const data = try gpa.alloc(u8, lo * src.row_bytes);
            @memset(index, .{});
            @memset(state, .init(0));
            s.* = .{ .source = src, .index = index, .ids = ids, .state = state, .data = data };
            made += 1;
        }
        self.* = .{ .gpa = gpa, .stores = stores, .bytes = bill(sources, lo) };
        return self;
    }

    /// Selection ends before start; duplicate rows retain their first position.
    pub fn add(self: *ReadAhead, layer: usize, id: u32) bool {
        const s = &self.stores[layer];
        const pos = s.find(id);
        if (s.index[pos].slot != empty) return true;
        if (s.count == s.ids.len) return false;
        s.index[pos] = .{ .id = id, .slot = @intCast(s.count) };
        s.ids[s.count] = id;
        s.count += 1;
        return true;
    }

    pub fn full(self: *const ReadAhead) bool {
        for (self.stores) |s| if (s.count < s.ids.len) return false;
        return true;
    }

    pub fn start(self: *ReadAhead) !void {
        errdefer {
            self.stop.store(true, .release);
            self.wait();
        }
        for (&self.threads) |*t| t.* = try std.Thread.spawn(.{ .stack_size = stack_bytes }, run, .{self});
    }

    fn run(self: *ReadAhead) void {
        var rows: usize = 0;
        for (self.stores) |s| rows = @max(rows, s.count);
        while (!self.stop.load(.acquire)) {
            const k = self.cursor.fetchAdd(1, .monotonic);
            if (k >= rows * self.stores.len) return;
            const s = &self.stores[k % self.stores.len];
            const row = k / self.stores.len;
            if (row >= s.count) continue;
            const out = s.data[row * s.source.row_bytes ..][0..s.source.row_bytes];
            const ok = s.source.read(s.source.ctx, s.ids[row], out);
            if (ok) _ = self.ready.fetchAdd(1, .monotonic);
            s.state[row].store(if (ok) 1 else 2, .release);
        }
    }

    /// A selected row waits for its reader; an unselected or failed row falls through to demand I/O.
    pub fn get(self: *ReadAhead, layer: usize, id: u32) ?[]const u8 {
        const s = &self.stores[layer];
        const slot = s.index[s.find(id)].slot;
        if (slot != empty) {
            while (s.state[slot].load(.acquire) == 0) {
                if (self.stop.load(.acquire)) {
                    _ = self.misses.fetchAdd(1, .monotonic);
                    return null;
                }
                const ts: std.c.timespec = .{ .sec = 0, .nsec = 100_000 };
                _ = std.c.nanosleep(&ts, null);
            }
            if (s.state[slot].load(.acquire) == 1) {
                _ = self.hits.fetchAdd(1, .monotonic);
                return s.data[slot * s.source.row_bytes ..][0..s.source.row_bytes];
            }
        }
        _ = self.misses.fetchAdd(1, .monotonic);
        return null;
    }

    pub fn wait(self: *ReadAhead) void {
        for (&self.threads) |*t| {
            if (t.*) |th| th.join();
            t.* = null;
        }
    }

    pub fn cancel(self: *ReadAhead) void {
        self.stop.store(true, .release);
    }

    pub fn deinit(self: *ReadAhead) void {
        self.cancel();
        self.wait();
        for (self.stores) |s| s.deinit(self.gpa);
        self.gpa.free(self.stores);
        self.gpa.destroy(self);
    }
};

const testing = std.testing;
const test_budget = ReadAhead.worker_bytes + (2 << 20);
const SlowSource = struct {
    caller: std.Thread.Id,
    reads: std.atomic.Value(usize) = .init(0),
    wrong_thread: std.atomic.Value(bool) = .init(false),
    fail: bool = false,

    fn read(ctx: *const anyopaque, id: u32, out: []u8) bool {
        const self: *@This() = @ptrCast(@alignCast(@constCast(ctx)));
        if (std.Thread.getCurrentId() == self.caller) self.wrong_thread.store(true, .release);
        _ = self.reads.fetchAdd(1, .monotonic);
        const ts: std.c.timespec = .{ .sec = 0, .nsec = 100_000 };
        _ = std.c.nanosleep(&ts, null);
        if (self.fail) return false;
        for (out, 0..) |*b, i| b.* = @truncate(id * 7 + i);
        return true;
    }
    fn source(self: *const @This()) Source {
        return .{ .ctx = self, .row_bytes = 64, .read = read };
    }
};

test "engram read-ahead: distinct rows from later chunks are resident, CPU workers only" {
    var source: SlowSource = .{ .caller = std.Thread.getCurrentId() };
    const ra = (try ReadAhead.create(testing.allocator, &.{ source.source(), source.source() }, test_budget, 8)) orelse return error.ReadAheadMissing;
    defer ra.deinit();
    for (0..2) |layer| {
        for ([_]u32{ 3, 1, 3, 7, 1 }) |id| _ = ra.add(layer, id);
    }
    try ra.start();
    ra.wait();
    try testing.expectEqual(@as(usize, 6), source.reads.load(.acquire));
    try testing.expect(!source.wrong_thread.load(.acquire));
    const before = source.reads.load(.acquire);
    for (0..2) |layer| for ([_]u32{ 7, 3, 1, 7 }) |id| {
        const raw = ra.get(layer, id) orelse return error.RowNotResident;
        for (raw, 0..) |b, i| try testing.expectEqual(@as(u8, @truncate(id * 7 + i)), b);
    };
    try testing.expectEqual(before, source.reads.load(.acquire));
    try testing.expectEqual(@as(usize, 8), ra.hits.load(.acquire));
}

test "engram read-ahead: cap includes metadata, unselected rows fall through" {
    var source: SlowSource = .{ .caller = std.Thread.getCurrentId() };
    const cap = ReadAhead.worker_bytes + 4096;
    const ra = (try ReadAhead.create(testing.allocator, &.{source.source()}, cap, 1000)) orelse return error.ReadAheadMissing;
    defer ra.deinit();
    for (0..1000) |id| _ = ra.add(0, @intCast(id));
    try testing.expect(ra.bytes <= cap);
    try ra.start();
    ra.wait();
    const count = source.reads.load(.acquire);
    try testing.expect(count > 0 and count < 1000);
    for (0..1000) |id| {
        const row = ra.get(0, @intCast(id));
        try testing.expectEqual(id < count, row != null);
    }
    try testing.expect((try ReadAhead.create(testing.allocator, &.{source.source()}, 0, 1000)) == null);
    try testing.expectEqual(@as(usize, 0), allowed(2 << 30, 12 << 30));
    try testing.expectEqual(@as(usize, 1 << 30), allowed(2 << 30, 13 << 30));
}

test "engram read-ahead: failed reads fall through and cancellation joins readers" {
    var source: SlowSource = .{ .caller = std.Thread.getCurrentId(), .fail = true };
    const ra = (try ReadAhead.create(testing.allocator, &.{source.source()}, test_budget, 8)) orelse return error.ReadAheadMissing;
    defer ra.deinit();
    _ = ra.add(0, 2);
    try ra.start();
    try testing.expect(ra.get(0, 2) == null);
    for (0..4) |_| {
        const cancelled = (try ReadAhead.create(testing.allocator, &.{source.source()}, test_budget, 1000)).?;
        for (0..1000) |id| _ = cancelled.add(0, @intCast(id));
        try cancelled.start();
        cancelled.cancel();
        try testing.expect(cancelled.get(0, 999) == null);
        cancelled.deinit();
    }
}
