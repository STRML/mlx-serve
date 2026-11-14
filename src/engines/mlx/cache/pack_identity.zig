//! The pack identity a peer prefix-cache transfer stamps and checks: config.json plus sampled weight content.

const std = @import("std");
const io_util = @import("../../../sdk/util/io_util.zig");
const log = @import("../../../sdk/util/log.zig");

const block_bytes = 64 * 1024;
const block_count = 64;
const whole_file_max = 4 * 1024 * 1024;

/// The identity samples content; exact pack equality is what bench/studio-cluster/pack-hash.sh checks.
/// Null when `model_dir` is not a folder with a config.json (a GGUF file): such a model cannot transfer its cache.
pub fn cached(io: std.Io, allocator: std.mem.Allocator, slot: *?u64, model_dir: []const u8) !?u64 {
    if (slot.*) |id| return id;
    const t0 = io_util.nowUsMonotonic(io);
    var bytes_read: u64 = 0;
    const id = try compute(io, allocator, model_dir, &bytes_read) orelse return null;
    slot.* = id;
    const ms = @max(@divTrunc(io_util.nowUsMonotonic(io) - t0, 1000), 0);
    log.info("[hot-cache] pack identity: {d} bytes read ({d} ms)\n", .{ bytes_read, ms });
    return id;
}

fn compute(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8, bytes_read: *u64) !?u64 {
    var dir = std.Io.Dir.cwd().openDir(io, model_dir, .{ .iterate = true }) catch |err| switch (err) {
        error.NotDir => return null,
        else => return err,
    };
    defer dir.close(io);
    const cfg = dir.readFileAlloc(io, "config.json", allocator, .limited(16 * 1024 * 1024)) catch |err| switch (err) {
        error.FileNotFound => return null,
        else => return err,
    };
    defer allocator.free(cfg);
    bytes_read.* += cfg.len;
    var h = std.hash.Wyhash.init(0x7061_636b_6964_656e);
    h.update(cfg);

    const paths = try weightPaths(io, allocator, dir);
    defer {
        for (paths) |p| allocator.free(p);
        allocator.free(paths);
    }
    const buf = try allocator.alloc(u8, block_bytes);
    defer allocator.free(buf);
    for (paths) |p| {
        const f = try dir.openFile(io, p, .{});
        defer f.close(io);
        const size = try f.length(io);
        h.update(p);
        std.hash.autoHash(&h, size);
        bytes_read.* += try hashSamples(io, f, size, buf, &h);
    }
    return h.final();
}

fn weightPaths(io: std.Io, allocator: std.mem.Allocator, dir: std.Io.Dir) ![][]u8 {
    var out: std.ArrayList([]u8) = .empty;
    errdefer {
        for (out.items) |p| allocator.free(p);
        out.deinit(allocator);
    }
    var walker = try dir.walk(allocator);
    defer walker.deinit();
    while (try walker.next(io)) |e| {
        if (e.kind != .file and e.kind != .sym_link) continue;
        if (!std.mem.endsWith(u8, e.basename, ".safetensors") and !std.mem.endsWith(u8, e.basename, ".bin")) continue;
        const p = try allocator.dupe(u8, e.path);
        errdefer allocator.free(p);
        try out.append(allocator, p);
    }
    std.mem.sort([]u8, out.items, {}, pathLess);
    return out.toOwnedSlice(allocator);
}

fn pathLess(_: void, a: []u8, b: []u8) bool {
    return std.mem.lessThan(u8, a, b);
}

fn hashSamples(io: std.Io, f: std.Io.File, size: u64, buf: []u8, h: *std.hash.Wyhash) !u64 {
    if (size <= whole_file_max) {
        var off: u64 = 0;
        while (off < size) {
            const n = try f.readPositionalAll(io, buf, off);
            if (n == 0) break;
            h.update(buf[0..n]);
            off += n;
        }
        return off;
    }
    var total: u64 = 0;
    const span = size - block_bytes;
    for (0..block_count) |i| {
        const n = try f.readPositionalAll(io, buf, span * i / (block_count - 1));
        h.update(buf[0..n]);
        total += n;
    }
    return total;
}

// ── tests ──

const testing = std.testing;

const TestPack = struct {
    tmp: testing.TmpDir,
    buf: [std.fs.max_path_bytes]u8 = undefined,
    len: usize = 0,

    fn init(weight_bytes: usize) !TestPack {
        const io = testing.io;
        var p: TestPack = .{ .tmp = testing.tmpDir(.{ .iterate = true }) };
        errdefer p.tmp.cleanup();
        try p.tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = "{\"model_type\":\"x\"}" });
        const w = try testing.allocator.alloc(u8, weight_bytes);
        defer testing.allocator.free(w);
        for (w, 0..) |*b, i| b.* = @truncate(i *% 31);
        try p.tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors", .data = w });
        p.len = try p.tmp.dir.realPath(io, &p.buf);
        return p;
    }

    fn path(p: *const TestPack) []const u8 {
        return p.buf[0..p.len];
    }

    fn flipByte(p: *TestPack, offset: u64) !void {
        const f = try p.tmp.dir.openFile(testing.io, "model.safetensors", .{ .mode = .read_write });
        defer f.close(testing.io);
        var b: [1]u8 = undefined;
        _ = try f.readPositionalAll(testing.io, &b, offset);
        b[0] ^= 0x5a;
        try f.writePositionalAll(testing.io, &b, offset);
    }

    fn identity(p: *const TestPack) !?u64 {
        var slot: ?u64 = null;
        return cached(testing.io, testing.allocator, &slot, p.path());
    }
};

test "pack identity: a changed byte inside a sampled block of a large weight file changes it" {
    const size: usize = 5 * 1024 * 1024;
    var p = try TestPack.init(size);
    defer p.tmp.cleanup();
    const before = (try p.identity()).?;
    try p.flipByte(32 * (size - 64 * 1024) / 63 + 100);
    const after_mid = (try p.identity()).?;
    try testing.expect(before != after_mid);
    try p.flipByte(size - 1);
    try testing.expect(after_mid != (try p.identity()).?);
}

test "pack identity: a changed byte anywhere in a small weight file changes it" {
    var p = try TestPack.init(1024 * 1024);
    defer p.tmp.cleanup();
    const before = (try p.identity()).?;
    try p.flipByte(700 * 1024);
    try testing.expect(before != (try p.identity()).?);
}

test "pack identity: the second call reads no file" {
    var p = try TestPack.init(4096);
    defer p.tmp.cleanup();
    var slot: ?u64 = null;
    const first = (try cached(testing.io, testing.allocator, &slot, p.path())).?;
    try p.tmp.dir.deleteFile(testing.io, "config.json");
    try p.tmp.dir.deleteFile(testing.io, "model.safetensors");
    try testing.expectEqual(first, (try cached(testing.io, testing.allocator, &slot, p.path())).?);
}

test "pack identity: a file path or a folder without config.json has none" {
    var p = try TestPack.init(4096);
    defer p.tmp.cleanup();
    const file = try std.fmt.allocPrint(testing.allocator, "{s}/model.safetensors", .{p.path()});
    defer testing.allocator.free(file);
    var slot: ?u64 = null;
    try testing.expectEqual(@as(?u64, null), try cached(testing.io, testing.allocator, &slot, file));
    try p.tmp.dir.deleteFile(testing.io, "config.json");
    try testing.expectEqual(@as(?u64, null), try p.identity());
}
