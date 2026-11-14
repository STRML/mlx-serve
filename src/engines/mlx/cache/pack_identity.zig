//! The pack identity a peer prefix-cache transfer stamps and checks, hashed on its own thread.

const std = @import("std");
const builtin = @import("builtin");
const io_util = @import("../../../sdk/util/io_util.zig");
const log = @import("../../../sdk/util/log.zig");

const chunk_bytes = 8 * 1024 * 1024;
const f_nocache: c_int = 48;
const qos_class_utility: c_uint = 0x11;

extern "c" fn pthread_set_qos_class_self_np(qos: c_uint, relative_priority: c_int) c_int;

pub var hold_for_test: std.atomic.Value(bool) = .init(false);

pub const Status = union(enum) { pending, unsupported, ready: u64 };

const State = enum(u8) { pending, ready, failed, cancelled };

/// The identity hashes all weight content in the background; bench/studio-cluster/pack-hash.sh checks pack equality across boxes.
pub const Job = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    path: []u8,
    thread: std.Thread = undefined,
    cancel: std.atomic.Value(bool) = .init(false),
    state: std.atomic.Value(State) = .init(.pending),
    id: u64 = 0,

    fn run(job: *Job) void {
        if (comptime builtin.os.tag == .macos) _ = pthread_set_qos_class_self_np(qos_class_utility, 0);
        const t0 = io_util.nowUsMonotonic(job.io);
        var bytes_read: u64 = 0;
        const id = hashPack(job, &bytes_read) catch |err| {
            if (err == error.Canceled) {
                job.state.store(.cancelled, .release);
                return;
            }
            log.warn("[hot-cache] pack identity unavailable for {s}: {s}\n", .{ job.path, @errorName(err) });
            job.state.store(.failed, .release);
            return;
        };
        job.id = id;
        job.state.store(.ready, .release);
        const ms = @max(@divTrunc(io_util.nowUsMonotonic(job.io) - t0, 1000), 0);
        log.info("[hot-cache] pack identity: {d} bytes read ({d} ms)\n", .{ bytes_read, ms });
    }
};

/// Starts hashing `path` for a freshly loaded model. The inference thread owns `slot`.
pub fn start(io: std.Io, allocator: std.mem.Allocator, slot: *?*Job, path: []const u8) !void {
    drop(slot);
    const job = try allocator.create(Job);
    errdefer allocator.destroy(job);
    job.* = .{ .allocator = allocator, .io = io, .path = try allocator.dupe(u8, path) };
    errdefer allocator.free(job.path);
    job.thread = try std.Thread.spawn(.{}, Job.run, .{job});
    slot.* = job;
}

/// Never blocks. A load that started no hash, or whose hash failed, cannot transfer its cache.
pub fn status(job: ?*const Job) Status {
    const j = job orelse return .unsupported;
    return switch (j.state.load(.acquire)) {
        .pending => .pending,
        .ready => .{ .ready = j.id },
        .failed, .cancelled => .unsupported,
    };
}

/// Cancels the hash, which checks between blocks, and joins its thread before freeing what it reads.
pub fn drop(slot: *?*Job) void {
    const job = slot.* orelse return;
    slot.* = null;
    job.cancel.store(true, .release);
    job.thread.join();
    job.allocator.free(job.path);
    job.allocator.destroy(job);
}

pub fn waitReady(io: std.Io, job: ?*const Job) !u64 {
    while (true) {
        switch (status(job)) {
            .ready => |id| return id,
            .unsupported => return error.PackIdentityUnsupported,
            .pending => io.sleep(.fromMilliseconds(1), .awake) catch {},
        }
    }
}

const Hasher = struct {
    job: *Job,
    h: std.hash.Wyhash = .init(0x7061_636b_6964_656e),
    buf: []u8,
    bytes_read: u64 = 0,

    fn file(self: *Hasher, dir: std.Io.Dir, sub_path: []const u8, name: []const u8) !void {
        const io = self.job.io;
        const f = try dir.openFile(io, sub_path, .{});
        defer f.close(io);
        if (comptime builtin.os.tag == .macos) _ = std.c.fcntl(f.handle, f_nocache, @as(c_int, 1));
        const st = try f.stat(io);
        if (st.kind != .file) return error.NotARegularFile;
        const size = st.size;
        self.h.update(name);
        std.hash.autoHash(&self.h, size);
        var off: u64 = 0;
        while (off < size) {
            while (hold_for_test.load(.acquire) and !self.job.cancel.load(.acquire)) {
                io.sleep(.fromMilliseconds(1), .awake) catch {};
            }
            if (self.job.cancel.load(.acquire)) return error.Canceled;
            const n = try f.readPositionalAll(io, self.buf, off);
            if (n == 0) return error.UnexpectedEndOfFile;
            self.h.update(self.buf[0..n]);
            off += n;
        }
        self.bytes_read += off;
    }
};

fn hashPack(job: *Job, bytes_read: *u64) !u64 {
    const io = job.io;
    const buf = try job.allocator.alloc(u8, chunk_bytes);
    defer job.allocator.free(buf);
    var hasher: Hasher = .{ .job = job, .buf = buf };
    const cwd = std.Io.Dir.cwd();
    const st = try cwd.statFile(io, job.path, .{});
    switch (st.kind) {
        .file => try hasher.file(cwd, job.path, ""),
        .directory => {
            var dir = try cwd.openDir(io, job.path, .{ .iterate = true });
            defer dir.close(io);
            hasher.file(dir, "config.json", "config.json") catch |err| switch (err) {
                error.FileNotFound => {},
                else => return err,
            };
            const paths = try weightPaths(io, job.allocator, dir);
            defer {
                for (paths) |p| job.allocator.free(p);
                job.allocator.free(paths);
            }
            for (paths) |p| try hasher.file(dir, p, p);
        },
        else => return error.UnsupportedPackPath,
    }
    bytes_read.* = hasher.bytes_read;
    return hasher.h.final();
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
        if (!isWeightFile(e.basename)) continue;
        const p = try allocator.dupe(u8, e.path);
        errdefer allocator.free(p);
        try out.append(allocator, p);
    }
    std.mem.sort([]u8, out.items, {}, pathLess);
    return out.toOwnedSlice(allocator);
}

fn isWeightFile(name: []const u8) bool {
    inline for (.{ ".safetensors", ".bin", ".gguf" }) |ext| {
        if (std.mem.endsWith(u8, name, ext)) return true;
    }
    return false;
}

fn pathLess(_: void, a: []u8, b: []u8) bool {
    return std.mem.lessThan(u8, a, b);
}

// ── tests ──

const testing = std.testing;

const TestDir = struct {
    tmp: testing.TmpDir,
    buf: [std.fs.max_path_bytes]u8 = undefined,
    len: usize = 0,

    fn init() !TestDir {
        var d: TestDir = .{ .tmp = testing.tmpDir(.{ .iterate = true }) };
        d.len = try d.tmp.dir.realPath(testing.io, &d.buf);
        return d;
    }

    fn path(d: *const TestDir) []const u8 {
        return d.buf[0..d.len];
    }

    fn write(d: *TestDir, name: []const u8, size: usize, flip: ?usize) !void {
        const w = try testing.allocator.alloc(u8, size);
        defer testing.allocator.free(w);
        for (w, 0..) |*b, i| b.* = @truncate(i *% 31);
        if (flip) |at| w[at] ^= 0x5a;
        try d.tmp.dir.writeFile(testing.io, .{ .sub_path = name, .data = w });
    }

    fn sub(d: *const TestDir, name: []const u8) ![]u8 {
        return std.fmt.allocPrint(testing.allocator, "{s}/{s}", .{ d.path(), name });
    }
};

fn identityOf(path: []const u8) !u64 {
    var slot: ?*Job = null;
    defer drop(&slot);
    try start(testing.io, testing.allocator, &slot, path);
    return waitReady(testing.io, slot);
}

fn safetensorsPack(flip: ?usize) !TestDir {
    var d = try TestDir.init();
    errdefer d.tmp.cleanup();
    try d.tmp.dir.writeFile(testing.io, .{ .sub_path = "config.json", .data = "{\"model_type\":\"x\"}" });
    try d.write("model.safetensors", 5 * 1024 * 1024, flip);
    return d;
}

test "pack identity: two 5 MiB packs that differ only at byte 65536 differ" {
    var a = try safetensorsPack(null);
    defer a.tmp.cleanup();
    var b = try safetensorsPack(65536);
    defer b.tmp.cleanup();
    try testing.expect(try identityOf(a.path()) != try identityOf(b.path()));
}

test "pack identity: GGUF folders with config.json and different .gguf files differ" {
    var dirs: [2]TestDir = .{ try TestDir.init(), try TestDir.init() };
    defer for (&dirs) |*d| d.tmp.cleanup();
    var ids: [2]u64 = undefined;
    for (&dirs, &ids, 0..) |*d, *id, i| {
        try d.tmp.dir.writeFile(testing.io, .{ .sub_path = "config.json", .data = "{\"model_type\":\"x\"}" });
        try d.write("a.gguf", 4096, null);
        try d.write("b.gguf", 4096, if (i == 0) null else 100);
        id.* = try identityOf(d.path());
    }
    try testing.expect(ids[0] != ids[1]);
}

test "pack identity: a file path gets one, and its content decides it" {
    var d = try TestDir.init();
    defer d.tmp.cleanup();
    try d.write("a.gguf", 4096, null);
    try d.write("b.gguf", 4096, 7);
    const a = try d.sub("a.gguf");
    defer testing.allocator.free(a);
    const b = try d.sub("b.gguf");
    defer testing.allocator.free(b);
    try testing.expect(try identityOf(a) != try identityOf(b));
}

test "pack identity: a path that does not open is unsupported" {
    var d = try TestDir.init();
    defer d.tmp.cleanup();
    const missing = try d.sub("missing");
    defer testing.allocator.free(missing);
    try testing.expectError(error.PackIdentityUnsupported, identityOf(missing));
}

test "pack identity: a second read of the status touches no file" {
    var d = try safetensorsPack(null);
    defer d.tmp.cleanup();
    var slot: ?*Job = null;
    defer drop(&slot);
    try start(testing.io, testing.allocator, &slot, d.path());
    const first = try waitReady(testing.io, slot);
    try d.tmp.dir.deleteFile(testing.io, "config.json");
    try d.tmp.dir.deleteFile(testing.io, "model.safetensors");
    try testing.expectEqual(Status{ .ready = first }, status(slot));
}

test "pack identity: it stays pending while the test seam holds it" {
    var d = try safetensorsPack(null);
    defer d.tmp.cleanup();
    hold_for_test.store(true, .release);
    defer hold_for_test.store(false, .release);
    var slot: ?*Job = null;
    defer drop(&slot);
    try start(testing.io, testing.allocator, &slot, d.path());
    try testing.io.sleep(.fromMilliseconds(50), .awake);
    try testing.expectEqual(Status.pending, status(slot));
    hold_for_test.store(false, .release);
    _ = try waitReady(testing.io, slot);
}

test "pack identity: dropping a job mid-hash joins its thread and frees it" {
    var d = try safetensorsPack(null);
    defer d.tmp.cleanup();
    hold_for_test.store(true, .release);
    defer hold_for_test.store(false, .release);
    var slot: ?*Job = null;
    try start(testing.io, testing.allocator, &slot, d.path());
    try testing.io.sleep(.fromMilliseconds(20), .awake);
    drop(&slot);
    try testing.expectEqual(@as(?*Job, null), slot);
    try testing.expectEqual(Status.unsupported, status(slot));
}

test "pack identity: a file that shrinks while it is read fails the hash" {
    var d = try safetensorsPack(null);
    defer d.tmp.cleanup();
    hold_for_test.store(true, .release);
    defer hold_for_test.store(false, .release);
    var slot: ?*Job = null;
    defer drop(&slot);
    try start(testing.io, testing.allocator, &slot, d.path());
    try testing.io.sleep(.fromMilliseconds(20), .awake);
    try d.tmp.dir.writeFile(testing.io, .{ .sub_path = "config.json", .data = "" });
    hold_for_test.store(false, .release);
    try testing.expectError(error.PackIdentityUnsupported, waitReady(testing.io, slot));
}
