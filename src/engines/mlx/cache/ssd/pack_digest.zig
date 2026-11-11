//! Content digests of pack files, cached in a sidecar. A file's mtime only validates its cache entry. No MLX calls.

const std = @import("std");
const builtin = @import("builtin");
const env = @import("../../../../env.zig");

pub const Digest = [16]u8;
pub const SIDECAR_NAME = "pack-digest.json";
pub const ALGORITHM = "xxh3x2-128";
pub const READ_BYTES: usize = 4 * 1024 * 1024;
const MAX_WORKERS: usize = 8;
const READ_ALIGN = std.mem.Alignment.fromByteUnits(16 * 1024);
const MAX_SIDECAR_BYTES: usize = 16 * 1024 * 1024;
const SEED_LO: u64 = 0x6d6c_7873_6572_7601;
const SEED_HI: u64 = 0x7061_636b_6469_6702;

const GUARD_SLOTS = 16;
var guard_lock: std.Io.Mutex = .init;
var guard_keys: [GUARD_SLOTS]u64 = @splat(0);

pub const Target = struct { name: []const u8, size: u64, mtime_ns: i128 };
const Cached = struct { name: []const u8, size: u64, mtime_ns: i128, digest: []const u8 };
const Sidecar = struct { v: u8 = 1, algo: []const u8 = ALGORITHM, files: []const Cached = &.{} };

/// Digests known from the pack directory's sidecar and the cache directory's sidecar.
pub const Store = struct {
    arena: std.heap.ArenaAllocator,
    entries: []const Cached = &.{},

    pub fn open(a: std.mem.Allocator, io: std.Io, root: []const u8, cache_dir: []const u8) !Store {
        var arena = std.heap.ArenaAllocator.init(a);
        errdefer arena.deinit();
        const aa = arena.allocator();
        var all = std.ArrayList(Cached).empty;
        const pack_path = try std.fs.path.join(aa, &.{ root, SIDECAR_NAME });
        const cache_path = try cacheSidecarPath(aa, root, cache_dir);
        for ([_][]const u8{ pack_path, cache_path }) |path| try readSidecarOrNothing(aa, io, path, &all);
        return .{ .arena = arena, .entries = all.items };
    }

    pub fn deinit(self: *Store) void {
        self.arena.deinit();
    }

    pub fn get(self: *const Store, name: []const u8, size: u64, mtime_ns: i128) ?Digest {
        for (self.entries) |e| {
            if (e.size != size or e.mtime_ns != mtime_ns or e.digest.len != 2 * @sizeOf(Digest)) continue;
            if (!std.mem.eql(u8, e.name, name)) continue;
            var digest: Digest = undefined;
            _ = std.fmt.hexToBytes(&digest, e.digest) catch continue;
            return digest;
        }
        return null;
    }
};

fn cacheSidecarPath(a: std.mem.Allocator, root: []const u8, cache_dir: []const u8) ![]u8 {
    return std.fmt.allocPrint(a, "{s}/pack-digest-{x:0>16}.json", .{ cache_dir, std.hash.Wyhash.hash(0, root) });
}

fn readSidecarOrNothing(aa: std.mem.Allocator, io: std.Io, path: []const u8, all: *std.ArrayList(Cached)) !void {
    const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, aa, .limited(MAX_SIDECAR_BYTES)) catch return;
    const parsed = std.json.parseFromSliceLeaky(Sidecar, aa, bytes, .{ .ignore_unknown_fields = true }) catch return;
    if (parsed.v != 1 or !std.mem.eql(u8, parsed.algo, ALGORITHM)) return;
    try all.appendSlice(aa, parsed.files);
}

pub const Options = struct {
    workers: usize = MAX_WORKERS,
    /// Called under the persist lock after each file's digest reached the sidecar, with how many this run holds.
    after_persist: ?*const fn (known: usize) void = null,
};

/// Hashes the targets the sidecars have no valid entry for and writes each digest to the sidecar as soon as
/// its file is done, so a killed run resumes. Returns how many files it hashed.
/// The allocator is used from worker threads, so it must be thread-safe.
pub fn fill(a: std.mem.Allocator, io: std.Io, root: []const u8, cache_dir: []const u8, targets: []const Target) !usize {
    return fillWith(a, io, root, cache_dir, targets, .{});
}

pub fn fillWith(a: std.mem.Allocator, io: std.Io, root: []const u8, cache_dir: []const u8, targets: []const Target, opts: Options) !usize {
    var store = try Store.open(a, io, root, cache_dir);
    defer store.deinit();
    const digests = try a.alloc(?Digest, targets.len);
    defer a.free(digests);
    var todo = std.ArrayList(usize).empty;
    defer todo.deinit(a);
    for (targets, digests, 0..) |t, *slot, i| {
        slot.* = store.get(t.name, t.size, t.mtime_ns);
        if (slot.* == null) try todo.append(a, i);
    }
    if (todo.items.len == 0) return 0;
    var dir = try std.Io.Dir.openDirAbsolute(io, root, .{});
    defer dir.close(io);
    var job: Job = .{
        .a = a, .io = io, .dir = dir, .root = root, .cache_dir = cache_dir, .opts = opts,
        .targets = targets, .todo = todo.items, .digests = digests,
    };
    try job.run();
    if (job.persist_error) |err| return err;
    for (digests) |d| if (d == null) return error.PackDigestFailed;
    return todo.items.len;
}

const Job = struct {
    a: std.mem.Allocator,
    io: std.Io,
    dir: std.Io.Dir,
    root: []const u8,
    cache_dir: []const u8,
    opts: Options,
    targets: []const Target,
    todo: []const usize,
    digests: []?Digest,
    next: std.atomic.Value(usize) = .init(0),
    persist_lock: std.Io.Mutex = .init,
    persist_error: ?anyerror = null,

    fn run(self: *Job) !void {
        const cpus = std.Thread.getCpuCount() catch 1;
        const workers = @max(1, @min(self.todo.len, @min(cpus, @min(self.opts.workers, MAX_WORKERS))));
        const threads = try self.a.alloc(std.Thread, workers - 1);
        defer self.a.free(threads);
        var started: usize = 0;
        for (threads) |*t| {
            t.* = std.Thread.spawn(.{}, Job.backgroundWork, .{self}) catch break;
            started += 1;
        }
        self.work();
        for (threads[0..started]) |t| t.join();
    }

    fn backgroundWork(self: *Job) void {
        lowerPriority();
        self.work();
    }

    fn work(self: *Job) void {
        const buf = self.a.alignedAlloc(u8, READ_ALIGN, READ_BYTES) catch return;
        defer self.a.free(buf);
        while (true) {
            const k = self.next.fetchAdd(1, .monotonic);
            if (k >= self.todo.len) return;
            const i = self.todo[k];
            const digest = hashFile(self.io, self.dir, self.targets[i].name, buf, true) catch continue;
            self.record(i, digest);
        }
    }

    fn record(self: *Job, i: usize, digest: Digest) void {
        self.persist_lock.lockUncancelable(self.io);
        defer self.persist_lock.unlock(self.io);
        self.digests[i] = digest;
        persistOne(self.a, self.io, self.root, self.cache_dir, self.targets[i], digest) catch |err| {
            if (self.persist_error == null) self.persist_error = err;
            return;
        };
        if (self.opts.after_persist) |hook| hook(self.known());
    }

    fn known(self: *const Job) usize {
        var n: usize = 0;
        for (self.digests) |d| n += @intFromBool(d != null);
        return n;
    }
};

/// Puts the calling thread in the background QoS class so a digest never competes with serving (Darwin only).
pub fn lowerPriority() void {
    if (comptime builtin.os.tag.isDarwin()) _ = std.c.pthread_set_qos_class_self_np(.BACKGROUND, 0);
}

/// Streams the file in READ_BYTES reads through two differently seeded XxHash3 lanes (std has no 128-bit variant).
/// `nocache` makes the reads bypass the unified buffer cache (F_NOCACHE, Darwin only), so a pack that is
/// being served keeps its cached pages.
pub fn hashFile(io: std.Io, dir: std.Io.Dir, name: []const u8, buf: []u8, nocache: bool) !Digest {
    var file = try dir.openFile(io, name, .{});
    defer file.close(io);
    if (comptime builtin.os.tag.isDarwin()) {
        if (nocache) _ = std.c.fcntl(file.handle, std.c.F.NOCACHE, @as(c_int, 1));
    }
    var lo = std.hash.XxHash3.init(SEED_LO);
    var hi = std.hash.XxHash3.init(SEED_HI);
    var offset: u64 = 0;
    while (true) {
        const n = try file.readPositionalAll(io, buf, offset);
        lo.update(buf[0..n]);
        hi.update(buf[0..n]);
        offset += n;
        if (n < buf.len) break;
    }
    var out: Digest = undefined;
    std.mem.writeInt(u64, out[0..8], lo.final(), .little);
    std.mem.writeInt(u64, out[8..16], hi.final(), .little);
    return out;
}

/// Read-modify-write: the sidecars' other entries stay, this file's entry replaces any earlier one of its name.
fn persistOne(a: std.mem.Allocator, io: std.Io, root: []const u8, cache_dir: []const u8, target: Target, digest: Digest) !void {
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const aa = arena.allocator();
    const store = try Store.open(aa, io, root, cache_dir);
    var files = std.ArrayList(Cached).empty;
    for (store.entries) |e| {
        if (!std.mem.eql(u8, e.name, target.name)) try files.append(aa, e);
    }
    const hex = try aa.dupe(u8, &std.fmt.bytesToHex(digest, .lower));
    try files.append(aa, .{ .name = target.name, .size = target.size, .mtime_ns = target.mtime_ns, .digest = hex });
    const body = try std.json.Stringify.valueAlloc(aa, Sidecar{ .files = files.items }, .{});
    writeInDir(io, root, SIDECAR_NAME, body) catch {
        const cache_path = try cacheSidecarPath(aa, root, cache_dir);
        try std.Io.Dir.cwd().createDirPath(io, cache_dir);
        return writeInDir(io, cache_dir, std.fs.path.basename(cache_path), body);
    };
}

fn writeInDir(io: std.Io, dir_path: []const u8, name: []const u8, body: []const u8) !void {
    var dir = try std.Io.Dir.openDirAbsolute(io, dir_path, .{});
    defer dir.close(io);
    var atomic = try dir.createFileAtomic(io, name, .{ .replace = true });
    defer atomic.deinit(io);
    try atomic.file.writeStreamingAll(io, body);
    try atomic.replace(io);
}

/// `MLX_SERVE_PACK_DIGEST=0` turns every digest run off; unset is on.
pub fn enabled() bool {
    return !env.present(.MLX_SERVE_PACK_DIGEST) or env.on(.MLX_SERVE_PACK_DIGEST);
}

pub fn packKey(root: []const u8) u64 {
    return std.hash.Wyhash.hash(0, root) | 1;
}

/// The process-wide in-flight guard: true for the one caller that may digest `root`, until `releasePack`.
pub fn claimPack(io: std.Io, root: []const u8) bool {
    const key = packKey(root);
    guard_lock.lockUncancelable(io);
    defer guard_lock.unlock(io);
    var free: ?usize = null;
    for (guard_keys, 0..) |k, i| {
        if (k == key) return false;
        if (k == 0 and free == null) free = i;
    }
    const slot = free orelse return false;
    guard_keys[slot] = key;
    return true;
}

pub fn releasePack(io: std.Io, root: []const u8) void {
    releaseKey(io, packKey(root));
}

pub fn releaseKey(io: std.Io, key: u64) void {
    guard_lock.lockUncancelable(io);
    defer guard_lock.unlock(io);
    for (&guard_keys) |*k| {
        if (k.* == key) k.* = 0;
    }
}
