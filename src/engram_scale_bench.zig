//! Disk-only companion to plan 75's Tier 1 table bench. No model or GPU initialization.
const std = @import("std");
const engram = @import("plugins/deepseek_v41/engram.zig");
const edir = @import("plugins/deepseek_v41/prompt/engram_direct.zig");
const testing = std.testing;

// Selection and mincore follow pp_step1_table.zig's Tier 1 harness.
fn selectedRows(grid: []const u32, rows: usize, layers: usize, slot: usize, cols: usize, out: []u32) ![]const u32 {
    if (slot >= layers or grid.len != rows * layers * cols or out.len < rows * cols) return error.InvalidGrid;
    for (0..rows) |i| @memcpy(out[i * cols ..][0..cols], grid[(i * layers + slot) * cols ..][0..cols]);
    const ids = out[0 .. rows * cols];
    std.mem.sort(u32, ids, {}, std.sort.asc(u32));
    var n: usize = 0;
    for (ids) |id| {
        if (n == 0 or ids[n - 1] != id) {
            ids[n] = id;
            n += 1;
        }
    }
    return ids[0..n];
}

fn rowShard(ids: []const u32, rank: usize, parts: usize) ![]const u32 {
    if (parts == 0 or rank >= parts) return error.InvalidShard;
    return ids[rank * ids.len / parts .. (rank + 1) * ids.len / parts];
}

extern "c" fn mincore(addr: *const anyopaque, len: usize, vec: [*]u8) c_int;

const Probe = struct {
    map: []align(std.heap.page_size_min) const u8,
    last: u64 = std.math.maxInt(u64),
    hit: bool = false,

    fn open(fd: c_int) !Probe {
        const size = std.c.lseek(fd, 0, std.c.SEEK.END);
        if (size <= 0) return error.TableSize;
        return .{ .map = try std.posix.mmap(null, @intCast(size), .{ .READ = true }, .{ .TYPE = .SHARED }, fd, 0) };
    }

    fn span(p: *Probe, off: u64, len: usize) !bool {
        if (off + len > p.map.len) return error.TableSize;
        const page = std.heap.pageSize();
        var at = off / page * page;
        var any = false;
        while (at < off + len) : (at += page) {
            if (p.last != at) {
                var vec: [1]u8 = .{0};
                if (mincore(p.map.ptr + @as(usize, @intCast(at)), @intCast(@min(page, p.map.len - at)), &vec) != 0) return error.MincoreFailed;
                p.last = at;
                p.hit = vec[0] & 1 != 0;
            }
            any = any or p.hit;
        }
        return any;
    }

    fn cached(p: *Probe, table: *const engram.Table, ids: []const u32) !usize {
        p.last = std.math.maxInt(u64);
        var count: usize = 0;
        for (ids) |id| if (try p.span(table.w_off + @as(u64, id) * table.w_stride, table.wlen)) {
            count += 1;
        };
        p.last = std.math.maxInt(u64);
        for (ids) |id| if (try p.span(table.s_off + @as(u64, id) * table.s_stride, table.slen)) {
            count += 1;
        };
        return count;
    }
};

fn loadConfig(io: std.Io, gpa: std.mem.Allocator, path: []const u8) !@import("plugins/deepseek_v41/config.zig").Config {
    const cfg = @import("plugins/deepseek_v41/config.zig");
    const abi = @import("abi/plugin.zig");
    var fs = try @import("sdk/util/pack_dir.zig").FsPackDir.open(io, path);
    defer fs.close();
    const pd = fs.packDir();
    const buf = try gpa.alloc(u8, 8 << 20);
    defer gpa.free(buf);
    var why: abi.Verdict = .{};
    const json = pd.readFile("config.json", buf, &why) orelse return error.ConfigUnreadable;
    var parsed: abi.Parsed = .{};
    if (!cfg.parse(&pd, .of(json), .{}, null, false, &parsed).isOk()) return error.ConfigRefused;
    defer cfg.freeParsed(&parsed);
    return cfg.of(&parsed).*;
}

fn jsonLine(gpa: std.mem.Allocator, value: anytype) !void {
    const bytes = try std.json.Stringify.valueAlloc(gpa, value, .{});
    defer gpa.free(bytes);
    var at: usize = 0;
    while (at < bytes.len) {
        const n = std.c.write(1, bytes[at..].ptr, bytes.len - at);
        if (n <= 0) return error.OutputFailed;
        at += @intCast(n);
    }
    if (std.c.write(1, "\n", 1) != 1) return error.OutputFailed;
}

fn nowNs() i128 {
    return std.Io.Timestamp.now(std.Io.Threaded.global_single_threaded.io(), .awake).toNanoseconds();
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len == 5 and std.mem.eql(u8, args[1], "tokenize")) {
        var tok = try @import("sdk/tokenizer/tokenizer.zig").loadTokenizerAny(io, gpa, args[2]);
        defer tok.deinit();
        const text = try std.Io.Dir.cwd().readFileAlloc(io, args[3], gpa, .limited(16 << 20));
        defer gpa.free(text);
        const ids = try tok.encode(gpa, text);
        defer gpa.free(ids);
        const raw = try gpa.alloc(u8, ids.len * 4);
        defer gpa.free(raw);
        for (ids, 0..) |id, i| std.mem.writeInt(u32, raw[i * 4 ..][0..4], id, .little);
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = args[4], .data = raw });
        try jsonLine(gpa, .{ .kind = "tokens", .count = ids.len, .text_bytes = text.len });
        return;
    }
    if (args.len != 9) return error.Usage_model_tokens_depth_readers_rank_shards_chunk;
    const depth = try std.fmt.parseInt(usize, args[3], 10);
    const readers = try std.fmt.parseInt(usize, args[4], 10);
    const rank = try std.fmt.parseInt(usize, args[5], 10);
    const shards = try std.fmt.parseInt(usize, args[6], 10);
    const chunk = try std.fmt.parseInt(usize, args[7], 10);
    const cold = std.mem.eql(u8, args[8], "cold");
    if (!cold and !std.mem.eql(u8, args[8], "warm-control")) return error.InvalidCacheMode;
    if (depth == 0 or depth > 131072 or chunk == 0 or chunk > depth or readers == 0 or readers > 32 or shards == 0 or shards > 4 or rank >= shards) return error.InvalidArgs;
    const cfg = try loadConfig(io, gpa, args[1]);
    if (cfg.dsv41_n_engram_layers != 2 or cfg.dsv41_engram_layers[0] != 1 or cfg.dsv41_engram_layers[1] != 14) return error.LayerMismatch;
    const map = try engram.loadTokenMap(gpa, args[1], cfg.vocab_size, cfg.dsv41_engram_compressed_vocab);
    defer gpa.free(map);
    const raw = try std.Io.Dir.cwd().readFileAlloc(io, args[2], gpa, .limited(16 << 20));
    defer gpa.free(raw);
    if (raw.len % 4 != 0 or raw.len / 4 < depth) return error.TokenFileTooShort;
    const comp = try gpa.alloc(u32, depth);
    defer gpa.free(comp);
    for (comp, 0..) |*id, i| {
        const t = std.mem.readInt(u32, raw[i * 4 ..][0..4], .little);
        if (t >= map.len) return error.TokenOutOfVocab;
        id.* = map[t];
    }
    const hash = try engram.Hash.init(&cfg, map[cfg.dsv41_engram_pad_id]);
    const cols: usize = hash.cols();
    const grid = try gpa.alloc(u32, chunk * 2 * cols);
    defer gpa.free(grid);
    const uniq_buf = try gpa.alloc(u32, chunk * cols);
    defer gpa.free(uniq_buf);
    var tables: [2]engram.Table = undefined;
    var opened: usize = 0;
    defer for (tables[0..opened]) |*table| table.close();
    for (&tables, 0..) |*table, slot| {
        table.* = try engram.Table.open(gpa, args[1], slot, cfg.dsv41_engram_layers[slot], cfg.dsv41_engram_rows[slot], cfg.dsv41_engram_head_dim);
        opened += 1;
        if (table.mode != .mxfp8 or table.records() or table.bits != 8 or table.group_size != 32) return error.UnsupportedTableLayout;
        table.readDirect();
        if (edir.blockOf(table.fd) == 0) return error.DirectUnavailable;
    }
    var pos: usize = 0;
    while (pos < depth) : (pos += chunk) {
        const n = @min(chunk, depth - pos);
        const t0 = nowNs();
        hash.rows(comp[0..pos], comp[pos .. pos + n], grid[0 .. n * 2 * cols]);
        const hash_ns = nowNs() - t0;
        for (&tables, 0..) |*table, slot| {
            const t1 = nowNs();
            const all_ids = try selectedRows(grid[0 .. n * 2 * cols], n, 2, slot, cols, uniq_buf);
            const ids = try rowShard(all_ids, rank, shards);
            const select_ns = nowNs() - t1;
            if (ids.len > (3 << 30) / (table.wlen + table.slen)) return error.BufferBudgetExceeded;
            var probe = try Probe.open(table.fd);
            defer std.posix.munmap(probe.map);
            const cached = try probe.cached(table, ids);
            if (cold and cached != 0) return error.CachedTablePages;
            const start = nowNs();
            const data = try table.readSorted(gpa, ids, readers);
            const read_ns = nowNs() - start;
            defer data.deinit(gpa);
            var checksum = std.hash.Wyhash.init(0);
            checksum.update(data.w);
            checksum.update(data.s);
            try jsonLine(gpa, .{
                .kind = "engram_scale",
                .depth = depth,
                .chunk = chunk,
                .pos = pos,
                .tokens = n,
                .layer = cfg.dsv41_engram_layers[slot],
                .readers = readers,
                .rank = rank,
                .shards = shards,
                .lookup_rows = n * cols,
                .distinct_rows = all_ids.len,
                .rows = ids.len,
                .payload_bytes = data.w.len + data.s.len,
                .table_bytes = table.rows * (table.w_stride + table.s_stride),
                .table_rows = table.rows,
                .code_bytes = table.wlen,
                .scale_bytes = table.slen,
                .read_ms = @as(f64, @floatFromInt(read_ns)) / 1e6,
                .hash_ms = @as(f64, @floatFromInt(hash_ns)) / 1e6,
                .select_ms = @as(f64, @floatFromInt(select_ns)) / 1e6,
                .cached_parts = cached,
                .block_bytes = edir.blockOf(table.fd),
                .first_row = if (ids.len > 0) ids[0] else null,
                .last_row = if (ids.len > 0) ids[ids.len - 1] else null,
                .row_digest = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(ids)),
                .checksum = checksum.final(),
            });
        }
    }
}

test "engram scale: sorted shards cover each distinct selected row exactly once" {
    const grid = [_]u32{ 9, 3, 100, 200, 3, 1, 200, 300, 5, 3, 400, 100 };
    var buf: [6]u32 = undefined;
    const ids = try selectedRows(&grid, 3, 2, 0, 2, &buf);
    try testing.expectEqualSlices(u32, &.{ 1, 3, 5, 9 }, ids);
    for ([_]usize{ 1, 2, 4, 8 }) |parts| {
        var at: usize = 0;
        for (0..parts) |rank| {
            const cut = try rowShard(ids, rank, parts);
            try testing.expectEqualSlices(u32, ids[at .. at + cut.len], cut);
            at += cut.len;
        }
        try testing.expectEqual(ids.len, at);
    }
    try testing.expectError(error.InvalidShard, rowShard(ids, 4, 4));
    try testing.expectError(error.InvalidShard, rowShard(ids, 0, 0));
    try testing.expectError(error.InvalidGrid, selectedRows(&grid, 3, 2, 2, 2, &buf));
    try testing.expectEqualSlices(u32, &.{ 100, 200, 300, 400 }, try selectedRows(&grid, 3, 2, 1, 2, &buf));
}

test "engram scale: every reader count returns codes and scales in selected order" {
    const io = std.Io.Threaded.global_single_threaded.io();
    var td = testing.tmpDir(.{});
    defer td.cleanup();
    var bytes: [8192]u8 = undefined;
    for (&bytes, 0..) |*b, i| b.* = @truncate(i * 7 + 3);
    try td.dir.writeFile(io, .{ .sub_path = "rows", .data = &bytes });
    var pb: [1024]u8 = undefined;
    const dir = pb[0..try td.dir.realPath(io, &pb)];
    const path = try std.fmt.allocPrintSentinel(testing.allocator, "{s}/rows", .{dir}, 0);
    defer testing.allocator.free(path);
    const fd = std.c.open(path, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
    try testing.expect(fd >= 0);
    defer _ = std.c.close(fd);
    var table: engram.Table = .{ .fd = fd, .rows = 1024, .head_dim = 4, .mode = .mxfp8, .bits = 8, .group_size = 4, .w_off = 0, .w_stride = 4, .wlen = 4, .s_off = 4096, .s_stride = 1, .slen = 1 };
    table.readDirect();
    defer edir.unregister(fd);
    var ids: [1024]u32 = undefined;
    for (&ids, 0..) |*id, i| id.* = @intCast(i);
    for ([_]usize{ 1, 2, 4, 8, 16, 32 }) |readers| {
        const raw = try table.readSorted(testing.allocator, &ids, readers);
        defer raw.deinit(testing.allocator);
        try testing.expectEqualSlices(u8, bytes[0..4096], raw.w);
        try testing.expectEqualSlices(u8, bytes[4096..5120], raw.s);
    }
}

test "engram scale: fresh APFS clones are cold and NOCACHE does not hide cached source pages" {
    const io = std.Io.Threaded.global_single_threaded.io();
    var td = testing.tmpDir(.{});
    defer td.cleanup();
    const bytes: [49152]u8 = @splat(37);
    try td.dir.writeFile(io, .{ .sub_path = "source", .data = &bytes });
    var pb: [1024]u8 = undefined;
    const dir = pb[0..try td.dir.realPath(io, &pb)];
    const src = try std.fmt.allocPrintSentinel(testing.allocator, "{s}/source", .{dir}, 0);
    defer testing.allocator.free(src);
    const dst = try std.fmt.allocPrintSentinel(testing.allocator, "{s}/clone", .{dir}, 0);
    defer testing.allocator.free(dst);
    const apfs = struct {
        extern "c" fn clonefile([*:0]const u8, [*:0]const u8, c_int) c_int;
    };
    try testing.expectEqual(@as(c_int, 0), apfs.clonefile(src, dst, 0));
    const source_fd = std.c.open(src, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
    try testing.expect(source_fd >= 0);
    defer _ = std.c.close(source_fd);
    var source: engram.Table = .{ .fd = source_fd, .rows = 1024, .head_dim = 4, .mode = .mxfp8, .bits = 8, .group_size = 4, .w_off = 0, .w_stride = 4, .wlen = 4, .s_off = 32768, .s_stride = 1, .slen = 1 };
    source.readDirect();
    defer edir.unregister(source_fd);
    var source_probe = try Probe.open(source_fd);
    defer std.posix.munmap(source_probe.map);
    try testing.expectEqual(@as(usize, 4), try source_probe.cached(&source, &.{ 0, 512 }));
    const clone_fd = std.c.open(dst, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
    try testing.expect(clone_fd >= 0);
    defer _ = std.c.close(clone_fd);
    var clone = source;
    clone.fd = clone_fd;
    clone.readDirect();
    defer edir.unregister(clone_fd);
    var clone_probe = try Probe.open(clone_fd);
    defer std.posix.munmap(clone_probe.map);
    try testing.expectEqual(@as(usize, 0), try clone_probe.cached(&clone, &.{ 0, 512 }));
    const data = try clone.readSorted(testing.allocator, &.{ 0, 512 }, 1);
    defer data.deinit(testing.allocator);
    try testing.expectEqualSlices(u8, &@as([8]u8, @splat(37)), data.w);
    try testing.expectEqual(@as(usize, 0), try clone_probe.cached(&clone, &.{ 0, 512 }));
}
