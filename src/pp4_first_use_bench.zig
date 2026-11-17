//! CPU-only first-use Engram reader for a borrowed Studio with one separate model server.
const std = @import("std");
const engram = @import("plugins/deepseek_v41/engram.zig");
const edir = @import("plugins/deepseek_v41/prompt/engram_direct.zig");
const testing = std.testing;

const Entry = struct { row: u32, chunk: u32 };

const Catalog = struct {
    storage: []Entry,
    unique: []Entry,

    fn init(gpa: std.mem.Allocator, grid: []const u32, rows: usize, layers: usize, slot: usize, cols: usize, chunk: usize) !Catalog {
        if (chunk == 0 or chunk > 8192 or layers == 0 or slot >= layers or cols == 0 or grid.len != rows * layers * cols) return error.InvalidGrid;
        const entries = try gpa.alloc(Entry, rows * cols);
        errdefer gpa.free(entries);
        for (0..rows) |i| for (0..cols) |j| {
            entries[i * cols + j] = .{ .row = grid[(i * layers + slot) * cols + j], .chunk = @intCast(i / chunk) };
        };
        std.mem.sort(Entry, entries, {}, byRow);
        var n: usize = 0;
        for (entries) |entry| {
            if (n > 0 and entries[n - 1].row == entry.row) continue;
            entries[n] = entry;
            n += 1;
        }
        return .{ .storage = entries, .unique = entries[0..n] };
    }

    fn deinit(cat: *Catalog, gpa: std.mem.Allocator) void {
        gpa.free(cat.storage);
    }

    fn shard(cat: *Catalog, rank: usize) ![]Entry {
        if (rank >= 4) return error.InvalidShard;
        return cat.unique[rank * cat.unique.len / 4 .. (rank + 1) * cat.unique.len / 4];
    }

    fn schedule(cat: *Catalog, rank: usize) ![]Entry {
        const entries = try cat.shard(rank);
        std.mem.sort(Entry, entries, {}, byChunk);
        return entries;
    }

    fn byRow(_: void, a: Entry, b: Entry) bool {
        return a.row < b.row or (a.row == b.row and a.chunk < b.chunk);
    }

    fn byChunk(_: void, a: Entry, b: Entry) bool {
        return a.chunk < b.chunk or (a.chunk == b.chunk and a.row < b.row);
    }
};

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

    fn parts(p: *Probe, table: *const engram.Table, id: u32) !usize {
        const w = try p.span(table.w_off + @as(u64, id) * table.w_stride, table.wlen);
        const sc = try p.span(table.s_off + @as(u64, id) * table.s_stride, table.slen);
        return @intFromBool(w) + @as(usize, @intFromBool(sc));
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

fn epochNs() i128 {
    return std.Io.Timestamp.now(std.Io.Threaded.global_single_threaded.io(), .real).toNanoseconds();
}

const ReadResult = struct {
    data: engram.Table.Raw,
    cached_parts: usize,
    selected_pages: usize = 0,
    cached_pages: usize = 0,
    cold_rows: usize,
    cold_read_ns: i128,
    probe_ns: i128,
    read_start_ns: i128,
    read_start_epoch_ns: i128,
    read_end_ns: i128,
};

const Coverage = struct {
    rows: usize = 0,
    cold: usize = 0,

    fn add(self: *Coverage, rows: usize, cold: usize) void {
        self.rows += rows;
        self.cold += cold;
    }

    fn qualify(self: Coverage) !void {
        if (self.rows == 0 or self.cold * 10 < self.rows * 9) return error.InsufficientColdCoverage;
    }
};

fn readCold(gpa: std.mem.Allocator, table: *const engram.Table, probe: *Probe, ids: []const u32, readers: usize) !ReadResult {
    if (readers == 0 or readers > 32) return error.InvalidReaders;
    if (table.mode != .mxfp8) return error.UnsupportedTableLayout;
    const probe_start_ns = nowNs();
    const order = try gpa.alloc(u32, ids.len * 2);
    defer gpa.free(order);
    const warm = try gpa.alloc(bool, ids.len);
    defer gpa.free(warm);
    var cold_n: usize = 0;
    var warm_n: usize = 0;
    var cached_parts: usize = 0;
    probe.last = std.math.maxInt(u64);
    for (ids, 0..) |id, i| {
        const parts = try probe.parts(table, id);
        cached_parts += parts;
        warm[i] = parts > 0;
        if (warm[i]) {
            order[ids.len + warm_n] = id;
            warm_n += 1;
        } else {
            order[cold_n] = id;
            cold_n += 1;
        }
    }
    const read_start_ns = nowNs();
    const read_start_epoch_ns = epochNs();
    const cold = try table.readSorted(gpa, order[0..cold_n], readers);
    const cold_read_ns = if (cold_n > 0) nowNs() - read_start_ns else 0;
    if (warm_n == 0) return .{ .data = cold, .cached_parts = 0, .cold_rows = cold_n, .cold_read_ns = cold_read_ns, .probe_ns = read_start_ns - probe_start_ns, .read_start_ns = read_start_ns, .read_start_epoch_ns = read_start_epoch_ns, .read_end_ns = nowNs() };
    defer cold.deinit(gpa);
    const cached = try table.readSorted(gpa, order[ids.len..][0..warm_n], readers);
    defer cached.deinit(gpa);
    const w = try gpa.alignedAlloc(u8, .@"4", ids.len * table.wlen);
    errdefer gpa.free(w);
    const sc = try gpa.alignedAlloc(u8, .@"4", ids.len * table.slen);
    errdefer gpa.free(sc);
    const bi = try gpa.alignedAlloc(u8, .@"4", 0);
    var at: [2]usize = .{ 0, 0 };
    for (warm, 0..) |is_warm, i| {
        const part: usize = @intFromBool(is_warm);
        const data = if (is_warm) cached else cold;
        @memcpy(w[i * table.wlen ..][0..table.wlen], data.w[at[part] * table.wlen ..][0..table.wlen]);
        @memcpy(sc[i * table.slen ..][0..table.slen], data.s[at[part] * table.slen ..][0..table.slen]);
        at[part] += 1;
    }
    return .{ .data = .{ .w = w, .s = sc, .b = bi }, .cached_parts = cached_parts, .cold_rows = cold_n, .cold_read_ns = cold_read_ns, .probe_ns = read_start_ns - probe_start_ns, .read_start_ns = read_start_ns, .read_start_epoch_ns = read_start_epoch_ns, .read_end_ns = nowNs() };
}

fn readWarm(gpa: std.mem.Allocator, table: *const engram.Table, probe: *Probe, ids: []const u32, readers: usize) !ReadResult {
    if (readers == 0 or readers > 32) return error.InvalidReaders;
    const probe_start_ns = nowNs();
    var pages: std.AutoHashMapUnmanaged(u64, void) = .empty;
    defer pages.deinit(gpa);
    var cached_pages: usize = 0;
    var cached_parts: usize = 0;
    var cold_rows: usize = 0;
    probe.last = std.math.maxInt(u64);
    const page = std.heap.pageSize();
    for (ids) |id| {
        const parts = try probe.parts(table, id);
        cached_parts += parts;
        cold_rows += @intFromBool(parts == 0);
        const offsets = [_]u64{ table.w_off + @as(u64, id) * table.w_stride, table.s_off + @as(u64, id) * table.s_stride };
        for (offsets, [_]usize{ table.wlen, table.slen }) |off, len| {
            var at = off / page * page;
            while (at < off + len) : (at += page) {
                const entry = try pages.getOrPut(gpa, at);
                if (!entry.found_existing) cached_pages += @intFromBool(try probe.span(at, @min(page, probe.map.len - at)));
            }
        }
    }
    const read_start_ns = nowNs();
    const read_start_epoch_ns = epochNs();
    const data = try table.readSorted(gpa, ids, readers);
    return .{ .data = data, .cached_parts = cached_parts, .selected_pages = pages.count(), .cached_pages = cached_pages, .cold_rows = cold_rows, .cold_read_ns = 0, .probe_ns = read_start_ns - probe_start_ns, .read_start_ns = read_start_ns, .read_start_epoch_ns = read_start_epoch_ns, .read_end_ns = nowNs() };
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
        const tokens = try tok.encode(gpa, text);
        defer gpa.free(tokens);
        const raw = try gpa.alloc(u8, tokens.len * 4);
        defer gpa.free(raw);
        for (tokens, 0..) |id, i| std.mem.writeInt(u32, raw[i * 4 ..][0..4], id, .little);
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = args[4], .data = raw });
        try jsonLine(gpa, .{ .kind = "pp4_tokens", .count = tokens.len, .text_bytes = text.len });
        return;
    }
    const warm = args.len == 8 and std.mem.eql(u8, args[7], "--warm");
    if (args.len != 7 and !warm) return error.Usage_pack_tokens_depth_readers_rank_chunk_optional_warm;
    const depth = try std.fmt.parseInt(usize, args[3], 10);
    const readers = try std.fmt.parseInt(usize, args[4], 10);
    const rank = try std.fmt.parseInt(usize, args[5], 10);
    const chunk = try std.fmt.parseInt(usize, args[6], 10);
    if ((depth != 32768 and depth != 131072) or chunk != 2048 or readers == 0 or readers > 32 or rank >= 4) return error.InvalidArgs;
    const start_ns = nowNs();
    const start_epoch_ns = epochNs();
    const cfg = try loadConfig(io, gpa, args[1]);
    if (cfg.dsv41_n_engram_layers != 2 or cfg.dsv41_engram_layers[0] != 1 or cfg.dsv41_engram_layers[1] != 14) return error.LayerMismatch;
    const map = try engram.loadTokenMap(gpa, args[1], cfg.vocab_size, cfg.dsv41_engram_compressed_vocab);
    defer gpa.free(map);
    const raw_tokens = try std.Io.Dir.cwd().readFileAlloc(io, args[2], gpa, .limited(16 << 20));
    defer gpa.free(raw_tokens);
    if (raw_tokens.len % 4 != 0 or raw_tokens.len / 4 < depth) return error.TokenFileTooShort;
    const n = depth - 1;
    const comp = try gpa.alloc(u32, n);
    defer gpa.free(comp);
    if (cfg.dsv41_engram_pad_id >= map.len) return error.TokenOutOfVocab;
    for (comp, 0..) |*id, i| {
        const token = std.mem.readInt(u32, raw_tokens[i * 4 ..][0..4], .little);
        if (token >= map.len) return error.TokenOutOfVocab;
        id.* = map[token];
    }
    const hash = try engram.Hash.init(&cfg, map[cfg.dsv41_engram_pad_id]);
    const cols: usize = hash.cols();
    if (cols != 24) return error.HashGeometry;
    const grid = try gpa.alloc(u32, n * 2 * cols);
    defer gpa.free(grid);
    const hash_start = nowNs();
    hash.rows(&.{}, comp, grid);
    const hash_ns = nowNs() - hash_start;
    const catalog_start = nowNs();
    var catalogs: [2]Catalog = undefined;
    var catalog_count: usize = 0;
    defer for (catalogs[0..catalog_count]) |*catalog| catalog.deinit(gpa);
    var entries: [2][]Entry = undefined;
    var catalog_digest: [2]u64 = undefined;
    var shard_digest: [2]u64 = undefined;
    for (&catalogs, 0..) |*catalog, slot| {
        catalog.* = try Catalog.init(gpa, grid, n, 2, slot, cols, chunk);
        catalog_count += 1;
        catalog_digest[slot] = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(catalog.unique));
        shard_digest[slot] = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(try catalog.shard(rank)));
        entries[slot] = try catalog.schedule(rank);
    }
    const catalog_ns = nowNs() - catalog_start;
    var tables: [2]engram.Table = undefined;
    var table_count: usize = 0;
    defer for (tables[0..table_count]) |*table| table.close();
    for (&tables, 0..) |*table, slot| {
        table.* = try engram.Table.open(gpa, args[1], slot, cfg.dsv41_engram_layers[slot], cfg.dsv41_engram_rows[slot], cfg.dsv41_engram_head_dim);
        table_count += 1;
        if (table.mode != .mxfp8 or table.records() or table.bits != 8 or table.group_size != 32 or table.wlen != 256 or table.slen != 8) return error.UnsupportedTableLayout;
        if (!warm) {
            table.readDirect();
            if (edir.blockOf(table.fd) == 0) return error.DirectUnavailable;
        }
    }
    var probes: [2]Probe = undefined;
    var probe_count: usize = 0;
    defer for (probes[0..probe_count]) |*probe| std.posix.munmap(probe.map);
    for (&probes, tables) |*probe, table| {
        probe.* = try Probe.open(table.fd);
        probe_count += 1;
    }
    var owners: [2][]u8 = undefined;
    var owner_count: usize = 0;
    defer for (owners[0..owner_count]) |owner| gpa.free(owner);
    for (&owners, entries) |*owner, list| {
        owner.* = try gpa.alloc(u8, list.len * 264);
        owner_count += 1;
    }
    const ready_ns = nowNs();
    try jsonLine(gpa, .{
        .kind = "pp4_catalog",
        .warm = warm,
        .schema = 2,
        .depth = depth,
        .prefill_tokens = n,
        .chunk_width = chunk,
        .rank = rank,
        .shards = 4,
        .readers = readers,
        .start_ns = start_ns,
        .start_epoch_ns = start_epoch_ns,
        .ready_ns = ready_ns,
        .hash_ns = hash_ns,
        .catalog_ns = catalog_ns,
        .prepare_ns = ready_ns - start_ns,
        .distinct_rows = [_]usize{ catalogs[0].unique.len, catalogs[1].unique.len },
        .shard_rows = [_]usize{ entries[0].len, entries[1].len },
        .catalog_digest = catalog_digest,
        .shard_digest = shard_digest,
        .owner_bytes = owners[0].len + owners[1].len,
        .token_digest = std.hash.Wyhash.hash(0, raw_tokens[0 .. depth * 4]),
    });
    const ids = try gpa.alloc(u32, chunk * cols);
    defer gpa.free(ids);
    var at: [2]usize = .{ 0, 0 };
    var copied: [2]usize = .{ 0, 0 };
    var coverage: Coverage = .{};
    var chunk_index: usize = 0;
    while (chunk_index * chunk < n) : (chunk_index += 1) {
        for (&tables, &probes, 0..) |*table, *probe, slot| {
            const begin = at[slot];
            while (at[slot] < entries[slot].len and entries[slot][at[slot]].chunk == chunk_index) : (at[slot] += 1) {}
            const bucket = entries[slot][begin..at[slot]];
            for (bucket, 0..) |entry, i| ids[i] = entry.row;
            const selected = ids[0..bucket.len];
            const result = if (warm) try readWarm(gpa, table, probe, selected, readers) else try readCold(gpa, table, probe, selected, readers);
            coverage.add(selected.len, result.cold_rows);
            const data = result.data;
            defer data.deinit(gpa);
            const copy_start_ns = nowNs();
            const owner = owners[slot][copied[slot]..][0 .. data.w.len + data.s.len];
            @memcpy(owner[0..data.w.len], data.w);
            @memcpy(owner[data.w.len..], data.s);
            copied[slot] += owner.len;
            const copy_end_ns = nowNs();
            try jsonLine(gpa, .{
                .kind = "pp4_bucket",
                .schema = 2,
                .depth = depth,
                .rank = rank,
                .shards = 4,
                .readers = readers,
                .chunk_width = chunk,
                .chunk_index = chunk_index,
                .p0 = chunk_index * chunk,
                .width = @min(chunk, n - chunk_index * chunk),
                .layer = cfg.dsv41_engram_layers[slot],
                .rows = selected.len,
                .payload_bytes = owner.len,
                .warm = warm,
                .cached_parts = result.cached_parts,
                .selected_pages = result.selected_pages,
                .cached_pages = result.cached_pages,
                .resident_fraction = if (warm and result.selected_pages > 0) @as(f64, @floatFromInt(result.cached_pages)) / @as(f64, @floatFromInt(result.selected_pages)) else @as(f64, 0),
                .cold_rows = result.cold_rows,
                .cold_read_ns = result.cold_read_ns,
                .read_start_ns = result.read_start_ns,
                .read_start_epoch_ns = result.read_start_epoch_ns,
                .read_end_ns = result.read_end_ns,
                .copy_end_ns = copy_end_ns,
                .read_ns = result.read_end_ns - result.read_start_ns,
                .copy_ns = copy_end_ns - copy_start_ns,
                .probe_ns = result.probe_ns,
                .elapsed_ns = copy_end_ns - ready_ns,
                .row_digest = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(selected)),
                .payload_digest = std.hash.Wyhash.hash(0, owner),
                .first_row = if (selected.len > 0) selected[0] else null,
                .last_row = if (selected.len > 0) selected[selected.len - 1] else null,
            });
        }
    }
    for (copied, owners, at, entries) |bytes, owner, end, list| {
        if (bytes != owner.len or end != list.len) return error.IncompleteSchedule;
    }
    try jsonLine(gpa, .{ .kind = "pp4_reader_coverage", .rows = coverage.rows, .cold_rows = coverage.cold, .minimum_cold_fraction = if (warm) @as(f64, 0) else @as(f64, 0.9) });
    if (!warm) try coverage.qualify();
    try jsonLine(gpa, .{ .kind = "pp4_reader_done", .schema = 2, .depth = depth, .rank = rank, .elapsed_ns = nowNs() - start_ns });
}

test "pp4 first use: earliest chunk dedup preserves global sorted shard ownership" {
    const grid = [_]u32{ 9, 3, 100, 200, 3, 1, 200, 300, 5, 3, 400, 100, 9, 7, 500, 100 };
    var cat = try Catalog.init(testing.allocator, &grid, 4, 2, 0, 2, 1);
    defer cat.deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 5), cat.unique.len);
    const rows = [_]u32{ 1, 3, 5, 7, 9 };
    const chunks = [_]u32{ 1, 0, 2, 3, 0 };
    for (cat.unique, rows, chunks) |entry, row, chunk| {
        try testing.expectEqual(row, entry.row);
        try testing.expectEqual(chunk, entry.chunk);
    }
    for (0..4) |rank| {
        const shard = try cat.shard(rank);
        const expected = switch (rank) {
            0 => &[_]u32{1},
            1 => &[_]u32{3},
            2 => &[_]u32{5},
            else => &[_]u32{ 7, 9 },
        };
        try testing.expectEqual(expected.len, shard.len);
        for (shard, expected) |entry, row| try testing.expectEqual(row, entry.row);
    }
    var other = try Catalog.init(testing.allocator, &grid, 4, 2, 1, 2, 1);
    defer other.deinit(testing.allocator);
    for (other.unique, [_]u32{ 100, 200, 300, 400, 500 }) |entry, row| try testing.expectEqual(row, entry.row);
    try testing.expectError(error.InvalidShard, cat.shard(4));
    try testing.expectError(error.InvalidGrid, Catalog.init(testing.allocator, &grid, 4, 2, 2, 2, 1));
    try testing.expectError(error.InvalidGrid, Catalog.init(testing.allocator, &grid, 4, 2, 0, 2, 0));
}

test "pp4 first use: scheduled quarter keeps first-use order without changing ownership" {
    var cat = try Catalog.init(testing.allocator, &.{ 7, 5, 1, 3, 7, 9, 9, 5 }, 4, 1, 0, 2, 1);
    defer cat.deinit(testing.allocator);
    const entries = try cat.schedule(3);
    try testing.expectEqual(@as(usize, 2), entries.len);
    try testing.expectEqual(@as(u32, 7), entries[0].row);
    try testing.expectEqual(@as(u32, 0), entries[0].chunk);
    try testing.expectEqual(@as(u32, 9), entries[1].row);
    try testing.expectEqual(@as(u32, 2), entries[1].chunk);
}

test "pp4 first use: cold clone and cached rows deliver exact bytes with separate cold timing" {
    const io = std.Io.Threaded.global_single_threaded.io();
    var td = testing.tmpDir(.{});
    defer td.cleanup();
    var bytes: [49152]u8 = undefined;
    for (&bytes, 0..) |*byte, i| byte.* = @truncate((i * 7 + 3) ^ (i >> 8));
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
    const warm = try readCold(testing.allocator, &source, &source_probe, &.{ 0, 512 }, 1);
    defer warm.data.deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 4), warm.cached_parts);
    try testing.expectEqual(@as(usize, 0), warm.cold_rows);
    try testing.expectEqual(@as(i128, 0), warm.cold_read_ns);
    try testing.expectEqualSlices(u8, bytes[0..4], warm.data.w[0..4]);
    const clone_fd = std.c.open(dst, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
    try testing.expect(clone_fd >= 0);
    defer _ = std.c.close(clone_fd);
    var clone = source;
    clone.fd = clone_fd;
    clone.readDirect();
    defer edir.unregister(clone_fd);
    var clone_probe = try Probe.open(clone_fd);
    defer std.posix.munmap(clone_probe.map);
    const result = try readCold(testing.allocator, &clone, &clone_probe, &.{ 0, 512 }, 1);
    const raw = result.data;
    defer raw.deinit(testing.allocator);
    try testing.expectEqualSlices(u8, bytes[0..4], raw.w[0..4]);
    try testing.expectEqualSlices(u8, bytes[2048..2052], raw.w[4..8]);
    try testing.expectEqualSlices(u8, &.{ bytes[32768], bytes[33280] }, raw.s);
    try testing.expectEqual(@as(usize, 0), try clone_probe.parts(&clone, 0));
    const first_page: *const volatile u8 = @ptrCast(clone_probe.map.ptr);
    try testing.expectEqual(bytes[0], first_page.*);
    clone.rows = 16384;
    const mixed = try readCold(testing.allocator, &clone, &clone_probe, &.{ 0, 8192 }, 1);
    defer mixed.data.deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 1), mixed.cold_rows);
    try testing.expectEqual(@as(usize, 1), mixed.cached_parts);
    try testing.expect(mixed.cold_read_ns > 0);
    try testing.expectEqualSlices(u8, bytes[0..4], mixed.data.w[0..4]);
    try testing.expectEqualSlices(u8, bytes[32768..32772], mixed.data.w[4..8]);
    try testing.expectEqualSlices(u8, &.{ bytes[32768], bytes[40960] }, mixed.data.s);
}

test "pp4 first use: cold coverage qualifies ninety percent over the full schedule" {
    var coverage: Coverage = .{};
    coverage.add(100, 90);
    try coverage.qualify();
    coverage.add(100, 89);
    try testing.expectError(error.InsufficientColdCoverage, coverage.qualify());
    coverage.add(100, 100);
    try coverage.qualify();
    try testing.expectError(error.InsufficientColdCoverage, (Coverage{}).qualify());
}

test "pp4 first use: warm buffered delivery counts unique resident pages and preserves payload" {
    const io = std.Io.Threaded.global_single_threaded.io();
    var td = testing.tmpDir(.{});
    defer td.cleanup();
    var bytes: [49152]u8 = undefined;
    for (&bytes, 0..) |*byte, i| byte.* = @truncate(i * 11 + 9);
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
    const fd = std.c.open(dst, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
    try testing.expect(fd >= 0);
    defer _ = std.c.close(fd);
    const table: engram.Table = .{ .fd = fd, .rows = 1024, .head_dim = 4, .mode = .mxfp8, .bits = 8, .group_size = 4, .w_off = 0, .w_stride = 4, .wlen = 4, .s_off = 32768, .s_stride = 1, .slen = 1 };
    var probe = try Probe.open(fd);
    defer std.posix.munmap(probe.map);
    const first: *const volatile u8 = @ptrCast(probe.map.ptr);
    try testing.expectEqual(bytes[0], first.*);
    const result = try readWarm(testing.allocator, &table, &probe, &.{ 0, 512 }, 2);
    defer result.data.deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 2), result.selected_pages);
    try testing.expectEqual(@as(usize, 1), result.cached_pages);
    try testing.expectEqual(@as(usize, 2), result.cached_parts);
    try testing.expectEqualSlices(u8, bytes[0..4], result.data.w[0..4]);
    try testing.expectEqualSlices(u8, bytes[2048..2052], result.data.w[4..8]);
    try testing.expectEqualSlices(u8, &.{ bytes[32768], bytes[33280] }, result.data.s);
    const cached = try readWarm(testing.allocator, &table, &probe, &.{ 0, 512 }, 2);
    defer cached.data.deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 2), cached.selected_pages);
    try testing.expectEqual(@as(usize, 2), cached.cached_pages);
    try testing.expectEqual(@as(usize, 0), cached.cold_rows);
}
