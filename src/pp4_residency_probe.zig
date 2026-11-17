//! CPU-only Engram page residency; mapped table bytes are never dereferenced.
const std = @import("std");
const reader = @import("pp4_first_use_bench.zig");
const engram = @import("plugins/deepseek_v41/engram.zig");
const testing = std.testing;
const Config = @import("plugins/deepseek_v41/config.zig").Config;

const allocation_cap = 384 << 20;
const mapping_cap = reader.Probe.mapping_cap;

const BoundedPages = struct {
    live: usize = 0,
    peak: usize = 0,

    fn allocator(self: *BoundedPages) std.mem.Allocator {
        return .{ .ptr = self, .vtable = &.{ .alloc = alloc, .resize = std.mem.Allocator.noResize, .remap = std.mem.Allocator.noRemap, .free = free } };
    }

    fn charge(len: usize, alignment: std.mem.Alignment) usize {
        const page = std.heap.pageSize();
        return std.mem.alignForward(usize, len, page) + (alignment.toByteUnits() -| page);
    }

    fn alloc(ctx: *anyopaque, len: usize, alignment: std.mem.Alignment, ra: usize) ?[*]u8 {
        const self: *BoundedPages = @ptrCast(@alignCast(ctx));
        const bytes = charge(len, alignment);
        if (bytes > allocation_cap - self.live) return null;
        const ptr = std.heap.page_allocator.rawAlloc(len, alignment, ra) orelse return null;
        self.live += bytes;
        self.peak = @max(self.peak, self.live);
        return ptr;
    }

    fn free(ctx: *anyopaque, buf: []u8, alignment: std.mem.Alignment, ra: usize) void {
        const self: *BoundedPages = @ptrCast(@alignCast(ctx));
        std.heap.page_allocator.rawFree(buf, alignment, ra);
        self.live -= charge(buf.len, alignment);
    }
};

const Fraction = struct {
    pages: usize,
    resident_pages: usize,
    resident_fraction: f64,

    fn of(pages: usize, resident: usize) Fraction {
        return .{ .pages = pages, .resident_pages = resident, .resident_fraction = if (pages == 0) 0 else @as(f64, @floatFromInt(resident)) / @as(f64, @floatFromInt(pages)) };
    }
};

fn selectedPart(bitmap: std.DynamicBitSet, table: *const engram.Table, entries: []const reader.Entry, off: u64, stride: u64, len: usize) !Fraction {
    const page = std.heap.pageSize();
    var pages: usize = 0;
    var resident: usize = 0;
    var last: ?u64 = null;
    var previous: u32 = 0;
    for (entries, 0..) |entry, i| {
        if (entry.row >= table.rows) return error.RowOutOfRange;
        if (i > 0 and entry.row < previous) return error.UnsortedRows;
        previous = entry.row;
        const start = off + @as(u64, entry.row) * stride;
        var at = start / page;
        const end = (start + len - 1) / page;
        if (end >= bitmap.capacity()) return error.TableSize;
        while (at <= end) : (at += 1) {
            if (last != null and at <= last.?) continue;
            pages += 1;
            resident += @intFromBool(bitmap.isSet(@intCast(at)));
            last = at;
        }
    }
    return .of(pages, resident);
}

fn selectedPages(bitmap: std.DynamicBitSet, table: *const engram.Table, entries: []const reader.Entry) !struct { code: Fraction, scale: Fraction } {
    return .{
        .code = try selectedPart(bitmap, table, entries, table.w_off, table.w_stride, table.wlen),
        .scale = try selectedPart(bitmap, table, entries, table.s_off, table.s_stride, table.slen),
    };
}

fn sampleOffsets(count: usize, depth: usize) ![3]usize {
    if (count < depth + 2) return error.CorpusTooShort;
    const span = count - depth;
    return .{ 0, span / 2, span };
}

fn sampleTokens(tokens: []const u32, depth: usize, offset: usize) ![]const u32 {
    if (depth < 2 or offset > tokens.len or depth > tokens.len - offset) return error.CorpusTooShort;
    return tokens[offset..][0 .. depth - 1];
}

fn validateLayout(spec: reader.TableLayout) !void {
    const w_end = @as(u128, spec.w_off) + @as(u128, spec.rows) * spec.w_stride;
    const s_end = @as(u128, spec.s_off) + @as(u128, spec.rows) * spec.s_stride;
    if ((spec.layer != 1 and spec.layer != 14) or spec.rows == 0 or spec.rows > std.math.maxInt(u32) or
        spec.wlen != 256 or spec.w_stride != 256 or spec.slen != 8 or spec.s_stride != 8 or
        w_end > spec.file_bytes or s_end > spec.file_bytes or
        !(w_end <= spec.s_off or s_end <= spec.w_off) or
        std.fs.path.isAbsolute(spec.relative_path) or std.mem.indexOf(u8, spec.relative_path, "..") != null)
        return error.InvalidLayout;
}

fn regionFraction(bitmap: std.DynamicBitSet, offset: u64, len: u64) Fraction {
    const page = std.heap.pageSize();
    const lo: usize = @intCast(offset / page);
    const hi: usize = @intCast((offset + len - 1) / page + 1);
    var resident: usize = 0;
    for (lo..hi) |i| resident += @intFromBool(bitmap.isSet(i));
    return .of(hi - lo, resident);
}

const Sample = struct { depth: usize, token_offset: usize, prefill_tokens: usize, token_digest: u64, rows: usize, code: Fraction, scale: Fraction };
const TableResult = struct {
    layer: usize,
    path: []const u8,
    device: u64,
    inode: u64,
    file_bytes: u64,
    snapshot_start_epoch_ns: i128,
    snapshot_end_epoch_ns: i128,
    whole: Fraction,
    code: Fraction,
    scale: Fraction,
    samples: [6]Sample,
};

fn clockNs() i128 {
    return std.Io.Timestamp.now(std.Io.Threaded.global_single_threaded.io(), .real).toNanoseconds();
}

fn probeTable(io: std.Io, gpa: std.mem.Allocator, pack: []const u8, spec: reader.TableLayout, slot: usize, cfg: *const Config, map: []const u32, tokens: []const u32) !TableResult {
    try validateLayout(spec);
    if (spec.rows != cfg.dsv41_engram_rows[slot] or spec.layer != cfg.dsv41_engram_layers[slot]) return error.LayoutConfigMismatch;
    const path = try std.fs.path.join(gpa, &.{ pack, spec.relative_path });
    defer gpa.free(path);
    const real = try std.Io.Dir.cwd().realPathFileAlloc(io, path, gpa);
    errdefer gpa.free(real);
    const fd = std.c.open(real, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
    if (fd < 0) return error.TableOpen;
    defer _ = std.c.close(fd);
    var st: std.c.Stat = undefined;
    if (std.c.fstat(fd, &st) != 0 or st.size <= 0 or @as(u64, @intCast(st.size)) != spec.file_bytes) return error.TableSize;
    const start = clockNs();
    var bitmap = try reader.Probe.snapshot(gpa, fd);
    defer bitmap.deinit();
    const end = clockNs();
    const table: engram.Table = .{ .fd = fd, .rows = spec.rows, .head_dim = 256, .mode = .mxfp8, .bits = 8, .group_size = 32, .w_off = spec.w_off, .w_stride = spec.w_stride, .wlen = spec.wlen, .s_off = spec.s_off, .s_stride = spec.s_stride, .slen = spec.slen };
    var result: TableResult = .{ .layer = spec.layer, .path = real, .device = @intCast(st.dev), .inode = st.ino, .file_bytes = spec.file_bytes, .snapshot_start_epoch_ns = start, .snapshot_end_epoch_ns = end, .whole = .of(bitmap.capacity(), bitmap.count()), .code = regionFraction(bitmap, spec.w_off, spec.rows * spec.w_stride), .scale = regionFraction(bitmap, spec.s_off, spec.rows * spec.s_stride), .samples = undefined };
    const hash = try engram.Hash.init(cfg, map[cfg.dsv41_engram_pad_id]);
    const cols: usize = hash.cols();
    if (cols != 24) return error.HashGeometry;
    var sample_index: usize = 0;
    for ([_]usize{ 32768, 131072 }) |depth| {
        for (try sampleOffsets(tokens.len, depth)) |offset| {
            var arena = std.heap.ArenaAllocator.init(gpa);
            defer arena.deinit();
            const a = arena.allocator();
            const sample = try sampleTokens(tokens, depth, offset);
            const comp = try a.alloc(u32, sample.len);
            for (comp, sample) |*id, token| {
                if (token >= map.len) return error.TokenOutOfVocab;
                id.* = map[token];
            }
            const grid = try a.alloc(u32, sample.len * 2 * cols);
            hash.rows(&.{}, comp, grid);
            const cat = try reader.Catalog.init(a, grid, sample.len, 2, slot, cols, 2048);
            const parts = try selectedPages(bitmap, &table, cat.unique);
            result.samples[sample_index] = .{ .depth = depth, .token_offset = offset, .prefill_tokens = sample.len, .token_digest = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(tokens[offset..][0..depth])), .rows = cat.unique.len, .code = parts.code, .scale = parts.scale };
            sample_index += 1;
        }
    }
    var after: std.c.Stat = undefined;
    if (std.c.fstat(fd, &after) != 0 or st.size != after.size or st.ino != after.ino) return error.TableChanged;
    return result;
}

pub fn main(init: std.process.Init) !void {
    if (comptime !@import("builtin").os.tag.isDarwin()) return error.DarwinOnly;
    _ = std.c.alarm(105);
    const start = clockNs();
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 4) return error.Usage_pack_corpus_exported_layout_json;
    var backing: BoundedPages = .{};
    var heap: std.heap.DebugAllocator(.{ .stack_trace_frames = 0, .thread_safe = false }) = .{ .backing_allocator = backing.allocator() };
    defer _ = heap.deinit();
    const gpa = heap.allocator();
    const io = init.io;
    const pack = try std.Io.Dir.cwd().realPathFileAlloc(io, args[1], gpa);
    defer gpa.free(pack);
    const layout_raw = try std.Io.Dir.cwd().readFileAlloc(io, args[3], gpa, .limited(1 << 20));
    defer gpa.free(layout_raw);
    const layout = try std.json.parseFromSlice(reader.Layout, gpa, layout_raw, .{});
    defer layout.deinit();
    if (layout.value.schema != 1) return error.LayoutSchema;
    const config_path = try std.fs.path.join(gpa, &.{ pack, "config.json" });
    defer gpa.free(config_path);
    const config_raw = try std.Io.Dir.cwd().readFileAlloc(io, config_path, gpa, .limited(8 << 20));
    defer gpa.free(config_raw);
    const config_digest = reader.sha256(config_raw);
    if (!std.mem.eql(u8, layout.value.config_sha256, &config_digest)) return error.LayoutConfigMismatch;
    const cfg = try reader.loadConfig(io, gpa, pack);
    if (cfg.dsv41_n_engram_layers != 2 or cfg.dsv41_engram_layers[0] != 1 or cfg.dsv41_engram_layers[1] != 14 or
        cfg.dsv41_engram_head_dim != 256) return error.LayerMismatch;
    const encoded = try reader.tokenize(io, gpa, pack, args[2]);
    defer gpa.free(encoded.ids);
    _ = try sampleOffsets(encoded.ids.len, 131072);
    const map = try engram.loadTokenMap(gpa, pack, cfg.vocab_size, cfg.dsv41_engram_compressed_vocab);
    defer gpa.free(map);
    if (cfg.dsv41_engram_pad_id >= map.len) return error.TokenOutOfVocab;
    var tables: [2]TableResult = undefined;
    var table_count: usize = 0;
    defer for (tables[0..table_count]) |table| gpa.free(table.path);
    for (&tables, layout.value.tables, 0..) |*result, spec, slot| {
        result.* = try probeTable(io, gpa, pack, spec, slot, &cfg, map, encoded.ids);
        table_count += 1;
    }
    var hostname: [256]u8 = @splat(0);
    if (std.c.gethostname(&hostname, hostname.len) != 0) return error.Hostname;
    const layout_digest = reader.sha256(layout_raw);
    const token_digest = reader.sha256(std.mem.sliceAsBytes(encoded.ids));
    try reader.jsonLine(gpa, .{
        .kind = "pp4_live_residency",
        .schema = 1,
        .host = std.mem.sliceTo(&hostname, 0),
        .table_payload_read = false,
        .table_header_read = false,
        .layout_sha256 = layout_digest[0..],
        .config_sha256 = config_digest[0..],
        .corpus_sha256 = encoded.corpus_sha256[0..],
        .tokens_sha256 = token_digest[0..],
        .corpus_tokens = encoded.ids.len,
        .page_bytes = std.heap.pageSize(),
        .started_epoch_ns = start,
        .elapsed_s = @as(f64, @floatFromInt(clockNs() - start)) / 1e9,
        .allocation_cap_bytes = allocation_cap,
        .peak_allocation_bytes = backing.peak,
        .mapping_cap_bytes = mapping_cap,
        .tables = tables,
    });
}

test "pp4 residency: fixture probe preserves cold pages and deduplicates code and scale spans" {
    const io = std.Io.Threaded.global_single_threaded.io();
    var td = testing.tmpDir(.{});
    defer td.cleanup();
    const page = std.heap.pageSize();
    const bytes = try testing.allocator.alloc(u8, page * 5);
    defer testing.allocator.free(bytes);
    @memset(bytes, 7);
    try td.dir.writeFile(io, .{ .sub_path = "source", .data = bytes });
    var pb: [1024]u8 = undefined;
    const dir = pb[0..try td.dir.realPath(io, &pb)];
    const src = try std.fmt.allocPrintSentinel(testing.allocator, "{s}/source", .{dir}, 0);
    defer testing.allocator.free(src);
    const dst = try std.fmt.allocPrintSentinel(testing.allocator, "{s}/clone", .{dir}, 0);
    defer testing.allocator.free(dst);
    const source = std.c.open(src, .{ .ACCMODE = .RDWR }, @as(std.c.mode_t, 0));
    try testing.expect(source >= 0);
    defer _ = std.c.close(source);
    try testing.expectEqual(@as(c_int, 0), std.c.ftruncate(source, @intCast(mapping_cap + page * 5 - 3)));
    const apfs = struct {
        extern "c" fn clonefile([*:0]const u8, [*:0]const u8, c_int) c_int;
    };
    try testing.expectEqual(@as(c_int, 0), apfs.clonefile(src, dst, 0));
    const fd = std.c.open(dst, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
    try testing.expect(fd >= 0);
    defer _ = std.c.close(fd);
    const map = try std.posix.mmap(null, bytes.len, .{ .READ = true }, .{ .TYPE = .SHARED }, fd, 0);
    defer std.posix.munmap(map);
    const first: *const volatile u8 = @ptrCast(map.ptr);
    try testing.expectEqual(@as(u8, 7), first.*);
    const edge_map = try std.posix.mmap(null, page, .{ .READ = true }, .{ .TYPE = .SHARED }, fd, mapping_cap - page);
    defer std.posix.munmap(edge_map);
    const edge: *const volatile u8 = @ptrCast(edge_map.ptr);
    try testing.expectEqual(@as(u8, 0), edge.*);
    const tail_map = try std.posix.mmap(null, page - 3, .{ .READ = true }, .{ .TYPE = .SHARED }, fd, mapping_cap + page * 4);
    defer std.posix.munmap(tail_map);
    const tail: *const volatile u8 = @ptrCast(tail_map.ptr);
    try testing.expectEqual(@as(u8, 0), tail.*);
    var before = try reader.Probe.snapshot(testing.allocator, fd);
    defer before.deinit();
    try testing.expectEqual(@as(usize, mapping_cap / page + 5), before.capacity());
    try testing.expectEqual(@as(usize, 3), before.count());
    try testing.expect(before.isSet(0));
    const table: engram.Table = .{ .fd = fd, .rows = 4, .head_dim = 256, .mode = .mxfp8, .bits = 8, .group_size = 32, .w_off = mapping_cap - 128, .w_stride = 256, .wlen = 256, .s_off = mapping_cap + page * 3 - 8, .s_stride = 8, .slen = 8 };
    const entries = [_]reader.Entry{ .{ .row = 0, .chunk = 0 }, .{ .row = 1, .chunk = 0 }, .{ .row = 1, .chunk = 1 }, .{ .row = 2, .chunk = 1 } };
    const result = try selectedPages(before, &table, &entries);
    const crossing = try selectedPages(before, &table, entries[0..1]);
    try testing.expectEqual(@as(usize, 2), crossing.code.pages);
    try testing.expectEqual(@as(usize, 1), crossing.scale.pages);
    try testing.expectEqual(@as(usize, 2), result.code.pages);
    try testing.expectEqual(@as(usize, 1), result.code.resident_pages);
    try testing.expectEqual(@as(f64, 0.5), result.code.resident_fraction);
    try testing.expectEqual(@as(usize, 2), result.scale.pages);
    try testing.expectEqual(@as(usize, 0), result.scale.resident_pages);
    try testing.expectEqual(@as(f64, 0), result.scale.resident_fraction);
    try testing.expectError(error.RowOutOfRange, selectedPages(before, &table, &.{.{ .row = 4, .chunk = 0 }}));
    var after = try reader.Probe.snapshot(testing.allocator, fd);
    defer after.deinit();
    try testing.expect(before.eql(after));
}

test "pp4 residency: backing budget rejects excess pages and reclaims freed allocations" {
    var budget: BoundedPages = .{};
    const a = budget.allocator();
    const bytes = try a.alloc(u8, allocation_cap);
    try testing.expectError(error.OutOfMemory, a.alloc(u8, 1));
    a.free(bytes);
    try testing.expectEqual(@as(usize, 0), budget.live);
    const reused = try a.alloc(u8, 1);
    defer a.free(reused);
    try testing.expectEqual(std.heap.pageSize(), budget.live);
    try testing.expectEqual(@as(usize, allocation_cap), budget.peak);
}

test "pp4 residency: samples use three distinct real token slices and exclude final token" {
    const offsets = try sampleOffsets(140000, 131072);
    try testing.expectEqualSlices(usize, &.{ 0, 4464, 8928 }, &offsets);
    const ids = try testing.allocator.alloc(u32, 140000);
    defer testing.allocator.free(ids);
    for (ids, 0..) |*id, i| id.* = @intCast(i);
    for (offsets) |offset| {
        const sample = try sampleTokens(ids, 131072, offset);
        try testing.expectEqual(@as(usize, 131071), sample.len);
        try testing.expectEqual(@as(u32, @intCast(offset)), sample[0]);
        try testing.expectEqual(@as(u32, @intCast(offset + 131070)), sample[sample.len - 1]);
    }
    try testing.expectError(error.CorpusTooShort, sampleOffsets(131073, 131072));
    try testing.expectError(error.CorpusTooShort, sampleTokens(ids, 131072, 8929));
}

test "pp4 residency: layout rejects overflow overlap and truncated split tables" {
    const good: reader.TableLayout = .{ .layer = 1, .relative_path = "engram/table", .file_bytes = 1000, .rows = 2, .w_off = 64, .w_stride = 256, .wlen = 256, .s_off = 576, .s_stride = 8, .slen = 8 };
    try validateLayout(good);
    var bad = good;
    bad.s_off = 500;
    try testing.expectError(error.InvalidLayout, validateLayout(bad));
    bad = good;
    bad.file_bytes = 580;
    try testing.expectError(error.InvalidLayout, validateLayout(bad));
    bad = good;
    bad.w_off = std.math.maxInt(u64);
    try testing.expectError(error.InvalidLayout, validateLayout(bad));
    bad = good;
    bad.relative_path = "../other";
    try testing.expectError(error.InvalidLayout, validateLayout(bad));
}
