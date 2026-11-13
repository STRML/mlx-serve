//! Plan 75 Step 1b Tier 1: standalone Engram table-read micro-bench, UBC residency probe, and u32 token-id serializer.
const std = @import("std");
const builtin = @import("builtin");
const crt = @import("../../sdk/util/crt.zig");
const engram = @import("engram.zig");
const edir = @import("prompt/engram_direct.zig");
const config_mod = @import("config.zig");
const tokenizer_mod = @import("../../sdk/tokenizer/tokenizer.zig");
const pp_step1 = @import("pp_step1.zig");

pub const TABLE_BENCH_P0: usize = 32768;
pub const TABLE_BENCH_WIDTHS = [_]usize{ 2048, 4096, 8192 };
pub const TABLE_BENCH_LAYERS = [_]usize{ 1, 14 };
pub const MAX_UBC_CACHED_RATIO: f64 = 0.05;

pub const TableBenchSource = enum {
    real_text,
    deterministic,

    pub fn name(self: TableBenchSource) []const u8 {
        return @tagName(self);
    }

    pub fn isGated(self: TableBenchSource) bool {
        return self == .real_text;
    }
};

pub const TableBenchSpec = struct {
    layer: usize,
    chunk_rows: usize,
    step1_pread_ms: f64,
    threshold_ms: f64,
};

pub const TABLE_BENCH_SPECS = [_]TableBenchSpec{
    .{ .layer = 1, .chunk_rows = 2048, .step1_pread_ms = 362.9, .threshold_ms = 172.8 },
    .{ .layer = 1, .chunk_rows = 4096, .step1_pread_ms = 628.0, .threshold_ms = 299.1 },
    .{ .layer = 1, .chunk_rows = 8192, .step1_pread_ms = 1190.8, .threshold_ms = 567.1 },
    .{ .layer = 14, .chunk_rows = 2048, .step1_pread_ms = 375.9, .threshold_ms = 179.0 },
    .{ .layer = 14, .chunk_rows = 4096, .step1_pread_ms = 636.9, .threshold_ms = 303.3 },
    .{ .layer = 14, .chunk_rows = 8192, .step1_pread_ms = 1288.0, .threshold_ms = 613.3 },
};

pub fn tableBenchSpecFor(layer: usize, chunk_rows: usize) ?TableBenchSpec {
    for (TABLE_BENCH_SPECS) |s| {
        if (s.layer == layer and s.chunk_rows == chunk_rows) return s;
    }
    return null;
}

pub fn extractSortDedupLayerRows(grid_ids: []const u32, chunk_rows: usize, n_layers: usize, layer_slot: usize, nc: usize, out_uniq: []u32) ![]const u32 {
    const total = chunk_rows * nc;
    if (layer_slot >= n_layers or grid_ids.len != chunk_rows * n_layers * nc) return error.InvalidGridShape;
    if (out_uniq.len < total) return error.BufferTooSmall;
    for (0..chunk_rows) |tok| @memcpy(out_uniq[tok * nc ..][0..nc], grid_ids[(tok * n_layers + layer_slot) * nc ..][0..nc]);
    std.mem.sort(u32, out_uniq[0..total], {}, std.sort.asc(u32));
    var nu: usize = 0;
    for (out_uniq[0..total]) |id| {
        if (nu == 0 or out_uniq[nu - 1] != id) {
            out_uniq[nu] = id;
            nu += 1;
        }
    }
    return out_uniq[0..nu];
}

pub fn countDistinctInPlace(ids: []u32) usize {
    if (ids.len == 0) return 0;
    std.mem.sort(u32, ids, {}, std.sort.asc(u32));
    var n: usize = 1;
    for (ids[1..]) |id| {
        if (id != ids[n - 1]) {
            ids[n] = id;
            n += 1;
        }
    }
    return n;
}

pub const MsStats = struct {
    median_ms: f64,
    p90_ms: f64,
};

pub fn sampleMsStats(samples_ms: []const f64, scratch: []f64) !MsStats {
    const n = samples_ms.len;
    if (n == 0 or scratch.len < n) return error.InvalidSampleCount;
    @memcpy(scratch[0..n], samples_ms);
    std.mem.sort(f64, scratch[0..n], {}, std.sort.asc(f64));
    for (scratch[0..n]) |v| {
        if (std.math.isNan(v) or std.math.isInf(v) or v < 0) return error.InvalidSampleValue;
    }
    const med = if (n % 2 == 1) scratch[n / 2] else 0.5 * (scratch[n / 2 - 1] + scratch[n / 2]);
    const p90_idx = @min(n - 1, (9 * n + 9) / 10 - 1);
    return .{ .median_ms = med, .p90_ms = scratch[p90_idx] };
}

fn medianUsize(vals: []const usize, scratch: []usize) !usize {
    const n = vals.len;
    if (n == 0 or scratch.len < n) return error.InvalidSampleCount;
    @memcpy(scratch[0..n], vals);
    std.mem.sort(usize, scratch[0..n], {}, std.sort.asc(usize));
    return scratch[n / 2];
}

pub fn coldAdjustedMs(median_ms: f64, ubc_cached_rows_max: usize, distinct_rows: usize) f64 {
    if (distinct_rows == 0 or ubc_cached_rows_max >= distinct_rows) return std.math.inf(f64);
    const ratio = @as(f64, @floatFromInt(ubc_cached_rows_max)) / @as(f64, @floatFromInt(distinct_rows));
    return median_ms / (1.0 - ratio);
}

pub const TableBenchCell = struct {
    source: TableBenchSource,
    chunk_rows: usize,
    layer: usize,
    p0_start: usize,
    chunk_tok_lo: usize,
    chunk_tok_hi: usize,
    reps: usize,
    block_bytes: u64,
    records: bool,
    structural_cache_possible: bool = false,
    ubc_verified: bool = true,
    ubc_cached_rows: usize = 0,
    ubc_cached_rows_max: usize = 0,
    ubc_populated_rows: usize = 0,
    page_cache_possible: bool,
    lookup_rows: usize,
    rows_read: usize,
    distinct_rows: usize,
    total_rows_read: u64,
    distinct_rows_across_reps: usize,
    median_ms: f64,
    p90_ms: f64,
    cold_adjusted_median_ms: f64,
    hash_sort_median_ms: f64,
    step1_pread_ms: f64,
    threshold_ms: f64,
    gate_pass: bool,
};

pub fn cellColdGatePass(spec: TableBenchSpec, block_bytes: u64, structural_cache_possible: bool, ubc_verified: bool, ubc_cached_rows_max: usize, distinct_rows: usize, median_ms: f64) bool {
    if (block_bytes == 0 or structural_cache_possible or !ubc_verified or distinct_rows == 0) return false;
    const ubc_ratio = @as(f64, @floatFromInt(ubc_cached_rows_max)) / @as(f64, @floatFromInt(distinct_rows));
    if (ubc_ratio > MAX_UBC_CACHED_RATIO) return false;
    const adj_ms = coldAdjustedMs(median_ms, ubc_cached_rows_max, distinct_rows);
    return median_ms <= spec.threshold_ms and adj_ms <= spec.threshold_ms;
}

pub fn evaluateTableBenchCell(
    source: TableBenchSource,
    chunk_rows: usize,
    layer: usize,
    block_bytes: u64,
    structural_cache_possible: bool,
    ubc_verified: bool,
    ubc_cached_rows_max: usize,
    distinct_rows: usize,
    samples_ms: []const f64,
    scratch: []f64,
) !struct { stats: MsStats, spec: TableBenchSpec, gated: bool, cold_adjusted_ms: f64, gate_pass: bool } {
    const spec = tableBenchSpecFor(layer, chunk_rows) orelse return error.UnknownTableBenchCell;
    const stats = try sampleMsStats(samples_ms, scratch);
    const adj_ms = coldAdjustedMs(stats.median_ms, ubc_cached_rows_max, distinct_rows);
    const pass = cellColdGatePass(spec, block_bytes, structural_cache_possible, ubc_verified, ubc_cached_rows_max, distinct_rows, stats.median_ms);
    return .{ .stats = stats, .spec = spec, .gated = source.isGated(), .cold_adjusted_ms = adj_ms, .gate_pass = pass };
}

pub fn evaluateTableBenchGate(cells: []const TableBenchCell) bool {
    for (TABLE_BENCH_SPECS) |spec| {
        var found = false;
        for (cells) |c| {
            if (c.source != .real_text) continue;
            if (c.layer != spec.layer or c.chunk_rows != spec.chunk_rows) continue;
            if (found) return false;
            if (!c.gate_pass) return false;
            if (!cellColdGatePass(spec, c.block_bytes, c.structural_cache_possible, c.ubc_verified, c.ubc_cached_rows_max, c.distinct_rows, c.median_ms)) return false;
            found = true;
        }
        if (!found) return false;
    }
    return true;
}

pub fn formatTableBenchJson(buf: []u8, c: TableBenchCell) ![]const u8 {
    const f_nocache = if (c.block_bytes > 0) "true" else "false";
    const rec = if (c.records) "true" else "false";
    const ubc_ver = if (c.ubc_verified) "true" else "false";
    const pc_poss = if (c.page_cache_possible) "true" else "false";
    const pc_note = if (c.structural_cache_possible or c.block_bytes == 0)
        "os_page_cache_fallback_active"
    else if (!c.ubc_verified)
        "ubc_mincore_unverified"
    else if (c.ubc_cached_rows_max > 0)
        "preexisting_ubc_pages_detected_cold_adjusted"
    else
        "verified_zero_ubc_pages_before_read";
    const gated = if (c.source.isGated()) "true" else "false";
    const gpass = if (c.gate_pass) "true" else "false";
    const verdict = if (c.gate_pass) "PASS" else "FAIL";
    const dist_ratio = if (c.lookup_rows > 0) @as(f64, @floatFromInt(c.distinct_rows)) / @as(f64, @floatFromInt(c.lookup_rows)) else 0.0;
    const ubc_ratio = if (c.distinct_rows > 0) @as(f64, @floatFromInt(c.ubc_cached_rows_max)) / @as(f64, @floatFromInt(c.distinct_rows)) else 1.0;
    const adj_out = if (std.math.isFinite(c.cold_adjusted_median_ms)) c.cold_adjusted_median_ms else 999999.0;
    const speedup = if (c.median_ms > 0) c.step1_pread_ms / c.median_ms else 0.0;
    var w = std.Io.Writer.fixed(buf);
    try w.print(
        "{{\"kind\":\"table_bench\",\"source\":\"{s}\",\"chunk_rows\":{d},\"layer\":{d},\"p0_start\":{d},\"chunk_tok_lo\":{d},\"chunk_tok_hi\":{d},\"reps\":{d},\"grid_threads\":{d},\"block_bytes\":{d},\"records\":{s},\"f_nocache\":{s},\"ubc_verified\":{s},\"ubc_cached_rows\":{d},\"ubc_cached_rows_max\":{d},\"ubc_cached_ratio\":{d:.4},\"ubc_populated_rows\":{d},\"page_cache_possible\":{s},\"page_cache_note\":\"{s}\",",
        .{ c.source.name(), c.chunk_rows, c.layer, c.p0_start, c.chunk_tok_lo, c.chunk_tok_hi, c.reps, engram.GRID_THREADS, c.block_bytes, rec, f_nocache, ubc_ver, c.ubc_cached_rows, c.ubc_cached_rows_max, ubc_ratio, c.ubc_populated_rows, pc_poss, pc_note },
    );
    try w.print(
        "\"lookup_rows\":{d},\"rows_read\":{d},\"distinct_rows\":{d},\"distinct_ratio\":{d:.4},\"total_rows_read\":{d},\"distinct_rows_across_reps\":{d},\"median_ms\":{d:.2},\"p90_ms\":{d:.2},\"cold_adjusted_median_ms\":{d:.2},\"hash_sort_median_ms\":{d:.2},\"step1_pread_ms\":{d:.1},\"threshold_ms\":{d:.1},\"speedup_vs_step1\":{d:.3},\"gated\":{s},\"gate_pass\":{s},\"gate_verdict\":\"{s}\"}}\n",
        .{ c.lookup_rows, c.rows_read, c.distinct_rows, dist_ratio, c.total_rows_read, c.distinct_rows_across_reps, c.median_ms, c.p90_ms, adj_out, c.hash_sort_median_ms, c.step1_pread_ms, c.threshold_ms, speedup, gated, gpass, verdict },
    );
    return w.buffered();
}

const darwin_mman = if (builtin.os.tag.isDarwin()) struct {
    extern "c" fn mincore(addr: *const anyopaque, len: usize, vec: [*]u8) c_int;
} else struct {};

pub const UbcProbe = struct {
    map: ?[]align(std.heap.page_size_min) const u8 = null,

    pub fn open(fd: c_int) UbcProbe {
        if (comptime !builtin.os.tag.isDarwin()) return .{};
        const sz = crt.seekEnd(fd);
        if (sz <= 0) return .{};
        const map = std.posix.mmap(null, @intCast(sz), .{ .READ = true }, .{ .TYPE = .SHARED }, fd, 0) catch return .{};
        return .{ .map = map };
    }

    pub fn close(self: *UbcProbe) void {
        if (comptime !builtin.os.tag.isDarwin()) return;
        if (self.map) |m| std.posix.munmap(m);
        self.map = null;
    }

    const PageMemo = struct {
        page_off: u64 = std.math.maxInt(u64),
        in_core: bool = false,
    };

    fn spanInCore(map: []const u8, off: u64, len: usize, page: u64, memo: *PageMemo) ?bool {
        if (comptime !builtin.os.tag.isDarwin()) return null;
        if (len == 0 or off >= map.len) return false;
        const page_lo = (off / page) * page;
        const end = @min(@as(u64, map.len), off + len);
        const page_hi = std.mem.alignForward(u64, end, page);
        var p = page_lo;
        var any = false;
        while (p < page_hi) : (p += page) {
            if (memo.page_off == p) {
                if (memo.in_core) any = true;
                continue;
            }
            const rem = @min(page, @as(u64, map.len) - p);
            var vec: [1]u8 = .{0};
            if (darwin_mman.mincore(@ptrCast(map.ptr + @as(usize, @intCast(p))), @intCast(rem), &vec) != 0) return null;
            const resident = (vec[0] & 0x1) != 0;
            memo.* = .{ .page_off = p, .in_core = resident };
            if (resident) any = true;
        }
        return any;
    }

    pub fn countCachedRows(self: *const UbcProbe, t: *const engram.Table, uniq: []const u32) ?usize {
        const map = self.map orelse return null;
        if (comptime !builtin.os.tag.isDarwin()) return null;
        const page: u64 = std.heap.pageSize();
        var w_memo: PageMemo = .{};
        var s_memo: PageMemo = .{};
        var b_memo: PageMemo = .{};
        var cached: usize = 0;
        for (uniq) |row_u32| {
            const row: u64 = row_u32;
            if (row >= t.rows) continue;
            if (t.records()) {
                if (spanInCore(map, t.w_off + row * t.w_stride, t.wlen + t.slen, page, &w_memo) orelse return null) cached += 1;
                continue;
            }
            const w_hit = spanInCore(map, t.w_off + row * t.w_stride, t.wlen, page, &w_memo) orelse return null;
            const s_hit = if (t.slen > 0) (spanInCore(map, t.s_off + row * t.s_stride, t.slen, page, &s_memo) orelse return null) else false;
            const b_hit = if (t.mode == .affine and t.slen > 0) (spanInCore(map, t.b_off + row * t.s_stride, t.slen, page, &b_memo) orelse return null) else false;
            if (w_hit or s_hit or b_hit) cached += 1;
        }
        return cached;
    }
};

fn monoUs() i64 {
    const io = std.Io.Threaded.global_single_threaded.io();
    return @intCast(@divTrunc(std.Io.Timestamp.now(io, .awake).toNanoseconds(), 1000));
}

fn writeFdAll(fd: c_int, bytes: []const u8) void {
    var off: usize = 0;
    while (off < bytes.len) {
        const n = crt.write(fd, bytes[off..]);
        if (n <= 0) return;
        off += @intCast(n);
    }
}

fn writeStderrFmt(comptime fmt: []const u8, args: anytype) void {
    var buf: [512]u8 = undefined;
    const line = std.fmt.bufPrint(&buf, fmt, args) catch return;
    writeFdAll(2, line);
}

pub fn readTokenIdsFile(io: std.Io, gpa: std.mem.Allocator, path: []const u8) ![]u32 {
    const raw = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(256 << 20));
    defer gpa.free(raw);
    if (raw.len == 0 or raw.len % 4 != 0) return error.InvalidTokenIdsFile;
    const n = raw.len / 4;
    const ids = try gpa.alloc(u32, n);
    for (ids, 0..) |*dst, idx| dst.* = std.mem.readInt(u32, raw[idx * 4 ..][0..4], .little);
    return ids;
}

pub fn runTokenizeIds(io: std.Io, gpa: std.mem.Allocator, dir: []const u8, in_path: []const u8, out_path: []const u8) !void {
    var opened = try std.Io.Dir.cwd().openDir(io, dir, .{});
    defer opened.close(io);
    var abs_buf: [std.fs.max_path_bytes]u8 = undefined;
    const abs_dir = abs_buf[0..try opened.realPath(io, &abs_buf)];

    var tok = try tokenizer_mod.loadTokenizerAny(io, gpa, abs_dir);
    defer tok.deinit();

    const text = try std.Io.Dir.cwd().readFileAlloc(io, in_path, gpa, .limited(64 << 20));
    defer gpa.free(text);
    if (text.len == 0) return error.EmptyInputText;

    const ids = try tok.encode(gpa, text);
    defer gpa.free(ids);
    if (ids.len == 0) return error.EmptyEncodedTokens;

    const raw_out = try gpa.alloc(u8, ids.len * 4);
    defer gpa.free(raw_out);
    for (ids, 0..) |id, idx| std.mem.writeInt(u32, raw_out[idx * 4 ..][0..4], id, .little);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = out_path, .data = raw_out });
}

fn mapCompressedIds(src: []const u32, token_map: []const u32, dst: []u32) void {
    for (src, dst) |t, *o| o.* = if (t < token_map.len) token_map[t] else 0;
}

fn fillDisjointDeterministicStream(dst: []u32, vocab: usize, p0: usize) void {
    const head = @min(p0, dst.len);
    pp_step1.fillDeterministicTokens(dst[0..head], vocab, 1);
    var at = head;
    var salt: usize = 1000;
    while (at < dst.len) : (salt += 1) {
        const step = @min(@as(usize, 100_000), dst.len - at);
        pp_step1.fillDeterministicTokens(dst[at .. at + step], vocab, salt);
        at += step;
    }
}

fn tableStructuralCachePossible(t: *const engram.Table, block: u64) bool {
    if (!t.direct or block == 0) return true;
    const part_max: u64 = if (t.records()) t.wlen + t.slen else @max(t.wlen, t.slen);
    const worst_span = std.mem.alignForward(u64, (block - 1) + part_max, block);
    return worst_span > edir.max_span;
}

fn countCrossRepDistinct(gpa: std.mem.Allocator, hash: *const engram.Hash, comp: []const u32, chunk_lo: usize, c_rows: usize, reps: usize, layer_slot: usize, grid_buf: []u32, uniq_buf: []u32) !usize {
    const nc: usize = hash.cols();
    const nt: usize = hash.n_layers;
    var all: std.ArrayList(u32) = .empty;
    defer all.deinit(gpa);
    try all.ensureTotalCapacity(gpa, reps * c_rows * nc);
    for (0..reps) |rep| {
        const pos = chunk_lo + rep * c_rows;
        const cur_grid = grid_buf[0 .. c_rows * nt * nc];
        hash.rows(comp[0..pos], comp[pos .. pos + c_rows], cur_grid);
        const uniq = try extractSortDedupLayerRows(cur_grid, c_rows, nt, layer_slot, nc, uniq_buf);
        try all.appendSlice(gpa, uniq);
    }
    return countDistinctInPlace(all.items);
}

pub fn runTableBench(io: std.Io, gpa: std.mem.Allocator, dir: []const u8, tokens_path: []const u8, reps: usize, warmup: usize, json_path: ?[]const u8, json_out: *const pp_step1.JsonOut) !void {
    if (reps == 0) return error.InvalidArgs;
    var opened_dir = try std.Io.Dir.cwd().openDir(io, dir, .{});
    defer opened_dir.close(io);
    var abs_buf: [std.fs.max_path_bytes]u8 = undefined;
    const abs_dir = abs_buf[0..try opened_dir.realPath(io, &abs_buf)];

    const cfg = try pp_step1.loadPackConfig(io, gpa, abs_dir);
    var slot_l1: ?usize = null;
    var slot_l14: ?usize = null;
    for (0..cfg.dsv41_n_engram_layers) |k| {
        if (cfg.dsv41_engram_layers[k] == 1) slot_l1 = k;
        if (cfg.dsv41_engram_layers[k] == 14) slot_l14 = k;
    }
    const slots = [2]usize{ slot_l1 orelse return error.EngramLayerMissing, slot_l14 orelse return error.EngramLayerMissing };

    const token_map = try engram.loadTokenMap(gpa, abs_dir, cfg.vocab_size, cfg.dsv41_engram_compressed_vocab);
    defer gpa.free(token_map);
    if (cfg.dsv41_engram_pad_id >= token_map.len) return error.EngramTokenMap;
    const hash = try engram.Hash.init(&cfg, token_map[cfg.dsv41_engram_pad_id]);
    const nc: usize = hash.cols();
    const nt: usize = hash.n_layers;

    var tables: [2]engram.Table = undefined;
    var probes: [2]UbcProbe = .{ .{}, .{} };
    var n_open: usize = 0;
    defer {
        for (probes[0..n_open]) |*pr| pr.close();
        for (tables[0..n_open]) |*t| t.close();
    }
    var blocks: [2]u64 = .{ 0, 0 };
    var struct_cache_flags: [2]bool = .{ true, true };

    for (0..2) |ti| {
        const slot = slots[ti];
        const layer = TABLE_BENCH_LAYERS[ti];
        tables[ti] = try engram.Table.open(gpa, abs_dir, slot, layer, cfg.dsv41_engram_rows[slot], cfg.dsv41_engram_head_dim);
        probes[ti] = UbcProbe.open(tables[ti].fd);
        n_open += 1;
        tables[ti].readDirect();
        blocks[ti] = edir.blockOf(tables[ti].fd);
        struct_cache_flags[ti] = tableStructuralCachePossible(&tables[ti], blocks[ti]);
        writeStderrFmt(
            "[table-bench] layer={d} fd={d} blockOf={d} direct={} records={} ubc_mincore_ready={} structural_cache_possible={}\n",
            .{ layer, tables[ti].fd, blocks[ti], tables[ti].direct, tables[ti].records(), probes[ti].map != null, struct_cache_flags[ti] },
        );
    }
    if (blocks[0] == 0 or blocks[1] == 0) {
        writeFdAll(1, "TABLE-BENCH-FAIL\n");
        return error.EngramDirectUnavailable;
    }

    var per_rep_tokens: usize = 0;
    var max_c: usize = 0;
    for (TABLE_BENCH_WIDTHS) |w| {
        per_rep_tokens += w;
        max_c = @max(max_c, w);
    }
    const needed_tokens = TABLE_BENCH_P0 + reps * per_rep_tokens;

    const real_comp = blk: {
        const real_toks = try readTokenIdsFile(io, gpa, tokens_path);
        defer gpa.free(real_toks);
        if (real_toks.len < needed_tokens) {
            writeStderrFmt("[table-bench] tokens-file too short: have {d} tokens, need >= {d} (p0={d}, reps={d})\n", .{ real_toks.len, needed_tokens, TABLE_BENCH_P0, reps });
            return error.TokenFileTooShort;
        }
        const comp = try gpa.alloc(u32, needed_tokens);
        mapCompressedIds(real_toks[0..needed_tokens], token_map, comp);
        break :blk comp;
    };
    defer gpa.free(real_comp);

    const det_comp = blk: {
        const det_toks = try gpa.alloc(u32, needed_tokens);
        defer gpa.free(det_toks);
        fillDisjointDeterministicStream(det_toks, cfg.vocab_size, TABLE_BENCH_P0);
        const comp = try gpa.alloc(u32, needed_tokens);
        mapCompressedIds(det_toks, token_map, comp);
        break :blk comp;
    };
    defer gpa.free(det_comp);

    const grid_buf = try gpa.alloc(u32, max_c * nt * nc);
    defer gpa.free(grid_buf);
    const uniq_buf = try gpa.alloc(u32, max_c * nc);
    defer gpa.free(uniq_buf);

    if (warmup > 0 and TABLE_BENCH_P0 >= 2048) {
        const w_c: usize = 2048;
        const w_pos = TABLE_BENCH_P0 - w_c;
        const w_grid = grid_buf[0 .. w_c * nt * nc];
        hash.rows(real_comp[0..w_pos], real_comp[w_pos .. w_pos + w_c], w_grid);
        for (0..2) |ti| {
            const uniq = try extractSortDedupLayerRows(w_grid, w_c, nt, slots[ti], nc, uniq_buf);
            const raw = try tables[ti].readSorted(gpa, uniq, engram.GRID_THREADS);
            raw.deinit(gpa);
        }
    }

    var samples_ms: [2][]f64 = .{ try gpa.alloc(f64, reps), try gpa.alloc(f64, reps) };
    defer for (samples_ms) |s| gpa.free(s);
    var hs_ms: [2][]f64 = .{ try gpa.alloc(f64, reps), try gpa.alloc(f64, reps) };
    defer for (hs_ms) |s| gpa.free(s);
    var nu_samples: [2][]usize = .{ try gpa.alloc(usize, reps), try gpa.alloc(usize, reps) };
    defer for (nu_samples) |s| gpa.free(s);
    var ubc_pre_samples: [2][]usize = .{ try gpa.alloc(usize, reps), try gpa.alloc(usize, reps) };
    defer for (ubc_pre_samples) |s| gpa.free(s);
    var ubc_pop_samples: [2][]usize = .{ try gpa.alloc(usize, reps), try gpa.alloc(usize, reps) };
    defer for (ubc_pop_samples) |s| gpa.free(s);
    const scratch_f64 = try gpa.alloc(f64, reps);
    defer gpa.free(scratch_f64);
    const scratch_usize = try gpa.alloc(usize, reps);
    defer gpa.free(scratch_usize);

    var cells: [2 * TABLE_BENCH_WIDTHS.len * TABLE_BENCH_LAYERS.len]TableBenchCell = undefined;
    var cell_idx: usize = 0;
    var line_buf: [1280]u8 = undefined;
    const has_separate_json_file = json_path != null and json_out.fd != 1;

    for ([_]TableBenchSource{ .real_text, .deterministic }) |source| {
        const comp = if (source == .real_text) real_comp else det_comp;
        var cursor: usize = TABLE_BENCH_P0;
        for (TABLE_BENCH_WIDTHS) |c_rows| {
            const chunk_lo = cursor;
            var ubc_ok: [2]bool = .{ true, true };
            for (0..reps) |rep| {
                const pos = cursor;
                cursor += c_rows;
                const cur_grid = grid_buf[0 .. c_rows * nt * nc];
                const t_h0 = monoUs();
                hash.rows(comp[0..pos], comp[pos .. pos + c_rows], cur_grid);
                const h_us = monoUs() - t_h0;
                for (0..2) |ti| {
                    const t_s0 = monoUs();
                    const uniq = try extractSortDedupLayerRows(cur_grid, c_rows, nt, slots[ti], nc, uniq_buf);
                    const s_us = monoUs() - t_s0;
                    hs_ms[ti][rep] = @as(f64, @floatFromInt(@max(h_us + s_us, 0))) / 1000.0;
                    nu_samples[ti][rep] = uniq.len;

                    const pre_ubc = probes[ti].countCachedRows(&tables[ti], uniq);
                    if (pre_ubc == null) ubc_ok[ti] = false;
                    const pre_n = pre_ubc orelse uniq.len;
                    ubc_pre_samples[ti][rep] = pre_n;

                    const t_r0 = monoUs();
                    const raw = try tables[ti].readSorted(gpa, uniq, engram.GRID_THREADS);
                    const r_us = monoUs() - t_r0;
                    raw.deinit(gpa);
                    samples_ms[ti][rep] = @as(f64, @floatFromInt(@max(r_us, 0))) / 1000.0;

                    const post_ubc = probes[ti].countCachedRows(&tables[ti], uniq) orelse pre_n;
                    ubc_pop_samples[ti][rep] = post_ubc -| pre_n;
                }
            }
            const chunk_hi = cursor;
            for (0..2) |ti| {
                const layer = TABLE_BENCH_LAYERS[ti];
                const med_nu = try medianUsize(nu_samples[ti], scratch_usize);
                const med_ubc = try medianUsize(ubc_pre_samples[ti], scratch_usize);
                const med_pop = try medianUsize(ubc_pop_samples[ti], scratch_usize);
                var max_ubc: usize = 0;
                for (ubc_pre_samples[ti]) |v| max_ubc = @max(max_ubc, v);
                var sum_nu: u64 = 0;
                for (nu_samples[ti]) |v| sum_nu += v;

                const ev = try evaluateTableBenchCell(source, c_rows, layer, blocks[ti], struct_cache_flags[ti], ubc_ok[ti], max_ubc, med_nu, samples_ms[ti], scratch_f64);
                const hs_st = try sampleMsStats(hs_ms[ti], scratch_f64);
                const distinct_all = try countCrossRepDistinct(gpa, &hash, comp, chunk_lo, c_rows, reps, slots[ti], grid_buf, uniq_buf);
                const pc_poss = struct_cache_flags[ti] or !ubc_ok[ti] or max_ubc > 0;
                const cell: TableBenchCell = .{
                    .source = source,
                    .chunk_rows = c_rows,
                    .layer = layer,
                    .p0_start = TABLE_BENCH_P0,
                    .chunk_tok_lo = chunk_lo,
                    .chunk_tok_hi = chunk_hi,
                    .reps = reps,
                    .block_bytes = blocks[ti],
                    .records = tables[ti].records(),
                    .structural_cache_possible = struct_cache_flags[ti],
                    .ubc_verified = ubc_ok[ti],
                    .ubc_cached_rows = med_ubc,
                    .ubc_cached_rows_max = max_ubc,
                    .ubc_populated_rows = med_pop,
                    .page_cache_possible = pc_poss,
                    .lookup_rows = c_rows * nc,
                    .rows_read = med_nu,
                    .distinct_rows = med_nu,
                    .total_rows_read = sum_nu,
                    .distinct_rows_across_reps = distinct_all,
                    .median_ms = ev.stats.median_ms,
                    .p90_ms = ev.stats.p90_ms,
                    .cold_adjusted_median_ms = ev.cold_adjusted_ms,
                    .hash_sort_median_ms = hs_st.median_ms,
                    .step1_pread_ms = ev.spec.step1_pread_ms,
                    .threshold_ms = ev.spec.threshold_ms,
                    .gate_pass = ev.gate_pass,
                };
                cells[cell_idx] = cell;
                cell_idx += 1;
                const line = try formatTableBenchJson(&line_buf, cell);
                writeFdAll(1, line);
                if (has_separate_json_file) json_out.writeLine(line);
            }
        }
    }

    if (evaluateTableBenchGate(cells[0..cell_idx])) {
        writeFdAll(1, "TABLE-BENCH-PASS\n");
        return;
    }
    writeFdAll(1, "TABLE-BENCH-FAIL\n");
    return error.TableBenchGateFailed;
}

// ── Tests ──

const testing = std.testing;

test "pp_step1_table: sort/dedup extracts per-layer rows, deduplicates repeated n-grams, and preserves 100%-distinct streams" {
    var hand_uniq: [12]u32 = undefined;
    const grid = [_]u32{
        9, 3, 9,  100, 200, 100,
        3, 1, 9,  200, 300, 100,
        5, 3, 1,  400, 100, 200,
        9, 5, 5,  300, 300, 400,
    };
    const l0 = try extractSortDedupLayerRows(&grid, 4, 2, 0, 3, &hand_uniq);
    try testing.expectEqualSlices(u32, &.{ 1, 3, 5, 9 }, l0);
    const l1 = try extractSortDedupLayerRows(&grid, 4, 2, 1, 3, &hand_uniq);
    try testing.expectEqualSlices(u32, &.{ 100, 200, 300, 400 }, l1);
    try testing.expectError(error.InvalidGridShape, extractSortDedupLayerRows(&grid, 4, 2, 2, 3, &hand_uniq));

    var cfg: config_mod.Config = .{};
    cfg.model_type = "deepseek_v41";
    cfg.dsv41_n_engram_layers = 2;
    cfg.dsv41_engram_layers[0] = 1;
    cfg.dsv41_engram_layers[1] = 14;
    cfg.dsv41_engram_rows[0] = 384006168;
    cfg.dsv41_engram_rows[1] = 384016682;
    cfg.dsv41_engram_max_ngram = 4;
    cfg.dsv41_engram_vocab_size = 16000000;
    cfg.dsv41_engram_n_heads = 8;
    cfg.dsv41_engram_compressed_vocab = 99092;
    const hash = try engram.Hash.init(&cfg, 2);
    const nc: usize = hash.cols();
    try testing.expectEqual(@as(usize, 24), nc);

    const gpa = testing.allocator;
    const p0: usize = 64;
    const c_len: usize = 64;
    const rep_stream = try gpa.alloc(u32, p0 + c_len);
    defer gpa.free(rep_stream);
    for (rep_stream, 0..) |*v, idx| v.* = @intCast(10 + (idx % 8));

    const det_stream = try gpa.alloc(u32, p0 + c_len);
    defer gpa.free(det_stream);
    pp_step1.fillDeterministicTokens(det_stream, cfg.dsv41_engram_compressed_vocab, 7);

    const grid_buf = try gpa.alloc(u32, c_len * 2 * nc);
    defer gpa.free(grid_buf);
    const uniq_buf = try gpa.alloc(u32, c_len * nc);
    defer gpa.free(uniq_buf);

    hash.rows(rep_stream[0..p0], rep_stream[p0 .. p0 + c_len], grid_buf);
    const rep_u0 = try extractSortDedupLayerRows(grid_buf, c_len, 2, 0, nc, uniq_buf);
    try testing.expect(rep_u0.len < (c_len * nc) / 4);
    for (1..rep_u0.len) |idx| try testing.expect(rep_u0[idx - 1] < rep_u0[idx]);

    hash.rows(det_stream[0..p0], det_stream[p0 .. p0 + c_len], grid_buf);
    const det_u0 = try extractSortDedupLayerRows(grid_buf, c_len, 2, 0, nc, uniq_buf);
    try testing.expectEqual(c_len * nc, det_u0.len);

    var union_ids = [_]u32{ 42, 7, 42, 19, 7, 1, 99, 19 };
    try testing.expectEqual(@as(usize, 5), countDistinctInPlace(&union_ids));
}

test "pp_step1_table: gate thresholds, UBC cold-adjusted latency, JSON schema, and fail-closed suite check" {
    try testing.expectEqual(@as(f64, 172.8), tableBenchSpecFor(1, 2048).?.threshold_ms);
    try testing.expectEqual(@as(f64, 299.1), tableBenchSpecFor(1, 4096).?.threshold_ms);
    try testing.expectEqual(@as(f64, 567.1), tableBenchSpecFor(1, 8192).?.threshold_ms);
    try testing.expectEqual(@as(f64, 179.0), tableBenchSpecFor(14, 2048).?.threshold_ms);
    try testing.expectEqual(@as(f64, 303.3), tableBenchSpecFor(14, 4096).?.threshold_ms);
    try testing.expectEqual(@as(f64, 613.3), tableBenchSpecFor(14, 8192).?.threshold_ms);
    try testing.expect(tableBenchSpecFor(2, 2048) == null);
    try testing.expect(tableBenchSpecFor(1, 1024) == null);

    var scratch: [10]f64 = undefined;
    const s5 = try sampleMsStats(&.{ 140.0, 110.0, 180.0, 120.0, 130.0 }, &scratch);
    try testing.expectApproxEqAbs(@as(f64, 130.0), s5.median_ms, 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 180.0), s5.p90_ms, 1e-9);

    const s10 = try sampleMsStats(&.{ 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0 }, &scratch);
    try testing.expectApproxEqAbs(@as(f64, 55.0), s10.median_ms, 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 90.0), s10.p90_ms, 1e-9);

    var cells: [12]TableBenchCell = undefined;
    var ci: usize = 0;
    for (TABLE_BENCH_SPECS) |spec| {
        const distinct = spec.chunk_rows * 10;
        const real_ev = try evaluateTableBenchCell(.real_text, spec.chunk_rows, spec.layer, 4096, false, true, 0, distinct, &.{ spec.threshold_ms - 5.0, spec.threshold_ms, spec.threshold_ms + 4.0 }, &scratch);
        try testing.expect(real_ev.gate_pass);
        cells[ci] = .{
            .source = .real_text,
            .chunk_rows = spec.chunk_rows,
            .layer = spec.layer,
            .p0_start = TABLE_BENCH_P0,
            .chunk_tok_lo = TABLE_BENCH_P0,
            .chunk_tok_hi = TABLE_BENCH_P0 + 3 * spec.chunk_rows,
            .reps = 3,
            .block_bytes = 4096,
            .records = false,
            .structural_cache_possible = false,
            .ubc_verified = true,
            .ubc_cached_rows = 0,
            .ubc_cached_rows_max = 0,
            .ubc_populated_rows = 0,
            .page_cache_possible = false,
            .lookup_rows = spec.chunk_rows * 24,
            .rows_read = distinct,
            .distinct_rows = distinct,
            .total_rows_read = distinct * 3,
            .distinct_rows_across_reps = distinct * 2,
            .median_ms = real_ev.stats.median_ms,
            .p90_ms = real_ev.stats.p90_ms,
            .cold_adjusted_median_ms = real_ev.cold_adjusted_ms,
            .hash_sort_median_ms = 1.2,
            .step1_pread_ms = spec.step1_pread_ms,
            .threshold_ms = spec.threshold_ms,
            .gate_pass = real_ev.gate_pass,
        };
        ci += 1;

        const det_ev = try evaluateTableBenchCell(.deterministic, spec.chunk_rows, spec.layer, 4096, false, true, 0, distinct, &.{ spec.threshold_ms + 25.0, spec.threshold_ms + 30.0, spec.threshold_ms + 40.0 }, &scratch);
        try testing.expect(!det_ev.gate_pass);
        cells[ci] = cells[ci - 1];
        cells[ci].source = .deterministic;
        cells[ci].median_ms = det_ev.stats.median_ms;
        cells[ci].p90_ms = det_ev.stats.p90_ms;
        cells[ci].cold_adjusted_median_ms = det_ev.cold_adjusted_ms;
        cells[ci].gate_pass = det_ev.gate_pass;
        ci += 1;
    }

    try testing.expect(evaluateTableBenchGate(&cells));
    try testing.expect(!evaluateTableBenchGate(cells[0 .. cells.len - 2]));

    // Borderline 168.0 ms <= 172.8 ms fails closed when 4% UBC hits push cold_adjusted_ms to 175.0 ms.
    const borderline = try evaluateTableBenchCell(.real_text, 2048, 1, 4096, false, true, 800, 20000, &.{ 167.0, 168.0, 169.0 }, &scratch);
    try testing.expect(!borderline.gate_pass);
    try testing.expectApproxEqAbs(@as(f64, 175.0), borderline.cold_adjusted_ms, 1e-6);

    // > 5% UBC residency fails closed even when fast.
    const warm_ubc = try evaluateTableBenchCell(.real_text, 2048, 1, 4096, false, true, 1200, 20000, &.{ 80.0, 82.0, 84.0 }, &scratch);
    try testing.expect(!warm_ubc.gate_pass);

    // Unverified UBC fails closed.
    const unverified = try evaluateTableBenchCell(.real_text, 2048, 1, 4096, false, false, 0, 20000, &.{ 80.0, 82.0, 84.0 }, &scratch);
    try testing.expect(!unverified.gate_pass);

    for (0..cells.len) |idx| {
        if (cells[idx].source != .real_text) continue;
        var mutated = cells;
        mutated[idx].median_ms = mutated[idx].threshold_ms + 0.1;
        mutated[idx].gate_pass = false;
        try testing.expect(!evaluateTableBenchGate(&mutated));

        var high_ubc = cells;
        high_ubc[idx].ubc_cached_rows_max = high_ubc[idx].distinct_rows / 10;
        try testing.expect(!evaluateTableBenchGate(&high_ubc));

        var unv = cells;
        unv[idx].ubc_verified = false;
        try testing.expect(!evaluateTableBenchGate(&unv));

        var zero_block = cells;
        zero_block[idx].block_bytes = 0;
        try testing.expect(!evaluateTableBenchGate(&zero_block));
    }

    var line_buf: [1280]u8 = undefined;
    const line = try formatTableBenchJson(&line_buf, cells[0]);
    var parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, std.mem.trim(u8, line, "\n"), .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("table_bench", parsed.value.object.get("kind").?.string);
    try testing.expectEqualStrings("real_text", parsed.value.object.get("source").?.string);
    try testing.expectEqual(@as(i64, 2048), parsed.value.object.get("chunk_rows").?.integer);
    try testing.expectEqual(@as(i64, 1), parsed.value.object.get("layer").?.integer);
    try testing.expectEqual(true, parsed.value.object.get("ubc_verified").?.bool);
    try testing.expectEqual(false, parsed.value.object.get("page_cache_possible").?.bool);
    try testing.expectEqualStrings("PASS", parsed.value.object.get("gate_verdict").?.string);
}

test "pp_step1_table: tokenize-ids, blockOf/readSorted, and Darwin UbcProbe mincore detect cached vs uncached pages" {
    if (comptime builtin.os.tag == .windows) return error.SkipZigTest;
    const gpa = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const byte_tok = @import("../../sdk/testing/byte_tokenizer.zig");

    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const tok_json = try byte_tok.json(arena.allocator(), &.{});

    var td = testing.tmpDir(.{});
    defer td.cleanup();
    try td.dir.writeFile(io, .{ .sub_path = "tokenizer.json", .data = tok_json });
    try td.dir.writeFile(io, .{ .sub_path = "sample.txt", .data = "Hello Plan 75!\nconst x: u32 = 42;\n" });

    var root_buf: [std.fs.max_path_bytes]u8 = undefined;
    const root = root_buf[0..try td.dir.realPath(io, &root_buf)];
    const in_path = try std.fmt.allocPrint(arena.allocator(), "{s}/sample.txt", .{root});
    const out_path = try std.fmt.allocPrint(arena.allocator(), "{s}/sample.u32", .{root});

    try pp_step1.cmdProbe(io, gpa, &.{ "tokenize-ids", "--model", root, "--in", in_path, "--out", out_path });
    const ids = try readTokenIdsFile(io, gpa, out_path);
    defer gpa.free(ids);
    try testing.expectEqual(@as(usize, 34), ids.len);
    try testing.expectEqual(@as(u32, 'H'), ids[0]);

    try td.dir.createDirPath(io, "engram");
    try td.dir.writeFile(io, .{
        .sub_path = "engram/engram-manifest.json",
        .data = "{\"layers\":[{\"layer_id\":1,\"file\":\"l1.bin\",\"rows\":8,\"record_bytes\":33,\"quant\":{\"bits\":8,\"group_size\":32,\"mode\":\"mxfp8\",\"head_dim\":32}}]}",
    });
    var l1_bytes: [8 * 33]u8 = undefined;
    for (&l1_bytes, 0..) |*b, idx| b.* = @truncate(idx *% 13 +% 5);
    try td.dir.writeFile(io, .{ .sub_path = "engram/l1.bin", .data = &l1_bytes });

    var tbl = try engram.Table.open(gpa, root, 0, 1, 8, 32);
    defer tbl.close();
    try testing.expectEqual(@as(u64, 0), edir.blockOf(tbl.fd));
    tbl.readDirect();
    if (comptime builtin.os.tag.isDarwin()) {
        try testing.expect(edir.blockOf(tbl.fd) > 0);
        var probe = UbcProbe.open(tbl.fd);
        defer probe.close();
        try testing.expect(probe.map != null);
        // writeFile wrote through the OS page cache, so mincore sees all 3 rows resident in UBC.
        const cached_n = probe.countCachedRows(&tbl, &.{ 1, 5, 7 }).?;
        try testing.expectEqual(@as(usize, 3), cached_n);
    }
    const raw = try tbl.readSorted(gpa, &.{ 1, 5, 7 }, engram.GRID_THREADS);
    defer raw.deinit(gpa);
    try testing.expectEqualSlices(u8, l1_bytes[1 * 33 ..][0..32], raw.w[0..32]);
    try testing.expectEqualSlices(u8, l1_bytes[5 * 33 ..][0..32], raw.w[32..64]);
    try testing.expectEqualSlices(u8, l1_bytes[7 * 33 ..][0..32], raw.w[64..96]);

    if (comptime builtin.os.tag.isDarwin()) {
        const sparse_z = try std.fmt.allocPrintSentinel(arena.allocator(), "{s}/engram/sparse.bin", .{root}, 0);
        const sfd = std.c.open(sparse_z, .{ .ACCMODE = .RDWR, .CREAT = true, .TRUNC = true, .CLOEXEC = true }, @as(std.c.mode_t, 0o600));
        try testing.expect(sfd >= 0);
        defer _ = std.c.close(sfd);
        try testing.expectEqual(@as(c_int, 0), std.c.ftruncate(sfd, 65536));
        const fake_tbl: engram.Table = .{
            .fd = sfd, .rows = 8, .head_dim = 32, .mode = .mxfp8, .bits = 8, .group_size = 32,
            .w_off = 0, .w_stride = 4096, .wlen = 32, .s_off = 32768, .s_stride = 4096, .slen = 1, .direct = true,
        };
        var sprobe = UbcProbe.open(sfd);
        defer sprobe.close();
        try testing.expectEqual(@as(usize, 0), sprobe.countCachedRows(&fake_tbl, &.{ 1, 5, 7 }).?);
    }
}
