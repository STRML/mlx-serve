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
pub const MIN_TABLE_BENCH_REPS: usize = 3;
pub const MAX_UBC_CACHED_RATIO: f64 = 0.05;
pub const MIN_REAL_TEXT_DISTINCT_RATIO: f64 = 0.30;

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

pub fn describeTableLayout(t: *const engram.Table) []const u8 {
    if (t.records()) return if (t.mode == .mxfp8) "mxfp8_records" else "records";
    return switch (t.mode) {
        .mxfp8 => if (t.bits == 8 and t.group_size == 32 and t.wlen > 0 and t.slen > 0) "iq_mxfp8_split" else "mxfp8_other",
        .affine => "affine_split",
        .gguf => "gguf",
        .nvfp4 => "nvfp4",
        .mxfp4 => "mxfp4",
    };
}

pub fn isSpecIqTableLayout(t: *const engram.Table) bool {
    return !t.records() and t.mode == .mxfp8 and t.bits == 8 and t.group_size == 32 and t.wlen > 0 and t.slen > 0;
}

pub fn hasRepresentativeDistinctRatio(distinct_rows: usize, lookup_rows: usize) bool {
    if (lookup_rows == 0) return false;
    return @as(u128, distinct_rows) * 10 >= @as(u128, lookup_rows) * 3;
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
    host: []const u8 = "unknown",
    pack_path: []const u8 = "",
    shard_path: []const u8 = "",
    shard_sha256: [64]u8 = @splat('0'),
    l1_shard_path: []const u8 = "",
    l1_shard_sha256: [64]u8 = @splat('0'),
    l14_shard_path: []const u8 = "",
    l14_shard_sha256: [64]u8 = @splat('0'),
    source: TableBenchSource,
    chunk_rows: usize,
    layer: usize,
    p0_start: usize,
    chunk_tok_lo: usize,
    chunk_tok_hi: usize,
    reps: usize,
    table_layout: []const u8 = "iq_mxfp8_split",
    block_bytes: u64,
    records: bool,
    ubc_control: bool = false,
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
    read_sorted_median_ms: f64,
    read_sorted_p90_ms: f64,
    extract_sort_median_ms: f64 = 0.0,
    hash_sort_median_ms: f64,
    step1_pread_ms: f64,
    threshold_ms: f64,
    gate_pass: ?bool = null,
    gate_reason: ?[]const u8 = null,
};

pub fn cellGateFailureReason(
    spec: TableBenchSpec,
    table_layout: []const u8,
    records: bool,
    block_bytes: u64,
    structural_cache_possible: bool,
    ubc_verified: bool,
    ubc_control: bool,
    ubc_cached_rows_max: usize,
    lookup_rows: usize,
    distinct_rows: usize,
    io_median_ms: f64,
) ?[]const u8 {
    if (records or !std.mem.eql(u8, table_layout, "iq_mxfp8_split")) return "invalid_table_layout";
    if (block_bytes == 0 or structural_cache_possible) return "direct_io_unavailable";
    if (!ubc_verified) return "ubc_unverified";
    if (!hasRepresentativeDistinctRatio(distinct_rows, lookup_rows)) return "unrepresentative ids";
    const ubc_ratio = @as(f64, @floatFromInt(ubc_cached_rows_max)) / @as(f64, @floatFromInt(distinct_rows));
    if (ubc_control or ubc_ratio > MAX_UBC_CACHED_RATIO) return "ubc_cached";
    const adj_ms = coldAdjustedMs(io_median_ms, ubc_cached_rows_max, distinct_rows);
    if (io_median_ms > spec.threshold_ms or adj_ms > spec.threshold_ms) return "threshold_exceeded";
    return null;
}

pub fn cellColdGatePass(
    spec: TableBenchSpec,
    table_layout: []const u8,
    records: bool,
    block_bytes: u64,
    structural_cache_possible: bool,
    ubc_verified: bool,
    ubc_control: bool,
    ubc_cached_rows_max: usize,
    lookup_rows: usize,
    distinct_rows: usize,
    io_median_ms: f64,
) bool {
    return cellGateFailureReason(
        spec,
        table_layout,
        records,
        block_bytes,
        structural_cache_possible,
        ubc_verified,
        ubc_control,
        ubc_cached_rows_max,
        lookup_rows,
        distinct_rows,
        io_median_ms,
    ) == null;
}

pub const CellEval = struct {
    io_stats: MsStats,
    read_stats: MsStats,
    sort_stats: MsStats,
    spec: TableBenchSpec,
    gated: bool,
    cold_adjusted_ms: f64,
    gate_pass: ?bool,
    gate_verdict: []const u8,
    gate_reason: ?[]const u8,
};

pub fn evaluateTableBenchCell(
    source: TableBenchSource,
    chunk_rows: usize,
    layer: usize,
    table_layout: []const u8,
    records: bool,
    block_bytes: u64,
    structural_cache_possible: bool,
    ubc_verified: bool,
    ubc_control: bool,
    ubc_cached_rows_max: usize,
    lookup_rows: usize,
    distinct_rows: usize,
    read_samples_ms: []const f64,
    sort_samples_ms: []const f64,
    io_scratch: []f64,
    scratch: []f64,
) !CellEval {
    const n = read_samples_ms.len;
    if (n == 0 or sort_samples_ms.len != n or io_scratch.len < n) return error.InvalidSampleCount;
    const spec = tableBenchSpecFor(layer, chunk_rows) orelse return error.UnknownTableBenchCell;
    const read_stats = try sampleMsStats(read_samples_ms, scratch);
    const sort_stats = try sampleMsStats(sort_samples_ms, scratch);
    for (read_samples_ms, sort_samples_ms, 0..) |r_ms, s_ms, idx| io_scratch[idx] = r_ms + s_ms;
    const io_stats = try sampleMsStats(io_scratch[0..n], scratch);
    const adj_ms = coldAdjustedMs(io_stats.median_ms, ubc_cached_rows_max, distinct_rows);
    if (!source.isGated()) {
        return .{
            .io_stats = io_stats,
            .read_stats = read_stats,
            .sort_stats = sort_stats,
            .spec = spec,
            .gated = false,
            .cold_adjusted_ms = adj_ms,
            .gate_pass = null,
            .gate_verdict = "n/a",
            .gate_reason = null,
        };
    }
    const fail_reason = cellGateFailureReason(
        spec,
        table_layout,
        records,
        block_bytes,
        structural_cache_possible,
        ubc_verified,
        ubc_control,
        ubc_cached_rows_max,
        lookup_rows,
        distinct_rows,
        io_stats.median_ms,
    );
    const pass = fail_reason == null;
    return .{
        .io_stats = io_stats,
        .read_stats = read_stats,
        .sort_stats = sort_stats,
        .spec = spec,
        .gated = true,
        .cold_adjusted_ms = adj_ms,
        .gate_pass = pass,
        .gate_verdict = if (pass) "PASS" else "FAIL",
        .gate_reason = fail_reason,
    };
}

pub fn evaluateTableBenchGate(cells: []const TableBenchCell) bool {
    for (cells) |c| {
        if (c.ubc_control) return false;
    }
    for (TABLE_BENCH_SPECS) |spec| {
        var found = false;
        for (cells) |c| {
            if (c.source != .real_text) continue;
            if (c.layer != spec.layer or c.chunk_rows != spec.chunk_rows) continue;
            if (found) return false;
            if (c.gate_pass != true) return false;
            if (c.median_ms + 1e-9 < c.read_sorted_median_ms) return false;
            if (!cellColdGatePass(
                spec,
                c.table_layout,
                c.records,
                c.block_bytes,
                c.structural_cache_possible,
                c.ubc_verified,
                c.ubc_control,
                c.ubc_cached_rows_max,
                c.lookup_rows,
                c.distinct_rows,
                c.median_ms,
            )) return false;
            found = true;
        }
        if (!found) return false;
    }
    return true;
}

fn writeJsonStr(w: *std.Io.Writer, s: []const u8) !void {
    try w.writeByte('"');
    for (s) |ch| {
        switch (ch) {
            '"' => try w.writeAll("\\\""),
            '\\' => try w.writeAll("\\\\"),
            '\n' => try w.writeAll("\\n"),
            '\r' => try w.writeAll("\\r"),
            '\t' => try w.writeAll("\\t"),
            else => {
                if (ch < 0x20) {
                    try w.print("\\u00{x:0>2}", .{ch});
                } else {
                    try w.writeByte(ch);
                }
            },
        }
    }
    try w.writeByte('"');
}

pub fn formatTableBenchJson(buf: []u8, c: TableBenchCell) ![]const u8 {
    const f_nocache = if (c.block_bytes > 0) "true" else "false";
    const rec = if (c.records) "true" else "false";
    const ubc_ctrl = if (c.ubc_control) "true" else "false";
    const ubc_ver = if (c.ubc_verified) "true" else "false";
    const pc_poss = if (c.page_cache_possible) "true" else "false";
    const pc_note = if (c.ubc_control)
        "ubc_control_page_cache_warmed"
    else if (c.structural_cache_possible or c.block_bytes == 0)
        "os_page_cache_fallback_active"
    else if (!c.ubc_verified)
        "ubc_mincore_unverified"
    else if (c.ubc_cached_rows_max > 0)
        "preexisting_ubc_pages_detected_cold_adjusted"
    else
        "verified_zero_ubc_pages_before_read";
    const dist_ratio = if (c.lookup_rows > 0) @as(f64, @floatFromInt(c.distinct_rows)) / @as(f64, @floatFromInt(c.lookup_rows)) else 0.0;
    const ubc_ratio = if (c.distinct_rows > 0) @as(f64, @floatFromInt(c.ubc_cached_rows_max)) / @as(f64, @floatFromInt(c.distinct_rows)) else 1.0;
    const adj_out = if (std.math.isFinite(c.cold_adjusted_median_ms)) c.cold_adjusted_median_ms else 999999.0;
    const speedup = if (c.median_ms > 0) c.step1_pread_ms / c.median_ms else 0.0;
    var w = std.Io.Writer.fixed(buf);
    try w.writeAll("{\"kind\":\"table_bench\",\"host\":");
    try writeJsonStr(&w, c.host);
    try w.writeAll(",\"pack_path\":");
    try writeJsonStr(&w, c.pack_path);
    try w.writeAll(",\"shard_path\":");
    try writeJsonStr(&w, c.shard_path);
    try w.print(",\"shard_sha256\":\"{s}\",\"l1_shard_path\":", .{&c.shard_sha256});
    try writeJsonStr(&w, c.l1_shard_path);
    try w.print(",\"l1_shard_sha256\":\"{s}\",\"l14_shard_path\":", .{&c.l1_shard_sha256});
    try writeJsonStr(&w, c.l14_shard_path);
    try w.print(",\"l14_shard_sha256\":\"{s}\",\"source\":\"{s}\",\"chunk_rows\":{d},\"layer\":{d},\"p0_start\":{d},\"chunk_tok_lo\":{d},\"chunk_tok_hi\":{d},\"reps\":{d},\"grid_threads\":{d},\"table_layout\":", .{
        &c.l14_shard_sha256,
        c.source.name(),
        c.chunk_rows,
        c.layer,
        c.p0_start,
        c.chunk_tok_lo,
        c.chunk_tok_hi,
        c.reps,
        engram.GRID_THREADS,
    });
    try writeJsonStr(&w, c.table_layout);
    try w.print(
        ",\"block_bytes\":{d},\"records\":{s},\"f_nocache\":{s},\"ubc_control\":{s},\"ubc_verified\":{s},\"ubc_cached_rows\":{d},\"ubc_cached_rows_max\":{d},\"ubc_cached_ratio\":{d:.4},\"ubc_populated_rows\":{d},\"page_cache_possible\":{s},\"page_cache_note\":\"{s}\",",
        .{ c.block_bytes, rec, f_nocache, ubc_ctrl, ubc_ver, c.ubc_cached_rows, c.ubc_cached_rows_max, ubc_ratio, c.ubc_populated_rows, pc_poss, pc_note },
    );
    try w.print(
        "\"lookup_rows\":{d},\"rows_read\":{d},\"distinct_rows\":{d},\"distinct_ratio\":{d:.4},\"total_rows_read\":{d},\"distinct_rows_across_reps\":{d},\"median_ms\":{d:.2},\"p90_ms\":{d:.2},\"cold_adjusted_median_ms\":{d:.2},\"read_sorted_median_ms\":{d:.2},\"read_sorted_p90_ms\":{d:.2},\"extract_sort_median_ms\":{d:.2},\"hash_sort_median_ms\":{d:.2},\"step1_pread_ms\":{d:.1},\"threshold_ms\":{d:.1},\"speedup_vs_step1\":{d:.3},",
        .{
            c.lookup_rows,
            c.rows_read,
            c.distinct_rows,
            dist_ratio,
            c.total_rows_read,
            c.distinct_rows_across_reps,
            c.median_ms,
            c.p90_ms,
            adj_out,
            c.read_sorted_median_ms,
            c.read_sorted_p90_ms,
            c.extract_sort_median_ms,
            c.hash_sort_median_ms,
            c.step1_pread_ms,
            c.threshold_ms,
            speedup,
        },
    );
    if (!c.source.isGated()) {
        try w.writeAll("\"gated\":false,\"gate_verdict\":\"n/a\"}\n");
        return w.buffered();
    }
    const pass = (c.gate_pass == true) and (c.gate_reason == null);
    const gpass = if (pass) "true" else "false";
    const verdict = if (pass) "PASS" else "FAIL";
    const reason = c.gate_reason orelse (if (pass) "ok" else "gate_failed");
    try w.print("\"gated\":true,\"gate_pass\":{s},\"gate_verdict\":\"{s}\",\"gate_reason\":", .{ gpass, verdict });
    try writeJsonStr(&w, reason);
    try w.writeAll("}\n");
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
    var buf: [768]u8 = undefined;
    const line = std.fmt.bufPrint(&buf, fmt, args) catch return;
    writeFdAll(2, line);
}

fn currentHostName(buf: *[std.posix.HOST_NAME_MAX]u8) []const u8 {
    if (comptime builtin.os.tag == .windows) return "windows";
    return std.posix.gethostname(buf) catch "unknown";
}

fn fdPathAlloc(gpa: std.mem.Allocator, fd: c_int, fallback_dir: []const u8, layer: usize) ![]u8 {
    if (comptime builtin.os.tag.isDarwin()) {
        const darwin_f_getpath: c_int = 50;
        var pbuf: [std.fs.max_path_bytes]u8 = undefined;
        if (std.c.fcntl(fd, darwin_f_getpath, &pbuf) != -1) {
            const end = std.mem.indexOfScalar(u8, &pbuf, 0) orelse pbuf.len;
            if (end > 0) return gpa.dupe(u8, pbuf[0..end]);
        }
    } else if (comptime builtin.os.tag == .linux) {
        var link_buf: [64]u8 = undefined;
        var out_buf: [std.fs.max_path_bytes]u8 = undefined;
        if (std.fmt.bufPrintSentinel(&link_buf, "/proc/self/fd/{d}", .{fd}, 0)) |lp| {
            const got = std.c.readlink(lp.ptr, &out_buf, out_buf.len);
            if (got > 0 and @as(usize, @intCast(got)) <= out_buf.len) {
                return gpa.dupe(u8, out_buf[0..@intCast(got)]);
            }
        } else |_| {}
    }
    return std.fmt.allocPrint(gpa, "{s}/engram-layer-{d}", .{ fallback_dir, layer });
}

pub fn sha256FdHex(fd: c_int) ![64]u8 {
    const end = crt.seekEnd(fd);
    if (end < 0) return error.ShardStatFailed;
    const size: u64 = @intCast(end);
    const chunk_len: usize = 4 << 20;
    const buf = try std.heap.page_allocator.alignedAlloc(u8, .fromByteUnits(16384), chunk_len);
    defer std.heap.page_allocator.free(buf);
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    var off: u64 = 0;
    while (off < size) {
        const req: usize = if (off % 4096 == 0) chunk_len else @intCast(@min(@as(u64, chunk_len), size - off));
        const got = crt.pread(fd, buf.ptr, req, @intCast(off));
        if (got <= 0) return error.ShardReadFailed;
        const n: usize = @intCast(got);
        h.update(buf[0..n]);
        off += n;
    }
    var digest: [32]u8 = undefined;
    h.final(&digest);
    return std.fmt.bytesToHex(digest, .lower);
}

fn plainReadRegionIntoPageCache(fd: c_int, start_off: u64, span_bytes: u64, buf: []u8) !void {
    if (comptime builtin.os.tag == .windows) return error.UnsupportedPlatform;
    if (span_bytes == 0) return;
    if (std.c.lseek(fd, @intCast(start_off), std.c.SEEK.SET) < 0) return error.UbcControlSeekFailed;
    var rem = span_bytes;
    var sink: u8 = 0;
    while (rem > 0) {
        const want: usize = @intCast(@min(rem, @as(u64, buf.len)));
        const got = crt.read(fd, buf[0..want]);
        if (got <= 0) return error.UbcControlReadFailed;
        sink ^= buf[0];
        rem -= @intCast(got);
    }
    std.mem.doNotOptimizeAway(sink);
}

fn warmTableShardPageCache(gpa: std.mem.Allocator, t: *const engram.Table) !void {
    if (comptime builtin.os.tag == .windows) return error.UnsupportedPlatform;
    if (comptime builtin.os.tag.isDarwin()) {
        if (std.c.fcntl(t.fd, std.c.F.NOCACHE, @as(c_int, 0)) == -1) return error.UbcControlFcntlFailed;
    }
    defer if (comptime builtin.os.tag.isDarwin()) {
        _ = std.c.fcntl(t.fd, std.c.F.NOCACHE, @as(c_int, 1));
    };
    const buf = try gpa.alloc(u8, 1 << 20);
    defer gpa.free(buf);
    if (t.slen > 0) {
        const s_bytes = @min(t.rows * t.s_stride, 512 << 20);
        try plainReadRegionIntoPageCache(t.fd, t.s_off, s_bytes, buf);
    }
    if (t.wlen > 0) {
        const w_bytes = @min(t.rows * t.w_stride, 64 << 20);
        try plainReadRegionIntoPageCache(t.fd, t.w_off, w_bytes, buf);
    }
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

pub fn mapCompressedIds(src: []const u32, token_map: []const u32, dst: []u32) !void {
    if (dst.len < src.len) return error.BufferTooSmall;
    for (src, dst[0..src.len]) |t, *o| {
        if (t >= token_map.len) return error.TokenIdOutOfVocab;
        o.* = token_map[t];
    }
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

fn hashTwoShardFds(fd0: c_int, fd1: c_int, same_path: bool) ![2][64]u8 {
    if (same_path) {
        const d0 = try sha256FdHex(fd0);
        return .{ d0, d0 };
    }
    const Worker = struct {
        fd: c_int,
        digest: [64]u8 = @splat('0'),
        err: ?anyerror = null,
        fn run(w: *@This()) void {
            w.digest = sha256FdHex(w.fd) catch |e| {
                w.err = e;
                return;
            };
        }
    };
    var w1: Worker = .{ .fd = fd1 };
    const th = std.Thread.spawn(.{}, Worker.run, .{&w1}) catch {
        const d0 = try sha256FdHex(fd0);
        const d1 = try sha256FdHex(fd1);
        return .{ d0, d1 };
    };
    const d0_res = sha256FdHex(fd0);
    th.join();
    const d0 = try d0_res;
    if (w1.err) |e| return e;
    return .{ d0, w1.digest };
}

pub fn runTableBench(
    io: std.Io,
    gpa: std.mem.Allocator,
    dir: []const u8,
    tokens_path: []const u8,
    reps: usize,
    warmup: usize,
    ubc_control: bool,
    json_path: ?[]const u8,
    json_out: *const pp_step1.JsonOut,
) !void {
    if (reps < MIN_TABLE_BENCH_REPS) {
        writeStderrFmt("[table-bench] invalid --reps={d}: must be >= {d}\n", .{ reps, MIN_TABLE_BENCH_REPS });
        return error.InvalidArgs;
    }
    var host_buf: [std.posix.HOST_NAME_MAX]u8 = undefined;
    const host = currentHostName(&host_buf);

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
    var shard_paths: [2][]u8 = .{ &.{}, &.{} };
    var n_open: usize = 0;
    defer {
        for (shard_paths[0..n_open]) |sp| if (sp.len > 0) gpa.free(sp);
        for (probes[0..n_open]) |*pr| pr.close();
        for (tables[0..n_open]) |*t| t.close();
    }
    var blocks: [2]u64 = .{ 0, 0 };
    var layouts: [2][]const u8 = .{ "", "" };
    var struct_cache_flags: [2]bool = .{ true, true };
    var bad_layout = false;

    for (0..2) |ti| {
        const slot = slots[ti];
        const layer = TABLE_BENCH_LAYERS[ti];
        tables[ti] = try engram.Table.open(gpa, abs_dir, slot, layer, cfg.dsv41_engram_rows[slot], cfg.dsv41_engram_head_dim);
        probes[ti] = UbcProbe.open(tables[ti].fd);
        n_open += 1;
        shard_paths[ti] = try fdPathAlloc(gpa, tables[ti].fd, abs_dir, layer);
        tables[ti].readDirect();
        blocks[ti] = edir.blockOf(tables[ti].fd);
        layouts[ti] = describeTableLayout(&tables[ti]);
        struct_cache_flags[ti] = tableStructuralCachePossible(&tables[ti], blocks[ti]);
        writeStderrFmt(
            "[table-bench] layer={d} fd={d} layout={s} mode={s} bits={d} group_size={d} records={} blockOf={d} direct={} ubc_mincore_ready={} structural_cache_possible={}\n",
            .{ layer, tables[ti].fd, layouts[ti], @tagName(tables[ti].mode), tables[ti].bits, tables[ti].group_size, tables[ti].records(), blocks[ti], tables[ti].direct, probes[ti].map != null, struct_cache_flags[ti] },
        );
        if (!isSpecIqTableLayout(&tables[ti])) {
            writeStderrFmt(
                "[table-bench] refusing layer={d} layout={s} (mode={s}, bits={d}, group_size={d}, records={}): spec requires layout=iq_mxfp8_split\n",
                .{ layer, layouts[ti], @tagName(tables[ti].mode), tables[ti].bits, tables[ti].group_size, tables[ti].records() },
            );
            bad_layout = true;
        }
    }
    if (bad_layout) {
        writeFdAll(1, "TABLE-BENCH-FAIL\n");
        return error.UnsupportedEngramTableLayout;
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
            writeFdAll(1, "TABLE-BENCH-FAIL\n");
            return error.TokenFileTooShort;
        }
        const comp = try gpa.alloc(u32, needed_tokens);
        errdefer gpa.free(comp);
        mapCompressedIds(real_toks[0..needed_tokens], token_map, comp) catch |err| {
            writeStderrFmt("[table-bench] tokens-file invalid ({s}): token id >= vocab_size ({d})\n", .{ @errorName(err), token_map.len });
            writeFdAll(1, "TABLE-BENCH-FAIL\n");
            return err;
        };
        break :blk comp;
    };
    defer gpa.free(real_comp);

    const det_comp = blk: {
        const det_toks = try gpa.alloc(u32, needed_tokens);
        defer gpa.free(det_toks);
        fillDisjointDeterministicStream(det_toks, cfg.vocab_size, TABLE_BENCH_P0);
        const comp = try gpa.alloc(u32, needed_tokens);
        errdefer gpa.free(comp);
        try mapCompressedIds(det_toks, token_map, comp);
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

    var read_samples_ms: [2][]f64 = .{ try gpa.alloc(f64, reps), try gpa.alloc(f64, reps) };
    defer for (read_samples_ms) |s| gpa.free(s);
    var sort_samples_ms: [2][]f64 = .{ try gpa.alloc(f64, reps), try gpa.alloc(f64, reps) };
    defer for (sort_samples_ms) |s| gpa.free(s);
    var hs_ms: [2][]f64 = .{ try gpa.alloc(f64, reps), try gpa.alloc(f64, reps) };
    defer for (hs_ms) |s| gpa.free(s);
    var nu_samples: [2][]usize = .{ try gpa.alloc(usize, reps), try gpa.alloc(usize, reps) };
    defer for (nu_samples) |s| gpa.free(s);
    var ubc_pre_samples: [2][]usize = .{ try gpa.alloc(usize, reps), try gpa.alloc(usize, reps) };
    defer for (ubc_pre_samples) |s| gpa.free(s);
    var ubc_pop_samples: [2][]usize = .{ try gpa.alloc(usize, reps), try gpa.alloc(usize, reps) };
    defer for (ubc_pop_samples) |s| gpa.free(s);
    const io_scratch = try gpa.alloc(f64, reps);
    defer gpa.free(io_scratch);
    const scratch_f64 = try gpa.alloc(f64, reps);
    defer gpa.free(scratch_f64);
    const scratch_usize = try gpa.alloc(usize, reps);
    defer gpa.free(scratch_usize);

    var cells: [2 * TABLE_BENCH_WIDTHS.len * TABLE_BENCH_LAYERS.len]TableBenchCell = undefined;
    var cell_idx: usize = 0;

    for ([_]TableBenchSource{ .real_text, .deterministic }) |source| {
        const comp = if (source == .real_text) real_comp else det_comp;
        var cursor: usize = TABLE_BENCH_P0;
        for (TABLE_BENCH_WIDTHS) |c_rows| {
            if (ubc_control and source == .real_text) {
                for (0..2) |ti| try warmTableShardPageCache(gpa, &tables[ti]);
            }
            const chunk_lo = cursor;
            var ubc_ok: [2]bool = .{ true, true };
            for (0..reps) |rep| {
                const pos = cursor;
                cursor += c_rows;
                const cur_grid = grid_buf[0 .. c_rows * nt * nc];
                const t_h0 = monoUs();
                hash.rows(comp[0..pos], comp[pos .. pos + c_rows], cur_grid);
                const h_us = @max(monoUs() - t_h0, 0);
                for (0..2) |ti| {
                    const t_s0 = monoUs();
                    const uniq = try extractSortDedupLayerRows(cur_grid, c_rows, nt, slots[ti], nc, uniq_buf);
                    const s_us = @max(monoUs() - t_s0, 0);
                    sort_samples_ms[ti][rep] = @as(f64, @floatFromInt(s_us)) / 1000.0;
                    hs_ms[ti][rep] = @as(f64, @floatFromInt(h_us + s_us)) / 1000.0;
                    nu_samples[ti][rep] = uniq.len;

                    const pre_ubc = probes[ti].countCachedRows(&tables[ti], uniq);
                    if (pre_ubc == null) ubc_ok[ti] = false;
                    const pre_n = pre_ubc orelse uniq.len;
                    ubc_pre_samples[ti][rep] = pre_n;

                    const t_r0 = monoUs();
                    const raw = try tables[ti].readSorted(gpa, uniq, engram.GRID_THREADS);
                    const r_us = @max(monoUs() - t_r0, 0);
                    raw.deinit(gpa);
                    read_samples_ms[ti][rep] = @as(f64, @floatFromInt(r_us)) / 1000.0;

                    const post_ubc = probes[ti].countCachedRows(&tables[ti], uniq) orelse pre_n;
                    ubc_pop_samples[ti][rep] = post_ubc -| pre_n;
                }
            }
            const chunk_hi = cursor;
            for (0..2) |ti| {
                const layer = TABLE_BENCH_LAYERS[ti];
                const lookup_rows = c_rows * nc;
                const med_nu = try medianUsize(nu_samples[ti], scratch_usize);
                const med_ubc = try medianUsize(ubc_pre_samples[ti], scratch_usize);
                const med_pop = try medianUsize(ubc_pop_samples[ti], scratch_usize);
                var max_ubc: usize = 0;
                for (ubc_pre_samples[ti]) |v| max_ubc = @max(max_ubc, v);
                var sum_nu: u64 = 0;
                for (nu_samples[ti]) |v| sum_nu += v;

                const ev = try evaluateTableBenchCell(
                    source,
                    c_rows,
                    layer,
                    layouts[ti],
                    tables[ti].records(),
                    blocks[ti],
                    struct_cache_flags[ti],
                    ubc_ok[ti],
                    ubc_control,
                    max_ubc,
                    lookup_rows,
                    med_nu,
                    read_samples_ms[ti],
                    sort_samples_ms[ti],
                    io_scratch,
                    scratch_f64,
                );
                if (ev.gate_reason) |reason| {
                    writeStderrFmt("[table-bench] layer={d} chunk_rows={d} source={s} gate fail: {s}\n", .{ layer, c_rows, source.name(), reason });
                }
                const hs_st = try sampleMsStats(hs_ms[ti], scratch_f64);
                const distinct_all = try countCrossRepDistinct(gpa, &hash, comp, chunk_lo, c_rows, reps, slots[ti], grid_buf, uniq_buf);
                const pc_poss = ubc_control or struct_cache_flags[ti] or !ubc_ok[ti] or max_ubc > 0;
                cells[cell_idx] = .{
                    .host = host,
                    .pack_path = abs_dir,
                    .shard_path = shard_paths[ti],
                    .l1_shard_path = shard_paths[0],
                    .l14_shard_path = shard_paths[1],
                    .source = source,
                    .chunk_rows = c_rows,
                    .layer = layer,
                    .p0_start = TABLE_BENCH_P0,
                    .chunk_tok_lo = chunk_lo,
                    .chunk_tok_hi = chunk_hi,
                    .reps = reps,
                    .table_layout = layouts[ti],
                    .block_bytes = blocks[ti],
                    .records = tables[ti].records(),
                    .ubc_control = ubc_control,
                    .structural_cache_possible = struct_cache_flags[ti],
                    .ubc_verified = ubc_ok[ti],
                    .ubc_cached_rows = med_ubc,
                    .ubc_cached_rows_max = max_ubc,
                    .ubc_populated_rows = med_pop,
                    .page_cache_possible = pc_poss,
                    .lookup_rows = lookup_rows,
                    .rows_read = med_nu,
                    .distinct_rows = med_nu,
                    .total_rows_read = sum_nu,
                    .distinct_rows_across_reps = distinct_all,
                    .median_ms = ev.io_stats.median_ms,
                    .p90_ms = ev.io_stats.p90_ms,
                    .cold_adjusted_median_ms = ev.cold_adjusted_ms,
                    .read_sorted_median_ms = ev.read_stats.median_ms,
                    .read_sorted_p90_ms = ev.read_stats.p90_ms,
                    .extract_sort_median_ms = ev.sort_stats.median_ms,
                    .hash_sort_median_ms = hs_st.median_ms,
                    .step1_pread_ms = ev.spec.step1_pread_ms,
                    .threshold_ms = ev.spec.threshold_ms,
                    .gate_pass = ev.gate_pass,
                    .gate_reason = ev.gate_reason,
                };
                cell_idx += 1;
            }
        }
    }

    const shard_digests = try hashTwoShardFds(tables[0].fd, tables[1].fd, std.mem.eql(u8, shard_paths[0], shard_paths[1]));
    var line_buf: [8192]u8 = undefined;
    const has_separate_json_file = json_path != null and json_out.fd != 1;
    var ubc_ctrl_missed = false;
    for (cells[0..cell_idx]) |*cell| {
        const ti: usize = if (cell.layer == TABLE_BENCH_LAYERS[0]) 0 else 1;
        cell.shard_sha256 = shard_digests[ti];
        cell.l1_shard_sha256 = shard_digests[0];
        cell.l14_shard_sha256 = shard_digests[1];
        if (ubc_control and cell.source.isGated() and (!cell.ubc_verified or cell.ubc_cached_rows_max == 0)) {
            writeStderrFmt("[table-bench] ubc-control error: layer={d} chunk_rows={d} reported ubc_cached_rows_max={d} (ubc_verified={})\n", .{ cell.layer, cell.chunk_rows, cell.ubc_cached_rows_max, cell.ubc_verified });
            ubc_ctrl_missed = true;
        }
        const line = try formatTableBenchJson(&line_buf, cell.*);
        writeFdAll(1, line);
        if (has_separate_json_file) json_out.writeLine(line);
    }

    if (ubc_ctrl_missed) {
        writeFdAll(1, "TABLE-BENCH-FAIL\n");
        return error.UbcControlMissedCachedPages;
    }
    if (!ubc_control and evaluateTableBenchGate(cells[0..cell_idx])) {
        writeFdAll(1, "TABLE-BENCH-PASS\n");
        return;
    }
    writeFdAll(1, "TABLE-BENCH-FAIL\n");
    return error.TableBenchGateFailed;
}

// ── Tests ──

const testing = std.testing;

test "pp_step1_table: mapCompressedIds rejects out-of-vocab ids and sort/dedup extracts sorted distinct rows" {
    const token_map = [_]u32{ 10, 20, 30, 40 };
    var mapped: [3]u32 = undefined;
    try mapCompressedIds(&.{ 3, 0, 2 }, &token_map, &mapped);
    try testing.expectEqualSlices(u32, &.{ 40, 10, 30 }, &mapped);
    try testing.expectError(error.TokenIdOutOfVocab, mapCompressedIds(&.{ 1, 4, 0 }, &token_map, &mapped));
    try testing.expectError(error.TokenIdOutOfVocab, mapCompressedIds(&.{ 1, 129280, 0 }, &token_map, &mapped));
    try testing.expectError(error.BufferTooSmall, mapCompressedIds(&.{ 0, 1, 2 }, &token_map, mapped[0..2]));

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

test "pp_step1_table: gate on read+sort, distinct_ratio >= 0.30 floor, ungated n/a JSON, layout check, and provenance fields" {
    const iq_tbl: engram.Table = .{
        .fd = -1, .rows = 100, .head_dim = 256, .mode = .mxfp8, .bits = 8, .group_size = 32,
        .w_off = 0, .w_stride = 256, .wlen = 256, .s_off = 25600, .s_stride = 8, .slen = 8,
    };
    const rec_tbl: engram.Table = .{
        .fd = -1, .rows = 100, .head_dim = 256, .mode = .mxfp8, .bits = 8, .group_size = 32,
        .w_off = 0, .w_stride = 264, .wlen = 256, .s_off = 256, .s_stride = 264, .slen = 8,
    };
    const aff_tbl: engram.Table = .{
        .fd = -1, .rows = 100, .head_dim = 256, .mode = .affine, .bits = 4, .group_size = 32,
        .w_off = 0, .w_stride = 128, .wlen = 128, .s_off = 12800, .s_stride = 16, .slen = 16, .b_off = 14400,
    };
    const gguf_tbl: engram.Table = .{
        .fd = -1, .rows = 100, .head_dim = 256, .mode = .gguf, .bits = 4, .group_size = 32,
        .w_off = 0, .w_stride = 144, .wlen = 144, .s_off = 0, .s_stride = 0, .slen = 0,
    };
    try testing.expect(isSpecIqTableLayout(&iq_tbl));
    try testing.expectEqualStrings("iq_mxfp8_split", describeTableLayout(&iq_tbl));
    try testing.expect(!isSpecIqTableLayout(&rec_tbl));
    try testing.expectEqualStrings("mxfp8_records", describeTableLayout(&rec_tbl));
    try testing.expect(!isSpecIqTableLayout(&aff_tbl));
    try testing.expectEqualStrings("affine_split", describeTableLayout(&aff_tbl));
    try testing.expect(!isSpecIqTableLayout(&gguf_tbl));
    try testing.expectEqualStrings("gguf", describeTableLayout(&gguf_tbl));

    try testing.expect(hasRepresentativeDistinctRatio(300, 1000));
    try testing.expect(!hasRepresentativeDistinctRatio(299, 1000));
    try testing.expect(!hasRepresentativeDistinctRatio(0, 1000));
    try testing.expect(!hasRepresentativeDistinctRatio(10, 0));

    var io_scratch: [10]f64 = undefined;
    var scratch: [10]f64 = undefined;
    const s5 = try sampleMsStats(&.{ 140.0, 110.0, 180.0, 120.0, 130.0 }, &scratch);
    try testing.expectApproxEqAbs(@as(f64, 130.0), s5.median_ms, 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 180.0), s5.p90_ms, 1e-9);

    var sha_l1: [64]u8 = @splat('a');
    sha_l1[0] = '1';
    var sha_l14: [64]u8 = @splat('b');
    sha_l14[0] = 'e';

    var cells: [12]TableBenchCell = undefined;
    var ci: usize = 0;
    for (TABLE_BENCH_SPECS) |spec| {
        const lookup = spec.chunk_rows * 24;
        const distinct = spec.chunk_rows * 10;
        const sort_arr = [_]f64{ 1.5, 1.5, 1.5 };
        const read_arr = [_]f64{ spec.threshold_ms - 6.5, spec.threshold_ms - 1.5, spec.threshold_ms + 2.5 };
        const real_ev = try evaluateTableBenchCell(
            .real_text,
            spec.chunk_rows,
            spec.layer,
            "iq_mxfp8_split",
            false,
            4096,
            false,
            true,
            false,
            0,
            lookup,
            distinct,
            &read_arr,
            &sort_arr,
            &io_scratch,
            &scratch,
        );
        try testing.expectEqual(@as(?bool, true), real_ev.gate_pass);
        try testing.expectEqualStrings("PASS", real_ev.gate_verdict);
        try testing.expect(real_ev.gate_reason == null);
        try testing.expectApproxEqAbs(spec.threshold_ms, real_ev.io_stats.median_ms, 1e-9);
        try testing.expectApproxEqAbs(spec.threshold_ms - 1.5, real_ev.read_stats.median_ms, 1e-9);

        const shard_sha = if (spec.layer == 1) sha_l1 else sha_l14;
        const shard_p = if (spec.layer == 1) "/pack/engram/l1.safetensors" else "/pack/engram/l14.safetensors";
        cells[ci] = .{
            .host = "studio2",
            .pack_path = "/pack",
            .shard_path = shard_p,
            .shard_sha256 = shard_sha,
            .l1_shard_path = "/pack/engram/l1.safetensors",
            .l1_shard_sha256 = sha_l1,
            .l14_shard_path = "/pack/engram/l14.safetensors",
            .l14_shard_sha256 = sha_l14,
            .source = .real_text,
            .chunk_rows = spec.chunk_rows,
            .layer = spec.layer,
            .p0_start = TABLE_BENCH_P0,
            .chunk_tok_lo = TABLE_BENCH_P0,
            .chunk_tok_hi = TABLE_BENCH_P0 + 3 * spec.chunk_rows,
            .reps = 3,
            .table_layout = "iq_mxfp8_split",
            .block_bytes = 4096,
            .records = false,
            .ubc_control = false,
            .structural_cache_possible = false,
            .ubc_verified = true,
            .ubc_cached_rows = 0,
            .ubc_cached_rows_max = 0,
            .ubc_populated_rows = 0,
            .page_cache_possible = false,
            .lookup_rows = lookup,
            .rows_read = distinct,
            .distinct_rows = distinct,
            .total_rows_read = distinct * 3,
            .distinct_rows_across_reps = distinct * 2,
            .median_ms = real_ev.io_stats.median_ms,
            .p90_ms = real_ev.io_stats.p90_ms,
            .cold_adjusted_median_ms = real_ev.cold_adjusted_ms,
            .read_sorted_median_ms = real_ev.read_stats.median_ms,
            .read_sorted_p90_ms = real_ev.read_stats.p90_ms,
            .extract_sort_median_ms = real_ev.sort_stats.median_ms,
            .hash_sort_median_ms = 2.4,
            .step1_pread_ms = spec.step1_pread_ms,
            .threshold_ms = spec.threshold_ms,
            .gate_pass = real_ev.gate_pass,
            .gate_reason = real_ev.gate_reason,
        };
        ci += 1;

        const det_read = [_]f64{ spec.threshold_ms + 25.0, spec.threshold_ms + 30.0, spec.threshold_ms + 40.0 };
        const det_ev = try evaluateTableBenchCell(
            .deterministic,
            spec.chunk_rows,
            spec.layer,
            "iq_mxfp8_split",
            false,
            4096,
            false,
            true,
            false,
            0,
            lookup,
            lookup,
            &det_read,
            &sort_arr,
            &io_scratch,
            &scratch,
        );
        try testing.expect(det_ev.gate_pass == null);
        try testing.expectEqualStrings("n/a", det_ev.gate_verdict);
        cells[ci] = cells[ci - 1];
        cells[ci].source = .deterministic;
        cells[ci].distinct_rows = lookup;
        cells[ci].rows_read = lookup;
        cells[ci].median_ms = det_ev.io_stats.median_ms;
        cells[ci].p90_ms = det_ev.io_stats.p90_ms;
        cells[ci].cold_adjusted_median_ms = det_ev.cold_adjusted_ms;
        cells[ci].read_sorted_median_ms = det_ev.read_stats.median_ms;
        cells[ci].read_sorted_p90_ms = det_ev.read_stats.p90_ms;
        cells[ci].gate_pass = det_ev.gate_pass;
        cells[ci].gate_reason = det_ev.gate_reason;
        ci += 1;
    }

    try testing.expect(evaluateTableBenchGate(&cells));
    try testing.expect(!evaluateTableBenchGate(cells[0 .. cells.len - 2]));

    const read_only_fast = try evaluateTableBenchCell(
        .real_text,
        2048,
        1,
        "iq_mxfp8_split",
        false,
        4096,
        false,
        true,
        false,
        0,
        49152,
        20000,
        &.{ 171.0, 171.5, 172.0 },
        &.{ 2.5, 2.5, 2.5 },
        &io_scratch,
        &scratch,
    );
    try testing.expectEqual(@as(?bool, false), read_only_fast.gate_pass);
    try testing.expectEqualStrings("threshold_exceeded", read_only_fast.gate_reason.?);
    try testing.expectApproxEqAbs(@as(f64, 171.5), read_only_fast.read_stats.median_ms, 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 174.0), read_only_fast.io_stats.median_ms, 1e-9);

    const low_distinct = try evaluateTableBenchCell(
        .real_text,
        2048,
        1,
        "iq_mxfp8_split",
        false,
        4096,
        false,
        true,
        false,
        0,
        49152,
        14000,
        &.{ 48.0, 50.0, 52.0 },
        &.{ 1.0, 1.0, 1.0 },
        &io_scratch,
        &scratch,
    );
    try testing.expectEqual(@as(?bool, false), low_distinct.gate_pass);
    try testing.expectEqualStrings("unrepresentative ids", low_distinct.gate_reason.?);

    const borderline = try evaluateTableBenchCell(
        .real_text,
        2048,
        1,
        "iq_mxfp8_split",
        false,
        4096,
        false,
        true,
        false,
        800,
        49152,
        20000,
        &.{ 166.0, 167.0, 168.0 },
        &.{ 1.0, 1.0, 1.0 },
        &io_scratch,
        &scratch,
    );
    try testing.expectEqual(@as(?bool, false), borderline.gate_pass);
    try testing.expectApproxEqAbs(@as(f64, 175.0), borderline.cold_adjusted_ms, 1e-6);

    var line_buf: [8192]u8 = undefined;
    const real_line = try formatTableBenchJson(&line_buf, cells[0]);
    var parsed_real = try std.json.parseFromSlice(std.json.Value, testing.allocator, std.mem.trim(u8, real_line, "\n"), .{});
    defer parsed_real.deinit();
    try testing.expectEqualStrings("table_bench", parsed_real.value.object.get("kind").?.string);
    try testing.expectEqualStrings("studio2", parsed_real.value.object.get("host").?.string);
    try testing.expectEqualStrings("/pack", parsed_real.value.object.get("pack_path").?.string);
    try testing.expectEqualStrings(&sha_l1, parsed_real.value.object.get("l1_shard_sha256").?.string);
    try testing.expectEqualStrings(&sha_l14, parsed_real.value.object.get("l14_shard_sha256").?.string);
    try testing.expectEqual(true, parsed_real.value.object.get("gate_pass").?.bool);
    try testing.expectEqualStrings("PASS", parsed_real.value.object.get("gate_verdict").?.string);
    try testing.expectApproxEqAbs(@as(f64, 172.8), parsed_real.value.object.get("median_ms").?.float, 1e-6);
    try testing.expectApproxEqAbs(@as(f64, 171.3), parsed_real.value.object.get("read_sorted_median_ms").?.float, 1e-6);

    const det_line = try formatTableBenchJson(&line_buf, cells[1]);
    try testing.expect(std.mem.indexOf(u8, det_line, "PASS") == null);
    try testing.expect(std.mem.indexOf(u8, det_line, "FAIL") == null);
    try testing.expect(std.mem.indexOf(u8, det_line, "\"gate_pass\"") == null);
    var parsed_det = try std.json.parseFromSlice(std.json.Value, testing.allocator, std.mem.trim(u8, det_line, "\n"), .{});
    defer parsed_det.deinit();
    try testing.expectEqual(false, parsed_det.value.object.get("gated").?.bool);
    try testing.expect(parsed_det.value.object.get("gate_pass") == null);
    try testing.expectEqualStrings("n/a", parsed_det.value.object.get("gate_verdict").?.string);

    var bad_cell = cells[0];
    bad_cell.distinct_rows = 14000;
    bad_cell.rows_read = 14000;
    bad_cell.gate_pass = false;
    bad_cell.gate_reason = "unrepresentative ids";
    const bad_line = try formatTableBenchJson(&line_buf, bad_cell);
    var parsed_bad = try std.json.parseFromSlice(std.json.Value, testing.allocator, std.mem.trim(u8, bad_line, "\n"), .{});
    defer parsed_bad.deinit();
    try testing.expectEqual(false, parsed_bad.value.object.get("gate_pass").?.bool);
    try testing.expectEqualStrings("FAIL", parsed_bad.value.object.get("gate_verdict").?.string);
    try testing.expectEqualStrings("unrepresentative ids", parsed_bad.value.object.get("gate_reason").?.string);
}

fn writeSparseSplitShard(dir: std.Io.Dir, io: std.Io, arena: std.mem.Allocator, sub_path: []const u8, layer: usize, rows: u64, head_dim: u32, tag_byte: u8) ![64]u8 {
    var hdr_json_buf: [512]u8 = undefined;
    const w_bytes = rows * head_dim;
    const s_cols = head_dim / 32;
    const s_bytes = rows * s_cols;
    const hdr_body = try std.fmt.bufPrint(
        &hdr_json_buf,
        "{{\"layers.{d}.engram.embed.weight\":{{\"dtype\":\"F8_E4M3\",\"shape\":[{d},{d}],\"data_offsets\":[0,{d}]}},\"layers.{d}.engram.embed.scale\":{{\"dtype\":\"F8_E8M0\",\"shape\":[{d},{d}],\"data_offsets\":[{d},{d}]}},\"__tag\":\"{x:0>2}\"}}",
        .{ layer, rows, head_dim, w_bytes, layer, rows, s_cols, w_bytes, w_bytes + s_bytes, tag_byte },
    );
    const base_off: usize = 16384;
    const hlen: u64 = base_off - 8;
    const hdr_page = try arena.alloc(u8, base_off);
    @memset(hdr_page, ' ');
    std.mem.writeInt(u64, hdr_page[0..8], hlen, .little);
    @memcpy(hdr_page[8 .. 8 + hdr_body.len], hdr_body);
    try dir.writeFile(io, .{ .sub_path = sub_path, .data = hdr_page });

    var root_buf: [std.fs.max_path_bytes]u8 = undefined;
    const root = root_buf[0..try dir.realPath(io, &root_buf)];
    const full_z = try std.fmt.allocPrintSentinel(arena, "{s}/{s}", .{ root, sub_path }, 0);
    const fd = std.c.open(full_z, .{ .ACCMODE = .RDWR, .CLOEXEC = true }, @as(std.c.mode_t, 0));
    if (fd < 0) return error.OpenShardFailed;
    defer _ = std.c.close(fd);
    const payload_bytes = w_bytes + s_bytes;
    const total_size: i64 = @intCast(@as(u64, base_off) + payload_bytes);
    if (std.c.ftruncate(fd, total_size) != 0) return error.TruncateShardFailed;

    var h = std.crypto.hash.sha2.Sha256.init(.{});
    h.update(hdr_page);
    var zero_blk: [4096]u8 = @splat(0);
    var rem = payload_bytes;
    while (rem > 0) {
        const step: usize = @intCast(@min(rem, zero_blk.len));
        h.update(zero_blk[0..step]);
        rem -= step;
    }
    var digest: [32]u8 = undefined;
    h.final(&digest);
    return std.fmt.bytesToHex(digest, .lower);
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

test "pp_step1_table: end-to-end reps>=3, records refusal, OOV refusal, cold pass vs --ubc-control fail, and shard sha256" {
    if (comptime builtin.os.tag == .windows) return error.SkipZigTest;
    const gpa = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();

    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    var td = testing.tmpDir(.{});
    defer td.cleanup();
    var root_buf: [std.fs.max_path_bytes]u8 = undefined;
    const root = root_buf[0..try td.dir.realPath(io, &root_buf)];
    const test_vocab: usize = 16381;
    const tmap_bytes = try arena.alloc(u8, test_vocab * 4);
    for (0..test_vocab) |idx| std.mem.writeInt(u32, tmap_bytes[idx * 4 ..][0..4], @intCast(idx), .little);
    try td.dir.writeFile(io, .{ .sub_path = "engram-token-map.u32", .data = tmap_bytes });
    try td.dir.createDirPath(io, "engram");

    const l1_rows: u64 = 32771;
    const l14_rows: u64 = 32779;
    const base_cfg =
        \\{"architectures":["DeepseekV41ForCausalLM"],"model_type":"deepseek_v41","bos_token_id":0,"eos_token_id":1,
        \\ "quantization":{"group_size":64,"bits":8},
        \\ "text_config":{"model_type":"deepseek_v41_text","vocab_size":16381,"hidden_size":128,
        \\  "moe_intermediate_size":128,"num_hidden_layers":16,"num_attention_heads":4,"num_key_value_heads":1,
        \\  "head_dim":64,"qk_rope_head_dim":16,"q_lora_rank":64,"o_lora_rank":32,"o_groups":2,"swiglu_limit":1.0,
        \\  "rms_norm_eps":1e-20,"max_position_embeddings":65536,"rope_theta":10000.0,
        \\  "n_routed_experts":8,"n_shared_experts":1,"num_experts_per_tok":3,"scoring_func":"sqrtsoftplus",
        \\  "topk_method":"noaux_tc","norm_topk_prob":true,"routed_scaling_factor":1.5,"sliding_window":8,
        \\  "compress_ratios":[0,0,2,2,2,2,2,2,1,1,1,1,1,1,1,1],"compress_rope_theta":40000.0,
        \\  "kv_source_layer_ids":[2,8],"index_source_layer_ids":[2,8,12],"index_n_heads":8,"index_head_dim":32,"index_topk":4,
        \\  "candidate_source_layer_id":8,"candidate_topk_blocks":3,"candidate_block_size":2,"hc_mult":4,
        \\  "hc_sinkhorn_iters":20,"hc_eps":1e-6,
        \\  "engram_layer_ids":[1,14],"engram_num_embeddings":[32771,32779],"engram_max_ngram_size":2,
        \\  "engram_vocab_size":32768,"engram_n_heads":1,"engram_head_dim":32,"engram_pad_token_id":2,
        \\  "engram_compressed_vocab_size":16381},
        \\ "omlx_deepseek_v41":{"version":1,"engram_tables":{
        \\  "language_model.layers.1.engram.embed":{"weight_key":"layers.1.engram.embed.weight","weight_file":"engram/l1.safetensors","scale_key":"layers.1.engram.embed.scale","scale_file":"engram/l1.safetensors","bits":null},
        \\  "language_model.layers.14.engram.embed":{"weight_key":"layers.14.engram.embed.weight","weight_file":"engram/l14.safetensors","scale_key":"layers.14.engram.embed.scale","scale_file":"engram/l14.safetensors","bits":null}
        \\ }}}
    ;
    try td.dir.writeFile(io, .{ .sub_path = "config.json", .data = base_cfg });
    const want_sha_l1 = try writeSparseSplitShard(td.dir, io, arena, "engram/l1.safetensors", 1, l1_rows, 32, 0x11);
    const want_sha_l14 = try writeSparseSplitShard(td.dir, io, arena, "engram/l14.safetensors", 14, l14_rows, 32, 0xee);

    const needed_toks: usize = TABLE_BENCH_P0 + 3 * (2048 + 4096 + 8192);
    const valid_raw = try arena.alloc(u8, needed_toks * 4);
    var prng_state: u64 = 0x9E3779B97F4A7C15;
    for (0..needed_toks) |idx| {
        prng_state +%= 0x9E3779B97F4A7C15;
        var z = prng_state;
        z = (z ^ (z >> 30)) *% 0xBF58476D1CE4E5B9;
        z = (z ^ (z >> 27)) *% 0x94D049BB133111EB;
        z = z ^ (z >> 31);
        const tid: u32 = @intCast(z % test_vocab);
        std.mem.writeInt(u32, valid_raw[idx * 4 ..][0..4], tid, .little);
    }
    const tok_file = try std.fmt.allocPrint(arena, "{s}/bench_tokens.u32", .{root});
    try td.dir.writeFile(io, .{ .sub_path = "bench_tokens.u32", .data = valid_raw });

    try testing.expectError(error.InvalidArgs, pp_step1.cmdProbe(io, gpa, &.{ "table-bench", "--model", root, "--tokens-file", tok_file, "--reps", "2" }));
    try testing.expectError(error.InvalidArgs, pp_step1.cmdProbe(io, gpa, &.{ "table-bench", "--model", root, "--tokens-file", tok_file, "--reps", "0" }));

    {
        var rec_td = testing.tmpDir(.{});
        defer rec_td.cleanup();
        try rec_td.dir.writeFile(io, .{ .sub_path = "config.json", .data = base_cfg });
        try rec_td.dir.writeFile(io, .{ .sub_path = "engram-token-map.u32", .data = tmap_bytes });
        try rec_td.dir.createDirPath(io, "engram");
        try rec_td.dir.writeFile(io, .{
            .sub_path = "engram/engram-manifest.json",
            .data = "{\"layers\":[{\"layer_id\":1,\"file\":\"r1.bin\",\"rows\":32771,\"record_bytes\":33,\"quant\":{\"bits\":8,\"group_size\":32,\"mode\":\"mxfp8\",\"head_dim\":32}},{\"layer_id\":14,\"file\":\"r14.bin\",\"rows\":32779,\"record_bytes\":33,\"quant\":{\"bits\":8,\"group_size\":32,\"mode\":\"mxfp8\",\"head_dim\":32}}]}",
        });
        var rec_root_buf: [std.fs.max_path_bytes]u8 = undefined;
        const rec_root = rec_root_buf[0..try rec_td.dir.realPath(io, &rec_root_buf)];
        for ([_]struct { name: []const u8, rows: u64 }{ .{ .name = "r1.bin", .rows = l1_rows }, .{ .name = "r14.bin", .rows = l14_rows } }) |entry| {
            const p_z = try std.fmt.allocPrintSentinel(arena, "{s}/engram/{s}", .{ rec_root, entry.name }, 0);
            const rfd = std.c.open(p_z, .{ .ACCMODE = .RDWR, .CREAT = true, .TRUNC = true, .CLOEXEC = true }, @as(std.c.mode_t, 0o600));
            try testing.expect(rfd >= 0);
            defer _ = std.c.close(rfd);
            try testing.expectEqual(@as(c_int, 0), std.c.ftruncate(rfd, @intCast(entry.rows * 33)));
        }
        try testing.expectError(
            error.UnsupportedEngramTableLayout,
            pp_step1.cmdProbe(io, gpa, &.{ "table-bench", "--model", rec_root, "--tokens-file", tok_file, "--reps", "3" }),
        );
    }

    if (comptime builtin.os.tag.isDarwin()) {
        const oov_raw = try arena.dupe(u8, valid_raw);
        std.mem.writeInt(u32, oov_raw[(needed_toks - 5) * 4 ..][0..4], @intCast(test_vocab), .little);
        try td.dir.writeFile(io, .{ .sub_path = "oov_tokens.u32", .data = oov_raw });
        const oov_file = try std.fmt.allocPrint(arena, "{s}/oov_tokens.u32", .{root});
        try testing.expectError(
            error.TokenIdOutOfVocab,
            pp_step1.cmdProbe(io, gpa, &.{ "table-bench", "--model", root, "--tokens-file", oov_file, "--reps", "3" }),
        );

        const cold_jsonl = try std.fmt.allocPrint(arena, "{s}/cold.jsonl", .{root});
        try pp_step1.cmdProbe(io, gpa, &.{ "table-bench", "--model", root, "--tokens-file", tok_file, "--reps", "3", "--warmup", "0", "--json-out", cold_jsonl });
        const cold_text = try td.dir.readFileAlloc(io, "cold.jsonl", gpa, .limited(1 << 20));
        defer gpa.free(cold_text);
        var cold_gated: usize = 0;
        var cold_ungated: usize = 0;
        var cold_it = std.mem.splitScalar(u8, cold_text, '\n');
        while (cold_it.next()) |line| {
            const trimmed = std.mem.trim(u8, line, " \r\t");
            if (trimmed.len == 0) continue;
            var parsed = try std.json.parseFromSlice(std.json.Value, gpa, trimmed, .{});
            defer parsed.deinit();
            const obj = parsed.value.object;
            try testing.expect(obj.get("host").?.string.len > 0);
            try testing.expectEqualStrings(root, obj.get("pack_path").?.string);
            try testing.expectEqualStrings(&want_sha_l1, obj.get("l1_shard_sha256").?.string);
            try testing.expectEqualStrings(&want_sha_l14, obj.get("l14_shard_sha256").?.string);
            try testing.expectEqual(@as(i64, 0), obj.get("ubc_cached_rows_max").?.integer);
            if (obj.get("gated").?.bool) {
                cold_gated += 1;
                try testing.expectEqual(true, obj.get("gate_pass").?.bool);
                try testing.expectEqualStrings("PASS", obj.get("gate_verdict").?.string);
            } else {
                cold_ungated += 1;
                try testing.expect(obj.get("gate_pass") == null);
                try testing.expectEqualStrings("n/a", obj.get("gate_verdict").?.string);
            }
        }
        try testing.expectEqual(@as(usize, 6), cold_gated);
        try testing.expectEqual(@as(usize, 6), cold_ungated);

        const warm_jsonl = try std.fmt.allocPrint(arena, "{s}/warm.jsonl", .{root});
        try testing.expectError(
            error.TableBenchGateFailed,
            pp_step1.cmdProbe(io, gpa, &.{ "table-bench", "--model", root, "--tokens-file", tok_file, "--reps", "3", "--warmup", "0", "--ubc-control", "--json-out", warm_jsonl }),
        );
        const warm_text = try td.dir.readFileAlloc(io, "warm.jsonl", gpa, .limited(1 << 20));
        defer gpa.free(warm_text);
        var warm_gated: usize = 0;
        var warm_it = std.mem.splitScalar(u8, warm_text, '\n');
        while (warm_it.next()) |line| {
            const trimmed = std.mem.trim(u8, line, " \r\t");
            if (trimmed.len == 0) continue;
            var parsed = try std.json.parseFromSlice(std.json.Value, gpa, trimmed, .{});
            defer parsed.deinit();
            const obj = parsed.value.object;
            try testing.expectEqual(true, obj.get("ubc_control").?.bool);
            if (obj.get("gated").?.bool) {
                warm_gated += 1;
                try testing.expect(obj.get("ubc_cached_rows_max").?.integer > 0);
                try testing.expectEqual(false, obj.get("gate_pass").?.bool);
                try testing.expectEqualStrings("FAIL", obj.get("gate_verdict").?.string);
            } else {
                try testing.expect(obj.get("gate_pass") == null);
                try testing.expectEqualStrings("n/a", obj.get("gate_verdict").?.string);
            }
        }
        try testing.expectEqual(@as(usize, 6), warm_gated);
    }
}
