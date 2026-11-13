//! Plan 75 Step 1 bound-measurement harness: opt-in stage/layer timing, overlapped GPU+TCP probe, and KV export check.
const std = @import("std");
const env = @import("../../env.zig");
const mlx = @import("../../mlx/ffi.zig");
const crt = @import("../../sdk/util/crt.zig");
const FsPackDir = @import("../../sdk/util/pack_dir.zig").FsPackDir;
const dsv41 = @import("deepseek_v41.zig");
const cache_codec = @import("cache_codec.zig");
const config_mod = @import("config.zig");
const plugin_abi = @import("../../abi/plugin.zig");
const tcp = @import("pp_step1_tcp.zig");

const A = mlx.mlx_array;
const none: A = .{ .ctx = null };

pub const CED_TAIL_ROWS: usize = 128;
pub const CED_MIN_CHUNK: usize = 384;
pub const PAIR_SEAM_ROW_BYTES: u64 = 4 * 5120 * 2 + 4 * 4;
pub const MID_GROUP_EXTRA_ROW_BYTES: u64 = 512 * 4 + 256 * 2;
pub const MID_GROUP_SEAM_ROW_BYTES: u64 = PAIR_SEAM_ROW_BYTES + MID_GROUP_EXTRA_ROW_BYTES;
pub const WIN_RING_BYTES: u64 = 144 * 512 * 2;
pub const TAIL_128_WIRE_BYTES: u64 = CED_TAIL_ROWS * PAIR_SEAM_ROW_BYTES;
pub const RELEASE_FIXED_EXPORT_BYTES: u64 = 40 * WIN_RING_BYTES + 3 * WIN_RING_BYTES + 15360 * 2;
pub const RELEASE_ODD_PEND_BYTES: u64 = 3 * 2 * 512 * 4;

pub const CutFamily = enum { A, B };

pub const SeamKind = enum {
    pair_cut,
    mid_group,
    tail,

    pub fn name(self: SeamKind) []const u8 {
        return @tagName(self);
    }
};

pub const StageSpec = struct {
    lo: usize,
    hi: usize,
    l20_comp: bool = false,
    seam: SeamKind,
};

pub const Schedule = enum {
    off,
    A2,
    A3,
    A4,
    B2,
    B3,
    B4,
    diag,

    pub fn parse(raw: ?[]const u8) Schedule {
        const s = std.mem.trim(u8, raw orelse return .off, " \t\r\n");
        if (s.len == 0 or std.mem.eql(u8, s, "0") or std.ascii.eqlIgnoreCase(s, "off")) return .off;
        if (std.ascii.eqlIgnoreCase(s, "A2")) return .A2;
        if (std.ascii.eqlIgnoreCase(s, "A3")) return .A3;
        if (std.ascii.eqlIgnoreCase(s, "A4")) return .A4;
        if (std.ascii.eqlIgnoreCase(s, "B2")) return .B2;
        if (std.ascii.eqlIgnoreCase(s, "B3")) return .B3;
        if (std.ascii.eqlIgnoreCase(s, "B4")) return .B4;
        if (std.ascii.eqlIgnoreCase(s, "diag")) return .diag;
        return .off;
    }

    pub fn family(self: Schedule) ?CutFamily {
        return switch (self) {
            .A2, .A3, .A4 => .A,
            .B2, .B3, .B4 => .B,
            .off, .diag => null,
        };
    }

    pub fn ppWidth(self: Schedule) usize {
        return switch (self) {
            .A2, .B2 => 2,
            .A3, .B3 => 3,
            .A4, .B4 => 4,
            .off, .diag => 0,
        };
    }

    pub fn stages(self: Schedule) []const StageSpec {
        return switch (self) {
            .A2 => &stages_a2,
            .A3 => &stages_a3,
            .A4 => &stages_a4,
            .B2 => &stages_b2,
            .B3 => &stages_b3,
            .B4 => &stages_b4,
            .off, .diag => &.{},
        };
    }
};

const stages_a2 = [_]StageSpec{
    .{ .lo = 0, .hi = 10, .seam = .mid_group },
    .{ .lo = 10, .hi = 20, .l20_comp = true, .seam = .tail },
};

const stages_a3 = [_]StageSpec{
    .{ .lo = 0, .hi = 7, .seam = .mid_group },
    .{ .lo = 7, .hi = 14, .seam = .pair_cut },
    .{ .lo = 14, .hi = 20, .l20_comp = true, .seam = .tail },
};

const stages_a4 = [_]StageSpec{
    .{ .lo = 0, .hi = 5, .seam = .mid_group },
    .{ .lo = 5, .hi = 10, .seam = .mid_group },
    .{ .lo = 10, .hi = 15, .seam = .mid_group },
    .{ .lo = 15, .hi = 20, .l20_comp = true, .seam = .tail },
};

const stages_b2 = [_]StageSpec{
    .{ .lo = 0, .hi = 8, .seam = .pair_cut },
    .{ .lo = 8, .hi = 20, .l20_comp = true, .seam = .tail },
};

const stages_b3 = [_]StageSpec{
    .{ .lo = 0, .hi = 8, .seam = .pair_cut },
    .{ .lo = 8, .hi = 14, .seam = .pair_cut },
    .{ .lo = 14, .hi = 20, .l20_comp = true, .seam = .tail },
};

const stages_b4 = [_]StageSpec{
    .{ .lo = 0, .hi = 2, .seam = .pair_cut },
    .{ .lo = 2, .hi = 8, .seam = .pair_cut },
    .{ .lo = 8, .hi = 14, .seam = .pair_cut },
    .{ .lo = 14, .hi = 20, .l20_comp = true, .seam = .tail },
};

pub fn seamPayloadBytes(rows: usize, mid_group: bool) u64 {
    const per_row = if (mid_group) MID_GROUP_SEAM_ROW_BYTES else PAIR_SEAM_ROW_BYTES;
    return @as(u64, rows) * per_row;
}

pub fn planPpChunks(n_snap: usize, chunk_c: usize, out: []usize) ![]const usize {
    if (n_snap < CED_MIN_CHUNK) return error.PromptTooShort;
    if (chunk_c < 2 * CED_MIN_CHUNK or chunk_c % 2 != 0) return error.InvalidChunkWidth;
    const m = (n_snap + chunk_c - 1) / chunk_c;
    if (out.len < m) return error.BufferTooSmall;
    if (m == 1) {
        out[0] = n_snap;
        return out[0..1];
    }
    for (0..m - 2) |i| out[i] = chunk_c;
    const rem = n_snap - (m - 2) * chunk_c;
    if (rem - chunk_c >= CED_MIN_CHUNK) {
        out[m - 2] = chunk_c;
        out[m - 1] = rem - chunk_c;
    } else {
        out[m - 2] = chunk_c - CED_MIN_CHUNK;
        out[m - 1] = rem - chunk_c + CED_MIN_CHUNK;
    }
    return out[0..m];
}

pub fn releaseCacheLayout(settings: u64) cache_codec.Layout {
    const kv_mask: u64 = (@as(u64, 1) << 2) | (@as(u64, 1) << 8) | (@as(u64, 1) << 14) | (@as(u64, 1) << 20);
    const idx_mask: u64 = kv_mask | (@as(u64, 1) << 24) | (@as(u64, 1) << 28) | (@as(u64, 1) << 32) | (@as(u64, 1) << 36);
    var l: cache_codec.Layout = .{
        .layers = 40,
        .stages = 3,
        .ring = 144,
        .width = 512,
        .index_width = 128,
        .hidden_width = 3 * 5120,
        .dtype = .bfloat16,
        .kv_sources = kv_mask,
        .index_sources = idx_mask,
        .engram = true,
        .settings = settings,
    };
    l.ratios[0] = 0;
    l.ratios[1] = 0;
    for (2..20) |i| l.ratios[i] = 2;
    for (20..40) |i| l.ratios[i] = 1;
    return l;
}

pub fn expectedPlanExportBytes(n: usize) u64 {
    const half: u64 = n / 2;
    const nn: u64 = n;
    const pend: u64 = if (n % 2 == 1) RELEASE_ODD_PEND_BYTES else 0;
    return RELEASE_FIXED_EXPORT_BYTES + pend + 3840 * half + 1284 * nn;
}

pub const NonStage0KvSizes = struct {
    rows_bytes: u64,
    pend_bytes: u64,
    win_bytes: u64,
    total_bytes: u64,
};

pub fn nonStage0KvBytes(n: usize) NonStage0KvSizes {
    const r2_used: u64 = n / 2;
    const r1_used: u64 = n;
    const rows_bytes = 2 * 1280 * r2_used + 1280 * r1_used;
    const pend_bytes: u64 = if (n % 2 == 1) 2 * 2 * 512 * 4 else 0;
    const win_bytes: u64 = 15 * WIN_RING_BYTES;
    return .{
        .rows_bytes = rows_bytes,
        .pend_bytes = pend_bytes,
        .win_bytes = win_bytes,
        .total_bytes = rows_bytes + pend_bytes + win_bytes,
    };
}

pub fn tbPeerIp4(src_node: u8, dst_node: u8) ![4]u8 {
    if (src_node == 0 and dst_node == 0) return .{ 127, 0, 0, 1 };
    if (src_node < 1 or src_node > 4 or dst_node < 1 or dst_node > 4 or src_node == dst_node) return error.InvalidNodePair;
    const lo = @min(src_node, dst_node);
    const hi = @max(src_node, dst_node);
    return .{ 10, 254, lo * 10 + hi, dst_node };
}

pub const JsonOut = struct {
    fd: c_int = -1,
    owns_fd: bool = false,

    pub fn openPath(path: ?[]const u8) !JsonOut {
        const p = path orelse return .{ .fd = 2, .owns_fd = false };
        if (p.len == 0 or std.mem.eql(u8, p, "-") or std.mem.eql(u8, p, "stderr")) return .{ .fd = 2, .owns_fd = false };
        if (std.mem.eql(u8, p, "stdout")) return .{ .fd = 1, .owns_fd = false };
        var zbuf: [1024]u8 = undefined;
        const z = try std.fmt.bufPrintSentinel(&zbuf, "{s}", .{p}, 0);
        const fd = crt.openWrite(z, false);
        if (fd < 0) return error.JsonOutOpenFailed;
        return .{ .fd = fd, .owns_fd = true };
    }

    pub fn close(self: *JsonOut) void {
        if (self.owns_fd and self.fd >= 0) crt.close(self.fd);
        self.* = .{};
    }

    pub fn writeLine(self: *const JsonOut, line: []const u8) void {
        if (self.fd < 0) return;
        _ = crt.write(self.fd, line);
    }
};

pub fn formatStageJson(buf: []u8, sched: Schedule, stage_idx: usize, spec: StageSpec, chunk_rows: usize, p0: usize, rep: usize, warmup: bool, elapsed_us: i64) ![]const u8 {
    const fam = if (sched.family()) |f| @tagName(f) else "none";
    const l20 = if (spec.l20_comp) "true" else "false";
    const wup = if (warmup) "true" else "false";
    return std.fmt.bufPrint(
        buf,
        "{{\"kind\":\"stage_timing\",\"family\":\"{s}\",\"pp_width\":{d},\"schedule\":\"{s}\",\"stage_idx\":{d},\"lo\":{d},\"hi\":{d},\"l20_comp\":{s},\"seam\":\"{s}\",\"chunk_rows\":{d},\"p0\":{d},\"rep\":{d},\"warmup\":{s},\"elapsed_us\":{d}}}\n",
        .{ fam, sched.ppWidth(), @tagName(sched), stage_idx, spec.lo, spec.hi, l20, spec.seam.name(), chunk_rows, p0, rep, wup, @max(elapsed_us, 0) },
    );
}

pub fn formatTailJson(buf: []u8, sched: Schedule, lo: usize, hi: usize, rows: usize, chunk_rows: usize, p0: usize, rep: usize, warmup: bool, elapsed_us: i64) ![]const u8 {
    const wup = if (warmup) "true" else "false";
    return std.fmt.bufPrint(
        buf,
        "{{\"kind\":\"tail_timing\",\"schedule\":\"{s}\",\"lo\":{d},\"hi\":{d},\"rows\":{d},\"chunk_rows\":{d},\"p0\":{d},\"rep\":{d},\"warmup\":{s},\"elapsed_us\":{d}}}\n",
        .{ @tagName(sched), lo, hi, rows, chunk_rows, p0, rep, wup, @max(elapsed_us, 0) },
    );
}

pub fn formatLayerDiagJson(buf: []u8, layer: usize, is_l20_comp: bool, chunk_rows: usize, p0: usize, rep: usize, warmup: bool, total_us: i64, engram_pread_us: i64, engram_wait_us: i64, engram_eval_us: i64, indexer_us: i64) ![]const u8 {
    const l20 = if (is_l20_comp) "true" else "false";
    const wup = if (warmup) "true" else "false";
    return std.fmt.bufPrint(
        buf,
        "{{\"kind\":\"layer_diag\",\"layer\":{d},\"is_l20_comp\":{s},\"chunk_rows\":{d},\"p0\":{d},\"rep\":{d},\"warmup\":{s},\"total_us\":{d},\"engram_pread_us\":{d},\"engram_wait_us\":{d},\"engram_eval_us\":{d},\"indexer_us\":{d}}}\n",
        .{ layer, l20, chunk_rows, p0, rep, wup, @max(total_us, 0), @max(engram_pread_us, 0), @max(engram_wait_us, 0), @max(engram_eval_us, 0), @max(indexer_us, 0) },
    );
}

pub fn formatTcpHopJson(buf: []u8, mode: []const u8, chunk_rows: usize, seam: SeamKind, bytes: u64, iters: usize, mean_us: f64, median_us: f64, best_us: f64, p95_us: f64, uma_memcpy_us: f64, uma_mlx_new_us: f64, iface: []const u8, local_ip: [4]u8, peer_ip: [4]u8, mtu: i32, sndbuf: i32, rcvbuf: i32) ![]const u8 {
    const gbps = if (median_us > 0) @as(f64, @floatFromInt(bytes)) / (median_us * 1000.0) else 0.0;
    return std.fmt.bufPrint(
        buf,
        "{{\"kind\":\"tcp_hop\",\"mode\":\"{s}\",\"chunk_rows\":{d},\"seam\":\"{s}\",\"bytes\":{d},\"iters\":{d},\"mean_us\":{d:.1},\"median_us\":{d:.1},\"best_us\":{d:.1},\"p95_us\":{d:.1},\"uma_memcpy_us\":{d:.1},\"uma_mlx_new_us\":{d:.1},\"gbps\":{d:.3},\"iface\":\"{s}\",\"local_ip\":\"{d}.{d}.{d}.{d}\",\"peer_ip\":\"{d}.{d}.{d}.{d}\",\"mtu\":{d},\"sndbuf\":{d},\"rcvbuf\":{d}}}\n",
        .{ mode, chunk_rows, seam.name(), bytes, iters, mean_us, median_us, best_us, p95_us, uma_memcpy_us, uma_mlx_new_us, gbps, iface, local_ip[0], local_ip[1], local_ip[2], local_ip[3], peer_ip[0], peer_ip[1], peer_ip[2], peer_ip[3], mtu, sndbuf, rcvbuf },
    );
}

pub fn formatOverlapJson(buf: []u8, chunk_rows: usize, seam: SeamKind, bytes: u64, moe_lo: usize, moe_hi: usize, p0: usize, iters: usize, gpu_solo_us: f64, tcp_solo_us: f64, gpu_overlap_us: f64, tcp_overlap_us: f64, step_wall_us: f64) ![]const u8 {
    const gpu_ratio = if (gpu_solo_us > 0) gpu_overlap_us / gpu_solo_us else 1.0;
    const tcp_ratio = if (tcp_solo_us > 0) tcp_overlap_us / tcp_solo_us else 1.0;
    return std.fmt.bufPrint(
        buf,
        "{{\"kind\":\"tcp_gpu_overlap\",\"chunk_rows\":{d},\"seam\":\"{s}\",\"bytes\":{d},\"moe_lo\":{d},\"moe_hi\":{d},\"p0\":{d},\"iters\":{d},\"gpu_solo_median_us\":{d:.1},\"tcp_solo_median_us\":{d:.1},\"gpu_overlap_median_us\":{d:.1},\"tcp_overlap_median_us\":{d:.1},\"step_wall_median_us\":{d:.1},\"gpu_slowdown_ratio\":{d:.4},\"tcp_slowdown_ratio\":{d:.4}}}\n",
        .{ chunk_rows, seam.name(), bytes, moe_lo, moe_hi, p0, iters, gpu_solo_us, tcp_solo_us, gpu_overlap_us, tcp_overlap_us, step_wall_us, gpu_ratio, tcp_ratio },
    );
}

pub fn formatKvExportJson(buf: []u8, prompt_tokens: usize, n_snap: usize, tensor_count: usize, export_bytes: u64, plan_bytes: u64, formula_bytes: u64, non_stage0_rows_bytes: u64, non_stage0_total_bytes: u64) ![]const u8 {
    const hist_bytes: u64 = @as(u64, n_snap) * 4;
    const ok = if (export_bytes == plan_bytes and plan_bytes == formula_bytes) "true" else "false";
    return std.fmt.bufPrint(
        buf,
        "{{\"kind\":\"kv_export_check\",\"prompt_tokens\":{d},\"n_snap\":{d},\"tensor_count\":{d},\"export_bytes\":{d},\"plan_bytes\":{d},\"formula_bytes\":{d},\"hist_local_bytes\":{d},\"non_stage0_rows_bytes\":{d},\"non_stage0_total_bytes\":{d},\"match\":{s}}}\n",
        .{ prompt_tokens, n_snap, tensor_count, export_bytes, plan_bytes, formula_bytes, hist_bytes, non_stage0_rows_bytes, non_stage0_total_bytes, ok },
    );
}

pub fn formatKvStreamJson(buf: []u8, subtest: []const u8, prompt_tokens: usize, chunk_rows: usize, wire_bytes: u64, rows_bytes: u64, iters: usize, tcp_median_us: f64, fill_rows_median_us: f64, total_median_us: f64) ![]const u8 {
    const gbps = if (tcp_median_us > 0) @as(f64, @floatFromInt(wire_bytes)) / (tcp_median_us * 1000.0) else 0.0;
    return std.fmt.bufPrint(
        buf,
        "{{\"kind\":\"kv_stream\",\"subtest\":\"{s}\",\"prompt_tokens\":{d},\"chunk_rows\":{d},\"wire_bytes\":{d},\"rows_bytes\":{d},\"iters\":{d},\"tcp_median_us\":{d:.1},\"fill_rows_median_us\":{d:.1},\"total_median_us\":{d:.1},\"gbps\":{d:.3}}}\n",
        .{ subtest, prompt_tokens, chunk_rows, wire_bytes, rows_bytes, iters, tcp_median_us, fill_rows_median_us, total_median_us, gbps },
    );
}

pub const TimingState = struct {
    schedule: Schedule = .off,
    rep: usize = 0,
    warmup: bool = false,
    out: JsonOut = .{},
    diag_engram_hit: bool = false,
    diag_engram_pread_us: i64 = 0,
    diag_engram_wait_us: i64 = 0,
    diag_engram_eval_us: i64 = 0,
    diag_indexer_us: i64 = 0,

    pub fn fromEnv() TimingState {
        const sched = Schedule.parse(env.span(.DSV41_PP_STAGE_TIMING));
        if (sched == .off) return .{};
        const out = JsonOut.openPath(env.span(.DSV41_PP_STAGE_JSON_OUT)) catch JsonOut{ .fd = 2, .owns_fd = false };
        return .{ .schedule = sched, .out = out };
    }

    pub fn close(self: *TimingState) void {
        self.out.close();
    }

    pub inline fn activeFor(self: *const TimingState, ok: bool) bool {
        return self.schedule != .off and ok;
    }

    pub inline fn isDiag(self: *const TimingState) bool {
        return self.schedule == .diag;
    }

    pub fn resetLayerDiag(self: *TimingState) void {
        self.diag_engram_hit = false;
        self.diag_engram_pread_us = 0;
        self.diag_engram_wait_us = 0;
        self.diag_engram_eval_us = 0;
        self.diag_indexer_us = 0;
    }

    pub fn noteEngramSpan(self: *TimingState, pread_us: i64, wait_us: i64) void {
        self.diag_engram_hit = true;
        self.diag_engram_pread_us += pread_us;
        self.diag_engram_wait_us += wait_us;
    }
};

fn monoUs() i64 {
    const io = std.Io.Threaded.global_single_threaded.io();
    return @intCast(@divTrunc(std.Io.Timestamp.now(io, .awake).toNanoseconds(), 1000));
}

fn evalStageInputs(gpa: std.mem.Allocator, sp: *const dsv41.SpansForBench) !void {
    var ev: std.ArrayList(A) = .empty;
    defer ev.deinit(gpa);
    for (sp.xs) |x| if (x.ctx != null) try ev.append(gpa, x);
    for (sp.pres) |x| if (x.ctx != null) try ev.append(gpa, x);
    if (ev.items.len > 0) try dsv41.evalAllForBench(ev.items);
}

fn drainStageCut(gpa: std.mem.Allocator, held: []const A, st: *const dsv41.State, lo: usize, hi: usize) !void {
    var ev: std.ArrayList(A) = .empty;
    defer ev.deinit(gpa);
    for (held) |x| if (x.ctx != null) try ev.append(gpa, x);
    for (lo..hi) |li| try dsv41.appendLayerStateForBench(gpa, &ev, &st.layers[li]);
    if (ev.items.len > 0) try dsv41.evalAllForBench(ev.items);
}

pub fn runTimedStages(
    m: *dsv41.Dsv41Model,
    gpa: std.mem.Allocator,
    st: *dsv41.State,
    sp: *dsv41.SpansForBench,
    held: []A,
    p0: usize,
    chunk_rows: usize,
    enc_hi: usize,
    tail_rows: usize,
    n_spans: usize,
    n_tail: usize,
    targets: usize,
    span_w: usize,
) !void {
    try evalStageInputs(gpa, sp);
    var line_buf: [512]u8 = undefined;
    const sched = m.pp_timing.schedule;
    const rep = m.pp_timing.rep;
    const warmup = m.pp_timing.warmup;

    if (sched == .diag) {
        for (0..enc_hi) |li| {
            m.pp_timing.resetLayerDiag();
            const t0 = monoUs();
            try dsv41.runLayersWithDrain(m, gpa, st, sp, held, li, li + 1, true);
            try drainStageCut(gpa, held, st, li, li + 1);
            const total_us = monoUs() - t0;
            m.pp_timing.out.writeLine(try formatLayerDiagJson(
                &line_buf, li, false, chunk_rows, p0, rep, warmup, total_us,
                m.pp_timing.diag_engram_pread_us, m.pp_timing.diag_engram_wait_us,
                m.pp_timing.diag_engram_eval_us, m.pp_timing.diag_indexer_us,
            ));
        }
        var tsp = dsv41.spansOfForBench(held[n_spans * (4 + targets) ..], n_tail, targets, span_w, tail_rows, p0 + chunk_rows - tail_rows);
        tsp.whole = held[held.len - 2 ..][0..2];
        tsp.floor = tsp.p0;
        const tc0 = monoUs();
        try dsv41.cedReplayInForBench(m, gpa, st, sp, &tsp, enc_hi);
        try drainStageCut(gpa, held, st, enc_hi, enc_hi + 1);
        const l20c_us = monoUs() - tc0;
        m.pp_timing.out.writeLine(try formatLayerDiagJson(&line_buf, enc_hi, true, chunk_rows, p0, rep, warmup, l20c_us, 0, 0, 0, 0));

        for (enc_hi..m.n_layers) |li| {
            m.pp_timing.resetLayerDiag();
            const t0 = monoUs();
            try dsv41.runLayersWithDrain(m, gpa, st, &tsp, held, li, li + 1, true);
            try drainStageCut(gpa, held, st, li, li + 1);
            const total_us = monoUs() - t0;
            m.pp_timing.out.writeLine(try formatLayerDiagJson(
                &line_buf, li, false, tail_rows, tsp.p0, rep, warmup, total_us,
                0, 0, 0, m.pp_timing.diag_indexer_us,
            ));
        }
        m.ced_stats.chunks += 1;
        m.ced_stats.skipped += chunk_rows - tail_rows;
        sp.* = tsp;
        if (!env.present(.DSV41_PP_STAGE_JSON_OUT)) m.pp_timing.rep += 1;
        return;
    }

    const list = sched.stages();
    var tsp = dsv41.spansOfForBench(held[n_spans * (4 + targets) ..], n_tail, targets, span_w, tail_rows, p0 + chunk_rows - tail_rows);
    tsp.whole = held[held.len - 2 ..][0..2];
    tsp.floor = tsp.p0;

    for (list, 0..) |stage, idx| {
        const lo = @min(stage.lo, enc_hi);
        const hi = @min(stage.hi, enc_hi);
        const t0 = monoUs();
        if (!stage.l20_comp) {
            if (lo < hi) try dsv41.runLayersWithDrain(m, gpa, st, sp, held, lo, hi, true);
            if (lo < hi and n_spans <= 1) try drainStageCut(gpa, held, st, lo, hi);
        } else {
            if (lo < hi) try dsv41.runLayersWithDrain(m, gpa, st, sp, held, lo, hi, false);
            try dsv41.cedReplayInForBench(m, gpa, st, sp, &tsp, enc_hi);
            try drainStageCut(gpa, held, st, lo, enc_hi + 1);
        }
        const elapsed = monoUs() - t0;
        m.pp_timing.out.writeLine(try formatStageJson(&line_buf, sched, idx, stage, chunk_rows, p0, rep, warmup, elapsed));
    }

    const t_tail0 = monoUs();
    try dsv41.runLayersWithDrain(m, gpa, st, &tsp, held, enc_hi, m.n_layers, true);
    try drainStageCut(gpa, held, st, enc_hi, m.n_layers);
    const tail_us = monoUs() - t_tail0;
    m.pp_timing.out.writeLine(try formatTailJson(&line_buf, sched, enc_hi, m.n_layers, tail_rows, chunk_rows, p0, rep, warmup, tail_us));

    m.ced_stats.chunks += 1;
    m.ced_stats.skipped += chunk_rows - tail_rows;
    sp.* = tsp;
    if (!env.present(.DSV41_PP_STAGE_JSON_OUT)) m.pp_timing.rep += 1;
}

const OverlapWorker = struct {
    fd: tcp.SocketFd,
    payload: []const u8,
    half_ping: f64,
    req_gen: std.atomic.Value(u32) = .init(0),
    done_gen: std.atomic.Value(u32) = .init(0),
    quit: std.atomic.Value(bool) = .init(false),
    failed: std.atomic.Value(bool) = .init(false),
    last_tcp_us: f64 = 0,

    fn run(self: *OverlapWorker) void {
        if (comptime @import("builtin").os.tag == .windows) return;
        var seen: u32 = 0;
        while (true) {
            while (self.req_gen.load(.acquire) == seen) {
                if (self.quit.load(.acquire)) return;
                std.atomic.spinLoopHint();
            }
            seen = self.req_gen.load(.acquire);
            if (self.quit.load(.acquire)) return;

            const hdr: tcp.FrameHdr = .{ .cmd = tcp.CMD_ONE_WAY, .bytes = self.payload.len, .arg0 = 1 };
            tcp.writeFull(self.fd, std.mem.asBytes(&hdr)) catch {
                self.failed.store(true, .release);
                self.done_gen.store(seen, .release);
                return;
            };
            var ready: [1]u8 = undefined;
            tcp.readFull(self.fd, &ready, 10_000) catch {
                self.failed.store(true, .release);
                self.done_gen.store(seen, .release);
                return;
            };
            const t0 = monoUs();
            tcp.writeFull(self.fd, self.payload) catch {
                self.failed.store(true, .release);
                self.done_gen.store(seen, .release);
                return;
            };
            var ack: tcp.FrameAck = undefined;
            tcp.readFull(self.fd, std.mem.asBytes(&ack), 60_000) catch {
                self.failed.store(true, .release);
                self.done_gen.store(seen, .release);
                return;
            };
            self.last_tcp_us = @max(0.0, @as(f64, @floatFromInt(monoUs() - t0)) - self.half_ping);
            self.done_gen.store(seen, .release);
        }
    }
};

const StateSave = struct {
    gpa: std.mem.Allocator,
    n: usize,
    hist_len: usize,
    comp_used: []usize,
    idx_used: []usize,
    wins: []A,
    pend_kv: []A,
    pend_gate: []A,

    fn capture(gpa: std.mem.Allocator, st: *const dsv41.State) !StateSave {
        const nl = st.layers.len;
        const comp_used = try gpa.alloc(usize, nl);
        errdefer gpa.free(comp_used);
        const idx_used = try gpa.alloc(usize, nl);
        errdefer gpa.free(idx_used);
        const wins = try gpa.alloc(A, nl);
        errdefer gpa.free(wins);
        @memset(wins, none);
        errdefer for (wins) |x| if (x.ctx != null) {
            _ = mlx.mlx_array_free(x);
        };
        const pend_kv = try gpa.alloc(A, nl);
        errdefer gpa.free(pend_kv);
        @memset(pend_kv, none);
        errdefer for (pend_kv) |x| if (x.ctx != null) {
            _ = mlx.mlx_array_free(x);
        };
        const pend_gate = try gpa.alloc(A, nl);
        errdefer gpa.free(pend_gate);
        @memset(pend_gate, none);

        for (st.layers, 0..) |*l, i| {
            comp_used[i] = if (l.comp) |r| r.used else 0;
            idx_used[i] = if (l.idx_k) |r| r.used else 0;
            wins[i] = dsv41.refForBench(l.win.buf);
            pend_kv[i] = dsv41.refForBench(l.pend_kv);
            pend_gate[i] = dsv41.refForBench(l.pend_gate);
        }
        return .{
            .gpa = gpa,
            .n = st.n,
            .hist_len = st.hist.items.len,
            .comp_used = comp_used,
            .idx_used = idx_used,
            .wins = wins,
            .pend_kv = pend_kv,
            .pend_gate = pend_gate,
        };
    }

    fn restore(self: *const StateSave, st: *dsv41.State) void {
        st.n = self.n;
        st.hist.items.len = self.hist_len;
        for (st.layers, 0..) |*l, i| {
            if (l.comp) |*r| r.used = self.comp_used[i];
            if (l.idx_k) |*r| r.used = self.idx_used[i];
            dsv41.holdForBench(&l.win.buf, self.wins[i]);
            dsv41.holdForBench(&l.pend_kv, self.pend_kv[i]);
            dsv41.holdForBench(&l.pend_gate, self.pend_gate[i]);
        }
    }

    fn deinit(self: *StateSave) void {
        for (self.wins) |x| if (x.ctx != null) {
            _ = mlx.mlx_array_free(x);
        };
        for (self.pend_kv) |x| if (x.ctx != null) {
            _ = mlx.mlx_array_free(x);
        };
        for (self.pend_gate) |x| if (x.ctx != null) {
            _ = mlx.mlx_array_free(x);
        };
        self.gpa.free(self.comp_used);
        self.gpa.free(self.idx_used);
        self.gpa.free(self.wins);
        self.gpa.free(self.pend_kv);
        self.gpa.free(self.pend_gate);
    }
};

fn runOverlapBench(
    m: *dsv41.Dsv41Model,
    st: *dsv41.State,
    sock: *const tcp.BoundSocketInfo,
    gpa: std.mem.Allocator,
    json_out: *const JsonOut,
    widths: []const usize,
    warmup: usize,
    iters: usize,
) !void {
    return runOverlapBenchRange(m, st, sock, gpa, json_out, widths, 5, @min(@as(usize, 10), m.n_layers), warmup, iters);
}

fn runOverlapBenchRange(
    m: *dsv41.Dsv41Model,
    st: *dsv41.State,
    sock: *const tcp.BoundSocketInfo,
    gpa: std.mem.Allocator,
    json_out: *const JsonOut,
    widths: []const usize,
    moe_lo: usize,
    moe_hi: usize,
    warmup: usize,
    iters: usize,
) !void {
    if (comptime @import("builtin").os.tag == .windows) return error.UnsupportedPlatform;
    if (moe_lo >= moe_hi) return;
    const half_ping = try tcp.measureHalfPingUs(sock.fd);
    const p0 = st.n;
    var base_save = try StateSave.capture(gpa, st);
    defer base_save.deinit();
    var line_buf: [512]u8 = undefined;

    for (widths) |c_rows| {
        const nbytes_u64 = seamPayloadBytes(c_rows, true);
        const nbytes: usize = @intCast(nbytes_u64);
        const payload = try gpa.alloc(u8, nbytes);
        defer gpa.free(payload);
        @memset(payload, 0x3C);

        const tcp_solo_samples = try gpa.alloc(f64, iters);
        defer gpa.free(tcp_solo_samples);
        for (0..warmup + iters) |rep| {
            const hdr: tcp.FrameHdr = .{ .cmd = tcp.CMD_ONE_WAY, .bytes = nbytes_u64, .arg0 = 1 };
            try tcp.writeFull(sock.fd, std.mem.asBytes(&hdr));
            var ready: [1]u8 = undefined;
            try tcp.readFull(sock.fd, &ready, 10_000);
            const t0 = monoUs();
            try tcp.writeFull(sock.fd, payload);
            var ack: tcp.FrameAck = undefined;
            try tcp.readFull(sock.fd, std.mem.asBytes(&ack), 60_000);
            if (rep >= warmup) tcp_solo_samples[rep - warmup] = @max(0.0, @as(f64, @floatFromInt(monoUs() - t0)) - half_ping);
        }
        const tcp_solo_med = tcp.statsOf(tcp_solo_samples).median;

        const span_w = m.span;
        const n_spans = (c_rows + span_w - 1) / span_w;
        const targets: usize = if (m.n_mtp > 0) m.cfg.dsv4_n_dspark_target_layers else 0;
        const held = try gpa.alloc(A, n_spans * (4 + targets) + 2);
        defer {
            for (held) |x| if (x.ctx != null) {
                _ = mlx.mlx_array_free(x);
            };
            gpa.free(held);
        }
        @memset(held, none);

        const init_xs = try gpa.alloc(A, n_spans);
        defer {
            for (init_xs) |x| if (x.ctx != null) {
                _ = mlx.mlx_array_free(x);
            };
            gpa.free(init_xs);
        }
        @memset(init_xs, none);
        const init_pres = try gpa.alloc(A, n_spans);
        defer {
            for (init_pres) |x| if (x.ctx != null) {
                _ = mlx.mlx_array_free(x);
            };
            gpa.free(init_pres);
        }
        @memset(init_pres, none);
        const init_topks = try gpa.alloc(A, n_spans);
        defer {
            for (init_topks) |x| if (x.ctx != null) {
                _ = mlx.mlx_array_free(x);
            };
            gpa.free(init_topks);
        }
        @memset(init_topks, none);

        base_save.restore(st);
        try dsv41.seedMidGroupSeamForBench(m, gpa, st, moe_lo, p0, c_rows, span_w, init_xs, init_pres, init_topks);
        var iter_save = try StateSave.capture(gpa, st);
        defer iter_save.deinit();

        const gpu_solo_samples = try gpa.alloc(f64, iters);
        defer gpa.free(gpu_solo_samples);
        for (0..warmup + iters) |rep| {
            iter_save.restore(st);
            for (held) |*h| dsv41.holdForBench(h, none);
            var sp = dsv41.spansOfForBench(held, n_spans, targets, span_w, c_rows, p0);
            sp.whole = held[held.len - 2 ..][0..2];
            for (0..n_spans) |k| {
                dsv41.holdForBench(&sp.xs[k], init_xs[k]);
                dsv41.holdForBench(&sp.pres[k], init_pres[k]);
                dsv41.holdForBench(&sp.topks[k], init_topks[k]);
            }
            const t0 = monoUs();
            try dsv41.runLayersWithDrain(m, gpa, st, &sp, held, moe_lo, moe_hi, true);
            if (n_spans <= 1) try drainStageCut(gpa, held, st, moe_lo, moe_hi);
            const dt = @as(f64, @floatFromInt(@max(monoUs() - t0, 0)));
            if (rep >= warmup) gpu_solo_samples[rep - warmup] = dt;
            _ = mlx.mlx_clear_cache();
        }
        const gpu_solo_med = tcp.statsOf(gpu_solo_samples).median;

        var worker: OverlapWorker = .{ .fd = sock.fd, .payload = payload, .half_ping = half_ping };
        const th = try std.Thread.spawn(.{ .stack_size = 1 << 20 }, OverlapWorker.run, .{&worker});
        defer {
            worker.quit.store(true, .release);
            _ = worker.req_gen.fetchAdd(1, .acq_rel);
            th.join();
        }

        const gpu_ov_samples = try gpa.alloc(f64, iters);
        defer gpa.free(gpu_ov_samples);
        const tcp_ov_samples = try gpa.alloc(f64, iters);
        defer gpa.free(tcp_ov_samples);
        const wall_samples = try gpa.alloc(f64, iters);
        defer gpa.free(wall_samples);

        var gen: u32 = 0;
        for (0..warmup + iters) |rep| {
            iter_save.restore(st);
            for (held) |*h| dsv41.holdForBench(h, none);
            var sp = dsv41.spansOfForBench(held, n_spans, targets, span_w, c_rows, p0);
            sp.whole = held[held.len - 2 ..][0..2];
            for (0..n_spans) |k| {
                dsv41.holdForBench(&sp.xs[k], init_xs[k]);
                dsv41.holdForBench(&sp.pres[k], init_pres[k]);
                dsv41.holdForBench(&sp.topks[k], init_topks[k]);
            }
            gen += 1;
            const t_wall0 = monoUs();
            worker.req_gen.store(gen, .release);
            try dsv41.runLayersWithDrain(m, gpa, st, &sp, held, moe_lo, moe_hi, true);
            if (n_spans <= 1) try drainStageCut(gpa, held, st, moe_lo, moe_hi);
            const gpu_dt = @as(f64, @floatFromInt(@max(monoUs() - t_wall0, 0)));
            while (worker.done_gen.load(.acquire) != gen) std.atomic.spinLoopHint();
            if (worker.failed.load(.acquire)) return error.OverlapTcpFailed;
            const wall_dt = @as(f64, @floatFromInt(@max(monoUs() - t_wall0, 0)));
            if (rep >= warmup) {
                gpu_ov_samples[rep - warmup] = gpu_dt;
                tcp_ov_samples[rep - warmup] = worker.last_tcp_us;
                wall_samples[rep - warmup] = @max(gpu_dt, @min(wall_dt, @max(gpu_dt, worker.last_tcp_us)));
            }
            _ = mlx.mlx_clear_cache();
        }
        base_save.restore(st);

        json_out.writeLine(try formatOverlapJson(
            &line_buf, c_rows, .mid_group, nbytes_u64, moe_lo, moe_hi, p0, iters,
            gpu_solo_med, tcp_solo_med,
            tcp.statsOf(gpu_ov_samples).median, tcp.statsOf(tcp_ov_samples).median, tcp.statsOf(wall_samples).median,
        ));
    }
}

pub fn verifyKvExportSizes(m: *const dsv41.Dsv41Model, gpa: std.mem.Allocator, json_out: *const JsonOut) !void {
    const prompt_ns = [_]usize{ 8192, 32768, 65536 };
    const layout = try dsv41.cacheLayoutForBench(m);
    const is_release = layout.layers == 40 and layout.stages == 3 and layout.ring == 144 and layout.width == 512 and layout.index_width == 128 and layout.hidden_width == 15360 and layout.engram;
    var line_buf: [512]u8 = undefined;

    for (prompt_ns) |prompt_n| {
        const n_snap = prompt_n - 1;
        const plan = try cache_codec.Plan.init(layout, n_snap, 0);
        const formula = if (is_release) expectedPlanExportBytes(n_snap) else plan.loaded;

        var sn = try dsv41.buildSyntheticSnapForBench(m, gpa, n_snap);
        defer sn.deinit(gpa);

        var tensors: [cache_codec.max_tensors]plugin_abi.AuxTensor = undefined;
        var names_buf: [cache_codec.max_tensors * 32]u8 = undefined;
        var header_buf: [cache_codec.max_header]u8 = undefined;
        var out: plugin_abi.AuxExport = .{
            .tensors = &tensors,
            .tensor_cap = tensors.len,
            .names = &names_buf,
            .names_cap = names_buf.len,
            .header = &header_buf,
            .header_cap = header_buf.len,
        };
        try dsv41.exportSnapshot(m, &sn, &out);
        defer for (out.tensors.?[0..out.tensor_len]) |t| {
            _ = mlx.mlx_array_free(.{ .ctx = t.array.ctx });
        };

        const decoded = try cache_codec.Plan.decode(layout, out.header.?[0..out.header_len]);
        _ = try decoded.validateArrays(out.tensors.?[0..out.tensor_len]);

        var export_bytes: u64 = 0;
        for (out.tensors.?[0..out.tensor_len]) |t| {
            const arr: A = .{ .ctx = t.array.ctx };
            export_bytes += @as(u64, mlx.mlx_array_size(arr)) * @as(u64, mlx.mlx_array_itemsize(arr));
        }

        const non_s0 = nonStage0KvBytes(n_snap);
        json_out.writeLine(try formatKvExportJson(
            &line_buf, prompt_n, n_snap, out.tensor_len, export_bytes, plan.loaded, formula,
            non_s0.rows_bytes, non_s0.total_bytes,
        ));
        if (export_bytes != plan.loaded or plan.loaded != formula) return error.KvExportByteMismatch;
    }
}

fn loadPackConfig(io: std.Io, gpa: std.mem.Allocator, dir: []const u8) !config_mod.Config {
    var fs = try FsPackDir.open(io, dir);
    defer fs.close();
    const pd = fs.packDir();
    const buf = try gpa.alloc(u8, 8 << 20);
    defer gpa.free(buf);
    var why: plugin_abi.Verdict = .{};
    const json = pd.readFile("config.json", buf, &why) orelse return error.ConfigUnreadable;
    var p: plugin_abi.Parsed = .{};
    if (!config_mod.parse(&pd, .of(json), .{}, null, false, &p).isOk()) return error.ConfigRefused;
    defer config_mod.freeParsed(&p);
    return config_mod.of(&p).*;
}

fn fillDeterministicTokens(dst: []u32, vocab: usize, salt: usize) void {
    const cap = @max(@as(usize, 1), @min(vocab, 100_000));
    for (dst, 0..) |*t, i| {
        t.* = @intCast(((i + 1) *% 7919 +% salt *% 1_000_003 +% 13) % cap);
    }
}

fn advanceStateTo(m: *dsv41.Dsv41Model, gpa: std.mem.Allocator, st: *dsv41.State, target_p0: usize, prompt_buf: []const u32) !void {
    m.pp_timing.schedule = .off;
    while (st.n < target_p0) {
        const step = @min(@as(usize, 8192), target_p0 - st.n);
        const logits = try dsv41.extendForBench(m, gpa, st, prompt_buf[st.n .. st.n + step]);
        try mlx.check(mlx.mlx_array_eval(logits));
        _ = mlx.mlx_array_free(logits);
        _ = mlx.mlx_clear_cache();
    }
}

pub fn cmdProbe(io: std.Io, gpa: std.mem.Allocator, args: []const []const u8) !void {
    if (comptime @import("builtin").os.tag == .windows) return error.UnsupportedPlatform;
    if (args.len == 0) return error.InvalidArgs;
    const sub = args[0];
    var listen_ep: ?[]const u8 = null;
    var peer_ep: ?[]const u8 = null;
    var model_dir: ?[]const u8 = null;
    var json_path: ?[]const u8 = null;
    var tcp_iters: usize = 50;
    var stage_warmup: usize = 3;
    var stage_reps: usize = 10;
    var diag_reps: usize = 3;

    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const a = args[i];
        if (i + 1 >= args.len) return error.InvalidArgs;
        const v = args[i + 1];
        i += 1;
        if (std.mem.eql(u8, a, "--listen")) {
            listen_ep = v;
        } else if (std.mem.eql(u8, a, "--peer")) {
            peer_ep = v;
        } else if (std.mem.eql(u8, a, "--model")) {
            model_dir = v;
        } else if (std.mem.eql(u8, a, "--json-out")) {
            json_path = v;
        } else if (std.mem.eql(u8, a, "--tcp-iters")) {
            tcp_iters = try std.fmt.parseInt(usize, v, 10);
        } else if (std.mem.eql(u8, a, "--stage-warmup")) {
            stage_warmup = try std.fmt.parseInt(usize, v, 10);
        } else if (std.mem.eql(u8, a, "--stage-reps")) {
            stage_reps = try std.fmt.parseInt(usize, v, 10);
        } else if (std.mem.eql(u8, a, "--diag-reps")) {
            diag_reps = try std.fmt.parseInt(usize, v, 10);
        } else {
            return error.InvalidArgs;
        }
    }

    if (std.mem.eql(u8, sub, "recv")) {
        const ep = try tcp.parseIp4Port(listen_ep orelse return error.InvalidArgs);
        const srv = try tcp.listenTcp(ep.ip4, ep.port);
        defer _ = std.c.close(srv.fd);
        try tcp.serveProbeListener(srv.fd, gpa);
        return;
    }

    var json_out = try JsonOut.openPath(json_path);
    defer json_out.close();

    const widths = [_]usize{ 8192, 4096, 2048 };
    const kv_ns = [_]usize{ 8192, 32768, 65536 };

    if (std.mem.eql(u8, sub, "tcp-bench")) {
        const ep = try tcp.parseIp4Port(peer_ep orelse return error.InvalidArgs);
        const sock = try tcp.bindAndConnectPeer(ep.ip4, ep.port, 5000);
        defer _ = std.c.close(sock.fd);
        try tcp.runTcpHopSuite(&sock, gpa, &json_out, &widths, 5, tcp_iters);
        try tcp.runKvStreamSuite(&sock, gpa, &json_out, &kv_ns, &widths, 2, 10);
        tcp.sendDone(sock.fd);
        return;
    }

    if (!std.mem.eql(u8, sub, "model-bench")) return error.InvalidArgs;
    const dir = model_dir orelse return error.InvalidArgs;

    @import("../../mlx/state.zig").applyMlxCacheLimit();
    _ = mlx.applyWiredPolicy();

    const cfg = try loadPackConfig(io, gpa, dir);
    var weights = try dsv41.loadPack(io, gpa, dir, false);
    defer weights.deinit();
    const m = try dsv41.init(gpa, &cfg, &weights, mlx.gpuStream(), .{
        .pack_dir = dir,
        .dspark = dsv41.dsparkWanted(&cfg, null, null),
        .head_8bit = dsv41.head8Wanted(),
    });
    defer m.deinit();

    try verifyKvExportSizes(m, gpa, &json_out);

    const max_p0: usize = 131_072;
    const max_chunk: usize = 8192;
    const prompt_buf = try gpa.alloc(u32, max_p0 + 2 * max_chunk);
    defer gpa.free(prompt_buf);
    fillDeterministicTokens(prompt_buf, m.vocab, 1);

    const rep_chunk = try gpa.alloc(u32, max_chunk);
    defer gpa.free(rep_chunk);

    var st = try dsv41.initState(m, gpa);
    defer st.deinit();

    m.pp_timing.out = json_out;
    defer m.pp_timing.out = .{};

    const schedules = [_]Schedule{ .A4, .A3, .A2, .B4, .B3, .B2 };
    const depths = [_]usize{ 0, 32768, 131072 };

    const peer_addr = if (peer_ep) |pep| try tcp.parseIp4Port(pep) else null;
    var peer_done_sent = peer_addr == null;
    defer if (!peer_done_sent) {
        if (peer_addr) |ep| {
            if (tcp.bindAndConnectPeer(ep.ip4, ep.port, 5000)) |sk| {
                tcp.sendDone(sk.fd);
                _ = std.c.close(sk.fd);
            } else |_| {}
        }
    };

    if (peer_addr) |ep| {
        const sock = try tcp.bindAndConnectPeer(ep.ip4, ep.port, 5000);
        defer _ = std.c.close(sock.fd);
        try tcp.runTcpHopSuite(&sock, gpa, &json_out, &widths, 5, tcp_iters);
        try tcp.runKvStreamSuite(&sock, gpa, &json_out, &kv_ns, &widths, 2, 10);
        try tcp.sendPark(sock.fd);
    }

    for (depths) |p0| {
        try advanceStateTo(m, gpa, &st, p0, prompt_buf);
        var save = try StateSave.capture(gpa, &st);
        defer save.deinit();

        for (schedules) |sched| {
            for (widths) |c_rows| {
                for (0..stage_warmup + stage_reps) |rep| {
                    save.restore(&st);
                    dsv41.dropAheadForBench(m);
                    fillDeterministicTokens(rep_chunk[0..c_rows], m.vocab, 1000 + rep + c_rows + p0);
                    m.pp_timing.schedule = sched;
                    m.pp_timing.rep = rep;
                    m.pp_timing.warmup = rep < stage_warmup;
                    const logits = try dsv41.extendForBench(m, gpa, &st, rep_chunk[0..c_rows]);
                    try mlx.check(mlx.mlx_array_eval(logits));
                    _ = mlx.mlx_array_free(logits);
                    _ = mlx.mlx_clear_cache();
                }
            }
        }

        for (widths) |c_rows| {
            for (0..1 + diag_reps) |rep| {
                save.restore(&st);
                dsv41.dropAheadForBench(m);
                fillDeterministicTokens(rep_chunk[0..c_rows], m.vocab, 9000 + rep + c_rows + p0);
                m.pp_timing.schedule = .diag;
                m.pp_timing.rep = rep;
                m.pp_timing.warmup = rep == 0;
                const logits = try dsv41.extendForBench(m, gpa, &st, rep_chunk[0..c_rows]);
                try mlx.check(mlx.mlx_array_eval(logits));
                _ = mlx.mlx_array_free(logits);
                _ = mlx.mlx_clear_cache();
            }
        }

        save.restore(&st);
        m.pp_timing.schedule = .off;
        if (p0 == 32768) {
            if (peer_addr) |ep| {
                const sock = try tcp.bindAndConnectPeer(ep.ip4, ep.port, 5000);
                defer _ = std.c.close(sock.fd);
                try runOverlapBench(m, &st, &sock, gpa, &json_out, &widths, 2, stage_reps);
                tcp.sendDone(sock.fd);
                peer_done_sent = true;
            }
        }
    }
}

// ── Tests ──

const testing = std.testing;

test "pp_step1: seam payload sizes and release Plan.init KV export bytes match exact formulas" {
    try testing.expectEqual(@as(u64, 83_918_848), seamPayloadBytes(2048, false));
    try testing.expectEqual(@as(u64, 89_161_728), seamPayloadBytes(2048, true));
    try testing.expectEqual(@as(u64, 167_837_696), seamPayloadBytes(4096, false));
    try testing.expectEqual(@as(u64, 178_323_456), seamPayloadBytes(4096, true));
    try testing.expectEqual(@as(u64, 335_675_392), seamPayloadBytes(8192, false));
    try testing.expectEqual(@as(u64, 356_646_912), seamPayloadBytes(8192, true));

    const layout = releaseCacheLayout(0x1234);
    for ([_]usize{ 8192, 32768, 65536 }) |prompt_n| {
        const n_snap = prompt_n - 1;
        const plan = try cache_codec.Plan.init(layout, n_snap, 0);
        try testing.expectEqual(plan.loaded, expectedPlanExportBytes(n_snap));
    }
    try testing.expectEqual(@as(u64, 32_625_660), expectedPlanExportBytes(8191));
    try testing.expectEqual(@as(u64, 111_367_164), expectedPlanExportBytes(32767));
    try testing.expectEqual(@as(u64, 216_355_836), expectedPlanExportBytes(65535));

    const even_plan = try cache_codec.Plan.init(layout, 8192, 0);
    try testing.expectEqual(even_plan.loaded, expectedPlanExportBytes(8192));

    try testing.expectEqual(@as(u64, 83_882_240), nonStage0KvBytes(32767).rows_bytes);
    try testing.expectEqual(@as(u64, 167_768_320), nonStage0KvBytes(65535).rows_bytes);
}

test "pp_step1: planPpChunks enforces even non-final chunks and >= 384 final CED tail" {
    var buf: [32]usize = undefined;

    const single = try planPpChunks(500, 2048, &buf);
    try testing.expectEqualSlices(usize, &.{500}, single);

    const split_8k = try planPpChunks(8292, 8192, &buf);
    try testing.expectEqualSlices(usize, &.{ 7808, 484 }, split_8k);

    const split_32k = try planPpChunks(32868, 8192, &buf);
    try testing.expectEqualSlices(usize, &.{ 8192, 8192, 8192, 7808, 484 }, split_32k);

    for ([_]usize{ 2048, 4096, 8192 }) |c| {
        for ([_]usize{ 384, 385, 2048, 2049, 8191, 8192, 8292, 16383, 32767, 32868 }) |n_snap| {
            const chunks = try planPpChunks(n_snap, c, &buf);
            var sum: usize = 0;
            for (chunks, 0..) |len, idx| {
                sum += len;
                if (idx + 1 < chunks.len) {
                    try testing.expectEqual(@as(usize, 0), len % 2);
                } else {
                    try testing.expect(len >= CED_MIN_CHUNK and len <= c);
                }
            }
            try testing.expectEqual(n_snap, sum);
        }
    }
}

test "pp_step1: schedule tables, tbPeerIp4 routing, and JSON line schemas parse cleanly" {
    try testing.expectEqual(Schedule.A4, Schedule.parse("A4"));
    try testing.expectEqual(Schedule.B3, Schedule.parse("b3"));
    try testing.expectEqual(Schedule.diag, Schedule.parse("diag"));
    try testing.expectEqual(Schedule.off, Schedule.parse(null));
    try testing.expectEqual(Schedule.off, Schedule.parse("0"));

    for ([_]Schedule{ .A2, .A3, .A4, .B2, .B3, .B4 }) |s| {
        const list = s.stages();
        try testing.expectEqual(s.ppWidth(), list.len);
        try testing.expectEqual(@as(usize, 0), list[0].lo);
        try testing.expectEqual(@as(usize, 20), list[list.len - 1].hi);
        try testing.expect(list[list.len - 1].l20_comp);
        for (1..list.len) |k| try testing.expectEqual(list[k - 1].hi, list[k].lo);
    }

    try testing.expectEqual([4]u8{ 10, 254, 12, 1 }, try tbPeerIp4(2, 1));
    try testing.expectEqual([4]u8{ 10, 254, 12, 2 }, try tbPeerIp4(1, 2));
    try testing.expectEqual([4]u8{ 10, 254, 24, 4 }, try tbPeerIp4(2, 4));
    try testing.expectEqual([4]u8{ 127, 0, 0, 1 }, try tbPeerIp4(0, 0));

    var buf: [768]u8 = undefined;
    const stage_line = try formatStageJson(&buf, .A4, 0, Schedule.A4.stages()[0], 2048, 32768, 3, false, 150_000);
    var parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, std.mem.trim(u8, stage_line, "\n"), .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("stage_timing", parsed.value.object.get("kind").?.string);
    try testing.expectEqual(@as(i64, 4), parsed.value.object.get("pp_width").?.integer);
    try testing.expectEqual(@as(i64, 150_000), parsed.value.object.get("elapsed_us").?.integer);

    const hop_line = try formatTcpHopJson(&buf, "one_way", 2048, .mid_group, 89_161_728, 50, 28_000, 27_900, 27_500, 29_100, 600, 700, "en2", .{ 10, 254, 12, 2 }, .{ 10, 254, 12, 1 }, 9000, 4194304, 4194304);
    var parsed_hop = try std.json.parseFromSlice(std.json.Value, testing.allocator, std.mem.trim(u8, hop_line, "\n"), .{});
    defer parsed_hop.deinit();
    try testing.expectEqualStrings("tcp_hop", parsed_hop.value.object.get("kind").?.string);
    try testing.expectEqual(@as(i64, 9000), parsed_hop.value.object.get("mtu").?.integer);
}

test "pp_step1: cmdProbe rejects invalid invocation cleanly across target OS" {
    const io = std.Io.Threaded.global_single_threaded.io();
    if (comptime @import("builtin").os.tag == .windows) {
        try testing.expectError(error.UnsupportedPlatform, cmdProbe(io, testing.allocator, &.{}));
    } else {
        try testing.expectError(error.InvalidArgs, cmdProbe(io, testing.allocator, &.{}));
    }
}

test "pp_step1: live DSV41_PP_STAGE_TIMING A4, diag, and verifyKvExportSizes match stock logits and emit valid JSONL" {
    if (comptime @import("builtin").os.tag == .windows) return error.SkipZigTest;
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const gpa = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const v4fixture = @import("../deepseek_v4/fixture.zig");

    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const files = try v4fixture.v41Files(arena.allocator(), 7);

    var td = testing.tmpDir(.{});
    defer td.cleanup();
    for (files) |f| try td.dir.writeFile(io, .{ .sub_path = f.name, .data = f.bytes });

    var root_buf: [512]u8 = undefined;
    const root = root_buf[0..try td.dir.realPath(io, &root_buf)];
    var jsonl_buf: [600]u8 = undefined;
    const jsonl_path = try std.fmt.bufPrint(&jsonl_buf, "{s}/pp_stage_test.jsonl", .{root});

    const cfg = try loadPackConfig(io, gpa, root);
    var w = try dsv41.loadPack(io, gpa, root, false);
    defer w.deinit();
    const m = try dsv41.init(gpa, &cfg, &w, mlx.gpuStream(), .{});
    defer m.deinit();
    m.span = 8;
    m.ced_min_skip = 8;

    var jout = try JsonOut.openPath(jsonl_path);
    defer jout.close();
    try verifyKvExportSizes(m, gpa, &jout);

    var prompt: [32]u32 = undefined;
    fillDeterministicTokens(&prompt, m.vocab, 42);

    var st_off = try dsv41.initState(m, gpa);
    defer st_off.deinit();
    m.pp_timing.schedule = .off;
    const logits_off = try dsv41.extendForBench(m, gpa, &st_off, &prompt);
    defer _ = mlx.mlx_array_free(logits_off);
    try mlx.check(mlx.mlx_array_eval(logits_off));

    m.pp_timing.out = jout;
    defer m.pp_timing.out = .{};

    for ([_]Schedule{ .A4, .diag }) |sched| {
        var st_t = try dsv41.initState(m, gpa);
        defer st_t.deinit();
        m.pp_timing.schedule = sched;
        m.pp_timing.rep = 1;
        m.pp_timing.warmup = false;
        const logits_t = try dsv41.extendForBench(m, gpa, &st_t, &prompt);
        defer _ = mlx.mlx_array_free(logits_t);
        try mlx.check(mlx.mlx_array_eval(logits_t));

        const p_off = mlx.mlx_array_data_float32(logits_off) orelse return error.NoData;
        const p_t = mlx.mlx_array_data_float32(logits_t) orelse return error.NoData;
        const n_elem = mlx.mlx_array_size(logits_off);
        for (0..n_elem) |idx| {
            try testing.expectApproxEqAbs(p_off[idx], p_t[idx], 1e-4);
        }
    }
    m.pp_timing.schedule = .off;
    jout.close();

    const raw = try td.dir.readFileAlloc(io, "pp_stage_test.jsonl", gpa, .limited(1 << 20));
    defer gpa.free(raw);
    var n_kv: usize = 0;
    var n_stage: usize = 0;
    var n_tail: usize = 0;
    var n_diag: usize = 0;
    var it = std.mem.splitScalar(u8, raw, '\n');
    while (it.next()) |line| {
        const trimmed = std.mem.trim(u8, line, " \r\t");
        if (trimmed.len == 0) continue;
        var parsed = try std.json.parseFromSlice(std.json.Value, gpa, trimmed, .{});
        defer parsed.deinit();
        const kind = parsed.value.object.get("kind").?.string;
        if (std.mem.eql(u8, kind, "kv_export_check")) n_kv += 1;
        if (std.mem.eql(u8, kind, "stage_timing")) n_stage += 1;
        if (std.mem.eql(u8, kind, "tail_timing")) n_tail += 1;
        if (std.mem.eql(u8, kind, "layer_diag")) n_diag += 1;
    }
    try testing.expectEqual(@as(usize, 3), n_kv);
    try testing.expectEqual(@as(usize, 4), n_stage);
    try testing.expectEqual(@as(usize, 1), n_tail);
    try testing.expectEqual(@as(usize, 9), n_diag);
}

test "pp_step1: seedMidGroupSeamForBench activates mid-group compressed-KV gather and runOverlapBenchRange emits overlap JSONL" {
    if (comptime @import("builtin").os.tag == .windows) return error.SkipZigTest;
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const gpa = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const v4fixture = @import("../deepseek_v4/fixture.zig");

    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const files = try v4fixture.v41Files(arena.allocator(), 11);

    var td = testing.tmpDir(.{});
    defer td.cleanup();
    for (files) |f| try td.dir.writeFile(io, .{ .sub_path = f.name, .data = f.bytes });

    var root_buf: [512]u8 = undefined;
    const root = root_buf[0..try td.dir.realPath(io, &root_buf)];
    var jsonl_buf: [600]u8 = undefined;
    const jsonl_path = try std.fmt.bufPrint(&jsonl_buf, "{s}/pp_overlap_test.jsonl", .{root});

    const cfg = try loadPackConfig(io, gpa, root);
    var w = try dsv41.loadPack(io, gpa, root, false);
    defer w.deinit();
    const m = try dsv41.init(gpa, &cfg, &w, mlx.gpuStream(), .{});
    defer m.deinit();
    m.span = 8;

    var st = try dsv41.initState(m, gpa);
    defer st.deinit();
    var prefix: [16]u32 = undefined;
    fillDeterministicTokens(&prefix, m.vocab, 99);
    try advanceStateTo(m, gpa, &st, prefix.len, &prefix);
    try testing.expectEqual(@as(usize, 16), st.n);
    try testing.expectEqual(@as(usize, 8), st.layers[2].comp.?.used);

    var base_save = try StateSave.capture(gpa, &st);
    defer base_save.deinit();

    const c_rows: usize = 16;
    const n_spans: usize = 2;
    var init_xs = [_]A{ none, none };
    defer for (init_xs) |x| if (x.ctx != null) {
        _ = mlx.mlx_array_free(x);
    };
    var init_pres = [_]A{ none, none };
    defer for (init_pres) |x| if (x.ctx != null) {
        _ = mlx.mlx_array_free(x);
    };
    var init_topks = [_]A{ none, none };
    defer for (init_topks) |x| if (x.ctx != null) {
        _ = mlx.mlx_array_free(x);
    };

    try dsv41.seedMidGroupSeamForBench(m, gpa, &st, 3, 16, c_rows, m.span, &init_xs, &init_pres, &init_topks);
    try testing.expect(init_topks[0].ctx != null);
    try testing.expect(init_topks[1].ctx != null);
    try testing.expectEqual(@as(usize, 16), st.layers[2].comp.?.used);
    var seeded_save = try StateSave.capture(gpa, &st);
    defer seeded_save.deinit();

    const held_a = try gpa.alloc(A, n_spans * 4 + 2);
    defer {
        for (held_a) |x| if (x.ctx != null) {
            _ = mlx.mlx_array_free(x);
        };
        gpa.free(held_a);
    }
    @memset(held_a, none);

    const held_b = try gpa.alloc(A, n_spans * 4 + 2);
    defer {
        for (held_b) |x| if (x.ctx != null) {
            _ = mlx.mlx_array_free(x);
        };
        gpa.free(held_b);
    }
    @memset(held_b, none);

    seeded_save.restore(&st);
    var sp_no_topk = dsv41.spansOfForBench(held_a, n_spans, 0, m.span, c_rows, 16);
    sp_no_topk.whole = held_a[held_a.len - 2 ..][0..2];
    for (0..n_spans) |k| {
        dsv41.holdForBench(&sp_no_topk.xs[k], init_xs[k]);
        dsv41.holdForBench(&sp_no_topk.pres[k], init_pres[k]);
    }
    try dsv41.runLayersWithDrain(m, gpa, &st, &sp_no_topk, held_a, 3, 5, true);

    seeded_save.restore(&st);
    var sp_with_topk = dsv41.spansOfForBench(held_b, n_spans, 0, m.span, c_rows, 16);
    sp_with_topk.whole = held_b[held_b.len - 2 ..][0..2];
    for (0..n_spans) |k| {
        dsv41.holdForBench(&sp_with_topk.xs[k], init_xs[k]);
        dsv41.holdForBench(&sp_with_topk.pres[k], init_pres[k]);
        dsv41.holdForBench(&sp_with_topk.topks[k], init_topks[k]);
    }
    try dsv41.runLayersWithDrain(m, gpa, &st, &sp_with_topk, held_b, 3, 5, true);

    var x_no = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x_no);
    try mlx.check(mlx.mlx_astype(&x_no, sp_no_topk.xs[0], .float32, mlx.gpuStream()));
    var x_tk = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x_tk);
    try mlx.check(mlx.mlx_astype(&x_tk, sp_with_topk.xs[0], .float32, mlx.gpuStream()));
    try dsv41.evalAllForBench(&.{ x_no, x_tk });
    const p_no = mlx.mlx_array_data_float32(x_no) orelse return error.NoData;
    const p_tk = mlx.mlx_array_data_float32(x_tk) orelse return error.NoData;
    var max_diff: f32 = 0.0;
    for (0..mlx.mlx_array_size(x_no)) |idx| {
        max_diff = @max(max_diff, @abs(p_no[idx] - p_tk[idx]));
    }
    try testing.expect(max_diff > 1e-4);

    base_save.restore(&st);

    const srv = try tcp.listenTcp(.{ 127, 0, 0, 1 }, 0);
    defer _ = std.c.close(srv.fd);
    const OverlapSrv = struct {
        listen_fd: tcp.SocketFd,
        err: ?anyerror = null,
        fn run(ctx: *@This()) void {
            tcp.serveProbeListener(ctx.listen_fd, std.heap.c_allocator) catch |e| {
                ctx.err = e;
            };
        }
    };
    var octx: OverlapSrv = .{ .listen_fd = srv.fd };
    const sth = try std.Thread.spawn(.{ .stack_size = 1 << 20 }, OverlapSrv.run, .{&octx});
    defer sth.join();

    var jout = try JsonOut.openPath(jsonl_path);
    defer jout.close();
    {
        const sock = try tcp.bindAndConnectPeer(.{ 127, 0, 0, 1 }, srv.port, 3000);
        defer _ = std.c.close(sock.fd);
        try runOverlapBenchRange(m, &st, &sock, gpa, &jout, &.{c_rows}, 3, 6, 1, 2);
        tcp.sendDone(sock.fd);
    }
    jout.close();
    if (octx.err) |e| return e;

    try testing.expectEqual(@as(usize, 16), st.n);
    try testing.expectEqual(@as(usize, 8), st.layers[2].comp.?.used);

    const raw = try td.dir.readFileAlloc(io, "pp_overlap_test.jsonl", gpa, .limited(1 << 20));
    defer gpa.free(raw);
    var parsed = try std.json.parseFromSlice(std.json.Value, gpa, std.mem.trim(u8, raw, " \r\n\t"), .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("tcp_gpu_overlap", parsed.value.object.get("kind").?.string);
    try testing.expectEqual(@as(i64, 3), parsed.value.object.get("moe_lo").?.integer);
    try testing.expectEqual(@as(i64, 6), parsed.value.object.get("moe_hi").?.integer);
}
