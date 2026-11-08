//! DeepSeek-V4.1's decode roofline under a tensor split (`DSV41_DEC_UBENCH`): one rank's fused decode layers at the
//! per-rank shapes of N = 1, 2, 4 (heads 64/N, expert width 2304/N; `Shard.axis` cuts), each stage's time in place.
//!
//! A step through the trunk runs once per `dsv41.DecCut`: at a cut every layer stops after that stage and hands its
//! input stream on (plus zero times one value of the stage's output, so the next layer waits on it). A stage's cost
//! is the step's time at its cut less the time at the cut before, so the rows add up to the whole step, launch gaps
//! and the plain MLX ops between the kernels included. The fixture holds no Engram, so layers 1 and 14 run as plain
//! layers: their Engram add (no cut of its own) is in no row. Each step is timed three ways: the host building the graph,
//! the graph built first and then evaluated (the device's time, the encoding hidden under it), and the serving
//! loop's way (an async eval a layer, `runLayers`). The sums are an identity on the GPU stream (`DSV41_TP_SUM=bf16`'s
//! casts, no hand-off): this is the floor the 4-Studio group's sums-off arm measures.
//!
//! Lines start `[roofline]` as key=value pairs, one a stage, for a script to fold into the table.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const iou = @import("../../../sdk/util/io_util.zig");
const dsv41 = @import("../deepseek_v41.zig");
const decode = @import("decode.zig");
const fixture = @import("fixture.zig");
const moe_up = @import("moe_up.zig");
const attn = @import("attn.zig");

const A = mlx.mlx_array;
const none: A = .{ .ctx = null };
const Cut = dsv41.DecCut;

/// The cuts in the layer's order, the whole layer last.
const cuts = [_]Cut{ .start, .in_proj, .wq_b, .win, .comp, .idx, .attn, .wo_a, .wo_b, .sum_attn, .ffn_pre, .moe_up, .moe_down, .none };

fn nowUs() i64 {
    return iou.nowUsMonotonic(std.Io.Threaded.global_single_threaded.io());
}

fn ci(v: usize) c_int {
    return @intCast(v);
}

fn nbytes(a: A) f64 {
    if (a.ctx == null) return 0;
    return @floatFromInt(mlx.mlx_array_size(a) * mlx.mlx_array_itemsize(a));
}

fn linBytes(l: dsv41.Lin) f64 {
    return nbytes(l.w) + nbytes(l.s) + nbytes(l.b);
}

fn bankBytes(b: dsv41.Bank) f64 {
    return nbytes(b.w) + nbytes(b.s) + nbytes(b.b);
}

/// The identity collective: the tensor sum's casts run, nothing crosses a wire.
fn identitySum(x: A) anyerror!A {
    var out = mlx.mlx_array_new();
    _ = mlx.mlx_array_set(&out, x);
    return out;
}

// ── the machine ──

/// Every thread XORs the 16-byte words at its stride and writes one word: a read with no arithmetic to hide behind.
const READ_SRC =
    \\const uint gid = thread_position_in_grid.x;
    \\const uint n = threads_per_grid.x;
    \\const device uint4* p = (const device uint4*)inp;
    \\uint4 acc = uint4(0);
    \\for (uint i = gid; i < WORDS; i += n) acc ^= p[i];
    \\out[gid] = acc.x ^ acc.y ^ acc.z ^ acc.w;
;

/// What this GPU streams from DRAM: the read kernel over 2 GiB, MLX's sum over it (read) and a copy-with-add of it
/// (read + write), best of 9, GB/s.
fn bandwidth() ![3]f64 {
    const s = mlx.gpuStream();
    const shape = [_]c_int{1 << 29};
    var key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_random_key(&key, 7));
    const lo = mlx.mlx_array_new_float(0);
    defer _ = mlx.mlx_array_free(lo);
    const hi = mlx.mlx_array_new_float(1);
    defer _ = mlx.mlx_array_free(hi);
    var x = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x);
    try mlx.check(mlx.mlx_random_uniform(&x, lo, hi, &shape, 1, .float32, key, s));
    var xu = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(xu);
    try mlx.check(mlx.mlx_view(&xu, x, .uint32, s));
    try mlx.check(mlx.mlx_array_eval(xu));
    const kern = try dsv41.buildKernel("dsv41_roofline_read", &.{"inp"}, &.{"out"}, READ_SRC, "");
    defer _ = mlx.mlx_fast_metal_kernel_free(kern);
    const threads: c_int = 1 << 18;
    const bytes = nbytes(x);
    var best = [3]f64{ std.math.inf(f64), std.math.inf(f64), std.math.inf(f64) };
    for (0..9) |_| for (&best, 0..) |*b, which| {
        var y = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(y);
        const t0 = nowUs();
        switch (which) {
            0 => {
                const cfg = mlx.mlx_fast_metal_kernel_config_new();
                defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
                try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{threads}, 1, .uint32));
                try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, threads, 1, 1));
                try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 256, 1, 1));
                try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "WORDS", @intCast(@divExact(shape[0], 4))));
                const iv = mlx.mlx_vector_array_new_data(&[_]A{xu}, 1);
                defer _ = mlx.mlx_vector_array_free(iv);
                var ov = mlx.mlx_vector_array_new();
                defer _ = mlx.mlx_vector_array_free(ov);
                try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, kern, iv, cfg, s));
                try mlx.check(mlx.mlx_vector_array_get(&y, ov, 0));
            },
            1 => try mlx.check(mlx.mlx_sum(&y, x, false, s)),
            else => try mlx.check(mlx.mlx_add(&y, x, hi, s)),
        }
        try mlx.check(mlx.mlx_array_eval(y));
        b.* = @min(b.*, @as(f64, @floatFromInt(nowUs() - t0)));
    };
    return .{ bytes / best[0] / 1e3, bytes / best[1] / 1e3, 2 * bytes / best[2] / 1e3 };
}

/// One tiny op on the last one's output: the device's time a dependent launch, the floor under any stage.
const Tiny = struct {
    sc: *dsv41.Scope,
    prev: A,
    zero: A,
    pub fn step(c: *Tiny, _: usize) !A {
        c.prev = try c.sc.add(c.prev, c.zero);
        return c.prev;
    }
};

fn launchFloorUs(gpa: std.mem.Allocator) !f64 {
    var sc = dsv41.Scope.init(gpa, mlx.gpuStream());
    defer sc.deinit();
    const zero = try sc.f(0);
    var t: Tiny = .{ .sc = &sc, .prev = try sc.f(1), .zero = zero };
    var best: f64 = std.math.inf(f64);
    for (0..3) |_| best = @min(best, try moe_up.deviceUs(&t, 256));
    return best;
}

// ── a step through the fixture's layers ──

const Times = struct { build: f64, eval: f64, pipe: f64 };

const Run = struct {
    gpa: std.mem.Allocator,
    fx: *fixture.Fixture,
    x: A,
    pre: A,
    rows: usize,
    base: usize,
    k: usize = 0,

    /// Steps a device timing chains before it waits: one step's sync would leave the GPU idle between steps.
    const chain = 4;

    /// One step's graph at `base` or `base + 1` (alternating, so a ratio-2 compressor commits on half the steps as in
    /// a decode), fed `x` and `pre`; `pipelined` hands each layer to the GPU as it is built (`runLayers`).
    fn build(r: *Run, sc: *dsv41.Scope, x0: A, pre0: A, pipelined: bool) ![2]A {
        const m = r.fx.m;
        const p = m.pair.?;
        const st = &m.dec_state.?;
        const p0 = r.base + r.k % 2;
        r.k += 1;
        var ev: std.ArrayList(A) = .empty;
        defer ev.deinit(r.gpa);
        var x = x0;
        var pre = pre0;
        var sh: dsv41.Shared = .{};
        for (p.lo..p.hi) |li| {
            x = try dsv41.block(sc, m, st, li, x, &pre, p0, &sh);
            if (!pipelined) continue;
            ev.clearRetainingCapacity();
            try ev.appendSlice(r.gpa, &.{ x, pre });
            if (sh.topk.ctx != null) try ev.append(r.gpa, sh.topk);
            try dsv41.appendLayerState(r.gpa, &ev, &st.layers[li]);
            const vec = mlx.mlx_vector_array_new_data(ev.items.ptr, ev.items.len);
            defer _ = mlx.mlx_vector_array_free(vec);
            try mlx.check(mlx.mlx_async_eval(vec));
        }
        st.n = p0 + r.rows;
        return .{ x, pre };
    }

    fn finish(r: *Run, x: A, pre: A) !void {
        const m = r.fx.m;
        const p = m.pair.?;
        var ev: std.ArrayList(A) = .empty;
        defer ev.deinit(r.gpa);
        try ev.appendSlice(r.gpa, &.{ x, pre });
        for (m.dec_state.?.layers[p.lo..p.hi]) |*ls| try dsv41.appendLayerState(r.gpa, &ev, ls);
        try dsv41.evalAll(ev.items);
    }

    /// `chain` steps built first, each fed the last one's stream (as a zero-weighted dependency on the same input, so
    /// no value drifts), then evaluated: host microseconds a step to build, and to evaluate.
    fn deferred(r: *Run) ![2]f64 {
        var sc = dsv41.Scope.init(r.gpa, r.fx.m.s);
        defer sc.deinit();
        var x = r.x;
        var pre = r.pre;
        const t0 = nowUs();
        for (0..chain) |_| {
            const out = try r.build(&sc, x, pre, false);
            const first = try sc.astype(try sc.slice(try sc.reshape(out[0], &.{-1}), 0, 0, 1), .float32);
            x = try sc.astype(try sc.add(r.x, try sc.mul(first, try sc.f(0))), mlx.mlx_array_dtype(r.x));
            pre = out[1];
        }
        const t1 = nowUs();
        try r.finish(x, pre);
        const t2 = nowUs();
        return .{ @as(f64, @floatFromInt(t1 - t0)) / chain, @as(f64, @floatFromInt(t2 - t1)) / chain };
    }

    /// One step the serving loop's way: built a layer at a time onto the GPU, then waited on.
    fn served(r: *Run) !f64 {
        var sc = dsv41.Scope.init(r.gpa, r.fx.m.s);
        defer sc.deinit();
        const t0 = nowUs();
        const out = try r.build(&sc, r.x, r.pre, true);
        try r.finish(out[0], out[1]);
        return @floatFromInt(nowUs() - t0);
    }

    fn median(v: []f64) f64 {
        std.mem.sort(f64, v, {}, std.sort.asc(f64));
        return v[v.len / 2];
    }

    /// The step's times with every layer stopped at `cut` (medians of 9 after 3 warm-ups each way), and the layers
    /// that reached it.
    fn at(r: *Run, cut: Cut) !struct { t: Times, hits: usize } {
        dsv41.dec_cut = cut;
        defer dsv41.dec_cut = .none;
        var b: [9]f64 = undefined;
        var e: [9]f64 = undefined;
        var pp: [9]f64 = undefined;
        for (0..3) |_| _ = try r.deferred();
        dsv41.dec_cut_hits = 0;
        for (&b, &e) |*bi, *ei| {
            const v = try r.deferred();
            bi.* = v[0];
            ei.* = v[1];
        }
        const hits = dsv41.dec_cut_hits / (b.len * chain);
        for (0..3) |_| _ = try r.served();
        for (&pp) |*pi| pi.* = try r.served();
        return .{ .t = .{ .build = median(&b), .eval = median(&e), .pipe = median(&pp) }, .hits = hits };
    }
};

/// Bytes stage `cut` reads a step over layer `li` at `rows` rows and context `ctx` (weights, the keys it scans, the
/// activations past a few KB); routed experts as the expected distinct picks of `rows` rows of 6 among E.
fn stageBytes(m: *const dsv41.Dsv41Model, st: *const dsv41.State, li: usize, cut: Cut, rows: usize, ctx: usize) f64 {
    const ly = &m.layers[li];
    const c = &m.cfg;
    const R: f64 = @floatFromInt(rows);
    const hd: f64 = @floatFromInt(c.head_dim);
    const H: f64 = @floatFromInt(c.num_attention_heads);
    const E: f64 = @floatFromInt(c.num_experts);
    const picks = E * (1 - std.math.pow(f64, 1 - 1 / E, R * @as(f64, @floatFromInt(ly.top_k))));
    const clen: f64 = if (ly.ratio > 0) @floatFromInt((ctx + rows) / ly.ratio) else 0;
    _ = st;
    return switch (cut) {
        .in_proj => linBytes(ly.q_kv.?) + nbytes(ly.hc_attn.fn_w) + (if (ly.idx) |iw| linBytes(iw.weights_proj) else 0) +
            (if (ly.comp) |cw| linBytes(cw.wkv) else 0) + R * @as(f64, @floatFromInt(c.dsv4_hc_mult * c.hidden_size)) * 2,
        .wq_b => linBytes(ly.wq_b) + (if (ly.idx) |iw| linBytes(iw.wq_b) else 0),
        // The index keys the scorer reads (bf16, index_head_dim wide), on the layers that own an indexer.
        .idx => if (ly.idx != null and ly.ratio > 0) clen * @as(f64, @floatFromInt(c.dsv4_index_head_dim)) * 2 else 0,
        // The window and the picked compressed rows (bf16), the queries.
        .attn => (@as(f64, @floatFromInt(c.sliding_window)) + (if (ly.ratio > 0) @min(clen, @as(f64, @floatFromInt(c.dsv4_index_topk))) else 0)) * hd * 2 + R * H * hd * 2,
        .wo_a => linBytes(ly.wo_a),
        .wo_b => linBytes(ly.wo_b),
        .ffn_pre => nbytes(ly.hc_ffn.fn_w) + nbytes(ly.gate_w) + nbytes(ly.gate_bias),
        .moe_up => (if (ly.shared_gu) |g| linBytes(g) else linBytes(ly.shared[0]) + linBytes(ly.shared[1])) +
            (if (ly.experts.gu) |g| bankBytes(g) / E * picks else 0),
        .moe_down => linBytes(ly.shared[2]) + bankBytes(ly.experts.down) / E * picks,
        else => 0,
    };
}

/// Every cut's times over `fx`'s layers at `rows` rows, one `[roofline]` line a stage.
fn table(gpa: std.mem.Allocator, fx: *fixture.Fixture, n: usize, rows: usize, ctx: usize, tag: []const u8) !void {
    const m = fx.m;
    const p = m.pair.?;
    const x = try fx.stream(rows);
    defer _ = mlx.mlx_array_free(x);
    const pre = try fx.cpre(rows);
    defer _ = mlx.mlx_array_free(pre);
    var r: Run = .{ .gpa = gpa, .fx = fx, .x = x, .pre = pre, .rows = rows, .base = ctx };
    var prev: Times = .{ .build = 0, .eval = 0, .pipe = 0 };
    for (cuts) |cut| {
        const got = try r.at(cut);
        var bytes: f64 = 0;
        for (p.lo..p.hi) |li| bytes += stageBytes(m, &m.dec_state.?, li, cut, rows, ctx);
        std.debug.print("[roofline] tag={s} N={d} R={d} ctx={d} layers={d}-{d} stage={s} hits={d} build_us={d:.1} eval_us={d:.1} pipe_us={d:.1} bytes={d:.0}\n", .{
            tag,                         n,                           rows,                       ctx, p.lo, p.hi, if (cut == .none) "sum_moe+hc_expand" else @tagName(cut), got.hits,
            got.t.build - prev.build, got.t.eval - prev.eval, got.t.pipe - prev.pipe, bytes,
        });
        prev = got.t;
    }
    std.debug.print("[roofline] tag={s} N={d} R={d} ctx={d} layers={d}-{d} stage=TOTAL hits=0 build_us={d:.1} eval_us={d:.1} pipe_us={d:.1} bytes=0\n", .{ tag, n, rows, ctx, p.lo, p.hi, prev.build, prev.eval, prev.pipe });
}

fn rankOpts(n: usize, lo: usize, hi: usize, ctx: usize, experts: u32) fixture.Opts {
    return .{ .lo = lo, .hi = hi, .experts = experts, .ctx = ctx, .fused = true, .experts_fp4 = true, .tp = .{ .rank = 0, .size = @intCast(n) } };
}

/// The arms a table runs over the same fixture: `old` the attention at its 8-split cap (`DSV41_ATTN_SPLITS=0`), `new`
/// the narrow rule (`attn.narrowCap`).
const Arm = enum { old, new };

fn armOn(a: Arm) void {
    attn.setNarrow(if (a == .old) 0 else null);
}

/// Every arm's table over `fx` at both row counts.
fn tables(gpa: std.mem.Allocator, fx: *fixture.Fixture, n: usize, ctx: usize, comptime tag: []const u8) !void {
    defer attn.setNarrow(null);
    for ([_]usize{ 1, decode.max_rows }) |rows| for ([_]Arm{ .old, .new }) |a| {
        armOn(a);
        try table(gpa, fx, n, rows, ctx, if (a == .old) tag ++ "-old" else tag ++ "-new");
    };
}

/// `DSV41_ROOFLINE_N` (default "1,2,4"): the rank counts to run.
fn wantN(n: usize) bool {
    const raw = std.c.getenv("DSV41_ROOFLINE_N") orelse return true;
    var it = std.mem.tokenizeScalar(u8, std.mem.span(raw), ',');
    while (it.next()) |t| if ((std.fmt.parseInt(usize, t, 10) catch 0) == n) return true;
    return false;
}

test "dsv41 dec roofline: each stage of a tensor-split decode layer in place, N = 1 2 4, 1 and 6 rows (DSV41_DEC_UBENCH)" {
    if (!decode.ubenchOn() or !mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const gpa = std.testing.allocator;
    dsv41.clusterHooks().tp_sum = &identitySum;
    defer dsv41.clusterHooks().tp_sum = null;
    dsv41.tp_sum_mode = .bf16;
    defer dsv41.tp_sum_mode = null;

    const bw = try bandwidth();
    const lf = try launchFloorUs(gpa);
    std.debug.print("[roofline] machine read_gbs={d:.0} sum_gbs={d:.0} copy_gbs={d:.0} launch_us={d:.2}\n", .{ bw[0], bw[1], bw[2], lf });

    // The trunk's two halves, each a fixture: [0, 20) holds the ratio-0 pair and the ratio-2 layers (index and KV
    // sources 2, 8, 14), [20, 40) the ratio-1 layers (KV source 20, index sources 20..36 by 4). A half holds ~36 GB
    // of experts: the release's 384 at four ranks, fewer at wider ranks (the gate reads E rows; see the report).
    // A smoke run: `DSV41_ROOFLINE_SMOKE=1` takes layers [20, 22) over 8 experts and skips the long context.
    const smoke = std.c.getenv("DSV41_ROOFLINE_SMOKE") != null;
    // `DSV41_ROOFLINE_LAYERS=lo-hi` runs that one range instead of the trunk's halves.
    var one: [1][2]usize = .{.{ 20, 22 }};
    if (std.c.getenv("DSV41_ROOFLINE_LAYERS")) |raw| {
        var it = std.mem.tokenizeScalar(u8, std.mem.span(raw), '-');
        one[0][0] = try std.fmt.parseInt(usize, it.next() orelse "20", 10);
        one[0][1] = try std.fmt.parseInt(usize, it.next() orelse "22", 10);
    }
    const halves: []const [2]usize = if (smoke or std.c.getenv("DSV41_ROOFLINE_LAYERS") != null) &one else &.{ .{ 0, 20 }, .{ 20, 40 } };
    const ctx = 2048;
    for ([_]usize{ 1, 2, 4 }) |n| {
        if (!wantN(n)) continue;
        const experts: u32 = if (smoke) 8 else @intCast(96 * n);
        for (halves) |half| {
            const t0 = nowUs();
            const fx = fixture.Fixture.init(gpa, rankOpts(n, half[0], half[1], ctx, experts)) catch |e| {
                std.debug.print("[roofline] fixture N={d} layers={d}-{d} experts={d} failed: {s}\n", .{ n, half[0], half[1], experts, @errorName(e) });
                return e;
            };
            defer fx.deinit();
            std.debug.print("[roofline] fixture N={d} layers={d}-{d} experts={d} built in {d:.0} s\n", .{ n, half[0], half[1], experts, @as(f64, @floatFromInt(nowUs() - t0)) / 1e6 });
            try tables(gpa, fx, n, ctx, "trunk");
        }
    }
    // Long context: the stages that scan keys (indexer, compressor, attention) at 2k and 32k over a KV + index source
    // and the plain layer after it (weights in the system cache: read only the context-driven rows).
    if (smoke or std.c.getenv("DSV41_ROOFLINE_NO_CTX") != null) return;
    for ([_]usize{ 1, 4 }) |n| {
        if (!wantN(n)) continue;
        for ([_]usize{ 2048, 32768 }) |c| {
            const fx = try fixture.Fixture.init(gpa, rankOpts(n, 20, 22, c, 8));
            defer fx.deinit();
            try tables(gpa, fx, n, c, "ctx");
        }
    }
}
