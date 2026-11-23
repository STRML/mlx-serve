//! DeepSeek-V4.1's decode kernels, one file a kernel (`~/plans/dsv41-decode-kernels-spec.md`): a layer's 1..6 rows in
//! about ten dependent launches, each at its weights' byte floor. What their tests and ubenches share lives here.

const std = @import("std");
const env = @import("../../../env.zig");
const mlx = @import("../../../mlx/ffi.zig");
const device = @import("../../../mlx/device.zig");
const kc = @import("../../../sdk/testing/kernel_contract/root.zig");

pub const fixture = @import("fixture.zig");
pub const in_proj = @import("in_proj.zig");
pub const wq_b = @import("wq_b.zig");
pub const idx_score = @import("idx_score.zig");
pub const topk = @import("topk.zig");
pub const attn = @import("attn.zig");
pub const wo_a = @import("wo_a.zig");
pub const ffn_in = @import("ffn_in.zig");
pub const moe_up = @import("moe_up.zig");
pub const moe_fp4 = @import("moe_fp4.zig");
pub const moe_down = @import("moe_down.zig");

const dsv41 = @import("../deepseek_v41.zig");

const A = mlx.mlx_array;

/// The most rows a decode kernel serves in one launch: a plain step, or a DSpark verify block.
pub const max_rows = 6;

/// What this machine streams from memory in practice (a plain reduction on the M5 Ultra), bytes a microsecond.
pub const stream_bytes_per_us: f64 = 1.065e6;

pub fn register(set: *kc.Set) !void {
    inline for (.{ in_proj, wq_b, idx_score, topk, attn, wo_a, ffn_in, moe_up, moe_fp4, moe_down }) |k| try k.register(set);
}

/// Root-mean-square of `got - want` over that of `want`.
pub fn relRms(got: []const f32, want: []const f32) f64 {
    var e: f64 = 0;
    var t: f64 = 0;
    for (got, want) |g, w| {
        const d = @as(f64, g) - @as(f64, w);
        e += d * d;
        t += @as(f64, w) * @as(f64, w);
    }
    return @sqrt(e / @max(t, 1e-300));
}

/// The fusion bar: `new` is no further from `truth` than today's `chain` (1.25x, plus 1e-4 of truth's scale).
pub fn expectNoWorse(label: []const u8, new: []const f32, chain: []const f32, truth: []const f32) !void {
    const en = relRms(new, truth);
    const ec = relRms(chain, truth);
    std.debug.print("[dec] {s}: new-truth {e:.2}, chain-truth {e:.2} (rel rms)\n", .{ label, en, ec });
    for (new) |v| if (!std.math.isFinite(v)) return error.NonFinite;
    if (en > 1.25 * ec + 1e-4) return error.FurtherFromTruth;
}

/// The ubenches run only with `DSV41_DEC_UBENCH` set (under `~/claude-tmp/dsdec/with.sh gpu`).
pub fn ubenchOn() bool {
    return env.get(.DSV41_DEC_UBENCH) != null;
}

/// Microseconds a step of a dependent chain: `ctx.step(i)` builds step i's graph (fed by step i - 1's output) and
/// returns the array the chain ends on; `n` steps evaluated at once, median of 7 chains after 2 warm-up chains.
pub fn chainUs(ctx: anytype, n: usize) !f64 {
    const io = std.Io.Threaded.global_single_threaded.io();
    const iou = @import("../../../sdk/util/io_util.zig");
    var times: [7]f64 = undefined;
    for (0..2 + times.len) |rep| {
        const t0 = iou.nowUsMonotonic(io);
        var last: A = .{ .ctx = null };
        for (0..n) |i| last = try ctx.step(i);
        try mlx.check(mlx.mlx_array_eval(last));
        const us = @as(f64, @floatFromInt(iou.nowUsMonotonic(io) - t0)) / @as(f64, @floatFromInt(n));
        if (rep >= 2) times[rep - 2] = us;
    }
    std.mem.sort(f64, &times, {}, std.sort.asc(f64));
    return times[times.len / 2];
}

/// The ubench line every kernel prints: its time, achieved bandwidth and floor, and the chain it replaces.
pub fn report(name: []const u8, rows: usize, us: f64, bytes: f64, chain_us: f64) void {
    const floor = bytes / stream_bytes_per_us;
    std.debug.print("[dec-ubench] {s} R={d}: {d:.1} us, {d:.0} GB/s ({d:.0}% of floor {d:.1} us), replaced chain {d:.1} us\n", .{ name, rows, us, bytes / us / 1e3, 100 * floor / us, floor, chain_us });
}

test "dsv41 dec layers: the fused kernels decode no further from an f32 model than today's chain over seeds, one row and a verify block" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const gpa = std.testing.allocator;
    // Ratio 1 (KV source, owner, candidate source, an index non-owner), ratio 2 (KV source pooling pairs), ratio 0.
    for ([_][2]usize{ .{ 20, 25 }, .{ 2, 4 }, .{ 0, 1 } }) |range| try layersAgree(gpa, range[0], range[1], false);
}

test "dsv41 dec layers: the fused kernels over mxfp4 experts decode no further from an f32 model than today's chain" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    // The mxfp4 kernels build only without NAX (`Dec.moe_fp4`): with NAX this compares the NAX attention against the
    // chain at a bf16 near-tie instead. The kernel contract checks the mxfp4 kernels on every GPU.
    if (device.naxAvailable()) return error.SkipZigTest;
    try layersAgree(std.testing.allocator, 20, 23, true);
}

fn layersAgree(gpa: std.mem.Allocator, lo: usize, hi: usize, fp4: bool) !void {
    // One seed lands up to 1.6x either side of the chain (five random layers amplify any rounding; under TF32 the f32
    // model itself moves 8% at the verify step), so the bar is each step's median over seeds.
    const seeds = [_]u64{ 9, 1, 2, 3, 4, 5, 6 };
    const steps = [_]usize{ 1, 1, 1, max_rows, 1 };
    var ratio: [steps.len][seeds.len]f64 = undefined;
    for (seeds, 0..) |seed, si| {
        // Same seed, so the same weights, prefill and step inputs: the fused kernels, today's chain, and the f32 model.
        var fxs: [3]*fixture.Fixture = undefined;
        var made: usize = 0;
        defer for (fxs[0..made]) |fx| fx.deinit();
        for (&fxs, [_]mlx.mlx_dtype{ .bfloat16, .bfloat16, .float32 }, [_]bool{ true, false, false }) |*fx, act, fused| {
            fx.* = try fixture.Fixture.init(gpa, .{ .lo = lo, .hi = hi, .ctx = 601, .seed = seed, .act = act, .fused = fused, .experts_fp4 = fp4 });
            made += 1;
        }
        var p0: usize = 601;
        for (steps, 0..) |rows, k| {
            var got: [3][]f32 = undefined;
            for (fxs, &got) |fx, *g| {
                const x = try fx.stream(rows);
                defer _ = mlx.mlx_array_free(x);
                const pre = try fx.cpre(rows);
                defer _ = mlx.mlx_array_free(pre);
                const out = try dsv41.workerStep(fx.m, gpa, x, pre, p0, false, false);
                defer for (out) |o| {
                    _ = mlx.mlx_array_free(o);
                };
                g.* = try dsv41.hostF32(out[0]);
            }
            defer for (got) |g| gpa.free(g);
            for (got[0]) |v| if (!std.math.isFinite(v)) return error.NonFinite;
            ratio[k][si] = relRms(got[0], got[2]) / @max(relRms(got[1], got[2]), 1e-4);
            p0 += rows;
        }
    }
    for (&ratio, steps) |*r, rows| {
        std.mem.sort(f64, r, {}, std.sort.asc(f64));
        std.debug.print("[dec] layers {d}..{d}, {d} rows: fused/chain distance to the f32 model, median {d:.3} (min {d:.2}, max {d:.2})\n", .{ lo, hi, rows, r[seeds.len / 2], r[0], r[seeds.len - 1] });
        if (r[seeds.len / 2] > 1.15) return error.FurtherFromTruth;
    }
}

test "dsv41 dec layers ubench: a decode step through layers 20..40, fused vs today's chain, 1 and 6 rows" {
    if (!ubenchOn() or !mlx.metalKernelsAvailable() or mlx.virtualGpu()) return error.SkipZigTest;
    const gpa = std.testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const iou = @import("../../../sdk/util/io_util.zig");
    // A fixture an arm: the compiled `.ffn` region keeps the MoE kernels of its first trace.
    for ([_]bool{ true, false }) |fused| {
        // 64 experts a layer: 20 layers hold ~20 GB, so a step's weights come from DRAM, not the SLC.
        const fx = try fixture.Fixture.init(gpa, .{ .lo = 20, .hi = 40, .experts = 64, .ctx = 4096, .fused = fused });
        defer fx.deinit();
        var p0: usize = 4096;
        for ([_]usize{ 1, max_rows }) |rows| {
            const x = try fx.stream(rows);
            defer _ = mlx.mlx_array_free(x);
            const pre = try fx.cpre(rows);
            defer _ = mlx.mlx_array_free(pre);
            var times: [9]f64 = undefined;
            for (0..3 + times.len) |rep| {
                const t0 = iou.nowUsMonotonic(io);
                const out = try dsv41.workerStep(fx.m, gpa, x, pre, p0, false, false);
                for (out) |o| _ = mlx.mlx_array_free(o);
                if (rep >= 3) times[rep - 3] = @floatFromInt(iou.nowUsMonotonic(io) - t0);
                p0 += rows;
            }
            std.mem.sort(f64, &times, {}, std.sort.asc(f64));
            std.debug.print("[dec-ubench] layers 20..40 R={d} {s}: {d:.1} us a layer (median step {d:.0} us)\n", .{ rows, if (fused) "fused" else "chain", times[times.len / 2] / 20, times[times.len / 2] });
        }
    }
}

test "dsv41 wide: hoisted indexer and compress match per-span" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const gpa = std.testing.allocator;
    const ctx: usize = 1024;
    const L: usize = 256;
    for ([_][2]usize{ .{ 20, 24 }, .{ 2, 5 } }) |range| {
        const lo = range[0];
        const hi = range[1];
        for ([_]u64{ 1, 2, 3 }) |seed| {
            var fxs: [3]*fixture.Fixture = undefined;
            var made: usize = 0;
            defer for (fxs[0..made]) |fx| fx.deinit();
            for (&fxs, [_]mlx.mlx_dtype{ .bfloat16, .bfloat16, .float32 }, [_]usize{ 64, L, L }) |*fx, act, sp| {
                fx.* = try fixture.Fixture.init(gpa, .{ .lo = lo, .hi = hi, .experts = 8, .ctx = ctx, .seed = seed, .act = act, .fused = false });
                fx.*.m.span = sp;
                made += 1;
            }
            var got: [3][]f32 = undefined;
            var n_got: usize = 0;
            defer for (got[0..n_got]) |g| gpa.free(g);
            for (fxs, &got) |fx, *g| {
                const x = try fx.stream(L);
                defer _ = mlx.mlx_array_free(x);
                const pre = try fx.cpre(L);
                defer _ = mlx.mlx_array_free(pre);
                const out = try dsv41.workerStep(fx.m, gpa, x, pre, ctx, false, false);
                defer for (out) |o| {
                    _ = mlx.mlx_array_free(o);
                };
                g.* = try dsv41.hostF32(out[0]);
                n_got += 1;
            }
            for (got) |slice| for (slice) |v| if (!std.math.isFinite(v)) return error.NonFinite;
            // Cache rows (comp + idx_k) produced by hoisted compress vs single-span compress, with nb > topk (512).
            var sc = dsv41.Scope.init(gpa, mlx.gpuStream());
            defer sc.deinit();
            for (lo..hi) |li| {
                const r = fxs[0].m.layers[li].ratio;
                if (r == 0) continue;
                const clen = (ctx + L) / r;
                if (clen <= fxs[0].m.cfg.dsv4_index_topk) return error.SelectionDidNotDiscriminate;
                const s0 = fxs[0].m.dec_state.?.layers[li];
                const s1 = fxs[1].m.dec_state.?.layers[li];
                for ([_]?@TypeOf(s0.comp.?){ s0.comp, s0.idx_k }, [_]?@TypeOf(s1.comp.?){ s1.comp, s1.idx_k }) |r0_opt, r1_opt| {
                    const r0 = r0_opt orelse continue;
                    const r1 = r1_opt orelse continue;
                    const k0 = try dsv41.hostF32(try sc.keep(try r0.view(clen, sc.s)));
                    defer gpa.free(k0);
                    const k1 = try dsv41.hostF32(try sc.keep(try r1.view(clen, sc.s)));
                    defer gpa.free(k1);
                    for (k0) |v| if (!std.math.isFinite(v)) return error.NonFinite;
                    for (k1) |v| if (!std.math.isFinite(v)) return error.NonFinite;
                    try std.testing.expectApproxEqAbs(@as(f64, 0), relRms(k0, k1), 1e-3);
                }
            }
            const ratio = relRms(got[0], got[2]) / @max(relRms(got[1], got[2]), 1e-4);
            std.debug.print("[wide] layers {d}..{d} seed={d}: wide/single distance to f32 = {d:.3}\n", .{ lo, hi, seed, ratio });
            if (ratio > 1.15) return error.FurtherFromTruth;
        }
    }
}

test "dsv41 wide ubench: engram gate, wide engram hoist, and indexer/comp hoist (DSV41_DEC_UBENCH)" {
    if (!ubenchOn() or !mlx.metalKernelsAvailable() or mlx.virtualGpu()) return error.SkipZigTest;
    const gpa = std.testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const iou = @import("../../../sdk/util/io_util.zig");
    const gpu = mlx.gpuStream();

    // 1. Vectorized engramGate vs engramGateOps across decode (L=1), single span (L=256), and chunk (L=2048).
    var key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_random_key(&key, 31));
    var gm: dsv41.Dsv41Model = undefined;
    gm.cfg = .{};
    gm.cfg.dsv4_hc_mult = 4;
    gm.cfg.hidden_size = 5120;
    gm.act = .bfloat16;
    gm.engram_kernel = try dsv41.buildKernel("dsv41_engram", &dsv41.ENGRAM_INS, &.{"out"}, dsv41.ENGRAM_SOURCE, "");
    defer _ = mlx.mlx_fast_metal_kernel_free(gm.engram_kernel);

    const D: c_int = 5120;
    const HC: c_int = 4;
    for ([_]usize{ 1, 256, 2048 }) |L| {
        var gsc = dsv41.Scope.init(gpa, gpu);
        defer gsc.deinit();
        var x0 = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_random_normal(&x0, &[_]c_int{ @intCast(L), HC, D }, 3, .bfloat16, 0, 1, key, gpu));
        x0 = try gsc.keep(x0);
        var kv = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_random_normal(&kv, &[_]c_int{ @intCast(L), (HC + 1) * D }, 2, .bfloat16, 0, 1, key, gpu));
        kv = try gsc.keep(kv);
        var qk = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_random_normal(&qk, &[_]c_int{ HC, D }, 2, .float32, 0, 1, key, gpu));
        qk = try gsc.keep(qk);
        const steps: usize = if (L == 1) 16 else if (L == 256) 8 else 4;
        var ops_med: [2]f64 = undefined;
        var kern_med: [2]f64 = undefined;
        // ABBA: Ops, Gate, Gate, Ops
        for ([_]bool{ false, true, true, false }, 0..) |use_kern, leg| {
            var times: [5]f64 = undefined;
            for (0..2 + times.len) |rep| {
                const t0 = iou.nowUsMonotonic(io);
                var esc = dsv41.Scope.init(gpa, gpu);
                defer esc.deinit();
                var cur = x0;
                for (0..steps) |_| cur = if (use_kern) try dsv41.engramGate(&esc, &gm, qk, cur, kv) else try dsv41.engramGateOps(&esc, &gm, qk, cur, kv);
                try mlx.check(mlx.mlx_array_eval(cur));
                if (rep >= 2) times[rep - 2] = @as(f64, @floatFromInt(iou.nowUsMonotonic(io) - t0)) / @as(f64, @floatFromInt(steps));
            }
            std.mem.sort(f64, &times, {}, std.sort.asc(f64));
            if (use_kern) kern_med[if (leg == 1) 0 else 1] = times[times.len / 2] else ops_med[if (leg == 0) 0 else 1] = times[times.len / 2];
        }
        const ops_us = (ops_med[0] + ops_med[1]) * 0.5;
        const kern_us = (kern_med[0] + kern_med[1]) * 0.5;
        std.debug.print("[wide-ubench] engramGate L={d}: kernel={d:.1} us, ops={d:.1} us (AB={d:.1}/{d:.1}, BA={d:.1}/{d:.1}), speedup={d:.2}x\n", .{ L, kern_us, ops_us, ops_med[0], ops_med[1], kern_med[0], kern_med[1], ops_us / kern_us });
    }

    // 2. Wide Engram hoist: 8x M=256 (per-span wkv+gate+concat) vs 1x M=2048 (release wkv [25600, 6144] bf16 = 314.6 MB).
    {
        var wsc = dsv41.Scope.init(gpa, gpu);
        defer wsc.deinit();
        gm.ablation = null;
        var wkv_w = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_random_normal(&wkv_w, &[_]c_int{ 25600, 6144 }, 2, .bfloat16, 0, 0.05, key, gpu));
        wkv_w = try wsc.keep(wkv_w);
        var qk = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_random_normal(&qk, &[_]c_int{ HC, D }, 2, .float32, 0, 1, key, gpu));
        qk = try wsc.keep(qk);
        const eng_layer: dsv41.EngramLayer = .{ .layer = 1, .wkv = .{ .w = wkv_w }, .qk = qk, .table = undefined };
        const chunk_L: usize = 2048;
        const span_L: usize = 256;
        const n_spans: usize = chunk_L / span_L;
        var x_chunk = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_random_normal(&x_chunk, &[_]c_int{ @intCast(chunk_L), HC, D }, 3, .bfloat16, 0, 1, key, gpu));
        x_chunk = try wsc.keep(x_chunk);
        var er_chunk = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_random_normal(&er_chunk, &[_]c_int{ @intCast(chunk_L), 6144 }, 2, .bfloat16, 0, 1, key, gpu));
        er_chunk = try wsc.keep(er_chunk);
        try mlx.check(mlx.mlx_array_eval(wkv_w));
        try mlx.check(mlx.mlx_array_eval(x_chunk));
        try mlx.check(mlx.mlx_array_eval(er_chunk));

        var span_med: [2]f64 = undefined;
        var hoist_med: [2]f64 = undefined;
        // ABBA: 8x256, 1x2048, 1x2048, 8x256
        for ([_]bool{ false, true, true, false }, 0..) |hoist, leg| {
            var times: [5]f64 = undefined;
            for (0..2 + times.len) |rep| {
                const t0 = iou.nowUsMonotonic(io);
                var esc = dsv41.Scope.init(gpa, gpu);
                defer esc.deinit();
                const out = if (hoist)
                    try dsv41.engramApply(&esc, &gm, &eng_layer, x_chunk, er_chunk)
                else blk: {
                    var parts: [n_spans]A = undefined;
                    for (&parts, 0..) |*p, k| {
                        const lo = k * span_L;
                        const hi = lo + span_L;
                        p.* = try dsv41.engramApply(&esc, &gm, &eng_layer, try esc.slice(x_chunk, 0, @intCast(lo), @intCast(hi)), try esc.slice(er_chunk, 0, @intCast(lo), @intCast(hi)));
                    }
                    break :blk try esc.concat(&parts, 0);
                };
                try mlx.check(mlx.mlx_array_eval(out));
                if (rep >= 2) times[rep - 2] = @floatFromInt(iou.nowUsMonotonic(io) - t0);
            }
            std.mem.sort(f64, &times, {}, std.sort.asc(f64));
            if (hoist) hoist_med[if (leg == 1) 0 else 1] = times[times.len / 2] else span_med[if (leg == 0) 0 else 1] = times[times.len / 2];
        }
        const sp_us = (span_med[0] + span_med[1]) * 0.5;
        const hs_us = (hoist_med[0] + hoist_med[1]) * 0.5;
        std.debug.print("[wide-ubench] engram M=2048 (8x256 vs 1x2048): hoist={d:.1} us, span8x={d:.1} us (AB={d:.1}/{d:.1}, BA={d:.1}/{d:.1}), speedup={d:.2}x\n", .{ hs_us, sp_us, span_med[0], span_med[1], hoist_med[0], hoist_med[1], sp_us / hs_us });
    }

    // 3. Release-shape wideLayer hoisted vs chunk-level compress+indexer projections (ratio-2 layer 2, L=2048, 8x256 vs 1x2048).
    {
        const fx = try fixture.Fixture.init(gpa, .{ .lo = 2, .hi = 3, .experts = 8, .ctx = 0, .seed = 5, .act = .bfloat16, .fused = false });
        defer fx.deinit();
        var wsc = dsv41.Scope.init(gpa, gpu);
        defer wsc.deinit();
        const chunk_L: usize = 2048;
        const span_L: usize = 256;
        const n_spans: usize = chunk_L / span_L;
        const ly = &fx.m.layers[2];
        const iw = &ly.idx.?;
        const cw = &ly.comp.?;
        var h_in = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_random_normal(&h_in, &[_]c_int{ @intCast(chunk_L), D }, 2, .bfloat16, 0, 1, key, gpu));
        h_in = try wsc.keep(h_in);
        var qr_in = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_random_normal(&qr_in, &[_]c_int{ @intCast(chunk_L), @intCast(fx.m.cfg.dsv4_q_lora_rank) }, 2, .bfloat16, 0, 1, key, gpu));
        qr_in = try wsc.keep(qr_in);
        try mlx.check(mlx.mlx_array_eval(h_in));
        try mlx.check(mlx.mlx_array_eval(qr_in));

        var span_med: [2]f64 = undefined;
        var hoist_med: [2]f64 = undefined;
        // ABBA: 8x256 (per-span compress + wq_b + ropeQat(f32) + weights_proj), 1x2048 (hoist + ropeQat(bf16))
        for ([_]bool{ false, true, true, false }, 0..) |hoist, leg| {
            var times: [5]f64 = undefined;
            for (0..2 + times.len) |rep| {
                var st = try dsv41.initState(fx.m, gpa);
                defer st.deinit();
                const t0 = iou.nowUsMonotonic(io);
                var csc = dsv41.Scope.init(gpa, gpu);
                defer csc.deinit();
                const pos = try csc.i(0);
                if (hoist) {
                    try dsv41.compress(&csc, fx.m, &st, 2, cw, h_in, 0);
                    const iq = try dsv41.ropeQat(&csc, fx.m, try csc.reshape(try iw.wq_b.apply(&csc, qr_in), &.{ @intCast(chunk_L), @intCast(fx.m.cfg.dsv4_index_n_heads), @intCast(fx.m.cfg.dsv4_index_head_dim) }), fx.m.rope_yarn.fwd, pos, 1.0, .fp4_ue8m0, fx.m.act);
                    const iwt = try iw.weights_proj.apply(&csc, h_in);
                    try mlx.check(mlx.mlx_array_eval(try csc.keep(try st.layers[2].comp.?.view(chunk_L / 2, gpu))));
                    try mlx.check(mlx.mlx_array_eval(try csc.keep(try st.layers[2].idx_k.?.view(chunk_L / 2, gpu))));
                    try mlx.check(mlx.mlx_array_eval(iq));
                    try mlx.check(mlx.mlx_array_eval(iwt));
                } else {
                    for (0..n_spans) |k| {
                        const lo = k * span_L;
                        const hi = (k + 1) * span_L;
                        const hs = try csc.slice(h_in, 0, lo, hi);
                        const qs = try csc.slice(qr_in, 0, lo, hi);
                        try dsv41.compress(&csc, fx.m, &st, 2, cw, hs, lo);
                        const iq = try dsv41.ropeQat(&csc, fx.m, try csc.reshape(try iw.wq_b.apply(&csc, qs), &.{ @intCast(span_L), @intCast(fx.m.cfg.dsv4_index_n_heads), @intCast(fx.m.cfg.dsv4_index_head_dim) }), fx.m.rope_yarn.fwd, try csc.i(@intCast(lo)), 1.0, .fp4_ue8m0, .float32);
                        const iwt = try iw.weights_proj.apply(&csc, hs);
                        try mlx.check(mlx.mlx_array_eval(iq));
                        try mlx.check(mlx.mlx_array_eval(iwt));
                    }
                    try mlx.check(mlx.mlx_array_eval(try csc.keep(try st.layers[2].comp.?.view(chunk_L / 2, gpu))));
                    try mlx.check(mlx.mlx_array_eval(try csc.keep(try st.layers[2].idx_k.?.view(chunk_L / 2, gpu))));
                }
                if (rep >= 2) times[rep - 2] = @floatFromInt(iou.nowUsMonotonic(io) - t0);
            }
            std.mem.sort(f64, &times, {}, std.sort.asc(f64));
            if (hoist) hoist_med[if (leg == 1) 0 else 1] = times[times.len / 2] else span_med[if (leg == 0) 0 else 1] = times[times.len / 2];
        }
        const sp_us = (span_med[0] + span_med[1]) * 0.5;
        const hs_us = (hoist_med[0] + hoist_med[1]) * 0.5;
        std.debug.print("[wide-ubench] idx+comp M=2048 (8x256 vs 1x2048): hoist={d:.1} us, span8x={d:.1} us (AB={d:.1}/{d:.1}, BA={d:.1}/{d:.1}), speedup={d:.2}x\n", .{ hs_us, sp_us, span_med[0], span_med[1], hoist_med[0], hoist_med[1], sp_us / hs_us });
    }
}

test {
    _ = fixture;
    _ = @import("exact_test.zig");
    inline for (.{ in_proj, wq_b, idx_score, topk, attn, wo_a, ffn_in, moe_up, moe_down }) |k| _ = k;
}
