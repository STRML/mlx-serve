//! A TP rank's whole decode step in a loop (`DSV41_DEC_UBENCH` + `DSV41_KMAP`), for a GPU trace to name every dispatch.
//!
//! It builds the trunk's layers at one rank's widths of a tensor split (random weights, mxfp4 experts), sums stubbed as
//! an identity, then runs `workerStep` as the serving loop does (a layer handed to the GPU as it is built). The setup's
//! GPU work is followed by a 2 s sleep, so a trace of the process splits at its idle gaps of 300 ms or more; a phase per
//! row count is 3 warm-up steps, 0.5 s, and `DSV41_KMAP_STEPS` timed ones, each on fresh random input rows (so the
//! routes move). A trace's last stretches are then [warm, timed] for each row count in order.
//!
//!   DSV41_KMAP_ROWS (1,6: row counts, one phase each)  DSV41_KMAP_STEPS (32, even)  DSV41_KMAP_CTX (2048)  DSV41_KMAP_TP (4)
//!   DSV41_KMAP_LAYERS (0-40)  DSV41_KMAP_EXPERTS (384)  DSV41_KMAP_GO (a file to wait for after the setup)
//!
//! Lines start `[kmap]`.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const iou = @import("../../../sdk/util/io_util.zig");
const dsv41 = @import("../deepseek_v41.zig");
const decode = @import("decode.zig");
const fixture = @import("fixture.zig");

const A = mlx.mlx_array;

fn nowUs() i64 {
    return iou.nowUsMonotonic(std.Io.Threaded.global_single_threaded.io());
}

/// The identity collective: the tensor sum's casts run, nothing crosses a wire.
fn identitySum(x: A) anyerror!A {
    var out = mlx.mlx_array_new();
    _ = mlx.mlx_array_set(&out, x);
    return out;
}

fn envInt(name: [*:0]const u8, default: usize) usize {
    const raw = std.c.getenv(name) orelse return default;
    return std.fmt.parseInt(usize, std.mem.span(raw), 10) catch default;
}

/// `DSV41_KMAP_GO=<path>`: wait for that file once the setup is done, so a trace can attach to a quiet process.
fn waitForGo() void {
    const go = std.c.getenv("DSV41_KMAP_GO") orelse return;
    std.debug.print("[kmap] ready\n", .{});
    while (std.c.access(go, 0) != 0) sleepSecs(100);
}

fn sleepSecs(ms: i64) void {
    std.Io.sleep(std.Io.Threaded.global_single_threaded.io(), .fromMilliseconds(ms), .real) catch {};
}

test "dsv41 kernel map: a TP rank's full decode step in a loop for a GPU trace (DSV41_DEC_UBENCH, DSV41_KMAP)" {
    if (!decode.ubenchOn() or std.c.getenv("DSV41_KMAP") == null) return error.SkipZigTest;
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const gpa = std.testing.allocator;
    dsv41.clusterHooks().tp_sum = &identitySum;
    defer dsv41.clusterHooks().tp_sum = null;

    const steps = envInt("DSV41_KMAP_STEPS", 32);
    const rows_list = std.mem.span(std.c.getenv("DSV41_KMAP_ROWS") orelse "1,6");
    const ctx = envInt("DSV41_KMAP_CTX", 2048);
    const tp = envInt("DSV41_KMAP_TP", 4);
    const experts: u32 = @intCast(envInt("DSV41_KMAP_EXPERTS", 384));
    var lo: usize = 0;
    var hi: usize = 40;
    if (std.c.getenv("DSV41_KMAP_LAYERS")) |raw| {
        var it = std.mem.tokenizeScalar(u8, std.mem.span(raw), '-');
        lo = try std.fmt.parseInt(usize, it.next() orelse "0", 10);
        hi = try std.fmt.parseInt(usize, it.next() orelse "40", 10);
    }
    const t0 = nowUs();
    const fx = try fixture.Fixture.init(gpa, .{
        .lo = lo,
        .hi = hi,
        .experts = experts,
        .ctx = ctx,
        .fused = true,
        .experts_fp4 = true,
        .tp = .{ .rank = 0, .size = @intCast(tp) },
    });
    defer fx.deinit();
    std.debug.print("[kmap] fixture tp={d} layers={d}-{d} experts={d} ctx={d} built in {d:.0} s\n", .{ tp, lo, hi, experts, ctx, @as(f64, @floatFromInt(nowUs() - t0)) / 1e6 });

    waitForGo();
    var p0: usize = ctx;
    var rows_it = std.mem.tokenizeScalar(u8, rows_list, ',');
    while (rows_it.next()) |tok| {
        const rows = try std.fmt.parseInt(usize, tok, 10);
        try phase(gpa, fx, rows, steps, &p0);
    }
}

/// One row count: 3 warm-up steps and `steps` timed ones from position `p0`, which moves past them.
fn phase(gpa: std.mem.Allocator, fx: *fixture.Fixture, rows: usize, steps: usize, p0: *usize) !void {
    const warm = 3;
    const total = warm + steps;
    const xs = try gpa.alloc(A, total);
    defer gpa.free(xs);
    const pres = try gpa.alloc(A, total);
    defer gpa.free(pres);
    for (xs, pres) |*x, *p| {
        x.* = try fx.stream(rows);
        p.* = try fx.cpre(rows);
    }
    defer for (xs, pres) |x, p| {
        _ = mlx.mlx_array_free(x);
        _ = mlx.mlx_array_free(p);
    };
    for (xs) |x| try mlx.check(mlx.mlx_array_eval(x));
    for (pres) |p| try mlx.check(mlx.mlx_array_eval(p));

    var us: [256]f64 = undefined;
    std.debug.assert(steps <= us.len);
    sleepSecs(2000);
    for (0..total) |k| {
        if (k == warm) sleepSecs(500);
        const s0 = nowUs();
        const out = try dsv41.workerStep(fx.m, gpa, xs[k], pres[k], p0.*, false, false);
        for (out) |o| _ = mlx.mlx_array_free(o);
        if (k >= warm) us[k - warm] = @floatFromInt(nowUs() - s0);
        p0.* += rows;
    }
    sleepSecs(500);
    var sorted: [256]f64 = undefined;
    @memcpy(sorted[0..steps], us[0..steps]);
    std.mem.sort(f64, sorted[0..steps], {}, std.sort.asc(f64));
    std.debug.print("[kmap] rows={d} steps={d} layers={d} host step us: min={d:.0} median={d:.0} max={d:.0}\n", .{ rows, steps, fx.m.pair.?.hi - fx.m.pair.?.lo, sorted[0], sorted[steps / 2], sorted[steps - 1] });
}

/// One linear's random weight, quantized: `mx` mxfp8 in groups of 32, else 8-bit affine.
fn randomLin(n: c_int, k: c_int, gs: u32, mx: bool, seed: u64) !dsv41.Lin {
    const s = mlx.gpuStream();
    var key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_random_key(&key, seed));
    const shape = [_]c_int{ n, k };
    var w = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w);
    try mlx.check(mlx.mlx_random_normal(&w, &shape, 2, .bfloat16, 0, 0.02, key, s));
    var qv = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(qv);
    const name: [*:0]const u8 = if (mx) "mxfp8" else "affine";
    try mlx.check(mlx.mlx_quantize(&qv, w, mlx.mlx_optional_int.some(@intCast(gs)), mlx.mlx_optional_int.some(8), name, .{ .ctx = null }, s));
    var out: [3]A = .{ .{ .ctx = null }, .{ .ctx = null }, .{ .ctx = null } };
    for (out[0..mlx.mlx_vector_array_size(qv)], 0..) |*a, i| {
        a.* = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(a, qv, i));
    }
    for (out) |a| if (a.ctx != null) try mlx.check(mlx.mlx_array_eval(a));
    return .{ .w = out[0], .s = out[1], .b = out[2], .qp = .{ .bits = 8, .group_size = gs, .mode = if (mx) .mxfp8 else .affine } };
}

/// One line: each layer's compress ratio, `K` where it sources the KV cache, `I` where it sources the index keys, `E`
/// where Engram sits.
fn printLayers(cfg: *const @import("../config.zig").Config) void {
    var line: [512]u8 = undefined;
    var n: usize = 0;
    for (0..cfg.num_hidden_layers) |l| {
        const kv: u8 = if ((cfg.dsv41_kv_sources >> @intCast(l)) & 1 != 0) 'K' else '-';
        const ix: u8 = if ((cfg.dsv41_index_sources >> @intCast(l)) & 1 != 0) 'I' else '-';
        var en: u8 = '-';
        for (cfg.dsv41_engram_layers[0..cfg.dsv41_n_engram_layers]) |e| if (e == l) {
            en = 'E';
        };
        const s = std.fmt.bufPrint(line[n..], "{d}:r{d}{c}{c}{c} ", .{ l, cfg.dsv4_compress_ratios[l], kv, ix, en }) catch break;
        n += s.len;
    }
    std.debug.print("[kmap] layers (ratio, K=KV source, I=index source, E=Engram): {s}\n", .{line[0..n]});
}

// The head and Engram's wkv linears, which the fixture's pair worker does not hold, as the trunk runs them: the
// final norm (head only), the quantized matmul and the f32 cast. `n_sets` distinct copies of each weight, 16 launches
// evaluated together over them, so every launch streams its weight from DRAM and the GPU stays busy between launches.
test "dsv41 kernel map: the head's and Engram's linears at TP4 (DSV41_DEC_UBENCH, DSV41_KMAP_HEAD)" {
    if (!decode.ubenchOn() or std.c.getenv("DSV41_KMAP_HEAD") == null) return error.SkipZigTest;
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const gpa = std.testing.allocator;
    const cfg = try fixture.releaseConfig();
    const d: c_int = @intCast(cfg.hidden_size);
    const rows_part: c_int = @intCast((cfg.vocab_size + 3) / 4);
    const eng_in: c_int = @intCast((cfg.dsv41_engram_max_ngram - 1) * cfg.dsv41_engram_n_heads * cfg.dsv41_engram_head_dim);
    const eng_out: c_int = @intCast((cfg.dsv4_hc_mult + 1) * cfg.hidden_size);
    printLayers(&cfg);
    const n_sets = 4;
    const launches = 16;
    var heads: [n_sets]dsv41.Lin = undefined;
    var wkvs: [n_sets]dsv41.Lin = undefined;
    // The head: rank 0's vocabulary quarter, requantized 8-bit affine in groups of 64 as `headPart` does.
    // Engram's wkv: mxfp8 in groups of 32 as the release's other fp8 linears are (uncut under a tensor split).
    for (&heads, &wkvs, 0..) |*h, *w, k| {
        h.* = try randomLin(rows_part, d, 64, false, 11 + k);
        w.* = try randomLin(eng_out, eng_in, 32, true, 31 + k);
    }
    var sc0 = dsv41.Scope.init(gpa, mlx.gpuStream());
    defer sc0.deinit();
    const norm = try sc0.keep(try sc0.astype(try sc0.broadcast(try sc0.f(1), &.{d}), .bfloat16));
    waitForGo();
    const stream = mlx.gpuStream();
    var key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_random_key(&key, 5));
    for ([_]bool{ true, false }) |is_head| for ([_]usize{ 1, 6 }) |rows| {
        const width: c_int = if (is_head) d else eng_in;
        var x = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x);
        try mlx.check(mlx.mlx_random_normal(&x, &[_]c_int{ @intCast(rows), width }, 2, .bfloat16, 0, 1, key, stream));
        try mlx.check(mlx.mlx_array_eval(x));
        var us: [9]f64 = undefined;
        sleepSecs(2000);
        for (0..3 + us.len) |k| {
            if (k == 3) sleepSecs(500);
            var sc = dsv41.Scope.init(gpa, stream);
            defer sc.deinit();
            var ys: [launches]A = undefined;
            const t0 = nowUs();
            for (&ys, 0..) |*y, j| {
                const xn = if (is_head) try sc.rms(x, norm, cfg.rms_norm_eps) else x;
                const lin = if (is_head) &heads[j % n_sets] else &wkvs[j % n_sets];
                y.* = try sc.astype(try lin.apply(&sc, xn), .float32);
            }
            try dsv41.evalAll(&ys);
            if (k >= 3) us[k - 3] = @as(f64, @floatFromInt(nowUs() - t0)) / launches;
        }
        sleepSecs(500);
        std.mem.sort(f64, &us, {}, std.sort.asc(f64));
        std.debug.print("[kmap] {s} rows={d}: us a launch (16 evaluated together over {d} weight sets): min={d:.1} median={d:.1}\n", .{ if (is_head) "head" else "engram_wkv", rows, n_sets, us[0], us[us.len / 2] });
    };
}
