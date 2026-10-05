//! Env-gated chained microbench of the two-launch hyper-connection read at 1..4 rows (production geometry:
//! hc 4, hidden 2560, lowrank 320, 8-bit g64, bf16): the generic up launch against the vectorized one, calls
//! chained the way a forward chains them, arms interleaved over several passes.
//!   HC_UBENCH=1 zig build test -Doptimize=ReleaseFast -Dtest-filter="hc one-row"

const std = @import("std");
const mlx = @import("mlx.zig");
const xfm = @import("transformer.zig");
const hc2 = @import("hc_decode2.zig");
const hc2_old = @import("hc_decode2_old.zig");
const io_util = @import("io_util.zig");

const HC: c_int = 4;
const H: c_int = 2560;
const K: c_int = HC * H;
const R: c_int = 320;
const BITS: u32 = 8;
const GS: u32 = 64;
const CHAIN = 80; // one forward: 40 layers, two reads each
const WARM = 3;
const PASSES = 9;
const MIN_W: c_int = 1;
const MAX_W: c_int = 4;

const Q = struct {
    w: mlx.mlx_array,
    s: mlx.mlx_array,
    b: mlx.mlx_array,
    fn deinit(q: Q) void {
        _ = mlx.mlx_array_free(q.w);
        _ = mlx.mlx_array_free(q.s);
        _ = mlx.mlx_array_free(q.b);
    }
};

fn randBf16(rnd: std.Random, shape: []const c_int, scale: f32, offset: f32, s: mlx.mlx_stream) !mlx.mlx_array {
    var key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_random_key(&key, rnd.int(u64)));
    const lo = mlx.mlx_array_new_float(offset - 0.5 * scale);
    defer _ = mlx.mlx_array_free(lo);
    const hi = mlx.mlx_array_new_float(offset + 0.5 * scale);
    defer _ = mlx.mlx_array_free(hi);
    var u = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(u);
    try mlx.check(mlx.mlx_random_uniform(&u, lo, hi, shape.ptr, shape.len, .float32, key, s));
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&out, u, .bfloat16, s));
    return out;
}

fn quantRandom(rnd: std.Random, rows: c_int, cols: c_int, s: mlx.mlx_stream) !Q {
    const w = try randBf16(rnd, &.{ rows, cols }, 0.2, 0.0, s);
    defer _ = mlx.mlx_array_free(w);
    var triple = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(triple);
    try mlx.check(mlx.mlx_quantize(&triple, w, mlx.mlx_optional_int.some(@intCast(GS)), mlx.mlx_optional_int.some(@intCast(BITS)), "affine", .{}, s));
    var q: Q = .{ .w = mlx.mlx_array_new(), .s = mlx.mlx_array_new(), .b = mlx.mlx_array_new() };
    try mlx.check(mlx.mlx_vector_array_get(&q.w, triple, 0));
    try mlx.check(mlx.mlx_vector_array_get(&q.s, triple, 1));
    try mlx.check(mlx.mlx_vector_array_get(&q.b, triple, 2));
    try mlx.check(mlx.mlx_array_eval(q.w));
    try mlx.check(mlx.mlx_array_eval(q.s));
    try mlx.check(mlx.mlx_array_eval(q.b));
    return q;
}

const Arm = enum { old, generic, vectorized };

const Weights = struct {
    norm_w: mlx.mlx_array,
    down_w: mlx.mlx_array,
    down_s: mlx.mlx_array,
    down_b: mlx.mlx_array,
    up_w: mlx.mlx_array,
    up_s: mlx.mlx_array,
    up_b: mlx.mlx_array,
    inject_flat: mlx.mlx_array,
};

const Inputs = struct {
    s: mlx.mlx_stream,
    sets: []Weights, // one per read of a forward: 40 layers x 2; HC_UBENCH_SETS=1 keeps one set cache-hot
    eps: mlx.mlx_array,
    x0: [MAX_W + 1]mlx.mlx_array, // [1, w, K] per width
    wo: [MAX_W + 1]mlx.mlx_array, // [1, w, H]
    wi: [MAX_W + 1]mlx.mlx_array, // [1, w, HC, 1]
};

/// One forward's worth of reads. Each read takes the stream the previous one returned, and applies the previous read's
/// `mixed` as the block output and its `inj` as the gate, as a forward does, so none of a read's work is dead and the
/// compiled graph cannot be pruned down to its elementwise tail.
const Timing = struct { build_ns: u64, total_ns: u64 };

fn chain(in: *const Inputs, arm: Arm, w: c_int) !Timing {
    const io = std.Io.Threaded.global_single_threaded.io();
    var sw = io_util.Stopwatch.init(io);
    const wu: usize = @intCast(w);
    var x = mlx.mlx_array_new();
    _ = mlx.mlx_array_set(&x, in.x0[wu]);
    var out = mlx.mlx_array_new();
    _ = mlx.mlx_array_set(&out, in.wo[wu]);
    var inj = mlx.mlx_array_new();
    _ = mlx.mlx_array_set(&inj, in.wi[wu]);
    defer {
        _ = mlx.mlx_array_free(x);
        _ = mlx.mlx_array_free(out);
        _ = mlx.mlx_array_free(inj);
    }
    for (0..CHAIN) |ci| {
        const pend: xfm.HcPending = .{ .out = out, .inj = inj };
        hc2.vec_override = arm == .vectorized;
        const wt = in.sets[ci % in.sets.len];
        const r: struct { mixed: mlx.mlx_array, inj: mlx.mlx_array, stream: mlx.mlx_array } = if (arm == .old) blk: {
            const o = (try hc2_old.read(in.s, x, wt.norm_w, wt.down_w, wt.down_s, wt.down_b, wt.up_w, wt.up_s, wt.up_b, wt.inject_flat, in.eps, HC, H, BITS, GS, .{ .out = pend.out, .inj = pend.inj })) orelse return error.OldDeclined;
            break :blk .{ .mixed = o.mixed, .inj = o.inj, .stream = o.stream };
        } else blk: {
            const o = (try hc2.read(in.s, x, wt.norm_w, wt.down_w, wt.down_s, wt.down_b, wt.up_w, wt.up_s, wt.up_b, wt.inject_flat, in.eps, w, HC, H, BITS, GS, .{ .out = pend.out, .inj = pend.inj })) orelse return error.TwoLaunchDeclined;
            break :blk .{ .mixed = o.mixed, .inj = o.inj, .stream = o.stream };
        };
        _ = mlx.mlx_array_free(x);
        _ = mlx.mlx_array_free(out);
        _ = mlx.mlx_array_free(inj);
        x = r.stream;
        out = r.mixed;
        inj = r.inj;
    }
    const build_ns = sw.read();
    try mlx.check(mlx.mlx_array_eval(x));
    try mlx.check(mlx.mlx_array_eval(out));
    try mlx.check(mlx.mlx_array_eval(inj));
    return .{ .build_ns = build_ns, .total_ns = sw.read() };
}

fn median(v: []u64) u64 {
    std.mem.sort(u64, v, {}, std.sort.asc(u64));
    return v[v.len / 2];
}

test "hc one-row: generic against vectorized up launch (HC_UBENCH=1)" {
    if (std.c.getenv("HC_UBENCH") == null) return error.SkipZigTest;
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    mlx.installErrorHandler();
    var trash: [512]u8 = undefined;
    if (mlx.errorPending()) _ = mlx.takeError(&trash);
    defer if (mlx.errorPending()) {
        _ = mlx.takeError(&trash);
    };

    const s = mlx.gpuStream();
    xfm.hc_fused_override = true;
    defer xfm.hc_fused_override = null;
    const gen = xfm.deviceGeneration();
    std.debug.print("[hc-ubench] device gen={d} phone={any} verifySharedHardware={any}\n", .{ gen.gen, gen.phone, xfm.verifySharedHardware() });

    var prng = std.Random.DefaultPrng.init(0x4C0DEB);
    const rnd = prng.random();
    var n_sets: usize = CHAIN;
    if (std.c.getenv("HC_UBENCH_SETS")) |raw| n_sets = std.fmt.parseInt(usize, std.mem.span(raw), 10) catch CHAIN;
    const sets = try std.testing.allocator.alloc(Weights, n_sets);
    defer std.testing.allocator.free(sets);
    for (sets) |*set| {
        const down = try quantRandom(rnd, R, K, s);
        const up = try quantRandom(rnd, K, R, s);
        const nw = try randBf16(rnd, &.{ HC, H }, 1.0, 1.0, s);
        const iw = try randBf16(rnd, &.{ K, HC }, 0.1, 0.0, s);
        for ([_]mlx.mlx_array{ nw, iw }) |a| try mlx.check(mlx.mlx_array_eval(a));
        set.* = .{ .norm_w = nw, .down_w = down.w, .down_s = down.s, .down_b = down.b, .up_w = up.w, .up_s = up.s, .up_b = up.b, .inject_flat = iw };
    }
    std.debug.print("[hc-ubench] {d} distinct weight sets ({d:.0} MB)\n", .{ n_sets, @as(f64, @floatFromInt(n_sets)) * 6.6 });
    const epsv = [_]f32{1e-6};
    const eps = mlx.mlx_array_new_data(&epsv, &.{1}, 1, .float32);
    defer _ = mlx.mlx_array_free(eps);

    var in: Inputs = undefined;
    in.s = s;
    in.eps = eps;
    in.sets = sets;
    var made: usize = 0;
    defer for (MIN_W..MIN_W + made) |i| {
        _ = mlx.mlx_array_free(in.x0[i]);
        _ = mlx.mlx_array_free(in.wo[i]);
        _ = mlx.mlx_array_free(in.wi[i]);
    };
    var w: c_int = MIN_W;
    while (w <= MAX_W) : (w += 1) {
        const i: usize = @intCast(w);
        in.x0[i] = try randBf16(rnd, &.{ 1, w, K }, 4.0, 0.0, s);
        in.wo[i] = try randBf16(rnd, &.{ 1, w, H }, 2.0, 0.0, s);
        in.wi[i] = try randBf16(rnd, &.{ 1, w, HC, 1 }, 1.0, 1.0, s);
        for ([_]mlx.mlx_array{ in.x0[i], in.wo[i], in.wi[i] }) |a| try mlx.check(mlx.mlx_array_eval(a));
        made += 1;
    }

    const arms = [_]Arm{ .old, .generic, .vectorized };
    var total: [3][MAX_W + 1][PASSES]u64 = undefined;
    var build: [3][MAX_W + 1][PASSES]u64 = undefined;
    w = MIN_W;
    while (w <= MAX_W) : (w += 1) {
        const i: usize = @intCast(w);
        if (w == 1 or true) {
            for (0..WARM) |_| for (arms) |a| {
                if (a == .old and w != 1) continue;
                _ = try chain(&in, a, w);
            };
        }
        for (0..PASSES) |p| {
            // Interleaved, rotated every pass so drift and ordering land on every arm.
            for (0..3) |k| {
                const ai = (k + p) % 3;
                if (arms[ai] == .old and w != 1) continue;
                const t = try chain(&in, arms[ai], w);
                total[ai][i][p] = t.total_ns;
                build[ai][i][p] = t.build_ns;
            }
        }
    }
    std.debug.print("[hc-ubench] us per read, {d} reads chained per pass, median of {d} interleaved passes (min in brackets)\n", .{ CHAIN, PASSES });
    std.debug.print("[hc-ubench] rows   old (pre-merge)    generic up         vectorized up\n", .{});
    w = MIN_W;
    while (w <= MAX_W) : (w += 1) {
        const i: usize = @intCast(w);
        var line: [3]f64 = .{ 0, 0, 0 };
        var mins: [3]f64 = .{ 0, 0, 0 };
        for (0..3) |a| {
            if (arms[a] == .old and w != 1) continue;
            var copy = total[a][i];
            mins[a] = @as(f64, @floatFromInt(std.mem.min(u64, &copy))) / CHAIN / 1000.0;
            line[a] = @as(f64, @floatFromInt(median(&copy))) / CHAIN / 1000.0;
        }
        std.debug.print("[hc-ubench] {d:>4}   {d:6.1} ({d:5.1})   {d:6.1} ({d:5.1})   {d:6.1} ({d:5.1})\n", .{ w, line[0], mins[0], line[1], mins[1], line[2], mins[2] });
    }
    try std.testing.expect(!mlx.errorPending());
}
