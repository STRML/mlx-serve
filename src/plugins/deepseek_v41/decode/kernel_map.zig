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

fn waitForGate() void {
    const gate = std.c.getenv("DSV41_KMAP_GATE") orelse return;
    _ = std.c.unlink(gate);
    std.debug.print("[kmap] gate\n", .{});
    while (std.c.access(gate, 0) != 0) sleepSecs(50);
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
    const ab_spec = std.c.getenv("DSV41_KMAP_AB") orelse {
        var rows_it = std.mem.tokenizeScalar(u8, rows_list, ',');
        while (rows_it.next()) |tok| try phase(gpa, fx, try std.fmt.parseInt(usize, tok, 10), steps, &p0, null);
        return;
    };
    const probe = try Probe.init(gpa, fx, lo, hi);
    defer probe.deinit();
    fx.m.probe = &probe.hook;
    defer fx.m.probe = null;
    var exps = std.mem.tokenizeScalar(u8, std.mem.span(ab_spec), ';');
    while (exps.next()) |exp| {
        var ab: Ab = .{ .probe = probe, .name = exp };
        var it = std.mem.tokenizeScalar(u8, exp, '/');
        while (it.next()) |tok| {
            ab.arms[ab.n] = try parseArm(tok);
            ab.n += 1;
        }
        var rows_it = std.mem.tokenizeScalar(u8, rows_list, ',');
        while (rows_it.next()) |tok| try phase(gpa, fx, try std.fmt.parseInt(usize, tok, 10), steps, &p0, &ab);
    }
    probe.arm = .{};
    if (std.c.getenv("DSV41_KMAP_CHAIN")) |raw| {
        var it = std.mem.tokenizeScalar(u8, std.mem.span(raw), ',');
        while (it.next()) |tok| try probe.reportChains(std.meta.stringToEnum(dsv41.ProjAt, tok) orelse return error.BadProjection);
    }
}

const ProjAt = dsv41.ProjAt;
const Mode = enum { base, ctl, twin, same, pre, precopy };
const Arm = struct { mode: Mode = .base, at: ProjAt = .in_proj, index: ?bool = null };

fn parseArm(tok: []const u8) !Arm {
    var it = std.mem.tokenizeAny(u8, tok, ":@");
    const mode = std.meta.stringToEnum(Mode, it.next() orelse "base") orelse return error.BadArm;
    if (mode == .base) return .{};
    const at = std.meta.stringToEnum(ProjAt, it.next() orelse return error.BadArm) orelse return error.BadArm;
    const where = it.next() orelse return .{ .mode = mode, .at = at };
    if (std.mem.eql(u8, where, "idx")) return .{ .mode = mode, .at = at, .index = true };
    if (std.mem.eql(u8, where, "plain")) return .{ .mode = mode, .at = at, .index = false };
    return error.BadArm;
}

const max_arms = 12;

const Ab = struct {
    probe: *Probe,
    arms: [max_arms]Arm = @splat(.{}),
    n: usize = 0,
    name: []const u8,

    fn armOf(ab: *const Ab, k: usize) usize {
        const block = k / ab.n;
        return (k % ab.n + block) % ab.n;
    }
};

const PASS_SOURCE =
    \\    uint i = thread_position_in_grid.x;
    \\    y[i] = x[i];
;

const max_layers = 64;
const Ins = struct { a: [4]A = @splat(none), n: usize = 0 };
const none: A = .{ .ctx = null };

const Probe = struct {
    gpa: std.mem.Allocator,
    m: *dsv41.Dsv41Model,
    lo: usize,
    hi: usize,
    held: dsv41.Scope,
    copies: []dsv41.Layer,
    pass: mlx.mlx_fast_metal_kernel,
    read: mlx.mlx_fast_metal_kernel,
    arm: Arm = .{},
    seen: [4][max_layers]Ins = @splat(@splat(.{})),
    hook: dsv41.ProjProbe,

    fn init(gpa: std.mem.Allocator, fx: *fixture.Fixture, lo: usize, hi: usize) !*Probe {
        std.debug.assert(hi <= max_layers);
        const p = try gpa.create(Probe);
        errdefer gpa.destroy(p);
        p.* = .{ .gpa = gpa, .m = fx.m, .lo = lo, .hi = hi, .held = dsv41.Scope.init(gpa, fx.m.s), .copies = try gpa.alloc(dsv41.Layer, hi - lo), .pass = try dsv41.buildKernel("dsv41_kmap_pass", &.{ "x", "dep" }, &.{"y"}, PASS_SOURCE, ""), .read = try dsv41.buildKernel("dsv41_kmap_read", &.{ "w", "s" }, &.{"out"}, READ_SOURCE, ""), .hook = .{ .ctx = p, .before = &before } };
        for (p.copies, lo..) |*c, li| c.* = try decode.layerCopy(&p.held, fx.layer(li));
        return p;
    }

    fn deinit(p: *Probe) void {
        for (&p.seen) |*row| for (row) |*ins| forget(ins);
        _ = mlx.mlx_fast_metal_kernel_free(p.pass);
        _ = mlx.mlx_fast_metal_kernel_free(p.read);
        p.held.deinit();
        p.gpa.free(p.copies);
        p.gpa.destroy(p);
    }

    fn forget(ins: *Ins) void {
        for (ins.a[0..ins.n]) |a| _ = mlx.mlx_array_free(a);
        ins.n = 0;
    }

    fn remember(p: *Probe, li: usize, at: ProjAt, ins: []const A) void {
        const slot = &p.seen[@backingInt(at)][li];
        forget(slot);
        for (ins, 0..) |a, i| {
            slot.a[i] = mlx.mlx_array_new();
            _ = mlx.mlx_array_set(&slot.a[i], a);
        }
        slot.n = ins.len;
    }

    fn applies(p: *const Probe, li: usize, at: ProjAt) bool {
        if (p.arm.mode == .base or p.arm.at != at) return false;
        const want = p.arm.index orelse return true;
        return (p.m.layers[li].idx != null) == want;
    }

    fn before(ctx: *anyopaque, sc: *dsv41.Scope, m: *dsv41.Dsv41Model, li: usize, at: ProjAt, ins: []const A) anyerror!A {
        const p: *Probe = @ptrCast(@alignCast(ctx));
        p.remember(li, at, ins);
        if (!p.applies(li, at)) return ins[0];
        const dep = switch (p.arm.mode) {
            .base, .ctl => ins[0],
            .twin => try twinLaunch(sc, m, &p.copies[li - p.lo], li, at, ins),
            .same => try twinLaunch(sc, m, &m.layers[li], li, at, ins),
            .pre => try readLaunch(sc, p.read, weightsOf(&m.layers[li], at)),
            .precopy => try readLaunch(sc, p.read, weightsOf(&p.copies[li - p.lo], at)),
        };
        return passLaunch(sc, p.pass, ins[0], dep);
    }

    fn chainUs(p: *Probe, at: ProjAt, index: bool, twin: bool) !f64 {
        var lis: [max_layers]usize = undefined;
        var n: usize = 0;
        for (p.lo..p.hi) |li| if (p.seen[@backingInt(at)][li].n > 0 and (p.m.layers[li].idx != null) == index) {
            lis[n] = li;
            n += 1;
        };
        if (n == 0) return 0;
        const links = 4 * n;
        var times: [7]f64 = undefined;
        for (0..2 + times.len) |rep| {
            var sc = dsv41.Scope.init(p.gpa, p.m.s);
            defer sc.deinit();
            var prev = p.seen[@backingInt(at)][lis[0]].a[0];
            for (0..links) |i| {
                const li = lis[i % n];
                const ins = &p.seen[@backingInt(at)][li];
                const g = try passLaunch(&sc, p.pass, ins.a[0], prev);
                if (!twin) {
                    prev = g;
                    continue;
                }
                var fed = ins.a;
                fed[0] = g;
                prev = try twinLaunch(&sc, p.m, &p.copies[li - p.lo], li, at, fed[0..ins.n]);
            }
            const t0 = nowUs();
            try mlx.check(mlx.mlx_array_eval(prev));
            if (rep >= 2) times[rep - 2] = @as(f64, @floatFromInt(nowUs() - t0)) / @as(f64, @floatFromInt(links));
        }
        return medianOf(&times);
    }

    fn reportChains(p: *Probe, at: ProjAt) !void {
        for ([_]bool{ false, true }) |index| {
            waitForGate();
            const pass_us = try p.chainUs(at, index, false);
            const both_us = try p.chainUs(at, index, true);
            std.debug.print("[kmap] chain {s} {s} layers: copy launch {d:.1} us, projection + copy {d:.1} us, projection {d:.1} us a link\n", .{ @tagName(at), if (index) "index" else "plain", pass_us, both_us, both_us - pass_us });
        }
    }
};

fn twinLaunch(sc: *dsv41.Scope, m: *dsv41.Dsv41Model, ly: *const dsv41.Layer, li: usize, at: ProjAt, ins: []const A) !A {
    return switch (at) {
        .in_proj => ((try decode.in_proj.run(sc, &m.dec.in_proj.?, m, ly, ins[0], ins[1])) orelse return error.Declined).z,
        .wq_b => ((try decode.wq_b.run(sc, m, m.dec.wq_b.?, ly, li, ins[0], ins[1], ins[2], ins[3])) orelse return error.Declined).q,
        .wo_a => (try decode.wo_a.apply(sc, m, ly, m.dec.wo_a.?, m.dec.wo_a_counter, dsv41.ropeFor(m, li).inv, ins[3], ins[0], ins[1], ins[2])) orelse error.Declined,
        .wo_b => ly.wo_b.apply(sc, ins[0]),
    };
}

const READ_SOURCE =
    \\    const uint tid = thread_position_in_grid.x, n = threads_per_grid.x;
    \\    const device uint4* wv = (const device uint4*)w;
    \\    const device uint* sv = (const device uint*)s;
    \\    uint acc = 0;
    \\    for (uint i = tid; i < NW; i += n) { const uint4 v = wv[i]; acc ^= v.x ^ v.y ^ v.z ^ v.w; }
    \\    for (uint i = tid; i < NS; i += n) acc ^= sv[i];
    \\    out[tid] = acc;
;

const read_threads = 32768;

fn weightsOf(ly: *const dsv41.Layer, at: ProjAt) dsv41.Lin {
    return switch (at) {
        .in_proj => ly.q_kv.?,
        .wq_b => ly.wq_b,
        .wo_a => ly.wo_a,
        .wo_b => ly.wo_b,
    };
}

fn bytesOf(a: A) usize {
    return mlx.mlx_array_size(a) * mlx.mlx_array_itemsize(a);
}

fn readLaunch(sc: *dsv41.Scope, k: mlx.mlx_fast_metal_kernel, l: dsv41.Lin) !A {
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{read_threads}, 1, .uint32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, read_threads, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 256, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "NW", @intCast(bytesOf(l.w) / 16)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "NS", @intCast(bytesOf(l.s) / 4)));
    const iv = mlx.mlx_vector_array_new_data(&[_]A{ l.w, l.s }, 2);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, k, iv, cfg, sc.s));
    var o = mlx.mlx_array_new();
    return sc.res(mlx.mlx_vector_array_get(&o, ov, 0), &o);
}

fn passLaunch(sc: *dsv41.Scope, k: mlx.mlx_fast_metal_kernel, x: A, dep: A) !A {
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    const shape = mlx.getShape(x);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, shape.ptr, shape.len, mlx.mlx_array_dtype(x)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, @intCast(mlx.mlx_array_size(x)), 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 256, 1, 1));
    const iv = mlx.mlx_vector_array_new_data(&[_]A{ x, dep }, 2);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, k, iv, cfg, sc.s));
    var o = mlx.mlx_array_new();
    return sc.res(mlx.mlx_vector_array_get(&o, ov, 0), &o);
}

fn medianOf(xs: []f64) f64 {
    std.mem.sort(f64, xs, {}, std.sort.asc(f64));
    return xs[xs.len / 2];
}

fn printArms(ab: *const Ab, rows: usize, us: []const f64) void {
    const blocks = us.len / ab.n;
    var by: [max_arms][max_steps]f64 = undefined;
    var diff: [max_arms][max_steps]f64 = undefined;
    for (0..blocks) |b| for (0..ab.n) |pos| {
        const k = b * ab.n + pos;
        by[ab.armOf(k)][b] = us[k];
    };
    for (0..blocks) |b| for (0..ab.n) |j| {
        diff[j][b] = by[j][b] - by[0][b];
    };
    for (0..ab.n) |j| {
        const d = medianOf(diff[j][0..blocks]);
        const q10 = quantileOf(by[j][0..blocks], 0.1);
        const q25 = quantileOf(by[j][0..blocks], 0.25);
        const med = quantileOf(by[j][0..blocks], 0.5);
        std.debug.print("[kmap] arm {s} #{d} rows={d} blocks={d}: q10 {d:.0} q25 {d:.0} median {d:.0} us, block diff from arm 0 median {d:.0} us\n", .{ ab.name, j, rows, blocks, q10, q25, med, d });
    }
    var line: [max_steps * 8]u8 = undefined;
    var n: usize = 0;
    for (us, 0..) |u, k| {
        const piece = std.fmt.bufPrint(line[n..], "{d}:{d:.0} ", .{ ab.armOf(k), u }) catch break;
        n += piece.len;
    }
    std.debug.print("[kmap] raw {s} rows={d}: {s}\n", .{ ab.name, rows, line[0..n] });
}

fn quantileOf(xs: []f64, f: f64) f64 {
    std.mem.sort(f64, xs, {}, std.sort.asc(f64));
    return xs[@intFromFloat(f * @as(f64, @floatFromInt(xs.len - 1)))];
}

const max_steps = 1024;

/// One row count: 3 warm-up steps and `steps` timed ones from position `p0`, which moves past them.
fn phase(gpa: std.mem.Allocator, fx: *fixture.Fixture, rows: usize, steps: usize, p0: *usize, ab: ?*const Ab) !void {
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

    var us: [max_steps]f64 = undefined;
    std.debug.assert(steps <= us.len);
    sleepSecs(2000);
    waitForGate();
    for (0..total) |k| {
        if (k == warm) sleepSecs(500);
        if (ab) |a| a.probe.arm = a.arms[a.armOf(k -| warm)];
        const s0 = nowUs();
        const out = try dsv41.workerStep(fx.m, gpa, xs[k], pres[k], p0.*, false, false);
        for (out) |o| _ = mlx.mlx_array_free(o);
        if (k >= warm) us[k - warm] = @floatFromInt(nowUs() - s0);
        p0.* += rows;
    }
    sleepSecs(500);
    if (ab) |a| printArms(a, rows, us[0 .. steps - steps % a.n]);
    var sorted: [max_steps]f64 = undefined;
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
