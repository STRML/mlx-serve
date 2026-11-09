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
const seg = @import("seg.zig");
const seg_bounds = @import("seg_bounds.zig");
const seg_ffn = @import("seg_ffn.zig");

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
    seg_bounds.cache_on = true;
    defer {
        seg_bounds.cache_on = false;
        seg_bounds.release();
        applyArm(fx, .{});
    }
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
const Mode = enum { base, ctl, twin, same, pre, precopy, ffn, seg, stop, stopm };

pub const Moe = struct { threads: usize, read: seg_ffn.ReadMode };

pub const Under = union(enum) { level: seg.Level, read: seg_ffn.ReadMode, moe: Moe, broken, nan };

const Arm = struct { mode: Mode = .base, at: ProjAt = .in_proj, index: ?bool = null, ffn: seg_bounds.Arm = .base, under: Under = .{ .level = .off }, stop: u32 = 4 };

fn parseUnder(tok: []const u8) !Under {
    if (std.mem.eql(u8, tok, "broken")) return .broken;
    if (std.mem.eql(u8, tok, "nan")) return .nan;
    if (std.mem.eql(u8, tok, "2m")) return .{ .moe = .{ .threads = 256, .read = .staged } };
    if (std.mem.eql(u8, tok, "2mp")) return .{ .moe = .{ .threads = 256, .read = .plain } };
    if (std.mem.eql(u8, tok, "2m64")) return .{ .moe = .{ .threads = 64, .read = .staged } };
    if (std.mem.eql(u8, tok, "2m64p")) return .{ .moe = .{ .threads = 64, .read = .plain } };
    if (std.mem.eql(u8, tok, "2c")) return .{ .read = .peer };
    if (std.mem.eql(u8, tok, "2p")) return .{ .read = .plain };
    if (std.mem.eql(u8, tok, "2s")) return .{ .read = .staged };
    const n = std.fmt.parseInt(u8, tok, 10) catch return error.BadArm;
    return .{ .level = std.enums.fromInt(seg.Level, n) orelse return error.BadArm };
}

fn parseArm(tok: []const u8) !Arm {
    var it = std.mem.tokenizeAny(u8, tok, ":@");
    const mode = std.meta.stringToEnum(Mode, it.next() orelse "base") orelse return error.BadArm;
    if (mode == .base) return .{};
    if (mode == .ffn) return .{ .mode = .ffn, .ffn = std.meta.stringToEnum(seg_bounds.Arm, it.next() orelse return error.BadArm) orelse return error.BadArm };
    if (mode == .seg) return .{ .mode = .seg, .under = try parseUnder(it.next() orelse return error.BadArm) };
    if (mode == .stopm) return .{ .mode = .stopm, .under = .{ .moe = .{ .threads = 256, .read = .staged } }, .stop = std.fmt.parseInt(u32, it.next() orelse return error.BadArm, 10) catch return error.BadArm };
    if (mode == .stop) return .{ .mode = .stop, .under = .{ .level = .ffn_seg }, .stop = std.fmt.parseInt(u32, it.next() orelse return error.BadArm, 10) catch return error.BadArm };
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
    pass_after: mlx.mlx_fast_metal_kernel,
    read: mlx.mlx_fast_metal_kernel,
    arm: Arm = .{},
    seen: [4][max_layers]Ins = @splat(@splat(.{})),
    hook: dsv41.ProjProbe,

    fn init(gpa: std.mem.Allocator, fx: *fixture.Fixture, lo: usize, hi: usize) !*Probe {
        std.debug.assert(hi <= max_layers);
        const p = try gpa.create(Probe);
        errdefer gpa.destroy(p);
        p.* = .{ .gpa = gpa, .m = fx.m, .lo = lo, .hi = hi, .held = dsv41.Scope.init(gpa, fx.m.s), .copies = try gpa.alloc(dsv41.Layer, hi - lo), .pass = try dsv41.buildKernel("dsv41_kmap_pass", &.{ "x", "dep" }, &.{"y"}, PASS_SOURCE, ""), .pass_after = try dsv41.buildKernel("dsv41_kmap_pass_after", &.{ "dep", "x" }, &.{"y"}, PASS_SOURCE, ""), .read = try dsv41.buildKernel("dsv41_kmap_read", &.{ "w", "s" }, &.{"out"}, READ_SOURCE, ""), .hook = .{ .ctx = p, .before = &before } };
        for (p.copies, lo..) |*c, li| c.* = try decode.layerCopy(&p.held, fx.layer(li));
        return p;
    }

    fn deinit(p: *Probe) void {
        for (&p.seen) |*row| for (row) |*ins| forget(ins);
        _ = mlx.mlx_fast_metal_kernel_free(p.pass);
        _ = mlx.mlx_fast_metal_kernel_free(p.read);
        _ = mlx.mlx_fast_metal_kernel_free(p.pass_after);
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
        switch (p.arm.mode) {
            .base, .ffn, .seg, .stop, .stopm => return false,
            else => {},
        }
        if (p.arm.at != at) return false;
        const want = p.arm.index orelse return true;
        return (p.m.layers[li].idx != null) == want;
    }

    fn before(ctx: *anyopaque, sc: *dsv41.Scope, m: *dsv41.Dsv41Model, li: usize, at: ProjAt, ins: []const A) anyerror!A {
        const p: *Probe = @ptrCast(@alignCast(ctx));
        p.remember(li, at, ins);
        if (!p.applies(li, at)) return ins[0];
        const dep = switch (p.arm.mode) {
            .base, .ctl, .ffn, .seg, .stop, .stopm => ins[0],
            .twin => try twinLaunch(sc, m, &p.copies[li - p.lo], li, at, ins),
            .same => try twinLaunch(sc, m, &m.layers[li], li, at, ins),
            .pre, .precopy => return passAfter(sc, p.pass_after, ins[0], try readLaunch(sc, p.read, weightsOf(if (p.arm.mode == .pre) &m.layers[li] else &p.copies[li - p.lo], at))),
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

fn passAfter(sc: *dsv41.Scope, k: mlx.mlx_fast_metal_kernel, x: A, dep: A) !A {
    return passWith(sc, k, x, &.{ dep, x });
}

fn passLaunch(sc: *dsv41.Scope, k: mlx.mlx_fast_metal_kernel, x: A, dep: A) !A {
    return passWith(sc, k, x, &.{ x, dep });
}

fn passWith(sc: *dsv41.Scope, k: mlx.mlx_fast_metal_kernel, x: A, ins: *const [2]A) !A {
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    const shape = mlx.getShape(x);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, shape.ptr, shape.len, mlx.mlx_array_dtype(x)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, @intCast(mlx.mlx_array_size(x)), 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 256, 1, 1));
    const iv = mlx.mlx_vector_array_new_data(ins, 2);
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

fn setUnder(m: *dsv41.Dsv41Model, u: Under) void {
    seg_ffn.read_mode = switch (u) {
        .read => |rm| rm,
        .moe => |mo| mo.read,
        else => .staged,
    };
    seg_ffn.moe_only = u == .moe;
    seg_ffn.moe_threads = switch (u) {
        .moe => |mo| mo.threads,
        else => 256,
    };
    seg.broken = switch (u) {
        .level, .read, .moe => .none,
        .broken => .zero_shared,
        .nan => .nan_shared,
    };
    m.fused_seg = switch (u) {
        .level => |l| l,
        .read, .moe => .ffn_seg,
        .broken, .nan => .off,
    };
}

fn applyArm(fx: *fixture.Fixture, arm: Arm) void {
    setUnder(fx.m, if (arm.mode == .seg or arm.mode == .stop or arm.mode == .stopm) arm.under else .{ .level = fx.opts.seg });
    seg_ffn.stop_after = if (arm.mode == .stop or arm.mode == .stopm) arm.stop else 4;
    seg_bounds.set(arm.ffn);
}

fn underName(buf: []u8, u: Under) []const u8 {
    return switch (u) {
        .level => |l| std.fmt.bufPrint(buf, "seg:{d}", .{@intFromEnum(l)}) catch "seg",
        .read => |rm| std.fmt.bufPrint(buf, "seg:2{s}", .{@tagName(rm)[0..1]}) catch "seg",
        .moe => |mo| std.fmt.bufPrint(buf, "seg:2m{d}{s}", .{ mo.threads, @tagName(mo.read)[0..1] }) catch "seg",
        .broken => "seg:broken",
        .nan => "seg:nan",
    };
}

const layer_rms_max = 5e-3;
const whole_seeds = 3;
const whole_layers = 6;
const whole_ratio = 1.15;

pub const wts_rel_tol = 1e-6;

const LayerStat = struct { layers: usize = 0, engaged: usize = 0, total: usize = 0, ndiff: usize = 0, nonfinite: usize = 0, route_rows: usize = 0, route_bad: usize = 0, wts_rel: f64 = 0, num: f64 = 0, den: f64 = 0, worst: f64 = 0, declined: ?[]const u8 = null };

fn dup(sc: *dsv41.Scope, a: A) !A {
    return sc.keep(sc.out(a));
}

fn routesOf(sc: *dsv41.Scope, li: usize) ![2]A {
    const c = seg_bounds.heldOf(li) orelse return error.NoCapture;
    return .{ try dup(sc, try sc.astype(c[1], .float32)), try dup(sc, try sc.astype(c[2], .float32)) };
}

/// Rows of `rows` x `topk` routes where the picks differ in any position (order counts, so a tie that went to the
/// higher index shows) or a weight moves by more than `wts_rel_tol` relative.
pub fn routeRowsBad(idx_a: []const f32, idx_b: []const f32, wts_a: []const f32, wts_b: []const f32, rows: usize, topk: usize) usize {
    var bad: usize = 0;
    for (0..rows) |r| {
        var row_bad = false;
        for (r * topk..(r + 1) * topk) |i| {
            const dw = @abs(wts_a[i] - wts_b[i]);
            row_bad = row_bad or idx_a[i] != idx_b[i] or !(dw <= wts_rel_tol * @max(@abs(wts_a[i]), 1e-30));
        }
        if (row_bad) bad += 1;
    }
    return bad;
}

fn wtsRelWorst(idx_a: []const f32, idx_b: []const f32, wts_a: []const f32, wts_b: []const f32, rows: usize, topk: usize) f64 {
    var worst: f64 = 0;
    for (0..rows) |r| {
        const lo = r * topk;
        if (!std.mem.eql(f32, idx_a[lo..][0..topk], idx_b[lo..][0..topk])) continue;
        for (wts_a[lo..][0..topk], wts_b[lo..][0..topk]) |u, v| worst = @max(worst, @abs(@as(f64, v) - u) / @max(@abs(@as(f64, u)), 1e-30));
    }
    return worst;
}

fn dumpRouteGap(gpa: std.mem.Allocator, ly: *const dsv41.Layer, li: usize, h_stock: []const f32, h_fused: []const f32, idx_a: []const f32, idx_b: []const f32, wts_a: []const f32, wts_b: []const f32, rows: usize) !void {
    const gate = try dsv41.hostF32(ly.gate_w);
    defer gpa.free(gate);
    const bias = try dsv41.hostF32(ly.gate_bias);
    defer gpa.free(bias);
    const e = bias.len;
    const d = gate.len / e;
    const topk = idx_a.len / rows;
    const order = try gpa.alloc(usize, e);
    defer gpa.free(order);
    const score = try gpa.alloc(f64, e);
    defer gpa.free(score);
    for (0..rows) |r| {
        const lo = r * topk;
        const same = std.mem.eql(f32, idx_a[lo..][0..topk], idx_b[lo..][0..topk]) and @abs(wts_a[lo] - wts_b[lo]) <= wts_rel_tol * @abs(wts_a[lo]);
        if (same) continue;
        std.debug.print("[kmap] route flip layer {d} row {d}: stock picks {any} fused picks {any}; stock weights {any} fused weights {any}\n", .{ li, r, idx_a[lo..][0..topk], idx_b[lo..][0..topk], wts_a[lo..][0..topk], wts_b[lo..][0..topk] });
        const same_h = std.mem.eql(f32, h_stock[r * d ..][0..d], h_fused[r * d ..][0..d]);
        std.debug.print("[kmap] route flip: h2 of the two paths {s}\n", .{if (same_h) "identical" else "differ"});
        for ([_][]const f32{ h_stock, h_fused }, [_][]const u8{ "stock h2", "fused h2" }) |h, name| {
            for (0..e) |x| {
                var z: f64 = 0;
                for (0..d) |c| z += @as(f64, h[r * d + c]) * @as(f64, gate[x * d + c]);
                const sp = if (z > 0) z + @log(1.0 + @exp(-z)) else @log(1.0 + @exp(z));
                score[x] = @sqrt(sp) + @as(f64, bias[x]);
                order[x] = x;
            }
            std.mem.sort(usize, order, score, struct {
                fn gt(sc_: []f64, a: usize, b: usize) bool {
                    return sc_[a] > sc_[b] or (sc_[a] == sc_[b] and a < b);
                }
            }.gt);
            std.debug.print("[kmap] route flip, f64 ranking from {s}:", .{name});
            for (order[0 .. topk + 3], 0..) |x, rank| std.debug.print(" #{d} e{d} {d:.9}", .{ rank + 1, x, score[x] });
            const g = score[order[topk - 1]] - score[order[topk]];
            std.debug.print("\n[kmap] route flip, gap between rank {d} and {d}: {e:.3} absolute, {e:.3} relative to rank {d}\n", .{ topk, topk + 1, g, g / @abs(score[order[topk - 1]]), topk });
            if (same_h) break;
        }
    }
}

fn layerStat(gpa: std.mem.Allocator, fx: *fixture.Fixture, lo: usize, hi: usize, under: Under) !LayerStat {
    const m = fx.m;
    var st: LayerStat = .{};
    for (lo..hi) |li| {
        const c = seg_bounds.inOf(li) orelse continue;
        var sc = dsv41.Scope.init(gpa, m.s);
        defer sc.deinit();
        const x = try dup(&sc, c[0]);
        const ao = try dup(&sc, c[1]);
        const sa: dsv41.Split = .{ .pre = try dup(&sc, c[2]), .post = try dup(&sc, c[3]), .comb = try dup(&sc, c[4]) };
        var pre_a = x;
        var pre_b = x;
        const rows: usize = @intCast(mlx.getShape(x)[0]);
        setUnder(m, .{ .level = .off });
        const ya = try dsv41.tpFfn(&sc, m, li, x, ao, &sa, &pre_a);
        const ra = try routesOf(&sc, li);
        const ha = try dup(&sc, try sc.astype(seg_bounds.heldOf(li).?[0], .float32));
        setUnder(m, under);
        const wanted = switch (under) {
            .level => |l| l != .off and seg.on(m, li, .ffn, rows, .ffn_entry),
            .read, .moe => seg.on(m, li, .ffn, rows, .ffn_entry),
            .broken, .nan => false,
        };
        const launched = seg.dispatches;
        const yb = try dsv41.tpFfn(&sc, m, li, x, ao, &sa, &pre_b);
        if (wanted and seg.dispatches > launched) st.engaged += 1 else if (st.declined == null) st.declined = seg.serves(m, rows) orelse if (wanted) "the segment kernel declined" else "layer not covered";
        const rb = try routesOf(&sc, li);
        const hb = try dup(&sc, try sc.astype(seg_bounds.heldOf(li).?[0], .float32));
        const af = try sc.astype(ya, .float32);
        const bf = try sc.astype(yb, .float32);
        try dsv41.evalAll(&.{ af, bf, ra[0], ra[1], rb[0], rb[1], ha, hb });
        const n = mlx.mlx_array_size(af);
        const xa = mlx.mlx_array_data_float32(af).?[0..n];
        const xb = mlx.mlx_array_data_float32(bf).?[0..n];
        for (xa, xb) |u, v| {
            if (!std.math.isFinite(u) or !std.math.isFinite(v)) st.nonfinite += 1;
            if (u != v) st.ndiff += 1;
            const d = @as(f64, u) - @as(f64, v);
            st.num += d * d;
            st.den += @as(f64, u) * @as(f64, u);
            st.worst = @max(st.worst, @abs(d));
        }
        const k = mlx.mlx_array_size(ra[0]);
        const idx_a = mlx.mlx_array_data_float32(ra[0]).?[0..k];
        const idx_b = mlx.mlx_array_data_float32(rb[0]).?[0..k];
        const wts_a = mlx.mlx_array_data_float32(ra[1]).?[0..k];
        const wts_b = mlx.mlx_array_data_float32(rb[1]).?[0..k];
        const bad = routeRowsBad(idx_a, idx_b, wts_a, wts_b, rows, k / rows);
        if (bad != 0) try dumpRouteGap(gpa, &m.layers[li], li, mlx.mlx_array_data_float32(ha).?[0..mlx.mlx_array_size(ha)], mlx.mlx_array_data_float32(hb).?[0..mlx.mlx_array_size(hb)], idx_a, idx_b, wts_a, wts_b, rows);
        st.route_bad += bad;
        st.wts_rel = @max(st.wts_rel, wtsRelWorst(idx_a, idx_b, wts_a, wts_b, rows, k / rows));
        st.route_rows += rows;
        st.total += n;
        st.layers += 1;
    }
    return st;
}

const Whole = struct { stock: f64, arm: f64 };

fn wholeStep(gpa: std.mem.Allocator, fx: *fixture.Fixture, lo: usize, hi: usize, under: Under) !Whole {
    const captured = seg_bounds.cache_on;
    seg_bounds.cache_on = false;
    defer seg_bounds.cache_on = captured;
    var stock: [whole_seeds]f64 = undefined;
    var armed: [whole_seeds]f64 = undefined;
    for (&stock, &armed, 1..) |*e_stock, *e_arm, seed| {
        var o = fx.opts;
        o.lo = lo;
        o.hi = @min(hi, lo + whole_layers);
        o.experts = @min(o.experts, 16);
        o.ctx = 0;
        o.seed = seed;
        o.fused = true;
        o.act = .bfloat16;
        o.seg = .off;
        var twin = o;
        twin.act = .float32;
        twin.fused = false;
        const truth = try fixture.Fixture.init(gpa, twin);
        defer truth.deinit();
        const run = try fixture.Fixture.init(gpa, o);
        defer run.deinit();
        const in = try fixture.Input.of(run, 1);
        defer in.deinit();
        const t = try fixture.streamOf(gpa, truth, in);
        defer gpa.free(t);
        defer setUnder(run.m, .{ .level = .off });
        const s = try fixture.streamOf(gpa, run, in);
        defer gpa.free(s);
        setUnder(run.m, under);
        const a = try fixture.streamOf(gpa, run, in);
        defer gpa.free(a);
        e_stock.* = decode.relRms(s, t);
        e_arm.* = decode.relRms(a, t);
    }
    std.mem.sort(f64, &stock, {}, std.sort.asc(f64));
    std.mem.sort(f64, &armed, {}, std.sort.asc(f64));
    return .{ .stock = stock[whole_seeds / 2], .arm = armed[whole_seeds / 2] };
}

pub fn segCheck(gpa: std.mem.Allocator, fx: *fixture.Fixture, lo: usize, hi: usize, under: Under) !void {
    defer applyArm(fx, .{});
    seg_bounds.set(.base);
    var buf: [16]u8 = undefined;
    const name = underName(&buf, under);
    const st = try layerStat(gpa, fx, lo, hi, under);
    const needs_seg = switch (under) {
        .level => |l| l != .off,
        .read, .moe => true,
        .broken, .nan => false,
    };
    if (st.layers > 0 and needs_seg and st.engaged == 0) {
        std.debug.print("SEG-CHECK-SKIP {s}: no captured layer takes the segment path ({s})\n", .{ name, st.declined.? });
        return error.SegCheckSkipped;
    }
    const whole = try wholeStep(gpa, fx, lo, hi, under);
    const rms = @sqrt(st.num / @max(st.den, 1e-30));
    std.debug.print("[kmap] {s} vs stock: {d} layers ({d} on the segment path), {d} of {d} stream values differ ({d} non-finite), rel rms {e:.3}, max abs {e:.3}, route rows differing {d} of {d} (largest weight difference {e:.2}); whole step from the f32 model (median of {d} seeds): stock {e:.3}, {s} {e:.3}\n", .{ name, st.layers, st.engaged, st.ndiff, st.total, st.nonfinite, rms, st.worst, st.route_bad, st.route_rows, st.wts_rel, whole_seeds, whole.stock, name, whole.arm });
    const why: ?[]const u8 = if (st.layers == 0)
        "no layer captured"
    else if (st.nonfinite != 0)
        "non-finite output"
    else if (st.route_bad != 0)
        "route rows differ"
    else if (!(rms <= layer_rms_max))
        "layer outputs differ from stock"
    else if (!(whole.arm <= whole_ratio * whole.stock + 1e-4))
        "whole step further from the f32 model than stock"
    else
        null;
    if (why) |w| {
        std.debug.print("SEG-CHECK-FAIL {s}: {s}\n", .{ name, w });
        return error.SegCheckFailed;
    }
    std.debug.print("SEG-CHECK-OK {s}\n", .{name});
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

fn printBuild(ab: *const Ab, rows: usize, us: []const f64) void {
    const blocks = us.len / ab.n;
    var by: [max_arms][max_steps]f64 = undefined;
    for (0..blocks) |b| for (0..ab.n) |pos| {
        const k = b * ab.n + pos;
        by[ab.armOf(k)][b] = us[k];
    };
    for (0..ab.n) |j| std.debug.print("[kmap] arm {s} #{d} rows={d}: host build median {d:.0} us\n", .{ ab.name, j, rows, quantileOf(by[j][0..blocks], 0.5) });
}

fn quantileOf(xs: []f64, f: f64) f64 {
    std.mem.sort(f64, xs, {}, std.sort.asc(f64));
    return xs[@intFromFloat(f * @as(f64, @floatFromInt(xs.len - 1)))];
}

const max_steps = 1024;

/// One row count: 3 warm-up steps and `steps` timed ones from position `p0`, which moves past them.
fn phase(gpa: std.mem.Allocator, fx: *fixture.Fixture, rows: usize, steps: usize, p0: *usize, ab: ?*const Ab) !void {
    const warm = 3 + if (ab) |a| 2 * a.n else 0;
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
    var bus: [max_steps]f64 = undefined;
    std.debug.assert(steps <= us.len);
    sleepSecs(2000);
    waitForGate();
    for (0..total) |k| {
        if (k == warm) sleepSecs(500);
        if (ab) |a| {
            a.probe.arm = a.arms[if (k < warm) k % a.n else a.armOf(k - warm)];
            applyArm(fx, a.probe.arm);
        }
        const s0 = nowUs();
        const out = try dsv41.workerStep(fx.m, gpa, xs[k], pres[k], p0.*, false, false);
        for (out) |o| _ = mlx.mlx_array_free(o);
        if (k >= warm) {
            us[k - warm] = @floatFromInt(nowUs() - s0);
            bus[k - warm] = @floatFromInt(seg_bounds.build_us);
        }
        p0.* += rows;
    }
    sleepSecs(500);
    if (ab) |a| {
        applyArm(fx, .{});
        const pair = fx.m.pair.?;
        for (a.arms[0..a.n]) |arm| if (arm.mode == .seg) segCheck(gpa, fx, pair.lo, pair.hi, arm.under) catch |e| {
            if (e != error.SegCheckSkipped) return e;
        };
        printArms(a, rows, us[0 .. steps - steps % a.n]);
        printBuild(a, rows, bus[0 .. steps - steps % a.n]);
    }
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
