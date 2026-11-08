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

    const o = try KmapOpts.fromEnv();
    const steps = o.steps;
    const rows_list = std.mem.span(std.c.getenv("DSV41_KMAP_ROWS") orelse "1,6");
    const lo = o.lo;
    const hi = o.hi;
    const fx = try o.build(gpa, 1, null);
    defer fx.deinit();

    waitForGo();
    var p0: usize = o.ctx;
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

const KmapOpts = struct {
    steps: usize,
    ctx: usize,
    tp: usize,
    experts: u32,
    lo: usize,
    hi: usize,

    fn fromEnv() !KmapOpts {
        var o: KmapOpts = .{ .steps = envInt("DSV41_KMAP_STEPS", 32), .ctx = envInt("DSV41_KMAP_CTX", 2048), .tp = envInt("DSV41_KMAP_TP", 4), .experts = @intCast(envInt("DSV41_KMAP_EXPERTS", 384)), .lo = 0, .hi = 40 };
        if (std.c.getenv("DSV41_KMAP_LAYERS")) |raw| {
            var it = std.mem.tokenizeScalar(u8, std.mem.span(raw), '-');
            o.lo = try std.fmt.parseInt(usize, it.next() orelse "0", 10);
            o.hi = try std.fmt.parseInt(usize, it.next() orelse "40", 10);
        }
        return o;
    }

    fn build(o: KmapOpts, gpa: std.mem.Allocator, seed: u64, stream: ?mlx.mlx_stream) !*fixture.Fixture {
        const t0 = nowUs();
        const fx = try fixture.Fixture.init(gpa, .{
            .lo = o.lo,
            .hi = o.hi,
            .experts = o.experts,
            .ctx = o.ctx,
            .seed = seed,
            .fused = true,
            .experts_fp4 = true,
            .tp = .{ .rank = 0, .size = @intCast(o.tp) },
            .stream = stream,
        });
        std.debug.print("[kmap] fixture tp={d} layers={d}-{d} experts={d} ctx={d} seed={d} built in {d:.0} s\n", .{ o.tp, o.lo, o.hi, o.experts, o.ctx, seed, @as(f64, @floatFromInt(nowUs() - t0)) / 1e6 });
        _ = mlx.mlx_clear_cache();
        memLine("after the fixture");
        return fx;
    }
};

fn memLine(tag: []const u8) void {
    var active: usize = 0;
    var peak: usize = 0;
    var cache: usize = 0;
    _ = mlx.mlx_get_active_memory(&active);
    _ = mlx.mlx_get_peak_memory(&peak);
    _ = mlx.mlx_get_cache_memory(&cache);
    std.debug.print("[kmap] memory {s}: active {d:.1} GB, cache {d:.1} GB, peak {d:.1} GB\n", .{ tag, @as(f64, @floatFromInt(active)) / 1e9, @as(f64, @floatFromInt(cache)) / 1e9, @as(f64, @floatFromInt(peak)) / 1e9 });
}

const ProjAt = dsv41.ProjAt;
const Mode = enum { base, ctl, twin, same, pre, precopy, sk };
const Stage = enum { hc, router, norm, attn, moe, proj };
const n_stages = std.enums.values(Stage).len;
const Skip = std.EnumSet(Stage);
const Arm = struct { mode: Mode = .base, at: ProjAt = .in_proj, index: ?bool = null, skip: Skip = .empty, raw: bool = false };

fn parseSkip(what: []const u8) !Arm {
    if (std.mem.eql(u8, what, "none")) return .{ .mode = .sk };
    if (std.mem.eql(u8, what, "raw")) return .{ .mode = .sk, .raw = true };
    if (std.mem.eql(u8, what, "all-small")) return .{ .mode = .sk, .skip = .initMany(&.{ .hc, .router, .norm }) };
    return .{ .mode = .sk, .skip = .initOne(std.meta.stringToEnum(Stage, what) orelse return error.BadArm) };
}

fn parseArm(tok: []const u8) !Arm {
    var it = std.mem.tokenizeAny(u8, tok, ":@");
    const mode = std.meta.stringToEnum(Mode, it.next() orelse "base") orelse return error.BadArm;
    if (mode == .base) return .{};
    if (mode == .sk) return parseSkip(it.next() orelse return error.BadArm);
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
        return armOfN(k, ab.n);
    }

    fn hasSk(ab: *const Ab) bool {
        for (ab.arms[0..ab.n]) |a| if (a.mode == .sk) return true;
        return false;
    }
};

fn armOfN(k: usize, n: usize) usize {
    return (k % n + k / n) % n;
}

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
    caps: [cap_sets][max_layers]Cap = @splat(@splat(.{})),
    ids: [n_ids][max_layers]A = @splat(@splat(none)),
    cap_rows: usize = 0,
    capturing: ?usize = null,
    step: usize = 0,
    arm_ix: usize = 0,
    fired: [max_arms][n_stages]usize = @splat(@splat(0)),
    calls: [max_arms]usize = @splat(0),
    last_entry: i64 = 0,
    stands: [max_outs + 1][max_deps + 1]?mlx.mlx_fast_metal_kernel = @splat(@splat(null)),

    fn init(gpa: std.mem.Allocator, fx: *fixture.Fixture, lo: usize, hi: usize) !*Probe {
        std.debug.assert(hi <= max_layers);
        const p = try gpa.create(Probe);
        errdefer gpa.destroy(p);
        p.* = .{ .gpa = gpa, .m = fx.m, .lo = lo, .hi = hi, .held = dsv41.Scope.init(gpa, fx.m.s), .copies = try gpa.alloc(dsv41.Layer, hi - lo), .pass = try dsv41.buildKernel("dsv41_kmap_pass", &.{ "x", "dep" }, &.{"y"}, PASS_SOURCE, ""), .pass_after = try dsv41.buildKernel("dsv41_kmap_pass_after", &.{ "dep", "x" }, &.{"y"}, PASS_SOURCE, ""), .read = try dsv41.buildKernel("dsv41_kmap_read", &.{ "w", "s" }, &.{"out"}, READ_SOURCE, ""), .hook = .{ .ctx = p, .before = &before, .layer = &layerHook } };
        for (p.copies, lo..) |*c, li| c.* = try decode.layerCopy(&p.held, fx.layer(li));
        return p;
    }

    fn deinit(p: *Probe) void {
        for (&p.seen) |*row| for (row) |*ins| forget(ins);
        for (&p.caps) |*set| for (set) |*c| for (c.slots()) |a| drop(a);
        for (&p.ids) |*set| for (set) |*a| drop(a);
        for (&p.stands) |*row| for (row) |k| if (k) |kk| {
            _ = mlx.mlx_fast_metal_kernel_free(kk);
        };
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
        if (p.arm.mode == .base or p.arm.mode == .sk or p.arm.at != at) return false;
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
            .pre, .precopy => return passAfter(sc, p.pass_after, ins[0], try readLaunch(sc, p.read, weightsOf(if (p.arm.mode == .pre) &m.layers[li] else &p.copies[li - p.lo], at))),
            .sk => return ins[0],
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

    fn resetCounts(p: *Probe) void {
        p.fired = @splat(@splat(0));
        p.calls = @splat(0);
    }

    fn fire(p: *Probe, st: Stage) void {
        if (p.capturing == null) p.fired[p.arm_ix][@intFromEnum(st)] += 1;
    }

    fn layerHook(ctx: *anyopaque, sc: *dsv41.Scope, m: *dsv41.Dsv41Model, st: *dsv41.State, li: usize, x: A, pre: *A, p0: usize, sh: *dsv41.Shared) anyerror!?A {
        const p: *Probe = @ptrCast(@alignCast(ctx));
        if (li + 1 == p.hi) p.last_entry = nowUs();
        if (p.capturing) |set| return try p.layerCopy(sc, m, st, li, x, pre, p0, sh, .{ .mode = .sk }, &p.caps[set][li]);
        if (p.arm.mode != .sk) return null;
        p.calls[p.arm_ix] += 1;
        return try p.layerCopy(sc, m, st, li, x, pre, p0, sh, p.arm, &p.caps[p.step % cap_sets][li]);
    }

    fn layerCopy(p: *Probe, sc: *dsv41.Scope, m: *dsv41.Dsv41Model, st: *dsv41.State, li: usize, x: A, pre: *A, p0: usize, sh: *dsv41.Shared, arm: Arm, c: *Cap) !A {
        if (m.tp == null) return error.KmapNeedsTp;
        const kp = &(m.dec.in_proj orelse return error.Declined);
        const kq = m.dec.wq_b orelse return error.Declined;
        const ka = m.dec.attn orelse return error.Declined;
        const kw = m.dec.wo_a orelse return error.Declined;
        const ly = &m.layers[li];
        const ls = &st.layers[li];
        const L: usize = @intCast(mlx.getShape(x)[0]);
        const sk = arm.skip;
        const rec = p.capturing != null;
        var w: Proj = undefined;
        if (sk.contains(.proj)) {
            w = try p.projStandIn(sc, c, x, pre.*);
        } else {
            const a = (try decode.in_proj.run(sc, kp, m, ly, x, pre.*)) orelse return error.Declined;
            const wb = (try decode.wq_b.run(sc, m, kq, ly, li, a.qa, a.kv, a.z, try decode.wq_b.rowsAt(sc, p0, L))) orelse return error.Declined;
            w = .{ .q = wb.q, .kv_row = wb.kv_row, .sa = wb.sa, .q_idx = wb.q_idx orelse none, .iw = a.iw orelse none, .comp = a.comp orelse none };
            if (rec) c.putProj(w);
        }
        if (sk.contains(.norm)) p.fire(.norm) else try ls.win.write(w.kv_row, p0, sc.s);
        var comp_rows = none;
        var topk = none;
        if (ly.ratio > 0) {
            const clen = (p0 + L) / ly.ratio;
            if (ly.comp) |*cw| if (!try dsv41.compressFused(sc, m, st, li, w.comp, p0)) try dsv41.compressFrom(sc, m, st, li, cw, w.comp, p0);
            if (ly.idx) |_| {
                sh.topk = if (clen == 0) none else if (sk.contains(.attn)) c.topk else try dsv41.indexerQ(sc, m, st, li, w.iw, w.q_idx, p0, sh);
                if (rec) hold(&c.topk, sh.topk);
            }
            topk = sh.topk;
            if (clen > 0 and topk.ctx != null) comp_rows = try sc.keep(try st.layers[m.latest(li, dsv41.Dsv41Model.isKvSource)].comp.?.view(clen, sc.s));
        }
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(m.cfg.head_dim)));
        var pa: [3]A = undefined;
        if (sk.contains(.attn)) {
            const o = try p.stand(sc, &.{ c.pm, c.pl }, &.{ w.q, ls.win.buf });
            pa = .{ c.acc, o[0], o[1] };
            p.fire(.attn);
        } else {
            const r = (try decode.attn.partials(sc, ka, m, w.q, ls.win, comp_rows, topk, p0, scale)) orelse return error.Declined;
            pa = .{ r.acc, r.m, r.l };
            if (rec) for ([_]*A{ &c.acc, &c.pm, &c.pl }, pa) |slot, a| hold(slot, a);
        }
        var ao: A = undefined;
        if (sk.contains(.proj)) {
            ao = (try p.stand(sc, &.{c.ao}, &pa))[0];
        } else {
            const meta = try dsv41.attnMeta(sc, m, w.q, ls.win, comp_rows, topk, p0);
            const oa = (try decode.wo_a.apply(sc, m, ly, kw, m.dec.wo_a_counter, dsv41.ropeFor(m, li).inv, meta, pa[0], pa[1], pa[2])) orelse return error.Declined;
            ao = try ly.wo_b.apply(sc, oa);
            if (rec) hold(&c.ao, ao);
        }
        const ys = try dsv41.tpSum(sc, m, ao);
        var hs: dsv41.Staged = undefined;
        var idx: A = undefined;
        var wts: A = undefined;
        if (!sk.contains(.hc) and !sk.contains(.router) and !arm.raw) {
            var o: [7]A = undefined;
            try dsv41.runRegion(sc, m, li, .ffn_pre, &.{ x, ys, w.sa.pre, w.sa.post, w.sa.comb }, &o);
            hs = .{ .x1 = o[0], .h = o[4], .sp = .{ .pre = o[1], .post = o[2], .comb = o[3] } };
            idx = o[5];
            wts = o[6];
        } else {
            if (sk.contains(.hc)) {
                const o = try p.stand(sc, &c.hc, &.{ x, ys, w.sa.pre, w.sa.post, w.sa.comb });
                hs = .{ .x1 = o[0], .h = o[1], .sp = .{ .pre = o[2], .post = o[3], .comb = o[4] } };
                p.fire(.hc);
            } else hs = try dsv41.hcStage(sc, m, x, .{ .y = ys, .sp = &w.sa }, w.sa.pre, &ly.hc_ffn, ly.ffn_norm);
            if (sk.contains(.router)) {
                idx = p.ids[p.step % n_ids][li];
                wts = c.wts;
                p.fire(.router);
            } else {
                const r = try dsv41.route(sc, m, ly, hs.h, none);
                idx = r.idx;
                wts = r.wts;
            }
        }
        if (rec) for ([_]*A{ &c.hc[0], &c.hc[1], &c.hc[2], &c.hc[3], &c.hc[4], &c.wts }, [_]A{ hs.x1, hs.h, hs.sp.pre, hs.sp.post, hs.sp.comb, wts }) |slot, a| hold(slot, a);
        var part: A = undefined;
        if (sk.contains(.moe)) {
            part = c.part;
            p.fire(.moe);
        } else part = (try dsv41.moeFused(sc, m, ly, hs.h, idx, wts)) orelse return error.Declined;
        if (rec) hold(&c.part, part);
        const y = try dsv41.tpSum(sc, m, part);
        const xo = if (sk.contains(.hc)) (try p.stand(sc, &.{c.xo}, &.{ hs.x1, y, hs.sp.post, hs.sp.comb }))[0] else try dsv41.hcExpand(sc, hs.x1, y, &hs.sp);
        if (rec) hold(&c.xo, xo);
        pre.* = hs.sp.pre;
        return xo;
    }

    fn projStandIn(p: *Probe, sc: *dsv41.Scope, c: *const Cap, x: A, pre: A) !Proj {
        var srcs: [max_outs]A = undefined;
        var n: usize = 0;
        for ([_]A{ c.q, c.kv_row, c.sa[0], c.sa[1], c.sa[2], c.q_idx, c.iw, c.comp }) |a| if (a.ctx != null) {
            srcs[n] = a;
            n += 1;
        };
        const o = try p.stand(sc, srcs[0..n], &.{ x, pre });
        p.fire(.proj);
        var w: Proj = .{ .q = o[0], .kv_row = o[1], .sa = .{ .pre = o[2], .post = o[3], .comb = o[4] }, .q_idx = none, .iw = none, .comp = none };
        var k: usize = 5;
        inline for (.{ "q_idx", "iw", "comp" }) |f| if (@field(c, f).ctx != null) {
            @field(w, f) = o[k];
            k += 1;
        };
        return w;
    }

    fn standKernel(p: *Probe, n: usize, d: usize) !mlx.mlx_fast_metal_kernel {
        if (p.stands[n][d]) |k| return k;
        var src: [1024]u8 = undefined;
        var len = (try std.fmt.bufPrint(&src, "    const uint i = thread_position_in_grid.x;\n", .{})).len;
        for (0..n) |j| len += (try std.fmt.bufPrint(src[len..], "    if (i < N{d}) out{d}[i] = src{d}[i];\n", .{ j, j, j })).len;
        src[len] = 0;
        var ins: [max_outs + max_deps][*:0]const u8 = undefined;
        for (0..n) |j| ins[j] = src_names[j];
        for (0..d) |j| ins[n + j] = dep_names[j];
        var nb: [48]u8 = undefined;
        const name = try std.fmt.bufPrintSentinel(&nb, "dsv41_kmap_stand_{d}_{d}", .{ n, d }, 0);
        const iv = mlx.mlx_vector_string_new_data(&ins, n + d);
        defer _ = mlx.mlx_vector_string_free(iv);
        const ov = mlx.mlx_vector_string_new_data(&out_names, n);
        defer _ = mlx.mlx_vector_string_free(ov);
        const k = mlx.mlx_fast_metal_kernel_new(name, iv, ov, src[0..len :0], "", false, false);
        if (k.ctx == null) return error.MetalKernelCompileFailed;
        p.stands[n][d] = k;
        return k;
    }

    fn stand(p: *Probe, sc: *dsv41.Scope, srcs: []const A, deps: []const A) ![max_outs]A {
        const k = try p.standKernel(srcs.len, deps.len);
        const cfg = mlx.mlx_fast_metal_kernel_config_new();
        defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
        var most: usize = 1;
        var ins: [max_outs + max_deps]A = undefined;
        for (srcs, 0..) |a, j| {
            const sh = mlx.getShape(a);
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, sh.ptr, sh.len, mlx.mlx_array_dtype(a)));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, tmpl_names[j], @intCast(mlx.mlx_array_size(a))));
            most = @max(most, mlx.mlx_array_size(a));
            ins[j] = a;
        }
        for (deps, 0..) |a, j| ins[srcs.len + j] = a;
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, @intCast(most), 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, @intCast(@min(most, 256)), 1, 1));
        const iv = mlx.mlx_vector_array_new_data(&ins, srcs.len + deps.len);
        defer _ = mlx.mlx_vector_array_free(iv);
        var ov = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(ov);
        try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, k, iv, cfg, sc.s));
        var outs: [max_outs]A = @splat(none);
        for (outs[0..srcs.len], 0..) |*o, j| {
            var a = mlx.mlx_array_new();
            o.* = try sc.res(mlx.mlx_vector_array_get(&a, ov, j), &a);
        }
        return outs;
    }

    fn capture(p: *Probe, gpa: std.mem.Allocator, fx: *fixture.Fixture, rows: usize, p0: *usize) !void {
        for (0..cap_sets) |set| {
            const x = try fx.stream(rows);
            defer _ = mlx.mlx_array_free(x);
            const pre = try fx.cpre(rows);
            defer _ = mlx.mlx_array_free(pre);
            p.capturing = set;
            defer p.capturing = null;
            const out = try dsv41.workerStep(fx.m, gpa, x, pre, p0.*, false, false);
            for (out) |o| _ = mlx.mlx_array_free(o);
            p0.* += rows;
        }
        var all: std.ArrayList(A) = .empty;
        defer all.deinit(gpa);
        for (&p.caps) |*set| for (set[p.lo..p.hi]) |*c| for (c.slots()) |a| if (a.ctx != null) {
            var o = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_contiguous(&o, a.*, false, p.m.s));
            _ = mlx.mlx_array_free(a.*);
            a.* = o;
            try all.append(gpa, o);
        };
        try p.makeIds(rows);
        for (&p.ids) |*set| for (set[p.lo..p.hi]) |a| try all.append(gpa, a);
        try dsv41.evalAll(all.items);
        p.cap_rows = rows;
        std.debug.print("[kmap] captured {d} steps of stage outputs at rows={d}\n", .{ cap_sets, rows });
    }

    fn makeIds(p: *Probe, rows: usize) !void {
        var rng = std.Random.DefaultPrng.init(0x5eed);
        const r = rng.random();
        var buf: [decode.max_rows * 16]u32 = undefined;
        for (&p.ids) |*set| for (p.lo..p.hi) |li| {
            const ly = &p.m.layers[li];
            const e: u32 = @intCast(mlx.getShape(ly.gate_w)[0]);
            const k: usize = @intCast(ly.top_k);
            std.debug.assert(k <= 16 and rows <= decode.max_rows);
            for (0..rows) |row| for (0..k) |j| {
                var v = r.uintLessThan(u32, e);
                while (std.mem.indexOfScalar(u32, buf[row * k .. row * k + j], v) != null) v = r.uintLessThan(u32, e);
                buf[row * k + j] = v;
            };
            drop(&set[li]);
            set[li] = mlx.mlx_array_new_data(&buf, &[_]c_int{ @intCast(rows), @intCast(k) }, 2, .uint32);
        };
    }
};

const cap_sets = 4;
const n_ids = 64;
const max_outs = 8;
const max_deps = 5;
const src_names = [_][*:0]const u8{ "src0", "src1", "src2", "src3", "src4", "src5", "src6", "src7" };
const dep_names = [_][*:0]const u8{ "dep0", "dep1", "dep2", "dep3", "dep4" };
const out_names = [_][*:0]const u8{ "out0", "out1", "out2", "out3", "out4", "out5", "out6", "out7" };
const tmpl_names = [_][*:0]const u8{ "N0", "N1", "N2", "N3", "N4", "N5", "N6", "N7" };

const Proj = struct { q: A, kv_row: A, sa: dsv41.Split, q_idx: A, iw: A, comp: A };

const Cap = struct {
    q: A = none,
    kv_row: A = none,
    sa: [3]A = @splat(none),
    q_idx: A = none,
    iw: A = none,
    comp: A = none,
    acc: A = none,
    pm: A = none,
    pl: A = none,
    topk: A = none,
    ao: A = none,
    hc: [5]A = @splat(none),
    wts: A = none,
    part: A = none,
    xo: A = none,

    fn slots(c: *Cap) [21]*A {
        return .{ &c.q, &c.kv_row, &c.sa[0], &c.sa[1], &c.sa[2], &c.q_idx, &c.iw, &c.comp, &c.acc, &c.pm, &c.pl, &c.topk, &c.ao, &c.hc[0], &c.hc[1], &c.hc[2], &c.hc[3], &c.hc[4], &c.wts, &c.part, &c.xo };
    }

    fn putProj(c: *Cap, w: Proj) void {
        for ([_]*A{ &c.q, &c.kv_row, &c.sa[0], &c.sa[1], &c.sa[2], &c.q_idx, &c.iw, &c.comp }, [_]A{ w.q, w.kv_row, w.sa.pre, w.sa.post, w.sa.comb, w.q_idx, w.iw, w.comp }) |slot, a| hold(slot, a);
    }
};

fn drop(a: *A) void {
    if (a.ctx != null) _ = mlx.mlx_array_free(a.*);
    a.* = none;
}

fn hold(slot: *A, x: A) void {
    if (slot.ctx == x.ctx) return;
    var o = none;
    if (x.ctx != null) {
        o = mlx.mlx_array_new();
        _ = mlx.mlx_array_set(&o, x);
    }
    drop(slot);
    slot.* = o;
}


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

fn printArms(ab: *const Ab, rows: usize, us: []const f64, host: []const f64) void {
    printBlocks(ab.name, ab.n, rows, us, host);
    for (ab.arms[0..ab.n], 0..) |arm, j| if (arm.mode == .sk) {
        const f = ab.probe.fired[j];
        std.debug.print("[kmap] arm {s} #{d} stand-ins over {d} layer calls: hc {d} router {d} norm {d} attn {d} moe {d} proj {d}\n", .{ ab.name, j, ab.probe.calls[j], f[0], f[1], f[2], f[3], f[4], f[5] });
    };
}

fn printBlocks(name: []const u8, n: usize, rows: usize, us: []const f64, host: []const f64) void {
    const blocks = us.len / n;
    var by: [max_arms][max_steps]f64 = undefined;
    var hb: [max_arms][max_steps]f64 = undefined;
    var diff: [max_arms][max_steps]f64 = undefined;
    for (0..blocks) |b| for (0..n) |pos| {
        const k = b * n + pos;
        by[armOfN(k, n)][b] = us[k];
        hb[armOfN(k, n)][b] = host[k];
    };
    for (0..blocks) |b| for (0..n) |j| {
        diff[j][b] = by[j][b] - by[0][b];
    };
    for (0..n) |j| {
        const d = quantileOf(diff[j][0..blocks], 0.5);
        const d25 = quantileOf(diff[j][0..blocks], 0.25);
        const d75 = quantileOf(diff[j][0..blocks], 0.75);
        const q10 = quantileOf(by[j][0..blocks], 0.1);
        const q25 = quantileOf(by[j][0..blocks], 0.25);
        const med = quantileOf(by[j][0..blocks], 0.5);
        const h = quantileOf(hb[j][0..blocks], 0.5);
        std.debug.print("[kmap] arm {s} #{d} rows={d} blocks={d}: q10 {d:.0} q25 {d:.0} median {d:.0} us, block diff from arm 0 median {d:.0} us (q25 {d:.0} q75 {d:.0}), host at the last layer {d:.0} us\n", .{ name, j, rows, blocks, q10, q25, med, d, d25, d75, h });
    }
    var line: [max_steps * 8]u8 = undefined;
    var len: usize = 0;
    for (us, 0..) |u, k| {
        const piece = std.fmt.bufPrint(line[len..], "{d}:{d:.0} ", .{ armOfN(k, n), u }) catch break;
        len += piece.len;
    }
    std.debug.print("[kmap] raw {s} rows={d}: {s}\n", .{ name, rows, line[0..len] });
}

fn blockRatio(us: []const f64, n: usize, num: usize, den: usize) f64 {
    var r: [max_steps]f64 = undefined;
    const blocks = us.len / n;
    var at: [max_arms]f64 = undefined;
    for (0..blocks) |b| {
        for (0..n) |pos| at[armOfN(b * n + pos, n)] = us[b * n + pos];
        r[b] = at[num] / at[den];
    }
    return quantileOf(r[0..blocks], 0.5);
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
    if (ab) |a| if (a.hasSk() and a.probe.cap_rows != rows) try a.probe.capture(gpa, fx, rows, p0);

    var us: [max_steps]f64 = undefined;
    var host: [max_steps]f64 = undefined;
    std.debug.assert(steps <= us.len);
    sleepSecs(2000);
    waitForGate();
    for (0..total) |k| {
        if (k == warm) sleepSecs(500);
        if (ab) |a| {
            const j = a.armOf(k -| warm);
            a.probe.arm = a.arms[j];
            a.probe.arm_ix = j;
            a.probe.step = k;
            if (k == warm) a.probe.resetCounts();
        }
        const s0 = nowUs();
        const out = try dsv41.workerStep(fx.m, gpa, xs[k], pres[k], p0.*, false, false);
        for (out) |o| _ = mlx.mlx_array_free(o);
        if (k >= warm) {
            us[k - warm] = @floatFromInt(nowUs() - s0);
            host[k - warm] = if (ab) |a| @floatFromInt(a.probe.last_entry - s0) else 0;
        }
        p0.* += rows;
    }
    sleepSecs(500);
    if (ab) |a| printArms(a, rows, us[0 .. steps - steps % a.n], host[0 .. steps - steps % a.n]);
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

const PairArm = enum { one, oneb, two, stock, onepre, twopre };

const TICK_SOURCE =
    \\    const uint i = thread_position_in_grid.x;
    \\    float v = x[i];
    \\    for (uint k = 0; k < ITERS; k++) v = fma(v, 0.999f, 0.001f);
    \\    y[i] = v + 1.0f;
;

const tick_links = 512;
const tick_n = 4096;

test "dsv41 kernel map: two fixtures stepped together on two GPU streams (DSV41_DEC_UBENCH, DSV41_KMAP_PAIR)" {
    if (!decode.ubenchOn() or std.c.getenv("DSV41_KMAP_PAIR") == null) return error.SkipZigTest;
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const gpa = std.testing.allocator;
    dsv41.clusterHooks().tp_sum = &identitySum;
    defer dsv41.clusterHooks().tp_sum = null;
    const o = try KmapOpts.fromEnv();
    const rows = envInt("DSV41_KMAP_ROWS", 1);
    const dev = mlx.mlx_device_new_type(.gpu, 0);
    defer _ = mlx.mlx_device_free(dev);
    const sb = mlx.mlx_stream_new_device(dev);
    defer _ = mlx.mlx_stream_free(sb);
    const fa = try o.build(gpa, 1, null);
    defer fa.deinit();
    const fb = try o.build(gpa, 2, sb);
    defer fb.deinit();
    waitForGo();
    try overlapCheck(gpa, sb);
    var arms: [max_arms]PairArm = undefined;
    var n: usize = 0;
    var it = std.mem.tokenizeScalar(u8, std.mem.span(std.c.getenv("DSV41_KMAP_PAIR_ARMS") orelse "one/one/oneb/two/stock/onepre/twopre"), '/');
    while (it.next()) |tok| : (n += 1) arms[n] = std.meta.stringToEnum(PairArm, tok) orelse return error.BadArm;
    try pairPhase(gpa, fa, fb, rows, o.steps, arms[0..n]);
}

fn overlapCheck(gpa: std.mem.Allocator, sb: mlx.mlx_stream) !void {
    const k = try dsv41.buildKernel("dsv41_kmap_tick", &.{"x"}, &.{"y"}, TICK_SOURCE, "");
    defer _ = mlx.mlx_fast_metal_kernel_free(k);
    for ([_]c_int{ 0, 2048 }) |iters| {
        const one = try tickChains(gpa, k, iters, &.{mlx.gpuStream()});
        const two = try tickChains(gpa, k, iters, &.{ mlx.gpuStream(), sb });
        std.debug.print("[kmap] streams: {d} dependent tiny launches of {d} fma steps on one stream {d:.0} us, the same on each of two streams at once {d:.0} us (x{d:.2})\n", .{ tick_links, iters, one, two, two / one });
    }
}

fn tickChains(gpa: std.mem.Allocator, k: mlx.mlx_fast_metal_kernel, iters: c_int, streams: []const mlx.mlx_stream) !f64 {
    var times: [7]f64 = undefined;
    for (0..2 + times.len) |rep| {
        var sc = dsv41.Scope.init(gpa, streams[0]);
        defer sc.deinit();
        var ends: [2]A = undefined;
        for (streams, 0..) |s, j| {
            var z = mlx.mlx_array_new();
            var x = try sc.res(mlx.mlx_zeros(&z, &[_]c_int{tick_n}, 1, .float32, s), &z);
            for (0..tick_links) |_| x = try tick(&sc, k, iters, x, s);
            ends[j] = x;
        }
        const t0 = nowUs();
        try dsv41.evalAll(ends[0..streams.len]);
        if (rep >= 2) times[rep - 2] = @floatFromInt(nowUs() - t0);
    }
    return medianOf(&times);
}

fn tick(sc: *dsv41.Scope, k: mlx.mlx_fast_metal_kernel, iters: c_int, x: A, s: mlx.mlx_stream) !A {
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "ITERS", iters));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{tick_n}, 1, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, tick_n, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 256, 1, 1));
    const iv = mlx.mlx_vector_array_new_data(&[_]A{x}, 1);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, k, iv, cfg, s));
    var o = mlx.mlx_array_new();
    return sc.res(mlx.mlx_vector_array_get(&o, ov, 0), &o);
}

fn inputs(gpa: std.mem.Allocator, fx: *fixture.Fixture, rows: usize, n: usize, mix: bool) ![]A {
    const out = try gpa.alloc(A, n);
    for (out, 0..) |*a, i| a.* = (if (mix) fx.cpre(rows) else fx.stream(rows)) catch |e| {
        for (out[0..i]) |b| _ = mlx.mlx_array_free(b);
        gpa.free(out);
        return e;
    };
    return out;
}

fn freeAll(gpa: std.mem.Allocator, xs: []A) void {
    for (xs) |a| _ = mlx.mlx_array_free(a);
    gpa.free(xs);
}

fn appendState(gpa: std.mem.Allocator, ev: *std.ArrayList(A), ls: *const dsv41.LState) !void {
    try ev.append(gpa, ls.win.buf);
    if (ls.comp) |r| if (r.buf.ctx != null) try ev.append(gpa, r.buf);
    if (ls.idx_k) |r| if (r.buf.ctx != null) try ev.append(gpa, r.buf);
    if (ls.pend_kv.ctx != null) try ev.append(gpa, ls.pend_kv);
    if (ls.pend_gate.ctx != null) try ev.append(gpa, ls.pend_gate);
}

fn stepTogether(gpa: std.mem.Allocator, fxs: []const *fixture.Fixture, xs: []const A, pres: []const A, p0s: []const *usize, whole: bool) !i64 {
    var cur: [2]A = @splat(none);
    var pr: [2]A = @splat(none);
    var tk: [2]A = @splat(none);
    var cd: [2]A = @splat(none);
    defer for (0..2) |k| {
        drop(&cur[k]);
        drop(&pr[k]);
        drop(&tk[k]);
        drop(&cd[k]);
    };
    for (fxs, 0..) |fx, k| {
        if (fx.m.dec_state.?.n != p0s[k].*) return error.KmapStateMoved;
        hold(&cur[k], xs[k]);
        hold(&pr[k], pres[k]);
    }
    const lo = fxs[0].m.pair.?.lo;
    const hi = fxs[0].m.pair.?.hi;
    var built: i64 = 0;
    var ev: std.ArrayList(A) = .empty;
    defer ev.deinit(gpa);
    var held: std.ArrayList(A) = .empty;
    defer {
        for (held.items) |a| _ = mlx.mlx_array_free(a);
        held.deinit(gpa);
    }
    for (lo..hi) |li| {
        if (!whole) ev.clearRetainingCapacity();
        for (fxs, 0..) |fx, k| {
            const m = fx.m;
            var sc = dsv41.Scope.init(gpa, m.s);
            defer sc.deinit();
            var sh: dsv41.Shared = .{ .topk = tk[k], .cand = cd[k] };
            var pre = pr[k];
            const y = try dsv41.block(&sc, m, &m.dec_state.?, li, cur[k], &pre, p0s[k].*, &sh);
            hold(&cur[k], y);
            hold(&pr[k], pre);
            hold(&tk[k], sh.topk);
            hold(&cd[k], sh.cand);
            if (whole) {
                const n0 = ev.items.len;
                try appendState(gpa, &ev, &m.dec_state.?.layers[li]);
                for (ev.items[n0..]) |*a| {
                    var o = mlx.mlx_array_new();
                    _ = mlx.mlx_array_set(&o, a.*);
                    try held.append(gpa, o);
                    a.* = o;
                }
                continue;
            }
            try ev.appendSlice(gpa, &.{ cur[k], pr[k] });
            try appendState(gpa, &ev, &m.dec_state.?.layers[li]);
        }
        if (whole and li + 1 == hi) {
            for (0..fxs.len) |k| try ev.appendSlice(gpa, &.{ cur[k], pr[k] });
            built = nowUs();
            try dsv41.evalAll(ev.items);
        } else if (whole) {
            continue;
        } else if (li + 1 == hi) {
            built = nowUs();
            try dsv41.evalAll(ev.items);
        } else {
            const vec = mlx.mlx_vector_array_new_data(ev.items.ptr, ev.items.len);
            defer _ = mlx.mlx_vector_array_free(vec);
            try mlx.check(mlx.mlx_async_eval(vec));
        }
    }
    const rows: usize = @intCast(mlx.getShape(xs[0])[0]);
    for (fxs, 0..) |fx, k| {
        fx.m.dec_state.?.n += rows;
        p0s[k].* += rows;
    }
    return built;
}

fn pairPhase(gpa: std.mem.Allocator, fa: *fixture.Fixture, fb: *fixture.Fixture, rows: usize, steps: usize, arms: []const PairArm) !void {
    const warm = 3;
    const total = warm + steps;
    std.debug.assert(steps <= max_steps and arms.len > 0);
    const xa = try inputs(gpa, fa, rows, total, false);
    defer freeAll(gpa, xa);
    const pa = try inputs(gpa, fa, rows, total, true);
    defer freeAll(gpa, pa);
    const xb = try inputs(gpa, fb, rows, total, false);
    defer freeAll(gpa, xb);
    const pb = try inputs(gpa, fb, rows, total, true);
    defer freeAll(gpa, pb);
    var p0a = fa.m.dec_state.?.n;
    var p0b = fb.m.dec_state.?.n;
    var us: [max_steps]f64 = undefined;
    var host: [max_steps]f64 = undefined;
    sleepSecs(2000);
    waitForGate();
    for (0..total) |k| {
        if (k == warm) sleepSecs(500);
        const s0 = nowUs();
        const built = switch (arms[armOfN(k -| warm, arms.len)]) {
            .one, .onepre => |a| try stepTogether(gpa, &.{fa}, &.{xa[k]}, &.{pa[k]}, &.{&p0a}, a == .onepre),
            .oneb => try stepTogether(gpa, &.{fb}, &.{xb[k]}, &.{pb[k]}, &.{&p0b}, false),
            .two, .twopre => |a| try stepTogether(gpa, &.{ fa, fb }, &.{ xa[k], xb[k] }, &.{ pa[k], pb[k] }, &.{ &p0a, &p0b }, a == .twopre),
            .stock => blk: {
                const out = try dsv41.workerStep(fa.m, gpa, xa[k], pa[k], p0a, false, false);
                for (out) |o| _ = mlx.mlx_array_free(o);
                p0a += rows;
                break :blk s0;
            },
        };
        if (k >= warm) {
            us[k - warm] = @floatFromInt(nowUs() - s0);
            host[k - warm] = @floatFromInt(built - s0);
        }
    }
    sleepSecs(500);
    var nb: [128]u8 = undefined;
    var len: usize = 0;
    for (arms, 0..) |a, j| len += (std.fmt.bufPrint(nb[len..], "{s}{s}", .{ if (j > 0) "/" else "", @tagName(a) }) catch break).len;
    const used = steps - steps % arms.len;
    printBlocks(nb[0..len], arms.len, rows, us[0..used], host[0..used]);
    if (std.mem.indexOfScalar(PairArm, arms, .one)) |one| if (std.mem.indexOfScalar(PairArm, arms, .two)) |two|
        std.debug.print("[kmap] pair rows={d}: A and B stepped together take x{d:.3} of A alone (block median; 1 = full overlap, 2 = none)\n", .{ rows, blockRatio(us[0..used], arms.len, two, one) });
    const one = std.mem.indexOfScalar(PairArm, arms, .onepre) orelse return;
    const two = std.mem.indexOfScalar(PairArm, arms, .twopre) orelse return;
    var gpu: [max_steps]f64 = undefined;
    for (gpu[0..used], us[0..used], host[0..used]) |*g, u, h| g.* = u - h;
    std.debug.print("[kmap] pair rows={d} whole graphs: evaluating A's and B's together takes x{d:.3} of A's alone (block median of the time after the host built them)\n", .{ rows, blockRatio(gpu[0..used], arms.len, two, one) });
}
