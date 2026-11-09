const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const kernelSource = @import("../../../mlx/kernels.zig").source;
const dsv41 = @import("../deepseek_v41.zig");
const ffn_in = @import("ffn_in.zig");
const moe_fp4 = @import("moe_fp4.zig");
const seg = @import("seg.zig");
const config = @import("../config.zig");
const seg_ffn_in = @import("seg_ffn_in.zig");

const A = mlx.mlx_array;

pub const SOURCE = kernelSource("plugins/deepseek_v41/kernels/dec_seg_ffn.metal");
pub const HEADER = kernelSource("plugins/deepseek_v41/kernels/seg_proto_header.metal") ++ moe_fp4.HEADER_BW ++
    kernelSource("plugins/deepseek_v41/kernels/dec_seg_ffn_phases_header.metal") ++
    kernelSource("plugins/deepseek_v41/kernels/dec_seg_ffn_moe_header.metal");
const INS = [_][*:0]const u8{
    "x",    "y",    "spre",  "spost", "scomb", "fn",     "scale", "base", "w",    "gate_w", "bias", "gu_w", "gu_s", "sgu_w", "sgu_s",
    "dn_w", "dn_s", "sdn_w", "sdn_s", "cnt",   "par",    "status", "dep",
};
const OUTS = [_][*:0]const u8{ "x1", "pre", "post", "comb", "scratch", "part" };

pub const ReadMode = enum(usize) { peer = 0, plain = 1, staged = 2 };

pub const Hook = struct { abort_at: u32 = 0, skew_at: u32 = 0 };

pub var hook: Hook = .{};

pub var read_mode: ReadMode = .staged;

pub var grid_extra: usize = 0;

pub var stop_after: u32 = 4;

pub const MOE_SOURCE = kernelSource("plugins/deepseek_v41/kernels/dec_seg_moe.metal");
const MOE_INS = [_][*:0]const u8{ "h2", "idx", "wts", "gu_w", "gu_s", "sgu_w", "sgu_s", "dn_w", "dn_s", "sdn_w", "sdn_s", "cnt", "par", "status", "dep" };
const MOE_OUTS = [_][*:0]const u8{ "act", "part" };

pub var moe_only = false;

pub var moe_threads: usize = 256;

pub fn buildMoe() !mlx.mlx_fast_metal_kernel {
    return dsv41.buildKernel("dsv41_dec_seg_moe", &MOE_INS, &MOE_OUTS, MOE_SOURCE, HEADER);
}

pub fn build() !mlx.mlx_fast_metal_kernel {
    return dsv41.buildKernel("dsv41_dec_seg_ffn", &INS, &OUTS, SOURCE, HEADER);
}

pub const Out = struct {
    x1: A,
    sf: dsv41.Split,
    part: A,
    scratch: A,
    layout: Layout,

    pub fn h2(o: Out, sc: *dsv41.Scope) !A {
        return sc.reshape(try sc.view(try sc.slice(o.scratch, 0, o.layout.h2, o.layout.h2 + o.layout.d / 2), .bfloat16), &.{ 1, ci(o.layout.d) });
    }

    pub fn idx(o: Out, sc: *dsv41.Scope) !A {
        return sc.reshape(try sc.view(try sc.slice(o.scratch, 0, o.layout.idx, o.layout.idx + o.layout.top_k), .uint32), &.{ 1, ci(o.layout.top_k) });
    }

    pub fn wts(o: Out, sc: *dsv41.Scope) !A {
        return sc.reshape(try sc.slice(o.scratch, 0, o.layout.wts, o.layout.wts + o.layout.top_k), &.{ 1, ci(o.layout.top_k) });
    }
};

pub const Layout = struct {
    parts: usize,
    logits: usize,
    h2: usize,
    idx: usize,
    wts: usize,
    act: usize,
    end: usize,
    d: usize,
    top_k: usize,

    fn of(p: ffn_in.Dims, n: usize) Layout {
        const logits = ffn_in.partsLen(p);
        const h2 = alignWords(logits + p.e);
        const idx = alignWords(h2 + p.d / 2);
        const wts = idx + p.top_k;
        const act = alignWords(wts + p.top_k);
        return .{ .parts = 0, .logits = logits, .h2 = h2, .idx = idx, .wts = wts, .act = act, .end = act + (p.top_k + 1) * n / 2, .d = p.d, .top_k = p.top_k };
    }
};

fn alignWords(w: usize) usize {
    return (w + 3) / 4 * 4;
}

pub const Geometry = struct { nin: usize, ncb: usize, items: usize };

const par_words = 16;

fn parWords(gen: u32, slot: usize, c: *const config.Config) [par_words]u32 {
    const f = [_]f32{ c.rms_norm_eps, c.dsv4_hc_eps, c.router_scaling_factor, c.dsv4_swiglu_limit, 1.0, 0.0, 0.0 };
    var w: [par_words]u32 = @splat(0);
    w[0] = gen;
    w[1] = @intCast(slot * seg.slot_words);
    w[2] = hook.abort_at;
    w[3] = hook.skew_at;
    for (f, 4..) |v, i| w[i] = @bitCast(v);
    return w;
}

pub fn geometryOf(p: ffn_in.Dims, n: usize, d: usize, nc: usize) Geometry {
    const nin = ffn_in.gridB(p) / ffn_in.b_threads;
    const ncb = n / 8;
    return .{ .nin = nin, .ncb = ncb, .items = nin + ncb + p.top_k * ncb + d / (2 * nc) };
}

fn ci(v: usize) c_int {
    return @intCast(v);
}

fn dim(a: A, axis: usize) usize {
    return @intCast(mlx.getShape(a)[axis]);
}

const Banks = struct { gu_w: A, gu_s: A, sgu_w: A, sgu_s: A, dn_w: A, dn_s: A, sdn_w: A, sdn_s: A, n: usize, nc: usize };

fn is(a: A, dt: mlx.mlx_dtype) bool {
    return a.ctx != null and mlx.mlx_array_dtype(a) == dt;
}

fn shapeIs(a: A, want: []const usize) bool {
    const s = mlx.getShape(a);
    if (s.len != want.len) return false;
    for (s, want) |u, v| if (@as(usize, @intCast(u)) != v) return false;
    return true;
}

fn banksOf(ly: *const dsv41.Layer, d: usize, topk: usize) ?Banks {
    const gu = ly.experts.gu orelse return null;
    const down = &ly.experts.down;
    const sgu = ly.shared_gu orelse return null;
    const sdn = &ly.shared[2];
    if (topk < 1 or topk > 6 or d % 1024 != 0) return null;
    if (gu.qp.mode != .mxfp4 or gu.qp.bits != 4 or gu.qp.group_size != 32 or gu.b.ctx != null or gu.s.ctx == null) return null;
    if (down.qp.mode != .mxfp4 or down.qp.bits != 4 or down.qp.group_size != 32 or down.b.ctx != null or down.s.ctx == null) return null;
    if (sgu.qp.mode != .mxfp8 or sgu.qp.bits != 8 or sgu.qp.group_size != 32 or sgu.s.ctx == null) return null;
    if (sdn.qp.mode != .mxfp8 or sdn.qp.bits != 8 or sdn.qp.group_size != 32 or sdn.s.ctx == null) return null;
    if (mlx.getShape(gu.w).len != 3 or dim(gu.w, 1) % 2 != 0) return null;
    const n = dim(gu.w, 1) / 2;
    const experts = dim(gu.w, 0);
    if (n % 8 != 0 or n % 32 != 0 or dim(ly.gate_w, 0) != experts) return null;
    const nc = moe_fp4.downCols(n, d, 1) orelse return null;
    if (d % (2 * nc) != 0) return null;
    const ok = shapeIs(gu.w, &.{ experts, 2 * n, d / 8 }) and shapeIs(gu.s, &.{ experts, 2 * n, d / 32 }) and
        shapeIs(down.w, &.{ experts, d, n / 8 }) and shapeIs(down.s, &.{ experts, d, n / 32 }) and
        shapeIs(sgu.w, &.{ 2 * n, d / 4 }) and shapeIs(sgu.s, &.{ 2 * n, d / 32 }) and
        shapeIs(sdn.w, &.{ d, n / 4 }) and shapeIs(sdn.s, &.{ d, n / 32 });
    if (!ok) return null;
    if (!is(gu.w, .uint32) or !is(gu.s, .uint8) or !is(down.w, .uint32) or !is(down.s, .uint8)) return null;
    if (!is(sgu.w, .uint32) or !is(sgu.s, .uint8) or !is(sdn.w, .uint32) or !is(sdn.s, .uint8)) return null;
    const vgu = if (ly.experts.views) |v| v[0] else gu;
    const vdn = if (ly.experts.views) |v| v[1] else down.*;
    return .{ .gu_w = vgu.w, .gu_s = vgu.s, .sgu_w = sgu.w, .sgu_s = sgu.s, .dn_w = vdn.w, .dn_s = vdn.s, .sdn_w = sdn.w, .sdn_s = sdn.s, .n = n, .nc = nc };
}

pub fn launch(sc: *dsv41.Scope, m: *const dsv41.Dsv41Model, ly: *const dsv41.Layer, li: usize, x: A, ao_sum: A, sa: *const dsv41.Split) !?Out {
    const kern = m.dec.seg_ffn orelse return null;
    if (m.dec.moe_fp4 == null) return null;
    const st = m.seg orelse return null;
    if (st.protocol != .ticket or dim(x, 0) != 1) return null;
    if (mlx.mlx_array_dtype(ly.gate_w) != .bfloat16 or mlx.mlx_array_dtype(x) != .bfloat16) return null;
    const c = &m.cfg;
    const p: ffn_in.Dims = .{ .rows = 1, .d = c.hidden_size, .e = dim(ly.gate_w, 0), .top_k = ly.top_k, .iters = c.dsv4_hc_sinkhorn_iters, .norm = c.moe_route_norm and ly.top_k > 1 };
    if (!ffn_in.servesB(p)) return null;
    const b = banksOf(ly, p.d, p.top_k) orelse return null;
    const geo = geometryOf(p, b.n, p.d, b.nc);
    const lay = Layout.of(p, b.n);
    const w = ffn_in.weightsOf(ly);
    const hc = ffn_in.hc;
    const act = ffn_in.act;
    const par = try sc.keep(mlx.mlx_array_new_data(&parWords(try st.counters.advance(), seg_ffn_in.slotOf(li), c), &[_]c_int{par_words}, 1, .uint32));
    const ins = [_]A{
        x,      try sc.astype(ao_sum, act), sa.pre, sa.post, sa.comb, w.fn_w, w.fn_scale, w.fn_base, w.norm_w, w.gate_w, w.gate_bias,
        b.gu_w, b.gu_s,                     b.sgu_w, b.sgu_s, b.dn_w,  b.dn_s, b.sdn_w,    b.sdn_s,   st.counters.buf, par, st.status, try st.after(sc),
    };
    const outs = [6]ffn_in.OutSpec{
        .{ .shape = &.{ 1, ci(hc), ci(p.d) }, .dt = act },
        .{ .shape = &.{ 1, ci(hc) }, .dt = .float32 },
        .{ .shape = &.{ 1, ci(hc) }, .dt = .float32 },
        .{ .shape = &.{ 1, ci(hc), ci(hc) }, .dt = .float32 },
        .{ .shape = &.{ci(lay.end)}, .dt = .float32 },
        .{ .shape = &.{ 1, ci(p.d) }, .dt = act },
    };
    const bt = ffn_in.bTargs(p);
    const extra = [_]ffn_in.TArg{
        .{ .name = "K", .v = p.d },          .{ .name = "N", .v = b.n },          .{ .name = "NC", .v = b.nc },         .{ .name = "XM", .v = @intFromEnum(read_mode) }, .{ .name = "STOP", .v = stop_after },
        .{ .name = "OFF_LOGITS", .v = lay.logits }, .{ .name = "OFF_H2", .v = lay.h2 }, .{ .name = "OFF_IDX", .v = lay.idx }, .{ .name = "OFF_WTS", .v = lay.wts }, .{ .name = "OFF_ACT", .v = lay.act },
    };
    var targs: [bt.len + extra.len]ffn_in.TArg = undefined;
    @memcpy(targs[0..bt.len], &bt);
    @memcpy(targs[bt.len..], &extra);
    const o = try ffn_in.run(6, sc, kern, &ins, outs, (geo.items + grid_extra) * ffn_in.b_threads, ffn_in.b_threads, &targs);
    st.chain(o[5]);
    seg.dispatches += 1;
    return .{ .x1 = o[0], .sf = .{ .pre = o[1], .post = o[2], .comb = o[3] }, .part = o[5], .scratch = o[4], .layout = lay };
}

pub fn launchMoe(sc: *dsv41.Scope, m: *const dsv41.Dsv41Model, ly: *const dsv41.Layer, li: usize, h2: A, idx: A, wts: A) !?A {
    const kern = m.dec.seg_moe orelse return null;
    const st = m.seg orelse return null;
    if (st.protocol != .ticket or dim(h2, 0) != 1) return null;
    const c = &m.cfg;
    const top_k = dim(idx, 1);
    const b = banksOf(ly, c.hidden_size, top_k) orelse return null;
    const nsg = moe_threads / 32;
    if (b.n % nsg != 0) return null;
    const items = b.n / nsg * (top_k + 1) + c.hidden_size / (2 * b.nc);
    const par = try sc.keep(mlx.mlx_array_new_data(&parWords(try st.counters.advance(), seg_ffn_in.slotOf(li), c), &[_]c_int{par_words}, 1, .uint32));
    const ins = [_]A{ h2, idx, wts, b.gu_w, b.gu_s, b.sgu_w, b.sgu_s, b.dn_w, b.dn_s, b.sdn_w, b.sdn_s, st.counters.buf, par, st.status, try st.after(sc) };
    const outs = [2]ffn_in.OutSpec{
        .{ .shape = &.{ 1, ci(top_k + 1), ci(b.n) }, .dt = ffn_in.act },
        .{ .shape = &.{ 1, ci(c.hidden_size) }, .dt = ffn_in.act },
    };
    const targs = [_]ffn_in.TArg{
        .{ .name = "K", .v = c.hidden_size }, .{ .name = "N", .v = b.n },          .{ .name = "NC", .v = b.nc },
        .{ .name = "TOPK", .v = top_k },      .{ .name = "XM", .v = @intFromEnum(read_mode) }, .{ .name = "TH", .v = moe_threads }, .{ .name = "STOP", .v = stop_after },
    };
    const o = try ffn_in.run(2, sc, kern, &ins, outs, items * moe_threads, moe_threads, &targs);
    st.chain(o[1]);
    seg.dispatches += 1;
    return o[1];
}
