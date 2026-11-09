//! DeepSeek-V4.1 fused decode segments, level 1 (`DSV41_FUSED_SEG=1`): a tensor rank's FFN entry in one launch. From the
//! stream `x`, the summed attention output and the attention's split it builds the new stream, the FFN's hc split, the
//! rms-normed collapse and the router's top K. Every mix and router threadgroup rebuilds the expand and collapse it
//! needs; the mixes' last arriver reduces the split and the router's the picks.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const kernelSource = @import("../../../mlx/kernels.zig").source;
const dsv41 = @import("../deepseek_v41.zig");
const ffn_in = @import("ffn_in.zig");
const seg = @import("seg.zig");

const A = mlx.mlx_array;

pub const SOURCE = kernelSource("plugins/deepseek_v41/kernels/dec_seg_ffn_in.metal");
pub const HEADER = kernelSource("plugins/deepseek_v41/kernels/seg_proto_header.metal");
const INS = [_][*:0]const u8{ "x", "y", "spre", "spost", "scomb", "fn", "scale", "base", "w", "gate_w", "bias", "scal", "eps", "cnt", "ctl", "dep" };
const OUTS = [_][*:0]const u8{ "x1", "h2", "pre", "post", "comb", "idx", "wts", "parts", "logits" };

pub fn build() !mlx.mlx_fast_metal_kernel {
    return dsv41.buildKernel("dsv41_dec_seg_ffn_in", &INS, &OUTS, SOURCE, HEADER);
}

fn ci(v: usize) c_int {
    return @intCast(v);
}

fn dim(a: A, axis: usize) usize {
    return @intCast(mlx.getShape(a)[axis]);
}

/// The counter slot of layer `li`'s FFN entry (`State.init` holds two a layer: attention, then FFN).
pub fn slotOf(li: usize) usize {
    return 2 * li + 1;
}

fn dimsOf(m: *const dsv41.Dsv41Model, ly: *const dsv41.Layer, x: A) ffn_in.Dims {
    const c = &m.cfg;
    return .{ .rows = dim(x, 0), .d = c.hidden_size, .e = dim(ly.gate_w, 0), .top_k = ly.top_k, .iters = c.dsv4_hc_sinkhorn_iters, .norm = c.moe_route_norm and ly.top_k > 1 };
}

/// The new stream `x1`, the FFN's normed input, split and routes for layer `li` (weights `ly`) from the stream `x`, the
/// summed attention output `ao_sum` and the attention's split `sa`, in one launch; null where `ffn_in.servesB` declines
/// or the model holds no kernel or counters for it.
pub fn launch(sc: *dsv41.Scope, m: *const dsv41.Dsv41Model, li: usize, ly: *const dsv41.Layer, x: A, ao_sum: A, sa: *const dsv41.Split) !?ffn_in.Out {
    const kern = m.dec.seg_ffn_in orelse return null;
    const st = m.seg orelse return null;
    if (mlx.mlx_array_dtype(ly.gate_w) != .bfloat16 or mlx.mlx_array_dtype(x) != .bfloat16) return null;
    const p = dimsOf(m, ly, x);
    if (!ffn_in.servesB(p)) return null;
    const w = ffn_in.weightsOf(ly);
    const hc = ffn_in.hc;
    const act = ffn_in.act;
    const scal = try sc.keep(mlx.mlx_array_new_data(&[_]f32{m.cfg.router_scaling_factor}, &[_]c_int{1}, 1, .float32));
    const ctl = try st.counters.ctl(sc, slotOf(li));
    const ins = [_]A{ x, try sc.astype(ao_sum, act), sa.pre, sa.post, sa.comb, w.fn_w, w.fn_scale, w.fn_base, w.norm_w, w.gate_w, w.gate_bias, scal, try dsv41.hcEps(sc, m), st.counters.buf, ctl, try st.after(sc) };
    const o = try ffn_in.run(9, sc, kern, &ins, .{
        .{ .shape = &.{ ci(p.rows), ci(hc), ci(p.d) }, .dt = act },
        .{ .shape = &.{ ci(p.rows), ci(p.d) }, .dt = act },
        .{ .shape = &.{ ci(p.rows), ci(hc) }, .dt = .float32 },
        .{ .shape = &.{ ci(p.rows), ci(hc) }, .dt = .float32 },
        .{ .shape = &.{ ci(p.rows), ci(hc), ci(hc) }, .dt = .float32 },
        .{ .shape = &.{ ci(p.rows), ci(p.top_k) }, .dt = .uint32 },
        .{ .shape = &.{ ci(p.rows), ci(p.top_k) }, .dt = .float32 },
        .{ .shape = &.{ci(ffn_in.partsLen(p))}, .dt = .float32 },
        .{ .shape = &.{ ci(p.rows), ci(p.e) }, .dt = .float32 },
    }, ffn_in.gridB(p), ffn_in.b_threads, &ffn_in.bTargs(p));
    st.chain(o[5]);
    seg.dispatches += 1;
    return .{ .x1 = o[0], .h2 = o[1], .sf = .{ .pre = o[2], .post = o[3], .comb = o[4] }, .idx = o[5], .wts = o[6] };
}
