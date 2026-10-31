//! Idol's fused elementwise glue for the denoiser block, each kernel one pass in place of an op chain that makes
//! many over [12k, 1024..4096] tensors: q/k RoPE then per-head RMS norm then the head-major layout SDPA reads, and
//! the fc1 bias add with the tanh GELU. Both decline (null) off the GPU and off f16/bf16; the caller keeps the chain.

const std = @import("std");
const kernelSource = @import("../../mlx/kernels.zig").source;
const mlx = @import("../../mlx/ffi.zig");
const log = @import("../../sdk/util/log.zig");

pub const QKV_SOURCE = kernelSource("plugins/idol/kernels/qkv_prep.metal");
pub const GELU_SOURCE = kernelSource("plugins/idol/kernels/bias_gelu.metal");

const HD: c_int = 64;

/// Lets a test run the chain these kernels replace.
pub var disabled_for_test = false;

fn eligible(dt: mlx.mlx_dtype) bool {
    return dt == .float16 or dt == .bfloat16;
}

fn makeKernel(name: [*:0]const u8, ins: []const [*:0]const u8, outs: []const [*:0]const u8, src: [*:0]const u8) !mlx.mlx_fast_metal_kernel {
    const iv = mlx.mlx_vector_string_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_string_free(iv);
    const ov = mlx.mlx_vector_string_new_data(outs.ptr, outs.len);
    defer _ = mlx.mlx_vector_string_free(ov);
    const k = mlx.mlx_fast_metal_kernel_new(name, iv, ov, src, "", true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    return k;
}

/// Runs `kernel`; `outs` receives one handle per declared output.
fn apply(kernel: mlx.mlx_fast_metal_kernel, cfg: mlx.mlx_fast_metal_kernel_config, arrs: []const mlx.mlx_array, outs: []mlx.mlx_array, s: mlx.mlx_stream) !void {
    const v = mlx.mlx_vector_array_new_data(arrs.ptr, arrs.len);
    defer _ = mlx.mlx_vector_array_free(v);
    var o = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(o);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&o, kernel, v, cfg, s));
    if (mlx.mlx_vector_array_size(o) != outs.len) return error.MetalKernelBadOutputCount;
    var got: usize = 0;
    errdefer for (outs[0..got]) |a| {
        _ = mlx.mlx_array_free(a);
    };
    for (outs, 0..) |*slot, i| {
        slot.* = mlx.mlx_array_new();
        got = i + 1;
        try mlx.check(mlx.mlx_vector_array_get(slot, o, i));
    }
}

var qkv_kernel: ?mlx.mlx_fast_metal_kernel = null;
var gelu_kernel: ?mlx.mlx_fast_metal_kernel = null;

pub const Qkv = struct { q: mlx.mlx_array, k: mlx.mlx_array, v: mlx.mlx_array };

/// `y` the fused qkv projection `[.., 3*H*64]` (q | k | v, head-major inside each); `cos` / `sin` f32 `[.., H, 32]`
/// per token and head; `q_gamma` / `k_gamma` f32 `[H, 64]`. q and k are rotated (adjacent pairs, f32 math, rounded to
/// y's dtype) and then RMS-normalized per head; all three come out heads-major `[1, H, L, 64]`.
pub fn qkvPrep(y: mlx.mlx_array, q_gamma: mlx.mlx_array, k_gamma: mlx.mlx_array, cos: mlx.mlx_array, sin: mlx.mlx_array, eps: f32, heads: c_int, s: mlx.mlx_stream) !?Qkv {
    if (disabled_for_test or !mlx.streamIsGpu(s)) return null;
    const dt = mlx.mlx_array_dtype(y);
    if (!eligible(dt)) return null;
    const width: c_int = 3 * heads * HD;
    const sh = mlx.getShape(y);
    if (sh.len == 0 or sh[sh.len - 1] != width) return null;
    const rows: c_int = @intCast(mlx.mlx_array_size(y) / @as(usize, @intCast(width)));
    const table: usize = @as(usize, @intCast(rows)) * @as(usize, @intCast(heads)) * @as(usize, @intCast(HD / 2));
    for ([_]mlx.mlx_array{ cos, sin }) |t| if (mlx.mlx_array_dtype(t) != .float32 or mlx.mlx_array_size(t) != table) return null;
    for ([_]mlx.mlx_array{ q_gamma, k_gamma }) |g| if (mlx.mlx_array_dtype(g) != .float32 or mlx.mlx_array_size(g) != @as(usize, @intCast(heads * HD))) return null;

    var y2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(y2);
    try mlx.check(mlx.mlx_reshape(&y2, y, &[_]c_int{ rows, width }, 2, s));
    if (qkv_kernel == null) qkv_kernel = try makeKernel("msv_idol_qkv_prep", &.{ "Y", "COS", "SIN", "QG", "KG", "eps" }, &.{ "OQ", "OK", "OV" }, QKV_SOURCE);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    for (0..3) |_| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, heads, rows, HD }, 4, dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, 256 * @divTrunc(3 * heads + 7, 8), rows, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 256, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "H", heads));
    // A 0-dim f32 binds `constant float&`; a [1] array would bind a pointer.
    const eps_a = mlx.mlx_array_new_float(eps);
    defer _ = mlx.mlx_array_free(eps_a);
    var outs: [3]mlx.mlx_array = undefined;
    try apply(qkv_kernel.?, cfg, &.{ y2, cos, sin, q_gamma, k_gamma, eps_a }, &outs, s);
    return .{ .q = outs[0], .k = outs[1], .v = outs[2] };
}

/// `gelu_tanh(y + bias)` for `y` `[.., N]` and `bias` `[N]` of one dtype: the sum rounds to that dtype as an add
/// would, the GELU runs in f32 and rounds once (as the reference's fused one does; the chain rounded each of its
/// eight ops).
pub fn biasGelu(y: mlx.mlx_array, bias: mlx.mlx_array, s: mlx.mlx_stream) !?mlx.mlx_array {
    if (disabled_for_test or !mlx.streamIsGpu(s)) return null;
    const dt = mlx.mlx_array_dtype(y);
    if (!eligible(dt) or mlx.mlx_array_dtype(bias) != dt) return null;
    const sh = mlx.getShape(y);
    if (sh.len == 0) return null;
    const n: c_int = sh[sh.len - 1];
    if (@rem(n, 4) != 0 or mlx.mlx_array_size(bias) != @as(usize, @intCast(n))) return null;
    const rows: c_int = @intCast(mlx.mlx_array_size(y) / @as(usize, @intCast(n)));

    var y2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(y2);
    try mlx.check(mlx.mlx_reshape(&y2, y, &[_]c_int{ rows, n }, 2, s));
    if (gelu_kernel == null) gelu_kernel = try makeKernel("msv_idol_bias_gelu", &.{ "Y", "B" }, &.{"O"}, GELU_SOURCE);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ rows, n }, 2, dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, @divExact(rows * n, 4), 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 256, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", dt));
    var out: [1]mlx.mlx_array = undefined;
    try apply(gelu_kernel.?, cfg, &.{ y2, bias }, &out, s);
    var o = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(o);
    try mlx.check(mlx.mlx_reshape(&o, out[0], sh.ptr, sh.len, s));
    _ = mlx.mlx_array_free(out[0]);
    return o;
}
