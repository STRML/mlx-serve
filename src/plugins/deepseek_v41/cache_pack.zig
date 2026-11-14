const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const codec = @import("cache_codec.zig");
const dsv41 = @import("deepseek_v41.zig");

const A = mlx.mlx_array;
const S = mlx.mlx_stream;

pub const Kernels = struct {
    pack: mlx.mlx_fast_metal_kernel,
    unpack: mlx.mlx_fast_metal_kernel,

    pub fn init() !Kernels {
        const header = dsv41.FP8_HEADER ++ dsv41.CACHE_CODEC_HEADER;
        const pack_kernel = try dsv41.buildKernel("dsv41_cache_pack", &.{"x"}, &.{"out"}, dsv41.CACHE_PACK_SOURCE, header);
        errdefer _ = mlx.mlx_fast_metal_kernel_free(pack_kernel);
        return .{
            .pack = pack_kernel,
            .unpack = try dsv41.buildKernel("dsv41_cache_unpack", &.{"packed"}, &.{ "out", "status" }, dsv41.CACHE_UNPACK_SOURCE, header),
        };
    }

    pub fn deinit(self: *Kernels) void {
        _ = mlx.mlx_fast_metal_kernel_free(self.pack);
        _ = mlx.mlx_fast_metal_kernel_free(self.unpack);
    }
};

pub const Unpacked = struct { bits: A, status: A };

pub fn encode(kernels: *const Kernels, stream: S, x: A, enc: codec.Encoding) !A {
    const shape = mlx.getShape(x);
    if (shape.len != 2 or mlx.mlx_array_dtype(x) != .uint16) return error.Dsv41CacheShape;
    const rows = std.math.cast(usize, shape[0]) orelse return error.Dsv41CacheShape;
    const width = std.math.cast(usize, shape[1]) orelse return error.Dsv41CacheShape;
    const row_bytes = codec.packedRowBytes(enc, width) orelse return error.Dsv41CacheGeometry;
    const groups = try groupCount(rows, width, enc);
    const q = try encId(enc);
    const outs = try apply(kernels.pack, stream, &.{x}, &.{.{ .shape = &.{ try ci(rows), try ci(row_bytes) }, .dtype = .uint8 }}, groups, q);
    return outs[0];
}

pub fn decode(kernels: *const Kernels, stream: S, packed_array: A, spec: codec.Spec, enc: codec.Encoding) !Unpacked {
    const row_bytes = codec.packedRowBytes(enc, spec.width) orelse return error.Dsv41CacheGeometry;
    const expected = [_]c_int{ try ci(spec.rows), try ci(row_bytes) };
    const shape = mlx.getShape(packed_array);
    if (shape.len != expected.len or shape[0] != expected[0] or shape[1] != expected[1] or mlx.mlx_array_dtype(packed_array) != .uint8) return error.Dsv41CacheShape;
    const groups = try groupCount(spec.rows, spec.width, enc);
    const outs = try apply(kernels.unpack, stream, &.{packed_array}, &.{
        .{ .shape = &.{ try ci(spec.rows), try ci(spec.width) }, .dtype = .uint16 },
        .{ .shape = &.{try ci(groups)}, .dtype = .uint32 },
    }, groups, try encId(enc));
    return .{ .bits = outs[0], .status = outs[1] };
}

const Output = struct { shape: []const c_int, dtype: mlx.mlx_dtype };

fn apply(kernel: mlx.mlx_fast_metal_kernel, stream: S, inputs: []const A, outputs: []const Output, groups: usize, enc: c_int) ![2]A {
    const grid = try ci(groups);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    for (outputs) |out| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, out.shape.ptr, out.shape.len, out.dtype));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, grid, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, @intCast(@min(groups, 256)), 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "ENC", enc));
    const group: c_int = if (enc == 1) 16 else 32;
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "G", group));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", mlx.mlx_array_dtype(inputs[0])));
    const iv = mlx.mlx_vector_array_new_data(inputs.ptr, inputs.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, kernel, iv, cfg, stream));
    var result: [2]A = .{ .{ .ctx = null }, .{ .ctx = null } };
    errdefer {
        for (result) |value| if (value.ctx != null) {
            _ = mlx.mlx_array_free(value);
        };
    }
    for (result[0..outputs.len], 0..) |*out, i| {
        var value = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(&value, ov, i));
        out.* = value;
    }
    return result;
}

fn groupCount(rows: usize, width: usize, enc: codec.Encoding) !usize {
    const group: usize = if (enc == .fp4_e4m3_g16) 16 else 32;
    return std.math.mul(usize, rows, width / group) catch return error.Dsv41CacheGeometry;
}

fn encId(enc: codec.Encoding) !c_int {
    return switch (enc) {
        .raw => error.Dsv41CacheEncoding,
        .fp4_e4m3_g16 => 1,
        .fp4_ue8m0_g32 => 2,
        .fp8_ue8m0_g32 => 3,
    };
}

fn ci(value: usize) !c_int {
    return std.math.cast(c_int, value) orelse error.Dsv41CacheGeometry;
}
