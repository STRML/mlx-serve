const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const dsv41 = @import("deepseek_v41.zig");

const testing = std.testing;
const A = mlx.mlx_array;
const Scope = dsv41.Scope;

const UINT8_SOURCE: [:0]const u8 =
    "const uint i = thread_position_in_grid.x;\n" ++
    "out[i] = uchar((x[i * 2] & 15) | ((x[i * 2 + 1] & 15) << 4));";

const PACK_SOURCE: [:0]const u8 =
    "const uint g = thread_position_in_grid.x;\n" ++
    "const uint start = g * GS;\n" ++
    "float amax = 0.0f;\n" ++
    "for (uint i = 0; i < GS; ++i) amax = metal::max(amax, metal::abs(float(x[start + i])));\n" ++
    "float scale = QK == 3 ? dsv41_fp8_roundtrip(metal::max(amax, 0.01171875f) / 6.0f) : dsv41_pow2ceil(metal::max(amax, 6.0f * 1.17549435e-38f) / 6.0f);\n" ++
    "uchar sb = QK == 3 ? fp8_bits(metal::max(amax, 0.01171875f) / 6.0f) : uchar(as_type<uint>(scale) >> 23);\n" ++
    "scales[g] = sb;\n" ++
    "uint bad = 0;\n" ++
    "for (uint j = 0; j < GS / 2; ++j) {\n" ++
    "  uchar codes[2];\n" ++
    "  for (uint k = 0; k < 2; ++k) {\n" ++
    "    uint i = start + j * 2 + k;\n" ++
    "    T v = x[i];\n" ++
    "    float q = metal::clamp(float(v) / scale, -6.0f, 6.0f);\n" ++
    "    float mag = metal::abs(dsv41_e2m1(q));\n" ++
    "    uchar level = uchar(mag == 0.0f ? 0 : mag == 0.5f ? 1 : mag == 1.0f ? 2 : mag == 1.5f ? 3 : mag == 2.0f ? 4 : mag == 3.0f ? 5 : mag == 4.0f ? 6 : 7);\n" ++
    "    uchar sign = (as_type<ushort>(v) & 0x8000) ? 8 : 0;\n" ++
    "    codes[k] = sign | level;\n" ++
    "    float decoded = (level == 0 ? 0.0f : level == 1 ? 0.5f : level == 2 ? 1.0f : level == 3 ? 1.5f : level == 4 ? 2.0f : level == 5 ? 3.0f : level == 6 ? 4.0f : 6.0f) * scale;\n" ++
    "    ushort db = as_type<ushort>(T((sign ? -decoded : decoded)));\n" ++
    "    bad |= uint(db != as_type<ushort>(v));\n" ++
    "  }\n" ++
    "  packed[g * (GS / 2) + j] = codes[0] | (codes[1] << 4);\n" ++
    "}\n" ++
    "flags[g] = bad;";

const FP8_BITS_HEADER: [:0]const u8 =
    "inline uchar fp8_bits(float f) {\n" ++
    "  uint fb = as_type<uint>(f); const uint sign = fb & 0x80000000u; fb ^= sign; uchar bits;\n" ++
    "  if (fb >= (543u << 21)) bits = 0x7E;\n" ++
    "  else if (fb < (121u << 23)) { const uint dm = 141u << 23; bits = uchar(as_type<uint>(as_type<float>(fb) + as_type<float>(dm)) - dm); }\n" ++
    "  else { const uint odd = (fb >> 20) & 1u; fb += (uint(7 - 127) << 23) + 0x7FFFFu; fb += odd; bits = uchar(fb >> 20); }\n" ++
    "  return bits | uchar(sign >> 24);\n" ++
    "}\n" ++ dsv41.FP8_HEADER;

const REDUCE_SOURCE: [:0]const u8 =
    "if (thread_position_in_grid.x == 0) { uint n = 0; for (uint i = 0; i < flags_shape[0]; ++i) n += flags[i]; count[0] = n; }";

fn gpuUnavailable() bool {
    return !mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu();
}

fn applyKernel(sc: *Scope, k: mlx.mlx_fast_metal_kernel, ins: []const A, outs: []const Output, grid: c_int, group: c_int, gs: c_int, qk: c_int) ![]A {
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    for (outs) |o| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, o.shape.ptr, o.shape.len, o.dtype));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, grid, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, group, 1, 1));
    if (gs != 0) try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "GS", gs));
    if (gs != 0) try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "QK", qk));
    if (gs != 0) try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", .bfloat16));
    const iv = mlx.mlx_vector_array_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, k, iv, cfg, sc.s));
    const result = try testing.allocator.alloc(A, outs.len);
    errdefer testing.allocator.free(result);
    for (result, 0..) |*a, i| {
        var out = mlx.mlx_array_new();
        a.* = try sc.res(mlx.mlx_vector_array_get(&out, ov, i), &out);
    }
    return result;
}

const Output = struct { shape: []const c_int, dtype: mlx.mlx_dtype };

fn kernel(name: [:0]const u8, ins: []const [*:0]const u8, outs: []const [*:0]const u8, src: [:0]const u8, header: [:0]const u8) !mlx.mlx_fast_metal_kernel {
    return dsv41.buildKernel(name, ins, outs, src, header);
}

fn bf16Bits(value: f32) u16 {
    const bits: u32 = @bitCast(value);
    return @truncate((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
}

fn scaleFromE4M3(bits: u8) f32 {
    const sign: f32 = if (bits & 0x80 == 0) 1 else -1;
    const exponent = (bits >> 3) & 0x0f;
    const mantissa = bits & 0x07;
    if (exponent == 0) return sign * @as(f32, @floatFromInt(mantissa)) * 0x1p-9;
    return sign * (1.0 + @as(f32, @floatFromInt(mantissa)) / 8.0) * std.math.pow(f32, 2, @as(f32, @floatFromInt(exponent)) - 7.0);
}

fn scaleFromPow2(bits: u8) f32 {
    return @bitCast(@as(u32, bits) << 23);
}

fn unpackBf16(packed_bytes: []const u8, scales: []const u8, group_size: usize, power_of_two: bool) ![]u16 {
    const decoded = try testing.allocator.alloc(u16, packed_bytes.len * 2);
    errdefer testing.allocator.free(decoded);
    const groups = decoded.len / group_size;
    for (0..groups) |g| {
        const scale = if (power_of_two) scaleFromPow2(scales[g]) else scaleFromE4M3(scales[g]);
        for (0..group_size) |j| {
            const code = (packed_bytes[g * (group_size / 2) + j / 2] >> @as(u3, @intCast((j & 1) * 4))) & 15;
            const level = code & 7;
            const magnitude: f32 = switch (level) {
                0 => 0,
                1 => 0.5,
                2 => 1,
                3 => 1.5,
                4 => 2,
                5 => 3,
                6 => 4,
                else => 6,
            };
            const signed = if (code & 8 != 0) -magnitude else magnitude;
            decoded[g * group_size + j] = bf16Bits(signed * scale);
        }
    }
    return decoded;
}

fn evaluateTogether(arrays: []const A) !void {
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    for (arrays) |a| _ = mlx.mlx_vector_array_append_value(vec, a);
    try mlx.check(mlx.mlx_eval(vec));
}

test "dsv41 pack probe: a metal kernel writes a uint8 output" {
    if (gpuUnavailable()) return error.SkipZigTest;
    var sc = Scope.init(testing.allocator, mlx.gpuStream());
    defer sc.deinit();
    const input = [_]u8{ 1, 2, 3, 4, 9, 10, 15, 0 };
    const x = try sc.keep(mlx.mlx_array_new_data(&input, &.{8}, 1, .uint8));
    const k = try kernel("dsv41_pack_probe_u8", &.{"x"}, &.{"out"}, UINT8_SOURCE, "");
    defer _ = mlx.mlx_fast_metal_kernel_free(k);
    const got = try applyKernel(&sc, k, &.{x}, &.{.{ .shape = &.{4}, .dtype = .uint8 }}, 4, 32, 0, 0);
    defer testing.allocator.free(got);
    try mlx.check(mlx.mlx_array_eval(got[0]));
    try testing.expectEqualSlices(u8, &.{ 0x21, 0x43, 0xa9, 0x0f }, mlx.mlx_array_data_uint8(got[0]).?[0..4]);
}

test "dsv41 pack probe: -0.0 survives the pack" {
    if (gpuUnavailable()) return error.SkipZigTest;
    var sc = Scope.init(testing.allocator, mlx.gpuStream());
    defer sc.deinit();
    const values = [_]f32{
        -0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        3.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    };
    var bf16: [values.len]u16 = undefined;
    for (values, 0..) |v, i| bf16[i] = bf16Bits(v);
    const x = try sc.keep(mlx.mlx_array_new_data(&bf16, &.{64}, 1, .bfloat16));
    const pack = try kernel("dsv41_pack_probe_fp4", &.{"x"}, &.{ "packed", "scales", "flags" }, PACK_SOURCE, FP8_BITS_HEADER);
    defer _ = mlx.mlx_fast_metal_kernel_free(pack);
    const got = try applyKernel(&sc, pack, &.{x}, &.{ .{ .shape = &.{32}, .dtype = .uint8 }, .{ .shape = &.{4}, .dtype = .uint8 }, .{ .shape = &.{4}, .dtype = .uint32 } }, 4, 32, 16, 3);
    defer testing.allocator.free(got);
    try evaluateTogether(got);
    const packed_bytes = mlx.mlx_array_data_uint8(got[0]).?[0..32];
    const scales = mlx.mlx_array_data_uint8(got[1]).?[0..4];
    const flags = mlx.mlx_array_data_uint32(got[2]).?[0..4];
    try testing.expectEqual(@as(u8, 0x08), packed_bytes[0]);
    try testing.expectEqual(@as(u8, 1), scales[0]);
    try testing.expectEqual(@as(u16, 0x8000), bf16Bits(-0.0));
    try testing.expectEqual(@as(u8, 0x06), packed_bytes[24]);
    try testing.expectEqualSlices(u32, &.{ 0, 0, 0, 0 }, flags);
    const restored = try unpackBf16(packed_bytes, scales, 16, false);
    defer testing.allocator.free(restored);
    try testing.expectEqualSlices(u16, &bf16, restored);
    const floor = scaleFromE4M3(scales[0]);
    try testing.expectEqual(@as(u32, @bitCast(@as(f32, 0x1p-9))), @as(u32, @bitCast(floor)));
    const level_six = scaleFromE4M3(scales[3]);
    try testing.expectEqual(@as(u16, bf16Bits(6.0 * level_six)), bf16[48]);

    const qk2_values = [_]f32{ 1.625, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.625, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
    var qk2_bf16: [qk2_values.len]u16 = undefined;
    for (qk2_values, 0..) |v, i| qk2_bf16[i] = bf16Bits(v);
    const qk2 = try sc.keep(mlx.mlx_array_new_data(&qk2_bf16, &.{32}, 1, .bfloat16));
    const qk2_pack = try applyKernel(&sc, pack, &.{qk2}, &.{ .{ .shape = &.{16}, .dtype = .uint8 }, .{ .shape = &.{1}, .dtype = .uint8 }, .{ .shape = &.{1}, .dtype = .uint32 } }, 1, 32, 32, 2);
    defer testing.allocator.free(qk2_pack);
    try evaluateTogether(qk2_pack);
    const qk2_packed = mlx.mlx_array_data_uint8(qk2_pack[0]).?[0..16];
    const qk2_scales = mlx.mlx_array_data_uint8(qk2_pack[1]).?[0..1];
    const qk2_decoded = try unpackBf16(qk2_packed, qk2_scales, 32, true);
    defer testing.allocator.free(qk2_decoded);
    try testing.expectEqual(bf16Bits(1.5), qk2_decoded[0]);
    try testing.expectEqual(bf16Bits(1.5), qk2_bf16[1]);
    try testing.expectEqual(@as(u32, 1), mlx.mlx_array_data_uint32(qk2_pack[2]).?[0]);
    try testing.expectEqual(@as(u8, 126), qk2_scales[0]);
}

test "dsv41 pack probe: mismatch count is read without a sync on the caller" {
    if (gpuUnavailable()) return error.SkipZigTest;
    var sc = Scope.init(testing.allocator, mlx.gpuStream());
    defer sc.deinit();
    var values: [64]f32 = @splat(0);
    values[16] = 0.5;
    values[32] = 0.25;
    var input: [values.len]u16 = undefined;
    for (values, 0..) |v, i| input[i] = bf16Bits(v);
    const x = try sc.keep(mlx.mlx_array_new_data(&input, &.{64}, 1, .bfloat16));
    const pack = try kernel("dsv41_pack_probe_lazy", &.{"x"}, &.{ "packed", "scales", "flags" }, PACK_SOURCE, FP8_BITS_HEADER);
    defer _ = mlx.mlx_fast_metal_kernel_free(pack);
    const pack_out = try applyKernel(&sc, pack, &.{x}, &.{ .{ .shape = &.{32}, .dtype = .uint8 }, .{ .shape = &.{4}, .dtype = .uint8 }, .{ .shape = &.{4}, .dtype = .uint32 } }, 4, 32, 16, 3);
    defer testing.allocator.free(pack_out);
    const reduce = try kernel("dsv41_pack_probe_mismatch_reduce", &.{"flags"}, &.{"count"}, REDUCE_SOURCE, "");
    defer _ = mlx.mlx_fast_metal_kernel_free(reduce);
    const count = try applyKernel(&sc, reduce, &.{pack_out[2]}, &.{.{ .shape = &.{1}, .dtype = .uint32 }}, 1, 32, 4, 0);
    defer testing.allocator.free(count);
    try evaluateTogether(&.{ pack_out[0], pack_out[1], count[0] });
    try testing.expectEqual(@as(u32, 2), mlx.mlx_array_data_uint32(count[0]).?[0]);
    const clean: [32]f32 = @splat(0);
    var clean_bf16: [clean.len]u16 = undefined;
    for (clean, 0..) |v, i| clean_bf16[i] = bf16Bits(v);
    const clean_x = try sc.keep(mlx.mlx_array_new_data(&clean_bf16, &.{32}, 1, .bfloat16));
    const clean_pack = try applyKernel(&sc, pack, &.{clean_x}, &.{ .{ .shape = &.{16}, .dtype = .uint8 }, .{ .shape = &.{2}, .dtype = .uint8 }, .{ .shape = &.{2}, .dtype = .uint32 } }, 2, 32, 16, 3);
    defer testing.allocator.free(clean_pack);
    const clean_count = try applyKernel(&sc, reduce, &.{clean_pack[2]}, &.{.{ .shape = &.{1}, .dtype = .uint32 }}, 1, 32, 2, 0);
    defer testing.allocator.free(clean_count);
    try evaluateTogether(&.{ clean_pack[0], clean_pack[1], clean_count[0] });
    try testing.expectEqual(@as(u32, 0), mlx.mlx_array_data_uint32(clean_count[0]).?[0]);
}
