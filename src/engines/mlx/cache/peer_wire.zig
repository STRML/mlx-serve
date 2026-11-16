//! Binary wire envelope and tensor-table codec for peer prefix-cache module transfer.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const kv_quant = @import("../../../sdk/blocks/attention/kv_quant.zig");
const module_record = @import("ssd/module_record.zig");

pub const MAGIC: [8]u8 = "MSVPFC\x00\x00".*;
pub const VERSION: u32 = 1;
pub const FIXED_HEADER_BYTES: usize = 96;
pub const MAX_TENSORS: usize = 1024;

pub const DecodeError = error{
    OutOfMemory,
    TruncatedEnvelope,
    BadEnvelope,
    UnsupportedEnvelopeVersion,
    BadQuantConfig,
    BadTensorName,
    BadTensorDtype,
    BadTensorShape,
    BadTensorOffsets,
    BadChecksum,
};

pub const WireDescriptor = struct {
    name: []const u8,
    dtype: []const u8,
    mlx_dtype: mlx.mlx_dtype,
    shape: []const u64,
    bytes: u64,
    offset: u64,
    end: u64,
};

pub const EnvelopeInput = struct {
    layout_key: u64,
    pack_identity: u64,
    settings: u64,
    prefix_key: u64,
    cache_key: u64,
    variant: u64,
    has_tools: bool,
    quant_config: kv_quant.KVQuantConfig,
    ids: []const u32,
    plugin_header: []const u8,
    tensors: []const module_record.Tensor,
};

pub const DecodedEnvelope = struct {
    arena: std.heap.ArenaAllocator,
    layout_key: u64,
    pack_identity: u64,
    settings: u64,
    prefix_key: u64,
    cache_key: u64,
    variant: u64,
    has_tools: bool,
    quant_config: kv_quant.KVQuantConfig,
    ids: []const u32,
    plugin_header: []const u8,
    tensors: []const module_record.Tensor,
    descriptors: []const WireDescriptor,
    payload: []const u8,

    pub fn deinit(self: *DecodedEnvelope) void {
        self.arena.deinit();
    }
};

pub const DtypeSpec = struct {
    name: []const u8,
    mlx_dtype: mlx.mlx_dtype,
    itemsize: u64,
};

const DTYPE_TABLE = [_]DtypeSpec{
    .{ .name = "BOOL", .mlx_dtype = .bool_, .itemsize = 1 },
    .{ .name = "U8", .mlx_dtype = .uint8, .itemsize = 1 },
    .{ .name = "I8", .mlx_dtype = .int8, .itemsize = 1 },
    .{ .name = "U16", .mlx_dtype = .uint16, .itemsize = 2 },
    .{ .name = "I16", .mlx_dtype = .int16, .itemsize = 2 },
    .{ .name = "F16", .mlx_dtype = .float16, .itemsize = 2 },
    .{ .name = "BF16", .mlx_dtype = .bfloat16, .itemsize = 2 },
    .{ .name = "U32", .mlx_dtype = .uint32, .itemsize = 4 },
    .{ .name = "I32", .mlx_dtype = .int32, .itemsize = 4 },
    .{ .name = "F32", .mlx_dtype = .float32, .itemsize = 4 },
    .{ .name = "U64", .mlx_dtype = .uint64, .itemsize = 8 },
    .{ .name = "I64", .mlx_dtype = .int64, .itemsize = 8 },
};

pub fn lookupDtype(name: []const u8) ?DtypeSpec {
    for (DTYPE_TABLE) |spec| {
        if (std.mem.eql(u8, spec.name, name)) return spec;
    }
    return null;
}

fn canonicalName(name: []const u8) bool {
    if (name.len == 0 or name.len > 256 or std.mem.eql(u8, name, "__metadata__")) return false;
    for (name) |c| {
        if (!std.ascii.isAlphanumeric(c) and c != '_' and c != '.' and c != '-') return false;
    }
    return true;
}

pub fn encodedMetaSize(n_tokens: usize, header_len: usize, tensors: []const module_record.Tensor) !usize {
    if (n_tokens == 0 or n_tokens > std.math.maxInt(u32)) return error.BadEnvelope;
    const token_bytes = try std.math.mul(usize, n_tokens, 4);
    if (@as(u64, token_bytes) > module_record.MAX_TOKEN_BYTES) return error.BadEnvelope;
    if (header_len == 0 or header_len > module_record.MAX_HEADER_BYTES) return error.BadEnvelope;
    if (tensors.len == 0 or tensors.len > MAX_TENSORS) return error.BadEnvelope;
    var size = try std.math.add(usize, FIXED_HEADER_BYTES, token_bytes);
    size = try std.math.add(usize, size, header_len);
    for (tensors) |t| {
        if (!canonicalName(t.name)) return error.BadTensorName;
        if (lookupDtype(t.dtype) == null) return error.BadTensorDtype;
        if (t.shape.len == 0 or t.shape.len > 8) return error.BadTensorShape;
        const desc_fixed = 2 + t.name.len + 1 + t.dtype.len + 1 + t.shape.len * 8 + 24;
        size = try std.math.add(usize, size, desc_fixed);
    }
    return size;
}

pub fn encodedSize(n_tokens: usize, header_len: usize, tensors: []const module_record.Tensor) !usize {
    var total = try encodedMetaSize(n_tokens, header_len, tensors);
    for (tensors) |t| {
        const tb = std.math.cast(usize, t.bytes) orelse return error.BadTensorShape;
        total = try std.math.add(usize, total, tb);
    }
    return total;
}

/// Writes everything up to the tensor payload into `out` and returns the byte offset where payload bytes begin.
/// Call `stampChecksum(out)` once the payload bytes are written.
pub fn writeEnvelopePrefix(out: []u8, in: EnvelopeInput) !usize {
    const meta_len = try encodedMetaSize(in.ids.len, in.plugin_header.len, in.tensors);
    var payload_bytes: u64 = 0;
    for (in.tensors, 0..) |t, i| {
        for (in.tensors[0..i]) |prev| {
            if (std.mem.eql(u8, prev.name, t.name)) return error.BadTensorName;
        }
        const spec = lookupDtype(t.dtype) orelse return error.BadTensorDtype;
        if (t.shape.len == 0 or t.shape.len > 8) return error.BadTensorShape;
        var prod: u64 = spec.itemsize;
        for (t.shape) |d| {
            if (d == 0 or d > std.math.maxInt(c_int)) return error.BadTensorShape;
            prod = std.math.mul(u64, prod, d) catch return error.BadTensorShape;
        }
        if (prod != t.bytes) return error.BadTensorShape;
        payload_bytes = std.math.add(u64, payload_bytes, t.bytes) catch return error.BadTensorOffsets;
    }
    const total_len = std.math.add(usize, meta_len, std.math.cast(usize, payload_bytes) orelse return error.BadTensorOffsets) catch return error.BadTensorOffsets;
    if (out.len != total_len) return error.BadEnvelope;

    const scheme_byte: u8 = switch (in.quant_config.scheme) {
        .off => 0,
        .affine => 1,
    };
    if (in.quant_config.scheme == .off) {
        if (in.quant_config.bits != 0 or in.quant_config.group_size != 0) return error.BadQuantConfig;
    } else {
        if ((in.quant_config.bits != 4 and in.quant_config.bits != 8) or in.quant_config.group_size == 0) return error.BadQuantConfig;
    }

    @memcpy(out[0..8], &MAGIC);
    std.mem.writeInt(u32, out[8..12], VERSION, .little);
    out[12] = if (in.has_tools) 1 else 0;
    out[13] = scheme_byte;
    out[14] = in.quant_config.bits;
    out[15] = 0;
    std.mem.writeInt(u32, out[16..20], in.quant_config.group_size, .little);
    std.mem.writeInt(u32, out[20..24], @intCast(in.ids.len), .little);
    std.mem.writeInt(u32, out[24..28], @intCast(in.plugin_header.len), .little);
    std.mem.writeInt(u32, out[28..32], @intCast(in.tensors.len), .little);
    std.mem.writeInt(u64, out[32..40], in.layout_key, .little);
    std.mem.writeInt(u64, out[40..48], in.pack_identity, .little);
    std.mem.writeInt(u64, out[48..56], in.settings, .little);
    std.mem.writeInt(u64, out[56..64], in.prefix_key, .little);
    std.mem.writeInt(u64, out[64..72], in.cache_key, .little);
    std.mem.writeInt(u64, out[72..80], in.variant, .little);
    std.mem.writeInt(u64, out[80..88], payload_bytes, .little);
    std.mem.writeInt(u64, out[88..96], 0, .little);

    var cursor: usize = FIXED_HEADER_BYTES;
    for (in.ids) |id| {
        std.mem.writeInt(u32, out[cursor..][0..4], id, .little);
        cursor += 4;
    }
    @memcpy(out[cursor..][0..in.plugin_header.len], in.plugin_header);
    cursor += in.plugin_header.len;

    var pay_off: u64 = 0;
    for (in.tensors) |t| {
        std.mem.writeInt(u16, out[cursor..][0..2], @intCast(t.name.len), .little);
        cursor += 2;
        @memcpy(out[cursor..][0..t.name.len], t.name);
        cursor += t.name.len;
        out[cursor] = @intCast(t.dtype.len);
        cursor += 1;
        @memcpy(out[cursor..][0..t.dtype.len], t.dtype);
        cursor += t.dtype.len;
        out[cursor] = @intCast(t.shape.len);
        cursor += 1;
        for (t.shape) |d| {
            std.mem.writeInt(u64, out[cursor..][0..8], d, .little);
            cursor += 8;
        }
        const pay_end = pay_off + t.bytes;
        std.mem.writeInt(u64, out[cursor..][0..8], t.bytes, .little);
        cursor += 8;
        std.mem.writeInt(u64, out[cursor..][0..8], pay_off, .little);
        cursor += 8;
        std.mem.writeInt(u64, out[cursor..][0..8], pay_end, .little);
        cursor += 8;
        pay_off = pay_end;
    }
    std.debug.assert(cursor == meta_len);
    return meta_len;
}

fn envelopeChecksum(body: []const u8) u64 {
    var h = std.hash.Wyhash.init(0);
    h.update(body[0..88]);
    h.update(body[96..]);
    return h.final();
}

pub fn stampChecksum(out: []u8) void {
    std.mem.writeInt(u64, out[88..96], envelopeChecksum(out), .little);
}

pub fn encodeAlloc(a: std.mem.Allocator, in: EnvelopeInput, tensor_payloads: []const []const u8) ![]u8 {
    if (tensor_payloads.len != in.tensors.len) return error.BadTensorOffsets;
    const total = try encodedSize(in.ids.len, in.plugin_header.len, in.tensors);
    const out = try a.alloc(u8, total);
    errdefer a.free(out);
    const payload_off = try writeEnvelopePrefix(out, in);
    var cursor = payload_off;
    for (in.tensors, tensor_payloads) |t, bytes| {
        if (@as(u64, bytes.len) != t.bytes) return error.BadTensorOffsets;
        @memcpy(out[cursor..][0..bytes.len], bytes);
        cursor += bytes.len;
    }
    stampChecksum(out);
    return out;
}

pub fn decode(a: std.mem.Allocator, body: []const u8) DecodeError!DecodedEnvelope {
    if (body.len < FIXED_HEADER_BYTES) return error.TruncatedEnvelope;
    if (!std.mem.eql(u8, body[0..8], &MAGIC)) return error.BadEnvelope;
    const version = std.mem.readInt(u32, body[8..12], .little);
    if (version != VERSION) return error.UnsupportedEnvelopeVersion;
    const flags = body[12];
    if (flags & ~@as(u8, 1) != 0 or body[15] != 0) return error.BadEnvelope;
    const has_tools = (flags & 1) != 0;
    const scheme_byte = body[13];
    const quant_bits = body[14];
    const quant_gs = std.mem.readInt(u32, body[16..20], .little);
    const quant_config: kv_quant.KVQuantConfig = switch (scheme_byte) {
        0 => blk: {
            if (quant_bits != 0 or quant_gs != 0) return error.BadQuantConfig;
            break :blk kv_quant.KVQuantConfig.dense;
        },
        1 => blk: {
            if ((quant_bits != 4 and quant_bits != 8) or quant_gs == 0) return error.BadQuantConfig;
            break :blk .{ .scheme = .affine, .bits = quant_bits, .group_size = quant_gs };
        },
        else => return error.BadQuantConfig,
    };

    const n_tokens: usize = @intCast(std.mem.readInt(u32, body[20..24], .little));
    const header_len: usize = @intCast(std.mem.readInt(u32, body[24..28], .little));
    const tensor_count: usize = @intCast(std.mem.readInt(u32, body[28..32], .little));
    if (n_tokens == 0 or @as(u64, n_tokens) * 4 > module_record.MAX_TOKEN_BYTES) return error.BadEnvelope;
    if (header_len == 0 or header_len > module_record.MAX_HEADER_BYTES) return error.BadEnvelope;
    if (tensor_count == 0 or tensor_count > MAX_TENSORS) return error.BadEnvelope;

    const layout_key = std.mem.readInt(u64, body[32..40], .little);
    const pack_identity = std.mem.readInt(u64, body[40..48], .little);
    const settings = std.mem.readInt(u64, body[48..56], .little);
    const prefix_key = std.mem.readInt(u64, body[56..64], .little);
    const cache_key = std.mem.readInt(u64, body[64..72], .little);
    const variant = std.mem.readInt(u64, body[72..80], .little);
    const payload_bytes = std.mem.readInt(u64, body[80..88], .little);
    const checksum = std.mem.readInt(u64, body[88..96], .little);

    var cursor: usize = FIXED_HEADER_BYTES;
    const token_bytes: usize = n_tokens * 4;
    if (body.len - cursor < token_bytes) return error.TruncatedEnvelope;

    var arena = std.heap.ArenaAllocator.init(a);
    errdefer arena.deinit();
    const aa = arena.allocator();

    const ids = try aa.alloc(u32, n_tokens);
    for (ids) |*id| {
        id.* = std.mem.readInt(u32, body[cursor..][0..4], .little);
        cursor += 4;
    }
    if (body.len - cursor < header_len) return error.TruncatedEnvelope;
    const plugin_header = body[cursor..][0..header_len];
    cursor += header_len;

    const tensors = try aa.alloc(module_record.Tensor, tensor_count);
    const descriptors = try aa.alloc(WireDescriptor, tensor_count);
    var expected_off: u64 = 0;
    for (0..tensor_count) |i| {
        if (body.len - cursor < 2) return error.TruncatedEnvelope;
        const name_len: usize = std.mem.readInt(u16, body[cursor..][0..2], .little);
        cursor += 2;
        if (body.len - cursor < name_len) return error.TruncatedEnvelope;
        const name = body[cursor..][0..name_len];
        cursor += name_len;
        if (!canonicalName(name)) return error.BadTensorName;
        for (descriptors[0..i]) |prev| {
            if (std.mem.eql(u8, prev.name, name)) return error.BadTensorName;
        }

        if (body.len - cursor < 1) return error.TruncatedEnvelope;
        const dtype_len: usize = body[cursor];
        cursor += 1;
        if (body.len - cursor < dtype_len) return error.TruncatedEnvelope;
        const dtype_str = body[cursor..][0..dtype_len];
        cursor += dtype_len;
        const spec = lookupDtype(dtype_str) orelse return error.BadTensorDtype;

        if (body.len - cursor < 1) return error.TruncatedEnvelope;
        const rank: usize = body[cursor];
        cursor += 1;
        if (rank == 0 or rank > 8) return error.BadTensorShape;
        const shape_bytes: usize = rank * 8;
        if (body.len - cursor < shape_bytes + 24) return error.TruncatedEnvelope;
        const shape = try aa.alloc(u64, rank);
        var prod: u64 = spec.itemsize;
        for (shape) |*dim| {
            const d = std.mem.readInt(u64, body[cursor..][0..8], .little);
            cursor += 8;
            if (d == 0 or d > std.math.maxInt(c_int)) return error.BadTensorShape;
            prod = std.math.mul(u64, prod, d) catch return error.BadTensorShape;
            dim.* = d;
        }
        const t_bytes = std.mem.readInt(u64, body[cursor..][0..8], .little);
        cursor += 8;
        const t_off = std.mem.readInt(u64, body[cursor..][0..8], .little);
        cursor += 8;
        const t_end = std.mem.readInt(u64, body[cursor..][0..8], .little);
        cursor += 8;

        if (prod != t_bytes) return error.BadTensorShape;
        if (t_off != expected_off or t_end < t_off or t_end - t_off != t_bytes) return error.BadTensorOffsets;
        expected_off = std.math.add(u64, expected_off, t_bytes) catch return error.BadTensorOffsets;

        tensors[i] = .{
            .name = name,
            .dtype = dtype_str,
            .shape = shape,
            .bytes = t_bytes,
        };
        descriptors[i] = .{
            .name = name,
            .dtype = dtype_str,
            .mlx_dtype = spec.mlx_dtype,
            .shape = shape,
            .bytes = t_bytes,
            .offset = t_off,
            .end = t_end,
        };
    }

    if (expected_off != payload_bytes) return error.BadTensorOffsets;
    const pay_len = std.math.cast(usize, payload_bytes) orelse return error.BadTensorOffsets;
    if (body.len - cursor < pay_len) return error.TruncatedEnvelope;
    if (body.len - cursor > pay_len) return error.BadTensorOffsets;
    const payload = body[cursor..][0..pay_len];
    if (envelopeChecksum(body) != checksum) return error.BadChecksum;

    return .{
        .arena = arena,
        .layout_key = layout_key,
        .pack_identity = pack_identity,
        .settings = settings,
        .prefix_key = prefix_key,
        .cache_key = cache_key,
        .variant = variant,
        .has_tools = has_tools,
        .quant_config = quant_config,
        .ids = ids,
        .plugin_header = plugin_header,
        .tensors = tensors,
        .descriptors = descriptors,
        .payload = payload,
    };
}

// ── tests ──

const testing = std.testing;

fn sampleEnvelope(a: std.mem.Allocator) ![]u8 {
    const s0 = [_]u64{ 2, 4 };
    const s1 = [_]u64{3};
    const tensors = [_]module_record.Tensor{
        .{ .name = "l0.win", .dtype = "BF16", .shape = &s0, .bytes = 16 },
        .{ .name = "hist", .dtype = "U32", .shape = &s1, .bytes = 12 },
    };
    const p0 = [_]u8{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
    const p1 = [_]u8{ 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32 };
    return encodeAlloc(a, .{
        .layout_key = 0x1111_2222_3333_4444,
        .pack_identity = 0x5555_6666_7777_8888,
        .settings = 0x9999_aaaa_bbbb_cccc,
        .prefix_key = 0xdead_beef,
        .cache_key = 0xcafe_babe,
        .variant = 7,
        .has_tools = true,
        .quant_config = kv_quant.KVQuantConfig.affine(8),
        .ids = &.{ 10, 20, 30 },
        .plugin_header = "DV41A001-hdr",
        .tensors = &tensors,
    }, &.{ &p0, &p1 });
}

test "dsv41 peer fetch: envelope and tensor descriptor round-trip" {
    const wire = try sampleEnvelope(testing.allocator);
    defer testing.allocator.free(wire);
    var dec = try decode(testing.allocator, wire);
    defer dec.deinit();

    try testing.expectEqual(@as(u64, 0x1111_2222_3333_4444), dec.layout_key);
    try testing.expectEqual(@as(u64, 0x5555_6666_7777_8888), dec.pack_identity);
    try testing.expectEqual(@as(u64, 0x9999_aaaa_bbbb_cccc), dec.settings);
    try testing.expectEqual(@as(u64, 0xdead_beef), dec.prefix_key);
    try testing.expectEqual(@as(u64, 0xcafe_babe), dec.cache_key);
    try testing.expectEqual(@as(u64, 7), dec.variant);
    try testing.expect(dec.has_tools);
    try testing.expectEqual(kv_quant.KVQuantConfig.affine(8), dec.quant_config);
    try testing.expectEqualSlices(u32, &.{ 10, 20, 30 }, dec.ids);
    try testing.expectEqualStrings("DV41A001-hdr", dec.plugin_header);
    try testing.expectEqual(@as(usize, 2), dec.tensors.len);
    try testing.expectEqualStrings("l0.win", dec.tensors[0].name);
    try testing.expectEqualStrings("BF16", dec.tensors[0].dtype);
    try testing.expectEqual(mlx.mlx_dtype.bfloat16, dec.descriptors[0].mlx_dtype);
    try testing.expectEqualSlices(u64, &.{ 2, 4 }, dec.tensors[0].shape);
    try testing.expectEqual(@as(u64, 16), dec.tensors[0].bytes);
    try testing.expectEqualStrings("hist", dec.tensors[1].name);
    try testing.expectEqual(mlx.mlx_dtype.uint32, dec.descriptors[1].mlx_dtype);
    try testing.expectEqualSlices(u64, &.{3}, dec.tensors[1].shape);
    try testing.expectEqual(@as(u64, 12), dec.tensors[1].bytes);
    try testing.expectEqual(@as(usize, 28), dec.payload.len);
}

test "dsv41 peer fetch: envelope rejects unknown version, short body and bad checksum" {
    const wire = try sampleEnvelope(testing.allocator);
    defer testing.allocator.free(wire);

    try testing.expectError(error.TruncatedEnvelope, decode(testing.allocator, wire[0..10]));
    try testing.expectError(error.TruncatedEnvelope, decode(testing.allocator, wire[0 .. wire.len - 1]));

    const bad_ver = try testing.allocator.dupe(u8, wire);
    defer testing.allocator.free(bad_ver);
    std.mem.writeInt(u32, bad_ver[8..12], 99, .little);
    try testing.expectError(error.UnsupportedEnvelopeVersion, decode(testing.allocator, bad_ver));

    const bad_magic = try testing.allocator.dupe(u8, wire);
    defer testing.allocator.free(bad_magic);
    bad_magic[0] = 'X';
    try testing.expectError(error.BadEnvelope, decode(testing.allocator, bad_magic));

    const bad_sum = try testing.allocator.dupe(u8, wire);
    defer testing.allocator.free(bad_sum);
    bad_sum[bad_sum.len - 1] ^= 0xff;
    try testing.expectError(error.BadChecksum, decode(testing.allocator, bad_sum));
}

test "dsv41 peer fetch: the checksum covers token ids and flags, not only the payload" {
    const wire = try sampleEnvelope(testing.allocator);
    defer testing.allocator.free(wire);

    const other_id = try testing.allocator.dupe(u8, wire);
    defer testing.allocator.free(other_id);
    std.mem.writeInt(u32, other_id[FIXED_HEADER_BYTES..][0..4], 11, .little);
    try testing.expectError(error.BadChecksum, decode(testing.allocator, other_id));

    const no_tools = try testing.allocator.dupe(u8, wire);
    defer testing.allocator.free(no_tools);
    no_tools[12] = 0;
    try testing.expectError(error.BadChecksum, decode(testing.allocator, no_tools));
}

test "dsv41 peer fetch: envelope rejects bad tensor dtype, shape and payload offsets" {
    const wire = try sampleEnvelope(testing.allocator);
    defer testing.allocator.free(wire);

    const desc_pos = FIXED_HEADER_BYTES + 3 * 4 + "DV41A001-hdr".len;
    const dtype_pos = desc_pos + 2 + "l0.win".len + 1;

    const bad_dt = try testing.allocator.dupe(u8, wire);
    defer testing.allocator.free(bad_dt);
    @memcpy(bad_dt[dtype_pos..][0..4], "FP64");
    try testing.expectError(error.BadTensorDtype, decode(testing.allocator, bad_dt));

    const rank_pos = dtype_pos + 4;
    const zero_dim = try testing.allocator.dupe(u8, wire);
    defer testing.allocator.free(zero_dim);
    std.mem.writeInt(u64, zero_dim[rank_pos + 1 ..][0..8], 0, .little);
    try testing.expectError(error.BadTensorShape, decode(testing.allocator, zero_dim));

    const huge_dim = try testing.allocator.dupe(u8, wire);
    defer testing.allocator.free(huge_dim);
    std.mem.writeInt(u64, huge_dim[rank_pos + 1 ..][0..8], @as(u64, std.math.maxInt(c_int)) + 1, .little);
    try testing.expectError(error.BadTensorShape, decode(testing.allocator, huge_dim));

    const bad_bytes = try testing.allocator.dupe(u8, wire);
    defer testing.allocator.free(bad_bytes);
    std.mem.writeInt(u64, bad_bytes[rank_pos + 1 + 16 ..][0..8], 14, .little);
    try testing.expectError(error.BadTensorShape, decode(testing.allocator, bad_bytes));

    const rank_9 = try testing.allocator.alloc(u8, wire.len + 7 * 8);
    defer testing.allocator.free(rank_9);
    @memcpy(rank_9[0..rank_pos], wire[0..rank_pos]);
    rank_9[rank_pos] = 9;
    @memset(rank_9[rank_pos + 1 ..][0 .. 9 * 8], 1);
    @memcpy(rank_9[rank_pos + 1 + 9 * 8 ..], wire[rank_pos + 1 + 2 * 8 ..]);
    try testing.expectError(error.BadTensorShape, decode(testing.allocator, rank_9));

    const bad_off = try testing.allocator.dupe(u8, wire);
    defer testing.allocator.free(bad_off);
    const off_pos = rank_pos + 1 + 16 + 8;
    std.mem.writeInt(u64, bad_off[off_pos..][0..8], 4, .little);
    std.mem.writeInt(u64, bad_off[off_pos + 8 ..][0..8], 20, .little);
    try testing.expectError(error.BadTensorOffsets, decode(testing.allocator, bad_off));

    const trailing = try testing.allocator.alloc(u8, wire.len + 1);
    defer testing.allocator.free(trailing);
    @memcpy(trailing[0..wire.len], wire);
    trailing[wire.len] = 0;
    try testing.expectError(error.BadTensorOffsets, decode(testing.allocator, trailing));
}
