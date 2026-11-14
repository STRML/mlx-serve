//! Bounded, allocation-free description of a V4.1 checkpoint's tensor inventory.
const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const mlx = @import("../../mlx/ffi.zig");

pub const max_header = 64 * 1024;
pub const max_tensors = 512;
const fixed_bytes = 48;
const v1_spec_bytes = 56;
const v2_spec_bytes = 64;
const magic_v1 = "DV41A001";
const magic_v2 = "DV41A002";
// SSD restores measured 2026-11-14 (M5 Ultra): footprint high-water = loaded + destination + a transient of
// 0.53-0.59 x destination (tokens: loaded / destination / measured bytes):
// 32,791: 111,444,060 / 137,672,540 / 321,896,520; 131,095: 426,410,076 / 531,281,756 / 1,267,482,912;
// 262,167: 846,364,764 / 1,056,094,044 / 2,524,856,896. Peak = 1.10 x (loaded + 1.6 x destination): the transient is
// charged to destination, which a packed (smaller) loaded size does not shrink. 1M is unmeasured.
const import_peak_loaded_pct: u64 = 110;
const import_peak_destination_pct: u64 = 176;

pub const Encoding = enum(u8) { raw, fp4_e4m3_g16, fp4_ue8m0_g32, fp8_ue8m0_g32 };

pub fn packedRowBytes(enc: Encoding, width: usize) ?usize {
    const group: usize = switch (enc) {
        .raw => return null,
        .fp4_e4m3_g16 => 16,
        .fp4_ue8m0_g32, .fp8_ue8m0_g32 => 32,
    };
    if (width == 0 or width % group != 0) return null;
    return switch (enc) {
        .raw => unreachable,
        .fp4_e4m3_g16, .fp4_ue8m0_g32 => width / 2 + width / group,
        .fp8_ue8m0_g32 => width + width / group,
    };
}

pub fn candidate(spec: Spec, qat_on: bool) Encoding {
    if (!qat_on or spec.dtype != .bfloat16) return .raw;
    const enc: Encoding = switch (spec.kind) {
        .comp => .fp4_e4m3_g16,
        .index => .fp4_ue8m0_g32,
        .win, .ds => .fp8_ue8m0_g32,
        .pend_kv, .pend_gate, .mh, .hist => .raw,
    };
    return if (enc == .raw or packedRowBytes(enc, spec.width) == null) .raw else enc;
}

pub const Layout = struct {
    layers: usize,
    stages: usize,
    ring: usize,
    width: usize,
    index_width: usize,
    hidden_width: usize,
    dtype: mlx.mlx_dtype,
    kv_sources: u64,
    index_sources: u64,
    ratios: [64]u8 = @splat(0),
    engram: bool,
    settings: u64,

    pub fn source(self: Layout, i: usize) bool {
        return self.kv_sources & (@as(u64, 1) << @intCast(i)) != 0;
    }

    pub fn owner(self: Layout, i: usize) bool {
        return self.source(i) and self.index_sources & (@as(u64, 1) << @intCast(i)) != 0;
    }

    fn validate(self: Layout) !void {
        if (self.layers == 0 or self.layers > 64 or self.stages > 128) return error.Dsv41CacheGeometry;
        for ([_]usize{ self.ring, self.width, self.index_width }) |d| try dimension(d);
        if (self.stages > 0) try dimension(self.hidden_width);
        _ = try itemBytes(self.dtype);
        if (self.layers < 64 and (self.kv_sources | self.index_sources) >> @intCast(self.layers) != 0) return error.Dsv41CacheGeometry;
        for (0..self.layers) |i| {
            if (self.source(i) and self.ratios[i] == 0) return error.Dsv41CacheGeometry;
        }
    }

    fn key(self: Layout) u64 {
        var h = std.hash.Wyhash.init(0x41ca);
        std.hash.autoHash(&h, self);
        return h.final();
    }
};

pub const Kind = enum(u8) { win, comp, index, pend_kv, pend_gate, ds, mh, hist };

pub const Spec = struct {
    kind: Kind,
    layer: usize = 0,
    rows: usize,
    width: usize,
    dtype: mlx.mlx_dtype,

    pub fn rank(self: Spec) usize {
        return if (self.kind == .hist) 1 else 2;
    }

    pub fn name(self: Spec, buf: []u8) ![]const u8 {
        return switch (self.kind) {
            .ds => std.fmt.bufPrint(buf, "d{d}.win", .{self.layer}),
            .mh => "mh",
            .hist => "hist",
            else => std.fmt.bufPrint(buf, "l{d}.{s}", .{ self.layer, @tagName(self.kind) }),
        };
    }

    pub fn bytes(self: Spec) !u64 {
        try dimension(self.rows);
        try dimension(self.width);
        return checkedMul(try checkedMul(self.rows, self.width), try itemBytes(self.dtype));
    }

    fn words(self: Spec) ![7]u64 {
        return .{ @backingInt(self.kind), self.layer, self.rows, self.width, self.rank(), @intCast(@backingInt(self.dtype)), try self.bytes() };
    }
};

pub const Plan = struct {
    layout: Layout,
    n: usize,
    vision_key: u64,
    items: [max_tensors]Spec = undefined,
    encodings: [max_tensors]Encoding = @splat(.raw),
    len: usize = 0,
    version: u8 = 1,
    qat_on: bool = true,
    loaded: u64 = 0,
    destination: u64 = 0,
    max_unpacked: u64 = 0,
    peak: u64 = 0,

    pub fn specs(self: *const Plan) []const Spec {
        return self.items[0..self.len];
    }

    pub fn init(l: Layout, n: usize, vision_key: u64) !Plan {
        return initWithPacking(l, n, vision_key, false);
    }

    pub fn initWithPacking(l: Layout, n: usize, vision_key: u64, qat_on: bool) !Plan {
        try l.validate();
        try dimension(n);
        var p: Plan = .{ .layout = l, .n = n, .vision_key = vision_key, .qat_on = qat_on };
        for (0..l.layers) |i| try p.layer(i);
        for (0..l.stages) |i| try p.add(.{ .kind = .ds, .layer = i, .rows = l.ring, .width = l.width, .dtype = l.dtype });
        if (l.stages > 0) try p.add(.{ .kind = .mh, .rows = 1, .width = l.hidden_width, .dtype = l.dtype });
        if (l.engram) try p.add(.{ .kind = .hist, .rows = n, .width = 1, .dtype = .uint32 });
        for (p.specs(), 0..) |spec, i| p.encodings[i] = candidate(spec, qat_on);
        try p.reprice();
        return p;
    }

    fn layer(self: *Plan, i: usize) !void {
        const l = self.layout;
        try self.add(.{ .kind = .win, .layer = i, .rows = l.ring, .width = l.width, .dtype = l.dtype });
        if (!l.source(i)) return;
        const used = self.n / l.ratios[i];
        try self.row(.comp, i, used, l.width);
        if (l.owner(i)) try self.row(.index, i, used, l.index_width);
        const pending = self.n % l.ratios[i];
        if (pending == 0) return;
        try self.add(.{ .kind = .pend_kv, .layer = i, .rows = pending, .width = l.width, .dtype = .float32 });
        try self.add(.{ .kind = .pend_gate, .layer = i, .rows = pending, .width = l.width, .dtype = .float32 });
    }

    fn row(self: *Plan, kind: Kind, i: usize, used: usize, width: usize) !void {
        const spec: Spec = .{ .kind = kind, .layer = i, .rows = used, .width = width, .dtype = self.layout.dtype };
        if (used > 0) try self.add(spec);
        const cap = try rowCapacity(used, try std.math.add(usize, used, 1));
        self.destination = try checkedAdd(self.destination, try checkedMul(try checkedMul(cap - used, width), try itemBytes(spec.dtype)));
    }

    fn add(self: *Plan, spec: Spec) !void {
        if (self.len == max_tensors) return error.Dsv41CacheGeometry;
        const bytes = try spec.bytes();
        self.loaded = try checkedAdd(self.loaded, bytes);
        self.destination = try checkedAdd(self.destination, bytes);
        self.items[self.len] = spec;
        self.len += 1;
    }

    pub fn encode(self: *const Plan, buf: []u8) ![]const u8 {
        const spec_bytes: usize = if (self.version == 1) v1_spec_bytes else if (self.version == 2) v2_spec_bytes else return error.Dsv41CacheHeaderVersion;
        const len = fixed_bytes + self.len * spec_bytes;
        if (len > max_header or len > buf.len) return error.Dsv41CacheStorage;
        @memcpy(buf[0..8], if (self.version == 1) magic_v1 else magic_v2);
        const words = [_]u64{ self.layout.key(), self.n, self.vision_key, self.n, self.len };
        for (words, 0..) |w, i| put(buf, 8 + i * 8, w);
        for (self.specs(), 0..) |spec, i| {
            for (try spec.words(), 0..) |w, k| put(buf, fixed_bytes + i * spec_bytes + k * 8, w);
            if (self.version == 2) put(buf, fixed_bytes + i * spec_bytes + 56, @backingInt(self.encodings[i]));
        }
        return buf[0..len];
    }

    pub fn decode(l: Layout, bytes: []const u8) !Plan {
        return decodeWithPacking(l, bytes, true);
    }

    pub fn decodeWithPacking(l: Layout, bytes: []const u8, qat_on: bool) !Plan {
        if (bytes.len < fixed_bytes or bytes.len > max_header) return error.Dsv41CacheHeader;
        const version: u8 = if (std.mem.eql(u8, bytes[0..8], magic_v1)) 1 else if (std.mem.eql(u8, bytes[0..8], magic_v2)) 2 else return error.Dsv41CacheHeaderVersion;
        const spec_bytes: usize = if (version == 1) v1_spec_bytes else v2_spec_bytes;
        const n = std.math.cast(usize, get(bytes, 16)) orelse return error.Dsv41CacheHeader;
        var p = Plan.initWithPacking(l, n, get(bytes, 24), qat_on) catch return error.Dsv41CacheHeader;
        p.version = version;
        if (version == 1) {
            @memset(p.encodings[0..p.len], .raw);
            try p.reprice();
        } else {
            if (bytes.len != fixed_bytes + p.len * spec_bytes) return error.Dsv41CacheHeader;
            for (p.specs(), 0..) |spec, i| {
                const raw = get(bytes, fixed_bytes + i * spec_bytes + 56);
                const value = std.math.cast(u8, raw) orelse return error.Dsv41CacheEncoding;
                if (value > @backingInt(Encoding.fp8_ue8m0_g32)) return error.Dsv41CacheEncoding;
                const enc: Encoding = @fromBackingInt(@intCast(value));
                if (enc != candidate(spec, qat_on)) return error.Dsv41CacheEncoding;
                p.encodings[i] = enc;
            }
            try p.reprice();
        }
        var buf: [max_header]u8 = undefined;
        const expected = try p.encode(&buf);
        if (!std.mem.eql(u8, expected, bytes)) return error.Dsv41CacheHeader;
        return p;
    }

    pub fn reprice(self: *Plan) !void {
        self.loaded = 0;
        self.max_unpacked = 0;
        for (self.specs(), 0..) |spec, i| {
            const logical = try spec.bytes();
            const physical = if (self.encodings[i] == .raw) logical else try checkedMul(spec.rows, packedRowBytes(self.encodings[i], spec.width) orelse return error.Dsv41CacheGeometry);
            self.loaded = try checkedAdd(self.loaded, physical);
            if (self.encodings[i] != .raw) self.max_unpacked = @max(self.max_unpacked, logical);
        }
        const host = try checkedAdd(try checkedMul(self.n, 12), try checkedAdd(2 * max_header, try checkedMul(self.len + self.layout.layers + self.layout.stages + 1, 4096)));
        self.peak = try checkedAdd(try importPeak(self.loaded, self.destination, host), self.max_unpacked);
    }

    /// An unordered borrowed list is checked in full before the owner allocates any destination.
    pub fn validateArrays(self: *const Plan, arrays: []const abi.AuxTensor) ![max_tensors]usize {
        if (arrays.len != self.len) return error.Dsv41CacheInventory;
        var order: [max_tensors]usize = undefined;
        var seen: [max_tensors]bool = @splat(false);
        for (arrays, 0..) |tensor, i| {
            if (tensor.size < @sizeOf(abi.AuxTensor) or tensor.array.ctx == null) return error.Dsv41CacheTensor;
            const k = try self.find(tensor.name.slice());
            if (seen[k]) return error.Dsv41CacheDuplicate;
            seen[k] = true;
            try validateEncodedArray(self.items[k], self.encodings[k], .{ .ctx = tensor.array.ctx });
            order[k] = i;
        }
        return order;
    }

    fn find(self: *const Plan, name: []const u8) !usize {
        var buf: [40]u8 = undefined;
        for (self.specs(), 0..) |spec, i| {
            if (std.mem.eql(u8, try spec.name(&buf), name)) return i;
        }
        return error.Dsv41CacheTensorName;
    }
};

pub fn validateArray(spec: Spec, x: mlx.mlx_array) !void {
    if (x.ctx == null or mlx.mlx_array_dtype(x) != spec.dtype) return error.Dsv41CacheDtype;
    const sh = mlx.getShape(x);
    if (sh.len != spec.rank()) return error.Dsv41CacheShape;
    if (sh[0] != (std.math.cast(c_int, spec.rows) orelse return error.Dsv41CacheShape)) return error.Dsv41CacheShape;
    if (sh.len == 2 and sh[1] != (std.math.cast(c_int, spec.width) orelse return error.Dsv41CacheShape)) return error.Dsv41CacheShape;
    if ((mlx.mlx_array_size(x) * mlx.mlx_array_itemsize(x)) != try spec.bytes()) return error.Dsv41CacheShape;
}

pub fn validateEncodedArray(spec: Spec, enc: Encoding, x: mlx.mlx_array) !void {
    if (enc == .raw) return validateArray(spec, x);
    if (x.ctx == null or mlx.mlx_array_dtype(x) != .uint8) return error.Dsv41CacheDtype;
    const sh = mlx.getShape(x);
    const row = packedRowBytes(enc, spec.width) orelse return error.Dsv41CacheEncoding;
    if (sh.len != 2 or sh[0] != (std.math.cast(c_int, spec.rows) orelse return error.Dsv41CacheShape) or sh[1] != (std.math.cast(c_int, row) orelse return error.Dsv41CacheShape)) return error.Dsv41CacheShape;
}

pub fn unpackBf16(spec: Spec, packed_bytes: []const u8, out: []u16) !void {
    const enc = candidate(spec, true);
    if (enc == .raw) return error.Dsv41CacheEncoding;
    const row_bytes = packedRowBytes(enc, spec.width) orelse return error.Dsv41CacheGeometry;
    const expected = try std.math.mul(usize, spec.rows, row_bytes);
    const elements = try std.math.mul(usize, spec.rows, spec.width);
    if (packed_bytes.len != expected or out.len != elements) return error.Dsv41CacheShape;
    const group: usize = if (enc == .fp4_e4m3_g16) 16 else 32;
    const code_bytes = if (enc == .fp8_ue8m0_g32) spec.width else spec.width / 2;
    const scales_per_row = spec.width / group;
    for (0..spec.rows) |row| {
        const src = packed_bytes[row * row_bytes ..][0..row_bytes];
        const dst = out[row * spec.width ..][0..spec.width];
        for (0..scales_per_row) |g| {
            const scale_byte = src[code_bytes + g];
            if (scale_byte == 0) return error.Dsv41CacheCorrupt;
            const scale = if (enc == .fp4_e4m3_g16) e4m3Value(scale_byte) else pow2Scale(scale_byte);
            if (!std.math.isFinite(scale) or scale <= 0) return error.Dsv41CacheCorrupt;
            for (0..group) |j| {
                const i = g * group + j;
                if (enc == .fp8_ue8m0_g32) {
                    const code = src[i];
                    if ((code & 0x7f) == 0x7f) return error.Dsv41CacheCorrupt;
                    const value = e4m3Value(code) * scale;
                    dst[i] = bf16Bits(value);
                } else {
                    const code_byte = src[i / 2];
                    const code: u8 = if (i & 1 == 0) code_byte & 0x0f else code_byte >> 4;
                    const level: f32 = switch (code & 7) {
                        0 => 0,
                        1 => 0.5,
                        2 => 1,
                        3 => 1.5,
                        4 => 2,
                        5 => 3,
                        6 => 4,
                        else => 6,
                    };
                    const signed = if (code & 8 == 0) level else -level;
                    dst[i] = bf16Bits(signed * scale);
                }
            }
        }
    }
}

fn bf16Bits(value: f32) u16 {
    const bits: u32 = @bitCast(value);
    return @truncate((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
}

fn e4m3Value(bits: u8) f32 {
    const sign: f32 = if (bits & 0x80 == 0) 1 else -1;
    const exponent = (bits >> 3) & 0x0f;
    const mantissa = bits & 7;
    if (exponent == 0) return sign * @as(f32, @floatFromInt(mantissa)) * 0x1p-9;
    return sign * (1 + @as(f32, @floatFromInt(mantissa)) / 8) * std.math.pow(f32, 2, @as(f32, @floatFromInt(exponent)) - 7);
}

fn pow2Scale(value: u8) f32 {
    return @bitCast(@as(u32, value) << 23);
}

/// The ordinary quarter-growth rule, also applied to an imported exact used capacity for one more row.
pub fn rowCapacity(cap: usize, need: usize) !usize {
    const grown = std.math.add(usize, cap, cap / 4) catch return error.Dsv41CacheGeometry;
    const n = @max(need, @max(64, grown));
    try dimension(n);
    return n;
}

fn dimension(n: usize) !void {
    if (n == 0 or n > std.math.maxInt(c_int)) return error.Dsv41CacheGeometry;
}

fn itemBytes(dt: mlx.mlx_dtype) !u64 {
    return switch (dt) {
        .float16, .bfloat16 => 2,
        .float32, .uint32 => 4,
        else => error.Dsv41CacheGeometry,
    };
}

fn checkedAdd(a: u64, b: u64) !u64 {
    return std.math.add(u64, a, b) catch return error.Dsv41CacheGeometry;
}

fn checkedMul(a: u64, b: u64) !u64 {
    return std.math.mul(u64, a, b) catch return error.Dsv41CacheGeometry;
}

fn importPeak(loaded: u64, destination: u64, host: u64) !u64 {
    const scaled = try checkedAdd(try checkedMul(loaded, import_peak_loaded_pct), try checkedMul(destination, import_peak_destination_pct));
    return checkedAdd((try checkedAdd(scaled, 99)) / 100, host);
}

fn put(buf: []u8, off: usize, n: u64) void {
    std.mem.writeInt(u64, buf[off..][0..8], n, .little);
}

fn get(buf: []const u8, off: usize) u64 {
    return std.mem.readInt(u64, buf[off..][0..8], .little);
}

// ── tests ──

const testing = std.testing;

fn testLayout() Layout {
    var l: Layout = .{ .layers = 3, .stages = 2, .ring = 24, .width = 64, .index_width = 32, .hidden_width = 256, .dtype = .bfloat16, .kv_sources = 6, .index_sources = 2, .engram = true, .settings = 123 };
    l.ratios[1] = 2;
    l.ratios[2] = 1;
    return l;
}

test "codec planning rejects malformed headers, geometry, settings and overflowing boundaries without MLX operations" {
    const l = testLayout();
    const p = try Plan.init(l, 9, 42);
    var buf: [max_header]u8 = undefined;
    const bytes = try p.encode(&buf);
    const ops = mlx.op_count.load(.monotonic);
    const decoded = try Plan.decode(l, bytes);
    try testing.expectEqual(ops, mlx.op_count.load(.monotonic));
    try testing.expectEqual(p.peak, decoded.peak);
    for (0..bytes.len) |len| try testing.expectError(error.Dsv41CacheHeader, Plan.decode(l, bytes[0..len]));
    // The media key varies by boundary; the remaining assertions match the loaded instance.
    for ([_]usize{ 0, 8, 16, 32, 40, 48, 56, 64, 72, 80, 88, 96 }) |off| {
        buf[off] ^= 0x80;
        try testing.expectError(error.Dsv41CacheHeader, Plan.decode(l, bytes));
        buf[off] ^= 0x80;
    }
    var changed = l;
    changed.dtype = .float16;
    try testing.expectError(error.Dsv41CacheHeader, Plan.decode(changed, bytes));
    changed = l;
    changed.settings += 1;
    try testing.expectError(error.Dsv41CacheHeader, Plan.decode(changed, bytes));
    changed = l;
    changed.kv_sources = 2;
    try testing.expectError(error.Dsv41CacheHeader, Plan.decode(changed, bytes));
    try testing.expectError(error.Dsv41CacheGeometry, Plan.init(l, 0, 0));
    try testing.expectError(error.Dsv41CacheGeometry, Plan.init(l, std.math.maxInt(usize), 0));
    changed = l;
    changed.width = std.math.maxInt(usize);
    try testing.expectError(error.Dsv41CacheGeometry, Plan.init(changed, 9, 0));
}

test "codec peak retains destination accounting and host storage" {
    const p = try Plan.init(testLayout(), 9, 0);
    // Source buffers grow to 64 rows; rings, pending f32 rows, DSpark and history all coexist.
    const loaded: u64 = 3 * 24 * 64 * 2 + (4 * 64 + 9 * 64 + 4 * 32) * 2 + 2 * 64 * 4 + 2 * 24 * 64 * 2 + 256 * 2 + 9 * 4;
    const destination: u64 = loaded + ((64 - 4) * 64 + (64 - 9) * 64 + (64 - 4) * 32) * 2;
    try testing.expectEqual(loaded, p.loaded);
    try testing.expectEqual(destination, p.destination);
    const host: u64 = 3 * 9 * 4 + 2 * max_header + (12 + 3 + 2 + 1) * 4096;
    try testing.expectEqual(try importPeak(loaded, destination, host), p.peak);
    const compressed = try Plan.initWithPacking(testLayout(), 9, 0, true);
    try testing.expect(compressed.max_unpacked > 0);
    try testing.expectEqual(try checkedAdd(try importPeak(compressed.loaded, compressed.destination, host), compressed.max_unpacked), compressed.peak);
    try testing.expectEqual(@as(usize, 64), try rowCapacity(0, 1));
    try testing.expectEqual(@as(usize, 80), try rowCapacity(64, 65));
    try testing.expectEqual(@as(usize, 100), try rowCapacity(80, 81));
    try testing.expectError(error.Dsv41CacheGeometry, rowCapacity(std.math.maxInt(usize), 1));
    try testing.expectError(error.Dsv41CacheGeometry, rowCapacity(0, @as(usize, std.math.maxInt(c_int)) + 1));
}

test "cache import estimate covers measured SSD restore peaks within 1.15x and a packed entry's destination" {
    const cases = .{
        .{ .n = 32_791, .loaded = 111_444_060, .destination = 137_672_540, .measured = 321_896_520 },
        .{ .n = 131_095, .loaded = 426_410_076, .destination = 531_281_756, .measured = 1_267_482_912 },
        .{ .n = 262_167, .loaded = 846_364_764, .destination = 1_056_094_044, .measured = 2_524_856_896 },
    };
    inline for (cases) |c| {
        const host = c.n * 12 + 2 * max_header + 100 * 4096;
        const peak = try importPeak(c.loaded, c.destination, host);
        try testing.expect(peak >= c.measured);
        try testing.expect(peak <= c.measured + c.measured * 15 / 100 + host);
        // A packed entry loads ~894/3204 of the raw bytes into the same destination: never below loaded + destination.
        const packed_loaded = c.loaded * 894 / 3204;
        const packed_peak = try importPeak(packed_loaded, c.destination, host);
        try testing.expect(packed_peak >= packed_loaded + c.destination);
    }
}

test "packed cache decoder preserves signed zero and FP4 levels and rejects zero scales" {
    const comp = Spec{ .kind = .comp, .rows = 1, .width = 16, .dtype = .bfloat16 };
    const comp_bytes = [_]u8{ 0x08, 0x21, 0x43, 0x65, 0xf7, 0, 0, 0, 0x01 };
    const expected = [_]u16{ 0x8000, 0, 0x3a80, 0x3b00, 0x3b40, 0x3b80, 0x3bc0, 0x3c00, 0x3c40, 0xbc40, 0, 0, 0, 0, 0, 0 };
    var decoded: [expected.len]u16 = undefined;
    try unpackBf16(comp, &comp_bytes, &decoded);
    try testing.expectEqualSlices(u16, &expected, &decoded);
    var corrupt = comp_bytes;
    corrupt[8] = 0;
    try testing.expectError(error.Dsv41CacheCorrupt, unpackBf16(comp, &corrupt, &decoded));

    const index = Spec{ .kind = .index, .rows = 1, .width = 32, .dtype = .bfloat16 };
    var index_bytes: [17]u8 = @splat(0);
    index_bytes[0] = 0x03;
    index_bytes[1] = 0x38;
    index_bytes[16] = 0x7f;
    var index_out: [32]u16 = undefined;
    try unpackBf16(index, &index_bytes, &index_out);
    try testing.expectEqual(@as(u16, 0x3fc0), index_out[0]);
    try testing.expectEqual(@as(u16, 0), index_out[1]);
    try testing.expectEqual(@as(u16, 0x8000), index_out[2]);
    try testing.expectEqual(@as(u16, 0x3fc0), index_out[3]);
}

test "codec v2 plans static encodings while v1 keeps its byte layout and raw bill" {
    const l = testLayout();
    var v1 = try Plan.init(l, 9, 42);
    var header: [max_header]u8 = undefined;
    const old = try v1.encode(&header);
    try testing.expectEqualStrings("DV41A001", old[0..8]);
    try testing.expectEqual(fixed_bytes + v1.len * v1_spec_bytes, old.len);
    try testing.expectEqual(l.key(), get(old, 8));
    try testing.expectEqual(@as(u64, 9), get(old, 16));
    try testing.expectEqual(@as(u64, 42), get(old, 24));
    try testing.expectEqual(@as(u64, 9), get(old, 32));
    try testing.expectEqual(@as(u64, v1.len), get(old, 40));
    try testing.expectEqualSlices(u64, &.{ 0, 0, 24, 64, 2, @intCast(@backingInt(mlx.mlx_dtype.bfloat16)), 3072 }, &.{
        get(old, fixed_bytes),      get(old, fixed_bytes + 8),  get(old, fixed_bytes + 16), get(old, fixed_bytes + 24),
        get(old, fixed_bytes + 32), get(old, fixed_bytes + 40), get(old, fixed_bytes + 48),
    });
    const decoded_v1 = try Plan.decode(l, old);
    try testing.expectEqual(v1.loaded, decoded_v1.loaded);
    try testing.expectEqual(@as(Encoding, .raw), decoded_v1.encodings[0]);

    var v2 = try Plan.initWithPacking(l, 9, 42, true);
    v2.version = 2;
    var raw_bytes: u64 = 0;
    for (v2.specs()) |spec| raw_bytes += try spec.bytes();
    try v2.reprice();
    const packed_bytes = v2.loaded;
    try testing.expect(packed_bytes < raw_bytes);
    const current = try v2.encode(&header);
    try testing.expectEqualStrings("DV41A002", current[0..8]);
    try testing.expectEqual(fixed_bytes + v2.len * v2_spec_bytes, current.len);
    const decoded_v2 = try Plan.decode(l, current);
    try testing.expectEqual(packed_bytes, decoded_v2.loaded);
    for (v2.specs(), 0..) |spec, i| try testing.expectEqual(candidate(spec, true), decoded_v2.encodings[i]);

    var bad = header;
    put(&bad, fixed_bytes + 56, 1);
    try testing.expectError(error.Dsv41CacheEncoding, Plan.decode(l, bad[0..current.len]));
    var unknown: [fixed_bytes]u8 = @splat(0);
    @memcpy(unknown[0..8], "NOPE0000");
    try testing.expectError(error.Dsv41CacheHeaderVersion, Plan.decode(l, &unknown));
}

test "codec candidates use only kind, bf16, QAT and divisible row widths" {
    const cases = .{
        .{ .spec = Spec{ .kind = .comp, .rows = 2, .width = 512, .dtype = .bfloat16 }, .enc = Encoding.fp4_e4m3_g16, .row = 288 },
        .{ .spec = Spec{ .kind = .index, .rows = 2, .width = 128, .dtype = .bfloat16 }, .enc = Encoding.fp4_ue8m0_g32, .row = 68 },
        .{ .spec = Spec{ .kind = .win, .rows = 2, .width = 512, .dtype = .bfloat16 }, .enc = Encoding.fp8_ue8m0_g32, .row = 528 },
        .{ .spec = Spec{ .kind = .ds, .rows = 2, .width = 512, .dtype = .bfloat16 }, .enc = Encoding.fp8_ue8m0_g32, .row = 528 },
    };
    inline for (cases) |case| {
        try testing.expectEqual(case.enc, candidate(case.spec, true));
        try testing.expectEqual(@as(?usize, case.row), packedRowBytes(case.enc, case.spec.width));
        try testing.expectEqual(Encoding.raw, candidate(case.spec, false));
    }
    for ([_]Spec{
        .{ .kind = .comp, .rows = 1, .width = 510, .dtype = .bfloat16 },
        .{ .kind = .index, .rows = 1, .width = 130, .dtype = .bfloat16 },
        .{ .kind = .win, .rows = 1, .width = 510, .dtype = .bfloat16 },
        .{ .kind = .pend_kv, .rows = 1, .width = 512, .dtype = .bfloat16 },
        .{ .kind = .mh, .rows = 1, .width = 512, .dtype = .bfloat16 },
        .{ .kind = .hist, .rows = 1, .width = 1, .dtype = .uint32 },
    }) |spec| try testing.expectEqual(Encoding.raw, candidate(spec, true));
}
