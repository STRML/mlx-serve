//! qwen4_exp n-gram PLE on the GPU: the whole `ngram_table.bin` mapping wrapped as ONE no-copy
//! Metal buffer, and one kernel that hashes the token ids, gathers the rows and rounds them to
//! bf16 exactly like `NgramHash.rowIds` + `NgramTable.gather` + `bf16Rne`. The kernel reads the
//! ids on the GPU, so a forward built on lazy ids needs no host read before it runs.

const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");
const qwen4 = @import("qwen4_exp.zig");

/// Why the loader picked its arm. Everything but `.gpu` serves the host gather.
pub const Arm = enum { gpu, env_off, bits, misaligned, too_large, low_memory };

pub const Budget = struct {
    page: usize,
    max_buffer: u64,
    working_set: u64,
    model_bytes: u64,
};

/// Room the working set keeps past weights + table for KV, transients and the buffer pool.
pub const HEADROOM: u64 = 16 << 30;

/// Kernel dispatches since start; the engagement counter the tests read.
pub var dispatches: u64 = 0;
/// Mappings MLX has unmapped after dropping their buffer.
pub var unmaps: u64 = 0;

pub fn chooseArm(env: ?[]const u8, bits: u32, base: usize, len: usize, b: Budget) Arm {
    _ = env;
    _ = bits;
    _ = base;
    _ = len;
    _ = b;
    return .gpu;
}

pub const Table = struct {
    arr: mlx.mlx_array,

    pub fn release(self: Table) void {
        _ = mlx.mlx_array_free(self.arr);
    }
};

pub fn wrap(map: []const u8) !Table {
    _ = map;
    return error.Unimplemented;
}

/// Load-time arm choice for `table`; logs one line naming the arm. `env` is `MLX_SERVE_PLE_GPU`,
/// `model_bytes` the weights already resident. Null = the host gather.
pub fn load(table: *qwen4.NgramTable, env: ?[]const u8, model_bytes: u64) ?Table {
    _ = table;
    _ = env;
    _ = model_bytes;
    return null;
}

/// bf16 `[S, n_heads * dim]` for the `S` ids of `ids` (any integer dtype, may be lazy),
/// hashed against `prev` (the `ngram_size - 1` tokens before them). GPU stream only.
pub fn embed(s: mlx.mlx_stream, tbl: Table, h: *const qwen4.NgramHash, t: *const qwen4.NgramTable, ids: mlx.mlx_array, prev: []const u32) !mlx.mlx_array {
    _ = s;
    _ = tbl;
    _ = h;
    _ = t;
    _ = ids;
    _ = prev;
    return error.Unimplemented;
}

// ── tests ──

const testing = std.testing;

/// A synthetic table file for tests; `deinit` closes the table and removes the file.
pub const Fixture = struct {
    td: std.testing.TmpDir,
    table: qwen4.NgramTable,

    pub fn deinit(self: *Fixture) void {
        self.table.close();
        self.td.cleanup();
    }
};

fn randBf16(r: std.Random) u16 {
    // Normal values only (exponent 2^-17 .. 2^13): real tables carry no NaN, Inf or subnormal.
    const sign: u16 = @as(u16, r.int(u1)) << 15;
    const exp: u16 = r.intRangeAtMost(u16, 110, 140);
    return sign | (exp << 7) | r.int(u7);
}

/// A `ngram_table.bin` with random packed words, scales and biases. The header length is odd so
/// every region starts off a 4-byte boundary, as nothing in the format promises alignment.
pub fn writeFixture(bits: u32, rows: u64, dim: u32, gs: u32, seed: u64) !Fixture {
    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();
    const a = testing.allocator;
    const hlen: usize = 509;
    const raw = bits == 16;
    const wbytes: u64 = if (raw) rows * dim * 2 else rows * (dim * bits / 32) * 4;
    const sbytes: u64 = if (raw) 0 else rows * (dim / gs) * 2;
    var hbuf: [hlen]u8 = @splat(' ');
    if (raw)
        _ = try std.fmt.bufPrint(&hbuf, "{{\"__metadata__\":{{\"format\":\"mlx-serve-ngram\",\"bits\":\"16\",\"group_size\":\"{d}\"}},\"weight\":{{\"dtype\":\"BF16\",\"shape\":[{d},{d}],\"data_offsets\":[0,{d}]}}}}", .{ gs, rows, dim, wbytes })
    else
        _ = try std.fmt.bufPrint(&hbuf, "{{\"__metadata__\":{{\"format\":\"mlx-serve-ngram\",\"bits\":\"{d}\",\"group_size\":\"{d}\"}},\"weight\":{{\"dtype\":\"U32\",\"shape\":[{d},{d}],\"data_offsets\":[0,{d}]}},\"scales\":{{\"dtype\":\"BF16\",\"shape\":[{d},{d}],\"data_offsets\":[{d},{d}]}},\"biases\":{{\"dtype\":\"BF16\",\"shape\":[{d},{d}],\"data_offsets\":[{d},{d}]}}}}", .{ bits, gs, rows, dim * bits / 32, wbytes, rows, dim / gs, wbytes, wbytes + sbytes, rows, dim / gs, wbytes + sbytes, wbytes + 2 * sbytes });
    const total: usize = @intCast(8 + hlen + wbytes + 2 * sbytes);
    const buf = try a.alloc(u8, total);
    defer a.free(buf);
    std.mem.writeInt(u64, buf[0..8], hlen, .little);
    @memcpy(buf[8 .. 8 + hlen], &hbuf);
    const data = buf[8 + hlen ..];
    if (raw) {
        var i: usize = 0;
        while (i < wbytes) : (i += 2) std.mem.writeInt(u16, data[i..][0..2], randBf16(r), .little);
    } else {
        r.bytes(data[0..@intCast(wbytes)]);
        var i: usize = @intCast(wbytes);
        while (i < data.len) : (i += 2) std.mem.writeInt(u16, data[i..][0..2], randBf16(r), .little);
    }
    var td = std.testing.tmpDir(.{});
    errdefer td.cleanup();
    const io = std.Io.Threaded.global_single_threaded.io();
    try td.dir.writeFile(io, .{ .sub_path = "ngram_table.bin", .data = buf });
    var pbuf: [std.fs.max_path_bytes]u8 = undefined;
    const root_len = try td.dir.realPath(io, &pbuf);
    var full: [std.fs.max_path_bytes]u8 = undefined;
    const path = try std.fmt.bufPrint(&full, "{s}/ngram_table.bin", .{pbuf[0..root_len]});
    qwen4.warm_override = false;
    defer qwen4.warm_override = null;
    return .{ .td = td, .table = try qwen4.NgramTable.open(path) };
}

fn cpuRows(h: *const qwen4.NgramHash, t: *const qwen4.NgramTable, prev: []const u32, ids: []const u32) ![]u16 {
    const a = testing.allocator;
    const rows = try a.alloc(i64, ids.len * h.n_heads);
    defer a.free(rows);
    h.rowIds(prev, ids, rows);
    const host = try a.alloc(f32, rows.len * t.dim);
    defer a.free(host);
    t.gather(rows, host, 0);
    const out = try a.alloc(u16, host.len);
    for (host, out) |v, *o| o.* = qwen4.bf16Rne(v);
    return out;
}

fn gpuRows(tbl: Table, h: *const qwen4.NgramHash, t: *const qwen4.NgramTable, prev: []const u32, ids: []const u32) ![]u16 {
    const a = testing.allocator;
    const ids_i = try a.alloc(i32, ids.len);
    defer a.free(ids_i);
    for (ids, ids_i) |v, *o| o.* = @intCast(v);
    const shape = [_]c_int{@intCast(ids.len)};
    const arr = mlx.mlx_array_new_data(ids_i.ptr, &shape, 1, .int32);
    defer _ = mlx.mlx_array_free(arr);
    const out = try embed(mlx.gpuStream(), tbl, h, t, arr, prev);
    defer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_array_eval(out));
    const want = ids.len * h.n_heads * t.dim;
    try testing.expectEqual(want, mlx.mlx_array_size(out));
    const d = mlx.mlx_array_data_bfloat16(out) orelse return error.MlxArrayDataNull;
    return a.dupe(u16, d[0..want]);
}

/// Both arms over one table; the GPU bits must equal the CPU bits.
fn expectArmsEqual(tbl: Table, h: *const qwen4.NgramHash, t: *const qwen4.NgramTable, prev: []const u32, ids: []const u32) !void {
    const want = try cpuRows(h, t, prev, ids);
    defer testing.allocator.free(want);
    const got = try gpuRows(tbl, h, t, prev, ids);
    defer testing.allocator.free(got);
    try testing.expectEqualSlices(u16, want, got);
}

const TEST_VOCAB: u32 = 1000;
const TEST_EOS: u32 = 999;

fn testHash(ngram: u32, heads: u32) !qwen4.NgramHash {
    return qwen4.NgramHash.init(TEST_VOCAB, ngram, heads, 500, 1, 1234, 0, TEST_EOS);
}

fn randIds(r: std.Random, out: []u32) void {
    for (out) |*v| v.* = r.uintLessThan(u32, TEST_VOCAB);
}

test "ple gpu: 10k random ids embed bit-identical to the CPU gather on every head" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const h = try testHash(3, 8);
    var fx = try writeFixture(4, h.total_rows, 64, 32, 1);
    defer fx.deinit();
    const tbl = try wrap(fx.table.map);
    defer tbl.release();
    var prng = std.Random.DefaultPrng.init(7);
    const ids = try testing.allocator.alloc(u32, 10_000);
    defer testing.allocator.free(ids);
    randIds(prng.random(), ids);
    const prev = [_]u32{ 17, 404 };
    try expectArmsEqual(tbl, &h, &fx.table, &prev, ids);
}

test "ple gpu: an eos at every chunk position and inside prev hashes like rowIds" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const h = try testHash(4, 4);
    var fx = try writeFixture(4, h.total_rows, 64, 32, 2);
    defer fx.deinit();
    const tbl = try wrap(fx.table.map);
    defer tbl.release();
    var prng = std.Random.DefaultPrng.init(8);
    const r = prng.random();
    var ids: [8]u32 = undefined;
    var prev: [3]u32 = undefined;
    for (0..prev.len + 1) |pe| {
        randIds(r, &prev);
        if (pe < prev.len) prev[pe] = TEST_EOS;
        for (0..ids.len) |p| {
            randIds(r, &ids);
            ids[p] = TEST_EOS;
            try expectArmsEqual(tbl, &h, &fx.table, &prev, &ids);
        }
    }
    // Back-to-back eos and a fresh (all-eos) history.
    const fresh = [_]u32{ TEST_EOS, TEST_EOS, TEST_EOS };
    const run = [_]u32{ 5, TEST_EOS, TEST_EOS, 7, 8, TEST_EOS, 9, 10 };
    try expectArmsEqual(tbl, &h, &fx.table, &fresh, &run);
}

test "ple gpu: every shipped width dequantizes like the CPU gather" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const h = try testHash(3, 8);
    var prng = std.Random.DefaultPrng.init(9);
    var ids: [512]u32 = undefined;
    randIds(prng.random(), &ids);
    const prev = [_]u32{ 3, 4 };
    for ([_]u32{ 2, 3, 4, 5, 6, 8, 16 }) |bits| {
        var fx = try writeFixture(bits, h.total_rows, 64, 32, 100 + bits);
        defer fx.deinit();
        try testing.expectEqual(bits, fx.table.bits);
        const tbl = try wrap(fx.table.map);
        defer tbl.release();
        try expectArmsEqual(tbl, &h, &fx.table, &prev, &ids);
    }
}

test "ple gpu: an 8192-token chunk that reaches the last table row matches the CPU gather" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const h = try testHash(3, 8);
    var fx = try writeFixture(4, h.total_rows, 64, 32, 3);
    defer fx.deinit();
    const tbl = try wrap(fx.table.map);
    defer tbl.release();
    var prng = std.Random.DefaultPrng.init(10);
    const ids = try testing.allocator.alloc(u32, 8192);
    defer testing.allocator.free(ids);
    randIds(prng.random(), ids);
    // Pick the last token so its widest head lands on the table's final row.
    const last_row: i64 = @intCast(h.total_rows - 1);
    var row: [16]i64 = undefined;
    var x: u32 = 0;
    while (true) : (x += 1) {
        h.rowIds(ids[8189..8191], &[_]u32{x}, &row);
        if (row[15] == last_row) break;
    }
    ids[8191] = x;
    try expectArmsEqual(tbl, &h, &fx.table, &[_]u32{ 1, 2 }, ids);
}

test "ple gpu arm gate: env off, a width the kernel lacks, a misaligned base, an over-long buffer, a tight working set" {
    const GB: u64 = 1 << 30;
    const b: Budget = .{ .page = 16384, .max_buffer = 64 * GB, .working_set = 200 * GB, .model_bytes = 70 * GB };
    const len: usize = 32 * GB;
    try testing.expectEqual(Arm.gpu, chooseArm(null, 4, 16384 * 7, len, b));
    try testing.expectEqual(Arm.gpu, chooseArm("1", 16, 16384 * 7, len, b));
    try testing.expectEqual(Arm.env_off, chooseArm("0", 4, 16384 * 7, len, b));
    try testing.expectEqual(Arm.bits, chooseArm(null, 7, 16384 * 7, len, b));
    try testing.expectEqual(Arm.misaligned, chooseArm(null, 4, 16384 * 7 + 4096, len, b));
    try testing.expectEqual(Arm.too_large, chooseArm(null, 4, 16384 * 7, 65 * GB, b));
    // The length rounds up to the page before the maxBufferLength check.
    try testing.expectEqual(Arm.too_large, chooseArm(null, 4, 16384 * 7, 64 * GB - 1, .{ .page = 16384, .max_buffer = 64 * GB - 1, .working_set = 200 * GB, .model_bytes = 0 }));
    var tight = b;
    tight.working_set = 70 * GB + 32 * GB + HEADROOM - 1;
    try testing.expectEqual(Arm.low_memory, chooseArm(null, 4, 16384 * 7, len, tight));
    tight.working_set += 1;
    try testing.expectEqual(Arm.gpu, chooseArm(null, 4, 16384 * 7, len, tight));
    tight.working_set = 0; // an unknown working set never pins 32 GB
    try testing.expectEqual(Arm.low_memory, chooseArm(null, 4, 16384 * 7, len, tight));
}

test "ple gpu wrap: the buffer IS the mapping, and a misaligned base never reaches MLX" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const h = try testHash(3, 8);
    var fx = try writeFixture(4, h.total_rows, 64, 32, 4);
    defer fx.deinit();
    const tbl = try wrap(fx.table.map);
    defer tbl.release();
    const d = mlx.mlx_array_data_uint8(tbl.arr) orelse return error.MlxArrayDataNull;
    try testing.expectEqual(@intFromPtr(fx.table.map.ptr), @intFromPtr(d));
    try testing.expect(mlx.mlx_array_size(tbl.arr) >= fx.table.map.len);
    try testing.expectError(error.PleMapMisaligned, wrap(fx.table.map[16..]));
}

test "ple gpu wrap: the mapping outlives the table until MLX drops its last reference" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const h = try testHash(3, 8);
    var fx = try writeFixture(4, h.total_rows, 64, 32, 5);
    const tbl = try wrap(fx.table.map);
    fx.table.gpu_owns_map = true;
    var extra = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_array_set(&extra, tbl.arr));
    const before = unmaps;
    fx.deinit(); // closes the table: fd and pool go, the mapping stays
    tbl.release();
    try testing.expectEqual(before, unmaps);
    _ = mlx.mlx_array_free(extra);
    try testing.expectEqual(before + 1, unmaps);
}
