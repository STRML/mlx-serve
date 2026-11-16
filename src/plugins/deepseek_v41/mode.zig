//! A serving mode: the whole model (`tp1`) or this box's rank of a tensor group (`tp`), each its own `Dsv41Model`
//! built by `init` at load and resident together. A switch moves no live slot state across them (`plugin.swapMode`).

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const dsv41 = @import("deepseek_v41.zig");
const Config = @import("config.zig").Config;
pub const Shard = @import("../../sdk/weights/shard.zig").Shard;
const Weights = @import("../../sdk/weights/resident.zig").Weights;

pub const Mode = enum(u8) { tp1, tp };

pub const Modes = struct {
    tp1: *dsv41.Dsv41Model,
    /// This box's rank of a tensor group, built beside `tp1`; null when the load named no group.
    tp: ?*dsv41.Dsv41Model = null,
    /// `tp`'s weights: views of this rank's part of the pack's tensors. The model reads its linears from here.
    tp_weights: ?Weights = null,
    active: Mode = .tp1,

    pub fn model(self: *const Modes, mode: Mode) ?*dsv41.Dsv41Model {
        return switch (mode) {
            .tp1 => self.tp1,
            .tp => self.tp,
        };
    }

    pub fn current(self: *const Modes) *dsv41.Dsv41Model {
        return self.model(self.active).?;
    }

    pub fn deinit(self: *Modes) void {
        if (self.tp) |t| t.deinit();
        if (self.tp_weights) |*w| w.deinit();
        self.tp1.deinit();
    }
};

/// Both modes from one weights map: `group`'s model first, over views of `w` (`shardViews`), then the whole model,
/// which consumes `w` as today's load does. `opts` is the whole model's; null `group` builds `tp1` alone.
pub fn build(gpa: std.mem.Allocator, whole: *const Config, w: *Weights, s: mlx.mlx_stream, opts: dsv41.InitOpts, group: ?Shard) !Modes {
    var out: Modes = .{ .tp1 = undefined };
    if (group) |g| {
        if (opts.tp != null or opts.pair != null) return error.Dsv41ModeBesideSplit;
        out.tp_weights = try shardViews(gpa, w, g, s);
        errdefer out.tp_weights.?.deinit();
        var o = opts;
        o.tp = g;
        out.tp = try dsv41.init(gpa, whole, &out.tp_weights.?, s, o);
    }
    errdefer if (out.tp) |t| {
        t.deinit();
        out.tp_weights.?.deinit();
    };
    out.tp1 = try dsv41.init(gpa, whole, w, s, opts);
    return out;
}

/// A map of `w`'s tensors as rank `sh` of a tensor group reads them: each one `Shard.axis` cuts as a slice of the
/// resident array, every other one the same array. Nothing is copied here; the arrays stay `w`'s.
pub fn shardViews(gpa: std.mem.Allocator, w: *const Weights, sh: Shard, s: mlx.mlx_stream) !Weights {
    var out = Weights.init(gpa);
    errdefer out.deinit();
    var it = w.map.iterator();
    while (it.next()) |kv| {
        const v = try viewOf(kv.value_ptr.*, Shard.axis(kv.key_ptr.*), sh, s);
        out.put(kv.key_ptr.*, v) catch |e| {
            _ = mlx.mlx_array_free(v);
            return e;
        };
    }
    return out;
}

fn viewOf(x: mlx.mlx_array, axis: ?Shard.Axis, sh: Shard, s: mlx.mlx_stream) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(o);
    const a = axis orelse {
        try mlx.check(mlx.mlx_array_set(&o, x));
        return o;
    };
    const shape = mlx.getShape(x);
    const ax = a.of(shape.len);
    const part = std.math.divExact(c_int, shape[ax], @intCast(sh.size)) catch return error.Dsv41ShardUneven;
    var start: [4]c_int = @splat(0);
    var stop: [4]c_int = undefined;
    const strides: [4]c_int = @splat(1);
    @memcpy(stop[0..shape.len], shape);
    start[ax] = part * @as(c_int, @intCast(sh.rank));
    stop[ax] = start[ax] + part;
    try mlx.check(mlx.mlx_slice(&o, x, &start, shape.len, &stop, shape.len, &strides, shape.len, s));
    return o;
}

test "mode.build: a tensor group's mode beside a --dist split is refused before it reads a weight" {
    var w = Weights.init(std.testing.allocator);
    defer w.deinit();
    const cfg: Config = undefined;
    try std.testing.expectError(error.Dsv41ModeBesideSplit, build(std.testing.allocator, &cfg, &w, .{ .ctx = null }, .{ .tp = .{ .rank = 0, .size = 2 } }, .{ .rank = 0, .size = 4 }));
}

/// A rank-0 model's tensor group with no peers: every rank's part is the sum.
const LoneGroup = struct {
    fn sum(x: mlx.mlx_array) anyerror!mlx.mlx_array {
        var o = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_array_set(&o, x));
        return o;
    }
    fn step(_: []const u32, _: usize, _: bool, _: bool, _: bool, _: u64, _: ?mlx.mlx_array) anyerror!void {}
    fn batch(_: []const u64, _: []const usize, _: []const []const u32, _: bool) anyerror!void {}
    fn free(_: u64) void {}
    fn snap(_: dsv41.SnapOp, _: u64, _: u64) void {}
};

/// A prompt and three decode steps on the whole model: each step's logits (host f32, caller frees).
fn wholeTurn(m: *dsv41.Dsv41Model, prompt: []const u32) ![4][]f32 {
    const gpa = std.testing.allocator;
    var out: [4][]f32 = undefined;
    const p = try dsv41.prefill(m, gpa, prompt, null);
    defer _ = mlx.mlx_array_free(p);
    out[0] = try dsv41.hostF32(p);
    for ([_]u32{ 5, 9, 13 }, out[1..]) |tok, *o| {
        const l = try dsv41.extendResumable(m, gpa, &.{tok}, null);
        defer _ = mlx.mlx_array_free(l);
        o.* = try dsv41.hostF32(l);
    }
    return out;
}

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;

test "mode: with Engram layers the whole model beside a tensor group's rank gives today's logits bit for bit (DSV41_TINY)" {
    const root = @import("../../env.zig").span(.DSV41_TINY) orelse return error.SkipZigTest;
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    _ = setenv("MLX_ENABLE_TF32", "0", 1);
    const gpa = std.testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const dir = try std.fmt.allocPrint(gpa, "{s}/pipe", .{root});
    defer gpa.free(dir);
    const cfg = try @import("../../sdk/testing/pack_config.zig").packConfig(@import("config.zig"), io, dir);
    const opts: dsv41.InitOpts = .{ .act = .float32, .pack_dir = dir };
    const prompt = [_]u32{ 3, 17, 42, 9, 7, 25, 1, 8, 2, 6, 4 };

    var want: [4][]f32 = undefined;
    {
        var w = try dsv41.loadPack(io, gpa, dir, false);
        defer w.deinit();
        const m = try dsv41.init(gpa, &cfg, &w, mlx.gpuStream(), opts);
        defer m.deinit();
        if (m.engrams.len == 0) return error.SkipZigTest;
        want = try wholeTurn(m, &prompt);
    }
    defer for (want) |x| gpa.free(x);

    const h = dsv41.clusterHooks();
    h.* = .{ .tp_sum = &LoneGroup.sum, .tp_step = &LoneGroup.step, .tp_batch = &LoneGroup.batch, .tp_free = &LoneGroup.free, .tp_snap = &LoneGroup.snap };
    defer h.* = .{};
    var w = try dsv41.loadPack(io, gpa, dir, false);
    defer w.deinit();
    var modes = build(gpa, &cfg, &w, mlx.gpuStream(), opts, .{ .rank = 0, .size = 2 }) catch |e| switch (e) {
        error.Dsv41ShardUneven => return error.SkipZigTest,
        else => return e,
    };
    defer modes.deinit();
    try std.testing.expect(modes.tp.?.engrams.len > 0);
    _ = mlx.mlx_array_free(try dsv41.prefill(modes.tp.?, gpa, &prompt, null));
    const got = try wholeTurn(modes.tp1, &prompt);
    defer for (got) |x| gpa.free(x);
    for (want, got) |a, b| try std.testing.expectEqualSlices(f32, a, b);
}
