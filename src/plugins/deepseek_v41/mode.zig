//! A serving mode: the whole model (`tp1`) or this box's rank of a tensor group (`tp`), each its own `Dsv41Model`
//! built by `init` at load and resident together. A switch moves no live slot state across them (`plugin.swapMode`).

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const dsv41 = @import("deepseek_v41.zig");
const Config = @import("config.zig").Config;
pub const Shard = @import("../../sdk/weights/shard.zig").Shard;
const Weights = @import("../../sdk/weights/resident.zig").Weights;
const rowcut = @import("rowcut.zig");

pub const Mode = enum(u8) { tp1, tp };

pub const Modes = struct {
    tp1: *dsv41.Dsv41Model,
    /// This box's rank of a tensor group, built beside `tp1`; null when the load named no group.
    tp: ?*dsv41.Dsv41Model = null,
    /// `tp`'s row cut of `tp1`'s layers, which its MoE runs at decode widths.
    row_cut: ?*rowcut.RowCut = null,
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
        if (self.row_cut) |rc| rc.destroy();
        self.tp1.deinit();
    }
};

/// Build the whole model once, then its tensor mode from the resident, transformed arrays.
pub fn build(gpa: std.mem.Allocator, whole: *const Config, w: *Weights, s: mlx.mlx_stream, opts: dsv41.InitOpts, group: ?Shard) !Modes {
    if (group) |g| {
        if (opts.tp != null or opts.pair != null) return error.Dsv41ModeBesideSplit;
        if (opts.streamed_experts) return error.Dsv41TpStreamedExperts;
        if (opts.quant != null) return error.Dsv41TpOutsideQuant;
        if (!dsv41.tpGeometryFits(whole, g.size)) return error.Dsv41ShardUneven;
    }
    var out: Modes = .{ .tp1 = try dsv41.init(gpa, whole, w, s, opts) };
    errdefer out.tp1.deinit();
    if (group) |g| {
        var o = opts;
        o.draft_shard = dsv41.draftShardWanted(g.size);
        o.tp = .{ .rank = g.rank, .size = g.size, .stages = o.draft_shard };
        const hi = if (o.draft_shard) out.tp1.layers.len else out.tp1.n_layers;
        out.row_cut = try rowcut.RowCut.create(gpa, s, out.tp1, 0, hi, g.rank, g.size);
        errdefer out.row_cut.?.destroy();
        o.resident = .{ .model = out.tp1, .rows = out.row_cut.? };
        out.tp = try dsv41.init(gpa, whole, w, s, o);
        if (g.rank == 0) out.tp.?.prefill_whole = out.tp1;
    }
    return out;
}

test "mode.build: a tensor group's mode beside a --dist split is refused before it reads a weight" {
    var w = Weights.init(std.testing.allocator);
    defer w.deinit();
    const cfg: Config = undefined;
    try std.testing.expectError(error.Dsv41ModeBesideSplit, build(std.testing.allocator, &cfg, &w, .{ .ctx = null }, .{ .tp = .{ .rank = 0, .size = 2 } }, .{ .rank = 0, .size = 4 }));
}

test "mode.build: a group size the release geometry's heads or output groups cannot split is refused by name before it reads a weight" {
    var w = Weights.init(std.testing.allocator);
    defer w.deinit();
    const cfg: Config = .{ .model_type = "deepseek_v41", .num_attention_heads = 64, .dsv4_o_groups = 8, .moe_intermediate_size = 2048, .shared_expert_intermediate_size = 2048 };
    try std.testing.expect(dsv41.tpGeometryFits(&cfg, 4));
    for ([_]u32{ 16, 3, 128 }) |size|
        try std.testing.expectError(error.Dsv41ShardUneven, build(std.testing.allocator, &cfg, &w, .{ .ctx = null }, .{}, .{ .rank = 0, .size = size }));
}

test "mode.build: streamed experts are refused before building either mode" {
    var w = Weights.init(std.testing.allocator);
    defer w.deinit();
    const cfg: Config = .{ .model_type = "deepseek_v41", .num_attention_heads = 64, .dsv4_o_groups = 8, .moe_intermediate_size = 2048, .shared_expert_intermediate_size = 2048 };
    try std.testing.expectError(error.Dsv41TpStreamedExperts, build(std.testing.allocator, &cfg, &w, .{ .ctx = null }, .{ .streamed_experts = true }, .{ .rank = 0, .size = 4 }));
}

/// A tensor group of `size` ranks in one process, for a rank-0 model's tests: every rank's part is rank 0's, so a sum
/// is the part and a gather is `size` copies of it on axis 0 (distributed.zig's order).
pub const LoneGroup = struct {
    pub var sums: usize = 0;
    var size: usize = 4;

    fn sum(x: mlx.mlx_array) anyerror!mlx.mlx_array {
        sums += 1;
        var o = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_array_set(&o, x));
        return o;
    }
    fn gather(x: mlx.mlx_array) anyerror!mlx.mlx_array {
        var xs: [8]mlx.mlx_array = @splat(x);
        const vec = mlx.mlx_vector_array_new_data(&xs, size);
        defer _ = mlx.mlx_vector_array_free(vec);
        var o = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_concatenate_axis(&o, vec, 0, mlx.gpuStream()));
        return o;
    }
    fn step(_: []const u32, _: usize, _: bool, _: bool, _: dsv41.VerifyOpts, _: u64, _: ?mlx.mlx_array) anyerror!void {}
    fn batch(_: []const u64, _: []const usize, _: []const []const u32, _: bool) anyerror!void {}
    fn draft(_: []const u64, _: []const usize, _: []const u32) anyerror!void {}
    fn load(_: u64, _: []const u32, _: []const mlx.mlx_array) anyerror!void {}
    fn prefill(_: u64, _: usize, _: usize, _: bool, _: []const mlx.mlx_array, _: usize) anyerror!void {}
    fn free(_: u64) void {}
    fn snap(_: dsv41.SnapOp, _: u64, _: u64) void {}

    pub fn arm(n: usize) void {
        size = n;
        sums = 0;
        dsv41.clusterHooks().* = .{ .tp_sum = &sum, .tp_gather = &gather, .tp_step = &step, .tp_batch = &batch, .tp_draft = &draft, .tp_load = &load, .tp_prefill = &prefill, .tp_free = &free, .tp_snap = &snap };
    }
    pub fn disarm() void {
        dsv41.clusterHooks().* = .{};
    }
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

    LoneGroup.arm(2);
    defer LoneGroup.disarm();
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
