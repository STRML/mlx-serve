//! The fused decode segments' switch (`DSV41_FUSED_SEG`): which level runs, which layers and shapes a segment kernel
//! serves, and the dispatch counter the tests read.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const env = @import("../../../env.zig");
const log = @import("../../../sdk/util/log.zig");
const config = @import("../config.zig");
const dsv41 = @import("../deepseek_v41.zig");

const A = mlx.mlx_array;
const Dsv41Model = dsv41.Dsv41Model;

pub const Level = enum(u8) { off = 0, ffn_entry = 1, ffn_seg = 2, attn_seg = 3, sum_overlap = 4, lossy = 5 };

pub const Which = enum { attn, ffn };

pub const State = struct {};

pub var dispatches: u64 = 0;

pub var broken = false;

pub fn levelFromEnv() Level {
    return levelOf(env.span(.DSV41_FUSED_SEG));
}

pub fn levelOf(raw: ?[]const u8) Level {
    const v = raw orelse return .off;
    const n = std.fmt.parseInt(u8, v, 10) catch return refused(v);
    return std.enums.fromInt(Level, n) orelse refused(v);
}

fn refused(v: []const u8) Level {
    log.warn("[dsv41] DSV41_FUSED_SEG={s} is not 0-5: fused segments off\n", .{v});
    return .off;
}

const tp_sizes = blk: {
    var reasons: [17][]const u8 = undefined;
    for (&reasons, 0..) |*r, n| r.* = std.fmt.comptimePrint("tp size {d}", .{n});
    break :blk reasons;
};

pub fn serves(m: *const Dsv41Model, rows: usize) ?[]const u8 {
    const t = m.tp orelse return "no tensor split";
    if (t.size != 4) return if (t.size < tp_sizes.len) tp_sizes[t.size] else "tp size over 16";
    if (m.dec.moe_fp4 == null) return "fused decode kernels off";
    if (m.attn_gather == null) return "no NAX";
    if (m.streamed != null or m.streamed_experts) return "streamed experts";
    if (rows != 1) return if (rows > 1) "rows > 1" else "no rows";
    if (m.act != .bfloat16) return "activations not bf16";
    if (!expertsMxfp4(m)) return "affine experts";
    return null;
}

fn expertsMxfp4(m: *const Dsv41Model) bool {
    for (m.layers[0..m.n_layers], 0..) |*ly, li| {
        if (m.pair) |p| if (!p.holds(li)) continue;
        const gu = ly.experts.gu orelse return false;
        if (gu.qp.mode != .mxfp4 or gu.b.ctx != null) return false;
    }
    return true;
}

pub fn covers(m: *const Dsv41Model, li: usize, which: Which) bool {
    return coversIn(&m.cfg, li, which);
}

pub fn coversIn(c: *const config.Config, li: usize, which: Which) bool {
    if (li >= c.num_hidden_layers) return false;
    if (which == .ffn) return true;
    const ratio = c.dsv4_compress_ratios[li];
    if (ratio != 1 and ratio != 2) return false;
    if (((c.dsv41_kv_sources | c.dsv41_index_sources) >> @intCast(li)) & 1 != 0) return false;
    return std.mem.indexOfScalar(u8, c.dsv41_engram_layers[0..c.dsv41_n_engram_layers], @intCast(li)) == null;
}

pub fn on(m: *const Dsv41Model, li: usize, which: Which, rows: usize, at: Level) bool {
    return @intFromEnum(m.fused_seg) >= @intFromEnum(at) and serves(m, rows) == null and covers(m, li, which);
}

pub fn announce(m: *const Dsv41Model) void {
    if (m.fused_seg == .off) return;
    if (serves(m, 1)) |why| log.info("[dsv41] fused seg off: {s}\n", .{why});
}

pub fn zeroShared(sc: *dsv41.Scope, act: A) !A {
    const shape = mlx.getShape(act);
    const routed = try sc.slice(act, 1, 0, @intCast(shape[1] - 1));
    const none = try sc.zeros(&.{ shape[0], 1, shape[2] }, mlx.mlx_array_dtype(act));
    return sc.concat(&.{ routed, none }, 1);
}
