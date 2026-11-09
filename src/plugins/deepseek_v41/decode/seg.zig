//! The fused decode segments' switch (`DSV41_FUSED_SEG`): which level runs, which layers and shapes a segment kernel
//! serves, and the dispatch counter the tests read.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const env = @import("../../../env.zig");
const log = @import("../../../sdk/util/log.zig");
const config = @import("../config.zig");
const dsv41 = @import("../deepseek_v41.zig");
const seg_proto = @import("seg_proto.zig");

const A = mlx.mlx_array;
const Dsv41Model = dsv41.Dsv41Model;

pub const Level = enum(u8) { off = 0, ffn_entry = 1, ffn_seg = 2, attn_seg = 3, sum_overlap = 4, lossy = 5 };

pub const Which = enum { attn, ffn };

pub const Protocol = enum { ticket, spin };

pub const Status = enum(u32) { ok = 0, spin_timeout = 1, sum_timeout = 2, injected_abort = 3 };

/// The protocol a fresh `State` starts with.
pub const default_protocol: Protocol = .ticket;

pub const slot_words = 16;

/// MLX binds a small input as constant memory, which a kernel cannot write: the status word keeps company.
const status_words = slot_words;

pub const max_gen: u32 = 0xffff;

const max_slots = 256;

/// The counter words every kernel of a step arrives on: `slot_words` a (layer, segment) slot, each word a generation stamp
/// over an arrival count, so a step never zeroes what the previous step left.
pub const Counters = struct {
    buf: A,
    gen: u32,

    pub fn init(n_slots: usize) !Counters {
        if (n_slots == 0 or n_slots > max_slots) return error.Dsv41FusedSegSlots;
        const zeros: [max_slots * slot_words]u32 = @splat(0);
        const buf = mlx.mlx_array_new_data(&zeros, &[_]c_int{@intCast(n_slots * slot_words)}, 1, .uint32);
        if (buf.ctx == null) return error.MlxArrayNew;
        return .{ .buf = buf, .gen = 0 };
    }

    /// The step's generation as a 1-element u32 input; the counters are cleared when the generation would wrap.
    pub fn next(self: *Counters, sc: *dsv41.Scope) !A {
        if (self.gen >= max_gen) try self.reset();
        self.gen += 1;
        const g = [1]u32{self.gen};
        return sc.keep(mlx.mlx_array_new_data(&g, &[_]c_int{1}, 1, .uint32));
    }

    /// Zero every slot and restart at generation 1, once the GPU has finished what was submitted.
    pub fn reset(self: *Counters) !void {
        try quiesce();
        zeroWords(self.buf);
        self.gen = 0;
    }

    pub fn deinit(self: *Counters) void {
        _ = mlx.mlx_array_free(self.buf);
    }
};

fn quiesce() !void {
    const s = mlx.gpuStream();
    defer _ = mlx.mlx_stream_free(s);
    try mlx.check(mlx.mlx_synchronize(s));
}

fn zeroWords(a: A) void {
    const words = mlx.mlx_array_data_uint32(a) orelse return;
    @memset(@constCast(words)[0..mlx.mlx_array_size(a)], 0);
}

/// What the fused segments of one model share: their counters, the status word every bounded wait reports into, and
/// the protocol wide phases use.
pub const State = struct {
    counters: Counters,
    status: A,
    protocol: Protocol,
    groups: u32,

    pub fn init(n_slots: usize, protocol: Protocol) !State {
        var counters = try Counters.init(n_slots);
        errdefer counters.deinit();
        const status = mlx.mlx_array_new_data(&std.mem.zeroes([status_words]u32), &[_]c_int{status_words}, 1, .uint32);
        if (status.ctx == null) return error.MlxArrayNew;
        return .{ .counters = counters, .status = status, .protocol = protocol, .groups = 0 };
    }

    pub fn deinit(self: *State) void {
        self.counters.deinit();
        _ = mlx.mlx_array_free(self.status);
    }

    pub fn code(self: *const State) u32 {
        const words = mlx.mlx_array_data_uint32(self.status) orelse return 0;
        return words[0];
    }

    pub fn clear(self: *State) !void {
        try self.counters.reset();
        zeroWords(self.status);
    }
};

/// The model's `State` where its level asks for one and `serves` takes the model; spin needs a probed grid, else ticket.
pub fn create(m: *Dsv41Model) !?*State {
    if (m.fused_seg == .off or serves(m, 1) != null) return null;
    const s = try m.gpa.create(State);
    errdefer m.gpa.destroy(s);
    s.* = try State.init(m.n_layers * 2, default_protocol);
    errdefer s.deinit();
    if (s.protocol != .spin) return s;
    s.groups = spinGrid(try seg_proto.probeResidency(spin_threads, spin_tgmem));
    if (s.groups == 0) s.protocol = .ticket;
    return s;
}

pub const spin_threads: u32 = 256;

/// Percent of the probed grid a spinning launch takes: the probed ceiling moved by 10-20% between runs and a launch at it
/// timed out in 1,000.
pub const spin_margin_pct: u32 = 75;

pub fn spinGrid(probed: u32) u32 {
    return probed * spin_margin_pct / 100;
}
pub const spin_tgmem: u32 = 16384;

/// After a step's evaluation: a status word that is not ok clears the counters and the status, turns the fused segments
/// off for good and fails the step, whose output the timed-out kernels did not finish.
pub fn afterStep(m: *Dsv41Model) !void {
    const s = m.seg orelse return;
    const code = s.code();
    if (code == @intFromEnum(Status.ok)) return;
    m.fused_seg = .off;
    try s.clear();
    log.warn("[dsv41] fused seg: status {s} ({d}): fused segments off\n", .{ if (std.enums.fromInt(Status, code)) |st| @tagName(st) else "unknown", code });
    return error.Dsv41FusedSegTimeout;
}

pub var dispatches: u64 = 0;

pub const Break = enum { none, zero_shared, nan_shared };

pub var broken: Break = .none;

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

pub fn breakShared(sc: *dsv41.Scope, act: A) !A {
    const shape = mlx.getShape(act);
    const dt = mlx.mlx_array_dtype(act);
    const routed = try sc.slice(act, 1, 0, @intCast(shape[1] - 1));
    const zero = try sc.zeros(&.{ shape[0], 1, shape[2] }, dt);
    const slot = if (broken == .nan_shared) try sc.astype(try sc.add(zero, try sc.f(std.math.nan(f32))), dt) else zero;
    return sc.concat(&.{ routed, slot }, 1);
}
