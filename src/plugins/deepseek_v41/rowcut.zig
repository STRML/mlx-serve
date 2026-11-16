//! TP4 mode's row cuts (plan 61c item 3): a rank holds row VIEWS of the resident TP1 tensors, never a copy, except
//! `wo_b`'s quarter. The routed and shared experts run row-cut: quarter rows of w1/w3 give a quarter of `h`, an
//! all_gather makes `h` whole, quarter rows of w2 give a quarter of the output, a second all_gather makes it whole.
//! Up to `max_rows` rows (decode, the DSpark verify and its stages); a prompt runs TP1 on the leader.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const dsv41 = @import("deepseek_v41.zig");
const moe_up = @import("decode/moe_up.zig");
const moe_down = @import("decode/moe_down.zig");

const A = mlx.mlx_array;
const S = mlx.mlx_stream;
const Scope = dsv41.Scope;
const Lin = dsv41.Lin;
const Bank = dsv41.Bank;
const none: A = .{ .ctx = null };

/// The widest call the row-cut forward takes: a batched DSpark verify.
pub const max_rows = 24;

/// The ranks a forward spans. `gather` is MLX's all_gather: every rank's `x` concatenated along axis 0 in rank order
/// (caller frees the result).
pub const Group = struct {
    rank: u32,
    size: u32,
    ctx: ?*anyopaque = null,
    gather: *const fn (ctx: ?*anyopaque, x: A) anyerror!A,

    pub const solo: Group = .{ .rank = 0, .size = 1, .gather = soloGather };

    fn soloGather(_: ?*anyopaque, x: A) anyerror!A {
        var o = mlx.mlx_array_new();
        _ = mlx.mlx_array_set(&o, x);
        return o;
    }
};

/// Rows `[lo, hi)` of `a`'s axis `axis` as a view (caller frees).
fn view(s: S, a: A, axis: usize, lo: usize, hi: usize) !A {
    if (a.ctx == null) return none;
    const sh = mlx.getShape(a);
    var start: [4]c_int = @splat(0);
    var stop: [4]c_int = undefined;
    const st: [4]c_int = @splat(1);
    @memcpy(stop[0..sh.len], sh);
    start[axis] = @intCast(lo);
    stop[axis] = @intCast(hi);
    var o = mlx.mlx_array_new();
    mlx.check(mlx.mlx_slice(&o, a, &start, sh.len, &stop, sh.len, &st, sh.len, s)) catch |e| {
        _ = mlx.mlx_array_free(o);
        return e;
    };
    return o;
}

/// The arrays a cut made, freed together.
const Owned = struct {
    items: std.ArrayList(A) = .empty,

    fn put(self: *Owned, gpa: std.mem.Allocator, a: A) !A {
        if (a.ctx == null) return a;
        self.items.append(gpa, a) catch |e| {
            _ = mlx.mlx_array_free(a);
            return e;
        };
        return a;
    }

    fn deinit(self: *Owned, gpa: std.mem.Allocator) void {
        for (self.items.items) |a| _ = mlx.mlx_array_free(a);
        self.items.deinit(gpa);
    }
};

/// A cut of one axis: part `rank` of `size` of the `len` rows from `base`, `ceil(len / size)` a part, the last what remains.
const Cut = struct {
    base: usize = 0,
    len: usize,
    rank: u32,
    size: u32,

    fn lo(c: Cut) usize {
        return c.base + dsv41.vocabRange(c.len, c.rank, c.size).lo;
    }
    fn hi(c: Cut) usize {
        return c.base + dsv41.vocabRange(c.len, c.rank, c.size).hi;
    }
};

const Cutter = struct {
    gpa: std.mem.Allocator,
    s: S,
    owned: *Owned,
    /// Every part the same width: an all_gather joins them.
    even: bool = false,

    fn arr(k: Cutter, a: A, axis: usize, c: Cut) !A {
        if (k.even and c.len % c.size != 0) return error.Dsv41RowCutUneven;
        return k.owned.put(k.gpa, try view(k.s, a, axis, c.lo(), c.hi()));
    }
    fn bank(k: Cutter, b: Bank, c: Cut) !Bank {
        return .{ .w = try k.arr(b.w, 1, c), .s = try k.arr(b.s, 1, c), .b = try k.arr(b.b, 1, c), .qp = b.qp };
    }
    fn lin(k: Cutter, l: Lin, axis: usize, c: Cut) !Lin {
        if (l.ext != null) return error.Dsv41RowCutOutsideQuant;
        return .{ .w = try k.arr(l.w, axis, c), .s = try k.arr(l.s, axis, c), .b = try k.arr(l.b, axis, c), .qp = l.qp };
    }
    /// Columns `c` of a linear in a buffer of its own: a column cut of a row-major weight is no view. A quantized
    /// weight's columns cut whole groups.
    fn linCols(k: Cutter, l: Lin, c: Cut) !Lin {
        if (l.ext != null) return error.Dsv41RowCutOutsideQuant;
        var out: Lin = .{ .w = none, .qp = l.qp };
        inline for (.{ "w", "s", "b" }) |f| {
            const a = @field(l, f);
            if (a.ctx != null) {
                const cols: usize = @intCast(dim(a, -1));
                if (cols % c.size != 0) return error.Dsv41RowCutGroups;
                const part: Cut = .{ .len = cols, .rank = c.rank, .size = c.size };
                const v = try view(k.s, a, 1, part.lo(), part.hi());
                defer _ = mlx.mlx_array_free(v);
                var o = mlx.mlx_array_new();
                mlx.check(mlx.mlx_contiguous(&o, v, false, k.s)) catch |e| {
                    _ = mlx.mlx_array_free(o);
                    return e;
                };
                @field(out, f) = try k.owned.put(k.gpa, o);
                try mlx.check(mlx.mlx_array_eval(o));
            }
        }
        return out;
    }
};

fn dim(a: A, i: isize) c_int {
    const sh = mlx.getShape(a);
    return sh[if (i < 0) @intCast(@as(isize, @intCast(sh.len)) + i) else @intCast(i)];
}

/// Whether any weight a cut views is an outside quant plugin's: it holds no MLX arrays to view.
fn readsOutside(ly: *const dsv41.Layer) bool {
    if (ly.experts.ext != null) return true;
    if (ly.shared_gu) |gu| if (gu.ext != null) return true;
    for (ly.shared) |l| if (l.ext != null) return true;
    return ly.wq_b.ext != null or ly.wo_a.ext != null or ly.wo_b.ext != null;
}

/// One rank's cut of a layer, built from the TP1 model's resident layer: the routed experts' w1/w3 a quarter of the
/// intermediate rows and w2 a quarter of the hidden rows (middle-axis views of `[E, rows, cols]`), the shared expert
/// the same, attention by heads (`wq_b` and `wo_a` rows, the per-head sink) and `wo_b`'s columns in a copy of its own.
pub const RowLayer = struct {
    gpa: std.mem.Allocator,
    owned: Owned = .{},
    /// gate, up and down as views; `gu` unset.
    experts: dsv41.Experts,
    /// w1, w3, w2.
    shared: [3]Lin,
    wq_b: Lin,
    wo_a: Lin,
    sink: A,
    wo_b: Lin,
    rank: u32,
    size: u32,
    /// The resident banks the decode kernels read this rank's rows of in place (`moe_up.moeUpRows`,
    /// `moe_down.moeDownRows`): the fused gate+up banks and the downs.
    fused: ?struct { gu: Bank, sgu: Lin, down: Bank, sdn: Lin } = null,

    pub fn init(gpa: std.mem.Allocator, s: S, ly: *const dsv41.Layer, rank: u32, size: u32) !RowLayer {
        const e = &ly.experts;
        if (readsOutside(ly)) return error.Dsv41RowCutOutsideQuant;
        if (e.down.qp.mode == .gguf) return error.Dsv41RowCutGguf;
        var rl: RowLayer = .{ .gpa = gpa, .experts = undefined, .shared = undefined, .wq_b = undefined, .wo_a = undefined, .sink = none, .wo_b = undefined, .rank = rank, .size = size };
        if (e.gu) |gu| if (ly.shared_gu) |sgu| {
            rl.fused = .{ .gu = gu, .sgu = sgu, .down = e.down, .sdn = ly.shared[2] };
        };
        errdefer rl.owned.deinit(gpa);
        const k: Cutter = .{ .gpa = gpa, .s = s, .owned = &rl.owned, .even = true };
        // Gate and up: views into the fused bank's halves where the load fused them.
        const gate, const up = if (e.gu) |gu| blk: {
            const inter: usize = @intCast(@divExact(dim(gu.w, 1), 2));
            break :blk .{ try k.bank(gu, .{ .len = inter, .rank = rank, .size = size }), try k.bank(gu, .{ .base = inter, .len = inter, .rank = rank, .size = size }) };
        } else .{ try k.bank(e.gate, .{ .len = @intCast(dim(e.gate.w, 1)), .rank = rank, .size = size }), try k.bank(e.up, .{ .len = @intCast(dim(e.up.w, 1)), .rank = rank, .size = size }) };
        rl.experts = .{ .gate = gate, .up = up, .down = try k.bank(e.down, .{ .len = @intCast(dim(e.down.w, 1)), .rank = rank, .size = size }) };
        const w1, const w3 = if (ly.shared_gu) |gu| blk: {
            const inter: usize = @intCast(@divExact(dim(gu.w, 0), 2));
            break :blk .{ try k.lin(gu, 0, .{ .len = inter, .rank = rank, .size = size }), try k.lin(gu, 0, .{ .base = inter, .len = inter, .rank = rank, .size = size }) };
        } else .{ try k.lin(ly.shared[0], 0, .{ .len = @intCast(dim(ly.shared[0].w, 0)), .rank = rank, .size = size }), try k.lin(ly.shared[1], 0, .{ .len = @intCast(dim(ly.shared[1].w, 0)), .rank = rank, .size = size }) };
        rl.shared = .{ w1, w3, try k.lin(ly.shared[2], 0, .{ .len = @intCast(dim(ly.shared[2].w, 0)), .rank = rank, .size = size }) };
        rl.wq_b = try k.lin(ly.wq_b, 0, .{ .len = @intCast(dim(ly.wq_b.w, 0)), .rank = rank, .size = size });
        rl.wo_a = try k.lin(ly.wo_a, 0, .{ .len = @intCast(dim(ly.wo_a.w, 0)), .rank = rank, .size = size });
        rl.sink = try k.arr(ly.sink, 1, .{ .len = @intCast(dim(ly.sink, 1)), .rank = rank, .size = size });
        rl.wo_b = try k.linCols(ly.wo_b, .{ .len = @intCast(dim(ly.wo_b.w, -1)), .rank = rank, .size = size });
        return rl;
    }

    pub fn deinit(self: *RowLayer) void {
        self.owned.deinit(self.gpa);
    }
};

/// A rank's vocabulary rows of the logit head and of DSpark's Markov head, as views.
pub const RowHead = struct {
    gpa: std.mem.Allocator,
    owned: Owned = .{},
    head: Lin,
    markov_head: A = none,

    /// From the model's `head` and DSpark's `markov_head` (none without DSpark).
    pub fn init(gpa: std.mem.Allocator, s: S, head: Lin, markov_head: A, rank: u32, size: u32) !RowHead {
        if (head.ext != null) return error.Dsv41RowCutOutsideQuant;
        var rh: RowHead = .{ .gpa = gpa, .head = .{ .w = none, .qp = head.qp } };
        errdefer rh.owned.deinit(gpa);
        const k: Cutter = .{ .gpa = gpa, .s = s, .owned = &rh.owned };
        if (head.w.ctx != null) rh.head = try k.lin(head, 0, .{ .len = @intCast(dim(head.w, 0)), .rank = rank, .size = size });
        if (markov_head.ctx != null) rh.markov_head = try k.arr(markov_head, 0, .{ .len = @intCast(dim(markov_head, 0)), .rank = rank, .size = size });
        return rh;
    }

    pub fn deinit(self: *RowHead) void {
        self.owned.deinit(self.gpa);
    }
};

/// A tensor mode's row cut of the whole model's layers (`mode.build` attaches it to the mode's model as `row_cut`): the
/// mode's MoE at up to `max_rows` rows runs `forward` over these views, gathered through the cluster's `tp_gather`.
pub const RowCut = struct {
    gpa: std.mem.Allocator,
    /// By layer index, DSpark stages after the trunk; null where the whole model holds no such layer.
    layers: []?RowLayer,
    /// Calls of `run`: the row cut engaged.
    calls: usize = 0,

    /// Layers `[lo, hi)` of `whole` cut for rank `rank` of `size`.
    pub fn create(gpa: std.mem.Allocator, s: S, whole: *const dsv41.Dsv41Model, lo: usize, hi: usize, rank: u32, size: u32) !*RowCut {
        const rc = try gpa.create(RowCut);
        errdefer gpa.destroy(rc);
        rc.* = .{ .gpa = gpa, .layers = try gpa.alloc(?RowLayer, whole.layers.len) };
        @memset(rc.layers, null);
        errdefer rc.deinitLayers();
        for (lo..hi) |li| rc.layers[li] = try RowLayer.init(gpa, s, &whole.layers[li], rank, size);
        return rc;
    }

    fn deinitLayers(self: *RowCut) void {
        for (self.layers) |*l| if (l.*) |*rl| rl.deinit();
        self.gpa.free(self.layers);
    }

    pub fn destroy(self: *RowCut) void {
        self.deinitLayers();
        self.gpa.destroy(self);
    }

    /// The cut `m` runs layer `li` through at `rows` rows, or null for today's sum. Every rank decides alike: the stages
    /// only when every rank holds them (`draft_shard`).
    pub fn layer(self: *const RowCut, m: *const dsv41.Dsv41Model, li: usize, rows: usize) ?*const RowLayer {
        if (rows > max_rows or li >= self.layers.len) return null;
        if (li >= m.n_layers and !m.draft_shard) return null;
        return if (self.layers[li]) |*rl| rl else null;
    }

    /// `forward` across `m`'s tensor group.
    pub fn run(self: *RowCut, sc: *Scope, m: *const dsv41.Dsv41Model, rl: *const RowLayer, x: A, idx: A, wts: A) !A {
        const t = m.tp orelse return error.Dsv41RowCutNoGroup;
        self.calls += 1;
        return forward(sc, m, rl, .{ .rank = t.rank, .size = t.size, .gather = hookGather }, x, idx, wts);
    }

    fn hookGather(_: ?*anyopaque, x: A) anyerror!A {
        const g = dsv41.clusterHooks().tp_gather orelse return error.DsvTpNotJoined;
        return g(x);
    }
};

/// The one-row lhs indices TP1's `routed` passes where they fit this call.
fn lhsOf(L: usize, k: usize, lhs: A) A {
    return if (L == 1 and lhs.ctx != null and k == @as(usize, @intCast(dim(lhs, 1)))) lhs else none;
}

/// This rank's quarter of every pick's activation, then the shared expert's: `[L, k + 1, inter / size]` in `m.act`.
pub fn stageUp(sc: *Scope, m: *const dsv41.Dsv41Model, rl: *const RowLayer, x: A, idx: A, wts: A) !A {
    const L: usize = @intCast(dim(x, 0));
    const k: usize = @intCast(dim(idx, 1));
    const d = dim(x, 1);
    const limit = m.cfg.dsv4_swiglu_limit;
    const uidx = try sc.astype(idx, .uint32);
    if (m.dec.moe_up) |ku| if (rl.fused) |f| if (try moe_up.moeUpRows(sc, ku, f.gu, f.sgu, rl.rank, rl.size, x, uidx, try sc.astype(wts, .float32))) |act| return act;
    const xe = try sc.reshape(x, &.{ @intCast(L), 1, 1, d });
    const gt, const up = try dsv41.gateUp(sc, &rl.experts, xe, lhsOf(L, k, m.lhs_gu), uidx, false);
    const iq = dim(gt, -1);
    const act = try sc.astype(try sc.mul(try dsv41.clampedSwiglu(sc, gt, up, limit), try sc.reshape(wts, &.{ @intCast(L), @intCast(k), 1, 1 })), m.act);
    const sh = try sc.astype(try dsv41.clampedSwiglu(sc, try rl.shared[0].apply(sc, x), try rl.shared[1].apply(sc, x), limit), m.act);
    return sc.concat(&.{ try sc.reshape(act, &.{ @intCast(L), @intCast(k), iq }), try sc.reshape(sh, &.{ @intCast(L), 1, iq }) }, 1);
}

/// This rank's quarter of the output rows from the whole `h` (`[L, k + 1, inter]`): `[L, hidden / size]` f32.
pub fn stageDown(sc: *Scope, m: *const dsv41.Dsv41Model, rl: *const RowLayer, h: A, idx: A) !A {
    const L: usize = @intCast(dim(h, 0));
    const k: usize = @intCast(dim(idx, 1));
    const inter = dim(h, -1);
    const uidx = try sc.astype(idx, .uint32);
    if (m.dec.moe_down) |kd| if (rl.fused) |f| if (try moe_down.moeDownRows(sc, kd, m.dec.moe_down_rb, &f.down, &f.sdn, rl.rank, rl.size, h, uidx)) |y| return sc.astype(y, .float32);
    const hr = try sc.reshape(try sc.slice(h, 1, 0, k), &.{ @intCast(L), @intCast(k), 1, inter });
    const dn = try dsv41.gatherMm(sc, hr, &rl.experts.down, lhsOf(L, k, m.lhs_dn), uidx, false);
    const dq = dim(dn, -1);
    const y = try sc.sum(try sc.astype(try sc.reshape(dn, &.{ @intCast(L), @intCast(k), dq }), .float32), 1, false);
    const hs = try sc.reshape(try sc.slice(h, 1, k, k + 1), &.{ @intCast(L), inter });
    return sc.add(y, try sc.astype(try rl.shared[2].apply(sc, hs), .float32));
}

/// `n` ranks' parts gathered along axis 0 (`[n * L, a, q]`) as one array `[L, a, n * q]`, rank-major on the last axis.
pub fn joinRanks(sc: *Scope, parts: A, n: usize) !A {
    const sh = mlx.getShape(parts);
    const L = @divExact(sh[0], @as(c_int, @intCast(n)));
    const r = try sc.reshape(parts, &.{ @intCast(n), L, sh[1], sh[2] });
    return sc.reshape(try sc.transpose(r, &.{ 1, 2, 0, 3 }), &.{ L, sh[1], @as(c_int, @intCast(n)) * sh[2] });
}

/// The routed and shared experts over `x` (`[L, hidden]`, L <= `max_rows`) routed by `idx` / `wts`, row-cut across
/// `g`: the whole layer's MoE output on every rank, in `m.act`.
pub fn forward(sc: *Scope, m: *const dsv41.Dsv41Model, rl: *const RowLayer, g: Group, x: A, idx: A, wts: A) !A {
    const L = dim(x, 0);
    if (L > max_rows) return error.Dsv41RowCutPrompt;
    const h = try joinRanks(sc, try sc.keep(try g.gather(g.ctx, try stageUp(sc, m, rl, x, idx, wts))), g.size);
    const y = try stageDown(sc, m, rl, h, idx);
    const ys = try joinRanks(sc, try sc.keep(try g.gather(g.ctx, try sc.reshape(y, &.{ L, 1, dim(y, 1) }))), g.size);
    return sc.astype(try sc.reshape(ys, &.{ L, dim(ys, 2) }), m.act);
}

/// The launch shapes every rank pins (`topk_tg`, `hc_tg`): the smallest any rank's GPU grants, so every rank reduces in
/// the same order and attends and routes its rows alike. `granted` is this box's grant (not a shape an earlier pin
/// lowered, or a second agreement could only go down); the caller applies the result (`pinLaunchShapes`).
pub fn agreeThreads(gpa: std.mem.Allocator, g: Group, granted: [2]usize) ![2]usize {
    const mine = [_]u32{ @intCast(granted[0]), @intCast(granted[1]) };
    const all = try gatherHost(gpa, g, &mine);
    defer gpa.free(all);
    var lo = granted;
    for (0..g.size) |r| for (&lo, all[r * 2 ..][0..2]) |*l, v| {
        l.* = @min(l.*, v);
    };
    return lo;
}

/// Every rank's `mine` (the same length on each), rank-major on the host (caller frees).
fn gatherHost(gpa: std.mem.Allocator, g: Group, mine: []const u32) ![]u32 {
    const a = mlx.mlx_array_new_data(mine.ptr, &[_]c_int{ 1, @intCast(mine.len) }, 2, .uint32);
    defer _ = mlx.mlx_array_free(a);
    const all = try g.gather(g.ctx, a);
    defer _ = mlx.mlx_array_free(all);
    try mlx.check(mlx.mlx_array_eval(all));
    const d = mlx.mlx_array_data_uint32(all) orelse return error.NoData;
    return gpa.dupe(u32, d[0 .. mine.len * g.size]);
}

/// A diagnostic of the routes each layer took, one CRC32 of its ids and weights a layer, compared across the ranks:
/// the row cut is right only while every rank routes alike. Reading it syncs the GPU; off in serving.
pub const RouteCrc = struct {
    gpa: std.mem.Allocator,
    crcs: std.ArrayList(u32) = .empty,

    pub fn deinit(self: *RouteCrc) void {
        self.crcs.deinit(self.gpa);
    }

    pub fn add(self: *RouteCrc, r: dsv41.Routes) !void {
        var h = std.hash.Crc32.init();
        inline for (.{ .{ r.idx, .uint32 }, .{ r.wts, .float32 } }) |p| {
            var c = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(c);
            try mlx.check(mlx.mlx_astype(&c, p[0], p[1], mlx.gpuStream()));
            var f = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(f);
            try mlx.check(mlx.mlx_contiguous(&f, c, false, mlx.gpuStream()));
            try mlx.check(mlx.mlx_array_eval(f));
            const n = mlx.mlx_array_size(f);
            const bytes: [*]const u8 = if (p[1] == .uint32) @ptrCast(mlx.mlx_array_data_uint32(f) orelse return error.NoData) else @ptrCast(mlx.mlx_array_data_float32(f) orelse return error.NoData);
            h.update(bytes[0 .. n * 4]);
        }
        try self.crcs.append(self.gpa, h.final());
    }

    /// The first layer some rank routed differently from rank 0, or null when all agree.
    pub fn firstDivergence(self: *const RouteCrc, g: Group) !?usize {
        const n = self.crcs.items.len;
        if (n == 0) return null;
        const all = try gatherHost(self.gpa, g, self.crcs.items);
        defer self.gpa.free(all);
        for (0..n) |li| for (1..g.size) |r| if (all[r * n + li] != all[li]) return li;
        return null;
    }
};

/// What a TP4 prompt chunk needs from the mode it runs in. Rank 0 runs the chunk on its TP1 model alone (`run`, which
/// hands back the slot's snapshot after it), then `ship` sends that snapshot's cache tensors to every other rank under
/// its id and the DSpark stages it holds: `dsv41.tpLoad`'s path through the cluster's `tp_load` hook in serving, a
/// recorder in a test.
pub const LeaderPrefill = struct {
    ctx: ?*anyopaque = null,
    whole: *const dsv41.Dsv41Model,
    run: *const fn (ctx: ?*anyopaque, ids: []const u32, p0: usize) anyerror!*const dsv41.Snap,
    ship: *const fn (ctx: ?*anyopaque, id: u64, ids: []const u32, tensors: []const A, stages: usize) anyerror!void,
};

/// Rank 0's half of a prompt chunk (`ids` from position `p0`): the chunk on the TP1 model, then its state to the other
/// ranks. Returns the snapshot `run` made.
pub fn prefillOnLeader(gpa: std.mem.Allocator, g: Group, lp: LeaderPrefill, ids: []const u32, p0: usize) !*const dsv41.Snap {
    if (g.rank != 0) return error.Dsv41RowCutNotLeader;
    const sn = try lp.run(lp.ctx, ids, p0);
    if (g.size == 1) return sn;
    const xs = try dsv41.planTensors(lp.whole, gpa, sn, try dsv41.shipPlan(lp.whole, sn.n, lp.whole.n_mtp));
    defer dsv41.freeArrays(gpa, xs);
    try lp.ship(lp.ctx, sn.tp_id, sn.ids, xs, lp.whole.n_mtp);
    return sn;
}

/// A follower's half: rank 0's snapshot `id` rebuilt from the tensors it shipped with `stages` DSpark stages, into
/// `snaps` under that id (the next TP4 turn restores it). The KV is whole on every rank, so the follower's TP1 model
/// takes it as is; a follower without stages (rank 0 drafts alone) skips their tensors.
pub fn importOnFollower(gpa: std.mem.Allocator, g: Group, whole: *const dsv41.Dsv41Model, snaps: *dsv41.Snaps, id: u64, ids: []const u32, tensors: []const A, stages: usize) !void {
    if (g.rank == 0) return error.Dsv41RowCutLeaderImport;
    const own = try dsv41.shipPlan(whole, ids.len, whole.n_mtp);
    if (stages == whole.n_mtp) return dsv41.importPlan(whole, gpa, snaps, id, ids, tensors, own);
    if (whole.n_mtp != 0) return error.Dsv41RowCutStagesMissing;
    const sent = try dsv41.shipPlan(whole, ids.len, stages);
    if (tensors.len != sent.len) return error.DistPeerDiverged;
    const kept = try gpa.alloc(A, own.len);
    defer gpa.free(kept);
    var k: usize = 0;
    for (sent.specs(), tensors) |spec, x| {
        if (spec.kind == .ds or spec.kind == .mh) continue;
        kept[k] = x;
        k += 1;
    }
    try dsv41.importPlan(whole, gpa, snaps, id, ids, kept, own);
}

// ── tests ────────────────────────────────────────────────────────────────────

const testing = std.testing;
const fixture = @import("decode/fixture.zig");
const decode = @import("decode/decode.zig");
const device = @import("../../mlx/device.zig");

fn host(gpa: std.mem.Allocator, a: A) ![]f32 {
    var c = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(c);
    try mlx.check(mlx.mlx_astype(&c, a, .float32, mlx.gpuStream()));
    var f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f);
    try mlx.check(mlx.mlx_contiguous(&f, c, false, mlx.gpuStream()));
    try mlx.check(mlx.mlx_array_eval(f));
    const d = mlx.mlx_array_data_float32(f) orelse return error.NoData;
    return gpa.dupe(f32, d[0..mlx.mlx_array_size(f)]);
}

/// Every rank's cut of layer `li`.
fn cutRanks(gpa: std.mem.Allocator, ly: *const dsv41.Layer, n: u32) ![]RowLayer {
    const rls = try gpa.alloc(RowLayer, n);
    var made: usize = 0;
    errdefer {
        for (rls[0..made]) |*r| r.deinit();
        gpa.free(rls);
    }
    for (rls, 0..) |*r, i| {
        r.* = try RowLayer.init(gpa, mlx.gpuStream(), ly, @intCast(i), n);
        made += 1;
    }
    return rls;
}

/// The row-cut MoE of `n` ranks run in one process: each rank's stage, the gathers as concatenations.
fn simulate(sc: *Scope, m: *const dsv41.Dsv41Model, rls: []const RowLayer, x: A, idx: A, wts: A) !A {
    var parts: [8]A = undefined;
    for (rls, 0..) |*rl, r| parts[r] = try stageUp(sc, m, rl, x, idx, wts);
    const h = try joinRanks(sc, try sc.concat(parts[0..rls.len], 0), rls.len);
    for (rls, 0..) |*rl, r| {
        const y = try stageDown(sc, m, rl, h, idx);
        parts[r] = try sc.reshape(y, &.{ dim(y, 0), 1, dim(y, 1) });
    }
    const y = try joinRanks(sc, try sc.concat(parts[0..rls.len], 0), rls.len);
    return sc.astype(try sc.reshape(y, &.{ dim(y, 0), dim(y, 2) }), m.act);
}

/// `got` within today's tensor-parallel bar of the whole model's `want` (`decode/fixture.zig`): 1e-2 of the peak on a
/// NAX GPU, whose f32 GEMMs run as TF32, 1e-3 elsewhere. Prints the measured error either way.
fn expectTpClose(label: []const u8, rows: usize, want: []const f32, got: []const f32) !void {
    try testing.expectEqual(want.len, got.len);
    var peak: f32 = 0;
    var err: f32 = 0;
    for (want, got) |u, v| {
        if (!std.math.isFinite(u) or !std.math.isFinite(v)) return error.NonFinite;
        peak = @max(peak, @abs(u));
        err = @max(err, @abs(u - v));
    }
    try testing.expect(peak > 0);
    const tol: f32 = if (device.naxAvailable()) 1e-2 else 1e-3;
    const rms = decode.relRms(got, want);
    std.debug.print("[rowcut] {s} R={d}: max-abs {e:.2} ({e:.2} of peak, bar {e:.0}), rel rms {e:.2}\n", .{ label, rows, err, err / peak, tol, rms });
    try testing.expect(err <= tol * peak);
    try testing.expect(rms <= tol);
}

fn routes(sc: *Scope, fx: *fixture.Fixture, ly: *const dsv41.Layer, rows: usize, dt: mlx.mlx_dtype) !struct { x: A, r: dsv41.Routes } {
    const x = try sc.keep(try fx.normal(&.{ @intCast(rows), @intCast(fx.m.cfg.hidden_size) }, dt, 0, 1));
    return .{ .x = x, .r = try dsv41.route(sc, fx.m, ly, x, none) };
}

test "dsv41 row cut: four ranks' row-cut MoE with its two gathers is the TP1 MoE at 1, 6 and 24 rows" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const gpa = testing.allocator;
    const fx = try fixture.Fixture.init(gpa, .{ .lo = 20, .hi = 21, .experts = 8, .ctx = 0, .act = .float32 });
    defer fx.deinit();
    const ly = fx.layer(20);
    const rls = try cutRanks(gpa, ly, 4);
    defer {
        for (rls) |*r| r.deinit();
        gpa.free(rls);
    }
    for ([_]usize{ 1, 6, max_rows }) |rows| {
        var sc = Scope.init(gpa, mlx.gpuStream());
        defer sc.deinit();
        const in = try routes(&sc, fx, ly, rows, .float32);
        const want = try dsv41.moeShare(fx.m, gpa, 20, in.x, in.r.idx, in.r.wts);
        defer _ = mlx.mlx_array_free(want);
        const a = try host(gpa, want);
        defer gpa.free(a);
        const b = try host(gpa, try simulate(&sc, fx.m, rls, in.x, in.r.idx, in.r.wts));
        defer gpa.free(b);
        try expectTpClose("four ranks", rows, a, b);
    }
}

test "dsv41 row cut: in bf16 the four ranks land no further from an f32 model than the TP1 MoE does" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const gpa = testing.allocator;
    // Today's tensor-parallel bf16 bar (`decode/fixture.zig`): the median over seeds, 1.25x the chain's error.
    for ([_]usize{ 1, 6, max_rows }) |rows| {
        var cut: [5]f64 = undefined;
        var chain: [5]f64 = undefined;
        for (&cut, &chain, 1..) |*e_cut, *e_chain, seed| {
            const truth = try fixture.Fixture.init(gpa, .{ .lo = 20, .hi = 21, .experts = 8, .ctx = 0, .act = .float32, .seed = seed });
            defer truth.deinit();
            const plain = try fixture.Fixture.init(gpa, .{ .lo = 20, .hi = 21, .experts = 8, .ctx = 0, .seed = seed });
            defer plain.deinit();
            const ly = plain.layer(20);
            const rls = try cutRanks(gpa, ly, 4);
            defer {
                for (rls) |*r| r.deinit();
                gpa.free(rls);
            }
            var sc = Scope.init(gpa, mlx.gpuStream());
            defer sc.deinit();
            // One set of routes for all three, so a near-tie flip cannot pick different experts.
            const in = try routes(&sc, plain, ly, rows, .bfloat16);
            const t32 = try dsv41.moeShare(truth.m, gpa, 20, try sc.astype(in.x, .float32), in.r.idx, in.r.wts);
            defer _ = mlx.mlx_array_free(t32);
            const c16 = try dsv41.moeShare(plain.m, gpa, 20, in.x, in.r.idx, in.r.wts);
            defer _ = mlx.mlx_array_free(c16);
            const t = try host(gpa, t32);
            defer gpa.free(t);
            const c = try host(gpa, c16);
            defer gpa.free(c);
            const g = try host(gpa, try simulate(&sc, plain.m, rls, in.x, in.r.idx, in.r.wts));
            defer gpa.free(g);
            e_cut.* = decode.relRms(g, t);
            e_chain.* = decode.relRms(c, t);
        }
        std.mem.sort(f64, &cut, {}, std.sort.asc(f64));
        std.mem.sort(f64, &chain, {}, std.sort.asc(f64));
        std.debug.print("[rowcut] R={d}: median cut-truth {e:.2}, chain-truth {e:.2} (rel rms)\n", .{ rows, cut[2], chain[2] });
        try testing.expect(cut[2] <= 1.25 * chain[2] + 1e-4);
    }
}

/// Every rank of a row-cut group in one process: `gather` computes the other ranks' parts from the same input, so each
/// rank's own `forward` runs whole, its two gathers included.
const Lockstep = struct {
    sc: *Scope,
    m: *const dsv41.Dsv41Model,
    rls: []const RowLayer,
    rank: u32,
    x: A,
    idx: A,
    wts: A,
    h: A = none,

    fn group(self: *Lockstep) Group {
        return .{ .rank = self.rank, .size = @intCast(self.rls.len), .ctx = self, .gather = gather };
    }

    fn gather(ctx: ?*anyopaque, mine: A) anyerror!A {
        const self: *Lockstep = @ptrCast(@alignCast(ctx.?));
        var parts: [8]A = undefined;
        const up = self.h.ctx == null;
        for (self.rls, 0..) |*rl, r| {
            if (r == self.rank) {
                parts[r] = mine;
            } else if (up) {
                parts[r] = try stageUp(self.sc, self.m, rl, self.x, self.idx, self.wts);
            } else {
                const y = try stageDown(self.sc, self.m, rl, self.h, self.idx);
                parts[r] = try self.sc.reshape(y, &.{ dim(y, 0), 1, dim(y, 1) });
            }
        }
        const all = try self.sc.concat(parts[0..self.rls.len], 0);
        if (up) self.h = try joinRanks(self.sc, all, self.rls.len);
        return self.sc.out(all);
    }
};

test "dsv41 row cut: every rank's forward over a four-rank group is the TP1 MoE" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const gpa = testing.allocator;
    const fx = try fixture.Fixture.init(gpa, .{ .lo = 20, .hi = 21, .experts = 8, .ctx = 0, .act = .float32 });
    defer fx.deinit();
    const ly = fx.layer(20);
    const rls = try cutRanks(gpa, ly, 4);
    defer {
        for (rls) |*r| r.deinit();
        gpa.free(rls);
    }
    var solo = try RowLayer.init(gpa, mlx.gpuStream(), ly, 0, 1);
    defer solo.deinit();
    for ([_]usize{ 1, 6, max_rows }) |rows| {
        var sc = Scope.init(gpa, mlx.gpuStream());
        defer sc.deinit();
        const in = try routes(&sc, fx, ly, rows, .float32);
        const want = try dsv41.moeShare(fx.m, gpa, 20, in.x, in.r.idx, in.r.wts);
        defer _ = mlx.mlx_array_free(want);
        const a = try host(gpa, want);
        defer gpa.free(a);
        for (0..4) |r| {
            var ls: Lockstep = .{ .sc = &sc, .m = fx.m, .rls = rls, .rank = @intCast(r), .x = in.x, .idx = in.r.idx, .wts = in.r.wts };
            const b = try host(gpa, try forward(&sc, fx.m, &rls[r], ls.group(), in.x, in.r.idx, in.r.wts));
            defer gpa.free(b);
            try expectTpClose("rank forward", rows, a, b);
        }
        const b = try host(gpa, try forward(&sc, fx.m, &solo, Group.solo, in.x, in.r.idx, in.r.wts));
        defer gpa.free(b);
        try expectTpClose("one-rank forward", rows, a, b);
    }
}

fn same(a: A, b: A) !bool {
    if (a.ctx == null or b.ctx == null) return a.ctx == b.ctx;
    if (!std.mem.eql(c_int, mlx.getShape(a), mlx.getShape(b))) return false;
    var o = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(o);
    try mlx.check(mlx.mlx_array_equal(&o, a, b, false, mlx.gpuStream()));
    try mlx.check(mlx.mlx_array_eval(o));
    var eq = false;
    try mlx.check(mlx.mlx_array_item_bool(&eq, o));
    return eq;
}

fn sameLin(a: Lin, b: Lin) !bool {
    return try same(a.w, b.w) and try same(a.s, b.s) and try same(a.b, b.b);
}

fn nbytes(a: A) usize {
    return mlx.mlx_array_size(a) * mlx.mlx_array_itemsize(a);
}

fn activeBytes() usize {
    var n: usize = 0;
    _ = mlx.mlx_get_active_memory(&n);
    return n;
}

test "dsv41 row cut: a rank's attention cut is today's tensor-parallel cut of the same tensors, gate and up its fused halves" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const gpa = testing.allocator;
    const whole = try fixture.Fixture.init(gpa, .{ .lo = 20, .hi = 21, .experts = 8, .ctx = 0 });
    defer whole.deinit();
    const rank = try fixture.Fixture.init(gpa, .{ .lo = 20, .hi = 21, .experts = 8, .ctx = 0, .tp = .{ .rank = 2, .size = 4 } });
    defer rank.deinit();
    var rl = try RowLayer.init(gpa, mlx.gpuStream(), whole.layer(20), 2, 4);
    defer rl.deinit();
    const ly = rank.layer(20);
    try testing.expect(try sameLin(rl.wq_b, ly.wq_b));
    try testing.expect(try sameLin(rl.wo_a, ly.wo_a));
    try testing.expect(try same(rl.sink, ly.sink));
    try testing.expect(try sameLin(rl.wo_b, ly.wo_b));
    const gu = ly.experts.gu.?;
    const half: usize = @intCast(@divExact(dim(gu.w, 1), 2));
    var sc = Scope.init(gpa, mlx.gpuStream());
    defer sc.deinit();
    inline for (.{ "w", "s", "b" }) |f| if (@field(gu, f).ctx != null) {
        try testing.expect(try same(@field(rl.experts.gate, f), try sc.slice(@field(gu, f), 1, 0, half)));
        try testing.expect(try same(@field(rl.experts.up, f), try sc.slice(@field(gu, f), 1, half, 2 * half)));
    };
}

/// Rows `[0, 2 * rows)` of a bank's axis 1, every other one: the same shape as a row view, but no row-contiguous matrix.
fn strided(gpa: std.mem.Allocator, owned: *Owned, b: Bank, rows: usize) !Bank {
    var out = b;
    inline for (.{ "w", "s", "b" }) |f| if (@field(b, f).ctx != null) {
        const a = @field(b, f);
        const sh = mlx.getShape(a);
        var o = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_slice(&o, a, &[_]c_int{ 0, 0, 0 }, 3, &[_]c_int{ sh[0], @intCast(2 * rows), sh[2] }, 3, &[_]c_int{ 1, 2, 1 }, 3, mlx.gpuStream()));
        @field(out, f) = try owned.put(gpa, o);
    };
    return out;
}

/// The memory peak above what is live of one rank's up stage over `in` (1 row).
fn stagePeak(gpa: std.mem.Allocator, fx: *fixture.Fixture, rl: *const RowLayer, x: A, idx: A, wts: A) !usize {
    var sc = Scope.init(gpa, mlx.gpuStream());
    defer sc.deinit();
    const base = activeBytes();
    _ = mlx.mlx_reset_peak_memory();
    const h = try stageUp(&sc, fx.m, rl, x, idx, wts);
    try mlx.check(mlx.mlx_array_eval(h));
    var peak: usize = 0;
    _ = mlx.mlx_get_peak_memory(&peak);
    return peak -| base;
}

test "dsv41 row cut: the views copy no weight bytes, and the row-cut forward reads them in place" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const gpa = testing.allocator;
    const fx = try fixture.Fixture.init(gpa, .{ .lo = 20, .hi = 21, .experts = 8, .ctx = 0 });
    defer fx.deinit();
    const ly = fx.layer(20);
    const before = activeBytes();
    const rls = try cutRanks(gpa, ly, 4);
    defer {
        for (rls) |*r| r.deinit();
        gpa.free(rls);
    }
    var wo_b: usize = 0;
    // The smallest weight view a matmul reads: a copy of any would allocate at least this much.
    var smallest: usize = std.math.maxInt(usize);
    for (rls) |*rl| {
        inline for (.{ "w", "s", "b" }) |f| if (@field(rl.wo_b, f).ctx != null) {
            wo_b += nbytes(@field(rl.wo_b, f));
        };
        for ([_]A{ rl.experts.gate.w, rl.experts.up.w, rl.experts.down.w, rl.shared[0].w, rl.shared[1].w, rl.shared[2].w }) |a| smallest = @min(smallest, nbytes(a));
        for (rl.owned.items.items) |a| try mlx.check(mlx.mlx_array_eval(a));
    }
    const grew = activeBytes() - before;
    try testing.expect(wo_b > 0 and grew >= wo_b and grew - wo_b < smallest);

    var sc = Scope.init(gpa, mlx.gpuStream());
    defer sc.deinit();
    const in = try routes(&sc, fx, ly, 1, .bfloat16);
    for ([_]A{ in.x, in.r.idx, in.r.wts }) |a| try mlx.check(mlx.mlx_array_eval(a));
    // The whole forward at one row: transients of a few KiB, far below any view.
    {
        var fsc = Scope.init(gpa, mlx.gpuStream());
        defer fsc.deinit();
        const base = activeBytes();
        _ = mlx.mlx_reset_peak_memory();
        try mlx.check(mlx.mlx_array_eval(try simulate(&fsc, fx.m, rls, in.x, in.r.idx, in.r.wts)));
        var peak: usize = 0;
        _ = mlx.mlx_get_peak_memory(&peak);
        try testing.expect(peak -| base < smallest / 2);
    }
    // The planted control: a gate view with a row stride makes the gather copy it, and the counter sees the copy.
    const in_place = try stagePeak(gpa, fx, &rls[0], in.x, in.r.idx, in.r.wts);
    var planted = rls[0];
    var extra: Owned = .{};
    defer extra.deinit(gpa);
    const iq: usize = @intCast(dim(rls[0].experts.gate.w, 1));
    planted.experts.gate = try strided(gpa, &extra, ly.experts.gu.?, iq);
    const copied = try stagePeak(gpa, fx, &planted, in.x, in.r.idx, in.r.wts);
    try testing.expect(in_place < smallest / 2);
    try testing.expect(copied >= nbytes(planted.experts.gate.w));
}

test "dsv41 row cut: a rank's head and Markov head are its vocabulary rows, as views" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const gpa = testing.allocator;
    var vals: [16 * 4]f32 = undefined;
    for (&vals, 0..) |*v, i| v.* = @floatFromInt(i);
    const w = mlx.mlx_array_new_data(&vals, &[_]c_int{ 16, 4 }, 2, .float32);
    defer _ = mlx.mlx_array_free(w);
    const mk = mlx.mlx_array_new_data(&vals, &[_]c_int{ 16, 4 }, 2, .float32);
    defer _ = mlx.mlx_array_free(mk);
    var rh = try RowHead.init(gpa, mlx.gpuStream(), .{ .w = w }, mk, 3, 4);
    defer rh.deinit();
    for ([_]A{ rh.head.w, rh.markov_head }) |a| {
        const got = try host(gpa, a);
        defer gpa.free(got);
        try testing.expectEqualSlices(f32, vals[12 * 4 ..], got);
    }
    var bare = try RowHead.init(gpa, mlx.gpuStream(), .{ .w = w }, none, 0, 4);
    defer bare.deinit();
    try testing.expect(bare.markov_head.ctx == null);
}

test "dsv41 row cut: a DSpark stage's MoE (its own experts, top-3) row-cut is the stage's TP1 MoE" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const gpa = testing.allocator;
    const fx = try fixture.Fixture.init(gpa, .{ .lo = 20, .hi = 21, .experts = 8, .ctx = 0, .act = .float32 });
    defer fx.deinit();
    var whole = try fixture.releaseConfig();
    whole.dsv41_dspark_experts = 8;
    fx.m.cfg.dsv41_dspark_experts = 8;
    try fx.putLayer(&whole, "mtp.{d}.", 0, 8, false, null);
    const m = fx.m;
    const stage = try dsv41.loadLayer(m, &fx.w, "mtp", 0, null);
    m.layers = try gpa.realloc(m.layers, m.n_layers + 1);
    m.layers[m.n_layers] = stage;
    const ly = &m.layers[m.n_layers];
    try testing.expectEqual(@as(usize, whole.dsv41_dspark_top_k), ly.top_k);
    const rls = try cutRanks(gpa, ly, 4);
    defer {
        for (rls) |*r| r.deinit();
        gpa.free(rls);
    }
    for ([_]usize{ 1, 6, max_rows }) |rows| {
        var sc = Scope.init(gpa, mlx.gpuStream());
        defer sc.deinit();
        const in = try routes(&sc, fx, ly, rows, .float32);
        const want = try dsv41.moeShare(m, gpa, m.n_layers, in.x, in.r.idx, in.r.wts);
        defer _ = mlx.mlx_array_free(want);
        const a = try host(gpa, want);
        defer gpa.free(a);
        const b = try host(gpa, try simulate(&sc, m, rls, in.x, in.r.idx, in.r.wts));
        defer gpa.free(b);
        try expectTpClose("DSpark stage", rows, a, b);
    }
}

test "dsv41 row cut: routes tied at the top-k cut-off pick the lower expert ids, the same on every call" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const gpa = testing.allocator;
    const fx = try fixture.Fixture.init(gpa, .{ .lo = 20, .hi = 21, .experts = 8, .ctx = 0 });
    defer fx.deinit();
    var ly = fx.layer(20).*;
    var sc = Scope.init(gpa, mlx.gpuStream());
    defer sc.deinit();
    var zb = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_zeros(&zb, &[_]c_int{ 1, 8 }, 2, .float32, mlx.gpuStream()));
    ly.gate_bias = try sc.keep(zb);
    // Rows of 8 logits: two clear winners, then six tied for the remaining picks.
    const k = ly.top_k;
    var vals: [3 * 8]f32 = undefined;
    for (0..3) |r| for (0..8) |e| {
        vals[r * 8 + e] = if (e == 3 or e == 6) 2.0 else 0.5;
    };
    const logits = try sc.keep(mlx.mlx_array_new_data(&vals, &[_]c_int{ 3, 8 }, 2, .float32));
    var first: ?[]u32 = null;
    defer if (first) |f| gpa.free(f);
    for (0..4) |_| {
        const r = try dsv41.routeFrom(&sc, fx.m, &ly, logits, none);
        const ids = try sc.astype(r.idx, .uint32);
        try mlx.check(mlx.mlx_array_eval(ids));
        const got = (mlx.mlx_array_data_uint32(ids) orelse return error.NoData)[0 .. 3 * k];
        const sorted = try gpa.dupe(u32, got[0..k]);
        defer gpa.free(sorted);
        std.mem.sort(u32, sorted, {}, std.sort.asc(u32));
        // The two winners, then the lowest of the tied ids.
        const tied = [_]u32{ 0, 1, 2, 4, 5, 7 };
        var want: [8]u32 = undefined;
        @memcpy(want[0 .. k - 2], tied[0 .. k - 2]);
        want[k - 2] = 3;
        want[k - 1] = 6;
        std.mem.sort(u32, want[0..k], {}, std.sort.asc(u32));
        try testing.expectEqualSlices(u32, want[0..k], sorted);
        if (first) |f| try testing.expectEqualSlices(u32, f, got) else first = try gpa.dupe(u32, got);
    }
}

/// A group of ranks whose other members' rows are fixed: `gather` stacks this rank's `x` with theirs, in rank order.
const Stub = struct {
    rank: u32,
    others: []const []const u32,

    fn group(self: *Stub) Group {
        return .{ .rank = self.rank, .size = @intCast(self.others.len + 1), .ctx = self, .gather = gather };
    }

    fn gather(ctx: ?*anyopaque, x: A) anyerror!A {
        const self: *Stub = @ptrCast(@alignCast(ctx.?));
        try mlx.check(mlx.mlx_array_eval(x));
        const n = mlx.mlx_array_size(x);
        const mine = (mlx.mlx_array_data_uint32(x) orelse return error.NoData)[0..n];
        var rows: [8 * 64]u32 = undefined;
        var o: usize = 0;
        for (0..self.others.len + 1) |r| {
            const src = if (r == self.rank) mine else self.others[if (r < self.rank) r else r - 1];
            @memcpy(rows[o..][0..n], src);
            o += n;
        }
        return mlx.mlx_array_new_data(&rows, &[_]c_int{ @intCast(self.others.len + 1), @intCast(n) }, 2, .uint32);
    }
};

test "dsv41 row cut: the ranks agree on the smallest launch shapes any of their GPUs grants" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const gpa = testing.allocator;
    var stub: Stub = .{ .rank = 1, .others = &.{ &.{ 1024, 512 }, &.{ 512, 512 }, &.{ 1024, 256 } } };
    try testing.expectEqual([2]usize{ 512, 256 }, try agreeThreads(gpa, stub.group(), .{ 1024, 512 }));
    var alone: Stub = .{ .rank = 0, .others = &.{} };
    try testing.expectEqual([2]usize{ 1024, 512 }, try agreeThreads(gpa, alone.group(), .{ 1024, 512 }));
}

test "dsv41 row cut: the route CRC names the first layer a rank routed differently" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const gpa = testing.allocator;
    const fx = try fixture.Fixture.init(gpa, .{ .lo = 20, .hi = 21, .experts = 8, .ctx = 0 });
    defer fx.deinit();
    const ly = fx.layer(20);
    var crc: RouteCrc = .{ .gpa = gpa };
    defer crc.deinit();
    var sc = Scope.init(gpa, mlx.gpuStream());
    defer sc.deinit();
    const in = try routes(&sc, fx, ly, 6, .bfloat16);
    try crc.add(in.r);
    try crc.add(try dsv41.route(&sc, fx.m, ly, in.x, none));
    try testing.expectEqual(crc.crcs.items[0], crc.crcs.items[1]);
    // A different input routes differently.
    try crc.add((try routes(&sc, fx, ly, 6, .bfloat16)).r);
    try testing.expect(crc.crcs.items[2] != crc.crcs.items[0]);
    const mine = crc.crcs.items;
    var agree: Stub = .{ .rank = 0, .others = &.{ mine, mine, mine } };
    try testing.expectEqual(@as(?usize, null), try crc.firstDivergence(agree.group()));
    const off = [_]u32{ mine[0], mine[1], mine[2] +% 1 };
    var apart: Stub = .{ .rank = 0, .others = &.{ mine, &off, mine } };
    try testing.expectEqual(@as(?usize, 2), try crc.firstDivergence(apart.group()));
}

test "dsv41 row cut: the fused up launch over a rank's rows of the resident banks is that rank's chain up stage" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const gpa = testing.allocator;
    const fx = try fixture.Fixture.init(gpa, .{ .lo = 20, .hi = 21, .experts = 8, .ctx = 0, .fused = true });
    defer fx.deinit();
    // The affine decode kernels are built on NAX GPUs only.
    const ku = fx.m.dec.moe_up orelse return error.SkipZigTest;
    const ly = fx.layer(20);
    const rls = try cutRanks(gpa, ly, 4);
    defer {
        for (rls) |*r| r.deinit();
        gpa.free(rls);
    }
    for ([_]usize{ 1, decode.max_rows }) |rows| {
        var sc = Scope.init(gpa, mlx.gpuStream());
        defer sc.deinit();
        const in = try routes(&sc, fx, ly, rows, .bfloat16);
        for (rls) |*rl| {
            const f = rl.fused.?;
            const got = (try moe_up.moeUpRows(&sc, ku, f.gu, f.sgu, rl.rank, rl.size, in.x, try sc.astype(in.r.idx, .uint32), in.r.wts)) orelse return error.Declined;
            var chain = rl.*;
            chain.fused = null;
            const want = try stageUp(&sc, fx.m, &chain, in.x, in.r.idx, in.r.wts);
            try testing.expectEqualSlices(c_int, mlx.getShape(want), mlx.getShape(got));
            const a = try host(gpa, want);
            defer gpa.free(a);
            const b = try host(gpa, got);
            defer gpa.free(b);
            try testing.expect(decode.relRms(b, a) < 5e-3);
        }
    }
}

test "dsv41 row cut: the fused down launch over a rank's rows of the resident banks is that rank's chain down stage" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const gpa = testing.allocator;
    const fx = try fixture.Fixture.init(gpa, .{ .lo = 20, .hi = 21, .experts = 8, .ctx = 0, .fused = true });
    defer fx.deinit();
    // The affine decode kernels are built on NAX GPUs only.
    const kd = fx.m.dec.moe_down orelse return error.SkipZigTest;
    const ly = fx.layer(20);
    const rls = try cutRanks(gpa, ly, 4);
    defer {
        for (rls) |*r| r.deinit();
        gpa.free(rls);
    }
    for ([_]usize{ 1, decode.max_rows }) |rows| {
        var sc = Scope.init(gpa, mlx.gpuStream());
        defer sc.deinit();
        const in = try routes(&sc, fx, ly, rows, .bfloat16);
        var parts: [4]A = undefined;
        for (rls, &parts) |*rl, *p| p.* = try stageUp(&sc, fx.m, rl, in.x, in.r.idx, in.r.wts);
        const h = try joinRanks(&sc, try sc.concat(&parts, 0), 4);
        for (rls) |*rl| {
            const f = rl.fused.?;
            const got = (try moe_down.moeDownRows(&sc, kd, fx.m.dec.moe_down_rb, &f.down, &f.sdn, rl.rank, rl.size, h, try sc.astype(in.r.idx, .uint32))) orelse return error.Declined;
            var chain = rl.*;
            chain.fused = null;
            const want = try stageDown(&sc, fx.m, &chain, h, in.r.idx);
            try testing.expectEqualSlices(c_int, mlx.getShape(want), mlx.getShape(got));
            const a = try host(gpa, want);
            defer gpa.free(a);
            const b = try host(gpa, got);
            defer gpa.free(b);
            try testing.expect(decode.relRms(b, a) < 5e-3);
        }
    }
}

/// The test seam of a TP4 prompt chunk: `run` hands back a snapshot built from `disk`, `ship` imports what the leader
/// sends into a follower's snapshots.
const PrefillRig = struct {
    gpa: std.mem.Allocator,
    leader: *dsv41.Dsv41Model,
    follower: *dsv41.Dsv41Model,
    lead: dsv41.Snaps,
    follow: dsv41.Snaps,
    disk: []const A,
    shipped: usize = 0,

    fn run(ctx: ?*anyopaque, ids: []const u32, _: usize) anyerror!*const dsv41.Snap {
        const self: *PrefillRig = @ptrCast(@alignCast(ctx.?));
        try dsv41.importPlan(self.leader, self.gpa, &self.lead, 7, ids, self.disk, try dsv41.shipPlan(self.leader, ids.len, self.leader.n_mtp));
        const sn = self.lead.getPtr(7).?;
        sn.tp_id = 7;
        return sn;
    }

    fn ship(ctx: ?*anyopaque, id: u64, ids: []const u32, tensors: []const A, stages: usize) anyerror!void {
        const self: *PrefillRig = @ptrCast(@alignCast(ctx.?));
        self.shipped += 1;
        try importOnFollower(self.gpa, .{ .rank = 1, .size = 4, .gather = Group.solo.gather }, self.follower, &self.follow, id, ids, tensors, stages);
    }
};

/// A prompt chunk on `lead` reaches `follow` as the same snapshot of every tensor the follower holds, under the
/// leader's id.
fn expectLeaderShips(gpa: std.mem.Allocator, lead: *dsv41.Dsv41Model, follow: *dsv41.Dsv41Model) !void {
    const n = 301;
    const plan = try dsv41.shipPlan(lead, n, lead.n_mtp);
    var sc = Scope.init(gpa, lead.s);
    defer sc.deinit();
    const disk = try gpa.alloc(A, plan.len);
    defer gpa.free(disk);
    for (plan.specs(), disk, 0..) |spec, *x, k| {
        const shape = [_]c_int{ @intCast(spec.rows), @intCast(spec.width) };
        x.* = try sc.reshape(try sc.astype(try sc.arange(k, k + spec.rows * spec.width), spec.dtype), shape[0..spec.rank()]);
    }
    const ids = try gpa.alloc(u32, n);
    defer gpa.free(ids);
    @memset(ids, 5);
    var rig: PrefillRig = .{ .gpa = gpa, .leader = lead, .follower = follow, .lead = .init(gpa), .follow = .init(gpa), .disk = disk };
    defer dsv41.dropSnaps(&rig.lead, gpa);
    defer dsv41.dropSnaps(&rig.follow, gpa);
    const lp: LeaderPrefill = .{ .ctx = &rig, .whole = lead, .run = PrefillRig.run, .ship = PrefillRig.ship };
    const four: Group = .{ .rank = 0, .size = 4, .gather = Group.solo.gather };
    _ = try prefillOnLeader(gpa, four, lp, ids, 0);
    try testing.expectEqual(@as(usize, 1), rig.shipped);
    const got = rig.follow.getPtr(7) orelse return error.NotShipped;
    try testing.expectEqual(@as(usize, n), got.n);
    const want = try dsv41.planTensors(lead, gpa, rig.lead.getPtr(7).?, try dsv41.shipPlan(lead, n, follow.n_mtp));
    defer dsv41.freeArrays(gpa, want);
    const back = try dsv41.planTensors(follow, gpa, got, try dsv41.shipPlan(follow, n, follow.n_mtp));
    defer dsv41.freeArrays(gpa, back);
    try testing.expectEqual(want.len, back.len);
    for (want, back) |a, b| try testing.expect(try same(a, b));
    // Only the leader prefills; only a follower imports; one rank ships nothing.
    try testing.expectError(error.Dsv41RowCutNotLeader, prefillOnLeader(gpa, .{ .rank = 2, .size = 4, .gather = Group.solo.gather }, lp, ids, 0));
    try testing.expectError(error.Dsv41RowCutLeaderImport, importOnFollower(gpa, four, follow, &rig.follow, 8, ids, want, follow.n_mtp));
    _ = try prefillOnLeader(gpa, Group.solo, lp, ids, 0);
    try testing.expectEqual(@as(usize, 1), rig.shipped);
}

test "dsv41 row cut: a prompt chunk on the leader reaches a follower as the same snapshot, under the leader's id" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const gpa = testing.allocator;
    const lead = try fixture.Fixture.init(gpa, .{ .lo = 20, .hi = 21, .experts = 8, .ctx = 0 });
    defer lead.deinit();
    const follow = try fixture.Fixture.init(gpa, .{ .lo = 20, .hi = 21, .experts = 8, .ctx = 0 });
    defer follow.deinit();
    try expectLeaderShips(gpa, lead.m, follow.m);
}

/// A two-layer whole model with `stages` DSpark stages. The fixture keeps the release's KV source layers past its own
/// two; a parsed config names only layers it has.
fn wholeWithStages(gpa: std.mem.Allocator, stages: u32) !*fixture.Fixture {
    const fx = try fixture.Fixture.initWhole(gpa, .{ .dspark_stages = stages });
    const have = (@as(u64, 1) << @intCast(fx.m.n_layers)) - 1;
    fx.m.cfg.dsv41_kv_sources &= have;
    fx.m.cfg.dsv41_index_sources &= have;
    return fx;
}

test "dsv41 row cut: a leader's prompt chunk ships the DSpark stage windows and main rows too" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const gpa = testing.allocator;
    const lead = try wholeWithStages(gpa, 1);
    defer lead.deinit();
    const follow = try wholeWithStages(gpa, 1);
    defer follow.deinit();
    try testing.expect(lead.m.n_mtp > 0);
    try expectLeaderShips(gpa, lead.m, follow.m);
}

test "dsv41 row cut: a follower without DSpark stages (rank 0 drafts alone) takes the leader's chunk without them" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const gpa = testing.allocator;
    const lead = try wholeWithStages(gpa, 1);
    defer lead.deinit();
    const follow = try wholeWithStages(gpa, 0);
    defer follow.deinit();
    try testing.expect(lead.m.n_mtp > 0 and follow.m.n_mtp == 0);
    try expectLeaderShips(gpa, lead.m, follow.m);
    // A follower holding stages the leader did not send is refused by name.
    try testing.expectError(error.Dsv41RowCutStagesMissing, expectLeaderShips(gpa, follow.m, lead.m));
}

test "dsv41 row cut: a layer with any linear in an outside quant plugin is refused by name" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const gpa = testing.allocator;
    inline for (.{ false, true }) |fused| {
        const fx = try fixture.Fixture.init(gpa, .{ .lo = 20, .hi = 21, .experts = 8, .ctx = 0, .fused = fused });
        defer fx.deinit();
        const ly = fx.layer(20);
        const w: @import("../../sdk/quant/outside.zig").Weight = .{ .t = undefined, .w = undefined };
        var rl = try RowLayer.init(gpa, mlx.gpuStream(), ly, 0, 4);
        rl.deinit();
        var marked: usize = 0;
        for (0..8) |which| {
            var bad = ly.*;
            switch (which) {
                0 => bad.experts.ext = .{ w, w, w },
                1, 2, 3 => bad.shared[which - 1].ext = w,
                4 => bad.wq_b.ext = w,
                5 => bad.wo_a.ext = w,
                6 => bad.wo_b.ext = w,
                else => if (bad.shared_gu) |*gu| {
                    gu.ext = w;
                } else continue,
            }
            marked += 1;
            try testing.expectError(error.Dsv41RowCutOutsideQuant, RowLayer.init(gpa, mlx.gpuStream(), &bad, 0, 4));
        }
        try testing.expectEqual(@as(usize, if (ly.shared_gu != null) 8 else 7), marked);
        if (fused) try testing.expect(ly.shared_gu != null);
    }
}

test "dsv41 row cut: the linear cutter refuses an outside quant weight instead of dropping it" {
    const gpa = testing.allocator;
    var owned: Owned = .{};
    defer owned.deinit(gpa);
    const k: Cutter = .{ .gpa = gpa, .s = undefined, .owned = &owned };
    const l: Lin = .{ .w = none, .ext = .{ .t = undefined, .w = undefined } };
    const c: Cut = .{ .len = 8, .rank = 0, .size = 4 };
    try testing.expectError(error.Dsv41RowCutOutsideQuant, k.lin(l, 0, c));
    try testing.expectError(error.Dsv41RowCutOutsideQuant, k.linCols(l, c));
}

/// A tensor group's sums and gathers in one process: each rank runs against the others' parts from the pass before
/// (its own where they have none yet), so passes fix the collectives one by one; the last pass changes none.
const PassGroup = struct {
    const max = 4;
    var rank: usize = 0;
    var size: usize = 4;
    var gathers: usize = 0;
    var parts: [max]std.ArrayList(A) = @splat(.empty);
    var prev: [max]std.ArrayList(A) = @splat(.empty);

    /// This rank's next collective input kept, and every rank's in rank order.
    fn meet(x: A) ![max]A {
        try mlx.check(mlx.mlx_array_eval(x));
        var kept = mlx.mlx_array_new();
        _ = mlx.mlx_array_set(&kept, x);
        const k = parts[rank].items.len;
        try parts[rank].append(testing.allocator, kept);
        var all: [max]A = undefined;
        for (0..size) |r| all[r] = if (r != rank and k < prev[r].items.len) prev[r].items[k] else x;
        return all;
    }

    fn sum(x: A) anyerror!A {
        const all = try meet(x);
        var acc = mlx.mlx_array_new();
        _ = mlx.mlx_array_set(&acc, all[0]);
        for (all[1..size]) |p| {
            var next = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_add(&next, acc, p, mlx.gpuStream()));
            _ = mlx.mlx_array_free(acc);
            acc = next;
        }
        return acc;
    }

    fn gather(x: A) anyerror!A {
        gathers += 1;
        const all = try meet(x);
        const vec = mlx.mlx_vector_array_new_data(&all, size);
        defer _ = mlx.mlx_vector_array_free(vec);
        var o = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_concatenate_axis(&o, vec, 0, mlx.gpuStream()));
        return o;
    }

    fn clear(l: *std.ArrayList(A)) void {
        for (l.items) |a| _ = mlx.mlx_array_free(a);
        l.clearRetainingCapacity();
    }

    /// Whether this pass's parts are the last pass's, bit for bit.
    fn settled() !bool {
        for (parts[0..size], prev[0..size]) |now, before| {
            if (now.items.len != before.items.len) return false;
            for (now.items, before.items) |a, b| if (!try same(a, b)) return false;
        }
        return true;
    }

    fn arm(n: usize) void {
        size = n;
        gathers = 0;
        dsv41.clusterHooks().tp_sum = &sum;
        dsv41.clusterHooks().tp_gather = &gather;
    }

    fn disarm(gpa: std.mem.Allocator) void {
        dsv41.clusterHooks().tp_sum = null;
        dsv41.clusterHooks().tp_gather = null;
        for (&parts, &prev) |*a, *b| {
            clear(a);
            clear(b);
            a.deinit(gpa);
            b.deinit(gpa);
            a.* = .empty;
            b.* = .empty;
        }
    }
};

test "dsv41 row cut: a tensor mode's decode step through the worker entry runs the row cut and gives the whole model's stream" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const gpa = testing.allocator;
    const base: fixture.Opts = .{ .lo = 20, .hi = 22, .experts = 8, .ctx = 0, .act = .float32 };
    const whole = try fixture.Fixture.init(gpa, base);
    defer whole.deinit();
    var ranks: [4]*fixture.Fixture = undefined;
    var cuts: [4]*RowCut = undefined;
    var made: usize = 0;
    defer for (ranks[0..made], cuts[0..made]) |r, c| {
        c.destroy();
        r.deinit();
    };
    for (&ranks, &cuts, 0..) |*r, *c, i| {
        var o = base;
        o.tp = .{ .rank = @intCast(i), .size = 4 };
        r.* = try fixture.Fixture.init(gpa, o);
        c.* = RowCut.create(gpa, mlx.gpuStream(), whole.m, base.lo, base.hi, @intCast(i), 4) catch |e| {
            r.*.deinit();
            return e;
        };
        r.*.m.row_cut = c.*;
        made += 1;
    }
    PassGroup.arm(4);
    defer PassGroup.disarm(gpa);
    for ([_]usize{ 1, 6 }) |rows| {
        const in = try fixture.Input.of(whole, rows);
        defer in.deinit();
        const want = try fixture.streamOf(gpa, whole, in);
        defer gpa.free(want);
        var got: [4][]f32 = @splat(&.{});
        defer for (got) |g| gpa.free(g);
        var settled = false;
        for (0..12) |_| {
            for (0..4) |r| {
                PassGroup.clear(&PassGroup.prev[r]);
                std.mem.swap(std.ArrayList(A), &PassGroup.prev[r], &PassGroup.parts[r]);
            }
            for (ranks, 0..) |rk, r| {
                PassGroup.rank = r;
                const out = try in.step(gpa, rk.m);
                defer for (out) |a| {
                    _ = mlx.mlx_array_free(a);
                };
                gpa.free(got[r]);
                got[r] = &.{};
                got[r] = try dsv41.hostF32(out[0]);
            }
            if (try PassGroup.settled()) {
                settled = true;
                break;
            }
        }
        try testing.expect(settled);
        for (got) |g| try expectTpClose("tensor mode step", rows, want, g);
        for (PassGroup.parts[0..4]) |*l| PassGroup.clear(l);
        for (PassGroup.prev[0..4]) |*l| PassGroup.clear(l);
    }
    // Two layers, two gathers each, on every rank in every pass.
    for (cuts) |c| try testing.expect(c.calls > 0);
    try testing.expect(PassGroup.gathers >= 2 * 2 * 4);
}

test "dsv41 row cut: the parity bar fails a result that is not finite" {
    const want = [_]f32{ 1, -2, 3, 0.5 };
    const nan = std.math.nan(f32);
    try testing.expectError(error.NonFinite, expectTpClose("nan", 1, &want, &[_]f32{ nan, nan, nan, nan }));
    try testing.expectError(error.NonFinite, expectTpClose("inf", 1, &want, &[_]f32{ 1, -2, std.math.inf(f32), 0.5 }));
    try expectTpClose("same", 1, &want, &want);
}

test "dsv41 row cut: a head's vocabulary rows that do not divide by the ranks all land on one, and a gathered cut refuses them" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const gpa = testing.allocator;
    var vals: [17 * 2]f32 = undefined;
    for (&vals, 0..) |*v, i| v.* = @floatFromInt(i);
    const w = mlx.mlx_array_new_data(&vals, &[_]c_int{ 17, 2 }, 2, .float32);
    defer _ = mlx.mlx_array_free(w);
    var seen: std.ArrayList(f32) = .empty;
    defer seen.deinit(gpa);
    for (0..4) |r| {
        var rh = try RowHead.init(gpa, mlx.gpuStream(), .{ .w = w }, w, @intCast(r), 4);
        defer rh.deinit();
        const got = try host(gpa, rh.head.w);
        defer gpa.free(got);
        try seen.appendSlice(gpa, got);
    }
    try testing.expectEqualSlices(f32, &vals, seen.items);
    var owned: Owned = .{};
    defer owned.deinit(gpa);
    const even: Cutter = .{ .gpa = gpa, .s = mlx.gpuStream(), .owned = &owned, .even = true };
    try testing.expectError(error.Dsv41RowCutUneven, even.arr(w, 0, .{ .len = 17, .rank = 0, .size = 4 }));
}
