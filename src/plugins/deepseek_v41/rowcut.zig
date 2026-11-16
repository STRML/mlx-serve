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

/// A cut of one axis: part `rank` of `size` of the rows from `base`.
const Cut = struct {
    base: usize = 0,
    len: usize,
    rank: u32,
    size: u32,

    fn lo(c: Cut) usize {
        return c.base + c.len / c.size * c.rank;
    }
    fn hi(c: Cut) usize {
        return c.lo() + c.len / c.size;
    }
};

const Cutter = struct {
    gpa: std.mem.Allocator,
    s: S,
    owned: *Owned,

    fn arr(k: Cutter, a: A, axis: usize, c: Cut) !A {
        return k.owned.put(k.gpa, try view(k.s, a, axis, c.lo(), c.hi()));
    }
    fn bank(k: Cutter, b: Bank, c: Cut) !Bank {
        return .{ .w = try k.arr(b.w, 1, c), .s = try k.arr(b.s, 1, c), .b = try k.arr(b.b, 1, c), .qp = b.qp };
    }
    fn lin(k: Cutter, l: Lin, axis: usize, c: Cut) !Lin {
        return .{ .w = try k.arr(l.w, axis, c), .s = try k.arr(l.s, axis, c), .b = try k.arr(l.b, axis, c), .qp = l.qp };
    }
    /// Columns `c` of a linear in a buffer of its own: a column cut of a row-major weight is no view. A quantized
    /// weight's columns cut whole groups.
    fn linCols(k: Cutter, l: Lin, c: Cut) !Lin {
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
        if (e.ext != null or ly.shared[2].ext != null or ly.wo_a.ext != null) return error.Dsv41RowCutOutsideQuant;
        if (e.down.qp.mode == .gguf) return error.Dsv41RowCutGguf;
        var rl: RowLayer = .{ .gpa = gpa, .experts = undefined, .shared = undefined, .wq_b = undefined, .wo_a = undefined, .sink = none, .wo_b = undefined, .rank = rank, .size = size };
        if (e.gu) |gu| if (ly.shared_gu) |sgu| {
            rl.fused = .{ .gu = gu, .sgu = sgu, .down = e.down, .sdn = ly.shared[2] };
        };
        errdefer rl.owned.deinit(gpa);
        const k: Cutter = .{ .gpa = gpa, .s = s, .owned = &rl.owned };
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
/// its id: `dsv41.tpLoad`'s path through the cluster's `tp_load` hook in serving, a recorder in a test.
pub const LeaderPrefill = struct {
    ctx: ?*anyopaque = null,
    whole: *const dsv41.Dsv41Model,
    run: *const fn (ctx: ?*anyopaque, ids: []const u32, p0: usize) anyerror!*const dsv41.Snap,
    ship: *const fn (ctx: ?*anyopaque, id: u64, ids: []const u32, tensors: []const A) anyerror!void,
};

/// Rank 0's half of a prompt chunk (`ids` from position `p0`): the chunk on the TP1 model, then its state to the other
/// ranks. Returns the snapshot `run` made.
pub fn prefillOnLeader(gpa: std.mem.Allocator, g: Group, lp: LeaderPrefill, ids: []const u32, p0: usize) !*const dsv41.Snap {
    if (g.rank != 0) return error.Dsv41RowCutNotLeader;
    const sn = try lp.run(lp.ctx, ids, p0);
    if (g.size == 1) return sn;
    const xs = try dsv41.tpLoadTensors(lp.whole, gpa, sn);
    defer dsv41.freeArrays(gpa, xs);
    try lp.ship(lp.ctx, sn.tp_id, sn.ids, xs);
    return sn;
}

/// A follower's half: rank 0's snapshot `id` rebuilt from the tensors it shipped, into `snaps` under that id (the next
/// TP4 turn restores it). The KV is whole on every rank, so the follower's TP1 model takes it as is.
pub fn importOnFollower(gpa: std.mem.Allocator, g: Group, whole: *const dsv41.Dsv41Model, snaps: *dsv41.Snaps, id: u64, ids: []const u32, tensors: []const A) !void {
    if (g.rank == 0) return error.Dsv41RowCutLeaderImport;
    try dsv41.tpWorkerLoad(whole, gpa, snaps, id, ids, tensors);
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
        peak = @max(peak, @abs(u));
        err = @max(err, @abs(u - v));
    }
    try testing.expect(peak > 0);
    const tol: f32 = if (device.naxAvailable()) 1e-2 else 1e-3;
    std.debug.print("[rowcut] {s} R={d}: max-abs {e:.2} ({e:.2} of peak, bar {e:.0}), rel rms {e:.2}\n", .{ label, rows, err, err / peak, tol, decode.relRms(got, want) });
    try testing.expect(err <= tol * peak);
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
        try dsv41.tpWorkerLoad(self.leader, self.gpa, &self.lead, 7, ids, self.disk);
        const sn = self.lead.getPtr(7).?;
        sn.tp_id = 7;
        return sn;
    }

    fn ship(ctx: ?*anyopaque, id: u64, ids: []const u32, tensors: []const A) anyerror!void {
        const self: *PrefillRig = @ptrCast(@alignCast(ctx.?));
        self.shipped += 1;
        try importOnFollower(self.gpa, .{ .rank = 1, .size = 4, .gather = Group.solo.gather }, self.follower, &self.follow, id, ids, tensors);
    }
};

test "dsv41 row cut: a prompt chunk on the leader reaches a follower as the same snapshot, under the leader's id" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const gpa = testing.allocator;
    const lead = try fixture.Fixture.init(gpa, .{ .lo = 20, .hi = 21, .experts = 8, .ctx = 0 });
    defer lead.deinit();
    const follow = try fixture.Fixture.init(gpa, .{ .lo = 20, .hi = 21, .experts = 8, .ctx = 0 });
    defer follow.deinit();
    const n = 301;
    const plan = try dsv41.tpLoadPlan(lead.m, n);
    var sc = Scope.init(gpa, lead.m.s);
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
    var rig: PrefillRig = .{ .gpa = gpa, .leader = lead.m, .follower = follow.m, .lead = .init(gpa), .follow = .init(gpa), .disk = disk };
    defer dsv41.dropSnaps(&rig.lead, gpa);
    defer dsv41.dropSnaps(&rig.follow, gpa);
    const lp: LeaderPrefill = .{ .ctx = &rig, .whole = lead.m, .run = PrefillRig.run, .ship = PrefillRig.ship };
    const four: Group = .{ .rank = 0, .size = 4, .gather = Group.solo.gather };
    _ = try prefillOnLeader(gpa, four, lp, ids, 0);
    try testing.expectEqual(@as(usize, 1), rig.shipped);
    const got = rig.follow.getPtr(7) orelse return error.NotShipped;
    try testing.expectEqual(@as(usize, n), got.n);
    const want = try dsv41.tpLoadTensors(lead.m, gpa, rig.lead.getPtr(7).?);
    defer dsv41.freeArrays(gpa, want);
    const back = try dsv41.tpLoadTensors(follow.m, gpa, got);
    defer dsv41.freeArrays(gpa, back);
    for (want, back) |a, b| try testing.expect(try same(a, b));
    // Only the leader prefills; only a follower imports; one rank ships nothing.
    try testing.expectError(error.Dsv41RowCutNotLeader, prefillOnLeader(gpa, .{ .rank = 2, .size = 4, .gather = Group.solo.gather }, lp, ids, 0));
    try testing.expectError(error.Dsv41RowCutLeaderImport, importOnFollower(gpa, four, follow.m, &rig.follow, 8, ids, want));
    _ = try prefillOnLeader(gpa, Group.solo, lp, ids, 0);
    try testing.expectEqual(@as(usize, 1), rig.shipped);
}
