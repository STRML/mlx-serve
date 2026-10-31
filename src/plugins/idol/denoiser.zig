//! Idol's denoiser (`LatentSeqMMFlowModel`): a rectified-flow transformer over the object latent (8192 tokens of 16)
//! and the input camera (1 token of 5), conditioned on the image tokens. The latent, image and camera tokens are
//! refined on their own (two blocks each), then run through the joint blocks with every token's rotary position
//! predicted from its own features (`RePo3D`). Blocks in the reference's f16 (Linear only), everything else f32.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const scope_mod = @import("../../mlx/scope.zig");
const L = @import("layers.zig");
const glue = @import("glue.zig");

const A = L.A;
const Scope = L.Scope;
const Linear = L.Linear;
const Weights = L.Weights;
const NONE = L.NONE;

pub const Cfg = struct {
    q_token_length: u32 = 8192,
    in_channels: u32 = 16,
    cam_channels: u32 = 5,
    model_channels: u32 = 1024,
    cond_channels: u32 = 1280,
    cond2_channels: u32 = 128,
    num_blocks: u32 = 24,
    num_refiner_blocks: u32 = 2,
    num_heads: u32 = 16,
    mlp_ratio: u32 = 4,
};

const Block = struct {
    shift_table: ?A,
    norm1_w: ?A,
    norm1_b: ?A,
    norm2_w: ?A,
    norm2_b: ?A,
    qkv: Linear,
    out: Linear,
    q_gamma: A,
    k_gamma: A,
    mlp: L.Mlp,

    fn load(w: *const Weights, comptime fmt: []const u8, i: usize, c: u32, hidden: u32, modulated: bool) !Block {
        var pbuf: [64]u8 = undefined;
        const p = try std.fmt.bufPrint(&pbuf, fmt, .{i});
        return .{
            .shift_table = if (modulated) try scope_mod.getW(w, "{s}.shift_table", .{p}) else null,
            .norm1_w = if (modulated) null else try scope_mod.getW(w, "{s}.norm1.weight", .{p}),
            .norm1_b = if (modulated) null else try scope_mod.getW(w, "{s}.norm1.bias", .{p}),
            .norm2_w = if (modulated) null else try scope_mod.getW(w, "{s}.norm2.weight", .{p}),
            .norm2_b = if (modulated) null else try scope_mod.getW(w, "{s}.norm2.bias", .{p}),
            .qkv = try scope_mod.linear(w, "{s}.attn.qkv", .{p}, c),
            .out = try scope_mod.linear(w, "{s}.attn.out", .{p}, c),
            .q_gamma = try scope_mod.getW(w, "{s}.attn.q_norm.gamma", .{p}),
            .k_gamma = try scope_mod.getW(w, "{s}.attn.k_norm.gamma", .{p}),
            .mlp = try L.Mlp.load(w, "{s}.mlp", .{p}, c, hidden),
        };
    }
};

/// `RePo3DRotaryEmbedding`: each token's features predict a 3D offset per head, turned into rotary angles.
const Repo = struct {
    norm_w: A,
    norm_b: A,
    gate: Linear,
    content: Linear,
    final: Linear,
    freqs: [3]A,

    fn load(w: *const Weights, comptime name: []const u8, i: usize, c: u32) !Repo {
        const hidden: u32 = c / 8;
        return .{
            .norm_w = try scope_mod.getW(w, name ++ ".{d}.norm.weight", .{i}),
            .norm_b = try scope_mod.getW(w, name ++ ".{d}.norm.bias", .{i}),
            .gate = try scope_mod.linear(w, name ++ ".{d}.gate_map", .{i}, c),
            .content = try scope_mod.linear(w, name ++ ".{d}.content_map", .{i}, c),
            .final = try scope_mod.linear(w, name ++ ".{d}.final_map", .{i}, hidden),
            .freqs = .{
                try scope_mod.getW(w, name ++ ".{d}.freqs_0", .{i}),
                try scope_mod.getW(w, name ++ ".{d}.freqs_1", .{i}),
                try scope_mod.getW(w, name ++ ".{d}.freqs_2", .{i}),
            },
        };
    }

    /// cos and sin `[1, L, H, 32]` f32 for `h` `[1, L, C]`.
    fn angles(self: *const Repo, sc: *Scope, h: A, heads: c_int) !struct { cos: A, sin: A } {
        const l = mlx.getShape(h)[1];
        const n = try sc.layerNorm(try L.asDtype(sc, h, .float32), self.norm_w, self.norm_b, 1e-5);
        const act = try sc.mul(try sc.silu(try self.gate.forward(sc, n)), try self.content.forward(sc, n));
        const delta = try sc.reshape(try self.final.forward(sc, act), &.{ 1, l, heads, 3 });
        var parts: [3]A = undefined;
        for (&parts, 0..) |*p, axis| {
            const d = try sc.slice(delta, 3, @intCast(axis), @intCast(axis + 1));
            p.* = try sc.mulS(try sc.mul(d, self.freqs[axis]), std.math.pi);
        }
        const ang = try sc.concat(&parts, 3);
        return .{ .cos = try sc.cos(ang), .sin = try sc.sin(ang) };
    }
};

/// `x` `[1, L, H, 64]` rotated pairwise (even, odd) by `cos`/`sin` `[1, L, H, 32]`, f32 math, x's dtype out.
pub fn applyRope(sc: *Scope, x: A, cos: A, sin: A) !A {
    const dt = mlx.mlx_array_dtype(x);
    const sh = mlx.getShape(x);
    const pair = [_]c_int{ sh[0], sh[1], sh[2], @divExact(sh[3], 2), 2 };
    const p = try sc.reshape(try L.asDtype(sc, x, .float32), &pair);
    const x0 = try sc.slice(p, 4, 0, 1);
    const x1 = try sc.slice(p, 4, 1, 2);
    const c = try sc.reshape(cos, &.{ sh[0], sh[1], sh[2], pair[3], 1 });
    const s = try sc.reshape(sin, &.{ sh[0], sh[1], sh[2], pair[3], 1 });
    const o0 = try sc.sub(try sc.mul(x0, c), try sc.mul(x1, s));
    const o1 = try sc.add(try sc.mul(x0, s), try sc.mul(x1, c));
    return L.asDtype(sc, try sc.reshape(try sc.concat(&.{ o0, o1 }, 4), sh), dt);
}

pub const Denoiser = struct {
    gpa: std.mem.Allocator,
    cfg: Cfg,
    act: mlx.mlx_dtype,
    input: Linear,
    cond1: Linear,
    cond2: Linear,
    t_fc0: Linear,
    t_fc2: Linear,
    ada: Linear,
    out: Linear,
    cam_out: Linear,
    cam0: Linear,
    cam1: Linear,
    shift_table: A,
    pos: A,
    noise_blocks: []Block,
    context_blocks: []Block,
    blocks: []Block,
    noise_repo: []Repo,
    context_repo: []Repo,
    repo: []Repo,
    ones: A,
    owned: std.ArrayList(A) = .empty,

    /// `w` is the denoiser's weight map (the caller keeps it alive); `act` the blocks' dtype (f16 as published).
    pub fn load(gpa: std.mem.Allocator, w: *const Weights, cfg: Cfg, act: mlx.mlx_dtype) !Denoiser {
        const c = cfg.model_channels;
        const hidden = c * cfg.mlp_ratio;
        var d: Denoiser = .{
            .gpa = gpa, .cfg = cfg, .act = act,
            .input = try scope_mod.linear(w, "input_layer", .{}, cfg.in_channels),
            .cond1 = try scope_mod.linear(w, "cond_embedder", .{}, cfg.cond_channels),
            .cond2 = try scope_mod.linear(w, "cond_embedder2", .{}, cfg.cond2_channels),
            .t_fc0 = try scope_mod.linear(w, "t_embedder.mlp.0", .{}, 256),
            .t_fc2 = try scope_mod.linear(w, "t_embedder.mlp.2", .{}, c),
            .ada = try scope_mod.linear(w, "adaLN_modulation.1", .{}, c),
            .out = try scope_mod.linear(w, "out_layer", .{}, c),
            .cam_out = try scope_mod.linear(w, "cam_out_layer", .{}, c),
            .cam0 = try scope_mod.linear(w, "cam_refiner.mlp.0", .{}, cfg.cam_channels),
            .cam1 = try scope_mod.linear(w, "cam_refiner.mlp.2", .{}, c),
            .shift_table = try scope_mod.getW(w, "shift_table", .{}),
            .pos = undefined,
            .noise_blocks = &.{}, .context_blocks = &.{}, .blocks = &.{},
            .noise_repo = &.{}, .context_repo = &.{}, .repo = &.{},
            .ones = undefined,
        };
        errdefer d.deinit();
        const n = cfg.num_blocks;
        const r = cfg.num_refiner_blocks;
        d.noise_blocks = try gpa.alloc(Block, r);
        d.context_blocks = try gpa.alloc(Block, r);
        d.blocks = try gpa.alloc(Block, n);
        d.noise_repo = try gpa.alloc(Repo, r);
        d.context_repo = try gpa.alloc(Repo, r);
        d.repo = try gpa.alloc(Repo, n);
        for (d.noise_blocks, 0..) |*b, i| b.* = try Block.load(w, "noise_refiner.{d}", i, c, hidden, true);
        for (d.context_blocks, 0..) |*b, i| b.* = try Block.load(w, "context_refiner.{d}", i, c, hidden, false);
        for (d.blocks, 0..) |*b, i| b.* = try Block.load(w, "blocks.{d}", i, c, hidden, true);
        for (d.noise_repo, 0..) |*b, i| b.* = try Repo.load(w, "noise_repo_layers", i, c);
        for (d.context_repo, 0..) |*b, i| b.* = try Repo.load(w, "context_repo_layers", i, c);
        for (d.repo, 0..) |*b, i| b.* = try Repo.load(w, "repo_layers", i, c);

        // The latent tokens' fixed positions, embedded once.
        const pe = try scope_mod.getW(w, "pos_pe", .{});
        try mlx.check(mlx.mlx_array_eval(pe));
        const host = try gpa.alloc(f32, cfg.q_token_length * 3);
        defer gpa.free(host);
        const src = mlx.mlx_array_data_float32(pe) orelse return error.NoData;
        @memcpy(host, src[0..host.len]);
        const emb = try L.absolutePosition(gpa, host, c);
        defer gpa.free(emb);
        d.pos = mlx.mlx_array_new_data(emb.ptr, &[_]c_int{ 1, @intCast(cfg.q_token_length), @intCast(c) }, 3, .float32);
        try d.owned.append(gpa, d.pos);
        const ones = try gpa.alloc(f32, L.HEAD_DIM);
        defer gpa.free(ones);
        @memset(ones, 1);
        d.ones = mlx.mlx_array_new_data(ones.ptr, &[_]c_int{L.HEAD_DIM}, 1, .float32);
        try d.owned.append(gpa, d.ones);
        return d;
    }

    pub fn deinit(self: *Denoiser) void {
        for (self.owned.items) |a| _ = mlx.mlx_array_free(a);
        self.owned.deinit(self.gpa);
        self.gpa.free(self.noise_blocks);
        self.gpa.free(self.context_blocks);
        self.gpa.free(self.blocks);
        self.gpa.free(self.noise_repo);
        self.gpa.free(self.context_repo);
        self.gpa.free(self.repo);
    }

    fn heads(self: *const Denoiser) c_int {
        return @intCast(self.cfg.num_heads);
    }

    /// Attention over the fused projection `y`: q and k rotated, then RMS-normalized per head (fused into one pass on
    /// the GPU), heads-major. Returns `[1, L, C]`.
    fn attention(self: *const Denoiser, sc: *Scope, b: *const Block, y: A, rope: anytype) !A {
        const h = self.heads();
        const d: c_int = L.HEAD_DIM;
        if (try glue.qkvPrep(y, b.q_gamma, b.k_gamma, rope.cos, rope.sin, L.HEAD_NORM_EPS, h, sc.s)) |f| {
            return L.attendHeads(sc, try sc.keep(f.q), try sc.keep(f.k), try sc.keep(f.v));
        }
        const q = try L.headNorm(sc, try applyRope(sc, try L.part(sc, y, 3, 0, h, d), rope.cos, rope.sin), b.q_gamma, self.ones);
        const k = try L.headNorm(sc, try applyRope(sc, try L.part(sc, y, 3, 1, h, d), rope.cos, rope.sin), b.k_gamma, self.ones);
        return L.attend(sc, q, k, try L.part(sc, y, 3, 2, h, d));
    }

    fn block(self: *const Denoiser, sc: *Scope, b: *const Block, x: A, mod: ?A, rope: anytype) !A {
        const c = self.cfg.model_channels;
        if (mod == null) {
            const n1 = try L.asDtype(sc, try sc.layerNorm(try L.asDtype(sc, x, .float32), b.norm1_w.?, b.norm1_b.?, L.LN_EPS), self.act);
            const x1 = try sc.add(x, try b.out.forward(sc, try self.attention(sc, b, try b.qkv.forward(sc, n1), rope)));
            const n2 = try L.asDtype(sc, try sc.layerNorm(try L.asDtype(sc, x1, .float32), b.norm2_w.?, b.norm2_b.?, L.LN_EPS), self.act);
            return sc.add(x1, try b.mlp.forward(sc, n2));
        }
        const m = try sc.add(mod.?, try L.asLike(sc, b.shift_table.?, mod.?));
        const shift_msa, const scale_msa, const gate_msa, const shift_mlp, const scale_mlp, const gate_mlp = try L.chunks6(sc, m, c);
        const n1 = try L.modulate(sc, try sc.layerNorm(x, NONE, NONE, L.LN_EPS), shift_msa, scale_msa);
        const att = try b.out.forward(sc, try self.attention(sc, b, try b.qkv.forward(sc, n1), rope));
        const x1 = try sc.add(x, try sc.mul(att, gate_msa));
        const n2 = try L.modulate(sc, try sc.layerNorm(x1, NONE, NONE, L.LN_EPS), shift_mlp, scale_mlp);
        return sc.add(x1, try sc.mul(try b.mlp.forward(sc, n2), gate_mlp));
    }

    /// The image tokens refined once (they depend on neither the noisy latent nor the time): `feature1`
    /// `[1, Lc, 1280]` and `feature2` `[1, Lc, 128]` f32 in, `[1, Lc, C]` in the block dtype out (caller frees).
    pub fn context(self: *const Denoiser, parent: *Scope, f1: A, f2: A) !A {
        var cur: ?A = null;
        errdefer if (cur) |x| {
            _ = mlx.mlx_array_free(x);
        };
        {
            var sc = Scope.init(parent.a, parent.s);
            defer sc.deinit();
            const e = try sc.add(try self.cond1.forward(&sc, f1), try self.cond2.forward(&sc, f2));
            const y = try L.asDtype(&sc, e, self.act);
            try mlx.check(mlx.mlx_array_eval(y));
            cur = sc.out(y);
        }
        for (self.context_blocks, self.context_repo) |*b, *r| {
            var sc = Scope.init(parent.a, parent.s);
            defer sc.deinit();
            const rope = try r.angles(&sc, cur.?, self.heads());
            const y = try self.block(&sc, b, cur.?, null, rope);
            try mlx.check(mlx.mlx_array_eval(y));
            _ = mlx.mlx_array_free(cur.?);
            cur = sc.out(y);
        }
        return cur.?;
    }

    pub const Velocity = struct { latent: A, camera: A };

    /// One velocity prediction at model time `t` (1000 x flow time). `z` `[1, 8192, 16]`, `cam` `[1, 1, 5]` f32,
    /// `ctx` from `context`. Both outputs belong to `sc`.
    pub fn velocity(self: *const Denoiser, sc: *Scope, z: A, cam: A, t: f32, ctx: A) !Velocity {
        const c: c_int = @intCast(self.cfg.model_channels);
        const nz: c_int = @intCast(self.cfg.q_token_length);
        var tbuf: [256]f32 = undefined;
        L.sinusoid(&tbuf, t, 10000, false);
        const e0 = try sc.fromF32(&tbuf, &.{ 1, 256 });
        const t_emb = try self.t_fc2.forward(sc, try sc.silu(try self.t_fc0.forward(sc, e0)));
        const t_mod = try L.asDtype(sc, try self.ada.forward(sc, try sc.silu(t_emb)), self.act);

        // The reference casts both terms to the block dtype before adding.
        const h_in = try L.asDtype(sc, try self.input.forward(sc, z), self.act);
        var hx = try sc.add(h_in, try L.asDtype(sc, self.pos, self.act));
        for (self.noise_blocks, self.noise_repo) |*b, *r| {
            var bs = Scope.init(sc.a, sc.s);
            defer bs.deinit();
            const rope = try r.angles(&bs, hx, self.heads());
            const y = try self.block(&bs, b, hx, t_mod, rope);
            try mlx.check(mlx.mlx_array_eval(y));
            hx = try sc.keep(bs.out(y));
        }
        const hc = try L.asDtype(sc, try self.cam1.forward(sc, try L.geluTanh(sc, try self.cam0.forward(sc, cam))), self.act);

        var h = try sc.concat(&.{ hx, ctx, hc }, 1);
        try mlx.check(mlx.mlx_array_eval(h));
        for (self.blocks, self.repo) |*b, *r| {
            var bs = Scope.init(sc.a, sc.s);
            defer bs.deinit();
            const rope = try r.angles(&bs, h, self.heads());
            const y = try self.block(&bs, b, h, t_mod, rope);
            try mlx.check(mlx.mlx_array_eval(y));
            h = try sc.keep(bs.out(y));
        }
        const total = mlx.getShape(h)[1];
        const lat = try sc.layerNorm(try L.asDtype(sc, try sc.slice(h, 1, 0, nz), .float32), NONE, NONE, L.FINAL_LN_EPS);
        const cm = try sc.layerNorm(try L.asDtype(sc, try sc.slice(h, 1, total - 1, total), .float32), NONE, NONE, L.FINAL_LN_EPS);
        const te = try sc.reshape(t_emb, &.{ 1, 1, c });
        const st = try sc.add(self.shift_table, te);
        const shift = try sc.slice(st, 1, 0, 1);
        const scale = try sc.slice(st, 1, 1, 2);
        const one = try sc.scalar(1);
        const lat_m = try sc.add(try sc.mul(lat, try sc.add(scale, one)), shift);
        const cam_m = try sc.add(try sc.mul(cm, try sc.add(scale, one)), shift);
        return .{ .latent = try self.out.forward(sc, lat_m), .camera = try self.cam_out.forward(sc, cam_m) };
    }
};
