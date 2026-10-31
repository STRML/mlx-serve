//! Idol's kernel contract: the denoiser block's fused q/k RoPE + per-head RMS norm + head-major layout (bit-equal to
//! the chain it replaces) and the fc1 bias add + tanh GELU. Listed in `kernel_contract/registry.zig`.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const kc = @import("../../sdk/testing/kernel_contract/root.zig");
const L = @import("layers.zig");
const den = @import("denoiser.zig");
const glue = @import("glue.zig");

const spec = kc.spec;
const metal = kc.metal;
const host = kc.host;
const Shape = spec.Shape;

const HD = 64;
const PAIRS = HD / 2;
const EPS: f32 = L.HEAD_NORM_EPS;

// ── q/k RoPE + RMS norm + head-major layout ──

const qkv_f16: metal.Launch = .{ .name = "msv_idol_qkv_prep", .inputs = &.{ "Y", "COS", "SIN", "QG", "KG", "eps" }, .outputs = &.{ "OQ", "OK", "OV" }, .source = glue.QKV_SOURCE, .tg_bytes = 0, .m1 = 1024, .m2 = 1024 };

/// y [rows, 3*H*64] -> q, k, v [1, H, rows, 64]: `rows` is the tokens, `n` the heads, `k` the head width (the norm's sum).
const QkvPrep = struct {
    rig: metal.Rig,
    launch: *const metal.Launch,
    dt: mlx.mlx_dtype,
    shape: Shape = undefined,
    arena: std.heap.ArenaAllocator,
    y: []f64 = &.{},
    cos: []f64 = &.{},
    sin: []f64 = &.{},
    qg: []f64 = &.{},
    kg: []f64 = &.{},
    args: [6]mlx.mlx_array = undefined,

    fn init(a: std.mem.Allocator, launch: *const metal.Launch, dt: mlx.mlx_dtype) QkvPrep {
        return .{ .rig = metal.Rig.init(a), .launch = launch, .dt = dt, .arena = .init(a) };
    }

    pub fn deinit(self: *QkvPrep) void {
        self.rig.deinit();
        self.arena.deinit();
    }

    pub fn geometry(self: *const QkvPrep, shape: Shape) ?metal.Geometry {
        if (shape.k != HD or shape.n == 0) return null;
        const h: c_int = @intCast(shape.n);
        const seq: c_int = @intCast(shape.rows);
        var g = metal.Geometry{ .grid = .{ 256 * @divTrunc(3 * h + 7, 8), seq, 1 }, .tg = .{ 256, 1, 1 } };
        for (0..3) |_| g.out(&.{ 1, h, seq, HD }, self.dt);
        g.dtype("T", self.dt);
        g.int("H", h);
        return g;
    }

    pub fn prepare(self: *QkvPrep, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        self.shape = shape;
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const width = 3 * shape.n * HD;
        const table = shape.rows * shape.n * PAIRS;
        self.y = try host.uniform(a, rng, shape.rows * width, -3, 3, self.dt);
        self.cos = try a.alloc(f64, table);
        self.sin = try a.alloc(f64, table);
        for (self.cos, self.sin) |*c, *s| {
            const ang = 6.3 * rng.float(f64);
            c.* = spec.roundTo(.f32, @cos(ang));
            s.* = spec.roundTo(.f32, @sin(ang));
        }
        self.qg = try host.uniform(a, rng, shape.n * HD, 0.5, 1.5, .float32);
        self.kg = try host.uniform(a, rng, shape.n * HD, 0.5, 1.5, .float32);
        switch (fill) {
            .random => {},
            .nonfinite => host.plant(self.y, shape.rows, width),
            .stress => {
                // Gate 4: integer rows, sums of squares exact in fp32; unit gammas and the identity rotation, so the
                // norm is the only rounding left.
                for (self.y) |*v| v.* = @floatFromInt(@as(i32, @intCast(rng.uintLessThan(u32, 65))) - 32);
                @memset(self.qg, 1);
                @memset(self.kg, 1);
                @memset(self.cos, 1);
                @memset(self.sin, 0);
            },
        }
        const rig = &self.rig;
        rig.reset();
        const seq: c_int = @intCast(shape.rows);
        const h: c_int = @intCast(shape.n);
        self.args = .{
            try rig.input(self.dt, &.{ seq, @intCast(width) }, self.y),
            try rig.input(.float32, &.{ seq, h, PAIRS }, self.cos),
            try rig.input(.float32, &.{ seq, h, PAIRS }, self.sin),
            try rig.input(.float32, &.{ h, HD }, self.qg),
            try rig.input(.float32, &.{ h, HD }, self.kg),
            try rig.scalar(EPS),
        };
        return true;
    }

    pub fn run(self: *QkvPrep) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, &self.args);
    }

    pub fn count(self: *const QkvPrep) usize {
        return 3 * self.shape.rows * self.shape.n * HD;
    }

    /// Every output, the norm's sum of squares run in `acc` and `order` (null: float64).
    fn outputsIn(self: *const QkvPrep, want: []f64, mag: ?[]f64, acc: ?spec.Accum, order: spec.Order) void {
        const e = host.elemOf(self.dt);
        const seq = self.shape.rows;
        const heads = self.shape.n;
        var rot: [HD]f64 = undefined;
        var sq: [HD]f64 = undefined;
        for (0..seq) |r| for (0..3) |which| for (0..heads) |h| {
            const src = self.y[r * 3 * heads * HD + (which * heads + h) * HD ..][0..HD];
            const at = ((which * heads + h) * seq + r) * HD;
            if (which == 2) {
                @memcpy(want[at..][0..HD], src);
                if (mag) |m| for (m[at..][0..HD], src) |*mo, x| {
                    mo.* = @abs(x);
                };
                continue;
            }
            const tab = (r * heads + h) * PAIRS;
            for (0..PAIRS) |i| {
                const c = self.cos[tab + i];
                const s = self.sin[tab + i];
                rot[2 * i] = spec.roundTo(e, src[2 * i] * c - src[2 * i + 1] * s);
                rot[2 * i + 1] = spec.roundTo(e, src[2 * i] * s + src[2 * i + 1] * c);
            }
            for (&sq, rot) |*q, x| q.* = x * x;
            const ss = if (acc) |ac| spec.sumIn(ac, order, &sq) else blk: {
                var t: f64 = 0;
                for (sq) |q| t += q;
                break :blk t;
            };
            const inv = 1 / @sqrt(ss / HD + EPS);
            const g = (if (which == 0) self.qg else self.kg)[h * HD ..][0..HD];
            for (want[at..][0..HD], rot, g) |*o, x, gi| o.* = x * inv * gi;
            if (mag) |m| for (m[at..][0..HD], want[at..][0..HD]) |*mo, o| {
                mo.* = @abs(o);
            };
        };
    }

    pub fn reference(self: *const QkvPrep, want: []f64, mag: []f64) !void {
        self.outputsIn(want, mag, null, .sequential);
    }

    pub fn simulate(self: *const QkvPrep, acc: spec.Accum, order: spec.Order, out: []f64) !void {
        self.outputsIn(out, null, acc, order);
    }

    /// The denoiser's chain: split, `applyRope`, `headNorm`, head-major transpose.
    pub fn replaced(self: *QkvPrep) ![]const u8 {
        var sc = L.Scope.init(self.rig.a, self.rig.stream());
        defer sc.deinit();
        const seq: c_int = @intCast(self.shape.rows);
        const h: c_int = @intCast(self.shape.n);
        const x = self.args;
        const y = try sc.reshape(x[0], &.{ 1, seq, 3 * h * HD });
        const cos = try sc.reshape(x[1], &.{ 1, seq, h, PAIRS });
        const sin = try sc.reshape(x[2], &.{ 1, seq, h, PAIRS });
        const unit: [HD]f32 = @splat(1);
        const ones = try sc.fromF32(&unit, &.{HD});
        var outs: [3]mlx.mlx_array = undefined;
        for (&outs, 0..) |*o, i| {
            const part = try L.part(&sc, y, 3, @intCast(i), h, HD);
            const ready = if (i == 2) part else try L.headNorm(&sc, try den.applyRope(&sc, part, cos, sin), x[3 + i], ones);
            o.* = try sc.transpose(ready, &.{ 0, 2, 1, 3 });
        }
        return self.rig.bytesOf(&outs);
    }

    /// The shipped wrapper on the same inputs.
    fn wrapper(self: *QkvPrep) ![]const u8 {
        const x = self.args;
        const got = (try glue.qkvPrep(x[0], x[3], x[4], x[1], x[2], EPS, @intCast(self.shape.n), self.rig.stream())) orelse return error.Declined;
        defer for ([_]mlx.mlx_array{ got.q, got.k, got.v }) |o| {
            _ = mlx.mlx_array_free(o);
        };
        return self.rig.bytesOf(&.{ got.q, got.k, got.v });
    }
};

// ── bias add + tanh GELU ──

const gelu_f16: metal.Launch = .{ .name = "msv_idol_bias_gelu", .inputs = &.{ "Y", "B" }, .outputs = &.{"O"}, .source = glue.GELU_SOURCE, .tg_bytes = 0, .m1 = 1024, .m2 = 1024 };

/// y [rows, N] + b [N] -> gelu_tanh: `k` is N.
const BiasGelu = struct {
    rig: metal.Rig,
    launch: *const metal.Launch,
    dt: mlx.mlx_dtype,
    shape: Shape = undefined,
    arena: std.heap.ArenaAllocator,
    y: []f64 = &.{},
    b: []f64 = &.{},
    args: [2]mlx.mlx_array = undefined,

    fn init(a: std.mem.Allocator, launch: *const metal.Launch, dt: mlx.mlx_dtype) BiasGelu {
        return .{ .rig = metal.Rig.init(a), .launch = launch, .dt = dt, .arena = .init(a) };
    }

    pub fn deinit(self: *BiasGelu) void {
        self.rig.deinit();
        self.arena.deinit();
    }

    pub fn geometry(self: *const BiasGelu, shape: Shape) ?metal.Geometry {
        const rows: c_int = @intCast(shape.rows);
        const n: c_int = @intCast(shape.k);
        if (n == 0 or @rem(n, 4) != 0) return null;
        var g = metal.Geometry{ .grid = .{ @divExact(rows * n, 4), 1, 1 }, .tg = .{ 256, 1, 1 } };
        g.out(&.{ rows, n }, self.dt);
        g.dtype("T", self.dt);
        return g;
    }

    pub fn prepare(self: *BiasGelu, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        self.shape = shape;
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        self.y = try host.uniform(a, rng, shape.rows * shape.k, -6, 6, self.dt);
        self.b = try host.uniform(a, rng, shape.k, -2, 2, self.dt);
        if (fill == .nonfinite) host.plant(self.y, shape.rows, shape.k);
        self.rig.reset();
        self.args = .{
            try self.rig.input(self.dt, &.{ @intCast(shape.rows), @intCast(shape.k) }, self.y),
            try self.rig.input(self.dt, &.{@intCast(shape.k)}, self.b),
        };
        return true;
    }

    pub fn run(self: *BiasGelu) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, &self.args);
    }

    pub fn count(self: *const BiasGelu) usize {
        return self.shape.rows * self.shape.k;
    }

    /// The sum rounded to T, then `0.5 x (1 + tanh a)` = `x / (1 + e^(-2a))` with a = sqrt(2/pi) (x + 0.044715 x^3).
    pub fn reference(self: *const BiasGelu, want: []f64, mag: []f64) !void {
        const e = host.elemOf(self.dt);
        const n = self.shape.k;
        for (0..self.shape.rows) |r| for (0..n) |i| {
            const x = spec.roundTo(e, self.y[r * n + i] + self.b[i]);
            const a = 0.7978845608028654 * (x + 0.044715 * x * x * x);
            const out = x / (1 + @exp(-2 * a));
            want[r * n + i] = out;
            mag[r * n + i] = @abs(out);
        };
    }
};

// ── registrations ──

fn headWidth(shape: Shape) usize {
    return shape.k;
}

fn noDepth(_: Shape) usize {
    return 0;
}

pub fn register(set: *kc.Set) !void {
    const a = set.a;
    try set.add(QkvPrep, .init(a, &qkv_f16, .float16), .{
        .name = "idol qkv_prep f16 (rope, then norm, head-major)",
        .accum = .f32,
        .depth = headWidth,
        // f32 roundings: the rotation's two products and add, then the norm's mean, + eps, rsqrt and two products.
        .ops = 7,
        // The rotation's rounding to T before the norm reads it, and the rounding of that read.
        .mids = 2,
        .out = .f16,
        .replaces = "split -> rope (slice, multiply, subtract, add, concat) -> rms_norm -> multiply by gamma -> transpose",
        .served = &.{ .{ .rows = 64, .n = 16, .k = HD }, .{ .rows = 9, .n = 2, .k = HD } },
        .sums = .{ .f16 = 0, .bf16 = 0, .why = "positive squares: a pairwise sum stays near the output's rounding" },
    }, null);
    try set.add(BiasGelu, .init(a, &gelu_f16, .float16), .{
        .name = "idol bias_gelu f16",
        .accum = .none,
        .depth = noDepth,
        // The exponent's f32 chain (cube, product, sum, scale), the exp, + 1 and the quotient.
        .ops = 10,
        // The bias sum's rounding to T.
        .mids = 1,
        .out = .f16,
        .tiles = .{ .{}, .{}, .{ .size = 4, .step = 4, .min = 4 } },
        .served = &.{ .{ .rows = 64, .n = 1, .k = 4096 }, .{ .rows = 7, .n = 1, .k = 512 } },
    }, null);
}

// ── tests ──

test "kernel contract idol: the fused kernels pass gates 1-4" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const a = std.testing.allocator;
    var set = kc.Set.init(a);
    defer set.deinit();
    try register(&set);
    var t: kc.gates.Tally = .{};
    try kc.gates.runAll(a, set.kernels.items, 0xc0de, &t);
    try std.testing.expectEqual(@as(usize, 0), t.problems());
    try std.testing.expectEqual(@as(usize, 2), t.passed);
}

test "kernel contract idol: each declared launch is the shipped wrapper's, byte for byte" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const a = std.testing.allocator;
    var q = QkvPrep.init(a, &qkv_f16, .float16);
    defer q.deinit();
    try std.testing.expect(try q.prepare(.{ .rows = 9, .n = 2, .k = HD }, .random, 3));
    try q.run();
    const ours = try a.dupe(u8, q.rig.outputBytes());
    defer a.free(ours);
    try std.testing.expectEqualSlices(u8, try q.wrapper(), ours);

    var g = BiasGelu.init(a, &gelu_f16, .float16);
    defer g.deinit();
    try std.testing.expect(try g.prepare(.{ .rows = 7, .n = 1, .k = 512 }, .random, 4));
    try g.run();
    const mine = try a.dupe(u8, g.rig.outputBytes());
    defer a.free(mine);
    const theirs = (try glue.biasGelu(g.args[0], g.args[1], g.rig.stream())) orelse return error.Declined;
    defer _ = mlx.mlx_array_free(theirs);
    try std.testing.expectEqualSlices(u8, try g.rig.bytesOf(&.{theirs}), mine);
}
