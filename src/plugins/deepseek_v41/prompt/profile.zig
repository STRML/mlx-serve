const std = @import("std");

pub const Stage = enum { setup, engram, hc, attn_in, q, compress, indexer, core, attn_out, shared, routed, expand, fused, head };
pub const Pass = enum { full, decoder };

pub const Profile = struct {
    started: i64,
    last: i64,
    us: [std.enums.values(Stage).len]u64 = @splat(0),

    pub fn start(enabled: bool, storage: *Profile, comptime now: fn () i64) ?*Profile {
        if (!enabled) return null;
        const tick = now();
        storage.* = .{ .started = tick, .last = tick };
        return storage;
    }

    pub inline fn mark(self: ?*Profile, stage: Stage, comptime evaluate: anytype, args: anytype, comptime now: fn () i64) !void {
        const p = self orelse return;
        try @call(.auto, evaluate, args);
        const tick = now();
        p.us[@backingInt(stage)] += elapsed(p.last, tick);
        p.last = tick;
    }

    pub fn line(self: *const Profile, pass: Pass, req: []const u8, pos: usize, rows: usize, layers: usize, end: i64) Line {
        return .{ .profile = self, .pass = pass, .req = req, .pos = pos, .rows = rows, .layers = layers, .total = elapsed(self.started, end) };
    }
};

fn elapsed(start: i64, end: i64) u64 {
    return @intCast(@max(end - start, 0));
}

const Line = struct {
    profile: *const Profile,
    pass: Pass,
    req: []const u8,
    pos: usize,
    rows: usize,
    layers: usize,
    total: u64,

    pub fn format(self: Line, w: *std.Io.Writer) std.Io.Writer.Error!void {
        const tag = if (self.pass == .full) "dsv41-prof" else "dsv41-prof-dec";
        try w.print("[{s}] req={s} pos={d} rows={d} layers={d}", .{ tag, self.req, self.pos, self.rows, self.layers });
        for (std.enums.values(Stage), self.profile.us) |stage, us| {
            try writeMs(w, @tagName(stage), us);
        }
        try writeMs(w, "total_ms", self.total);
    }
};

fn writeMs(w: *std.Io.Writer, key: []const u8, us: u64) std.Io.Writer.Error!void {
    try w.print(" {s}={d}.{d:0>3}", .{ key, us / 1000, us % 1000 });
}

test "prefill profile formats both passes with their chunk position and zero stages" {
    var p: Profile = .{ .started = 100, .last = 100 };
    p.us[@backingInt(Stage.indexer)] = 1234;
    var buf: [512]u8 = undefined;
    const full = p.line(.full, "chatcmpl-0123456789abcdef-7", 8192, 4096, 20, 2100);
    try std.testing.expectEqualStrings(
        "[dsv41-prof] req=chatcmpl-0123456789abcdef-7 pos=8192 rows=4096 layers=20 setup=0.000 engram=0.000 hc=0.000 attn_in=0.000 q=0.000 compress=0.000 indexer=1.234 core=0.000 attn_out=0.000 shared=0.000 routed=0.000 expand=0.000 fused=0.000 head=0.000 total_ms=2.000",
        try std.fmt.bufPrint(&buf, "{f}", .{full}),
    );
    const tail = p.line(.decoder, "-", 8192, 128, 23, 2100);
    const out = try std.fmt.bufPrint(&buf, "{f}", .{tail});
    try std.testing.expectStringStartsWith(out, "[dsv41-prof-dec] req=- pos=8192 rows=128 layers=23 ");
}

test "prefill profile off never reads the clock or evaluates a stage" {
    const Calls = struct {
        var clocks: usize = 0;
        var evals: usize = 0;

        fn now() i64 {
            clocks += 1;
            return @intCast(clocks);
        }

        fn eval() !void {
            evals += 1;
        }
    };
    var storage: Profile = undefined;
    const off = Profile.start(false, &storage, Calls.now);
    try Profile.mark(off, .indexer, Calls.eval, .{}, Calls.now);
    try std.testing.expectEqual(@as(usize, 0), Calls.clocks);
    try std.testing.expectEqual(@as(usize, 0), Calls.evals);

    const on = Profile.start(true, &storage, Calls.now);
    try Profile.mark(on, .indexer, Calls.eval, .{}, Calls.now);
    try std.testing.expectEqual(@as(usize, 2), Calls.clocks);
    try std.testing.expectEqual(@as(usize, 1), Calls.evals);
    try std.testing.expectEqual(@as(u64, 1), storage.us[@backingInt(Stage.indexer)]);
}
