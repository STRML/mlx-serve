//! `/v1/internal/*` (plan 61c items 5, 10 and 11): the router's switch commands and mode poll, behind an internal key
//! that is never the serving key, plus the pieces of the public path that the switch gates (admission, `/health`).
//! Pure over a `Switch`: the connection glue is in `http/router.zig`.

const std = @import("std");
const attempt = @import("attempt.zig");
const sw_mod = @import("switch.zig");

pub const prefix = "/v1/internal/";

pub const Request = struct {
    method: []const u8,
    path: []const u8,
    /// The `Authorization` header value, or null.
    authorization: ?[]const u8,
    body: []const u8,
};

pub const Response = struct { status: u16, body: []const u8 };

const max_body = 1024;
/// A formation deadline further out than this is a router bug, not a deadline.
const max_deadline_ahead_ms: i64 = 10_000;
const max_drain_wait_ms: i64 = 1000;

pub fn isInternalPath(path: []const u8) bool {
    return std.mem.startsWith(u8, path, prefix);
}

const not_inference = [_][]const u8{
    "/tokenize",
    "/detokenize",
    "/v1/models/rescan",
    "/v1/providers/reload",
    "/v1/load-model",
    "/v1/unload-model",
    "/v1/responses/compact",
    "/api/pull",
    "/api/show",
};

/// Public inference, which a switch drains and a TP4 follower refuses: every POST that is not a model-free utility,
/// plus the Responses WebSocket upgrade.
pub fn isInference(method: []const u8, path: []const u8) bool {
    if (isInternalPath(path)) return false;
    if (std.mem.eql(u8, method, "GET")) return std.mem.eql(u8, path, "/v1/responses");
    if (!std.mem.eql(u8, method, "POST")) return false;
    for (not_inference) |p| if (std.mem.eql(u8, path, p)) return false;
    return true;
}

pub fn statusLine(status: u16) []const u8 {
    return switch (status) {
        200 => "200 OK",
        400 => "400 Bad Request",
        401 => "401 Unauthorized",
        404 => "404 Not Found",
        405 => "405 Method Not Allowed",
        409 => "409 Conflict",
        501 => "501 Not Implemented",
        502 => "502 Bad Gateway",
        503 => "503 Service Unavailable",
        else => "500 Internal Server Error",
    };
}

/// The internal key is held as its sha256 (`ablation.readKey`); digests compare in constant time, so neither length nor content leaks.
pub fn keyMatches(digest: [32]u8, presented: []const u8) bool {
    var d: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(std.mem.trim(u8, presented, " \t\r\n"), &d, .{});
    return std.crypto.timing_safe.eql([32]u8, d, digest);
}

fn authorized(digest: [32]u8, authorization: ?[]const u8) bool {
    const h = authorization orelse return false;
    const bearer = "Bearer ";
    if (!std.mem.startsWith(u8, h, bearer)) return false;
    return keyMatches(digest, h[bearer.len..]);
}

const Params = struct { n: u64, deadline_ms: ?i64 = null, wait_ms: i64 = max_drain_wait_ms };

/// Only the attempt id, the router's deadline and the drain wait are read; every other field, a device or a
/// coordinator included, is ignored.
fn parseParams(body: []const u8) ?Params {
    if (body.len == 0 or body.len > max_body) return null;
    var fba_buf: [4096]u8 = undefined;
    var fba = std.heap.FixedBufferAllocator.init(&fba_buf);
    const parsed = std.json.parseFromSlice(std.json.Value, fba.allocator(), body, .{}) catch return null;
    if (parsed.value != .object) return null;
    const obj = parsed.value.object;
    const n = obj.get("n") orelse return null;
    if (n != .integer or n.integer <= 0) return null;
    var p: Params = .{ .n = @intCast(n.integer) };
    if (obj.get("deadline_ms")) |d| {
        if (d != .integer) return null;
        p.deadline_ms = d.integer;
    }
    if (obj.get("wait_ms")) |w| {
        if (w != .integer) return null;
        p.wait_ms = std.math.clamp(w.integer, 0, max_drain_wait_ms);
    }
    return p;
}

fn commandOf(name: []const u8) ?attempt.Cmd {
    return std.meta.stringToEnum(attempt.Cmd, name);
}

fn plain(buf: []u8, status: u16, msg: []const u8) Response {
    return .{ .status = status, .body = std.fmt.bufPrint(buf, "{{\"ok\":false,\"error\":\"{s}\"}}", .{msg}) catch "{}" };
}

fn render(buf: []u8, status: u16, reason: []const u8, drained: ?bool, snap: sw_mod.Snapshot) Response {
    var tmp: [512]u8 = undefined;
    const body = sw_mod.renderSnapshot(&tmp, snap);
    var dr: [24]u8 = undefined;
    const drained_field = if (drained) |d| std.fmt.bufPrint(&dr, ",\"drained\":{}", .{d}) catch "" else "";
    return .{ .status = status, .body = std.fmt.bufPrint(buf, "{{\"ok\":{},\"reason\":\"{s}\"{s},{s}", .{ status == 200, reason, drained_field, body[1..] }) catch "{}" };
}

/// The router's deadline is wall-clock and shared by every rank; it is refused when past or absurdly far.
fn validDeadline(wall_deadline_ms: i64, wall_now_ms: i64) bool {
    const ahead = wall_deadline_ms - wall_now_ms;
    return ahead > 0 and ahead <= max_deadline_ahead_ms;
}

pub fn handle(sw: *sw_mod.Switch, internal_key: ?[32]u8, req: Request, wall_now_ms: i64, buf: []u8) Response {
    const key = internal_key orelse return plain(buf, 404, "not found");
    if (!authorized(key, req.authorization)) return plain(buf, 401, "unauthorized");
    const route = req.path[@min(prefix.len, req.path.len)..];
    const is_get = std.mem.eql(u8, req.method, "GET");
    if (std.mem.eql(u8, route, "mode")) {
        if (!is_get) return plain(buf, 405, "method not allowed");
        return render(buf, 200, "", null, sw.snapshot());
    }
    const cmd = commandOf(route) orelse return plain(buf, 404, "not found");
    if (!std.mem.eql(u8, req.method, "POST")) return plain(buf, 405, "method not allowed");
    const p = parseParams(req.body) orelse return plain(buf, 400, "body needs a positive integer n");
    var deadline: i64 = 0;
    if (p.deadline_ms) |d| {
        if (!validDeadline(d, wall_now_ms)) return plain(buf, 400, "deadline is past or too far ahead");
        deadline = d;
    }
    const reply = sw.command(cmd, p.n, deadline);
    var drained: ?bool = null;
    var snap = reply.snap;
    if (cmd == .drain and reply.status == 200) {
        drained = sw.waitDrained(p.wait_ms);
        snap = sw.snapshot();
    }
    return render(buf, reply.status, reply.reason, drained, snap);
}

/// `/health`: 200 `ok` with the mode fields, 503 naming quarantine, an abort in progress, or a stall.
pub fn healthResponse(sw: *sw_mod.Switch, buf: []u8) Response {
    const h = sw.health();
    const s = sw.snapshot();
    return .{
        .status = if (h == .ok) 200 else 503,
        .body = std.fmt.bufPrint(buf, "{{\"status\":\"{s}\",\"mode\":\"{s}\",\"n\":{d},\"role\":\"{s}\",\"peers\":{d}}}", .{
            @tagName(h), @tagName(s.mode), s.n, @tagName(s.role), s.peers,
        }) catch "{}",
    };
}

const t = std.testing;

const KEY = "internal-key-0123456789";

var test_now: i64 = 50_000;
fn testNow(_: std.Io) i64 {
    return test_now;
}

const Rig = struct {
    sw: sw_mod.Switch,
    calls: [8]u8 = undefined,
    n_calls: usize = 0,
    deadline: i64 = 0,

    fn me(ctx: *anyopaque) *Rig {
        return @ptrCast(@alignCast(ctx));
    }
    fn start(ctx: *anyopaque, _: attempt.Role, _: u64, deadline_epoch_ms: i64) sw_mod.StartError!void {
        me(ctx).deadline = deadline_epoch_ms;
        me(ctx).note('s');
    }
    fn cancel(ctx: *anyopaque, _: u64) sw_mod.CancelResult {
        me(ctx).note('c');
        return .cancelled;
    }
    fn poll(_: *anyopaque, _: u64) ?sw_mod.FormOutcome {
        return null;
    }
    fn release(ctx: *anyopaque, _: u64, _: *anyopaque, _: *?u32) sw_mod.ReleaseError!void {
        me(ctx).note('r');
    }
    fn fall(ctx: *anyopaque, _: u64, _: *anyopaque) void {
        me(ctx).note('f');
    }
    fn quarantined(_: *anyopaque) bool {
        return false;
    }
    fn exitProcess(ctx: *anyopaque, _: u64) void {
        me(ctx).note('x');
    }
    fn peers(_: *anyopaque) u8 {
        return 0;
    }
    fn stallNote(_: *anyopaque, _: []u8) []const u8 {
        return "";
    }
    fn note(r: *Rig, c: u8) void {
        r.calls[r.n_calls] = c;
        r.n_calls += 1;
    }
};

fn wallNow(_: std.Io) i64 {
    return WALL;
}

fn rig() Rig {
    test_now = 50_000;
    var r: Rig = .{ .sw = sw_mod.Switch.init(t.io) };
    r.sw.now_fn = testNow;
    r.sw.wall_fn = wallNow;
    r.sw.spawn_watcher = false;
    return r;
}

fn wire(r: *Rig) void {
    r.sw.formation = .{
        .ctx = r,
        .start = Rig.start,
        .cancel = Rig.cancel,
        .poll = Rig.poll,
        .release = Rig.release,
        .fall = Rig.fall,
        .quarantined = Rig.quarantined,
        .exit_process = Rig.exitProcess,
        .peers = Rig.peers,
        .stall_note = Rig.stallNote,
    };
}

const WALL: i64 = 1_700_000_000_000;

fn digestOf(key: []const u8) [32]u8 {
    var d: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(key, &d, .{});
    return d;
}

fn call(r: *Rig, key: ?[]const u8, method: []const u8, path: []const u8, auth: ?[]const u8, body: []const u8) Response {
    const S = struct {
        var buf: [1024]u8 = undefined;
    };
    return handle(&r.sw, if (key) |k| digestOf(k) else null, .{ .method = method, .path = path, .authorization = auth, .body = body }, WALL, &S.buf);
}

const BEARER = "Bearer " ++ KEY;

fn expectField(body: []const u8, field: []const u8, want: []const u8) !void {
    const p = try std.json.parseFromSlice(std.json.Value, t.allocator, body, .{});
    defer p.deinit();
    const v = p.value.object.get(field) orelse return error.MissingField;
    switch (v) {
        .string => |s| try t.expectEqualStrings(want, s),
        .integer => |i| try t.expectEqual(try std.fmt.parseInt(i64, want, 10), i),
        .bool => |b| try t.expectEqualStrings(want, if (b) "true" else "false"),
        else => return error.UnexpectedType,
    }
}

test "no internal key configured: /v1/internal/* does not exist" {
    var r = rig();
    try t.expectEqual(@as(u16, 404), call(&r, null, "GET", "/v1/internal/mode", BEARER, "").status);
}

test "the serving key, no key and a wrong key are all 401; the internal key reaches the mode" {
    var r = rig();
    try t.expectEqual(@as(u16, 401), call(&r, KEY, "GET", "/v1/internal/mode", "Bearer serving-key", "").status);
    try t.expectEqual(@as(u16, 401), call(&r, KEY, "GET", "/v1/internal/mode", null, "").status);
    try t.expectEqual(@as(u16, 401), call(&r, KEY, "GET", "/v1/internal/mode", KEY, "").status);
    try t.expectEqual(@as(u16, 401), call(&r, KEY, "POST", "/v1/internal/abort", "Bearer serving-key", "{\"n\":1}").status);
    const ok = call(&r, KEY, "GET", "/v1/internal/mode", BEARER, "");
    try t.expectEqual(@as(u16, 200), ok.status);
    try expectField(ok.body, "mode", "tp1");
}

test "routes: wrong method is 405, unknown path is 404" {
    var r = rig();
    try t.expectEqual(@as(u16, 405), call(&r, KEY, "POST", "/v1/internal/mode", BEARER, "{}").status);
    try t.expectEqual(@as(u16, 405), call(&r, KEY, "GET", "/v1/internal/drain", BEARER, "").status);
    try t.expectEqual(@as(u16, 404), call(&r, KEY, "POST", "/v1/internal/reboot", BEARER, "{\"n\":1}").status);
    try t.expectEqual(@as(u16, 404), call(&r, KEY, "GET", "/v1/internal/", BEARER, "").status);
}

test "a command needs a positive integer n" {
    var r = rig();
    for ([_][]const u8{ "", "{}", "{\"n\":0}", "{\"n\":-1}", "{\"n\":\"7\"}", "not json", "{\"n\":1.5}" }) |b| {
        try t.expectEqual(@as(u16, 400), call(&r, KEY, "POST", "/v1/internal/drain", BEARER, b).status);
    }
}

test "drain: closes admission, reports drained, and a body naming devices or coordinators is ignored" {
    var r = rig();
    const res = call(&r, KEY, "POST", "/v1/internal/drain", BEARER, "{\"n\":3,\"coordinator\":\"10.0.0.9:5000\",\"device\":\"mlx\"}");
    try t.expectEqual(@as(u16, 200), res.status);
    try expectField(res.body, "ok", "true");
    try expectField(res.body, "drained", "true");
    try expectField(res.body, "state", "draining");
    try t.expect(!r.sw.enter());
}

test "drain with a request still in flight answers drained false after wait_ms" {
    var r = rig();
    try t.expect(r.sw.enter());
    const res = call(&r, KEY, "POST", "/v1/internal/drain", BEARER, "{\"n\":3,\"wait_ms\":10}");
    try t.expectEqual(@as(u16, 200), res.status);
    try expectField(res.body, "drained", "false");
    try expectField(res.body, "in_flight", "1");
}

test "a refused command is 409 with the reason and the box's N" {
    var r = rig();
    _ = call(&r, KEY, "POST", "/v1/internal/drain", BEARER, "{\"n\":9}");
    const res = call(&r, KEY, "POST", "/v1/internal/drain", BEARER, "{\"n\":2}");
    try t.expectEqual(@as(u16, 409), res.status);
    try expectField(res.body, "ok", "false");
    try expectField(res.body, "reason", "stale");
    try expectField(res.body, "n", "9");
}

test "tp4 hands the formation the router's wall-clock deadline, not a duration" {
    var r = rig();
    wire(&r);
    _ = call(&r, KEY, "POST", "/v1/internal/drain", BEARER, "{\"n\":1}");
    const res = call(&r, KEY, "POST", "/v1/internal/tp4", BEARER, "{\"n\":1,\"deadline_ms\":1700000001900}");
    try t.expectEqual(@as(u16, 200), res.status);
    try t.expectEqual(@as(i64, 1_700_000_001_900), r.deadline);
}

test "a deadline in the past or more than ten seconds out is a 400" {
    var r = rig();
    wire(&r);
    _ = call(&r, KEY, "POST", "/v1/internal/drain", BEARER, "{\"n\":1}");
    try t.expectEqual(@as(u16, 400), call(&r, KEY, "POST", "/v1/internal/join", BEARER, "{\"n\":1,\"deadline_ms\":1699999999000}").status);
    try t.expectEqual(@as(u16, 400), call(&r, KEY, "POST", "/v1/internal/join", BEARER, "{\"n\":1,\"deadline_ms\":1700000011000}").status);
    try t.expectEqual(attempt.State.draining, r.sw.snapshot().state);
}

test "tp1 on a box with no formation backend is 501" {
    var r = rig();
    try t.expectEqual(@as(u16, 501), call(&r, KEY, "POST", "/v1/internal/tp1", BEARER, "{\"n\":1}").status);
}

test "mode poll shows the attempt the router needs to adopt or fence" {
    var r = rig();
    wire(&r);
    _ = call(&r, KEY, "POST", "/v1/internal/drain", BEARER, "{\"n\":4}");
    _ = call(&r, KEY, "POST", "/v1/internal/join", BEARER, "{\"n\":4}");
    const m = call(&r, KEY, "GET", "/v1/internal/mode", BEARER, "");
    try expectField(m.body, "state", "forming");
    try expectField(m.body, "role", "follower");
    try expectField(m.body, "n", "4");
    try expectField(m.body, "high_water", "4");
    try expectField(m.body, "admitting", "false");
}

test "every command and the mode poll are served on any repeat" {
    var r = rig();
    wire(&r);
    for ([_][]const u8{ "drain", "drain", "tp4", "tp4", "abort", "abort", "tp1", "join" }) |c| {
        var path: [40]u8 = undefined;
        const p = try std.fmt.bufPrint(&path, "/v1/internal/{s}", .{c});
        const res = call(&r, KEY, "POST", p, BEARER, "{\"n\":2}");
        try t.expect(res.status == 200 or res.status == 409 or res.status == 501);
    }
}

test "which requests count as public inference" {
    try t.expect(isInference("POST", "/v1/chat/completions"));
    try t.expect(isInference("POST", "/v1/messages"));
    try t.expect(isInference("POST", "/api/generate"));
    try t.expect(isInference("POST", "/v1/embeddings"));
    try t.expect(isInference("GET", "/v1/responses"));
    try t.expect(!isInference("GET", "/v1/models"));
    try t.expect(!isInference("GET", "/health"));
    try t.expect(!isInference("GET", "/metrics"));
    try t.expect(!isInference("POST", "/tokenize"));
    try t.expect(!isInference("POST", "/v1/internal/drain"));
    try t.expect(!isInference("POST", "/v1/models/rescan"));
    try t.expect(!isInference("POST", "/v1/providers/reload"));
    try t.expect(!isInference("OPTIONS", "/v1/chat/completions"));
}

test "the health body carries mode, n, role and peers; 503 names the cause" {
    var r = rig();
    var buf: [256]u8 = undefined;
    const ok = healthResponse(&r.sw, &buf);
    try t.expectEqual(@as(u16, 200), ok.status);
    try expectField(ok.body, "status", "ok");
    try expectField(ok.body, "mode", "tp1");
    _ = r.sw.command(.abort, 5, 0);
    _ = r.sw.command(.drain, 6, 0);
    wire(&r);
    _ = r.sw.command(.join, 6, 0);
    test_now += 2500;
    const q = healthResponse(&r.sw, &buf);
    try t.expectEqual(@as(u16, 503), q.status);
    try expectField(q.body, "status", "quarantined");
}
