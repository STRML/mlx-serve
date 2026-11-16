//! `--internal-key-file`: builds the layout-switch controller and loads the key its `/v1/internal/*` takes.

const std = @import("std");
const ablation = @import("../api/ablation.zig");
const http_server = @import("../http/server.zig");
const log = @import("../../sdk/util/log.zig");
const sw_mod = @import("switch.zig");

pub const Installed = struct { sw: *sw_mod.Switch, key: [32]u8 };

pub const LaunchError = error{ InternalKeyRefused, InternalKeyIsServingKey, OutOfMemory };

const Gauges = struct { generated: u64, prefill_done: u64, prefilling: u64 };

/// Generation plus prefill tokens, in flight, from the scheduler's own atomics: only a change in the sum matters. The
/// token counts move without `--metrics`; the prefill phase is published only with it, so without it the box holds the
/// longer prefill limit.
fn progressOf(g: Gauges, metrics_on: bool) sw_mod.Progress {
    return .{ .sum = g.generated + g.prefill_done, .prefilling = g.prefilling > 0 or !metrics_on };
}

fn progress() sw_mod.Progress {
    const sch = http_server.launch().global_scheduler orelse return .{ .sum = 0, .prefilling = true };
    const eg = &sch.engine_gauges;
    return progressOf(.{
        .generated = sch.inflight_generated_tokens.load(.monotonic) + eg.generated.load(.monotonic),
        .prefill_done = sch.inflight_prefill_tokens.load(.monotonic) + eg.prefill_done.load(.monotonic),
        .prefilling = sch.requests_prefilling.load(.monotonic) + eg.prefilling.load(.monotonic),
    }, http_server.g_metrics != null);
}

/// Null without the flag. The key file follows the ablation key's rules (0600, owned, at least 32 bytes) and must not be
/// the serving key. Stall detection for `/health` always runs; `--metrics` only sharpens its prefill limit.
pub fn launch(gpa: std.mem.Allocator, io: std.Io, key_file: ?[]const u8, serving_key: ?[]const u8) LaunchError!?Installed {
    const path = key_file orelse return null;
    const key = ablation.readKey(gpa, path) catch |e| {
        log.err("[internal] key file {s} refused: {s}\n", .{ path, switch (e) {
            error.NotRegular => "not a regular file",
            error.NotOwner => "not owned by the server's user",
            error.GroupOrOtherAccess => "mode must be 0600 (no group or other access)",
            error.TooShort => "the key must be at least 32 bytes",
            error.Unreadable, error.OutOfMemory => "unreadable",
            error.Unsupported => "the layout switch needs a POSIX host",
        } });
        return error.InternalKeyRefused;
    };
    if (serving_key) |s| if (ablation.Set.digestMatches(key, s)) {
        log.err("[internal] --internal-key-file holds the serving key; /v1/internal/* needs a key of its own\n", .{});
        return error.InternalKeyIsServingKey;
    };
    const sw = try gpa.create(sw_mod.Switch);
    sw.* = sw_mod.Switch.init(io);
    sw.progress = progress;
    log.info("[internal] layout switch enabled: /v1/internal/* takes the key in {s}\n", .{path});
    return .{ .sw = sw, .key = key };
}

const testing = std.testing;

test "without --metrics the switch still has a progress source, and it holds the prefill limit" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var root_buf: [std.fs.max_path_bytes]u8 = undefined;
    const root = root_buf[0..try tmp.dir.realPath(testing.io, &root_buf)];
    const a = testing.allocator;
    const path = try std.fs.path.join(a, &.{ root, "key" });
    defer a.free(path);
    const path_z = try a.dupeSentinel(u8, path, 0);
    defer a.free(path_z);
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "key", .data = "internal-key-0123456789-0123456789-0123\n" });
    _ = std.c.chmod(path_z, 0o600);
    const inst = (try launch(a, testing.io, path, null)).?;
    defer a.destroy(inst.sw);
    try testing.expect(inst.sw.progress != null);
}

test "progress is the in-flight token sum; without metrics the phase is unknown so it reads as prefilling" {
    const idle = Gauges{ .generated = 0, .prefill_done = 0, .prefilling = 0 };
    try testing.expect(progressOf(idle, false).prefilling);
    try testing.expect(!progressOf(idle, true).prefilling);
    try testing.expect(progressOf(.{ .generated = 3, .prefill_done = 4, .prefilling = 1 }, true).prefilling);
    try testing.expectEqual(@as(u64, 7), progressOf(.{ .generated = 3, .prefill_done = 4, .prefilling = 0 }, true).sum);
}
