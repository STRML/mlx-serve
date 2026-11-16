//! `--internal-key-file`: builds the layout-switch controller and loads the key its `/v1/internal/*` takes.

const std = @import("std");
const ablation = @import("../api/ablation.zig");
const metrics = @import("../metrics/metrics.zig");
const log = @import("../../sdk/util/log.zig");
const sw_mod = @import("switch.zig");

pub const Installed = struct { sw: *sw_mod.Switch, key: [32]u8 };

pub const LaunchError = error{ InternalKeyRefused, InternalKeyIsServingKey, OutOfMemory };

var progress_metrics: ?*metrics.Metrics = null;

/// The router's progress definition: generation plus prefill tokens, total and live.
fn progress() sw_mod.Progress {
    const m = progress_metrics.?;
    return .{
        .sum = m.generation_tokens_total.load() + m.prefill_tokens_total.load() + m.generation_tokens_live.load() + m.prefill_tokens_live.load(),
        .prefilling = m.requests_prefilling.load() > 0,
    };
}

/// Null without the flag. The key file follows the ablation key's rules (0600, owned, at least 32 bytes) and must not be
/// the serving key. Stall detection for `/health` needs `--metrics`, the counters it reads.
pub fn launch(gpa: std.mem.Allocator, io: std.Io, key_file: ?[]const u8, serving_key: ?[]const u8, m: ?*metrics.Metrics) LaunchError!?Installed {
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
    if (m) |mm| {
        progress_metrics = mm;
        sw.progress = progress;
    } else log.warn("[internal] no --metrics: /health cannot detect a stalled request\n", .{});
    log.info("[internal] layout switch enabled: /v1/internal/* takes the key in {s}\n", .{path});
    return .{ .sw = sw, .key = key };
}
