const std = @import("std");
const builtin = @import("builtin");
const env = @import("../../../env.zig");
const mlx = @import("../../../mlx/ffi.zig");
const system = @import("../../../sdk/util/system.zig");
const io_util = @import("../../../sdk/util/io_util.zig");
const log = @import("../../../sdk/util/log.zig");
const format = @import("import_peak_format.zig");

extern "c" fn getpid() c_int;
extern "c" fn proc_pid_rusage(pid: c_int, flavor: c_int, buffer: *anyopaque) c_int;
extern "c" fn proc_reset_footprint_interval(pid: c_int) c_int;

const RusageV4 = extern struct {
    uuid: [16]u8,
    preceding: [33]u64,
    interval_max_phys_footprint: u64,
};

pub const Start = struct {
    io: std.Io,
    started_us: i64,
    active: u64,
    cache: u64,
    footprint: u64,
    footprint_before_body: ?u64 = null,
    body_bytes: ?u64 = null,
};

var initialized = false;
var footprint_interval_supported = false;

pub fn init() void {
    if (initialized or !enabled()) return;
    initialized = true;
    if (comptime builtin.os.tag == .macos) {
        var usage: RusageV4 = undefined;
        footprint_interval_supported = proc_reset_footprint_interval(getpid()) == 0 and
            proc_pid_rusage(getpid(), 4, @ptrCast(&usage)) == 0;
    }
    log.info("[import-peak] footprint interval reset: {s}\n", .{if (footprint_interval_supported) "supported" else "unsupported"});
}

pub fn enabled() bool {
    return env.on(.DSV41_IMPORT_PEAK_LOG);
}

pub fn begin(io: std.Io) ?Start {
    if (!enabled()) return null;
    init();
    synchronize();
    _ = mlx.mlx_reset_peak_memory();
    if (footprint_interval_supported and proc_reset_footprint_interval(getpid()) != 0) footprint_interval_supported = false;
    var active: usize = 0;
    var cache: usize = 0;
    _ = mlx.mlx_get_active_memory(&active);
    _ = mlx.mlx_get_cache_memory(&cache);
    return .{
        .io = io,
        .started_us = io_util.nowUsMonotonic(io),
        .active = active,
        .cache = cache,
        .footprint = system.getAppMemFootprintBytes(),
    };
}

pub fn beginPeer(io: std.Io, footprint_before_body: ?u64, body_bytes: ?u64) ?Start {
    var start = begin(io) orelse return null;
    start.footprint_before_body = footprint_before_body;
    start.body_bytes = body_bytes;
    return start;
}

pub fn finish(start: ?Start, path: []const u8, n_tokens: usize, entry_bytes: u64, plan: anytype, refused: ?[]const u8) void {
    const s = start orelse return;
    if (refused) |reason| {
        synchronize();
        const elapsed: u64 = @intCast(@max(@divTrunc(io_util.nowUsMonotonic(s.io) - s.started_us, 1000), 0));
        var active: usize = 0;
        var cache: usize = 0;
        _ = mlx.mlx_get_active_memory(&active);
        _ = mlx.mlx_get_cache_memory(&cache);
        write(.{
            .path = path,
            .n_tokens = n_tokens,
            .entry_bytes = entry_bytes,
            .estimate = plan.peak,
            .mlx_peak = null,
            .peak_over_active_before = null,
            .active_before = s.active,
            .active_after = active,
            .cache_before = s.cache,
            .cache_after = cache,
            .footprint_before = s.footprint,
            .footprint_after = system.getAppMemFootprintBytes(),
            .footprint_peak = intervalPeak(),
            .loaded_bytes = plan.loaded,
            .destination_bytes = plan.destination,
            .ms = elapsed,
            .refused = reason,
            .footprint_before_body = s.footprint_before_body,
            .body_bytes = s.body_bytes,
        });
        return;
    }
    synchronize();
    const elapsed: u64 = @intCast(@max(@divTrunc(io_util.nowUsMonotonic(s.io) - s.started_us, 1000), 0));
    var peak: usize = 0;
    var active: usize = 0;
    var cache: usize = 0;
    _ = mlx.mlx_get_peak_memory(&peak);
    _ = mlx.mlx_get_active_memory(&active);
    _ = mlx.mlx_get_cache_memory(&cache);
    const footprint = system.getAppMemFootprintBytes();
    const interval = intervalPeak();
    write(.{
        .path = path,
        .n_tokens = n_tokens,
        .entry_bytes = entry_bytes,
        .estimate = plan.peak,
        .mlx_peak = peak,
        .peak_over_active_before = @as(u64, @intCast(peak)) -| s.active,
        .active_before = s.active,
        .active_after = active,
        .cache_before = s.cache,
        .cache_after = cache,
        .footprint_before = s.footprint,
        .footprint_after = footprint,
        .footprint_peak = interval,
        .loaded_bytes = plan.loaded,
        .destination_bytes = plan.destination,
        .ms = elapsed,
        .footprint_before_body = s.footprint_before_body,
        .body_bytes = s.body_bytes,
    });
}

fn intervalPeak() ?u64 {
    if (!footprint_interval_supported) return null;
    if (comptime builtin.os.tag != .macos) return null;
    var usage: RusageV4 = undefined;
    if (proc_pid_rusage(getpid(), 4, @ptrCast(&usage)) != 0) return null;
    return usage.interval_max_phys_footprint;
}

fn synchronize() void {
    const stream = mlx.gpuStream();
    defer _ = mlx.mlx_stream_free(stream);
    _ = mlx.mlx_synchronize(stream);
}

fn write(values: format.Values) void {
    var buf: [1024]u8 = undefined;
    const line = format.format(&buf, values) catch return;
    log.info("{s}\n", .{line});
}
