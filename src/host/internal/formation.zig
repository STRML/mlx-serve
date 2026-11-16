//! Named-error bridge for the layout-switch formation API.
const std = @import("std");
const controller = @import("switch.zig");
const Role = @import("attempt.zig").Role;

pub const Config = struct {
    coordinator: []const u8,
    device: []const u8,
    model_dir: []const u8,
    rank: u32 = 0,
    size: u32 = 2,
};

pub const Backend = struct {
    ctx: *anyopaque,
    start: *const fn (*anyopaque, Role, u64, i64) ?[]const u8,
    cancel: *const fn (*anyopaque, u64) controller.CancelResult,
    poll: *const fn (*anyopaque, u64) ?controller.FormOutcome,
    activate: *const fn (*anyopaque, u64, *anyopaque) ?[]const u8,
    abort_formed: *const fn (*anyopaque, u64) void,
    release: *const fn (*anyopaque, u64, *anyopaque, *?u32) ?[]const u8,
    fall: *const fn (*anyopaque, u64, *anyopaque) void,
    quarantined: *const fn (*anyopaque) bool,
    exit_process: *const fn (*anyopaque, u64) void,
    peers: *const fn (*anyopaque) u8,
    stall_note: *const fn (*anyopaque, []u8) []const u8,

    pub fn formation(self: *Backend) controller.Formation {
        return .{ .ctx = self, .start = start, .cancel = cancel, .poll = poll, .activate = activate, .abort_formed = abortFormed, .release = release, .fall = fall, .quarantined = quarantined, .exit_process = exitProcess, .peers = peers, .stall_note = stallNote };
    }
};

fn backend(ctx: *anyopaque) *Backend {
    return @ptrCast(@alignCast(ctx));
}

fn start(ctx: *anyopaque, role: Role, n: u64, deadline: i64) controller.StartError!void {
    const b = backend(ctx);
    const name = b.start(b.ctx, role, n, deadline) orelse return;
    inline for (@typeInfo(controller.StartError).error_set.error_names.?) |e| {
        if (std.mem.eql(u8, name, e)) return @field(controller.StartError, e);
    }
    return error.Unavailable;
}

fn cancel(ctx: *anyopaque, n: u64) controller.CancelResult {
    const b = backend(ctx);
    return b.cancel(b.ctx, n);
}

fn poll(ctx: *anyopaque, n: u64) ?controller.FormOutcome {
    const b = backend(ctx);
    return b.poll(b.ctx, n);
}

fn activate(ctx: *anyopaque, n: u64, link: *anyopaque) controller.ActivateError!void {
    const b = backend(ctx);
    if (b.activate(b.ctx, n, link) != null) return error.SwapFailed;
}

fn abortFormed(ctx: *anyopaque, n: u64) void {
    const b = backend(ctx);
    b.abort_formed(b.ctx, n);
}

fn release(ctx: *anyopaque, n: u64, link: *anyopaque, rank: *?u32) controller.ReleaseError!void {
    const b = backend(ctx);
    if (b.release(b.ctx, n, link, rank) != null) return error.ReleaseSendFailed;
}

fn fall(ctx: *anyopaque, n: u64, link: *anyopaque) void {
    const b = backend(ctx);
    b.fall(b.ctx, n, link);
}

fn quarantined(ctx: *anyopaque) bool {
    const b = backend(ctx);
    return b.quarantined(b.ctx);
}

fn exitProcess(ctx: *anyopaque, n: u64) void {
    const b = backend(ctx);
    b.exit_process(b.ctx, n);
}

fn peers(ctx: *anyopaque) u8 {
    const b = backend(ctx);
    return b.peers(b.ctx);
}

fn stallNote(ctx: *anyopaque, buf: []u8) []const u8 {
    const b = backend(ctx);
    return b.stall_note(b.ctx, buf);
}
