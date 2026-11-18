//! Forms one distributed switch attempt and owns it until its outcome is retired.
const std = @import("std");
const builtin = @import("builtin");
const distributed = @import("distributed.zig");
const crt = @import("../../../sdk/util/crt.zig");

pub const Role = enum { leader, follower };
pub const Context = struct { cfg: distributed.Config, model_dir: []const u8 };
const Request = struct { n: u64, role: Role, deadline_epoch_ms: i64, cfg: distributed.Config };
pub const FormOutcome = union(enum) {
    formed: struct { link: *distributed.Link, open_ms: u32, ready_to_go_ms: ?u32 },
    failed: struct { err: []const u8, rank: ?u32, elapsed_ms: u32 },
    cancelled,
    quarantined,
};
pub const Cancel = enum { cancelled, already_formed, already_finished, unknown };
const State = enum(u8) { forming, cancelling, formed, releasing, failed, cancelled };
const Open = *const fn (std.Io, distributed.Config) anyerror!distributed.Link;

const Attempt = struct {
    req: Request,
    control: distributed.OpenControl,
    deadline_ms: u64,
    started_ms: u64,
    model_dir: []const u8,
    state: std.atomic.Value(State) = .init(.forming),
    returned: std.atomic.Value(bool) = .init(false),
    finished_ms: u64 = 0,
    pending_abort: bool = false,
    thread: ?std.Thread = null,
    link: distributed.Link = undefined,
    failure: anyerror = error.InvalidArgs,
    open: Open,

    fn run(self: *Attempt) void {
        defer {
            lock();
            defer mu.unlock();
            self.finished_ms = @intCast(crt.monotonicMs());
            if (self.finished_ms > self.deadline_ms +| 1000) poisoned = true;
            if (self.state.load(.acquire) == .formed and !poisoned) {
                distributed.setSwitchGroup(&self.link, self.model_dir);
                if (self.pending_abort) self.link.abortFormed();
            }
            self.returned.store(true, .release);
        }
        const io = std.Io.Threaded.global_single_threaded.io();
        self.link = self.open(io, self.req.cfg) catch |err| {
            self.failure = err;
            if (self.state.cmpxchgStrong(.forming, .failed, .acq_rel, .acquire) != null)
                self.state.store(.cancelled, .release);
            return;
        };
        if (self.state.cmpxchgStrong(.forming, .formed, .acq_rel, .acquire) == null) {
            if (comptime builtin.is_test) if (testing.after_formed) |hook| hook();
            return;
        }
        self.link.cancelledAfterGo();
        self.link.deinit();
        self.state.store(.cancelled, .release);
    }

    fn destroy(self: *Attempt) void {
        if (self.thread) |t| t.join();
        if (self.state.load(.acquire) == .formed or self.state.load(.acquire) == .releasing) self.link.deinit();
        std.heap.page_allocator.free(self.req.cfg.coordinator);
        std.heap.page_allocator.free(self.req.cfg.device);
        std.heap.page_allocator.free(self.model_dir);
        std.heap.page_allocator.destroy(self);
    }
};

var mu: std.atomic.Mutex = .unlocked;
var current: ?*Attempt = null;
var poisoned = false;
var fence: u64 = 0;

fn lock() void {
    while (!mu.tryLock()) std.atomic.spinLoopHint();
}

pub fn start(ctx: Context, role: Role, n: u64, deadline_epoch_ms: i64) error{ Busy, InvalidArgs, DistEnvNotPrimed }!void {
    const req: Request = .{ .cfg = ctx.cfg, .role = role, .n = n, .deadline_epoch_ms = deadline_epoch_ms };
    lock();
    defer mu.unlock();
    refreshQuarantine();
    if (poisoned) return error.Busy;
    if (current) |attempt| {
        if (!attempt.returned.load(.acquire) or (attempt.state.load(.acquire) == .formed or attempt.state.load(.acquire) == .releasing)) return error.Busy;
    }
    if (req.n <= fence) return error.InvalidArgs;
    const wall = crt.wallMs();
    if (req.cfg.size < 2 or req.cfg.size > distributed.max_ranks or req.cfg.rank >= req.cfg.size or req.deadline_epoch_ms <= wall or ctx.model_dir.len == 0) return error.InvalidArgs;
    if ((req.role == .leader) != (req.cfg.rank == 0)) return error.InvalidArgs;
    if (!distributed.envPrimed(req.cfg)) return error.DistEnvNotPrimed;
    if (current) |attempt| {
        attempt.destroy();
        current = null;
    }
    const a = std.heap.page_allocator.create(Attempt) catch return error.InvalidArgs;
    errdefer std.heap.page_allocator.destroy(a);
    const coord = std.heap.page_allocator.dupe(u8, req.cfg.coordinator) catch return error.InvalidArgs;
    errdefer std.heap.page_allocator.free(coord);
    const device = std.heap.page_allocator.dupe(u8, req.cfg.device) catch return error.InvalidArgs;
    errdefer std.heap.page_allocator.free(device);
    const model_dir = std.heap.page_allocator.dupe(u8, ctx.model_dir) catch return error.InvalidArgs;
    errdefer std.heap.page_allocator.free(model_dir);
    const started_ms: u64 = @intCast(crt.monotonicMs());
    const remaining: u64 = @intCast(req.deadline_epoch_ms - wall);
    a.* = .{ .req = req, .control = .{ .n = req.n }, .started_ms = started_ms, .model_dir = model_dir, .deadline_ms = @as(u64, @intCast(crt.monotonicMs())) +| remaining, .open = if (builtin.is_test) testing.open else distributed.Link.open };
    a.req.cfg.coordinator = coord;
    a.req.cfg.device = device;
    a.req.cfg.formation_deadline_ms = a.deadline_ms;
    a.req.cfg.ready_in_open = true;
    a.req.cfg.open_control = &a.control;
    a.req.cfg.switch_model = model_dir;
    a.thread = std.Thread.spawn(.{ .stack_size = 1 << 20 }, Attempt.run, .{a}) catch return error.InvalidArgs;
    current = a;
}

pub fn cancel(n: u64) Cancel {
    lock();
    defer mu.unlock();
    const a = current orelse return .unknown;
    if (a.req.n != n) return .unknown;
    if (poisoned) return .already_finished;
    if (a.state.cmpxchgStrong(.forming, .cancelling, .acq_rel, .acquire)) |state| {
        return switch (state) {
            .formed => .already_formed,
            .cancelling => .cancelled,
            .cancelled, .failed, .releasing => .already_finished,
            else => .unknown,
        };
    }
    a.control.cancel();
    return .cancelled;
}

/// Called on the inference thread after the mode swap.
pub fn activate(n: u64) error{DistTeardownWrongThread}!void {
    lock();
    defer mu.unlock();
    const a = current orelse return;
    if (a.req.n != n or !a.returned.load(.acquire) or a.state.load(.acquire) != .formed) return;
    try a.link.activate();
}

pub fn poll(n: u64) ?FormOutcome {
    lock();
    defer mu.unlock();
    const a = current orelse return null;
    if (a.req.n != n) return null;
    refreshQuarantine();
    if (poisoned) return .quarantined;
    if (!a.returned.load(.acquire)) return null;
    if (a.thread) |t| {
        t.join();
        a.thread = null;
    }
    return switch (a.state.load(.acquire)) {
        .formed => .{ .formed = .{ .link = &a.link, .open_ms = elapsed(a), .ready_to_go_ms = a.link.ready_to_go_ms } },
        .failed => .{ .failed = .{ .err = @errorName(a.failure), .rank = a.control.failedRank(), .elapsed_ms = elapsed(a) } },
        .cancelled => .cancelled,
        else => null,
    };
}

/// Inference thread only, with no receive or collective in flight.
pub fn release(n: u64, link: *distributed.Link, rank: *u32) distributed.ReleaseError!void {
    const a = try beginRelease(n, link) orelse return;
    try link.drain();
    try link.release(rank);
    retire(a);
}

/// Inference thread only, with no receive or collective in flight.
pub fn fall(n: u64, link: *distributed.Link) distributed.TeardownError!void {
    const a = try beginRelease(n, link) orelse return;
    try link.drain();
    retire(a);
}

pub fn abortFormed(n: u64) void {
    lock();
    defer mu.unlock();
    fence = @max(fence, n);
    const a = current orelse return;
    if (a.req.n != n or a.req.role != .follower or a.state.load(.acquire) != .formed or poisoned) return;
    a.pending_abort = true;
    if (a.returned.load(.acquire)) a.link.abortFormed();
}

fn beginRelease(n: u64, link: *distributed.Link) distributed.TeardownError!?*Attempt {
    lock();
    defer mu.unlock();
    const a = current orelse return null;
    if (a.req.n != n or &a.link != link or !a.returned.load(.acquire) or a.state.load(.acquire) != .formed or poisoned) return null;
    try link.checkTeardownThread();
    a.state.store(.releasing, .release);
    link.disarm();
    return a;
}

fn retire(a: *Attempt) void {
    a.link.deinit();
    lock();
    defer mu.unlock();
    current = null;
    a.destroy();
}

fn elapsed(a: *const Attempt) u32 {
    return @intCast(@min(a.finished_ms -| a.started_ms, std.math.maxInt(u32)));
}

fn refreshQuarantine() void {
    if (poisoned) return;
    const a = current orelse return;
    const cutoff = a.deadline_ms +| 1000;
    if (a.returned.load(.acquire)) {
        if (a.finished_ms > cutoff) poisoned = true;
        return;
    }
    if (@as(u64, @intCast(crt.monotonicMs())) <= cutoff) return;
    if (a.returned.load(.acquire) and a.finished_ms <= cutoff) return;
    poisoned = true;
}

pub fn quarantined() bool {
    lock();
    defer mu.unlock();
    refreshQuarantine();
    return poisoned;
}

pub const testing = if (builtin.is_test) struct {
    pub var open: Open = distributed.Link.open;
    pub var after_formed: ?*const fn () void = null;

    pub fn reset() void {
        lock();
        const a = current;
        current = null;
        mu.unlock();
        if (a) |attempt| attempt.destroy();
        lock();
        defer mu.unlock();
        poisoned = false;
        fence = 0;
        open = distributed.Link.open;
        after_formed = null;
    }
} else struct {};
