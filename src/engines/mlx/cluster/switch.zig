//! Forms one distributed switch attempt and owns it until its outcome is retired.
const std = @import("std");
const builtin = @import("builtin");
const distributed = @import("distributed.zig");
const crt = @import("../../../sdk/util/crt.zig");

pub const Role = enum { leader, follower };
pub const Request = struct { n: u64, role: Role, deadline_epoch_ms: i64, cfg: distributed.Config };
pub const Outcome = union(enum) { formed: *distributed.Link, failed: anyerror, cancelled, quarantined };
const Cancel = enum { cancelled, already_formed, unknown };
const State = enum(u8) { forming, cancelling, formed, failed, cancelled };
const Open = *const fn (std.Io, distributed.Config) anyerror!distributed.Link;

const Attempt = struct {
    req: Request,
    control: distributed.OpenControl,
    deadline_ms: u64,
    state: std.atomic.Value(State) = .init(.forming),
    returned: std.atomic.Value(bool) = .init(false),
    finished_ms: u64 = 0,
    thread: ?std.Thread = null,
    link: distributed.Link = undefined,
    failure: anyerror = error.InvalidArgs,
    open: Open,

    fn run(self: *Attempt) void {
        defer {
            self.finished_ms = @intCast(crt.monotonicMs());
            self.returned.store(true, .release);
        }
        const io = std.Io.Threaded.global_single_threaded.io();
        self.link = self.open(io, self.req.cfg) catch |err| {
            self.failure = err;
            if (self.state.cmpxchgStrong(.forming, .failed, .acq_rel, .acquire) != null)
                self.state.store(.cancelled, .release);
            return;
        };
        if (self.state.cmpxchgStrong(.forming, .formed, .acq_rel, .acquire) == null) return;
        self.link.cancelledAfterGo();
        self.link.deinit();
        self.state.store(.cancelled, .release);
    }

    fn destroy(self: *Attempt) void {
        if (self.thread) |t| t.join();
        if (self.state.load(.acquire) == .formed) self.link.deinit();
        std.heap.page_allocator.free(self.req.cfg.coordinator);
        std.heap.page_allocator.free(self.req.cfg.device);
        std.heap.page_allocator.destroy(self);
    }
};

var mu: std.atomic.Mutex = .unlocked;
var current: ?*Attempt = null;
var poisoned = false;

fn lock() void {
    while (!mu.tryLock()) std.atomic.spinLoopHint();
}

pub fn start(req: Request) error{ Busy, InvalidArgs }!void {
    lock();
    defer mu.unlock();
    refreshQuarantine();
    if (poisoned) return error.Busy;
    if (current) |attempt| {
        if (!attempt.returned.load(.acquire) or attempt.state.load(.acquire) == .formed) return error.Busy;
    }
    const wall = crt.wallMs();
    if (req.cfg.size < 2 or req.cfg.size > distributed.max_ranks or req.cfg.rank >= req.cfg.size or req.deadline_epoch_ms <= wall) return error.InvalidArgs;
    if ((req.role == .leader) != (req.cfg.rank == 0)) return error.InvalidArgs;
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
    const remaining: u64 = @intCast(req.deadline_epoch_ms - wall);
    a.* = .{ .req = req, .control = .{ .n = req.n }, .deadline_ms = @as(u64, @intCast(crt.monotonicMs())) +| remaining, .open = if (builtin.is_test) testing.open else distributed.Link.open };
    a.req.cfg.coordinator = coord;
    a.req.cfg.device = device;
    a.req.cfg.formation_deadline_ms = a.deadline_ms;
    a.req.cfg.ready_in_open = true;
    a.req.cfg.open_control = &a.control;
    a.thread = std.Thread.spawn(.{ .stack_size = 1 << 20 }, Attempt.run, .{a}) catch return error.InvalidArgs;
    current = a;
}

pub fn cancel(n: u64) Cancel {
    lock();
    defer mu.unlock();
    const a = current orelse return .unknown;
    if (a.req.n != n or poisoned) return .unknown;
    if (a.state.cmpxchgStrong(.forming, .cancelling, .acq_rel, .acquire)) |state| {
        return switch (state) {
            .formed => .already_formed,
            .cancelling, .cancelled => .cancelled,
            else => .unknown,
        };
    }
    a.control.cancel();
    return .cancelled;
}

pub fn poll(n: u64) ?Outcome {
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
        .formed => .{ .formed = &a.link },
        .failed => .{ .failed = a.failure },
        .cancelled => .cancelled,
        else => null,
    };
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

    pub fn reset() void {
        lock();
        defer mu.unlock();
        if (current) |a| a.destroy();
        current = null;
        poisoned = false;
        open = distributed.Link.open;
    }
} else struct {};
