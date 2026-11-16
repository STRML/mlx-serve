//! Inference-thread adapter for distributed switch attempts.
const std = @import("std");
const distributed = @import("distributed.zig");
const switcher = @import("switch.zig");
const controller = @import("../../../host/internal/switch.zig");
const bridge = @import("../../../host/internal/formation.zig");
const Scheduler = @import("../../../host/gpu_thread/scheduler.zig").Scheduler;
const abi = @import("../../../abi/plugin.zig");
const driver = @import("../plugin/driver.zig");
const log = @import("../../../sdk/util/log.zig");

pub const SwapHook = *const fn (u64, *distributed.Link, switcher.Role, abi.ServingMode) anyerror!void;
var swap_hook: std.atomic.Value(?SwapHook) = .init(null);
var warned: std.atomic.Value(bool) = .init(false);

pub fn setSwapHook(hook: ?SwapHook) void {
    swap_hook.store(hook, .release);
}

pub const Adapter = struct {
    sw: *controller.Switch,
    context: switcher.Context,
    backend: bridge.Backend,

    pub fn init(self: *Adapter, sw: *controller.Switch, context: switcher.Context) void {
        self.* = .{
            .sw = sw,
            .context = context,
            .backend = .{ .ctx = self, .start = start, .cancel = cancel, .poll = poll, .activate = activate, .abort_formed = abortFormed, .release = release, .fall = fall, .quarantined = quarantined, .exit_process = exitProcess, .peers = peers, .stall_note = stallNote },
        };
        installed = self;
    }

    pub fn formation(self: *Adapter) controller.Formation {
        return self.backend.formation();
    }
};

pub fn install(sw: *controller.Switch, cfg: bridge.Config, out: *?*bridge.Backend) ?[]const u8 {
    if (cfg.size < 2 or cfg.size > distributed.max_ranks or cfg.rank >= cfg.size or cfg.model_dir.len == 0) return "InvalidArgs";
    const self = std.heap.page_allocator.create(Adapter) catch return "OutOfMemory";
    self.init(sw, .{ .cfg = .{ .rank = cfg.rank, .size = cfg.size, .coordinator = cfg.coordinator, .device = cfg.device }, .model_dir = cfg.model_dir });
    out.* = &self.backend;
    installed = self;
    return null;
}

var installed: ?*Adapter = null;

fn adapter(ctx: *anyopaque) *Adapter {
    return @ptrCast(@alignCast(ctx));
}

fn start(ctx: *anyopaque, role: @import("../../../host/internal/attempt.zig").Role, n: u64, deadline: i64) ?[]const u8 {
    lock();
    const available = inference_thread != null;
    mu.unlock();
    if (!available) return "Unavailable";
    const r: switcher.Role = switch (role) {
        .leader => .leader,
        .follower => .follower,
        .none => return "InvalidArgs",
    };
    switcher.start(adapter(ctx).context, r, n, deadline) catch |err| return @errorName(err);
    return null;
}

fn cancel(_: *anyopaque, n: u64) controller.CancelResult {
    return switch (switcher.cancel(n)) {
        .cancelled => .cancelled,
        .already_formed => .already_formed,
        .already_finished => .already_finished,
        .unknown => .unknown,
    };
}

fn poll(_: *anyopaque, n: u64) ?controller.FormOutcome {
    const out = switcher.poll(n) orelse return null;
    return switch (out) {
        .formed => |f| .{ .formed = .{ .ptr = f.link, .open_ms = f.open_ms, .ready_to_go_ms = f.ready_to_go_ms orelse 0 } },
        .failed => |f| .{ .failed = .{ .err = f.err, .rank = f.rank, .elapsed_ms = f.elapsed_ms } },
        .cancelled => .cancelled,
        .quarantined => .quarantined,
    };
}

fn activate(ctx: *anyopaque, n: u64, link: *anyopaque) ?[]const u8 {
    var work: Work = .{ .owner = adapter(ctx), .n = n, .link = @ptrCast(@alignCast(link)), .op = .activate };
    dispatch(&work);
    if (work.failure) |name| if (std.mem.eql(u8, name, "Unavailable")) distributed.exitSwitchAbort(n);
    return work.failure;
}

fn abortFormed(_: *anyopaque, n: u64) void {
    switcher.abortFormed(n);
}

fn release(ctx: *anyopaque, n: u64, link: *anyopaque, rank: *?u32) ?[]const u8 {
    var work: Work = .{ .owner = adapter(ctx), .n = n, .link = @ptrCast(@alignCast(link)), .op = .release };
    dispatch(&work);
    rank.* = work.rank;
    return work.failure;
}

fn fall(ctx: *anyopaque, n: u64, link: *anyopaque) void {
    var work: Work = .{ .owner = adapter(ctx), .n = n, .link = @ptrCast(@alignCast(link)), .op = .fall };
    dispatch(&work);
    if (work.failure != null) distributed.exitSwitchAbort(n);
}

fn quarantined(_: *anyopaque) bool {
    return switcher.quarantined();
}

fn exitProcess(_: *anyopaque, n: u64) void {
    distributed.exitSwitchAbort(n);
}

fn peers(ctx: *anyopaque) u8 {
    return @intCast(adapter(ctx).context.cfg.size -| 1);
}

fn stallNote(_: *anyopaque, buf: []u8) []const u8 {
    return distributed.switchStallNote(buf);
}

pub fn followerFell(n: u64, link: *distributed.Link) void {
    const self = installed orelse distributed.exitSwitchAbort(n);
    var work: Work = .{ .owner = self, .n = n, .link = link, .op = .follower_fell };
    dispatch(&work);
    if (work.failure != null) distributed.exitSwitchAbort(n);
}

const Work = struct {
    owner: *Adapter,
    n: u64,
    link: *distributed.Link,
    op: enum { activate, release, fall, follower_fell },
    next: ?*Work = null,
    done: std.atomic.Value(u32) = .init(0),
    failure: ?[]const u8 = null,
    rank: ?u32 = null,

    fn run(self: *Work) void {
        distributed.parkInferenceLoop();
        defer distributed.tickInferenceLoop();
        self.execute() catch |err| {
            self.failure = @errorName(err);
        };
    }

    fn execute(self: *Work) !void {
        switch (self.op) {
            .activate => {
                self.swap() catch {
                    self.leave() catch distributed.exitSwitchAbort(self.n);
                    return error.SwapFailed;
                };
                active_mode = .{ .n = self.n, .link = self.link };
                switcher.activate(self.n) catch {
                    self.restore() catch distributed.exitSwitchAbort(self.n);
                    self.leave() catch distributed.exitSwitchAbort(self.n);
                    return error.SwapFailed;
                };
            },
            .release => {
                try self.restore();
                _ = self.owner.sw.releasing(self.n);
                try self.releaseLink();
                _ = self.owner.sw.released(self.n);
            },
            .fall => {
                try self.restore();
                try switcher.fall(self.n, self.link);
            },
            .follower_fell => {
                try self.restore();
                try switcher.fall(self.n, self.link);
                _ = self.owner.sw.formationFailed(self.n, .release_timeout);
            },
        }
    }

    fn swap(self: *Work) !void {
        const hook = swap_hook.load(.acquire) orelse {
            if (!warned.swap(true, .acq_rel)) log.err("no TP mode is wired (step 1)\n", .{});
            return error.SwapFailed;
        };
        try hook(self.n, self.link, if (self.link.rank == 0) .leader else .follower, .tp);
    }

    fn restore(self: *Work) !void {
        const live = active_mode orelse return;
        if (live.n != self.n or live.link != self.link) return;
        active_mode = null;
        follower_pending.store(false, .release);
        const hook = swap_hook.load(.acquire) orelse return;
        try hook(self.n, self.link, if (self.link.rank == 0) .leader else .follower, .tp1);
    }

    fn leave(self: *Work) !void {
        if (self.link.rank == 0) return self.releaseLink();
        try switcher.fall(self.n, self.link);
    }

    fn releaseLink(self: *Work) !void {
        var rank: u32 = 0;
        switcher.release(self.n, self.link, &rank) catch |err| {
            if (err == error.ReleaseSendFailed) self.rank = rank;
            return err;
        };
    }
};

var mu: std.atomic.Mutex = .unlocked;
var inference_thread: ?std.Thread.Id = null;
var scheduler: ?*Scheduler = null;
var head: ?*Work = null;
var tail: ?*Work = null;
var pending: std.atomic.Value(bool) = .init(false);
var follower_pending: std.atomic.Value(bool) = .init(false);
var follower: Work = undefined;
var active_mode: ?struct { n: u64, link: *distributed.Link } = null;

fn lock() void {
    while (!mu.tryLock()) std.atomic.spinLoopHint();
}

pub fn bindInferenceThread(sch: ?*Scheduler) void {
    distributed.bindInferenceThread();
    lock();
    defer mu.unlock();
    inference_thread = std.Thread.getCurrentId();
    scheduler = sch;
}

pub fn installSwapHook() void {
    setSwapHook(&swapMode);
}

fn modelTransformer() !*@import("../transformer.zig").Transformer {
    const self = installed orelse return error.SwapFailed;
    const sch = scheduler orelse return error.SwapFailed;
    const entry = sch.registry.peekByPath(self.context.model_dir) orelse return error.SwapFailed;
    sch.registry.mutex.lockUncancelable(sch.io);
    defer sch.registry.mutex.unlock(sch.io);
    if (entry.state != .ready) return error.SwapFailed;
    return entry.transformer orelse error.SwapFailed;
}

fn modelBinding() !*driver.Bound {
    const xfm = try modelTransformer();
    return xfm.plugin orelse error.SwapFailed;
}

pub fn serveFollower() bool {
    if (!follower_pending.load(.acquire)) return false;
    var work = follower;
    const snap = work.owner.sw.snapshot();
    if (snap.quarantined) distributed.exitSwitchAbort(work.n);
    if (snap.mode != .tp4) return false;
    if (!follower_pending.swap(false, .acq_rel)) return false;
    distributed.parkInferenceLoop();
    defer distributed.tickInferenceLoop();
    const xfm = modelTransformer() catch |err| distributed.exitWorker(err);
    const m = @import("../plugin/intree.zig").dsv41(xfm) orelse distributed.exitWorker(error.NotPluginServed);
    const released = @import("../modules/call.zig").call("dsv41_serve_steps", .{ work.owner.sw.io, std.heap.page_allocator, work.link, m }) catch |err| {
        if (err != error.FallRequested) distributed.exitWorker(err);
        followerFell(work.n, work.link);
        return true;
    };
    if (!released) distributed.exitWorker(null);
    work.run();
    if (work.failure != null) distributed.exitSwitchAbort(work.n);
    return true;
}

fn swapMode(n: u64, link: *distributed.Link, role: switcher.Role, to: abi.ServingMode) !void {
    if (to == .tp1) {
        defer distributed.clearDsv41Tp();
        const b = try modelBinding();
        return b.swapMode(to);
    }
    distributed.installDsv41Tp(link);
    errdefer distributed.clearDsv41Tp();
    const b = try modelBinding();
    try b.swapMode(to);
    if (role == .follower) {
        follower = .{ .owner = installed.?, .n = n, .link = link, .op = .release };
        follower_pending.store(true, .release);
    }
}

pub fn unbindInferenceThread() void {
    lock();
    defer mu.unlock();
    inference_thread = null;
    scheduler = null;
    active_mode = null;
    follower_pending.store(false, .release);
    while (head) |work| {
        head = work.next;
        work.failure = "Unavailable";
        complete(work);
    }
    tail = null;
    pending.store(false, .release);
}

pub fn hasPending() bool {
    return pending.load(.acquire) or follower_pending.load(.acquire);
}

fn dispatch(work: *Work) void {
    lock();
    if (inference_thread == std.Thread.getCurrentId()) {
        mu.unlock();
        work.run();
        return;
    }
    if (inference_thread == null) {
        mu.unlock();
        work.failure = "Unavailable";
        return;
    }
    if (tail) |last| last.next = work else head = work;
    tail = work;
    pending.store(true, .release);
    if (scheduler) |s| {
        s.queue_mu.lockUncancelable(s.io);
        s.queue_cond.signal(s.io);
        s.queue_mu.unlock(s.io);
    }
    mu.unlock();
    while (work.done.load(.acquire) == 0) std.Io.Threaded.global_single_threaded.io().futexWaitUncancelable(u32, &work.done.raw, 0);
}

fn complete(work: *Work) void {
    work.done.store(1, .release);
    std.Io.Threaded.global_single_threaded.io().futexWake(u32, &work.done.raw, 1);
}

pub fn drain() bool {
    var ran = false;
    while (true) {
        lock();
        const work = head orelse {
            mu.unlock();
            if (serveFollower()) return true;
            if (!follower_pending.load(.acquire)) return ran;
            std.Io.sleep(std.Io.Threaded.global_single_threaded.io(), .fromMilliseconds(1), .real) catch {};
            continue;
        };
        head = work.next;
        if (head == null) tail = null;
        pending.store(head != null, .release);
        mu.unlock();
        work.run();
        complete(work);
        ran = true;
    }
}
