const std = @import("std");
const builtin = @import("builtin");

const LOCK_EX: c_int = 2;
const LOCK_NB: c_int = 4;
extern "c" fn flock(fd: c_int, operation: c_int) c_int;
extern "c" fn close(fd: c_int) c_int;
extern "c" fn unlink(path: [*:0]const u8) c_int;

pub const default_path = "/private/tmp/mlx-serve-load.lock";

pub fn pathFromOverride(override: ?[]const u8) []const u8 {
    return override orelse default_path;
}

pub const Lock = struct {
    fd: c_int = -1,

    pub fn acquire(path: []const u8) !Lock {
        if (comptime builtin.os.tag != .macos) return .{};
        const zpath = try std.heap.page_allocator.dupeZ(u8, path);
        defer std.heap.page_allocator.free(zpath);
        const fd = std.c.open(zpath, .{ .ACCMODE = .RDWR, .CREAT = true }, @as(std.c.mode_t, 0o600));
        if (fd < 0) return error.LoadLockOpenFailed;
        errdefer _ = close(fd);
        if (flock(fd, LOCK_EX | LOCK_NB) != 0) {
            std.log.info("[load-lock] waiting for another mlx-serve load\n", .{});
            if (flock(fd, LOCK_EX) != 0) return error.LoadLockAcquireFailed;
            std.log.info("[load-lock] acquired after waiting\n", .{});
        }
        return .{ .fd = fd };
    }

    pub fn release(self: *Lock) void {
        if (self.fd >= 0) _ = close(self.fd);
        self.fd = -1;
    }
};

test "load lock path uses the override when set" {
    try std.testing.expectEqualStrings("/custom/load.lock", pathFromOverride("/custom/load.lock"));
    try std.testing.expectEqualStrings(default_path, pathFromOverride(null));
}

test "separate load lock file descriptions serialize concurrent holders" {
    if (comptime builtin.os.tag != .macos) return error.SkipZigTest;
    const path = try std.fmt.allocPrintSentinel(std.testing.allocator, ".mlx-serve-load-lock-test-{d}", .{std.c.getpid()}, 0);
    defer std.testing.allocator.free(path);
    defer _ = unlink(path);

    const first_fd = try openAt(path);
    var first = Lock{ .fd = first_fd };
    defer first.release();
    const second_fd = try openAt(path);
    var second_fd_owned = true;
    errdefer {
        if (second_fd_owned) _ = close(second_fd);
    }
    if (flock(first_fd, LOCK_EX) != 0) return error.LoadLockAcquireFailed;
    var second_contended = std.atomic.Value(bool).init(false);
    var second_acquired = std.atomic.Value(bool).init(false);
    const thread = try std.Thread.spawn(.{}, acquireSecond, .{ second_fd, &second_contended, &second_acquired });
    second_fd_owned = false;
    var wait_ms: u8 = 0;
    while (!second_contended.load(.acquire) and !second_acquired.load(.acquire) and wait_ms < 100) : (wait_ms += 1) {
        std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
    }
    const contended = second_contended.load(.acquire);
    first.release();
    thread.join();
    try std.testing.expect(contended);
    try std.testing.expect(second_acquired.load(.acquire));
}

fn openAt(path: [:0]const u8) !c_int {
    const fd = std.c.open(path, .{ .ACCMODE = .RDWR, .CREAT = true }, @as(std.c.mode_t, 0o600));
    if (fd < 0) return error.LoadLockOpenFailed;
    return fd;
}

fn acquireSecond(fd: c_int, contended: *std.atomic.Value(bool), acquired: *std.atomic.Value(bool)) void {
    defer _ = close(fd);
    if (flock(fd, LOCK_EX | LOCK_NB) == 0) {
        acquired.store(true, .release);
        return;
    }
    contended.store(true, .release);
    if (flock(fd, LOCK_EX) != 0) return;
    acquired.store(true, .release);
}
