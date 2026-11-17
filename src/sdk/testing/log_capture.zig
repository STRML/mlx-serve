//! Capture the real log sink without writing test diagnostics to stderr.
const std = @import("std");
const log = @import("../util/log.zig");

pub fn capture(a: std.mem.Allocator, comptime run: anytype, args: anytype) ![]u8 {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var root_buf: [1024]u8 = undefined;
    const root = root_buf[0..try tmp.dir.realPath(std.testing.io, &root_buf)];
    var path_buf: [1100]u8 = undefined;
    const path = try std.fmt.bufPrint(&path_buf, "{s}/request.log", .{root});
    try log.openFile(path, 0);
    defer log.closeFile();
    @call(.auto, run, args);
    log.syncFile();
    return tmp.dir.readFileAlloc(std.testing.io, "request.log", a, .limited(1 << 20));
}
