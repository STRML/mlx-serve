const std = @import("std");

pub const Values = struct {
    path: []const u8,
    n_tokens: usize,
    entry_bytes: u64,
    estimate: u64,
    mlx_peak: ?u64,
    peak_over_active_before: ?u64,
    active_before: u64,
    active_after: ?u64,
    cache_before: u64,
    cache_after: ?u64,
    footprint_before: u64,
    footprint_after: ?u64,
    footprint_peak: ?u64,
    loaded_bytes: u64,
    destination_bytes: u64,
    ms: u64,
    footprint_before_body: ?u64 = null,
    body_bytes: ?u64 = null,
    refused: ?[]const u8 = null,
};

pub fn format(buf: []u8, v: Values) ![]const u8 {
    var b: Builder = .{ .buf = buf };
    try b.append("[import-peak] path={s} n_tokens={d} entry_bytes={d} estimate={d} mlx_peak=", .{ v.path, v.n_tokens, v.entry_bytes, v.estimate });
    try b.optional(v.mlx_peak);
    try b.append(" peak_over_active_before=", .{});
    try b.optional(v.peak_over_active_before);
    try b.append(" active_before={d} active_after=", .{v.active_before});
    try b.optional(v.active_after);
    try b.append(" cache_before={d} cache_after=", .{v.cache_before});
    try b.optional(v.cache_after);
    try b.append(" footprint_before={d} footprint_after=", .{v.footprint_before});
    try b.optional(v.footprint_after);
    try b.append(" footprint_peak=", .{});
    try b.optional(v.footprint_peak);
    try b.append(" loaded_bytes={d} destination_bytes={d} ms={d}", .{ v.loaded_bytes, v.destination_bytes, v.ms });
    if (std.mem.eql(u8, v.path, "peer")) {
        try b.append(" footprint_before_body=", .{});
        try b.optional(v.footprint_before_body);
        try b.append(" body_bytes=", .{});
        try b.optional(v.body_bytes);
    } else {
        try b.append(" footprint_before_body=NA body_bytes=NA", .{});
    }
    if (v.refused) |reason| try b.append(" refused={s}", .{reason});
    return b.buf[0..b.used];
}

const Builder = struct {
    buf: []u8,
    used: usize = 0,

    fn append(self: *Builder, comptime fmt: []const u8, args: anytype) !void {
        const out = try std.fmt.bufPrint(self.buf[self.used..], fmt, args);
        self.used += out.len;
    }

    fn optional(self: *Builder, value: ?u64) !void {
        if (value) |n| return self.append("{d}", .{n});
        try self.append("NA", .{});
    }
};

test "format renders one measured import peak line" {
    var buf: [1024]u8 = undefined;
    const line = try format(&buf, .{
        .path = "ssd",
        .n_tokens = 1_000_000,
        .entry_bytes = 3_367_160_524,
        .estimate = 17_664_435_168,
        .mlx_peak = 8_589_934_592,
        .peak_over_active_before = 2_147_483_648,
        .active_before = 6_442_450_944,
        .active_after = 7_516_192_768,
        .cache_before = 536_870_912,
        .cache_after = 268_435_456,
        .footprint_before = 80_000_000_000,
        .footprint_after = 82_000_000_000,
        .footprint_peak = 83_000_000_000,
        .loaded_bytes = 3_000_000_000,
        .destination_bytes = 2_000_000_000,
        .ms = 12_345,
    });
    try std.testing.expectEqualStrings("[import-peak] path=ssd n_tokens=1000000 entry_bytes=3367160524 estimate=17664435168 mlx_peak=8589934592 peak_over_active_before=2147483648 active_before=6442450944 active_after=7516192768 cache_before=536870912 cache_after=268435456 footprint_before=80000000000 footprint_after=82000000000 footprint_peak=83000000000 loaded_bytes=3000000000 destination_bytes=2000000000 ms=12345 footprint_before_body=NA body_bytes=NA", line);
}

test "format marks an admission refusal without a measured MLX peak" {
    var buf: [1024]u8 = undefined;
    const line = try format(&buf, .{
        .path = "peer",
        .n_tokens = 8,
        .entry_bytes = 4096,
        .estimate = 16384,
        .mlx_peak = null,
        .peak_over_active_before = null,
        .active_before = 100,
        .active_after = 100,
        .cache_before = 20,
        .cache_after = 20,
        .footprint_before = 1000,
        .footprint_after = 1000,
        .footprint_peak = null,
        .loaded_bytes = 4096,
        .destination_bytes = 2048,
        .ms = 1,
        .footprint_before_body = 900,
        .body_bytes = 8192,
        .refused = "no_memory",
    });
    try std.testing.expectEqualStrings("[import-peak] path=peer n_tokens=8 entry_bytes=4096 estimate=16384 mlx_peak=NA peak_over_active_before=NA active_before=100 active_after=100 cache_before=20 cache_after=20 footprint_before=1000 footprint_after=1000 footprint_peak=NA loaded_bytes=4096 destination_bytes=2048 ms=1 footprint_before_body=900 body_bytes=8192 refused=no_memory", line);
}
