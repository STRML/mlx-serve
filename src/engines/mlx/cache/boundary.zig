//! A prompt boundary's identity: its prefix key and its tokens. The key alone is zero for every text-only prompt of
//! some plugins, so two different prompts of one length share it.

const std = @import("std");

pub fn id(prefix_key: u64, tokens: []const u32) u64 {
    var h = std.hash.Wyhash.init(prefix_key);
    h.update(std.mem.sliceAsBytes(tokens));
    h.update(std.mem.asBytes(&@as(u64, tokens.len)));
    return h.final();
}

test "boundary id: equal-length prompts under one key differ, and a key or a length changes it" {
    const a = id(0, &.{ 1, 2, 3, 4 });
    try std.testing.expect(a != id(0, &.{ 5, 6, 7, 8 }));
    try std.testing.expect(a != id(9, &.{ 1, 2, 3, 4 }));
    try std.testing.expect(a != id(0, &.{ 1, 2, 3 }));
    try std.testing.expectEqual(a, id(0, &.{ 1, 2, 3, 4 }));
}
