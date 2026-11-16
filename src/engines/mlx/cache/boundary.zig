//! A prompt boundary's identity: its prefix key, its tokens and the images inside them. The key alone is zero for every
//! text-only prompt of some plugins, so two different prompts of one length share it, and a plugin's key may not cover
//! the pixels of an image its tokens only name.

const std = @import("std");
const MediaSpan = @import("../../../sdk/kv/media_span.zig").MediaSpan;

pub fn id(prefix_key: u64, tokens: []const u32, media: []const MediaSpan) u64 {
    var h = std.hash.Wyhash.init(prefix_key);
    h.update(std.mem.sliceAsBytes(tokens));
    h.update(std.mem.asBytes(&@as(u64, tokens.len)));
    for (media) |span| {
        h.update(std.mem.asBytes(&@as(u64, span.start)));
        h.update(std.mem.asBytes(&span.key));
    }
    return h.final();
}

test "boundary id: equal-length prompts under one key differ, and a key, a length or an image changes it" {
    const a = id(0, &.{ 1, 2, 3, 4 }, &.{});
    try std.testing.expect(a != id(0, &.{ 5, 6, 7, 8 }, &.{}));
    try std.testing.expect(a != id(9, &.{ 1, 2, 3, 4 }, &.{}));
    try std.testing.expect(a != id(0, &.{ 1, 2, 3 }, &.{}));
    try std.testing.expectEqual(a, id(0, &.{ 1, 2, 3, 4 }, &.{}));
    const first = [_]MediaSpan{.{ .start = 1, .key = 7 }};
    const second = [_]MediaSpan{.{ .start = 1, .key = 8 }};
    const moved = [_]MediaSpan{.{ .start = 2, .key = 7 }};
    const with_first = id(0, &.{ 1, 2, 3, 4 }, &first);
    try std.testing.expect(a != with_first);
    try std.testing.expect(with_first != id(0, &.{ 1, 2, 3, 4 }, &second));
    try std.testing.expect(with_first != id(0, &.{ 1, 2, 3, 4 }, &moved));
}
