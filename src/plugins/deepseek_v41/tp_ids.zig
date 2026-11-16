//! The ids rank 0 gives snapshots for the other tensor ranks, across the groups a process forms and drops.
const std = @import("std");

pub const TpIds = struct {
    /// The last id handed out. Never reset: an id from a dropped group cannot name a snapshot of the next.
    last: u64 = 0,
    /// `last` when the current group formed. The other ranks' copies die with their group, so a snapshot
    /// whose id is at or below this (a TP1-era snapshot has 0) has no copy in this one.
    formed_at: u64 = 0,

    pub fn next(self: *TpIds) u64 {
        self.last += 1;
        return self.last;
    }

    pub fn formed(self: *TpIds) void {
        self.formed_at = self.last;
    }

    /// True: the other ranks of the current group hold no copy of the snapshot named `id`.
    pub fn stale(self: TpIds, id: u64) bool {
        return id <= self.formed_at;
    }
};

test "tp ids: an id handed out before a group formed is stale in it, one handed out after is not" {
    var ids: TpIds = .{};
    const old = ids.next();
    try std.testing.expect(ids.stale(0));
    ids.formed();
    try std.testing.expect(ids.stale(0));
    try std.testing.expect(ids.stale(old));
    const fresh = ids.next();
    try std.testing.expect(!ids.stale(fresh));
}

test "tp ids: a second group makes the first group's ids stale and never reuses one" {
    var ids: TpIds = .{};
    ids.formed();
    const a = ids.next();
    const b = ids.next();
    try std.testing.expect(!ids.stale(a) and !ids.stale(b));
    ids.formed();
    try std.testing.expect(ids.stale(a) and ids.stale(b));
    const c = ids.next();
    try std.testing.expect(c > b and !ids.stale(c));
}
