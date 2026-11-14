//! Pure GPU memory ceiling term calculation.

const std = @import("std");

pub const Binding = enum { working_set, footprint_plus_free, wired_floor };

pub const Components = struct {
    working_set_limit: u64,
    mlx_footprint: u64,
    free_system: u64,
    wired_floor: u64,
    ane_bytes: u64,
    ceiling: u64,
    binds: Binding,
};

pub fn calculate(
    working_set_limit: u64,
    mlx_footprint: u64,
    free_system: u64,
    wired_floor: u64,
    ane_bytes: u64,
) Components {
    const physical = @min(working_set_limit, mlx_footprint +| free_system);
    const binding: Binding = if (wired_floor > physical)
        .wired_floor
    else if (working_set_limit <= mlx_footprint +| free_system)
        .working_set
    else
        .footprint_plus_free;
    return .{
        .working_set_limit = working_set_limit,
        .mlx_footprint = mlx_footprint,
        .free_system = free_system,
        .wired_floor = wired_floor,
        .ane_bytes = ane_bytes,
        .ceiling = @max(physical, wired_floor) -| ane_bytes,
        .binds = binding,
    };
}

test "memory ceiling components report the binding term" {
    const t = std.testing;
    try t.expectEqual(Binding.working_set, calculate(100, 30, 80, 0, 0).binds);
    try t.expectEqual(Binding.footprint_plus_free, calculate(100, 30, 20, 0, 0).binds);
    try t.expectEqual(Binding.wired_floor, calculate(100, 30, 20, 80, 0).binds);
}
