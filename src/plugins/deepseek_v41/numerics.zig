//! DeepSeek-V4.1's env levers that change the prefill or decode numerics, resolved as its kernels read them.

const env = @import("../../env.zig");
const flash_plan = @import("prompt/flash_plan.zig");
const seg = @import("decode/seg.zig");

pub fn attnF16() bool {
    return if (env.get(.MLX_SERVE_DSV41_ATTN_F16)) |v| v[0] != '0' else true;
}

pub fn ced() bool {
    return if (env.get(.MLX_SERVE_DSV41_CED)) |v| v[0] != '0' else true;
}

/// Null when the setting is invalid, which refuses the load.
pub fn flashAttn() ?flash_plan.Mode {
    return flash_plan.parse(env.span(.MLX_SERVE_DSV41_FLASH_ATTN)) catch null;
}

pub const Resolved = struct { attn_f16: bool, ced: bool, flash_attn: ?flash_plan.Mode, fused_seg: seg.Level };

pub fn resolved() Resolved {
    return .{ .attn_f16 = attnF16(), .ced = ced(), .flash_attn = flashAttn(), .fused_seg = seg.levelFromEnv() };
}
