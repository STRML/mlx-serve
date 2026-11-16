#include <mlx/config.h>

extern "C" int msv_test_swap_tf32(int enabled) {
    const int previous = mlx::core::config::get("MLX_ENABLE_TF32", 1);
    mlx::core::config::update("MLX_ENABLE_TF32", enabled);
    return previous;
}
