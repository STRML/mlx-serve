const uint lane = thread_index_in_simdgroup;
const uint sg = thread_position_in_grid.x / 32;
constexpr uint NS = PART == 1 ? 1u : PART == 2 ? uint(TOPK * R) : uint(TOPK * R + 1);
const uint u = (PART == 1 ? uint(TOPK * R) : 0u) + sg % NS;
const uint n_col = sg / NS;
if (n_col >= uint(N)) return;
if (u == uint(TOPK * R)) {
  float acc[R][2];
  dec_up_shared<R, K, N, T>(sw, sscales, n_col, x, lane, acc);
  if (lane == 0) {
#pragma unroll
    for (int r = 0; r < R; ++r) out[(PART == 1 ? r : r * (TOPK + 1) + TOPK) * N + n_col] = dec_up_act<T>(acc[r][0], acc[r][1], lim[0], 1.0f);
  }
  return;
}
const uint r = u / TOPK;
float acc[2];
dec_fp4_up_pair<K, N, T>(w, scales, size_t(inds[u]) * (2 * N) + n_col, x + size_t(r) * K, lane, acc);
if (lane == 0) out[(PART == 2 ? r * TOPK + u % TOPK : r * (TOPK + 1) + u % TOPK) * N + n_col] = dec_up_act<T>(acc[0], acc[1], lim[0], wts[u]);
