// dec_moe_fp4_up_bw: dec_moe_fp4_up's sums with a routed unit's gate and up codes read against one decode of its
// activations (dec_fp4_up_pair). One simdgroup per (output column, slot): slots 0 .. TOPK * R - 1 are the (row, pick)
// pairs, slot TOPK * R the shared expert (every row).
const uint lane = thread_index_in_simdgroup;
const uint sg = thread_position_in_grid.x / 32;
const uint u = sg % uint(TOPK * R + 1);
const uint n_col = sg / uint(TOPK * R + 1);
if (n_col >= uint(N)) return;
if (u == uint(TOPK * R)) {
  T acts[R];
  dec_fp4_up_shared_acts<R, K, N, T>(sw, sscales, n_col, (const device uint4*)x, lim[0], lane, acts);
  if (lane == 0) {
#pragma unroll
    for (int r = 0; r < R; ++r) out[(r * (TOPK + 1) + TOPK) * N + n_col] = acts[r];
  }
  return;
}
const uint r = u / TOPK;
const T act = dec_fp4_up_routed_act<K, N, T>(w, scales, inds[u], n_col, (const device uint4*)(x + size_t(r) * K), lim[0], wts[u], lane);
if (lane == 0) out[(r * (TOPK + 1) + u % TOPK) * N + n_col] = act;
