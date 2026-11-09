// dec_moe_fp4_up_rg: dec_moe_fp4_up_bw's sums with an expert several rows picked read once. One simdgroup per (output
// column, slot), as dec_moe_fp4_up_bw: of the slots that picked one expert the first (row major) decodes each unit of
// its gate and up codes once and runs dec_fp4_up_pair's sums over every row that picked it, the same fmas in the same
// order a row, then each row's simd_sum and activation; the other slots return. The shared expert's slot is
// dec_moe_fp4_up_bw's.
const uint lane = thread_index_in_simdgroup;
const uint sg = thread_position_in_grid.x / 32;
const uint u = sg % uint(TOPK * R + 1);
const uint n_col = sg / uint(TOPK * R + 1);
if (n_col >= uint(N)) return;
if (u == uint(TOPK * R)) {
  float acc[R][2];
  dec_up_shared<R, K, N, T>(sw, sscales, n_col, x, lane, acc);
  if (lane == 0) {
#pragma unroll
    for (int r = 0; r < R; ++r) out[(r * (TOPK + 1) + TOPK) * N + n_col] = dec_up_act<T>(acc[r][0], acc[r][1], lim[0], 1.0f);
  }
  return;
}
const uint e = inds[u];
for (uint s = 0; s < u; ++s) {
  if (inds[s] == e) return;
}
uint slots[R];
int m = 0;
for (uint s = u; s < uint(TOPK * R); ++s) {
  if (inds[s] == e) slots[m++] = s;
}
constexpr uint KW = K / 8, KG = K / 32, UPL = K / 1024;
const size_t row = size_t(e) * (2 * N) + n_col;
const device uint4* w0 = (const device uint4*)(w + row * KW) + lane;
const device uint4* w1 = (const device uint4*)(w + (row + N) * KW) + lane;
const device uint8_t* s0 = scales + row * KG + lane;
const device uint8_t* s1 = scales + (row + N) * KG + lane;
float acc[R][2];
for (int i = 0; i < m; ++i) acc[i][0] = acc[i][1] = 0.0f;
#pragma unroll
for (uint k = 0; k < UPL; ++k) {
  const uint4 q[2] = {w0[32 * k], w1[32 * k]};
  const float f0 = dec_fp4_scale(uint(s0[32 * k])), f1 = dec_fp4_scale(uint(s1[32 * k]));
  for (int i = 0; i < m; ++i) {
    float d[2];
    dec_fp4_dots<2>(q, (const device uint4*)(x + size_t(slots[i] / uint(TOPK)) * K) + (32 * k + lane) * 4, d);
    acc[i][0] = fma(f0, d[0], acc[i][0]);
    acc[i][1] = fma(f1, d[1], acc[i][1]);
  }
}
for (int i = 0; i < m; ++i) {
  const float g = simd_sum(acc[i][0]);
  const float v = simd_sum(acc[i][1]);
  const uint s = slots[i];
  if (lane == 0) out[(s / uint(TOPK) * (TOPK + 1) + s % uint(TOPK)) * N + n_col] = dec_up_act<T>(g, v, lim[0], wts[s]);
}
