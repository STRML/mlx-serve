// dec_moe_fp4_down_rg: dec_moe_fp4_down_bw's routed and shared down sums with an expert several rows picked read once,
// each (row, slot)'s column sums written to its fixed f32 place in `out` [R, TOPK + 1, N] for dec_moe_fp4_combine_rg.
// Threadgroup t (one simdgroup) is slot u = t % (TOPK * R + R) of columns t / (TOPK * R + R) * 2 NC .. + 2 NC: below
// TOPK * R the (row, pick) pairs, where the first slot that picked an expert reads each unit of its two column halves
// once and runs dec_fp4_down_rows' sums for every row that picked it (the other slots return); TOPK * R + r is row r's
// shared expert, both halves. Every sum is dec_moe_fp4_down_bw's, fma for fma, shuffle for shuffle.
const uint S = uint(TOPK * R + R);
const uint t = threadgroup_position_in_grid.x;
const uint u = t % S;
const uint d0 = t / S * (2 * NC);
const uint lane = thread_index_in_simdgroup;
const uint c = lane % NC;
if (u >= uint(TOPK * R)) {
  const uint r = u - uint(TOPK * R);
  for (uint hf = 0; hf < 2; ++hf) {
    float a = dec_fp4_down_shared<K, T>(sw, sscales, d0 + hf * NC + c, x + size_t(r * (TOPK + 1) + TOPK) * K, lane / NC, 32 / NC);
#pragma unroll
    for (uint o = 16; o >= uint(NC); o >>= 1) a += simd_shuffle_xor(a, o);
    if (lane < uint(NC)) out[(r * (TOPK + 1) + TOPK) * N + d0 + hf * NC + lane] = a;
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
constexpr uint NU = K / 32;
const size_t row = size_t(e) * N + d0 + c;
float acc[R][2];
for (int j = 0; j < m; ++j) acc[j][0] = acc[j][1] = 0.0f;
for (uint v = lane / NC; v < NU; v += 32 / NC) {
  uint4 q[2];
  float f[2];
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    q[i] = ((const device uint4*)(w + (row + i * NC) * (K / 8)))[v];
    f[i] = dec_fp4_scale(uint(scales[(row + i * NC) * (K / 32) + v]));
  }
  for (int j = 0; j < m; ++j) {
    const uint s = slots[j];
    float d[2];
    dec_fp4_dots<2>(q, (const device uint4*)(x + size_t(s / uint(TOPK) * (TOPK + 1) + s % uint(TOPK)) * K) + v * 4, d);
#pragma unroll
    for (int i = 0; i < 2; ++i) acc[j][i] = fma(f[i], d[i], acc[j][i]);
  }
}
for (int j = 0; j < m; ++j) {
  const uint s = slots[j];
#pragma unroll
  for (uint hf = 0; hf < 2; ++hf) {
    float a = acc[j][hf];
#pragma unroll
    for (uint o = 16; o >= uint(NC); o >>= 1) a += simd_shuffle_xor(a, o);
    if (lane < uint(NC)) out[(s / uint(TOPK) * (TOPK + 1) + s % uint(TOPK)) * N + d0 + hf * NC + lane] = a;
  }
}
