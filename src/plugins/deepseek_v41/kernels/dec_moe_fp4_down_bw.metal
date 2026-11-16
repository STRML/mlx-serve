// dec_moe_fp4_down_bw: dec_moe_fp4_down's sums over twice the columns a threadgroup, the R rows of a column group
// adjacent in the grid (threadgroup t is row t % R of columns t / R * 2 NC ..) so the rows' reads of the shared expert
// and of the experts they share come from the cache. Simdgroup j < TOPK reads the row's j-th pick for both column
// halves (dec_fp4_down_rows); simdgroups TOPK and TOPK + 1 the shared expert (twice the bytes a unit), a half each.
const uint t = threadgroup_position_in_grid.x;
const uint r = t % uint(R);
const uint d0 = t / uint(R) * (2 * NC);
const uint sgi = simdgroup_index_in_threadgroup;
const uint lane = thread_index_in_simdgroup;
threadgroup float part[2 * NC * (TOPK + 1)];
const uint slot = metal::min(sgi, uint(TOPK));
dec_fp4_down_slot<K, N, NC, TOPK>(w, scales, sw, sscales, sgi < uint(TOPK) ? inds[r * TOPK + sgi] : 0u, d0, sgi, lane, (const device uint4*)(x + size_t(r * (TOPK + 1) + slot) * K), part);
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint i = sgi * 32 + lane; i < uint(2 * NC * HC); i += 32 * (TOPK + 2)) {
  const uint cc = i / HC;
  const uint h = i % HC;
  const uint d = d0 + cc;
  const float y = float(T(dec_fp4_down_sum<TOPK>(part, cc)));
  float a = 0.0f;
  for (int j = 0; j < HC; ++j) a = metal::fma(comb[(r * HC + j) * HC + h], float(res[(r * HC + j) * N + d]), a);
  out[(r * HC + h) * N + d] = T(metal::fma(post[r * HC + h], y, a));
}
