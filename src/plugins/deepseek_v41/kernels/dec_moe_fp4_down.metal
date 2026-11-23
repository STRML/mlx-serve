// dec_moe_fp4_down: the routed (mxfp4) and shared (mxfp8) down projections of R decode rows, the combine and the hc
// expand into the next stream. Grid covers N / NC column tiles across grid.y = 1 so the mxfp8 shared weights and
// repeated routed experts load once per batch instead of R times, while accumulating in exact R = 1 slot order.
const uint d0 = threadgroup_position_in_grid.x * NC;
const uint sgi = simdgroup_index_in_threadgroup;
const uint lane = thread_index_in_simdgroup;
const uint c = lane % NC;
threadgroup float part[R * NC * (TOPK + 1)];
threadgroup uint uniq_e[(R > 1 && R * TOPK <= 64) ? R * TOPK : 1];
threadgroup ulong uniq_mask[(R > 1 && R * TOPK <= 64) ? R * TOPK : 1];
threadgroup uint num_uniq;
if (R == 1) {
  float acc;
  if (sgi == uint(TOPK)) {
    acc = dec_fp4_down_shared<K, T>(sw, sscales, d0 + c, x + size_t(TOPK) * K, lane / NC, 32 / NC);
  } else {
    acc = dec_fp4_down_routed<K, T>(w, scales, size_t(inds[sgi]) * N + d0 + c, x + size_t(sgi) * K, lane / NC, 32 / NC);
  }
#pragma unroll
  for (uint o = 16; o >= uint(NC); o >>= 1) acc += simd_shuffle_xor(acc, o);
  if (lane < uint(NC)) part[lane * (TOPK + 1) + sgi] = acc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sgi == 0 && lane < uint(NC * HC)) {
    const uint cc = lane / HC;
    const uint h = lane % HC;
    const uint d = d0 + cc;
    float s = 0.0f;
    for (int j = 0; j <= TOPK; ++j) s += part[cc * (TOPK + 1) + j];
    const float y = float(T(s));
    float a = 0.0f;
    for (int j = 0; j < HC; ++j) a = metal::fma(comb[j * HC + h], float(res[j * N + d]), a);
    out[h * N + d] = T(metal::fma(post[h], y, a));
  }
  return;
}
if (R * TOPK <= 64) {
  constexpr int PAIRS = R * TOPK;
  if (sgi == 0u) {
    const uint p0 = lane, p1 = lane + 32u;
    const uint e0 = p0 < uint(PAIRS) ? uint(inds[p0]) : 0xFFFFFFFFu;
    const uint e1 = p1 < uint(PAIRS) ? uint(inds[p1]) : 0xFFFFFFFFu;
    ulong m0 = 0ul, m1 = 0ul;
    for (uint q = 0u; q < uint(PAIRS); ++q) {
      const uint eq = uint(inds[q]);
      if (eq == e0) m0 |= 1ul << q;
      if (eq == e1) m1 |= 1ul << q;
    }
    const bool rep0 = p0 < uint(PAIRS) && (m0 & ((1ul << p0) - 1ul)) == 0ul;
    const bool rep1 = p1 < uint(PAIRS) && (m1 & ((1ul << p1) - 1ul)) == 0ul;
    const uint cnt0 = metal::simd_sum(uint(rep0));
    const uint cnt1 = metal::simd_sum(uint(rep1));
    const uint off0 = metal::simd_prefix_exclusive_sum(uint(rep0));
    const uint off1 = cnt0 + metal::simd_prefix_exclusive_sum(uint(rep1));
    if (rep0) { uniq_e[off0] = e0; uniq_mask[off0] = m0; }
    if (rep1) { uniq_e[off1] = e1; uniq_mask[off1] = m1; }
    if (lane == 0u) num_uniq = cnt0 + cnt1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sgi == uint(TOPK)) {
    float acc_s[R];
    dec_fp4_down_shared_multi<R, K, T>(sw, sscales, d0 + c, x, uint(TOPK), lane / NC, 32 / NC, acc_s);
    for (int r = 0; r < R; ++r) {
      float a = acc_s[r];
#pragma unroll
      for (uint o = 16; o >= uint(NC); o >>= 1) a += simd_shuffle_xor(a, o);
      if (lane < uint(NC)) part[(uint(r) * uint(NC) + lane) * uint(TOPK + 1) + uint(TOPK)] = a;
    }
  } else {
    const uint nu = num_uniq;
    for (uint ui = sgi; ui < nu; ui += uint(TOPK)) {
      const uint e = uniq_e[ui];
      ulong mask = uniq_mask[ui];
      while (mask != 0ul) {
        uint ps[R];
        int np = 0;
        for (; np < R && mask != 0ul; ++np) {
          ps[np] = uint(ctz(mask));
          mask &= mask - 1ul;
        }
        float acc_p[R];
        for (int i = 0; i < np; ++i) acc_p[i] = 0.0f;
        dec_fp4_down_routed_multi<K, R, T>(w, scales, size_t(e) * N + d0 + c, x, ps, np, uint(TOPK), lane / NC, 32 / NC, acc_p);
        for (int i = 0; i < np; ++i) {
          float a = acc_p[i];
#pragma unroll
          for (uint o = 16; o >= uint(NC); o >>= 1) a += simd_shuffle_xor(a, o);
          if (lane < uint(NC)) {
            const uint p = ps[i];
            part[((p / uint(TOPK)) * uint(NC) + lane) * uint(TOPK + 1) + (p % uint(TOPK))] = a;
          }
        }
      }
    }
  }
} else {
  if (sgi == uint(TOPK)) {
    float acc_s[R];
    dec_fp4_down_shared_multi<R, K, T>(sw, sscales, d0 + c, x, uint(TOPK), lane / NC, 32 / NC, acc_s);
    for (int r = 0; r < R; ++r) {
      float a = acc_s[r];
#pragma unroll
      for (uint o = 16; o >= uint(NC); o >>= 1) a += simd_shuffle_xor(a, o);
      if (lane < uint(NC)) part[(uint(r) * uint(NC) + lane) * uint(TOPK + 1) + uint(TOPK)] = a;
    }
  } else {
    for (int r = 0; r < R; ++r) {
      float a = dec_fp4_down_routed<K, T>(w, scales, size_t(inds[r * TOPK + int(sgi)]) * N + d0 + c, x + size_t(r * (TOPK + 1) + int(sgi)) * K, lane / NC, 32 / NC);
#pragma unroll
      for (uint o = 16; o >= uint(NC); o >>= 1) a += simd_shuffle_xor(a, o);
      if (lane < uint(NC)) part[(uint(r) * uint(NC) + lane) * uint(TOPK + 1) + sgi] = a;
    }
  }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
for (int r = int(sgi); r < R; r += TOPK + 1) if (lane < uint(NC * HC)) {
  const uint cc = lane / HC;
  const uint h = lane % HC;
  const uint d = d0 + cc;
  float s = 0.0f;
  for (int j = 0; j <= TOPK; ++j) s += part[(uint(r) * uint(NC) + cc) * uint(TOPK + 1) + uint(j)];
  const float y = float(T(s));
  float a = 0.0f;
  for (int j = 0; j < HC; ++j) a = metal::fma(comb[(r * HC + j) * HC + h], float(res[(r * HC + j) * N + d]), a);
  out[(r * HC + h) * N + d] = T(metal::fma(post[r * HC + h], y, a));
}
