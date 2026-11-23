// dec_ffn_in, launch 1: y = wo_b(oa) for R rows (mxfp8: e4m3 words, an e8m0 exponent per 32), then each output column's
// hc expand x1 = post * y + comb . x and the collapse hv = pre . x1. A simdgroup owns NR columns across the lanes' K slices.
threadgroup float part_nax[NAX ? (NSG - 1) * 8 * NR : 1];
const uint lane = thread_index_in_simdgroup;
const uint sg = simdgroup_index_in_threadgroup;
const int tg = int(threadgroup_position_in_grid.x);
const int tid = int(thread_index_in_threadgroup);
if constexpr (NAX) {
  if (tg == 0 && tid < 16) ctr[tid] = 0u;
  const short qid = short(lane >> 2);
  const short fm = (qid & 4) | short((lane >> 1) & 3);
  const short f4 = (qid & 2) | short(lane & 1);
  const uint col0 = uint(tg) * NR;
  const bool live = int(fm) < R;
  const device uint4* xp = (const device uint4*)(oa + (live ? (size_t)fm * K : 0) + sg * 64) + f4 * 2;
  dsv41_nax_rows r0;
  for (int j = 0; j < 4; ++j) {
    const uint row = col0 + uint(fm) + 8u * uint(j);
    r0.w[j] = (const device uint4*)(w_q + (size_t)row * (K / 4)) + sg * 4 + f4;
    r0.s[j] = scales + (size_t)row * (K / 32) + sg * 2 + (f4 >> 1);
  }
  const dsv41_nax_first f0 = dsv41_nax_head(r0);
  float acc[16] = {0};
  dsv41_nax_mxfp8<K / (NSG * 64), NSG * 4, NSG * 2, NSG * 8>(acc, r0, f0, xp, live);
  if (sg > 0 && live) {
    for (int i = 0; i < 4; ++i) {
      part_nax[((sg - 1) * 8 + uint(fm)) * NR + uint(f4) * 4u + i] = acc[i];
      part_nax[((sg - 1) * 8 + uint(fm)) * NR + uint(f4) * 4u + 16u + i] = acc[8 + i];
    }
  }
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  if (sg == 0 && live) {
    const int r = int(fm);
    const auto po = spost + r * HC;
    const auto cb = scomb + r * HC * HC;
    const auto pr = spre + r * HC;
    for (int half_i = 0; half_i < 2; ++half_i) for (int i = 0; i < 4; ++i) {
      float sum = acc[half_i * 8 + i];
      for (int s = 1; s < NSG; ++s) sum += part_nax[((s - 1) * 8 + uint(r)) * NR + uint(f4) * 4u + uint(half_i * 16 + i)];
      const int col = int(col0) + int(f4) * 4 + half_i * 16 + i;
      const float yv = float(T(sum));
      float h = 0.0f;
      for (int a = 0; a < HC; ++a) {
        float ax = 0.0f;
        for (int b = 0; b < HC; ++b) ax = metal::fma(cb[b * HC + a], float(x[((size_t)r * HC + b) * D + col]), ax);
        const T o = T(metal::fma(po[a], yv, ax));
        x1[((size_t)r * HC + a) * D + col] = o;
        h = metal::fma(pr[a], float(o), h);
      }
      hv[(size_t)r * D + col] = T(h);
    }
  }
  return;
}
const int nb = tg * (NSG * NR) + int(sg) * NR;
constexpr int KW = K / 4, KG = K / 32, STEPS = K / 512;
const device uint4* wr[NR];
const device uchar* sr[NR];
for (int j = 0; j < NR; ++j) {
  wr[j] = (const device uint4*)(w_q + (size_t)(nb + j) * KW);
  sr[j] = scales + (size_t)(nb + j) * KG;
}
// The epilogue's stream columns, loaded before the K loop.
float sv[HC];
if (tid < R * NSG * NR) {
  const int r = tid / (NSG * NR), c = tid % (NSG * NR);
  for (int j = 0; j < HC; ++j) sv[j] = float(x[((size_t)r * HC + j) * D + tg * (NSG * NR) + c]);
}
float acc[NR][R];
for (int j = 0; j < NR; ++j) for (int r = 0; r < R; ++r) acc[j][r] = 0.0f;
for (int s0 = 0; s0 < STEPS; s0 += U) {
  uint4 q[U][NR];
  uint e[U][NR];
  for (int u = 0; u < U; ++u) for (int j = 0; j < NR; ++j) {
    q[u][j] = wr[j][(s0 + u) * 32 + lane];
    e[u][j] = uint(sr[j][(s0 + u) * 16 + lane / 2]);
  }
  for (int u = 0; u < U; ++u) {
    const int s = s0 + u;
    float wv[NR][16], sc[NR];
    for (int j = 0; j < NR; ++j) {
      // e8m0 as 2^(e - 127), times the 2^8 the e4m3 bit placement leaves out.
      sc[j] = as_type<float>(e[u][j] == 0u ? 0x400000u : (e[u][j] << 23)) * 256.0f;
      const uint words[4] = {q[u][j].x, q[u][j].y, q[u][j].z, q[u][j].w};
      for (int i = 0; i < 4; ++i) {
        const float2 lo = mlxserve_e4m3x2(words[i]);
        const float2 hi = mlxserve_e4m3x2(words[i] >> 8);
        wv[j][4 * i + 0] = lo.x;
        wv[j][4 * i + 1] = hi.x;
        wv[j][4 * i + 2] = lo.y;
        wv[j][4 * i + 3] = hi.y;
      }
    }
    for (int r = 0; r < R; ++r) {
      const device uint4* xp = (const device uint4*)(oa + (size_t)r * K) + s * 64 + lane * 2;
      const uint4 a = xp[0], b = xp[1];
      const uint wd[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
      float xf[16];
      for (int i = 0; i < 8; ++i) {
        xf[2 * i] = as_type<float>(wd[i] << 16);
        xf[2 * i + 1] = as_type<float>(wd[i] & 0xffff0000u);
      }
      for (int j = 0; j < NR; ++j) {
        float d = 0.0f;
        for (int i = 0; i < 16; ++i) d += xf[i] * wv[j][i];
        acc[j][r] += d * sc[j];
      }
    }
  }
}
threadgroup float ys[NAX ? 1 : R * NSG * NR];
for (int j = 0; j < NR; ++j) for (int r = 0; r < R; ++r) {
  const float v = metal::simd_sum(acc[j][r]);
  if (lane == 0) ys[r * (NSG * NR) + int(sg) * NR + j] = v;
}
threadgroup_barrier(metal::mem_flags::mem_threadgroup);
// The arrival counters of launch 2.
if (tg == 0 && tid < 16) ctr[tid] = 0u;
if (tid < R * NSG * NR) {
  const int r = tid / (NSG * NR), c = tid % (NSG * NR);
  const int n = tg * (NSG * NR) + c;
  const float yv = float(T(ys[tid]));
  const auto po = spost + r * HC;
  const auto cb = scomb + r * HC * HC;
  const auto pr = spre + r * HC;
  float h = 0.0f;
  for (int i = 0; i < HC; ++i) {
    float a = 0.0f;
    for (int j = 0; j < HC; ++j) a = metal::fma(cb[j * HC + i], sv[j], a);
    const T o = T(metal::fma(po[i], yv, a));
    x1[((size_t)r * HC + i) * D + n] = o;
    h = metal::fma(pr[i], float(o), h);
  }
  hv[(size_t)r * D + n] = T(h);
}
