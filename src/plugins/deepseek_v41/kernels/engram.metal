#pragma clang fp contract(off)
const uint t = thread_index_in_threadgroup, lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
const int TG = int(threads_per_threadgroup.x), l = int(threadgroup_position_in_grid.y);
const int nsg = TG / 32;
const int HC = qk_shape[0], D = qk_shape[1];
const size_t x_base = size_t(l) * HC * D;
const size_t k_base = size_t(l) * (HC + 1) * D;
const size_t vo = k_base + size_t(HC) * D;
constexpr int MAX_HC = 8;
constexpr int MAX_ND = 8;
threadgroup float part[MAX_HC][3][8];
threadgroup float gate_s[MAX_HC];

const bool fast_vec = (D % 4 == 0) && (HC <= MAX_HC);
if (fast_vec) {
  const int nd = (D + TG * 4 - 1) / (TG * 4);
  const int c0_base = int(t) * 4;
  using XT4 = metal::vec<OT, 4>;
  using KVT4 = metal::vec<KT, 4>;
  KVT4 vr[MAX_ND];
#if CACHE_X
  XT4 xr[MAX_HC][MAX_ND];
#endif
  for (int b = 0; b < nd; ++b) {
    const int c = b * TG * 4 + c0_base;
    if (b < MAX_ND && c < D) vr[b] = *(const device KVT4*)(kv + vo + c);
  }
  for (int j = 0; j < HC; ++j) {
    const size_t xo = x_base + size_t(j) * D;
    const size_t ko = k_base + size_t(j) * D;
    const device float* qkj = qk + size_t(j) * D;
    float sh = 0.0f, sk = 0.0f, sd = 0.0f;
    for (int b = 0; b < nd; ++b) {
      const int c = b * TG * 4 + c0_base;
      if (c < D) {
        const XT4 hv4 = *(const device XT4*)(x + xo + c);
#if CACHE_X
        if (b < MAX_ND) xr[j][b] = hv4;
#endif
        const KVT4 kv4 = *(const device KVT4*)(kv + ko + c);
        const float4 qk4 = *(const device float4*)(qkj + c);
        #pragma clang loop unroll(full)
        for (int i = 0; i < 4; ++i) {
          const float hv = float(hv4[i]), kk = float(kv4[i]);
          const float h2 = hv * hv, k2 = kk * kk, hq = hv * qk4[i];
          const float p = hq * kk;
          sh = h2 + sh;
          sk = k2 + sk;
          sd = p + sd;
        }
      }
    }
    if (TG > 1) {
      sh = metal::simd_sum(sh);
      sk = metal::simd_sum(sk);
      sd = metal::simd_sum(sd);
    }
    if (lane == 0) {
      part[j][0][sg] = sh;
      part[j][1][sg] = sk;
      part[j][2][sg] = sd;
    }
  }
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  for (int j = int(sg); j < HC; j += nsg) {
    const bool own = int(lane) < nsg;
    const float sh = metal::simd_sum(own ? part[j][0][lane] : 0.0f);
    const float sk = metal::simd_sum(own ? part[j][1][lane] : 0.0f);
    const float sd = metal::simd_sum(own ? part[j][2][lane] : 0.0f);
    if (lane == 0) {
      const float mh = sh * cst[0], mk = sk * cst[0];
      const float eh = mh + cst[1], ek = mk + cst[1];
      const float rh = metal::precise::rsqrt(eh), rk = metal::precise::rsqrt(ek);
      const float rr = rh * rk;
      const float d1 = sd * rr;
      const float dot = d1 * cst[2];
      const float ad = metal::abs(dot);
      const float mg = metal::precise::sqrt(metal::isnan(ad) || ad > 1e-6f ? ad : 1e-6f);
      const float s = dot < 0.0f ? -mg : mg;
      const float y = 1.0f / (1.0f + metal::precise::exp(metal::abs(s)));
      gate_s[j] = s < 0.0f ? y : 1.0f - y;
    }
  }
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  for (int j = 0; j < HC; ++j) {
    const float g = gate_s[j];
    const size_t xo = x_base + size_t(j) * D;
    for (int b = 0; b < nd; ++b) {
      const int c = b * TG * 4 + c0_base;
      if (c < D) {
#if CACHE_X
        const float4 hv4 = float4(b < MAX_ND ? xr[j][b] : *(const device XT4*)(x + xo + c));
#else
        const float4 hv4 = float4(*(const device XT4*)(x + xo + c));
#endif
        const float4 vv4 = float4(b < MAX_ND ? vr[b] : *(const device KVT4*)(kv + vo + c));
        *(device XT4*)(out + xo + c) = XT4(hv4 + g * vv4);
      }
    }
  }
  return;
}

for (int j = 0; j < HC; ++j) {
  const size_t xo = x_base + size_t(j) * D;
  const size_t ko = k_base + size_t(j) * D;
  float sh = 0.0f, sk = 0.0f, sd = 0.0f;
  for (int c = int(t); c < D; c += TG) {
    const float hv = float(x[xo + c]), kk = float(kv[ko + c]);
    const float h2 = hv * hv, k2 = kk * kk, hq = hv * qk[size_t(j) * D + c];
    const float p = hq * kk;
    sh = h2 + sh;
    sk = k2 + sk;
    sd = p + sd;
  }
  if (TG > 1) {
    sh = metal::simd_sum(sh);
    sk = metal::simd_sum(sk);
    sd = metal::simd_sum(sd);
  }
  if (TG > 32) {
    if (lane == 0) { part[0][0][sg] = sh; part[0][1][sg] = sk; part[0][2][sg] = sd; }
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    const bool own = int(t) < nsg;
    sh = metal::simd_sum(own ? part[0][0][t] : 0.0f);
    sk = metal::simd_sum(own ? part[0][1][t] : 0.0f);
    sd = metal::simd_sum(own ? part[0][2][t] : 0.0f);
  }
  if (t == 0) {
    const float mh = sh * cst[0], mk = sk * cst[0];
    const float eh = mh + cst[1], ek = mk + cst[1];
    const float rh = metal::precise::rsqrt(eh), rk = metal::precise::rsqrt(ek);
    const float rr = rh * rk;
    const float d1 = sd * rr;
    const float dot = d1 * cst[2];
    const float ad = metal::abs(dot);
    const float mg = metal::precise::sqrt(metal::isnan(ad) || ad > 1e-6f ? ad : 1e-6f);
    const float s = dot < 0.0f ? -mg : mg;
    const float y = 1.0f / (1.0f + metal::precise::exp(metal::abs(s)));
    gate_s[0] = s < 0.0f ? y : 1.0f - y;
  }
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  const float g = gate_s[0];
  for (int c = int(t); c < D; c += TG) {
    const float gv = g * float(kv[vo + c]);
    out[xo + c] = OT(float(x[xo + c]) + gv);
  }
  if (j + 1 < HC) threadgroup_barrier(metal::mem_flags::mem_threadgroup);
}
