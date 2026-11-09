constexpr int MIX = (2 + HC) * HC;
constexpr int PL = MIX + 1;
constexpr int NCH = D / 8;
constexpr int NC = D / 512;
constexpr int W = D / P;
constexpr int NRT = G * ((R + RG - 1) / RG);
constexpr int TH = RG * NCH * 4;
constexpr int T0 = TH + 16 * RG;
constexpr int T1 = R * P * PL + R * PL;
constexpr int T2 = 2 * R * E;
constexpr int TGM = (T0 > T1 ? (T0 > T2 ? T0 : T2) : (T1 > T2 ? T1 : T2));
const int tg = int(threadgroup_position_in_grid.x);
const int tid = int(thread_index_in_threadgroup);
const uint lane = thread_index_in_simdgroup;
const int sg = int(simdgroup_index_in_threadgroup);
const uint gen_v = ctl[0];
device atomic_uint* slot = (device atomic_uint*)cnt + ctl[1];
threadgroup float4 tgm4[(TGM + 3) / 4];
threadgroup float* tgm = (threadgroup float*)tgm4;
threadgroup uint last;
const bool is_mix = tg < P;

if (tid == 0 && tg == 0 && gen_v == 0xffffffffu && dep[0] == 0xffffffffu) logits[0] = 0.0f;

if (is_mix) {
  const int d = tg * W + 4 * int(lane);
  const int m0 = sg * (MIX / 8);
  float4 f[MIX / 8][HC];
  for (int a = 0; a < MIX / 8; ++a) for (int i = 0; i < HC; ++i)
    f[a][i] = *(const device float4*)(fn + (size_t)(m0 + a) * (HC * D) + i * D + d);
  #pragma clang loop unroll(disable)
  for (int r = 0; r < R; ++r) {
    const auto po = spost + r * HC;
    const auto cb = scomb + r * HC * HC;
    float sv[HC][4];
    for (int j = 0; j < HC; ++j) {
      const uint2 v = *(const device uint2*)(x + ((size_t)r * HC + j) * D + d);
      sv[j][0] = as_type<float>(v.x << 16);
      sv[j][1] = as_type<float>(v.x & 0xffff0000u);
      sv[j][2] = as_type<float>(v.y << 16);
      sv[j][3] = as_type<float>(v.y & 0xffff0000u);
    }
    const uint2 yw = *(const device uint2*)(y + (size_t)r * D + d);
    const float yv[4] = {as_type<float>(yw.x << 16), as_type<float>(yw.x & 0xffff0000u), as_type<float>(yw.y << 16), as_type<float>(yw.y & 0xffff0000u)};
    float xs[HC * 4];
    uint2 packed[HC];
    for (int i = 0; i < HC; ++i) {
      uint bits[4];
      for (int c = 0; c < 4; ++c) {
        float a = 0.0f;
        for (int j = 0; j < HC; ++j) a = metal::fma(cb[j * HC + i], sv[j][c], a);
        const T o = T(metal::fma(po[i], yv[c], a));
        xs[4 * i + c] = float(o);
        bits[c] = uint(as_type<ushort>(o));
      }
      packed[i] = uint2(bits[0] | (bits[1] << 16), bits[2] | (bits[3] << 16));
    }
    if (sg == 0) for (int i = 0; i < HC; ++i) *(device uint2*)(x1 + ((size_t)r * HC + i) * D + d) = packed[i];
    float dd[MIX / 8] = {0.0f, 0.0f, 0.0f};
    float s2 = 0.0f;
    for (int k = 0; k < HC * 4; ++k) {
      const float v = xs[k];
      s2 = metal::fma(v, v, s2);
      for (int a = 0; a < MIX / 8; ++a) dd[a] = metal::fma(v, f[a][k / 4][k % 4], dd[a]);
    }
    for (int a = 0; a < MIX / 8; ++a) {
      const float t = metal::simd_sum(dd[a]);
      if (lane == 0) parts[((size_t)r * P + tg) * PL + m0 + a] = t;
    }
    if (sg == 0) {
      const float t = metal::simd_sum(s2);
      if (lane == 0) parts[((size_t)r * P + tg) * PL + MIX] = t;
    }
  }
} else {
  threadgroup uint4* h2s = (threadgroup uint4*)tgm4;
  threadgroup float* red = tgm + TH;
  const int rt = tg - P;
  const int gi = rt % G;
  const int r0 = (rt / G) * RG;
  const int e = gi * 4 + sg / 2;
  const int half_c = (sg & 1) * NC;
  uint4 gw[NC];
  for (int c = 0; c < NC; ++c) gw[c] = e < E ? ((const device uint4*)(gate_w + (size_t)e * D))[(half_c + c) * 32 + lane] : uint4(0u);
  float ss[RG];
  for (int rr = 0; rr < RG; ++rr) ss[rr] = 0.0f;
  for (int j = 0; j < 3; ++j) {
    const int q = tid + 256 * j;
    if (q < NCH) for (int rr = 0; rr < RG; ++rr) if (r0 + rr < R) {
      const int r = r0 + rr;
      const auto po = spost + r * HC;
      const auto cb = scomb + r * HC * HC;
      const auto pr = spre + r * HC;
      float sv[HC][8];
      for (int s = 0; s < HC; ++s) {
        const uint4 v = *(const device uint4*)(x + ((size_t)r * HC + s) * D + 8 * q);
        const uint wd[4] = {v.x, v.y, v.z, v.w};
        for (int i = 0; i < 4; ++i) {
          sv[s][2 * i] = as_type<float>(wd[i] << 16);
          sv[s][2 * i + 1] = as_type<float>(wd[i] & 0xffff0000u);
        }
      }
      const uint4 yq = *(const device uint4*)(y + (size_t)r * D + 8 * q);
      const uint yd[4] = {yq.x, yq.y, yq.z, yq.w};
      uint hw[4];
      for (int i = 0; i < 4; ++i) {
        uint hb[2];
        for (int c = 0; c < 2; ++c) {
          const int col = 2 * i + c;
          const float yv = c == 0 ? as_type<float>(yd[i] << 16) : as_type<float>(yd[i] & 0xffff0000u);
          float h = 0.0f;
          for (int s = 0; s < HC; ++s) {
            float a = 0.0f;
            for (int k = 0; k < HC; ++k) a = metal::fma(cb[k * HC + s], sv[k][col], a);
            const T o = T(metal::fma(po[s], yv, a));
            h = metal::fma(pr[s], float(o), h);
          }
          const T hv = T(h);
          const float hf = float(hv);
          ss[rr] = metal::fma(hf, hf, ss[rr]);
          hb[c] = uint(as_type<ushort>(hv));
        }
        hw[i] = hb[0] | (hb[1] << 16);
      }
      h2s[rr * NCH + q] = uint4(hw[0], hw[1], hw[2], hw[3]);
    }
  }
  for (int rr = 0; rr < RG; ++rr) {
    const float t = metal::simd_sum(ss[rr]);
    if (lane == 0) red[sg * RG + rr] = t;
  }
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  float inv[RG];
  for (int rr = 0; rr < RG; ++rr) {
    float tot = 0.0f;
    for (int g = 0; g < 8; ++g) tot += red[g * RG + rr];
    inv[rr] = metal::precise::rsqrt(tot / float(D) + eps[0]);
  }
  for (int j = 0; j < 3; ++j) {
    const int q = tid + 256 * j;
    if (q < NCH) {
      const uint4 wq = ((const device uint4*)w)[q];
      const uint wn[4] = {wq.x, wq.y, wq.z, wq.w};
      for (int rr = 0; rr < RG; ++rr) if (r0 + rr < R) {
        const uint4 h = h2s[rr * NCH + q];
        const uint wd[4] = {h.x, h.y, h.z, h.w};
        uint o[4];
        for (int i = 0; i < 4; ++i) {
          const float a = as_type<float>(wd[i] << 16), b = as_type<float>(wd[i] & 0xffff0000u);
          const float wa = as_type<float>(wn[i] << 16), wb = as_type<float>(wn[i] & 0xffff0000u);
          const T ha = T(wa * float(T(a * inv[rr]))), hb = T(wb * float(T(b * inv[rr])));
          o[i] = (uint(as_type<ushort>(hb)) << 16) | uint(as_type<ushort>(ha));
        }
        const uint4 ov = uint4(o[0], o[1], o[2], o[3]);
        h2s[rr * NCH + q] = ov;
        if (q % G == gi) ((device uint4*)(h2 + (size_t)(r0 + rr) * D))[q] = ov;
      }
    }
  }
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  float acc[RG];
  for (int rr = 0; rr < RG; ++rr) acc[rr] = 0.0f;
  for (int c = 0; c < NC; ++c) {
    const uint gv[4] = {gw[c].x, gw[c].y, gw[c].z, gw[c].w};
    for (int rr = 0; rr < RG; ++rr) {
      const uint4 hx = h2s[rr * NCH + (half_c + c) * 32 + lane];
      const uint hw[4] = {hx.x, hx.y, hx.z, hx.w};
      for (int i = 0; i < 4; ++i) {
        acc[rr] = metal::fma(as_type<float>(hw[i] << 16), as_type<float>(gv[i] << 16), acc[rr]);
        acc[rr] = metal::fma(as_type<float>(hw[i] & 0xffff0000u), as_type<float>(gv[i] & 0xffff0000u), acc[rr]);
      }
    }
  }
  for (int rr = 0; rr < RG; ++rr) {
    const float t = metal::simd_sum(acc[rr]);
    if (lane == 0) red[8 * RG + sg * RG + rr] = t;
  }
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  if (lane == 0 && (sg & 1) == 0 && e < E) for (int rr = 0; rr < RG; ++rr) if (r0 + rr < R) {
    const float z = red[8 * RG + sg * RG + rr] + red[8 * RG + (sg + 1) * RG + rr];
    const float ez = metal::exp(-metal::abs(z));
    const float lp = ez < 1e-4f ? ez * (1.0f - ez * (0.5f - ez / 3.0f)) : metal::log(1.0f + ez);
    logits[(size_t)(r0 + rr) * E + e] = metal::sqrt(metal::max(z, 0.0f) + lp);
  }
}

seg_release();
if (tid == 0) last = seg_arrive_last(slot + (is_mix ? 0 : 1), gen_v, is_mix ? uint(P) : uint(NRT)) ? 1u : 0u;
threadgroup_barrier(metal::mem_flags::mem_threadgroup);
if (last == 0u) return;
seg_acquire();

if (is_mix) {
  threadgroup float* zr = tgm + R * P * PL;
  for (int i = tid; i < R * P * PL; i += 256) tgm[i] = seg_read_peer_f32(parts, uint(i));
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  #pragma clang loop unroll(disable)
  for (int pair = sg; pair < R * PL; pair += 8) {
    const int r = pair / PL, m = pair % PL;
    float v = 0.0f;
    for (int p = int(lane); p < P; p += 32) v += tgm[(r * P + p) * PL + m];
    v = metal::simd_sum(v);
    if (lane == 0) zr[pair] = v;
  }
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  if (sg == 0 && lane < 4 * R) {
    const int r = int(lane) / 4, i = int(lane) % 4;
    const threadgroup float* zz = zr + r * PL;
    const float inv = metal::precise::rsqrt(zz[MIX] / float(HC * D) + eps[0]);
    const float e1 = eps[1];
    pre[r * HC + i] = 1.0f / (1.0f + metal::exp(-(zz[i] * inv * scale[i] + base[i]))) + e1;
    post[r * HC + i] = 2.0f / (1.0f + metal::exp(-(zz[HC + i] * inv * scale[HC + i] + base[HC + i])));
    float v[HC];
    float mx = -INFINITY;
    for (int j = 0; j < HC; ++j) {
      const int m = 2 * HC + i * HC + j;
      v[j] = zz[m] * inv * scale[m] + base[m];
      mx = metal::max(mx, v[j]);
    }
    float sm = 0.0f;
    for (int j = 0; j < HC; ++j) {
      v[j] = metal::exp(v[j] - mx);
      sm += v[j];
    }
    for (int j = 0; j < HC; ++j) v[j] = v[j] / sm + e1;
    #pragma clang loop unroll(disable)
    for (int t = 0; t < 2 * ITERS - 1; ++t) {
      if (t % 2 == 0) {
        for (int j = 0; j < HC; ++j) v[j] = v[j] / (metal::quad_sum(v[j]) + e1);
      } else {
        const float rs = v[0] + v[1] + v[2] + v[3];
        for (int j = 0; j < HC; ++j) v[j] = v[j] / (rs + e1);
      }
    }
    *(device float4*)(comb + (r * HC + i) * HC) = float4(v[0], v[1], v[2], v[3]);
  }
} else {
  for (int i = tid; i < R * E; i += 256) {
    const float s = seg_read_peer_f32(logits, uint(i));
    const float b = s + bias[i % E];
    tgm[i] = s;
    tgm[R * E + i] = b == b ? b : -3.4e38f;
  }
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  if (sg < R) {
    const int r = sg;
    threadgroup float* vt = tgm + R * E + r * E;
    float sum = 0.0f, mine_s = 0.0f;
    int mine_e = 0;
    #pragma clang loop unroll(disable)
    for (int t = 0; t < TOPK; ++t) {
      float bv = -INFINITY;
      int be = 0x7fffffff;
      for (int ex = int(lane); ex < E; ex += 32) {
        const float xv = vt[ex];
        if (xv > bv) { bv = xv; be = ex; }
      }
      const float gv = metal::simd_max(bv);
      const int pick = metal::simd_min(bv == gv ? be : 0x7fffffff);
      const float s = tgm[r * E + pick];
      if (int(lane) == t) { mine_s = s; mine_e = pick; }
      if (lane == 0) vt[pick] = -INFINITY;
      sum += s;
      metal::simdgroup_barrier(metal::mem_flags::mem_threadgroup);
    }
    if (lane < TOPK) {
      const float div = NORM ? sum + 1e-20f : 1.0f;
      idx[r * TOPK + lane] = uint(mine_e);
      wts[r * TOPK + lane] = mine_s / div * scal[0];
    }
  }
}
