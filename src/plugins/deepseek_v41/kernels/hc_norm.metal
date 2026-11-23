const int l = threadgroup_position_in_grid.y;
const uint tid = thread_index_in_threadgroup, lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
const int D = x_shape[2];
constexpr int MIX = (2 + HC) * HC;
const float e = eps[1];
if (sg == 0) {
  // Lane a * HC + b holds comb[a][b]; each sum gathers its HC terms in the serial order.
  const device float* zr = z + l * MIX;
  if (lane < HC) {
    pre[l * HC + lane] = 1.0f / (1.0f + metal::exp(-zr[lane])) + e;
    post[l * HC + lane] = 2.0f / (1.0f + metal::exp(-zr[HC + lane]));
  }
  const bool act = lane < HC * HC;
  const ushort a = act ? lane / HC : 0, b = act ? lane % HC : 0;
  float v = zr[2 * HC + a * HC + b];
  float mx = metal::simd_shuffle(v, a * HC);
  for (ushort k = 1; k < HC; ++k) mx = metal::max(mx, metal::simd_shuffle(v, a * HC + k));
  v = metal::exp(v - mx);
  float sm = 0.0f;
  for (ushort k = 0; k < HC; ++k) sm += metal::simd_shuffle(v, a * HC + k);
  v = v / sm + e;
  for (int t = 0; t < 2 * ITERS - 1; ++t) {
    const bool rows = t % 2 == 1;
    sm = 0.0f;
    for (ushort k = 0; k < HC; ++k) sm += metal::simd_shuffle(v, rows ? a * HC + k : k * HC + b);
    v = v / (sm + e);
  }
  if (act) comb[l * HC * HC + lane] = v;
}
float cp[HC];
for (int j = 0; j < HC; ++j) cp[j] = cpre[l * HC + j];
float ss = 0.0f;
threadgroup float t_s[TG / 32];
if ((D & 3) == 0) {
  const int D4 = D >> 2;
  vec<T, 4> hb0 = vec<T, 4>(0), hb1 = vec<T, 4>(0), hb2 = vec<T, 4>(0), hb3 = vec<T, 4>(0), hb4 = vec<T, 4>(0);
  int k = 0;
  for (int d4 = tid; d4 < D4; d4 += TG, ++k) {
    const int d = d4 << 2;
    float4 h = 0.0f;
    for (int j = 0; j < HC; ++j) h += float4(cp[j]) * float4(*(const device vec<T, 4>*)(x + (size_t(l) * HC + j) * D + d));
    const vec<T, 4> hb = vec<T, 4>(h);
    if (k == 0) hb0 = hb;
    else if (k == 1) hb1 = hb;
    else if (k == 2) hb2 = hb;
    else if (k == 3) hb3 = hb;
    else if (k == 4) hb4 = hb;
    else *(device vec<T, 4>*)(h_out + size_t(l) * D + d) = hb;
    const float4 hf = float4(hb);
    ss += (hf.x * hf.x + hf.y * hf.y) + (hf.z * hf.z + hf.w * hf.w);
  }
  ss = metal::simd_sum(ss);
  if (lane == 0) t_s[sg] = ss;
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  float tot = 0.0f;
  for (int g = 0; g < TG / 32; ++g) tot += t_s[g];
  const float inv = metal::precise::rsqrt(tot / float(D) + eps[0]);
  k = 0;
  for (int d4 = tid; d4 < D4; d4 += TG, ++k) {
    const int d = d4 << 2;
    const size_t o = size_t(l) * D + d;
    const vec<T, 4> hb = k == 0 ? hb0 : (k == 1 ? hb1 : (k == 2 ? hb2 : (k == 3 ? hb3 : (k == 4 ? hb4 : *(const device vec<T, 4>*)(h_out + o)))));
    *(device vec<T, 4>*)(h_out + o) = *(const device vec<T, 4>*)(w + d) * vec<T, 4>(float4(hb) * inv);
  }
} else {
  for (int d = tid; d < D; d += TG) {
    float h = 0.0f;
    for (int j = 0; j < HC; ++j) h += cp[j] * float(x[(size_t(l) * HC + j) * D + d]);
    const T hb = T(h);
    h_out[size_t(l) * D + d] = hb;
    ss += float(hb) * float(hb);
  }
  ss = metal::simd_sum(ss);
  if (lane == 0) t_s[sg] = ss;
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  float tot = 0.0f;
  for (int g = 0; g < TG / 32; ++g) tot += t_s[g];
  const float inv = metal::precise::rsqrt(tot / float(D) + eps[0]);
  for (int d = tid; d < D; d += TG) {
    const size_t o = size_t(l) * D + d;
    h_out[o] = w[d] * T(float(h_out[o]) * inv);
  }
}