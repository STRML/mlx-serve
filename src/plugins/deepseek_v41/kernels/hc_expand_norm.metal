// the hc expand (kernels/hc_expand_into.metal's loads, fmas and stores, a
// column at a time) and hc_norm.metal's collapse of the new stream with `cpre` and its rms norm with `w`, one threadgroup
// a row (hc_norm's): each thread expands its columns of every copy, then collapses them from its registers, so the new
// stream is not read back; the split comes from the mixes after (hc_mix over the new stream, then hc_split). The header
// picks where the new stream goes: X1_AT stores into `dst` (a per-prompt region bound as an input, with a one-word
// `carrier`, scratch.zig) or into the `x1` output. Each thread reads a column's copies of `res` before it stores the
// column's copies of the new stream, so `dst` may be `res`'s own buffer.
const int l = threadgroup_position_in_grid.y;
const uint tid = thread_index_in_threadgroup, lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
const int D = y_shape[1];
CARRIER_INIT
// `auto`: MLX binds an input of fewer than 8 elements (`post` [1, HC] and `cpre` [1, HC] at one row) in the constant
// address space, a larger one in the device space; the row pointers take the space their input has.
const auto pr = post + l * HC;
const auto cb = comb + l * HC * HC;
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
    const float4 xv = float4(*(const device vec<T, 4>*)(y + size_t(l) * D + d));
    float4 sv[HC];
    for (int j = 0; j < HC; ++j) sv[j] = float4(*(const device vec<T, 4>*)(res + (size_t(l) * HC + j) * D + d));
    float4 h = 0.0f;
    for (int i = 0; i < HC; ++i) {
      float4 acc = 0.0f;
      for (int j = 0; j < HC; ++j) acc = fma(float4(cb[j * HC + i]), sv[j], acc);
      const vec<T, 4> o = vec<T, 4>(fma(float4(pr[i]), xv, acc));
      *(device vec<T, 4>*)(&X1_AT((size_t(l) * HC + i) * D + d)) = o;
      h += float4(cp[i]) * float4(o);
    }
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
    const float xv = float(y[size_t(l) * D + d]);
    float sv[HC];
    for (int j = 0; j < HC; ++j) sv[j] = float(res[(size_t(l) * HC + j) * D + d]);
    float h = 0.0f;
    for (int i = 0; i < HC; ++i) {
      float acc = 0.0f;
      for (int j = 0; j < HC; ++j) acc = fma(cb[j * HC + i], sv[j], acc);
      const T o = T(fma(pr[i], xv, acc));
      X1_AT((size_t(l) * HC + i) * D + d) = o;
      h += cp[i] * float(o);
    }
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
