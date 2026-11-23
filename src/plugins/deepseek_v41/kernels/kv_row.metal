const int l = threadgroup_position_in_grid.y, d4 = thread_position_in_threadgroup.x * 4;
const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
const int D = x_shape[1];
const size_t o = size_t(l) * D + d4;
const metal::vec<T, 4> xv4 = *(const device metal::vec<T, 4>*)(x + o);
const metal::vec<T, 4> w4 = *(const device metal::vec<T, 4>*)(w + d4);
float4 fv = float4(float(xv4[0]), float(xv4[1]), float(xv4[2]), float(xv4[3]));
threadgroup float t_s[32];
const float ss = metal::simd_sum(metal::dot(fv, fv));
if (lane == 0) t_s[sg] = ss;
threadgroup_barrier(metal::mem_flags::mem_threadgroup);
float tot = 0.0f;
const int n_sg = (D / 4 + 31) / 32;
for (int g = 0; g < n_sg; ++g) tot += t_s[g];
const float inv = metal::precise::rsqrt(tot / float(D) + eps[0]);
const metal::vec<T, 4> y4(w4[0] * T(fv[0] * inv), w4[1] * T(fv[1] * inv), w4[2] * T(fv[2] * inv), w4[3] * T(fv[3] * inv));
float4 r4 = float4(float(y4[0]), float(y4[1]), float(y4[2]), float(y4[3]));
const int head = D - RD;
if (d4 >= head) {
  const int p0 = (d4 - head) >> 1;
  const float pos = float(l + off);
  const float t0 = pos * (1.0f / periods[p0]);
  const float t1 = pos * (1.0f / periods[p0 + 1]);
  const float c0 = metal::fast::cos(t0), s0 = metal::fast::sin(t0);
  const float c1 = metal::fast::cos(t1), s1 = metal::fast::sin(t1);
  const float x0 = r4[0], x1 = r4[1], x2 = r4[2], x3 = r4[3];
  r4[0] = float(T(x0 * c0 - x1 * s0));
  r4[1] = float(T(x0 * s0 + x1 * c0));
  r4[2] = float(T(x2 * c1 - x3 * s1));
  r4[3] = float(T(x2 * s1 + x3 * c1));
}
if (QAT) {
  float m32 = metal::max(metal::max(metal::abs(r4[0]), metal::abs(r4[1])), metal::max(metal::abs(r4[2]), metal::abs(r4[3])));
  m32 = metal::max(m32, metal::simd_shuffle_xor(m32, ushort(1)));
  m32 = metal::max(m32, metal::simd_shuffle_xor(m32, ushort(2)));
  m32 = metal::max(m32, metal::simd_shuffle_xor(m32, ushort(4)));
  const float scale = dsv41_pow2ceil(metal::max(m32, 1e-4f) * (1.0f / 448.0f));
  const float inv_s = 1.0f / scale;
  r4[0] = dsv41_fp8_roundtrip(metal::clamp(r4[0] * inv_s, -448.0f, 448.0f)) * scale;
  r4[1] = dsv41_fp8_roundtrip(metal::clamp(r4[1] * inv_s, -448.0f, 448.0f)) * scale;
  r4[2] = dsv41_fp8_roundtrip(metal::clamp(r4[2] * inv_s, -448.0f, 448.0f)) * scale;
  r4[3] = dsv41_fp8_roundtrip(metal::clamp(r4[3] * inv_s, -448.0f, 448.0f)) * scale;
}
*(device metal::vec<T, 4>*)(out + o) = metal::vec<T, 4>(T(r4[0]), T(r4[1]), T(r4[2]), T(r4[3]));