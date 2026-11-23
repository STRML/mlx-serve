// dec_moe_fp4: the decode MoE over mxfp4 routed experts (and the mxfp8 shared expert, whose gate/up reader is
// `dec_up_shared` of dec_moe_up_header.metal). A lane owns the units of 32 inputs u = 32 k + lane, so one load covers 32
// adjacent units: 16 bytes of codes and one e8m0 scale byte a unit.

// The two e2m1 codes of byte `b` (low nibble first) as floats, each 2^-14 of its value: sign, then the other three bits
// placed as an fp16's top exponent and mantissa bits (a zero exponent field is the 0.5 subnormal).
inline float2 dec_fp4_pair(uint b) {
  const uint v = ((b & 0x07u) << 9) | ((b & 0x08u) << 12) | ((b & 0x70u) << 21) | ((b & 0x80u) << 24);
  return float2(as_type<half2>(v));
}

// e8m0 `e` as 2^(e - 127), times the 2^14 `dec_fp4_pair` leaves out.
inline float dec_fp4_scale(uint e) {
  return as_type<float>(e == 0u ? 0x400000u : (e << 23)) * 16384.0f;
}

// One unit's dot: the 32 codes in `q` against the 32 activations at `x` (bf16).
inline float dec_fp4_dot(uint4 q, const device uint4* x) {
  float d = 0.0f;
#pragma unroll
  for (int c = 0; c < 4; ++c) {
    const uint4 xq = x[c];
    const float xf[8] = {dec_up_bf(xq.x, 0), dec_up_bf(xq.x, 1), dec_up_bf(xq.y, 0), dec_up_bf(xq.y, 1), dec_up_bf(xq.z, 0), dec_up_bf(xq.z, 1), dec_up_bf(xq.w, 0), dec_up_bf(xq.w, 1)};
    const uint a = c == 0 ? q.x : c == 1 ? q.y : c == 2 ? q.z : q.w;
#pragma unroll
    for (int p = 0; p < 4; ++p) {
      const float2 wv = dec_fp4_pair((a >> (8 * p)) & 0xFFu);
      d = fma(xf[2 * p], wv.x, d);
      d = fma(xf[2 * p + 1], wv.y, d);
    }
  }
  return d;
}

// The gate (acc[rr][0]) and up (acc[rr][1]) sums of expert row `row` and `row + N` against RG rows of `x` (`xrow[rr]`).
template <int K, int N, int RG, typename T>
inline void dec_fp4_up_routed(const device uint32_t* w, const device uint8_t* sc, size_t row, const device T* x, thread const uint* xrow, uint lane, thread float (*acc)[2]) {
  constexpr uint KW = K / 8, KG = K / 32, UPL = K / 1024;
  const device uint4* w0 = (const device uint4*)(w + row * KW) + lane;
  const device uint4* w1 = (const device uint4*)(w + (row + N) * KW) + lane;
  const device uint8_t* s0 = sc + row * KG + lane;
  const device uint8_t* s1 = sc + (row + N) * KG + lane;
#pragma unroll
  for (int rr = 0; rr < RG; ++rr) acc[rr][0] = acc[rr][1] = 0.0f;
#pragma unroll
  for (uint k = 0; k < UPL; ++k) {
    const uint4 q0 = w0[32 * k], q1 = w1[32 * k];
    const float f0 = dec_fp4_scale(uint(s0[32 * k])), f1 = dec_fp4_scale(uint(s1[32 * k]));
#pragma unroll
    for (int rr = 0; rr < RG; ++rr) {
      const device uint4* xu = (const device uint4*)(x + size_t(xrow[rr]) * K) + (32 * k + lane) * 4;
      acc[rr][0] = fma(f0, dec_fp4_dot(q0, xu), acc[rr][0]);
      acc[rr][1] = fma(f1, dec_fp4_dot(q1, xu), acc[rr][1]);
    }
  }
#pragma unroll
  for (int rr = 0; rr < RG; ++rr) {
    acc[rr][0] = simd_sum(acc[rr][0]);
    acc[rr][1] = simd_sum(acc[rr][1]);
  }
}

// A lane's part of expert row `row` (mxfp4, K inputs) decoded once against NP `(row, slot)` activation vectors.
template <int K, int MAXP, typename T>
inline void dec_fp4_down_routed_multi(const device uint32_t* w, const device uint8_t* sc, size_t row, const device T* x, thread const uint* ps, int np, uint topk, uint first, uint step, thread float* acc) {
  constexpr uint NU = K / 32;
  const device uint4* wp = (const device uint4*)(w + row * (K / 8));
  const device uint8_t* sp = sc + row * (K / 32);
  for (uint u = first; u < NU; u += step) {
    const float f = dec_fp4_scale(uint(sp[u]));
    const uint4 q = wp[u];
    const uint qw[4] = {q.x, q.y, q.z, q.w};
    half2 wv[4][4];
#pragma unroll
    for (int c = 0; c < 4; ++c) {
#pragma unroll
      for (int p = 0; p < 4; ++p) {
        const uint b = (qw[c] >> (8 * p)) & 0xFFu;
        const uint v = ((b & 0x07u) << 9) | ((b & 0x08u) << 12) | ((b & 0x70u) << 21) | ((b & 0x80u) << 24);
        wv[c][p] = as_type<half2>(v);
      }
    }
    for (int i = 0; i < np && i < MAXP; ++i) {
      const uint idx = ps[i];
      const device uint4* xu = (const device uint4*)(x + size_t((idx / topk) * (topk + 1) + idx % topk) * K) + u * 4;
      float d = 0.0f;
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        const uint4 xq = xu[c];
        const float xf[8] = {dec_up_bf(xq.x, 0), dec_up_bf(xq.x, 1), dec_up_bf(xq.y, 0), dec_up_bf(xq.y, 1), dec_up_bf(xq.z, 0), dec_up_bf(xq.z, 1), dec_up_bf(xq.w, 0), dec_up_bf(xq.w, 1)};
#pragma unroll
        for (int p = 0; p < 4; ++p) {
          const float2 fp = float2(wv[c][p]);
          d = fma(xf[2 * p], fp.x, d);
          d = fma(xf[2 * p + 1], fp.y, d);
        }
      }
      acc[i] = fma(f, d, acc[i]);
    }
  }
}

// A lane's part of the expert row `row` of the fused-down bank (mxfp4, K inputs) against `x`: units first, first + step...
template <int K, typename T>
inline float dec_fp4_down_routed(const device uint32_t* w, const device uint8_t* sc, size_t row, const device T* x, uint first, uint step) {
  constexpr uint NU = K / 32;
  const device uint4* wp = (const device uint4*)(w + row * (K / 8));
  const device uint8_t* sp = sc + row * (K / 32);
  float acc = 0.0f;
  for (uint u = first; u < NU; u += step) acc = fma(dec_fp4_scale(uint(sp[u])), dec_fp4_dot(wp[u], (const device uint4*)x + u * 4), acc);
  return acc;
}

// A lane's part of the shared expert's down row `row` (mxfp8, K inputs: a unit is 32 bytes) decoded once across all R rows.
template <int R, int K, typename T>
inline void dec_fp4_down_shared_multi(const device uint32_t* sw, const device uint8_t* ss, size_t row, const device T* x, uint topk, uint first, uint step, thread float (&acc)[R]) {
  constexpr uint NU = K / 32;
  const device uint4* wp = (const device uint4*)(sw + row * (K / 4));
  const device uint8_t* sp = ss + row * (K / 32);
  for (int r = 0; r < R; ++r) acc[r] = 0.0f;
  for (uint u = first; u < NU; u += step) {
    const uint4 q0 = wp[2 * u], q1 = wp[2 * u + 1];
    const uint qs[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
    float4 wt[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) wt[i] = dec_up_e4m3x4(qs[i]);
    const uint e = uint(sp[u]);
    const float f = as_type<float>(e == 0u ? 0x400000u : (e << 23)) * 256.0f;
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const device uint4* xu = (const device uint4*)(x + size_t(r * (topk + 1) + topk) * K) + u * 4;
      float d = 0.0f;
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        const uint4 xq = xu[c];
        const float xf[8] = {dec_up_bf(xq.x, 0), dec_up_bf(xq.x, 1), dec_up_bf(xq.y, 0), dec_up_bf(xq.y, 1), dec_up_bf(xq.z, 0), dec_up_bf(xq.z, 1), dec_up_bf(xq.w, 0), dec_up_bf(xq.w, 1)};
        const float4 a = wt[2 * c], b = wt[2 * c + 1];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          d = fma(xf[i], a[i], d);
          d = fma(xf[4 + i], b[i], d);
        }
      }
      acc[r] = fma(f, d, acc[r]);
    }
  }
}

// A lane's part of the shared expert's down row `row` (mxfp8, K inputs: a unit is 32 bytes) against `x`, as above.
template <int K, typename T>
inline float dec_fp4_down_shared(const device uint32_t* sw, const device uint8_t* ss, size_t row, const device T* x, uint first, uint step) {
  constexpr uint NU = K / 32;
  const device uint4* wp = (const device uint4*)(sw + row * (K / 4));
  const device uint8_t* sp = ss + row * (K / 32);
  float acc = 0.0f;
  for (uint u = first; u < NU; u += step) {
    const uint4 q0 = wp[2 * u], q1 = wp[2 * u + 1];
    const uint qs[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
    float d = 0.0f;
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const uint4 xq = ((const device uint4*)x)[u * 4 + c];
      const float xf[8] = {dec_up_bf(xq.x, 0), dec_up_bf(xq.x, 1), dec_up_bf(xq.y, 0), dec_up_bf(xq.y, 1), dec_up_bf(xq.z, 0), dec_up_bf(xq.z, 1), dec_up_bf(xq.w, 0), dec_up_bf(xq.w, 1)};
      const float4 a = dec_up_e4m3x4(qs[2 * c]), b = dec_up_e4m3x4(qs[2 * c + 1]);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        d = fma(xf[i], a[i], d);
        d = fma(xf[4 + i], b[i], d);
      }
    }
    const uint e = uint(sp[u]);
    acc = fma(as_type<float>(e == 0u ? 0x400000u : (e << 23)) * 256.0f, d, acc);
  }
  return acc;
}
