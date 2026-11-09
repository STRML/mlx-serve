// dec_moe_fp4_bw: the DSV41_MOE_BW readers of dec_moe_fp4, the same sums in the same order: a word's eight e2m1
// codes decoded four half2 at a time, and the rows a lane reads sharing one decode of its activations.

// The eight e2m1 codes of word `a` (code i in bits 4i..4i+3) as floats, each 2^-14 of its value as `dec_fp4_pair`
// gives them: codes i and i + 4 land in one half2.
inline void dec_fp4_word(uint a, thread float* w) {
  const half2 h0 = as_type<half2>(((a << 9) & 0x0E000E00u) | ((a << 12) & 0x80008000u));
  const half2 h1 = as_type<half2>(((a << 5) & 0x0E000E00u) | ((a << 8) & 0x80008000u));
  const half2 h2 = as_type<half2>(((a << 1) & 0x0E000E00u) | ((a << 4) & 0x80008000u));
  const half2 h3 = as_type<half2>(((a >> 3) & 0x0E000E00u) | (a & 0x80008000u));
  w[0] = h0.x;
  w[1] = h1.x;
  w[2] = h2.x;
  w[3] = h3.x;
  w[4] = h0.y;
  w[5] = h1.y;
  w[6] = h2.y;
  w[7] = h3.y;
}

// `dec_fp4_dot` of NR rows' unit codes `q[i]` against one unit of activations at `x` (bf16) into d[i].
template <int NR, typename XP>
inline void dec_fp4_dots(thread const uint4* q, XP x, thread float* d) {
#pragma unroll
  for (int i = 0; i < NR; ++i) d[i] = 0.0f;
#pragma unroll
  for (int c = 0; c < 4; ++c) {
    const uint4 xq = x[c];
    const float xf[8] = {dec_up_bf(xq.x, 0), dec_up_bf(xq.x, 1), dec_up_bf(xq.y, 0), dec_up_bf(xq.y, 1), dec_up_bf(xq.z, 0), dec_up_bf(xq.z, 1), dec_up_bf(xq.w, 0), dec_up_bf(xq.w, 1)};
#pragma unroll
    for (int i = 0; i < NR; ++i) {
      float wv[8];
      dec_fp4_word(q[i][c], wv);
#pragma unroll
      for (int e = 0; e < 8; ++e) d[i] = fma(xf[e], wv[e], d[i]);
    }
  }
}

// `dec_fp4_down_routed` of NR expert rows `row`, `row + rs`, ... against `x` into acc[i]: units first, first + step...
template <int K, int NR, typename XP>
inline void dec_fp4_down_rows(const device uint32_t* w, const device uint8_t* sc, size_t row, size_t rs, XP x, uint first, uint step, thread float* acc) {
  constexpr uint NU = K / 32;
#pragma unroll
  for (int i = 0; i < NR; ++i) acc[i] = 0.0f;
  for (uint u = first; u < NU; u += step) {
    uint4 q[NR];
    float f[NR], d[NR];
#pragma unroll
    for (int i = 0; i < NR; ++i) {
      q[i] = ((const device uint4*)(w + (row + i * rs) * (K / 8)))[u];
      f[i] = dec_fp4_scale(uint(sc[(row + i * rs) * (K / 32) + u]));
    }
    dec_fp4_dots<NR>(q, x + u * 4, d);
#pragma unroll
    for (int i = 0; i < NR; ++i) acc[i] = fma(f[i], d[i], acc[i]);
  }
}

// `dec_fp4_up_routed` with a unit's gate and up codes read against one decode of its activations.
template <int K, int N, typename XP>
inline void dec_fp4_up_pair(const device uint32_t* w, const device uint8_t* sc, size_t row, XP x, uint lane, thread float (&acc)[2]) {
  constexpr uint KW = K / 8, KG = K / 32, UPL = K / 1024;
  const device uint4* w0 = (const device uint4*)(w + row * KW) + lane;
  const device uint4* w1 = (const device uint4*)(w + (row + N) * KW) + lane;
  const device uint8_t* s0 = sc + row * KG + lane;
  const device uint8_t* s1 = sc + (row + N) * KG + lane;
  acc[0] = acc[1] = 0.0f;
#pragma unroll
  for (uint k = 0; k < UPL; ++k) {
    const uint4 q[2] = {w0[32 * k], w1[32 * k]};
    const float f0 = dec_fp4_scale(uint(s0[32 * k])), f1 = dec_fp4_scale(uint(s1[32 * k]));
    float d[2];
    dec_fp4_dots<2>(q, x + (32 * k + lane) * 4, d);
    acc[0] = fma(f0, d[0], acc[0]);
    acc[1] = fma(f1, d[1], acc[1]);
  }
  acc[0] = simd_sum(acc[0]);
  acc[1] = simd_sum(acc[1]);
}

template <int K, int N, typename T, typename XP>
inline T dec_fp4_up_routed_act(const device uint32_t* w, const device uint8_t* sc, uint expert, uint n_col, XP x, float lim, float wt, uint lane) {
  float acc[2];
  dec_fp4_up_pair<K, N>(w, sc, size_t(expert) * (2 * N) + n_col, x, lane, acc);
  return dec_up_act<T>(acc[0], acc[1], lim, wt);
}

template <int R, int K, int N, typename T, typename XP>
inline void dec_fp4_up_shared_acts(const device uint32_t* sw, const device uint8_t* ss, uint n_col, XP x, float lim, uint lane, thread T* o) {
  float acc[R][2];
  dec_up_shared_x<R, K, N>(sw, ss, n_col, x, lane, acc);
#pragma unroll
  for (int r = 0; r < R; ++r) o[r] = dec_up_act<T>(acc[r][0], acc[r][1], lim, 1.0f);
}

template <int K, int N, int NC, int TOPK, typename XP>
inline void dec_fp4_down_slot(const device uint32_t* w, const device uint8_t* sc, const device uint32_t* sw, const device uint8_t* ss, uint expert, uint d0, uint sgi, uint lane, XP xs, threadgroup float* part) {
  const uint c = lane % NC;
  if (sgi >= uint(TOPK)) {
    const uint hf = sgi - TOPK;
    float a = dec_fp4_down_shared_x<K>(sw, ss, d0 + hf * NC + c, xs, lane / NC, 32 / NC);
#pragma unroll
    for (uint o = 16; o >= uint(NC); o >>= 1) a += simd_shuffle_xor(a, o);
    if (lane < uint(NC)) part[(hf * NC + lane) * (TOPK + 1) + TOPK] = a;
  } else {
    float acc[2];
    dec_fp4_down_rows<K, 2>(w, sc, size_t(expert) * N + d0 + c, NC, xs, lane / NC, 32 / NC, acc);
#pragma unroll
    for (int hf = 0; hf < 2; ++hf) {
#pragma unroll
      for (uint o = 16; o >= uint(NC); o >>= 1) acc[hf] += simd_shuffle_xor(acc[hf], o);
      if (lane < uint(NC)) part[(hf * NC + lane) * (TOPK + 1) + sgi] = acc[hf];
    }
  }
}

template <int TOPK>
inline float dec_fp4_down_sum(threadgroup const float* part, uint cc) {
  float s = 0.0f;
  for (int j = 0; j <= TOPK; ++j) s += part[cc * (TOPK + 1) + j];
  return s;
}
