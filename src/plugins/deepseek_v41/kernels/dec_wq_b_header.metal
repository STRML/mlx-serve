// e8m0 as 2^(e - 127), times the 2^8 the e4m3 bit placement leaves out.
inline float dsv41_wqb_scale(uint e) {
  return as_type<float>(e == 0u ? 0x400000u : (e << 23)) * 256.0f;
}

inline float dsv41_wqb_sum16(float v) {
  v += metal::simd_shuffle_xor(v, 8);
  v += metal::simd_shuffle_xor(v, 4);
  v += metal::simd_shuffle_xor(v, 2);
  return v + metal::simd_shuffle_xor(v, 1);
}

// The four e4m3 bytes of w as floats (value x 2^-8): bytes 0 and 2, then 1 and 3, each pair's fields shifted into a half2.
inline float4 dsv41_wqb_e4m3x4(uint w) {
  const float2 lo = float2(as_type<half2>(((w << 7) & 0x3F803F80u) | ((w << 8) & 0x80008000u)));
  const float2 hi = float2(as_type<half2>(((w >> 1) & 0x3F803F80u) | (w & 0x80008000u)));
  return float4(lo.x, hi.x, lo.y, hi.y);
}

// What a lane reads of NP row pairs of a [N, 1280] linear (mxfp8 when !AFF, affine 8-bit group GS when AFF):
// lanes 0-15 the even row of each pair, lanes 16-31 the odd one, 16 consecutive k a step (5 steps of 256).
template <int NP, bool AFF>
struct dsv41_wqb_w {
  uint4 q[NP][5];
  metal::conditional_t<AFF, float, uint> e[NP][5];
  metal::conditional_t<AFF, float, uchar> b[NP][5];
};

template <int NP, bool AFF, int GS, typename T>
METAL_FUNC void dsv41_wqb_load(thread dsv41_wqb_w<NP, AFF>& w, const device uint4* wp, const device void* sp, const device void* bp, int n_base, uint lane) {
  const uint h = lane >> 4, j = lane & 15;
  for (int p = 0; p < NP; ++p) {
    const int row = n_base + 2 * p + int(h);
    const device uint4* wr = wp + row * 80 + j;
    if constexpr (AFF) {
      constexpr int KG = 1280 / GS, LPG = GS / 16, SSTEP = 256 / GS;
      const device T* sr = (const device T*)sp + row * KG + (j / LPG);
      const device T* br = (const device T*)bp + row * KG + (j / LPG);
      for (int s = 0; s < 5; ++s) {
        w.q[p][s] = wr[16 * s];
        w.e[p][s] = float(sr[SSTEP * s]);
        w.b[p][s] = float(br[SSTEP * s]);
      }
    } else {
      const device uchar* sr = (const device uchar*)sp + row * 40 + (j >> 1);
      for (int s = 0; s < 5; ++s) {
        w.q[p][s] = wr[16 * s];
        w.e[p][s] = uint(sr[8 * s]);
      }
    }
  }
}

// The bits of `v`, a value bf16 holds exactly.
inline ushort dsv41_wqb_bf(float v) {
  return ushort(as_type<uint>(v) >> 16);
}

inline float4 dsv41_wqb_unbf(ushort4 v) {
  return float4(as_type<float>(uint(v.x) << 16), as_type<float>(uint(v.y) << 16), as_type<float>(uint(v.z) << 16), as_type<float>(uint(v.w) << 16));
}

inline float4 dsv41_wqb_u8x4(uint w) {
  return float4(float(w & 0xFFu), float((w >> 8) & 0xFFu), float((w >> 16) & 0xFFu), float(w >> 24));
}

// Those rows against R rows of x, held as chunks of four bf16: chunk (r, s, t, j) at ((r * 5 + s) * 4 + t) * 16 + j for
// k = 256 s + 16 j + 4 t, so the 16 lanes of a half read consecutive chunks.
template <int R, int NP, bool AFF>
METAL_FUNC void dsv41_wqb_dots(const thread dsv41_wqb_w<NP, AFF>& w, const threadgroup ushort4* xs, uint lane, thread float (&acc)[NP][R]) {
  const uint j = lane & 15;
  for (int p = 0; p < NP; ++p) for (int r = 0; r < R; ++r) acc[p][r] = 0.0f;
  for (int s = 0; s < 5; ++s) {
    if constexpr (AFF) {
      for (int p = 0; p < NP; ++p) {
        const float sv = w.e[p][s], bv = w.b[p][s];
        const float4 w0 = metal::fma(dsv41_wqb_u8x4(w.q[p][s].x), sv, bv);
        const float4 w1 = metal::fma(dsv41_wqb_u8x4(w.q[p][s].y), sv, bv);
        const float4 w2 = metal::fma(dsv41_wqb_u8x4(w.q[p][s].z), sv, bv);
        const float4 w3 = metal::fma(dsv41_wqb_u8x4(w.q[p][s].w), sv, bv);
        for (int r = 0; r < R; ++r) {
          const threadgroup ushort4* xr = xs + (r * 5 + s) * 64 + j;
          const float4 x0 = dsv41_wqb_unbf(xr[0]), x1 = dsv41_wqb_unbf(xr[16]), x2 = dsv41_wqb_unbf(xr[32]), x3 = dsv41_wqb_unbf(xr[48]);
          float d = acc[p][r];
          d = metal::fma(x0.x, w0.x, d);
          d = metal::fma(x0.y, w0.y, d);
          d = metal::fma(x0.z, w0.z, d);
          d = metal::fma(x0.w, w0.w, d);
          d = metal::fma(x1.x, w1.x, d);
          d = metal::fma(x1.y, w1.y, d);
          d = metal::fma(x1.z, w1.z, d);
          d = metal::fma(x1.w, w1.w, d);
          d = metal::fma(x2.x, w2.x, d);
          d = metal::fma(x2.y, w2.y, d);
          d = metal::fma(x2.z, w2.z, d);
          d = metal::fma(x2.w, w2.w, d);
          d = metal::fma(x3.x, w3.x, d);
          d = metal::fma(x3.y, w3.y, d);
          d = metal::fma(x3.z, w3.z, d);
          d = metal::fma(x3.w, w3.w, d);
          acc[p][r] = d;
        }
      }
    } else {
      for (int p = 0; p < NP; ++p) {
        const uint u[4] = {w.q[p][s].x, w.q[p][s].y, w.q[p][s].z, w.q[p][s].w};
        float4 wv[4];
        for (int i = 0; i < 4; ++i) wv[i] = dsv41_wqb_e4m3x4(u[i]);
        const float sf = dsv41_wqb_scale(w.e[p][s]);
        for (int r = 0; r < R; ++r) {
          const threadgroup ushort4* xr = xs + (r * 5 + s) * 64 + j;
          const float4 x0 = dsv41_wqb_unbf(xr[0]), x1 = dsv41_wqb_unbf(xr[16]), x2 = dsv41_wqb_unbf(xr[32]), x3 = dsv41_wqb_unbf(xr[48]);
          float d = x0.x * wv[0].x;
          d = metal::fma(x0.y, wv[0].y, d);
          d = metal::fma(x0.z, wv[0].z, d);
          d = metal::fma(x0.w, wv[0].w, d);
          d = metal::fma(x1.x, wv[1].x, d);
          d = metal::fma(x1.y, wv[1].y, d);
          d = metal::fma(x1.z, wv[1].z, d);
          d = metal::fma(x1.w, wv[1].w, d);
          d = metal::fma(x2.x, wv[2].x, d);
          d = metal::fma(x2.y, wv[2].y, d);
          d = metal::fma(x2.z, wv[2].z, d);
          d = metal::fma(x2.w, wv[2].w, d);
          d = metal::fma(x3.x, wv[3].x, d);
          d = metal::fma(x3.y, wv[3].y, d);
          d = metal::fma(x3.z, wv[3].z, d);
          d = metal::fma(x3.w, wv[3].w, d);
          acc[p][r] = metal::fma(sf, d, acc[p][r]);
        }
      }
    }
  }
}
