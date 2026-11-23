inline float2 dsv41_bf2(uint w) { return float2(as_type<float>(w << 16), as_type<float>(w & 0xFFFF0000u)); }

// Affine 8-bit dot over one thread's 8 positions against RT rows and R activation rows: w = q * s + b.
template <int R, int RT>
METAL_FUNC void dsv41_in_proj_aff8(
    const thread uint2 (&wq)[RT],
    const thread float (&sc)[RT],
    const thread float (&bi)[RT],
    const thread float (&hf)[R][8],
    thread float (&dv)[RT][R]) {
  #pragma unroll
  for (int i = 0; i < RT; ++i) {
    const uint w0 = wq[i].x, w1 = wq[i].y;
    const float s = sc[i], b = bi[i];
    const float w8[8] = {
      metal::fma(float(w0 & 0xFFu), s, b),
      metal::fma(float((w0 >> 8) & 0xFFu), s, b),
      metal::fma(float((w0 >> 16) & 0xFFu), s, b),
      metal::fma(float(w0 >> 24), s, b),
      metal::fma(float(w1 & 0xFFu), s, b),
      metal::fma(float((w1 >> 8) & 0xFFu), s, b),
      metal::fma(float((w1 >> 16) & 0xFFu), s, b),
      metal::fma(float(w1 >> 24), s, b)
    };
    #pragma unroll
    for (int r = 0; r < R; ++r) {
      float d = 0.0f;
      #pragma unroll
      for (int k = 0; k < 8; ++k) d = metal::fma(hf[r][k], w8[k], d);
      dv[i][r] = d;
    }
  }
}
