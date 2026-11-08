inline float2 dsv41_bf2(uint w) { return float2(as_type<float>(w << 16), as_type<float>(w & 0xFFFF0000u)); }

inline void dsv41_unpack8(uint4 xv, thread float (&v)[8]) {
  const uint w[4] = {xv.x, xv.y, xv.z, xv.w};
  for (int q = 0; q < 4; ++q) {
    const float2 u = dsv41_bf2(w[q]);
    v[2 * q] = u.x;
    v[2 * q + 1] = u.y;
  }
}

// Row r's 4 stream copies at positions p..p + 7 of a D-wide stream, as floats.
inline void dsv41_load8(const device bfloat16_t* x, int r, int p, int d, thread float (&v)[4][8]) {
  for (int j = 0; j < 4; ++j) dsv41_unpack8(*(const device uint4*)(x + (size_t(r) * 4 + j) * d + p), v[j]);
}

// The same positions of the stream the previous block's hc expand makes, copy i = T(post[i] * y + sum_j comb[j][i] x_j)
// with glm5's expand kernel's multiply-adds in its order. `post` and `comb` come as MLX binds them: a one-row post (4
// floats) in the constant address space.
template <typename P, typename C>
inline void dsv41_expand8(const device bfloat16_t* x, const device bfloat16_t* y, P post, C comb, int r, int p, int d, thread float (&v)[4][8]) {
  uint4 xw[4];
  for (int j = 0; j < 4; ++j) xw[j] = *(const device uint4*)(x + (size_t(r) * 4 + j) * d + p);
  const uint4 yw = *(const device uint4*)(y + size_t(r) * d + p);
  for (int q = 0; q < 4; ++q) {
    float2 xs[4];
    for (int j = 0; j < 4; ++j) xs[j] = dsv41_bf2(xw[j][q]);
    const float2 yv = dsv41_bf2(yw[q]);
    for (int i = 0; i < 4; ++i) {
      float a0 = 0.0f, a1 = 0.0f;
      for (int j = 0; j < 4; ++j) {
        const float c = comb[(r * 4 + j) * 4 + i];
        a0 = metal::fma(c, xs[j].x, a0);
        a1 = metal::fma(c, xs[j].y, a1);
      }
      const float pi = post[r * 4 + i];
      v[i][2 * q] = float(bfloat16_t(metal::fma(pi, yv.x, a0)));
      v[i][2 * q + 1] = float(bfloat16_t(metal::fma(pi, yv.y, a1)));
    }
  }
}

inline void dsv41_keep8(device bfloat16_t* x1, int r, int p, int d, thread float (&v)[4][8]) {
  for (int i = 0; i < 4; ++i) {
    uint o[4];
    for (int q = 0; q < 4; ++q) o[q] = (as_type<uint>(v[i][2 * q]) >> 16) | (as_type<uint>(v[i][2 * q + 1]) & 0xFFFF0000u);
    *(device uint4*)(x1 + (size_t(r) * 4 + i) * d + p) = uint4(o[0], o[1], o[2], o[3]);
  }
}
