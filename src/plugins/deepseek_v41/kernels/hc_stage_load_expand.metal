  if (use4) {
    if (d < D) {
      const auto pr = spost + l * HC;
      const auto cb = scomb + l * HC * HC;
      float4 sv[HC];
      for (int j = 0; j < HC; ++j) sv[j] = float4(*(const device vec<T, 4>*)(xr + j * D + d));
      const float4 yv = float4(*(const device vec<T, 4>*)(y + size_t(l) * D + d));
      for (int i = 0; i < HC; ++i) {
        float4 acc = 0.0f;
        for (int j = 0; j < HC; ++j) acc = metal::fma(float4(cb[j * HC + i]), sv[j], acc);
        const vec<T, 4> o = vec<T, 4>(metal::fma(float4(pr[i]), yv, acc));
        v4[i][c] = float4(o);
        if (role == 0) *(device vec<T, 4>*)(x1 + (size_t(l) * HC + i) * D + d) = o;
      }
    }
  } else {
    if (d < D) {
      const auto pr = spost + l * HC;
      const auto cb = scomb + l * HC * HC;
      float sv[HC];
      for (int j = 0; j < HC; ++j) sv[j] = float(xr[j * D + d]);
      const float yv = float(y[size_t(l) * D + d]);
      for (int i = 0; i < HC; ++i) {
        float acc = 0.0f;
        for (int j = 0; j < HC; ++j) acc = metal::fma(cb[j * HC + i], sv[j], acc);
        const T o = T(metal::fma(pr[i], yv, acc));
        v4[i][c >> 2][c & 3] = float(o);
        if (role == 0) x1[(size_t(l) * HC + i) * D + d] = o;
      }
    }
  }
}