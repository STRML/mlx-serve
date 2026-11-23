  if (use4) {
    if (d < D) for (int i = 0; i < HC; ++i) v4[i][c] = float4(*(const device vec<T, 4>*)(xr + i * D + d));
  } else {
    if (d < D) for (int i = 0; i < HC; ++i) v4[i][c >> 2][c & 3] = float(xr[i * D + d]);
  }
}