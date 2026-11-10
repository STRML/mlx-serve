const int n = int(thread_position_in_grid.x);
const int r = int(thread_position_in_grid.y);
if (n == 0 && r == 0) for (int i = 0; i < 16; ++i) ctr[i] = 0u;
if (n >= D) return;
float sv[HC];
for (int j = 0; j < HC; ++j) sv[j] = float(x[((size_t)r * HC + j) * D + n]);
const float yv = float(y[(size_t)r * D + n]);
const auto po = spost + r * HC;
const auto cb = scomb + r * HC * HC;
const auto pr = spre + r * HC;
float h = 0.0f;
for (int i = 0; i < HC; ++i) {
  float a = 0.0f;
  for (int j = 0; j < HC; ++j) a = metal::fma(cb[j * HC + i], sv[j], a);
  const T o = T(metal::fma(po[i], yv, a));
  x1[((size_t)r * HC + i) * D + n] = o;
  h = metal::fma(pr[i], float(o), h);
}
hv[(size_t)r * D + n] = T(h);
