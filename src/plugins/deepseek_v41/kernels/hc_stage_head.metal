const uint role = threadgroup_position_in_grid.x;
const int l = threadgroup_position_in_grid.y;
const uint tid = thread_index_in_threadgroup, lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
const int D = x_shape[2], N = fn_shape[1];
threadgroup float t_a[TG / 32], t_b[TG / 32];
const device auto* xr = x + size_t(l) * HC * D;
constexpr int ND4 = (ND + 3) / 4;
const int D4 = D >> 2;
const bool use4 = (D & 3) == 0;
const int NC = use4 ? ND4 : ND;
float4 v4[HC][ND4];
for (int i = 0; i < HC; ++i)
  for (int c = 0; c < ND4; ++c) v4[i][c] = float4(0.0f);
for (int c = 0; c < NC; ++c) {
  const int d = use4 ? ((int(tid) + c * TG) << 2) : (int(tid) + c * TG);