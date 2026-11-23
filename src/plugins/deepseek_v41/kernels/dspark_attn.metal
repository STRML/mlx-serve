// dspark_attn: a DSpark stage's attention over the ring's committed window [p0 - nw, p0) and the B draft keys kv,
// online softmax with the sink, and the inverse tail rope epilogue. One threadgroup per (draft row i, head h).
const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
const int i = threadgroup_position_in_grid.y, h = threadgroup_position_in_grid.x;
const int p0 = meta[0], nw = meta[1], cap = meta[2];
const int B = q_shape[0], H = q_shape[1];
constexpr int V = D / 32;
const int lo = p0 - nw;
const int total_keys = nw + B;
float qv[V], acc[V];
for (int v = 0; v < V; ++v) {
  qv[v] = float(q[(size_t(i) * H + h) * D + lane * V + v]);
  acc[v] = 0.0f;
}
float mx = -INFINITY, den = 0.0f;
const float scale = scal[0];
for (int t = int(sg); t < total_keys; t += NSG) {
  const device T* row = t < nw ? ring + size_t((lo + t) % cap) * D : kv + size_t(t - nw) * D;
  float kvec[V], dot = 0.0f;
  for (int v = 0; v < V; ++v) {
    kvec[v] = float(row[lane * V + v]);
    dot += qv[v] * kvec[v];
  }
  const float s = metal::simd_sum(dot) * scale;
  const float nm = metal::max(mx, s);
  const bool both_ninf = mx == -INFINITY && s == -INFINITY;
  const float corr = both_ninf ? 0.0f : metal::exp(mx - nm);
  const float p = both_ninf ? 0.0f : metal::exp(s - nm);
  den = den * corr + p;
  for (int v = 0; v < V; ++v) acc[v] = acc[v] * corr + p * kvec[v];
  mx = nm;
}
threadgroup float t_acc[NSG * D];
threadgroup float t_mx[NSG], t_den[NSG];
for (int v = 0; v < V; ++v) t_acc[sg * D + lane * V + v] = acc[v];
if (lane == 0) {
  t_mx[sg] = mx;
  t_den[sg] = den;
}
threadgroup_barrier(metal::mem_flags::mem_threadgroup);
float M = sink[h];
for (int g = 0; g < NSG; ++g) M = metal::max(M, t_mx[g]);
float total = metal::exp(sink[h] - M);
float w[NSG];
for (int g = 0; g < NSG; ++g) {
  w[g] = t_den[g] > 0.0f ? metal::exp(t_mx[g] - M) : 0.0f;
  total += t_den[g] * w[g];
}
for (int d = sg * 32 + lane; d < D; d += NSG * 32) {
  float o = 0.0f;
  for (int g = 0; g < NSG; ++g) o += t_acc[g * D + d] * w[g];
  out[(size_t(i) * H + h) * D + d] = dsv41_unrope<OT, RD, D>(OT(o / total), d, p0 + i, periods);
}
