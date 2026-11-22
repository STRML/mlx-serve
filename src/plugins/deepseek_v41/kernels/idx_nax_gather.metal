constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
    16, 32, 16, false, true, true, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
const uint lane = thread_index_in_simdgroup;
const int R = w_shape[0];
const int nkeys = keys_shape[0];
const int ratio = meta[1];
const int kb = cand_shape[1];
const int ncols = kb * BS;
const int row = int(threadgroup_position_in_grid.x) % R;
const int nranges = int(threadgroups_per_grid.x) / R;
const int nstreams = int(threads_per_threadgroup.x / 32);
const ushort qid = ushort(lane >> 2);
const ushort fm = (qid & 4) | ((ushort(lane) >> 1) & 3);
const ushort f4 = (qid & 2) | (ushort(lane) & 1);

constexpr int KSTEPS = IHD / 16;
constexpr int C8 = IHD / 32;
auto ct_a = op.template get_left_input_cooperative_tensor<T, T, float>();
auto ct_b0 = op.template get_right_input_cooperative_tensor<T, T, float>();
auto ct_b1 = op.template get_right_input_cooperative_tensor<T, T, float>();
auto ct_b2 = op.template get_right_input_cooperative_tensor<T, T, float>();
auto ct_b3 = op.template get_right_input_cooperative_tensor<T, T, float>();
auto ct_b4 = op.template get_right_input_cooperative_tensor<T, T, float>();
auto ct_b5 = op.template get_right_input_cooperative_tensor<T, T, float>();
auto ct_b6 = op.template get_right_input_cooperative_tensor<T, T, float>();
auto ct_b7 = op.template get_right_input_cooperative_tensor<T, T, float>();
auto ct_c = op.template get_destination_cooperative_tensor<
    metal::remove_addrspace_t<decltype(ct_a)>,
    metal::remove_addrspace_t<decltype(ct_b0)>, float>();

#define LOAD_B(KS, CT) if ((KS) < KSTEPS) { \
  const int d0_ = ((KS) >> 1) * 32 + int(f4) * 8 + ((KS) & 1) * 4; \
  _Pragma("unroll") for (ushort tn = 0; tn < 2; ++tn) _Pragma("unroll") for (ushort i = 0; i < 2; ++i) { \
    const int h_ = int(tn) * 16 + int(fm) + int(i) * 8; \
    const metal::vec<float, 4> v = h_ < IH ? metal::vec<float, 4>(*(const device metal::vec<QT, 4>*)(q + (size_t(row) * IH + h_) * IHD + d0_)) : metal::vec<float, 4>(0.0f); \
    _Pragma("unroll") for (ushort j = 0; j < 4; ++j) CT[tn * 8 + i * 4 + j] = T(v[j]); \
  } \
}
LOAD_B(0, ct_b0) LOAD_B(1, ct_b1) LOAD_B(2, ct_b2) LOAD_B(3, ct_b3)
LOAD_B(4, ct_b4) LOAD_B(5, ct_b5) LOAD_B(6, ct_b6) LOAD_B(7, ct_b7)
#undef LOAD_B

float wv[8];
#pragma clang loop unroll(full)
for (ushort tn = 0; tn < 2; ++tn)
  #pragma clang loop unroll(full)
  for (ushort j = 0; j < 4; ++j) {
    const int h_ = int(tn) * 16 + int(f4) * 4 + int(j);
    wv[tn * 4 + j] = h_ < IH ? float(w[size_t(row) * IH + h_]) * wsc[0] : 0.0f;
  }

const int seen = metal::min((meta[0] + row + 1) / ratio, nkeys);
device float* out_row = scores + size_t(row) * ncols;
const int ntiles = (ncols + 15) / 16;
const int stride = nranges * nstreams;
for (int tile = (int(threadgroup_position_in_grid.x) / R) * nstreams + int(simdgroup_index_in_threadgroup); tile < ntiles; tile += stride) {
  int n0, n1;
  bool v0, v1;
  if (BS == 8) {
    const int c0 = cand[size_t(row) * kb + tile * 2];
    const int c1 = (tile * 2 + 1 < kb) ? cand[size_t(row) * kb + tile * 2 + 1] : -1;
    n0 = c0 * 8 + int(fm);
    n1 = c1 * 8 + int(fm);
    v0 = c0 >= 0;
    v1 = c1 >= 0;
  } else if (BS == 16) {
    const int c0 = cand[size_t(row) * kb + tile];
    n0 = c0 * 16 + int(fm);
    n1 = n0 + 8;
    v0 = c0 >= 0;
    v1 = v0;
  } else {
    const int c0 = cand[size_t(row) * kb + (tile >> 1)];
    const int sub = (tile & 1) * 16;
    n0 = c0 * BS + sub + int(fm);
    n1 = n0 + 8;
    v0 = c0 >= 0;
    v1 = v0;
  }
  const int safe_n0 = metal::max(0, metal::min(n0, nkeys - 1));
  const int safe_n1 = metal::max(0, metal::min(n1, nkeys - 1));
  const device metal::vec<T, 8>* k0 = (const device metal::vec<T, 8>*)(keys + size_t(safe_n0) * IHD + int(f4) * 8);
  const device metal::vec<T, 8>* k1 = (const device metal::vec<T, 8>*)(keys + size_t(safe_n1) * IHD + int(f4) * 8);
  metal::vec<T, 8> a0[C8], a1[C8];
  #pragma clang loop unroll(full)
  for (ushort c = 0; c < C8; ++c) {
    a0[c] = v0 ? k0[c * 4] : metal::vec<T, 8>(T(0));
    a1[c] = v1 ? k1[c * 4] : metal::vec<T, 8>(T(0));
  }
  #pragma clang loop unroll(full)
  for (short e = 0; e < 16; ++e) ct_c[e] = 0.0f;
#define STEP(KS, CT_B) if ((KS) < KSTEPS) { \
    _Pragma("unroll") for (ushort j = 0; j < 4; ++j) { \
      ct_a[j] = a0[(KS) >> 1][((KS) & 1) * 4 + j]; \
      ct_a[4 + j] = a1[(KS) >> 1][((KS) & 1) * 4 + j]; \
    } \
    op.run(ct_a, CT_B, ct_c); \
  }
  STEP(0, ct_b0) STEP(1, ct_b1) STEP(2, ct_b2) STEP(3, ct_b3)
  STEP(4, ct_b4) STEP(5, ct_b5) STEP(6, ct_b6) STEP(7, ct_b7)
#undef STEP
  float p0[8], p1[8];
  #pragma clang loop unroll(full)
  for (ushort tn = 0; tn < 2; ++tn)
    #pragma clang loop unroll(full)
    for (ushort j = 0; j < 4; ++j) {
      const float d0 = ct_c[tn * 8 + j], d1 = ct_c[tn * 8 + 4 + j];
      const float wt = wv[tn * 4 + j];
      p0[tn * 4 + j] = wt * ((d0 < 0.0f) ? 0.0f : d0);
      p1[tn * 4 + j] = wt * ((d1 < 0.0f) ? 0.0f : d1);
    }
  float s0 = 0.0f, s1 = 0.0f;
  #pragma clang loop unroll(full)
  for (ushort tn = 0; tn < 2; ++tn) {
    #pragma clang loop unroll(full)
    for (ushort src = 0; src < 4; ++src) {
      const ushort xmask = (src & 1) | ((src & 2) << 2);
      #pragma clang loop unroll(full)
      for (ushort j = 0; j < 4; ++j) {
        const float q0 = xmask ? metal::simd_shuffle_xor(p0[tn * 4 + j], xmask) : p0[tn * 4 + j];
        const float q1 = xmask ? metal::simd_shuffle_xor(p1[tn * 4 + j], xmask) : p1[tn * 4 + j];
        if (int(tn) * 16 + int(src) * 4 + int(j) < IH) {
          s0 += q0;
          s1 += q1;
        }
      }
    }
  }
  if (f4 == 0) {
    const int m0 = tile * 16 + int(fm), m1 = m0 + 8;
    if (m0 < ncols) out_row[m0] = (v0 && n0 >= 0 && n0 < seen) ? s0 : -INFINITY;
    if (m1 < ncols) out_row[m1] = (v1 && n1 >= 0 && n1 < seen) ? s1 : -INFINITY;
  }
}
