template <int K, int N, int TOPK, typename T, typename XP>
inline T seg_up_compute(XP x, uint slot, uint n_col, uint expert, float wt, float lim, const device uint32_t* w, const device uint8_t* sc, const device uint32_t* sw, const device uint8_t* ss, uint lane) {
  if (slot == uint(TOPK)) {
    T acts[1];
    dec_fp4_up_shared_acts<1, K, N, T>(sw, ss, n_col, x, lim, lane, acts);
    return acts[0];
  }
  return dec_fp4_up_routed_act<K, N, T>(w, sc, expert, n_col, x, lim, wt, lane);
}

template <int XM, int SGN, int K, int N, int TOPK, typename T, typename PH>
inline void seg_up_item(int tid, uint lane, uint sg, uint cb, uint slot, uint expert, float wt, float lim, const device uint32_t* w, const device uint8_t* sc, const device uint32_t* sw, const device uint8_t* ss, PH h2, device T* act, threadgroup float4* tgm4) {
  const uint n_col = cb * SGN + sg;
  threadgroup T* stage = (threadgroup T*)(tgm4 + (XM == 2 ? K / 8 : 0));
  T v;
  if constexpr (XM == 2) {
    threadgroup uint4* hs = (threadgroup uint4*)tgm4;
    const seg_peer4 src = seg_peer4{(const device uint4*)h2};
    for (uint i = uint(tid); i < uint(K / 8); i += SGN * 32) hs[i] = src[i];
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    v = seg_up_compute<K, N, TOPK, T>((threadgroup const uint4*)hs, slot, n_col, expert, wt, lim, w, sc, sw, ss, lane);
  } else if constexpr (XM == 1) {
    v = seg_up_compute<K, N, TOPK, T>((const device uint4*)h2, slot, n_col, expert, wt, lim, w, sc, sw, ss, lane);
  } else {
    v = seg_up_compute<K, N, TOPK, T>(seg_peer4{(const device uint4*)h2}, slot, n_col, expert, wt, lim, w, sc, sw, ss, lane);
  }
  if (lane == 0) stage[sg] = v;
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  if (tid < SGN / 2) seg_write_peer_output((device uint*)(act + slot * uint(N) + cb * SGN), uint(tid), ((threadgroup const uint*)stage)[tid]);
}

template <int XM, int K, int N, int NC, int TOPK, typename PA>
inline void seg_down_slot(uint tile, uint sgi, uint lane, uint expert, PA act, const device uint32_t* w, const device uint8_t* sc, const device uint32_t* sw, const device uint8_t* ss, threadgroup float* part) {
  const uint d0 = tile * (2 * NC);
  const uint slot = metal::min(sgi, uint(TOPK));
  const device uint4* xs = (const device uint4*)(act + size_t(slot) * K);
  if constexpr (XM == 1) {
    dec_fp4_down_slot<K, N, NC, TOPK>(w, sc, sw, ss, expert, d0, sgi, lane, xs, part);
  } else {
    dec_fp4_down_slot<K, N, NC, TOPK>(w, sc, sw, ss, expert, d0, sgi, lane, seg_peer4{xs}, part);
  }
}

template <int NC, int TOPK, typename T>
inline void seg_down_final(uint tile, uint tid, const device float* unit, device T* out, threadgroup float* part) {
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  if (tid < uint(2 * NC)) {
    const uint d = tile * (2 * NC) + tid;
    const float y = float(T(dec_fp4_down_sum<TOPK>(part, tid)));
    const float a = metal::fma(unit[1], unit[2], 0.0f);
    out[d] = T(metal::fma(unit[0], y, a));
  }
}

template <int XM, int K, int N, int NC, int TOPK, typename T, typename PA>
inline void seg_down_item(uint tile, uint tid, uint sgi, uint lane, uint expert, PA act, const device uint32_t* w, const device uint8_t* sc, const device uint32_t* sw, const device uint8_t* ss, const device float* unit, device T* out, threadgroup float* part) {
  seg_down_slot<XM, K, N, NC, TOPK>(tile, sgi, lane, expert, act, w, sc, sw, ss, part);
  seg_down_final<NC, TOPK, T>(tile, tid, unit, out, part);
}
