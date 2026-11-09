#define SEG_SPIN_POLLS_DEFAULT 40000
#define SEG_POLL_PAUSE 128

inline uint seg_stamp(uint gen, uint n) {
  return (gen << 16) | n;
}

inline void seg_fail(device atomic_uint* status, uint code) {
  uint seen = 0u;
  while (!atomic_compare_exchange_weak_explicit(status, &seen, code, metal::memory_order_relaxed, metal::memory_order_relaxed)) {
    if (seen != 0u) return;
  }
}

inline void seg_release() {
  metal::atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst, metal::thread_scope_device);
  threadgroup_barrier(metal::mem_flags::mem_device);
}

inline void seg_acquire() {
  metal::atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst, metal::thread_scope_device);
}

inline uint seg_arrive(device atomic_uint* c, uint gen) {
  const uint old = atomic_fetch_add_explicit(c, 1u, metal::memory_order_relaxed);
  if ((old >> 16) == gen) return old & 0xffffu;
  while (true) {
    uint v = atomic_load_explicit(c, metal::memory_order_relaxed);
    if ((v >> 16) == gen) return atomic_fetch_add_explicit(c, 1u, metal::memory_order_relaxed) & 0xffffu;
    if (atomic_compare_exchange_weak_explicit(c, &v, seg_stamp(gen, 1u), metal::memory_order_relaxed, metal::memory_order_relaxed)) return 0u;
  }
}

inline bool seg_arrive_last(device atomic_uint* c, uint gen, uint n) {
  return seg_arrive(c, gen) + 1u == n;
}

inline int seg_pause(int seed) {
  uint x = uint(seed);
  for (int j = 0; j < SEG_POLL_PAUSE; ++j) x = x * 1664525u + 1013904223u;
  return int(x == 0x9e3779b9u);
}

template <int POLLS = SEG_SPIN_POLLS_DEFAULT>
inline bool seg_wait(device atomic_uint* c, uint target, device atomic_uint* status, uint code) {
  const uint want_gen = target >> 16;
  const uint want_n = target & 0xffffu;
  for (int i = 0; i < POLLS; ++i) {
    const uint v = atomic_load_explicit(c, metal::memory_order_relaxed);
    if ((v >> 16) == want_gen && (v & 0xffffu) >= want_n) return true;
    if ((i & 15) == 0 && atomic_load_explicit(status, metal::memory_order_relaxed) != 0u) return false;
    i += seg_pause(i);
  }
  seg_fail(status, code);
  return false;
}

inline bool seg_wait_live(device atomic_uint* c, uint target, device atomic_uint* status, uint code) {
  return seg_wait(c, target, status, code) && atomic_load_explicit(status, metal::memory_order_relaxed) == 0u;
}

inline void seg_touch(device uint* sink, uint v) {
  if (v == 0x9e3779b9u) sink[14] = v;
}

inline uint seg_ticket(device atomic_uint* t) {
  return atomic_fetch_add_explicit(t, 1u, metal::memory_order_relaxed);
}

#if __METAL_VERSION__ >= 320
#define SEG_COHERENT 1
inline uint seg_read_peer_output(const device uint* p, uint i) {
  return ((const device coherent(device) uint*)p)[i];
}
inline void seg_write_peer_output(device uint* p, uint i, uint v) {
  ((device coherent(device) uint*)p)[i] = v;
}
#else
#define SEG_COHERENT 0
inline uint seg_read_peer_output(const device uint* p, uint i) {
  return atomic_load_explicit((device atomic_uint*)(p + i), metal::memory_order_relaxed);
}
inline void seg_write_peer_output(device uint* p, uint i, uint v) {
  atomic_store_explicit((device atomic_uint*)(p + i), v, metal::memory_order_relaxed);
}
#endif

inline float seg_read_peer_f32(const device float* p, uint i) {
  return as_type<float>(seg_read_peer_output((const device uint*)p, i));
}

inline void seg_write_peer_f32(device float* p, uint i, float v) {
  seg_write_peer_output((device uint*)p, i, as_type<uint>(v));
}

inline float2 seg_read_peer_bf16x2(const device uint* p, uint i) {
  const uint v = seg_read_peer_output(p, i);
  return float2(as_type<float>(v << 16), as_type<float>(v & 0xffff0000u));
}

struct seg_peer4 {
  const device uint4* p;
  inline uint4 operator[](uint i) const {
#if SEG_COHERENT
    return ((const device coherent(device) uint4*)p)[i];
#else
    const device uint* q = (const device uint*)(p + i);
    return uint4(seg_read_peer_output(q, 0), seg_read_peer_output(q, 1), seg_read_peer_output(q, 2), seg_read_peer_output(q, 3));
#endif
  }
  inline seg_peer4 operator+(uint n) const { return seg_peer4{p + n}; }
};

inline void seg_write_peer_uint4(device uint4* p, uint i, uint4 v) {
  device uint* q = (device uint*)(p + i);
  for (int j = 0; j < 4; ++j) seg_write_peer_output(q, uint(j), v[j]);
}

inline uint seg_get(device atomic_uint* p, uint i) {
  return atomic_load_explicit(p + i, metal::memory_order_relaxed);
}

inline void seg_put(device atomic_uint* p, uint i, uint v) {
  atomic_store_explicit(p + i, v, metal::memory_order_relaxed);
}
