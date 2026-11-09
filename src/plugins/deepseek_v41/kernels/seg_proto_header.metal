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
  for (int i = 0; i < POLLS; ++i) {
    if (atomic_load_explicit(c, metal::memory_order_relaxed) >= target) return true;
    if ((i & 15) == 0 && atomic_load_explicit(status, metal::memory_order_relaxed) != 0u) return false;
    i += seg_pause(i);
  }
  seg_fail(status, code);
  return false;
}

inline uint seg_ticket(device atomic_uint* t) {
  return atomic_fetch_add_explicit(t, 1u, metal::memory_order_relaxed);
}

inline uint seg_get(device atomic_uint* p, uint i) {
  return atomic_load_explicit(p + i, metal::memory_order_relaxed);
}

inline void seg_put(device atomic_uint* p, uint i, uint v) {
  atomic_store_explicit(p + i, v, metal::memory_order_relaxed);
}
