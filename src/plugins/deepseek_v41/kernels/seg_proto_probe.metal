constexpr int N = GROUPS * THREADS;
constexpr int SERIAL = ITEMS > 0 ? ITEMS : GROUPS;
constexpr int WAIT_POLLS = POLLS > 0 ? POLLS : SEG_SPIN_POLLS_DEFAULT;
const uint tid = thread_position_in_threadgroup.x;
const uint grp = threadgroup_position_in_grid.x;
const uint gen_v = gen[0];
device atomic_uint* c = (device atomic_uint*)cnt;
device atomic_uint* st = (device atomic_uint*)status;
device atomic_uint* data = (device atomic_uint*)buf;
device uint* pdata = (device uint*)buf;
#if SEG_COHERENT
device coherent(device) uint* cdata = (device coherent(device) uint*)buf;
#endif
threadgroup uint pad[TGMEM >= 4 ? TGMEM / 4 : 1];
threadgroup uint ticket_slot;
threadgroup uint go;

#if SEG_COHERENT
#define CLOAD(IDX) cdata[IDX]
#define CSTORE(IDX, V) cdata[IDX] = (V)
#else
#define CLOAD(IDX) seg_get(data, IDX)
#define CSTORE(IDX, V) seg_put(data, IDX, V)
#endif

#define LOAD(IDX) (ACCESS == 0 ? seg_get(data, IDX) : ACCESS == 1 ? pdata[IDX] : ACCESS == 2 ? CLOAD(IDX) : ACCESS == 3 ? seg_get(data, IDX) : seg_read_peer_output(pdata, IDX))

#define STORE(IDX, V) do { \
  if (ACCESS == 0) seg_put(data, IDX, V); \
  else if (ACCESS == 2) CSTORE(IDX, V); \
  else if (ACCESS == 5) seg_write_peer_output(pdata, IDX, V); \
  else pdata[IDX] = (V); \
} while (0)

#define SPOT(P, GG, T) (uint(P) * N + (LAYOUT == 1 ? uint(T) * GROUPS + uint(GG) : uint(GG) * THREADS + uint(T)))

#define ITEM(P, GG) do { \
  const uint gid_ = uint(GG) * THREADS + tid; \
  const uint at_ = SPOT(P, GG, tid); \
  uint v_ = gid_ + gen_v * SALT; \
  if ((P) > 0) v_ = LOAD(SPOT(uint(P) - 1, (uint(GG) + uint(P)) % GROUPS, tid)) + uint(P) * STEP + gen_v * SALT; \
  STORE(at_, v_); \
} while (0)

#define PRELOAD(IDX) do { \
  if (ACCESS != 0) { \
    const uint pv_ = pdata[IDX]; \
    if (pv_ == 0xA5A5A5A5u) seg_put(st, 14u, pv_); \
  } \
} while (0)

#define ARRIVE_COUNT(SLOT) do { \
  if (seg_arrive_last(c + (SLOT), gen_v, GROUPS)) atomic_fetch_add_explicit(st + (SLOT), 1u, metal::memory_order_relaxed); \
} while (0)

#define PUBLISH_AND_WAIT(SLOT) do { \
  seg_release(); \
  if (tid == 0) { \
    ARRIVE_COUNT(SLOT); \
    go = seg_wait<WAIT_POLLS>(c + (SLOT), seg_stamp(gen_v, GROUPS), st, 1u) ? 1u : 0u; \
  } \
  threadgroup_barrier(metal::mem_flags::mem_threadgroup); \
  if (go == 0u) return; \
  seg_acquire(); \
} while (0)

uint acc = 0u;
if (TGMEM >= 4) {
  const uint words = TGMEM / 4;
  pad[tid % words] = tid + gen_v;
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  acc = pad[(tid + 1) % words];
}
if (grp == 0 && tid == 0) {
  sink[0] = dep[0] + 1u;
  if (gen_v == 0xffffffffu) sink[1] = acc;
}

if (PROTO == 0) {
  if (tid == 0) ticket_slot = seg_ticket(c) - (gen_v - 1u) * uint(PHASES * GROUPS);
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  const uint t = ticket_slot;
  if (t >= uint(PHASES * GROUPS)) {
    if (tid == 0) seg_fail(st, 1u);
    return;
  }
  const uint p = t / GROUPS;
  const uint g = t % GROUPS;
  if (p > 0) {
    PRELOAD(SPOT(p - 1, (g + p) % GROUPS, tid));
    if (tid == 0) go = seg_wait<WAIT_POLLS>(c + p, seg_stamp(gen_v, GROUPS), st, 1u) ? 1u : 0u;
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    if (go == 0u) return;
    seg_acquire();
  }
  ITEM(p, g);
  if (p + 1 < PHASES) {
    seg_release();
    if (tid == 0) ARRIVE_COUNT(1 + p);
  }
} else if (PROTO == 1) {
  for (int p = 0; p < PHASES; ++p) {
    ITEM(p, grp);
    if (p + 1 < PHASES) {
      PRELOAD(SPOT(p, (grp + p + 1) % GROUPS, tid));
      PUBLISH_AND_WAIT(1 + p);
    }
  }
} else if (PROTO == 2) {
  ITEM(0, grp);
  if (PHASES > 1) {
    PRELOAD(SPOT(0, (grp + 1) % GROUPS, tid));
    seg_release();
    if (tid == 0) {
      go = seg_arrive_last(c + 1, gen_v, GROUPS) ? 1u : 0u;
      if (go != 0u) atomic_fetch_add_explicit(st + 1, 1u, metal::memory_order_relaxed);
    }
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    if (go == 0u) return;
    seg_acquire();
    for (int p = 1; p < PHASES; ++p) {
      for (int g = 0; g < SERIAL; ++g) ITEM(p, g);
      seg_release();
    }
  }
} else if (PROTO == 3) {
  ITEM(arg[0], grp);
} else if (PROTO == 4) {
  PUBLISH_AND_WAIT(1);
} else {
  if (tid == 0) go = seg_wait<WAIT_POLLS>(c + 1, seg_stamp(gen_v, 60000u), st, 1u) ? 1u : 0u;
}
