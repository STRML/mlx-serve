constexpr int NSG = TH / 32;
constexpr int NCB = N / NSG;
constexpr int SH0 = 0;
constexpr int RO0 = SH0 + NCB;
constexpr int DN0 = RO0 + TOPK * NCB;
constexpr int NDN = K / (2 * NC);
constexpr int TOTAL = DN0 + NDN;
constexpr int W_TICKET = 0;
constexpr int W_ACT = 4;
constexpr int T3 = (XM == 2 ? K / 2 : 0) + 4;
constexpr int T4 = 2 * NC * (TOPK + 1);
constexpr int TGM = T3 > T4 ? T3 : T4;
const int tid = int(thread_index_in_threadgroup);
const uint lane = thread_index_in_simdgroup;
const int sg = int(simdgroup_index_in_threadgroup);
const uint gen_v = par[0];
device atomic_uint* slot = (device atomic_uint*)cnt + par[1];
device atomic_uint* st = (device atomic_uint*)status;
const float lim = as_type<float>(par[7]);
const device float* unit = (const device float*)(par + 8);
threadgroup float4 tgm4[(TGM + 3) / 4];
threadgroup float* tgm = (threadgroup float*)tgm4;
threadgroup uint item_s;
threadgroup uint flag;
threadgroup uint oks[8];

if (tid == 0 && gen_v == 0xffffffffu && dep[0] == 0xffffffffu) seg_put(st, 15u, 0u);
if (int(threadgroups_per_grid.x) != TOTAL) {
  if (tid == 0) seg_fail(st, 4u);
  return;
}
if (tid == 0) item_s = seg_arrive(slot + W_TICKET, gen_v);
threadgroup_barrier(metal::mem_flags::mem_threadgroup);
const int item = int(item_s);
if ((item < RO0 ? 2 : item < DN0 ? 3 : 4) > STOP) return;

if (item < DN0) {
  const bool shared = item < RO0;
  const int j = shared ? TOPK : (item - RO0) / NCB;
  const int cb = shared ? item - SH0 : (item - RO0) % NCB;
  const uint expert = shared ? 0u : uint(idx[j]);
  const float wt = shared ? 1.0f : float(wts[j]);
  seg_up_item<XM, NSG, K, N, TOPK, T>(tid, lane, uint(sg), uint(cb), uint(j), expert, wt, lim, gu_w, gu_s, sgu_w, sgu_s, h2, act, tgm4);
  seg_release();
  if (tid == 0) seg_arrive(slot + W_ACT + j, gen_v);
  return;
}

const uint tile = uint(item - DN0);
if (lane == 0) {
  for (uint s = uint(sg); s < uint(TOPK + 2); s += uint(NSG)) oks[s] = seg_wait_live(slot + W_ACT + metal::min(s, uint(TOPK)), seg_stamp(gen_v, uint(NCB)), st, 1u) ? 1u : 0u;
}
threadgroup_barrier(metal::mem_flags::mem_threadgroup);
uint all_ok = 1u;
for (int i = 0; i < TOPK + 2; ++i) all_ok &= oks[i];
if (all_ok == 0u) return;
seg_acquire();
for (uint s = uint(sg); s < uint(TOPK + 2); s += uint(NSG)) {
  seg_down_slot<XM, N, K, NC, TOPK>(tile, s, lane, s < uint(TOPK) ? uint(idx[s]) : 0u, act, dn_w, dn_s, sdn_w, sdn_s, tgm);
}
seg_down_final<NC, TOPK, T>(tile, uint(tid), unit, part, tgm);
