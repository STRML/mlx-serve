constexpr int MIX = (2 + HC) * HC;
constexpr int PL = MIX + 1;
constexpr int NCH = D / 8;
constexpr int NRT = G * ((R + RG - 1) / RG);
constexpr int TH = RG * NCH * 4;
constexpr int T0 = TH + 16 * RG;
constexpr int T1 = R * P * PL + R * PL;
constexpr int T2 = 2 * R * E;
constexpr int T3 = K / 2 + 4;
constexpr int T4 = 2 * NC * (TOPK + 1);
constexpr int T01 = T0 > T1 ? T0 : T1;
constexpr int T23 = T3 > T4 ? T3 : T4;
constexpr int T0123 = T01 > T23 ? T01 : T23;
constexpr int TGM = T0123 > T2 ? T0123 : T2;
constexpr int NIN = P + NRT;
constexpr int NCB = N / 8;
constexpr int SH0 = NIN;
constexpr int RO0 = SH0 + NCB;
constexpr int DN0 = RO0 + TOPK * NCB;
constexpr int NDN = D / (2 * NC);
constexpr int TOTAL = DN0 + NDN;
constexpr int W_TICKET = 0;
constexpr int W_MIX = 1;
constexpr int W_ROUTER = 2;
constexpr int W_TOPK = 3;
constexpr int W_ACT = 4;
const int tid = int(thread_index_in_threadgroup);
const uint lane = thread_index_in_simdgroup;
const int sg = int(simdgroup_index_in_threadgroup);
const uint gen_v = par[0];
device atomic_uint* slot = (device atomic_uint*)cnt + par[1];
device atomic_uint* st = (device atomic_uint*)status;
const uint abort_at = par[2];
const uint skew_at = par[3];
const device float* eps = (const device float*)(par + 4);
const device float* scal = (const device float*)(par + 6);
const float lim = as_type<float>(par[7]);
const device float* unit = (const device float*)(par + 8);
device float* parts = scratch;
device float* logits = scratch + OFF_LOGITS;
device T* h2 = (device T*)(scratch + OFF_H2);
device uint* idx = (device uint*)(scratch + OFF_IDX);
device float* wts = scratch + OFF_WTS;
device T* act = (device T*)(scratch + OFF_ACT);
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
if ((item < NIN ? 1 : item < RO0 ? 2 : item < DN0 ? 3 : 4) > STOP) return;

if (item < NIN) {
  const bool is_mix = item < P;
  if (is_mix) {
    seg_ffn_mix<R, D, HC, P, T>(item, lane, sg, x, y, spost, scomb, fn, x1, parts);
  } else {
    seg_ffn_router<R, D, E, HC, P, G, RG, T>(item, tid, lane, sg, x, y, spre, spost, scomb, w, gate_w, eps, h2, logits, tgm4);
  }
  seg_release();
  if (tid == 0) flag = seg_arrive_last(slot + (is_mix ? W_MIX : W_ROUTER), gen_v, is_mix ? uint(P) : uint(NRT)) ? 1u : 0u;
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  if (flag != 0u) {
    seg_acquire();
    if (is_mix) {
      seg_ffn_split<R, D, HC, P, ITERS>(tid, lane, sg, tgm, parts, eps, scale, base, pre, post, comb);
    } else {
      seg_ffn_topk<R, E, TOPK, NORM>(tid, lane, sg, tgm, logits, bias, scal, idx, wts);
      seg_release();
      if (tid == 0) seg_arrive(slot + W_TOPK, gen_v);
    }
  }
  if (abort_at == 1u && tid == 0) seg_fail(st, 3u);
  return;
}

if (item < DN0) {
  const bool shared = item < RO0;
  const int j = shared ? TOPK : (item - RO0) / NCB;
  const int cb = shared ? item - SH0 : (item - RO0) % NCB;
  const uint n_col = uint(cb * 8 + sg);
  uint pf = 0u;
  uint expert = 0u;
  float wt = 1.0f;
  if (shared) {
    pf = sgu_w[size_t(n_col) * (K / 4) + lane * 8] ^ sgu_w[size_t(n_col + N) * (K / 4) + lane * 8];
    if (tid == 0) flag = seg_wait_live(slot + W_ROUTER, seg_stamp(gen_v, uint(NRT) + (skew_at == 2u ? 1u : 0u)), st, 1u) ? 1u : 0u;
  } else {
    if (tid == 0) flag = seg_wait_live(slot + W_TOPK, seg_stamp(gen_v, 1u + (skew_at == 3u ? 1u : 0u)), st, 1u) ? 1u : 0u;
  }
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  if (flag == 0u) return;
  seg_acquire();
  if (!shared) {
    expert = seg_read_peer_output(idx, uint(j));
    wt = seg_read_peer_f32(wts, uint(j));
  }
  seg_up_item<XM, 8, K, N, TOPK, T>(tid, lane, uint(sg), uint(cb), uint(j), expert, wt, lim, gu_w, gu_s, sgu_w, sgu_s, h2, act, tgm4);
  seg_touch((device uint*)status, pf);
  seg_release();
  if (tid == 0) seg_arrive(slot + W_ACT + j, gen_v);
  if (abort_at == (shared ? 2u : 3u) && tid == 0) seg_fail(st, 3u);
  return;
}

const uint tile = uint(item - DN0);
if (tid == 0) flag = seg_wait_live(slot + W_TOPK, seg_stamp(gen_v, 1u), st, 1u) ? 1u : 0u;
threadgroup_barrier(metal::mem_flags::mem_threadgroup);
if (flag == 0u) return;
seg_acquire();
const uint sgi = uint(sg);
const uint slot_j = metal::min(sgi, uint(TOPK));
const uint expert = sgi < uint(TOPK) ? seg_read_peer_output(idx, sgi) : 0u;
const uint dcol = tile * (2 * NC) + lane % NC;
const uint pf = sgi < uint(TOPK) ? dn_w[(size_t(expert) * K + dcol) * (N / 8) + (lane / NC) * 4] : sdn_w[size_t(dcol) * (N / 4) + (lane / NC) * 8];
if (lane == 0) oks[sg] = seg_wait_live(slot + W_ACT + slot_j, seg_stamp(gen_v, uint(NCB) + (skew_at == 4u ? 1u : 0u)), st, 1u) ? 1u : 0u;
threadgroup_barrier(metal::mem_flags::mem_threadgroup);
uint all_ok = 1u;
for (int i = 0; i < 8; ++i) all_ok &= oks[i];
if (all_ok == 0u) return;
seg_acquire();
seg_down_item<XM, N, K, NC, TOPK, T>(tile, uint(tid), sgi, lane, expert, act, dn_w, dn_s, sdn_w, sdn_s, unit, part, tgm);
seg_touch((device uint*)status, pf);
if (abort_at == 4u && tid == 0) seg_fail(st, 3u);
