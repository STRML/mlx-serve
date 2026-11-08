#!/usr/bin/env python3
"""The per-kernel table of one 40-layer decode step from kernel_map.py's two halves.

Usage: kernel_map_report.py --rows R --patched-a A.trace --patched-b B.trace --stock-a A.trace --stock-b B.trace
           --seg K [--top 15]
(--seg is the stretch kernel_map.py takes: 4 for the 1-row phase, 1 for the 6-row phase of the kernel_map test).

The patched traces (MLX labelling each encoder with its kernels, one primitive a command buffer) name the kernels; the
stock traces give the step's real span, busy time and idle. Halves are layers 0-20 and 20-40 and add up to the step.
"""
import argparse
import collections
import re

import kernel_map as km

GROUPS = [
    ("MoE", r"dsv41_dec_moe_fp4_(up|down)"),
    ("attention projections", r"dsv41_dec_(in_proj|wq_b|wo_a)|^mxfp8_qmv|^affine_qmv|steel_gemm"),
    ("attention core", r"dsv41_dec_attn"),
    ("indexer", r"dsv41_dec_(idx_score|topk)"),
    ("compressor", r"dsv41_dec_idx_pool"),
    ("hc stages", r"dsv41_hc_|glm5_hc_"),
    ("router", r"dsv41_(gate|route)"),
    ("norms/rope/casts/KV writes", r"_copy|^rms|^ss_|^vs_|^sv_|^vv_|rope|kv_row"),
]


def group_of(key):
    for g, pat in GROUPS:
        if re.search(pat, key):
            return g
    return "other"


def short(key):
    """The kernel's name without its template arguments and without the `custom_kernel_` prefix."""
    k = key.removeprefix("custom_kernel_")
    k = re.sub(r"_(\d+|bfloat16_t|float|floatc|uint32_t|int32_t|uint8_t|int32_tc|f|t)(_|$).*", "", k)
    return k


def load(trace, steps, seg):
    """kernel -> [encoders/step, dispatches/step, us/step]; the stretch's span and busy us/step."""
    pid = km.launched_pid(trace)
    enc_all, _ = km.encoders(trace, pid)
    enc = km.stretches(enc_all)[-seg]
    span = max(e[1] for e in enc) - enc[0][0]
    busy = km.union_busy(enc)
    agg = collections.defaultdict(lambda: [0.0, 0.0, 0.0])
    for s, e, lab in enc:
        names = [x for x in lab.split(";") if x]
        key = "+".join(names) if names and not lab.startswith("Compute Command") else (lab or "?")
        agg[key][0] += 1 / steps
        agg[key][1] += max(1, len(names)) / steps
        agg[key][2] += (e - s) / 1e3 / steps
    return agg, span / 1e3 / steps, busy / 1e3 / steps, len(enc) / steps


def merge_halves(a, b, steps, seg):
    """kernel -> [encoders, dispatches, us] per step summed over the two halves; plus the halves' span and busy."""
    merged = collections.defaultdict(lambda: [0.0, 0.0, 0.0])
    span = busy = enc = 0.0
    for t in (a, b):
        agg, sp, bu, n = load(t, steps, seg)
        span += sp
        busy += bu
        enc += n
        for k, v in agg.items():
            m = merged[short(k)]
            for i in range(3):
                m[i] += v[i]
    return merged, span, busy, enc


def mean_runs(runs):
    out = collections.defaultdict(lambda: [0.0, 0.0, 0.0])
    for m in runs:
        for k, v in m.items():
            for i in range(3):
                out[k][i] += v[i] / len(runs)
    return out


def group_totals(merged):
    grp = collections.defaultdict(lambda: [0.0, 0.0])
    for k, (ne, nd, us) in merged.items():
        grp[group_of(k)][0] += us
        grp[group_of(k)][1] += nd
    return grp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, required=True)
    ap.add_argument("--patched", required=True, help="A1.trace:B1.trace,A2.trace:B2.trace (repeats, halves 0-20 : 20-40)")
    ap.add_argument("--stock", required=True, help="the same for the stock-MLX traces")
    ap.add_argument("--seg", type=int, required=True)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--top", type=int, default=15)
    a = ap.parse_args()

    pr = [merge_halves(*p.split(":"), a.steps, a.seg) for p in a.patched.split(",")]
    sr = [merge_halves(*p.split(":"), a.steps, a.seg) for p in a.stock.split(",")]
    merged = mean_runs([r[0] for r in pr])
    lab_span = sum(r[1] for r in pr) / len(pr)
    st_span = sum(r[1] for r in sr) / len(sr)
    st_busy = sum(r[2] for r in sr) / len(sr)
    st_enc = sum(r[3] for r in sr) / len(sr)
    totals = [sum(v[2] for v in r[0].values()) for r in pr]
    total = sum(v[2] for v in merged.values())
    disp = sum(v[1] for v in merged.values())
    scale = st_busy / total
    print(f"rows = {a.rows}, 40 layers (layers 0-20 + 20-40), per step, mean of {len(pr)} labelled and {len(sr)} stock runs")
    print("stock MLX runs: " + "; ".join(f"span {r[1]:.0f} busy {r[2]:.0f} us" for r in sr))
    print(f"real step (stock MLX): span {st_span:.0f} us, GPU busy {st_busy:.0f} us ({100 * st_busy / st_span:.1f}%), "
          f"idle {st_span - st_busy:.0f} us ({100 * (st_span - st_busy) / st_span:.1f}%), {st_enc:.0f} encoders")
    print(f"labelled runs: kernels sum to {', '.join(f'{t:.0f}' for t in totals)} us in {disp:.0f} dispatches; serialized span "
          f"{lab_span:.0f} us (one command buffer a primitive); scale to the real busy time x{scale:.3f}")
    print(f"{'kernel':<34} {'launches':>8} {'us/step':>9} {'scaled':>8} {'us/launch':>9} {'share':>6}  group")
    rows = sorted(merged.items(), key=lambda kv: -kv[1][2])
    for k, (ne, nd, us) in rows[:a.top]:
        print(f"{k[:34]:<34} {nd:8.1f} {us:9.0f} {us * scale:8.0f} {us / max(nd, 1e-9):9.1f} {100 * us / total:5.1f}%  {group_of(k)}")
    rest = rows[a.top:]
    if rest:
        us = sum(v[2] for _, v in rest)
        nd = sum(v[1] for _, v in rest)
        print(f"{'(' + str(len(rest)) + ' more)':<34} {nd:8.1f} {us:9.0f} {us * scale:8.0f} {'':>9} {100 * us / total:5.1f}%")
    grp = group_totals(merged)
    per_run = [group_totals(r[0]) for r in pr]
    print("groups (us/step, scaled, dispatches, share, spread across the labelled runs):")
    for g, (us, nd) in sorted(grp.items(), key=lambda kv: -kv[1][0]):
        vals = [x[g][0] for x in per_run if g in x]
        print(f"  {g:<30} {us:8.0f} us {us * scale:7.0f} scaled {nd:7.1f} dispatches {100 * us / total:5.1f}%  "
              f"[{min(vals):.0f}..{max(vals):.0f}]")


if __name__ == "__main__":
    main()
