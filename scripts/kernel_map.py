#!/usr/bin/env python3
"""Per-kernel GPU time of one decode step from a Metal System Trace of the kernel_map test.

Usage: kernel_map.py <trace> --steps N --seg K [--top 15] [--json out.json]

The test (`decode/kernel_map.zig`) runs, per row count, 2 s of sleep, 3 warm-up steps, 0.5 s, N timed steps, 0.5 s.
The process's compute encoders split at GPU idle gaps of 300 ms or more; --seg K takes the K-th stretch from the end
(with row counts 1,6: 3 is the timed steps at 1 row, 1 the timed steps at 6 rows).

With an MLX built to label each encoder with the kernels it dispatched (`docs/dsv41-kernel-map.mlx-labels.patch`) and
MLX_MAX_OPS_PER_BUFFER=0 (one command buffer a primitive), an encoder's label is `name;name;`: the table groups by
those names. Without the patch the labels are generic and only the busy / idle / encoder totals mean anything.
"""
import argparse
import collections
import json
import re
import subprocess
import sys
import xml.etree.ElementTree as ET


def export(trace, schema):
    xml = subprocess.run(
        ["xcrun", "xctrace", "export", "--input", trace, "--xpath",
         f'/trace-toc/run[1]/data/table[@schema="{schema}"]'],
        check=True, capture_output=True).stdout
    return ET.fromstring(xml)


def rows(root):
    cols = [c.findtext("mnemonic") for c in root.iter("col")]
    refs = {}
    for r in root.iter("row"):
        for x in r.iter():
            if "id" in x.attrib:
                refs[x.attrib["id"]] = x
        rec = {}
        for n, cell in zip(cols, list(r)):
            rec[n] = refs.get(cell.attrib["ref"]) if "ref" in cell.attrib else cell
        yield rec


def launched_pid(trace):
    toc = subprocess.run(["xcrun", "xctrace", "export", "--input", trace, "--toc"], check=True, capture_output=True).stdout
    for p in ET.fromstring(toc).iter("process"):
        if p.get("type") in ("launched", "attached"):
            return p.get("pid")
    sys.exit("no launched or attached process in the trace")


def encoders(trace, pid):
    """(start_ns, end_ns, label) of the launched process's compute intervals (any depth: concurrent encoders nest),
    and the same for every other process, in start order."""
    out, other = [], []
    for d in rows(export(trace, "metal-gpu-intervals")):
        p = d["process"].get("fmt") or ""
        if not p:
            continue
        if not p.endswith(f"({pid})"):
            other.append((int(d["start"].text), int(d["start"].text) + int(d["duration"].text), p))
            continue
        if d["channel-name"].get("fmt") != "Compute":
            continue
        s = int(d["start"].text)
        lab = d["event-label"].get("fmt") or ""
        lab = re.sub(r"\s+\(\S+ \(\d+\)\) 0x[0-9a-f]+$", "", lab)
        lab = re.sub(r"^Command Buffer \d+:", "", lab)
        out.append((s, s + int(d["duration"].text), lab))
    out.sort()
    other.sort()
    return out, other


def stretches(enc, gap_ns=300_000_000):
    """The encoders split wherever the GPU sat idle for gap_ns or more."""
    out, cur, end = [], [], None
    for e in enc:
        if cur and e[0] - end > gap_ns:
            out.append(cur)
            cur, end = [], None
        cur.append(e)
        end = e[1] if end is None else max(end, e[1])
    out.append(cur)
    return out


def union_busy(enc):
    busy, end = 0, None
    for s, e, _ in enc:
        if end is None or s > end:
            busy += e - s
            end = e
        elif e > end:
            busy += e - end
            end = e
    return busy


GROUPS = [
    ("MoE", r"moe_|gather_qmv|gather_qmm|expert|routed|swiglu|shared_"),
    ("attention projections", r"in_proj|wq_b|wo_a|wo_b|q_kv|qmv|qmm|steel_gemm|gemv|bf16_?mm|matmul"),
    ("attention core", r"dec_attn|attn_|sdpa|nax|softmax|attn$|win_"),
    ("indexer", r"idx_|topk|index"),
    ("compressor", r"comp|pool"),
    ("hc stages", r"hc_|sinkhorn|ffn_in"),
    ("router", r"gate|route|router"),
    ("norms/rope/casts/KV writes", r"norm|rope|copy|cast|kv_row|scatter|concat|slice|add|mul|astype|rms|\bg\d"),
    ("Engram", r"engram"),
    ("head", r"head|lm_"),
]


def group_of(key):
    k = key.lower()
    for g, pat in GROUPS:
        if re.search(pat, k):
            return g
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--seg", type=int, required=True)
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--json")
    a = ap.parse_args()

    pid = launched_pid(a.trace)
    enc_all, other = encoders(a.trace, pid)
    segs = stretches(enc_all)
    print("stretches from the end (encoders):", [len(x) for x in segs[::-1][:6]])
    enc = segs[-a.seg]
    if not enc:
        sys.exit("no compute encoders for the process")
    t0, t1 = enc[0][0], max(e[1] for e in enc)
    span = t1 - t0
    foreign = collections.defaultdict(int)
    for s0, e0, p in other:
        if e0 > t0 and s0 < t1:
            foreign[p] += min(e0, t1) - max(s0, t0)
    print("other processes' GPU intervals inside the stretch (ms, summed over their encoders):",
          {k: round(v / 1e6, 1) for k, v in sorted(foreign.items(), key=lambda kv: -kv[1])[:5]})
    busy = union_busy(enc)
    n = a.steps
    print(f"timed stretch: {len(enc)} encoders, span {span / 1e3 / n:.0f} us/step, GPU busy {busy / 1e3 / n:.0f} us/step "
          f"({100 * busy / span:.1f}%), idle {(span - busy) / 1e3 / n:.0f} us/step; {len(enc) / n:.1f} encoders/step")
    gaps = []
    for (s0, e0, _), (s1, e1, _) in zip(enc, enc[1:]):
        gaps.append(max(0, s1 - e0))
    print(f"mean gap between consecutive encoders {sum(gaps) / max(1, len(gaps)) / 1e3:.1f} us; "
          f"gaps over 100 us: {sum(1 for g in gaps if g > 100_000)}")

    agg = collections.defaultdict(lambda: [0, 0, 0])  # encoders, dispatches, ns
    for s, e, lab in enc:
        names = [x for x in lab.split(";") if x]
        key = "+".join(names) if names and not lab.startswith("Compute Command") else (lab or "?")
        agg[key][0] += 1
        agg[key][1] += max(1, len(names))
        agg[key][2] += e - s
    total = sum(v[2] for v in agg.values())
    disp = sum(v[1] for v in agg.values())
    print(f"dispatches/step {disp / n:.1f} in {len(enc) / n:.1f} encoders; summed encoder time {total / 1e3 / n:.0f} us/step")
    print(f"{'kernel':<70} {'enc/step':>8} {'us/step':>9} {'us/enc':>8} {'share':>6}  group")
    table = []
    for k, (ne, nd, ns) in sorted(agg.items(), key=lambda kv: -kv[1][2]):
        table.append({"kernel": k, "encoders_per_step": ne / n, "dispatches_per_step": nd / n,
                      "us_per_step": ns / 1e3 / n, "us_per_encoder": ns / 1e3 / ne, "share": ns / total,
                      "group": group_of(k)})
    for t in table[:a.top]:
        print(f"{t['kernel'][:70]:<70} {t['encoders_per_step']:8.1f} {t['us_per_step']:9.1f} {t['us_per_encoder']:8.1f} "
              f"{100 * t['share']:5.1f}%  {t['group']}")
    grp = collections.defaultdict(lambda: [0.0, 0.0])
    for t in table:
        grp[t["group"]][0] += t["us_per_step"]
        grp[t["group"]][1] += t["dispatches_per_step"]
    print("groups:")
    for g, (us, nd) in sorted(grp.items(), key=lambda kv: -kv[1][0]):
        print(f"  {g:<30} {us:9.1f} us/step {nd:7.1f} dispatches  {100 * us / (total / 1e3 / n):5.1f}%")
    if a.json:
        json.dump({"steps": n, "span_us": span / 1e3 / n, "busy_us": busy / 1e3 / n, "table": table}, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
