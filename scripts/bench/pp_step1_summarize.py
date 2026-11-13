#!/usr/bin/env python3
"""Summarize Plan 75 Step 1 bound-measurement JSONL into R_bound(N, P, C) and the Go/No-Go gate."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from typing import Any

CED_MIN_CHUNK = 384
CED_TAIL_ROWS = 128
PAIR_SEAM_ROW_BYTES = 4 * 5120 * 2 + 4 * 4
MID_GROUP_SEAM_ROW_BYTES = PAIR_SEAM_ROW_BYTES + 512 * 4 + 256 * 2
WIN_RING_BYTES = 144 * 512 * 2
RELEASE_FIXED_EXPORT_BYTES = 40 * WIN_RING_BYTES + 3 * WIN_RING_BYTES + 15360 * 2
RELEASE_ODD_PEND_BYTES = 3 * 2 * 512 * 4
LOSSLESS_TP4_REF_TOK_S = (3623.0, 3697.0)

SCHEDULES: dict[tuple[str, int], list[tuple[int, int, bool, str]]] = {
    ("A", 2): [(0, 10, False, "mid_group"), (10, 20, True, "tail")],
    ("A", 3): [(0, 7, False, "mid_group"), (7, 14, False, "pair_cut"), (14, 20, True, "tail")],
    ("A", 4): [
        (0, 5, False, "mid_group"),
        (5, 10, False, "mid_group"),
        (10, 15, False, "mid_group"),
        (15, 20, True, "tail"),
    ],
    ("B", 2): [(0, 8, False, "pair_cut"), (8, 20, True, "tail")],
    ("B", 3): [(0, 8, False, "pair_cut"), (8, 14, False, "pair_cut"), (14, 20, True, "tail")],
    ("B", 4): [
        (0, 2, False, "pair_cut"),
        (2, 8, False, "pair_cut"),
        (8, 14, False, "pair_cut"),
        (14, 20, True, "tail"),
    ],
}

PROMPT_LADDER = [8192, 16384, 32768, 49152, 65536, 131072]
CHUNK_WIDTHS = [2048, 4096, 8192]
PP_WIDTHS = [2, 3, 4]


def seam_payload_bytes(rows: int, seam: str) -> int:
    per_row = MID_GROUP_SEAM_ROW_BYTES if seam == "mid_group" else PAIR_SEAM_ROW_BYTES
    return rows * per_row


def expected_plan_export_bytes(n_snap: int) -> int:
    pend = RELEASE_ODD_PEND_BYTES if (n_snap % 2 == 1) else 0
    return RELEASE_FIXED_EXPORT_BYTES + pend + 3840 * (n_snap // 2) + 1284 * n_snap


def non_stage0_kv_rows_bytes(n_snap: int) -> int:
    return 2 * 1280 * (n_snap // 2) + 1280 * n_snap


def plan_pp_chunks(n_snap: int, chunk_c: int) -> list[int]:
    if n_snap < CED_MIN_CHUNK:
        raise ValueError(f"n_snap={n_snap} < {CED_MIN_CHUNK}")
    if chunk_c < 2 * CED_MIN_CHUNK or chunk_c % 2 != 0:
        raise ValueError(f"invalid chunk_c={chunk_c}")
    m = (n_snap + chunk_c - 1) // chunk_c
    if m == 1:
        return [n_snap]
    chunks = [chunk_c] * (m - 2)
    rem = n_snap - (m - 2) * chunk_c
    if rem - chunk_c >= CED_MIN_CHUNK:
        chunks.extend([chunk_c, rem - chunk_c])
    else:
        chunks.extend([chunk_c - CED_MIN_CHUNK, rem - chunk_c + CED_MIN_CHUNK])
    return chunks


def interp_1d(points: dict[int, float], x: float) -> float:
    if not points:
        raise KeyError("empty interpolation table")
    xs = sorted(points.keys())
    if x <= xs[0]:
        return points[xs[0]]
    if x >= xs[-1]:
        return points[xs[-1]]
    for i in range(len(xs) - 1):
        x0, x1 = xs[i], xs[i + 1]
        if x0 <= x <= x1:
            if x1 == x0:
                return points[x0]
            t = (x - x0) / float(x1 - x0)
            return points[x0] * (1.0 - t) + points[x1] * t
    return points[xs[-1]]


REQUIRED_FIELDS: dict[str, set[str]] = {
    "load_check": {"ctx_size", "prefix_cache_mem", "prefill_chunk", "narrowed", "cache_ram_enabled", "cache_mem_bytes"},
    "single_box_cold": {"prompt_tokens", "cached_tokens", "warmup", "elapsed_us", "tok_per_s"},
    "stage_timing": {
        "family",
        "pp_width",
        "schedule",
        "stage_idx",
        "lo",
        "hi",
        "l20_comp",
        "seam",
        "chunk_rows",
        "p0",
        "warmup",
        "elapsed_us",
    },
    "tail_timing": {"schedule", "lo", "hi", "rows", "chunk_rows", "p0", "warmup", "elapsed_us"},
    "layer_diag": {
        "layer",
        "is_l20_comp",
        "chunk_rows",
        "p0",
        "warmup",
        "total_us",
        "engram_pread_us",
        "engram_wait_us",
        "engram_eval_us",
        "indexer_us",
    },
    "tcp_hop": {"mode", "chunk_rows", "seam", "bytes", "iters", "median_us", "gbps", "mtu"},
    "rdma_hop": {"bytes", "iters", "one_way_mean_us", "gbps"},
    "tcp_gpu_overlap": {
        "chunk_rows",
        "seam",
        "bytes",
        "gpu_solo_median_us",
        "tcp_solo_median_us",
        "gpu_overlap_median_us",
        "tcp_overlap_median_us",
        "step_wall_median_us",
    },
    "kv_export_check": {
        "prompt_tokens",
        "n_snap",
        "export_bytes",
        "plan_bytes",
        "formula_bytes",
        "match",
    },
    "kv_stream": {
        "subtest",
        "prompt_tokens",
        "chunk_rows",
        "wire_bytes",
        "tcp_median_us",
        "fill_rows_median_us",
        "total_median_us",
    },
}


def validate_record(rec: dict[str, Any]) -> None:
    if not isinstance(rec, dict) or "kind" not in rec:
        raise ValueError("JSONL record missing 'kind'")
    kind = rec["kind"]
    req = REQUIRED_FIELDS.get(kind)
    if req is None:
        raise ValueError(f"unknown record kind: {kind!r}")
    missing = req - set(rec.keys())
    if missing:
        raise ValueError(f"record kind={kind!r} missing fields: {sorted(missing)}")


def load_jsonl_paths(paths: list[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    files: list[str] = []
    for p in paths:
        if os.path.isdir(p):
            for name in sorted(os.listdir(p)):
                if name.endswith(".jsonl"):
                    files.append(os.path.join(p, name))
        else:
            files.append(p)
    for fp in files:
        with open(fp, "r", encoding="utf-8") as fh:
            for lineno, raw in enumerate(fh, 1):
                line = raw.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    validate_record(rec)
                except Exception as exc:
                    raise ValueError(f"{fp}:{lineno}: {exc}") from exc
                records.append(rec)
    return records


class Dataset:
    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = records
        self.load_checks: list[dict[str, Any]] = []
        self.kv_checks: dict[int, dict[str, Any]] = {}
        self.single_box_tps: dict[int, float] = {}
        self.warm_cold_hits: list[tuple[int, int]] = []
        self.stage_us: dict[tuple[str, int, int, int], dict[int, float]] = {}
        self.tail_us: dict[int, float] = {}
        self.tcp_one_way_us: dict[tuple[int, str], float] = {}
        self.tcp_pipelined_us: dict[tuple[int, str], float] = {}
        self.tcp_mtu: int = 0
        self.tcp_iface: str = ""
        self.rdma_one_way_us: dict[int, float] = {}
        self.overlap_gpu_ratio: dict[int, float] = {}
        self.overlap_tcp_ratio: dict[int, float] = {}
        self.kv_drain_us: dict[int, float] = {}
        self.kv_full_us: dict[int, float] = {}
        self.layer_diag: dict[tuple[int, int, int, bool], dict[str, float]] = {}
        self._aggregate()

    def _aggregate(self) -> None:
        cold_samples: dict[int, list[float]] = {}
        stage_samples: dict[tuple[str, int, int, int, int], list[float]] = {}
        tail_samples: dict[int, list[float]] = {}
        diag_samples: dict[tuple[int, int, int, bool], list[dict[str, Any]]] = {}

        for r in self.records:
            k = r["kind"]
            if k == "load_check":
                self.load_checks.append(r)
            elif k == "kv_export_check":
                self.kv_checks[int(r["prompt_tokens"])] = r
            elif k == "single_box_cold":
                if int(r["cached_tokens"]) != 0:
                    self.warm_cold_hits.append((int(r["prompt_tokens"]), int(r["cached_tokens"])))
                if not r["warmup"]:
                    cold_samples.setdefault(int(r["prompt_tokens"]), []).append(float(r["tok_per_s"]))
            elif k == "stage_timing":
                if not r["warmup"]:
                    key = (
                        str(r["family"]),
                        int(r["pp_width"]),
                        int(r["chunk_rows"]),
                        int(r["stage_idx"]),
                        int(r["p0"]),
                    )
                    stage_samples.setdefault(key, []).append(float(r["elapsed_us"]))
            elif k == "tail_timing":
                if not r["warmup"]:
                    tail_samples.setdefault(int(r["p0"]), []).append(float(r["elapsed_us"]))
            elif k == "layer_diag":
                if not r["warmup"]:
                    dkey = (int(r["chunk_rows"]), int(r["p0"]), int(r["layer"]), bool(r["is_l20_comp"]))
                    diag_samples.setdefault(dkey, []).append(r)
            elif k == "tcp_hop":
                key2 = (int(r["chunk_rows"]), str(r["seam"]))
                if r["mode"] == "one_way":
                    self.tcp_one_way_us[key2] = float(r["median_us"])
                elif r["mode"] == "pipelined":
                    self.tcp_pipelined_us[key2] = float(r["median_us"])
                self.tcp_mtu = max(self.tcp_mtu, int(r.get("mtu", 0)))
                if r.get("iface"):
                    self.tcp_iface = str(r["iface"])
            elif k == "rdma_hop":
                self.rdma_one_way_us[int(r["bytes"])] = float(r["one_way_mean_us"])
            elif k == "tcp_gpu_overlap":
                c = int(r["chunk_rows"])
                gpu_solo = max(1.0, float(r["gpu_solo_median_us"]))
                tcp_solo = max(1.0, float(r["tcp_solo_median_us"]))
                self.overlap_gpu_ratio[c] = max(1.0, float(r["gpu_overlap_median_us"]) / gpu_solo)
                self.overlap_tcp_ratio[c] = max(1.0, float(r["tcp_overlap_median_us"]) / tcp_solo)
            elif k == "kv_stream":
                if r["subtest"] == "final_chunk_drain":
                    self.kv_drain_us[int(r["chunk_rows"])] = float(r["total_median_us"])
                elif r["subtest"] == "full_non_stage0":
                    self.kv_full_us[int(r["prompt_tokens"])] = float(r["total_median_us"])

        for n_tok, vals in cold_samples.items():
            self.single_box_tps[n_tok] = statistics.median(vals)
        for (fam, pp_w, c_rows, s_idx, p0), vals in stage_samples.items():
            self.stage_us.setdefault((fam, pp_w, c_rows, s_idx), {})[p0] = statistics.median(vals)
        for p0, vals in tail_samples.items():
            self.tail_us[p0] = statistics.median(vals)
        for dkey, rows in diag_samples.items():
            self.layer_diag[dkey] = {
                "total_us": statistics.median(float(x["total_us"]) for x in rows),
                "engram_pread_us": statistics.median(float(x["engram_pread_us"]) for x in rows),
                "engram_eval_us": statistics.median(float(x["engram_eval_us"]) for x in rows),
                "indexer_us": statistics.median(float(x["indexer_us"]) for x in rows),
            }

    def hop_solo_us(self, chunk_rows: int, seam: str, transport: str) -> float:
        if transport == "rdma":
            b = seam_payload_bytes(chunk_rows, seam)
            if b in self.rdma_one_way_us:
                return self.rdma_one_way_us[b]
            return interp_1d(self.rdma_one_way_us, float(b))
        key = (chunk_rows, seam)
        if key in self.tcp_one_way_us:
            return self.tcp_one_way_us[key]
        fallback = {c: v for (c, s), v in self.tcp_one_way_us.items() if s == seam}
        return interp_1d(fallback, float(chunk_rows))

    def hop_steady_us(self, chunk_rows: int, seam: str, transport: str) -> float:
        if transport == "rdma":
            return self.hop_solo_us(chunk_rows, seam, "rdma")
        key = (chunk_rows, seam)
        base = self.tcp_pipelined_us.get(key, self.hop_solo_us(chunk_rows, seam, "tcp"))
        ratio = self.overlap_tcp_ratio.get(chunk_rows, 1.0)
        return base * ratio

    def choose_cut_family(self, chunk_rows: int, p0: int = 32768) -> dict[str, Any]:
        stages_a = [
            interp_1d(self.stage_us[("A", 4, chunk_rows, s)], float(p0))
            for s in range(4)
        ]
        stages_b = [
            interp_1d(self.stage_us[("B", 4, chunk_rows, s)], float(p0))
            for s in range(4)
        ]
        bot_a = max(stages_a)
        bot_b = max(stages_b)
        stage_gain_us = bot_b - bot_a
        hop_delta_us = max(
            0.0,
            self.hop_solo_us(chunk_rows, "mid_group", "tcp")
            - self.hop_solo_us(chunk_rows, "pair_cut", "tcp"),
        )
        winner = "A" if stage_gain_us > hop_delta_us else "B"
        return {
            "chunk_rows": chunk_rows,
            "p0": p0,
            "bottleneck_a_us": bot_a,
            "bottleneck_b_us": bot_b,
            "stage_gain_us": stage_gain_us,
            "hop_delta_us": hop_delta_us,
            "winner": winner,
        }

    def compute_r_bound(
        self,
        prompt_n: int,
        pp_width: int,
        chunk_rows: int,
        family: str,
        transport: str = "tcp",
    ) -> dict[str, float]:
        n_snap = prompt_n - 1
        chunks = plan_pp_chunks(n_snap, chunk_rows)
        m_chunks = len(chunks)
        sched = SCHEDULES[(family, pp_width)]
        seams = [s[3] for s in sched[:-1]]
        gpu_ratio = 1.0 if transport == "rdma" else self.overlap_gpu_ratio.get(chunk_rows, 1.0)

        step_times: list[float] = []
        step_nocomm_times: list[float] = []
        p0 = 0
        for lc in chunks:
            w = lc / float(chunk_rows)
            solos = [
                w * interp_1d(self.stage_us[(family, pp_width, chunk_rows, s)], float(p0))
                for s in range(pp_width)
            ]
            step_nocomm_times.append(max(solos))
            effs: list[float] = []
            for s in range(pp_width):
                wire_seam = seams[s] if s < pp_width - 1 else seams[-1]
                wire_us = w * self.hop_steady_us(chunk_rows, wire_seam, transport)
                effs.append(max(solos[s] * gpu_ratio, wire_us))
            step_times.append(max(effs))
            p0 += lc

        mean_stage_us = sum(step_times) / float(m_chunks)
        mean_stage_nocomm_us = sum(step_nocomm_times) / float(m_chunks)
        pipeline_us = (m_chunks + pp_width - 1) * mean_stage_us
        pipeline_nocomm_us = (m_chunks + pp_width - 1) * mean_stage_nocomm_us
        overlap_excess_us = max(0.0, pipeline_us - pipeline_nocomm_us)

        w0 = chunks[0] / float(chunk_rows)
        fill_hops_us = sum(w0 * self.hop_solo_us(chunk_rows, sm, transport) for sm in seams)

        if transport == "rdma" and self.rdma_one_way_us:
            drain_bytes = (
                2 * 1280 * (chunk_rows // 2)
                + 1280 * chunk_rows
                + 15 * WIN_RING_BYTES
                + CED_TAIL_ROWS * PAIR_SEAM_ROW_BYTES
            )
            drain_us = interp_1d(self.rdma_one_way_us, float(drain_bytes))
        elif chunk_rows in self.kv_drain_us:
            drain_us = self.kv_drain_us[chunk_rows]
        elif self.kv_full_us:
            drain_us = interp_1d(self.kv_full_us, float(prompt_n))
        else:
            drain_us = 0.0

        tail_p0 = max(0, n_snap - CED_TAIL_ROWS)
        tail_us = interp_1d(self.tail_us, float(tail_p0)) if self.tail_us else 0.0

        total_us = pipeline_us + fill_hops_us + drain_us + tail_us
        r_bound = (prompt_n * 1_000_000.0 / total_us) if total_us > 0 else 0.0
        exposed_comm_us = fill_hops_us + overlap_excess_us + drain_us
        exposed_comm_frac = (exposed_comm_us / total_us) if total_us > 0 else 0.0

        return {
            "prompt_tokens": float(prompt_n),
            "pp_width": float(pp_width),
            "chunk_rows": float(chunk_rows),
            "chunks": float(m_chunks),
            "mean_stage_us": mean_stage_us,
            "fill_hops_us": fill_hops_us,
            "overlap_excess_us": overlap_excess_us,
            "drain_us": drain_us,
            "tail_us": tail_us,
            "total_us": total_us,
            "exposed_comm_us": exposed_comm_us,
            "exposed_comm_frac": exposed_comm_frac,
            "r_bound_tok_s": r_bound,
        }


def find_10k_threshold(ds: Dataset, chunk_star: int, family_star: str) -> int | None:
    for n_tok in range(8192, 131072 + 1, 256):
        res = ds.compute_r_bound(n_tok, 4, chunk_star, family_star, "tcp")
        if res["r_bound_tok_s"] >= 10_000.0:
            return n_tok
    return None


def evaluate_gate(ds: Dataset) -> dict[str, Any]:
    reasons: list[str] = []

    if not ds.load_checks:
        reasons.append("missing_load_check")
    for lc in ds.load_checks:
        if not lc.get("cache_ram_enabled") or int(lc.get("cache_mem_bytes", 0)) < 7_000_000_000:
            reasons.append(
                f"load_check_cache_disabled(ram_enabled={lc.get('cache_ram_enabled')},mem_bytes={lc.get('cache_mem_bytes')})"
            )
        if lc.get("narrowed") or int(lc.get("prefill_chunk", 0)) < 8192:
            reasons.append(f"load_check_narrowed(prefill_chunk={lc.get('prefill_chunk')})")
    if ds.warm_cold_hits:
        reasons.append(f"warm_hit_in_cold_bench({ds.warm_cold_hits[0][0]}:{ds.warm_cold_hits[0][1]})")
    for req_n in (8192, 32768, 65536):
        if req_n not in ds.kv_checks:
            reasons.append(f"missing_kv_export_check({req_n})")
            continue
        chk = ds.kv_checks[req_n]
        want = expected_plan_export_bytes(req_n - 1)
        if not chk["match"] or int(chk["export_bytes"]) != want or int(chk["plan_bytes"]) != want:
            reasons.append(f"kv_export_mismatch({req_n})")

    available_widths = [
        c
        for c in CHUNK_WIDTHS
        if ("A", 4, c, 0) in ds.stage_us and ("B", 4, c, 0) in ds.stage_us
    ]
    if not available_widths:
        raise ValueError("no complete PP4 stage_timing measurements found in JSONL")

    family_choices = {c: ds.choose_cut_family(c) for c in available_widths}

    pp4_32k = {
        c: ds.compute_r_bound(32768, 4, c, family_choices[c]["winner"], "tcp")
        for c in available_widths
    }
    pp4_64k = {
        c: ds.compute_r_bound(65536, 4, c, family_choices[c]["winner"], "tcp")
        for c in available_widths
    }

    c_star_64k = max(available_widths, key=lambda c: pp4_64k[c]["r_bound_tok_s"])
    c_star_32k = max(available_widths, key=lambda c: pp4_32k[c]["r_bound_tok_s"])
    fam_star = family_choices[c_star_64k]["winner"]

    best_32k = pp4_32k[c_star_32k]
    best_64k = pp4_64k[c_star_64k]

    r1_32k = interp_1d(ds.single_box_tps, 32768.0)
    r1_64k = interp_1d(ds.single_box_tps, 65536.0)

    speedup_32k = best_32k["r_bound_tok_s"] / r1_32k if r1_32k > 0 else 0.0
    speedup_64k = best_64k["r_bound_tok_s"] / r1_64k if r1_64k > 0 else 0.0
    tp4_ratio_lo = best_64k["r_bound_tok_s"] / LOSSLESS_TP4_REF_TOK_S[1]
    tp4_ratio_hi = best_64k["r_bound_tok_s"] / LOSSLESS_TP4_REF_TOK_S[0]
    comm_frac_32k = best_32k["exposed_comm_frac"]

    if speedup_32k < 2.2:
        reasons.append(f"speedup_32k={speedup_32k:.2f}x<2.20x")
    if speedup_64k < 2.6:
        reasons.append(f"speedup_64k={speedup_64k:.2f}x<2.60x")
    if best_64k["r_bound_tok_s"] < 6500.0:
        reasons.append(f"r_bound_64k={best_64k['r_bound_tok_s']:.0f}<6500_tok_s")
    if comm_frac_32k >= 0.20:
        reasons.append(f"exposed_comm_32k={comm_frac_32k * 100.0:.1f}%>=20.0%")

    has_rdma = bool(ds.rdma_one_way_us)
    rdma_64k = (
        ds.compute_r_bound(65536, 4, c_star_64k, fam_star, "rdma")["r_bound_tok_s"]
        if has_rdma
        else None
    )
    rdma_gap_64k = (rdma_64k - best_64k["r_bound_tok_s"]) if rdma_64k is not None else None

    n_10k = find_10k_threshold(ds, c_star_32k, family_choices[c_star_32k]["winner"])

    table_rows: list[dict[str, Any]] = []
    for n_tok in PROMPT_LADDER:
        row: dict[str, Any] = {
            "prompt_tokens": n_tok,
            "r1_tok_s": interp_1d(ds.single_box_tps, float(n_tok)) if ds.single_box_tps else 0.0,
        }
        for pp_w in PP_WIDTHS:
            for c_rows in available_widths:
                fam = family_choices[c_rows]["winner"]
                if (fam, pp_w, c_rows, 0) not in ds.stage_us:
                    continue
                res = ds.compute_r_bound(n_tok, pp_w, c_rows, fam, "tcp")
                row[f"PP{pp_w}_C{c_rows}"] = round(res["r_bound_tok_s"], 1)
        if has_rdma:
            res_rdma = ds.compute_r_bound(n_tok, 4, c_star_64k, fam_star, "rdma")
            row[f"PP4_C{c_star_64k}_RDMA"] = round(res_rdma["r_bound_tok_s"], 1)
        table_rows.append(row)

    verdict = "GO" if not reasons else "NO-GO"
    ask_sam_below_10k = verdict == "GO" and n_10k is None

    return {
        "verdict": verdict,
        "fail_reasons": reasons,
        "cut_family_gate": family_choices,
        "selected_family": fam_star,
        "best_chunk_32k": c_star_32k,
        "best_chunk_64k": c_star_64k,
        "r1_32k_tok_s": round(r1_32k, 1),
        "r1_64k_tok_s": round(r1_64k, 1),
        "r_bound_32k_tok_s": round(best_32k["r_bound_tok_s"], 1),
        "r_bound_64k_tok_s": round(best_64k["r_bound_tok_s"], 1),
        "speedup_32k": round(speedup_32k, 3),
        "speedup_64k": round(speedup_64k, 3),
        "lossless_tp4_speedup_range": [round(tp4_ratio_lo, 3), round(tp4_ratio_hi, 3)],
        "exposed_comm_frac_32k": round(comm_frac_32k, 4),
        "n_10k_threshold": n_10k,
        "ask_sam_6_5k_9_5k": ask_sam_below_10k,
        "rdma_64k_tok_s": round(rdma_64k, 1) if rdma_64k is not None else None,
        "rdma_gap_64k_tok_s": round(rdma_gap_64k, 1) if rdma_gap_64k is not None else None,
        "tcp_iface": ds.tcp_iface,
        "tcp_mtu": ds.tcp_mtu,
        "bound_table": table_rows,
    }


def render_markdown(summary: dict[str, Any], ds: Dataset) -> str:
    lines: list[str] = []
    lines.append(f"# Plan 75 Step 1 Bound Summary — **{summary['verdict']}**")
    lines.append("")
    if summary["fail_reasons"]:
        lines.append(f"- Failure reasons: `{', '.join(summary['fail_reasons'])}`")
    lines.append(f"- Cut-family winner: **Family {summary['selected_family']}**")
    lines.append(
        f"- 32k PP4 (`C={summary['best_chunk_32k']}`): **{summary['r_bound_32k_tok_s']} tok/s** "
        f"(`{summary['speedup_32k']:.2f}x` vs `{summary['r1_32k_tok_s']}` single-box; "
        f"exposed TCP+KV `{summary['exposed_comm_frac_32k'] * 100.0:.1f}%`, bar `< 20.0%`)"
    )
    lines.append(
        f"- 64k PP4 (`C={summary['best_chunk_64k']}`): **{summary['r_bound_64k_tok_s']} tok/s** "
        f"(`{summary['speedup_64k']:.2f}x` vs `{summary['r1_64k_tok_s']}` single-box; "
        f"Lossless TP4 ratio `{summary['lossless_tp4_speedup_range'][0]:.2f}x..{summary['lossless_tp4_speedup_range'][1]:.2f}x`)"
    )
    if summary["n_10k_threshold"] is not None:
        lines.append(f"- 10k+ threshold $N_{{10\\text{{k}}}}$: **{summary['n_10k_threshold']} tokens**")
    else:
        lines.append(
            "- 10k+ threshold $N_{10\\text{k}}$: **unreachable <= 128k** "
            "(ask Sam before Step 2 if Go passed in 6.5k-9.5k)"
        )
    if summary["rdma_64k_tok_s"] is not None:
        lines.append(
            f"- RDMA ceiling at 64k: **{summary['rdma_64k_tok_s']} tok/s** "
            f"(gap `{summary['rdma_gap_64k_tok_s']:+.1f} tok/s`)"
        )
    lines.append("")

    if summary["bound_table"]:
        cols = list(summary["bound_table"][0].keys())
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "|".join(["---"] * len(cols)) + "|")
        for row in summary["bound_table"]:
            lines.append("| " + " | ".join(str(row.get(c, "-")) for c in cols) + " |")
        lines.append("")

    if ds.layer_diag:
        lines.append("## Diagnostic Per-Layer Breakdown (not summed into R_bound)")
        lines.append("")
        lines.append("| chunk_rows | p0 | layer | L20c | total_us | engram_pread_us | engram_eval_us | indexer_us |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for (c_rows, p0, layer, is_l20c), vals in sorted(ds.layer_diag.items()):
            if layer in (1, 2, 8, 14, 20) or is_l20c:
                lines.append(
                    f"| {c_rows} | {p0} | {layer} | {is_l20c} | "
                    f"{vals['total_us']:.0f} | {vals['engram_pread_us']:.0f} | "
                    f"{vals['engram_eval_us']:.0f} | {vals['indexer_us']:.0f} |"
                )
        lines.append("")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="JSONL file(s) or directory")
    parser.add_argument("--json-out", help="Write structured summary JSON")
    parser.add_argument("--md-out", help="Write markdown summary table")
    args = parser.parse_args(argv)

    try:
        records = load_jsonl_paths(args.inputs)
        ds = Dataset(records)
        summary = evaluate_gate(ds)
    except Exception as exc:
        print(f"SUMMARIZE-ERROR: {exc}", file=sys.stderr)
        return 2

    md = render_markdown(summary, ds)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, sort_keys=True)
            fh.write("\n")
    if args.md_out:
        with open(args.md_out, "w", encoding="utf-8") as fh:
            fh.write(md)

    print(md)
    if summary["verdict"] == "GO":
        return 0
    print("GATE-FAIL: " + "; ".join(summary["fail_reasons"]), file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
