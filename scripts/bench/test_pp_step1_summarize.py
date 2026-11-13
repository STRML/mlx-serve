#!/usr/bin/env python3
"""Unit tests for Plan 75 Step 1 formulas, JSONL schema, R_bound(N,P,C), and Go/No-Go gate."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import pp_step1_summarize as summ  # noqa: E402


def make_synthetic_records(
    *,
    r1: float = 3300.0,
    hop_gbps: float = 3.0,
    rdma_gbps: float = 9.28,
    gpu_overlap_ratio: float = 1.0,
    tcp_overlap_ratio: float = 1.0,
    drain_s: float = 0.05,
    tail_s: float = 0.15,
    narrowed: bool = False,
    cache_disabled: bool = False,
    warm_hit: bool = False,
    bad_kv_export: bool = False,
    family_b_faster: bool = False,
) -> list[dict]:
    recs: list[dict] = [
        {
            "kind": "load_check",
            "ctx_size": 262144,
            "prefix_cache_mem": "8GB",
            "prefill_chunk": 4096 if narrowed else 8192,
            "narrowed": narrowed,
            "cache_ram_enabled": not cache_disabled,
            "cache_mem_bytes": 0 if cache_disabled else 8_000_000_000,
        }
    ]

    for n_tok in (8192, 32768, 65536, 131072):
        recs.append(
            {
                "kind": "single_box_cold",
                "prompt_tokens": n_tok,
                "cached_tokens": 0,
                "rep": 0,
                "warmup": True,
                "elapsed_us": int(n_tok * 1_000_000.0 / (r1 * 0.8)),
                "tok_per_s": r1 * 0.8,
            }
        )
        for rep in (1, 2, 3):
            recs.append(
                {
                    "kind": "single_box_cold",
                    "prompt_tokens": n_tok,
                    "cached_tokens": 512 if (warm_hit and rep == 2) else 0,
                    "rep": rep,
                    "warmup": False,
                    "elapsed_us": int(n_tok * 1_000_000.0 / r1),
                    "tok_per_s": r1,
                }
            )

    for n_tok in (8192, 32768, 65536):
        n_snap = n_tok - 1
        want = summ.expected_plan_export_bytes(n_snap)
        got = want + (4096 if bad_kv_export else 0)
        recs.append(
            {
                "kind": "kv_export_check",
                "prompt_tokens": n_tok,
                "n_snap": n_snap,
                "tensor_count": 57,
                "export_bytes": got,
                "plan_bytes": want,
                "formula_bytes": want,
                "hist_local_bytes": n_snap * 4,
                "non_stage0_rows_bytes": summ.non_stage0_kv_rows_bytes(n_snap),
                "non_stage0_total_bytes": summ.non_stage0_kv_rows_bytes(n_snap) + 15 * summ.WIN_RING_BYTES + 8192,
                "match": not bad_kv_export,
            }
        )

    for (fam, pp_w), stages in summ.SCHEDULES.items():
        for c_rows in (2048, 4096, 8192):
            full_chunk_us = (c_rows / r1) * 1_000_000.0
            for p0 in (0, 32768, 131072):
                for s_idx, (lo, hi, l20c, seam) in enumerate(stages):
                    share = (hi - lo) / 20.0
                    if family_b_faster and fam == "A":
                        share *= 1.30
                    stage_us = full_chunk_us * share
                    recs.append(
                        {
                            "kind": "stage_timing",
                            "family": fam,
                            "pp_width": pp_w,
                            "schedule": f"{fam}{pp_w}",
                            "stage_idx": s_idx,
                            "lo": lo,
                            "hi": hi,
                            "l20_comp": l20c,
                            "seam": seam,
                            "chunk_rows": c_rows,
                            "p0": p0,
                            "rep": 0,
                            "warmup": True,
                            "elapsed_us": int(stage_us * 1.5),
                        }
                    )
                    recs.append(
                        {
                            "kind": "stage_timing",
                            "family": fam,
                            "pp_width": pp_w,
                            "schedule": f"{fam}{pp_w}",
                            "stage_idx": s_idx,
                            "lo": lo,
                            "hi": hi,
                            "l20_comp": l20c,
                            "seam": seam,
                            "chunk_rows": c_rows,
                            "p0": p0,
                            "rep": 1,
                            "warmup": False,
                            "elapsed_us": int(round(stage_us)),
                        }
                    )

    for p0 in (0, 32768, 131072):
        recs.append(
            {
                "kind": "tail_timing",
                "schedule": "A4",
                "lo": 20,
                "hi": 40,
                "rows": 128,
                "chunk_rows": 2048,
                "p0": p0,
                "rep": 1,
                "warmup": False,
                "elapsed_us": int(round(tail_s * 1_000_000.0)),
            }
        )
        recs.append(
            {
                "kind": "layer_diag",
                "layer": 1,
                "is_l20_comp": False,
                "chunk_rows": 2048,
                "p0": p0,
                "rep": 1,
                "warmup": False,
                "total_us": 999_999,
                "engram_pread_us": 4200,
                "engram_wait_us": 100,
                "engram_eval_us": 5100,
                "indexer_us": 0,
            }
        )

    for c_rows in (2048, 4096, 8192):
        for seam in ("pair_cut", "mid_group"):
            b = summ.seam_payload_bytes(c_rows, seam)
            hop_us = b / (hop_gbps * 1000.0)
            rdma_us = b / (rdma_gbps * 1000.0)
            for mode in ("one_way", "pipelined"):
                recs.append(
                    {
                        "kind": "tcp_hop",
                        "mode": mode,
                        "chunk_rows": c_rows,
                        "seam": seam,
                        "bytes": b,
                        "iters": 50,
                        "mean_us": hop_us,
                        "median_us": hop_us,
                        "best_us": hop_us * 0.98,
                        "p95_us": hop_us * 1.03,
                        "uma_memcpy_us": 500.0,
                        "uma_mlx_new_us": 600.0,
                        "gbps": hop_gbps,
                        "iface": "en2",
                        "local_ip": "10.254.12.2",
                        "peer_ip": "10.254.12.1",
                        "mtu": 9000,
                        "sndbuf": 4194304,
                        "rcvbuf": 4194304,
                    }
                )
            recs.append(
                {
                    "kind": "rdma_hop",
                    "bytes": b,
                    "iters": 50,
                    "one_way_mean_us": rdma_us,
                    "one_way_best_us": rdma_us * 0.98,
                    "gbps": rdma_gbps,
                }
            )

        b_mid = summ.seam_payload_bytes(c_rows, "mid_group")
        gpu_solo = (c_rows / r1) * 0.25 * 1_000_000.0
        tcp_solo = b_mid / (hop_gbps * 1000.0)
        gpu_ov = gpu_solo * gpu_overlap_ratio
        tcp_ov = tcp_solo * tcp_overlap_ratio
        recs.append(
            {
                "kind": "tcp_gpu_overlap",
                "chunk_rows": c_rows,
                "seam": "mid_group",
                "bytes": b_mid,
                "moe_lo": 5,
                "moe_hi": 10,
                "p0": 32768,
                "iters": 10,
                "gpu_solo_median_us": gpu_solo,
                "tcp_solo_median_us": tcp_solo,
                "gpu_overlap_median_us": gpu_ov,
                "tcp_overlap_median_us": tcp_ov,
                "step_wall_median_us": max(gpu_ov, tcp_ov),
                "gpu_slowdown_ratio": gpu_overlap_ratio,
                "tcp_slowdown_ratio": tcp_overlap_ratio,
            }
        )
        recs.append(
            {
                "kind": "kv_stream",
                "subtest": "final_chunk_drain",
                "prompt_tokens": 0,
                "chunk_rows": c_rows,
                "wire_bytes": 12_000_000,
                "rows_bytes": 5_000_000,
                "iters": 10,
                "tcp_median_us": drain_s * 800_000.0,
                "fill_rows_median_us": drain_s * 200_000.0,
                "total_median_us": drain_s * 1_000_000.0,
                "gbps": 3.0,
            }
        )

    return recs


class PpStep1SummarizeTests(unittest.TestCase):
    def test_exact_byte_formulas_and_chunks(self) -> None:
        self.assertEqual(summ.seam_payload_bytes(2048, "pair_cut"), 83_918_848)
        self.assertEqual(summ.seam_payload_bytes(2048, "mid_group"), 89_161_728)
        self.assertEqual(summ.seam_payload_bytes(4096, "pair_cut"), 167_837_696)
        self.assertEqual(summ.seam_payload_bytes(4096, "mid_group"), 178_323_456)
        self.assertEqual(summ.seam_payload_bytes(8192, "pair_cut"), 335_675_392)
        self.assertEqual(summ.seam_payload_bytes(8192, "mid_group"), 356_646_912)

        self.assertEqual(summ.expected_plan_export_bytes(8191), 32_625_660)
        self.assertEqual(summ.expected_plan_export_bytes(32767), 111_367_164)
        self.assertEqual(summ.expected_plan_export_bytes(65535), 216_355_836)
        self.assertEqual(summ.non_stage0_kv_rows_bytes(32767), 83_882_240)
        self.assertEqual(summ.non_stage0_kv_rows_bytes(65535), 167_768_320)

        self.assertEqual(summ.plan_pp_chunks(8292, 8192), [7808, 484])
        self.assertEqual(summ.plan_pp_chunks(32868, 8192), [8192, 8192, 8192, 7808, 484])

    def test_r_bound_matches_spec_section1_table_and_ignores_layer_diag(self) -> None:
        recs = make_synthetic_records(r1=3300.0, hop_gbps=2.972, drain_s=0.05, tail_s=0.15)
        ds = summ.Dataset(recs)
        res_32k = ds.compute_r_bound(32768, 4, 2048, "A", "tcp")
        res_64k = ds.compute_r_bound(65536, 4, 2048, "A", "tcp")
        self.assertAlmostEqual(res_32k["r_bound_tok_s"], 10120.0, delta=25.0)
        self.assertAlmostEqual(res_64k["r_bound_tok_s"], 11460.0, delta=25.0)

        summary = summ.evaluate_gate(ds)
        self.assertEqual(summary["verdict"], "GO")
        self.assertEqual(summary["selected_family"], "A")
        self.assertIsNotNone(summary["n_10k_threshold"])
        self.assertLessEqual(summary["n_10k_threshold"], 32768)
        self.assertFalse(summary["ask_sam_6_5k_9_5k"])

    def test_go_below_10k_flags_sam_check(self) -> None:
        recs = make_synthetic_records(r1=2400.0, hop_gbps=3.0, drain_s=0.05, tail_s=0.15)
        ds = summ.Dataset(recs)
        summary = summ.evaluate_gate(ds)
        self.assertEqual(summary["verdict"], "GO")
        self.assertGreater(summary["r_bound_64k_tok_s"], 6500.0)
        self.assertLess(summary["r_bound_64k_tok_s"], 10000.0)
        self.assertIsNone(summary["n_10k_threshold"])
        self.assertTrue(summary["ask_sam_6_5k_9_5k"])

    def test_cut_family_gate_switches_to_b_when_stage_gain_smaller_than_hop_delta(self) -> None:
        recs = make_synthetic_records(r1=3300.0, family_b_faster=True)
        ds = summ.Dataset(recs)
        choice = ds.choose_cut_family(2048)
        self.assertEqual(choice["winner"], "B")

    def test_no_go_conditions_and_shell_harness_verdict_lines(self) -> None:
        slow_recs = make_synthetic_records(r1=1500.0)
        slow_summary = summ.evaluate_gate(summ.Dataset(slow_recs))
        self.assertEqual(slow_summary["verdict"], "NO-GO")
        self.assertTrue(any("6500" in r for r in slow_summary["fail_reasons"]))

        comm_recs = make_synthetic_records(r1=3300.0, hop_gbps=0.15)
        comm_summary = summ.evaluate_gate(summ.Dataset(comm_recs))
        self.assertEqual(comm_summary["verdict"], "NO-GO")
        self.assertTrue(any("exposed_comm_32k" in r for r in comm_summary["fail_reasons"]))

        narrow_recs = make_synthetic_records(narrowed=True)
        narrow_summary = summ.evaluate_gate(summ.Dataset(narrow_recs))
        self.assertEqual(narrow_summary["verdict"], "NO-GO")
        self.assertTrue(any("load_check_narrowed" in r for r in narrow_summary["fail_reasons"]))

        kv_recs = make_synthetic_records(bad_kv_export=True)
        kv_summary = summ.evaluate_gate(summ.Dataset(kv_recs))
        self.assertEqual(kv_summary["verdict"], "NO-GO")
        self.assertTrue(any("kv_export_mismatch" in r for r in kv_summary["fail_reasons"]))

        cache_off_recs = make_synthetic_records(cache_disabled=True)
        cache_off_summary = summ.evaluate_gate(summ.Dataset(cache_off_recs))
        self.assertEqual(cache_off_summary["verdict"], "NO-GO")
        self.assertTrue(any("load_check_cache_disabled" in r for r in cache_off_summary["fail_reasons"]))

        warm_recs = make_synthetic_records(warm_hit=True)
        warm_summary = summ.evaluate_gate(summ.Dataset(warm_recs))
        self.assertEqual(warm_summary["verdict"], "NO-GO")
        self.assertTrue(any("warm_hit_in_cold_bench" in r for r in warm_summary["fail_reasons"]))

        sh_script = os.path.join(SCRIPT_DIR, "pp-step1-bound.sh")
        with tempfile.TemporaryDirectory() as tmp:
            ok_path = os.path.join(tmp, "measurements.jsonl")
            with open(ok_path, "w", encoding="utf-8") as fh:
                for row in make_synthetic_records(r1=3300.0):
                    fh.write(json.dumps(row) + "\n")
            proc_ok = subprocess.run(
                ["bash", sh_script, "--summarize-only", "--results-dir", tmp],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc_ok.returncode, 0, msg=proc_ok.stderr)
            self.assertEqual(proc_ok.stdout.strip().splitlines()[-1], "PP-STEP1-OK")

        with tempfile.TemporaryDirectory() as tmp:
            bad_path = os.path.join(tmp, "measurements.jsonl")
            with open(bad_path, "w", encoding="utf-8") as fh:
                for row in slow_recs:
                    fh.write(json.dumps(row) + "\n")
            proc_bad = subprocess.run(
                ["bash", sh_script, "--summarize-only", "--results-dir", tmp],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc_bad.returncode, 1)
            self.assertTrue(proc_bad.stdout.strip().splitlines()[-1].startswith("PP-STEP1-FAIL "))


if __name__ == "__main__":
    unittest.main()
