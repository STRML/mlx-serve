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
    uma_memcpy_us: float = 500.0,
    uma_mlx_new_us: float = 600.0,
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
                        "uma_memcpy_us": uma_memcpy_us,
                        "uma_mlx_new_us": uma_mlx_new_us,
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
        recs = make_synthetic_records(
            r1=3300.0,
            hop_gbps=2.972,
            drain_s=0.05,
            tail_s=0.15,
            uma_memcpy_us=0.0,
            uma_mlx_new_us=0.0,
        )
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


    def test_stage_only_stages_all_rpath_dylibs_from_relocated_script_and_fails_on_missing_lib(self) -> None:
        sh_src = os.path.join(SCRIPT_DIR, "pp-step1-bound.sh")
        required = ("libmlx.dylib", "libmlxc.dylib", "libjaccl.dylib", "libllama.dylib", "mlx.metallib")
        with tempfile.TemporaryDirectory() as tmp:
            relocated_dir = os.path.join(tmp, "standalone-bench")
            os.makedirs(relocated_dir, exist_ok=True)
            relocated_sh = os.path.join(relocated_dir, "pp-step1-bound.sh")
            with open(sh_src, "r", encoding="utf-8") as f_in, open(relocated_sh, "w", encoding="utf-8") as f_out:
                f_out.write(f_in.read())
            os.chmod(relocated_sh, 0o755)

            fake_repo = os.path.join(tmp, "fake-repo")
            bin_dir = os.path.join(fake_repo, "zig-out", "bin")
            mlx_lib_dir = os.path.join(fake_repo, "lib", "mlx", "lib")
            llama_lib_dir = os.path.join(fake_repo, "lib", "llama", "lib")
            os.makedirs(bin_dir, exist_ok=True)
            os.makedirs(mlx_lib_dir, exist_ok=True)
            os.makedirs(llama_lib_dir, exist_ok=True)

            fake_bin = os.path.join(bin_dir, "mlx-serve")
            with open(fake_bin, "w", encoding="utf-8") as fh:
                fh.write("#!/bin/sh\nexit 0\n")
            os.chmod(fake_bin, 0o755)

            for name in ("libmlx.dylib", "libmlxc.dylib", "libjaccl.dylib", "mlx.metallib"):
                with open(os.path.join(mlx_lib_dir, name), "w", encoding="utf-8") as fh:
                    fh.write(f"payload:{name}")
            with open(os.path.join(llama_lib_dir, "libllama.dylib"), "w", encoding="utf-8") as fh:
                fh.write("payload:libllama.dylib")

            stage_dir = os.path.join(tmp, "remote-stage")
            res_dir = os.path.join(tmp, "results")
            env = dict(os.environ)
            env.update(
                {
                    "STUDIO1_HOST": "local",
                    "STUDIO2_HOST": "local",
                    "MLX_SERVE_LOCAL_BIN": fake_bin,
                    "REMOTE_STAGE_DIR": stage_dir,
                    "PP_STEP1_RESULTS_DIR": res_dir,
                }
            )
            proc = subprocess.run(
                ["bash", relocated_sh, "--stage-only"],
                cwd=relocated_dir,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)
            self.assertEqual(proc.stdout.strip().splitlines()[-1], "PP-STEP1-STAGE-OK")
            self.assertTrue(os.access(os.path.join(stage_dir, "mlx-serve"), os.X_OK))
            for name in required:
                staged_file = os.path.join(stage_dir, "lib", name)
                self.assertTrue(os.path.isfile(staged_file), msg=f"missing staged {name}")
                with open(staged_file, "r", encoding="utf-8") as fh:
                    self.assertEqual(fh.read(), f"payload:{name}")

            for missing_name, missing_dir in (
                ("libmlxc.dylib", mlx_lib_dir),
                ("libjaccl.dylib", mlx_lib_dir),
                ("libllama.dylib", llama_lib_dir),
            ):
                target_path = os.path.join(missing_dir, missing_name)
                saved_path = target_path + ".bak"
                os.rename(target_path, saved_path)
                try:
                    proc_miss = subprocess.run(
                        ["bash", relocated_sh, "--stage-only"],
                        cwd=relocated_dir,
                        env=env,
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    self.assertEqual(proc_miss.returncode, 1)
                    self.assertEqual(
                        proc_miss.stdout.strip().splitlines()[-1],
                        f"PP-STEP1-FAIL missing_runtime_lib:{missing_name}",
                    )
                finally:
                    os.rename(saved_path, target_path)

    def test_missing_required_records_rejected_without_zero_fallbacks_or_clamping(self) -> None:
        base = make_synthetic_records(r1=3300.0)
        drop_cases = [
            ("missing_tail", lambda r: r["kind"] != "tail_timing"),
            ("missing_kv_stream", lambda r: r["kind"] != "kv_stream"),
            ("missing_overlap", lambda r: r["kind"] != "tcp_gpu_overlap"),
            ("missing_pipelined_hop", lambda r: not (r["kind"] == "tcp_hop" and r["mode"] == "pipelined")),
            (
                "missing_cold_32k_64k",
                lambda r: not (r["kind"] == "single_box_cold" and r["prompt_tokens"] in (32768, 65536)),
            ),
            (
                "missing_stage_nonzero_idx",
                lambda r: not (r["kind"] == "stage_timing" and r["stage_idx"] == 2),
            ),
            (
                "missing_stage_high_p0",
                lambda r: not (r["kind"] == "stage_timing" and r["p0"] in (32768, 131072)),
            ),
        ]
        sh_script = os.path.join(SCRIPT_DIR, "pp-step1-bound.sh")
        for label, pred in drop_cases:
            partial = [r for r in base if pred(r)]
            ds = summ.Dataset(partial)
            with self.assertRaises(ValueError, msg=label):
                summ.evaluate_gate(ds)
            if label in ("missing_tail", "missing_kv_stream", "missing_overlap", "missing_pipelined_hop", "missing_stage_nonzero_idx", "missing_stage_high_p0"):
                with self.assertRaises(ValueError, msg=f"compute_r_bound:{label}"):
                    ds.compute_r_bound(32768, 4, 2048, "A", "tcp")

        with tempfile.TemporaryDirectory() as tmp:
            partial_path = os.path.join(tmp, "measurements.jsonl")
            with open(partial_path, "w", encoding="utf-8") as fh:
                for row in [r for r in base if r["kind"] != "kv_stream"]:
                    fh.write(json.dumps(row) + "\n")
            proc = subprocess.run(
                ["bash", sh_script, "--summarize-only", "--results-dir", tmp],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 1)
            self.assertTrue(proc.stdout.strip().splitlines()[-1].startswith("PP-STEP1-FAIL "))

    def test_host_staging_copies_charged_in_r_bound_and_exposed_comm(self) -> None:
        ds_zero = summ.Dataset(make_synthetic_records(r1=3300.0, uma_memcpy_us=0.0, uma_mlx_new_us=0.0))
        ds_staged = summ.Dataset(make_synthetic_records(r1=3300.0, uma_memcpy_us=12_000.0, uma_mlx_new_us=8_000.0))

        res_zero = ds_zero.compute_r_bound(32768, 4, 2048, "A", "tcp")
        res_staged = ds_staged.compute_r_bound(32768, 4, 2048, "A", "tcp")

        self.assertGreater(res_staged["mean_stage_us"], res_zero["mean_stage_us"] + 19_000.0)
        self.assertGreater(res_staged["overlap_excess_us"], 350_000.0)
        self.assertGreater(res_staged["exposed_comm_frac"], res_zero["exposed_comm_frac"])
        self.assertLess(res_staged["r_bound_tok_s"], res_zero["r_bound_tok_s"])

        heavy_staging = summ.Dataset(make_synthetic_records(r1=3300.0, uma_memcpy_us=30_000.0, uma_mlx_new_us=20_000.0))
        heavy_summary = summ.evaluate_gate(heavy_staging)
        self.assertEqual(heavy_summary["verdict"], "NO-GO")
        self.assertTrue(any("exposed_comm_32k" in r for r in heavy_summary["fail_reasons"]))

    def test_find_10k_threshold_sweeps_max_across_chunk_widths(self) -> None:
        recs = make_synthetic_records(r1=2450.0, hop_gbps=3.2, uma_memcpy_us=0.0, uma_mlx_new_us=0.0)
        adjusted: list[dict] = []
        for r in recs:
            if r["kind"] == "stage_timing":
                r = dict(r)
                if r["chunk_rows"] == 4096:
                    r["elapsed_us"] = int(round(r["elapsed_us"] * 1.05))
                elif r["chunk_rows"] == 8192:
                    r["elapsed_us"] = int(round(r["elapsed_us"] * 0.65))
            adjusted.append(r)

        ds = summ.Dataset(adjusted)
        c2k_128k = ds.compute_r_bound(131072, 4, 2048, "A", "tcp")["r_bound_tok_s"]
        self.assertLess(c2k_128k, 10_000.0)

        summary = summ.evaluate_gate(ds)
        self.assertEqual(summary["verdict"], "GO")
        self.assertEqual(summary["best_chunk_32k"], 2048)
        self.assertEqual(summary["best_chunk_64k"], 8192)
        self.assertGreaterEqual(summary["r_bound_64k_tok_s"], 10_000.0)
        self.assertIsNotNone(summary["n_10k_threshold"])
        self.assertLessEqual(summary["n_10k_threshold"], 65536)
        self.assertFalse(summary["ask_sam_6_5k_9_5k"])
        md = summ.render_markdown(summary, ds)
        self.assertNotIn("unreachable <= 128k", md)

    def test_make_pp_step1b_tokens_validates_vocab_and_rejects_byte_level_or_oov_output(self) -> None:
        import contextlib
        import io as pyio
        import make_pp_step1b_tokens as mktok

        with tempfile.TemporaryDirectory() as tmp:
            docs_dir = os.path.join(tmp, "docs")
            src_dir = os.path.join(tmp, "src")
            os.makedirs(docs_dir)
            os.makedirs(src_dir)
            for idx in range(6):
                with open(os.path.join(docs_dir, f"doc_{idx}.md"), "w", encoding="utf-8") as fh:
                    fh.write((f"# Architecture Note {idx}\nPipeline parallel prefill hides SSD reads.\n") * 60)
                with open(os.path.join(src_dir, f"mod_{idx}.zig"), "w", encoding="utf-8") as fh:
                    fh.write((f"pub fn step{idx}(x: u32) u32 {{ return x + {idx}; }}\n") * 60)

            with open(os.path.join(tmp, "config.json"), "w", encoding="utf-8") as fh:
                json.dump({"model_type": "deepseek_v41", "text_config": {"vocab_size": 129280}}, fh)

            corpus = mktok.collect_mixed_corpus(tmp, target_bytes=4096)
            self.assertGreaterEqual(len(corpus.encode("utf-8")), 4096)
            self.assertIn("# Architecture Note 0", corpus)
            self.assertIn("pub fn step0", corpus)

            fake_stage = os.path.join(tmp, "stage")
            os.makedirs(os.path.join(fake_stage, "lib"))
            fake_bin = os.path.join(fake_stage, "mlx-serve")
            built_env = mktok.build_tokenize_env(fake_bin, {})
            self.assertTrue(built_env["DYLD_LIBRARY_PATH"].endswith("/stage/lib"))

            def write_fake_bin(mode_expr: str) -> None:
                with open(fake_bin, "w", encoding="utf-8") as fh:
                    fh.write(
                        "#!/usr/bin/env python3\n"
                        "import struct, sys\n"
                        "assert sys.argv[1:3] == ['pp-probe', 'tokenize-ids']\n"
                        "args = sys.argv[3:]\n"
                        "kv = {args[i]: args[i + 1] for i in range(0, len(args), 2)}\n"
                        "data = open(kv['--in'], 'rb').read()\n"
                        f"ids = {mode_expr}\n"
                        "with open(kv['--out'], 'wb') as out:\n"
                        "    out.write(struct.pack(f'<{len(ids)}I', *ids))\n"
                    )
                os.chmod(fake_bin, 0o755)

            write_fake_bin("[int.from_bytes(data[i:i+4], 'little') % 129280 for i in range(0, len(data), 4)]")
            out_u32 = os.path.join(tmp, "tokens.u32")
            out_txt = os.path.join(tmp, "tokens.txt")
            rc = mktok.main([
                "--repo-root", tmp,
                "--model", tmp,
                "--mlx-serve", fake_bin,
                "--out", out_u32,
                "--text-out", out_txt,
                "--target-kb", "12",
                "--min-tokens", "2000",
            ])
            self.assertEqual(rc, 0)
            with open(out_u32, "rb") as fh:
                raw = fh.read()
            self.assertEqual(len(raw) % 4, 0)
            self.assertGreaterEqual(len(raw) // 4, 2000)
            self.assertTrue(os.path.isfile(out_txt))

            write_fake_bin("[b for b in data]")
            err_byte = pyio.StringIO()
            with contextlib.redirect_stderr(err_byte):
                rc_byte = mktok.main([
                    "--repo-root", tmp,
                    "--model", tmp,
                    "--mlx-serve", fake_bin,
                    "--out", out_u32,
                    "--target-kb", "12",
                    "--min-tokens", "2000",
                ])
            self.assertNotEqual(rc_byte, 0)
            self.assertIn("0.6", err_byte.getvalue())

            write_fake_bin(
                "[int.from_bytes(data[i:i+4], 'little') % 129280 for i in range(0, len(data) - 8, 4)] + [129280]"
            )
            err_oov = pyio.StringIO()
            with contextlib.redirect_stderr(err_oov):
                rc_oov = mktok.main([
                    "--repo-root", tmp,
                    "--model", tmp,
                    "--mlx-serve", fake_bin,
                    "--out", out_u32,
                    "--target-kb", "12",
                    "--min-tokens", "2000",
                ])
            self.assertNotEqual(rc_oov, 0)
            self.assertIn("129280", err_oov.getvalue())

            bad_model_dir = os.path.join(tmp, "bad-pack")
            os.makedirs(bad_model_dir)
            with open(os.path.join(bad_model_dir, "config.json"), "w", encoding="utf-8") as fh:
                json.dump({"model_type": "deepseek_v41", "text_config": {"vocab_size": 32000}}, fh)
            write_fake_bin("[int.from_bytes(data[i:i+4], 'little') % 32000 for i in range(0, len(data), 4)]")
            err_cfg = pyio.StringIO()
            with contextlib.redirect_stderr(err_cfg):
                rc_cfg = mktok.main([
                    "--repo-root", tmp,
                    "--model", bad_model_dir,
                    "--mlx-serve", fake_bin,
                    "--out", out_u32,
                    "--target-kb", "12",
                    "--min-tokens", "2000",
                ])
            self.assertNotEqual(rc_cfg, 0)
            self.assertIn("129280", err_cfg.getvalue())
if __name__ == "__main__":
    unittest.main()
