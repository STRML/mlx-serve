#!/usr/bin/env python3
"""Disk-safe incremental fetch+convert driver for DeepSeek-V4-Flash-0731.

The full source is ~167 GB and the box does not have room for it alongside
the mirror it produces, so nothing ever holds the whole checkpoint: for each
converter GROUP (layer.N / mtp / top) this downloads only the shards that
group reads, converts it, then deletes every shard no REMAINING group needs.
Peak source-on-disk is a handful of shards.

Resumable: the converter's own `.convert-manifest.json` skips finished
groups, and `hf download` skips files already present. Re-run the same
command after any interruption.

  python3 tests/fetch_convert_dsv4_0731.py \
      --src ~/.mlx-serve/staging/DeepSeek-V4-Flash-0731 \
      --out ~/.mlx-serve/models/ddalcu/DeepSeek-V4-Flash-0731-MLX-Serve-mixed-2-3-8bit
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time

REPO = "deepseek-ai/DeepSeek-V4-Flash-0731"
# Everything except the weight shards — small, and the converter needs the
# index + config + tokenizer to emit a loadable mirror.
META = [
    "config.json", "generation_config.json", "model.safetensors.index.json",
    "tokenizer.json", "tokenizer_config.json", "README.md",
    "encoding/encoding_dsv4.py", "encoding/test_encoding_dsv4.py",
    "inference/model.py", "inference/generate.py", "inference/convert.py",
    "inference/kernel.py", "inference/config.json", "inference/README.md",
]
MIN_FREE_GIB = 12.0  # abort rather than wedge the machine


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def free_gib(path):
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / (1 << 30)


def hf_download(src, patterns):
    """Fetch specific files into `src` (flat repo layout, resumable)."""
    cmd = ["hf", "download", REPO, *patterns, "--local-dir", src]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)


def group_of(name):
    m = re.match(r"^layers\.(\d+)\.", name)
    if m:
        return f"layer.{m.group(1)}"
    if name.startswith("mtp."):
        return "mtp"
    return "top"


def converter_cmd(converter, src, out, groups=None, imatrix=None):
    cmd = [sys.executable, converter, "--src", src, "--out", out]
    if imatrix is not None:
        cmd += ["--imatrix", imatrix]
    if groups is not None:
        cmd += ["--groups", groups]
    return cmd


def self_test():
    ok = True

    def check(cond, name):
        nonlocal ok
        print(("PASS" if cond else "FAIL") + f"  {name}")
        ok = ok and cond

    # Without --imatrix the driver must not inject the flag: the converter's
    # own default applies (and a converter-side default change is one place).
    c = converter_cmd("conv.py", "/s", "/o", groups="layer.3")
    check("--imatrix" not in c, "no --imatrix flag when none requested")
    check(c[-2:] == ["--groups", "layer.3"], "group pass carries --groups")
    # With one, BOTH the per-group and the finalize invocations must carry it —
    # the finalize pass re-runs the converter and would otherwise stamp the
    # mirror's `calibration` field from the wrong .dat.
    c = converter_cmd("conv.py", "/s", "/o", groups="mtp", imatrix="/im.dat")
    check(c[c.index("--imatrix") + 1] == "/im.dat", "group pass carries --imatrix")
    f = converter_cmd("conv.py", "/s", "/o", imatrix="/im.dat")
    check("--groups" not in f, "finalize pass has no --groups")
    check(f[f.index("--imatrix") + 1] == "/im.dat", "finalize pass carries --imatrix")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--converter", default=os.path.join(os.path.dirname(__file__), "convert_dsv4_weights.py"))
    ap.add_argument("--imatrix", default=None,
                    help="imatrix .dat passed through to the converter on every "
                         "invocation (default: the converter's own default)")
    args = ap.parse_args()
    src = os.path.expanduser(args.src)
    out = os.path.expanduser(args.out)
    os.makedirs(src, exist_ok=True)
    os.makedirs(out, exist_ok=True)

    log(f"metadata → {src}")
    hf_download(src, META)

    wm = json.load(open(os.path.join(src, "model.safetensors.index.json")))["weight_map"]
    cfg = json.load(open(os.path.join(src, "config.json")))
    n_layers = cfg["num_hidden_layers"]

    # Shards each group reads, in conversion order.
    order = [f"layer.{i}" for i in range(n_layers)] + ["mtp", "top"]
    need = {g: set() for g in order}
    for name, shard in wm.items():
        need[group_of(name)].add(shard)

    # Groups the converter has already finished. Checked HERE as well as
    # inside the converter: this loop downloads a group's shards BEFORE
    # invoking it, so without this a resume would re-fetch all ~167 GB only
    # for every group to skip.
    manifest_path = os.path.join(out, ".convert-manifest.json")
    done = set(json.load(open(manifest_path))) if os.path.exists(manifest_path) else set()

    for gi, g in enumerate(order):
        if g in done:
            continue
        # Only groups still to come may keep a shard alive.
        future = set().union(*(need[x] for x in order[gi + 1:] if x not in done)) if gi + 1 < len(order) else set()
        missing = sorted(s for s in need[g] if not os.path.exists(os.path.join(src, s)))
        if missing:
            log(f"{g}: fetching {len(missing)} shard(s): {', '.join(missing)}")
            hf_download(src, missing)
        fg = free_gib(out)
        if fg < MIN_FREE_GIB:
            log(f"ABORT: only {fg:.1f} GiB free (floor {MIN_FREE_GIB}). "
                f"Free space (e.g. remove the superseded preview mirror) and re-run — this is resumable.")
            return 1
        log(f"{g}: converting ({fg:.1f} GiB free)")
        r = subprocess.run(converter_cmd(args.converter, src, out, groups=g, imatrix=args.imatrix))
        if r.returncode != 0:
            log(f"ABORT: converter failed on {g}")
            return r.returncode
        # Drop every shard no later group reads.
        for s in sorted(need[g] - future):
            p = os.path.join(src, s)
            if os.path.exists(p):
                os.remove(p)
                log(f"{g}: dropped source shard {s}")

    # The per-group invocations above deliberately skip the index/config
    # writers (the converter only emits them on a full run), so a final
    # no-`--groups` pass is REQUIRED: every group short-circuits through the
    # manifest and it writes model.safetensors.index.json, the rebuilt
    # per-path `quantization` dict, and the tokenizer/template files. Without
    # it the mirror is a pile of shards no loader can open.
    log("finalizing: index + config + tokenizer")
    r = subprocess.run(converter_cmd(args.converter, src, out, imatrix=args.imatrix))
    if r.returncode != 0:
        log("ABORT: finalize pass failed")
        return r.returncode

    # The blobs hf keeps under .cache double every shard on disk; they are
    # useless once the shard files are gone.
    cache = os.path.join(src, ".cache")
    if os.path.isdir(cache):
        shutil.rmtree(cache, ignore_errors=True)
    log(f"DONE — mirror at {out} ({free_gib(out):.1f} GiB free)")
    return 0


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        sys.exit(self_test())
    sys.exit(main())
