#!/usr/bin/env python3
"""One cold disk-only arm. Run through borrow/Latch; never starts a server.

Build on the Mac or a borrowed Studio:
  zig build-exe src/engram_scale_bench.zig -lc -O ReleaseFast -femit-bin=engram-scale
Tokenize once with the pack tokenizer:
  engram-scale tokenize PACK CORPUS TOKENS.u32
Each arm clones only the two Engram files with APFS clonefile(), then requires
mincore=0 for every selected code/scale span. F_NOCACHE prevents repopulation.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import resource
import struct
import subprocess
import tempfile
import time


def cold_pack(pack, parent):
    config = json.loads((pack / "config.json").read_text())
    specs = config["omlx_deepseek_v41"]["engram_tables"]
    paths = sorted({s["weight_file"] for s in specs.values()})
    if len(paths) != 2:
        raise ValueError("expected two split MXFP8 table files")
    root = Path(tempfile.mkdtemp(prefix="cold-", dir=parent))
    try:
        for name in ("config.json", "engram-token-map.u32", "engram_token_map.json"):
            if (pack / name).exists():
                (root / name).symlink_to(pack / name)
        libc = ctypes.CDLL(None, use_errno=True)
        libc.clonefile.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
        libc.clonefile.restype = ctypes.c_int
        provenance = []
        for rel in paths:
            if Path(rel).is_absolute() or ".." in Path(rel).parts:
                raise ValueError("table path escapes pack")
            src, dst = pack / rel, root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if libc.clonefile(os.fsencode(src), os.fsencode(dst), 0):
                raise OSError(ctypes.get_errno(), "APFS clonefile", str(src))
            a, b = src.stat(), dst.stat()
            if a.st_ino == b.st_ino or a.st_size != b.st_size:
                raise ValueError("clone must have a distinct inode and identical size")
            provenance.append(dict(path=str(src), bytes=a.st_size,
                                   allocated_bytes=a.st_blocks * 512,
                                   source_inode=a.st_ino, clone_inode=b.st_ino))
        return root, provenance
    except BaseException:
        subprocess.run(["rm", "-rf", str(root)], check=True)
        raise


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pack", type=Path, required=True)
    ap.add_argument("--binary", type=Path, required=True)
    ap.add_argument("--tokens", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--depth", type=int, required=True, choices=(32768, 131072))
    ap.add_argument("--readers", type=int, required=True, choices=(1, 2, 4, 8, 16, 32))
    ap.add_argument("--shards", type=int, default=1, choices=(1, 4))
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--start-at", type=float, default=0, help="Common Unix timestamp after all four Latch jobs are admitted")
    ap.add_argument("--start-file", type=Path, help="Write ready after cloning; wait for a common start timestamp in this file")
    ap.add_argument("--warm-control", action="store_true", help="Populate selected pages before the binary's cold refusal check")
    args = ap.parse_args()
    if not 0 <= args.rank < args.shards:
        ap.error("rank outside shard count")
    if args.warm_control and args.shards != 1:
        ap.error("warm control requires the full row list")
    args.pack = args.pack.expanduser().resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    root, provenance = cold_pack(args.pack, args.out)
    try:
        meta = dict(host=os.uname().nodename, cold_method="APFS fresh vnode clone + F_NOCACHE + mincore-zero",
                    tables=provenance, tokens_sha256=hashlib.sha256(args.tokens.read_bytes()).hexdigest(),
                    binary_sha256=hashlib.sha256(args.binary.read_bytes()).hexdigest(),
                    config_sha256=hashlib.sha256((args.pack / "config.json").read_bytes()).hexdigest(),
                    depth=args.depth, readers=args.readers, rank=args.rank, shards=args.shards,
                    warm_control=args.warm_control,
                    chunk=args.chunk or args.depth)
        (args.out / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
        cmd = [str(args.binary.resolve()), str(root), str(args.tokens.resolve()), str(args.depth),
               str(args.readers), str(args.rank), str(args.shards),
               str(args.chunk or args.depth), "cold"]
        if args.warm_control:
            specs = json.loads((args.pack / "config.json").read_text())["omlx_deepseek_v41"]["engram_tables"]
            for spec in specs.values():
                with (root / spec["weight_file"]).open("rb") as f:
                    header_len, = struct.unpack("<Q", f.read(8))
                    if header_len > 16 << 20:
                        raise ValueError("oversized table header")
                    header = json.loads(f.read(header_len))
                    for key in (spec["weight_key"], spec["scale_key"]):
                        lo, hi = header[key]["data_offsets"]
                        f.seek(8 + header_len + lo)
                        f.read(min(64 << 20, hi - lo))
        (args.out / "ready").touch()
        deadline = time.monotonic() + 600
        while args.start_file and not args.start_file.exists():
            if time.monotonic() >= deadline:
                raise TimeoutError("four-box start barrier")
            time.sleep(.2)
        if args.start_file:
            args.start_at = float(args.start_file.read_text())
        while time.time() < args.start_at:
            time.sleep(max(0, min(.2, args.start_at - time.time())))
        start = time.time()
        with (args.out / "rows.jsonl").open("w") as out, (args.out / "stderr.log").open("w") as err:
            proc = subprocess.run(cmd, stdout=out, stderr=err)
        meta.update(start_epoch=start, end_epoch=time.time(), exit_code=proc.returncode,
                    peak_rss_bytes=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
                    parent_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        meta["combined_peak_upper_bytes"] = meta["peak_rss_bytes"] + meta["parent_peak_rss_bytes"]
        (args.out / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
        if proc.returncode:
            raise SystemExit(proc.returncode)
        records = [json.loads(line) for line in (args.out / "rows.jsonl").read_text().splitlines()]
        if not records or any(r["cached_parts"] for r in records):
            raise RuntimeError("missing results or cached table pages")
        if meta["combined_peak_upper_bytes"] >= 4 << 30:
            raise RuntimeError("4 GiB process budget exceeded")
        print(json.dumps(dict(out=str(args.out), rows=sum(r["rows"] for r in records),
                              read_ms=sum(r["read_ms"] for r in records),
                              peak_rss_bytes=meta["peak_rss_bytes"])), flush=True)
    finally:
        subprocess.run(["rm", "-rf", str(root)], check=True)


if __name__ == "__main__":
    main()
