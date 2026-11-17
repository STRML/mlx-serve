#!/usr/bin/env python3
"""Run on studio1: four disk-only Latch jobs, a quarter of each table per box.

Stage engram-scale, engram_scale.py and corpus.u32 under ~/bench/engram-scale
on every box first. Live model servers may remain loaded: bench RSS <4 GiB.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shlex
import subprocess
import time


HOME = Path.home()
BASE = HOME / "bench/engram-scale"
PACK = HOME / "models-iq/DeepSeek-V4.1-Flash-1017-MLX-Serve-iQ-MLX-3.0bpw"


def endpoint(box):
    if box == 1:
        return None
    return subprocess.check_output(["bash", "-c",
        ". ~/bench/cluster-addr.sh; tb_addr studio%d studio1" % box], text=True).strip()


def command(addr, argv):
    return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", addr, shlex.join(map(str, argv))] if addr else list(map(str, argv))


def arm(out, depth, readers, rep, chunk, borrowed):
    tag = "d%d-n%d-r%d-c%d" % (depth, readers, rep, chunk)
    jobs = []
    try:
        for rank in range(4):
            box = rank + 1
            addr = endpoint(box)
            remote = out / tag
            start_file = remote / "start"
            argv = ["/usr/sbin/taskpolicy", "-c", "utility",
                    "/usr/bin/python3", BASE / "engram_scale.py", "--pack", PACK,
                    "--binary", BASE / "engram-scale", "--tokens", BASE / "corpus.u32", "--depth", depth,
                    "--readers", readers, "--rank", rank, "--shards", 4,
                    "--out", remote, "--start-file", start_file]
            if chunk:
                argv += ["--chunk", chunk]
            if not borrowed:
                argv = [HOME / ".local/bin/latch", "schedule", "--name", "engram-scale-" + tag,
                        "--cpu", "8", "--memory-mib", "512", "--io", "--timeout", "1200", "--"] + argv
            log = (out / (tag + "-studio%d.log" % box)).open("w")
            proc = subprocess.Popen(command(addr, argv), stdout=log, stderr=subprocess.STDOUT)
            jobs.append(dict(box=box, addr=addr, root=remote, proc=proc, log=log))
        deadline = time.monotonic() + 1200
        while True:
            if any(j["proc"].poll() is not None for j in jobs):
                raise RuntimeError("a worker exited before the barrier; inspect arm logs")
            with ThreadPoolExecutor(max_workers=4) as pool:
                ready = list(pool.map(lambda j: subprocess.run(command(j["addr"], ["test", "-f", j["root"] / "ready"])).returncode == 0, jobs))
            if all(ready):
                break
            if time.monotonic() > deadline:
                raise TimeoutError("Latch admission")
            time.sleep(2)
        start = time.time() + 3
        def release(j):
            code = "from pathlib import Path; Path(%r).write_text(%r)" % (str(j["root"] / "start"), str(start))
            subprocess.run(command(j["addr"], ["/usr/bin/python3", "-c", code]), check=True)
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(release, jobs))
        codes = [j["proc"].wait() for j in jobs]
        if any(codes):
            raise RuntimeError("incomplete four-box arm: %r" % codes)
        meta = []
        for j in jobs:
            for name in ("metadata.json", "rows.jsonl", "stderr.log"):
                data = subprocess.check_output(command(j["addr"], ["cat", j["root"] / name]))
                (out / (tag + "-studio%d-" % j["box"] + name)).write_bytes(data)
                if name == "metadata.json":
                    meta.append(json.loads(data))
        (out / (tag + "-barrier.json")).write_text(json.dumps(dict(start=start, boxes=meta), indent=2) + "\n")
        print("FOUR-ARM-OK", tag, flush=True)
    finally:
        for j in jobs:
            if j["proc"].poll() is None:
                code = "from pathlib import Path; p=Path(%r); p.parent.mkdir(parents=True,exist_ok=True); p.write_text('abort')" % str(j["root"] / "start")
                subprocess.run(command(j["addr"], ["/usr/bin/python3", "-c", code]), check=False)
        for j in jobs:
            j["proc"].wait()
        for j in jobs:
            j["log"].close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--borrowed", action="store_true", help="use an enclosing four-box borrow's Latch holds")
    args = ap.parse_args()
    if args.borrowed and (not os.environ.get("BORROW") or
                         set(os.environ.get("BORROW_BOXES", "").split(",")) !=
                         {"studio1", "studio2", "studio3", "studio4"}):
        ap.error("--borrowed requires all four boxes in BORROW_BOXES")
    args.out = args.out.expanduser().resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    for rep in range(1, 4):
        for depth in (32768, 131072):
            for readers in ((16, 32) if rep % 2 else (32, 16)):
                arm(args.out, depth, readers, rep, 0, args.borrowed)
    for depth in (32768, 131072):
        arm(args.out, depth, 16, 1, 2048, args.borrowed)
    print("ENGRAM-SCALE-FOUR-OK", flush=True)


if __name__ == "__main__":
    main()
