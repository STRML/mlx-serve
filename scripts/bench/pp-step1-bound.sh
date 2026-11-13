#!/usr/bin/env bash
# Plan 75 Step 1 bound-measurement harness.
# Runs inside an existing `borrow A` (studio1 + studio2 on 10.254.12.1 <-> 10.254.12.2).
# Never calls `borrow` or `latch`. Writes all JSONL measurements into one results directory,
# runs pp_step1_summarize.py, and prints PP-STEP1-OK or PP-STEP1-FAIL <why> last.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SUMMARIZER="${PP_STEP1_SUMMARIZER:-$SCRIPT_DIR/pp_step1_summarize.py}"

STUDIO1_HOST="${STUDIO1_HOST:-studio1}"
STUDIO2_HOST="${STUDIO2_HOST:-studio2}"
S1_TB_IP="${S1_TB_IP:-10.254.12.1}"
S2_TB_IP="${S2_TB_IP:-10.254.12.2}"
PROBE_PORT="${PROBE_PORT:-29751}"
RDMA_PORT="${RDMA_PORT:-29753}"
SERVE_PORT="${SERVE_PORT:-18751}"

MODEL_DIR_REMOTE="${MODEL_DIR_REMOTE:-\$HOME/models-iq/DeepSeek-V4.1-Flash-1017-MLX-Serve-iQ-MLX-3.0bpw}"
CTX_SIZE="${CTX_SIZE:-262144}"
PREFIX_CACHE_MEM="${PREFIX_CACHE_MEM:-8GB}"
COLD_REPS="${COLD_REPS:-3}"

MLX_SERVE_LOCAL_BIN="${MLX_SERVE_LOCAL_BIN:-$REPO_ROOT/zig-out/bin/mlx-serve}"
REMOTE_STAGE_DIR="${REMOTE_STAGE_DIR:-/tmp/pp75-step1-bin}"
RESULTS_DIR="${PP_STEP1_RESULTS_DIR:-$HOME/claude-tmp/pp-step1-$(date +%Y%m%d-%H%M%S)}"
SUMMARIZE_ONLY=0

while [ $# -gt 0 ]; do
    case "$1" in
        --results-dir)
            RESULTS_DIR="$2"
            shift 2
            ;;
        --summarize-only)
            SUMMARIZE_ONLY=1
            shift
            ;;
        *)
            RESULTS_DIR="$1"
            shift
            ;;
    esac
done

fail_step1() {
    local why="$1"
    echo "PP-STEP1-FAIL $why"
    exit 1
}

mkdir -p "$RESULTS_DIR" || fail_step1 "mkdir_results_dir_failed"

if [ "$SUMMARIZE_ONLY" = "1" ]; then
    ERR_FILE="$RESULTS_DIR/summarize.err"
    SUM_IN="$RESULTS_DIR"
    if [ -f "$RESULTS_DIR/measurements.jsonl" ]; then
        SUM_IN="$RESULTS_DIR/measurements.jsonl"
    fi
    if python3 "$SUMMARIZER" "$SUM_IN" \
        --json-out "$RESULTS_DIR/summary.json" \
        --md-out "$RESULTS_DIR/summary.md" 2>"$ERR_FILE"; then
        echo "PP-STEP1-OK"
        exit 0
    fi
    WHY="$(tr '\n' ' ' < "$ERR_FILE" | sed 's/[[:space:]]\+/ /g; s/^ //; s/ $//')"
    fail_step1 "${WHY:-gate_failed}"
fi

LOCAL_SHORT="$(hostname -s 2>/dev/null || hostname)"

run_on() {
    local target="$1"
    shift
    if [ "$LOCAL_SHORT" = "$target" ] || [ "$target" = "local" ]; then
        bash -c "$*"
    else
        ssh -o BatchMode=yes -o ConnectTimeout=10 "$target" "$*"
    fi
}

copy_to() {
    local src="$1"
    local target="$2"
    local dst="$3"
    if [ "$LOCAL_SHORT" = "$target" ] || [ "$target" = "local" ]; then
        cp -f "$src" "$dst"
    else
        scp -q -o BatchMode=yes "$src" "${target}:${dst}"
    fi
}

copy_from() {
    local target="$1"
    local src="$2"
    local dst="$3"
    if [ "$LOCAL_SHORT" = "$target" ] || [ "$target" = "local" ]; then
        cp -f "$src" "$dst"
    else
        scp -q -o BatchMode=yes "${target}:${src}" "$dst"
    fi
}

cleanup() {
    run_on "$STUDIO1_HOST" "pkill -f '$REMOTE_STAGE_DIR/mlx-serve pp-probe recv' 2>/dev/null || true; pkill -f '$REMOTE_STAGE_DIR/mlx-serve dist-probe' 2>/dev/null || true" >/dev/null 2>&1 || true
    run_on "$STUDIO2_HOST" "pkill -f '$REMOTE_STAGE_DIR/mlx-serve.*--port $SERVE_PORT' 2>/dev/null || true; pkill -f '$REMOTE_STAGE_DIR/mlx-serve pp-probe' 2>/dev/null || true; pkill -f '$REMOTE_STAGE_DIR/mlx-serve dist-probe' 2>/dev/null || true" >/dev/null 2>&1 || true
}
trap cleanup EXIT

if [ ! -x "$MLX_SERVE_LOCAL_BIN" ]; then
    fail_step1 "missing_local_binary:$MLX_SERVE_LOCAL_BIN"
fi

for box in "$STUDIO1_HOST" "$STUDIO2_HOST"; do
    run_on "$box" "mkdir -p '$REMOTE_STAGE_DIR/lib'" || fail_step1 "stage_mkdir_failed:$box"
    copy_to "$MLX_SERVE_LOCAL_BIN" "$box" "$REMOTE_STAGE_DIR/mlx-serve" || fail_step1 "stage_bin_failed:$box"
    if [ -f "$REPO_ROOT/lib/mlx/lib/libmlx.dylib" ]; then
        copy_to "$REPO_ROOT/lib/mlx/lib/libmlx.dylib" "$box" "$REMOTE_STAGE_DIR/lib/libmlx.dylib" || fail_step1 "stage_dylib_failed:$box"
    fi
    if [ -f "$REPO_ROOT/lib/mlx/lib/mlx.metallib" ]; then
        copy_to "$REPO_ROOT/lib/mlx/lib/mlx.metallib" "$box" "$REMOTE_STAGE_DIR/lib/mlx.metallib" || fail_step1 "stage_metallib_failed:$box"
    fi
    run_on "$box" "chmod +x '$REMOTE_STAGE_DIR/mlx-serve'" || fail_step1 "stage_chmod_failed:$box"
done

# ── 1a (Part 1): Server boot check (--ctx-size 262144 --prefix-cache-mem 8GB) + whole-prompt cold tok/s on studio2 ──

S2_SERVE_LOG="$REMOTE_STAGE_DIR/serve-1a.log"
S2_COLD_JSONL="$REMOTE_STAGE_DIR/1a-cold.jsonl"
S2_COLD_DRIVER="$REMOTE_STAGE_DIR/drive_cold_1a.py"

cat > "$RESULTS_DIR/drive_cold_1a.py" <<'PY'
import json
import sys
import time
import urllib.request

port = int(sys.argv[1])
log_path = sys.argv[2]
out_path = sys.argv[3]
ctx_size = int(sys.argv[4])
cache_mem = sys.argv[5]
reps = int(sys.argv[6])
base = f"http://127.0.0.1:{port}"

deadline = time.time() + 300.0
ready = False
while time.time() < deadline:
    try:
        with urllib.request.urlopen(f"{base}/v1/models", timeout=5) as r:
            if r.status == 200:
                ready = True
                break
    except Exception:
        pass
    time.sleep(1.0)

if not ready:
    raise SystemExit("server_boot_timeout")
with urllib.request.urlopen(f"{base}/props", timeout=10) as r:
    props = json.loads(r.read().decode("utf-8"))
settings = props.get("settings", {})
prefill_chunk = int(settings.get("prefill_chunk", 8192))
pcache = settings.get("prefix_cache", {})
cache_ram_enabled = bool(pcache.get("ram_enabled", False))
cache_mem_bytes = int(pcache.get("mem_bytes", 0))

log_text = open(log_path, "r", encoding="utf-8", errors="replace").read()
narrowed = ("Prefill chunk:" in log_text and "memory-sized down" in log_text) or prefill_chunk < 8192

with open(out_path, "w", encoding="utf-8") as out:
    out.write(json.dumps({
        "kind": "load_check",
        "ctx_size": ctx_size,
        "prefix_cache_mem": cache_mem,
        "prefill_chunk": prefill_chunk,
        "narrowed": narrowed,
        "cache_ram_enabled": cache_ram_enabled,
        "cache_mem_bytes": cache_mem_bytes,
    }) + "\n")
    out.flush()
    if not cache_ram_enabled or cache_mem_bytes < 7_000_000_000:
        raise SystemExit(f"prefix_cache_disabled_or_clamped:{cache_ram_enabled}:{cache_mem_bytes}")
    if narrowed:
        raise SystemExit(f"prefill_chunk_narrowed:{prefill_chunk}")
    with urllib.request.urlopen(f"{base}/v1/models", timeout=10) as r:
        model_id = json.loads(r.read().decode("utf-8"))["data"][0]["id"]

    def count_tokens(txt: str) -> int:
        treq = urllib.request.Request(
            f"{base}/tokenize",
            data=json.dumps({"model": model_id, "content": txt}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(treq, timeout=30) as tr:
            return len(json.loads(tr.read().decode("utf-8")).get("tokens", []))

    vocab_words = [
        "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel",
        "india", "juliet", "kilo", "lima", "mike", "november", "oscar", "papa",
    ]
    sample_words = [vocab_words[i % len(vocab_words)] for i in range(1024)]
    toks_per_word = max(1.0, count_tokens(" ".join(sample_words)) / 1024.0)

    for target_n in (8192, 32768, 65536, 131072):
        for rep in range(1 + reps):
            est_words = max(64, int((target_n - 16) / toks_per_word))
            salt = f"coldnonce_{target_n}_{rep}_{time.time_ns()}"
            words = [salt] + [vocab_words[(i * 7 + rep * 3) % len(vocab_words)] for i in range(est_words)]
            got_n = count_tokens(" ".join(words))
            if abs(got_n - (target_n - 16)) > 64 and got_n > 0:
                est_words = max(64, int(est_words * float(target_n - 16) / float(got_n)))
                words = [salt] + [vocab_words[(i * 7 + rep * 3) % len(vocab_words)] for i in range(est_words)]
            final_prompt = " ".join(words)
            body = json.dumps({
                "model": model_id,
                "messages": [
                    {"role": "system", "content": salt},
                    {"role": "user", "content": final_prompt},
                ],
                "max_tokens": 1,
                "temperature": 0.0,
                "stream": False,
                "enable_drafter": False,
            }).encode("utf-8")
            req = urllib.request.Request(
                f"{base}/v1/chat/completions",
                data=body,
                headers={"Content-Type": "application/json"},
            )
            t0 = time.monotonic()
            with urllib.request.urlopen(req, timeout=600) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            wall_us = int((time.monotonic() - t0) * 1_000_000)
            timings = payload.get("timings", {})
            usage = payload.get("usage", {})
            p_toks = int(timings.get("prompt_n") or usage.get("prompt_tokens") or target_n)
            cached_n = int(timings.get("cached_n") or 0)
            p_ms = float(timings.get("prompt_ms") or (wall_us / 1000.0))
            tps = float(timings.get("prompt_per_second") or (p_toks * 1000.0 / max(p_ms, 0.001)))
            if cached_n != 0:
                raise SystemExit(f"unexpected_warm_prefix_hit:{target_n}:rep{rep}:cached={cached_n}")
            out.write(json.dumps({
                "kind": "single_box_cold",
                "prompt_tokens": target_n,
                "actual_prompt_tokens": p_toks,
                "cached_tokens": cached_n,
                "rep": rep,
                "warmup": rep == 0,
                "elapsed_us": int(p_ms * 1000.0),
                "tok_per_s": round(tps, 2),
            }) + "\n")
            out.flush()
PY

copy_to "$RESULTS_DIR/drive_cold_1a.py" "$STUDIO2_HOST" "$S2_COLD_DRIVER" || fail_step1 "copy_cold_driver_failed"

run_on "$STUDIO2_HOST" "rm -f '$S2_SERVE_LOG' '$S2_COLD_JSONL' && DYLD_LIBRARY_PATH='$REMOTE_STAGE_DIR/lib' '$REMOTE_STAGE_DIR/mlx-serve' --model $MODEL_DIR_REMOTE --ctx-size $CTX_SIZE --prefix-cache-mem $PREFIX_CACHE_MEM --no-drafter --port $SERVE_PORT --log-file '$S2_SERVE_LOG' --serve >/dev/null 2>&1 & SPID=\$!; trap 'kill \$SPID 2>/dev/null || true; wait \$SPID 2>/dev/null || true' EXIT; python3 '$S2_COLD_DRIVER' '$SERVE_PORT' '$S2_SERVE_LOG' '$S2_COLD_JSONL' '$CTX_SIZE' '$PREFIX_CACHE_MEM' '$COLD_REPS'" \
    || fail_step1 "step1a_cold_serve_failed"

copy_from "$STUDIO2_HOST" "$S2_COLD_JSONL" "$RESULTS_DIR/1a-cold.jsonl" || fail_step1 "fetch_1a_cold_failed"
copy_from "$STUDIO2_HOST" "$S2_SERVE_LOG" "$RESULTS_DIR/1a-serve.log" || true

# ── 1b (RDMA Ceiling Reference): dist-probe --pp-sizes over 1-2 cable ──

detect_rdma_dev() {
    local box="$1"
    local ip="$2"
    run_on "$box" "python3 -c '
import re, subprocess, sys
ip = sys.argv[1]
out = subprocess.check_output([\"/sbin/ifconfig\"], text=True)
cur = None
for line in out.splitlines():
    m = re.match(r\"^([a-z0-9]+):\", line)
    if m:
        cur = m.group(1)
    elif cur and (\"inet \" + ip + \" \") in line:
        print(\"rdma_\" + cur)
        sys.exit(0)
sys.exit(1)
' '$ip'"
}

RDMA_DEV_S1="${RDMA_DEV_S1:-}"
RDMA_DEV_S2="${RDMA_DEV_S2:-}"
if [ -z "$RDMA_DEV_S1" ]; then
    RDMA_DEV_S1="$(detect_rdma_dev "$STUDIO1_HOST" "$S1_TB_IP" || true)"
fi
if [ -z "$RDMA_DEV_S2" ]; then
    RDMA_DEV_S2="$(detect_rdma_dev "$STUDIO2_HOST" "$S2_TB_IP" || true)"
fi

if [ -n "$RDMA_DEV_S1" ] && [ -n "$RDMA_DEV_S2" ]; then
    S1_RDMA_JSONL="$REMOTE_STAGE_DIR/1b-rdma.jsonl"
    run_on "$STUDIO1_HOST" "rm -f '$S1_RDMA_JSONL' && nohup env DYLD_LIBRARY_PATH='$REMOTE_STAGE_DIR/lib' '$REMOTE_STAGE_DIR/mlx-serve' dist-probe --pp-sizes --rank 0 --coordinator '${S1_TB_IP}:${RDMA_PORT}' --device '$RDMA_DEV_S1' --json-out '$S1_RDMA_JSONL' >'$REMOTE_STAGE_DIR/rdma-r0.log' 2>&1 &" \
        || fail_step1 "rdma_rank0_start_failed"
    sleep 1
    run_on "$STUDIO2_HOST" "DYLD_LIBRARY_PATH='$REMOTE_STAGE_DIR/lib' '$REMOTE_STAGE_DIR/mlx-serve' dist-probe --pp-sizes --rank 1 --coordinator '${S1_TB_IP}:${RDMA_PORT}' --device '$RDMA_DEV_S2' >'$REMOTE_STAGE_DIR/rdma-r1.log' 2>&1" \
        || fail_step1 "rdma_rank1_failed"
    sleep 1
    copy_from "$STUDIO1_HOST" "$S1_RDMA_JSONL" "$RESULTS_DIR/1b-rdma.jsonl" || fail_step1 "fetch_1b_rdma_failed"
fi

# ── 1a (Stage + Diag) + 1b (POSIX TCP solo, pipelined, overlapped 5 MoE) + 1c (KV export + TCP stream + fillImportedRows) ──

S2_MODEL_JSONL="$REMOTE_STAGE_DIR/1abc-model.jsonl"

run_on "$STUDIO1_HOST" "pkill -f '$REMOTE_STAGE_DIR/mlx-serve pp-probe recv' 2>/dev/null || true; nohup env DYLD_LIBRARY_PATH='$REMOTE_STAGE_DIR/lib' '$REMOTE_STAGE_DIR/mlx-serve' pp-probe recv --listen '${S1_TB_IP}:${PROBE_PORT}' >'$REMOTE_STAGE_DIR/pp-recv.log' 2>&1 &" \
    || fail_step1 "pp_recv_start_failed"
sleep 1

run_on "$STUDIO2_HOST" "rm -f '$S2_MODEL_JSONL' && DYLD_LIBRARY_PATH='$REMOTE_STAGE_DIR/lib' '$REMOTE_STAGE_DIR/mlx-serve' pp-probe model-bench --model $MODEL_DIR_REMOTE --peer '${S1_TB_IP}:${PROBE_PORT}' --json-out '$S2_MODEL_JSONL'" \
    || fail_step1 "pp_model_bench_failed"

copy_from "$STUDIO2_HOST" "$S2_MODEL_JSONL" "$RESULTS_DIR/1abc-model.jsonl" || fail_step1 "fetch_model_jsonl_failed"

# ── Merge all JSON lines and run Go/No-Go gate ──

MERGED_JSONL="$RESULTS_DIR/measurements.jsonl"
: > "$MERGED_JSONL"
for f in "$RESULTS_DIR"/*.jsonl; do
    [ "$f" = "$MERGED_JSONL" ] && continue
    [ -f "$f" ] && cat "$f" >> "$MERGED_JSONL"
done

SUM_ERR="$RESULTS_DIR/summarize.err"
if python3 "$SUMMARIZER" "$MERGED_JSONL" \
    --json-out "$RESULTS_DIR/summary.json" \
    --md-out "$RESULTS_DIR/summary.md" 2>"$SUM_ERR"; then
    echo "PP-STEP1-OK"
    exit 0
fi

WHY="$(tr '\n' ' ' < "$SUM_ERR" | sed 's/[[:space:]]\+/ /g; s/^ //; s/ $//')"
fail_step1 "${WHY:-gate_failed}"
