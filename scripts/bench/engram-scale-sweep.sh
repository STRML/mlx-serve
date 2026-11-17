#!/bin/bash
# Single-box sweep: launch on studio1 with borrow studioN -- ssh <cable> ... .
set -euo pipefail
OUT=${1:?output directory}
BASE="$HOME/bench/engram-scale"
PACK="$HOME/models-iq/DeepSeek-V4.1-Flash-1017-MLX-Serve-iQ-MLX-3.0bpw"
mkdir -p "$OUT"
for rep in 1 2 3; do
    order="1 2 4 8 16 32"
    [ "$rep" != 2 ] || order="32 16 8 4 2 1"
    [ "$rep" != 3 ] || order="8 1 32 4 16 2"
    for depth in 32768 131072; do
        for n in $order; do
            echo "ARM full d=$depth n=$n rep=$rep $(date -u +%FT%TZ)"
            taskpolicy -c utility /usr/bin/python3 "$BASE/engram_scale.py" \
                --pack "$PACK" --binary "$BASE/engram-scale" --tokens "$BASE/corpus.u32" \
                --depth "$depth" --readers "$n" --out "$OUT/full-d$depth-n$n-r$rep"
        done
    done
done
# Chunk-width control quantifies the extra reads from deduplicating each chunk separately.
for depth in 32768 131072; do
    for n in 1 16 32; do
        echo "ARM chunks d=$depth n=$n $(date -u +%FT%TZ)"
        taskpolicy -c utility /usr/bin/python3 "$BASE/engram_scale.py" \
            --pack "$PACK" --binary "$BASE/engram-scale" --tokens "$BASE/corpus.u32" \
            --depth "$depth" --readers "$n" --chunk 2048 --out "$OUT/chunks-d$depth-n$n"
    done
done
echo "ENGRAM-SCALE-SWEEP-OK $(date -u +%FT%TZ)"
