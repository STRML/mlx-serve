#!/bin/bash
# Invoke on studio4 ONLY as the command of studio1's single-box borrow.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
case ${1:---dry-run} in
--dry-run)
    cat <<'EOF'
No commands executed. Stage this tree at studio4:build/pp4-step0-src first.
On studio1 (the borrow owns studio4's Latch hold):
nohup bash ~/bench/ds41-tp2x2.sh borrow studio4 --max 10800 -- bash -c 'ssh -o BatchMode=yes 10.254.14.4 "BORROW=$BORROW BORROW_BOXES=$BORROW_BOXES bash ~/build/pp4-step0-src/scripts/bench/pp4-step0.sh --run"' > ~/bench/pp4-step0-window.log 2>&1 &
Inside borrow: clone matching integ-61c build template, build ReleaseFast,
run CPU unit tests in ReleaseFast and Debug, sign server with ~/bench/signbin.sh.
One server at a time, loopback :18754, prefix caches off, chunk=2048.
Read real ~/bench/engram-corpus.md with pack tokenizer. Depths 32768/131072.
Before model load: 30s offline check of 75 distinct sample/warm-up prefixes.
Controls before/after: 6 per depth. Profile: 6 per depth per shard (4 shards).
CPU-only cold first-use reads finish before any server starts; memory floor16GiB.
--run --depths 131072 selects only that depth, including warmups.
PP4_BINARY selects an existing signed server for a matching-SHA continuation.
No live binary install. Cleanup drains private request, stops owned PIDs.
Output ~/bench/pp4-step0-<timestamp>/bound.{json,md}; borrow returns S4.
EOF
    exit 0 ;;
--run) shift ;;
*) echo 'usage: pp4-step0.sh --dry-run|--run [--depths 32768 131072]' >&2; exit 2 ;;
esac
if [ "${BORROW:-}" != S4 ] || [ "${BORROW_BOXES:-}" != studio4 ] || [ "$(hostname -s)" != studio4 ]; then
    echo 'requires studio4 borrow (BORROW=S4 BORROW_BOXES=studio4)' >&2
    exit 2
fi
export PATH="$HOME/opt/zig-0.17.1:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"
TPL=${PP4_TEMPLATE:-$HOME/build/integ-61c-runner-2335}
STAGE=${PP4_MLX_STAGE:-$HOME/opt/mlx-serve-integ-2335/lib/mlx}
OUT=${PP4_OUT:-$HOME/bench/pp4-step0-$(date +%Y%m%d-%H%M%S)}
[ -d "$TPL/lib" ] && [ -d "$STAGE" ] && [ ! -e "$OUT" ] || { echo "PP4-STEP0-REFUSED: missing $TPL/lib or $STAGE, or $OUT exists" >&2; exit 2; }
mkdir -p "$OUT"
BUILD="$OUT/build"
cp -Rc "$TPL/." "$BUILD/"
for d in src scripts build tests; do
    rsync -a --delete "$ROOT/$d/" "$BUILD/$d/"
done
cp "$ROOT/build.zig" "$ROOT/build.zig.zon" "$BUILD/"
. "$HOME/bench/tree-lib.sh"
sync_lib "$ROOT" "$BUILD"
[ ! -L "$BUILD/lib" ]
if [ -d "$BUILD/lib/mlx" ] && [ ! -L "$BUILD/lib/mlx" ]; then
    mv "$BUILD/lib/mlx" "$BUILD/lib/mlx.template"
fi
rm -f "$BUILD/lib/mlx"
ln -s "$STAGE" "$BUILD/lib/mlx"
cd "$BUILD"
zig version > "$OUT/zig-version.txt"
for mode in ReleaseFast Debug; do
    zig test src/plugins/deepseek_v41/pp4_step0.zig -lc -O "$mode" > "$OUT/diag-test-$mode.log" 2>&1
    zig test src/pp4_first_use_bench.zig -lc -O "$mode" --test-filter 'pp4 first use' > "$OUT/reader-test-$mode.log" 2>&1
done
SERVER_BINARY=${PP4_BINARY:-$BUILD/zig-out/bin/mlx-serve}
if [ -z "${PP4_BINARY:-}" ]; then
    zig build -Doptimize=ReleaseFast -Dunits=false > "$OUT/build.log" 2>&1
    bash "$HOME/bench/signbin.sh" "$SERVER_BINARY" > "$OUT/sign.log" 2>&1
fi
zig build-exe src/pp4_first_use_bench.zig -lc -O ReleaseFast -femit-bin="$OUT/pp4-first-use" > "$OUT/reader-build.log" 2>&1
imports=$(dyld_info -imports "$SERVER_BINARY" | grep -c _mlx_)
[ "$imports" -gt 200 ]
shasum -a 256 "$SERVER_BINARY" "$OUT/pp4-first-use" > "$OUT/binaries.sha256"
echo "PP4-STEP0-BUILT $OUT $(date -u +%FT%TZ)"
window_rc=0
taskpolicy -c utility /usr/bin/python3 scripts/bench/pp4_window.py --binary "$SERVER_BINARY" \
    --reader "$OUT/pp4-first-use" --out "$OUT" \
    --pack "$HOME/models-iq/DeepSeek-V4.1-Flash-1017-MLX-Serve-iQ-MLX-3.0bpw" \
    --corpus "$HOME/bench/engram-corpus.md" "$@" || window_rc=$?
/usr/bin/python3 scripts/bench/pp4_step0.py "$OUT"
[ "$window_rc" -eq 0 ] || exit "$window_rc"
echo "PP4-STEP0-OK $OUT $(date -u +%FT%TZ)"
