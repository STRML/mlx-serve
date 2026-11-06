#!/bin/bash
# Two dist-probe ranks on this Mac over MLX's ring backend (TCP, no RDMA): round trips at every size, all_sum checks.
#   tests/test_dist_ring.sh [path/to/mlx-serve]
set -u
cd "$(dirname "$0")/.." || exit 1
BIN=${1:-zig-out/bin/mlx-serve}
[ -x "$BIN" ] || { echo "FAIL: no binary at $BIN (zig build first)"; exit 1; }
DIR=$(mktemp -d)
trap 'kill $P0 $P1 2>/dev/null; rm -rf "$DIR"' EXIT
base=$((20000 + RANDOM % 20000))
hosts="127.0.0.1:$base,127.0.0.1:$((base + 1))"
coord="127.0.0.1:$((base + 10))"
"$BIN" dist-probe --rank 0 --device "tcp:$hosts" --coordinator "$coord" > "$DIR/r0.log" 2>&1 &
P0=$!
"$BIN" dist-probe --rank 1 --device "tcp:$hosts" --coordinator "$coord" > "$DIR/r1.log" 2>&1 &
P1=$!
for _ in $(seq 1 120); do
    kill -0 $P0 2>/dev/null || kill -0 $P1 2>/dev/null || break
    sleep 1
done
fail=0
for r in 0 1; do
    log="$DIR/r$r.log"
    grep -q "\[dist-probe\] done" "$log" || { echo "FAIL: rank $r did not finish"; tail -20 "$log"; fail=1; continue; }
    [ "$(grep -c "\[dist-probe\] round trip" "$log")" = 5 ] || { echo "FAIL: rank $r round trips"; fail=1; }
    while read -r a b; do
        [ "$a" = "$b" ] || { echo "FAIL: rank $r all_sum check $a != $b"; fail=1; }
    done < <(grep -o "sum check [0-9]* == [0-9]*" "$log" | awk '{print $3, $5}')
    [ "$(grep -c "sum check" "$log")" = 2 ] || { echo "FAIL: rank $r all_sum count"; fail=1; }
done
[ $fail = 0 ] && { echo "PASS: two ranks over the ring backend"; grep -h "round trip\|all_sum" "$DIR/r0.log"; }
exit $fail
