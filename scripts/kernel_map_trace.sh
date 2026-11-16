#!/bin/bash
# Record a Metal System Trace of the kernel_map test's timed steps, attached after the fixture is built.
#   scripts/kernel_map_trace.sh <out-dir> <tag> <lo-hi> <patched|stock> [rows] [steps]
# `patched` needs KMAP_MLX_LIB: a dir with a libmlx.dylib built from lib/mlx-src + docs/dsv41-kernel-map.mlx-labels.patch,
# beside the stock libmlxc.dylib, libjaccl.dylib and mlx.metallib (the patch labels each encoder with its kernels'
# names); it runs one primitive a command buffer (MLX_MAX_OPS_PER_BUFFER=0). `stock` is the unmodified build.
# Run outside the Claude Code sandbox. Needs `zig build test-build -Doptimize=ReleaseFast -Dtest-filter="dsv41 kernel map"`.
# A `KMAPVAR=DSV41_KMAP_HEAD` run times the head's and Engram's linears instead of the trunk.
set -u
out=$1
tag=$2
layers=$3
mode=$4
rows=${5:-1,6}
steps=${6:-32}
cd "$(dirname "$0")/.." || exit 1
mkdir -p "$out"
go="$out/$tag.go"
extra=()
if [ "$mode" = patched ]; then extra=(DYLD_LIBRARY_PATH="${KMAP_MLX_LIB:?}" MLX_MAX_OPS_PER_BUFFER=0); fi
env DSV41_DEC_UBENCH=1 "${KMAPVAR:-DSV41_KMAP}=1" DSV41_KMAP_ROWS="$rows" DSV41_KMAP_LAYERS="$layers" \
  DSV41_KMAP_STEPS="$steps" DSV41_KMAP_GO="$go" "${extra[@]}" ./zig-out/tests/test > "$out/$tag.out" 2>&1 &
pid=$!
for _ in $(seq 1 900); do grep -aq "\[kmap\] ready" "$out/$tag.out" && break; sleep 1; done
grep -aq "\[kmap\] ready" "$out/$tag.out" || { echo "never ready" >&2; kill $pid; exit 1; }
xcrun xctrace record --template 'Metal System Trace' --no-prompt --attach $pid --time-limit 40s \
  --output "$out/$tag.trace" > "$out/$tag.log" 2>&1 &
xp=$!
sleep 5
touch "$go"
wait $pid
wait $xp
