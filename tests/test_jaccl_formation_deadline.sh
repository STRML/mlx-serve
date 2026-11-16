#!/usr/bin/env bash
# Compile and exercise an already-patched MLX source tree; never modify it.
# Usage: bash tests/test_jaccl_formation_deadline.sh <patched-mlx-src>
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MLX_SRC="${1:?usage: $0 <patched-mlx-src>}"
JACCL="$MLX_SRC/mlx/distributed/jaccl/lib"
TEST_TMP="$(mktemp -d "${TMPDIR:-/private/tmp}/jaccl-deadline-test.XXXXXX")"
echo "JACCL test artifacts: $TEST_TMP"
clang++ -std=c++17 -O1 -I "$JACCL" \
  "$JACCL/jaccl/tcp.cpp" "$HERE/jaccl-deadline-behavior.cpp" \
  -o "$TEST_TMP/behavior"
"$TEST_TMP/behavior"
