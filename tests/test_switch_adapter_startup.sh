#!/bin/bash
# test_switch_adapter_startup.sh — the server boots with the layout-switch adapter installed (plan 61c step 4 <-> step 5):
# --internal-key-file plus a 4-rank tensor dist config. [1] it comes up and stays up (the path a Zig pre-release
# toolchain crashed at startup); [2] the mode poll reports TP1, admitting, with 3 peers (the formation backend's count);
# [3] tp4 is not 501 (a formation backend is installed); [4] drain goes to draining and closes public inference;
# [5] abort goes to aborted and reopens admission.
#
#   BINARY=./zig-out/bin/mlx-serve ./tests/test_switch_adapter_startup.sh [port]
#
# FULLY HERMETIC: an empty --model-dir loads nothing and no peer is contacted; runs in seconds.

set -u
source "$(dirname "$0")/lib/server.sh"

PORT="${1:-11268}"
BINARY="${BINARY:-./zig-out/bin/mlx-serve}"
PASS=0
FAIL=0
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

check() {
    if [ "$2" = "1" ]; then PASS=$((PASS + 1)); echo -e "  ${GREEN}PASS${NC} $1"; else FAIL=$((FAIL + 1)); echo -e "  ${RED}FAIL${NC} $1"; fi
}

if [ ! -x "$BINARY" ]; then
    echo "[fail] $BINARY not found — build first: zig build -Doptimize=ReleaseFast"
    exit 1
fi

WORK="$(mktemp -d)"
LOG="$WORK/server.log"
SRV=""
BASE="http://127.0.0.1:$PORT"
KEY="internal-key-$(head -c 24 /dev/urandom | base64 | tr -d '/+=')-padding"
printf '%s\n' "$KEY" > "$WORK/internal.key"; chmod 600 "$WORK/internal.key"
mkdir "$WORK/models" "$WORK/tp-model"

cleanup() {
    srv_stop "$SRV"
    srv_preclean "$PORT"
    rm -rf "$WORK"
}
trap cleanup EXIT

# post <path> <json>  -> "<status> <body>"
post() { curl -s -m 10 -o "$WORK/body" -w '%{http_code}' -H 'Content-Type: application/json' -H "Authorization: Bearer $KEY" -d "$2" "$BASE$1" | tr -d '\n'; printf ' %s' "$(cat "$WORK/body")"; }
mode() { curl -s -m 10 -H "Authorization: Bearer $KEY" "$BASE/v1/internal/mode"; }
field() { python3 -c 'import json,sys; v=json.loads(sys.argv[1]).get(sys.argv[2]); print(str(v).lower() if isinstance(v,bool) else v)' "$1" "$2" 2>/dev/null; }

echo "[1] boot with --internal-key-file and a 4-rank tensor dist config"
srv_preclean "$PORT"
"$BINARY" --serve --host 127.0.0.1 --model-dir "$WORK/models" --port "$PORT" --log-file off \
    --internal-key-file "$WORK/internal.key" \
    --dist tensor --dist-coordinator "127.0.0.1:$((PORT + 30000))" --dist-device "-,rdma_en2,rdma_en3,rdma_en4" \
    --dist-model "$WORK/tp-model" --dist-size 4 > "$LOG" 2>&1 &
SRV=$!
UP=0
srv_wait_health "$BASE" 60 0.5 "$SRV" && UP=1
check "the server comes up healthy" "$UP"
sleep 2
ALIVE=0
kill -0 "$SRV" 2> /dev/null && ALIVE=1
check "it is still running 2 s later (no startup crash)" "$ALIVE"
if [ "$UP" != 1 ] || [ "$ALIVE" != 1 ]; then
    wait "$SRV" 2> /dev/null; echo "  exit status $?"; tail -20 "$LOG"
    echo "TESTS: $PASS passed, $FAIL failed"; exit 1
fi

echo "[2] mode poll"
M="$(mode)"
check "mode is tp1" "$([ "$(field "$M" mode)" = tp1 ] && echo 1)"
check "admitting is true" "$([ "$(field "$M" admitting)" = true ] && echo 1)"
check "peers is 3 (the formation backend's count for a 4-rank group)" "$([ "$(field "$M" peers)" = 3 ] && echo 1)"

echo "[3] a formation backend is installed"
R="$(post /v1/internal/tp4 '{"n":1}')"
check "tp4 is not 501 (got ${R%% *})" "$([ "${R%% *}" != 501 ] && echo 1)"

echo "[4] drain"
R="$(post /v1/internal/drain '{"n":2}')"
check "drain answers 200 (got ${R%% *})" "$([ "${R%% *}" = 200 ] && echo 1)"
check "the box is draining" "$([ "$(field "$(mode)" state)" = draining ] && echo 1)"
PUB="$(curl -s -m 10 -o /dev/null -w '%{http_code}' -H 'Content-Type: application/json' -d '{"messages":[{"role":"user","content":"hi"}]}' "$BASE/v1/chat/completions")"
check "public inference is closed while draining (503, got $PUB)" "$([ "$PUB" = 503 ] && echo 1)"

echo "[5] abort"
R="$(post /v1/internal/abort '{"n":2}')"
check "abort answers 200 (got ${R%% *})" "$([ "${R%% *}" = 200 ] && echo 1)"
M="$(mode)"
check "the box is aborted" "$([ "$(field "$M" state)" = aborted ] && echo 1)"
check "admission is reopened" "$([ "$(field "$M" admitting)" = true ] && echo 1)"

echo "TESTS: $PASS passed, $FAIL failed"
[ "$FAIL" = 0 ]
