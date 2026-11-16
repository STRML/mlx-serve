#!/bin/bash
# test_internal_switch.sh — /v1/internal/* and the layout-switch admission gate (plan 61c step 5), over a real HTTP
# server. [1] without --internal-key-file the paths 404; [2] startup refuses a loose-mode key file and a key equal to
# --api-key; [3] the serving key, no key and a wrong key are 401 even from loopback, the internal key reads the mode;
# [4] drain closes public inference and model operations (tokenize, load) with a 503 while /health, /v1/models and the
# mode poll stay up, abort reopens it and chat reaches the model path (the missing model's 503, asserted by body);
# [5] repeats, a stale N (409 carrying the box's N), an abort that fences a later drain, and tp4 with no formation
# backend (501, state untouched); [6] a drain nobody follows expires on its own and reopens admission.
#
#   ./tests/test_internal_switch.sh [port]
#
# FULLY HERMETIC: an empty --model-dir loads nothing; runs in seconds. Hermetic twins: attempt.zig, switch.zig and
# endpoints.zig tests.

set -u
source "$(dirname "$0")/lib/server.sh"

PORT="${1:-11266}"
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
SERVING="serving-key-0123456789"
printf '%s\n' "$KEY" > "$WORK/internal.key"; chmod 600 "$WORK/internal.key"
mkdir "$WORK/models"

cleanup() {
    srv_stop "$SRV"
    srv_preclean "$PORT"
    rm -rf "$WORK"
}
trap cleanup EXIT

boot() {
    srv_stop "$SRV"; SRV=""
    srv_preclean "$PORT"
    : > "$LOG"
    "$BINARY" --serve --model-dir "$WORK/models" --port "$PORT" --log-file off "$@" > "$LOG" 2>&1 &
    SRV=$!
    srv_wait_health "$BASE" 60 0.5 "$SRV"
}

# post <path> <json> [key]  -> "<status> <body>"
post() { curl -s -m 10 -o "$WORK/body" -w '%{http_code}' -H 'Content-Type: application/json' ${3:+-H "Authorization: Bearer $3"} -d "$2" "$BASE$1" | tr -d '\n'; printf ' %s' "$(cat "$WORK/body")"; }
get() { curl -s -m 10 -o "$WORK/body" -w '%{http_code}' ${2:+-H "Authorization: Bearer $2"} "$BASE$1" | tr -d '\n'; printf ' %s' "$(cat "$WORK/body")"; }
status() { echo "${1%% *}"; }
field() { python3 -c 'import json,sys; print(json.loads(sys.argv[1].split(" ",1)[1]).get(sys.argv[2]))' "$1" "$2"; }
has() { case "$1" in *"$2"*) echo 1 ;; *) echo 0 ;; esac; }
eq() { [ "$1" = "$2" ] && echo 1 || echo 0; }

echo "Layout switch internal endpoints (port $PORT)"

echo "[1/6] no flag: /v1/internal/* does not exist"
boot || { echo "server did not come up"; cat "$LOG"; exit 1; }
check "GET /v1/internal/mode is 404" "$(eq "$(status "$(get /v1/internal/mode "$KEY")")" 404)"
check "/health keeps its plain body" "$(eq "$(curl -s "$BASE/health")" '{"status":"ok"}')"

echo "[2/6] startup refuses a bad key file"
cp "$WORK/internal.key" "$WORK/loose.key"; chmod 644 "$WORK/loose.key"
srv_stop "$SRV"; SRV=""; srv_preclean "$PORT"
"$BINARY" --serve --model-dir "$WORK/models" --port "$PORT" --log-file off --internal-key-file "$WORK/loose.key" > "$LOG" 2>&1
check "mode 0644 key file: exit non-zero" "$([ $? -ne 0 ] && echo 1 || echo 0)"
check "the log names the mode rule" "$(has "$(cat "$LOG")" '0600')"
printf '%s\n' "$SERVING-padding-padding-padding" > "$WORK/same.key"; chmod 600 "$WORK/same.key"
"$BINARY" --serve --model-dir "$WORK/models" --port "$PORT" --log-file off --api-key "$SERVING-padding-padding-padding" --internal-key-file "$WORK/same.key" > "$LOG" 2>&1
check "key equal to --api-key: exit non-zero" "$([ $? -ne 0 ] && echo 1 || echo 0)"

echo "[3/6] only the internal key reads /v1/internal/*"
boot --api-key "$SERVING" --internal-key-file "$WORK/internal.key" || { echo "server did not come up"; cat "$LOG"; exit 1; }
check "serving key is 401 on the mode poll (loopback included)" "$(eq "$(status "$(get /v1/internal/mode "$SERVING")")" 401)"
check "no key is 401" "$(eq "$(status "$(get /v1/internal/mode)")" 401)"
check "serving key is 401 on abort" "$(eq "$(status "$(post /v1/internal/abort '{"n":1}' "$SERVING")")" 401)"
M="$(get /v1/internal/mode "$KEY")"
check "internal key reads the mode: tp1, idle, admitting" "$([ "$(status "$M")" = 200 ] && [ "$(field "$M" mode)" = tp1 ] && [ "$(field "$M" state)" = idle ] && [ "$(field "$M" admitting)" = True ] && echo 1 || echo 0)"
check "no --metrics: the mode poll still carries a progress count (stall detection runs)" "$([ "$(field "$M" progress)" != None ] && echo 1 || echo 0)"
check "/health carries the mode fields" "$(has "$(curl -s "$BASE/health")" '"mode":"tp1"')"

echo "[4/6] drain closes public inference; abort reopens it"
D="$(post /v1/internal/drain '{"n":7}' "$KEY")"
check "drain 7 is 200, drained, state draining" "$([ "$(status "$D")" = 200 ] && [ "$(field "$D" drained)" = True ] && [ "$(field "$D" state)" = draining ] && echo 1 || echo 0)"
C="$(post /v1/chat/completions '{"messages":[{"role":"user","content":"hi"}]}')"
check "chat is 503 naming the switch" "$([ "$(status "$C")" = 503 ] && [ "$(has "$C" 'Layout switch')" = 1 ] && echo 1 || echo 0)"
for op in /tokenize /detokenize /v1/load-model /v1/unload-model; do
    O="$(post "$op" '{}')"
    check "$op is refused while draining (model operations count as inference)" "$([ "$(status "$O")" = 503 ] && [ "$(has "$O" 'Layout switch')" = 1 ] && echo 1 || echo 0)"
done
check "/v1/models still answers" "$(eq "$(curl -s -o /dev/null -w '%{http_code}' "$BASE/v1/models")" 200)"
check "/health still 200 while draining" "$(eq "$(curl -s -o /dev/null -w '%{http_code}' "$BASE/health")" 200)"
A="$(post /v1/internal/abort '{"n":7}' "$KEY")"
check "abort 7 is 200, state aborted" "$([ "$(status "$A")" = 200 ] && [ "$(field "$A" state)" = aborted ] && echo 1 || echo 0)"
C="$(post /v1/chat/completions '{"messages":[{"role":"user","content":"hi"}]}')"
check "chat reaches the model path again: 503 from the missing model, not the switch" "$([ "$(status "$C")" = 503 ] && [ "$(has "$C" 'No default model configured')" = 1 ] && [ "$(has "$C" 'Layout switch')" = 0 ] && echo 1 || echo 0)"

echo "[5/6] repeats, stale N, fences, no formation backend"
check "abort 7 again is 200 (idempotent)" "$(eq "$(status "$(post /v1/internal/abort '{"n":7}' "$KEY")")" 200)"
S="$(post /v1/internal/drain '{"n":3}' "$KEY")"
check "drain 3 is 409 stale and reports N 7" "$([ "$(status "$S")" = 409 ] && [ "$(field "$S" reason)" = stale ] && [ "$(field "$S" n)" = 7 ] && echo 1 || echo 0)"
check "abort 12 fences an attempt the box never saw" "$([ "$(field "$(post /v1/internal/abort '{"n":12}' "$KEY")" n)" = 12 ] && echo 1 || echo 0)"
check "a late drain 12 is refused" "$(eq "$(status "$(post /v1/internal/drain '{"n":12}' "$KEY")")" 409)"
post /v1/internal/drain '{"n":13}' "$KEY" > /dev/null
T="$(post /v1/internal/tp4 '{"n":13}' "$KEY")"
check "tp4 with no formation backend is 501 naming the missing adapter" "$([ "$(status "$T")" = 501 ] && [ "$(has "$T" 'the switch adapter is not installed')" = 1 ] && echo 1 || echo 0)"
check "…and the box stays draining" "$(eq "$(field "$(get /v1/internal/mode "$KEY")" state)" draining)"
check "a body naming a coordinator is ignored, not obeyed" "$(eq "$(status "$(post /v1/internal/join '{"n":13,"coordinator":"10.9.9.9:1"}' "$KEY")")" 501)"

echo "[6/6] a drain nobody follows expires into aborted"
sleep 4
E="$(get /v1/internal/mode "$KEY")"
check "after 3.5 s the box aborted 13 itself and admits" "$([ "$(field "$E" state)" = aborted ] && [ "$(field "$E" admitting)" = True ] && echo 1 || echo 0)"

echo
echo "internal switch: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
