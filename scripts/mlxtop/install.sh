#!/bin/sh
# Install mlxtop: a venv with Textual, a `mlxtop` launcher, and (macOS) a LaunchAgent for the collector.
#
#   sh install.sh [--venv DIR] [--bin DIR] [collector/viewer flags...]
#
# Flags other than --venv and --bin (--url, --log, --db, --launchd-label, --api-key) are passed
# to the collector and the launcher. A key given with --api-key is stored in the LaunchAgent plist.
set -eu

SRC="$(cd "$(dirname "$0")" && pwd)"
VENV="$HOME/.local/share/mlxtop/venv"
BIN="$HOME/.local/bin"
while [ $# -gt 0 ]; do
  case "$1" in
    --venv) VENV="$2"; shift 2 ;;
    --bin) BIN="$2"; shift 2 ;;
    *) break ;;
  esac
done

quote() { printf "'%s'" "$(printf %s "$1" | sed "s/'/'\\\\''/g")"; }
ARGS=""
XML_ARGS=""
for a in "$@"; do
  ARGS="$ARGS $(quote "$a")"
  XML_ARGS="$XML_ARGS<string>$(printf %s "$a" | sed 's/&/\&amp;/g; s/</\&lt;/g')</string>"
done

[ -x "$VENV/bin/python" ] || python3 -m venv "$VENV"
"$VENV/bin/python" -m pip install --quiet textual

mkdir -p "$BIN"
cat > "$BIN/mlxtop" <<WRAP
#!/bin/sh
cd "$SRC" && exec "$VENV/bin/python" mlxtop.py$ARGS "\$@"
WRAP
chmod +x "$BIN/mlxtop"

if [ "$(uname)" != "Darwin" ]; then
  echo "viewer installed at $BIN/mlxtop; start the collector with: $VENV/bin/python $SRC/collect.py$ARGS"
  exit 0
fi

LABEL="dev.mlxtop.collect"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
mkdir -p "$HOME/Library/LaunchAgents" "$HOME/Library/Logs"
cat > "$PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key><array>
    <string>$VENV/bin/python</string><string>$SRC/collect.py</string>$XML_ARGS
  </array>
  <key>WorkingDirectory</key><string>$SRC</string>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>$HOME/Library/Logs/mlxtop-collect.log</string>
  <key>StandardErrorPath</key><string>$HOME/Library/Logs/mlxtop-collect.log</string>
</dict></plist>
PLIST
chmod 600 "$PLIST"

launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"
echo "collector $LABEL loaded; viewer at $BIN/mlxtop"
