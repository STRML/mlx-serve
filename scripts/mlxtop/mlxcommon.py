"""Shared pieces for the mlxtop collector and viewer: options, DB schema, parsing, rates, key, lock."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import plistlib
import re
import sqlite3
import urllib.error
import urllib.parse
import urllib.request

HOME = os.path.expanduser("~")
DB_PATH = os.path.join(HOME, ".local/share/mlxtop/mlxtop.db")
DEFAULT_URL = "http://127.0.0.1:11234"
DEFAULT_LABEL = "local.mlx-serve"
RETENTION_S = 7 * 86400


def metrics_url(base: str) -> str:
    return base.rstrip("/") + "/metrics.json"


def port_of(base: str) -> int:
    parts = urllib.parse.urlparse(base)
    return parts.port or (443 if parts.scheme == "https" else 80)


def default_log(port: int) -> str:
    """The server's own per-port log when it exists, else the LaunchAgent log location."""
    own = os.path.join(HOME, f".mlx-serve/logs/mlx-serve-{port}.log")
    return own if os.path.exists(own) else os.path.join(HOME, "Library/Logs/mlx-serve.log")


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--url", default=DEFAULT_URL, help="mlx-serve base URL (default %(default)s)")
    parser.add_argument("--api-key", help="server API key (else $MLX_SERVE_API_KEY, else the LaunchAgent plist)")
    parser.add_argument("--launchd-label", default=DEFAULT_LABEL,
                        help="LaunchAgent label whose plist holds --api-key (default %(default)s)")
    parser.add_argument("--db", default=DB_PATH, help="history database (default %(default)s)")


def key_source(args):
    """Callable returning the current API key; re-read on each call so a fixed key is picked up."""
    return lambda: find_key(args.api_key, args.launchd_label)


SCHEMA = """
CREATE TABLE IF NOT EXISTS samples (
  ts REAL PRIMARY KEY, up INTEGER, prefill_rate REAL, decode_rate REAL,
  gpu REAL, mem_mb REAL, running INTEGER, waiting INTEGER,
  prompt_tok REAL, cached_tok REAL);
CREATE TABLE IF NOT EXISTS requests (
  ts REAL, prompt INTEGER, gen INTEGER, cached INTEGER, prefill_rate REAL,
  decode_rate REAL, finish TEXT, ms INTEGER);
CREATE INDEX IF NOT EXISTS requests_ts ON requests (ts);
CREATE TABLE IF NOT EXISTS peers (ip TEXT, port INTEGER, host TEXT, ts REAL);
CREATE TABLE IF NOT EXISTS live (id INTEGER PRIMARY KEY CHECK (id = 1), ts REAL, json TEXT);
CREATE TABLE IF NOT EXISTS status (
  id INTEGER PRIMARY KEY CHECK (id = 1), ts REAL, state TEXT, down_since REAL,
  log_path TEXT, log_ok INTEGER, unparsed INTEGER, peers_ok INTEGER);
"""


def connect(path: str) -> sqlite3.Connection:
    """Open the writer connection: WAL, short busy timeout, schema ensured."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path, timeout=5)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(SCHEMA)
    have = {row[1] for row in conn.execute("PRAGMA table_info(samples)")}
    for col in ("prompt_tok", "cached_tok"):
        if col not in have:
            conn.execute(f"ALTER TABLE samples ADD COLUMN {col} REAL")
    return conn


def connect_readonly(path: str) -> sqlite3.Connection:
    """Open a read-only viewer connection; raises sqlite3.OperationalError if missing."""
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=0.2)
    conn.row_factory = sqlite3.Row
    return conn


def acquire_lock(path: str):
    """Take an exclusive non-blocking flock; return the open file, or None if held."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fh = open(path, "w")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        return None
    return fh


REQUEST_RE = re.compile(
    r"^\s*<- (\d+)\+(\d+) tokens(?: streamed| \((\d+)ms\))? "
    r"\[prefill: ([\d.]+) tok/s(?: \((\d+) cached / (\d+) total\))?, "
    r"decode: ([\d.]+) tok/s\] \[(\w+)\]"
)


def is_request_line(line: str) -> bool:
    return line.lstrip().startswith("<-")


def parse_request(line: str):
    """Parse one mlx-serve request line into a dict, or None when the form is unknown."""
    m = REQUEST_RE.match(line)
    if not m:
        return None
    prompt, gen, ms, prate, cached, _total, drate, finish = m.groups()
    return {
        "prompt": int(prompt), "gen": int(gen), "ms": int(ms) if ms else None,
        "cached": int(cached) if cached else 0, "prefill_rate": float(prate),
        "decode_rate": float(drate), "finish": finish,
    }


def rate(prev, cur, dt: float):
    """Tokens/s from two counter values; None when the counter went backwards (a restart)."""
    if prev is None or cur is None or dt <= 0 or cur < prev:
        return None
    return (cur - prev) / dt


class PrefillMeter:
    """Forwarded prefill tokens per poll. Live growth counts as it happens; a completion adds only
    what live did not show (the total advances at completion, after live has already fallen to 0).
    """

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.total = self.live = None
        self.pending = 0  # live tokens from prefills that ended but are not in total yet

    def update(self, metrics: dict):
        """Forwarded tokens since the previous call, or None on the first call or a counter reset."""
        c, g = metrics.get("counters", {}), metrics.get("gauges", {})
        total, live = c.get("prefill_tokens_total", 0), g.get("prefill_tokens_live", 0)
        if self.total is None or total < self.total:
            self.total, self.live, self.pending = total, live, 0
            return None
        grown = live - self.live if live >= self.live else live
        if live < self.live:
            self.pending += self.live
        done = total - self.total
        unseen = max(0, done - self.pending)
        self.pending = 0 if idle(g) else max(0, self.pending - done)
        self.total, self.live = total, live
        return grown + unseen


def idle(gauges: dict) -> bool:
    return not (gauges.get("requests_running") or gauges.get("requests_waiting"))


def delta(prev, cur):
    """Counter increase, or None when there is no previous value or the counter went backwards."""
    return None if prev is None or cur is None or cur < prev else cur - prev


def find_key(flag, label: str):
    """API key from --api-key, then $MLX_SERVE_API_KEY, then the LaunchAgent plist; None if none."""
    return flag or os.environ.get("MLX_SERVE_API_KEY") or read_api_key(
        os.path.join(HOME, f"Library/LaunchAgents/{label}.plist"))


def read_api_key(plist_path: str):
    """The argument after --api-key in the mlx-serve LaunchAgent plist, or None."""
    try:
        with open(plist_path, "rb") as fh:
            args = plistlib.load(fh).get("ProgramArguments", [])
        return args[args.index("--api-key") + 1]
    except (OSError, ValueError, IndexError, plistlib.InvalidFileException):
        return None


class MetricsError(Exception):
    """Carries the collector state name: down, no_metrics, no_key or bad_key."""

    def __init__(self, state: str):
        super().__init__(state)
        self.state = state


def fetch_metrics(url: str, key, timeout: float = 2.0) -> dict:
    """GET /metrics.json; raises MetricsError with the state to record."""
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"} if key else {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as err:
        denied = "bad_key" if key else "no_key"
        raise MetricsError({401: denied, 403: denied, 404: "no_metrics"}.get(err.code, "down")) from err
    except (OSError, ValueError) as err:
        raise MetricsError("down") from err
