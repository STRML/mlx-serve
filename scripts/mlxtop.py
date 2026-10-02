#!/usr/bin/env python3
"""mlxtop: a terminal dashboard for a running mlx-serve.

Reads only `/metrics.json` and keeps its own history, the way a Prometheus
scraper does: every figure is a delta between two polled counter snapshots.
Raw samples are kept for 1 hour and one per minute up to 24 hours, in memory and
in a snapshot at ~/.cache/mlxtop/<host>-<port>.json, so restarting mlxtop keeps
the window.

    python3 -m pip install textual          # Python 3.9+
    mlx-serve --model <model> --metrics     # the feed is off without --metrics
    python3 scripts/mlxtop.py [--url http://127.0.0.1:11234] [--api-key KEY]

The key comes from `--api-key` or `$MLX_SERVE_API_KEY`. Keys 1/2/3/4 switch the
window between 5m, 15m, 1h and 24h; q quits.

Prefill and decode tok/s are tokens over the time the server spent in that phase
(the histogram sums), so they read per request and do not dilute while idle.
A server restart (a counter going down, or a new process_start_time_seconds)
draws a gap: no delta is taken across it. The `sessions[].request_id` and
`client` fields and the failed / rejected counters appear on newer servers;
without them the client column is blank and those rates read zero.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from textual.app import App, ComposeResult
from textual.widgets import DataTable, Static

DEFAULT_URL = "http://127.0.0.1:11234"
WINDOWS = {"1": (300, "5m"), "2": (900, "15m"), "3": (3600, "1h"), "4": (86400, "24h")}
RAW_KEEP_S = 3600
MAX_KEEP_S = 86400
LIVE_WINDOW_S = 10
SAVE_EVERY_S = 30
DEFAULT_REFRESH_S = 1.0
MIN_REFRESH_S = 0.2
BLOCKS = " ▁▂▃▄▅▆▇█"
AUTH_MESSAGE = "API key missing or rejected: pass --api-key or set MLX_SERVE_API_KEY"
NOTICES = {
    "unreachable": "mlx-serve unreachable: showing the last window, will resume",
    "metrics_off": "metrics are off: start mlx-serve with --metrics",
}
COUNTERS = ("prompt_tokens_total", "prefill_tokens_total", "prefix_cache_tokens_total",
            "generation_tokens_total", "requests_success_total", "requests_cancelled_total",
            "requests_failed_total", "requests_rejected_total", "prefix_cache_queries_total",
            "prefix_cache_hits_total")
PHASE_SUMS = {"prefill_s": "prefill_time_seconds", "decode_s": "decode_time_seconds"}
QUANTILE_HISTS = {"ttft": "time_to_first_token_seconds", "e2e": "e2e_request_latency_seconds"}


class FetchError(Exception):
    """The feed could not be read; `state` is unauthorized, metrics_off or unreachable."""

    def __init__(self, state: str):
        super().__init__(state)
        self.state = state


class AuthError(Exception):
    """The server rejected the key; polling cannot continue."""


def fetch_metrics(url: str, key: str | None, timeout: float = 2.0) -> dict:
    target = url if url.endswith("/metrics.json") else url.rstrip("/") + "/metrics.json"
    request = urllib.request.Request(target, headers={"Authorization": f"Bearer {key}"} if key else {})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.load(response)
    except urllib.error.HTTPError as err:
        state = {401: "unauthorized", 403: "unauthorized", 503: "metrics_off"}.get(err.code, "unreachable")
        raise FetchError(state) from err
    except (OSError, ValueError) as err:
        raise FetchError("unreachable") from err
    if not isinstance(data, dict):
        raise FetchError("unreachable")
    return data


# -- history ------------------------------------------------------------------
def make_sample(data: dict, t: float, epoch: int) -> dict:
    counters = data.get("counters") or {}
    hists = data.get("histograms") or {}
    c = {k: counters.get(k) or 0 for k in COUNTERS}
    for key, name in PHASE_SUMS.items():
        c[key] = (hists.get(name) or {}).get("sum") or 0
    h = {}
    for key, name in QUANTILE_HISTS.items():
        hist = hists.get(name) or {}
        h[key] = {"b": hist.get("bounds") or [], "c": hist.get("bucket_counts") or []}
    start = (data.get("gauges") or {}).get("process_start_time_seconds")
    return {"t": t, "e": epoch, "start": start, "c": c, "h": h}


def is_restart(prev: dict, cur: dict) -> bool:
    if prev["start"] is not None and cur["start"] is not None and prev["start"] != cur["start"]:
        return True
    return any(cur["c"][k] < prev["c"][k] for k in cur["c"])


def valid_sample(s) -> bool:
    return (isinstance(s, dict) and isinstance(s.get("t"), (int, float)) and isinstance(s.get("e"), int)
            and isinstance(s.get("c"), dict) and isinstance(s.get("h"), dict)
            and all(isinstance(v, (int, float)) for v in s["c"].values()))


class History:
    """Samples oldest first: raw for an hour, then one per minute and epoch up to a day.

    `samples[:older]` is the per-minute part; the rest is raw.
    """

    def __init__(self, samples: list | None = None):
        self.samples: list = []
        self.older = 0
        for s in samples or []:
            self.add(s)

    def add(self, sample: dict) -> None:
        self.samples.append(sample)
        raw_from = sample["t"] - RAW_KEEP_S
        while self.older < len(self.samples) and self.samples[self.older]["t"] <= raw_from:
            self.fold_into_minutes()
        expired = 0
        while self.samples[expired]["t"] < sample["t"] - MAX_KEEP_S:
            expired += 1
        del self.samples[:expired]
        self.older -= expired

    def fold_into_minutes(self) -> None:
        """Move the oldest raw sample into the per-minute part, replacing its minute's earlier one."""
        cur = self.samples[self.older]
        prev = self.samples[self.older - 1] if self.older else None
        if prev and prev["e"] == cur["e"] and prev["t"] // 60 == cur["t"] // 60:
            self.samples[self.older - 1] = cur
            del self.samples[self.older]
        else:
            self.older += 1

    def pairs(self, start: float, end: float):
        """Consecutive samples of one uninterrupted counter run whose later end is in (start, end]."""
        for a, b in zip(self.samples, self.samples[1:]):
            if a["e"] == b["e"] and start < b["t"] <= end:
                yield a, b


def hist_delta(a: dict, b: dict) -> list | None:
    if a["b"] != b["b"] or len(a["c"]) != len(b["c"]) or not b["c"]:
        return None
    return [y - x for x, y in zip(a["c"], b["c"])]


def accumulate(pairs) -> dict | None:
    total = {"dt": 0.0, "c": dict.fromkeys(COUNTERS, 0), "h": {}}
    total["c"].update(dict.fromkeys(PHASE_SUMS, 0))
    seen = False
    for a, b in pairs:
        seen = True
        total["dt"] += b["t"] - a["t"]
        for k in total["c"]:
            total["c"][k] += b["c"][k] - a["c"][k]
        for key in QUANTILE_HISTS:
            delta = hist_delta(a["h"].get(key) or {"b": [], "c": []}, b["h"].get(key) or {"b": [], "c": []})
            if delta is None:
                continue
            prior = total["h"].get(key)
            bounds = b["h"][key]["b"]
            total["h"][key] = (bounds, delta if prior is None else [x + y for x, y in zip(prior[1], delta)])
    return total if seen else None


def quantile(hist, q: float) -> float | None:
    """Prometheus-style estimate from cumulative bucket counts; None without observations."""
    if hist is None:
        return None
    bounds, cum = hist
    total = cum[-1]
    if total <= 0:
        return None
    rank = q * total
    i = next(i for i, c in enumerate(cum) if c >= rank)
    if i >= len(bounds):
        return bounds[-1]
    before = cum[i - 1] if i else 0
    lower = bounds[i - 1] if i else 0.0
    return lower + (bounds[i] - lower) * (rank - before) / (cum[i] - before)


def ratio(num, den, scale: float = 1.0) -> float | None:
    return num * scale / den if den and den > 0 and num >= 0 else None


def window_rates(total: dict | None) -> dict:
    keys = ("prefill", "decode", "cache_hit", "failed_per_min", "rejected_per_min", "cancelled_per_min",
            "ttft_p50", "ttft_p95", "e2e_p50", "e2e_p95")
    if total is None:
        return dict.fromkeys(keys)
    c, dt = total["c"], total["dt"]
    return {
        "prefill": ratio(c["prefill_tokens_total"], c["prefill_s"]),
        "decode": ratio(c["generation_tokens_total"], c["decode_s"]),
        "cache_hit": ratio(c["prefix_cache_hits_total"], c["prefix_cache_queries_total"], 100.0),
        "failed_per_min": ratio(c["requests_failed_total"], dt, 60.0),
        "rejected_per_min": ratio(c["requests_rejected_total"], dt, 60.0),
        "cancelled_per_min": ratio(c["requests_cancelled_total"], dt, 60.0),
        "ttft_p50": quantile(total["h"].get("ttft"), 0.5), "ttft_p95": quantile(total["h"].get("ttft"), 0.95),
        "e2e_p50": quantile(total["h"].get("e2e"), 0.5), "e2e_p95": quantile(total["h"].get("e2e"), 0.95),
    }


def bucket_series(history: History, start: float, end: float, width: int) -> dict:
    """Per-bucket prefill, decode tok/s and cache hit %; None where no valid interval landed."""
    span = (end - start) / width
    grouped: list = [[] for _ in range(width)]
    for a, b in history.pairs(start, end):
        grouped[min(width - 1, max(0, math.ceil((b["t"] - start) / span) - 1))].append((a, b))
    rates = [window_rates(accumulate(g)) if g else None for g in grouped]
    return {key: [r[key] if r else None for r in rates] for key in ("prefill", "decode", "cache_hit")}


# -- tracker ------------------------------------------------------------------
def snapshot_path(url: str) -> Path:
    parsed = urlparse(url)
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    host = re.sub(r"[^A-Za-z0-9._-]", "_", parsed.hostname or "localhost")
    return Path.home() / ".cache" / "mlxtop" / f"{host}-{port}.json"


def server_ident(url: str) -> str:
    parsed = urlparse(url)
    return f"{parsed.hostname or 'localhost'}:{parsed.port or (443 if parsed.scheme == 'https' else 80)}"


class Tracker:
    """History plus the live state of the last poll."""

    def __init__(self, path: Path | None = None, ident: str = ""):
        self.path, self.ident = path, ident
        self.history = History()
        self.state = "ok"
        self.note: str | None = None
        self.sessions: list = []
        self.gauges: dict = {}
        self.live: dict = {}
        self.last_saved = 0.0
        self.session_marks: dict = {}
        if path is not None:
            self.load()

    def load(self) -> None:
        try:
            raw = json.loads(self.path.read_text())
        except FileNotFoundError:
            return
        except (OSError, ValueError):
            self.note = "snapshot unreadable: starting with an empty window"
            return
        if not isinstance(raw, dict) or not isinstance(raw.get("samples"), list) \
                or not all(valid_sample(s) for s in raw["samples"]):
            self.note = "snapshot corrupt: starting with an empty window"
        elif raw.get("ident") != self.ident:
            self.note = "snapshot is from another server: starting with an empty window"
        else:
            self.history = History(raw["samples"])

    def save(self) -> None:
        tmp = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
            with os.fdopen(fd, "w") as f:
                json.dump({"ident": self.ident, "samples": self.history.samples}, f)
            os.replace(tmp, self.path)
        except OSError as err:
            self.note = f"snapshot not saved: {err.strerror or err}"
            if tmp:
                Path(tmp).unlink(missing_ok=True)

    def maybe_save(self, t: float) -> None:
        if self.path is not None and t - self.last_saved >= SAVE_EVERY_S:
            self.last_saved = t
            self.save()

    def observe(self, data: dict, t: float) -> None:
        prev = self.history.samples[-1] if self.history.samples else None
        sample = make_sample(data, t, prev["e"] if prev else 0)
        if prev and is_restart(prev, sample):
            sample["e"] += 1
        self.history.add(sample)
        self.state = "ok"
        self.gauges = data.get("gauges") or {}
        self.sessions = [s for s in data.get("sessions") or [] if isinstance(s, dict)]
        self.live = self.live_rates(sample["e"], t)

    def live_rates(self, epoch: int, t: float) -> dict:
        marks, rates = {}, {}
        for s in self.sessions:
            rid, gen = s.get("request_id"), s.get("generated_tokens")
            if rid is None or not isinstance(gen, (int, float)):
                continue
            marks[(epoch, rid)] = (t, gen)
            before = self.session_marks.get((epoch, rid))
            if before and t > before[0] and gen >= before[1]:
                rates[rid] = (gen - before[1]) / (t - before[0])
        self.session_marks = marks
        return rates

    def unreachable(self) -> None:
        self.fail("unreachable")

    def fail(self, state: str) -> None:
        self.state = state
        self.live = {}
        self.session_marks = {}


def poll(tracker: Tracker, fetch, url: str, key: str | None, t: float) -> None:
    try:
        data = fetch(url, key)
    except FetchError as err:
        if err.state == "unauthorized":
            raise AuthError(AUTH_MESSAGE) from err
        tracker.fail(err.state)
        return
    tracker.observe(data, t)


def probe(url: str, key: str | None) -> None:
    """Fail fast on a rejected key; any other fetch problem is left to the screen."""
    try:
        fetch_metrics(url, key)
    except FetchError as err:
        if err.state == "unauthorized":
            raise AuthError(AUTH_MESSAGE) from err


# -- view model ---------------------------------------------------------------
def num(value, spec: str = "{:,.0f}") -> str:
    return "-" if value is None else spec.format(value)


def bar(fraction: float, width: int = 30) -> str:
    fraction = min(max(fraction, 0.0), 1.0)
    full = round(fraction * width)
    return "█" * full + "░" * (width - full)


def sparkline(values: list, width: int, top: float | None = None) -> str:
    """One block per bucket, scaled to `top` (default the series maximum); gaps and idle buckets are blank."""
    shown = values[-width:]
    top = top or max((v for v in shown if v is not None), default=0) or 1
    return "".join(" " if v is None else BLOCKS[min(8, round(v / top * 8))] for v in shown)


def session_row(s: dict, rates: dict) -> list:
    rate = rates.get(s.get("request_id"))
    return [s.get("phase", "-"), s.get("model", "-"), num(s.get("context_tokens")), num(s.get("generated_tokens")),
            num(rate, "{:,.1f}"), num(s.get("cached_tokens")), num((s.get("state_bytes") or 0) / 1e6),
            s.get("client") or "-"]


def model_rows(sessions: list, rates: dict) -> list:
    groups: dict = {}
    for s in sessions:
        count, rate = groups.get(s.get("model", "-"), (0, None))
        r = rates.get(s.get("request_id"))
        groups[s.get("model", "-")] = (count + 1, rate if r is None else (rate or 0) + r)
    return [[model, str(count), num(rate, "{:,.1f}")] for model, (count, rate) in sorted(groups.items())]


def header_text(g: dict) -> str:
    return (f"GPU {num(g.get('gpu_utilization_pct'))}%  mem {num(g.get('memory_mb'))} MB  "
            f"running {num(g.get('requests_running'))}  waiting {num(g.get('requests_waiting'))}")


def build_view(tracker: Tracker, window_s: int, now: float, width: int) -> dict:
    """Everything the screen shows, from the history and the last poll."""
    start = now - window_s
    pairs = list(tracker.history.pairs(start, now))
    total = accumulate(pairs)
    live = window_rates(accumulate(tracker.history.pairs(now - LIVE_WINDOW_S, now)))
    counts = {k: (total["c"][f"requests_{k}_total"] if total else None)
              for k in ("success", "failed", "rejected", "cancelled")}
    return {
        "notice": NOTICES.get(tracker.state),
        "note": tracker.note,
        "header": header_text(tracker.gauges),
        "rates": window_rates(total),
        "live": {"prefill": live["prefill"], "decode": live["decode"]},
        "counts": counts,
        "spark": bucket_series(tracker.history, start, now, width),
        "sessions": [session_row(s, tracker.live) for s in tracker.sessions],
        "models": model_rows(tracker.sessions, tracker.live),
    }


# -- rendering ----------------------------------------------------------------
def secs(value) -> str:
    return "-" if value is None else f"{value:.2f}s"


def rates_text(view: dict, label: str) -> str:
    r, live, n = view["rates"], view["live"], view["counts"]
    return (f"now: prefill {num(live['prefill'])} tok/s   decode {num(live['decode'])} tok/s\n"
            f"{label}: prefill {num(r['prefill'])} tok/s   decode {num(r['decode'])} tok/s   "
            f"cache hit {num(r['cache_hit'], '{:.1f}%')}\n"
            f"{label}: ttft p50 {secs(r['ttft_p50'])} p95 {secs(r['ttft_p95'])}   "
            f"e2e p50 {secs(r['e2e_p50'])} p95 {secs(r['e2e_p95'])}\n"
            f"{label}: failed {num(r['failed_per_min'], '{:.1f}')}/min ({num(n['failed'])})   "
            f"rejected {num(r['rejected_per_min'], '{:.1f}')}/min ({num(n['rejected'])})   "
            f"cancelled {num(r['cancelled_per_min'], '{:.1f}')}/min ({num(n['cancelled'])})")


def spark_text(view: dict, label: str, width: int) -> str:
    spark = view["spark"]
    rows = [f"window {label}   [1] 5m  [2] 15m  [3] 1h  [4] 24h"]
    for title, key in (("prefill tok/s", "prefill"), ("decode tok/s", "decode")):
        peak = max((v for v in spark[key] if v is not None), default=None)
        rows.append(f"{title:>14} {sparkline(spark[key], width)} max {num(peak)}")
    rows.append(f"{'cache hit %':>14} {sparkline(spark['cache_hit'], width, top=100)}")
    return "\n".join(rows)


class MlxTop(App):
    CSS = """
    #header, #live, #spark, #footer { height: auto; padding: 0 1; }
    #banner { height: auto; padding: 0 1; color: red; }
    #sessions { height: 8; }
    #models { height: 5; }
    """
    BINDINGS = [("q", "quit", "Quit"), ("1", "window('1')", "5m"), ("2", "window('2')", "15m"),
                ("3", "window('3')", "1h"), ("4", "window('4')", "24h")]

    def __init__(self, url: str, key: str | None, fetch=fetch_metrics, interval: float = DEFAULT_REFRESH_S,
                 tracker: Tracker | None = None, clock=time.time):
        super().__init__()
        self.url, self.key, self.fetch, self.interval, self.clock = url, key, fetch, interval, clock
        self.tracker = tracker or Tracker()
        self.window_key = "3"
        self.stop_message: str | None = None

    def compose(self) -> ComposeResult:
        for name in ("header", "banner", "live"):
            yield Static(id=name)
        yield DataTable(id="sessions")
        yield DataTable(id="models")
        yield Static(id="spark")
        yield Static("q quit   1/2/3/4 window", id="footer")

    def on_mount(self) -> None:
        self.query_one("#sessions", DataTable).add_columns(
            "phase", "model", "context", "generated", "tok/s", "cached", "state MB", "client")
        self.query_one("#models", DataTable).add_columns("model", "sessions", "tok/s")
        self.set_interval(self.interval, self.refresh_view)
        self.call_after_refresh(self.refresh_view)

    def action_window(self, key: str) -> None:
        self.window_key = key
        self.call_after_refresh(self.refresh_view)

    def show(self, name: str, text: str) -> None:
        self.query_one(f"#{name}", Static).update(text)

    def fill(self, name: str, rows: list) -> None:
        table = self.query_one(f"#{name}", DataTable)
        table.clear()
        for row in rows:
            table.add_row(*row)

    async def refresh_view(self) -> None:
        now = self.clock()
        try:
            await asyncio.to_thread(poll, self.tracker, self.fetch, self.url, self.key, now)
        except AuthError as err:
            self.stop_message = str(err)
            self.exit()
            return
        self.tracker.maybe_save(now)
        seconds, label = WINDOWS[self.window_key]
        width = max(10, self.size.width - 34)
        view = build_view(self.tracker, seconds, now, width)
        self.show("banner", " ".join(filter(None, (view["notice"], view["note"]))))
        self.show("header", f"mlxtop  {self.url}  {view['header']}")
        self.show("live", rates_text(view, label))
        self.show("spark", spark_text(view, label, width))
        self.fill("sessions", view["sessions"])
        self.fill("models", view["models"])


def refresh_seconds(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number of seconds")
    if value < MIN_REFRESH_S:
        raise argparse.ArgumentTypeError(f"must be at least {MIN_REFRESH_S} seconds")
    return value


def main(argv: list | None = None) -> None:
    parser = argparse.ArgumentParser(description="Terminal dashboard for mlx-serve (/metrics.json).")
    parser.add_argument("--url", default=DEFAULT_URL, help=f"server base URL (default {DEFAULT_URL})")
    parser.add_argument("--api-key", default=None, help="API key (default $MLX_SERVE_API_KEY)")
    parser.add_argument("--interval", type=refresh_seconds, default=DEFAULT_REFRESH_S, metavar="SECONDS",
                        help=f"poll and refresh period (default {DEFAULT_REFRESH_S:g}, minimum {MIN_REFRESH_S:g})")
    args = parser.parse_args(argv)
    key = args.api_key or os.environ.get("MLX_SERVE_API_KEY")
    try:
        probe(args.url, key)
    except AuthError as err:
        parser.exit(2, f"mlxtop: {err}\n")
    tracker = Tracker(snapshot_path(args.url), server_ident(args.url))
    app = MlxTop(args.url, key, interval=args.interval, tracker=tracker)
    try:
        app.run()
    finally:
        tracker.save()
    if app.stop_message:
        parser.exit(2, f"mlxtop: {app.stop_message}\n")


if __name__ == "__main__":
    main()
