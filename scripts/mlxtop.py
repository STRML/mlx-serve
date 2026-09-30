#!/usr/bin/env python3
"""mlxtop: a terminal dashboard for a running mlx-serve.

A stateless client of `/metrics.json`: it polls once a second, keeps nothing,
and reads every chart from the server's own `monitor` history, so it shows the
same past no matter when it was started.

    python3 -m pip install textual          # Python 3.9+
    mlx-serve --model <model> --metrics     # the feed is off without --metrics
    python3 scripts/mlxtop.py [--url http://127.0.0.1:11234] [--api-key KEY]

The key comes from `--api-key` or `$MLX_SERVE_API_KEY`. Keys 1/2/3 switch the
sparklines between 5m, 15m and 1h; q quits. The client column (peer and
User-Agent) is filled only when the server shows it to you: a loopback reader,
or any reader of a server started with `--api-key`. An older server without the
`monitor` object gets the live values only.

Prefill tok/s counts forwarded tokens (those the model actually computed, not
prefix-cache hits) over the time the server spent prefilling.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import urllib.error
import urllib.request
from datetime import datetime

from textual.app import App, ComposeResult
from textual.widgets import DataTable, Static

DEFAULT_URL = "http://127.0.0.1:11234"
WINDOWS = {"1": (300, "5m"), "2": (900, "15m"), "3": (3600, "1h")}
SAMPLE_INTERVAL_MS = 2000
DEFAULT_REFRESH_S = 1.0
MIN_REFRESH_S = 0.2
LIVE_RATE_S = 10
REQUEST_ROWS = 100
BLOCKS = " ▁▂▃▄▅▆▇█"
NO_MONITOR = "this server has no monitor history (older mlx-serve): live values only"
BANNERS = {
    "unauthorized": "API key missing or rejected (--api-key or $MLX_SERVE_API_KEY)",
    "metrics_off": "metrics are off: start mlx-serve with --metrics",
    "unreachable": "mlx-serve unreachable at {url}",
}
PREFILL = ("prefill_tokens_forwarded_live_total", "prefill_active_ns_total")
DECODE = ("generation_tokens_live", "decode_active_ns_total")
CACHE = ("cache_hits_total", "cache_queries_total")


class FetchError(Exception):
    """The feed could not be read; `state` keys into BANNERS."""

    def __init__(self, state: str):
        super().__init__(state)
        self.state = state


def fetch_metrics(url: str, key: str | None, timeout: float = 2.0) -> dict:
    target = url if url.endswith("/metrics.json") else url.rstrip("/") + "/metrics.json"
    request = urllib.request.Request(target, headers={"Authorization": f"Bearer {key}"} if key else {})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as err:
        state = {401: "unauthorized", 403: "unauthorized", 503: "metrics_off"}.get(err.code, "unreachable")
        raise FetchError(state) from err
    except (OSError, ValueError) as err:
        raise FetchError("unreachable") from err


# -- history math -------------------------------------------------------------
def merged_history(monitor: dict) -> list:
    """Archived samples older than the raw ring, then the raw ring."""
    raw = monitor.get("history") or []
    oldest = raw[0]["at_ms"] if raw else None
    archive = [s for s in monitor.get("history_archive") or [] if oldest is None or s["at_ms"] < oldest]
    return archive + raw


def valid_pair(a: dict, b: dict) -> bool:
    """Both samples belong to one uninterrupted counter run."""
    dt = b["at_ms"] - a["at_ms"]
    ids = (a.get("continuity_id"), b.get("continuity_id"))
    if dt <= 0:
        return False
    if all(isinstance(i, int) for i in ids):
        return ids[0] == ids[1]
    return dt <= 3 * SAMPLE_INTERVAL_MS


def counter_deltas(a: dict, b: dict, keys: tuple) -> dict | None:
    """Per-key increase between two samples; None if a counter is missing or went backwards."""
    deltas = {}
    for key in keys:
        before, after = a.get(key), b.get(key)
        if not isinstance(before, int) or not isinstance(after, int) or after < before:
            return None
        deltas[key] = after - before
    return deltas


def bucket_sums(history: list, keys: tuple, start_ms: int, end_ms: int, width: int) -> list:
    """Counter increases per time bucket over [start, end]; None where no valid interval landed."""
    sums: list = [None] * width
    span = (end_ms - start_ms) / width
    for a, b in zip(history, history[1:]):
        if a["at_ms"] < start_ms or b["at_ms"] > end_ms or not valid_pair(a, b):
            continue
        deltas = counter_deltas(a, b, keys)
        if deltas is None:
            continue
        index = min(int((b["at_ms"] - start_ms) / span), width - 1)
        total = sums[index] or dict.fromkeys(keys, 0)
        sums[index] = {key: total[key] + deltas[key] for key in keys}
    return sums


def ratio_series(sums: list, num: str, den: str, scale: float) -> list:
    return [s[num] * scale / s[den] if s and s[den] else None for s in sums]


def rate_series(history: list, pair: tuple, start_ms: int, end_ms: int, width: int) -> list:
    """tok/s over the time the server spent in that phase, None while it was idle."""
    return ratio_series(bucket_sums(history, pair, start_ms, end_ms, width), pair[0], pair[1], 1e9)


def hit_series(history: list, start_ms: int, end_ms: int, width: int) -> list:
    """Percent of prefix-cache lookups that hit."""
    return ratio_series(bucket_sums(history, CACHE, start_ms, end_ms, width), CACHE[0], CACHE[1], 100.0)


def gauge_series(history: list, key: str, start_ms: int, end_ms: int, width: int) -> list:
    span = (end_ms - start_ms) / width
    buckets: list = [[] for _ in range(width)]
    for sample in history:
        value = sample.get(key)
        if isinstance(value, (int, float)) and start_ms <= sample["at_ms"] <= end_ms:
            buckets[min(int((sample["at_ms"] - start_ms) / span), width - 1)].append(value)
    return [sum(b) / len(b) if b else None for b in buckets]


# -- formatting ---------------------------------------------------------------
def num(value, spec: str = "{:,.0f}") -> str:
    return "-" if value is None else spec.format(value)


def clock(ms) -> str:
    return datetime.fromtimestamp(ms / 1000).strftime("%H:%M:%S") if ms else "-"


def pct(part, whole) -> str:
    return "-" if not whole else f"{100 * part / whole:.1f}%"


def bar(fraction: float, width: int = 30) -> str:
    fraction = min(max(fraction, 0.0), 1.0)
    full = round(fraction * width)
    return "█" * full + "░" * (width - full)


def sparkline(values: list, width: int, top: float | None = None) -> str:
    """One block per bucket, scaled to `top` (default the series maximum); idle buckets are blank."""
    shown = values[-width:]
    top = top or max((v for v in shown if v is not None), default=0) or 1
    return "".join(" " if v is None else BLOCKS[min(8, round(v / top * 8))] for v in shown)


def client_of(row: dict) -> str:
    """Peer, User-Agent and cache key, each only when the server sent it."""
    parts = [row.get(key) for key in ("peer", "user_agent", "cache_key")]
    return " ".join(str(p)[:32] for p in parts if p) or "-"


# -- view model ---------------------------------------------------------------
def rate_between(history: list, pair: tuple, now_ms: int) -> float | None:
    start = now_ms - LIVE_RATE_S * 1000
    return rate_series(history, pair, start, now_ms, 1)[0]


def loaded_model(data: dict) -> dict:
    models = (data.get("monitor") or {}).get("models") or []
    ready = [m for m in models if m.get("state") == "ready"]
    session = (data.get("sessions") or [{}])[0]
    model = ready[0] if ready else {}
    return {"model": model.get("id") or session.get("model"),
            "context": session.get("context_length") or model.get("context_length")}


def live_view(data: dict, history: list, now_ms: int) -> dict:
    gauges = data.get("gauges") or {}
    expected = gauges.get("prefill_tokens_expected") or 0
    done = gauges.get("prefill_tokens_live") or 0
    return {
        **loaded_model(data),
        "gpu": gauges.get("gpu_utilization_pct"),
        "memory_mb": gauges.get("memory_mb"),
        "prefill_done": done,
        "prefill_expected": expected,
        "prefill_rate": rate_between(history, PREFILL, now_ms),
        "decode_rate": rate_between(history, DECODE, now_ms),
        "running": gauges.get("requests_running"),
        "waiting": gauges.get("requests_waiting"),
    }


def session_rows(data: dict) -> list:
    return [[s.get("phase", "-"), num(s.get("context_tokens")), num(s.get("cached_tokens")),
             num(s.get("generated_tokens")), num((s.get("state_bytes") or 0) / 1e6), client_of(s)]
            for s in data.get("sessions") or []]


def per_second(count, ms) -> float | None:
    return count * 1000 / ms if ms else None


def request_row(r: dict) -> list:
    prompt, cached = r.get("prompt_tokens") or 0, r.get("cached_tokens") or 0
    forwarded = max(prompt - cached, 0)
    outcome = r.get("outcome", "-") + (f" {r['error_code']}" if r.get("error_code") else "")
    return [clock(r.get("finished_at_ms")), f"{prompt:,}+{r.get('output_tokens') or 0:,}", num(cached),
            pct(cached, prompt), num(forwarded), num(r.get("prefill_ms")),
            num(per_second(forwarded, r.get("prefill_ms")), "{:,.1f}"),
            num(per_second(r.get("output_tokens") or 0, r.get("decode_ms")), "{:,.1f}"),
            outcome, client_of(r)]


def request_rows(monitor: dict) -> list:
    recent = monitor.get("recent_requests") or []
    return [request_row(r) for r in reversed(recent[-REQUEST_ROWS:])]


def spark_view(history: list, now_ms: int, window_key: str, width: int) -> dict:
    seconds, label = WINDOWS[window_key]
    start = now_ms - seconds * 1000
    hit = hit_series(history, start, now_ms, width)
    queries = bucket_sums(history, CACHE, start, now_ms, 1)[0]
    return {
        "label": label,
        "prefill": rate_series(history, PREFILL, start, now_ms, width),
        "decode": rate_series(history, DECODE, start, now_ms, width),
        "gpu": gauge_series(history, "gpu_pct", start, now_ms, width),
        "hit": hit,
        "hit_window": pct(queries[CACHE[0]], queries[CACHE[1]]) if queries else "-",
    }


def build_view(data: dict, window_key: str, width: int) -> dict:
    """Everything the screen shows, derived from one /metrics.json body."""
    monitor = data.get("monitor")
    history = merged_history(monitor) if monitor else []
    sampled = ((monitor or {}).get("server") or {}).get("sampled_at_ms")
    now_ms = sampled or (history[-1]["at_ms"] if history else 0)
    view = {"live": live_view(data, history, now_ms), "sessions": session_rows(data),
            "notice": None, "spark": None, "requests": []}
    if monitor is None:
        view["notice"] = NO_MONITOR
        return view
    view["spark"] = spark_view(history, now_ms, window_key, width)
    view["requests"] = request_rows(monitor)
    return view


# -- rendering ----------------------------------------------------------------
def header_text(live: dict) -> str:
    return (f"mlxtop  {live['model'] or '-'}  ctx limit {num(live['context'])}"
            f"  GPU {num(live['gpu'])}%  mem {num(live['memory_mb'])} MB")


def live_text(live: dict, spark: dict | None) -> str:
    expected, done = live["prefill_expected"], live["prefill_done"]
    progress = f"{bar(done / expected)} {done / expected:4.0%}  {done:,}/{expected:,}" if expected else "idle"
    hit = spark["hit_window"] + f" ({spark['label']})" if spark else "-"
    return (f"prefill {progress}\n"
            f"prefill {num(live['prefill_rate'])} tok/s   decode {num(live['decode_rate'])} tok/s   "
            f"running {num(live['running'])}  waiting {num(live['waiting'])}   cache hit {hit}")


def spark_text(spark: dict | None, width: int) -> str:
    if spark is None:
        return ""
    rows = [f"window {spark['label']}   [1] 5m  [2] 15m  [3] 1h"]
    for title, key in (("prefill tok/s", "prefill"), ("decode tok/s", "decode"), ("GPU %", "gpu")):
        peak = max((v for v in spark[key] if v is not None), default=None)
        rows.append(f"{title:>14} {sparkline(spark[key], width)} max {num(peak)}")
    rows.append(f"{'cache hit %':>14} {sparkline(spark['hit'], width, top=100)} window {spark['hit_window']}")
    return "\n".join(rows)


def banner_text(state: str | None, url: str) -> str:
    return BANNERS[state].format(url=url) if state else ""


class MlxTop(App):
    CSS = """
    #header, #live, #spark, #footer { height: auto; padding: 0 1; }
    #banner { height: auto; padding: 0 1; color: red; }
    #sessions { height: 6; }
    #requests { height: 1fr; }
    """
    BINDINGS = [("q", "quit", "Quit"), ("1", "window('1')", "5m"),
                ("2", "window('2')", "15m"), ("3", "window('3')", "1h")]

    def __init__(self, url: str, key: str | None, fetch=fetch_metrics, interval: float = DEFAULT_REFRESH_S):
        super().__init__()
        self.url, self.key, self.fetch, self.interval = url, key, fetch, interval
        self.window_key = "1"

    def compose(self) -> ComposeResult:
        for name in ("header", "banner", "live"):
            yield Static(id=name)
        yield DataTable(id="sessions")
        yield Static(id="spark")
        yield DataTable(id="requests")
        yield Static("q quit   1/2/3 window", id="footer")

    def on_mount(self) -> None:
        self.query_one("#sessions", DataTable).add_columns(
            "phase", "context", "cached", "generated", "state MB", "client")
        self.query_one("#requests", DataTable).add_columns(
            "time", "prompt+out", "cached", "hit", "forwarded", "prefill ms", "prefill/s",
            "decode/s", "outcome", "client")
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
        try:
            data = await asyncio.to_thread(self.fetch, self.url, self.key)
        except FetchError as err:
            view = build_view({}, self.window_key, 10)
            view["notice"] = banner_text(err.state, self.url)
        else:
            view = build_view(data, self.window_key, max(10, self.size.width - 34))
        self.draw(view)

    def draw(self, view: dict) -> None:
        self.show("banner", view["notice"] or "")
        self.show("header", header_text(view["live"]))
        self.show("live", live_text(view["live"], view["spark"]))
        self.show("spark", spark_text(view["spark"], max(10, self.size.width - 34)))
        self.fill("sessions", view["sessions"])
        self.fill("requests", view["requests"])


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
                        help=f"screen refresh period (default {DEFAULT_REFRESH_S:g}, minimum {MIN_REFRESH_S:g}); "
                             f"the server samples every {SAMPLE_INTERVAL_MS // 1000} s, so history only moves that often")
    args = parser.parse_args(argv)
    MlxTop(args.url, args.api_key or os.environ.get("MLX_SERVE_API_KEY"), interval=args.interval).run()


if __name__ == "__main__":
    main()
