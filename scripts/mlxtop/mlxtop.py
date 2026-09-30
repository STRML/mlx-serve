#!/usr/bin/env python3
"""mlxtop: live Textual dashboard for mlx-serve, reading the collector's SQLite history."""
from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import time
from datetime import datetime

from rich.text import Text
from textual.app import App, ComposeResult
from textual.widgets import DataTable, Static

import mlxcommon as mc

WINDOWS = {"1": (900, "15 min"), "2": (3600, "1 h"), "3": (86400, "24 h")}
STALE_AFTER_S = 5
RATE_AVG_S = 5
SMALL_HEIGHT = 22
BLOCKS = " ▁▂▃▄▅▆▇█"
SERIES = (("prefill tok/s", "prefill_rate"), ("decode tok/s", "decode_rate"), ("GPU %", "gpu"))
HIT_TITLE = "cache hit %"
STATE_BANNER = {
    "down": "mlx-serve down since {since}",
    "no_metrics": "start mlx-serve with --metrics",
    "no_key": "no API key",
    "bad_key": "API key rejected",
}
OPTIONAL_IDS = ("sessions", "peers", "spark", "requests")


class Unreadable(Exception):
    """The DB is missing or corrupt."""


class Busy(Exception):
    """The DB is locked; retry next tick."""


# -- pure formatting ---------------------------------------------------------
def hms(ts) -> str:
    return datetime.fromtimestamp(ts).strftime("%H:%M:%S") if ts else "-"


def num(value, fmt="{:,.0f}") -> str:
    return "-" if value is None else fmt.format(value)


def bar(frac: float, width: int = 30) -> str:
    frac = min(max(frac, 0.0), 1.0)
    full = round(frac * width)
    return "█" * full + "░" * (width - full)


def pct(part, whole) -> str:
    return "-" if not whole else f"{100 * part / whole:.1f}%"


def sparkline(values: list, width: int, top=None) -> str:
    """One block char per bucket, scaled to `top` (default the series maximum); None buckets are blank."""
    known = [v for v in values if v is not None]
    top = top or max(known, default=0) or 1
    return "".join(" " if v is None else BLOCKS[min(8, round(v / top * 8))] for v in values[-width:])


def client_of(session: dict) -> str:
    parts = [session.get(k) for k in ("peer", "user_agent", "cache_key")]
    return " ".join(str(p)[:24] for p in parts if p)


def banners(status, sample_ts, now: float, have_db: bool, fallback_state) -> list:
    """Warning lines for the header, in priority order."""
    if not have_db:
        state = STATE_BANNER.get(fallback_state or "", "")
        return ["no history (collector not running?)"] + ([state.format(since="?")] if state else [])
    lines = []
    age = now - (status["ts"] or 0)
    if age > STALE_AFTER_S:
        lines.append(f"collector stale: last sample {age:.0f} s ago")
    if status["state"] in STATE_BANNER:
        lines.append(STATE_BANNER[status["state"]].format(since=hms(status["down_since"])))
    if not status["log_ok"]:
        lines.append(f"no log at {status['log_path']}")
    return lines


def header_text(live: dict, sample, lines: list) -> Text:
    sessions = live.get("sessions") or [{}]
    g = live.get("gauges", {})
    title = (f"mlxtop  {sessions[0].get('model', '-')}  ctx limit {num(sessions[0].get('context_length'))}"
             f"  GPU {num(g.get('gpu_utilization_pct'))}%  mem {num(g.get('memory_mb'))} MB")
    text = Text(title, style="bold")
    for line in lines:
        text.append("\n" + line, style="bold red")
    return text


def live_text(live: dict, sample, hit: str, label: str) -> Text:
    g = live.get("gauges", {})
    done, expected = g.get("prefill_tokens_live", 0), g.get("prefill_tokens_expected", 0)
    frac = done / expected if expected else 0
    prefill = f"{bar(frac)} {frac:4.0%}  {done:,}/{expected:,}" if expected else "idle"
    prate = sample["prefill_rate"] if sample else None
    drate = sample["decode_rate"] if sample else None
    return Text(
        f"prefill {prefill}\n"
        f"prefill {num(prate)} tok/s   decode {num(drate)} tok/s   "
        f"running {num(g.get('requests_running'))}  waiting {num(g.get('requests_waiting'))}   "
        f"cache hit {hit} ({label})"
    )


def peers_text(peers: list, ok: bool) -> Text:
    if not ok:
        return Text("peers unavailable")
    if not peers:
        return Text("peers: none connected")
    counts: dict = {}
    for p in peers:
        label = p["host"] if p["host"] == p["ip"] else f"{p['host']} ({p['ip']})"
        counts[label] = counts.get(label, 0) + 1
    return Text("peers: " + "   ".join(f"{k} x{n}" for k, n in counts.items()))


def spark_text(series: dict, label: str, width: int, hit: str) -> Text:
    rows = [f"window {label}   [1] 15 min  [2] 1 h  [3] 24 h"]
    for title, col in SERIES:
        values = series[col]
        known = [v for v in values if v is not None]
        rows.append(f"{title:>14} {sparkline(values, width)} max {num(max(known, default=None))}")
    rows.append(f"{HIT_TITLE:>14} {sparkline(series['hit'], width, top=100)} window {hit}")
    return Text("\n".join(rows))


# -- data access -------------------------------------------------------------
def bucket_series(conn, now: float, window_s: int, width: int) -> dict:
    """Per-bucket averages for the sparklines, plus the token-weighted cache hit rate and its totals."""
    bucket = window_s / width
    grouped = conn.execute(
        "SELECT CAST((? - ts) / ? AS INTEGER) AS k, AVG(prefill_rate) p, AVG(decode_rate) d, AVG(gpu) g, "
        "SUM(prompt_tok) pt, SUM(cached_tok) ct "
        "FROM samples WHERE ts >= ? AND up = 1 GROUP BY k", (now, bucket, now - window_s)).fetchall()
    series = {col: [None] * width for _, col in SERIES}
    series["hit"] = [None] * width
    prompt_sum = cached_sum = 0
    for row in grouped:
        idx = width - 1 - min(int(row["k"]), width - 1)
        for col, key in (("prefill_rate", "p"), ("decode_rate", "d"), ("gpu", "g")):
            series[col][idx] = row[key]
        if row["pt"]:
            series["hit"][idx] = 100 * (row["ct"] or 0) / row["pt"]
            prompt_sum += row["pt"]
            cached_sum += row["ct"] or 0
    return {"series": series, "hit_tokens": (cached_sum, prompt_sum)}


def query_all(conn, now: float, window_s: int, width: int) -> dict:
    live_row = conn.execute("SELECT json FROM live WHERE id = 1").fetchone()
    return {
        **bucket_series(conn, now, window_s, width),
        "live": json.loads(live_row["json"]) if live_row else {},
        "sample": conn.execute(  # 5 s average: the server updates its token gauges in bursts
            "SELECT AVG(prefill_rate) AS prefill_rate, AVG(decode_rate) AS decode_rate "
            "FROM samples WHERE up = 1 AND ts >= ?", (now - RATE_AVG_S,)).fetchone(),
        "status": conn.execute("SELECT * FROM status WHERE id = 1").fetchone(),
        "peers": conn.execute("SELECT * FROM peers ORDER BY host, port").fetchall(),
        "requests": conn.execute("SELECT * FROM requests ORDER BY ts DESC LIMIT 100").fetchall(),
    }


def read_snapshot(db_path: str, now: float, window_s: int, width: int) -> dict:
    """Read everything the screen needs; raises Unreadable (missing/corrupt) or Busy (locked)."""
    try:
        conn = mc.connect_readonly(db_path)
        try:
            return query_all(conn, now, window_s, width)
        finally:
            conn.close()
    except sqlite3.OperationalError as err:
        raise (Busy() if "locked" in str(err) else Unreadable(str(err))) from err
    except (sqlite3.DatabaseError, ValueError) as err:
        raise Unreadable(str(err)) from err


# -- app ---------------------------------------------------------------------
class MlxTop(App):
    CSS = """
    #header { height: auto; padding: 0 1; }
    #live { height: auto; padding: 0 1; }
    #sessions { height: 6; }
    #peers { height: auto; padding: 0 1; }
    #spark { height: 6; padding: 0 1; }
    #requests { height: 1fr; }
    #footer { height: 1; padding: 0 1; color: $text-muted; }
    """
    BINDINGS = [("q", "quit", "Quit"), ("1", "window('1')", "15 min"),
                ("2", "window('2')", "1 h"), ("3", "window('3')", "24 h")]

    def __init__(self, db_path, url, key_fn, clock=time.time):
        """`url` is the full /metrics.json URL."""
        super().__init__()
        self.db_path, self.url, self.key_fn, self.clock = db_path, url, key_fn, clock
        self.window = "1"
        self.rendered: dict = {}

    def compose(self) -> ComposeResult:
        yield Static(id="header")
        yield Static(id="live")
        yield DataTable(id="sessions")
        yield Static(id="peers")
        yield Static(id="spark")
        yield DataTable(id="requests")
        yield Static(id="footer")

    def on_mount(self) -> None:
        self.query_one("#sessions", DataTable).add_columns("phase", "context", "cached", "generated", "state MB", "client")
        self.query_one("#requests", DataTable).add_columns(
            "time", "prompt+gen", "cached", "hit", "forwarded", "prefill/s", "decode/s", "finish")
        self.set_interval(1.0, self.refresh_view)
        self.call_after_refresh(self.refresh_view)

    def on_resize(self) -> None:
        small = self.size.height < SMALL_HEIGHT
        for wid in OPTIONAL_IDS:
            self.query_one(f"#{wid}").display = not small

    def action_window(self, key: str) -> None:
        self.window = key
        self.call_after_refresh(self.refresh_view)

    def show(self, wid: str, text: Text) -> None:
        self.rendered[wid] = text
        self.query_one(f"#{wid}", Static).update(text)

    async def refresh_view(self) -> None:
        now = self.clock()
        window_s, label = WINDOWS[self.window]
        width = max(10, self.size.width - 34)
        try:
            snap = read_snapshot(self.db_path, now, window_s, width)
        except Busy:
            return
        except Unreadable:
            snap = await self.fallback_snapshot()
        self.draw(snap, now, label, width)

    async def fallback_snapshot(self) -> dict:
        """No history: read /metrics.json directly so the live panel still works."""
        snap = {"series": dict.fromkeys(("prefill_rate", "decode_rate", "gpu", "hit"), []), "hit_tokens": (0, 0), "live": {}, "sample": None, "status": None,
                "peers": [], "requests": [], "state": None}
        try:
            snap["live"] = await asyncio.to_thread(mc.fetch_metrics, self.url, self.key_fn(), 1.0)
        except mc.MetricsError as err:
            snap["state"] = err.state
        return snap

    def draw(self, snap: dict, now: float, label: str, width: int) -> None:
        status = snap["status"]
        lines = banners(status, snap["sample"], now, status is not None, snap.get("state"))
        live = snap["live"]
        hit = pct(*snap["hit_tokens"])
        self.show("header", header_text(live, snap["sample"], lines))
        self.show("live", live_text(live, snap["sample"], hit, label))
        self.show("peers", peers_text(snap["peers"], status is None or bool(status["peers_ok"])))
        self.show("spark", spark_text(snap["series"], label, width, hit))
        unparsed = status["unparsed"] if status else 0
        self.show("footer", Text(f"q quit  1/2/3 window  unparsed: {unparsed}"))
        self.fill_sessions(live.get("sessions") or [])
        self.fill_requests(snap["requests"])

    def fill_sessions(self, sessions: list) -> None:
        table = self.query_one("#sessions", DataTable)
        table.clear()
        for s in sessions:
            table.add_row(s.get("phase", "-"), num(s.get("context_tokens")), num(s.get("cached_tokens")),
                          num(s.get("generated_tokens")), num((s.get("state_bytes") or 0) / 1e6), client_of(s))

    def fill_requests(self, requests: list) -> None:
        table = self.query_one("#requests", DataTable)
        table.clear()
        for r in requests:
            table.add_row(hms(r["ts"]), f"{r['prompt']:,}+{r['gen']:,}", num(r["cached"]),
                          pct(r["cached"], r["prompt"]), num(r["prompt"] - r["cached"]),
                          num(r["prefill_rate"], "{:,.1f}"), num(r["decode_rate"], "{:,.1f}"), r["finish"])


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="mlxtop: live dashboard for mlx-serve.")
    mc.add_common_args(parser)
    args = parser.parse_args(argv)
    MlxTop(args.db, mc.metrics_url(args.url), mc.key_source(args)).run()


if __name__ == "__main__":
    main()
