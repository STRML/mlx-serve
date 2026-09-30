#!/usr/bin/env python3
"""mlxtop collector: polls /metrics.json, tails the mlx-serve log, lists TCP peers, writes SQLite."""
from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import mlxcommon as mc

PEER_RE = re.compile(r"TCP \S+:(\d+)->(\S+):(\d+) \(ESTABLISHED\)")
PEER_EVERY_S = 2.0
PRUNE_EVERY_S = 3600.0


def lsof_peers(port: int) -> list:
    """Remote (ip, port) pairs of established connections to our port; raises OSError if lsof fails."""
    out = subprocess.run(
        ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:ESTABLISHED"],
        capture_output=True, text=True, timeout=3,
    ).stdout
    found = []
    for line in out.splitlines():
        m = PEER_RE.search(line)
        if m and int(m.group(1)) == port:
            found.append((m.group(2).strip("[]"), int(m.group(3))))
    return found


class Resolver:
    """Non-blocking cached reverse DNS: the bare IP until a lookup lands; one that takes >0.5 s stays the IP."""

    BUDGET_S = 0.5

    def __init__(self, lookup=None):
        self.lookup = lookup or (lambda ip: socket.gethostbyaddr(ip)[0])
        self.cache: dict = {}  # ip -> (future or None, submitted_at, final_name or None)
        self.pool = ThreadPoolExecutor(max_workers=4)

    def name(self, ip: str) -> str:
        if ip not in self.cache:
            self.cache[ip] = [self.pool.submit(self._safe, ip), time.monotonic(), None]
        entry = self.cache[ip]
        if entry[2] is None:
            entry[2] = self._settle(ip, entry)
        return entry[2] or ip

    def _settle(self, ip: str, entry: list):
        fut, started, _ = entry
        if fut.done():
            return fut.result()
        if time.monotonic() - started > self.BUDGET_S:
            return ip  # too slow: give up on this IP for good
        return None

    def _safe(self, ip: str) -> str:
        try:
            return self.lookup(ip) or ip
        except Exception:  # no PTR or resolver error: show the IP
            return ip


class LogTail:
    """Follow a log by (inode, offset): starts at the end, restarts at 0 on rotation or truncation."""

    def __init__(self, path: str):
        self.path = path
        self.inode = None
        self.offset = 0
        self.buf = b""

    def read_lines(self) -> list:
        """New complete lines; raises FileNotFoundError when the log is missing."""
        st = os.stat(self.path)
        if self.inode is None:
            self.inode, self.offset = st.st_ino, st.st_size
            return []
        if st.st_ino != self.inode or st.st_size < self.offset:
            self.inode, self.offset, self.buf = st.st_ino, 0, b""
        with open(self.path, "rb") as fh:
            fh.seek(self.offset)
            data = fh.read()
        self.offset += len(data)
        *lines, self.buf = (self.buf + data).split(b"\n")
        return [ln.decode("utf-8", "replace") for ln in lines]

    def reset(self) -> None:
        self.inode, self.buf = None, b""


class Collector:
    def __init__(self, db_path, url, log_path, key_fn, peers_fn, clock=time.time, resolver=None):
        """`url` is the full /metrics.json URL."""
        self.conn = mc.connect(db_path)
        self.url, self.key_fn, self.peers_fn, self.clock = url, key_fn, peers_fn, clock
        self.key = key_fn()
        self.tail = LogTail(log_path)
        self.log_path = log_path
        self.resolver = resolver or Resolver()
        self.meter = mc.PrefillMeter()
        self.prev = None  # (ts, generated_live, counters) of the previous good poll
        self.state, self.down_since = "ok", None
        self.unparsed, self.log_ok, self.peers_ok = 0, 1, 1
        self.last_peers = self.last_prune = -1e18

    # -- one tick ---------------------------------------------------------
    def tick(self) -> None:
        now = self.clock()
        with self.conn:
            self.poll(now)
            self.follow_log(now)
            self.refresh_peers(now)
            self.prune(now)
            self.write_status(now)

    def poll(self, now: float) -> None:
        try:
            metrics = mc.fetch_metrics(self.url, self.key)
        except mc.MetricsError as err:
            self.record_failure(now, err.state)
            return
        self.record_success(now, metrics)

    def record_failure(self, now: float, state: str) -> None:
        if state in ("no_key", "bad_key"):
            self.key = self.key_fn()  # the plist may have been fixed
        if self.down_since is None:
            self.down_since = now
        self.state, self.prev = state, None
        self.meter.reset()
        self.conn.execute("INSERT OR REPLACE INTO samples (ts, up) VALUES (?, 0)", (now,))

    def record_success(self, now: float, metrics: dict) -> None:
        self.state, self.down_since = "ok", None
        g, counters = metrics.get("gauges", {}), metrics.get("counters", {})
        forwarded = self.meter.update(metrics)
        gen = g.get("generation_tokens_live", 0)
        prate = drate = prompt = cached = None
        if self.prev:
            dt = now - self.prev[0]
            prate = None if forwarded is None else forwarded / dt
            drate = mc.rate(self.prev[1], gen, dt)
            prompt = mc.delta(self.prev[2].get("prompt_tokens_total"), counters.get("prompt_tokens_total"))
            cached = mc.delta(self.prev[2].get("prefix_cache_tokens_total"), counters.get("prefix_cache_tokens_total"))
        self.prev = (now, gen, counters)
        self.conn.execute(
            "INSERT OR REPLACE INTO samples VALUES (?, 1, ?, ?, ?, ?, ?, ?, ?, ?)",
            (now, prate, drate, g.get("gpu_utilization_pct"), g.get("memory_mb"),
             g.get("requests_running"), g.get("requests_waiting"), prompt, cached),
        )
        live = {k: metrics.get(k, {} if k != "sessions" else []) for k in ("counters", "gauges", "sessions")}
        self.conn.execute("INSERT OR REPLACE INTO live VALUES (1, ?, ?)", (now, json.dumps(live)))

    # -- log --------------------------------------------------------------
    def follow_log(self, now: float) -> None:
        try:
            lines = self.tail.read_lines()
        except OSError:
            self.log_ok = 0
            self.tail.reset()
            return
        self.log_ok = 1
        for line in lines:
            self.record_line(now, line)

    def record_line(self, now: float, line: str) -> None:
        if not mc.is_request_line(line):
            return
        req = mc.parse_request(line)
        if req is None:
            self.unparsed += 1
            return
        self.conn.execute(
            "INSERT INTO requests VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (now, req["prompt"], req["gen"], req["cached"], req["prefill_rate"],
             req["decode_rate"], req["finish"], req["ms"]),
        )

    # -- peers ------------------------------------------------------------
    def refresh_peers(self, now: float) -> None:
        if now - self.last_peers < PEER_EVERY_S:
            return
        self.last_peers = now
        try:
            found = self.peers_fn()
        except (OSError, subprocess.SubprocessError):
            self.peers_ok = 0
            self.conn.execute("DELETE FROM peers")
            return
        self.peers_ok = 1
        self.conn.execute("DELETE FROM peers")
        rows = [(ip, port, self.resolver.name(ip), now) for ip, port in found]
        self.conn.executemany("INSERT INTO peers VALUES (?, ?, ?, ?)", rows)

    # -- housekeeping -----------------------------------------------------
    def prune(self, now: float) -> None:
        if now - self.last_prune < PRUNE_EVERY_S:
            return
        self.last_prune = now
        cutoff = now - mc.RETENTION_S
        self.conn.execute("DELETE FROM samples WHERE ts < ?", (cutoff,))
        self.conn.execute("DELETE FROM requests WHERE ts < ?", (cutoff,))

    def write_status(self, now: float) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO status VALUES (1, ?, ?, ?, ?, ?, ?, ?)",
            (now, self.state, self.down_since, self.log_path, self.log_ok, self.unparsed, self.peers_ok),
        )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="mlxtop collector: samples mlx-serve into SQLite.")
    mc.add_common_args(parser)
    parser.add_argument("--log", help="mlx-serve log to tail (default: the server's per-port log, else "
                                      "~/Library/Logs/mlx-serve.log)")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    lock = mc.acquire_lock(args.db + ".lock")
    if lock is None:
        print("mlxtop collector already running (lock held); exiting", file=sys.stderr)
        return 1
    port = mc.port_of(args.url)
    collector = Collector(args.db, mc.metrics_url(args.url), args.log or mc.default_log(port),
                          mc.key_source(args), lambda: lsof_peers(port))
    while True:
        start = time.time()
        try:
            collector.tick()
        except Exception as err:  # keep the collector alive through transient DB or IO trouble
            print(f"tick failed: {err!r}", file=sys.stderr, flush=True)
        time.sleep(max(0.0, 1.0 - (time.time() - start)))


if __name__ == "__main__":
    sys.exit(main())
