"""Collector end to end: fake server, temp log, temp DB."""
import os
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest

import collect
import mlxcommon as mc
from tests.helpers import HERE, KEY, FakeServer, free_port

ROOT = os.path.dirname(HERE)
REQ = "  <- 27+3 tokens (520ms) [prefill: 55.9 tok/s, decode: 187.3 tok/s] [stop]\n"


class Clock:
    def __init__(self, t=1_000_000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, s=1.0):
        self.t += s


class CollectorCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.log = os.path.join(self.tmp.name, "mlx-serve.log")
        self.db = os.path.join(self.tmp.name, "sub", "mlxtop.db")
        self.server = FakeServer()
        self.addCleanup(self.server.close)
        self.clock = Clock()
        self.peers = []
        self.collector = self.make()

    def make(self, url=None, key=KEY):
        with open(self.log, "a"):
            pass
        return collect.Collector(
            db_path=self.db, url=url or self.server.url, log_path=self.log, key_fn=lambda: key,
            peers_fn=lambda: self.peers, clock=self.clock,
            resolver=collect.Resolver(lookup=lambda ip: "box.ts.net"))

    def tick(self, advance=1.0):
        self.clock.advance(advance)
        self.collector.tick()

    def rows(self, sql):
        with sqlite3.connect(self.db) as conn:
            conn.row_factory = sqlite3.Row
            return conn.execute(sql).fetchall()

    def append(self, text):
        with open(self.log, "a") as fh:
            fh.write(text)


class PollTests(CollectorCase):
    def test_sample_live_and_status_rows(self):
        self.tick()
        s = self.rows("SELECT * FROM samples")[0]
        self.assertEqual((s["up"], s["gpu"], s["running"]), (1, 99, 1))
        self.assertIsNone(s["prefill_rate"])  # first sample has no previous
        self.assertEqual(self.rows("SELECT state FROM status")[0]["state"], "ok")
        self.assertIn("sessions", self.rows("SELECT json FROM live")[0]["json"])

    def test_rates_from_counter_deltas(self):
        self.tick()
        self.server.bump(prefill=2000, gen=300)
        self.tick(2.0)
        s = self.rows("SELECT * FROM samples ORDER BY ts DESC")[0]
        self.assertEqual((s["prefill_rate"], s["decode_rate"]), (1000.0, 150.0))

    def test_completion_does_not_double_count_the_prefill_it_already_showed(self):
        # Live climbs to the forwarded length, falls to 0 when prefill ends, and the total
        # only catches up when the request completes.
        self.server.restart()
        self.tick()
        for live in (4000, 8000, 12000, 0, 0):
            self.server.payload["gauges"]["prefill_tokens_live"] = live
            self.tick()
        self.server.payload["counters"]["prefill_tokens_total"] = 12000
        self.tick()
        rates = [r["prefill_rate"] for r in self.rows("SELECT * FROM samples ORDER BY ts")]
        self.assertEqual(rates, [None, 4000.0, 4000.0, 4000.0, 0.0, 0.0, 0.0])

    def test_cache_restores_are_not_prefill_and_feed_the_hit_rate(self):
        self.server.restart()
        self.tick()
        c = self.server.payload["counters"]
        c.update(prompt_tokens_total=435897, prefill_tokens_total=278, prefix_cache_tokens_total=435619)
        self.tick()
        s = self.rows("SELECT * FROM samples ORDER BY ts DESC")[0]
        self.assertEqual((s["prefill_rate"], s["prompt_tok"], s["cached_tok"]), (278.0, 435897, 435619))

    def test_server_restart_drops_the_rate_not_negative(self):
        self.tick()
        self.server.bump(prefill=5000, gen=500)
        self.tick()
        self.server.restart()
        self.tick()
        self.server.bump(prefill=100, gen=10)
        self.tick()
        rates = [(r["prefill_rate"], r["decode_rate"]) for r in self.rows("SELECT * FROM samples ORDER BY ts")]
        self.assertEqual(rates, [(None, None), (5000.0, 500.0), (None, None), (100.0, 10.0)])

    def test_404_then_401_then_down_then_recovery(self):
        self.server.mode = "404"
        self.tick()
        self.assertEqual(self.rows("SELECT state FROM status")[0]["state"], "no_metrics")
        self.server.mode = "401"
        self.tick()
        self.assertEqual(self.rows("SELECT state FROM status")[0]["state"], "bad_key")
        self.server.mode = "ok"
        self.tick()
        self.assertEqual(self.rows("SELECT state FROM status")[0]["state"], "ok")
        self.collector.url = f"http://127.0.0.1:{free_port()}/metrics.json"
        self.tick()
        st = self.rows("SELECT * FROM status")[0]
        self.assertEqual((st["state"], st["down_since"]), ("down", self.clock.t))
        ups = [r["up"] for r in self.rows("SELECT up FROM samples ORDER BY ts")]
        self.assertEqual(ups, [0, 0, 1, 0])
        self.collector.url = self.server.url
        self.tick()
        self.tick()
        last = self.rows("SELECT * FROM samples ORDER BY ts DESC")[0]
        self.assertEqual(last["up"], 1)
        self.assertIsNotNone(self.rows("SELECT * FROM status")[0]["ts"])
        self.assertIsNone(self.rows("SELECT down_since FROM status")[0]["down_since"])

    def test_down_since_stays_at_first_failure(self):
        self.collector.url = f"http://127.0.0.1:{free_port()}/metrics.json"
        self.tick()
        first = self.clock.t
        self.tick()
        self.tick()
        self.assertEqual(self.rows("SELECT down_since FROM status")[0]["down_since"], first)

    def test_missing_key_records_no_key(self):
        self.collector = self.make(key=None)
        self.tick()
        self.assertEqual(self.rows("SELECT state FROM status")[0]["state"], "no_key")

    def test_wrong_key_is_rejected_state(self):
        self.collector = self.make(key="wrong")
        self.tick()
        self.assertEqual(self.rows("SELECT state FROM status")[0]["state"], "bad_key")


class LogTests(CollectorCase):
    def requests(self):
        return self.rows("SELECT * FROM requests ORDER BY rowid")

    def test_existing_history_is_not_replayed_and_new_lines_land(self):
        self.append(REQ * 5)
        self.collector = self.make()
        self.tick()
        self.assertEqual(len(self.requests()), 0)
        self.append(REQ + "  pld=enabled\n" + REQ)
        self.tick()
        self.assertEqual(len(self.requests()), 2)
        self.assertEqual(self.requests()[0]["ms"], 520)

    def test_partial_line_waits_for_newline(self):
        self.tick()
        self.append(REQ[:30])
        self.tick()
        self.assertEqual(len(self.requests()), 0)
        self.append(REQ[30:])
        self.tick()
        self.assertEqual(len(self.requests()), 1)

    def test_unknown_request_form_counts_as_unparsed(self):
        self.tick()
        self.append("  <- something new [weird]\n" + REQ + "  <- and another\n")
        self.tick()
        self.assertEqual(len(self.requests()), 1)
        self.assertEqual(self.rows("SELECT unparsed FROM status")[0]["unparsed"], 2)

    def test_truncation_restarts_at_the_top(self):
        self.tick()
        self.append(REQ * 3)
        self.tick()
        with open(self.log, "w") as fh:
            fh.write(REQ)
        self.tick()
        self.assertEqual(len(self.requests()), 4)

    def test_rotation_by_inode_is_followed(self):
        self.tick()
        self.append(REQ)
        self.tick()
        os.rename(self.log, self.log + ".1")
        self.append(REQ * 2)
        self.tick()
        self.assertEqual(len(self.requests()), 3)
        self.assertEqual(self.rows("SELECT unparsed FROM status")[0]["unparsed"], 0)

    def test_missing_log_flags_status_and_recovers(self):
        self.tick()
        os.remove(self.log)
        self.tick()
        st = self.rows("SELECT * FROM status")[0]
        self.assertEqual((st["log_ok"], st["log_path"], st["state"]), (0, self.log, "ok"))
        self.assertEqual(self.rows("SELECT up FROM samples ORDER BY ts DESC")[0]["up"], 1)  # live unaffected
        self.append("")
        self.tick()
        self.append(REQ)
        self.tick()
        self.assertEqual(self.rows("SELECT log_ok FROM status")[0]["log_ok"], 1)
        self.assertEqual(len(self.requests()), 1)


class PeerTests(CollectorCase):
    def test_peers_written_with_resolved_names(self):
        self.peers = [("100.64.0.5", 50000), ("192.168.1.9", 50001)]
        self.tick()
        rows = self.rows("SELECT * FROM peers ORDER BY port")
        self.assertEqual([r["ip"] for r in rows], ["100.64.0.5", "192.168.1.9"])
        deadline = time.time() + 2
        while time.time() < deadline and self.collector.resolver.name("100.64.0.5") != "box.ts.net":
            time.sleep(0.02)
        self.tick(3.0)
        self.assertEqual(self.rows("SELECT host FROM peers")[0]["host"], "box.ts.net")

    def test_peers_refresh_only_every_two_seconds(self):
        self.peers = [("1.1.1.1", 1)]
        self.tick()
        self.peers = []
        self.tick(0.5)
        self.assertEqual(len(self.rows("SELECT * FROM peers")), 1)
        self.tick(2.0)
        self.assertEqual(len(self.rows("SELECT * FROM peers")), 0)

    def test_lsof_missing_marks_peers_unavailable(self):
        def broken():
            raise FileNotFoundError("lsof")
        self.collector.peers_fn = broken
        self.tick()
        self.assertEqual(self.rows("SELECT peers_ok FROM status")[0]["peers_ok"], 0)
        self.assertEqual(self.rows("SELECT up FROM samples")[0]["up"], 1)

    def test_lsof_output_parser(self):
        line = "mlx-serve 6012 u 7u IPv4 0xe6 0t0 TCP 192.168.86.184:11234->192.168.86.67:49882 (ESTABLISHED)"
        class Out:
            stdout = line + "\nmlx-serve 6012 u 9u IPv4 0x1 0t0 TCP *:11234 (LISTEN)\n"
        orig = collect.subprocess.run
        collect.subprocess.run = lambda *a, **k: Out()
        self.addCleanup(setattr, collect.subprocess, "run", orig)
        self.assertEqual(collect.lsof_peers(11234), [("192.168.86.67", 49882)])


class ResolverTests(unittest.TestCase):
    def test_no_ptr_shows_bare_ip(self):
        def nope(ip):
            raise OSError("no PTR")
        r = collect.Resolver(lookup=nope)
        r.name("9.9.9.9")
        time.sleep(0.1)
        self.assertEqual(r.name("9.9.9.9"), "9.9.9.9")

    def test_slow_lookup_gives_up_after_budget_and_stays_ip(self):
        r = collect.Resolver(lookup=lambda ip: (time.sleep(1.2), "late.example")[1])
        self.assertEqual(r.name("8.8.8.8"), "8.8.8.8")
        time.sleep(collect.Resolver.BUDGET_S + 0.15)
        self.assertEqual(r.name("8.8.8.8"), "8.8.8.8")
        time.sleep(1.2)
        self.assertEqual(r.name("8.8.8.8"), "8.8.8.8")  # settled as the IP, never flips

    def test_fast_lookup_is_cached(self):
        calls = []
        r = collect.Resolver(lookup=lambda ip: calls.append(ip) or "h.ts.net")
        r.name("1.2.3.4")
        time.sleep(0.1)
        self.assertEqual((r.name("1.2.3.4"), r.name("1.2.3.4")), ("h.ts.net", "h.ts.net"))
        self.assertEqual(len(calls), 1)


class DbTests(CollectorCase):
    def test_prune_removes_rows_older_than_seven_days(self):
        old = self.clock.t - mc.RETENTION_S - 10
        self.collector.conn.execute("INSERT INTO samples (ts, up) VALUES (?, 1)", (old,))
        self.collector.conn.execute("INSERT INTO requests VALUES (?, 1, 1, 0, 1, 1, 'stop', NULL)", (old,))
        self.collector.conn.commit()
        self.tick()
        self.assertEqual(len(self.rows("SELECT * FROM samples WHERE ts < %f" % (old + 1))), 0)
        self.assertEqual(len(self.rows("SELECT * FROM requests")), 0)

    def test_database_from_before_the_token_columns_is_upgraded(self):
        path = os.path.join(self.tmp.name, "old.db")
        with sqlite3.connect(path) as conn:
            conn.execute("CREATE TABLE samples (ts REAL PRIMARY KEY, up INTEGER, prefill_rate REAL, "
                         "decode_rate REAL, gpu REAL, mem_mb REAL, running INTEGER, waiting INTEGER)")
        conn = mc.connect(path)
        cols = [row[1] for row in conn.execute("PRAGMA table_info(samples)")]
        conn.close()
        self.assertEqual(cols[-2:], ["prompt_tok", "cached_tok"])

    def test_wal_mode(self):
        self.assertEqual(self.rows("PRAGMA journal_mode")[0][0], "wal")

    def test_crash_mid_tick_rolls_back_the_whole_tick(self):
        def boom():
            raise RuntimeError("killed")
        self.collector.peers_fn = boom
        with self.assertRaises(RuntimeError):
            self.tick()
        self.assertEqual(len(self.rows("SELECT * FROM samples")), 0)
        self.collector.peers_fn = lambda: []
        self.tick()
        self.assertEqual(len(self.rows("SELECT * FROM samples")), 1)

    def test_reader_is_not_blocked_by_an_open_write_transaction(self):
        self.tick()
        self.collector.conn.execute("BEGIN IMMEDIATE")
        self.collector.conn.execute("INSERT INTO samples (ts, up) VALUES (1, 1)")
        ro = mc.connect_readonly(self.db)
        self.addCleanup(ro.close)
        self.assertEqual(ro.execute("SELECT COUNT(*) FROM samples").fetchone()[0], 1)
        self.collector.conn.rollback()


class LockTests(unittest.TestCase):
    def test_second_lock_holder_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "x", "mlxtop.lock")
            first = mc.acquire_lock(path)
            self.assertIsNotNone(first)
            self.assertIsNone(mc.acquire_lock(path))
            first.close()
            self.assertIsNotNone(mc.acquire_lock(path))

    def test_second_collector_process_exits_with_message(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "mlxtop.db")
            held = mc.acquire_lock(db + ".lock")
            self.assertIsNotNone(held)
            env = dict(os.environ, HOME=tmp)
            out = subprocess.run([sys.executable, os.path.join(ROOT, "collect.py"), "--db", db], env=env,
                                 capture_output=True, text=True, timeout=20, cwd=ROOT)
            self.assertEqual(out.returncode, 1)
            self.assertIn("already running", out.stderr)
            held.close()
