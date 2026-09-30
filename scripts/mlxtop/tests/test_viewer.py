"""Viewer against a seeded DB via Textual's run_test."""
import json
import os
import sqlite3
import tempfile
import unittest

import mlxcommon as mc
import mlxtop
from tests.helpers import KEY, FakeServer, load_metrics

NOW = 2_000_000.0
SIZE = (140, 45)


def seed(path, status=None, sessions_extra=None, samples_at=(NOW - 2,), peers=(("100.64.1.2", 5, "laptop.example"),),
         tokens=(1000, 900)):
    conn = mc.connect(path)
    live = load_metrics()
    if sessions_extra:
        live["sessions"][0].update(sessions_extra)
    live["gauges"].update(prefill_tokens_live=207362, prefill_tokens_expected=414724)
    with conn:
        for ts in samples_at:
            conn.execute("INSERT INTO samples VALUES (?, 1, 4200.0, 120.5, 88, 112000, 1, 0, ?, ?)", (ts, *tokens))
        conn.execute("INSERT INTO live VALUES (1, ?, ?)", (NOW - 1, json.dumps(live)))
        for ip, port, host in peers:
            conn.execute("INSERT INTO peers VALUES (?, ?, ?, ?)", (ip, port, host, NOW))
        conn.execute("INSERT INTO requests VALUES (?, 396482, 561, 396451, 1378.6, 126.8, 'tool_calls', NULL)", (NOW - 30,))
        conn.execute("INSERT INTO requests VALUES (?, 27, 3, 0, 55.9, 187.3, 'stop', 520)", (NOW - 10,))
        s = dict(ts=NOW - 1, state="ok", down_since=None, log_path="/x/log", log_ok=1, unparsed=3, peers_ok=1)
        s.update(status or {})
        conn.execute("INSERT INTO status VALUES (1, :ts, :state, :down_since, :log_path, :log_ok, :unparsed, :peers_ok)", s)
    conn.close()


class Snap:
    """Everything the tests read from a finished app run (widgets are gone after exit)."""

    def __init__(self, app):
        self.text = {k: str(v) for k, v in app.rendered.items()}
        self.sessions = self.rows(app.query_one("#sessions"))
        self.requests = self.rows(app.query_one("#requests"))
        self.display = {w: app.query_one(f"#{w}").display for w in ("header", "live") + mlxtop.OPTIONAL_IDS}

    @staticmethod
    def rows(table):
        return [table.get_row_at(i) for i in range(table.row_count)]


class ViewerCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db = os.path.join(self.tmp.name, "mlxtop.db")

    def app(self, url="http://127.0.0.1:1/metrics.json"):
        return mlxtop.MlxTop(self.db, url, lambda: KEY, clock=lambda: NOW)

    async def run_app(self, app, size=SIZE, keys=()):
        async with app.run_test(size=size) as pilot:
            await pilot.pause(0.3)
            for key in keys:
                await pilot.press(key)
                await pilot.pause(0.3)
            return Snap(app)


class RenderTests(ViewerCase):
    async def test_prefill_bar_rates_peer_session_and_request_rows(self):
        seed(self.db)
        snap = await self.run_app(self.app())
        self.assertIn("50%", snap.text["live"])
        self.assertIn("207,362/414,724", snap.text["live"])
        self.assertIn("█", snap.text["live"])
        self.assertIn("4,200 tok/s", snap.text["live"])
        self.assertIn("decode 120 tok/s", snap.text["live"])
        self.assertIn("laptop.example (100.64.1.2) x1", snap.text["peers"])
        self.assertIn("unparsed: 3", snap.text["footer"])
        self.assertIn("ctx limit 1,048,576", snap.text["header"])
        self.assertEqual(len(snap.sessions), 4)
        first = snap.sessions[0]
        self.assertEqual(first[0], "prefill")
        self.assertEqual(len(snap.requests), 2)
        self.assertEqual(snap.requests[0][1], "27+3")  # newest first
        self.assertEqual(snap.requests[1][7], "tool_calls")
        self.assertNotIn("collector stale", snap.text["header"])

    async def test_session_client_fields_shown_when_present_and_absent_is_fine(self):
        seed(self.db, sessions_extra={"peer": "100.64.1.2:5", "user_agent": "claude-cli/2", "cache_key": "abc123"})
        snap = await self.run_app(self.app())
        client = snap.sessions[0][5]
        self.assertIn("claude-cli/2", client)
        self.assertIn("abc123", client)
        self.assertEqual(snap.sessions[1][5], "")


class CacheHitTests(ViewerCase):
    async def test_request_log_shows_forwarded_tokens_and_hit_percent(self):
        seed(self.db)
        snap = await self.run_app(self.app())
        cold, warm = snap.requests[1], snap.requests[0]
        self.assertEqual((cold[2], cold[3], cold[4]), ("396,451", "100.0%", "31"))
        self.assertEqual((warm[2], warm[3], warm[4]), ("0", "0.0%", "27"))

    async def test_live_panel_and_sparkline_show_token_weighted_hit_rate(self):
        seed(self.db, samples_at=(NOW - 2, NOW - 1), tokens=(1000, 900))
        snap = await self.run_app(self.app())
        self.assertIn("cache hit 90.0% (15 min)", snap.text["live"])
        line = snap.text["spark"].split("\n")[4]
        self.assertIn("cache hit %", line)
        self.assertIn("window 90.0%", line)

    async def test_buckets_without_prompt_tokens_are_skipped(self):
        seed(self.db, tokens=(0, 0))
        snap = await self.run_app(self.app())
        self.assertIn("cache hit - (15 min)", snap.text["live"])
        self.assertEqual(snap.text["spark"].split("\n")[4].count("█"), 0)


class WindowTests(ViewerCase):
    async def test_window_keys_change_the_range(self):
        seed(self.db, samples_at=(NOW - 1800, NOW - 1790))  # 30 min ago: outside 15 min, inside 1 h
        snap = await self.run_app(self.app())
        self.assertIn("window 15 min", snap.text["spark"])
        self.assertEqual(snap.text["spark"].split("\n")[1].count("█"), 0)
        snap = await self.run_app(self.app(), keys=("2",))
        self.assertIn("window 1 h", snap.text["spark"])
        self.assertIn("█", snap.text["spark"].split("\n")[1])
        snap = await self.run_app(self.app(), keys=("3",))
        self.assertIn("window 24 h", snap.text["spark"])
        self.assertIn("█", snap.text["spark"].split("\n")[1])
        snap = await self.run_app(self.app(), keys=("3", "1"))
        self.assertIn("window 15 min", snap.text["spark"])

    async def test_q_quits(self):
        seed(self.db)
        app = self.app()
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("q")
        self.assertFalse(app.is_running)


class BannerTests(ViewerCase):
    async def test_server_down(self):
        seed(self.db, status=dict(state="down", down_since=NOW - 90))
        snap = await self.run_app(self.app())
        self.assertIn("mlx-serve down since", snap.text["header"])

    async def test_no_metrics_flag(self):
        seed(self.db, status=dict(state="no_metrics"))
        snap = await self.run_app(self.app())
        self.assertIn("start mlx-serve with --metrics", snap.text["header"])
        self.assertEqual(len(snap.requests), 2)  # request log still works

    async def test_key_states(self):
        seed(self.db, status=dict(state="no_key"))
        snap = await self.run_app(self.app())
        self.assertIn("no API key", snap.text["header"])
        os.remove(self.db)
        seed(self.db, status=dict(state="bad_key"))
        snap = await self.run_app(self.app())
        self.assertIn("API key rejected", snap.text["header"])

    async def test_log_missing(self):
        seed(self.db, status=dict(log_ok=0, log_path="/nope/mlx-serve.log"))
        snap = await self.run_app(self.app())
        self.assertIn("no log at /nope/mlx-serve.log", snap.text["header"])
        self.assertIn("4,200 tok/s", snap.text["live"])  # live panel unaffected

    async def test_collector_stale(self):
        seed(self.db, status=dict(ts=NOW - 42))
        snap = await self.run_app(self.app())
        self.assertIn("collector stale: last sample 42 s ago", snap.text["header"])

    async def test_peers_unavailable(self):
        seed(self.db, status=dict(peers_ok=0), peers=())
        snap = await self.run_app(self.app())
        self.assertIn("peers unavailable", snap.text["peers"])

    async def test_no_peers_connected(self):
        seed(self.db, peers=())
        snap = await self.run_app(self.app())
        self.assertIn("none connected", snap.text["peers"])


class DbTroubleTests(ViewerCase):
    async def test_missing_db_falls_back_to_live_metrics(self):
        server = FakeServer()
        self.addCleanup(server.close)
        snap = await self.run_app(self.app(url=server.url))
        self.assertIn("no history (collector not running?)", snap.text["header"])
        self.assertIn("prefill", snap.text["live"])
        self.assertIn("/414,724", snap.text["live"])
        self.assertEqual(len(snap.sessions), 4)

    async def test_missing_db_and_server_down(self):
        snap = await self.run_app(self.app(url="http://127.0.0.1:1/metrics.json"))
        self.assertIn("no history", snap.text["header"])

    async def test_corrupt_db_is_treated_as_missing(self):
        with open(self.db, "wb") as fh:
            fh.write(b"this is not a sqlite database" * 100)
        snap = await self.run_app(self.app(url="http://127.0.0.1:1/metrics.json"))
        self.assertIn("no history", snap.text["header"])

    async def test_locked_db_keeps_last_frame_and_does_not_raise(self):
        seed(self.db)
        app = self.app()
        async with app.run_test(size=SIZE) as pilot:
            await pilot.pause(0.3)
            before = str(app.rendered["live"])
            orig = mlxtop.query_all

            def locked(*a, **k):
                raise sqlite3.OperationalError("database is locked")
            mlxtop.query_all = locked
            self.addCleanup(setattr, mlxtop, "query_all", orig)
            await app.refresh_view()
            self.assertEqual(str(app.rendered["live"]), before)
        self.assertRaises(mlxtop.Busy, mlxtop.read_snapshot, self.db, NOW, 900, 20)

    def test_snapshot_errors_are_classified(self):
        self.assertRaises(mlxtop.Unreadable, mlxtop.read_snapshot, self.db, NOW, 900, 20)


class LayoutTests(ViewerCase):
    async def test_tiny_terminal_collapses_to_header_and_live(self):
        seed(self.db)
        snap = await self.run_app(self.app(), size=(50, 10))
        self.assertTrue(snap.display["header"] and snap.display["live"])
        for wid in mlxtop.OPTIONAL_IDS:
            self.assertFalse(snap.display[wid])


class PureTests(unittest.TestCase):
    def test_sparkline_scales_and_blanks_none(self):
        self.assertEqual(mlxtop.sparkline([None, 0, 5, 10], 10), "  ▄█")
        self.assertEqual(mlxtop.sparkline([], 5), "")

    def test_sparkline_fixed_top_scales_percentages(self):
        self.assertEqual(mlxtop.sparkline([50, 100], 5, top=100), "▄█")
        self.assertEqual(mlxtop.sparkline([10], 5, top=100), "▁")

    def test_pct(self):
        self.assertEqual(mlxtop.pct(435619, 435897), "99.9%")
        self.assertEqual(mlxtop.pct(0, 0), "-")

    def test_bar(self):
        self.assertEqual(mlxtop.bar(0.5, 10), "█████░░░░░")
        self.assertEqual(mlxtop.bar(2, 4), "████")
