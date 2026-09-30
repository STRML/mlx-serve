"""Failure rows: unknown log line form; counters going backwards."""
import os
import unittest

import mlxcommon as mc
from tests.helpers import FIXTURES, load_metrics


def fixture_lines():
    with open(os.path.join(FIXTURES, "requests.log")) as fh:
        return fh.read().splitlines()


class ParserTests(unittest.TestCase):
    def parse(self, needle):
        return mc.parse_request(next(ln for ln in fixture_lines() if needle in ln))

    def test_non_streamed_stop(self):
        r = self.parse("27+3 tokens (520ms)")
        self.assertEqual((r["prompt"], r["gen"], r["ms"], r["cached"], r["finish"]), (27, 3, 520, 0, "stop"))
        self.assertAlmostEqual(r["prefill_rate"], 55.9)
        self.assertAlmostEqual(r["decode_rate"], 187.3)

    def test_cold_streamed_tool_calls(self):
        r = self.parse("396482+561")
        self.assertEqual((r["prompt"], r["gen"], r["cached"], r["finish"]), (396482, 561, 0, "tool_calls"))
        self.assertIsNone(r["ms"])

    def test_cached_form(self):
        r = self.parse("397144+312")
        self.assertEqual((r["cached"], r["finish"]), (396451, "tool_calls"))
        self.assertAlmostEqual(r["prefill_rate"], 1378.6)
        self.assertAlmostEqual(r["decode_rate"], 126.8)

    def test_disconnect_and_length(self):
        self.assertEqual(self.parse("0+0")["finish"], "client_disconnect")
        self.assertEqual(self.parse("4497+2048")["finish"], "length")

    def test_non_request_lines_are_not_requests(self):
        for line in ("[mtp] regime gate: two-chunk", "  pld=enabled (draft_len=5, key_len=3)", ""):
            self.assertFalse(mc.is_request_line(line))
            self.assertIsNone(mc.parse_request(line))

    def test_unknown_form_is_a_request_line_that_does_not_parse(self):
        line = "  <- some future format we do not know"
        self.assertTrue(mc.is_request_line(line))
        self.assertIsNone(mc.parse_request(line))

    def test_every_real_request_fixture_line_parses_except_the_unknown_one(self):
        reqs = [ln for ln in fixture_lines() if mc.is_request_line(ln)]
        parsed = [mc.parse_request(ln) for ln in reqs]
        self.assertEqual(len(reqs), 8)
        self.assertEqual(sum(p is None for p in parsed), 1)


class RateTests(unittest.TestCase):
    def test_normal_delta(self):
        self.assertEqual(mc.rate(100, 350, 2.0), 125.0)

    def test_reset_drops_the_interval(self):
        series = [1000, 2000, 50, 150]  # server restarted between 2000 and 50
        rates = [mc.rate(a, b, 1.0) for a, b in zip(series, series[1:])]
        self.assertEqual(rates, [1000.0, None, 100.0])
        self.assertTrue(all(r is None or r >= 0 for r in rates))

    def test_missing_previous_or_zero_dt(self):
        self.assertIsNone(mc.rate(None, 5, 1.0))
        self.assertIsNone(mc.rate(1, 5, 0))


def snap(total=0, live=0, running=1):
    return {"counters": {"prefill_tokens_total": total}, "gauges": {
        "prefill_tokens_live": live, "requests_running": running, "requests_waiting": 0}}


class PrefillMeterTests(unittest.TestCase):
    def feed(self, *metrics):
        meter = mc.PrefillMeter()
        return [meter.update(m) for m in metrics]

    def test_first_call_has_no_interval(self):
        self.assertEqual(self.feed(snap(100, 50)), [None])

    def test_live_growth_counts_and_completion_adds_nothing_already_seen(self):
        out = self.feed(snap(), snap(0, 4000), snap(0, 8000), snap(0, 0), snap(8000, 0))
        self.assertEqual(out, [None, 4000, 4000, 0, 0])

    def test_prefill_shorter_than_a_poll_is_counted_at_completion(self):
        out = self.feed(snap(), snap(278, 0, running=0))
        self.assertEqual(out, [None, 278])

    def test_unseen_tail_of_a_long_prefill_is_added_once(self):
        out = self.feed(snap(), snap(0, 4000), snap(0, 0), snap(4500, 0))
        self.assertEqual(out, [None, 4000, 0, 500])

    def test_second_prefill_starting_as_the_first_ends(self):
        out = self.feed(snap(), snap(0, 4000), snap(0, 1000), snap(4000, 2000))
        self.assertEqual(out, [None, 4000, 1000, 1000])

    def test_counter_reset_drops_the_interval(self):
        out = self.feed(snap(5000, 100), snap(0, 0), snap(300, 0))
        self.assertEqual(out, [None, None, 300])

    def test_cancelled_prefill_does_not_hide_a_later_request(self):
        out = self.feed(snap(), snap(0, 4000), snap(0, 0, running=0), snap(700, 0, running=0))
        self.assertEqual(out, [None, 4000, 0, 700])


class DeltaTests(unittest.TestCase):
    def test_delta(self):
        self.assertEqual(mc.delta(10, 25), 15)
        self.assertIsNone(mc.delta(None, 5))
        self.assertIsNone(mc.delta(25, 10))


class UrlTests(unittest.TestCase):
    def test_metrics_url_and_port(self):
        self.assertEqual(mc.metrics_url("http://h:8000/"), "http://h:8000/metrics.json")
        self.assertEqual(mc.port_of("http://h:8000"), 8000)
        self.assertEqual(mc.port_of("https://h"), 443)


class KeyTests(unittest.TestCase):
    def test_missing_plist_gives_none(self):
        self.assertIsNone(mc.read_api_key("/nonexistent/plist"))

    def test_key_after_flag(self):
        import plistlib, tempfile
        with tempfile.NamedTemporaryFile(suffix=".plist") as fh:
            plistlib.dump({"ProgramArguments": ["x", "--api-key", "abc", "--metrics"]}, fh)
            fh.flush()
            self.assertEqual(mc.read_api_key(fh.name), "abc")

    def test_plist_without_flag_gives_none(self):
        import plistlib, tempfile
        with tempfile.NamedTemporaryFile(suffix=".plist") as fh:
            plistlib.dump({"ProgramArguments": ["x", "--metrics"]}, fh)
            fh.flush()
            self.assertIsNone(mc.read_api_key(fh.name))

    def test_key_precedence_flag_env_plist(self):
        import os, plistlib, tempfile
        from unittest import mock
        with tempfile.TemporaryDirectory() as home:
            agents = os.path.join(home, "Library/LaunchAgents")
            os.makedirs(agents)
            with open(os.path.join(agents, "x.mlx.plist"), "wb") as fh:
                plistlib.dump({"ProgramArguments": ["x", "--api-key", "from-plist"]}, fh)
            with mock.patch.object(mc, "HOME", home), mock.patch.dict(os.environ, {}, clear=True):
                self.assertEqual(mc.find_key(None, "x.mlx"), "from-plist")
                self.assertIsNone(mc.find_key(None, "other"))
                os.environ["MLX_SERVE_API_KEY"] = "from-env"
                self.assertEqual(mc.find_key(None, "x.mlx"), "from-env")
                self.assertEqual(mc.find_key("from-flag", "x.mlx"), "from-flag")
