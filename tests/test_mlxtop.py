"""mlxtop: view model built from /metrics.json, fetch error states, and the on-screen banner.

Fixtures follow the shapes `monitor.zig` (`writeSample`, `renderJson`) and `metrics.zig` emit.
Run: python -m pytest tests/test_mlxtop.py
"""
import asyncio
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import mlxtop  # noqa: E402

NOW = 1_000_000


def sample(at_ms, **over):
    base = {"at_ms": at_ms, "continuity_id": 0, "running": 0, "queued": 0,
            "prefill_tokens_total": 0, "prefill_tokens_forwarded_live_total": 0,
            "generation_tokens_live": 0, "prefill_active_ns_total": 0, "decode_active_ns_total": 0,
            "requests_completed_total": 0, "cache_queries_total": 0, "cache_hits_total": 0,
            "ttft_ns_sum": 0, "ttft_count": 0, "cpu_pct": None, "gpu_pct": 40.0,
            "process_bytes": None, "mlx_active_bytes": None, "mlx_cache_bytes": None}
    base.update(over)
    return base


def request(rid, **over):
    base = {"id": rid, "model": "m", "outcome": "success", "error_code": None,
            "started_at_ms": NOW - 5000, "finished_at_ms": NOW - 1000, "queue_ms": 5,
            "prefill_ms": 500, "decode_ms": 2000, "ttft_ms": 505, "e2e_ms": 4000,
            "prompt_tokens": 1000, "output_tokens": 100, "cached_tokens": 600}
    base.update(over)
    return base


def feed(history=(), requests=(), archive=(), sessions=(), monitor=True, **gauges):
    body = {"counters": {}, "gauges": {"gpu_utilization_pct": 61, "memory_mb": 2048,
                                       "prefill_tokens_live": 0, "prefill_tokens_expected": 0,
                                       "requests_running": 1, "requests_waiting": 0, **gauges},
            "sessions": list(sessions)}
    if monitor:
        body["monitor"] = {"server": {"sampled_at_ms": NOW}, "history": list(history),
                           "history_archive": list(archive), "recent_requests": list(requests),
                           "models": [{"id": "m", "state": "ready", "context_length": 65536}]}
    return body


def test_prefill_rate_counts_forwarded_tokens_over_prefill_time():
    history = [sample(NOW - 4000),
               sample(NOW - 2000, prefill_tokens_forwarded_live_total=400, prefill_active_ns_total=2_000_000_000),
               # completion adds the 600 cached tokens to prefill_tokens_total; they were never computed
               sample(NOW, prefill_tokens_forwarded_live_total=400, prefill_tokens_total=1000,
                      prefill_active_ns_total=2_000_000_000)]
    view = mlxtop.build_view(feed(history), "1", 10)
    assert view["live"]["prefill_rate"] == pytest.approx(200.0)
    assert view["live"]["decode_rate"] is None


def test_decode_rate_and_idle_buckets():
    history = [sample(NOW - 4000), sample(NOW - 2000, generation_tokens_live=60, decode_active_ns_total=1_000_000_000),
               sample(NOW, generation_tokens_live=60, decode_active_ns_total=1_000_000_000)]
    view = mlxtop.build_view(feed(history), "1", 5)
    assert view["live"]["decode_rate"] == pytest.approx(60.0)
    assert view["spark"]["decode"] == [None, None, None, None, pytest.approx(60.0)]


def test_counter_reset_and_gap_make_no_rate():
    history = [sample(NOW - 6000, generation_tokens_live=500, decode_active_ns_total=9_000_000_000),
               sample(NOW - 4000, continuity_id=1),
               sample(NOW - 2000, continuity_id=1, generation_tokens_live=10, decode_active_ns_total=1_000_000_000),
               sample(NOW, continuity_id=2, generation_tokens_live=99, decode_active_ns_total=2_000_000_000)]
    view = mlxtop.build_view(feed(history), "1", 10)
    assert view["live"]["decode_rate"] == pytest.approx(10.0)


def test_window_hit_rate_and_archive_extends_the_long_window():
    archive = [sample(NOW - 3_000_000, cache_queries_total=0, cache_hits_total=0),
               sample(NOW - 2_000_000, cache_queries_total=10, cache_hits_total=2)]
    history = [sample(NOW - 4000, cache_queries_total=10, cache_hits_total=2),
               sample(NOW, cache_queries_total=14, cache_hits_total=5)]
    spark = mlxtop.build_view(feed(history, archive=archive), "1", 10)["spark"]
    assert spark["hit_window"] == "75.0%"
    assert spark["hit"][-1] == pytest.approx(75.0)
    merged = mlxtop.merged_history(feed(history, archive=archive)["monitor"])
    assert [s["at_ms"] for s in merged] == [NOW - 3_000_000, NOW - 2_000_000, NOW - 4000, NOW]


def test_archive_overlapping_the_raw_ring_is_dropped():
    archive = [sample(NOW - 10_000), sample(NOW - 4000)]
    merged = mlxtop.merged_history(feed([sample(NOW - 4000), sample(NOW)], archive=archive)["monitor"])
    assert [s["at_ms"] for s in merged] == [NOW - 10_000, NOW - 4000, NOW]


def test_request_log_row():
    row = mlxtop.build_view(feed(requests=[request(1, peer="10.0.0.9:4242", user_agent="claude-cli/2.1")]),
                            "1", 10)["requests"][0]
    assert row[1:] == ["1,000+100", "600", "60.0%", "400", "500", "800.0", "50.0", "success",
                       "10.0.0.9:4242 claude-cli/2.1"]


def test_request_log_is_newest_first_and_shows_failures():
    rows = mlxtop.build_view(feed(requests=[request(1), request(2, outcome="failed", error_code="Metal")]),
                             "1", 10)["requests"]
    assert rows[0][-2] == "failed Metal" and rows[1][-2] == "success"


def test_withheld_client_fields_render_as_a_dash():
    view = mlxtop.build_view(feed(requests=[request(1)], sessions=[{"model": "m", "phase": "decode"}]), "1", 10)
    assert view["requests"][0][-1] == "-"
    assert view["sessions"][0][-1] == "-"


def test_session_row_names_its_client():
    session = {"model": "m", "phase": "prefill", "context_tokens": 10, "cached_tokens": 4, "generated_tokens": 0,
               "state_bytes": 2_000_000, "peer": "10.0.0.9:4242", "user_agent": "curl/8", "cache_key": "00ab"}
    assert mlxtop.build_view(feed(sessions=[session]), "1", 10)["sessions"][0] == \
        ["prefill", "10", "4", "0", "2", "10.0.0.9:4242 curl/8 00ab"]


def test_live_panel_reports_prefill_progress_and_model():
    live = mlxtop.build_view(feed(sessions=[{"model": "m", "phase": "prefill", "context_length": 32768}],
                                  prefill_tokens_live=500, prefill_tokens_expected=1000), "1", 10)["live"]
    assert (live["model"], live["context"], live["gpu"], live["memory_mb"]) == ("m", 32768, 61, 2048)
    assert "50%" in mlxtop.live_text(live, None) and "500/1,000" in mlxtop.live_text(live, None)


def test_server_without_monitor_shows_live_values_and_a_notice():
    view = mlxtop.build_view(feed(monitor=False, sessions=[{"model": "m", "phase": "decode"}]), "1", 10)
    assert view["notice"] == mlxtop.NO_MONITOR
    assert view["spark"] is None and view["requests"] == []
    assert view["live"]["gpu"] == 61 and len(view["sessions"]) == 1
    assert mlxtop.spark_text(None, 10) == ""


def test_sparkline_scales_to_peak_and_blanks_idle():
    assert mlxtop.sparkline([None, 0, 5, 10], 4) == " " + mlxtop.BLOCKS[0] + mlxtop.BLOCKS[4] + mlxtop.BLOCKS[8]
    assert mlxtop.sparkline([50], 1, top=100) == mlxtop.BLOCKS[4]


class Feed(BaseHTTPRequestHandler):
    status, seen = 200, []

    def do_GET(self):
        Feed.seen.append((self.path, self.headers.get("Authorization")))
        self.send_response(Feed.status)
        self.end_headers()
        self.wfile.write(json.dumps(feed()).encode() if Feed.status == 200 else b"{}")

    def log_message(self, *args):
        pass


@pytest.fixture
def server():
    httpd = HTTPServer(("127.0.0.1", 0), Feed)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    Feed.status, Feed.seen = 200, []
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()


def test_fetch_sends_the_key_as_a_bearer_token(server):
    assert mlxtop.fetch_metrics(server, "s3cret")["gauges"]["memory_mb"] == 2048
    assert Feed.seen == [("/metrics.json", "Bearer s3cret")]


@pytest.mark.parametrize("status,state", [(401, "unauthorized"), (403, "unauthorized"), (503, "metrics_off")])
def test_fetch_maps_http_errors_to_states(server, status, state):
    Feed.status = status
    with pytest.raises(mlxtop.FetchError) as err:
        mlxtop.fetch_metrics(server, None)
    assert err.value.state == state


def test_fetch_reports_an_unreachable_server():
    with pytest.raises(mlxtop.FetchError) as err:
        mlxtop.fetch_metrics("http://127.0.0.1:1", None, timeout=0.5)
    assert err.value.state == "unreachable"


@pytest.mark.parametrize("state,text", [("unauthorized", "API key missing or rejected"),
                                        ("unreachable", "unreachable at http://x:1")])
def test_screen_shows_the_banner_when_the_feed_fails(state, text):
    def failing(url, key):
        raise mlxtop.FetchError(state)

    async def run():
        app = mlxtop.MlxTop("http://x:1", None, fetch=failing)
        async with app.run_test() as pilot:
            await pilot.pause(0.3)
            return str(app.query_one("#banner").render())

    assert text in asyncio.run(run())


def test_key_comes_from_the_flag_then_the_environment(monkeypatch):
    seen = []
    monkeypatch.setattr(mlxtop.MlxTop, "run", lambda self: seen.append(self.key))
    monkeypatch.setenv("MLX_SERVE_API_KEY", "from-env")
    mlxtop.main([])
    mlxtop.main(["--api-key", "from-flag"])
    assert seen == ["from-env", "from-flag"]
