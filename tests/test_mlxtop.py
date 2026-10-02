"""mlxtop: history kept by the client from /metrics.json counter deltas.

Fixtures follow `renderJson` in src/metrics.zig. Run: python -m pytest tests/test_mlxtop.py
"""
import asyncio
import copy
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import mlxtop  # noqa: E402

BOUNDS = [0.1, 0.5, 1.0, 5.0]


def hist(sum_s=0.0, cum=(0, 0, 0, 0, 0)):
    return {"count": cum[-1], "sum": sum_s, "bounds": BOUNDS, "bucket_counts": list(cum)}


def body(*, prefill=0, cached=0, gen=0, queries=0, hits=0, success=0, cancelled=0,
         prefill_s=0.0, decode_s=0.0, ttft=None, e2e=None, sessions=(), extra_counters=None,
         start=None):
    """A /metrics.json body shaped like renderJson on main."""
    counters = {"prompt_tokens_total": prefill + cached, "prefill_tokens_total": prefill,
                "prefix_cache_tokens_total": cached, "generation_tokens_total": gen,
                "requests_success_total": success, "requests_cancelled_total": cancelled,
                "prefix_cache_queries_total": queries, "prefix_cache_hits_total": hits}
    counters.update(extra_counters or {})
    gauges = {"requests_running": 1, "requests_waiting": 0, "gpu_utilization_pct": 61, "memory_mb": 2048,
              "generation_tokens_live": 0, "prefill_tokens_live": 0, "prefill_tokens_expected": 0,
              "requests_prefilling": 0, "mlx_active_bytes": 0, "mlx_cache_bytes": 0, "ane_int8_bytes": 0,
              "ane_layers": 0, "ngram_warm_bytes": 0, "batched_group_size": 0}
    if start is not None:
        gauges["process_start_time_seconds"] = start
    return {"counters": counters, "gauges": gauges, "decode_serial": {},
            "histograms": {"time_to_first_token_seconds": ttft or hist(),
                           "e2e_request_latency_seconds": e2e or hist(),
                           "prefill_time_seconds": hist(prefill_s), "decode_time_seconds": hist(decode_s),
                           "prompt_tokens": hist(), "output_tokens": hist()},
            "sessions": list(sessions)}


def session(model="m", gen=0, **over):
    base = {"model": model, "phase": "decode", "context_tokens": 100, "context_length": 4096,
            "cached_tokens": 20, "generated_tokens": gen, "state_bytes": 2_000_000}
    base.update(over)
    return base


def tracker(*points, **kw):
    tr = mlxtop.Tracker(**kw)
    for t, b in points:
        tr.observe(b, t)
    return tr


def rates(tr, now, window=3600):
    return mlxtop.build_view(tr, window, now, 10)["rates"]


def test_rates_come_from_counter_deltas_over_the_window():
    tr = tracker((0, body()),
                 (10, body(prefill=500, prefill_s=2.0, gen=300, decode_s=10.0, queries=10, hits=4,
                           cancelled=3, extra_counters={"requests_failed_total": 2,
                                                        "requests_rejected_total": 1})))
    r = rates(tr, 10)
    assert r["prefill"] == pytest.approx(250.0)
    assert r["decode"] == pytest.approx(30.0)
    assert r["cache_hit"] == pytest.approx(40.0)
    assert (r["failed_per_min"], r["rejected_per_min"], r["cancelled_per_min"]) == pytest.approx((12.0, 6.0, 18.0))


def test_percentiles_use_bucket_deltas_inside_the_window_only():
    old = hist(1.0, (100, 100, 100, 100, 100))
    new = hist(6.0, (100, 100, 110, 110, 110))
    tr = tracker((0, body(ttft=old, e2e=old)), (100, body(ttft=old, e2e=old)), (200, body(ttft=new, e2e=new)))
    r = rates(tr, 200, window=150)
    assert r["ttft_p50"] == pytest.approx(0.75)
    assert r["ttft_p95"] == pytest.approx(0.975)
    assert r["e2e_p50"] == pytest.approx(0.75)


# matrix 15: server unreachable mid-run
def test_unreachable_keeps_the_window_and_resumes():
    tr = tracker((0, body()), (10, body(gen=100, decode_s=2.0)))
    before = rates(tr, 10)
    tr.unreachable()
    view = mlxtop.build_view(tr, 3600, 11, 10)
    assert "unreachable" in view["notice"]
    assert view["rates"] == before
    tr.observe(body(gen=160, decode_s=3.0), 20)
    view = mlxtop.build_view(tr, 3600, 20, 10)
    assert view["notice"] is None
    assert view["rates"]["decode"] == pytest.approx(160 / 3.0)


def test_poll_marks_unreachable_and_recovers():
    tr = tracker((0, body()))

    def down(url, key):
        raise mlxtop.FetchError("unreachable")

    mlxtop.poll(tr, down, "u", None, 5)
    assert tr.state == "unreachable"
    mlxtop.poll(tr, lambda url, key: body(gen=10, decode_s=1.0), "u", None, 10)
    assert tr.state == "ok"
    assert rates(tr, 10)["decode"] == pytest.approx(10.0)


# matrix 16: HTTP 401
class Feed(BaseHTTPRequestHandler):
    status = 200

    def do_GET(self):
        self.send_response(Feed.status)
        self.end_headers()
        self.wfile.write(json.dumps(body()).encode() if Feed.status == 200 else b"{}")

    def log_message(self, *args):
        pass


@pytest.fixture
def server():
    httpd = HTTPServer(("127.0.0.1", 0), Feed)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    Feed.status = 200
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()


def test_http_401_stops_with_a_line_naming_api_key(server, monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    Feed.status = 401
    with pytest.raises(SystemExit) as stop:
        mlxtop.main(["--url", server])
    assert stop.value.code != 0
    err = capsys.readouterr().err.strip()
    assert "--api-key" in err and len(err.splitlines()) == 1


def test_poll_raises_on_401_mid_run():
    def denied(url, key):
        raise mlxtop.FetchError("unauthorized")

    with pytest.raises(mlxtop.AuthError) as err:
        mlxtop.poll(tracker(), denied, "u", None, 1)
    assert "--api-key" in str(err.value)


def test_key_comes_from_the_flag_then_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    seen = []
    monkeypatch.setattr(mlxtop, "probe", lambda url, key: None)
    monkeypatch.setattr(mlxtop.MlxTop, "run", lambda self: seen.append(self.key))
    monkeypatch.setenv("MLX_SERVE_API_KEY", "from-env")
    mlxtop.main([])
    mlxtop.main(["--api-key", "from-flag"])
    assert seen == ["from-env", "from-flag"]


def test_fetch_sends_the_key_as_a_bearer_token_only_when_given(server):
    seen = []
    orig = Feed.do_GET
    Feed.do_GET = lambda self: (seen.append(self.headers.get("Authorization")), orig(self))[1]
    try:
        mlxtop.fetch_metrics(server, "s3cret")
        mlxtop.fetch_metrics(server, None)
    finally:
        Feed.do_GET = orig
    assert seen == ["Bearer s3cret", None]


# matrix 17: server restarted
RESTART = [("counter", None, None), ("start_time", 100.0, 200.0)]


@pytest.mark.parametrize("kind,s1,s2", RESTART)
def test_restart_draws_a_gap_and_no_negative_rate(kind, s1, s2):
    tr = tracker((0, body(start=s1)),
                 (10, body(gen=100, decode_s=2.0, start=s1)),
                 (20, body(gen=300 if kind == "start_time" else 5, decode_s=3.0 if kind == "start_time" else 0.5,
                           start=s2)),
                 (30, body(gen=(310 if kind == "start_time" else 15), decode_s=(4.0 if kind == "start_time" else 1.5),
                           start=s2)))
    view = mlxtop.build_view(tr, 30, 30, 3)
    assert view["spark"]["decode"] == [pytest.approx(50.0), None, pytest.approx(10.0)]
    assert view["rates"]["decode"] == pytest.approx(110 / 3.0)
    assert all(v is None or v >= 0 for v in view["spark"]["prefill"])


# matrix 18: on-disk snapshot
def snapshot_tracker(path, ident="127.0.0.1:11234"):
    return mlxtop.Tracker(path=path, ident=ident)


def test_snapshot_restores_the_window_after_a_restart(tmp_path):
    path = tmp_path / "snap.json"
    first = snapshot_tracker(path)
    first.observe(body(), 0)
    first.observe(body(gen=100, decode_s=2.0), 10)
    first.save()
    assert [p.name for p in tmp_path.iterdir()] == ["snap.json"]
    second = snapshot_tracker(path)
    assert second.note is None
    second.observe(body(gen=160, decode_s=3.0), 20)
    assert rates(second, 20)["decode"] == pytest.approx(160 / 3.0)


@pytest.mark.parametrize("content", ["{not json", '{"ident":"x"', json.dumps({"ident": "127.0.0.1:11234", "samples": 7}),
                                     json.dumps({"ident": "127.0.0.1:11234", "samples": [{"t": "bad"}]})])
def test_corrupt_snapshot_starts_empty_with_a_note(tmp_path, content):
    path = tmp_path / "snap.json"
    path.write_text(content)
    tr = snapshot_tracker(path)
    assert tr.history.samples == [] and tr.note
    tr.observe(body(), 0)


def test_snapshot_from_another_server_is_ignored(tmp_path):
    path = tmp_path / "snap.json"
    other = snapshot_tracker(path, ident="10.0.0.2:11234")
    other.observe(body(), 0)
    other.save()
    tr = snapshot_tracker(path)
    assert tr.history.samples == [] and "another server" in tr.note


def test_unwritable_snapshot_does_not_raise(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    tr = snapshot_tracker(blocker / "snap.json")
    tr.observe(body(), 0)
    tr.save()
    assert "snapshot" in tr.note


def test_snapshot_path_is_keyed_by_host_and_port():
    path = mlxtop.snapshot_path("http://10.0.0.2:8080")
    assert path == Path.home() / ".cache" / "mlxtop" / "10.0.0.2-8080.json"


# matrix 19: older servers
def test_server_without_request_id_client_or_start_time_renders():
    tr = tracker((0, body(sessions=[session(gen=10)])), (10, body(gen=50, decode_s=2.0, sessions=[session(gen=50)])))
    view = mlxtop.build_view(tr, 3600, 10, 10)
    assert view["sessions"][0][-1] == "-"
    assert view["sessions"][0][3] == "50"
    assert view["rates"]["failed_per_min"] == 0 and view["rates"]["rejected_per_min"] == 0
    assert view["sessions"][0][4] == "-"


def test_monitor_block_is_ignored():
    pts = [(0, body()), (10, body(gen=50, decode_s=2.0, sessions=[session(gen=50)]))]
    plain = mlxtop.build_view(tracker(*pts), 3600, 10, 10)
    withmon = []
    for t, b in pts:
        b = copy.deepcopy(b)
        b["monitor"] = {"server": {"sampled_at_ms": 1}, "history": [{"at_ms": 1, "gpu_pct": 99.0}],
                        "history_archive": [], "recent_requests": [{"id": 1}],
                        "models": [{"id": "other", "state": "ready", "context_length": 1}]}
        withmon.append((t, b))
    assert mlxtop.build_view(tracker(*withmon), 3600, 10, 10) == plain


def test_sessions_with_request_id_and_client_get_live_rates_and_a_client_column():
    tr = tracker((0, body(sessions=[session(gen=10, request_id=7, client="claude-code"),
                                    session(gen=0, request_id=8, client="codex")])),
                 (4, body(sessions=[session(gen=50, request_id=7, client="claude-code"),
                                    session(gen=8, request_id=8, client="codex")])))
    rows = mlxtop.build_view(tr, 3600, 4, 10)["sessions"]
    assert [(r[-1], r[4]) for r in rows] == [("claude-code", "10.0"), ("codex", "2.0")]


def test_reused_request_id_after_a_restart_makes_no_live_rate():
    tr = tracker((0, body(start=1.0, sessions=[session(gen=500, request_id=7)])),
                 (4, body(start=2.0, sessions=[session(gen=3, request_id=7)])))
    assert mlxtop.build_view(tr, 3600, 4, 10)["sessions"][0][4] == "-"


def test_per_model_breakdown_from_sessions():
    tr = tracker((0, body(sessions=[session("a", gen=0, request_id=1), session("a", gen=0, request_id=2),
                                    session("b", gen=0, request_id=3)])),
                 (2, body(sessions=[session("a", gen=20, request_id=1), session("a", gen=20, request_id=2),
                                    session("b", gen=10, request_id=3)])))
    assert mlxtop.build_view(tr, 3600, 2, 10)["models"] == [["a", "2", "20.0"], ["b", "1", "5.0"]]


def test_history_keeps_raw_for_an_hour_then_one_per_minute_up_to_a_day():
    tr = mlxtop.Tracker()
    for t in range(0, 26 * 3600, 5):
        tr.observe(body(gen=t), t)
    now = 26 * 3600 - 5
    times = [s["t"] for s in tr.history.samples]
    assert min(times) >= now - 24 * 3600
    raw = [t for t in times if t > now - 3600]
    older = [t for t in times if t <= now - 3600]
    assert len(raw) == 3600 // 5
    assert len({t // 60 for t in older}) == len(older)


def test_screen_registers_the_refresh_timer_with_the_interval(monkeypatch):
    registered = []
    orig = mlxtop.MlxTop.set_interval

    def spy(self, interval, *a, **k):
        registered.append(interval)
        return orig(self, interval, *a, **k)

    monkeypatch.setattr(mlxtop.MlxTop, "set_interval", spy)

    def fetch(url, key):
        raise mlxtop.FetchError("unreachable")

    async def run():
        app = mlxtop.MlxTop("http://x:1", None, fetch=fetch, interval=0.5, tracker=mlxtop.Tracker())
        async with app.run_test() as pilot:
            await pilot.pause()

    asyncio.run(run())
    assert registered == [0.5]


@pytest.mark.parametrize("bad", ["0", "0.1", "-2", "fast"])
def test_refresh_interval_rejects_values_below_the_floor_or_not_numbers(bad, capsys):
    with pytest.raises(SystemExit):
        mlxtop.main(["--interval", bad])
    assert "--interval" in capsys.readouterr().err
