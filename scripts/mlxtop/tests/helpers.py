"""Shared test fixtures: a fake mlx-serve HTTP server and a seeded DB."""
import copy
import json
import os
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

HERE = os.path.dirname(__file__)
FIXTURES = os.path.join(HERE, "fixtures")
KEY = "testkey"


def load_metrics() -> dict:
    with open(os.path.join(FIXTURES, "metrics.json")) as fh:
        return json.load(fh)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class FakeServer:
    """Serves the metrics fixture; `mode` is ok, 404 or 401. `payload` can be swapped between ticks."""

    def __init__(self):
        self.mode = "ok"
        self.payload = load_metrics()
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                authed = self.headers.get("Authorization") == f"Bearer {KEY}"
                if owner.mode == "404":
                    return self.reply(404, b"not found")
                if owner.mode == "401" or not authed:
                    return self.reply(401, b"unauthorized")
                self.reply(200, json.dumps(owner.payload).encode())

            def reply(self, code, body):
                self.send_response(code)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        self.httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}/metrics.json"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def bump(self, prefill=0, gen=0):
        self.payload = copy.deepcopy(self.payload)
        self.payload["gauges"]["prefill_tokens_live"] += prefill
        self.payload["gauges"]["generation_tokens_live"] += gen

    def restart(self):
        self.payload = copy.deepcopy(self.payload)
        for key in ("prefill_tokens_total", "prompt_tokens_total", "prefix_cache_tokens_total"):
            self.payload["counters"][key] = 0
        self.payload["gauges"].update(prefill_tokens_live=0, generation_tokens_live=0)

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()
