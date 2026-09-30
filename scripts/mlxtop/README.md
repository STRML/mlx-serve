# mlxtop

A live terminal dashboard for a running `mlx-serve`. A small collector samples the server once a
second into SQLite; the viewer reads that history read-only, so any number of viewers can be open.

```
mlxtop  Qwen3.8-27B-MLX-Serve-4bit  ctx limit 262,144  GPU 97%  mem 41,210 MB
prefill ██████████████░░░░░░░░░░░░░░░░  46%  190,464/414,724
prefill 4,210 tok/s   decode 38 tok/s   running 1  waiting 0   cache hit 92.4% (15 min)
phase    context  cached   generated  state MB  client
prefill  414,755  0        0          10,660    100.64.1.2:5 claude-cli/2 abc123
peers: laptop.local x2   desktop.local x1
      prefill tok/s ▁▁▁▆█▂▁▁▁▁▅▇▁▁▁▁▁▁▁▁▁▁▁▁▁ max 4,300
      decode tok/s  ▁▃▃▃▃▃▃▄▃▃▃▃▃▃▂▃▃▃▃▃▃▃▃▃▃ max 61
      GPU %         ▂▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ max 99
      cache hit %   ▇▇▇█▆▇▇▇▇▇▇█▇▇▇▇▇▇▇▇▇▇▇▇▇ window 92.4%
time      prompt+gen   cached   hit    forwarded  prefill/s  decode/s  finish
12:45:19  435,897+22   435,619  99.9%  278        714.2      59.0      stop
```

- **Live**: prefill progress, prefill and decode tok/s, running and waiting requests, and the
  token-weighted prefix-cache hit rate for the selected window. Prefill tok/s counts only tokens the
  server forwarded, never tokens restored from the prefix cache.
- **Sessions**: phase, context, cached and generated tokens, state size, and the client
  (`peer`, `user_agent`, `cache_key`) when the server reports them.
- **Peers**: TCP clients connected to the server port (`lsof`), reverse-resolved.
- **Sparklines** for prefill tok/s, decode tok/s, GPU % and cache hit rate over 15 min, 1 h or 24 h.
- **Request log**, newest first: cached tokens and hit percent, forwarded tokens (prompt minus
  cached) beside the server's own prefill tok/s.
- Banners for server down, missing `--metrics`, a missing or rejected API key, a missing log,
  and a stale collector. Without a history DB the live panel reads `/metrics.json` directly.

## Requirements

Python 3.9+, `textual`. The server needs `--metrics` (`/metrics.json`). The request log and the
peers panel need read access to the server log and `lsof`.

## Install

```sh
sh scripts/mlxtop/install.sh
```

Creates a venv at `~/.local/share/mlxtop/venv`, writes the `mlxtop` launcher to `~/.local/bin`, and on
macOS loads a LaunchAgent (`dev.mlxtop.collect`) that runs the collector. Elsewhere it prints the
collector command to run yourself. Options go before the flags below: `--venv DIR`, `--bin DIR`.
Any other flags are passed through to the collector and the launcher:

```sh
sh scripts/mlxtop/install.sh --url http://127.0.0.1:8098 --log ~/logs/mlx-serve.log
```

Or run it in place: `python3 scripts/mlxtop/collect.py &` then `python3 scripts/mlxtop/mlxtop.py`.

## Flags

| Flag | Default |
|---|---|
| `--url` | `http://127.0.0.1:11234` |
| `--api-key` | `$MLX_SERVE_API_KEY`, else the `--api-key` argument in `~/Library/LaunchAgents/<label>.plist`, else none |
| `--launchd-label` | `local.mlx-serve` |
| `--db` | `~/.local/share/mlxtop/mlxtop.db` (7 days of history) |
| `--log` (collector) | `~/.mlx-serve/logs/mlx-serve-<port>.log` if it exists, else `~/Library/Logs/mlx-serve.log` |

The key is read at runtime and not stored, unless you pass `--api-key` to `install.sh`. A flag on the
command line is visible in `ps`; prefer the environment variable. Peers are the `lsof`
connections on the port from `--url`.

## Keys

`1` 15 min, `2` 1 h, `3` 24 h window; `q` quit.

## Tests

```sh
python3 -m pytest scripts/mlxtop/tests -q
```

Needs `pytest` and `textual`; no network or server (the tests use a fake `/metrics.json` server).
