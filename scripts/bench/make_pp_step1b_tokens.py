#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
import sys
import tempfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
MAX_SINGLE_FILE_BYTES = 131_072
EXPECTED_VOCAB_SIZE = 129_280
MAX_TOKENS_PER_BYTE = 0.6


def list_candidate_files(root: str, subdir: str, ext: str) -> list[str]:
    base = os.path.join(root, subdir)
    if not os.path.isdir(base):
        return []
    found: list[str] = []
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames.sort()
        rel_dir = os.path.relpath(dirpath, root)
        if "fixtures" in rel_dir.split(os.sep):
            continue
        for name in sorted(filenames):
            if not name.endswith(ext):
                continue
            if name == "registry.zig" or name.startswith("unit_"):
                continue
            full = os.path.join(dirpath, name)
            try:
                sz = os.path.getsize(full)
            except OSError:
                continue
            if 0 < sz <= MAX_SINGLE_FILE_BYTES:
                found.append(full)
    return found


def collect_mixed_corpus(repo_root: str, target_bytes: int) -> str:
    docs = list_candidate_files(repo_root, "docs", ".md")
    code = list_candidate_files(repo_root, "src", ".zig")
    if not docs and not code:
        raise ValueError(f"no .md or .zig corpus files found under {repo_root}")

    interleaved: list[str] = []
    limit = max(len(docs), len(code))
    for i in range(limit):
        if i < len(docs):
            interleaved.append(docs[i])
        if i < len(code):
            interleaved.append(code[i])

    chunks: list[str] = []
    total = 0
    for path in interleaved:
        if total >= target_bytes:
            break
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                body = fh.read()
        except OSError:
            continue
        if not body.strip():
            continue
        rel = os.path.relpath(path, repo_root)
        piece = f"\n// --- {rel} ---\n{body}\n"
        chunks.append(piece)
        total += len(piece.encode("utf-8"))

    if not chunks:
        raise ValueError(f"empty corpus collected from {repo_root}")
    return "".join(chunks)


def resolve_mlx_serve_bin(explicit: str | None, repo_root: str) -> str:
    if explicit:
        return explicit
    env_bin = os.environ.get("MLX_SERVE_LOCAL_BIN")
    if env_bin:
        return env_bin
    sibling = os.path.join(SCRIPT_DIR, "mlx-serve")
    if os.path.isfile(sibling) and os.access(sibling, os.X_OK):
        return sibling
    return os.path.join(repo_root, "zig-out", "bin", "mlx-serve")


def build_tokenize_env(mlx_serve_bin: str, base_env: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ if base_env is None else base_env)
    staged_lib = os.path.join(os.path.dirname(os.path.abspath(mlx_serve_bin)), "lib")
    if os.path.isdir(staged_lib):
        prev = env.get("DYLD_LIBRARY_PATH", "")
        env["DYLD_LIBRARY_PATH"] = f"{staged_lib}:{prev}" if prev else staged_lib
    return env


def load_pack_vocab_size(model_dir: str, expected_vocab: int = EXPECTED_VOCAB_SIZE) -> int:
    cfg_path = os.path.join(os.path.expanduser(model_dir), "config.json")
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"invalid config.json root in {cfg_path}")
    text_cfg = cfg.get("text_config")
    if isinstance(text_cfg, dict) and "vocab_size" in text_cfg:
        raw_vocab = text_cfg["vocab_size"]
    else:
        raw_vocab = cfg.get("vocab_size")
    if isinstance(raw_vocab, bool) or not isinstance(raw_vocab, int) or raw_vocab <= 0:
        raise ValueError(f"missing or invalid vocab_size in {cfg_path}: {raw_vocab!r}")
    if expected_vocab > 0 and raw_vocab != expected_vocab:
        raise ValueError(f"pack vocab_size={raw_vocab} in {cfg_path}, expected {expected_vocab}")
    return raw_vocab


def validate_tokens_output(out_u32_path: str, vocab_size: int, text_bytes: int) -> int:
    if text_bytes <= 0:
        raise ValueError(f"invalid input text_bytes={text_bytes}")
    size = os.path.getsize(out_u32_path)
    if size <= 0 or size % 4 != 0:
        raise ValueError(f"invalid u32 output byte length: {size}")
    with open(out_u32_path, "rb") as fh:
        raw = fh.read()
    if len(raw) != size:
        raise ValueError(f"short read on {out_u32_path}: got {len(raw)} of {size}")
    tok_count = size // 4
    ids = struct.unpack(f"<{tok_count}I", raw)
    for idx, tid in enumerate(ids):
        if tid >= vocab_size:
            raise ValueError(f"token id out of vocab at index {idx}: {tid} >= vocab_size={vocab_size}")
    if tok_count > MAX_TOKENS_PER_BYTE * text_bytes:
        ratio = tok_count / text_bytes
        raise ValueError(
            f"tokens ({tok_count}) > {MAX_TOKENS_PER_BYTE} * text_bytes ({text_bytes}, ratio={ratio:.4f}): byte-level output suspected"
        )
    return tok_count


def run_tokenize_ids(mlx_serve_bin: str, model_dir: str, in_text_path: str, out_u32_path: str, vocab_size: int) -> int:
    if not (os.path.isfile(mlx_serve_bin) and os.access(mlx_serve_bin, os.X_OK)):
        raise FileNotFoundError(f"mlx-serve binary not executable: {mlx_serve_bin}")
    text_bytes = os.path.getsize(in_text_path)
    if text_bytes <= 0:
        raise ValueError(f"input text file is empty: {in_text_path}")
    env = build_tokenize_env(mlx_serve_bin)
    out_dir = os.path.dirname(os.path.abspath(out_u32_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    proc = subprocess.run(
        [
            mlx_serve_bin,
            "pp-probe",
            "tokenize-ids",
            "--model",
            os.path.expanduser(model_dir),
            "--in",
            in_text_path,
            "--out",
            out_u32_path,
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"tokenize-ids failed (rc={proc.returncode}): {detail}")

    return validate_tokens_output(out_u32_path, vocab_size=vocab_size, text_bytes=text_bytes)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build little-endian u32 real-text token IDs for Plan 75 Step 1b")
    ap.add_argument("--model", required=True, help="Path to DeepSeek-V4.1 pack directory")
    ap.add_argument("--out", required=True, help="Output little-endian u32 token-id file")
    ap.add_argument("--in-text", help="Optional existing UTF-8 corpus file to tokenize")
    ap.add_argument("--text-out", help="Optional path to save assembled UTF-8 corpus text")
    ap.add_argument("--mlx-serve", help="Path to mlx-serve binary")
    ap.add_argument("--repo-root", default=DEFAULT_REPO_ROOT, help="Repo root with docs/ and src/")
    ap.add_argument("--target-kb", type=int, default=768, help="Target corpus UTF-8 size in KiB (default: 768)")
    ap.add_argument("--min-tokens", type=int, default=150000, help="Minimum required output tokens (default: 150000)")
    args = ap.parse_args(argv)

    repo_root = os.path.abspath(os.path.expanduser(args.repo_root))
    mlx_serve_bin = resolve_mlx_serve_bin(args.mlx_serve, repo_root)
    out_u32 = os.path.abspath(os.path.expanduser(args.out))

    try:
        vocab_size = load_pack_vocab_size(args.model)
        if args.in_text:
            in_path = os.path.abspath(os.path.expanduser(args.in_text))
            tok_count = run_tokenize_ids(mlx_serve_bin, args.model, in_path, out_u32, vocab_size)
            if tok_count < args.min_tokens:
                print(f"MAKE-TOKENS-FAIL: got {tok_count} tokens < min_tokens={args.min_tokens}", file=sys.stderr)
                return 1
            text_bytes = os.path.getsize(in_path)
            print(f"MAKE-TOKENS-OK text_bytes={text_bytes} tokens={tok_count} vocab_size={vocab_size} out={out_u32}")
            return 0

        target_bytes = max(4096, args.target_kb * 1024)
        for attempt in range(3):
            corpus = collect_mixed_corpus(repo_root, target_bytes=target_bytes)
            text_bytes = len(corpus.encode("utf-8"))
            if args.text_out:
                text_path = os.path.abspath(os.path.expanduser(args.text_out))
                text_dir = os.path.dirname(text_path)
                if text_dir:
                    os.makedirs(text_dir, exist_ok=True)
                with open(text_path, "w", encoding="utf-8") as fh:
                    fh.write(corpus)
                tok_count = run_tokenize_ids(mlx_serve_bin, args.model, text_path, out_u32, vocab_size)
            else:
                with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as tf:
                    tf.write(corpus)
                    temp_path = tf.name
                try:
                    tok_count = run_tokenize_ids(mlx_serve_bin, args.model, temp_path, out_u32, vocab_size)
                finally:
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)

            if tok_count >= args.min_tokens or attempt == 2:
                if tok_count < args.min_tokens:
                    print(f"MAKE-TOKENS-FAIL: got {tok_count} tokens < min_tokens={args.min_tokens}", file=sys.stderr)
                    return 1
                print(f"MAKE-TOKENS-OK text_bytes={text_bytes} tokens={tok_count} vocab_size={vocab_size} out={out_u32}")
                return 0
            target_bytes *= 2
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"MAKE-TOKENS-FAIL: {exc}", file=sys.stderr)
        return 1

    return 1


if __name__ == "__main__":
    sys.exit(main())
