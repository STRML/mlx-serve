#!/usr/bin/env python3
"""Studio4-only finite window. Starts exactly one private model server at a time."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import struct
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tests'))
from lib.server import wait_health

PORT = 18754
PREFIX_TOKENS = 32


def read_tokens(path):
    raw = path.read_bytes()
    if len(raw) % 4:
        raise ValueError('corpus token file is not u32 aligned')
    return list(struct.unpack('<' + 'I' * (len(raw) // 4), raw))


def sample_schedule(depths=(32768, 131072)):
    if not depths or len(set(depths)) != len(depths) or any(d not in (32768, 131072) for d in depths):
        raise ValueError("depths must select 32768 and/or 131072 once")
    for block, arm in enumerate(('control', 'profile', 'control')):
        yield dict(block=block, arm=arm, depth=depths[0], rep=-1, rank=None, warmup=True)
        for depth in depths:
            for rep in range(6):
                ranks = (None,)
                if arm == 'profile':
                    ranks = (0, 1, 2, 3) if rep % 2 == 0 else (3, 2, 1, 0)
                for rank in ranks:
                    yield dict(block=block, arm=arm, depth=depth,
                               rep=rep + (6 if block == 2 else 0), rank=rank, warmup=False)


def sample_ids(ids, sample):
    offset = sample['token_offset']
    return ids[offset:offset + sample['depth']]


def plan_samples(ids, deadline=None, depths=(32768, 131072)):
    deadline = time.monotonic() + 30 if deadline is None else deadline
    plan, first_tokens = [], set()
    for ordinal, sample in enumerate(sample_schedule(depths)):
        span = len(ids) - sample['depth'] + 1
        if span <= 0:
            raise ValueError('real corpus too short')
        for probe in range(span):
            if time.monotonic() > deadline:
                raise TimeoutError('offline sample check exceeded 30 seconds')
            offset = (ordinal * 4093 + probe) % span
            if ids[offset] not in first_tokens:
                break
        else:
            raise ValueError('corpus lacks distinct first tokens for all samples')
        first_tokens.add(ids[offset])
        plan.append(dict(sample, token_offset=offset))
    prefixes = set()
    for sample in plan:
        if time.monotonic() > deadline:
            raise TimeoutError('offline sample check exceeded 30 seconds')
        prompt = sample_ids(ids, sample)
        prefix = tuple(prompt[:PREFIX_TOKENS])
        if len(prompt) != sample['depth'] or prefix in prefixes:
            raise ValueError('sample has a short or repeated corpus prefix')
        prefixes.add(prefix)
    return plan


def http(path, body=None, timeout=10):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(f'http://127.0.0.1:{PORT}' + path, data=data,
                                    headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read()
    return json.loads(raw) if path != '/metrics' else raw.decode()


def free_bytes():
    text = subprocess.check_output(['vm_stat'], text=True)
    size = int(re.search(r'page size of (\d+) bytes', text)[1])
    counts = [int(re.search(r'^' + name + r':\s+(\d+)\.', text, re.M)[1])
              for name in ('Pages free', 'Pages inactive')]
    return sum(counts) * size


def model_pids():
    text = subprocess.check_output(['ps', '-axo', 'pid=,comm='], text=True)
    found = []
    for line in text.splitlines():
        pid, name = line.strip().split(None, 1)
        if Path(name).name.startswith(('mlx-serve', 'llama-server', 'vllm')):
            found.append(int(pid))
    return found


def cold_pack(pack, parent):
    """Reuse engram-scale's APFS clone/mincore method (33ee00f0)."""
    specs = json.loads((pack / 'config.json').read_text())['omlx_deepseek_v41']['engram_tables']
    paths = sorted({s['weight_file'] for s in specs.values()})
    if len(paths) != 2:
        raise ValueError('expected two split MXFP8 tables')
    root = Path(tempfile.mkdtemp(prefix='cold-', dir=parent))
    try:
        for name in ('config.json', 'engram-token-map.u32', 'engram_token_map.json'):
            if (pack / name).exists():
                (root / name).symlink_to(pack / name)
        libc = ctypes.CDLL(None, use_errno=True)
        libc.clonefile.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
        libc.clonefile.restype = ctypes.c_int
        provenance = []
        for rel in paths:
            if Path(rel).is_absolute() or '..' in Path(rel).parts:
                raise ValueError('table path escapes pack')
            src, dst = pack / rel, root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if libc.clonefile(os.fsencode(src), os.fsencode(dst), 0):
                raise OSError(ctypes.get_errno(), 'APFS clonefile', str(src))
            a, b = src.stat(), dst.stat()
            if a.st_ino == b.st_ino or a.st_size != b.st_size:
                raise ValueError('clone inode/size mismatch')
            provenance.append(dict(path=rel, source_inode=a.st_ino,
                                   clone_inode=b.st_ino, bytes=a.st_size))
        return root, provenance
    except BaseException:
        subprocess.run(['/usr/bin/trash', str(root)], check=True)
        raise


class Window:
    def __init__(self, args):
        self.args = args
        self.server = None
        self.reader = None
        self.stop_watch = threading.Event()
        self.failure = None
        self.samples = []
        self.warmups = []
        self.readers_ready = set()
        self.binary_sha = None
        self.active_request = False
        self.stopping = False
        self.base = args.out
        self.log = None

    def save(self):
        if self.binary_sha is None:
            self.binary_sha = hashlib.sha256(self.args.binary.read_bytes()).hexdigest()
        meta = dict(samples=self.samples, warmups=self.warmups, failure=self.failure, chunk=2048,
                    depths=self.args.depths, binary_sha256=self.binary_sha,
                    prefix_cold=True, selected_rows='reader clones probed before server startup; >=90% fully cold rows; server pages uncontrolled',
                    corpus_sha256=hashlib.sha256(self.args.corpus.read_bytes()).hexdigest(),
                    tokens_sha256=hashlib.sha256((self.base / 'corpus.u32').read_bytes()).hexdigest())
        (self.base / 'manifest.json').write_text(json.dumps(meta, indent=2) + '\n')

    def log_tail(self, path, offset):
        with path.open('rb') as stream:
            stream.seek(offset)
            return stream.read()

    def watch(self):
        while not self.stop_watch.wait(1):
            server = self.server
            own = server.pid if server else None
            try:
                free = free_bytes()
                foreign = [p for p in model_pids() if p != own]
                with (self.base / 'memory.jsonl').open('a') as out:
                    row = dict(epoch=time.time(), free_inactive_bytes=free, server_pid=own,
                               foreign_model_pids=foreign)
                    if own:
                        row['rss_kib'] = subprocess.check_output(['ps', '-o', 'rss=', '-p', str(own)], text=True).strip()
                        row['footprint_top'] = subprocess.check_output(
                            ['top', '-l', '1', '-pid', str(own), '-stats', 'pid,mem'], text=True)
                    out.write(json.dumps(row) + '\n')
                if free < 16 << 30 or foreign:
                    raise RuntimeError('memory floor16GiB or one-server invariant failed')
            except Exception as error:
                current_server = self.server
                if self.stopping or current_server is not server:
                    continue
                self.failure = str(error)
                for process in (self.reader, server):
                    if process and process.poll() is None:
                        try:
                            process.terminate()
                        except ProcessLookupError:
                            pass
                return

    def check(self):
        if self.failure:
            raise RuntimeError(self.failure)
        if self.server and self.server.poll() is not None:
            raise RuntimeError(f'server exited {self.server.returncode}')

    def stop_server(self):
        if not self.server:
            return
        deadline = time.monotonic() + 120
        while self.active_request and self.server.poll() is None and time.monotonic() < deadline:
            time.sleep(.1)
        if self.active_request:
            self.failure = 'request failed to drain before server shutdown'
        self.stopping = True
        self.server.terminate()
        try:
            self.server.wait(timeout=120)
        except subprocess.TimeoutExpired:
            self.server.kill()
            self.server.wait()
            raise RuntimeError('server required forced shutdown; window invalid')
        finally:
            self.server = None
            self.log.close()

    def start_server(self, arm):
        if model_pids():
            raise RuntimeError('existing model process before private server start')
        self.stopping = False
        env = dict(os.environ)
        env['MLX_SERVE_DIAG'] = 'prefill-trace' + (',pp4-step0' if arm == 'profile' else '')
        env['MLX_SERVE_GPU_CEILING_MB'] = '225000'
        self.log = (self.base / f'{arm}.log').open('ab', buffering=0)
        cmd = [str(self.args.binary), '--serve', '--model', str(self.args.pack),
               '--host', '127.0.0.1', '--port', str(PORT), '--ctx-size', '131200',
               '--max-concurrent', '1', '--prefill-chunk', '2048', '--no-drafter',
               '--prefix-cache-mem', '0', '--prefix-cache-disk', '0', '--dspark',
               '--warmup-full', '--log-level', 'debug']
        self.server = subprocess.Popen(cmd, env=env, stdout=self.log, stderr=self.log)
        wait_health(f'http://127.0.0.1:{PORT}', 900, self.server)
        self.check()
        (self.base / f'{arm}-props.json').write_text(json.dumps(http('/props'), indent=2) + '\n')

    def request(self, ids):
        self.check()
        self.active_request = True
        begin = time.time_ns()
        try:
            response = http('/v1/completions', dict(prompt=ids, max_tokens=1,
                            temperature=0, stream=False), timeout=1200)
        finally:
            self.active_request = False
        end = time.time_ns()
        self.check()
        if response.get('usage', {}).get('prompt_tokens') != len(ids):
            raise ValueError('server did not use exact token count')
        if not response.get('choices') or response.get('error'):
            raise ValueError('missing completion or server error')
        detail = response['usage'].get('prompt_tokens_details', {})
        if detail.get('cached_tokens') != 0:
            raise ValueError('prefix cache hit')
        return response, begin, end

    def sample_input(self, spec):
        arm, depth, rep, rank = (spec[k] for k in ('arm', 'depth', 'rep', 'rank'))
        tag = f'{arm}-d{depth}-r{rep}' + (f'-s{rank}' if rank is not None else '')
        if spec['warmup']:
            tag = f'{arm}-b{spec["block"]}-warmup'
        dest = self.base / tag
        dest.mkdir(exist_ok=True)
        ids = sample_ids(self.ids, spec)
        token_file = dest / 'tokens.u32'
        raw = struct.pack('<' + 'I' * len(ids), *ids)
        token_file.write_bytes(raw)
        input_meta = dict(spec, tokens=f'{tag}/tokens.u32',
                          tokens_sha256=hashlib.sha256(raw).hexdigest())
        (dest / 'input.json').write_text(json.dumps(input_meta, indent=2) + '\n')
        return dest, ids, input_meta

    def read_sample(self, spec):
        dest, _, _ = self.sample_input(spec)
        depth, rank = spec['depth'], spec['rank']
        clone = None
        try:
            clone, provenance = cold_pack(self.args.pack, dest)
            (dest / 'clone.json').write_text(json.dumps(provenance, indent=2) + '\n')
            with (dest / 'reader.jsonl').open('w') as out, (dest / 'reader.err').open('w') as err:
                self.reader = subprocess.Popen([str(self.args.reader), str(clone),
                    str(dest / 'tokens.u32'), str(depth), '32', str(rank), '2048'],
                    stdout=out, stderr=err)
                rc = self.reader.wait(timeout=120)
                self.reader = None
            if rc:
                raise RuntimeError(f'first-use reader exited {rc}')
            from pp4_step0 import records, deliveries
            deliveries(records(dest / 'reader.jsonl'), depth, rank)
            self.readers_ready.add(dest.name)
        finally:
            if self.reader and self.reader.poll() is None:
                self.reader.terminate()
                try:
                    self.reader.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    self.reader.kill()
                    self.reader.wait()
            self.reader = None
            if clone:
                subprocess.run(['/usr/bin/trash', str(clone)], check=True)

    def sample(self, spec):
        arm, depth, rank = spec['arm'], spec['depth'], spec['rank']
        dest, ids, input_meta = self.sample_input(spec)
        tag = dest.name
        if spec['warmup']:
            response, begin, end = self.request(ids)
            (dest / 'response.json').write_text(json.dumps(response, indent=2) + '\n')
            self.warmups.append(dict(input_meta, request_start_epoch_ns=begin,
                                     request_end_epoch_ns=end))
            self.save()
            return
        logpath = self.base / f'{arm}.log'
        offset = logpath.stat().st_size
        if rank is not None and tag not in self.readers_ready:
            self.read_sample(spec)
        response, begin, end = self.request(ids)
        (dest / 'response.json').write_text(json.dumps(response, indent=2) + '\n')
        trace = self.log_tail(logpath, offset)
        (dest / 'trace.log').write_bytes(trace)
        # Existing production trace reports prefill wall time in milliseconds.
        trace_text = trace.decode(errors='replace')
        matches = re.findall(r'\[prefill-trace\].*', trace_text)
        (dest / 'prefill-lines.txt').write_text('\n'.join(matches))
        timing = re.findall(r'\[prefill-trace\] tokens=(\d+).*?total=(\d+)ms', trace_text)
        if len(timing) != 1 or int(timing[0][0]) != depth or int(timing[0][1]) <= 0:
            raise ValueError('missing or mismatched production prefill timing')
        elapsed = (end - begin) / 1e9
        sample = dict(input_meta,
                      request_start_epoch_ns=begin, request_end_epoch_ns=end,
                      wall_s=elapsed, prefill_s=int(timing[0][1]) / 1000,
                      trace=f'{tag}/trace.log', reader=f'{tag}/reader.jsonl')
        from pp4_step0 import records, chunk_costs, deliveries
        if rank is not None:
            events = records(dest / 'trace.log')
            chunk_costs(events)
            read = records(dest / 'reader.jsonl')
            deliveries(read, depth, rank)
            sample['reader_mode'] = 'before_request_idle_gpu'
        self.samples.append(sample)
        self.save()
        print('PP4-SAMPLE', tag, elapsed, flush=True)

    def run(self):
        deadline = time.monotonic() + 30
        subprocess.run([str(self.args.reader), 'tokenize', str(self.args.pack),
                        str(self.args.corpus), str(self.base / 'corpus.u32')], check=True, timeout=30)
        self.ids = read_tokens(self.base / 'corpus.u32')
        plan = plan_samples(self.ids, deadline, self.args.depths)
        (self.base / 'sample-plan.json').write_text(json.dumps(plan, indent=2) + '\n')
        print(f'PP4-OFFLINE-CHECK OK samples={len(plan)} prefix_tokens={PREFIX_TOKENS}', flush=True)
        if model_pids() or free_bytes() < 16 << 30:
            raise RuntimeError('one-server/memory preflight refused')
        watch = threading.Thread(target=self.watch, daemon=True)
        watch.start()
        try:
            for spec in plan:
                if spec['rank'] is not None:
                    self.read_sample(spec)
            for spec in plan:
                if spec['warmup']:
                    self.stop_server()
                    self.start_server(spec['arm'])
                self.sample(spec)
        except BaseException as error:
            self.failure = self.failure or repr(error)
            raise
        finally:
            try:
                self.stop_server()
            finally:
                self.stop_watch.set()
                watch.join(15)
                self.save()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--depths', type=int, nargs='+', choices=(32768, 131072), default=[32768, 131072])
    ap.add_argument('--check-tokens', type=Path, help='check every sample offline within 30 seconds')
    for name in ('binary', 'reader', 'pack', 'corpus', 'out'):
        ap.add_argument('--' + name, type=Path)
    args = ap.parse_args()
    if args.check_tokens:
        deadline = time.monotonic() + 30
        plan = plan_samples(read_tokens(args.check_tokens), deadline, args.depths)
        print(f'PP4-OFFLINE-CHECK OK samples={len(plan)} prefix_tokens={PREFIX_TOKENS}')
        return
    for name in ('binary', 'reader', 'pack', 'corpus', 'out'):
        if getattr(args, name) is None:
            ap.error('--' + name + ' is required for a window')
    if os.environ.get('BORROW') != 'S4' or os.environ.get('BORROW_BOXES') != 'studio4' or os.uname().nodename.split('.')[0] != 'studio4':
        ap.error('requires studio4 borrow')
    for name in ('binary', 'reader', 'pack', 'corpus', 'out'):
        setattr(args, name, getattr(args, name).resolve())
    window = Window(args)
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    window.run()


if __name__ == '__main__':
    main()
