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
        subprocess.run(['rm', '-rf', str(root)], check=True)
        raise


class Window:
    def __init__(self, args):
        self.args = args
        self.server = None
        self.reader = None
        self.stop_watch = threading.Event()
        self.failure = None
        self.samples = []
        self.active_request = False
        self.stopping = False
        self.base = args.out
        self.log = None

    def save(self):
        meta = dict(samples=self.samples, failure=self.failure, chunk=2048,
                    prefix_cold=True, selected_rows='reader clones cold, server pages uncontrolled',
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
                if free < 16 << 30 or (own is not None and foreign):
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

    def sample(self, arm, depth, rep, rank=None):
        tag = f'{arm}-d{depth}-r{rep}' + (f'-s{rank}' if rank is not None else '')
        dest = self.base / tag
        dest.mkdir()
        ids = self.ids[:depth]
        logpath = self.base / f'{arm}.log'
        offset = logpath.stat().st_size
        clone = None
        worker = None
        try:
            if rank is not None:
                clone, provenance = cold_pack(self.args.pack, dest)
                (dest / 'clone.json').write_text(json.dumps(provenance, indent=2) + '\n')
                out = (dest / 'reader.jsonl').open('w')
                err = (dest / 'reader.err').open('w')
                # Send the request first; begin reader when GPU layer trace proves active work.
                result = {}
                def run_request():
                    try:
                        result['value'] = self.request(ids)
                    except BaseException as error:
                        result['error'] = error
                worker = threading.Thread(target=run_request)
                worker.start()
                deadline = time.monotonic() + 120
                while 'pp4_layer' not in self.log_tail(logpath, offset).decode(errors='replace'):
                    self.check()
                    if not worker.is_alive() or time.monotonic() > deadline:
                        raise RuntimeError('no GPU trace before reader start')
                    time.sleep(.01)
                self.reader = subprocess.Popen([str(self.args.reader), str(clone),
                    str(self.base / 'corpus.u32'), str(depth), '32', str(rank), '2048'],
                    stdout=out, stderr=err)
                rc = self.reader.wait(timeout=120)
                self.reader = None
                out.close()
                err.close()
                worker.join(1200)
                if worker.is_alive():
                    raise TimeoutError('request stalled')
                if 'error' in result:
                    raise result['error']
                if rc:
                    raise RuntimeError(f'first-use reader exited {rc}')
                response, begin, end = result['value']
            else:
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
            sample = dict(arm=arm, depth=depth, rep=rep, rank=rank,
                          request_start_epoch_ns=begin, request_end_epoch_ns=end,
                          wall_s=elapsed, prefill_s=int(timing[0][1]) / 1000,
                          trace=f'{tag}/trace.log', reader=f'{tag}/reader.jsonl')
            from pp4_step0 import records, chunk_costs, deliveries
            if rank is not None:
                events = records(dest / 'trace.log')
                chunk_costs(events)
                read = records(dest / 'reader.jsonl')
                deliveries(read, depth, rank)
                reads = [e for e in read if e['kind'] == 'pp4_bucket' and e['rows']]
                layers = [e for e in events if e['kind'] == 'pp4_layer']
                if not reads or not layers:
                    raise ValueError('missing read/GPU activity')
                gpu_start = min(e['epoch_us'] for e in layers) * 1000
                gpu_end = max(e['epoch_us'] + e['elapsed_us'] for e in layers) * 1000
                if any(e['read_start_epoch_ns'] < gpu_start or
                       e['read_start_epoch_ns'] + e['read_ns'] > gpu_end for e in reads):
                    raise ValueError('first-use reader did not remain under GPU prefill load')
                compute = [(e['epoch_us'] * 1000 + e['engram_wait_us'] * 1000,
                            (e['epoch_us'] + e['elapsed_us']) * 1000) for e in layers]
                read_time = sum(e['read_ns'] for e in reads)
                overlap = sum(max(0, min(e['read_start_epoch_ns'] + e['read_ns'], hi) -
                                  max(e['read_start_epoch_ns'], lo))
                              for e in reads for lo, hi in compute)
                sample['compute_overlap_fraction'] = overlap / read_time if read_time else 0
                if sample['compute_overlap_fraction'] < .5:
                    raise ValueError('less than50% of reader time overlaps synchronized compute intervals')
            self.samples.append(sample)
            self.save()
            print('PP4-SAMPLE', tag, elapsed, flush=True)
        finally:
            if self.reader and self.reader.poll() is None:
                self.reader.terminate()
                try:
                    self.reader.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    self.reader.kill()
                    self.reader.wait()
            self.reader = None
            if worker and worker.is_alive():
                worker.join(120)
                if worker.is_alive():
                    self.failure = 'request did not drain within120s'
            if clone:
                subprocess.run(['rm', '-rf', str(clone)], check=True)

    def run(self):
        subprocess.run([str(self.args.reader), 'tokenize', str(self.args.pack),
                        str(self.args.corpus), str(self.base / 'corpus.u32')], check=True)
        raw = (self.base / 'corpus.u32').read_bytes()
        self.ids = list(struct.unpack('<' + 'I' * (len(raw) // 4), raw))
        if len(self.ids) < 131072:
            raise ValueError('real corpus too short')
        if model_pids() or free_bytes() < 16 << 30:
            raise RuntimeError('one-server/memory preflight refused')
        watch = threading.Thread(target=self.watch, daemon=True)
        watch.start()
        try:
            for block, arm in enumerate(('control', 'profile', 'control')):
                self.start_server(arm)
                self.request(self.ids[:32768])  # Discard kernel warm-up, prefix cache remains disabled.
                for depth in (32768, 131072):
                    for rep in range(6):
                        if arm == 'profile':
                            for rank in ((0, 1, 2, 3) if rep % 2 == 0 else (3, 2, 1, 0)):
                                self.sample(arm, depth, rep, rank)
                        else:
                            self.sample(arm, depth, rep + (6 if block == 2 else 0))
                self.stop_server()
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
    for name in ('binary', 'reader', 'pack', 'corpus', 'out'):
        ap.add_argument('--' + name, type=Path, required=True)
    args = ap.parse_args()
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
