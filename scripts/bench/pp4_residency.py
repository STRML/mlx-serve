#!/usr/bin/env python3
"""Read-only live Engram residency. Run beside the selected server without a borrow."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import tempfile
import time


def open_inodes(text):
    result, device = set(), None
    for line in text.splitlines():
        if line.startswith('f'):
            device = None
        elif line.startswith('D'):
            device = int(line[1:], 16)
        elif line.startswith('i') and device is not None:
            result.add((device, int(line[1:])))
    return result


def require_open_tables(expected, text):
    if not expected <= open_inodes(text):
        raise ValueError('both original Engram tables must be open by the selected live server')


def require_layout_paths(pack, layout, expected):
    specs = layout['tables']
    if ([s['layer'] for s in specs] != [1, 14] or
            [(pack / s['relative_path']).resolve() for s in specs] != expected):
        raise ValueError('exported layout paths must match the original live tables')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pid', type=int, required=True)
    ap.add_argument('--pack', type=Path, required=True)
    ap.add_argument('--corpus', type=Path, default=Path.home() / 'bench/engram-corpus.md')
    ap.add_argument('--binary', type=Path, required=True)
    ap.add_argument('--layout', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    start = time.monotonic()
    if args.pid <= 1 or args.out.exists():
        raise ValueError('requires a live PID and a fresh output path')
    config = json.loads((args.pack / 'config.json').read_text())
    specs = config['omlx_deepseek_v41']['engram_tables']
    paths = []
    for layer in (1, 14):
        spec = specs[f'language_model.layers.{layer}.engram.embed']
        path = (args.pack / spec['weight_file']).resolve()
        if (spec.get('bits') is not None or spec['scale_file'] != spec['weight_file'] or
                Path(spec['weight_file']).is_absolute() or '..' in Path(spec['weight_file']).parts):
            raise ValueError('requires original split code/scale mxfp8 tables')
        paths.append(path)
    layout = json.loads(args.layout.read_text())
    require_layout_paths(args.pack, layout, paths)
    identities = [(p.stat().st_dev, p.stat().st_ino, p.stat().st_size) for p in paths]
    expected = {(dev, inode) for dev, inode, _ in identities}
    def check_server():
        text = subprocess.check_output(['/usr/sbin/lsof', '-a', '-p', str(args.pid), '-F', 'Di'],
                                       text=True, timeout=5)
        require_open_tables(expected, text)
        if identities != [(p.stat().st_dev, p.stat().st_ino, p.stat().st_size) for p in paths]:
            raise ValueError('Engram table identity changed during probe')
    process = subprocess.check_output(['ps', '-p', str(args.pid), '-o', 'lstart=', '-o', 'comm='], text=True, timeout=2)
    check_server()
    run = subprocess.run([str(args.binary.resolve()), str(args.pack.resolve()), str(args.corpus.resolve()), str(args.layout.resolve())],
                         capture_output=True, text=True, check=True, timeout=100)
    result = json.loads(run.stdout)
    check_server()
    if [t['path'] for t in result['tables']] != list(map(str, paths)):
        raise ValueError('probe table paths differ from live files')
    if process != subprocess.check_output(['ps', '-p', str(args.pid), '-o', 'lstart=', '-o', 'comm='], text=True, timeout=2):
        raise ValueError('live server process changed during probe')
    result['server_pid'] = args.pid
    result['server_process'] = process.strip()
    for table, identity in zip(result['tables'], identities):
        if (table['device'], table['inode'], table['file_bytes']) != identity:
            raise ValueError('probe descriptor differs from live table inode')
    result['elapsed_s'] = time.monotonic() - start
    result['peak_rss_bytes'] = (resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss +
                                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if result['elapsed_s'] >= 120 or result['peak_rss_bytes'] >= 512 << 20:
        raise ValueError('probe exceeded live time or memory budget')
    result['binary_sha256'] = hashlib.sha256(args.binary.read_bytes()).hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=args.out.parent, prefix='.pp4-residency-', delete=False) as stream:
        temp = Path(stream.name)
        try:
            json.dump(result, stream, indent=2)
            stream.write('\n')
            stream.close()
            os.link(temp, args.out)
        finally:
            temp.unlink(missing_ok=True)
    print(args.out)


if __name__ == '__main__':
    main()
