"""Warm windows keep prompt identity and measure the existing page cache."""
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import pp4_window as window_module
import pp4_step0 as pp4
from test_pp4_step0 import write_artifact


class WarmWindow(unittest.TestCase):
    def test_warm_plan_reserves_distinct_32k_prompts_before_each_depth(self):
        ids = list(range(469416))
        plan = window_module.plan_samples(ids, depths=[32768, 131072], warm=True)
        self.assertEqual(len(plan), 96)
        self.assertEqual(len({ids[s['token_offset']] for s in plan}), 96)
        self.assertEqual(len({tuple(window_module.sample_ids(ids, s)[:32]) for s in plan}), 96)
        for block in range(3):
            for depth in (32768, 131072):
                group = [s for s in plan if s['block'] == block and s.get('warm_depth', s['depth']) == depth]
                self.assertEqual([s['warmup'] for s in group[:4]], [True] * 4)
                self.assertEqual([s['depth'] for s in group[:4]], [32768] * 4)
                self.assertTrue(all(not s['warmup'] for s in group[4:]))
        with tempfile.TemporaryDirectory() as temp:
            tokens = Path(temp) / 'tokens.u32'
            tokens.write_bytes(struct.pack('<469416I', *ids))
            run = subprocess.run([sys.executable, window_module.__file__, '--check-tokens',
                str(tokens), '--warm', '--depths', '32768,131072', '--readers', '32'],
                capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertIn('samples=96', run.stdout)

    def test_warm_window_warms_each_depth_before_reading_or_measuring(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_artifact(root)
            (root / 'corpus.u32').write_bytes(struct.pack('<140000I', *range(140000)))
            (root / 'binary').write_bytes(b'server')
            (root / 'corpus').write_text('real corpus stand-in')
            window = window_module.Window(SimpleNamespace(out=root, reader='reader', binary=root / 'binary',
                pack=root, corpus=root / 'corpus', depths=[32768, 131072], readers=32, warm=True))
            plan = window_module.plan_samples(list(range(140000)), warm=True)
            specs = {s['token_offset']: s for s in plan}
            warmups, read = {}, set()
            def start(arm):
                (root / f'{arm}.log').touch()
            window.start_server = Mock(side_effect=start)
            def request(ids):
                spec = specs[ids[0]]
                key = (spec['block'], spec.get('warm_depth', spec['depth']))
                if spec['warmup']:
                    warmups[key] = warmups.get(key, 0) + 1
                else:
                    self.assertEqual(warmups[key], 4)
                    if spec['rank'] is not None:
                        self.assertIn((spec['depth'], spec['rep'], spec['rank']), read)
                return {}, 0, 10**10
            window.request = Mock(side_effect=request)
            def reader(command, stdout, **kwargs):
                spec = json.loads(Path(command[2]).with_name('input.json').read_text())
                key = (spec['block'], spec['depth'])
                self.assertEqual(warmups[key], 4)
                read.add((spec['depth'], spec['rep'], spec['rank']))
                events = pp4.records(root / f'reader-{spec["depth"]}-{spec["rank"]}-{spec["rep"]}.jsonl')
                for e in events:
                    e['warm'] = True
                    if e['kind'] == 'pp4_bucket':
                        e.update(selected_pages=2, cached_pages=0, resident_fraction=0)
                stdout.write(''.join(json.dumps(e) + '\n' for e in events))
                stdout.flush()
                return Mock(wait=Mock(return_value=0), poll=Mock(return_value=0))
            def trace(path, offset):
                depth = window_module.sample_ids(window.ids, specs[window.request.call_args.args[0][0]])
                return (root / f'trace-{len(depth)}.log').read_bytes() + f'[prefill-trace] tokens={len(depth)} total=9000ms\n'.encode()
            window.log_tail = Mock(side_effect=trace)
            with patch.object(window_module.subprocess, 'run'), \
                    patch.object(window_module.subprocess, 'Popen', side_effect=reader), \
                    patch.object(window_module, 'cold_pack', side_effect=AssertionError('warm clone')), \
                    patch.object(window_module, 'model_pids', return_value=[]), \
                    patch.object(window_module, 'free_bytes', return_value=32 << 30), \
                    patch.object(window_module.threading, 'Thread'):
                window.run()
            self.assertEqual(window.start_server.call_count, 3)
            self.assertEqual(len(read), 48)
            meta = json.loads((root / 'manifest.json').read_text())
            self.assertTrue(meta['warm'])
            self.assertEqual(len(meta['warmups']), 24)
            self.assertEqual(len(meta['samples']), 72)
            self.assertEqual({s['reader_mode'] for s in meta['samples'] if s['arm'] == 'profile'},
                             {'warm_before_request_idle_gpu'})
            self.assertEqual(len({s['tokens_sha256'] for s in meta['samples'] + meta['warmups']}), 96)

    def test_warm_reader_uses_original_pack_and_accepts_resident_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_artifact(root, (32768,))
            events = pp4.records(root / 'reader-32768-0-0.jsonl')
            for e in events:
                e['warm'] = True
                if e['kind'] == 'pp4_bucket':
                    e.update(cached_parts=2, cold_rows=0, cold_read_ns=0,
                             selected_pages=4, cached_pages=3, resident_fraction=.75)
            window = window_module.Window(SimpleNamespace(out=root, pack=root, reader='reader',
                                                          readers=32, warm=True))
            window.ids = list(range(40000))
            def reader(command, stdout, **kwargs):
                self.assertEqual(command[1], str(root))
                self.assertEqual(command[-1], '--warm')
                stdout.write(''.join(json.dumps(e) + '\n' for e in events))
                stdout.flush()
                return Mock(wait=Mock(return_value=0), poll=Mock(return_value=0))
            with patch.object(window_module, 'cold_pack', side_effect=AssertionError('warm clone')), \
                    patch.object(window_module.subprocess, 'Popen', side_effect=reader):
                window.read_sample(dict(arm='profile', depth=32768, rep=0, rank=0,
                                        block=1, warmup=False, token_offset=0))
            self.assertEqual(window.readers_ready, {'profile-d32768-r0-s0'})

    def test_warmup_floor_checks_before_and_after_request(self):
        for free, requests in (([15 << 30], 0), ([32 << 30, 15 << 30], 1)):
            with self.subTest(free=free), tempfile.TemporaryDirectory() as temp:
                window = window_module.Window(SimpleNamespace(out=Path(temp), warm=True))
                window.ids = list(range(40000))
                window.request = Mock(return_value=({}, 0, 1))
                window.save = Mock()
                with patch.object(window_module, 'free_bytes', side_effect=free), \
                        self.assertRaisesRegex(RuntimeError, 'memory floor'):
                    window.sample(dict(arm='control', depth=32768, rep=-1, rank=None,
                        block=0, warmup=True, warm_depth=131072, token_offset=0))
                self.assertEqual(window.request.call_count, requests)


class WarmAnalysis(unittest.TestCase):
    def test_cli_separates_warm_bounds_and_compares_residency_with_cold(self):
        with tempfile.TemporaryDirectory() as temp:
            roots = [Path(temp) / mode for mode in ('cold', 'warm')]
            for root in roots:
                root.mkdir()
                write_artifact(root, (32768,))
            manifest = roots[1] / 'manifest.json'
            meta = json.loads(manifest.read_text())
            meta['warm'] = True
            manifest.write_text(json.dumps(meta))
            for path in roots[1].glob('reader-*.jsonl'):
                events = pp4.records(path)
                for e in events:
                    e['warm'] = True
                    if e['kind'] == 'pp4_bucket':
                        e.update(cached_parts=2, cold_rows=0, cold_read_ns=0,
                                 selected_pages=4, cached_pages=3, resident_fraction=.75)
                path.write_text(''.join(json.dumps(e) + '\n' for e in events))
            out = Path(temp) / 'analysis'
            run = subprocess.run([sys.executable, pp4.__file__, *map(str, roots), '--out', str(out)],
                                 capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            result = json.loads((out / 'bound.json').read_text())
            warm = result['warm']['depths']['32768']
            self.assertEqual(warm['cache_mode'], 'WARM')
            self.assertEqual(warm['resident_fraction'], .75)
            self.assertEqual(result['comparison']['32768']['warm_resident_fraction'], .75)
            self.assertAlmostEqual(result['comparison']['32768']['cuts']['2/8/14']['delivery_bound_ratio'], 1.)
            text = (out / 'bound.md').read_text()
            self.assertIn('WARM', text)
            self.assertIn('75.000%', text)
            self.assertIn('Cold vs warm', text)
            events = pp4.records(roots[1] / 'reader-32768-0-0.jsonl')
            with self.assertRaisesRegex(ValueError, 'mode'):
                pp4.deliveries(events, 32768, 0, warm=False)
            self.assertEqual(pp4.summarize(roots[1])['depths'][32768]['cache_mode'], 'WARM')
            bad = [dict(e) for e in events]
            bad[1]['selected_pages'] = None
            path = roots[1] / 'reader-32768-0-0.jsonl'
            path.write_text(''.join(json.dumps(e) + '\n' for e in bad))
            partial = pp4.summarize(roots[1])
            self.assertEqual(partial['missing_depths'], [32768, 131072])
            self.assertIn('residency', partial['missing_reasons'][32768])
            events[1]['resident_fraction'] = 1.1
            with self.assertRaisesRegex(ValueError, 'residen'):
                pp4.deliveries(events, 32768, 0, warm=True)


if __name__ == '__main__':
    unittest.main()
