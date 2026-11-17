"""Offline bound refuses incomplete evidence and charges the measured tail."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location('pp4', Path(__file__).with_name('pp4_step0.py'))
pp4 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pp4)


class BoundContract(unittest.TestCase):
    def test_tail_and_hops_change_both_cut_bounds(self):
        # Independent serial pipeline: e=.2, stages=(.1,.2,.3,.4), M=16.
        row = pp4.envelope(16, [.1, .2, .3, .4], .2, .5, 10)
        self.assertAlmostEqual(row['pipe_s'], 7.2)
        self.assertAlmostEqual(row['tp1_s'], 7.7)
        self.assertAlmostEqual(row['speedup'], 10 / 7.7)
        layers = [.01] * 20
        a = pp4.stage_costs(layers, .2, '2/8/14', .03)
        b = pp4.stage_costs(layers, .2, '8/14/20', .03)
        self.assertEqual([round(x, 4) for x in a], [.05, .09, .09, .26])
        self.assertEqual([round(x, 4) for x in b], [.11, .09, .09, .2])
        charged = pp4.stage_costs(layers, .2, '8/14/20', .03, head_s=.05)
        self.assertAlmostEqual(charged[3], .25)

    def test_delivery_waits_for_both_tables_and_slowest_shard(self):
        # The second chunk's late table must stall its owner, despite fast average IO.
        stages = [[.1, .1, .1, .1], [.1, .1, .1, .1]]
        result = pp4.simulate(stages, [[.2, .3], [.4, 1.2]], 3)
        self.assertAlmostEqual(result['finish_s'], 1.3)
        self.assertAlmostEqual(result['io_stall_s'], .8)
        self.assertAlmostEqual(result['needs'][1]['layer14_s'], .7)
        self.assertAlmostEqual(result['needs'][1]['layer14_late_s'], .5)

    def test_missing_and_duplicate_layer_or_x_refuses_summary(self):
        layers = [dict(kind='pp4_layer', p0=0, rows=2048, layer=i,
                       phase='encoder', elapsed_us=1000, compute_us=1000)
                  for i in range(20)]
        chunk = dict(kind='pp4_chunk', p0=0, rows=2048, elapsed_us=25000,
                     encoder_us=20000, compressor_us=1000, decoder_us=2000,
                     dspark_us=1000, head_us=1000, x_us=4000, dspark_enabled=True)
        self.assertEqual(pp4.chunk_costs(layers + [chunk])[0]['x_s'], .004)
        for bad in (layers[:-1] + [chunk], layers + [layers[0], chunk], layers):
            with self.subTest(records=len(bad)), self.assertRaises(ValueError):
                pp4.chunk_costs(bad)
        with self.assertRaises(ValueError):
            pp4.chunk_costs(layers + [dict(chunk, x_us=0)])


    def test_full_artifact_summary_refuses_partial_reader_and_recomputes_gate(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            samples = []
            for depth in (32768, 131072):
                trace, reader = [], []
                for p0 in range(0, depth - 1, 2048):
                    width = min(2048, depth - 1 - p0)
                    for layer in range(20):
                        trace.append(dict(kind='pp4_layer', p0=p0, rows=width,
                                          phase='encoder', layer=layer, compute_us=1000))
                    trace += [dict(kind='pp4_chunk', p0=p0, rows=width,
                                   elapsed_us=31000, encoder_us=20000, compressor_us=1000,
                                   decoder_us=6000, dspark_us=3000, head_us=1000,
                                   x_us=10000, dspark_enabled=True),
                              dict(kind='pp4_engram', p0=p0, needed_us=p0 * 10)]
                    k = p0 // 2048
                    for layer, step in ((1, 1), (14, 2)):
                        start = (k * 2 + step) * 50000000
                        reader.append(dict(kind='pp4_bucket', depth=depth, rank=0,
                            chunk_index=k, p0=p0, width=width, layer=layer,
                            cached_parts=0, rows=1, payload_bytes=264,
                            read_start_ns=start, read_end_ns=start + 1000000,
                            copy_end_ns=start + 2000000))
                trace_name = f'trace-{depth}.log'
                (root / trace_name).write_text(''.join(json.dumps(e) + '\n' for e in trace))
                for rep in range(6):
                    for rank in range(4):
                        name = f'reader-{depth}-{rank}-{rep}.jsonl'
                        events = [dict(kind='pp4_catalog', depth=depth, rank=rank, start_ns=0, ready_ns=0,
                                       prefill_tokens=depth - 1, chunk_width=2048, readers=32, shards=4,
                                       token_digest=123, catalog_digest=[456, 789], distinct_rows=[4, 4])]
                        events += [dict(e, rank=rank) for e in reader]
                        events += [dict(kind='pp4_reader_done', depth=depth, rank=rank, elapsed_ns=10**12)]
                        (root / name).write_text(''.join(json.dumps(e) + '\n' for e in events))
                        samples.append(dict(depth=depth, arm='profile', rank=rank,
                                            trace=trace_name, reader=name))
                samples += [dict(depth=depth, arm='control', prefill_s=9 if depth == 32768 else 30)
                            for _ in range(12)]
            manifest = root / 'manifest.json'
            manifest.write_text(json.dumps(dict(samples=samples)))
            result = pp4.summarize(root)
            self.assertEqual(result['verdict'], 'NO_GO')
            self.assertAlmostEqual(result['depths'][32768]['x_share'], .16 / 9)
            for sample in samples:
                if sample['arm'] == 'control':
                    sample['prefill_s'] = 100
            manifest.write_text(json.dumps(dict(samples=samples)))
            self.assertEqual(pp4.summarize(root)['verdict'], 'HOLD_PENDING_PREREQUISITES')
            target = root / samples[0]['reader']
            original = target.read_text().splitlines()
            variants = [original[:-1], [original[0], *reversed(original[1:-1]), original[-1]]]
            for lines in variants:
                target.write_text('\n'.join(lines) + '\n')
                with self.subTest(kind='missing-or-reordered'), self.assertRaises(ValueError):
                    pp4.summarize(root)

    def test_two_chunk_credit_horizon_holds_early_reader(self):
        result = pp4.simulate([[1, 0, 0, 0]] * 3,
                              [[.1, .2], [.3, .4], [.5, .6]], 3)
        self.assertAlmostEqual(result['paced_ready_s'][2][0], 1.2)
        self.assertAlmostEqual(result['paced_ready_s'][2][1], 1.3)


    def test_watchdog_first_poll_error_publishes_failure_and_stops_owned_processes(self):
        from types import SimpleNamespace
        from unittest.mock import Mock, patch
        spec = importlib.util.spec_from_file_location('window', Path(__file__).with_name('pp4_window.py'))
        window_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(window_module)
        with tempfile.TemporaryDirectory() as temp:
            window = window_module.Window(SimpleNamespace(out=Path(temp)))
            window.server = Mock(pid=123, poll=Mock(return_value=None))
            window.reader = Mock(poll=Mock(return_value=None))
            window.stop_watch.wait = Mock(side_effect=[False, True])
            with patch.object(window_module, 'free_bytes', side_effect=OSError('vm_stat failed')):
                window.watch()
            self.assertEqual(window.failure, 'vm_stat failed')
            window.server.terminate.assert_called_once_with()
            window.reader.terminate.assert_called_once_with()

    def test_window_overlap_excludes_waits_at_their_recorded_positions(self):
        from types import SimpleNamespace
        from unittest.mock import Mock, patch
        import pp4_window as window_module

        cases = [
            ('middle-wait', [(50, 90)], (50, 90), None),
            ('start-and-middle-waits', [(0, 20), (50, 90)], (50, 90), None),
            ('compute-between-waits', [(0, 20), (50, 90)], (20, 50), 1.),
            ('partial-compute', [(50, 90)], (0, 100), .6),
            ('overlapping-unsorted-waits', [(60, 90), (50, 80)], (0, 100), .6),
            ('waits-in-previous-layer', [(0, 20), (50, 90)], (100, 200), 1.),
            ('fully-waiting-layer', [(0, 100)], (0, 100), None),
            ('half-compute-boundary', [(0, 50)], (0, 100), .5),
            ('no-waits', [], (0, 100), 1.),
        ]
        for name, waits, (read_lo, read_hi), fraction in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                epoch, mono, depth = 900000, 10000, 101
                layers = [dict(kind='pp4_layer', p0=0, rows=100, layer=i,
                               phase='encoder', start_us=mono + i * 100,
                               end_us=mono + (i + 1) * 100,
                               epoch_us=epoch + i * 100, elapsed_us=100,
                               engram_wait_us=sum(hi - lo for lo, hi in waits) if i == 0 else 0,
                               compute_us=100) for i in range(20)]
                events = layers + [dict(kind='pp4_chunk', p0=0, rows=100,
                    elapsed_us=5000, encoder_us=2000, compressor_us=1000,
                    decoder_us=1000, dspark_us=1000, head_us=0, x_us=3000,
                    dspark_enabled=True)]
                events += [dict(kind='pp4_engram', p0=0, rows=100, table=0,
                                needed_us=mono + lo, taken_us=mono + hi,
                                epoch_us=epoch + lo) for lo, hi in waits]
                # Other chunks' waits must not remove this chunk's compute.
                events.append(dict(kind='pp4_engram', p0=2048, rows=100, table=0,
                                   needed_us=mono, taken_us=mono + 100, epoch_us=epoch))
                trace = ('[prefill-trace] tokens=101 total=1ms\n' +
                         ''.join(json.dumps(e) + '\n' for e in events)).encode()
                reader = [dict(kind='pp4_catalog', depth=depth, rank=0, start_ns=0,
                               ready_ns=0, prefill_tokens=100, chunk_width=2048,
                               shards=4, readers=32),
                          dict(kind='pp4_bucket', chunk_index=0, layer=1, p0=0,
                               width=100, depth=depth, rank=0, cached_parts=0,
                               rows=1, payload_bytes=264, read_start_ns=0,
                               read_end_ns=1, copy_end_ns=1,
                               read_start_epoch_ns=(epoch + read_lo) * 1000,
                               read_ns=(read_hi - read_lo) * 1000),
                          dict(kind='pp4_bucket', chunk_index=0, layer=14, p0=0,
                               width=100, depth=depth, rank=0, cached_parts=0,
                               rows=0, payload_bytes=0, read_start_ns=1,
                               read_end_ns=1, copy_end_ns=1),
                          dict(kind='pp4_reader_done', depth=depth, rank=0, elapsed_ns=1)]
                reads = [e for e in reader if e['kind'] == 'pp4_bucket' and e['rows']]
                self.assertAlmostEqual(window_module.compute_overlap_fraction(events, reads),
                                       0. if fraction is None else fraction)
                (root / 'profile.log').touch()
                window = window_module.Window(SimpleNamespace(out=root, pack=root, reader='reader'))
                window.ids = list(range(depth))
                window.request = Mock(return_value=({}, 0, 1000000))
                window.log_tail = Mock(return_value=trace)
                window.save = Mock()

                def run_reader(*args, stdout, **kwargs):
                    stdout.write(''.join(json.dumps(e) + '\n' for e in reader))
                    stdout.flush()
                    return Mock(wait=Mock(return_value=0), poll=Mock(return_value=0))

                with patch.object(window_module, 'cold_pack', return_value=(None, [])), \
                        patch.object(window_module.subprocess, 'Popen', side_effect=run_reader):
                    if fraction is None:
                        with self.assertRaisesRegex(ValueError, 'less than50%'):
                            window.sample('profile', depth, 0, 0)
                        self.assertEqual(window.samples, [])
                    else:
                        window.sample('profile', depth, 0, 0)
                        self.assertAlmostEqual(window.samples[0]['compute_overlap_fraction'], fraction)

    def test_dry_run_has_no_effects_and_unborrowed_run_refuses(self):
        script = Path(__file__).with_name('pp4-step0.sh')
        p = subprocess.run(['bash', str(script), '--dry-run'], capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn('borrow studio4', p.stdout)
        self.assertIn('2048', p.stdout)
        import os
        env = dict(os.environ, BORROW='', BORROW_BOXES='')
        p = subprocess.run(['bash', str(script), '--run'], env=env, capture_output=True, text=True)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn('requires studio4 borrow', p.stderr)


if __name__ == '__main__':
    unittest.main()
