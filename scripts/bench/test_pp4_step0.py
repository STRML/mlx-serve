"""Offline bound refuses incomplete evidence and charges the measured tail."""
import importlib.util
import json
from pathlib import Path
import subprocess
import struct
import sys
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location('pp4', Path(__file__).with_name('pp4_step0.py'))
pp4 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pp4)


def write_artifact(root, depths=(32768, 131072)):
    samples = []
    for depth in depths:
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
                               token_digest=rep * 4 + rank,
                               catalog_digest=[456 + rep, 789 + rank], distinct_rows=[4, 4])]
                events += [dict(e, rank=rank) for e in reader]
                events += [dict(kind='pp4_reader_done', depth=depth, rank=rank, elapsed_ns=10**12)]
                (root / name).write_text(''.join(json.dumps(e) + '\n' for e in events))
                samples.append(dict(depth=depth, arm='profile', rank=rank,
                                    trace=trace_name, reader=name))
        samples += [dict(depth=depth, arm='control', prefill_s=9 if depth == 32768 else 30)
                    for _ in range(12)]
    (root / "manifest.json").write_text(json.dumps(dict(samples=samples, corpus_sha256="corpus")))
    (root / "binaries.sha256").write_text("binary  build/zig-out/bin/mlx-serve\nreader  pp4-first-use\n")
    return samples


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
            samples = write_artifact(root)
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
                with self.subTest(kind='missing-or-reordered'):
                    partial = pp4.summarize(root)
                    self.assertEqual(partial['missing_depths'], [32768])
                    self.assertEqual(set(partial['depths']), {131072})
                    self.assertEqual(partial['status'], 'PARTIAL')
                    self.assertRegex(partial['missing_reasons'][32768], 'completion|bucket')

    def test_failed_window_preserves_complete_depth_with_one_control_block(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            samples = write_artifact(root, (32768,))
            profiles = [s for s in samples if s['arm'] == 'profile']
            controls = [s for s in samples if s['arm'] == 'control'][:6]
            (root / 'manifest.json').write_text(json.dumps(dict(
                samples=controls + profiles, failure='first-use reader exited 1')))
            result = pp4.summarize(root)
            self.assertEqual(result['status'], 'PARTIAL')
            self.assertEqual(result['missing_depths'], [131072])
            self.assertEqual(result['verdict'], 'PARTIAL')
            row = result['depths'][32768]
            self.assertEqual(row['control_coverage'], 'preceding_only')
            self.assertAlmostEqual(row['x_share'], .16 / 9)
            self.assertEqual(row['cuts']['2/8/14']['ratio_gate'], 'NO_GO')
            self.assertIn('131072', result['missing_reasons'][131072])

    def test_analysis_cli_writes_outside_read_only_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'evidence'
            out = Path(temp) / 'analysis'
            root.mkdir()
            write_artifact(root, (32768,))
            root.chmod(0o555)
            try:
                run = subprocess.run([sys.executable, pp4.__file__, str(root), '--out', str(out)],
                                     capture_output=True, text=True)
            finally:
                root.chmod(0o755)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertEqual(json.loads((out / 'bound.json').read_text())['missing_depths'], [131072])
            self.assertIn('PARTIAL', (out / 'bound.md').read_text())
            self.assertFalse((root / 'bound.json').exists())

    def test_merge_depth_runs_requires_matching_binary_and_corpus(self):
        with tempfile.TemporaryDirectory() as temp:
            roots = [Path(temp) / str(i) for i in range(2)]
            for root, depth in zip(roots, (32768, 131072)):
                root.mkdir()
                write_artifact(root, (depth,))
            result = pp4.summarize(roots)
            self.assertEqual(set(result['depths']), {32768, 131072})
            self.assertEqual(result['status'], 'PROJECTION_ONLY')
            self.assertEqual(len(result['sources']), 2)
            for mismatch in ('binary', 'corpus', 'missing'):
                with self.subTest(mismatch=mismatch):
                    write_artifact(roots[1], (131072,))
                    if mismatch == 'binary':
                        (roots[1] / 'binaries.sha256').write_text('different  mlx-serve\n')
                    else:
                        path = roots[1] / 'manifest.json'
                        manifest = json.loads(path.read_text())
                        manifest['corpus_sha256'] = None if mismatch == 'missing' else 'different'
                        path.write_text(json.dumps(manifest))
                    with self.assertRaisesRegex(ValueError, 'identity'):
                        pp4.summarize(roots)
            write_artifact(roots[1], (32768,))
            with self.assertRaisesRegex(ValueError, 'overlapping depth'):
                pp4.summarize(roots)

    def test_depth_only_cli_checks_only_requested_depth(self):
        import pp4_window as window_module
        with tempfile.TemporaryDirectory() as temp:
            tokens = Path(temp) / 'tokens.u32'
            tokens.write_bytes(struct.pack('<140000I', *range(140000)))
            run = subprocess.run([sys.executable, window_module.__file__, '--check-tokens',
                str(tokens), '--depths', '131072'], capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertIn('samples=39', run.stdout)
            plan = window_module.plan_samples(list(range(140000)), depths=[131072])
            self.assertEqual({s['depth'] for s in plan}, {131072})
            with self.assertRaisesRegex(ValueError, 'depth'):
                window_module.plan_samples(list(range(140000)), depths=[8192])

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

    def test_reader_finishes_before_matching_request_and_failure_prevents_request(self):
        from types import SimpleNamespace
        from unittest.mock import Mock, patch
        import pp4_window as window_module
        for exitcode in (0, 1):
            with self.subTest(exitcode=exitcode), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                write_artifact(root, (32768,))
                (root / 'profile.log').touch()
                trace = (root / 'trace-32768.log').read_bytes()
                trace += b'[prefill-trace] tokens=32768 total=9000ms\n'
                reader = (root / 'reader-32768-0-0.jsonl').read_text()
                window = window_module.Window(SimpleNamespace(out=root, pack=root, reader='reader'))
                window.ids = list(range(40000))
                window.save = Mock()
                window.log_tail = Mock(return_value=trace)
                order = []
                def request(ids):
                    order.append('request')
                    self.assertEqual(order[:2], ['reader-start', 'reader-done'])
                    return {}, 0, 10**10
                window.request = Mock(side_effect=request)
                def run_reader(command, stdout, **kwargs):
                    order.append('reader-start')
                    self.assertEqual(Path(command[2]).read_bytes(), struct.pack('<32768I', *range(32768)))
                    stdout.write(reader)
                    stdout.flush()
                    def wait(timeout):
                        order.append('reader-done')
                        return exitcode
                    return Mock(wait=wait, poll=Mock(return_value=exitcode))
                with patch.object(window_module, 'cold_pack', return_value=(None, [])), \
                        patch.object(window_module.subprocess, 'Popen', side_effect=run_reader):
                    spec = dict(arm='profile', depth=32768, rep=0, rank=0,
                                block=1, warmup=False, token_offset=0)
                    if exitcode:
                        with self.assertRaisesRegex(RuntimeError, 'reader exited 1'):
                            window.sample(spec)
                        window.request.assert_not_called()
                    else:
                        window.sample(spec)
                        self.assertEqual(order, ['reader-start', 'reader-done', 'request'])
                        self.assertEqual(window.samples[0]['reader_mode'], 'before_request_idle_gpu')

    def test_window_reads_all_depths_before_any_server_can_touch_the_rows(self):
        from types import SimpleNamespace
        from unittest.mock import Mock, patch
        import pp4_window as window_module
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'corpus.u32').write_bytes(struct.pack('<140000I', *range(140000)))
            window = window_module.Window(SimpleNamespace(out=root, reader='reader',
                pack=root, corpus=root / 'corpus', depths=[32768, 131072]))
            read = []
            window.read_sample = Mock(side_effect=lambda spec: read.append((spec['depth'], spec['rep'], spec['rank'])))
            def start(arm):
                self.assertEqual(len(read), 48)
                self.assertEqual({d for d, _, _ in read}, {32768, 131072})
            window.start_server = Mock(side_effect=start)
            window.sample = Mock()
            window.save = Mock()
            with patch.object(window_module.subprocess, 'run'), \
                    patch.object(window_module, 'model_pids', return_value=[]), \
                    patch.object(window_module, 'free_bytes', return_value=32 << 30), \
                    patch.object(window_module.threading, 'Thread'):
                window.run()
            self.assertEqual(window.start_server.call_count, 3)

    def test_cached_parts_require_cold_timing_and_ninety_percent_coverage(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_artifact(root, (32768,))
            events = pp4.records(root / 'reader-32768-0-0.jsonl')
            for event in events:
                if event['kind'] == 'pp4_bucket':
                    event.update(rows=100, payload_bytes=26400, cached_parts=5,
                                 cold_rows=95, cold_read_ns=500000)
            ready = pp4.deliveries(events, 32768, 0)
            self.assertEqual(len(ready), 16)
            events[1]['cold_rows'] = 89
            for event in events:
                if event['kind'] == 'pp4_bucket':
                    event.update(cold_rows=89, cached_parts=11)
            with self.assertRaisesRegex(ValueError, 'cold coverage'):
                pp4.deliveries(events, 32768, 0)
            events[1].pop('cold_read_ns')
            with self.assertRaisesRegex(ValueError, 'cold'):
                pp4.deliveries(events, 32768, 0)

    def test_offline_plan_covers_warmups_and_rejects_repeated_prefixes_before_load(self):
        from types import SimpleNamespace
        from unittest.mock import Mock, patch
        import pp4_window as window_module

        # Frequent repeated words require collision resolution as well as offset spacing.
        ids = [i // 4096 for i in range(469416)]
        plan = window_module.plan_samples(ids)
        self.assertEqual(plan, window_module.plan_samples(ids))
        self.assertEqual(len(plan), 75)
        self.assertEqual(sum(s['warmup'] for s in plan), 3)
        self.assertEqual(len({ids[s['token_offset']] for s in plan}), 75)
        for sample in plan:
            offset, depth = sample['token_offset'], sample['depth']
            self.assertGreaterEqual(offset, 0)
            self.assertLessEqual(offset + depth, len(ids))
        wrapped_ids = [i % 4096 for i in range(140000)]
        wrapped = window_module.plan_samples(wrapped_ids)
        self.assertEqual(len({wrapped_ids[s['token_offset']] for s in wrapped}), 75)
        with tempfile.TemporaryDirectory() as temp:
            tokens = Path(temp) / 'corpus.u32'
            tokens.write_bytes(struct.pack('<' + 'I' * len(ids), *ids))
            result = subprocess.run([sys.executable, window_module.__file__,
                '--check-tokens', str(tokens)], capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('PP4-OFFLINE-CHECK OK samples=75 prefix_tokens=32', result.stdout)
        for bad in (ids[:131071], [7] * len(ids)):
            with self.subTest(corpus=len(bad)), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                token_file = root / 'corpus.u32'
                token_file.write_bytes(struct.pack('<' + 'I' * len(bad), *bad))
                window = window_module.Window(SimpleNamespace(out=root, reader='reader',
                    corpus=root / 'corpus.md', pack=root, depths=[32768, 131072]))
                window.start_server = Mock()
                with patch.object(window_module.subprocess, 'run'), \
                        self.assertRaisesRegex(ValueError, 'corpus|distinct'):
                    window.run()
                window.start_server.assert_not_called()
        with patch.object(window_module.time, 'monotonic', side_effect=[0, 31]):
            with self.assertRaisesRegex(TimeoutError, '30'):
                window_module.plan_samples(ids)

    def test_warmup_records_and_requests_its_own_exact_slice(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        import pp4_window as window_module

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            window = window_module.Window(SimpleNamespace(out=root))
            window.ids = list(range(469416))
            warmup = [s for s in window_module.plan_samples(window.ids) if s['warmup']][1]
            offset, depth = warmup['token_offset'], warmup['depth']
            expected = window.ids[offset:offset + depth]
            window.request = Mock(return_value=({}, 0, 1))
            window.save = Mock()
            window.sample(warmup)
            self.assertTrue(window.request.call_args.args[0] == expected, 'warm-up request slice differs')
            self.assertEqual(window.samples, [])
            recorded = window.warmups[0]
            raw = (root / recorded['tokens']).read_bytes()
            self.assertTrue(list(struct.unpack('<' + 'I' * depth, raw)) == expected,
                            'warm-up token artifact differs')
            meta = json.loads((root / recorded['tokens']).with_name('input.json').read_text())
            self.assertEqual(meta['token_offset'], offset)

    def test_cached_tokens_guard_still_refuses_a_live_slot_hit(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        import pp4_window as window_module

        window = window_module.Window(SimpleNamespace(out=Path('/unused')))
        for cached in (1, None):
            response = dict(choices=[{}], usage=dict(prompt_tokens=2,
                prompt_tokens_details=dict(cached_tokens=cached)))
            with self.subTest(cached=cached), \
                    patch.object(window_module, 'http', return_value=response), \
                    self.assertRaisesRegex(ValueError, 'prefix cache hit'):
                window.request([11, 22])

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
