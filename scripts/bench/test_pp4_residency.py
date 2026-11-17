"""Live residency evidence must remain separate from the WARM bucket measure."""
import copy
import json
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest

import pp4_step0 as pp4


def live_fixture():
    def fraction(pages, resident):
        return dict(pages=pages, resident_pages=resident, resident_fraction=resident / pages)
    return dict(kind='pp4_live_residency', schema=1, host='fixture-box',
                corpus_sha256='corpus', tokens_sha256='tokens', peak_rss_bytes=100 << 20, table_payload_read=False, table_header_read=False, elapsed_s=1,
                tables=[dict(layer=layer, path=f'/pack/layer-{layer}', whole=fraction(100, 25),
                    code=fraction(90, 20), scale=fraction(10, 5),
                    samples=[dict(depth=depth, token_offset=offset, code=fraction(8, 2),
                                  scale=fraction(2, 1))
                             for depth in (32768, 131072) for offset in (0, 4093, 8186)])
                        for layer in (1, 14)])


class LiveAnalysis(unittest.TestCase):
    def test_live_comparison_preserves_box_depth_and_warm_denominators(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'manifest.json').write_text(json.dumps(dict(corpus_sha256='corpus', tokens_sha256='tokens')))
            result = dict(depths={'32768': dict(source=str(root), cache_mode='WARM', resident_fraction=.75)})
            artifact = root / 'live.json'
            artifact.write_text(json.dumps(live_fixture()))
            text = pp4.attach_live_residency(result, [artifact])
            row = result['live_residency'][0]['comparison']['32768']
            self.assertAlmostEqual(row['live_resident_fraction'], .3)
            self.assertEqual(row['warm_arm_resident_fraction'], .75)
            self.assertIn('live resident fraction = 30.000% vs warm-arm resident fraction = 75.000%', '\n'.join(text))
            self.assertIn('fixture-box', '\n'.join(text))
            original = live_fixture()
            mutations = [dict(original, corpus_sha256='other'), dict(original, table_payload_read=True),
                         dict(original, tables=original['tables'][:1]), dict(original, elapsed_s=120), dict(original, tokens_sha256='other'),
                         dict(original, peak_rss_bytes=512 << 20), dict(original, table_header_read=True)]
            duplicate = copy.deepcopy(original)
            duplicate['tables'][0]['samples'][1]['token_offset'] = 0
            mutations.append(duplicate)
            wrong = copy.deepcopy(original)
            wrong['tables'][0]['samples'][0]['code']['resident_pages'] = 9
            mutations.append(wrong)
            for bad in mutations:
                with self.subTest(bad=bad):
                    artifact.write_text(json.dumps(bad))
                    with self.assertRaises(ValueError):
                        pp4.attach_live_residency(result, [artifact])

    def test_cli_includes_live_tables_and_retains_warm_bound(self):
        from test_pp4_step0 import write_artifact
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_artifact(root, (32768,))
            manifest = json.loads((root / 'manifest.json').read_text())
            manifest.update(warm=True, tokens_sha256='tokens')
            (root / 'manifest.json').write_text(json.dumps(manifest))
            for path in root.glob('reader-*.jsonl'):
                events = pp4.records(path)
                for event in events:
                    event['warm'] = True
                    if event['kind'] == 'pp4_bucket':
                        event.update(cached_parts=2, cold_rows=0, cold_read_ns=0,
                                     selected_pages=4, cached_pages=3, resident_fraction=.75)
                path.write_text(''.join(json.dumps(e) + '\n' for e in events))
            live = root / 'live.json'
            live.write_text(json.dumps(live_fixture()))
            expected = pp4.summarize(root)['depths'][32768]['cuts']
            run = subprocess.run([sys.executable, pp4.__file__, str(root), '--live-residency', str(live)],
                                 capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            result = json.loads((root / 'bound.json').read_text())
            self.assertEqual(result['depths']['32768']['cuts'], expected)
            self.assertEqual(result['live_residency'][0]['comparison']['32768']['live_resident_fraction'], .3)
            self.assertIn('| 1 | 25.000% |', (root / 'bound.md').read_text())


class LiveRunner(unittest.TestCase):
    def test_live_file_identity_requires_both_tables_on_the_selected_pid(self):
        import pp4_residency as runner
        opened = 'p123\nf4\nD1000004\ni42\nf5\nD1000004\ni43\n'
        expected = {(0x1000004, 42), (0x1000004, 43)}
        self.assertEqual(runner.open_inodes(opened), expected)
        runner.require_open_tables(expected, opened)
        with self.assertRaisesRegex(ValueError, 'live server'):
            runner.require_open_tables(expected | {(1, 2)}, opened)

    def test_layout_paths_must_match_live_tables_before_child_launch(self):
        import pp4_residency as runner
        pack = Path('/fixture')
        layout = dict(tables=[dict(layer=layer, relative_path=f'engram/{layer}') for layer in (1, 14)])
        expected = [pack / f'engram/{layer}' for layer in (1, 14)]
        runner.require_layout_paths(pack, layout, expected)
        layout['tables'][0]['relative_path'] = 'different-inode'
        with self.assertRaisesRegex(ValueError, 'layout paths'):
            runner.require_layout_paths(pack, layout, expected)
