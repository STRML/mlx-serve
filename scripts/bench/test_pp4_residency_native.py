"""Exercise the CPU probe under a table mapping cap and a table-read trap."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest


@unittest.skipUnless(sys.platform == 'darwin', 'Darwin mincore and dyld interposition')
class NativeResidency(unittest.TestCase):
    def test_large_bpe_corpus_and_tables_fit_memory_and_mapping_budgets(self):
        repo = Path(__file__).resolve().parents[2]
        zig = repo / '.zig-toolchain/bin/zig'
        if not zig.exists() or not shutil.which('cc'):
            self.skipTest('requires the pinned Zig toolchain and cc')
        with tempfile.TemporaryDirectory(prefix='pp4-residency-') as temp:
            root = Path(temp)
            binary = root / 'probe'
            subprocess.run([str(zig), 'build-exe', 'src/pp4_residency_probe.zig',
                            '-lc', '-O', 'ReleaseFast', f'-femit-bin={binary}',
                            '--cache-dir', str(root / 'cache'),
                            '--global-cache-dir', str(Path(tempfile.gettempdir()) / 'pp4-zig-global')],
                           cwd=repo, check=True, capture_output=True)
            guard = root / 'guard.dylib'
            subprocess.run(['cc', '-dynamiclib', '-O2', str(Path(__file__).with_name('pp4_residency_guard.c')),
                            '-o', str(guard)], check=True, capture_output=True)
            config = (repo / 'src/fixtures/model-configs/deepseek_v41-iq-mlx-3.0bpw.json').read_bytes()
            (root / 'config.json').write_bytes(config)
            tables = []
            for layer, rows in ((1, 384006168), (14, 384016682)):
                path = root / f'engram/{layer}.safetensors'
                path.parent.mkdir(exist_ok=True)
                size = 101535150936 + (rows - 384006168) * 264
                with path.open('wb') as stream:
                    stream.truncate(size)
                tables.append(dict(layer=layer, relative_path=str(path.relative_to(root)),
                                   file_bytes=size, rows=rows, w_off=232 + rows * 8,
                                   w_stride=256, wlen=256, s_off=232, s_stride=8, slen=8))
            layout = root / 'layout.json'
            layout.write_text(json.dumps(dict(schema=1, config_sha256=hashlib.sha256(config).hexdigest(), tables=tables)))
            byte_values = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
            chars = byte_values.copy()
            for byte in range(256):
                if byte not in byte_values:
                    chars.append(256 + len(byte_values) - 188)
                    byte_values.append(byte)
            vocab = {chr(char): byte for byte, char in zip(byte_values, chars)}
            vocab.update({f'fixture{i}': i for i in range(256, 129280)})
            (root / 'tokenizer.json').write_text(json.dumps(dict(
                model=dict(type='BPE', vocab=vocab, merges=[]),
                pre_tokenizer=dict(type='ByteLevel'), added_tokens=[])))
            corpus = b'hello world\n' * 400000
            (root / 'corpus.md').write_bytes(corpus)
            measure = ('import os,resource,subprocess,sys; '
                       'r=subprocess.run(sys.argv[2:],env=dict(os.environ,DYLD_INSERT_LIBRARIES=sys.argv[1])); '
                       'print("PP4-RSS",resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,file=sys.stderr); '
                       'sys.exit(r.returncode)')
            process = subprocess.run([sys.executable, '-c', measure, str(guard), str(binary), str(root), str(root / 'corpus.md'), str(layout)],
                                     capture_output=True, text=True, timeout=100)
            self.assertEqual(process.returncode, 0, process.stderr)
            result = json.loads(process.stdout)
            self.assertEqual(result['corpus_tokens'], len(corpus))
            self.assertEqual(result['corpus_sha256'], hashlib.sha256(corpus).hexdigest())
            self.assertFalse(result['table_header_read'])
            self.assertFalse(result['table_payload_read'])
            windows = re.findall(r'PP4-GUARD windows=(\d+)', process.stderr)
            self.assertEqual(max(map(int, windows)), sum((t['file_bytes'] + (64 << 20) - 1) // (64 << 20) for t in tables))
            rss = re.search(r'PP4-RSS (\d+)', process.stderr)
            self.assertIsNotNone(rss, process.stderr)
            self.assertLess(int(rss[1]), 512 << 20)
            self.assertLessEqual(result['peak_allocation_bytes'], result['allocation_cap_bytes'])
            print(f"PP4 native fixture: rss={rss[1]} bytes, "
                  f"peak_backing={result['peak_allocation_bytes']} bytes, windows={max(map(int, windows))}")
            for table, spec in zip(result['tables'], tables):
                self.assertEqual(table['whole']['pages'], (spec['file_bytes'] + os.sysconf('SC_PAGESIZE') - 1) // os.sysconf('SC_PAGESIZE'))
                self.assertEqual(table['whole']['resident_pages'], 0)
                self.assertEqual(len(table['samples']), 6)
                for sample in table['samples']:
                    self.assertGreater(sample['rows'], 0)
                    self.assertGreater(sample['code']['pages'], 0)
                    self.assertGreater(sample['scale']['pages'], 0)
                    self.assertEqual(sample['code']['resident_pages'], 0)
                    self.assertEqual(sample['scale']['resident_pages'], 0)


if __name__ == '__main__':
    unittest.main()
