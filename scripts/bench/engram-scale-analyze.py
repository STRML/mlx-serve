#!/usr/bin/env python3
"""Verify collected Engram sweep arms and write throughput tables and JSON."""
import json
from pathlib import Path
from statistics import median
import sys

root = Path(sys.argv[1])
single = root / 'single-final'
four = root / 'four-final'
checksums = {}
binary_hashes = set()
token_hashes = set()
peak_rss = []


def arm(meta_path, rows_path):
    meta = json.loads(meta_path.read_text())
    assert meta['exit_code'] == 0, meta_path
    binary_hashes.add(meta['binary_sha256'])
    token_hashes.add(meta['tokens_sha256'])
    peak_rss.append(meta.get('combined_peak_upper_bytes', meta['peak_rss_bytes']))
    records = [json.loads(line) for line in rows_path.read_text().splitlines()]
    expected = {(p, l) for p in range(0, meta['depth'], meta['chunk']) for l in (1, 14)}
    assert {(r['pos'], r['layer']) for r in records} == expected
    assert len(records) == len(expected)
    for r in records:
        assert r['cached_parts'] == 0 and r['block_bytes'] == 4096
        assert r['depth'] == meta['depth'] and r['readers'] == meta['readers']
        assert r['rank'] == meta['rank'] and r['shards'] == meta['shards']
        assert r['payload_bytes'] == r['rows'] * 264
        assert r['read_ms'] > 0
        key = (r['depth'], r['chunk'], r['pos'], r['layer'], r['rank'], r['shards'])
        proof = (r['rows'], r['row_digest'], r['checksum'], r['first_row'], r['last_row'])
        assert checksums.setdefault(key, proof) == proof, (key, rows_path)
    return meta, records


points = {}
for depth in (32768, 131072):
    for readers in (1, 2, 4, 8, 16, 32):
        samples = []
        for rep in (1, 2, 3):
            d = single / ('full-d%d-n%d-r%d' % (depth, readers, rep))
            samples.append(arm(d / 'metadata.json', d / 'rows.jsonl'))
        points[depth, readers] = samples

lines = ['## Single SSD: whole-prompt read-ahead', '',
         'Three cold repetitions per cell, median read time. MB/s counts 264-byte payloads. Chunk time is amortized per 2,048 token positions.', '']
for depth in (32768, 131072):
    chunks = depth / 2048
    lines += ['### %s tokens' % f'{depth:,}', '',
              '| Readers | Layer 1 rows/s | MB/s | ms/chunk | Layer 14 rows/s | MB/s | ms/chunk | Both ms/chunk | Both time min–max (s) |',
              '|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for readers in (1, 2, 4, 8, 16, 32):
        samples = points[depth, readers]
        pairs = [sum(r['read_ms'] for r in rs) for m, rs in samples]
        fields = []
        for layer in (1, 14):
            ms = median(next(r['read_ms'] for r in rs if r['layer'] == layer) for m, rs in samples)
            row = next(r for r in samples[0][1] if r['layer'] == layer)
            rate = row['rows'] * 1000 / ms
            fields += [f'{rate:,.0f}', f'{rate*264/1e6:.2f}', f'{ms/chunks:.1f}']
        lines.append('| %d | %s | %.1f | %.2f–%.2f |' % (readers, ' | '.join(fields), median(pairs)/chunks, min(pairs)/1000, max(pairs)/1000))
    t1 = median(sum(r['read_ms'] for r in rs) for m, rs in points[depth, 1])
    t16 = median(sum(r['read_ms'] for r in rs) for m, rs in points[depth, 16])
    lines += ['', f'16-reader scaling: **{t1/t16:.2f}x** over one reader for both tables at {depth:,} tokens.', '']

lines += ['## Four SSDs: concurrent disjoint quarters', '',
          'Each arm starts after all four workers are ready under Latch holds. Disk time is the slowest box’s sum of both table reads. Wall time is the longest host duration, including hashing, sorting, cache verification and checksum work.', '',
          '| Tokens | Readers/SSD | Aggregate rows/s | Payload MB/s | Disk factor over one SSD | Wall factor | Layer 1 ms/chunk | Layer 14 ms/chunk | Both SSD budget ms/chunk | Start spread max (ms) |',
          '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
four_summary = []
for depth in (32768, 131072):
    chunks = depth / 2048
    for readers in (16, 32):
        critical, wall, times1, times14, spreads = [], [], [], [], []
        for rep in (1, 2, 3):
            tag = 'd%d-n%d-r%d-c0' % (depth, readers, rep)
            boxes = [arm(four / (tag+'-studio%d-metadata.json'%b), four / (tag+'-studio%d-rows.jsonl'%b)) for b in (1,2,3,4)]
            assert {m['host'] for m,rs in boxes} == {'studio1','studio2','studio3','studio4'}
            for layer in (1, 14):
                records = [next(r for r in rs if r['layer']==layer) for m,rs in boxes]
                full = next(r for r in points[depth,readers][0][1] if r['layer']==layer)
                assert sum(r['rows'] for r in records)==full['rows']
                assert all(r['distinct_rows']==full['rows'] for r in records)
                assert all(a['last_row']<b['first_row'] for a,b in zip(records,records[1:]))
            critical.append(max(sum(r['read_ms'] for r in rs) for m,rs in boxes))
            wall.append(max(m['end_epoch']-m['start_epoch'] for m,rs in boxes)*1000)
            times1.append(max(next(r['read_ms'] for r in rs if r['layer']==1) for m,rs in boxes))
            times14.append(max(next(r['read_ms'] for r in rs if r['layer']==14) for m,rs in boxes))
            starts = [m['start_epoch'] for m,rs in boxes]
            spreads.append((max(starts)-min(starts))*1000)
        one = points[depth,readers]
        one_ms = median(sum(r['read_ms'] for r in rs) for m,rs in one)
        one_wall = median((m['end_epoch']-m['start_epoch'])*1000 for m,rs in one)
        rows = sum(r['rows'] for r in one[0][1])
        ms = median(critical)
        rate = rows*1000/ms
        lines.append('| %s | %d | %s | %.2f | %.2fx | %.2fx | %.1f | %.1f | %.1f | %.1f |' % (f'{depth:,}',readers,f'{rate:,.0f}',rate*264/1e6,one_ms/ms,one_wall/median(wall),median(times1)/chunks,median(times14)/chunks,ms/chunks,max(spreads)))
        four_summary.append(dict(depth=depth, readers=readers, factor=one_ms/ms, combined_ms_per_chunk=ms/chunks, total_ms=ms, layer1_ms=median(times1)/chunks, layer14_ms=median(times14)/chunks))

lines += ['', '## Actual 2,048-token chunk controls', '',
          'One cold whole-prompt sweep per control; each chunk deduplicates independently. Time is the sum of chunk reads divided by the number of chunks. The four-box control uses the slowest box’s two-table sum.', '',
          '| Tokens | Readers | Single SSD both ms/chunk | Table rows read | Full-prompt distinct rows | Repeated reads factor |', '|---:|---:|---:|---:|---:|---:|']
four_controls = []
for depth in (32768,131072):
    for readers in (1,16,32):
        d=single/('chunks-d%d-n%d'%(depth,readers))
        m, rs=arm(d/'metadata.json',d/'rows.jsonl')
        rows=sum(r['rows'] for r in rs)
        distinct=sum(r['rows'] for r in points[depth,readers][0][1])
        lines.append('| %s | %d | %.1f | %s | %s | %.2fx |' % (f'{depth:,}',readers,sum(r['read_ms'] for r in rs)/(depth/2048),f'{rows:,}',f'{distinct:,}',rows/distinct))
    tag='d%d-n16-r1-c2048'%depth
    boxes=[arm(four/(tag+'-studio%d-metadata.json'%b),four/(tag+'-studio%d-rows.jsonl'%b)) for b in (1,2,3,4)]
    assert {m['host'] for m,rs in boxes} == {'studio1','studio2','studio3','studio4'}
    source=arm(single/('chunks-d%d-n16'%depth)/'metadata.json',single/('chunks-d%d-n16'%depth)/'rows.jsonl')[1]
    for full in source:
        cuts=[next(r for r in rs if (r['pos'],r['layer'])==(full['pos'],full['layer'])) for m,rs in boxes]
        assert sum(r['rows'] for r in cuts)==full['rows']
        assert all(a['last_row']<b['first_row'] for a,b in zip(cuts,cuts[1:]))
    critical=max(sum(r['read_ms'] for r in rs) for m,rs in boxes)
    four_controls.append(f'Four-box chunk control, {depth:,} tokens, 16 readers/SSD: **{critical/(depth/2048):.1f} ms per chunk** for both tables.')

for control in four_controls:
    lines += ['', control, '']

assert len(binary_hashes)==1, binary_hashes
assert len(token_hashes)==1, token_hashes
assert max(peak_rss)<4*2**30
lines += ['## Run identity', '', f'- Binary SHA-256: `{next(iter(binary_hashes))}`.', f'- Token SHA-256: `{next(iter(token_hashes))}`.', f'- Maximum recorded process/RSS bound: {max(peak_rss)/2**20:.1f} MiB.', '- Every accepted arm has zero cached selected code/scale spans. Reader-count and repetition changes preserve selected rows and payload checksums. Four-box row counts sum to the full prompt and adjacent shard row ranges do not overlap.']
(root/'measurement-tables.md').write_text('\n'.join(lines)+'\n')
(root/'summary.json').write_text(json.dumps(dict(four=four_summary,binary_sha256=list(binary_hashes),tokens_sha256=list(token_hashes),peak_rss=max(peak_rss)),indent=2)+'\n')
print('\n'.join(lines))
