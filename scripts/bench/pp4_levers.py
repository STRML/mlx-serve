#!/usr/bin/env python3
"""Offline 32k reader/cut sensitivity. Non-source cuts omit unmeasured shared-state transport."""
import argparse
import functools
import itertools
import json
from pathlib import Path
import statistics

import pp4_step0 as pp4

# Single-SSD, whole-prompt 32k rows/s, layers 1 and 14 (engram-scale report).
ROWS_PER_S = {4: (22340, 22663), 8: (45356, 45294),
              16: (67802, 68304), 32: (76279, 75970)}
ROWS_PER_S[6] = tuple((a + b) / 2 for a, b in zip(ROWS_PER_S[4], ROWS_PER_S[8]))
HOP_S = 83918848 / 3e9


def read_factors(source, target):
    return tuple(a / b for a, b in zip(ROWS_PER_S[source], ROWS_PER_S[target]))


def scale_reads(events, factors):
    origin = next(e['start_ns'] for e in events if e['kind'] == 'pp4_catalog')
    saved, flat = 0., []
    for e in events:
        if e['kind'] != 'pp4_bucket':
            continue
        factor = factors[int(e['layer'] == 14)]
        saved += (e['read_end_ns'] - e['read_start_ns']) * (1 - factor)
        flat.append((e['copy_end_ns'] - origin - saved) / 1e9)
    return [flat[k:k + 2] for k in range(0, len(flat), 2)]


def decoder_chunks(events):
    chunks = pp4.chunk_costs(events)
    measured = {}
    for e in events:
        if e['kind'] != 'pp4_layer' or e['phase'] != 'decoder' or e['rows'] <= 1:
            continue
        dst = measured.setdefault((e['p0'], e['rows']), {})
        if (e['layer'] in dst or e['layer_rows'] != 128 or
                e['compute_us'] < 0 or not isinstance(e['compute_us'], int)):
            raise ValueError('invalid decoder layer timing or replay rows')
        dst[e['layer']] = e['compute_us'] / 1e6
    if set(measured) != {(c['p0'], c['rows']) for c in chunks}:
        raise ValueError('missing decoder chunks')
    for c in chunks:
        layers = measured[c['p0'], c['rows']]
        if set(layers) != set(range(20, 40)):
            raise ValueError('missing decoder layer')
        c['decoder_layer_s'] = [layers[i] for i in range(20, 40)]
        if abs(sum(c['decoder_layer_s']) - c['decoder_s']) > 20e-6:
            raise ValueError('decoder layer sum differs from total')
    return chunks


def cost_model(chunks, reduce, hop_s=HOP_S):
    @functools.lru_cache(maxsize=None)
    def segment(i, lo, hi):
        cost = sum(reduce(c['layers_s'][j] for c in chunks) for j in range(lo, min(20, hi)))
        if lo <= 20 and hi == 40:
            cost += reduce(c['x_s'] for c in chunks)
        else:
            if lo <= 20 < hi:
                cost += reduce(c['compressor_s'] for c in chunks)
            if hi > 20:
                cost += reduce(sum(c['decoder_layer_s'][max(lo, 20) - 20:hi - 20]) for c in chunks)
            if hi == 40:
                cost += reduce(c['dspark_s'] for c in chunks)
        if i == 0:
            cost += reduce(c['overhead_s'] for c in chunks)
        if i == 3:
            cost += reduce(c['head_s'] for c in chunks)
        return cost + (hop_s if i < 3 else 0)

    def costs(cuts):
        bounds = (0, *cuts, 40)
        return [segment(i, lo, hi) for i, (lo, hi) in enumerate(zip(bounds, bounds[1:]))]
    return costs


def sensitivity(root):
    validated = pp4.summarize(root)
    if 32768 not in validated['depths']:
        raise ValueError(validated['missing_reasons'].get(32768, '32k evidence missing'))
    row = validated['depths'][32768]
    manifest = json.loads((root / 'manifest.json').read_text())
    samples = [s for s in manifest['samples'] if s['arm'] == 'profile' and s['depth'] == 32768]
    runs = [decoder_chunks(pp4.records(root / s['trace'])) for s in samples]
    readers = [pp4.records(root / s['reader']) for s in samples]
    flat = [c for run in runs for c in run]
    by_chunk = [[run[k] for run in runs] for k in range(len(runs[0]))]
    count = len(by_chunk)
    mean_cost = cost_model(flat, statistics.mean)
    chunk_cost = [cost_model(chunks, max) for chunks in by_chunk]
    # Layer 1 stays on stage 0, as in the step-0 credit/delivery contract.
    candidates = []
    for cuts in itertools.combinations(range(2, 40), 3):
        stages = mean_cost(cuts)
        scalar = pp4.envelope(count, stages, pp4.SSD[32768], .5, row['control_s'])
        candidates.append((cuts, scalar, [cost(cuts) for cost in chunk_cost]))
    scalar_best = min(candidates, key=lambda c: (c[1]['tp1_s'], c[0]))
    result = dict(status='PROJECTION_ONLY', control_coverage=row['control_coverage'],
        recorded_readers=row['reader_counts'], control_s=row['control_s'],
        target_s=row['control_s'] / 2.2, decoder_samples_per_layer=len(flat),
        decoder_layer_s=[statistics.mean(c['decoder_layer_s'][i] for c in flat) for i in range(20)],
        scalar_best=dict(cuts=scalar_best[0], **scalar_best[1]),
        assumptions=['read start aligned to request arrival; preparation retained',
            'only read intervals scale; throughput-linear interpolation at 6 readers',
            'two-chunk credits; storage floor .2411 s/chunk; .5 s final allowance',
            'first cut >=2 keeps layer1 on stage0; head and DSpark retained on stage3',
            'non-source cuts omit extra KV/index/candidate transport and are optimistic'],
        scenarios={})
    for anchor in sorted({6, *row['reader_counts']}):
        arms = {}
        for target in (6, 16, 32):
            timelines = [scale_reads(ev, read_factors(anchor, target)) for ev in readers]
            ready = [[max(t[k][j] for t in timelines) for j in range(2)] for k in range(count)]
            evaluated = []
            for cuts, scalar, stages in candidates:
                owner14 = next(i for i, hi in enumerate((*cuts, 40)) if hi > 14)
                sim = pp4.simulate(stages, ready, owner14)
                total = max(scalar['tp1_s'], sim['finish_s'] + .5)
                evaluated.append(dict(cuts=cuts, tp1_s=total, speedup=row['control_s'] / total,
                    scalar_s=scalar['tp1_s'], simulated_s=sim['finish_s'] + .5,
                    stages_s=scalar['stages_s'], clears_2_2=total <= result['target_s']))
            # Favor the smaller scalar envelope when delivery makes finishes tie.
            def order(v):
                return round(v['tp1_s'], 12), v['scalar_s'], v['cuts']
            arms[target] = dict(read_factors=read_factors(anchor, target),
                final_delivery_s=ready[-1][-1],
                current=[v for v in evaluated if v['cuts'] in ((2, 8, 14), (8, 14, 20))],
                best=min(evaluated, key=order),
                best_decoder_cut=min((v for v in evaluated if v['cuts'][-1] > 20), key=order),
                best_activation_only=min((v for v in evaluated if set(v['cuts']) <= {2, 8, 14, 20}), key=order))
        result['scenarios'][anchor] = arms
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = sensitivity(args.directory)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(args.out)


if __name__ == '__main__':
    main()
