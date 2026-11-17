#!/usr/bin/env python3
"""Validate step-0 artifacts and recompute plan75 envelopes, without a model."""
import argparse
import json
import math
from pathlib import Path
import statistics

CUTS = {'2/8/14': (0, 2, 8, 14, 20), '8/14/20': (0, 8, 14, 20, 20)}
SSD = {32768: .2411, 131072: .1593}


def envelope(chunks, stages, storage_s, allowance_s, control_s):
    pipe = storage_s + sum(stages) + (chunks - 1) * max(storage_s, *stages)
    total = pipe + allowance_s
    return dict(pipe_s=pipe, tp1_s=total, speedup=control_s / total,
                bottleneck_s=max(storage_s, *stages), stages_s=stages)


def stage_costs(layers, x_s, cuts, hop_s, embedding_s=0, head_s=0):
    bounds = CUTS[cuts]
    result = [sum(layers[bounds[i]:bounds[i + 1]]) for i in range(4)]
    result[0] += embedding_s
    result[3] += x_s + head_s
    return [v + (hop_s if i < 3 else 0) for i, v in enumerate(result)]


def simulate(stages, ready, owner14, credit=2):
    """Flow-shop simulation; each ready pair is cumulative table1/table14 delivery."""
    free = [0.] * 4
    needs, stalled = [], 0.
    stage0_done, paced_ready = [], []
    storage_done, previous = 0., 0.
    for k, (cost, delivered) in enumerate(zip(stages, ready)):
        release = stage0_done[k - credit] if k >= credit else 0.
        base = max(storage_done, release)
        delivered = [base + t - previous for t in delivered]
        previous = ready[k][1]
        storage_done = delivered[1]
        paced_ready.append(delivered)
        upstream = 0.
        record = {'chunk': k}
        for i in range(4):
            need = max(upstream, free[i])
            arrival = delivered[0] if i == 0 else delivered[1] if i == owner14 else 0.
            if i in (0, owner14):
                table = 'layer1' if i == 0 else 'layer14'
                record[table + '_s'] = need
                record[table + '_late_s'] = max(0., arrival - need)
            stalled += max(0., arrival - need)
            free[i] = max(need, arrival) + cost[i]
            upstream = free[i]
            if i == 0:
                stage0_done.append(free[i])
        needs.append(record)
    return dict(finish_s=free[-1], io_stall_s=stalled, needs=needs, paced_ready_s=paced_ready)


def records(path):
    result = []
    for line in Path(path).read_text().splitlines():
        at = line.find('{')
        if at < 0:
            continue
        try:
            item = json.loads(line[at:])
        except json.JSONDecodeError:
            continue
        if str(item.get('kind', '')).startswith('pp4_'):
            result.append(item)
    return result


def chunk_costs(events):
    chunks = {}
    layers = {}
    for e in events:
        if e['kind'] not in ('pp4_chunk', 'pp4_layer') or e['rows'] <= 1:
            continue
        key = (e['p0'], e['rows'])
        if e['kind'] == 'pp4_chunk':
            if key in chunks:
                raise ValueError('duplicate chunk')
            chunks[key] = e
        elif e['phase'] == 'encoder':
            dst = layers.setdefault(key, {})
            if e['layer'] in dst:
                raise ValueError('duplicate encoder layer')
            dst[e['layer']] = e['compute_us'] / 1e6
    if not chunks or set(chunks) != set(layers):
        raise ValueError('missing chunk/X record')
    out = []
    for key, e in sorted(chunks.items()):
        if set(layers[key]) != set(range(20)):
            raise ValueError('missing encoder layer')
        if e['dspark_enabled'] is not True:
            raise ValueError('native DSpark capture disabled')
        x = sum(e[k] for k in ('compressor_us', 'decoder_us', 'dspark_us'))
        if abs(x - e['x_us']) > 3 or not math.isfinite(x):
            raise ValueError('X component mismatch')
        measured = [layers[key][i] for i in range(20)]
        if any(v < 0 or not math.isfinite(v) for v in measured + [x]):
            raise ValueError('negative/nonfinite timing')
        # Embedding/preparation and sync gaps stay charged, rather than disappearing.
        residual = max(0., (e['elapsed_us'] - e['encoder_us'] - x - e['head_us']) / 1e6)
        out.append(dict(p0=key[0], rows=key[1], layers_s=measured,
                        x_s=x / 1e6, overhead_s=residual, head_s=e['head_us'] / 1e6,
                        compressor_s=e['compressor_us'] / 1e6,
                        decoder_s=e['decoder_us'] / 1e6, dspark_s=e['dspark_us'] / 1e6))
    return out


def deliveries(events, depth, rank, readers=None, warm=False):
    catalog = [e for e in events if e['kind'] == 'pp4_catalog']
    buckets = [e for e in events if e['kind'] == 'pp4_bucket']
    if len(catalog) != 1:
        raise ValueError('expected one first-use catalog')
    expected_catalog = dict(depth=depth, rank=rank, prefill_tokens=depth - 1,
                            chunk_width=2048, shards=4)
    if any(catalog[0].get(k) != v for k, v in expected_catalog.items()):
        raise ValueError('reader catalog identity mismatch')
    if catalog[0].get('warm', False) is not warm:
        raise ValueError('reader cache mode mismatch')
    count_readers = catalog[0].get('readers')
    if (type(count_readers) is not int or not 1 <= count_readers <= 32 or
            (readers is not None and count_readers != readers)):
        raise ValueError('reader count mismatch or outside 1..32')
    if sum(e['kind'] == 'pp4_reader_done' for e in events) != 1:
        raise ValueError('reader completion record missing')
    done = next(e for e in events if e['kind'] == 'pp4_reader_done')
    if done['depth'] != depth or done['rank'] != rank:
        raise ValueError('reader completion identity mismatch')
    count = math.ceil((depth - 1) / 2048)
    expected_order = [(k, layer) for k in range(count) for layer in (1, 14)]
    if [(e['chunk_index'], e['layer']) for e in buckets] != expected_order:
        raise ValueError('first-use bucket order changed')
    origin = catalog[0]['start_ns']
    previous_end = catalog[0]['ready_ns']
    if origin < 0 or previous_end < origin:
        raise ValueError('invalid catalog clock')
    found = {}
    selected_rows = cold_rows = 0
    for e in buckets:
        key = (e['chunk_index'], e['layer'])
        if key in found or e['rank'] != rank or e['depth'] != depth:
            raise ValueError('duplicate or wrong-shard bucket')
        if e.get('readers', count_readers) != count_readers:
            raise ValueError('bucket reader count mismatch')
        numeric = [e[k] for k in ('read_start_ns', 'read_end_ns', 'copy_end_ns', 'rows', 'payload_bytes')]
        if any(not isinstance(v, int) or v < 0 for v in numeric):
            raise ValueError('nonfinite or negative reader value')
        if (e['read_start_ns'] < previous_end or
                e['read_end_ns'] < e['read_start_ns'] or e['read_start_ns'] < origin or
                e['copy_end_ns'] < e['read_end_ns'] or
                e['p0'] != e['chunk_index'] * 2048 or
                e['width'] != min(2048, depth - 1 - e['p0']) or
                e['payload_bytes'] != e['rows'] * 264):
            raise ValueError('invalid reader timeline')
        if e.get('warm', False) is not warm:
            raise ValueError('bucket cache mode mismatch')
        if warm:
            pages, cached = e.get('selected_pages'), e.get('cached_pages')
            fraction = e.get('resident_fraction')
            if (type(pages) is not int or type(cached) is not int or not 0 <= cached <= pages or
                    (e['rows'] > 0 and pages == 0) or
                    not isinstance(fraction, (int, float)) or not math.isfinite(fraction) or
                    abs(fraction - (cached / pages if pages else 0)) > 1e-12):
                raise ValueError('invalid page residency accounting')
        cold = e.get('cold_rows', e['rows'] if not e['cached_parts'] else None)
        if (not isinstance(e['cached_parts'], int) or not 0 <= e['cached_parts'] <= e['rows'] * 2 or
                not isinstance(cold, int) or not 0 <= cold <= e['rows'] or
                not e['rows'] - cold <= e['cached_parts'] <= (e['rows'] - cold) * 2):
            raise ValueError('invalid cold row accounting')
        if not warm and ('cold_rows' in e or e['cached_parts']):
            ns = e.get('cold_read_ns')
            if not isinstance(ns, int) or ns < 0 or (cold > 0 and ns == 0) or ns > e['read_end_ns'] - e['read_start_ns']:
                raise ValueError('missing or invalid cold timing')
        selected_rows += e['rows']
        cold_rows += cold
        found[key] = (e['copy_end_ns'] - origin) / 1e9
        previous_end = e['copy_end_ns']
    if not selected_rows or (not warm and cold_rows / selected_rows < .9):
        raise ValueError('cold coverage below 90%')
    if not isinstance(done['elapsed_ns'], int) or done['elapsed_ns'] < previous_end - origin:
        raise ValueError('reader completion precedes final copy')
    if set(found) != {(k, layer) for k in range(count) for layer in (1, 14)}:
        raise ValueError('incomplete first-use buckets')
    return [[found[k, layer] for layer in (1, 14)] for k in range(count)]


def run_identity(root, manifest):
    binary = manifest.get('binary_sha256')
    if not binary and (root / 'binaries.sha256').exists():
        for line in (root / 'binaries.sha256').read_text().splitlines():
            digest, name = line.split(None, 1)
            if Path(name).name == 'mlx-serve':
                binary = digest
    return binary, manifest.get('corpus_sha256')


def summarize_depth(root, depth, controls, profiles, coverage, readers=None, warm=False):
    e = SSD[depth]
    # With both blocks, choose the faster median to keep the ratio conservative.
    if any(not math.isfinite(v) or v <= 0 for v in controls):
        raise ValueError('invalid control timing')
    control = min(statistics.median(controls[:6]), statistics.median(controls[-6:]))
    all_chunks, ready_by_rank = [], [[] for _ in range(4)]
    actual_demands, reader_evidence, reader_counts = [], [], set()
    for sample in profiles:
        chunks = chunk_costs(records(root / sample['trace']))
        expected = [(p0, min(2048, depth - 1 - p0)) for p0 in range(0, depth - 1, 2048)]
        if [(c['p0'], c['rows']) for c in chunks] != expected:
            raise ValueError('wrong prefill depth/chunk width')
        rows = records(root / sample['reader'])
        ready = deliveries(rows, depth, sample['rank'], readers, warm=warm)
        buckets = [e for e in rows if e['kind'] == 'pp4_bucket']
        selected = sum(e['rows'] for e in buckets)
        cold = sum(e.get('cold_rows', e['rows'] if not e['cached_parts'] else 0) for e in buckets)
        pages = sum(e.get('selected_pages', 0) for e in buckets)
        cached_pages = sum(e.get('cached_pages', 0) for e in buckets)
        reader_evidence.append(dict(rank=sample['rank'], rows=selected, cold_rows=cold,
            selected_pages=pages, cached_pages=cached_pages,
            resident_fraction=cached_pages / pages if pages else (0 if not any(e['cached_parts'] for e in buckets) else None),
            cached_parts=sum(e['cached_parts'] for e in buckets), cold_fraction=cold / selected if selected else 0,
            cold_read_s=sum(e.get('cold_read_ns', e['read_end_ns'] - e['read_start_ns']) for e in buckets) / 1e9))
        demands = [e for e in records(root / sample['trace']) if e['kind'] == 'pp4_engram']
        if not demands:
            raise ValueError('Engram demand trace missing')
        actual_demands.append(demands)
        ready_by_rank[sample['rank']].append(ready)
        reader_counts.add(next(e['readers'] for e in rows if e['kind'] == 'pp4_catalog'))
        all_chunks.append(chunks)
    if len(reader_counts) != 1:
        raise ValueError('mixed reader counts within depth')
    count = len(all_chunks[0])
    if any(len(c) != count for c in all_chunks):
        raise ValueError('chunk descriptor count changed')
    # Upper observed delivery over repetitions and separately replayed shard ranks.
    ready = [[max(run[k][j] for rank in ready_by_rank for run in rank)
              for j in (0, 1)] for k in range(count)]
    layers = [statistics.mean(c['layers_s'][i] for run in all_chunks for c in run) for i in range(20)]
    x = statistics.mean(c['x_s'] for run in all_chunks for c in run)
    head = statistics.mean(c['head_s'] for run in all_chunks for c in run)
    overhead = statistics.mean(c['overhead_s'] for run in all_chunks for c in run)
    profiled = [s['prefill_s'] for s in profiles if 'prefill_s' in s]
    pages = sum(e['selected_pages'] for e in reader_evidence)
    resident = (sum(e['cached_pages'] for e in reader_evidence) / pages if pages else
                (0 if all(e['resident_fraction'] == 0 for e in reader_evidence) else None))
    row = dict(cache_mode='WARM' if warm else 'COLD', resident_fraction=resident, control_coverage=coverage, source=str(root), control_samples=len(controls),
               reader_modes=sorted({s.get('reader_mode', 'concurrent_gpu_prefill') for s in profiles}),
               reader_counts=sorted(reader_counts),
               control_s=control, t_s=control / count, x_s=x,
               profile_prefill_s=statistics.median(profiled) if profiled else None,
               trace_inflation=statistics.median(profiled) / control if profiled else None,
               x_share=x / (control / count), head_s=head, overhead_s=overhead,
               x_components_s={k: statistics.mean(c[k] for run in all_chunks for c in run)
                               for k in ('compressor_s', 'decoder_s', 'dspark_s')},
               reader_evidence=reader_evidence, encoder_layer_s=layers,
               measured_local_delivery_s=ready, tp1_demands=actual_demands, cuts={})
    for cuts in CUTS:
        stages = stage_costs(layers, x, cuts, 83918848 / 3e9, overhead, head)
        bound = envelope(count, stages, e, .5, control)
        chunk_stages = [stage_costs(
            [max(run[k]['layers_s'][i] for run in all_chunks) for i in range(20)],
            max(run[k]['x_s'] for run in all_chunks), cuts, 83918848 / 3e9,
            max(run[k]['overhead_s'] for run in all_chunks),
            max(run[k]['head_s'] for run in all_chunks)) for k in range(count)]
        sim = simulate(chunk_stages, ready, 3 if cuts == '2/8/14' else 2)
        bound['first_use_projection'] = sim
        bound['tp1_with_delivery_s'] = max(bound['tp1_s'], sim['finish_s'] + .5)
        bound['tp4_s'] = bound['tp1_with_delivery_s'] + (.4 if depth == 32768 else .7)
        bound['speedup_with_delivery'] = control / bound['tp1_with_delivery_s']
        bar = 2.2 if depth == 32768 else 2.6
        bound['ratio_gate'] = 'PASS_PROJECTION' if bound['speedup_with_delivery'] >= bar else 'NO_GO'
        bound['tp4_ratio_gate'] = 'PASS_PROJECTION' if control / bound['tp4_s'] >= bar else 'NO_GO'
        row['cuts'][cuts] = bound
    return row


def summarize_runs(roots):
    roots = [roots] if isinstance(roots, Path) else list(roots)
    runs = [(root, json.loads((root / 'manifest.json').read_text())) for root in roots]
    identities = [run_identity(root, manifest) for root, manifest in runs]
    if len(runs) > 1 and (any(not all(identity) for identity in identities) or
                          len(set(identities)) != 1):
        raise ValueError('merge identity requires matching binary and corpus SHA256')
    output = {'status': 'PROJECTION_ONLY', 'prerequisites': [
        '#261 hardware ceiling pending', 'four simultaneous SSD/GPU/network proof pending',
        'TCP hop, row wire, full checkpoint/import remain plan allowances',
        'intermediate head work retained until skip proof; final head also covered by allowance'],
        'depths': {}, 'missing_depths': [], 'missing_reasons': {},
        'sources': [dict(directory=str(root), failure=manifest.get('failure'),
                         binary_sha256=identity[0], corpus_sha256=identity[1])
                    for (root, manifest), identity in zip(runs, identities)]}
    for depth, e in SSD.items():
        candidates = []
        for root, manifest in runs:
            samples = manifest['samples']
            controls = [s['prefill_s'] for s in samples if s['depth'] == depth and s['arm'] == 'control']
            profiles = [s for s in samples if s['depth'] == depth and s['arm'] == 'profile']
            if len(controls) not in (6, 12) or any(sum(s['rank'] == r for s in profiles) != 6 for r in range(4)):
                continue
            coverage = 'neighboring'
            if len(controls) == 6:
                positions = [i for i, sample in enumerate(samples) if sample['depth'] == depth]
                arms = [samples[i]['arm'] for i in positions]
                if arms[:6] == ['control'] * 6:
                    coverage = 'preceding_only'
                elif arms[-6:] == ['control'] * 6:
                    coverage = 'following_only'
                else:
                    raise ValueError('control block split around profiles')
            candidates.append((root, controls, profiles, coverage, manifest.get('readers'), manifest.get('warm', False)))
        if len(candidates) > 1:
            raise ValueError(f'overlapping depth {depth}; supply one evidence set per depth')
        if not candidates:
            output['missing_depths'].append(depth)
            output['missing_reasons'][depth] = f'{depth}: requires six profiles per shard and a six-sample control block'
            continue
        root, controls, profiles, coverage, readers, warm = candidates[0]
        try:
            output['depths'][depth] = summarize_depth(root, depth, controls, profiles, coverage, readers, warm)
        except (ValueError, OSError, KeyError) as error:
            output['missing_depths'].append(depth)
            output['missing_reasons'][depth] = f'{depth}: {error}'
    output['verdict'] = 'NO_GO' if output['depths'] and all(
        any(row['cuts'][c]['ratio_gate'] == 'NO_GO' for row in output['depths'].values())
        for c in CUTS) else 'HOLD_PENDING_PREREQUISITES'
    if output['missing_depths'] or any(manifest.get('failure') for _, manifest in runs) or any(row['control_coverage'] != 'neighboring' for row in output['depths'].values()):
        output['status'] = 'PARTIAL'
        output['observed_verdict'] = output['verdict']
        output['verdict'] = 'PARTIAL'
    return output


def summarize(roots):
    roots = [roots] if isinstance(roots, Path) else list(roots)
    groups = {False: [], True: []}
    identities = set()
    for root in roots:
        manifest = json.loads((root / 'manifest.json').read_text())
        warm = manifest.get('warm', False)
        if type(warm) is not bool:
            raise ValueError('invalid manifest cache mode')
        groups[warm].append(root)
        identities.add(run_identity(root, manifest))
    if len(roots) > 1 and (len(identities) != 1 or any(not all(i) for i in identities)):
        raise ValueError('merge identity requires matching binary and corpus SHA256')
    if not all(groups.values()):
        return summarize_runs(roots)
    result = summarize_runs(groups[False])
    result['warm'] = summarize_runs(groups[True])
    result['comparison'] = {}
    for depth in result['depths'].keys() & result['warm']['depths'].keys():
        cold, warm = result['depths'][depth], result['warm']['depths'][depth]
        if cold['reader_counts'] != warm['reader_counts']:
            raise ValueError('cold vs warm reader count mismatch')
        result['comparison'][depth] = dict(
            cold_reader_modes=cold['reader_modes'], warm_reader_modes=warm['reader_modes'],
            cold_resident_fraction=cold['resident_fraction'], warm_resident_fraction=warm['resident_fraction'],
            cuts={cuts: dict(cold_s=cold['cuts'][cuts]['tp1_with_delivery_s'],
                             warm_s=warm['cuts'][cuts]['tp1_with_delivery_s'],
                             delivery_bound_ratio=cold['cuts'][cuts]['tp1_with_delivery_s'] /
                                                  warm['cuts'][cuts]['tp1_with_delivery_s']) for cuts in CUTS})
    return result


def bound_table(result):
    text = [result['verdict'], '', f"Missing depths: {result['missing_depths']}", '',
            '| Mode | Depth | Resident pages | Cuts | X/t | TP1 projected s | Speedup | Ratio gate |',
            '|---|---|---:|---|---:|---:|---:|---|']
    for depth, row in result['depths'].items():
        resident = row['resident_fraction']
        fraction = f'{resident:.3%}' if resident is not None else 'unrecorded'
        for cuts, bound in row['cuts'].items():
            text.append(f"| {row['cache_mode']} | {depth} | {fraction} | {cuts} | {row['x_share']:.3%} | "
                        f"{bound['tp1_with_delivery_s']:.3f} | {bound['speedup_with_delivery']:.3f}x | {bound['ratio_gate']} |")
    text += ['', *[f'{d}: {reason}' for d, reason in result['missing_reasons'].items()], '',
             *[f"{d}: controls={row['control_coverage']}; reader modes={row['reader_modes']}"
               for d, row in result['depths'].items()]]
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('directories', type=Path, nargs='+')
    ap.add_argument('--out', type=Path, help='write bound files here; evidence can stay read-only')
    args = ap.parse_args()
    result = summarize(args.directories)
    destination = args.out or args.directories[0]
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'bound.json').write_text(json.dumps(result, indent=2) + '\n')
    text = ['# PP4 step 0', '', *bound_table(result)]
    if 'warm' in result:
        text += ['', '## WARM', '', *bound_table(result['warm']), '', '## Cold vs warm', '',
                 '| Depth | Cuts | Cold s | Warm s | Cold/warm bound |',
                 '|---|---|---:|---:|---:|']
        for depth, row in result['comparison'].items():
            for cuts, pair in row['cuts'].items():
                text.append(f"| {depth} | {cuts} | {pair['cold_s']:.3f} | {pair['warm_s']:.3f} | {pair['delivery_bound_ratio']:.3f}x |")
    text += ['', *result['prerequisites'], '',
             'Local shard replays share one SSD. Max cumulative delivery is a projection,',
             'not a measurement of four-box cadence. Layer timers synchronize the GPU;',
             'their perturbation is visible against uninstrumented controls.',
             'Residency is weighted by unique selected pages within each bucket, probed before reads.',
             'Cold vs warm compares recorded runs; reader-load modes above qualify the comparison.']
    (destination / 'bound.md').write_text('\n'.join(text) + '\n')
    print(json.dumps({'verdict': result['verdict'], 'bound': str(destination / 'bound.md')}))


if __name__ == '__main__':
    main()
