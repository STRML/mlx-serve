# PP4 step 0: prepare one-box evidence

Run `bash scripts/bench/pp4-step0.sh --dry-run` to inspect the command. It starts
nothing. The actual window runs only on studio4 inside studio1's single-box
borrow. Build, sign, and test before loading the sole private server. Keep the
live launcher and installed binaries untouched; the borrow returns S4 afterward.

```text
cold readers (all selected depths, before any server starts)
    |
    v
control block       profile block                       control block
6/depth        ->    6/depth/global ordinal quarter   ->  6/depth
                    sole server; readers already finished
                                  |
              exact corpus ids, C=2048, N=32768/131072
                                  |
                   trace.log + reader.jsonl
                                  |
                pp4_step0.py -> bound.json / bound.md
```

The source tree must include `src/`, `scripts/`, `build/`, `tests/`, `lib/`
tracked files, and `build.zig*`. A `git archive` of the committed tree is a
suitable source. The runner clones the matching integ-61c template
`~/build/integ-61c-runner-2335`, links
`~/opt/mlx-serve-integ-2335/lib/mlx`, and uses `~/opt/zig-0.17.1`.
`PP4_TEMPLATE` and `PP4_MLX_STAGE` can select another matching pair; the build's
stage-stamp check remains authoritative. Only studio4 builds or loads a model.
Allow 60–120 minutes including a cold build and three loads. The borrow has a
three-hour limit. Claude starts the window and arms its progress/completion
watchers; this preparation does not start it.

The private server binds loopback 18754. Prefix RAM/disk caches are disabled;
responses must report exactly N prompt tokens and zero cached tokens. The reader
tokenizes `~/bench/engram-corpus.md` with the same pack tokenizer and sends those
exact ids through `/v1/completions`. Kernel warm-up is discarded. Selected table
pages for the CPU reader are cold APFS clones, tested with `mincore`; the sole
server's own selected-page cache is uncontrolled and labeled separately. There
is no global purge or memory-pressure probe.

Before loading a model, the runner checks all 75 inputs (72 measured samples
and three warm-ups) within a 30-second budget including tokenization. Each input
gets a deterministic corpus offset, with collisions resolved until its first
token differs from every other input. The check builds every full slice and
refuses repeated first-32-token prefixes. Reuse a tokenized corpus to run the
same check offline on the Mac:

```bash
python3 scripts/bench/pp4_window.py --check-tokens CORPUS.u32
```

`sample-plan.json` records the offsets for the full schedule. Every sample and
warm-up also writes `input.json` and `tokens.u32`; the manifest retains its
`token_offset` and token SHA256. The HTTP request and CPU reader consume that
same slice. Disabling prefix caches alone does not clear a slot's live decode
state, so the zero-cached-token response guard still applies to every request.

`MLX_SERVE_DIAG=prefill-trace` supplies the control's production prefill time.
The `pp4-step0` diagnostic adds synchronized layer timings, layer-20 compressor,
decoder, DSpark commit/evaluation, head, and stock Engram read/ready/demand
clocks. Encoder compute excludes measured row wait. Native DSpark must be
present. Actual chunk descriptors include the final 2,047-row chunk. Only
2,048-row requests are qualified here; the plan's later 4,096-row lead/subchunk
case needs a separate descriptor mapping.

The [first-use reader](pp4-first-use-reader.md) executes every selected sample
on fresh table clones before starting any server. Each bucket probes both codes
and scales, times fully cold rows separately, then reads cached rows and restores
the sorted payload order. A completion marker requires at least 90% fully cold
rows across that reader sample. Buckets report `cached_parts`, `cold_rows`, and
`cold_read_ns`, including on a coverage failure. The manifest records
`reader_mode=before_request_idle_gpu`. These reads measure an idle GPU; the
older concurrent readers remain labeled `concurrent_gpu_prefill` in analysis.
`memory.jsonl` samples free+inactive, server RSS, and `top`'s footprint. The
watchdog aborts the owned processes below 16 GiB or on a foreign model process.

## Recompute the bound

```bash
python3 scripts/bench/pp4_step0.py ~/bench/pp4-step0-YYYYMMDD-HHMMSS
```

The analyzer validates each depth independently: six profiles per rank, a full
six-sample control block, complete chunks and encoder layers, native DSpark/X
fields, ordered reader buckets, at least 90% fully cold rows, and a completion
marker. A failed window retains its valid depths as `PARTIAL` and names missing
depths and reasons. A single control block is labeled `preceding_only` or
`following_only`; the overall result remains partial even after both depths
are present. Corrupt evidence excludes its depth from the bound.

For each depth it uses the faster available control-block median `T1`,
`M=ceil((N-1)/2048)`, `t=T1/M`, and reports `X/t`, where
`X=compressor+decoder+DSpark`. It sums actual layer compute for cuts 2/8/14 and
8/14/20. Residual embedding/preparation stays on stage 0. Every measured head
stays on stage 3 until a skip proof exists. Each outgoing stage pays the plan's
83,918,848 B / 3 GB/s hop. The scalar envelope remains:

```text
q = max(e, d0, d1, d2, d3)
Tpipe = e + sum(d) + (M-1)*q
TtoTP1 = Tpipe + 0.50s
TtoTP4 = TtoTP1 + 0.40s (32k) or 0.70s (131k)
e = 0.2411s (32k), 0.1593s (131k), measured four-SSD report
```

A second flow-shop projection takes the maximum observed cumulative local
completion for each first-use bucket across ranks/repetitions, includes real
hash/catalog/cache-probe/copy time, and enforces two-chunk credits. Stage 0 waits
for layer 1; layer 14 gates stage 3 or stage 2 according to the cuts. The larger
of this projection and the scalar envelope supplies the reported ratio. The
output preserves per-chunk demand/late-delivery times and raw TP1 demand clocks.

Sequential quarters on distinct corpus slices and one SSD are a surrogate for
four simultaneous SSDs. Catalog digests may differ between samples; the bound
takes the maximum observed delivery across those samples.
The hop, row wire and final-state/import costs remain allowances. Per-layer
synchronization perturbs execution; compare profile/control time before relying
on the projection. This window does not complete #261's hardware ceiling or
prove four-box coexistence. A schedule that misses 2.2x at 32k or 2.6x at131k
is `NO_GO`. Passing projected ratios produce `HOLD_PENDING_PREREQUISITES`.
TP4 continuation is evaluated independently. No executor is authorized by a
passing projection alone.

## Validation

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/bench/test_pp4_step0.py
bash -n scripts/bench/pp4-step0.sh
zig test src/plugins/deepseek_v41/pp4_step0.zig -lc
zig test src/pp4_first_use_bench.zig -lc --test-filter 'pp4 first use'
zig build check -Doptimize=ReleaseFast
zig build test -Doptimize=ReleaseFast -Dtest-filter='pp4 step0'
zig build test -Doptimize=Debug -Dtest-filter='pp4 step0'
```

| State/input | Operation | Failure | Result/proof |
|---|---|---|---|
| Dry-run | Print borrow and configuration | Accidental execution | Offline command test; no remote work |
| Wrong host/borrow | Refuse before build/load | Second server | Offline refusal check |
| Missing/duplicate layer/X | Reject artifact | Optimistic tail bound | Parsed fixture and X mutation |
| Missing/reordered reader data | Reject artifact | Early delivery from corrupt timeline | Full artifact fixture and completion mutation |
| Head/credits | Charge all measured work; pace storage | Head omitted or unbounded read-ahead | Numeric pipeline oracle and head mutation |
| First memory poll fails | Publish failure, stop owned processes | Silent watchdog death | Injected vm_stat failure, red before fix |
| Reader/request fails | Join/reap owned work; invalidate window | Orphan process or partial result | Bounded cleanup; actual window still pending |
| Memory/foreign load | Abort owned processes | Host pressure/contamination | Runtime watchdog, never a pressure test |

The physical fixture compares logits and committed DSpark rings with and
without synchronized diagnostics. Tests and dry-run establish preparation
correctness; measured performance remains pending the window.

## Resume one depth and merge evidence

The shell forwards `--depths` and `--readers` to the window. Reader counts are
1–32, default 32, recorded in the manifest and checked against each catalog.
A 131k-only run has 39 inputs
(36 measured requests and three warm-ups), all at 131,072 tokens. Reuse the
original signed server with `PP4_BINARY` so the server SHA stays identical; the
shell builds the updated reader separately. Run only inside the same Studio
borrow described above:

```bash
PP4_BINARY=EXISTING_RUN/build/zig-out/bin/mlx-serve \
  bash scripts/bench/pp4-step0.sh --run --depths 131072 --readers 32
```

`EXISTING_RUN` is the absolute path of the original run. Reader binaries may
differ because the cold-page policy changed. The analyzer requires identical
server-binary and corpus SHA256 values across inputs and refuses overlapping
complete depths. Older manifests use `binaries.sha256` for the server identity.
A failed window also gets `bound.json` and `bound.md`; the shell preserves the
window's failure exit status.

Read-only copies stay untouched when `--out` selects another directory:

```bash
python3 scripts/bench/pp4_step0.py /private/tmp/claude/pp4-run2 \
  --out /private/tmp/claude/pp4-run2-analysis
python3 scripts/bench/pp4_step0.py /private/tmp/claude/pp4-run2 RUN_131K \
  --out /private/tmp/claude/pp4-merged-analysis
```

`RUN_131K` is the copied 131k-only evidence directory. The merged output keeps
source failures, one-sided controls, and reader-load modes visible. Cold-only
read time excludes cached-row reads; the first-use simulation uses full bucket
completion, including cached rows and copies. Cached-row timings do not establish
an all-cold delivery guarantee.

## Offline reader and layer-cut sensitivity

```bash
python3 scripts/bench/pp4_levers.py /private/tmp/claude/pp4-run2 \
  --out /private/tmp/claude/pp4-levers.json
```

The 32k sensitivity uses measured decoder layers 20–39 and searches contiguous
four-stage partitions. Layer 1 stays on stage 0. The objective is the larger
of the scalar envelope and the conservative per-chunk delivery simulation.
Encoder costs take each layer's maximum across repetitions/ranks at that chunk;
decoder segments take the maximum of their measured sum. An intact decoder
retains the original maximum X, so existing cuts reproduce the original bound.
Compressor work follows layer 20; DSpark and the head remain on stage 3.

Only bucket read intervals scale with the single-SSD 32k rows/s curve from the
Engram scaling report. Six-reader throughput is linearly interpolated between
four and eight readers, per table. Preparation, probes, copies and inter-bucket
gaps stay fixed. Catalog start is request arrival; two-chunk credits still apply.
The measured four-SSD storage floor stays 0.2411 s/chunk. No additional factor
of four applies to timelines that already describe a quarter per box.

The output anchors to the artifact's recorded reader count. It also includes
a conditional six-reader anchor to check that premise separately. Run 2 records
32 readers, with six repetitions per rank. Ratios against a six-reader anchor
therefore describe a hypothetical source measurement.

Arbitrary cuts need shared compressed KV, index keys, picks and candidate state
in addition to activations. Those transfers are unmeasured and omitted from
this optimistic sensitivity. `best_activation_only` restricts boundaries to
2/8/14/20; `best_decoder_cut` requires a boundary inside layers 20–39. Neither
output implements a distributed executor or qualifies its memory use.
