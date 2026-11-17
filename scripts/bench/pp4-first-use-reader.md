# PP4 first-use row reader

This CPU-only companion measures sorted first-use Engram buckets before the
window starts its model server. It reuses the production hash,
compressed-token map and `Table.readSorted`; cold-page probing and tokenizer CLI
come from `engram_scale_bench.zig` on `perf/engram-read-scale`.

```bash
zig build-exe src/pp4_first_use_bench.zig -lc -O ReleaseFast \
  -femit-bin=pp4-first-use
zig test src/pp4_first_use_bench.zig -lc --test-filter 'pp4 first use'
zig test src/pp4_first_use_bench.zig -lc -O ReleaseFast \
  --test-filter 'pp4 first use'

pp4-first-use tokenize PACK CORPUS TOKENS.u32
pp4-first-use CLONED_PACK TOKENS.u32 32768 32 0 2048
```

The second command reads one global sorted ordinal quarter (`rank` 0–3).
Run every selected sample and quarter sequentially before any server starts. `CLONED_PACK` must hold fresh APFS clones of the two table files and the
source config/token map. Before every bucket, `mincore` classifies selected code/scale spans. A row is
fully cold only when both parts have zero resident pages. The reader times fully
cold rows separately, reads cached rows afterward, and restores the original
sorted payload order. Direct reads do not fill those caches. A reader completes
only with at least 90% fully cold rows across both tables and all chunks; it
prints every bucket and a coverage record before refusing lower coverage. Clone cleanup belongs to the window.

For the warm window, run `pp4-first-use PACK TOKENS.u32 32768 32 0 2048
--warm` (one command). Use the original pack after the server's distinct
real-text prefills. Warm mode opens buffered tables, probes residency, and reads
each sorted bucket once as-is. It omits the 90% cold-coverage gate. The catalog
and buckets record `warm=true`; `cached_parts` remains an integer span count.
`cached_pages / selected_pages` gives `resident_fraction`, deduplicating OS pages
within the bucket before any reads. Fully cold rows remain reported, while
`cold_read_ns=0` because warm mode does not time a separate cold subset.

The window passes a sample's offset slice as its own `tokens.u32`, identical to
the HTTP prompt. The reader uses the first `N-1` exact tokenizer ids, preserving
full preceding mapped history. Each table deduplicates the whole request and
retains each row's earliest chunk. It assigns contiguous sorted ordinal quarters
before ordering that quarter by first-use chunk, then row. Buckets execute chunk 0 layer 1,
chunk 0 layer 14, chunk 1 layer 1, and so on. One pool of at most 32 readers runs
at a time. The window's `--readers N` selects 1–32 (default 32). Each catalog
and bucket records that cap. All selected shard bytes remain in owner buffers
until exit.

Only 2,048-token chunks are supported. The 4,096-row path's stock lead split needs
its own descriptor mapping before qualification.

| JSONL kind | Fields |
|---|---|
| `pp4_tokens` | Exact encoded count and corpus byte count |
| `pp4_catalog` | Depth, `prefill_tokens=N-1`, rank, chunk width, readers; start/ready timestamps, hash/catalog/prepare ns; global/shard row counts and digests, owner bytes, token digest |
| `pp4_bucket` | Rank/table/chunk, `p0`, width; rows/bytes; `cached_parts`, `cold_rows`, `cold_read_ns`; raw read timestamps, local-copy completion, elapsed/probe/read/copy ns; row and payload digests |
| `pp4_reader_coverage` | Total selected/cold rows and required fraction (0.9), emitted before qualification |
| `pp4_reader_done` | Depth, rank and total elapsed ns; absent on failure |

`*_ns` timestamps use the same awake monotonic clock on this host.
`start_epoch_ns` / `read_start_epoch_ns` provide Unix-clock correlation for the
server log. `elapsed_ns` starts at catalog readiness and includes cache probes,
reads, host copies and prior bucket/report overhead. `prepare_ns` precedes it.
The copy timer measures one real local host copy into retained owner storage.
It excludes network, dequantization and scatter into model spans.

These are one-SSD measurements with the GPU idle. Older schema-1 concurrent
captures retain their original load qualification. A max of sequential quarter
completion timelines is a modeled four-SSD input. Keep that projection, cable
transport, two-chunk read credits and other boxes' speed separate from measured
claims. First-use service cannot be inferred by dividing whole-request time by
four or by averaging per-chunk times.

| Input/state | Operation | Failure | Caller result/test |
|---|---|---|---|
| Cross-chunk repeated row | Dedup whole request; earliest use wins | Assign latest use/drop first row | Catalog oracle checks rows and earliest chunks |
| Sorted ordinal quarters | Assign before first-use ordering | Per-chunk re-shard/overlap/omission | Uneven fixture proves exact global quarters |
| First-use order | Read chunks monotonically | Reverse chunk comparator | Schedule oracle exposes reordered rows |
| Cached selected span | Partition fully cold rows; read both subsets | Count cached rows as cold or drop bytes | Warm/cold payload oracle and 90% schedule coverage guard |
| Read/output failure | Abort utility | Short table read or stdout write | Nonzero exit; no completion record |

Catalog and schedule tests protect earliest use and global shard ownership.
Reader tests check exact cold/cached payload bytes, separate cold timing, and the
90% coverage boundary. Window tests enforce all readers finishing before server
startup. Existing production reader tests own concurrency and byte ordering.
