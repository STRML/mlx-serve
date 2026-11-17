# Measure Engram read scaling

This disk-only companion to plan 75's Tier 1 bench uses the production Engram
hash and `Table.readSorted`. It loads the pack config and token map, then reads
only the selected code and scale rows of layers 1 and 14. It initializes neither
MLX nor a model, and opens no network sockets.

Build with the repository's pinned Zig compiler on the Mac or a borrowed Studio:

```bash
zig build-exe src/engram_scale_bench.zig -lc -O ReleaseFast \
  -femit-bin=engram-scale
zig test src/engram_scale_bench.zig -lc --test-filter 'engram scale'
zig test src/engram_scale_bench.zig -lc -O ReleaseFast \
  --test-filter 'engram scale'
```

Create real-text ids with the pack's tokenizer:

```bash
./engram-scale tokenize /absolute/pack /absolute/corpus.md /absolute/corpus.u32
```

An arm creates APFS clones of the two table files. Each clone has its own inode
and a fresh page cache while sharing the source's physical disk extents. The
binary requires `mincore` to report zero cached selected code and scale spans
before reading. Aligned 4 KiB `F_NOCACHE` reads use the same production path as
Tier 1. This avoids `sudo purge` and a memory-filling eviction workload. The
runner deletes only its temporary clones when it exits. A populated output
directory is refused so stale barrier files cannot release a new arm early.

Run an arm through a Studio borrow or Latch as the Studio rules require:

```bash
python3 scripts/bench/engram_scale.py \
  --binary /absolute/engram-scale --pack /absolute/pack \
  --tokens /absolute/corpus.u32 --depth 32768 --readers 16 \
  --out /absolute/new-results-directory
```

`--warm-control` caches up to 64 MiB from each clone's code and scale regions. The binary must then
fail with `CachedTablePages`. This verifies that `F_NOCACHE` alone cannot make
preexisting cache pages count as cold.

Stage `engram-scale`, `engram_scale.py` and `corpus.u32` under
`~/bench/engram-scale` on each Studio. The sweep and four-box coordinator use
the iQ pack's standard Studio path. Launch the single-box sweep from studio1:

```bash
. ~/bench/cluster-addr.sh
addr=$(tb_addr studio4 studio1)
bash ~/bench/ds41-tp2x2.sh borrow studio4 --max 7200 -- \
  ssh -o BatchMode=yes "$addr" \
  'bash ~/bench/engram-scale/engram-scale-sweep.sh ~/bench/engram-scale/single'
```

After that borrow returns, run `engram-scale-four.py --out NEW_DIRECTORY` on
studio1. It queues one finite disk job per box through Latch, waits for all four
to be admitted and ready, then releases them at a common timestamp. If Latch's
memory reserve prevents admission beside live servers, borrow all four boxes
and use those holds instead of nesting Latch:

```bash
bash ~/bench/ds41-tp2x2.sh borrow studio1,studio2,studio3,studio4 \
  --both-down --max 1800 -- /usr/bin/python3 \
  ~/bench/engram-scale/engram-scale-four.py --borrowed \
  --out "$HOME/bench/engram-scale/four"
```

`--borrowed` requires all four boxes in the enclosing borrow environment. Each box
reads its disjoint quarter of each layer's sorted distinct row list. These
disk-only jobs may run beside the live servers within the 4 GiB budget. An arm
fails if any box fails. An error releases other workers from the start barrier
with an abort marker and waits for their cleanup.

Every arm saves raw JSONL, source and clone inode/size provenance, token and
binary SHA-256, process peak RSS, and timing boundaries. JSONL records include
selected-row and payload checksums, cold-page counts, row counts and bytes.
The timer covers output allocation, thread creation, reads, copies and joins;
hashing, sorting, `mincore`, payload checksums and clone creation sit outside it.

The default scope deduplicates across the whole prompt, matching request-wide
read-ahead. `--chunk 2048` deduplicates each 2,048-token chunk separately. For
the whole-prompt scope, milliseconds per 2,048 tokens is the layer's read time
divided by `depth / 2048`. Rows/s counts distinct table rows, each 264 payload
bytes for this pack. MB/s is decimal payload throughput, not physical SSD
traffic. Four-box aggregate throughput uses total rows divided by the slowest
box's time, so an uneven or slow box stays visible. Network transport to the
stage, dequantization and GPU compute are outside this disk benchmark.

Collect the single sweep into `RAW_ROOT/single-final` and the four-box coordinator
output into `RAW_ROOT/four-final`, excluding any `cold-*` directories. Run
`python3 scripts/bench/engram-scale-analyze.py RAW_ROOT` to verify completeness,
cold pages, checksums and shard coverage, then write measurement tables and a
JSON summary. Older compiler runs and failed launches belong outside these paths.
