# Live Engram residency

Run this CPU-only probe beside a serving PID without a borrow. It does not load
weights, send requests, read table headers or payloads, change cache policy, or
stop a server. It verifies that both original table inodes are open by the
selected PID before and after probing. A PID or inode change refuses the result.

```text
offline identical pack -> first-use layout export -> small layout.json
                                                        |
live original files -> read-only mmap -> mincore -> unmap -> JSON
                       64 MiB windows
```

Build both standalone binaries locally, using the same Zig as the first-use
reader. Neither command needs MLX libraries:

```bash
zig build-exe src/pp4_residency_probe.zig -lc -O ReleaseFast -femit-bin=pp4-residency
zig build-exe src/pp4_first_use_bench.zig -lc -O ReleaseFast -femit-bin=pp4-first-use
zig test src/pp4_residency_probe.zig -lc --test-filter 'pp4 residency'
```

Export layout metadata once from a **non-serving copy of the identical pack**.
Set `OFFLINE_PACK` to that copy's absolute directory. The exporter reads only
safetensors headers, using the first-use reader's table opener. Keep the pack
immutable between exporting and probing. Stage the resulting JSON with the
probe binary and `scripts/bench/pp4_residency.py`. The live command refuses a
config SHA256, table-size, row-count, path, or layout mismatch:

```bash
./pp4-first-use layout "$OFFLINE_PACK" > pp4-engram-layout.json
```

The live runner is one command. Set `SERVER_PID` to the serving process and
`PACK` to its original iQ-MLX pack directory:

```bash
python3 scripts/bench/pp4_residency.py --pid "$SERVER_PID" --pack "$PACK" --binary ./pp4-residency --layout ./pp4-engram-layout.json --out live-residency.json
```

The default corpus is `~/bench/engram-corpus.md`; `--corpus` selects another
file. Output must be a fresh path. Failed runs leave no completed JSON. The
binary has a 384 MiB allocation buffer and a 105-second alarm. The runner caps
the child at 100 seconds, checks combined parent/child peak RSS below 512 MiB,
and rejects an elapsed time of 120 seconds or more. These are refusal limits;
measure actual runtime and RSS in the resulting JSON.

Each layer (1 and 14) records a whole-file fraction, whole code/scale region
fractions, and six exact sample selections. Each depth, 32768 and 131072, uses
three token offsets: 0, half the available start range, and its end. Corpus
length must permit three distinct offsets. The reader's tokenizer, compressed
token map, production hash and sorted unique `Catalog` select the first `N-1`
tokens of each slice with empty preceding history. Each sample includes its
offset, token digest and row count. It covers the complete request across all
ordinal quarters. Code and scale fractions deduplicate OS pages within each
part, including shared scale pages and row spans crossing page boundaries.

A bounded `mincore` scan produces one bitmap per table. Sample counts use that
bitmap, so hashing cannot warm or change its result. The JSON records each
scan's start/end epoch timestamps; traffic can change residency between windows
and tables. File paths, device/inode identities, page size, corpus/config/layout
SHA256, PID, elapsed time, and combined peak RSS qualify the capture. The probe
opens table files read-only and never dereferences their mappings.

Compare captured boxes with a later WARM window using the existing analyzer:

```bash
python3 scripts/bench/pp4_step0.py WARM_RUN --live-residency studio1.json --live-residency studio2.json --live-residency studio3.json --out analysis
```

`bound.json` retains every box's table/part/sample counts and adds depth-level
live/WARM comparisons. `bound.md` prints `live resident fraction = X vs warm-arm
resident fraction = Y` per box and depth. A cold-only input labels WARM
unmeasured. The analyzer requires matching corpus and encoded-token SHA256 values and complete,
consistent samples. Live counts deduplicate within a complete sample/part;
WARM counts deduplicate within each first-use bucket. Pages can recur across
buckets, so the two denominators are explicitly qualified. This comparison
helps choose a warm level; it does not change the projected pipeline bound.
