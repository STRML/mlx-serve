# DeepSeek-V4.1 verify MoE: read each picked expert once per verify

Status: spec, nothing built. Scope: the fused mxfp4 decode MoE (`decode/moe_fp4.zig`, `kernels/dec_moe_fp4_{up,down}{,_bw}.metal`)
for R = 2..6 rows (a DSpark verify; `decode.max_rows` is 6). "Shared" below means an expert two or more rows picked, not the
model's always-on shared expert, which every row already reads through one simdgroup.

## What today's kernels read

- Gate/up (`dec_moe_fp4_up[_bw]`): one simdgroup per (output column, slot), slots `0 .. TOPK*R-1` the (row, pick) pairs.
  Each slot decodes its expert's gate and up codes for that column from memory, so an expert picked by k rows is read k times.
  The grid is column-major with a column's slots adjacent, so the k reads of one tile land close in time.
- Down (`dec_moe_fp4_down[_bw]`): a threadgroup per (column group, row), simdgroup j the row's j-th pick. The BW layout already
  puts a column group's R rows adjacent in the grid "so the rows' reads ... of the experts they share come from the cache".
  So part of the duplicate traffic may already be served by the cache. The microbench below measures how much.

Per expert, per layer, per TP4 rank (each rank holds a quarter of every expert's columns; mxfp4 is 0.5 B a weight plus a
1-byte scale per 32):

| Part | Weights | Bytes |
|---|---|---|
| gate + up | 2 x 2304 x 5120 / 4 = 5.90 M | 2.95 MB + 0.18 MB scales |
| down | 5120 x 2304 / 4 = 2.95 M | 1.47 MB + 0.09 MB scales |
| one expert | | ~4.70 MB |

A row's 6 picks are ~28.2 MB a layer per rank, ~1.13 GB over 40 layers if every layer routes (my arithmetic from the config,
not measured).

## What it would save

Measured duplicate share of a verify's (row, expert) picks (four Studios, `dsv41-verify-picks`, 2026-11-09): 2 rows 13%,
3 rows 21-29%, 4 rows 27-28%, 5 rows 30-32%, 6 rows 33-39%; 27-32% overall.

Reading each distinct expert once removes that share of the routed bytes:

| R | dup share | routed reads a verify (rows' worth) | saved (rows' worth) | saved per rank |
|---|---|---|---|---|
| 2 | 13% | 2 | 0.26 | ~0.29 GB |
| 4 | 27-28% | 4 | 1.1 | ~1.2 GB |
| 5 | 30-32% | 5 | 1.5-1.6 | ~1.7-1.8 GB |
| 6 | 33-39% | 6 | 2.0-2.3 | ~2.2-2.6 GB |

In time: if routed reads run at the review's estimate of ~2.1 ms a row (Fable 5.2, a code-read estimate), a 5-row verify
saves up to ~3.2 ms and a 6-row ~4.5 ms. That is an upper bound. It assumes today's duplicate reads go to DRAM; any that the
BW layout already serves from the cache are not there to save.

## Exactness

Each row's output must be bitwise today's, for every R, both `DSV41_MOE_BW` arms, finite and non-finite inputs:

- **Gate/up:** row r's pick j value for column n is today's `dec_fp4_up_pair` (or `dec_fp4_up_routed`) over `x[r]`: the same
  K split across lanes, the same simd reduction tree, the same `dec_up_act` with `wts[r*TOPK+j]`, written to the same
  `out[(r*(TOPK+1)+j)*N+n]`. Reading the weight tile once and running that same per-lane, per-row sequence for each row that
  picked it keeps every float op unchanged.
- **Down:** each (row, slot) partial must come from the same lane/K mapping (`lane / NC`, stride `32 / NC`) and the same
  `simd_shuffle_xor` tree. The slots must then sum in today's order: j = 0 .. TOPK-1, then the shared expert. Then the same
  `T(s)` rounding, and the same `fma` chain for the hc combine.
- **No arrival-order sums.** Partials from different threadgroups go to fixed slots in a scratch `[R, TOPK+1, N]` f32
  buffer. They are summed by a fixed-index loop, either in a second launch or by the last arriver, as `dec_wo_a` does with
  its counter. Never with atomic float adds.

## Shape of the kernel

- **Same static grid as today.** The thread group for slot u = (r, j) owns expert `e = inds[u]` only if no slot u' < u (row
  major) has `inds[u'] == e`: at most 35 comparisons in registers, no extra pass and no host sync. A slot that doesn't own
  its expert exits.
- **The owner works per tile, not per row.** It decodes each K tile of its column once, then for each slot s >= u with
  `inds[s] == e` it accumulates row `s / TOPK`'s dot product in today's lane order. It writes each slot's result where today
  writes it.
- **Register budget.** The owner keeps one accumulator pair per row it serves (at most R = 6). That fits; today's shared-expert
  simdgroup already keeps `acc[R][2]`.
- **Load balance.** An expert picked by all 6 rows does 6x the arithmetic of a singleton and reads the same bytes. If that
  stalls the tail of the launch, order owners by multiplicity in the grid, or split a heavy owner's rows across two
  simdgroups that both read the tile (still fewer reads than today).
- **Down:** the same owner rule per (column group, expert). The row combine moves to a fixed-order pass over the scratch
  partials.

## With TP4 expert sharding

Routing is identical on every rank (the gate is whole, the ids are computed from the same summed stream). So the duplicates,
the owner choice and the grid are identical too, and every rank saves the same share of its own quarter. The rank's MoE
output (`moeFused` -> `tpSum`) and its shape are unchanged, so the collectives keep their order and shape. The DSpark stages
(`li >= n_layers`, 128 experts, top-3) are not in scope; their 5-row drafts would need the same rule over TOPK = 3.

## Microbench on one Ultra, before any window

Follow the `DSV41_DEC_UBENCH` pattern (`decode/fixture.zig` kernel ubench against its byte floor). Use TP4 rank-slice shapes
(K = 5120, N = 576 for gate/up; K = 576, N = 5120 for down) and R = 2..6.

- **Weights:** cycle 80 distinct weight sets (the house rule: a read-level bench on one set stays in cache and flatters the
  kernel).
- **Today's kernels (both BW arms), at each R, three id patterns:**
  - all 6R picks distinct;
  - the measured duplicate share for that R, drawn from recorded `dsv41-verify-picks` id lists;
  - every row picking the same 6 experts.
- **What it shows:**
  - The time drop from "distinct" to "measured" is what the cache already saves today.
  - The bound for the new kernel is today's measured-share time minus distinct-bytes / bandwidth (the byte floor of the
    distinct experts alone).
  - If the cache already closes most of that gap, stop here.
- **The prototype:** at the same three patterns. Bit-equal to today on every row via the kernel contract harness
  (`kc.Set`, as `moe_fp4.zig` registers `up_l` / `down_l`). Report time against today and against the distinct-bytes floor.
- **Gate to a window:** the measured-share time beats today's by at least 5% at R = 5 and 6 on the Ultra. Then a four-Studio
  verify-only A/B with `DSPARK_ROWS` pinned.
