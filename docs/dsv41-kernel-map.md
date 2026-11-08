# DeepSeek-V4.1 decode: a kernel map of one TP4 rank (M5 Max)

Where one decode step of one tensor-parallel rank (TP4 widths, 384 mxfp4 experts, context 2048, sums stubbed as an
identity) goes, kernel by kernel, at 1 and 6 rows. The 40 layers run as two fixtures (layers 0-20 and 20-40; a
40-layer fixture swaps on a 128 GB Mac); the tables add the halves.

## Where the step goes

At 1 row the three groups that had no name, hc stages, router and attention core, take about 27% of the busy time
(13% + 8% + 6%). The rest of the gap to the ubench per-launch numbers is that the named kernels cost more in the step
than alone: wo_a 35.8 us against 22.2, wo_b 35.1 against 23.9, in_proj 38.3 against 24.9, wq_b 39.8 against 25.8, MoE up
59.1 against 49.6, down 37.7 against 30.3 (scaling by the stock busy time brings the MoE launches to the ubench numbers
and leaves the projections 30-40% above them). On the 8 index-owner layers `wq_b` runs at 46-55 us against 30-36 on the
others. At 6 rows `hc_split` grows to 62 us a launch, 8.8% of the step, and the MoE takes 52%.

## Method

`src/plugins/deepseek_v41/decode/kernel_map.zig` builds the fixture, then runs `workerStep` as the serving loop does (a
layer handed to the GPU as it is built) for 3 warm-up and 32 timed steps per row count, on fresh random inputs.
`scripts/kernel_map_trace.sh` attaches `xctrace record --template 'Metal System Trace'` after the setup, and
`scripts/kernel_map.py` / `scripts/kernel_map_report.py` read the `metal-gpu-intervals` table.

Two builds of MLX, the same step in both:
- stock: one compute encoder a layer. It gives the real span, GPU busy time and the idle between encoders.
- labelled (`docs/dsv41-kernel-map.mlx-labels.patch`, `MLX_MAX_OPS_PER_BUFFER=0`): one primitive a command buffer, each
  encoder labelled with the kernels it dispatched, so every dispatch is its own interval with its name. No result
  changes; the extra command-buffer gaps are not in the kernel times (the serialized span is 1.8x the real one at 1
  row, 1.2x at 6).

Idle: the stock trace has one encoder a layer, so the idle in the tables (8.2% at 1 row, 5.4% at 6) is the GPU waiting
between layer encoders. Gaps between dispatches inside a layer cannot be seen in `metal-gpu-intervals`; they sit in the
busy time. The labelled run's gaps are the one-command-buffer-a-primitive cost and are not idle of the real step. A
`share` is a share of the labelled kernel sum (the GPU busy time), not of the span. No other process but WindowServer
used the GPU inside any stretch used here.

The labelled kernels sum to 108-121% of the stock busy time. The `scaled` column spreads the stock busy time over the
kernels by their share. Two repeats of each: shares move 1-2 points; at 1 row the labelled total moved 13.0 to 15.3 ms
(the GPU ran slower in the second repeat), at 6 rows 28.2 and 27.9 ms. Quote shares, and absolute microseconds to
about 15%.

## What is not in the fixture

The pair worker holds no embed, head or Engram: layers 1 and 14 (Engram) run as plain layers, and the head's gather and
the hc head are absent. `DSV41_KMAP_HEAD=1` times the two linears that cost bandwidth, 16 launches evaluated together
over 4 weight sets (each streams from DRAM), min..median per launch:

- head (rank 0's quarter of the vocabulary, 8-bit affine, norm + matmul + f32 cast): 1 row 296..369 us, 6 rows 328..367 us.
- Engram wkv, per Engram layer (mxfp8 6144 -> 25600, uncut by the tensor split, a guess at its quant): 1 row 299..342 us,
  6 rows 457..516 us. Its table gather (host, SSD) and gate kernel are not measured.

As group rows, standalone estimates (not in the fixture's step): head 0.30-0.37 ms, Engram 2 x 0.30-0.34 ms at 1 row;
at 6 rows head 0.33-0.37 ms, Engram 2 x 0.46-0.52 ms. The Engram quant is a guess.

A real 1-row step therefore adds about 0.3 ms (head) + 2 x 0.3 ms (Engram wkv) = 0.9-1.0 ms to the tables below, plus
the head's all-gather.

## Layers

`layer:r<compress ratio><K=KV source><I=index source><E=Engram>`, printed by the test:

    0:r0--- 1:r0--E 2:r2KI- 3:r2--- 4:r2--- 5:r2--- 6:r2--- 7:r2--- 8:r2KI- 9:r2--- 10:r2--- 11:r2--- 12:r2--- 13:r2---
    14:r2KIE 15:r2--- 16:r2--- 17:r2--- 18:r2--- 19:r2--- 20:r1KI- 21:r1--- 22:r1--- 23:r1--- 24:r1-I- 25:r1--- 26:r1---
    27:r1--- 28:r1-I- 29:r1--- 30:r1--- 31:r1--- 32:r1-I- 33:r1--- 34:r1--- 35:r1--- 36:r1-I- 37:r1--- 38:r1--- 39:r1---

The index-owner layers (wider `wq_b`, the index scorer and top-k) are 2, 8, 14, 20, 24, 28, 32, 36: 8 of 40. Layers 2, 8,
14 and 20 also source the KV cache.

## Run

    zig build test-build -Doptimize=ReleaseFast -Dtest-filter="dsv41 kernel map"
    scripts/kernel_map_trace.sh OUT stock_a 0-20 stock;  scripts/kernel_map_trace.sh OUT stock_b 20-40 stock
    KMAP_MLX_LIB=<labelled lib dir> scripts/kernel_map_trace.sh OUT patched_a 0-20 patched   # and 20-40
    scripts/kernel_map_report.py --rows 1 --seg 4 --patched A.trace:B.trace[,A2:B2] --stock A.trace:B.trace[,..]
    # --seg 4 is the 1-row phase of the test, --seg 1 the 6-row phase

## 1 row

```
rows = 1, 40 layers (layers 0-20 + 20-40), per step, mean of 2 labelled and 2 stock runs
stock MLX runs: span 13133 busy 12030 us; span 13797 busy 12697 us
real step (stock MLX): span 13465 us, GPU busy 12364 us (91.8%), idle 1101 us (8.2%), 40 encoders
labelled runs: kernels sum to 13040, 15344 us in 544 dispatches; serialized span 24057 us; scale to the real busy time x0.871
kernel                             launches   us/step   scaled us/launch  share  group
dsv41_dec_moe_fp4_up_bw                40.0      2363     2058      59.1  16.6%  MoE
dsv41_dec_wq_b                         40.0      1592     1387      39.8  11.2%  attention projections
dsv41_dec_in_proj                      40.0      1532     1334      38.3  10.8%  attention projections
dsv41_dec_moe_fp4_down_bw              40.0      1511     1316      37.7  10.6%  MoE
dsv41_dec_wo_a_bw                      40.0      1431     1247      35.8  10.1%  attention projections
mxfp8_qmv_fast (wo_b)                  40.0      1404     1223      35.1   9.9%  attention projections
dsv41_dec_attn                         40.0       891      776      22.3   6.3%  attention core
dsv41_hc_split                         40.0       854      744      21.4   6.0%  hc stages
dsv41_hc_stage_x                       40.0       811      707      20.3   5.7%  hc stages
dsv41_route                            40.0       585      510      14.6   4.1%  router
dsv41_gate                             40.0       564      491      14.1   4.0%  router
glm5_hc_expand                         40.0       201      175       5.0   1.4%  hc stages
gg1_copybfloat16bfloat16               45.0       175      152       3.9   1.2%  norms/rope/casts/KV writes
dsv41_dec_topk                          8.0        96       84      12.0   0.7%  indexer
dsv41_dec_idx_score                     8.0        88       77      11.0   0.6%  indexer
dsv41_dec_idx_pool2                     1.5        56       48      37.0   0.4%  compressor
(4 more)                                1.1        39       34             0.3%
groups (us/step, scaled, dispatches, share, spread across the labelled runs):
  attention projections              5959 us    5192 scaled   160.1 dispatches  42.0%  [5469..6449]
  MoE                                3873 us    3374 scaled    80.0 dispatches  27.3%  [3675..4071]
  hc stages                          1866 us    1626 scaled   119.9 dispatches  13.2%  [1665..2068]
  router                             1149 us    1001 scaled    80.0 dispatches   8.1%  [1034..1264]
  attention core                      891 us     776 scaled    40.0 dispatches   6.3%  [783..999]
  indexer                             184 us     160 scaled    16.0 dispatches   1.3%  [175..193]
  norms/rope/casts/KV writes          175 us     152 scaled    45.0 dispatches   1.2%  [149..200]
  compressor                           91 us      80 scaled     2.5 dispatches   0.6%  [87..96]
  other                                 3 us       2 scaled     0.1 dispatches   0.0%  [3..3]
```

## 6 rows

```
rows = 6, 40 layers (layers 0-20 + 20-40), per step, mean of 2 labelled and 2 stock runs
stock MLX runs: span 26124 busy 24672 us; span 26137 busy 24753 us
real step (stock MLX): span 26131 us, GPU busy 24713 us (94.6%), idle 1418 us (5.4%), 40 encoders
labelled runs: kernels sum to 28185, 27894 us in 590 dispatches; serialized span 30488 us; scale to the real busy time x0.881
kernel                             launches   us/step   scaled us/launch  share  group
dsv41_dec_moe_fp4_up_bw                40.2      9290     8188     231.0  33.1%  MoE
dsv41_dec_moe_fp4_down_bw              40.1      5301     4672     132.1  18.9%  MoE
dsv41_hc_split                         40.0      2475     2182      61.9   8.8%  hc stages
dsv41_dec_wq_b                         40.1      2343     2065      58.4   8.4%  attention projections
dsv41_dec_in_proj                      40.0      1578     1391      39.4   5.6%  attention projections
mxfp8_qmv_nax (wo_b)                   40.0      1536     1354      38.4   5.5%  attention projections
dsv41_dec_wo_a_bw                      40.0      1513     1334      37.8   5.4%  attention projections
dsv41_hc_stage_x                       40.1      1379     1215      34.4   4.9%  hc stages
dsv41_gate                             40.0       733      646      18.3   2.6%  router
dsv41_dec_attn                         40.0       689      608      17.2   2.5%  attention core
dsv41_route                            40.0       378      333       9.4   1.3%  router
gg1_copybfloat16bfloat16               49.2       233      205       4.7   0.8%  norms/rope/casts/KV writes
dsv41_dec_in_proj_stage                40.0       170      150       4.2   0.6%  attention projections
glm5_hc_expand                         40.0       137      121       3.4   0.5%  hc stages
dsv41_dec_idx_pool2                     3.0       115      102      38.2   0.4%  compressor
dsv41_dec_idx_score                     8.0        76       67       9.5   0.3%  indexer
(5 more)                                9.1        92       81             0.3%
groups (us/step, scaled, dispatches, share, spread across the labelled runs):
  MoE                               14591 us   12860 scaled    80.4 dispatches  52.0%  [14538..14645]
  attention projections              7140 us    6293 scaled   200.1 dispatches  25.5%  [7127..7152]
  hc stages                          3992 us    3518 scaled   120.1 dispatches  14.2%  [3894..4090]
  router                             1111 us     979 scaled    80.0 dispatches   4.0%  [1109..1113]
  attention core                      689 us     608 scaled    40.0 dispatches   2.5%  [689..690]
  norms/rope/casts/KV writes          233 us     205 scaled    49.3 dispatches   0.8%  [222..243]
  indexer                             143 us     126 scaled    16.0 dispatches   0.5%  [140..146]
  compressor                          139 us     122 scaled     4.0 dispatches   0.5%  [134..143]
  other                                 3 us       2 scaled     0.0 dispatches   0.0%  [0..4]
```

The indexer's and compressor's projections are inside the projection rows: the index `wq_b` in `wq_b`, the compressor
`wkv` and the indexer `weights_proj` in `in_proj`. `wq_b` is the attention `wq_b` and, on the 8 index-owner layers, the index `wq_b` (extra launch widths); `in_proj` is the
fused q/kv projection plus, where a layer has them, the compressor's `wkv` and the indexer's `weights_proj` rows.
`dsv41_hc_stage_x` is one hc stage launch a layer and `hc_split` the mix split (sinkhorn); `glm5_hc_expand` the hc expand
after the MoE. `dec_idx_pool` is the compressor pooling (it commits on alternate steps at ratio 2). The step is 544
dispatches at 1 row (13.6 a layer) and 590 at 6 rows. The attn-side rms norms, ropes and the hc collapse sit inside the
`in_proj`, `wq_b` and `wo_a` launches (their kernel headers say so), so the "norms/rope/casts/KV writes" row is only
the stray copies.
