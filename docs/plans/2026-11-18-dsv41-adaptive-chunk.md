# DeepSeek V4.1 per-request prefill width

For resident dsv41, choose the widest legal prefill rung that fits the full
request bill. With `--prefill-chunk` omitted, the candidates are 8192 and
4096. A cold 1M prompt should admit at 4096 at the g14 budget. A 32k prompt
with room should use 8192. An explicit width remains a pin, subject to the
model's row limit. [Issue #50](https://github.com/STRML/mlx-serve/issues/50)

```text
connection bill ─▶ queue ─▶ restore / evict ─▶ inference-thread re-ask
                                                     │
                                                     ▼
                                              generator slices
                                                     │
                                              plugin row guard
```

## Build order

1. In `src/plugins/deepseek_v41/plugin.zig`, test and reject single-slot
   forwards over 8192 before activating slot state. Use the model's shared
   prefill maximum. The guard covers prefill, verify, and all-logits forwards.
   `forwardBatched` already requires one decode row per slot and caps the
   number of slots at 16.
2. Share the rung filter between the load, request, and per-chunk ladders.
   Keep the score and CUDA limits. Reuse `ModelConfig.cappedPrefillChunk`
   for the width cap and update the stale `prefillChunkMax` references.
3. Opt resident dsv41 into request sizing. Floor its ladder at 4096 because
   the Engram encoder lead split has a 4096-row minimum on TP1. Keep
   per-chunk repricing and the hot-cache reserve clamp on `longCtxGated`.
   The omitted-flag startup still sizes the hot cache from the load pin.
4. Include media bytes, pending promises and reserve, and the omitted-flag
   cap beside decoders when pricing candidates for resident dsv41. Apply
   these constraints in the relevant prefill bill and re-ask. Admission
   still owns eviction, warm deferral and refusal. If the final re-ask
   measures less headroom than the floor needs, it returns the floor and
   execution proceeds, as with qwen4.
5. Pass the final choice through the existing generator width path. Keep
   explicit flag precedence and bound a merged dsv41 tail by its billed
   width. Keep the full-context dsv41 estimate, including on restore.
   Handle the streamed bank's internal call width and matching load bill
   in a follow-up issue. The plugin row guard rejects oversized inputs
   at the ABI boundary.

## Failure matrix

| Case | Expected behavior | Verification |
| --- | --- | --- |
| Single-slot plugin forward | 8192 reaches forward and 8193 is refused before slot activation, including all-row logits. | Exercise the plugin ABI with fake or tiny-model state. |
| Cold 1M at g14 | Reported budget is about 13062 MiB against about 13309 MiB needed at 8192. The omitted-flag request admits at 4096. | Use the production estimator and the release config. Check the fit boundary and the bill. Run the live omitted-flag gate before rollout. |
| 32k with room | From a 4096 load pin, the request chooses 8192. | Check the chooser and bill. Verify forwarded slice sizes on TP1 during rollout. |
| Image request under pressure | The image claim pushes the 8192 bill over budget. The full 4096 bill fits and admits. | Exercise the admission path with a media claim between the two fit boundaries. |
| Pending prefill beside decoders | Price the 4096 company cap and include other promises and reserve. | Exercise the pending hold. Check the width against the scheduler cap and test a shortfall with existing claims. |
| Restored prefix and cold retry | Price the full context after restore. Re-evaluate the remaining tail after eviction or a cold retry. | Cover warm and cold fit inputs and the re-ask. Check restored position and completion in the live gate. |
| Cache eviction and late headroom drop | Sizing steps down when possible. Admission can reclaim or refuse. An already-admitted request gets the 4096 floor on a late shortfall. | Vary headroom around both rungs and verify the floor fallback and normal admission refusal. |
| Short final tail | A merged tail remains within the selected dsv41 width. | Exercise a merge candidate just above the selected rung. |
| Explicit pin and stale width | Explicit 4096 pins. Explicit 8192 preserves its normal fit check. Oversized pins cannot produce a chooser result, bill, or plugin forward above 8192. | Exercise omitted, 4096, 8192 and 16384 width inputs. |
| First launch with flag omitted | The load pin sizes the hot-cache reserve. The admitted request can widen independently. | Exercise real reserve sizing with explicit and omitted flag inputs. Include a flag-omitted launch in the rollout gate. |
| qwen4 and a second architecture | Existing per-request selections, explicit precedence, and long-context policy hold. | Compare old and new decisions over a grid of headroom, prompt, warm-prefix and pin inputs. |
| TP4 collective and leader-only prefill | Rank 0 determines the slice. Collective workers forward the IDs sent by rank 0. In leader-only mode, workers apply the position and row-count update. | Verify position and row agreement on 4096, 8192, restored and final-short slices with a progress timeout. |
| TP4 worker capacity | Rank 0 can widen using its own headroom. Every worker must fit the resulting 8192-row collective. All ranks need matching `MLX_SERVE_DSV41_CED` behavior. | Gate rollout on per-rank headroom and forward progress at 8192. |

## Validation

Run red tests before each behavior change. On the Mac, run the Debug
admission, prefill, chunk, dsv41, deepseek, peer and qwen4 filters, plus
`zig build check` and strict lint. Use fake or fixture paths for scheduler and
plugin coverage.

Before rollout, run the iQ 3.0bpw 1M gate with `--prefill-chunk` omitted.
`local-llm/bench/studio-cluster/gate1m.sh:56-58` always supplies the flag, so
the omitted-flag run needs a separate launch arm. Check the load pin and
hot-cache reserve on that first launch, then record selected and forwarded
widths for four cold prompts and a restored-prefix session. Repeat on target
TP4 modes. The collective header carries `ids.len` and workers forward that
slice (`distributed.zig:2803,3021-3041` on the plan base). CED reads each
process's environment (`deepseek_v41.zig:879-890`). TP disables the encoder
lead split.

For the decode stall gate, hold the model, prompt and arrival schedule
constant while comparing 4096 and 8192. Record maximum and p95 inter-token
gaps, prefill chunk wall time, and time to first token for a small request
arriving during prefill. Confirm the 4096 cap when decode is already active.
Use a clean Studio window and the existing Latch protocol. Keep explicit
4096 as the rollout fallback until the live gates pass.
