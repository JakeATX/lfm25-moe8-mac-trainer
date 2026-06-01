# Epoch04 Aggressive Tool-Use Semi-Full Fine-Tune: Go/No-Go

Date: 2026-06-01

## Decision

No-go for release, fusion, quantization, or upload.

Epoch04 completed cleanly as a grouped direct expert/router update, but the parser-disabled 200-case evaluation did not improve over the fixed `step_01746_pythonic` base. The final summary matched the fixed-base summary on all headline metrics, so this checkpoint is retained only as an experiment artifact.

## Start Point

- Base runtime: `/Users/jkooker/Documents/Codex/2026-05-28/get-this-new-model-up-and/release_work/model_runtime_step01746_pythonic`
- Base weights: preserved `step_01746_final`
- Endpoint used for baseline: `http://127.0.0.1:8080/v1`
- Eval endpoint for epoch04: `http://127.0.0.1:8084/v1`
- Parser mode: `HERMES_PARSE_TEXT_TOOL_CALLS=0`
- Runtime metadata requirement: `tool_parser_type: "pythonic"`

## Dataset

Dataset: `semi_epoch04_tool_aggressive_10k`

- Train: 582 examples
- Valid: 60 examples
- Test: 60 examples
- Token cap: 10K per example
- XML tool-call targets: false
- Native target format: `<|tool_call_start|>[tool_name(arg="value")]<|tool_call_end|>`

Training mix:

| Category | Train rows |
|---|---:|
| live_failure_repair | 232 |
| normal_retention | 145 |
| original_trace_converted | 60 |
| aggressive_browser | 47 |
| aggressive_terminal_file | 47 |
| aggressive_computer_use | 27 |
| dont_give_up_recovery | 21 |
| tool_result_finalization | 3 |

## Training Run

- Run dir: `/Users/jkooker/Documents/Codex/2026-05-28/get-this-new-model-up-and/runtime_moe8_artifacts/runs/moe8_epoch04_tool_aggressive_g4s3_10k`
- Final checkpoint: `checkpoints/step_00582_final`
- Steps: 582/582
- Epochs: 1
- Group size: 4
- Stride: 3
- Learning rate: `5e-8`
- Trainable scope: MoE expert projections plus routers
- Frozen scope: attention, conv, embeddings, norms, and non-router dense layers
- Final logged `loss_before`: 5.510509
- Final logged `loss_after`: 5.521232
- Final logged tokens: 2,929
- Final step elapsed: 21.03 seconds
- Peak memory: 39.934 GB
- NaN/OOM/hard-memory markers: none found

## Evaluation

Baseline result file:

`artifacts/live_hermes_eval/results/epoch04_pretrain_fixed_base.summary.json`

Epoch04 result file:

`artifacts/live_hermes_eval/results/epoch04_final_parser_disabled_rerun.summary.json`

| Metric | Fixed base | Epoch04 final | Delta |
|---|---:|---:|---:|
| Overall pass rate | 47.5% | 47.5% | 0.0 |
| Passed / total | 95 / 200 | 95 / 200 | 0 |
| Structured tool-call rate | 37.5% | 37.5% | 0.0 |
| Valid structured tool calls | 54 / 144 | 54 / 144 | 0 |
| No-tool false-positive rate | 11.54% | 11.54% | 0.0 |
| Text tool leaks | 0 | 0 | 0 |
| Invented tool names | 0 | 0 | 0 |
| Invented computer actions | 1 | 1 | 0 |

Category metrics:

| Category | Fixed base | Epoch04 final | Delta |
|---|---:|---:|---:|
| terminal_file_patch | 58.54% | 58.54% | 0.0 |
| browser_search_current | 22.22% | 22.22% | 0.0 |
| normal_chat | 80.00% | 80.00% | 0.0 |
| computer_use_browser_control | 27.59% | 27.59% | 0.0 |
| correction_recovery | 45.45% | 45.45% | 0.0 |
| tool_result_finalization | 25.00% | 25.00% | 0.0 |

## Gate Results

| Gate | Required | Result |
|---|---|---|
| Overall live/direct pass rate improves by at least 10 points | yes | fail |
| Structured tool-call rate improves by at least 15 points | yes | fail |
| Browser/current improves over fixed base | yes | fail |
| Terminal/file improves over fixed base | yes | fail |
| Computer-use improves over fixed base | yes | fail |
| Correction/recovery improves over fixed base | yes | fail |
| Finalization improves over fixed base | yes | fail |
| Normal chat at least fixed-base minus 2 points | yes | pass |
| Zero text-tool leaks | yes | pass |
| Zero invented tool names | yes | pass |
| Zero invented computer-use actions | yes | fail |

## Interpretation

This run proved that the grouped epoch04 path can complete within memory, but it did not measurably move behavior under the parser-disabled live/direct suite. The most likely explanation is that the update magnitude was too small at `5e-8`, or that the grouped projection/requantization path did not produce enough effective behavioral change for this specific dataset.

Because the checkpoint does not beat the fixed base, it should not replace the served model, and no KXL GGUF variants should be generated from it.

## Next Actions

- Keep `step_00582_final` only as a local experiment artifact.
- Keep serving the restored fixed base on `8080`.
- Do not upload, fuse, quantize, or publish epoch04.
- If continuing this line, first add a weight-delta and logit-delta diagnostic before another overnight run, so we can prove the semi-full update is actually changing the model in behaviorally relevant places before spending time on full live suites.
