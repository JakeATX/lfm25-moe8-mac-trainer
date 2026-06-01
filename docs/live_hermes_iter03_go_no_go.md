# Live Hermes Iter03 Go/No-Go

Date: 2026-05-31

## Decision

**No-go for fusion, quantization, or upload.**

The fixed `step_01746_final` runtime with `tool_parser_type: "pythonic"` remains the rollback target. The Iter03 LoRA adapter is retained as an experiment artifact, but it should not be served as the default Hermes model and should not be exported to GGUF/MLX quant variants.

## Current Runtime State

- Stable endpoint: `http://127.0.0.1:8080/v1`
- Stable model path: `/Users/jkooker/Documents/Codex/2026-05-28/get-this-new-model-up-and/release_work/model_runtime_step01746_pythonic`
- Hermes config restored to port `8080`.
- Rejected Iter03 adapter server stopped.
- Hermes gateway restarted with `~/.local/bin` on `PATH` so `computer_use` discovery can see `cua-driver`.

## Why The Runtime Copy Exists

The preserved `step_01746_final` checkpoint did not have `tool_parser_type: "pythonic"` in tokenizer metadata. Without that metadata, MLX emits native LFM tool syntax as plain assistant content instead of structured OpenAI `message.tool_calls`.

The runtime copy at `release_work/model_runtime_step01746_pythonic` is an APFS clone with tokenizer metadata corrected. This is a runtime packaging fix, not a model-weight change.

## Iter03 Training Artifact

Adapter:

`artifacts/adapters/step01746_live_tool_repair_iter03_balanced_r8_8k`

Dataset:

`artifacts/repair_datasets/live_iter03_balanced_runtime`

Training settings:

- Base: `model_runtime_step01746_pythonic`
- Fine-tune type: MLX LoRA
- LoRA rank: 8
- LoRA scale: 16
- Layers: all layers
- Learning rate: `2e-7`
- Iterations: 200
- Max sequence length: 8192
- Prompt masking: enabled
- Peak memory: 17.419 GB
- Final validation loss: 0.937

Dataset manifest:

- Rows: 690
- Train/valid/test: 585 / 59 / 46
- Main categories: browser/current, computer_use, terminal, correction, file, finalization, normal retention, hard negatives.

## Eval Comparison

Acceptance target:

- Overall live-case pass rate >= 90%
- Structured tool-call pass rate >= 95% on tool-required cases
- Zero text-tool leaks
- Zero invented tool names
- Zero invented `computer_use` actions
- No-tool false-positive rate <= 5%

| Metric | Fixed Base | Iter03 Adapter |
|---|---:|---:|
| Overall pass rate | 37 / 83 = 44.58% | 38 / 83 = 45.78% |
| Structured tool-call rate | 20 / 57 = 35.09% | 22 / 57 = 38.60% |
| No-tool false-positive rate | 14.29% | 19.05% |
| Text-tool leaks | 0 | 0 |
| Invented tool names | 1 | 0 |
| Invented computer actions | 0 | 1 |
| Normal chat | 16 / 20 = 80.00% | 13 / 20 = 65.00% |
| Browser/current | 4 / 21 = 19.05% | 6 / 21 = 28.57% |
| Terminal/file/patch | 12 / 17 = 70.59% | 9 / 17 = 52.94% |
| Computer-use/browser-control | 1 / 10 = 10.00% | 3 / 10 = 30.00% |
| Correction/recovery | 3 / 10 = 30.00% | 5 / 10 = 50.00% |
| Tool-result finalization | 1 / 5 = 20.00% | 2 / 5 = 40.00% |

## Interpretation

Iter03 moved a few tool categories in the right direction, but the gain was only one additional pass out of 83 cases and it regressed normal chat plus terminal/file behavior. It also introduced one invented `computer_use` action. This is not an acceptable tradeoff for a release candidate.

## Live Hermes Smoke Notes

With Hermes parser bridge disabled and Hermes pointed at Iter03:

- Normal greeting worked.
- Weather lookup executed through live Hermes and returned a current-style answer.
- Terminal echo executed and returned output.
- Repo search executed and summarized matches.
- `read_file` executed successfully for `/tmp/hermes_live_eval_marker.txt`.
- `computer_use` still routed incorrectly or hung on the capture path.
- A normal Apollo 13 answer contained major factual errors, indicating retention damage risk.

## Quantization Decision

Quantization/export is deferred. Producing `Q8KXL`, `Q6KXL`, `Q5KXL`, or `Q4KXL` variants from Iter03 would preserve and possibly amplify the current routing and chat regressions.

## Recommended Next Iteration

Do not continue broad mixed-objective LoRA in the same shape. The next attempt should separate objectives and add tighter live-process controls:

1. Keep fixed `step_01746` runtime as base.
2. Build a smaller browser/current adapter with heavy no-tool retention and immediate live eval.
3. Build a separate computer-use adapter only after validating exact live `computer_use` schemas and timeouts.
4. Treat factual chat retention as a hard gate.
5. Add per-case Hermes process timeouts so hung computer-use attempts cannot stall an iteration.
6. Only stack or merge adapters after each objective passes independently.
