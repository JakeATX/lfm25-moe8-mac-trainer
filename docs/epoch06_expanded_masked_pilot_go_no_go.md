# Epoch06 Expanded Masked Pilot Go/No-Go

## Decision

Epoch06 is a no-go for 500-step continuation, full epoch training, fusion,
quantization, upload, or served-model replacement.

The expanded dataset and two pilot branches produced small movement, but neither
pilot met the gate: browser/current-info did not improve by at least 10 points
over the fixed base while preserving no-tool behavior.

`8080` remained on the fixed rollback base:
`release_work/model_runtime_step01746_pythonic`.

## Dataset

Built `semi_epoch06_tool_repair_expanded_10k` as structured `messages` + `tools`
JSONL with prompt masking.

- Train/valid/test: 2400 / 240 / 240.
- No XML tool-call targets.
- No tokenizer-render rejections.
- Tool surface: live Hermes core subset, including `computer_use` and its fixed
  action enum.

Train mix:

| Category | Rows |
| --- | ---: |
| browser_current_info | 796 |
| normal_chat_retention | 611 |
| live_failure_repair | 518 |
| computer_use_exact_schema | 242 |
| terminal_file_patch | 121 |
| correction_recovery | 72 |
| tool_result_finalization | 40 |

The live Hermes schema renders around 5.2K tokens per row with the core tool
subset, so rows remain under the 10K cap.

## Candidate-Loss Diagnostic

Candidate-loss diagnostics compared fixed base, Epoch05 `1e-6`, and Epoch06
pilot checkpoints. In all checked model variants, the correct tool continuation
was already lowest-loss in five of six diagnostic cases. The hard no-tool prompt
still preferred a tool continuation over normal text.

This means the failure is not simply that correct browser/tool-call tokens are
globally impossible. The remaining issue is sampled routing under the full
schema and ambiguous current-info prompts, especially confusion between
`browser_navigate`, `x_search`, local file search, and refusal.

## Pilot A

Warm-started from:
`runtime_moe8_artifacts/runs/moe8_epoch05_pilot_lr1e-6_masked/checkpoints/step_00025`.

Training:

- Layers: all MoE layers, grouped `g4/s3`.
- LR: `5e-7`.
- Prompt masking: enabled.
- Steps: 100, run in 25-step chunks after one long-process Metal hang at step 45.
- Peak memory: 23.586 GB.
- Unique steps: 100.
- Mean loss delta: `+0.00173`.
- Improved/worse/flat steps: 48 / 51 / 1.

Parser-disabled 50-case eval:

| Metric | Fixed Base | Epoch05 `1e-6` | Epoch06 Pilot A |
| --- | ---: | ---: | ---: |
| Overall | 26/50 | 27/50 | 28/50 |
| Structured tool calls | 10/30 | 12/30 | 11/30 |
| Normal chat | 16/20 | 15/20 | 17/20 |
| Browser/current | 2/21 | 4/21 | 3/21 |
| Terminal/file | 8/9 | 8/9 | 8/9 |
| No-tool false positives | 10% | 15% | 10% |
| Text-tool leaks | 0 | 0 | 0 |
| Invented tools/actions | 0 | 0 | 0 |

Pilot A improved normal chat and overall score, but browser/current improved by
only one case over fixed base. This failed the gate.

## Pilot B

Restarted from fixed base:
`release_work/model_runtime_step01746_pythonic`.

Training:

- Layers: late MoE layers `14-23`, grouped `g4/s3`.
- LR: `1e-6`.
- Prompt masking: enabled.
- Steps: 100, run in 25-step chunks plus one extra chunk after a Metal recovery
  event around step 54.
- Peak memory: 23.586 GB.
- Unique steps: 100.
- Mean loss delta: `-0.00314`.
- Improved/worse/flat steps: 50 / 40 / 10.

Parser-disabled 50-case eval:

| Metric | Fixed Base | Epoch06 Pilot B |
| --- | ---: | ---: |
| Overall | 26/50 | 27/50 |
| Structured tool calls | 10/30 | 11/30 |
| Normal chat | 16/20 | 16/20 |
| Browser/current | 2/21 | 3/21 |
| Terminal/file | 8/9 | 8/9 |
| No-tool false positives | 10% | 10% |
| Text-tool leaks | 0 | 0 |
| Invented tools/actions | 0 | 0 |

Pilot B had a cleaner loss trace than Pilot A and was faster, but it did not
move browser/current routing enough and produced more wrong-tool failures than
Pilot A.

## Failure Pattern

The expanded semi-full direct update still did not teach robust general web
routing. Common failures:

- General web/current/weather prompts routed to `x_search`.
- Current-info prompts routed to local `search_files`.
- Weather prompts sometimes produced malformed browser URLs.
- Some current-info prompts still refused despite web tools being available.
- No-tool hard negatives still had a 10% false-positive tool-call rate.

The model can use tools structurally: there were zero text-tool leaks, zero
invented tool names, and zero invented `computer_use` actions. The problem is
tool selection and semantic argument construction under ambiguous live-Hermes
prompts.

## Follow-Up Recommendation

Do not run a larger semi-full epoch from these pilots. The next useful direction
is a smaller, more targeted intervention:

- Build a browser-vs-X-vs-files contrastive dataset where near-identical prompts
  differ only by intent words: `web/current/latest/weather` -> `browser_navigate`,
  `X/Twitter/posts` -> `x_search`, `repo/local/files` -> `search_files`.
- Add many no-tool hard negatives that mention current, weather, browser,
  terminal, X, and files but explicitly ask for conceptual answers.
- Train a narrow adapter or a much smaller late-layer/router pilot first, then
  evaluate at 25/50 steps before any 100-step run.
- Consider runtime-side tool affordance ordering or tool-description shaping only
  if Hermes fixed-schema constraints allow it; do not rename tools or change
  schemas.

## Artifacts

- Dataset: `artifacts/repair_datasets/semi_epoch06_tool_repair_expanded_10k`.
- Candidate diagnostics:
  - `artifacts/evals/epoch06_candidate_losses_fixed_base.json`
  - `artifacts/evals/epoch06_candidate_losses_epoch05_lr1e-6_step00025.json`
  - `artifacts/evals/epoch06_candidate_losses_pilotA_warm_lr5e-7_step100.json`
  - `artifacts/evals/epoch06_candidate_losses_pilotB_late14-23_lr1e-6_step100.json`
- Eval results:
  - `artifacts/live_hermes_eval/results/epoch06_pilotA_warm_lr5e-7_step100_limit50.jsonl`
  - `artifacts/live_hermes_eval/results/epoch06_pilotB_late14-23_lr1e-6_step100_limit50.jsonl`
- Logs:
  - `runtime_moe8_artifacts/logs/moe8_epoch06_pilotA_warm_lr5e-7_masked_retry25.log`
  - `runtime_moe8_artifacts/logs/moe8_epoch06_pilotB_late14-23_lr1e-6_masked_retry25.log`

Only the best diagnostic checkpoint should be retained locally:
`runtime_moe8_artifacts/runs/moe8_epoch06_pilotA_warm_lr5e-7_masked_retry25/checkpoints/step_00100_final`.
