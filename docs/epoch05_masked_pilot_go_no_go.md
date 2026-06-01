# Epoch05 Masked Semi-Full Tool Repair: Pilot Go/No-Go

Date: 2026-06-01

## Decision

No-go for full Epoch05 training.

The masked runtime and dataset path worked, but none of the three 25-step pilot runs satisfied the ladder gate. The low-LR pilots improved normal chat and modestly improved structured tool calls, but did not reach the required +5 tool-call-point improvement. The high-LR pilot reached the tool-call improvement threshold, but regressed normal chat and increased no-tool false positives.

## What Changed

- Added `--mask-prompt` support to grouped semi-full training.
- Built `semi_epoch05_tool_repair_masked_10k` as `messages` + `tools` JSONL.
- Repaired eval scoring to report:
  - `tool_selected_rate`
  - `schema_valid_rate`
  - `args_semantically_valid_rate`
  - strict task pass rate
- Kept fixed Hermes tool schemas unchanged.

Dataset summary:

| Split | Rows |
|---|---:|
| train | 720 |
| valid | 80 |
| test | 80 |

Train mix:

| Category | Rows |
|---|---:|
| live_failure_repair | 407 |
| normal_chat_retention | 182 |
| computer_use_exact_schema | 64 |
| browser_current_info | 45 |
| tool_result_finalization | 16 |
| terminal_file_patch | 6 |

The dataset had no XML tool-call targets and no pre-rendered `text` rows.

## Pilot Results

Baseline was the restored `step_01746_pythonic` fixed base on the same 50-case mini-suite.

| Run | Overall | Structured tool calls | Tool selected | Schema valid | Args valid | Normal chat | Browser/current | No-tool FP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed base | 26/50 | 10/30 | 50.00% | 83.33% | 33.33% | 16/20 | 2/21 | 10.00% |
| LR `1e-7` | 28/50 | 11/30 | 56.67% | 86.67% | 36.67% | 17/20 | 3/21 | 10.00% |
| LR `3e-7` | 28/50 | 11/30 | 56.67% | 86.67% | 36.67% | 17/20 | 3/21 | 10.00% |
| LR `1e-6` | 27/50 | 12/30 | 53.33% | 90.00% | 40.00% | 15/20 | 4/21 | 15.00% |

Training diagnostics:

| Run | Mean loss delta | Improved steps | Worse steps | Flat steps | Peak memory |
|---|---:|---:|---:|---:|---:|
| LR `1e-7` | -0.02818 | 16 | 7 | 2 | 23.586 GB |
| LR `3e-7` | -0.02777 | 16 | 7 | 2 | 23.586 GB |
| LR `1e-6` | -0.02871 | 16 | 8 | 1 | 23.586 GB |

## Gate Check

| Gate | Result |
|---|---|
| Lowest LR improves tool-call rate by at least 5 points | fail |
| No normal-chat regression | fail for `1e-6` |
| No invented tool names | pass |
| No invented `computer_use` actions | pass |
| No text-tool leaks | pass |
| No-tool false positives stay controlled | fail for `1e-6` |

## Interpretation

Prompt masking fixed a real training-objective bug and made pilot training cheaper in memory, but the semi-full int8 expert/router update still did not produce enough reliable behavior movement in 25 steps to justify a full epoch. The key remaining weakness is browser/current-info routing, where the best pilot reached only `4/21`.

The next useful path is not another blind full epoch. It should first add a stronger diagnostic around why browser-current rows do not move: compare logits for `browser_navigate` vs `x_search` vs refusal tokens before and after pilots, and test whether a BF16-router-only or final-layer-router-focused update moves the routing logits more directly than the current grouped expert update.

## Cleanup Notes

The `1e-6` pilot failed while writing a duplicate `_final` checkpoint because the disk was full. The regular `step_00025` checkpoint existed and was evaluated. The corrupt partial `_final` checkpoint was deleted. The full epoch was not launched, and no fusion, quantization, upload, or replacement of the served 8080 model was performed.
