# Iter14 Focused Contrast Router Go/No-Go

## Decision

No-go. Iter14 built the intended focused browser-vs-X-vs-local-file dataset and
added stricter eval/margin tooling, but neither adapter attempt passed the
normal-chat and tool-routing gates. No fusion, quantization, upload, or served
model replacement was performed.

The stable endpoint was restored to:

`/Users/jkooker/Documents/Codex/2026-05-28/get-this-new-model-up-and/release_work/model_runtime_step01746_pythonic`

## What Changed

- Added `iter14_browser_x_files_contrast_router`:
  - `1200 train / 120 valid / 120 test`
  - structured `messages` + fixed Hermes `tools`
  - prompt-masked MLX LoRA format
  - no XML tool-call targets
  - exact contrast rows for general web, X/Twitter search, local file search, and no-tool wording
- Added a focused 80-case direct API eval with a confusion matrix.
- Added a 12-case candidate-loss margin diagnostic.
- Added an adapter-search orchestrator that sends adapter paths through the HTTP `adapters` field.

The first rank-8 eval exposed a runtime bug in the orchestrator: it sent a
relative adapter path, while the MLX server resolved paths from a different
working directory. The orchestrator now resolves dataset, tool, and adapter
paths to absolute paths before evaluation.

## Fixed-Base Baseline

Focused Iter14 suite:

| Metric | Result |
|---|---:|
| Overall | 50/80 |
| Structured tool calls | 34/60 |
| Browser/general web | 4/20 |
| X social search | 17/20 |
| Local file search | 13/15 |
| No-tool retention | 16/20 |
| No-tool false positives | 1/20 |
| Text leaks / invented tools | 0 / 0 |

The main baseline failure is exactly the boundary problem: general web/current
prompts route to `x_search`, `search_files`, `terminal`, or refusal instead of
`browser_navigate`.

Broad 50-case suite:

| Metric | Result |
|---|---:|
| Overall | 26/50 |
| Normal chat | 16/20 |
| Browser/current | 2/21 |
| Terminal/file | 8/9 |
| No-tool false positives | 2/20 |

Candidate margins:

| Metric | Result |
|---|---:|
| Correct candidate best | 7/12 |
| Mean best-wrong minus correct margin | -0.004295 |
| Positive-margin cases | 7/12 |

## Adapter Attempts

### Rank 8, LR 1e-6, Scale 16

Training shape:

- LoRA rank: `8`
- Trainable params: `68.098M`
- Max sequence length: `8192`
- Steps: `50`
- Peak memory: `18.831 GB`
- Final train loss: `0.477`
- Final validation loss: `0.389`

Focused eval:

| Metric | Fixed Base | Rank 8 Step 50 |
|---|---:|---:|
| Overall | 50/80 | 48/80 |
| Structured tool calls | 34/60 | 33/60 |
| Browser/general web | 4/20 | 5/20 |
| X social search | 17/20 | 17/20 |
| Local file search | 13/15 | 11/15 |
| No-tool retention | 16/20 | 15/20 |
| No-tool false positives | 1/20 | 2/20 |
| Invented tools | 0 | 4 |

Broad eval:

| Metric | Fixed Base | Rank 8 Step 50 |
|---|---:|---:|
| Overall | 26/50 | 25/50 |
| Normal chat | 16/20 | 15/20 |
| Browser/current | 2/21 | 8/21 |
| Terminal/file | 8/9 | 2/9 |
| No-tool false positives | 2/20 | 3/20 |
| Invented tools | 0 | 2 |

Candidate margins improved:

| Metric | Fixed Base | Rank 8 Step 50 |
|---|---:|---:|
| Correct candidate best | 7/12 | 8/12 |
| Mean margin | -0.004295 | 0.195831 |
| Positive-margin cases | 7/12 | 8/12 |

Interpretation: rank 8 moved some logits in the desired direction and improved
browser/current on the broad suite, but the update was too blunt. It introduced
invented tool names, increased no-tool false positives, damaged terminal/file
behavior, and failed normal-chat retention.

### Rank 4, LR 3e-7, Scale 8

Training shape:

- LoRA rank: `4`
- Trainable params: `34.049M`
- Max sequence length: `8192`
- Steps: `50`
- Peak memory: `18.420 GB`
- Final train loss: `1.256`
- Final validation loss: `1.302`

Focused eval:

| Metric | Fixed Base | Rank 4 Step 50 |
|---|---:|---:|
| Overall | 50/80 | 46/80 |
| Structured tool calls | 34/60 | 31/60 |
| Browser/general web | 4/20 | 2/20 |
| X social search | 17/20 | 17/20 |
| Local file search | 13/15 | 12/15 |
| No-tool retention | 16/20 | 15/20 |
| No-tool false positives | 1/20 | 1/20 |
| Invented tools | 0 | 0 |

Broad eval:

| Metric | Fixed Base | Rank 4 Step 50 |
|---|---:|---:|
| Overall | 26/50 | 26/50 |
| Normal chat | 16/20 | 15/20 |
| Browser/current | 2/21 | 3/21 |
| Terminal/file | 8/9 | 8/9 |
| No-tool false positives | 2/20 | 2/20 |
| Invented tools | 0 | 0 |

Candidate margins were mostly unchanged:

| Metric | Fixed Base | Rank 4 Step 50 |
|---|---:|---:|
| Correct candidate best | 7/12 | 7/12 |
| Mean margin | -0.004295 | 0.030090 |
| Positive-margin cases | 7/12 | 7/12 |

Interpretation: rank 4 avoided the worst schema damage but did not teach the
browser boundary. It also still regressed normal chat by one case.

## Artifact Ledger

- Dataset: `artifacts/repair_datasets/iter14_browser_x_files_contrast_router`
- Focused eval script: `scripts/eval_iter14_contrast_router.py`
- Margin diagnostic: `scripts/diagnose_iter14_contrast_margins.py`
- Orchestrator: `scripts/run_iter14_adapter_search.py`
- Kept diagnostic adapter: `artifacts/adapters/iter14_browser_x_files_contrast_r8`
- Removed failed bulky adapter: `artifacts/adapters/iter14_browser_x_files_contrast_r4_lr3e-7`

## Recommendation

Do not continue Iter14 SFT-style LoRA on this dataset. The focused data revealed
the problem, but SFT on tool-call targets is not precise enough to shift the
browser boundary without collateral damage.

The next attempt should be preference-style or margin-style, not another larger
SFT pass:

1. Build paired candidate examples where the same prompt has ranked alternatives:
   `browser_navigate` preferred over `x_search`, `search_files`, `terminal`, and
   refusal for general web/current prompts.
2. Optimize a small adapter against the preference margin directly, or emulate
   this with a custom MLX loss over candidate continuations.
3. Keep the 80-case focused suite as the first gate and the broad 50-case suite
   as the second gate.
4. Stop immediately if normal chat drops below `16/20`, no-tool false positives
   exceed `2/20`, or invented tools appear.

This is more likely to work than adding more positive browser rows, because the
base model already knows how to emit each tool. The defect is a ranking problem
among valid alternatives under similar wording.
