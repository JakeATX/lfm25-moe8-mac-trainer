# Overnight Live Hermes Repair Report

- Started: `20260531_215209`
- Base model: `/Users/jkooker/Documents/Codex/2026-05-28/get-this-new-model-up-and/release_work/model_runtime_step01746_pythonic`
- Endpoint: `http://127.0.0.1:8080/v1/chat/completions`
- Decision: **no-go: no adapter beat fixed base without violating gates**

## Baseline

- Overall: `94/200 (0.47)`
- Structured tool-call rate: `54/144 (0.375)`
- No-tool false positive rate: `0.1296`

## Adapter Attempts

### `browser`

- Adapter: `/Users/jkooker/Documents/Codex/2026-05-28/get-this-new-model-up-and/release_work/lfm25-moe8-mac-trainer/artifacts/adapters/overnight_20260531_215209_browser_r4`
- Decision: `rejected`
- Overall: `87/200 (0.435)`
- Structured tool-call rate: `48/144 (0.3333)`
- No-tool false positive rate: `0.1852`
- Rejection reasons: `invented_computer_actions, no_tool_false_positive_regression, normal_chat_regression, no_browser_search_current_improvement`

### `terminal_file`

- Adapter: `/Users/jkooker/Documents/Codex/2026-05-28/get-this-new-model-up-and/release_work/lfm25-moe8-mac-trainer/artifacts/adapters/overnight_20260531_215209_terminal_file_r4`
- Decision: `rejected`
- Overall: `92/200 (0.46)`
- Structured tool-call rate: `52/144 (0.3611)`
- No-tool false positive rate: `0.1667`
- Rejection reasons: `no_tool_false_positive_regression, normal_chat_regression`

### `computer_use`

- Adapter: `/Users/jkooker/Documents/Codex/2026-05-28/get-this-new-model-up-and/release_work/lfm25-moe8-mac-trainer/artifacts/adapters/overnight_20260531_215209_computer_use_r4`
- Decision: `rejected`
- Overall: `90/200 (0.45)`
- Structured tool-call rate: `51/144 (0.3542)`
- No-tool false positive rate: `0.1667`
- Rejection reasons: `no_tool_false_positive_regression, normal_chat_regression`

## Quantization

Quantization was deferred because no adapter has passed the live-Hermes acceptance gates.

## Artifacts

- `baseline_jsonl`: `/Users/jkooker/Documents/Codex/2026-05-28/get-this-new-model-up-and/release_work/lfm25-moe8-mac-trainer/artifacts/live_hermes_eval/results/overnight_20260531_215209_baseline.jsonl`
- `baseline_report`: `/Users/jkooker/Documents/Codex/2026-05-28/get-this-new-model-up-and/release_work/lfm25-moe8-mac-trainer/artifacts/live_hermes_eval/reports/overnight_20260531_215209_baseline.md`
- `hermes_smoke`: `/Users/jkooker/Documents/Codex/2026-05-28/get-this-new-model-up-and/release_work/lfm25-moe8-mac-trainer/artifacts/live_hermes_eval/transcripts/overnight_20260531_215209_hermes_oneshot_smoke.txt`
