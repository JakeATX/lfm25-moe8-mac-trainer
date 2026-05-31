# LFM2.5 MoE8 Mac Trainer

Research-grade MLX runtime for local Apple Silicon fine-tuning of `LiquidAI/LFM2.5-8B-A1B` style MoE models. The current focus is Hermes/tool-use behavior on Mac hardware with unified memory constraints.

## What This Repo Contains

- Expert-only quantization for LFM2.5 MoE checkpoints.
- Grouped block-coordinate training for int8 expert weights plus BF16 routers.
- Hardware-adaptive group-size sweeping for different Mac RAM capacities.
- Native LFM tool-call failure mining and LoRA/DoRA repair utilities.
- Evaluation scripts for parser-disabled and parser-enabled tool-call reliability.

Large artifacts are intentionally excluded from git. Recreate them from the scripts or download from the associated Hugging Face repos.

## Quick Start

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python scripts/hardware_profile.py --out artifacts/machine_profile.json
```

Quantize only MoE experts:

```bash
python runtime_moe8/quantize_experts.py \
  --model LiquidAI/LFM2.5-8B-A1B-MLX-bf16 \
  --out artifacts/checkpoints/lfm25_experts_int8_mlx \
  --bits 8 \
  --group-size 64
```

Find the largest safe grouped training size for this Mac:

```bash
python runtime_moe8/group_sweep.py \
  --model artifacts/checkpoints/lfm25_experts_int8_mlx \
  --data artifacts/datasets/hermes_filtered_text_10k \
  --out artifacts/reports/group_sweep_10k.json \
  --work-dir artifacts/group_sweeps/10k \
  --group-sizes auto \
  --max-seq-length 10000 \
  --train-router \
  --grad-checkpoint
```

Run native tool-call repair:

```bash
python scripts/build_lfm_repair_dataset.py --out artifacts/repair_datasets/iter01
python scripts/run_lora_repair.py \
  --model artifacts/runs/moe8_overlap_g4s3_3epochs_10k/checkpoints/step_01746_final \
  --data artifacts/repair_datasets/iter01 \
  --adapter-path artifacts/adapters/lfm_tool_repair_iter01 \
  --iters 300 \
  --max-seq-length 4096
```

Run the colloquial tool-router repair loop:

```bash
python scripts/build_colloquial_tool_router_dataset.py \
  --out artifacts/repair_datasets/iter02_colloquial_router

python scripts/eval_colloquial_tool_router.py \
  --endpoint http://127.0.0.1:8081/v1/chat/completions \
  --model release_work/model_upload \
  --out artifacts/evals/iter02_pretrain_colloquial_openai_parser_disabled.json \
  --allow-fail

python scripts/run_lora_repair.py \
  --model release_work/model_upload \
  --data artifacts/repair_datasets/iter02_colloquial_router \
  --adapter-path artifacts/adapters/lfm_tool_router_iter02 \
  --iters 800 \
  --max-seq-length 4096 \
  --learning-rate 3e-6
```

For prompt-masked repair, convert the text dataset to MLX chat JSONL first:

```bash
PYTHONPATH=scripts python scripts/convert_text_dataset_to_chat_tools.py \
  --src artifacts/repair_datasets/iter03_colloquial_router_server_template \
  --out artifacts/repair_datasets/iter04_colloquial_router_chat_masked

python scripts/run_lora_repair.py \
  --model release_work/model_upload \
  --data artifacts/repair_datasets/iter04_colloquial_router_chat_masked \
  --adapter-path artifacts/adapters/lfm_tool_router_iter04_masked \
  --resume-adapter-file artifacts/adapters/lfm_tool_router_iter03/adapters.safetensors \
  --iters 600 \
  --mask-prompt
```

Run the fixed-Hermes contrastive router repair:

```bash
python scripts/export_fixed_hermes_tools.py \
  --hermes-repo ../hermes-agent-lfm-tool-parser \
  --out artifacts/tool_surfaces/fixed_hermes_terminal_file_browser_tools.json \
  --toolsets terminal_tools file_tools browser_tools

python scripts/build_fixed_hermes_contrast_router_dataset.py \
  --tools-json artifacts/tool_surfaces/fixed_hermes_terminal_file_browser_tools.json \
  --out artifacts/repair_datasets/iter05_fixed_hermes_contrast_router

python scripts/eval_fixed_hermes_tool_router.py \
  --endpoint http://127.0.0.1:8081/v1/chat/completions \
  --model release_work/model_upload \
  --tools-json artifacts/tool_surfaces/fixed_hermes_terminal_file_browser_tools.json \
  --out artifacts/evals/iter05_pretrain_fixed_hermes_parser_disabled.json \
  --allow-fail

python scripts/run_lora_repair.py \
  --model release_work/model_upload \
  --data artifacts/repair_datasets/iter05_fixed_hermes_contrast_router \
  --adapter-path artifacts/adapters/iter05_fixed_hermes_contrast_router_r32 \
  --iters 1200 \
  --num-layers -1 \
  --lora-rank 32 \
  --lora-scale 64 \
  --learning-rate 3e-6 \
  --max-seq-length 4096 \
  --mask-prompt
```

When evaluating an unfused MLX adapter through `mlx_lm.server`, pass the adapter
in the request body with `--adapter-path` on the eval script. The server's CLI
`--adapter-path` is not enough for requests whose `model` field is a concrete
model path, because adapter selection is exposed as the HTTP `adapters` field.

Build the GGUF export plan after a fused adapter passes:

```bash
python scripts/build_gguf_calibration_set.py \
  --datasets artifacts/repair_datasets/iter05_fixed_hermes_contrast_router \
  --out artifacts/gguf/calibration/hermes_tool_router_calibration.txt

python scripts/plan_gguf_xl_quant.py \
  --out-dir artifacts/gguf/xl_quant_policies \
  --bf16-gguf artifacts/gguf/bf16/LFM-2.5-8B-1B-hermes-ft-BF16.gguf \
  --imatrix artifacts/gguf/calibration/hermes_tool_router_imatrix.gguf

python scripts/run_gguf_export_pipeline.py \
  --hf-model artifacts/fused/iter05_fixed_hermes \
  --llama-cpp ../../tools/llama-current \
  --out-dir artifacts/gguf/final_iter05 \
  --calibration artifacts/gguf/calibration/hermes_tool_router_calibration.txt \
  --xl-plan artifacts/gguf/xl_quant_policies/xl_quant_plan.json
```

The XL quant policies are honest stock-llama.cpp approximations of Unsloth-style Dynamic XL buckets. They use `--tensor-type-file` regex overrides and optional imatrix calibration, and should be published as `*_XL_approx.gguf` unless generated by the actual Unsloth Dynamic 2.0 tooling.

## Current Known Result

The first completed local run is an expert/router direct-weight fine-tune:

- 582 train examples, 10K token cap.
- 3 epochs, 1,746 steps.
- MoE expert projections and routers changed.
- Attention, conv, embeddings, norms, and dense non-MoE layers stayed frozen.
- Peak memory was about 40.43 GB on a 64 GB Mac.

This is not mathematically equivalent to a simultaneous whole-model full fine-tune. It is a grouped semi-full-gradient update over MoE experts and routers.

The first LFM native tool-call repair pass then trained a LoRA adapter over the grouped checkpoint:

- Base checkpoint: `step_01746_final`.
- Adapter: rank-16 MLX LoRA, 300 iterations, 4K sequence length, learning rate `5e-6`.
- Repair data: 392 train / 32 valid / 32 test examples generated from observed native tool-call failures and no-tool counterexamples.
- Parser-disabled eval improved from `5/6` overall and `2/3` tool cases to `6/6` overall and `3/3` tool cases.
- Fused, 8-bit MLX, and 6-bit MLX variants all passed the same parser-disabled native LFM eval.

The repair target format is:

```text
<|tool_call_start|>[tool_name(arg="value")]<|tool_call_end|>
```

## Fixed-Hermes Iter10 Release Result

The accepted release adapter is `iter10_balanced_holdout_repair_r32`, trained after correcting the target data to use structured `assistant.tool_calls` rows so MLX renders canonical LFM pythonic calls during prompt-masked training.

- Final adapter: `artifacts/adapters/iter10_balanced_holdout_repair_r32`.
- Fused MLX checkpoint: `release_work/model_upload_iter10_fused`.
- Dequantized HF/safetensors source for GGUF export: `release_work/model_upload_iter10_fused_dequantized`.
- Parser metadata: `tokenizer_config.json` preserves `tool_parser_type: "pythonic"`.
- MLX parser-disabled fixed-Hermes eval: `43/43`, with `28/28` structured tool calls, `10/10` no-tool cases, and `0` text-tool leaks.

The final fixed-Hermes eval covers browser, terminal, file read/search/write, no-tool counterexamples, and post-tool finalization. Hermes tool names and schemas are treated as fixed.

## GGUF Export Result

The GGUF path exports from the dequantized fused HF/safetensors checkpoint, not from an already quantized MLX checkpoint. A current llama.cpp checkout needed one local converter patch for LFM2/LFM2MoE short-conv weights:

```python
if "conv.conv" in name:
    data_torch = data_torch.squeeze(-1)
```

Without that patch, the converter left `conv.conv.weight` as a 3D tensor and llama.cpp failed in `ggml_ssm_conv`. With the patch, the BF16 parent loads and generates.

Generated GGUF artifacts:

| Artifact | Size | llama.cpp 64K fixed-Hermes eval |
|---|---:|---:|
| `LFM-2.5-8B-1B-hermes-ft-BF16.gguf` | 16G | load/generation smoke passed |
| `LFM-2.5-8B-1B-hermes-ft-Q8_K_XL_APPROX.gguf` | 8.6G | 43/43 |
| `LFM-2.5-8B-1B-hermes-ft-Q6_K_XL_APPROX.gguf` | 7.1G | 43/43 |
| `LFM-2.5-8B-1B-hermes-ft-Q5_K_XL_APPROX.gguf` | 6.5G | 43/43 |
| `LFM-2.5-8B-1B-hermes-ft-Q4_K_XL_APPROX.gguf` | 5.7G | 43/43 |

The XL quant policies use stock llama.cpp `--tensor-type-file` overrides plus the Hermes/tool-router imatrix. They are size- and behavior-targeted approximations of Unsloth-style Dynamic XL buckets, not byte-identical Unsloth Dynamic 2.0 outputs.

## Colloquial Tool-Router Repair Result

A second repair loop attempted to teach more aggressive natural-language routing for Hermes-style tools. The loop built three local datasets/adapters:

- `iter02_colloquial_router`: 2,608 train / 212 valid / 213 test rows, text JSONL, native LFM pythonic targets.
- `iter03_colloquial_router_server_template`: same scale, but with server-shaped tool JSON in the prompt.
- `iter04_colloquial_router_chat_masked`: chat JSONL converted for MLX prompt masking so loss targets assistant completions only.

The loop did not meet acceptance thresholds. All variants preserved the smaller 12-case structured OpenAI tool-call suite at `12/12`, but the broader colloquial eval stayed at `16/20` with the same failures:

- `run echo hermes-tool-test` routed to `browser_navigate`.
- `pwd in terminal` routed to `browser_navigate`.
- `list files here from shell` routed to a non-Hermes `bash` tool.
- `search this repo for browser_navigate` routed to `browser_navigate`.

This means the adapters improved or preserved structured parser behavior but did not reliably change colloquial terminal/search tool selection. The iter02-iter04 adapters should be treated as analysis artifacts, not final model-release adapters.

## Safety Boundary

This repo does not contain API keys or Hugging Face tokens. Do not commit local `config.yaml`, `.env`, generated checkpoints, or raw private logs.
