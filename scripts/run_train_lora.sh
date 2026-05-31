#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/venv/bin/python"
MODEL="LiquidAI/LFM2.5-8B-A1B-MLX-bf16"
DATA="$ROOT/datasets/hermes_filtered_text_12k"
ADAPTER="$ROOT/checkpoints/hermes_lora_r16_12k"
LOG="$ROOT/logs/train_lora.log"
CONFIG="$ROOT/artifacts/train_config.json"

mkdir -p "$ROOT/checkpoints" "$ROOT/logs" "$ROOT/artifacts"

cat > "$CONFIG" <<JSON
{
  "model": "$MODEL",
  "data": "$DATA",
  "adapter_path": "$ADAPTER",
  "fine_tune_type": "lora",
  "lora_rank": 16,
  "num_layers": 16,
  "batch_size": 1,
  "gradient_accumulation_steps": 1,
  "max_seq_length": 12000,
  "iters": 880,
  "epochs_approx": 1.0,
  "learning_rate": 0.00001,
  "val_batches": 1,
  "save_every": 100,
  "steps_per_report": 20,
  "steps_per_eval": 100,
  "grad_checkpoint": true,
  "seed": 42
}
JSON

echo "Training config written to $CONFIG"
echo "Logging to $LOG"

export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.org/simple}"
exec "$PY" -m mlx_lm.lora \
  --model "$MODEL" \
  --train \
  --data "$DATA" \
  --fine-tune-type lora \
  --adapter-path "$ADAPTER" \
  --batch-size 1 \
  --iters 880 \
  --val-batches 1 \
  --max-seq-length 12000 \
  --num-layers 16 \
  --learning-rate 1e-5 \
  --steps-per-report 20 \
  --steps-per-eval 100 \
  --grad-accumulation-steps 1 \
  --save-every 100 \
  --grad-checkpoint \
  --seed 42 \
  2>&1 | tee "$LOG"
