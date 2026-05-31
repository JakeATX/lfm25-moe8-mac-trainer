#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH="$ROOT/runtime_moe8"
PY="$ROOT/lfm25_hermes_ft/venv/bin/python"
RUN_DIR="$ROOT/runtime_moe8_artifacts/runs/moe8_overlap_g4s3_3epochs_10k"
LOG="$ROOT/runtime_moe8_artifacts/logs/moe8_overlap_g4s3_3epochs_10k.log"

mkdir -p "$(dirname "$LOG")" "$RUN_DIR"

exec "$PY" "$ROOT/runtime_moe8/grouped_epoch_train.py" \
  --model "$ROOT/runtime_moe8_artifacts/checkpoints/lfm25_experts_int8_mlx" \
  --data "$ROOT/runtime_moe8_artifacts/datasets/hermes_filtered_text_10k" \
  --run-dir "$RUN_DIR" \
  --epochs 3 \
  --max-seq-length 10000 \
  --group-size 4 \
  --stride 3 \
  --lr 1e-7 \
  --train-router \
  --grad-checkpoint \
  --save-every-steps 25 \
  --keep-last-checkpoints 2 \
  --target-memory-limit-gb 55 \
  --hard-memory-limit-gb 60 \
  "$@" \
  2>&1 | tee -a "$LOG"
