#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

OUT_DIR="${OUTPUT_DIR:-outputs/run5_stable}"
CKPT_DIR="${CHECKPOINT_DIR:-checkpoints/run5_stable}"
mkdir -p "$OUT_DIR" "$CKPT_DIR"

# 1순위 안정화: AdamW + weight_decay + ReduceLROnPlateau + early stopping
ARGS=(
  --data_dir "${DATA_DIR:-data}"
  --target_col soh
  --seq_len "${SEQ_LEN:-96}"
  --patch_len "${PATCH_LEN:-16}"
  --stride "${STRIDE:-8}"
  --sample_stride "${SAMPLE_STRIDE:-10}"
  --window_stride "${WINDOW_STRIDE:-8}"
  --batch_size "${BATCH_SIZE:-128}"
  --epochs "${EPOCHS:-30}"
  --lr "${LR:-1e-3}"
  --weight_decay "${WEIGHT_DECAY:-1e-4}"
  --dropout "${DROPOUT:-0.2}"
  --patience "${PATIENCE:-5}"
  --lr_factor "${LR_FACTOR:-0.5}"
  --lr_patience "${LR_PATIENCE:-2}"
  --output_dir "$OUT_DIR"
  --checkpoint_dir "$CKPT_DIR"
)

if [[ -n "${DATA_PATH:-}" ]]; then
  ARGS+=(--data_path "$DATA_PATH")
fi
if [[ -n "${FEATURE_COLS:-}" ]]; then
  ARGS+=(--feature_cols "$FEATURE_COLS")
fi

PYTHON_BIN="${PYTHON_BIN:-./.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi

"$PYTHON_BIN" train.py "${ARGS[@]}" 2>&1 | tee "$OUT_DIR/train.log"
