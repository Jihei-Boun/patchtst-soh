#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

mkdir -p outputs checkpoints

python train.py \
  --data_path "${DATA_PATH:-data/01241225186.csv.gz}" \
  --target_col soh \
  ${FEATURE_COLS:+--feature_cols "$FEATURE_COLS"} \
  --seq_len "${SEQ_LEN:-96}" \
  --patch_len "${PATCH_LEN:-16}" \
  --stride "${STRIDE:-8}" \
  --batch_size "${BATCH_SIZE:-32}" \
  --epochs "${EPOCHS:-20}" \
  --lr "${LR:-1e-3}" \
  2>&1 | tee outputs/train.log
