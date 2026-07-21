#!/usr/bin/env bash
# 220 few-shot(Test head) + 178 temp fix 재실행용
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate

CKPT220=checkpoints/lovo_holdout_220/best_model.pt

echo "========== 220 few-shot on TEST head =========="
for FRAC in 0.05 0.10 0.20; do
  TAG=$(python3 -c "print(f'{float(\"$FRAC\")*100:.0f}')")
  python train.py \
    --data_dir data \
    --holdout_vehicle 220 \
    --calibrate_frac "$FRAC" \
    --vehicle_bias_correct \
    --sample_stride 10 --window_stride 8 \
    --batch_size 128 \
    --weight_decay 0 --dropout 0.1 --patience 0 \
    --eval_checkpoint "$CKPT220" \
    --output_dir "outputs/calib_220_testh_f${TAG}" \
    --checkpoint_dir "checkpoints/calib_220_testh_f${TAG}"
done

python train.py \
  --data_dir data \
  --holdout_vehicle 220 \
  --calibrate_frac 0.10 \
  --finetune_epochs 5 --finetune_lr 1e-4 \
  --vehicle_bias_correct \
  --sample_stride 10 --window_stride 8 \
  --batch_size 128 \
  --weight_decay 0 --dropout 0.1 --patience 0 \
  --eval_checkpoint "$CKPT220" \
  --output_dir outputs/calib_220_testh_f10_ft5 \
  --checkpoint_dir checkpoints/calib_220_testh_f10_ft5

echo "========== 178 fix_zero_temp =========="
python train.py \
  --data_dir data --fix_zero_temp --no-vehicle_bias_correct \
  --sample_stride 10 --window_stride 8 --epochs 8 --lr 1e-3 \
  --weight_decay 0 --dropout 0.1 --patience 0 \
  --output_dir outputs/run10_fix_temp \
  --checkpoint_dir checkpoints/run10_fix_temp

python train.py \
  --data_dir data --fix_zero_temp --vehicle_bias_correct \
  --sample_stride 10 --window_stride 8 \
  --weight_decay 0 --dropout 0.1 --patience 0 \
  --eval_checkpoint checkpoints/run10_fix_temp/best_model.pt \
  --output_dir outputs/run10b_fix_temp_bias \
  --checkpoint_dir checkpoints/run10b_fix_temp_bias

python train.py \
  --data_dir data --fix_zero_temp --per_vehicle_norm --vehicle_bias_correct \
  --sample_stride 10 --window_stride 8 --epochs 8 --lr 1e-3 \
  --weight_decay 0 --dropout 0.1 --patience 0 \
  --output_dir outputs/run10c_fix_temp_norm_bias \
  --checkpoint_dir checkpoints/run10c_fix_temp_norm_bias

echo "[DONE]"
