# SOH 예측 배포 프로토콜

실험(`outputs/Experiment_Report.md`) 기준 운영 규칙입니다.

## 한 줄 요약

| 차량 유형 | 파이프라인 | 기대 |
|-----------|------------|------|
| **알려진 차량** (학습에 포함된 ID) | `run7b_norm_bias` | Test MAE ≈ **0.65** |
| **새 차량** (미학습) | LOVO 베이스 + **캘리브레이션** | 220처럼 domain이 다르면 zero-shot 불가 |

---

## A. 알려진 차량 (Known-vehicle)

1. 체크포인트: `checkpoints/run7_vehicle_norm/best_model.pt`
2. 전처리: **차량별** feature/target StandardScaler (`--per_vehicle_norm`)
3. 추론 후: Validation에서 추정한 차량별 bias 보정  
   `pred_corrected = pred − bias_val`
4. (선택) median filter는 MAE에 거의 영향 없음 → 필수 아님

```bash
python train.py \
  --data_dir data \
  --sample_stride 10 --window_stride 8 \
  --per_vehicle_norm --vehicle_bias_correct \
  --eval_checkpoint checkpoints/run7_vehicle_norm/best_model.pt \
  --output_dir outputs/deploy_known \
  --checkpoint_dir checkpoints/deploy_known
```

**주의:** 178은 val↔test bias 부호가 어긋날 수 있어, 상수 bias만으로는 한계가 있습니다.  
온도 0 고정 차량은 `--fix_zero_temp`로 `int_temp`/`ext_temp`를 `mod_avg_temp` 등으로 대체하세요.

---

## B. 새 차량 (Unknown / LOVO)

### B-0. Zero-shot (캘리브레이션 없음) — 기본 비권장

- 학습 차량만으로 학습한 글로벌 모델로 바로 추론
- LOVO 결과: **178 ≈ 0.71 (가능 쪽)**, **220 ≈ 2.75 (실패)**
- Domain shift가 큰 차는 **배포 전 캘리브레이션 필수**

### B-1. Few-shot 캘리브레이션 (권장 기본)

새 차량의 **Test(또는 운영 직전) 구간 앞부분 라벨**로 level을 맞춘 뒤 나머지에 적용합니다.  
(전체 시계열 초반으로 잡으면 bias가 비정상이라 악화될 수 있음 → **Test head** 사용)

| 단계 | 내용 |
|------|------|
| 1 | 기존 차량으로 베이스 모델 학습 (또는 `lovo_holdout_*` 체크포인트) |
| 2 | 새 차량 Test 앞 `calibrate_frac` (권장 **5~10%**)을 캘리브 구간으로 확보 |
| 3 | (선택) 캘리브 구간으로 `finetune_epochs` 짧은 fine-tune |
| 4 | 캘리브 구간에서 `bias = mean(pred − true)` 추정 |
| 5 | 이후 Test 구간에 `pred − bias` 적용 후 운영 |

```bash
# 예: 220을 새 차량으로 가정, Test 5% 캘리브 + bias
python train.py \
  --data_dir data \
  --holdout_vehicle 220 \
  --calibrate_frac 0.05 \
  --vehicle_bias_correct \
  --sample_stride 10 --window_stride 8 \
  --eval_checkpoint checkpoints/lovo_holdout_220/best_model.pt \
  --output_dir outputs/calib_220_testh_f5 \
  --checkpoint_dir checkpoints/calib_220_testh_f5
```

### 캘리브레이션 윈도우 가이드 (220 LOVO 기준)

| 설정 | Holdout MAE | 비고 |
|------|------------:|------|
| Zero-shot | 2.751 | 캘리브 없음 |
| **Test 5% + bias** | **1.198** | **권장 기본** (≈ −56%) |
| Test 10% + bias | 1.291 | |
| Test 20% + bias | 1.398 | 길수록 이득 감소(비정상 bias) |
| Test 10% + FT5 + bias | 1.252 | FT 후 raw≈1.07이면 bias 생략 검토 |

실측 주기가 다르면 Test 구간의 5~10%를 “N일”로 환산하면 됩니다.

---

## C. 의사결정 플로우

```text
새 차량 인가?
  ├─ No  → Known 파이프라인 (per-vehicle norm + val bias)
  └─ Yes → 캘리브 라벨 확보 가능한가?
            ├─ No  → Zero-shot (위험: 220형 domain은 실패 가능) → 모니터링 강화
            └─ Yes → calibrate_frac 0.1~0.2
                      ├─ bias만으로 MAE 목표 충족? → 배포
                      └─ 미달 → + short fine-tune 후 재평가
```

---

## D. 운영 체크리스트

- [ ] 차량 ID가 학습 집합에 있는지 확인
- [ ] 알려진 차: per-vehicle scaler + val bias 파일/메타 저장
- [ ] 새 차: 캘리브 구간 길이·bias 값·(선택) finetuned 체크포인트 저장
- [ ] `int_temp`/`ext_temp` 0 고정 여부 점검 → `--fix_zero_temp`
- [ ] 차량별 Test MAE / bias 부호를 대시보드에 기록 (178형 부호 반전 감시)

---

## E. 현재 기준 수치 (참고)

| 설정 | MAE |
|------|----:|
| Known best (`run7b_norm_bias`) | 0.647 |
| LOVO 178 zero-shot | 0.710 |
| LOVO 220 zero-shot | 2.751 |
| 220 few-shot Test 5% + bias | **1.198** |
| 178 `--fix_zero_temp` (bias 없음) | 차량 178 MAE **0.546** (run2 1.101 대비 개선) |
