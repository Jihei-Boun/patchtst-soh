# PatchTST SOH 예측 실험 보고 (run3_finer)

- 작성일: 2026-07-21
- 실험명: `run3_finer`
- 프로젝트 경로: `Project/patchtst-soh`
- 예측 대상: `soh` (State of Health)
- 모델: PatchTST 기반 회귀 모델 (Transformer Encoder)

---

## 1. 실험 목적

전처리된 전기차 충전 구간 CSV 데이터를 이용해 **배터리 SOH를 예측**하는 PatchTST 학습 파이프라인을 구성하고, 차량 4대 데이터로 학습·평가 결과를 확보한다.

`run3`(기본 속도 옵션) 대비 **다운샘플·윈도우를 더 세밀하게** 두고 epoch를 늘려, 해상도 증가가 Test MAE/RMSE에 미치는 영향을 확인한다.

성능 지표는 **Test MAE / Test RMSE**로 보고한다.

---

## 2. 프로젝트 구성

```text
patchtst-soh/
├── data/                          # 학습용 차량 CSV (.csv.gz)
├── outputs/
│   └── run3_finer/                # 본 실험 결과
│       ├── metrics.json
│       ├── prediction.csv
│       ├── prediction.png
│       ├── training_loss.png
│       └── REPORT.md
├── checkpoints/
│   └── run3_finer/
│       └── best_model.pt          # 최고 성능 모델 가중치 (바이너리)
├── train.py                       # 학습/평가 메인 코드
├── inspect_data.py                # 데이터 컬럼·결측치 확인 스크립트
├── requirements.txt
└── run_train.sh
```

---

## 3. 주요 파일 역할

### 3.1 `inspect_data.py`

**역할:** 학습 전에 CSV 데이터 구조를 확인하는 점검용 스크립트.

하는 일:
- `|` 구분자 CSV/`.csv.gz` 로드
- 데이터 크기(`shape`) 출력
- 컬럼 목록, 앞부분 샘플, dtype 확인
- `soh` 관련 컬럼 탐지
- 숫자형 컬럼 목록 출력
- 결측치가 많은 컬럼 상위 출력
- SOH 기본 통계 출력

실행 예:

```bash
python inspect_data.py --data_path data/01241225178.csv.gz
```

### 3.2 `train.py`

**역할:** PatchTST 기반 SOH 예측 모델의 **데이터 전처리 → 학습 → 평가 → 결과 저장**을 한 번에 수행하는 메인 코드.

처리 흐름:

```text
충전구간 CSV (차량 1대 이상)
    ↓
컬럼 선택 / 결측 제거 / 시간순 정렬
    ↓
차량별 Train / Validation / Test 분할 (시간순 70% / 15% / 15%)
    ↓
Sliding Window 생성
    ↓
PatchTST 학습
    ↓
SOH 예측
    ↓
MAE / RMSE 및 그래프 저장
```

본 실험에서 사용한 설정:
- `--sample_stride 5` : 원본 시계열 5행마다 1행 사용 (`run3`의 10보다 촘촘)
- `--window_stride 4` : 슬라이딩 윈도우 시작 간격 (`run3`의 8보다 촘촘)
- `--batch_size 128`
- `--epochs 10`

실행 명령:

```bash
source .venv/bin/activate
python train.py \
  --data_dir data \
  --sample_stride 5 \
  --window_stride 4 \
  --epochs 10 \
  --output_dir outputs/run3_finer \
  --checkpoint_dir checkpoints/run3_finer
```

### 3.3 `best_model.pt`

학습 중 validation 성능이 가장 좋았던 모델 체크포인트이다.
에디터로 열어볼 수 있는 텍스트 파일이 아니라 **PyTorch 바이너리**이다.

포함 내용 예:
- `model_state_dict` (모델 가중치)
- feature / target scaler
- feature 컬럼 목록
- 모델 하이퍼파라미터 설정

경로:

```text
checkpoints/run3_finer/best_model.pt
```

본 실험에서는 **Epoch 2**의 validation loss가 최저(0.183)였고, 해당 체크포인트로 평가했다.

---

## 4. 사용 데이터

| 차량 ID | 파일 | SOH 범위 (확인값) | 다운샘플 후 행 수 |
|---|---|---|---:|
| 01241225178 | `data/01241225178.csv.gz` | 96.7 ~ 100.0 | 270,040 |
| 01241225211 | `data/01241225211.csv.gz` | 94.7 ~ 100.0 | 457,818 |
| 01241225220 | `data/01241225220.csv.gz` | 94.0 ~ 100.0 | 419,683 |
| 01241225226 | `data/01241225226.csv.gz` | 91.1 ~ 94.0 | 42,102 |

- 데이터 형식: `|` 구분자 `.csv.gz`
- 예측 대상: `soh`
- 분할: 차량별 시간순 70% / 15% / 15%
- `sample_stride=5`로 `run3` 대비 약 2배 행 수 사용

### 입력 Feature (20개)

`soc`, `socd`, `pack_volt`, `pack_current`, `batt_pw`,  
`mod_avg_temp`, `mod_max_temp`, `mod_min_temp`,  
`batt_internal_temp`, `ext_temp`, `int_temp`,  
`cell_volt_dispersion`, `max_cell_volt`, `min_cell_volt`,  
`odometer`, `chrg_cnt`, `cumul_energy_chrgd`, `cumul_pw_chrgd`,  
`insul_resistance`, `sub_batt_volt`

---

## 5. 학습 설정

| 항목 | run3 (비교) | **run3_finer** |
|---|---|---|
| 모델 | PatchTST Regressor | 동일 |
| device | CUDA | 동일 |
| seq_len | 96 | 96 |
| patch_len | 16 | 16 |
| patch stride | 8 | 8 |
| **sample_stride** | 10 | **5** |
| **window_stride** | 8 | **4** |
| batch_size | 128 | 128 |
| **epochs** | 8 | **10** |
| learning_rate | 0.001 | 0.001 |
| 손실함수 | MSE | MSE |
| 최적화 | Adam | Adam |

다운샘플 후 규모:
- train/val/test rows: 832,749 / 178,446 / 178,448 (`run3`의 약 2배)
- train/val windows: 208,092 / 44,517 (`run3` 52,001 / 11,106 대비 약 4배)

### Epoch별 loss

| Epoch | train_loss | val_loss |
|---:|---:|---:|
| 1 | 0.094407 | 0.216516 |
| 2 | 0.042255 | **0.183329** (best) |
| 3 | 0.048539 | 0.248213 |
| 4 | 0.026678 | 0.240848 |
| 5 | 0.023508 | 0.200897 |
| 6 | 0.044386 | 0.314972 |
| 7 | 0.028071 | 0.324359 |
| 8 | 0.049463 | 0.251019 |
| 9 | 0.032845 | 0.540501 |
| 10 | 0.026341 | 0.253521 |

Epoch 2 이후 validation loss가 다시 올라가는 구간이 있어, 조기 최고점(Epoch 2) 체크포인트가 최종 평가에 사용되었다.

---

## 6. 결과 요약

### 6.1 전체 성능

| 지표 | 값 |
|---|---|
| **Test MAE** | **1.101** |
| **Test RMSE** | **1.388** |
| Best Validation Loss (scaled MSE) | 0.183 |
| Test Loss (scaled MSE) | 0.438 |

### 6.2 차량별 성능

| 차량 ID | MAE | RMSE | Test 샘플 수 |
|---|---:|---:|---:|
| 01241225178 | 1.057 | 1.124 | 40,410 |
| 01241225211 | 0.282 | 0.361 | 68,577 |
| 01241225220 | 2.095 | 2.118 | 62,857 |
| 01241225226 | 0.372 | 0.401 | 6,220 |

### 6.3 `run3`와의 비교

| 실험 | sample_stride | window_stride | epochs | Test MAE | Test RMSE |
|---|---:|---:|---:|---:|---:|
| run3 | 10 | 8 | 8 | **1.036** | **1.236** |
| run3_finer | 5 | 4 | 10 | 1.101 | 1.388 |

차량별 MAE 비교:

| 차량 ID | run3 MAE | run3_finer MAE | 변화 |
|---|---:|---:|---|
| 01241225178 | 1.101 | 1.057 | 개선 |
| 01241225211 | 0.359 | 0.282 | 개선 |
| 01241225220 | 1.771 | 2.095 | 악화 |
| 01241225226 | 0.656 | 0.372 | 개선 |

세밀 샘플링으로 일부 차량은 좋아졌으나, `01241225220` 오차 증가가 전체 MAE/RMSE를 `run3`보다 높게 만들었다.

### 6.4 결과 파일

| 파일 | 설명 |
|---|---|
| `outputs/run3_finer/metrics.json` | 전체/차량별 지표 및 설정값 |
| `outputs/run3_finer/prediction.csv` | 실제 SOH / 예측 SOH / 절대오차 |
| `outputs/run3_finer/prediction.png` | 실제 vs 예측 SOH 그래프 |
| `outputs/run3_finer/training_loss.png` | Train / Validation loss 곡선 |
| `checkpoints/run3_finer/best_model.pt` | 최고 성능 모델 체크포인트 |

---

## 7. 실험 과정에서 확인한 점

1. `sample_stride`/`window_stride`를 줄이면 윈도우 수가 크게 늘어나 학습 시간이 길어진다.
2. 데이터가 촘촘해져도 **전체 Test 성능이 자동으로 좋아지지는 않았다** (`run3` 대비 MAE·RMSE 상승).
3. 차량 `01241225211`, `01241225226` 등은 개선되었으나, `01241225220`이 전체 지표를 끌어올렸다.
4. Best validation은 Epoch 2에서 나왔고, 이후 val_loss가 불안정해 early-stop에 가까운 선택이 유효했다.
5. `best_model.pt`는 바이너리이므로 에디터로 열지 않고, 재추론/재사용용으로 보관한다.

---

## 8. 재현 방법

```bash
cd ~/Jihei/Project/patchtst-soh
source .venv/bin/activate

# (선택) 데이터 확인
python inspect_data.py --data_path data/01241225178.csv.gz

# 학습 (run3_finer)
python train.py \
  --data_dir data \
  --sample_stride 5 \
  --window_stride 4 \
  --epochs 10 \
  --output_dir outputs/run3_finer \
  --checkpoint_dir checkpoints/run3_finer
```

필요 패키지: `requirements.txt`  
(`torch`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`)

---

## 9. 결론

- 세밀 설정(`sample_stride=5`, `window_stride=4`, `epochs=10`)으로 차량 4대 학습·평가를 완료했다.
- 최종 성능은 **Test MAE 1.101 / Test RMSE 1.388**이다.
- 동일 데이터·동일 feature 기준으로는 **기본 속도 옵션의 `run3`(MAE 1.036)가 전체 지표에서 더 좋았다.**
- 코드·결과·체크포인트는 `outputs/run3_finer/` 및 `checkpoints/run3_finer/`에 정리되어 있다.
