# PatchTST SOH 예측 실험 보고

- 작성일: 2026-07-21
- 프로젝트 경로: `Project/patchtst-soh`
- 예측 대상: `soh` (State of Health)
- 모델: PatchTST 기반 회귀 모델 (Transformer Encoder)

---

## 1. 실험 목적

전처리된 전기차 충전 구간 CSV 데이터를 이용해 **배터리 SOH를 예측**하는 PatchTST 학습 파이프라인을 구성하고, 차량 4대 데이터로 학습·평가 결과를 확보한다.

하이퍼파라미터와 입력 feature는 자율 설정하며, 성능 지표는 **Test MAE / Test RMSE**로 보고한다.

---

## 2. 프로젝트 구성

```text
patchtst-soh/
├── data/                          # 학습용 차량 CSV (.csv.gz)
├── outputs/
│   └── run2_fast/                 # 최종 학습 결과
│       ├── metrics.json
│       ├── prediction.csv
│       ├── prediction.png
│       ├── training_loss.png
│       └── train.log
├── checkpoints/
│   └── run2_fast/
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

학습 feature / target 컬럼을 정하기 전에 사용한다.

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

주요 기능:
- `data/` 폴더의 여러 차량 CSV를 한 번에 학습
- 예측 대상 기본값: `soh`
- 배터리 관련 기본 feature 자동 선택 (또는 `--feature_cols`로 직접 지정)
- 차량별로 시간순 분할 후 합쳐 학습 (윈도우가 차량 경계를 넘지 않음)
- Validation loss가 가장 낮은 모델을 `best_model.pt`로 저장
- 차량별 / 전체 Test MAE·RMSE 계산
- `metrics.json`, `prediction.csv`, `prediction.png`, `training_loss.png` 저장

마감용 속도 옵션 (기본값):
- `--sample_stride 10` : 원본 시계열 10행마다 1행 사용
- `--window_stride 8` : 슬라이딩 윈도우 시작 간격
- `--batch_size 128`
- `--epochs 8`

실행 예:

```bash
source .venv/bin/activate
python train.py --data_dir data --output_dir outputs/run2_fast --checkpoint_dir checkpoints/run2_fast
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
checkpoints/run2_fast/best_model.pt
```

---

## 4. 사용 데이터

| 차량 ID | 파일 | SOH 범위 (확인값) |
|---|---|---|
| 01241225178 | `data/01241225178.csv.gz` | 96.7 ~ 100.0 |
| 01241225211 | `data/01241225211.csv.gz` | 94.7 ~ 100.0 |
| 01241225220 | `data/01241225220.csv.gz` | 94.0 ~ 100.0 |
| 01241225226 | `data/01241225226.csv.gz` | 91.1 ~ 94.0 |

- 데이터 형식: `|` 구분자 `.csv.gz`
- 예측 대상: `soh`
- 분할: 차량별 시간순 70% / 15% / 15%

### 입력 Feature (20개)

`soc`, `socd`, `pack_volt`, `pack_current`, `batt_pw`,  
`mod_avg_temp`, `mod_max_temp`, `mod_min_temp`,  
`batt_internal_temp`, `ext_temp`, `int_temp`,  
`cell_volt_dispersion`, `max_cell_volt`, `min_cell_volt`,  
`odometer`, `chrg_cnt`, `cumul_energy_chrgd`, `cumul_pw_chrgd`,  
`insul_resistance`, `sub_batt_volt`

---

## 5. 학습 설정

| 항목 | 값 |
|---|---|
| 모델 | PatchTST Regressor |
| device | CUDA |
| seq_len | 96 |
| patch_len | 16 |
| patch stride | 8 |
| sample_stride | 10 |
| window_stride | 8 |
| batch_size | 128 |
| epochs | 8 |
| learning_rate | 0.001 |
| 손실함수 | MSE |
| 최적화 | Adam |

다운샘플 후 대략적인 규모:
- train/val/test rows: 416,374 / 89,223 / 89,225
- train/val windows: 52,001 / 11,106

---

## 6. 결과 요약

### 6.1 전체 성능

| 지표 | 값 |
|---|---|
| **Test MAE** | **1.036** |
| **Test RMSE** | **1.236** |
| Best Validation Loss (scaled MSE) | 0.168 |

### 6.2 차량별 성능

| 차량 ID | MAE | RMSE | Test 샘플 수 |
|---|---:|---:|---:|
| 01241225178 | 1.101 | 1.173 | 20,157 |
| 01241225211 | 0.359 | 0.389 | 34,241 |
| 01241225220 | 1.771 | 1.797 | 31,381 |
| 01241225226 | 0.656 | 0.686 | 3,062 |

### 6.3 결과 파일

| 파일 | 설명 |
|---|---|
| `outputs/run2_fast/metrics.json` | 전체/차량별 지표 및 설정값 |
| `outputs/run2_fast/prediction.csv` | 실제 SOH / 예측 SOH / 절대오차 |
| `outputs/run2_fast/prediction.png` | 실제 vs 예측 SOH 그래프 |
| `outputs/run2_fast/training_loss.png` | Train / Validation loss 곡선 |
| `checkpoints/run2_fast/best_model.pt` | 최고 성능 모델 체크포인트 |

---

## 7. 실험 과정에서 확인한 점

1. 원본 CSV는 쉼표가 아니라 **`|` 구분자**를 사용한다.
2. 결측이 많은 컬럼(`seq` 등)을 feature에 넣으면 `dropna` 후 유효 행이 사라질 수 있다.
3. 차량 1대만 시간순으로 나누면, train 구간에 SOH가 거의 일정한 경우 모델이 상수에 가깝게 예측할 수 있다.
4. 차량 4대 + 다운샘플/윈도우 stride 적용으로 **약 2분 내** 학습·평가를 완료할 수 있었다.
5. `best_model.pt`는 바이너리이므로 에디터로 열지 않고, 재추론/재사용용으로 보관한다.

---

## 8. 재현 방법

```bash
cd ~/Jihei/Project/patchtst-soh
source .venv/bin/activate

# (선택) 데이터 확인
python inspect_data.py --data_path data/01241225178.csv.gz

# 학습
python train.py \
  --data_dir data \
  --output_dir outputs/run2_fast \
  --checkpoint_dir checkpoints/run2_fast
```

필요 패키지: `requirements.txt`  
(`torch`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`)

---

## 9. 결론

- PatchTST 기반 SOH 예측 파이프라인을 구성하고, **차량 4대**로 학습·평가를 완료했다.
- 최종 성능은 **Test MAE 1.036 / Test RMSE 1.236**이다.
- 차량별로 성능 차이가 있으며, `01241225211`이 가장 낮고 `01241225220`이 상대적으로 높다.
- 코드·결과·체크포인트는 `outputs/run2_fast/` 및 `checkpoints/run2_fast/`에 정리되어 있다.
