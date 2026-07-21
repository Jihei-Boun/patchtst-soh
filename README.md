# PatchTST SOH

전기차 충전 구간 시계열 데이터로 **배터리 SOH(State of Health)** 를 예측하는 PatchTST 학습 파이프라인입니다.

- 모델: PatchTST 기반 회귀 (Transformer Encoder)
- 입력: `|` 구분자 CSV / `.csv.gz` (차량 1대 이상)
- 출력: MAE / RMSE, 예측 CSV·그래프, 체크포인트

## 빠른 시작

```bash
cd patchtst-soh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 학습 데이터를 data/ 에 배치한 뒤
python inspect_data.py --data_path data/01241225178.csv.gz

python train.py \
  --data_dir data \
  --output_dir outputs/run1 \
  --checkpoint_dir checkpoints/run1
```

GPU(CUDA)가 있으면 자동으로 사용합니다.

배포(알려진 차 / 새 차 캘리브레이션) 규칙은 [`DEPLOYMENT.md`](DEPLOYMENT.md)를 참고하세요.

## 프로젝트 구조

```text
patchtst-soh/
├── data/                 # 학습용 차량 CSV (.csv.gz) — Git 제외
├── outputs/              # 실험 결과 (metrics, REPORT, 그래프 등)
├── checkpoints/          # best_model.pt
├── train.py              # 전처리 → 학습 → 평가 메인
├── inspect_data.py       # 컬럼·결측·SOH 통계 점검
├── run_train.sh          # 단일 파일 빠른 실행 스크립트
└── requirements.txt
```

## 데이터

`data/` 아래에 차량별 `.csv` 또는 `.csv.gz` 를 둡니다. **용량·민감 정보(VIN 등) 때문에 Git에는 올리지 않습니다.** 배치 방법은 [`data/README.md`](data/README.md)를 참고하세요.

| 항목 | 내용 |
|------|------|
| 구분자 | `\|` |
| 타깃 | `soh` |
| 분할 | 차량별 시간순 Train 70% / Val 15% / Test 15% |
| 기본 feature (20) | `soc`, `socd`, `pack_volt`, `pack_current`, `batt_pw`, `mod_avg_temp`, `mod_max_temp`, `mod_min_temp`, `batt_internal_temp`, `ext_temp`, `int_temp`, `cell_volt_dispersion`, `max_cell_volt`, `min_cell_volt`, `odometer`, `chrg_cnt`, `cumul_energy_chrgd`, `cumul_pw_chrgd`, `insul_resistance`, `sub_batt_volt` |

처리 흐름:

```text
충전구간 CSV
  → 컬럼 선택 / 결측 제거 / 시간순 정렬
  → 다운샘플 (sample_stride)
  → 차량별 Train/Val/Test 분할
  → Sliding Window
  → PatchTST 학습 · SOH 예측
  → MAE/RMSE · 그래프 · 체크포인트 저장
```

## 학습 예시

### 여러 차량 (`data/` 전체)

```bash
source .venv/bin/activate
python train.py \
  --data_dir data \
  --sample_stride 10 \
  --window_stride 8 \
  --epochs 8 \
  --output_dir outputs/run3 \
  --checkpoint_dir checkpoints/run3
```

### 해상도를 올린 실험 (`run3_finer`)

```bash
python train.py \
  --data_dir data \
  --sample_stride 5 \
  --window_stride 4 \
  --epochs 10 \
  --output_dir outputs/run3_finer \
  --checkpoint_dir checkpoints/run3_finer
```

### 단일 파일 (`run_train.sh`)

```bash
DATA_PATH=data/01241225178.csv.gz ./run_train.sh
```

### 특정 feature만 사용

```bash
python train.py \
  --data_dir data \
  --feature_cols "soc,pack_volt,pack_current,odometer" \
  --output_dir outputs/feat_ablation
```

## 주요 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--data_dir` | `data` | 폴더 내 모든 `.csv`/`.csv.gz` 사용 |
| `--data_path` | (비움) | 단일 또는 쉼표 구분 다중 파일 |
| `--target_col` | `soh` | 예측 대상 컬럼 |
| `--feature_cols` | (기본 20개) | 쉼표 구분 입력 feature |
| `--seq_len` | `96` | 입력 시퀀스 길이 |
| `--patch_len` | `16` | 패치 길이 |
| `--stride` | `8` | 패치 stride |
| `--sample_stride` | `10` | 원본 시계열 다운샘플 (클수록 빠름) |
| `--window_stride` | `8` | 슬라이딩 윈도우 시작 간격 |
| `--batch_size` | `128` | 배치 크기 |
| `--epochs` | `8` | 학습 epoch |
| `--lr` | `1e-3` | 학습률 |
| `--d_model` / `--nhead` / `--num_layers` | `128` / `8` / `3` | Transformer 크기 |
| `--output_dir` | `outputs` | 메트릭·예측·그래프 저장 |
| `--checkpoint_dir` | `checkpoints` | `best_model.pt` 저장 |

전체 목록은 `python train.py --help` 로 확인할 수 있습니다.

## 산출물

실험마다 `outputs/<실험명>/` 에 다음이 생성됩니다.

| 파일 | 내용 |
|------|------|
| `metrics.json` | Test MAE/RMSE, 차량별 지표, 하이퍼파라미터 |
| `prediction.csv` | 실제값·예측값 |
| `prediction.png` | 예측 시각화 |
| `training_loss.png` | 학습/검증 loss |
| `REPORT.md` | 실험 보고 (작성 시) |

체크포인트: `checkpoints/<실험명>/best_model.pt`  
(모델 가중치, scaler, feature 목록, 하이퍼파라미터 포함)

참고 실험 결과: [`outputs/run3/REPORT.md`](outputs/run3/REPORT.md), [`outputs/run3_finer/REPORT.md`](outputs/run3_finer/REPORT.md)

## 의존성

- Python 3.10+ 권장
- `torch`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`

```bash
pip install -r requirements.txt
```

## 라이선스 / 데이터

학습 원본 데이터는 저장소에 포함되지 않습니다. 사내·연구용 텔레메트리이므로 외부 공유 시 정책을 확인하세요.
