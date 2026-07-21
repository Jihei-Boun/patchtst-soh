# PatchTST SOH 학습 결과 공유용

> 짧게 말할 때 이 문서만 열면 됩니다. (상세 실험: [`Experiment_Report.md`](Experiment_Report.md))

---

## 한 줄 요약

전기차 충전 시계열로 **PatchTST → SOH 예측**.  
알려진 차는 MAE **0.65**, 새 차(220)는 zero-shot **2.75** → **5% 캘리브로 1.20**.

---

## 말할 멘트 (약 40초)

> 충전 구간 시계열로 PatchTST로 배터리 SOH를 예측했습니다.  
> 기본 모델 MAE는 **1.04**였고, 차량별 정규화와 bias 보정으로 **0.65**까지 내렸습니다.  
> 다만 차량을 빼고 테스트하면(새 차량 가정) 220호는 **2.75**로 크게 나빠집니다.  
> Test 앞부분 **5% 라벨만**으로 bias를 맞추면 **1.20**까지 회복됩니다.  
> 그래서 운영은 **알려진 차 = 보정 파이프라인**, **새 차 = 짧은 캘리브레이션**으로 가져가면 됩니다.

---

## 핵심 숫자 (이 표만 보여줘도 됨)

| 구분 | 실험 | Test MAE | 의미 |
|------|------|---------:|------|
| Baseline | `run2_fast` | 1.036 | 기본 학습 |
| **Known best** | `run7b_norm_bias` | **0.647** | 차량별 norm + bias |
| New-vehicle 실패 | `lovo_holdout_220` | **2.751** | 220 제외 학습 → zero-shot |
| **New-vehicle 회복** | `calib_220_testh_f5` | **1.198** | Test 5% + bias 캘리브 |

```text
1.04  →  0.65  →  (새 차) 2.75  →  (캘리브) 1.20
기본      known      zero-shot         5% bias
```

---

## 보여줄 그래프

### 1) Known-vehicle (잘 되는 경우)

`run7b` — 차량별 정규화 + validation bias 보정

![run7b prediction](run7b_norm_bias/prediction.png)

### 2) New-vehicle zero-shot (실패)

220을 학습에서 빼고 바로 예측 → level이 크게 어긋남

![lovo 220](lovo_holdout_220/prediction.png)

### 3) New-vehicle + 5% 캘리브 (회복)

Test 앞 5%로 bias만 맞춘 뒤 나머지 평가

![calib 220 f5](calib_220_testh_f5/prediction.png)

---

## 운영 결론 (질문 나오면)

| 차량 유형 | 방법 | 기대 |
|-----------|------|------|
| 알려진 차량 | per-vehicle norm + val bias (`run7b`) | MAE ≈ **0.65** |
| 새 차량 | Test(운영) 앞 **5~10%** 라벨로 bias 캘리브 | 220 기준 **2.75 → 1.20** |

상세 규칙: [`../DEPLOYMENT.md`](../DEPLOYMENT.md)

---

## Q&A 짧게

| 질문 | 답 |
|------|----|
| 왜 차량별로 나눠? | 차마다 SOH level bias가 다름 (특히 220은 체계적 저추정) |
| Embedding/Huber는? | 시도했지만 known best(0.65)를 못 넘김 |
| 178은? | 온도 0 고정 → `--fix_zero_temp`로 해당 차 MAE 1.10→0.55 개선. 전체 best는 여전히 run7b |
| 다음에 뭐 하면? | 새 차 캘리브 길이(일 단위) 현장 기준 확정, 178형 시간 가변 bias |

---

## 파일 위치

| 용도 | 경로 |
|------|------|
| 이 공유 자료 | `outputs/Share_Summary.md` |
| 전체 실험 리포트 | `outputs/Experiment_Report.md` |
| 배포 프로토콜 | `DEPLOYMENT.md` |
| Known best 그래프 | `outputs/run7b_norm_bias/prediction.png` |
| 220 zero-shot | `outputs/lovo_holdout_220/prediction.png` |
| 220 캘리브 | `outputs/calib_220_testh_f5/prediction.png` |
