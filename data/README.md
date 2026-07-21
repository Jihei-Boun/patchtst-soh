# 데이터 배치 안내

이 폴더에는 학습용 차량 CSV(`.csv.gz`)를 둡니다.

Git에는 **올리지 않습니다** (용량 + VIN 등 민감 정보).

예시:
```bash
# data/ 아래에 파일 배치 후
DATA_PATH=data/01241225178.csv.gz ./run_train.sh
```

필요 컬럼은 `train.py` / `inspect_data.py`를 참고하세요.
