"""전처리된 CSV 컬럼/결측치 확인용 스크립트."""

import argparse

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/01241225186.csv.gz")
    args = parser.parse_args()

    df = pd.read_csv(args.data_path, sep="|", low_memory=False)

    print("=" * 60)
    print(f"파일: {args.data_path}")
    print(f"데이터 크기: {df.shape}")
    print("=" * 60)

    print("\n[컬럼 목록]")
    for col in df.columns:
        print(col)

    print("\n[앞부분 5행]")
    print(df.head())

    print("\n[데이터 타입]")
    print(df.dtypes)

    soh_cols = [c for c in df.columns if c.lower() == "soh" or "soh" in c.lower()]
    print("\n[SOH 관련 컬럼]")
    print(soh_cols if soh_cols else "없음")

    print("\n[숫자형 컬럼]")
    num_cols = df.select_dtypes(include="number").columns.tolist()
    print(num_cols)

    print("\n[결측치 상위 20개 컬럼]")
    print(df.isna().sum().sort_values(ascending=False).head(20))

    if soh_cols:
        target = soh_cols[0]
        print(f"\n[SOH({target}) 기본 통계]")
        print(df[target].describe())


if __name__ == "__main__":
    main()
