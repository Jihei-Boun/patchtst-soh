#!/usr/bin/env python3
"""차량별 residual / bias / feature-gap 진단 스크립트."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FEATURE_COLS = [
    "soc",
    "socd",
    "pack_volt",
    "pack_current",
    "batt_pw",
    "mod_avg_temp",
    "mod_max_temp",
    "mod_min_temp",
    "batt_internal_temp",
    "ext_temp",
    "int_temp",
    "cell_volt_dispersion",
    "max_cell_volt",
    "min_cell_volt",
    "odometer",
    "chrg_cnt",
    "cumul_energy_chrgd",
    "cumul_pw_chrgd",
    "insul_resistance",
    "sub_batt_volt",
]
TARGET = "soh"
SAMPLE_STRIDE = 10


def normalize_vid(x: str) -> str:
    s = str(x)
    mapping = {
        "1241225178": "01241225178",
        "1241225211": "01241225211",
        "1241225220": "01241225220",
        "1241225226": "01241225226",
    }
    if s in mapping:
        return mapping[s]
    if s.startswith("0"):
        return s
    return s


def load_vehicle(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="|", low_memory=False)
    time_col = "time" if "time" in df.columns else ("msg_time" if "msg_time" in df.columns else None)
    if time_col:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        df = df.sort_values(time_col)
    cols = [c for c in FEATURE_COLS + [TARGET] if c in df.columns]
    work = df[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return work.iloc[::SAMPLE_STRIDE].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vehicle", required=True, help="대상 차량 ID 또는 끝자리 (예: 178)")
    parser.add_argument("--pred_csv", default="outputs/run2_fast/prediction.csv")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--out_dir", default="")
    args = parser.parse_args()

    pred = pd.read_csv(args.pred_csv)
    pred["vehicle"] = pred["device_no"].astype(str).map(normalize_vid)
    vehicles = sorted(pred["vehicle"].unique().tolist())
    target = None
    for v in vehicles:
        if v == args.vehicle or v.endswith(args.vehicle):
            target = v
            break
    if target is None:
        raise SystemExit(f"vehicle '{args.vehicle}' not in {vehicles}")

    out = Path(args.out_dir or f"outputs/analysis_{target[-3:]}")
    out.mkdir(parents=True, exist_ok=True)

    pred["residual"] = pred["y_pred"] - pred["y_true"]
    rows = []
    for vid, g in pred.groupby("vehicle"):
        r = g["residual"]
        rows.append(
            {
                "vehicle": vid,
                "n": len(g),
                "y_true_mean": g["y_true"].mean(),
                "y_pred_mean": g["y_pred"].mean(),
                "bias_mean": r.mean(),
                "bias_median": r.median(),
                "mae": g["absolute_error"].mean(),
                "rmse": float(np.sqrt((r**2).mean())),
                "residual_std": r.std(),
                "pct_ae_gt1": (g["absolute_error"] > 1).mean() * 100,
                "pct_ae_gt2": (g["absolute_error"] > 2).mean() * 100,
                "corr": g["y_true"].corr(g["y_pred"]),
            }
        )
    stats = pd.DataFrame(rows).sort_values("mae", ascending=False)
    stats.to_csv(out / "residual_stats_by_vehicle.csv", index=False)

    g_t = pred[pred["vehicle"] == target].sort_values("time_index").copy()
    g_t["seg"] = pd.qcut(np.arange(len(g_t)), 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
    seg = (
        g_t.groupby("seg", observed=True)
        .agg(
            n=("residual", "size"),
            y_true_mean=("y_true", "mean"),
            y_pred_mean=("y_pred", "mean"),
            bias=("residual", "mean"),
            mae=("absolute_error", "mean"),
        )
        .reset_index()
    )
    seg.to_csv(out / f"{target[-3:]}_bias_by_time_segment.csv", index=False)

    # val vs test bias from prediction? prediction is test-only.
    # Approximate using raw vehicle series split 70/15/15 for feature gap.
    data_dir = Path(args.data_dir)
    feat_rows = []
    test_frames = {}
    for vid in vehicles:
        path = data_dir / f"{vid}.csv.gz"
        if not path.exists():
            continue
        work = load_vehicle(path)
        n = len(work)
        tr = work.iloc[: int(n * 0.7)]
        va = work.iloc[int(n * 0.7) : int(n * 0.85)]
        te = work.iloc[int(n * 0.85) :]
        test_frames[vid] = te
        for split_name, split_df in [("train", tr), ("val", va), ("test", te)]:
            for c in FEATURE_COLS + [TARGET]:
                if c not in split_df.columns:
                    continue
                feat_rows.append(
                    {
                        "vehicle": vid,
                        "split": split_name,
                        "feature": c,
                        "mean": split_df[c].mean(),
                        "std": split_df[c].std(),
                    }
                )
    feat_df = pd.DataFrame(feat_rows)
    feat_df.to_csv(out / "feature_stats_by_split.csv", index=False)

    # test-split feature gap: target vs others
    te = feat_df[feat_df["split"] == "test"]
    pivot = te.pivot(index="feature", columns="vehicle", values="mean")
    others = [v for v in vehicles if v != target and v in pivot.columns]
    pivot["others_mean"] = pivot[others].mean(axis=1)
    pivot["target_minus_others"] = pivot[target] - pivot["others_mean"]
    std_pivot = te.pivot(index="feature", columns="vehicle", values="std")
    pivot["gap_z"] = pivot["target_minus_others"] / (
        std_pivot[others].mean(axis=1).replace(0, np.nan)
    )
    gap = pivot.sort_values("gap_z", key=lambda s: s.abs(), ascending=False)
    gap.to_csv(out / f"{target[-3:]}_feature_mean_gap_vs_others.csv")

    # SOH mean drift train→val→test for target
    soh_drift = feat_df[(feat_df["feature"] == TARGET) & (feat_df["vehicle"] == target)]
    soh_by_split = {r["split"]: r["mean"] for _, r in soh_drift.iterrows()}

    # plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    ax = axes[0, 0]
    for vid, g in pred.groupby("vehicle"):
        ax.hist(g["residual"], bins=50, alpha=0.45, label=vid[-3:], density=True)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_title("Residual density (pred - true)")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(g_t["time_index"], g_t["y_true"], label="true", lw=1.2)
    ax.plot(g_t["time_index"], g_t["y_pred"], label="pred", lw=0.7, alpha=0.85)
    ax.set_title(f"{target[-3:]} SOH true vs pred")
    ax.legend()

    ax = axes[1, 0]
    ax.plot(g_t["time_index"], g_t["residual"], color="C3", lw=0.55)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title(f"{target[-3:]} residual over time")

    ax = axes[1, 1]
    for vid, g in pred.groupby("vehicle"):
        ax.scatter(g["y_true"], g["y_pred"], s=1, alpha=0.12, label=vid[-3:])
    lims = [
        pred[["y_true", "y_pred"]].min().min() - 0.2,
        pred[["y_true", "y_pred"]].max().max() + 0.2,
    ]
    ax.plot(lims, lims, "k--", lw=1)
    ax.set_title("True vs Pred")
    ax.legend(markerscale=5, fontsize=8)
    plt.tight_layout()
    plt.savefig(out / f"{target[-3:]}_residual_overview.png", dpi=140)
    plt.close()

    top = gap.head(10).reset_index()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = ["C3" if x < 0 else "C0" for x in top["gap_z"]]
    ax.barh(top["feature"], top["gap_z"], color=colors)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_title(f"{target[-3:]} feature gap vs others (z)")
    plt.tight_layout()
    plt.savefig(out / f"{target[-3:]}_feature_gap.png", dpi=140)
    plt.close()

    g_t[["time_index", "y_true", "y_pred", "residual", "absolute_error"]].to_csv(
        out / f"{target[-3:]}_residual_series.csv", index=False
    )

    st = stats[stats["vehicle"] == target].iloc[0]
    md = f"""# Vehicle {target} Residual Analysis

기준 예측: `{args.pred_csv}`

## 1. 차량별 residual 요약

| vehicle | n | true mean | pred mean | bias (pred-true) | MAE | RMSE | corr | % AE>1 |
|---------|--:|----------:|----------:|-----------------:|----:|-----:|-----:|-------:|
"""
    for _, r in stats.iterrows():
        md += (
            f"| {r['vehicle']} | {int(r['n'])} | {r['y_true_mean']:.3f} | {r['y_pred_mean']:.3f} | "
            f"{r['bias_mean']:.3f} | {r['mae']:.3f} | {r['rmse']:.3f} | {r['corr']:.3f} | {r['pct_ae_gt1']:.1f} |\n"
        )

    md += f"""
## 2. {target[-3:]} 핵심 소견

- bias ≈ **{st['bias_mean']:.3f}** (pred − true)
- Test true mean ≈ **{st['y_true_mean']:.3f}**, pred mean ≈ **{st['y_pred_mean']:.3f}**
- 상관 ≈ **{st['corr']:.3f}**
- AE>1 ≈ **{st['pct_ae_gt1']:.1f}%**, AE>2 ≈ **{st['pct_ae_gt2']:.1f}%**

### SOH level drift (차량 내부 train→val→test mean)

| split | SOH mean |
|-------|---------:|
| train | {soh_by_split.get('train', float('nan')):.3f} |
| val | {soh_by_split.get('val', float('nan')):.3f} |
| test | {soh_by_split.get('test', float('nan')):.3f} |

> val↔test SOH mean 차이가 크면 validation bias 보정이 test에서 어긋날 수 있음.

## 3. 시간 구간별 bias (5분위)

| seg | n | true mean | pred mean | bias | MAE |
|-----|--:|----------:|----------:|-----:|----:|
"""
    for _, r in seg.iterrows():
        md += (
            f"| {r['seg']} | {int(r['n'])} | {r['y_true_mean']:.3f} | "
            f"{r['y_pred_mean']:.3f} | {r['bias']:.3f} | {r['mae']:.3f} |\n"
        )

    md += f"""
## 4. Feature 분포 gap (test split, {target[-3:]} vs 타차량)

| feature | {target[-3:]} mean | others mean | diff | gap_z |
|---------|---------:|------------:|-----:|------:|
"""
    for feat, r in gap.head(10).iterrows():
        md += (
            f"| {feat} | {r[target]:.4f} | {r['others_mean']:.4f} | "
            f"{r['target_minus_others']:.4f} | {r['gap_z']:.3f} |\n"
        )

    md += f"""
## 5. 해석

1. {target[-3:]}의 주 오차 형태를 residual/bias/상관으로 위 표에서 확인.
2. train→val→test SOH drift가 크면 **val bias ≠ test bias**.
3. feature gap이 크면 domain shift로 공용 모델이 레벨을 틀릴 수 있음.

## Graphs
- `{target[-3:]}_residual_overview.png`
- `{target[-3:]}_feature_gap.png`
"""
    (out / "ANALYSIS.md").write_text(md, encoding="utf-8")
    print(f"[saved] {out}/ANALYSIS.md")
    print(stats.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("SOH drift", soh_by_split)


if __name__ == "__main__":
    main()
