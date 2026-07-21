import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import ConcatDataset, DataLoader, Dataset


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_vehicle_csv(csv_path: str) -> pd.DataFrame:
    """'|' 구분자로 차량 CSV를 읽습니다."""
    return pd.read_csv(csv_path, sep="|", low_memory=False)


def vehicle_id_from_path(csv_path: str) -> str:
    name = os.path.basename(csv_path)
    if name.endswith(".csv.gz"):
        return name[: -len(".csv.gz")]
    if name.endswith(".csv"):
        return name[: -len(".csv")]
    return os.path.splitext(name)[0]


# 기본 feature에서 제외할 컬럼 (ID/리스트/문자열/전부결측)
EXCLUDE_FEATURE_COLS = {
    "chg_seg",
    "chg_mode",
    "device_no",
    "measured_month",
    "msg_time",
    "time",
    "start_time",
    "msg_id",
    "vin",
    "seq",
    "cell_volt_list",
    "hvac_list1",
    "hvac_list2",
    "mod_temp_list",
}

# SOH 예측에 쓸 기본 입력 필드 (정확도 실험 시작점)
DEFAULT_FEATURE_COLS = [
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


def infer_time_column(df: pd.DataFrame) -> str:
    candidates = [
        "timestamp",
        "time",
        "msg_time",
        "start_time",
        "datetime",
        "date",
        "created_at",
    ]
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in lower_map:
            return lower_map[cand]
    return ""


def resolve_column(df: pd.DataFrame, name: str) -> str:
    if name in df.columns:
        return name
    lower_map = {c.lower(): c for c in df.columns}
    if name.lower() in lower_map:
        return lower_map[name.lower()]
    raise ValueError(f"컬럼 '{name}' 을(를) 찾지 못했습니다. 사용 가능: {list(df.columns)}")


def infer_target_column(df: pd.DataFrame, user_target: str = "soh") -> str:
    return resolve_column(df, user_target)


def select_feature_columns(
    df: pd.DataFrame, target_col: str, user_features: List[str] | None = None
) -> List[str]:
    if user_features:
        resolved = [resolve_column(df, col) for col in user_features]
        if target_col in resolved:
            raise ValueError("feature_cols에 예측 대상(soh)은 포함할 수 없습니다.")
        return resolved

    # 기본: SOH 관련 배터리 센서 우선 사용
    defaults = [c for c in DEFAULT_FEATURE_COLS if c in df.columns and c != target_col]
    if defaults:
        return defaults

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in num_cols:
        num_cols.remove(target_col)
    num_cols = [c for c in num_cols if c not in EXCLUDE_FEATURE_COLS]
    if not num_cols:
        raise ValueError("사용 가능한 숫자형 feature 컬럼이 없습니다.")
    return num_cols


def parse_time_series(series: pd.Series) -> pd.Series:
    """aicar_charge time 형식 예: 22-12-08 14:22:04"""
    parsed = pd.to_datetime(series, format="%y-%m-%d %H:%M:%S", errors="coerce")
    if parsed.isna().mean() > 0.5:
        parsed = pd.to_datetime(series, errors="coerce")
    return parsed


def filter_usable_features(work: pd.DataFrame, feature_cols: List[str], max_missing: float = 0.2) -> List[str]:
    kept = []
    for col in feature_cols:
        miss = work[col].isna().mean()
        if miss <= max_missing:
            kept.append(col)
        else:
            print(f"[WARN] feature 제외 (결측 {miss:.1%}): {col}")
    if not kept:
        raise ValueError("결측 비율이 낮은 feature가 없습니다.")
    return kept


def split_by_time(
    n: int, train_ratio: float = 0.7, val_ratio: float = 0.15
) -> Tuple[slice, slice, slice]:
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return slice(0, train_end), slice(train_end, val_end), slice(val_end, n)


class SlidingWindowDataset(Dataset):
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        seq_len: int,
        pred_len: int = 1,
        window_stride: int = 1,
    ) -> None:
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.window_stride = max(1, int(window_stride))
        max_start = len(x) - seq_len - pred_len + 1
        if max_start <= 0:
            raise ValueError(
                f"데이터 길이가 너무 짧습니다. len={len(x)}, seq_len={seq_len}, pred_len={pred_len}"
            )
        self.starts = np.arange(0, max_start, self.window_stride, dtype=np.int64)
        # CPU 병목 줄이기: float32 텐서로 미리 보관
        self.x = torch.from_numpy(np.asarray(x, dtype=np.float32))
        self.y = torch.from_numpy(np.asarray(y, dtype=np.float32))

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        s = int(self.starts[idx])
        e = s + self.seq_len
        y_idx = e + self.pred_len - 1
        return self.x[s:e], self.y[y_idx]


class PatchEmbedding(nn.Module):
    def __init__(self, num_features: int, patch_len: int, stride: int, d_model: int):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.proj = nn.Linear(num_features * patch_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, F]
        b, l, f = x.shape
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        # [B, N, patch_len, F]
        n = patches.shape[1]
        patches = patches.contiguous().view(b, n, self.patch_len * f)
        return self.proj(patches)


class PatchTSTRegressor(nn.Module):
    def __init__(
        self,
        num_features: int,
        seq_len: int,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        ff_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed = PatchEmbedding(num_features, patch_len, stride, d_model)
        num_patches = math.floor((seq_len - patch_len) / stride) + 1
        if num_patches <= 0:
            raise ValueError("seq_len/patch_len/stride 조합이 유효하지 않습니다.")

        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.embed(x)
        z = z + self.pos_embed[:, : z.size(1), :]
        z = self.encoder(z)
        z = z[:, -1, :]
        out = self.head(z).squeeze(-1)
        return out


@dataclass
class DataBundle:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    feature_scaler: StandardScaler
    target_scaler: StandardScaler
    feature_cols: List[str]
    target_col: str
    # 차량별 구간 (슬라이딩 윈도우가 차량 경계를 넘지 않도록)
    train_segments: List[Tuple[np.ndarray, np.ndarray]]
    val_segments: List[Tuple[np.ndarray, np.ndarray]]
    test_segments: List[Tuple[np.ndarray, np.ndarray]]
    vehicle_ids: List[str]
    data_paths: List[str]


def resolve_data_paths(data_path: str = "", data_dir: str = "") -> List[str]:
    paths: List[str] = []
    if data_dir:
        if not os.path.isdir(data_dir):
            raise ValueError(f"data_dir 이 없습니다: {data_dir}")
        for name in sorted(os.listdir(data_dir)):
            if name.endswith(".csv") or name.endswith(".csv.gz"):
                paths.append(os.path.join(data_dir, name))
    if data_path:
        # 쉼표로 여러 파일 지정 가능
        for p in [x.strip() for x in data_path.split(",") if x.strip()]:
            if p not in paths:
                paths.append(p)
    if not paths:
        raise ValueError("--data_path 또는 --data_dir 로 데이터 파일을 지정해주세요.")
    for p in paths:
        if not os.path.exists(p):
            raise ValueError(f"데이터 파일이 없습니다: {p}")
    return paths


def prepare_one_vehicle_df(
    csv_path: str,
    target_col: str,
    time_col: str,
    feature_cols: List[str] | None,
    sample_stride: int = 1,
) -> Tuple[pd.DataFrame, List[str], str]:
    df = read_vehicle_csv(csv_path)
    if df.empty:
        raise ValueError(f"CSV 파일이 비어 있습니다: {csv_path}")

    print(f"[INFO] loaded {os.path.basename(csv_path)} shape: {df.shape}")

    local_time = time_col or infer_time_column(df)
    if local_time and local_time in df.columns:
        df[local_time] = parse_time_series(df[local_time])
        df = df.sort_values(local_time).reset_index(drop=True)
        print(f"[INFO]   time_col: {local_time}")
    else:
        print(f"[WARN] 시간 컬럼을 찾지 못했습니다: {csv_path}")

    t_col = infer_target_column(df, target_col)
    f_cols = select_feature_columns(df, t_col, feature_cols)
    use_cols = f_cols + [t_col]
    work = df[use_cols].copy()
    for col in use_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    f_cols = filter_usable_features(work, f_cols, max_missing=0.2)
    use_cols = f_cols + [t_col]
    work = work[use_cols].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    sample_stride = max(1, int(sample_stride))
    if sample_stride > 1:
        before = len(work)
        work = work.iloc[::sample_stride].reset_index(drop=True)
        print(f"[INFO]   downsample stride={sample_stride}: {before} -> {len(work)}")

    print(f"[INFO]   usable rows: {len(work)} | soh range: {work[t_col].min():.2f}~{work[t_col].max():.2f}")
    return work, f_cols, t_col


def load_and_prepare_data(
    data_paths: List[str],
    seq_len: int,
    target_col: str = "soh",
    time_col: str = "",
    feature_cols: List[str] | None = None,
    sample_stride: int = 1,
) -> DataBundle:
    prepared = []
    shared_features: List[str] | None = None
    t_col = target_col

    for path in data_paths:
        work, f_cols, t_col = prepare_one_vehicle_df(
            path, target_col, time_col, feature_cols, sample_stride=sample_stride
        )
        if shared_features is None:
            shared_features = f_cols
        else:
            # 여러 차량에서 공통 feature만 사용
            shared_features = [c for c in shared_features if c in f_cols]
        prepared.append((path, work))

    if not shared_features:
        raise ValueError("차량들 간 공통 feature가 없습니다.")

    print(f"[INFO] common features ({len(shared_features)}): {shared_features}")

    train_x_list, train_y_list = [], []
    val_x_list, val_y_list = [], []
    test_x_list, test_y_list = [], []
    train_segments, val_segments, test_segments = [], [], []
    vehicle_ids: List[str] = []

    for path, work in prepared:
        if len(work) < seq_len + 10:
            print(f"[WARN] skip short file: {path}")
            continue
        x = work[shared_features].values
        y = work[[t_col]].values
        tr, va, te = split_by_time(len(work), train_ratio=0.7, val_ratio=0.15)
        x_tr, x_va, x_te = x[tr], x[va], x[te]
        y_tr, y_va, y_te = y[tr], y[va], y[te]
        if min(len(x_tr), len(x_va), len(x_te)) <= seq_len:
            print(f"[WARN] skip short split: {path}")
            continue
        vehicle_ids.append(vehicle_id_from_path(path))
        train_x_list.append(x_tr)
        train_y_list.append(y_tr)
        val_x_list.append(x_va)
        val_y_list.append(y_va)
        test_x_list.append(x_te)
        test_y_list.append(y_te)

    if not train_x_list:
        raise ValueError("학습에 사용할 차량 데이터가 없습니다.")

    x_train = np.concatenate(train_x_list, axis=0)
    y_train = np.concatenate(train_y_list, axis=0)
    x_val = np.concatenate(val_x_list, axis=0)
    y_val = np.concatenate(val_y_list, axis=0)
    x_test = np.concatenate(test_x_list, axis=0)
    y_test = np.concatenate(test_y_list, axis=0)

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    x_scaler.fit(x_train)
    y_scaler.fit(y_train)

    for x_tr, y_tr in zip(train_x_list, train_y_list):
        train_segments.append(
            (
                x_scaler.transform(x_tr),
                y_scaler.transform(y_tr).reshape(-1),
            )
        )
    for x_va, y_va in zip(val_x_list, val_y_list):
        val_segments.append(
            (
                x_scaler.transform(x_va),
                y_scaler.transform(y_va).reshape(-1),
            )
        )
    for x_te, y_te in zip(test_x_list, test_y_list):
        test_segments.append(
            (
                x_scaler.transform(x_te),
                y_scaler.transform(y_te).reshape(-1),
            )
        )

    print(
        f"[INFO] vehicles used: {len(train_segments)} | "
        f"ids: {vehicle_ids} | "
        f"train/val/test rows: {len(x_train)}/{len(x_val)}/{len(x_test)}"
    )

    return DataBundle(
        x_train=x_scaler.transform(x_train),
        y_train=y_scaler.transform(y_train).reshape(-1),
        x_val=x_scaler.transform(x_val),
        y_val=y_scaler.transform(y_val).reshape(-1),
        x_test=x_scaler.transform(x_test),
        y_test=y_scaler.transform(y_test).reshape(-1),
        feature_scaler=x_scaler,
        target_scaler=y_scaler,
        feature_cols=shared_features,
        target_col=t_col,
        train_segments=train_segments,
        val_segments=val_segments,
        test_segments=test_segments,
        vehicle_ids=vehicle_ids,
        data_paths=data_paths,
    )


def build_concat_dataset(
    segments: List[Tuple[np.ndarray, np.ndarray]],
    seq_len: int,
    window_stride: int = 1,
) -> Dataset:
    datasets = [
        SlidingWindowDataset(
            x, y, seq_len=seq_len, pred_len=1, window_stride=window_stride
        )
        for x, y in segments
    ]
    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(datasets)


def train_one_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    losses = []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses))


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> Tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    losses = []
    preds, trues = [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        pred = model(xb)
        loss = criterion(pred, yb)
        losses.append(loss.item())
        preds.append(pred.detach().cpu().numpy())
        trues.append(yb.detach().cpu().numpy())
    return float(np.mean(losses)), np.concatenate(preds), np.concatenate(trues)


def resolve_report_path(out_dir: str) -> str:
    """실험별 폴더가 아니라 outputs/Experiment_Report.md 하나만 유지합니다."""
    abs_out = os.path.abspath(out_dir)
    parent_name = os.path.basename(os.path.dirname(abs_out))
    base_name = os.path.basename(abs_out)
    if parent_name == "outputs":
        return os.path.join(os.path.dirname(abs_out), "Experiment_Report.md")
    if base_name == "outputs":
        return os.path.join(abs_out, "Experiment_Report.md")
    return os.path.join(abs_out, "Experiment_Report.md")


def _existing_report_mae(report_path: str) -> float | None:
    if not os.path.isfile(report_path):
        return None
    try:
        with open(report_path, encoding="utf-8") as f:
            for line in f:
                if line.strip().startswith("| MAE |"):
                    return float(line.split("|")[2].strip())
    except (OSError, ValueError, IndexError):
        return None
    return None


def write_experiment_report(
    out_dir: str,
    metrics: dict,
    train_losses: List[float],
    val_losses: List[float],
    env_info: dict | None = None,
) -> str | None:
    """단일 outputs/Experiment_Report.md에 best(MAE 최저) 결과만 기록합니다."""
    env_info = env_info or {}
    best_epoch = int(metrics.get("best_epoch", 0) or 0)
    best_val = float(metrics.get("best_val_loss", float("nan")))
    mae = float(metrics.get("test_mae", float("nan")))
    mse = float(metrics.get("test_mse", float("nan")))
    rmse = float(metrics.get("test_rmse", float("nan")))
    feature_cols = metrics.get("feature_cols", [])
    vehicle_ids = metrics.get("vehicle_ids", [])
    data_paths = metrics.get("data_paths", [])

    report_path = resolve_report_path(out_dir)
    prev_mae = _existing_report_mae(report_path)
    if prev_mae is not None and mae >= prev_mae:
        print(
            f"[INFO] Experiment_Report.md 유지 "
            f"(기존 MAE={prev_mae:.6f} <= 현재 MAE={mae:.6f})"
        )
        return None

    run_name = os.path.basename(os.path.abspath(out_dir).rstrip(os.sep))
    report_dir = os.path.dirname(os.path.abspath(report_path))
    if os.path.basename(report_dir) == "outputs" and run_name != "outputs":
        img_prefix = f"{run_name}/"
    else:
        img_prefix = ""

    epoch_rows = []
    for i, (tr, va) in enumerate(zip(train_losses, val_losses), start=1):
        mark = " **(best)**" if i == best_epoch else ""
        epoch_rows.append(f"| {i} | {tr:.6f} | {va:.6f}{mark} |")

    vehicle_rows = []
    for vid, vm in (metrics.get("vehicles") or {}).items():
        vehicle_rows.append(
            f"| {vid} | {vm['mae']:.4f} | {vm['rmse']:.4f} | {vm.get('n', '-')} |"
        )

    feature_text = ", ".join(feature_cols) if feature_cols else "-"
    data_text = ", ".join(os.path.basename(p) for p in data_paths) if data_paths else "-"
    period = metrics.get("data_period", "2022-12-15 ~ 2023-08-31")

    report = f"""# PatchTST Experiment Report

> 실험 결과 중 **Test MAE 기준 best** 기록 (run: `{run_name}`, Best Epoch {best_epoch})

## Dataset
- 사용 데이터 : {data_text}
- 데이터 기간 : {period}
- 차량 수 : {metrics.get('num_vehicles', len(vehicle_ids))}
- Feature : {feature_text}
- Target : {metrics.get('target_col', 'soh')}

## Environment
- Python : {env_info.get('python', '-')}
- PyTorch : {env_info.get('pytorch', '-')}
- CUDA : {env_info.get('cuda', '-')}
- GPU : {env_info.get('gpu', metrics.get('device', '-'))}

## Hyperparameters

| Parameter | Value |
|-----------|------|
| seq_len | {metrics.get('seq_len', '')} |
| pred_len | {metrics.get('pred_len', 1)} |
| patch_len | {metrics.get('patch_len', '')} |
| stride | {metrics.get('stride', '')} |
| sample_stride | {metrics.get('sample_stride', '')} |
| window_stride | {metrics.get('window_stride', '')} |
| batch_size | {metrics.get('batch_size', '')} |
| learning_rate | {metrics.get('learning_rate', '')} |
| epochs | {metrics.get('epochs', '')} |
| best_epoch | {best_epoch} |

## Result

> Validation loss 기준 **Best Epoch = {best_epoch}** (val_loss={best_val:.6f}) 체크포인트로 Test 평가.

| Metric | Value |
|---------|------|
| MAE | {mae:.6f} |
| MSE | {mse:.6f} |
| RMSE | {rmse:.6f} |
| Best Val Loss (scaled MSE) | {best_val:.6f} |

### Epoch별 Loss

| Epoch | train_loss | val_loss |
|------:|----------:|---------:|
{chr(10).join(epoch_rows)}

### 차량별 Test 성능

| 차량 ID | MAE | RMSE | n |
|---------|----:|-----:|--:|
{chr(10).join(vehicle_rows) if vehicle_rows else '| - | - | - | - |'}

## Graphs

### Training / Validation Loss
![training_loss]({img_prefix}training_loss.png)

### SOH Prediction (Actual vs Predicted)
![prediction]({img_prefix}prediction.png)

## Observation

### 좋았던 점
- Best validation epoch 체크포인트로 Test를 평가해, 마지막 epoch 과적합 영향을 줄였.
- 차량별 MAE/RMSE를 함께 기록해 성능 편차를 확인할 수 있음.

### 아쉬웠던 점
- 차량 간 성능 편차가 큼 (일부 차량에서 예측이 평탄화되거나 bias가 큼).
- Validation loss가 epoch마다 흔들려 학습이 불안정한 구간이 있음.

### 다음 실험
- Early stopping / learning rate scheduler 적용
- 차량별 bias 보정 또는 성능이 낮은 차량 feature 재검토
- pred_len / seq_len / patch 설정 비교 실험
"""

    os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    return report_path



def collect_env_info(device: torch.device) -> dict:
    import sys

    gpu_name = "-"
    if device.type == "cuda" and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
    return {
        "python": sys.version.split()[0],
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda or "-",
        "gpu": gpu_name,
    }


def save_artifacts(
    out_dir: str,
    train_losses: List[float],
    val_losses: List[float],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    device_nos: List[str],
    time_indices: List[int],
    metrics: dict,
    env_info: dict | None = None,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    abs_err = np.abs(y_true - y_pred)
    pred_df = pd.DataFrame(
        {
            "device_no": device_nos,
            "time_index": time_indices,
            "y_true": y_true,
            "y_pred": y_pred,
            "absolute_error": abs_err,
        }
    )
    pred_df.to_csv(os.path.join(out_dir, "prediction.csv"), index=False)

    plt.figure(figsize=(10, 4))
    plt.plot(train_losses, label="train_loss")
    plt.plot(val_losses, label="val_loss")
    plt.title("Training / Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_loss.png"), dpi=140)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(y_true, label="Actual SOH")
    plt.plot(y_pred, label="Predicted SOH")
    # 차량 경계 표시
    y_top = float(np.nanmax(y_true)) if len(y_true) else 0.0
    offset = 0
    for vid in dict.fromkeys(device_nos):
        n = sum(1 for d in device_nos if d == vid)
        if offset > 0:
            plt.axvline(offset - 0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
        mid = offset + max(n, 1) / 2
        plt.text(mid, y_top, vid[-3:], ha="center", va="bottom", fontsize=8)
        offset += n
    plt.title("SOH Prediction")
    plt.xlabel("Time Index")
    plt.ylabel("SOH")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "prediction.png"), dpi=140)
    plt.close()

    report_path = write_experiment_report(
        out_dir=out_dir,
        metrics=metrics,
        train_losses=train_losses,
        val_losses=val_losses,
        env_info=env_info,
    )
    if report_path:
        print(f"[DONE] experiment report: {report_path}")


def build_checkpoint(
    model: nn.Module,
    feature_scaler: StandardScaler,
    target_scaler: StandardScaler,
    feature_cols: List[str],
    target_col: str,
    args,
) -> dict:
    return {
        "model_state_dict": model.state_dict(),
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
        "x_scaler": feature_scaler,
        "y_scaler": target_scaler,
        "feature_cols": feature_cols,
        "target_col": target_col,
        "model_config": {
            "seq_len": args.seq_len,
            "patch_len": args.patch_len,
            "stride": args.stride,
            "d_model": args.d_model,
            "nhead": args.nhead,
            "num_layers": args.num_layers,
            "ff_dim": args.ff_dim,
            "dropout": args.dropout,
            "num_features": len(feature_cols),
        },
    }


def load_model_state(model: nn.Module, checkpoint_path: str, device: torch.device) -> None:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)


def evaluate_per_vehicle(
    model,
    segments: List[Tuple[np.ndarray, np.ndarray]],
    vehicle_ids: List[str],
    seq_len: int,
    batch_size: int,
    criterion,
    device,
    target_scaler: StandardScaler,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[int], dict, float]:
    """차량별 test 예측/지표를 계산하고 전체 결과를 합칩니다."""
    all_true: List[np.ndarray] = []
    all_pred: List[np.ndarray] = []
    device_nos: List[str] = []
    time_indices: List[int] = []
    vehicle_metrics: dict = {}
    scaled_losses: List[float] = []

    for vid, (x_seg, y_seg) in zip(vehicle_ids, segments):
        ds = SlidingWindowDataset(
            x_seg, y_seg, seq_len=seq_len, pred_len=1, window_stride=1
        )
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=2,
            pin_memory=device.type == "cuda",
        )
        loss, pred_sc, true_sc = evaluate(model, loader, criterion, device)
        scaled_losses.append(loss)

        pred = target_scaler.inverse_transform(pred_sc.reshape(-1, 1)).reshape(-1)
        true = target_scaler.inverse_transform(true_sc.reshape(-1, 1)).reshape(-1)
        mae = float(mean_absolute_error(true, pred))
        rmse = float(math.sqrt(mean_squared_error(true, pred)))
        vehicle_metrics[vid] = {
            "mae": mae,
            "rmse": rmse,
            "n": int(len(true)),
            "test_loss_scaled_mse": float(loss),
        }

        all_true.append(true)
        all_pred.append(pred)
        device_nos.extend([vid] * len(true))
        time_indices.extend(list(range(len(true))))

        print(f"[TEST] {vid}: MAE={mae:.6f} RMSE={rmse:.6f} n={len(true)}")

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    overall_loss = float(np.mean(scaled_losses)) if scaled_losses else float("nan")
    return y_true, y_pred, device_nos, time_indices, vehicle_metrics, overall_loss


def parse_feature_cols(raw: str) -> List[str] | None:
    if not raw.strip():
        return None
    return [c.strip() for c in raw.split(",") if c.strip()]


def parse_args():
    parser = argparse.ArgumentParser(
        description="PatchTST 기반 SOH 예측 학습 (예측 대상: soh, feature/하이퍼파라미터는 자유 설정)"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="",
        help="단일 파일 또는 쉼표로 구분한 여러 파일 경로",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="data 폴더의 모든 .csv/.csv.gz 를 학습에 사용",
    )
    parser.add_argument("--target_col", type=str, default="soh", help="예측 대상 컬럼 (기본: soh)")
    parser.add_argument(
        "--feature_cols",
        type=str,
        default="",
        help="입력 feature 목록 (쉼표 구분). 비우면 기본 배터리 feature 사용",
    )
    parser.add_argument("--time_col", type=str, default="")
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--patch_len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument(
        "--sample_stride",
        type=int,
        default=10,
        help="원본 시계열 다운샘플 간격 (10이면 10행마다 1행 사용). 마감용 기본값=10",
    )
    parser.add_argument(
        "--window_stride",
        type=int,
        default=8,
        help="슬라이딩 윈도우 시작점 간격. 클수록 샘플 수↓ 속도↑",
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--ff_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_paths = resolve_data_paths(data_path=args.data_path, data_dir=args.data_dir)
    print(f"[INFO] device: {device}")
    print(f"[INFO] data files ({len(data_paths)}):")
    for p in data_paths:
        print(f"  - {p}")

    feature_cols = parse_feature_cols(args.feature_cols)

    data = load_and_prepare_data(
        data_paths=data_paths,
        seq_len=args.seq_len,
        target_col=args.target_col,
        time_col=args.time_col,
        feature_cols=feature_cols,
        sample_stride=args.sample_stride,
    )

    print(f"[INFO] target_col: {data.target_col}")
    print(f"[INFO] feature_count: {len(data.feature_cols)}")
    print(f"[INFO] feature_cols: {data.feature_cols}")
    print(
        f"[INFO] speed opts: sample_stride={args.sample_stride} "
        f"window_stride={args.window_stride} batch_size={args.batch_size} epochs={args.epochs}"
    )

    train_ds = build_concat_dataset(
        data.train_segments, seq_len=args.seq_len, window_stride=args.window_stride
    )
    val_ds = build_concat_dataset(
        data.val_segments, seq_len=args.seq_len, window_stride=args.window_stride
    )
    print(f"[INFO] train/val windows: {len(train_ds)}/{len(val_ds)}")

    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=pin,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=max(1, args.num_workers // 2),
        pin_memory=pin,
        persistent_workers=args.num_workers > 0,
    )

    model = PatchTSTRegressor(
        num_features=len(data.feature_cols),
        seq_len=args.seq_len,
        patch_len=args.patch_len,
        stride=args.stride,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = float("inf")
    best_epoch = 0
    train_losses: List[float] = []
    val_losses: List[float] = []
    best_path = os.path.join(args.checkpoint_dir, "best_model.pt")
    env_info = collect_env_info(device)

    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        va_loss, _, _ = evaluate(model, val_loader, criterion, device)
        train_losses.append(tr_loss)
        val_losses.append(va_loss)

        if va_loss < best_val:
            best_val = va_loss
            best_epoch = epoch
            torch.save(
                build_checkpoint(
                    model=model,
                    feature_scaler=data.feature_scaler,
                    target_scaler=data.target_scaler,
                    feature_cols=data.feature_cols,
                    target_col=data.target_col,
                    args=args,
                ),
                best_path,
            )

        print(f"[Epoch {epoch:03d}/{args.epochs}] train_loss={tr_loss:.6f} val_loss={va_loss:.6f}")

    load_model_state(model, best_path, device)
    (
        true,
        pred,
        device_nos,
        time_indices,
        vehicle_metrics,
        test_loss,
    ) = evaluate_per_vehicle(
        model=model,
        segments=data.test_segments,
        vehicle_ids=data.vehicle_ids,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        criterion=criterion,
        device=device,
        target_scaler=data.target_scaler,
    )

    mae = float(mean_absolute_error(true, pred))
    mse = float(mean_squared_error(true, pred))
    rmse = float(math.sqrt(mse))

    metrics = {
        "device": str(device),
        "data_paths": data.data_paths,
        "num_vehicles": len(data.train_segments),
        "vehicle_ids": data.vehicle_ids,
        "target_col": data.target_col,
        "feature_cols": data.feature_cols,
        "seq_len": args.seq_len,
        "pred_len": 1,
        "patch_len": args.patch_len,
        "stride": args.stride,
        "sample_stride": args.sample_stride,
        "window_stride": args.window_stride,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "num_workers": args.num_workers,
        "learning_rate": args.lr,
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "test_loss_scaled_mse": test_loss,
        "test_mae": mae,
        "test_mse": mse,
        "test_rmse": rmse,
        "overall_mae": mae,
        "overall_rmse": rmse,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "vehicles": vehicle_metrics,
        "python": env_info.get("python"),
        "pytorch": env_info.get("pytorch"),
        "cuda": env_info.get("cuda"),
        "gpu": env_info.get("gpu"),
    }

    save_artifacts(
        out_dir=args.output_dir,
        train_losses=train_losses,
        val_losses=val_losses,
        y_true=true,
        y_pred=pred,
        device_nos=device_nos,
        time_indices=time_indices,
        metrics=metrics,
        env_info=env_info,
    )

    print(f"[DONE] Best Epoch: {best_epoch} (val_loss={best_val:.6f})")
    print(f"[DONE] Test MAE:  {mae:.6f}")
    print(f"[DONE] Test MSE:  {mse:.6f}")
    print(f"[DONE] Test RMSE: {rmse:.6f}")
    for vid, vm in vehicle_metrics.items():
        print(f"[DONE]   {vid}: MAE={vm['mae']:.6f} RMSE={vm['rmse']:.6f}")
    print(f"[DONE] outputs saved to: {args.output_dir}")
    print(f"[DONE] best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
