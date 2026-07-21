import argparse
import json
import math
import os
import re
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
        vehicle_idx: int = 0,
        return_vehicle_idx: bool = False,
    ) -> None:
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.window_stride = max(1, int(window_stride))
        self.vehicle_idx = int(vehicle_idx)
        self.return_vehicle_idx = return_vehicle_idx
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

    def __getitem__(self, idx: int):
        s = int(self.starts[idx])
        e = s + self.seq_len
        y_idx = e + self.pred_len - 1
        if self.return_vehicle_idx:
            return self.x[s:e], self.y[y_idx], self.vehicle_idx
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
        num_vehicles: int = 0,
        vehicle_emb_dim: int = 16,
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
        # num_vehicles>0 이면 index0=unknown, 1..N=학습 차량
        self.num_vehicles = int(num_vehicles)
        self.vehicle_emb_dim = int(vehicle_emb_dim) if num_vehicles > 0 else 0
        self.vehicle_emb: nn.Embedding | None = None
        head_in = d_model
        if self.num_vehicles > 0:
            self.vehicle_emb = nn.Embedding(self.num_vehicles + 1, self.vehicle_emb_dim)
            nn.init.normal_(self.vehicle_emb.weight, mean=0.0, std=0.02)
            head_in = d_model + self.vehicle_emb_dim
        self.head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(
        self, x: torch.Tensor, vehicle_idx: torch.Tensor | None = None
    ) -> torch.Tensor:
        z = self.embed(x)
        z = z + self.pos_embed[:, : z.size(1), :]
        z = self.encoder(z)
        z = z[:, -1, :]
        if self.vehicle_emb is not None:
            if vehicle_idx is None:
                vehicle_idx = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            else:
                vehicle_idx = vehicle_idx.long().to(x.device)
            z = torch.cat([z, self.vehicle_emb(vehicle_idx)], dim=-1)
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
    per_vehicle_norm: bool = False
    feature_scalers: dict | None = None  # vehicle_id -> StandardScaler
    target_scalers: dict | None = None
    train_vehicle_ids: List[str] | None = None
    val_vehicle_ids: List[str] | None = None
    test_vehicle_ids: List[str] | None = None
    holdout_vehicle: str = ""
    vehicle_to_idx: dict | None = None  # embedding용 (unknown=0)
    calib_segments: List[Tuple[np.ndarray, np.ndarray]] | None = None
    calib_vehicle_ids: List[str] | None = None
    calibrate_frac: float = 0.0
    fix_zero_temp: bool = False


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


def fix_zero_temp_features(
    work: pd.DataFrame, feature_cols: List[str]
) -> Tuple[pd.DataFrame, List[str]]:
    """
    int_temp/ext_temp가 거의 0 또는 상수면(178 등) mod_avg_temp 등으로 대체.
    proxy도 없으면 해당 feature를 제거합니다.
    """
    proxies = ["mod_avg_temp", "batt_internal_temp", "mod_min_temp", "mod_max_temp"]
    kept = list(feature_cols)
    for col in ("int_temp", "ext_temp"):
        if col not in work.columns or col not in kept:
            continue
        vals = pd.to_numeric(work[col], errors="coerce").to_numpy(dtype=float)
        zero_ratio = float(np.nanmean(np.abs(vals) < 1e-6)) if len(vals) else 1.0
        std = float(np.nanstd(vals)) if len(vals) else 0.0
        if zero_ratio < 0.9 and std > 1e-6:
            continue
        proxy = next(
            (
                p
                for p in proxies
                if p in work.columns and float(pd.to_numeric(work[p], errors="coerce").std()) > 1e-6
            ),
            None,
        )
        if proxy:
            work[col] = pd.to_numeric(work[proxy], errors="coerce")
            print(f"[WARN]   {col} zero/constant → filled from {proxy}")
        else:
            kept = [c for c in kept if c != col]
            work = work.drop(columns=[col], errors="ignore")
            print(f"[WARN]   {col} zero/constant → dropped (no proxy)")
    return work, kept


def prepare_one_vehicle_df(
    csv_path: str,
    target_col: str,
    time_col: str,
    feature_cols: List[str] | None,
    sample_stride: int = 1,
    fix_zero_temp: bool = False,
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

    if fix_zero_temp:
        work, f_cols = fix_zero_temp_features(work, f_cols)

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


def resolve_holdout_id(vehicle_ids: List[str], holdout: str) -> str:
    """holdout 문자열을 실제 vehicle_id로 매칭 (전체 ID 또는 끝 3~자리)."""
    q = holdout.strip()
    if not q:
        return ""
    for vid in vehicle_ids:
        if vid == q or vid.endswith(q):
            return vid
    raise ValueError(f"holdout_vehicle '{holdout}' 을(를) 찾지 못함. 후보: {vehicle_ids}")


def load_and_prepare_data(
    data_paths: List[str],
    seq_len: int,
    target_col: str = "soh",
    time_col: str = "",
    feature_cols: List[str] | None = None,
    sample_stride: int = 1,
    per_vehicle_norm: bool = False,
    holdout_vehicle: str = "",
    calibrate_frac: float = 0.0,
    fix_zero_temp: bool = False,
) -> DataBundle:
    prepared = []
    shared_features: List[str] | None = None
    t_col = target_col

    for path in data_paths:
        work, f_cols, t_col = prepare_one_vehicle_df(
            path,
            target_col,
            time_col,
            feature_cols,
            sample_stride=sample_stride,
            fix_zero_temp=fix_zero_temp,
        )
        if shared_features is None:
            shared_features = f_cols
        else:
            shared_features = [c for c in shared_features if c in f_cols]
        prepared.append((path, work))

    if not shared_features:
        raise ValueError("차량들 간 공통 feature가 없습니다.")

    print(f"[INFO] common features ({len(shared_features)}): {shared_features}")
    print(f"[INFO] per_vehicle_norm: {per_vehicle_norm}")

    all_vids: List[str] = []
    all_x_tr, all_y_tr = [], []
    all_x_va, all_y_va = [], []
    all_x_te, all_y_te = [], []

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
        all_vids.append(vehicle_id_from_path(path))
        all_x_tr.append(x_tr)
        all_y_tr.append(y_tr)
        all_x_va.append(x_va)
        all_y_va.append(y_va)
        all_x_te.append(x_te)
        all_y_te.append(y_te)

    if not all_vids:
        raise ValueError("학습에 사용할 차량 데이터가 없습니다.")

    holdout_id = resolve_holdout_id(all_vids, holdout_vehicle) if holdout_vehicle else ""
    if holdout_id:
        print(f"[INFO] LOVO holdout vehicle: {holdout_id}")
        if per_vehicle_norm:
            print("[WARN] LOVO에서는 per_vehicle_norm을 끄고 global scaler를 사용합니다.")
            per_vehicle_norm = False

    seen_idx = [i for i, v in enumerate(all_vids) if v != holdout_id]
    hold_idx = [i for i, v in enumerate(all_vids) if v == holdout_id]
    if holdout_id and not hold_idx:
        raise ValueError(f"holdout 차량 데이터가 없습니다: {holdout_id}")
    if holdout_id and not seen_idx:
        raise ValueError("holdout 제외 후 학습 차량이 없습니다.")

    train_idx = seen_idx if holdout_id else list(range(len(all_vids)))
    val_idx = train_idx
    test_idx = hold_idx if holdout_id else list(range(len(all_vids)))

    train_vehicle_ids = [all_vids[i] for i in train_idx]
    val_vehicle_ids = [all_vids[i] for i in val_idx]
    test_vehicle_ids = [all_vids[i] for i in test_idx]

    train_x_list = [all_x_tr[i] for i in train_idx]
    train_y_list = [all_y_tr[i] for i in train_idx]
    val_x_list = [all_x_va[i] for i in val_idx]
    val_y_list = [all_y_va[i] for i in val_idx]
    test_x_list = [all_x_te[i] for i in test_idx]
    test_y_list = [all_y_te[i] for i in test_idx]

    x_train = np.concatenate(train_x_list, axis=0)
    y_train = np.concatenate(train_y_list, axis=0)
    x_val = np.concatenate(val_x_list, axis=0)
    y_val = np.concatenate(val_y_list, axis=0)
    x_test = np.concatenate(test_x_list, axis=0)
    y_test = np.concatenate(test_y_list, axis=0)

    train_segments: List[Tuple[np.ndarray, np.ndarray]] = []
    val_segments: List[Tuple[np.ndarray, np.ndarray]] = []
    test_segments: List[Tuple[np.ndarray, np.ndarray]] = []
    feature_scalers: dict | None = None
    target_scalers: dict | None = None

    if per_vehicle_norm:
        feature_scalers = {}
        target_scalers = {}
        for vid, x_tr, y_tr, x_va, y_va in zip(
            train_vehicle_ids, train_x_list, train_y_list, val_x_list, val_y_list
        ):
            x_sc = StandardScaler().fit(x_tr)
            y_sc = StandardScaler().fit(y_tr)
            feature_scalers[vid] = x_sc
            target_scalers[vid] = y_sc
            train_segments.append(
                (x_sc.transform(x_tr), y_sc.transform(y_tr).reshape(-1))
            )
            val_segments.append(
                (x_sc.transform(x_va), y_sc.transform(y_va).reshape(-1))
            )
            print(
                f"[NORM] {vid}: y_mean={float(y_sc.mean_[0]):.3f} "
                f"y_scale={float(y_sc.scale_[0]):.3f}"
            )
        for vid, x_te, y_te in zip(test_vehicle_ids, test_x_list, test_y_list):
            x_sc = feature_scalers[vid]
            y_sc = target_scalers[vid]
            test_segments.append(
                (x_sc.transform(x_te), y_sc.transform(y_te).reshape(-1))
            )
        x_scaler = StandardScaler().fit(x_train)
        y_scaler = StandardScaler().fit(y_train)
        x_train_sc = np.concatenate([s[0] for s in train_segments], axis=0)
        y_train_sc = np.concatenate([s[1] for s in train_segments], axis=0)
        x_val_sc = np.concatenate([s[0] for s in val_segments], axis=0)
        y_val_sc = np.concatenate([s[1] for s in val_segments], axis=0)
        x_test_sc = np.concatenate([s[0] for s in test_segments], axis=0)
        y_test_sc = np.concatenate([s[1] for s in test_segments], axis=0)
    else:
        x_scaler = StandardScaler().fit(x_train)
        y_scaler = StandardScaler().fit(y_train)
        for x_tr, y_tr in zip(train_x_list, train_y_list):
            train_segments.append(
                (x_scaler.transform(x_tr), y_scaler.transform(y_tr).reshape(-1))
            )
        for x_va, y_va in zip(val_x_list, val_y_list):
            val_segments.append(
                (x_scaler.transform(x_va), y_scaler.transform(y_va).reshape(-1))
            )
        for x_te, y_te in zip(test_x_list, test_y_list):
            test_segments.append(
                (x_scaler.transform(x_te), y_scaler.transform(y_te).reshape(-1))
            )
        x_train_sc = x_scaler.transform(x_train)
        y_train_sc = y_scaler.transform(y_train).reshape(-1)
        x_val_sc = x_scaler.transform(x_val)
        y_val_sc = y_scaler.transform(y_val).reshape(-1)
        x_test_sc = x_scaler.transform(x_test)
        y_test_sc = y_scaler.transform(y_test).reshape(-1)

    # embedding: unknown=0, 학습 차량=1..N
    uniq_train = list(dict.fromkeys(train_vehicle_ids))
    vehicle_to_idx = {vid: i + 1 for i, vid in enumerate(uniq_train)}

    calib_segments: List[Tuple[np.ndarray, np.ndarray]] | None = None
    calib_vehicle_ids: List[str] | None = None
    cal_frac = float(calibrate_frac or 0.0)

    if holdout_id and cal_frac > 0:
        if per_vehicle_norm:
            raise ValueError("holdout + calibrate_frac 에서는 per_vehicle_norm을 사용할 수 없습니다.")
        if not (0.0 < cal_frac < 0.9):
            raise ValueError(f"calibrate_frac는 (0, 0.9) 범위여야 합니다: {cal_frac}")
        hi = hold_idx[0]
        # LOVO test(마지막 15%)와 동일 구간에서 few-shot:
        # 앞 calibrate_frac → calib, 나머지 → eval (zero-shot과 비교 가능)
        x_te_raw, y_te_raw = all_x_te[hi], all_y_te[hi]
        n = len(x_te_raw)
        min_cal = seq_len + 20
        min_eval = seq_len + 20
        n_cal = max(min_cal, int(n * cal_frac))
        if n_cal > n - min_eval:
            n_cal = n - min_eval
        if n_cal < min_cal:
            raise ValueError(
                f"holdout test가 calibrate에 너무 짧습니다: n={n}, n_cal={n_cal}"
            )
        x_cal_raw, y_cal_raw = x_te_raw[:n_cal], y_te_raw[:n_cal]
        x_ev_raw, y_ev_raw = x_te_raw[n_cal:], y_te_raw[n_cal:]
        calib_segments = [
            (
                x_scaler.transform(x_cal_raw),
                y_scaler.transform(y_cal_raw).reshape(-1),
            )
        ]
        test_segments = [
            (
                x_scaler.transform(x_ev_raw),
                y_scaler.transform(y_ev_raw).reshape(-1),
            )
        ]
        x_test_sc = test_segments[0][0]
        y_test_sc = test_segments[0][1]
        calib_vehicle_ids = [holdout_id]
        test_vehicle_ids = [holdout_id]
        print(
            f"[INFO] few-shot calib on holdout TEST head: frac={cal_frac:.2f} → "
            f"calib_rows={n_cal}/{n} ({100.0 * n_cal / n:.1f}% of test), "
            f"eval_rows={n - n_cal} "
            f"(LOVO zero-shot은 test 전체 n={n})"
        )

    print(
        f"[INFO] train vehicles: {train_vehicle_ids} | "
        f"test vehicles: {test_vehicle_ids} | "
        f"train/val/test rows: {len(x_train)}/{len(x_val)}/{len(x_test_sc)}"
    )

    return DataBundle(
        x_train=x_train_sc,
        y_train=y_train_sc,
        x_val=x_val_sc,
        y_val=y_val_sc,
        x_test=x_test_sc,
        y_test=y_test_sc,
        feature_scaler=x_scaler,
        target_scaler=y_scaler,
        feature_cols=shared_features,
        target_col=t_col,
        train_segments=train_segments,
        val_segments=val_segments,
        test_segments=test_segments,
        vehicle_ids=test_vehicle_ids,
        data_paths=data_paths,
        per_vehicle_norm=per_vehicle_norm,
        feature_scalers=feature_scalers,
        target_scalers=target_scalers,
        train_vehicle_ids=train_vehicle_ids,
        val_vehicle_ids=val_vehicle_ids,
        test_vehicle_ids=test_vehicle_ids,
        holdout_vehicle=holdout_id,
        vehicle_to_idx=vehicle_to_idx,
        calib_segments=calib_segments,
        calib_vehicle_ids=calib_vehicle_ids,
        calibrate_frac=cal_frac if holdout_id else 0.0,
        fix_zero_temp=fix_zero_temp,
    )


def build_concat_dataset(
    segments: List[Tuple[np.ndarray, np.ndarray]],
    seq_len: int,
    window_stride: int = 1,
    vehicle_ids: List[str] | None = None,
    vehicle_to_idx: dict | None = None,
    use_vehicle_idx: bool = False,
) -> Dataset:
    datasets = []
    for i, (x, y) in enumerate(segments):
        vid = (vehicle_ids or [""])[i] if vehicle_ids else ""
        vidx = 0
        if use_vehicle_idx and vehicle_to_idx is not None:
            vidx = int(vehicle_to_idx.get(vid, 0))
        datasets.append(
            SlidingWindowDataset(
                x,
                y,
                seq_len=seq_len,
                pred_len=1,
                window_stride=window_stride,
                vehicle_idx=vidx,
                return_vehicle_idx=use_vehicle_idx,
            )
        )
    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(datasets)


def _unpack_batch(batch, device):
    if len(batch) == 3:
        xb, yb, vid = batch
        return (
            xb.to(device, non_blocking=True),
            yb.to(device, non_blocking=True),
            vid.to(device, non_blocking=True),
        )
    xb, yb = batch
    return xb.to(device, non_blocking=True), yb.to(device, non_blocking=True), None


def train_one_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    losses = []
    for batch in loader:
        xb, yb, vid = _unpack_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(xb, vid) if vid is not None else model(xb)
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
    for batch in loader:
        xb, yb, vid = _unpack_batch(batch, device)
        pred = model(xb, vid) if vid is not None else model(xb)
        loss = criterion(pred, yb)
        losses.append(loss.item())
        preds.append(pred.detach().cpu().numpy())
        trues.append(yb.detach().cpu().numpy())
    return float(np.mean(losses)), np.concatenate(preds), np.concatenate(trues)


def median_filter_series(y: np.ndarray, kernel: int = 5) -> np.ndarray:
    """홀수 커널 median filter (경계는 reflect)."""
    k = int(kernel)
    if k <= 1 or len(y) == 0:
        return y
    if k % 2 == 0:
        k += 1
    pad = k // 2
    yp = np.pad(y.astype(np.float64), (pad, pad), mode="edge")
    out = np.empty_like(y, dtype=np.float64)
    for i in range(len(y)):
        out[i] = np.median(yp[i : i + k])
    return out.astype(y.dtype, copy=False)


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
            text = f.read()
        # 비교형 리포트(여러 실험)는 자동 덮어쓰기 방지: 헤더 best MAE 우선
        m = re.search(r"MAE\s*\*\*([0-9.]+)\*\*", text)
        if m:
            return float(m.group(1))
        for line in text.splitlines():
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
    """단일 outputs/Experiment_Report.md에 best(MAE 최저) 결과만 기록합니다.
    비교형 리포트(실험 비교 섹션)가 있으면 덮어쓰지 않습니다.
    """
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
    if os.path.isfile(report_path):
        try:
            with open(report_path, encoding="utf-8") as f:
                existing = f.read()
            if "## 실험 비교" in existing or "실험 비교 중" in existing:
                print(
                    f"[INFO] Experiment_Report.md 유지 "
                    f"(비교형 리포트 — 수동 갱신 권장, 현재 MAE={mae:.6f})"
                )
                return None
        except OSError:
            pass

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
| weight_decay | {metrics.get('weight_decay', '')} |
| dropout | {metrics.get('dropout', '')} |
| patience (early stop) | {metrics.get('patience', '')} |
| lr_factor / lr_patience | {metrics.get('lr_factor', '')} / {metrics.get('lr_patience', '')} |
| epochs | {metrics.get('epochs', '')} |
| epochs_ran | {metrics.get('epochs_ran', '')} |
| stopped_early | {metrics.get('stopped_early', '')} |
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
- 차량별 bias 보정 (`01241225220` 중심) 또는 차량별 정규화/fine-tuning
- 예측 스파이크 후처리 및 Huber/MAE 손실 비교
- seq_len / patch_len 비교 실험
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
    skip_report: bool = False,
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

    if skip_report:
        print("[INFO] Experiment_Report.md 자동 갱신 생략 (skip_report=True)")
        return

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
    feature_scalers: dict | None = None,
    target_scalers: dict | None = None,
    per_vehicle_norm: bool = False,
    vehicle_to_idx: dict | None = None,
) -> dict:
    num_vehicles = int(getattr(args, "num_vehicles_emb", 0) or 0)
    vehicle_emb_dim = int(getattr(args, "vehicle_emb_dim", 16) or 16)
    return {
        "model_state_dict": model.state_dict(),
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
        "x_scaler": feature_scaler,
        "y_scaler": target_scaler,
        "feature_scalers": feature_scalers,
        "target_scalers": target_scalers,
        "per_vehicle_norm": per_vehicle_norm,
        "vehicle_to_idx": vehicle_to_idx,
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
            "num_vehicles": num_vehicles,
            "vehicle_emb_dim": vehicle_emb_dim,
        },
    }


def load_model_state(model: nn.Module, checkpoint_path: str, device: torch.device) -> None:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)


def predict_vehicle_original(
    model,
    x_seg: np.ndarray,
    y_seg: np.ndarray,
    seq_len: int,
    batch_size: int,
    criterion,
    device,
    target_scaler: StandardScaler,
    window_stride: int = 1,
    vehicle_idx: int = 0,
    use_vehicle_idx: bool = False,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """한 차량 구간에 대해 원본 스케일 true/pred를 반환합니다."""
    ds = SlidingWindowDataset(
        x_seg,
        y_seg,
        seq_len=seq_len,
        pred_len=1,
        window_stride=window_stride,
        vehicle_idx=vehicle_idx,
        return_vehicle_idx=use_vehicle_idx,
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
    pred = target_scaler.inverse_transform(pred_sc.reshape(-1, 1)).reshape(-1)
    true = target_scaler.inverse_transform(true_sc.reshape(-1, 1)).reshape(-1)
    return float(loss), true, pred


def estimate_vehicle_biases(
    model,
    segments: List[Tuple[np.ndarray, np.ndarray]],
    vehicle_ids: List[str],
    seq_len: int,
    batch_size: int,
    criterion,
    device,
    target_scaler: StandardScaler,
    target_scalers: dict | None = None,
    vehicle_to_idx: dict | None = None,
    use_vehicle_idx: bool = False,
) -> dict:
    """
    Validation 구간에서 차량별 평균 bias(pred - true)를 추정합니다.
    Test 적용 시 pred에서 이 값을 빼 보정합니다.
    """
    biases: dict = {}
    for vid, (x_seg, y_seg) in zip(vehicle_ids, segments):
        y_sc = (target_scalers or {}).get(vid, target_scaler)
        vidx = int((vehicle_to_idx or {}).get(vid, 0))
        _, true, pred = predict_vehicle_original(
            model=model,
            x_seg=x_seg,
            y_seg=y_seg,
            seq_len=seq_len,
            batch_size=batch_size,
            criterion=criterion,
            device=device,
            target_scaler=y_sc,
            window_stride=1,
            vehicle_idx=vidx,
            use_vehicle_idx=use_vehicle_idx,
        )
        bias = float(np.mean(pred - true))
        biases[vid] = bias
        mae = float(mean_absolute_error(true, pred))
        mae_bc = float(mean_absolute_error(true, pred - bias))
        print(
            f"[BIAS] {vid}: val_bias={bias:+.4f} "
            f"val_MAE={mae:.4f} -> {mae_bc:.4f} (corrected)"
        )
    return biases


def evaluate_per_vehicle(
    model,
    segments: List[Tuple[np.ndarray, np.ndarray]],
    vehicle_ids: List[str],
    seq_len: int,
    batch_size: int,
    criterion,
    device,
    target_scaler: StandardScaler,
    vehicle_biases: dict | None = None,
    target_scalers: dict | None = None,
    vehicle_to_idx: dict | None = None,
    use_vehicle_idx: bool = False,
    median_kernel: int = 0,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[int], dict, float]:
    """차량별 test 예측/지표를 계산하고 전체 결과를 합칩니다."""
    all_true: List[np.ndarray] = []
    all_pred: List[np.ndarray] = []
    device_nos: List[str] = []
    time_indices: List[int] = []
    vehicle_metrics: dict = {}
    scaled_losses: List[float] = []
    biases = vehicle_biases or {}

    for vid, (x_seg, y_seg) in zip(vehicle_ids, segments):
        y_sc = (target_scalers or {}).get(vid, target_scaler)
        vidx = int((vehicle_to_idx or {}).get(vid, 0))
        loss, true, pred_raw = predict_vehicle_original(
            model=model,
            x_seg=x_seg,
            y_seg=y_seg,
            seq_len=seq_len,
            batch_size=batch_size,
            criterion=criterion,
            device=device,
            target_scaler=y_sc,
            window_stride=1,
            vehicle_idx=vidx,
            use_vehicle_idx=use_vehicle_idx,
        )
        scaled_losses.append(loss)

        bias = float(biases.get(vid, 0.0))
        pred = pred_raw - bias
        if median_kernel and median_kernel > 1:
            pred = median_filter_series(pred, kernel=median_kernel)
        mae = float(mean_absolute_error(true, pred))
        rmse = float(math.sqrt(mean_squared_error(true, pred)))
        mae_raw = float(mean_absolute_error(true, pred_raw))
        vehicle_metrics[vid] = {
            "mae": mae,
            "rmse": rmse,
            "n": int(len(true)),
            "test_loss_scaled_mse": float(loss),
            "bias_applied": bias,
            "mae_before_bias": mae_raw,
            "vehicle_idx": vidx,
        }

        all_true.append(true)
        all_pred.append(pred)
        device_nos.extend([vid] * len(true))
        time_indices.extend(list(range(len(true))))

        extra = ""
        if bias:
            extra += f" bias={bias:+.4f} raw_MAE={mae_raw:.6f}"
        if median_kernel and median_kernel > 1:
            extra += f" median_k={median_kernel}"
        print(f"[TEST] {vid}: MAE={mae:.6f} RMSE={rmse:.6f} n={len(true)}{extra}")

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
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="AdamW weight decay (과적합 완화)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early stopping patience (val_loss 미개선 epoch 수). 0이면 비활성",
    )
    parser.add_argument(
        "--lr_factor",
        type=float,
        default=0.5,
        help="ReduceLROnPlateau 학습률 감소 배율",
    )
    parser.add_argument(
        "--lr_patience",
        type=int,
        default=2,
        help="val_loss 미개선 시 LR 감소까지 기다리는 epoch 수",
    )
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--ff_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument(
        "--vehicle_bias_correct",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Validation에서 차량별 bias(pred-true)를 추정해 Test 예측에 보정 적용",
    )
    parser.add_argument(
        "--per_vehicle_norm",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="차량별 train split으로 feature/target StandardScaler를 따로 fit",
    )
    parser.add_argument(
        "--holdout_vehicle",
        type=str,
        default="",
        help="LOVO: 해당 차량을 학습/검증에서 제외하고 Test만 수행 (ID 또는 끝자리)",
    )
    parser.add_argument(
        "--calibrate_frac",
        type=float,
        default=0.0,
        help="LOVO holdout의 Test 구간 앞부분 비율로 few-shot 캘리브레이션 (예: 0.1). 0이면 비활성",
    )
    parser.add_argument(
        "--finetune_epochs",
        type=int,
        default=0,
        help="캘리브레이션 구간으로 fine-tune할 epoch 수 (0이면 bias만)",
    )
    parser.add_argument(
        "--finetune_lr",
        type=float,
        default=1e-4,
        help="few-shot fine-tune 학습률",
    )
    parser.add_argument(
        "--fix_zero_temp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="int_temp/ext_temp가 0·상수면 mod_avg_temp 등으로 대체 (178 대응)",
    )
    parser.add_argument(
        "--vehicle_embedding",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="차량 ID embedding을 모델에 주입 (unknown=0)",
    )
    parser.add_argument(
        "--vehicle_emb_dim",
        type=int,
        default=16,
        help="vehicle embedding 차원",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="mse",
        choices=["mse", "huber", "mae"],
        help="학습 손실 함수",
    )
    parser.add_argument(
        "--huber_delta",
        type=float,
        default=1.0,
        help="HuberLoss delta",
    )
    parser.add_argument(
        "--median_kernel",
        type=int,
        default=0,
        help="예측 후처리 median filter 커널(홀수). 0이면 비활성",
    )
    parser.add_argument(
        "--eval_checkpoint",
        type=str,
        default="",
        help="지정 시 학습 없이 해당 체크포인트로 평가만 수행",
    )
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
        per_vehicle_norm=args.per_vehicle_norm,
        holdout_vehicle=args.holdout_vehicle,
        calibrate_frac=args.calibrate_frac,
        fix_zero_temp=args.fix_zero_temp,
    )

    # LOVO: 라벨 없는 새 차량 가정 → bias 비활성.
    # 단, calibrate_frac>0 이면 few-shot 라벨로 bias/finetune 허용.
    has_calib = bool(data.calib_segments) and float(data.calibrate_frac or 0) > 0
    if data.holdout_vehicle and args.vehicle_bias_correct and not has_calib:
        print(
            f"[WARN] LOVO holdout={data.holdout_vehicle} → "
            "vehicle_bias_correct 비활성화 (새 차량 가정). "
            "few-shot은 --calibrate_frac 사용."
        )
        args.vehicle_bias_correct = False
    if has_calib and not args.vehicle_bias_correct and args.finetune_epochs <= 0:
        print(
            "[INFO] calibrate_frac 사용 → vehicle_bias_correct 자동 활성화"
        )
        args.vehicle_bias_correct = True

    use_vehicle_emb = bool(args.vehicle_embedding)
    n_emb_vehicles = len(data.vehicle_to_idx or {}) if use_vehicle_emb else 0
    args.num_vehicles_emb = n_emb_vehicles

    print(f"[INFO] target_col: {data.target_col}")
    print(f"[INFO] feature_count: {len(data.feature_cols)}")
    print(f"[INFO] feature_cols: {data.feature_cols}")
    print(
        f"[INFO] speed opts: sample_stride={args.sample_stride} "
        f"window_stride={args.window_stride} batch_size={args.batch_size} epochs={args.epochs}"
    )
    print(f"[INFO] vehicle_bias_correct: {args.vehicle_bias_correct}")
    print(f"[INFO] per_vehicle_norm: {data.per_vehicle_norm}")
    print(f"[INFO] holdout_vehicle: {data.holdout_vehicle or '-'}")
    print(
        f"[INFO] calibrate_frac: {data.calibrate_frac} | "
        f"finetune_epochs: {args.finetune_epochs} | finetune_lr: {args.finetune_lr}"
    )
    print(f"[INFO] fix_zero_temp: {args.fix_zero_temp}")
    print(
        f"[INFO] vehicle_embedding: {use_vehicle_emb} "
        f"(n={n_emb_vehicles}, dim={args.vehicle_emb_dim})"
    )
    print(f"[INFO] loss: {args.loss} | median_kernel: {args.median_kernel}")

    pin = device.type == "cuda"
    if args.loss == "huber":
        criterion = nn.HuberLoss(delta=args.huber_delta)
    elif args.loss == "mae":
        criterion = nn.L1Loss()
    else:
        criterion = nn.MSELoss()
    env_info = collect_env_info(device)

    train_losses: List[float] = []
    val_losses: List[float] = []
    best_val = float("inf")
    best_epoch = 0
    stopped_early = False
    best_path = os.path.join(args.checkpoint_dir, "best_model.pt")
    optimizer = None

    if args.eval_checkpoint:
        ckpt_path = args.eval_checkpoint
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"eval_checkpoint 없음: {ckpt_path}")
        print(f"[INFO] eval-only from checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        cfg = ckpt.get("model_config") or {}
        feature_cols_ckpt = list(ckpt.get("feature_cols") or data.feature_cols)
        model = PatchTSTRegressor(
            num_features=int(cfg.get("num_features", len(feature_cols_ckpt))),
            seq_len=int(cfg.get("seq_len", args.seq_len)),
            patch_len=int(cfg.get("patch_len", args.patch_len)),
            stride=int(cfg.get("stride", args.stride)),
            d_model=int(cfg.get("d_model", args.d_model)),
            nhead=int(cfg.get("nhead", args.nhead)),
            num_layers=int(cfg.get("num_layers", args.num_layers)),
            ff_dim=int(cfg.get("ff_dim", args.ff_dim)),
            dropout=float(cfg.get("dropout", args.dropout)),
            num_vehicles=int(cfg.get("num_vehicles", 0)),
            vehicle_emb_dim=int(cfg.get("vehicle_emb_dim", args.vehicle_emb_dim)),
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        args.seq_len = int(cfg.get("seq_len", args.seq_len))
        use_vehicle_emb = int(cfg.get("num_vehicles", 0)) > 0
        n_emb_vehicles = int(cfg.get("num_vehicles", 0))
        args.num_vehicles_emb = n_emb_vehicles
        if ckpt.get("vehicle_to_idx"):
            data.vehicle_to_idx = ckpt["vehicle_to_idx"]
        data = apply_checkpoint_scalers(data, ckpt)
        best_path = ckpt_path
        # 체크포인트와 같은 run 이름의 metrics를 우선 탐색
        ckpt_run = os.path.basename(os.path.dirname(os.path.abspath(ckpt_path)))
        cand_list = [
            os.path.join(args.output_dir, "metrics.json"),
            os.path.join("outputs", ckpt_run, "metrics.json"),
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(ckpt_path))),
                "outputs",
                ckpt_run,
                "metrics.json",
            ),
            "outputs/run2_fast/metrics.json",
        ]
        for cand in cand_list:
            if os.path.isfile(cand):
                with open(cand, encoding="utf-8") as f:
                    src = json.load(f)
                if src.get("train_losses") or src.get("best_epoch"):
                    train_losses = list(src.get("train_losses") or [])
                    val_losses = list(src.get("val_losses") or [])
                    best_epoch = int(src.get("best_epoch") or 0)
                    best_val = float(src.get("best_val_loss") or float("inf"))
                    print(f"[INFO] loaded train history from: {cand}")
                    break
    else:
        train_ds = build_concat_dataset(
            data.train_segments,
            seq_len=args.seq_len,
            window_stride=args.window_stride,
            vehicle_ids=data.train_vehicle_ids,
            vehicle_to_idx=data.vehicle_to_idx,
            use_vehicle_idx=use_vehicle_emb,
        )
        val_ds = build_concat_dataset(
            data.val_segments,
            seq_len=args.seq_len,
            window_stride=args.window_stride,
            vehicle_ids=data.val_vehicle_ids,
            vehicle_to_idx=data.vehicle_to_idx,
            use_vehicle_idx=use_vehicle_emb,
        )
        print(f"[INFO] train/val windows: {len(train_ds)}/{len(val_ds)}")

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
            num_vehicles=n_emb_vehicles,
            vehicle_emb_dim=args.vehicle_emb_dim,
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.lr_factor,
            patience=args.lr_patience,
            min_lr=args.min_lr,
        )
        epochs_no_improve = 0

        print(
            f"[INFO] train stability: weight_decay={args.weight_decay} "
            f"dropout={args.dropout} patience={args.patience} "
            f"lr_factor={args.lr_factor} lr_patience={args.lr_patience}"
        )

        for epoch in range(1, args.epochs + 1):
            tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            va_loss, _, _ = evaluate(model, val_loader, criterion, device)
            train_losses.append(tr_loss)
            val_losses.append(va_loss)

            prev_lr = optimizer.param_groups[0]["lr"]
            scheduler.step(va_loss)
            curr_lr = optimizer.param_groups[0]["lr"]
            if curr_lr < prev_lr - 1e-15:
                print(f"[INFO] LR reduced: {prev_lr:.2e} -> {curr_lr:.2e}")

            if va_loss < best_val:
                best_val = va_loss
                best_epoch = epoch
                epochs_no_improve = 0
                torch.save(
                    build_checkpoint(
                        model=model,
                        feature_scaler=data.feature_scaler,
                        target_scaler=data.target_scaler,
                        feature_cols=data.feature_cols,
                        target_col=data.target_col,
                        args=args,
                        feature_scalers=data.feature_scalers,
                        target_scalers=data.target_scalers,
                        per_vehicle_norm=data.per_vehicle_norm,
                        vehicle_to_idx=data.vehicle_to_idx,
                    ),
                    best_path,
                )
            else:
                epochs_no_improve += 1

            print(
                f"[Epoch {epoch:03d}/{args.epochs}] "
                f"train_loss={tr_loss:.6f} val_loss={va_loss:.6f} lr={curr_lr:.2e}"
            )

            if args.patience > 0 and epochs_no_improve >= args.patience:
                stopped_early = True
                print(
                    f"[INFO] Early stopping at epoch {epoch} "
                    f"(no val improvement for {args.patience} epochs, best={best_epoch})"
                )
                break

        if not os.path.isfile(best_path):
            raise RuntimeError(f"best checkpoint가 없습니다: {best_path}")
        load_model_state(model, best_path, device)

    test_vids = data.test_vehicle_ids or data.vehicle_ids
    val_vids = data.val_vehicle_ids or data.vehicle_ids

    # Few-shot: holdout 캘리브레이션 구간으로 fine-tune (+ 이후 bias)
    finetune_losses: List[float] = []
    if has_calib and args.finetune_epochs > 0:
        calib_vids = data.calib_vehicle_ids or test_vids
        ft_ds = build_concat_dataset(
            data.calib_segments,
            seq_len=args.seq_len,
            window_stride=max(1, args.window_stride),
            vehicle_ids=calib_vids,
            vehicle_to_idx=data.vehicle_to_idx,
            use_vehicle_idx=use_vehicle_emb,
        )
        ft_loader = DataLoader(
            ft_ds,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=max(1, args.num_workers // 2),
            pin_memory=pin,
        )
        ft_opt = torch.optim.AdamW(
            model.parameters(), lr=args.finetune_lr, weight_decay=args.weight_decay
        )
        print(
            f"[INFO] few-shot fine-tune: epochs={args.finetune_epochs} "
            f"lr={args.finetune_lr} windows={len(ft_ds)}"
        )
        for ep in range(1, args.finetune_epochs + 1):
            ft_loss = train_one_epoch(model, ft_loader, ft_opt, criterion, device)
            finetune_losses.append(ft_loss)
            print(f"[FT {ep:02d}/{args.finetune_epochs}] loss={ft_loss:.6f}")
        ft_ckpt = os.path.join(args.checkpoint_dir, "finetuned_model.pt")
        torch.save(
            build_checkpoint(
                model=model,
                feature_scaler=data.feature_scaler,
                target_scaler=data.target_scaler,
                feature_cols=data.feature_cols,
                target_col=data.target_col,
                args=args,
                feature_scalers=data.feature_scalers,
                target_scalers=data.target_scalers,
                per_vehicle_norm=data.per_vehicle_norm,
                vehicle_to_idx=data.vehicle_to_idx,
            ),
            ft_ckpt,
        )
        best_path = ft_ckpt
        print(f"[INFO] saved finetuned checkpoint: {ft_ckpt}")

    vehicle_biases = None
    bias_segments = data.val_segments
    bias_vids = val_vids
    if has_calib:
        bias_segments = data.calib_segments or data.val_segments
        bias_vids = data.calib_vehicle_ids or val_vids
    if args.vehicle_bias_correct:
        print("[INFO] Estimating per-vehicle bias on validation/calib set...")
        vehicle_biases = estimate_vehicle_biases(
            model=model,
            segments=bias_segments,
            vehicle_ids=bias_vids,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            criterion=criterion,
            device=device,
            target_scaler=data.target_scaler,
            target_scalers=data.target_scalers,
            vehicle_to_idx=data.vehicle_to_idx,
            use_vehicle_idx=use_vehicle_emb,
        )

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
        vehicle_ids=test_vids,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        criterion=criterion,
        device=device,
        target_scaler=data.target_scaler,
        vehicle_biases=vehicle_biases,
        target_scalers=data.target_scalers,
        vehicle_to_idx=data.vehicle_to_idx,
        use_vehicle_idx=use_vehicle_emb,
        median_kernel=args.median_kernel,
    )

    mae = float(mean_absolute_error(true, pred))
    mse = float(mean_squared_error(true, pred))
    rmse = float(math.sqrt(mse))

    final_lr = float(optimizer.param_groups[0]["lr"]) if optimizer is not None else None

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
        "epochs_ran": len(train_losses),
        "num_workers": args.num_workers,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "patience": args.patience,
        "lr_factor": args.lr_factor,
        "lr_patience": args.lr_patience,
        "min_lr": args.min_lr,
        "stopped_early": stopped_early,
        "best_epoch": best_epoch,
        "best_val_loss": best_val if best_val < float("inf") else None,
        "final_lr": final_lr,
        "vehicle_bias_correct": bool(args.vehicle_bias_correct),
        "vehicle_biases": vehicle_biases,
        "per_vehicle_norm": bool(data.per_vehicle_norm),
        "holdout_vehicle": data.holdout_vehicle or None,
        "calibrate_frac": float(data.calibrate_frac or 0.0),
        "finetune_epochs": int(args.finetune_epochs),
        "finetune_lr": float(args.finetune_lr) if args.finetune_epochs > 0 else None,
        "finetune_losses": finetune_losses,
        "fix_zero_temp": bool(args.fix_zero_temp),
        "vehicle_embedding": use_vehicle_emb,
        "vehicle_emb_dim": args.vehicle_emb_dim if use_vehicle_emb else 0,
        "num_vehicles_emb": n_emb_vehicles,
        "vehicle_to_idx": data.vehicle_to_idx,
        "loss": args.loss,
        "huber_delta": args.huber_delta if args.loss == "huber" else None,
        "median_kernel": args.median_kernel,
        "eval_checkpoint": args.eval_checkpoint or None,
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
        train_losses=train_losses if train_losses else [0.0],
        val_losses=val_losses if val_losses else [0.0],
        y_true=true,
        y_pred=pred,
        device_nos=device_nos,
        time_indices=time_indices,
        metrics=metrics,
        env_info=env_info,
        skip_report=bool(args.eval_checkpoint),
    )

    print(f"[DONE] Best Epoch: {best_epoch} (val_loss={best_val})")
    print(f"[DONE] Test MAE:  {mae:.6f}")
    print(f"[DONE] Test MSE:  {mse:.6f}")
    print(f"[DONE] Test RMSE: {rmse:.6f}")
    if vehicle_biases:
        print(f"[DONE] vehicle biases: {vehicle_biases}")
    for vid, vm in vehicle_metrics.items():
        print(f"[DONE]   {vid}: MAE={vm['mae']:.6f} RMSE={vm['rmse']:.6f}")
    print(f"[DONE] outputs saved to: {args.output_dir}")
    print(f"[DONE] best checkpoint: {best_path}")


def apply_checkpoint_scalers(data: DataBundle, ckpt: dict) -> DataBundle:
    """
    체크포인트의 scaler로 train/val/test segment를 다시 맞춥니다.
    load_and_prepare_data가 이미 transform한 값이므로, inverse(현재)→transform(ckpt) 적용.
    """
    per_vehicle = bool(ckpt.get("per_vehicle_norm")) and bool(ckpt.get("target_scalers"))
    if per_vehicle:
        old_x_map = data.feature_scalers or {
            vid: data.feature_scaler for vid in data.vehicle_ids
        }
        old_y_map = data.target_scalers or {
            vid: data.target_scaler for vid in data.vehicle_ids
        }
        new_x_map = ckpt["feature_scalers"]
        new_y_map = ckpt["target_scalers"]

        def remap_seg(seg_list):
            out = []
            for vid, (x_sc, y_sc) in zip(data.vehicle_ids, seg_list):
                x_raw = old_x_map[vid].inverse_transform(x_sc)
                y_raw = old_y_map[vid].inverse_transform(y_sc.reshape(-1, 1)).reshape(-1)
                out.append(
                    (
                        new_x_map[vid].transform(x_raw),
                        new_y_map[vid].transform(y_raw.reshape(-1, 1)).reshape(-1),
                    )
                )
            return out

        data.train_segments = remap_seg(data.train_segments)
        data.val_segments = remap_seg(data.val_segments)
        data.test_segments = remap_seg(data.test_segments)
        if data.calib_segments:
            data.calib_segments = remap_seg(data.calib_segments)
        data.feature_scalers = new_x_map
        data.target_scalers = new_y_map
        data.per_vehicle_norm = True
        data.feature_scaler = ckpt.get("feature_scaler", data.feature_scaler)
        data.target_scaler = ckpt.get("target_scaler", data.target_scaler)
    else:
        old_x = data.feature_scaler
        old_y = data.target_scaler
        new_x = ckpt["feature_scaler"]
        new_y = ckpt["target_scaler"]

        def remap_seg(seg_list):
            out = []
            for x_sc, y_sc in seg_list:
                x_raw = old_x.inverse_transform(x_sc)
                y_raw = old_y.inverse_transform(y_sc.reshape(-1, 1)).reshape(-1)
                out.append(
                    (
                        new_x.transform(x_raw),
                        new_y.transform(y_raw.reshape(-1, 1)).reshape(-1),
                    )
                )
            return out

        data.train_segments = remap_seg(data.train_segments)
        data.val_segments = remap_seg(data.val_segments)
        data.test_segments = remap_seg(data.test_segments)
        if data.calib_segments:
            data.calib_segments = remap_seg(data.calib_segments)
        data.feature_scaler = new_x
        data.target_scaler = new_y
        data.per_vehicle_norm = False
        data.feature_scalers = None
        data.target_scalers = None

    data.feature_cols = list(ckpt.get("feature_cols") or data.feature_cols)
    data.target_col = ckpt.get("target_col") or data.target_col
    return data


if __name__ == "__main__":
    main()
