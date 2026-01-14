# multi_lake_dataset.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


# =========================================================
# Depth feature builder (area-by-bin)
# =========================================================
def build_depth_features_from_area(area_ha: np.ndarray) -> np.ndarray:
    """
    area_ha: (Dz,) or (Dz,1)
    Returns depth_feat: (Dz, Dd) with simple, robust features:
      - z_norm in (0,1)
      - area_norm
      - cum_area (0..1)
      - log_area
    """
    a = np.asarray(area_ha).reshape(-1).astype(np.float32)
    Dz = a.shape[0]
    eps = 1e-8

    z_norm = (np.arange(Dz, dtype=np.float32) + 0.5) / float(Dz)
    area_norm = a / (a.sum() + eps)
    cum_area = np.cumsum(area_norm)
    log_area = np.log(a + eps)

    feat = np.stack([z_norm, area_norm, cum_area, log_area], axis=-1).astype(np.float32)  # (Dz, 4)
    return feat


# =========================================================
# Helpers: padding
# =========================================================
def pad_2d(arr: np.ndarray, target_len: int, pad_value: float = 0.0) -> np.ndarray:
    """
    arr: (L, D) -> (target_len, D)
    """
    arr = np.asarray(arr)
    L, D = arr.shape
    if L == target_len:
        return arr
    out = np.full((target_len, D), pad_value, dtype=arr.dtype)
    out[:L, :] = arr
    return out


# =========================================================
# Multi-lake data container
# =========================================================
@dataclass
class LakeData:
    name: str
    year: np.ndarray          # (T,)
    doy: np.ndarray           # (T,)
    drivers: np.ndarray       # (T, Du)
    params: np.ndarray        # (N, P)
    temps: np.ndarray         # (N, T, Dz)
    depth_feat: np.ndarray    # (Dz, Dd)
    Dz: int


# =========================================================
# Load one lake directory
# =========================================================
def load_one_lake(lake_dir: str, lake_name: Optional[str] = None) -> LakeData:
    """
    Expects files in lake_dir:
      weather.npy      (T, 2+Du) with [YEAR, DOY, ...drivers]
      outputs.npy      (N, T, 2+Dz) with [YEAR, DOY, temps...]
      parameters.npy   (N, P)
      bathymetry.npy   (Dz,1) or (Dz,)
    """
    if lake_name is None:
        lake_name = os.path.basename(os.path.normpath(lake_dir))

    weather = np.load(os.path.join(lake_dir, "weather.npy"))
    outputs = np.load(os.path.join(lake_dir, "outputs.npy"))
    params = np.load(os.path.join(lake_dir, "parameters.npy"))
    bathy = np.load(os.path.join(lake_dir, "bathymetry.npy"))

    weather = weather.astype(np.float32, copy=False)
    outputs = outputs.astype(np.float32, copy=False)
    params = params.astype(np.float32, copy=False)

    year = weather[:, 0].astype(np.int32)
    doy = weather[:, 1].astype(np.float32)
    drivers = weather[:, 2:].astype(np.float32)  # (T, Du)

    # outputs: (N, T, 2+Dz) => temps (N, T, Dz)
    temps = outputs[:, :, 2:].astype(np.float32)
    Dz = temps.shape[2]

    depth_feat = build_depth_features_from_area(bathy)  # (Dz, Dd)

    return LakeData(
        name=lake_name,
        year=year,
        doy=doy,
        drivers=drivers,
        params=params,
        temps=temps,
        depth_feat=depth_feat,
        Dz=Dz,
    )


# =========================================================
# Global (multi-lake) normalization
# =========================================================
def compute_global_norms(
    lakes: List[LakeData],
    split_years: Tuple[int, int, int] = (2018, 2020, 2025),
    eps: float = 1e-6,
) -> Dict[str, np.ndarray]:
    """
    Train-only normalization across multiple lakes.

    - Weather drivers: concatenated across all lakes' TRAIN periods.
    - Params: concatenated across all lakes (all sims).
    - Outputs: variable depth -> pad to Dz_max and use nanmean/nanstd per depth index.
      (So output_mean/output_std are (1,1,Dz_max)).
    """
    train_end = split_years[0]

    # Weather
    w_all = []
    for lk in lakes:
        idx_train = lk.year <= train_end
        w_all.append(lk.drivers[idx_train, :].astype(np.float32))
    W = np.concatenate(w_all, axis=0)  # (sum(Ttrain), Du)
    w_mean = W.mean(axis=0, keepdims=True)
    w_std = W.std(axis=0, keepdims=True) + eps

    # Params
    p_all = [lk.params.astype(np.float32) for lk in lakes]
    P = np.concatenate(p_all, axis=0)  # (sum(N), P)
    p_mean = P.mean(axis=0, keepdims=True)
    p_std = P.std(axis=0, keepdims=True) + eps

    # Outputs (variable depth)
    Dz_max = max(lk.Dz for lk in lakes)
    o_rows = []
    for lk in lakes:
        idx_train = lk.year <= train_end  # same time axis for temps/drivers
        temps = lk.temps[:, idx_train, :]  # (N, Ttrain, Dz_lake)
        N, T, Dz_lake = temps.shape
        temps2d = temps.reshape(N * T, Dz_lake)  # (N*T, Dz_lake)

        pad = np.full((temps2d.shape[0], Dz_max), np.nan, dtype=np.float32)
        pad[:, :Dz_lake] = temps2d
        o_rows.append(pad)

    O = np.concatenate(o_rows, axis=0)  # (sum(N*Ttrain), Dz_max)
    o_mean_1d = np.nanmean(O, axis=0, keepdims=True)  # (1, Dz_max)
    o_std_1d = np.nanstd(O, axis=0, keepdims=True) + eps

    norms = {
        "weather_mean": w_mean.astype(np.float32),           # (1, Du)
        "weather_std": w_std.astype(np.float32),             # (1, Du)
        "params_mean": p_mean.astype(np.float32),            # (1, P)
        "params_std": p_std.astype(np.float32),              # (1, P)
        "output_mean": o_mean_1d.reshape(1, 1, Dz_max).astype(np.float32),  # (1,1,Dz_max)
        "output_std": o_std_1d.reshape(1, 1, Dz_max).astype(np.float32),    # (1,1,Dz_max)
        "Dz_max": np.array([Dz_max], dtype=np.int32),
        "split_years": np.array(list(split_years), dtype=np.int32),
    }
    return norms


def apply_norms_inplace(lk: LakeData, norms: Dict[str, np.ndarray]) -> None:
    """
    Applies norms computed above:
      drivers: (T,Du)
      params: (N,P)
      temps: depth-wise normalization using first Dz bins of global (Dz_max)
    """
    lk.drivers = (lk.drivers - norms["weather_mean"]) / norms["weather_std"]
    lk.params = (lk.params - norms["params_mean"]) / norms["params_std"]

    mu = norms["output_mean"][0, 0, :]  # (Dz_max,)
    sd = norms["output_std"][0, 0, :]   # (Dz_max,)

    # Only normalize existing depths
    mu_l = mu[: lk.Dz].reshape(1, 1, lk.Dz)
    sd_l = sd[: lk.Dz].reshape(1, 1, lk.Dz)
    lk.temps = (lk.temps - mu_l) / sd_l


# =========================================================
# Dataset
# =========================================================
class MultiLakeWindowDataset(Dataset):
    """
    Padded + masked depth dataset across multiple lakes.

    Each item returns:
      lake_id: int
      x_win: (Wx, Du)
      p: (P,)
      depth_feat: (Dz_max, Dd)
      depth_mask: (Dz_max,)
      y: (Wy, Dz_max)

    Alignment:
      y corresponds to last Wy time steps of the Wx window:
        y = temps[sim, start+Wx-Wy : start+Wx, :]
    """

    def __init__(
        self,
        lakes: List[LakeData],
        split: str,
        split_years: Tuple[int, int, int],
        Wx: int,
        Wy: int,
        precompute_weather_windows: bool = True,
        window_stride: int = 1,  # you can raise this to reduce overlap
    ):
        super().__init__()
        assert split in ("train", "val", "test")
        assert 1 <= Wy <= Wx
        assert window_stride >= 1

        self.lakes = lakes
        self.split = split
        self.Wx = int(Wx)
        self.Wy = int(Wy)
        self.window_stride = int(window_stride)
        self.precompute_weather_windows = bool(precompute_weather_windows)

        # Basic dims
        self.Du = lakes[0].drivers.shape[1]
        self.P = lakes[0].params.shape[1]
        self.Dd = lakes[0].depth_feat.shape[1]

        # Pad depths to Dz_max across lakes
        self.Dz_max = max(lk.Dz for lk in lakes)

        # Pre-pad per-lake depth_feat and mask
        self.depth_feat_padded: List[np.ndarray] = []
        self.depth_mask_padded: List[np.ndarray] = []
        for lk in lakes:
            df = pad_2d(lk.depth_feat, self.Dz_max, pad_value=0.0)  # (Dz_max, Dd)
            m = np.zeros((self.Dz_max,), dtype=np.float32)
            m[: lk.Dz] = 1.0
            self.depth_feat_padded.append(df.astype(np.float32))
            self.depth_mask_padded.append(m)

        tr_end, va_end, _ = split_years

        # Time indices per lake for the split
        self.time_index_per_lake: List[np.ndarray] = []
        for lk in lakes:
            if split == "train":
                idx = np.where(lk.year <= tr_end)[0]
            elif split == "val":
                idx = np.where((lk.year > tr_end) & (lk.year <= va_end))[0]
            else:
                idx = np.where(lk.year > va_end)[0]
            self.time_index_per_lake.append(idx.astype(np.int32))

        # Precompute weather windows per lake/split if requested
        self.weather_wins: List[Optional[torch.Tensor]] = [None] * len(lakes)
        self.valid_window_starts: List[np.ndarray] = []

        for lake_id, lk in enumerate(lakes):
            t_idx = self.time_index_per_lake[lake_id]
            T = t_idx.shape[0]
            if T < self.Wx:
                raise ValueError(f"Lake {lk.name} split={split} too short T={T} for Wx={self.Wx}")

            starts = np.arange(0, T - self.Wx + 1, self.window_stride, dtype=np.int32)
            self.valid_window_starts.append(starts)

            if self.precompute_weather_windows:
                drivers_split = lk.drivers[t_idx, :]  # (T, Du)

                # SAFE sliding window: (W, Wx, Du)
                wv = np.lib.stride_tricks.sliding_window_view(
                    drivers_split, window_shape=self.Wx, axis=0
                )
                # wv is a view; make writable contiguous copy to avoid torch warning
                w_np = np.ascontiguousarray(wv).copy()
                self.weather_wins[lake_id] = torch.from_numpy(w_np).float()

        # Build global index over (lake_id, sim_id, start_idx_in_split)
        self.index: List[Tuple[int, int, int]] = []
        for lake_id, lk in enumerate(lakes):
            N = lk.params.shape[0]
            starts = self.valid_window_starts[lake_id]
            for sim_id in range(N):
                for s in starts:
                    self.index.append((lake_id, sim_id, int(s)))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, k: int):
        lake_id, sim_id, start = self.index[k]
        lk = self.lakes[lake_id]
        t_idx = self.time_index_per_lake[lake_id]

        # Window indices (global time)
        global_slice = t_idx[start : start + self.Wx]

        # Targets correspond to last Wy within that window
        y_start = start + (self.Wx - self.Wy)
        y_end = start + self.Wx
        y_global = t_idx[y_start:y_end]  # length Wy

        # Weather window
        if self.precompute_weather_windows and (self.weather_wins[lake_id] is not None):
            x_win = self.weather_wins[lake_id][start]  # (Wx, Du)
        else:
            x_np = lk.drivers[global_slice, :]  # (Wx, Du)
            x_win = torch.from_numpy(np.ascontiguousarray(x_np).copy()).float()

        # Params
        p = torch.from_numpy(np.ascontiguousarray(lk.params[sim_id]).copy()).float()  # (P,)

        # y: (Wy, Dz_lake) -> pad to (Wy, Dz_max)
        y_np = lk.temps[sim_id, y_global, :]  # (Wy, Dz_lake)
        y_pad = np.zeros((self.Wy, self.Dz_max), dtype=np.float32)
        y_pad[:, : lk.Dz] = y_np.astype(np.float32, copy=False)

        depth_feat = torch.from_numpy(np.ascontiguousarray(self.depth_feat_padded[lake_id]).copy()).float()
        depth_mask = torch.from_numpy(np.ascontiguousarray(self.depth_mask_padded[lake_id]).copy()).float()
        y = torch.from_numpy(np.ascontiguousarray(y_pad).copy()).float()

        return int(lake_id), x_win, p, depth_feat, depth_mask, y


# =========================================================
# High-level loader
# =========================================================
def load_multi_lake_data(
    root_dir: str,
    lake_names: List[str],
    split_years: Tuple[int, int, int],
    normalize: bool = True,
) -> Tuple[List[LakeData], Optional[Dict[str, np.ndarray]]]:
    lakes: List[LakeData] = []
    for nm in lake_names:
        lk_dir = os.path.join(root_dir, nm)
        lakes.append(load_one_lake(lk_dir, lake_name=nm))

    norms = None
    if normalize:
        norms = compute_global_norms(lakes, split_years=split_years)
        for lk in lakes:
            apply_norms_inplace(lk, norms)

    return lakes, norms
