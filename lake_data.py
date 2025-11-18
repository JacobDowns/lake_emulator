# lake_data.py
import math
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# ---------- Meta / Fourier helpers ----------
def split_meta_and_drivers(weather_with_meta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Input: weather_with_meta shape (T, 2 + Du) with columns [YEAR, DOY, drivers...]
    Returns:
        meta:    (T, 2) int32 (YEAR, DOY)
        drivers: (T, Du) float32
    """
    meta = weather_with_meta[:, :2].astype(np.int32)
    drivers = weather_with_meta[:, 2:].astype(np.float32)
    return meta, drivers


def build_fourier_from_year_doy(year: np.ndarray, doy: np.ndarray, harmonics=(1, 2)) -> np.ndarray:
    """
    Construct seasonal Fourier features of shape (T, 2*K) for given YEAR, DOY.
    Handles leap years by using year_len 366 when appropriate.
    """
    year = year.astype(int)
    doy = doy.astype(int)
    year_len = np.where(((year % 4 == 0) & (year % 100 != 0)) | (year % 400 == 0), 366, 365).astype(float)

    cols = []
    for k in harmonics:
        theta = 2 * np.pi * k * (doy.astype(float) / year_len)
        cols += [np.sin(theta), np.cos(theta)]
    return np.column_stack(cols).astype(np.float32) if cols else np.zeros((len(year), 0), dtype=np.float32)


def strict_test_split_by_year_from_meta(
    meta_year: np.ndarray, L: int, lead: int, test_years: tuple[int, int], val_frac_pretest: float = 0.2
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build strict (train, val, test) time indices using YEAR from the meta columns.
    meta_year: (T,) array of per-timestep YEAR (int)
    """
    T = len(meta_year)
    t_lo = L - 1
    t_hi = T - 1 - lead
    valid = np.arange(t_lo, t_hi + 1, dtype=int)

    # YEAR label corresponds to the target (t+lead), so select by meta_year[t+lead]
    mask_test = (meta_year[valid + lead] >= test_years[0]) & (meta_year[valid + lead] <= test_years[1])
    t_test = valid[mask_test]
    remain = valid[~mask_test]

    n_val = max(1, int(len(remain) * val_frac_pretest))
    t_val = remain[-n_val:] if n_val > 0 else np.array([], dtype=int)
    t_train = remain[:-n_val] if n_val > 0 else remain
    return t_train, t_val, t_test


# ---------- Normalization helpers ----------
def fit_norms_on_train(
    drivers_with_fourier: np.ndarray, params: np.ndarray, t_train: np.ndarray, L: int, exclude_tail_cols: int = 0
):
    """
    Fit mean/std on TRAIN windows only for drivers (excluding the last 'exclude_tail_cols' columns, i.e., Fourier).
    Returns means/stds for drivers core and for params, and Du_core.
    """
    rows = np.unique(np.concatenate([np.arange(t - L + 1, t + 1) for t in t_train]))
    Du_total = drivers_with_fourier.shape[1]
    Du_core = Du_total - int(exclude_tail_cols)

    core = drivers_with_fourier[:, :Du_core]
    x_mean = core[rows].mean(axis=0, keepdims=True)
    x_std = core[rows].std(axis=0, keepdims=True) + 1e-6
    p_mean = params.mean(axis=0, keepdims=True)
    p_std = params.std(axis=0, keepdims=True) + 1e-6

    return x_mean.astype(np.float32), x_std.astype(np.float32), p_mean.astype(np.float32), p_std.astype(np.float32), Du_core


def apply_norms(
    drivers_with_fourier: np.ndarray,
    params: np.ndarray,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    p_mean: np.ndarray,
    p_std: np.ndarray,
    exclude_tail_cols: int = 0,
):
    """
    Standardize drivers core columns using (x_mean, x_std) and leave the tail (Fourier) untouched.
    """
    Du_total = drivers_with_fourier.shape[1]
    Du_core = Du_total - int(exclude_tail_cols)

    core = (drivers_with_fourier[:, :Du_core] - x_mean) / x_std
    if exclude_tail_cols > 0:
        tail = drivers_with_fourier[:, Du_core:]
        drivers_std = np.concatenate([core, tail], axis=1)
    else:
        drivers_std = core

    params_std = (params - p_mean) / p_std
    return drivers_std.astype(np.float32), params_std.astype(np.float32)


# ---------- Dataset ----------
class WindowNextDayDataset(Dataset):
    """
    Returns (drivers_win, params_vec, target):
      drivers_win: (L, Du_use)
      params_vec : (P,)
      target     : (Dz,)
    Assumes inputs are already normalized; YEAR/DOY are NOT included here.
    """
    def __init__(self, drivers_std: np.ndarray, params_std: np.ndarray, outputs: np.ndarray,
                 L: int = 90, lead: int = 1, t_indices: np.ndarray | None = None, trials: list[int] | None = None):
        self.drivers = drivers_std.astype(np.float32)    # (T, Du_use)
        self.params  = params_std.astype(np.float32)     # (N, P)
        self.outputs = outputs.astype(np.float32)        # (N, T, Dz)

        self.T, self.Du = self.drivers.shape
        self.N, T2, self.Dz = self.outputs.shape
        assert self.T == T2, "Drivers and outputs time lengths must match."

        self.L = int(L)
        self.lead = int(lead)
        t_lo = self.L - 1
        t_hi = self.T - 1 - self.lead
        valid = np.arange(t_lo, t_hi + 1, dtype=int)

        self.t_idx = valid if t_indices is None else np.array([t for t in t_indices if t_lo <= t <= t_hi], dtype=int)
        self.trials = list(range(self.N)) if trials is None else list(trials)

        self.index = [(i, t) for i in self.trials for t in self.t_idx]

    def __len__(self): return len(self.index)

    def __getitem__(self, k: int):
        i, t = self.index[k]
        x_win = self.drivers[t - self.L + 1 : t + 1, :]  # (L, Du)
        p_vec = self.params[i, :]                         # (P,)
        y     = self.outputs[i, t + self.lead, :]         # (Dz,)
        return torch.from_numpy(x_win), torch.from_numpy(p_vec), torch.from_numpy(y)
