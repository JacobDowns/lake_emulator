from torch.utils.data import Dataset
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd


# =========================================================
# Dataset: windowing over arrays returned by load_data
# =========================================================
class LakeWindowDataset(Dataset):
    """
    Given:
      weather: (T, Du)
      outputs: (N, T, Dz)
      params : (N, P)
      extra_features: (T, D_extra) or None

    For window length W_x and horizon W_y (<= W_x):
      x_win: (W_x, Du_total = Du + D_extra)
      p_vec: (P,)
      y    : (W_y, Dz)  (we squeeze y to (Dz) in the loop if W_y==1)

    If return_ids=True, __getitem__ returns:
      sim_id, x_win, p_vec, y
    otherwise:
      x_win, p_vec, y
    """
    def __init__(
        self,
        weather: np.ndarray,
        outputs: np.ndarray,
        params: np.ndarray,
        W_x: int = 90,
        W_y: int = 1,
        trial_ids=None,
        extra_features: np.ndarray | None = None,
        return_ids: bool = False

    ):
        super().__init__()
        self.return_ids = return_ids

        # Concatenate extra time-dependent features if provided
        if extra_features is not None:
            assert extra_features.shape[0] == weather.shape[0], \
                "extra_features must have same T as weather"
            weather = np.concatenate([weather, extra_features], axis=-1)

        assert 1 <= W_y <= W_x, "Require 1 <= W_y <= W_x"
        assert weather.ndim == 2 and outputs.ndim == 3 and params.ndim == 2

        T, Du = weather.shape
        N, T2, Dz = outputs.shape
        assert T == T2 and N == params.shape[0]

        self.W_x, self.W_y = W_x, W_y
        self.Du, self.Dz = Du, Dz

        self.params = torch.from_numpy(params).float()  # (N, P)
        self.outputs = outputs                           # (N, T, Dz), numpy

        if trial_ids is None:
            self.trial_ids = np.arange(N, dtype=int)
        else:
            self.trial_ids = np.array(trial_ids, dtype=int)

        # Precompute weather windows (T - W_x + 1, W_x, Du)
        if T < W_x:
            raise ValueError(f"Not enough timesteps T={T} for W_x={W_x}")
        W = T - W_x + 1
        self.W = W

        # sliding_window_view gives shape (W, 1, W_x, Du) with this call
        wv = np.lib.stride_tricks.sliding_window_view(
            weather,
            window_shape=(W_x, Du),
            axis=(0, 1)
        )
        self.weather_wins = torch.from_numpy(wv[:, 0, :, :]).float()  # (W, W_x, Du)

        # index over all (n, w)
        self.index = [(i, w) for i in range(len(self.trial_ids)) for w in range(W)]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, k: int):
        i_idx, w = self.index[k]
        n = self.trial_ids[i_idx]   # global simulation index

        x_win = self.weather_wins[w]                         # (W_x, Du_total)
        p_vec = self.params[n]                               # (P,)

        start = w + self.W_x - self.W_y
        end   = w + self.W_x
        y_np  = self.outputs[n, start:end, :]                # (W_y, Dz)
        y = torch.from_numpy(y_np).float()

        if self.return_ids:
            return int(n), x_win, p_vec, y
        else:
            return x_win, p_vec, y
    
import numpy as np

def smooth_weather_timeseries(
    weather: np.ndarray,
    window_days: list[int],
    pad_mode: str = "reflect",   # "reflect" | "edge" | "wrap"
) -> dict[str, np.ndarray]:
    """
    Moving-average smooths of weather drivers across multiple window sizes
    with explicit padding (avoids np.convolve(mode="same") zero-padding artifacts).
    Returns arrays of same length T.

    weather: (T, Du)
    """
    if weather.ndim != 2:
        raise ValueError(f"weather must be 2D (T, Du); got shape {weather.shape}")

    if pad_mode not in ("reflect", "edge", "wrap"):
        raise ValueError(f"pad_mode must be one of 'reflect', 'edge', 'wrap'; got {pad_mode}")

    T, Du = weather.shape
    out: dict[str, np.ndarray] = {}

    for w in window_days:
        if not isinstance(w, (int, np.integer)) or w <= 0:
            raise ValueError(f"window sizes must be positive ints; got {w}")

        if w == 1:
            out[f"weather_smoothed_{w}"] = weather.astype(np.float32, copy=False)
            continue

        left = w // 2
        right = w - 1 - left  # left+right = w-1 ensures output length T with 'valid'

        padded = np.pad(weather, ((left, right), (0, 0)), mode=pad_mode)
        kernel = np.ones(w, dtype=np.float32) / float(w)

        sm = np.empty((T, Du), dtype=np.float32)
        for j in range(Du):
            sm[:, j] = np.convolve(padded[:, j], kernel, mode="valid").astype(np.float32)

        out[f"weather_smoothed_{w}"] = sm

    return out


def load_data(
    weather_path='data/parsed_data/weather_data.npy',
    output_path ='data/parsed_data/output_data.npy',
    params_path ='data/parsed_data/parameter_data.npy',
    bathymetry_path='data/BearLake_inputs_outputs/inputs/BearLake_bathy.csv',
    split_years = [2018, 2020, 2025],  # [train_end, val_end, max]
    normalize = True,
    smooth_windows: list[int] | None = None,   # e.g. [7, 30]
    smooth_pad_mode: str = "reflect",
):
    """
    Returns dict of splits with train-only normalization applied:
      weather_data_* : (T_split, Du_total)  # YEAR/DOY removed; drivers + optional smooths
      output_data_*  : (N, T_split, Dz)
      doy_*          : (T_split,)
      params_data    : (N, P)
      norms          : dict with means/stds for inverse-transform
    """
    weather = np.load(weather_path)   # (T, 2 + Du_raw): [YEAR, DOY, drivers...]
    outputs = np.load(output_path)    # (N, T, Dz)
    params  = np.load(params_path)    # (N, P)
    bathymetry = pd.read_csv(bathymetry_path)['area_ha'].to_numpy() / 100.0

    year = weather[:, 0]
    doy  = weather[:, 1].astype(np.float32)
    drivers = weather[:, 2:].astype(np.float32)  # (T, Du_raw)

    # ---- optional smoothing (append features) ----
    if smooth_windows is None:
        smooth_windows = []
    # allow arbitrary list; remove duplicates while keeping order
    seen = set()
    smooth_windows = [int(w) for w in smooth_windows if int(w) not in seen and not seen.add(int(w))]

    if len(smooth_windows) > 0:
        smooth_dict = smooth_weather_timeseries(drivers, smooth_windows, pad_mode=smooth_pad_mode)
        smooth_feats = [smooth_dict[f"weather_smoothed_{w}"] for w in smooth_windows]  # each (T, Du_raw)
        drivers = np.concatenate([drivers] + smooth_feats, axis=1)  # (T, Du_raw*(1+len(windows)))

    # ---- splits ----
    idx_train = (year <= split_years[0])
    idx_val   = (year > split_years[0]) & (year <= split_years[1])
    idx_test  = (year > split_years[1])

    weather_train = drivers[idx_train, :]
    weather_val   = drivers[idx_val, :]
    weather_test  = drivers[idx_test, :]

    outputs_train = outputs[:, idx_train, :].astype(np.float32)
    outputs_val   = outputs[:, idx_val, :].astype(np.float32)
    outputs_test  = outputs[:, idx_test, :].astype(np.float32)

    doy_train = doy[idx_train]
    doy_val   = doy[idx_val]
    doy_test  = doy[idx_test]

    params = params.astype(np.float32)

    # ---- normalization (train-only) ----
    if normalize:
        eps = 1e-6
        w_mean = weather_train.mean(axis=0, keepdims=True)               # (1, Du_total)
        w_std  = weather_train.std(axis=0, keepdims=True) + eps          # (1, Du_total)

        o_mean = outputs_train.mean(axis=(0, 1), keepdims=True)          # (1, 1, Dz)
        o_std  = outputs_train.std(axis=(0, 1), keepdims=True) + eps     # (1, 1, Dz)

        p_mean = params.mean(axis=0, keepdims=True)                      # (1, P)
        p_std  = params.std(axis=0, keepdims=True) + eps                 # (1, P)

        weather_train = (weather_train - w_mean) / w_std
        weather_val   = (weather_val   - w_mean) / w_std
        weather_test  = (weather_test  - w_mean) / w_std

        outputs_train = (outputs_train - o_mean) / o_std
        outputs_val   = (outputs_val   - o_mean) / o_std
        outputs_test  = (outputs_test  - o_mean) / o_std

        params = (params - p_mean) / p_std

        norms = {
            "weather_mean": w_mean.astype(np.float32),
            "weather_std":  w_std.astype(np.float32),
            "output_mean":  o_mean.astype(np.float32),
            "output_std":   o_std.astype(np.float32),
            "params_mean":  p_mean.astype(np.float32),
            "params_std":   p_std.astype(np.float32),
            "smooth_windows": np.array(smooth_windows, dtype=np.int32),
            "smooth_pad_mode": smooth_pad_mode,
        }
    else:
        norms = None

    return {
        "weather_data_train": weather_train.astype(np.float32),
        "output_data_train":  outputs_train.astype(np.float32),
        "weather_data_val":   weather_val.astype(np.float32),
        "output_data_val":    outputs_val.astype(np.float32),
        "weather_data_test":  weather_test.astype(np.float32),
        "output_data_test":   outputs_test.astype(np.float32),
        "doy_train":          doy_train.astype(np.float32),
        "doy_val":            doy_val.astype(np.float32),
        "doy_test":           doy_test.astype(np.float32),
        "params_data":        params.astype(np.float32),
        "norms":              norms,
        "bathymetry":         bathymetry.astype(np.float32),
    }


load_data()