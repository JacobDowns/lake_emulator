import math
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# =========================================================
# Data loading 
# =========================================================
def load_data(
    weather_path='data/parsed_data/weather_data.npy',
    output_path ='data/parsed_data/output_data.npy',
    params_path ='data/parsed_data/parameter_data.npy',
    split_years = [2018, 2020, 2025],  # [train_end, val_end, max_year]
    normalize = True
):
    """
    Returns dict of normalized splits:
      weather_data_{train,val,test}: (T_split, Du=7)  # YEAR/DOY removed
      output_data_{train,val,test}:  (N, T_split, Dz=8)
      params_data:                   (N, P)
    Normalization is fit on TRAIN only (feature-wise for weather, per-depth for outputs, per-feature for params).
    """
    weather = np.load(weather_path)   # (T, 2 + Du) : [YEAR, DOY, drivers...]
    outputs = np.load(output_path)    # (N, T, Dz)
    params  = np.load(params_path)    # (N, P)

    year = weather[:, 0]
    # doy  = weather[:, 1]  # not used in this loader
    drivers = weather[:, 2:]  # (T, Du)

    # splits
    idx_train = (year <= split_years[0])
    idx_val   = (year > split_years[0]) & (year <= split_years[1])
    idx_test  = (year > split_years[1])

    weather_train = drivers[idx_train, :]
    weather_val   = drivers[idx_val, :]
    weather_test  = drivers[idx_test, :]

    outputs_train = outputs[:, idx_train, :]
    outputs_val   = outputs[:, idx_val, :]
    outputs_test  = outputs[:, idx_test, :]

    if normalize:
        eps = 1e-6
        w_mean = weather_train.mean(axis=0, keepdims=True)
        w_std  = weather_train.std(axis=0, keepdims=True) + eps

        o_mean = outputs_train.mean(axis=(0,1), keepdims=True)   # (1,1,Dz)
        o_std  = outputs_train.std(axis=(0,1), keepdims=True) + eps

        p_mean = params.mean(axis=0, keepdims=True)              # (1,P)
        p_std  = params.std(axis=0, keepdims=True) + eps

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
            "output_mean":  o_mean.astype(np.float32).squeeze(0),  # (1,Dz)
            "output_std":   o_std.astype(np.float32).squeeze(0),   # (1,Dz)
            "params_mean":  p_mean.astype(np.float32),
            "params_std":   p_std.astype(np.float32),
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
        "params_data":        params.astype(np.float32),
        "norms":              norms
    }

# =========================================================
# Dataset: windowing over arrays returned by load_data
# =========================================================
class LakeWindowDataset(Dataset):
    """
    Given:
      weather: (T, Du)
      outputs: (N, T, Dz)
      params : (N, P)

    For window length W_x and horizon W_y (<= W_x):
      x_win: (W_x, Du)
      p_vec: (P,)
      y    : (W_y, Dz)  (we’ll squeeze to (Dz) in the loop if W_y==1)
    """
    def __init__(self, weather: np.ndarray, outputs: np.ndarray, params: np.ndarray,
                 W_x: int = 90, W_y: int = 1, trial_ids=None):
        super().__init__()
        assert 1 <= W_y <= W_x, "Require 1 <= W_y <= W_x"
        assert weather.ndim == 2 and outputs.ndim == 3 and params.ndim == 2
        T, Du = weather.shape
        N, T2, Dz = outputs.shape
        assert T == T2 and N == params.shape[0]
        self.W_x, self.W_y = W_x, W_y
        self.Du, self.Dz = Du, Dz
        self.params = torch.from_numpy(params).float()
        self.outputs = outputs  # keep numpy for cheap slicing
        if trial_ids is None:
            self.trial_ids = np.arange(N, dtype=int)
        else:
            self.trial_ids = np.array(trial_ids, dtype=int)

        # Precompute weather windows (T - W_x + 1, W_x, Du)
        if T < W_x:
            raise ValueError(f"Not enough timesteps T={T} for W_x={W_x}")
        W = T - W_x + 1
        self.W = W
        wv = np.lib.stride_tricks.sliding_window_view(weather, window_shape=(W_x, Du), axis=(0, 1))
        self.weather_wins = torch.from_numpy(wv[:, 0, :, :]).float()  # (W, W_x, Du)

        # index over all (n, w)
        self.index = [(i, w) for i in range(len(self.trial_ids)) for w in range(W)]

    def __len__(self): return len(self.index)

    def __getitem__(self, k: int):
        i_idx, w = self.index[k]
        n = self.trial_ids[i_idx]
        x_win = self.weather_wins[w]                         # (W_x, Du)
        p_vec = self.params[n]                               # (P,)
        start = w + self.W_x - self.W_y
        end   = w + self.W_x
        y_np  = self.outputs[n, start:end, :]                # (W_y, Dz)
        y = torch.from_numpy(y_np).float()
        return x_win, p_vec, y