from torch.utils.data import Dataset
import numpy as np
import torch


# =========================================================
# Dataset: windowing over arrays returned by load_data
# =========================================================
class LakeWindowDataset(Dataset):
    """2
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
        return_ids: bool = False,
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
    

# =========================================================
# Data loading (train-only normalization; strip YEAR, keep DOY)
# =========================================================
def load_data(
    weather_path='data/parsed_data/weather_data.npy',
    output_path ='data/parsed_data/output_data.npy',
    params_path ='data/parsed_data/parameter_data.npy',
    split_years = [2018, 2020, 2025],  # [train_end, val_end, max]
    normalize = True
):
    """
    Returns dict of splits with train-only normalization applied:
      weather_data_* : (T_split, Du)  # YEAR/DOY removed
      output_data_*  : (N, T_split, Dz)
      doy_*          : (T_split,)       # day-of-year retained for features
      params_data    : (N, P)
      norms          : dict with means/stds for inverse-transform

    YEAR used only for splitting.
    DOY is *not* included in weather_* but is returned separately so
    you can build seasonal features (e.g. sin/cos(doy)) if desired.
    """
    weather = np.load(weather_path)   # (T, 2 + Du) : [YEAR, DOY, drivers...]
    outputs = np.load(output_path)    # (N, T, Dz)
    params  = np.load(params_path)    # (N, P)

    year = weather[:, 0]
    doy  = weather[:, 1].astype(np.float32)
    drivers = weather[:, 2:]          # (T, Du)

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

    doy_train = doy[idx_train]
    doy_val   = doy[idx_val]
    doy_test  = doy[idx_test]

    if normalize:
        eps = 1e-6
        w_mean = weather_train.mean(axis=0, keepdims=True)         # (1, Du)
        w_std  = weather_train.std(axis=0, keepdims=True) + eps    # (1, Du)

        o_mean = outputs_train.mean(axis=(0,1), keepdims=True)     # (1, 1, Dz)
        o_std  = outputs_train.std(axis=(0,1), keepdims=True) + eps# (1, 1, Dz)

        p_mean = params.mean(axis=0, keepdims=True)                # (1, P)
        p_std  = params.std(axis=0, keepdims=True) + eps           # (1, P)

        weather_train = (weather_train - w_mean) / w_std
        weather_val   = (weather_val   - w_mean) / w_std
        weather_test  = (weather_test  - w_mean) / w_std

        outputs_train = (outputs_train - o_mean) / o_std
        outputs_val   = (outputs_val   - o_mean) / o_std
        outputs_test  = (outputs_test  - o_mean) / o_std

        params = (params - p_mean) / p_std

        norms = {
            "weather_mean": w_mean.astype(np.float32),   # (1, Du)
            "weather_std":  w_std.astype(np.float32),    # (1, Du)
            "output_mean":  o_mean.astype(np.float32),   # (1, 1, Dz)
            "output_std":   o_std.astype(np.float32),    # (1, 1, Dz)
            "params_mean":  p_mean.astype(np.float32),   # (1, P)
            "params_std":   p_std.astype(np.float32),    # (1, P)
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
        "norms":              norms
    }


# =========================================================
# Extra feature builders (seasonality etc.)
# =========================================================
def build_seasonal_features(doy: np.ndarray) -> np.ndarray:
    """
    Build a cyclic encoding of day-of-year.
    doy: (T,) with values in [1,365] or [0,364]
    Returns: (T, 2) array with sin/cos(2π * doy / 365).
    """
    angle = 2.0 * np.pi * (doy / 365.0)
    sin_doy = np.sin(angle)
    cos_doy = np.cos(angle)
    return np.stack([sin_doy, cos_doy], axis=-1).astype(np.float32)  # (T, 2)

