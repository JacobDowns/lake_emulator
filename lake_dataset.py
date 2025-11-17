import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class WindowNextDayDataset(Dataset):
    """
    Returns:
      drivers_win: (L, Du)
      params_vec : (P,)
      target     : (Dz,)
    Assumes arrays:
      weather: (T, Du)     # unnormalized here
      params : (N, P)
      outputs: (N, T, Dz)
    Dates TSV:
      aligned_dates.tsv with columns YEAR, DOY for each t in [0..T-1]
    """
    def __init__(self, 
                 weather, 
                 params, 
                 outputs, 
                 dates_tsv,
                 L=90, 
                 lead=1, 
                 t_indices=None,
                 x_mean=None, 
                 x_std=None, 
                 p_mean=None, 
                 p_std=None, 
                 fit_norm=True
        ):
        
        super().__init__()
        self.weather_raw = weather.astype(np.float32)
        self.params_raw  = params.astype(np.float32)
        self.outputs     = outputs.astype(np.float32)
        self.T, self.Du  = weather.shape
        self.N, T2, self.Dz = outputs.shape
        assert self.T == T2, "Time mismatch weather vs outputs"

        d = pd.read_csv(dates_tsv, sep="\t")
        assert len(d) == self.T and {"YEAR","DOY"} <= set(d.columns)
        self.year = d["YEAR"].to_numpy(int)

        self.L = int(L)
        self.lead = int(lead)
        # valid window end times
        t_lo = self.L - 1
        t_hi = self.T - 1 - self.lead

        if t_indices is None:
            self.t_idx = np.arange(t_lo, t_hi + 1, dtype=int)
        else:
            self.t_idx = np.array([t for t in t_indices if t_lo <= t <= t_hi], dtype=int)

        # normalization
        if fit_norm:
            x_train = self.weather_raw[:t_hi+1]  # we’ll subset to train outside before calling
            x_mean = x_train.mean(axis=0, keepdims=True)
            x_std  = x_train.std(axis=0, keepdims=True) + 1e-6
            p_mean = self.params_raw.mean(axis=0, keepdims=True)
            p_std  = self.params_raw.std(axis=0, keepdims=True) + 1e-6
            self.norms = (x_mean, x_std, p_mean, p_std)
        else:
            assert x_mean is not None and x_std is not None and p_mean is not None and p_std is not None
            self.norms = (x_mean, x_std, p_mean, p_std)

        x_mean, x_std, p_mean, p_std = self.norms
        self.weather = (self.weather_raw - x_mean) / x_std
        self.params  = (self.params_raw  - p_mean)  / p_std

        # build (trial, t) index list over provided t_idx
        self.index = [(i, t) for i in range(self.N) for t in self.t_idx]

    def __len__(self): return len(self.index)

    def __getitem__(self, k):
        i, t = self.index[k]
        win = self.weather[t - self.L + 1 : t + 1, :]      # (L, Du)
        p   = self.params[i, :]                             # (P,)
        y   = self.outputs[i, t + self.lead, :]             # (Dz,)
        return torch.from_numpy(win), torch.from_numpy(p), torch.from_numpy(y)
