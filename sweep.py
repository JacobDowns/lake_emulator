#!/usr/bin/env python3
"""
sweep_compare_cells.py

Compares GRU/LSTM/RNN at matched parameter budgets (low/med/high),
using the refactored data layout:

- weather_data.npy: [YEAR, DOY, drivers...]
- parameter_data.npy: (N, P)
- output_data.npy: (N, T, Dz)

Adds seasonal Fourier features from YEAR/DOY (excluded from normalization),
strict year split (default TEST 2020–2025), and MLflow logging.

Example:
  export MLFLOW_TRACKING_URI="file:./mlruns"
  export MLFLOW_EXPERIMENT_NAME="Lake-Compare"
  python sweep_compare_cells.py --budgets 80k,160k,320k --cells gru,lstm,rnn \
      --Ls 60,90 --lrs 1e-3,5e-4 --epochs 30 --seasonal_harmonics 1,2
"""
import os, math, argparse, itertools, socket, subprocess
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import mlflow
import mlflow.pytorch

from lake_models import LakeProfileSeq


# ----------------------------
# Data helpers (mirror lake_data.py)
# ----------------------------
def split_meta_and_drivers(weather_with_meta: np.ndarray):
    meta = weather_with_meta[:, :2].astype(np.int32)       # YEAR, DOY
    drivers = weather_with_meta[:, 2:].astype(np.float32)  # raw drivers
    return meta, drivers

def build_fourier_from_year_doy(year: np.ndarray, doy: np.ndarray, harmonics=(1, 2)) -> np.ndarray:
    year = year.astype(int)
    doy = doy.astype(int)
    year_len = np.where(((year % 4 == 0) & (year % 100 != 0)) | (year % 400 == 0), 366, 365).astype(float)
    cols = []
    for k in harmonics:
        theta = 2 * np.pi * k * (doy.astype(float) / year_len)
        cols += [np.sin(theta), np.cos(theta)]
    return np.column_stack(cols).astype(np.float32) if cols else np.zeros((len(year), 0), dtype=np.float32)

def strict_test_split_by_year_from_meta(meta_year: np.ndarray, L: int, lead: int,
                                        test_years: tuple[int,int], val_frac_pretest: float = 0.2):
    T = len(meta_year)
    t_lo = L - 1
    t_hi = T - 1 - lead
    valid = np.arange(t_lo, t_hi + 1, dtype=int)
    mask_test = (meta_year[valid + lead] >= test_years[0]) & (meta_year[valid + lead] <= test_years[1])
    t_test = valid[mask_test]
    remain = valid[~mask_test]
    n_val = max(1, int(len(remain) * val_frac_pretest))
    t_val = remain[-n_val:] if n_val > 0 else np.array([], dtype=int)
    t_train = remain[:-n_val] if n_val > 0 else remain
    return t_train, t_val, t_test

def fit_norms_on_train(drivers_with_fourier: np.ndarray, params: np.ndarray, t_train: np.ndarray, L: int,
                       exclude_tail_cols: int = 0):
    rows = np.unique(np.concatenate([np.arange(t - L + 1, t + 1) for t in t_train]))
    Du_total = drivers_with_fourier.shape[1]
    Du_core = Du_total - int(exclude_tail_cols)
    core = drivers_with_fourier[:, :Du_core]
    x_mean = core[rows].mean(axis=0, keepdims=True)
    x_std  = core[rows].std(axis=0, keepdims=True) + 1e-6
    p_mean = params.mean(axis=0, keepdims=True)
    p_std  = params.std(axis=0, keepdims=True) + 1e-6
    return x_mean.astype(np.float32), x_std.astype(np.float32), p_mean.astype(np.float32), p_std.astype(np.float32), Du_core

def apply_norms(drivers_with_fourier, params, x_mean, x_std, p_mean, p_std, exclude_tail_cols: int = 0):
    Du_total = drivers_with_fourier.shape[1]
    Du_core = Du_total - int(exclude_tail_cols)
    core_std = (drivers_with_fourier[:, :Du_core] - x_mean) / x_std
    if exclude_tail_cols > 0:
        tail = drivers_with_fourier[:, Du_core:]
        drivers_std = np.concatenate([core_std, tail], axis=1)
    else:
        drivers_std = core_std
    params_std = (params - p_mean) / p_std
    return drivers_std.astype(np.float32), params_std.astype(np.float32)

class WindowNextDayDataset(Dataset):
    def __init__(self, drivers_std, params_std, outputs, L=90, lead=1, t_indices=None, trials=None):
        self.drivers = drivers_std.astype(np.float32)  # (T, Du)
        self.params  = params_std.astype(np.float32)   # (N, P)
        self.outputs = outputs.astype(np.float32)      # (N, T, Dz)
        self.T, self.Du = self.drivers.shape
        self.N, T2, self.Dz = self.outputs.shape
        assert self.T == T2
        self.L, self.lead = int(L), int(lead)
        t_lo = self.L - 1; t_hi = self.T - 1 - self.lead
        valid = np.arange(t_lo, t_hi + 1, dtype=int)
        self.t_idx = valid if t_indices is None else np.array([t for t in t_indices if t_lo <= t <= t_hi], dtype=int)
        self.trials = list(range(self.N)) if trials is None else list(trials)
        self.index = [(i, t) for i in self.trials for t in self.t_idx]

    def __len__(self): return len(self.index)
    def __getitem__(self, k):
        i, t = self.index[k]
        x_win = self.drivers[t - self.L + 1 : t + 1, :]
        p_vec = self.params[i, :]
        y     = self.outputs[i, t + self.lead, :]
        return torch.from_numpy(x_win), torch.from_numpy(p_vec), torch.from_numpy(y)


# ----------------------------
# Metrics / logging helpers
# ----------------------------
def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

def per_depth_rmse(loader, model, device) -> float:
    sse = None; n = 0
    with torch.no_grad():
        for x, p, y in loader:
            x = x.float().to(device); p = p.float().to(device); y = y.float().to(device)
            pred = model(x, p)
            err2 = (pred - y) ** 2
            sse = err2.sum(dim=0) if sse is None else sse + err2.sum(dim=0)
            n += y.shape[0]
    rmse_depth = torch.sqrt(sse / max(1, n))
    return float(rmse_depth.mean().item())

def log_git_tags_safe():
    try:
        commit = subprocess.check_output(["git","rev-parse","HEAD"], text=True).strip()
        dirty  = subprocess.check_output(["git","status","--porcelain"], text=True).strip()
        mlflow.set_tag("git.commit", commit)
        mlflow.set_tag("git.dirty", "1" if dirty else "0")
    except Exception:
        pass


# ----------------------------
# Hidden-size matcher (param budget)
# ----------------------------
def find_hidden_for_budget(cell_type: str, budget_params: int, Du: int, P: int, Dz: int,
                           H_min=16, H_max=768, num_layers=1) -> int:
    best_H, best_diff = None, 1e18
    for step in [16, 8, 4, 2, 1]:
        start = H_min if best_H is None else max(H_min, best_H - 7)
        end   = H_max if best_H is None else min(H_max, best_H + 7)
        for H in range(start, end + 1, step):
            m = LakeProfileSeq(driver_dim=Du, param_dim=P, hidden=H, depth_out=Dz,
                               cell_type=cell_type, num_layers=num_layers)
            diff = abs(count_params(m) - budget_params)
            if diff < best_diff:
                best_diff, best_H = diff, H
    return int(best_H)


# ----------------------------
# Train once
# ----------------------------
@dataclass
class TrainConfig:
    L: int
    lr: float
    epochs: int
    batch_size: int
    hidden: int
    cell_type: str
    num_layers: int = 1
    smooth_lambda: float = 0.0
    lead: int = 1
    val_frac_pretest: float = 0.2
    test_years: Tuple[int,int] = (2020, 2025)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    artifacts_dir: str = "artifacts_compare"
    tag_bucket: str = ""  # "low"/"med"/"high"
    harmonics: Tuple[int,...] = (1, 2)

def train_once(cfg: TrainConfig,
               weather_with_meta: np.ndarray, params: np.ndarray, outputs: np.ndarray) -> Dict:
    os.makedirs(cfg.artifacts_dir, exist_ok=True)

    meta, drivers = split_meta_and_drivers(weather_with_meta)
    year, doy = meta[:,0], meta[:,1]
    T, Du_base = drivers.shape
    N, T2, Dz = outputs.shape
    P = params.shape[1]
    assert T == T2

    # split
    t_train, t_val, t_test = strict_test_split_by_year_from_meta(
        meta_year=year, L=cfg.L, lead=cfg.lead, test_years=cfg.test_years, val_frac_pretest=cfg.val_frac_pretest
    )

    # features
    if len(cfg.harmonics) > 0:
        fourier = build_fourier_from_year_doy(year, doy, harmonics=cfg.harmonics)  # (T, 2K)
        drivers_all = np.concatenate([drivers, fourier], axis=1)
        n_fourier = fourier.shape[1]
    else:
        drivers_all = drivers; n_fourier = 0

    Du_total = drivers_all.shape[1]

    # norms
    x_mean, x_std, p_mean, p_std, _ = fit_norms_on_train(drivers_all, params, t_train, cfg.L, exclude_tail_cols=n_fourier)
    drivers_std, params_std = apply_norms(drivers_all, params, x_mean, x_std, p_mean, p_std, exclude_tail_cols=n_fourier)

    # datasets
    train_ds = WindowNextDayDataset(drivers_std, params_std, outputs, L=cfg.L, lead=cfg.lead, t_indices=t_train)
    val_ds   = WindowNextDayDataset(drivers_std, params_std, outputs, L=cfg.L, lead=cfg.lead, t_indices=t_val)
    test_ds  = WindowNextDayDataset(drivers_std, params_std, outputs, L=cfg.L, lead=cfg.lead, t_indices=t_test)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=max(1, cfg.batch_size//2), shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=max(1, cfg.batch_size//2), shuffle=False, num_workers=0)

    # model/optim
    model = LakeProfileSeq(driver_dim=Du_total, param_dim=P, depth_out=Dz,
                           hidden=cfg.hidden, cell_type=cfg.cell_type,
                           num_layers=cfg.num_layers).to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    mse = nn.MSELoss()

    num_params = count_params(model)
    run_name = f"{cfg.cell_type.upper()}_H{cfg.hidden}_L{cfg.L}_lr{cfg.lr:g}_TEST{cfg.test_years[0]}-{cfg.test_years[1]}"
    if n_fourier > 0:
        run_name += f"_fourierK{len(cfg.harmonics)}"

    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "Lake-Compare"))
    with mlflow.start_run(run_name=run_name):
        log_git_tags_safe()
        mlflow.set_tag("host", socket.gethostname())
        mlflow.set_tag("budget_bucket", cfg.tag_bucket or "")
        mlflow.set_tag("split_mode", "strict_years")
        mlflow.set_tag("fourier_enabled", str(n_fourier > 0))
        mlflow.log_params({
            "cell_type": cfg.cell_type, "hidden": cfg.hidden, "num_layers": cfg.num_layers,
            "L": cfg.L, "lead": cfg.lead, "epochs": cfg.epochs, "batch_size": cfg.batch_size,
            "lr": cfg.lr, "Du_base": Du_base, "Du_total": Du_total, "P": P, "Dz": Dz,
            "smooth_lambda": cfg.smooth_lambda,
            "train_len": len(train_ds), "val_len": len(val_ds), "test_len": len(test_ds),
            "val_frac_pretest": cfg.val_frac_pretest,
            "fourier_harmonics": ",".join(map(str, cfg.harmonics)) if n_fourier > 0 else "",
        })
        mlflow.log_metric("num_params", int(num_params))

        best_val = float("inf")
        ckpt_path = os.path.join(cfg.artifacts_dir, f"best_{run_name}.pt")

        for ep in range(1, cfg.epochs + 1):
            # train
            model.train()
            train_sse, train_n = 0.0, 0
            for x, p, y in train_loader:
                x = x.float().to(cfg.device); p = p.float().to(cfg.device); y = y.float().to(cfg.device)
                pred = model(x, p)
                loss = mse(pred, y)
                if cfg.smooth_lambda > 0 and Dz > 1:
                    loss = loss + cfg.smooth_lambda * (pred[:,1:] - pred[:,:-1]).abs().mean()
                opt.zero_grad(); loss.backward(); opt.step()
                train_sse += (pred - y).pow(2).sum().item(); train_n += y.numel()
            train_mse = train_sse / max(1, train_n)

            # val
            model.eval()
            val_sse, val_n = 0.0, 0
            with torch.no_grad():
                for x, p, y in val_loader:
                    pred = model(x.float().to(cfg.device), p.float().to(cfg.device))
                    val_sse += (pred - y.float().to(cfg.device)).pow(2).sum().item()
                    val_n   += y.numel()
            val_mse = val_sse / max(1, val_n)
            val_rmse = per_depth_rmse(val_loader, model, cfg.device)

            mlflow.log_metrics({"train_mse": train_mse, "val_mse": val_mse, "val_rmse_mean_depths": val_rmse}, step=ep)

            if val_mse < best_val:
                best_val = val_mse
                torch.save({
                    "model_state": model.state_dict(),
                    "Du_base": Du_base, "Du_total": Du_total, "P": P, "Dz": Dz,
                    "norms": {
                        "driver_mean": x_mean, "driver_std": x_std,
                        "param_mean": p_mean, "param_std": p_std,
                        "n_fourier": n_fourier
                    },
                    "config": cfg.__dict__,
                }, ckpt_path)
                mlflow.log_artifact(ckpt_path, artifact_path="checkpoints")

        # test
        model.eval()
        test_sse, test_n = 0.0, 0
        with torch.no_grad():
            for x, p, y in test_loader:
                pred = model(x.float().to(cfg.device), p.float().to(cfg.device))
                test_sse += (pred - y.float().to(cfg.device)).pow(2).sum().item()
                test_n   += y.numel()
        test_mse = test_sse / max(1, test_n)
        mlflow.log_metric("test_mse", float(test_mse))
        mlflow.log_metric("best_val_mse", float(best_val))

        return {"num_params": int(num_params), "best_val_mse": float(best_val), "test_mse": float(test_mse), "run_name": run_name}


# ----------------------------
# CLI & sweep
# ----------------------------
def parse_budgets(s: str) -> List[int]:
    out = []
    for tok in s.split(","):
        tok = tok.strip().lower()
        mult = 1
        if tok.endswith("k"): mult, tok = 1000, tok[:-1]
        elif tok.endswith("m"): mult, tok = 1_000_000, tok[:-1]
        out.append(int(float(tok) * mult))
    return out

def main():
    ap = argparse.ArgumentParser()
    # Data paths
    ap.add_argument("--weather_npy", default="data/parsed_data/weather_data.npy")
    ap.add_argument("--params_npy",  default="data/parsed_data/parameter_data.npy")
    ap.add_argument("--outputs_npy", default="data/parsed_data/output_data.npy")

    # Seasonal Fourier
    ap.add_argument("--seasonal_harmonics", default="1,2", help="'' to disable, else comma list e.g. 1,2,3")

    # Sweep knobs
    ap.add_argument("--cells", default="gru,lstm,rnn")
    ap.add_argument("--budgets", default="50k,100k,200k")
    ap.add_argument("--Ls", default="90")
    ap.add_argument("--lrs", default="0.001,0.0005")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lead", type=int, default=1)
    ap.add_argument("--val_frac_pretest", type=float, default=0.2)
    ap.add_argument("--test_years", nargs=2, type=int, default=[2020, 2025])
    ap.add_argument("--num_layers", type=int, default=1)
    ap.add_argument("--smooth_lambda", type=float, default=0.0)
    ap.add_argument("--artifacts", default="artifacts_compare")
    args = ap.parse_args()

    # Load arrays
    weather_with_meta = np.load(args.weather_npy)     # (T, 2+Du)
    params  = np.load(args.params_npy)                # (N, P)
    outputs = np.load(args.outputs_npy)               # (N, T, Dz)
    meta, drivers = split_meta_and_drivers(weather_with_meta)
    T, Du_base = drivers.shape
    N, T2, Dz = outputs.shape
    P = params.shape[1]
    assert T == T2

    cells   = [c.strip().lower() for c in args.cells.split(",") if c.strip()]
    budgets = parse_budgets(args.budgets)
    Ls      = [int(x) for x in args.Ls.split(",")]
    lrs     = [float(x) for x in args.lrs.split(",")]
    harm    = tuple(int(h) for h in args.seasonal_harmonics.split(",") if h.strip()) if args.seasonal_harmonics.strip() != "" else tuple()

    # Precompute hidden sizes per (cell_type, budget) using Du_total with Fourier (worst-case K)
    # We don't yet know K at this point; approximate with provided harmonics.
    if len(harm) > 0:
        year, doy = meta[:,0], meta[:,1]
        fourier = build_fourier_from_year_doy(year, doy, harmonics=harm)  # (T, 2K)
        Du_total_for_budget = Du_base + fourier.shape[1]
    else:
        Du_total_for_budget = Du_base

    hidden_for = {}
    print("Selecting hidden sizes for budgets (includes head params):")
    for cell in cells:
        for b in budgets:
            H = find_hidden_for_budget(cell, b, Du=Du_total_for_budget, P=P, Dz=Dz, H_min=16, H_max=768, num_layers=args.num_layers)
            cnt = count_params(LakeProfileSeq(driver_dim=Du_total_for_budget, param_dim=P, hidden=H, depth_out=Dz,
                                              cell_type=cell, num_layers=args.num_layers))
            hidden_for[(cell, b)] = H
            print(f"  {cell.upper()} ~{b:,} params -> H={H} (actual {cnt:,} using Du={Du_total_for_budget})")

    # Run sweep
    os.makedirs(args.artifacts, exist_ok=True)
    for L, lr, b in itertools.product(Ls, lrs, budgets):
        for cell in cells:
            H = hidden_for[(cell, b)]
            cfg = TrainConfig(
                L=L, lr=lr, epochs=args.epochs, batch_size=args.batch_size,
                hidden=H, cell_type=cell, num_layers=args.num_layers,
                smooth_lambda=args.smooth_lambda, lead=args.lead,
                val_frac_pretest=args.val_frac_pretest, test_years=tuple(args.test_years),
                artifacts_dir=args.artifacts, harmonics=harm,
                tag_bucket={budgets[0]:"low", budgets[1]:"med" if len(budgets)>1 else "mid",
                            budgets[2]:"high" if len(budgets)>2 else "hi"}.get(b, f"{b}")
            )
            _ = train_once(cfg, weather_with_meta, params, outputs)


if __name__ == "__main__":
    main()
