#!/usr/bin/env python3
import os, json, argparse, subprocess
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List

# non-interactive backend for HPC/headless
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- your models ----
from models import ModelMLP, ModelRNN, ModelCNN1D

# ---- MLflow ----
import mlflow
import mlflow.pytorch


# =========================================================
# Data loading (train-only normalization; strip YEAR/DOY)
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
      weather_data_* : (T_split, Du=7)  # YEAR/DOY removed
      output_data_*  : (N, T_split, Dz=8)
      params_data    : (N, P)
      norms          : dict with means/stds for inverse-transform

    YEAR used only for splitting. DOY is ignored here (you can add cyc features upstream if desired).
    """
    weather = np.load(weather_path)   # (T, 2 + Du) : [YEAR, DOY, drivers...]
    outputs = np.load(output_path)    # (N, T, Dz)
    params  = np.load(params_path)    # (N, P)

    year = weather[:, 0]
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
      y    : (W_y, Dz)  (we squeeze y to (Dz) in the loop if W_y==1)
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


# =========================================================
# Training / eval / plotting
# =========================================================
def train_one_epoch(loader, model, opt, device, smooth_lambda=0.0):
    model.train()
    mse = nn.MSELoss()
    total, count = 0.0, 0
    for x, p, y in loader:
        x = x.to(device).float()    # (B, W_x, Du)
        p = p.to(device).float()    # (B, P)
        y = y.to(device).float()    # (B, W_y, Dz) or (B, Dz) later

        if y.dim() == 3 and y.size(1) == 1:
            y = y[:, 0, :]          # (B, Dz)

        pred = model(x, p)          # (B, Dz) for Wy=1; else (B, W_y, Dz)
        loss = mse(pred, y)

        if smooth_lambda > 0.0 and pred.dim() == 2 and pred.size(1) > 1:
            loss = loss + smooth_lambda * (pred[:, 1:] - pred[:, :-1]).abs().mean()

        opt.zero_grad()
        loss.backward()
        # Optional: gradient clipping for RNN/LSTM stability
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        total += loss.item() * x.size(0)
        count += x.size(0)
    return total / max(1, count)

@torch.no_grad()
def eval_epoch(loader, model, device):
    model.eval()
    mse = nn.MSELoss()
    total, count = 0.0, 0
    for x, p, y in loader:
        x = x.to(device).float()
        p = p.to(device).float()
        y = y.to(device).float()
        if y.dim() == 3 and y.size(1) == 1:
            y = y[:, 0, :]
        pred = model(x, p)
        loss = mse(pred, y)
        total += loss.item() * x.size(0)
        count += x.size(0)
    return total / max(1, count)

@torch.no_grad()
def per_depth_rmse(loader, model, device, Dz):
    model.eval()
    sse = torch.zeros(Dz, device=device)
    n   = 0
    for x, p, y in loader:
        x = x.to(device).float()
        p = p.to(device).float()
        y = y.to(device).float()
        if y.dim() == 3 and y.size(1) == 1:
            y = y[:, 0, :]
        pred = model(x, p)
        sse += ((pred - y) ** 2).sum(dim=0)
        n   += y.size(0)
    return torch.sqrt(sse / max(1, n)).detach().cpu().numpy()

def _norms_output_arrays(norms):
    """
    Normalize shapes: return (1, Dz) arrays for output mean/std regardless of storage convention.
    Accepts (Dz,), (1, Dz) or (1,1,Dz) and returns (1, Dz).
    """
    mu = norms["output_mean"]
    sd = norms["output_std"]
    mu = np.array(mu)
    sd = np.array(sd)
    if mu.ndim == 3:   # (1,1,Dz)
        mu = mu.reshape(1, -1)
        sd = sd.reshape(1, -1)
    elif mu.ndim == 1: # (Dz,)
        mu = mu.reshape(1, -1)
        sd = sd.reshape(1, -1)
    # if (1, Dz) already, keep
    return mu.astype(np.float32), sd.astype(np.float32)

def denorm_outputs(y_norm: np.ndarray, norms: dict) -> np.ndarray:
    """
    y_norm: (B, Dz) or (B, Wy, Dz). Returns same shape in real units.
    """
    if norms is None:
        return y_norm
    mu, sd = _norms_output_arrays(norms)  # (1, Dz)
    if y_norm.ndim == 2:
        return y_norm * sd + mu
    elif y_norm.ndim == 3:
        return y_norm * sd[None, :, :] + mu[None, :, :]
    else:
        return y_norm

@torch.no_grad()
def make_val_depth_profile_plot(model, val_loader, device, norms, Dz, epoch,
                                outdir="plots", max_samples=4):
    """
    Takes the first batch from val_loader, computes predictions,
    de-normalizes to real units, and logs a depth profile comparison plot
    for up to max_samples samples. Returns the saved PNG path or None.
    """
    model.eval()
    os.makedirs(outdir, exist_ok=True)

    try:
        x, p, y = next(iter(val_loader))
    except StopIteration:
        return None

    x = x.to(device).float()
    p = p.to(device).float()
    y = y.to(device).float()

    # Targets → (B, Dz) for plotting (use last step if multi-step)
    if y.dim() == 3:
        if y.size(1) == 1:
            y = y[:, 0, :]
        else:
            y = y[:, -1, :]

    pred = model(x, p)
    if pred.dim() == 3:
        if pred.size(1) == 1:
            pred = pred[:, 0, :]
        else:
            pred = pred[:, -1, :]

    y_np = y.detach().cpu().numpy()
    pr_np = pred.detach().cpu().numpy()

    # De-normalize back to °C
    y_den  = denorm_outputs(y_np, norms)
    pr_den = denorm_outputs(pr_np, norms)

    # Depth axis (bin centers)
    depth = np.arange(Dz) + 0.5

    K = min(max_samples, y_den.shape[0])
    plt.figure(figsize=(6.5, 4.5))
    for i in range(K):
        plt.plot(y_den[i],  depth, linestyle='-',  label="True" if i == 0 else None)
        plt.plot(pr_den[i], depth, linestyle='--', label="Pred" if i == 0 else None)

    plt.gca().invert_yaxis()
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Depth (m)")
    plt.title(f"Validation profiles (epoch {epoch})")
    plt.legend(loc="best")
    fname = os.path.join(outdir, f"val_profiles_epoch_{epoch:03d}.png")
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()
    return fname


# =========================================================
# Utilities
# =========================================================
def parse_hidden_list(s: str) -> List[int]:
    return [int(x) for x in s.split(",") if x.strip()]

def log_git_info_as_tags():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty  = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        mlflow.set_tag("git.commit", commit)
        mlflow.set_tag("git.dirty", "1" if dirty else "0")
    except Exception:
        pass


# =========================================================
# Main
# =========================================================
def main():
    ap = argparse.ArgumentParser()
    # data & split
    ap.add_argument("--weather", default="data/parsed_data/weather_data.npy")
    ap.add_argument("--outputs", default="data/parsed_data/output_data.npy")
    ap.add_argument("--params",  default="data/parsed_data/parameter_data.npy")
    ap.add_argument("--split_years", type=int, nargs=3, default=[2018, 2020, 2025],
                    help="YYYY_train_end YYYY_val_end YYYY_max")
    ap.add_argument("--normalize", type=int, default=1)

    # windowing
    ap.add_argument("--Wx", type=int, default=90, help="window length (days)")
    ap.add_argument("--Wy", type=int, default=1, help="prediction horizon (days)")

    # model choice & hypers
    ap.add_argument("--model", choices=["mlp","rnn","cnn"], default="rnn")
    ap.add_argument("--head_hidden", type=str, default="256,256")
    ap.add_argument("--head_dropout", type=float, default=0.0)

    # RNN specifics
    ap.add_argument("--cell", choices=["gru","lstm","rnn"], default="gru")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--rnn_dropout", type=float, default=0.0)

    # CNN specifics
    ap.add_argument("--cnn_layers", type=int, default=2)
    ap.add_argument("--cnn_channels", type=int, default=128)
    ap.add_argument("--cnn_kernel", type=int, default=5)

    # training
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=7.5e-3)
    ap.add_argument("--smooth_lambda", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)

    # infra
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--artifacts", default="artifacts_run")
    ap.add_argument("--experiment", default=None)
    ap.add_argument("--log_model_every_improvement", action="store_true")
    args = ap.parse_args()

    # seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.artifacts, exist_ok=True)

    # ---------- load & normalize
    data = load_data(
        weather_path=args.weather,
        output_path=args.outputs,
        params_path=args.params,
        split_years=args.split_years,
        normalize=bool(args.normalize)
    )
    w_tr = data["weather_data_train"];  o_tr = data["output_data_train"]
    w_va = data["weather_data_val"];    o_va = data["output_data_val"]
    w_te = data["weather_data_test"];   o_te = data["output_data_test"]
    params = data["params_data"]
    norms  = data["norms"]

    Du = w_tr.shape[1]
    Dz = o_tr.shape[2]
    P  = params.shape[1]

    # ---------- datasets & loaders
    train_ds = LakeWindowDataset(w_tr, o_tr, params, W_x=args.Wx, W_y=args.Wy)
    val_ds   = LakeWindowDataset(w_va, o_va, params, W_x=args.Wx, W_y=args.Wy)
    test_ds  = LakeWindowDataset(w_te, o_te, params, W_x=args.Wx, W_y=args.Wy)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=max(1, args.batch_size//2), shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=max(1, args.batch_size//2), shuffle=False, num_workers=0)

    # ---------- model
    head_hidden = parse_hidden_list(args.head_hidden)
    if args.model == "mlp":
        model = ModelMLP(W_x=args.Wx, Du=Du, P=P, Dz=Dz, W_y=args.Wy,
                         mlp_hidden=head_hidden, mlp_dropout=args.head_dropout)
    elif args.model == "rnn":
        model = ModelRNN(cell=args.cell, Du=Du, P=P, Dz=Dz,
                         hidden=args.hidden, num_layers=args.num_layers,
                         rnn_dropout=args.rnn_dropout,
                         W_y=args.Wy, head_hidden=head_hidden, head_dropout=args.head_dropout)
    else:
        model = ModelCNN1D(Du=Du, P=P, Dz=Dz, W_y=args.Wy,
                           layers=args.cnn_layers, channels=args.cnn_channels, kernel=args.cnn_kernel,
                           head_hidden=head_hidden, head_dropout=args.head_dropout)
    device = args.device
    model = model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_val = float("inf")
    ckpt_path = os.path.join(args.artifacts, "best.pt")
    plots_dir = os.path.join(args.artifacts, "plots")

    # ---------- MLflow
    exp_name = args.experiment or os.getenv("MLFLOW_EXPERIMENT_NAME", "Lake-Emu")
    mlflow.set_experiment(exp_name)
    run_name = f"{args.model.upper()}_Wx{args.Wx}_Wy{args.Wy}"
    with mlflow.start_run(run_name=run_name):
        log_git_info_as_tags()
        run_key = f"{args.model}_cell{args.cell}_Wx{args.Wx}_Wy{args.Wy}_H{args.hidden}_LR{args.lr}_head{args.head_hidden}"
        mlflow.set_tag("run_key", run_key)
        mlflow.log_params({
            "model": args.model, 
            "cell": args.cell,
            "Wx": args.Wx, 
            "Wy": args.Wy,
            "Du": Du, 
            "Dz": Dz, "P": P,
            "hidden": args.hidden, 
            "num_layers": args.num_layers, 
            "rnn_dropout": args.rnn_dropout,
            "cnn_layers": args.cnn_layers, 
            "cnn_channels": args.cnn_channels, 
            "cnn_kernel": args.cnn_kernel,
            "head_hidden": args.head_hidden, 
            "head_dropout": args.head_dropout,
            "epochs": args.epochs, 
            "batch_size": args.batch_size, 
            "lr": args.lr,
            "smooth_lambda": args.smooth_lambda, 
            "device": device,
            "split_years": ",".join(map(str, args.split_years))
        })
        mlflow.log_metric("num_params", sum(p.numel() for p in model.parameters()))

        # training loop
        for ep in range(1, args.epochs + 1):
            tr_mse = train_one_epoch(train_loader, model, opt, device, smooth_lambda=args.smooth_lambda)
            va_mse = eval_epoch(val_loader, model, device)
            mlflow.log_metrics({"train_mse": tr_mse, "val_mse": va_mse}, step=ep)

            # Optional per-depth RMSE (Wy==1)
            if args.Wy == 1:
                rmse_depth = per_depth_rmse(val_loader, model, device, Dz)
                mlflow.log_metric("val_rmse_mean_depths", float(rmse_depth.mean()), step=ep)

            # Evolving validation plot (true vs pred profiles), logged each epoch
            plot_path = make_val_depth_profile_plot(
                model, val_loader, device, norms, Dz, ep,
                outdir=plots_dir, max_samples=4
            )
            if plot_path is not None:
                mlflow.log_artifact(plot_path, artifact_path="plots")

            print(f"Epoch {ep:03d} | train MSE {tr_mse:.5f} | val MSE {va_mse:.5f}")

            # save best
            if va_mse < best_val:
                best_val = va_mse
                torch.save({
                    "state_dict": model.state_dict(),
                    "config": vars(args),
                    "Du": Du, "Dz": Dz, "P": P
                }, ckpt_path)
                mlflow.log_artifact(ckpt_path, artifact_path="checkpoints")
                if args.log_model_every_improvement:
                    mlflow.pytorch.log_model(model, artifact_path="model")

        # save norms/config for inference
        if data["norms"] is not None:
            norms_path = os.path.join(args.artifacts, "norms.npz")
            np.savez(norms_path, **data["norms"])
            mlflow.log_artifact(norms_path, artifact_path="artifacts")

        cfg_path = os.path.join(args.artifacts, "config.json")
        with open(cfg_path, "w") as f:
            json.dump(vars(args), f, indent=2)
        mlflow.log_artifact(cfg_path, artifact_path="artifacts")

        # final test
        te_mse = eval_epoch(test_loader, model, device)
        mlflow.log_metric("test_mse", te_mse)
        print(f"TEST MSE: {te_mse:.6f}")
        mlflow.log_metric("best_val_mse", best_val)
        print(f"Best val MSE: {best_val:.6f}")
        print(f"Artifacts at: {mlflow.get_artifact_uri()}")


if __name__ == "__main__":
    main()
