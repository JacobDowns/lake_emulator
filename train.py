#!/usr/bin/env python3
import os, json, math, argparse, subprocess
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import mlflow
import mlflow.pytorch
import pandas as pd
from lake_dataset import WindowNextDayDataset

# =========================
# Model
# =========================
class LakeProfileGRU(nn.Module):
    def __init__(self, driver_dim=7, param_dim=4, hidden=64, depth_out=8):
        super().__init__()
        self.gru = nn.GRU(input_size=driver_dim, hidden_size=hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden + param_dim, 256),
            nn.ReLU(),
            nn.Linear(256, depth_out),
        )

    def forward(self, drivers_win, params_vec):
        h_all, _ = self.gru(drivers_win)         # (B, L, H)
        h_last = h_all[:, -1, :]                 # (B, H)
        fused = torch.cat([h_last, params_vec], dim=-1)
        return self.head(fused)                  # (B, Dz)

# =========================
# Helpers
# =========================
def build_time_splits_by_year(year_array, L, lead, test_years, val_frac_of_pretest=0.2):
    """
    Reserve strict TEST by year range (inclusive) using the TARGET day (t+lead).
    Return t_train, t_val, t_test arrays of window-end indices.
    """
    T = len(year_array)
    t_lo = L - 1
    t_hi = T - 1 - lead
    valid = np.arange(t_lo, t_hi + 1, dtype=int)

    y_target = year_array  # year at index corresponds to day t (we use at t+lead below)
    test_mask = (y_target[valid + lead] >= test_years[0]) & (y_target[valid + lead] <= test_years[1])
    t_test = valid[test_mask]

    remain = valid[~test_mask]
    n_val = max(1, int(len(remain) * val_frac_of_pretest))
    t_val = remain[-n_val:] if n_val > 0 else np.array([], dtype=int)
    t_train = remain[:-n_val] if n_val > 0 else remain
    return t_train, t_val, t_test

def log_git_info_as_tags():
    """Optional: tag the run with git commit hash & dirty flag if in a git repo."""
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        mlflow.set_tag("git.commit", commit)
        mlflow.set_tag("git.dirty", "1" if dirty else "0")
    except Exception:
        # no git or not a repo; ignore
        pass

# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weather_npy", default="data/parsed_data/weather_data.npy")
    ap.add_argument("--params_npy",  default="data/parsed_data/parameter_data.npy")
    ap.add_argument("--outputs_npy", default="data/parsed_data/output_data.npy")
    ap.add_argument("--dates_tsv",   default="data/parsed_data/aligned_dates.tsv",
                    help="TSV with YEAR, DOY per timestep (optional but required for --test_years)")

    # Model/training
    ap.add_argument("--L", type=int, default=60)
    ap.add_argument("--lead", type=int, default=1)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--smooth_lambda", type=float, default=0.0)

    # Splits
    ap.add_argument("--train_frac", type=float, default=0.8,
                    help="time split if not using --test_years")
    ap.add_argument("--test_years", nargs=2, type=int, default=None,
                    help="e.g., --test_years 2018 2025 for strict holdout by year (uses dates_tsv)")
    ap.add_argument("--val_frac_pretest", type=float, default=0.2,
                    help="fraction of pre-test time for validation when using --test_years")

    # Infra
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--artifacts", default="artifacts_gru")
    ap.add_argument("--experiment", default=None, help="MLflow experiment name override")
    ap.add_argument("--log_model_every_improvement", action="store_true",
                    help="Log the pytorch model artifact when val improves (can be large).")

    args = ap.parse_args()
    os.makedirs(args.artifacts, exist_ok=True)

    # ---------- Load arrays
    weather = np.load(args.weather_npy)      # (T, Du)
    params  = np.load(args.params_npy)       # (N, P)
    outputs = np.load(args.outputs_npy)      # (N, T, Dz)
    T, Du = weather.shape
    N, T2, Dz = outputs.shape
    assert T == T2, "Time mismatch weather vs outputs"
    P = params.shape[1]

    # ---------- Build splits
    if args.test_years is not None:
        if not os.path.exists(args.dates_tsv):
            raise FileNotFoundError("--dates_tsv is required when using --test_years")
        years = pd.read_csv(args.dates_tsv, sep="\t")["YEAR"].to_numpy(int)
        t_train, t_val, t_test = build_time_splits_by_year(
            years, L=args.L, lead=args.lead,
            test_years=(args.test_years[0], args.test_years[1]),
            val_frac_of_pretest=args.val_frac_pretest
        )
    else:
        # simple chronological train/val split by time indices
        t_lo = args.L - 1
        t_hi = T - 1 - args.lead
        valid = np.arange(t_lo, t_hi + 1)
        n_train = int(math.floor(len(valid) * args.train_frac))
        t_train = valid[:n_train]
        t_val   = valid[n_train:]
        t_test  = None

    # ---------- Fit normalization on TRAIN windows only (weather), params across trials
    # gather unique weather indices that contribute to train windows
    train_weather_rows = np.unique(
        np.concatenate([np.arange(t - args.L + 1, t + 1) for t in t_train])
    )
    x_mean = weather[train_weather_rows].mean(axis=0, keepdims=True)
    x_std  = weather[train_weather_rows].std(axis=0, keepdims=True) + 1e-6
    p_mean = params.mean(axis=0, keepdims=True)
    p_std  = params.std(axis=0, keepdims=True) + 1e-6

    weather_std = (weather - x_mean) / x_std
    params_std  = (params - p_mean) / p_std

    # ---------- Datasets & loaders
    train_ds = WindowNextDayDataset(weather_std, params_std, outputs, dates_tsv=args.dates_tsv, L=args.L, lead=args.lead, t_indices=t_train)
    val_ds   = WindowNextDayDataset(weather_std, params_std, outputs, dates_tsv=args.dates_tsv, L=args.L, lead=args.lead, t_indices=t_val)
    test_ds  = None
    if t_test is not None and len(t_test) > 0:
        test_ds = WindowNextDayDataset(weather_std, params_std, outputs, dates_tsv=args.dates_tsv, L=args.L, lead=args.lead, t_indices=t_test)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=max(1, args.batch_size), shuffle=False, num_workers=0)
    test_loader  = None if test_ds is None else DataLoader(test_ds, batch_size=max(1, args.batch_size), shuffle=False)

    # ---------- Model & optim
    model = LakeProfileGRU(driver_dim=Du, param_dim=P, hidden=args.hidden, depth_out=Dz).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()

    best_val = float("inf")
    ckpt_path = os.path.join(args.artifacts, "best_model.pt")

    # ---------- MLflow
    exp_name = args.experiment or os.getenv("MLFLOW_EXPERIMENT_NAME", "Lake-GRU")
    mlflow.set_experiment(exp_name)
    run_name = f"GRU_L{args.L}_lead{args.lead}_H{args.hidden}"

    with mlflow.start_run(run_name=run_name):
        log_git_info_as_tags()  # optional
        mlflow.set_tag("framework", "pytorch")
        mlflow.set_tag("task", "lake_temp_profile_nextday")

        # log run-level params
        mlflow.log_params({
            "L": args.L, 
            "lead": args.lead, 
            "hidden": args.hidden,
            "epochs": args.epochs, 
            "batch_size": args.batch_size, 
            "lr": args.lr,
            "Du": Du, 
            "P": P, 
            "Dz": Dz, 
            "device": args.device,
            "smooth_lambda": args.smooth_lambda,
            "split_mode": "strict_years" if args.test_years is not None else "time_frac",
            "train_len": int(len(t_train)), 
            "val_len": int(len(t_val)),
            "test_len": int(len(t_test)) if t_test is not None else 0
        })

        # Training loop
        for epoch in range(1, args.epochs + 1):
            model.train()
            train_loss_sum, train_n = 0.0, 0
            for x_win, p_vec, y in train_loader:
                x_win = x_win.float().to(args.device)
                p_vec = p_vec.float().to(args.device)
                y     = y.float().to(args.device)

                pred = model(x_win, p_vec)
                loss = mse(pred, y)
                if args.smooth_lambda > 0.0 and Dz > 1:
                    dz_smooth = (pred[:, 1:] - pred[:, :-1]).abs().mean()
                    loss = loss + args.smooth_lambda * dz_smooth

                opt.zero_grad()
                loss.backward()
                opt.step()

                train_loss_sum += loss.item() * x_win.size(0)
                train_n += x_win.size(0)

            avg_train = train_loss_sum / max(1, train_n)

            # Validation
            model.eval()
            val_loss_sum, val_n = 0.0, 0
            with torch.no_grad():
                for x_win, p_vec, y in val_loader:
                    x_win = x_win.float().to(args.device)
                    p_vec = p_vec.float().to(args.device)
                    y     = y.float().to(args.device)
                    pred = model(x_win, p_vec)
                    loss = mse(pred, y)
                    val_loss_sum += loss.item() * x_win.size(0)
                    val_n += x_win.size(0)
            avg_val = val_loss_sum / max(1, val_n)
            
            # Log metrics
            mlflow.log_metrics({
                "train_mse": float(avg_train),
                "val_mse": float(avg_val),
            }, step=epoch)

            print(f"Epoch {epoch:03d} | train MSE {avg_train:.5f} | val MSE {avg_val:.5f} | ")

            # Save best
            if avg_val < best_val:
                best_val = avg_val
                torch.save({
                    "model_state": model.state_dict(),
                    "Du": Du, "Dz": Dz, "P": P,
                    "config": vars(args)
                }, ckpt_path)
                mlflow.log_artifact(ckpt_path, artifact_path="checkpoints")
                if args.log_model_every_improvement:
                    mlflow.pytorch.log_model(model, artifact_path="model")

        # Log norms & config as artifacts
        norms_path = os.path.join(args.artifacts, "norms.npz")
        np.savez(norms_path,
                 driver_mean=x_mean.astype(np.float32),
                 driver_std=x_std.astype(np.float32),
                 param_mean=p_mean.astype(np.float32),
                 param_std=p_std.astype(np.float32))
        mlflow.log_artifact(norms_path, artifact_path="artifacts")

        cfg_path = os.path.join(args.artifacts, "config.json")
        with open(cfg_path, "w") as f:
            json.dump(vars(args), f, indent=2)
        mlflow.log_artifact(cfg_path, artifact_path="artifacts")

        # Final test (if strict split was used)
        if test_loader is not None:
            model.eval()
            test_loss_sum, test_n = 0.0, 0
            with torch.no_grad():
                for x_win, p_vec, y in test_loader:
                    pred = model(x_win.float().to(args.device), p_vec.float().to(args.device))
                    loss = mse(pred, y.float().to(args.device))
                    test_loss_sum += loss.item() * x_win.size(0)
                    test_n += x_win.size(0)
            test_mse = test_loss_sum / max(1, test_n)
            mlflow.log_metric("test_mse", float(test_mse))
            print(f"Strict TEST MSE: {test_mse:.6f}")

        mlflow.log_metric("best_val_mse", float(best_val))
        print(f"\nBest val MSE: {best_val:.6f}")
        print(f"MLflow artifacts at: {mlflow.get_artifact_uri()}")

if __name__ == "__main__":
    main()
