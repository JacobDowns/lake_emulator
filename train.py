#!/usr/bin/env python3
import os, json, argparse, subprocess
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# non-interactive backend for HPC/headless
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- your model ----
from models import ModelRNN

# ---- dataset / loader ----
from lake_dataset import LakeWindowDataset, load_data

# ---- MLflow ----
import mlflow
import mlflow.pytorch


# =========================================================
# Optional seasonal features (train.py-local, since your lake_dataset.py
# snippet doesn't include it)
# =========================================================
def build_seasonal_features(doy: np.ndarray) -> np.ndarray:
    """
    doy: (T,) values ~ [1..365] or [0..364]
    returns (T, 2): sin/cos encoding
    """
    angle = 2.0 * np.pi * (doy.astype(np.float32) / 365.0)
    return np.stack([np.sin(angle), np.cos(angle)], axis=-1).astype(np.float32)


# =========================================================
# Training / eval / plotting
# =========================================================
def train_one_epoch(loader, model, opt, device, smooth_lambda: float = 0.0):
    model.train()
    mse = nn.MSELoss()
    total, count = 0.0, 0

    for batch in loader:
        # Dataset may return (sim_id, x, p, y) if return_ids=True, otherwise (x,p,y)
        if len(batch) == 4:
            _, x, p, y = batch
        else:
            x, p, y = batch

        x = x.to(device).float()    # (B, Wx, Du_total)
        p = p.to(device).float()    # (B, P)
        y = y.to(device).float()    # (B, Wy, Dz) or (B, Dz) if Wy==1 squeeze later

        if y.dim() == 3 and y.size(1) == 1:
            y = y[:, 0, :]          # (B, Dz)

        pred = model(x, p)          # (B, Dz) if Wy==1 else (B, Wy, Dz)
        loss = mse(pred, y)

        # Smoothness penalty across depth dimension for (B, Dz) outputs
        if smooth_lambda > 0.0 and pred.dim() == 2 and pred.size(1) > 1:
            loss = loss + smooth_lambda * (pred[:, 1:] - pred[:, :-1]).abs().mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        total += loss.item() * x.size(0)
        count += x.size(0)

    return total / max(1, count)


@torch.no_grad()
def eval_epoch(loader, model, device):
    model.eval()
    mse = nn.MSELoss()
    total, count = 0.0, 0

    for batch in loader:
        if len(batch) == 4:
            _, x, p, y = batch
        else:
            x, p, y = batch

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
def per_depth_rmse(loader, model, device, Dz: int):
    model.eval()
    sse = torch.zeros(Dz, device=device)
    n = 0

    for batch in loader:
        if len(batch) == 4:
            _, x, p, y = batch
        else:
            x, p, y = batch

        x = x.to(device).float()
        p = p.to(device).float()
        y = y.to(device).float()

        if y.dim() == 3 and y.size(1) == 1:
            y = y[:, 0, :]

        pred = model(x, p)
        sse += ((pred - y) ** 2).sum(dim=0)
        n += y.size(0)

    return torch.sqrt(sse / max(1, n)).detach().cpu().numpy()


def _norms_output_arrays(norms):
    mu = np.array(norms["output_mean"])
    sd = np.array(norms["output_std"])
    if mu.ndim == 3:      # (1,1,Dz)
        mu = mu.reshape(1, -1)
        sd = sd.reshape(1, -1)
    elif mu.ndim == 1:    # (Dz,)
        mu = mu.reshape(1, -1)
        sd = sd.reshape(1, -1)
    return mu.astype(np.float32), sd.astype(np.float32)


def denorm_outputs(y_norm: np.ndarray, norms: dict) -> np.ndarray:
    if norms is None:
        return y_norm
    mu, sd = _norms_output_arrays(norms)
    if y_norm.ndim == 2:
        return y_norm * sd + mu
    elif y_norm.ndim == 3:
        return y_norm * sd[None, :, :] + mu[None, :, :]
    return y_norm


@torch.no_grad()
def make_val_depth_profile_plot(model, val_loader, device, norms, Dz, epoch,
                                outdir="plots", max_samples=4):
    model.eval()
    os.makedirs(outdir, exist_ok=True)

    try:
        batch = next(iter(val_loader))
    except StopIteration:
        return None

    if len(batch) == 4:
        _, x, p, y = batch
    else:
        x, p, y = batch

    x = x.to(device).float()
    p = p.to(device).float()
    y = y.to(device).float()

    # Targets -> (B, Dz) for plotting (use last step if multi-step)
    if y.dim() == 3:
        y_plot = y[:, -1, :] if y.size(1) > 1 else y[:, 0, :]
    else:
        y_plot = y

    pred = model(x, p)
    if pred.dim() == 3:
        pred_plot = pred[:, -1, :] if pred.size(1) > 1 else pred[:, 0, :]
    else:
        pred_plot = pred

    y_den = denorm_outputs(y_plot.detach().cpu().numpy(), norms)
    p_den = denorm_outputs(pred_plot.detach().cpu().numpy(), norms)

    depth = np.arange(Dz) + 0.5
    K = min(max_samples, y_den.shape[0])

    plt.figure(figsize=(6.5, 4.5))
    for i in range(K):
        plt.plot(y_den[i], depth, linestyle="-",  label="True" if i == 0 else None)
        plt.plot(p_den[i], depth, linestyle="--", label="Pred" if i == 0 else None)

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
    return [int(x) for x in str(s).split(",") if x.strip()]


def log_git_info_as_tags():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        mlflow.set_tag("git.commit", commit)
        mlflow.set_tag("git.dirty", "1" if dirty else "0")
    except Exception:
        pass


def parse_int_list_csv(s: str) -> List[int]:
    """
    Parses "7,30" -> [7,30]; "" -> []
    """
    s = (s or "").strip()
    if not s:
        return []
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if tok:
            out.append(int(tok))
    return out


# =========================================================
# Main
# =========================================================
def main():
    ap = argparse.ArgumentParser()

    # data & split
    ap.add_argument("--weather", default="data/parsed_data/weather_data.npy")
    ap.add_argument("--outputs", default="data/parsed_data/output_data.npy")
    ap.add_argument("--params",  default="data/parsed_data/parameter_data.npy")
    ap.add_argument("--bathymetry_path", default="data/BearLake_inputs_outputs/inputs/BearLake_bathy.csv")
    ap.add_argument("--split_years", type=int, nargs=3, default=[2018, 2020, 2025])
    ap.add_argument("--normalize", type=int, default=1)

    # smoothing (MUST match lake_dataset.py: pad_mode reflect|edge|wrap)
    ap.add_argument("--smooth_windows", type=str, default="",
                    help='Comma-separated smoothing windows, e.g. "7,30". Empty = none.')
    ap.add_argument("--smooth_pad_mode", type=str, default="reflect",
                    choices=["reflect", "edge", "wrap"])

    # windowing
    ap.add_argument("--Wx", type=int, default=90)
    ap.add_argument("--Wy", type=int, default=1)

    # model hypers
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--rnn_dropout", type=float, default=0.0)

    ap.add_argument("--head_hidden", type=str, default="256,256")
    ap.add_argument("--head_dropout", type=float, default=0.0)

    # training
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--smooth_lambda", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)

    # extra features (optional, not in your pasted loader; we add at dataset stage)
    ap.add_argument("--use_seasonal_features", action="store_true")

    # infra
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--artifacts", default="artifacts_run")
    ap.add_argument("--experiment", default=None)
    ap.add_argument("--log_model_every_improvement", action="store_true")

    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.artifacts, exist_ok=True)

    smooth_windows = parse_int_list_csv(args.smooth_windows)

    # ---------- load data (smoothing is applied+appended inside load_data)
    data = load_data(
        weather_path=args.weather,
        output_path=args.outputs,
        params_path=args.params,
        bathymetry_path=args.bathymetry_path,
        split_years=args.split_years,
        normalize=bool(args.normalize),
        smooth_windows=smooth_windows,
        smooth_pad_mode=args.smooth_pad_mode,
    )

    w_tr = data["weather_data_train"];  o_tr = data["output_data_train"]
    w_va = data["weather_data_val"];    o_va = data["output_data_val"]
    w_te = data["weather_data_test"];   o_te = data["output_data_test"]

    doy_tr = data["doy_train"]; doy_va = data["doy_val"]; doy_te = data["doy_test"]
    params = data["params_data"]
    norms  = data["norms"]

    Dz = o_tr.shape[2]
    P  = params.shape[1]

    # ---------- optional seasonal extra features appended at Dataset stage
    extra_tr = extra_va = extra_te = None
    if args.use_seasonal_features:
        extra_tr = build_seasonal_features(doy_tr)
        extra_va = build_seasonal_features(doy_va)
        extra_te = build_seasonal_features(doy_te)

    # ---------- datasets & loaders
    train_ds = LakeWindowDataset(w_tr, o_tr, params, W_x=args.Wx, W_y=args.Wy, extra_features=extra_tr)
    val_ds   = LakeWindowDataset(w_va, o_va, params, W_x=args.Wx, W_y=args.Wy, extra_features=extra_va)
    test_ds  = LakeWindowDataset(w_te, o_te, params, W_x=args.Wx, W_y=args.Wy, extra_features=extra_te)

    Du = train_ds.Du

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=max(1, args.batch_size // 2), shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=max(1, args.batch_size // 2), shuffle=False, num_workers=0)

    # ---------- model (GRU only)
    head_hidden = parse_hidden_list(args.head_hidden)
    model = ModelRNN(
        cell="gru",
        Du=Du, P=P, Dz=Dz,
        hidden=args.hidden,
        num_layers=args.num_layers,
        rnn_dropout=args.rnn_dropout,
        W_y=args.Wy,
        head_hidden=head_hidden,
        head_dropout=args.head_dropout,
    )

    device = args.device
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = float("inf")
    ckpt_path = os.path.join(args.artifacts, "best.pt")
    plots_dir = os.path.join(args.artifacts, "plots")

    # ---------- MLflow
    exp_name = args.experiment or os.getenv("MLFLOW_EXPERIMENT_NAME", "Lake-Emu")
    mlflow.set_experiment(exp_name)

    run_name = f"GRU_Wx{args.Wx}_Wy{args.Wy}"
    with mlflow.start_run(run_name=run_name):
        log_git_info_as_tags()

        # sweep can pass RUN_KEY env var
        env_run_key = os.getenv("RUN_KEY", "")
        if env_run_key:
            mlflow.set_tag("run_key", env_run_key)

        # log params
        mlflow.log_params({
            "model": "gru",
            "Wx": args.Wx,
            "Wy": args.Wy,
            "Du": Du,
            "Dz": Dz,
            "P": P,
            "hidden": args.hidden,
            "num_layers": args.num_layers,
            "rnn_dropout": args.rnn_dropout,
            "head_hidden": args.head_hidden,
            "head_dropout": args.head_dropout,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "smooth_lambda": args.smooth_lambda,
            "device": device,
            "split_years": ",".join(map(str, args.split_years)),
            "use_seasonal_features": int(args.use_seasonal_features),
            "smooth_windows": ",".join(map(str, smooth_windows)),
            "smooth_pad_mode": args.smooth_pad_mode,
        })
        mlflow.log_metric("num_params", sum(p.numel() for p in model.parameters()))

        # training loop
        for ep in range(1, args.epochs + 1):
            tr_mse = train_one_epoch(train_loader, model, opt, device, smooth_lambda=args.smooth_lambda)
            va_mse = eval_epoch(val_loader, model, device)
            mlflow.log_metrics({"train_mse": tr_mse, "val_mse": va_mse}, step=ep)

            if args.Wy == 1:
                rmse_depth = per_depth_rmse(val_loader, model, device, Dz)
                mlflow.log_metric("val_rmse_mean_depths", float(rmse_depth.mean()), step=ep)

            plot_path = make_val_depth_profile_plot(
                model, val_loader, device, norms, Dz, ep,
                outdir=plots_dir, max_samples=4
            )
            if plot_path is not None:
                mlflow.log_artifact(plot_path, artifact_path="plots")

            print(f"Epoch {ep:03d} | train MSE {tr_mse:.6f} | val MSE {va_mse:.6f}")

            if va_mse < best_val:
                best_val = va_mse
                torch.save({
                    "state_dict": model.state_dict(),
                    "config": vars(args),
                    "Du": Du, "Dz": Dz, "P": P,
                }, ckpt_path)
                mlflow.log_artifact(ckpt_path, artifact_path="checkpoints")
                if args.log_model_every_improvement:
                    mlflow.pytorch.log_model(model, artifact_path="model")

        # save norms
        if norms is not None:
            norms_path = os.path.join(args.artifacts, "norms.npz")
            # norms contains a string "smooth_pad_mode"; np.savez will store it as object/string fine
            np.savez(norms_path, **norms)
            mlflow.log_artifact(norms_path, artifact_path="artifacts")

        cfg_path = os.path.join(args.artifacts, "config.json")
        with open(cfg_path, "w") as f:
            json.dump(vars(args), f, indent=2)
        mlflow.log_artifact(cfg_path, artifact_path="artifacts")

        te_mse = eval_epoch(test_loader, model, device)
        mlflow.log_metric("test_mse", te_mse)
        mlflow.log_metric("best_val_mse", best_val)

        print(f"TEST MSE: {te_mse:.6f}")
        print(f"Best val MSE: {best_val:.6f}")
        print(f"Artifacts at: {mlflow.get_artifact_uri()}")


if __name__ == "__main__":
    main()
