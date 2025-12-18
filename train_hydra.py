#!/usr/bin/env python3
import os, json, subprocess
from dataclasses import dataclass, field
from typing import List
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig, OmegaConf
import mlflow
import mlflow.pytorch
from models import ModelRNN
from lake_dataset import LakeWindowDataset, load_data


# -----------------------------
# Training / eval
# -----------------------------
def train_one_epoch(loader, model, opt, device, smooth_lambda: float = 0.0):
    model.train()
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

        # smoothness penalty across depth if output is (B, Dz)
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


def log_git_info():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        mlflow.set_tag("git.commit", commit)
        mlflow.set_tag("git.dirty", "1" if dirty else "0")
    except Exception:
        pass


# -----------------------------
# Hydra main
# -----------------------------
@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # cfg is assembled from your YAMLs
    # Expected structure:
    # cfg.data, cfg.model, cfg.train, cfg.mlflow (see example config below)

    # Repro
    torch.manual_seed(int(cfg.train.seed))
    np.random.seed(int(cfg.train.seed))

    device = str(cfg.train.device)
    os.makedirs(str(cfg.train.artifacts), exist_ok=True)

    # ---- Load data (smooth_windows is a LIST)
    smooth_windows = list(cfg.data.smooth_windows) if cfg.data.smooth_windows is not None else []
    data = load_data(
        weather_path=str(cfg.data.weather),
        output_path=str(cfg.data.outputs),
        params_path=str(cfg.data.params),
        bathymetry_path=str(cfg.data.bathymetry_path),
        split_years=list(cfg.data.split_years),
        normalize=bool(cfg.data.normalize),
        smooth_windows=smooth_windows,
        smooth_pad_mode=str(cfg.data.smooth_pad_mode),
    )

    w_tr, o_tr = data["weather_data_train"], data["output_data_train"]
    w_va, o_va = data["weather_data_val"], data["output_data_val"]
    w_te, o_te = data["weather_data_test"], data["output_data_test"]
    doy_tr, doy_va, doy_te = data["doy_train"], data["doy_val"], data["doy_test"]
    params = data["params_data"]
    norms = data["norms"]

    Dz = o_tr.shape[2]
    P = params.shape[1]

    # ---- Datasets/loaders
    Wx = int(cfg.model.Wx)
    Wy = int(cfg.model.Wy)

    train_ds = LakeWindowDataset(w_tr, o_tr, params, W_x=Wx, W_y=Wy)
    val_ds   = LakeWindowDataset(w_va, o_va, params, W_x=Wx, W_y=Wy)
    test_ds  = LakeWindowDataset(w_te, o_te, params, W_x=Wx, W_y=Wy)

    Du = train_ds.Du

    train_loader = DataLoader(train_ds, batch_size=int(cfg.train.batch_size), shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=max(1, int(cfg.train.batch_size)//2), shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=max(1, int(cfg.train.batch_size)//2), shuffle=False, num_workers=0)

    # ---- Model (GRU only)
    head_hidden = list(cfg.model.head_hidden)
    model = ModelRNN(
        Du=Du, 
        P=P, 
        Dz=Dz,
        hidden=int(cfg.model.hidden),
        num_layers=int(cfg.model.num_layers),
        rnn_dropout=float(cfg.model.rnn_dropout),
        W_y=Wy,
        head_hidden=head_hidden,
        head_dropout=float(cfg.model.head_dropout),
        # if your simplified ModelRNN supports these:
        param_in_rnn=bool(cfg.model.get("param_in_rnn", True)),
        param_in_head=bool(cfg.model.get("param_in_head", True)),
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=float(cfg.train.lr))

    # ---- MLflow
    mlflow.set_experiment(str(cfg.mlflow.experiment))
    run_name = str(cfg.mlflow.get("run_name", f"GRU_Wx{Wx}_Wy{Wy}"))

    with mlflow.start_run(run_name=run_name):
        log_git_info()

        # log full resolved config
        mlflow.log_text(OmegaConf.to_yaml(cfg), "hydra_config.yaml")

        # log key params (optional; MLflow params must be simple types)
        mlflow.log_params({
            "Wx": Wx, 
            "Wy": Wy,
            "hidden": int(cfg.model.hidden),
            "num_layers": int(cfg.model.num_layers),
            "rnn_dropout": float(cfg.model.rnn_dropout),
            "head_hidden": ",".join(map(str, head_hidden)),
            "head_dropout": float(cfg.model.head_dropout),
            "lr": float(cfg.train.lr),
            "batch_size": int(cfg.train.batch_size),
            "epochs": int(cfg.train.epochs),
            "smooth_lambda": float(cfg.train.smooth_lambda),
            "smooth_windows": ",".join(map(str, smooth_windows)),
            "smooth_pad_mode": str(cfg.data.smooth_pad_mode),
            "Du": Du, 
            "Dz": Dz, 
            "P": P,
            "device": device,
        })

        best_val = float("inf")
        ckpt_path = os.path.join(str(cfg.train.artifacts), "best.pt")
        os.makedirs(str(cfg.train.artifacts), exist_ok=True)

        for ep in range(1, int(cfg.train.epochs) + 1):
            tr_mse = train_one_epoch(train_loader, model, opt, device, smooth_lambda=float(cfg.train.smooth_lambda))
            va_mse = eval_epoch(val_loader, model, device)
            mlflow.log_metrics({"train_mse": tr_mse, "val_mse": va_mse}, step=ep)

            print(f"Epoch {ep:03d} | train MSE {tr_mse:.6f} | val MSE {va_mse:.6f}")

            if va_mse < best_val:
                best_val = va_mse
                torch.save({
                    "state_dict": model.state_dict(),
                    "config": OmegaConf.to_container(cfg, resolve=True),
                    "Du": Du, "Dz": Dz, "P": P,
                }, ckpt_path)
                mlflow.log_artifact(ckpt_path, artifact_path="checkpoints")

        te_mse = eval_epoch(test_loader, model, device)
        mlflow.log_metric("test_mse", te_mse)
        mlflow.log_metric("best_val_mse", best_val)

        print(f"TEST MSE: {te_mse:.6f}")
        print(f"Best val MSE: {best_val:.6f}")
        print(f"Artifacts at: {mlflow.get_artifact_uri()}")


if __name__ == "__main__":
    main()
