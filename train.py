#!/usr/bin/env python3
# train.py
from __future__ import annotations

import os
import json
import argparse
import subprocess
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mlflow

from multi_lake_dataset import load_multi_lake_data, MultiLakeWindowDataset
from models import ModelRNNDepth


# -------------------------
# Utils
# -------------------------
def parse_int_tuple3(x: str) -> Tuple[int, int, int]:
    parts = [int(t.strip()) for t in x.split(",")]
    if len(parts) != 3:
        raise ValueError("split_years must be 'train_end,val_end,max' e.g. 2018,2020,2025")
    return parts[0], parts[1], parts[2]


def parse_hidden_list(s: str) -> List[int]:
    s = str(s).strip()
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def log_git_info_as_tags():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        mlflow.set_tag("git.commit", commit)
        mlflow.set_tag("git.dirty", "1" if dirty else "0")
    except Exception:
        pass


def masked_mse(pred: torch.Tensor, target: torch.Tensor, depth_mask: torch.Tensor) -> torch.Tensor:
    """
    pred/target: (B, Wy, Dz_max) or (B, Dz_max) if Wy==1 squeezed
    depth_mask: (B, Dz_max) or (Dz_max,)

    Computes MSE over valid depths only. Averages over batch/time/depth(valid).
    """
    if pred.dim() == 2:
        # (B, Dz)
        if depth_mask.dim() == 1:
            m = depth_mask.unsqueeze(0)  # (1, Dz)
        else:
            m = depth_mask               # (B, Dz)
        se = (pred - target) ** 2
        se = se * m
        denom = m.sum().clamp_min(1.0)
        return se.sum() / denom

    # (B, Wy, Dz)
    if depth_mask.dim() == 1:
        m = depth_mask.view(1, 1, -1)  # (1,1,Dz)
    else:
        m = depth_mask.unsqueeze(1)    # (B,1,Dz)
    se = (pred - target) ** 2
    se = se * m
    denom = m.sum().clamp_min(1.0)
    return se.sum() / denom


def masked_depth_smoothness(pred: torch.Tensor, depth_mask: torch.Tensor) -> torch.Tensor:
    """
    Smoothness penalty across depth (adjacent bins), respecting mask.
    pred: (B, Dz) or (B, Wy, Dz)
    depth_mask: (B, Dz) or (Dz,)
    """
    if pred.dim() == 2:
        # (B, Dz)
        if depth_mask.dim() == 1:
            m = depth_mask.unsqueeze(0)  # (1, Dz)
        else:
            m = depth_mask
        dif = (pred[:, 1:] - pred[:, :-1]).abs()  # (B, Dz-1)
        m_pair = (m[:, 1:] * m[:, :-1])           # valid adjacent pairs
        denom = m_pair.sum().clamp_min(1.0)
        return (dif * m_pair).sum() / denom

    # (B, Wy, Dz)
    if depth_mask.dim() == 1:
        m = depth_mask.view(1, 1, -1)
    else:
        m = depth_mask.unsqueeze(1)
    dif = (pred[:, :, 1:] - pred[:, :, :-1]).abs()  # (B, Wy, Dz-1)
    m_pair = (m[:, :, 1:] * m[:, :, :-1])
    denom = m_pair.sum().clamp_min(1.0)
    return (dif * m_pair).sum() / denom


# -------------------------
# Train / eval
# -------------------------
def train_one_epoch(loader, model, opt, device, smooth_lambda: float = 0.0):
    model.train()
    total = 0.0
    count = 0

    for lake_id, x, p, depth_feat, depth_mask, y in loader:
        x = x.to(device).float()                 # (B, Wx, Du)
        p = p.to(device).float()                 # (B, P)
        depth_feat = depth_feat.to(device).float()  # (B, Dz, Dd)
        depth_mask = depth_mask.to(device).float()  # (B, Dz)
        y = y.to(device).float()                 # (B, Wy, Dz)

        # Safety: ensure correct orientation (prevents Du/Wx swap bugs)
        # Expect x = (B, Wx, Du) and second dim should be Wx
        # If accidentally (B, Du, Wx), transpose.
        if x.dim() == 3 and x.shape[1] < x.shape[2]:
            x = x.transpose(1, 2).contiguous()

        pred = model(x, p, depth_feat)           # (B, Wy, Dz) or (B, Dz) if Wy==1 squeeze

        loss = masked_mse(pred, y if pred.dim() == 3 else y[:, 0, :], depth_mask)
        if smooth_lambda > 0.0:
            loss = loss + smooth_lambda * masked_depth_smoothness(pred, depth_mask)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        total += loss.item() * x.size(0)
        count += x.size(0)

    return total / max(1, count)


@torch.no_grad()
def eval_epoch(loader, model, device):
    model.eval()
    total = 0.0
    count = 0

    for lake_id, x, p, depth_feat, depth_mask, y in loader:
        x = x.to(device).float()
        p = p.to(device).float()
        depth_feat = depth_feat.to(device).float()
        depth_mask = depth_mask.to(device).float()
        y = y.to(device).float()

        if x.dim() == 3 and x.shape[1] < x.shape[2]:
            x = x.transpose(1, 2).contiguous()

        pred = model(x, p, depth_feat)
        loss = masked_mse(pred, y if pred.dim() == 3 else y[:, 0, :], depth_mask)

        total += loss.item() * x.size(0)
        count += x.size(0)

    return total / max(1, count)


# -------------------------
# Optional small debug plot
# -------------------------
@torch.no_grad()
def make_debug_depth_plot(model, loader, device, epoch, outdir="plots", max_samples=3):
    """
    Plots predicted vs true last-step depth profiles for a few samples.
    (Uses normalized values; quick sanity check.)
    """
    os.makedirs(outdir, exist_ok=True)
    model.eval()

    batch = next(iter(loader), None)
    if batch is None:
        return None

    lake_id, x, p, depth_feat, depth_mask, y = batch
    x = x.to(device).float()
    p = p.to(device).float()
    depth_feat = depth_feat.to(device).float()
    depth_mask = depth_mask.to(device).float()
    y = y.to(device).float()

    if x.dim() == 3 and x.shape[1] < x.shape[2]:
        x = x.transpose(1, 2).contiguous()

    pred = model(x, p, depth_feat)  # (B, Wy, Dz) or (B, Dz)

    if pred.dim() == 3:
        pred_last = pred[:, -1, :]
        y_last = y[:, -1, :]
    else:
        pred_last = pred
        y_last = y[:, 0, :]

    pred_np = pred_last.detach().cpu().numpy()
    y_np = y_last.detach().cpu().numpy()
    m_np = depth_mask.detach().cpu().numpy()

    B = pred_np.shape[0]
    K = min(max_samples, B)

    plt.figure(figsize=(7, 4))
    for i in range(K):
        valid = m_np[i] > 0.5
        z = np.where(valid)[0]
        plt.plot(y_np[i, valid], z, label="true" if i == 0 else None, linewidth=1.5)
        plt.plot(pred_np[i, valid], z, label="pred" if i == 0 else None, linestyle="--", linewidth=1.2)

    plt.gca().invert_yaxis()
    plt.xlabel("Temp (normalized)")
    plt.ylabel("Depth index")
    plt.title(f"Depth profiles (epoch {epoch})")
    plt.legend(loc="best")
    path = os.path.join(outdir, f"depth_profiles_epoch_{epoch:03d}.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()

    # data
    ap.add_argument("--root_dir", type=str, default="data/parsed_data")
    ap.add_argument("--lake_names", type=str, default="BearLake,RedPond",
                    help="Comma-separated lake directory names inside root_dir")
    ap.add_argument("--split_years", type=str, default="2018,2021,2025")
    ap.add_argument("--normalize", type=int, default=1)

    # windowing
    ap.add_argument("--Wx", type=int, default=360)
    ap.add_argument("--Wy", type=int, default=60)
    ap.add_argument("--window_stride", type=int, default=1, help="stride between window starts (>=1)")

    # model
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--rnn_dropout", type=float, default=0.0)
    ap.add_argument("--head_hidden", type=str, default="256,256,256")
    ap.add_argument("--head_dropout", type=float, default=0.0)
    ap.add_argument("--param_in_rnn", type=int, default=1)
    ap.add_argument("--param_in_head", type=int, default=1)

    # training
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--smooth_lambda", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)

    # infra
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--pin_memory", type=int, default=1)
    ap.add_argument("--precompute_windows", type=int, default=1)

    # logging / artifacts
    ap.add_argument("--artifacts", type=str, default="artifacts_run")
    ap.add_argument("--experiment", type=str, default="test")
    ap.add_argument("--log_debug_plot", type=int, default=1)

    args = ap.parse_args()

    # seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.artifacts, exist_ok=True)
    plots_dir = os.path.join(args.artifacts, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    lake_names = [s.strip() for s in args.lake_names.split(",") if s.strip()]
    split_years = parse_int_tuple3(args.split_years)

    # Load + normalize multi-lake data
    lakes, norms = load_multi_lake_data(
        root_dir=args.root_dir,
        lake_names=lake_names,
        split_years=split_years,
        normalize=bool(args.normalize),
    )

    # Build datasets
    train_ds = MultiLakeWindowDataset(
        lakes=lakes,
        split="train",
        split_years=split_years,
        Wx=args.Wx,
        Wy=args.Wy,
        precompute_weather_windows=bool(args.precompute_windows),
        window_stride=args.window_stride,
    )
    val_ds = MultiLakeWindowDataset(
        lakes=lakes,
        split="val",
        split_years=split_years,
        Wx=args.Wx,
        Wy=args.Wy,
        precompute_weather_windows=bool(args.precompute_windows),
        window_stride=args.window_stride,
    )
    test_ds = MultiLakeWindowDataset(
        lakes=lakes,
        split="test",
        split_years=split_years,
        Wx=args.Wx,
        Wy=args.Wy,
        precompute_weather_windows=bool(args.precompute_windows),
        window_stride=args.window_stride,
    )

    Du = train_ds.Du
    P = train_ds.P
    Dd = train_ds.Dd
    Dz_max = train_ds.Dz_max

    # DataLoaders
    # NOTE: pin_memory warnings you saw were tied to pin_memory_device usage.
    # We do not pass pin_memory_device here; pin_memory=True is fine.
    pin_memory = bool(args.pin_memory) and (args.device.startswith("cuda"))
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_ds, batch_size=max(1, args.batch_size // 2), shuffle=False,
        num_workers=args.num_workers, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_ds, batch_size=max(1, args.batch_size // 2), shuffle=False,
        num_workers=args.num_workers, pin_memory=pin_memory
    )

    # Model
    head_hidden = parse_hidden_list(args.head_hidden)
    model = ModelRNNDepth(
        Du=Du,
        P=P,
        Dd=Dd,
        hidden=args.hidden,
        num_layers=args.num_layers,
        rnn_dropout=args.rnn_dropout,
        Wy=args.Wy,
        head_hidden=head_hidden,
        head_dropout=args.head_dropout,
        param_in_rnn=bool(args.param_in_rnn),
        param_in_head=bool(args.param_in_head),
    ).to(args.device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # MLflow
    mlflow.set_experiment(args.experiment)
    run_name = f"MultiLake_GRU_Wx{args.Wx}_Wy{args.Wy}"
    best_val = float("inf")
    best_path = os.path.join(args.artifacts, "best.pt")

    with mlflow.start_run(run_name=run_name):
        log_git_info_as_tags()

        # log params
        mlflow.log_params({
            "root_dir": args.root_dir,
            "lake_names": ",".join(lake_names),
            "split_years": args.split_years,
            "normalize": int(args.normalize),

            "Wx": args.Wx,
            "Wy": args.Wy,
            "window_stride": args.window_stride,

            "Du": Du,
            "P": P,
            "Dd": Dd,
            "Dz_max": Dz_max,

            "hidden": args.hidden,
            "num_layers": args.num_layers,
            "rnn_dropout": args.rnn_dropout,
            "head_hidden": args.head_hidden,
            "head_dropout": args.head_dropout,
            "param_in_rnn": int(args.param_in_rnn),
            "param_in_head": int(args.param_in_head),

            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "smooth_lambda": args.smooth_lambda,

            "device": args.device,
            "num_workers": args.num_workers,
            "pin_memory": int(pin_memory),
            "precompute_windows": int(args.precompute_windows),
        })

        mlflow.log_metric("num_params", sum(p.numel() for p in model.parameters()))

        # training loop
        for ep in range(1, args.epochs + 1):
            tr = train_one_epoch(train_loader, model, opt, args.device, smooth_lambda=args.smooth_lambda)
            va = eval_epoch(val_loader, model, args.device)

            mlflow.log_metrics({"train_mse": tr, "val_mse": va}, step=ep)
            print(f"Epoch {ep:03d} | train MSE {tr:.6f} | val MSE {va:.6f}")

            if args.log_debug_plot:
                pth = make_debug_depth_plot(model, val_loader, args.device, ep, outdir=plots_dir, max_samples=3)
                if pth is not None:
                    mlflow.log_artifact(pth, artifact_path="plots")

            if va < best_val:
                best_val = va
                torch.save({
                    "state_dict": model.state_dict(),
                    "config": vars(args),
                    "Du": Du, "P": P, "Dd": Dd, "Dz_max": Dz_max,
                    "norms": norms,  # optional; can be large but convenient
                }, best_path)
                mlflow.log_artifact(best_path, artifact_path="checkpoints")

        te = eval_epoch(test_loader, model, args.device)
        mlflow.log_metric("test_mse", te)
        mlflow.log_metric("best_val_mse", best_val)

        # save config as json artifact too
        cfg_path = os.path.join(args.artifacts, "config.json")
        with open(cfg_path, "w") as f:
            json.dump(vars(args), f, indent=2)
        mlflow.log_artifact(cfg_path, artifact_path="artifacts")

        print(f"TEST MSE: {te:.6f}")
        print(f"Best val MSE: {best_val:.6f}")
        print(f"Artifacts at: {mlflow.get_artifact_uri()}")


if __name__ == "__main__":
    main()
