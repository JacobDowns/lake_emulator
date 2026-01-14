#!/usr/bin/env python3
# sanity_plot_dataloader.py
"""
Sanity-check visualization for the multi-lake variable-depth dataloader.

This script:
  - loads multi-lake data (train-only normalization optional)
  - builds a MultiLakeWindowDataset (train/val/test)
  - samples a handful of items
  - produces a quick set of plots:
      1) Weather window (Wx) for a few driver dims
      2) Target temperature profile(s) for each forecast step (Wy) as a heatmap
      3) Bathymetry-derived depth features + mask

Outputs PNGs into --outdir.
"""

from __future__ import annotations

import os
import argparse
import random
from typing import List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch

from multi_lake_dataset import load_multi_lake_data, MultiLakeWindowDataset


def _ensure_dir(d: str) -> None:
    os.makedirs(d, exist_ok=True)


def _plot_weather_window(x_win: torch.Tensor, title: str, outpath: str, max_dims: int = 6) -> None:
    """
    x_win: (Wx, Du)
    Plot a few driver dimensions over time.
    """
    x = x_win.detach().cpu().numpy()
    Wx, Du = x.shape
    dims = list(range(min(Du, max_dims)))

    plt.figure(figsize=(10, 4.5))
    for j in dims:
        plt.plot(np.arange(Wx), x[:, j], label=f"driver[{j}]")
    plt.title(title)
    plt.xlabel("Window time index (0..Wx-1)")
    plt.ylabel("Normalized driver value")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best", ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def _plot_target_heatmap(y: torch.Tensor, depth_mask: torch.Tensor, title: str, outpath: str) -> None:
    """
    y: (Wy, Dz_max)
    depth_mask: (Dz_max,)
    Heatmap of temperature targets over (time, depth), masking padded depths.
    """
    yy = y.detach().cpu().numpy()  # (Wy, Dz)
    m = depth_mask.detach().cpu().numpy().astype(bool)

    # Mask padded depths (set to nan so colormap ignores)
    yy_masked = yy.copy()
    yy_masked[:, ~m] = np.nan

    # transpose to (Dz, Wy) for imshow with depth on y-axis
    img = yy_masked.T  # (Dz, Wy)

    plt.figure(figsize=(6.5, 4.5))
    im = plt.imshow(img, aspect="auto", origin="upper", interpolation="nearest")
    plt.title(title)
    plt.xlabel("Forecast step (0..Wy-1)")
    plt.ylabel("Depth index (padded)")
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label("Normalized temperature")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def _plot_depth_features(depth_feat: torch.Tensor, depth_mask: torch.Tensor, title: str, outpath: str) -> None:
    """
    depth_feat: (Dz_max, Dd)
    depth_mask: (Dz_max,)
    Plot each depth feature vs depth index and show mask.
    """
    df = depth_feat.detach().cpu().numpy()
    m = depth_mask.detach().cpu().numpy()
    Dz, Dd = df.shape

    plt.figure(figsize=(10, 4.5))
    for k in range(Dd):
        plt.plot(np.arange(Dz), df[:, k], label=f"feat[{k}]")
    plt.plot(np.arange(Dz), m, label="mask", linestyle="--", linewidth=2)
    plt.title(title)
    plt.xlabel("Depth index (padded)")
    plt.ylabel("Depth feature value")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best", ncol=3, fontsize=9)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/parsed_data", help="Root containing lake subdirs")
    ap.add_argument("--lakes", default="BearLake,RedPond", help="Comma-separated lake names")
    ap.add_argument("--split_years", type=int, nargs=3, default=[2018, 2020, 2025])

    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--normalize", type=int, default=1, help="1=global train-only normalize, 0=no normalize")

    ap.add_argument("--Wx", type=int, default=365)
    ap.add_argument("--Wy", type=int, default=30)
    ap.add_argument("--precompute_weather_windows", type=int, default=1)

    ap.add_argument("--num_samples", type=int, default=5)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--outdir", default="plots_dataloader_sanity")

    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    lakes = [x.strip() for x in args.lakes.split(",") if x.strip()]
    _ensure_dir(args.outdir)

    # Load data (and optionally normalize)
    lake_datas, norms = load_multi_lake_data(
        root_dir=args.data_root,
        lake_names=lakes,
        split_years=list(args.split_years),
        normalize=bool(args.normalize),
    )

    # Build dataset for requested split
    ds = MultiLakeWindowDataset(
        lakes=lake_datas,
        split=args.split,
        split_years=list(args.split_years),
        Wx=args.Wx,
        Wy=args.Wy,
        precompute_weather_windows=bool(args.precompute_weather_windows),
    )

    print(f"Dataset split={args.split}: len={len(ds)}")
    print(f"Du={ds.Du}, P={ds.P}, Dd={ds.Dd}, Dz_max={ds.Dz_max}, Wx={ds.Wx}, Wy={ds.Wy}")

    # Sample indices (random)
    idxs = [random.randrange(len(ds)) for _ in range(min(args.num_samples, len(ds)))]

    for i, k in enumerate(idxs):
        lake_id, x_win, p, depth_feat, depth_mask, y = ds[k]
        lake_name = lake_datas[lake_id].name

        # Basic prints
        Dz_actual = int(depth_mask.sum().item())
        print(
            f"[{i}] idx={k} lake={lake_name} lake_id={lake_id} "
            f"x_win={tuple(x_win.shape)} p={tuple(p.shape)} "
            f"depth_feat={tuple(depth_feat.shape)} y={tuple(y.shape)} Dz_actual={Dz_actual}"
        )

        # Plot weather window
        _plot_weather_window(
            x_win,
            title=f"{lake_name} | sample {i} | weather window (Wx={args.Wx})",
            outpath=os.path.join(args.outdir, f"sample_{i:02d}_{lake_name}_weather.png"),
            max_dims=6,
        )

        # Plot target heatmap
        _plot_target_heatmap(
            y,
            depth_mask,
            title=f"{lake_name} | sample {i} | targets heatmap (Wy={args.Wy}, Dz_actual={Dz_actual})",
            outpath=os.path.join(args.outdir, f"sample_{i:02d}_{lake_name}_targets_heatmap.png"),
        )

        # Plot depth features + mask
        _plot_depth_features(
            depth_feat,
            depth_mask,
            title=f"{lake_name} | sample {i} | depth features + mask (Dz_max={ds.Dz_max})",
            outpath=os.path.join(args.outdir, f"sample_{i:02d}_{lake_name}_depth_features.png"),
        )

        # Optional: plot a single forecast step temperature profile
        # (use last step)
        yy = y.detach().cpu().numpy()  # (Wy, Dz_max)
        m = depth_mask.detach().cpu().numpy().astype(bool)
        profile = yy[-1, :].copy()
        profile[~m] = np.nan

        plt.figure(figsize=(6, 4))
        plt.plot(profile, np.arange(ds.Dz_max), marker="o")
        plt.gca().invert_yaxis()
        plt.title(f"{lake_name} | sample {i} | last-step profile (step Wy-1)")
        plt.xlabel("Normalized temperature")
        plt.ylabel("Depth index (padded)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, f"sample_{i:02d}_{lake_name}_profile_last_step.png"), dpi=150)
        plt.close()

    print(f"Saved plots to: {args.outdir}")


if __name__ == "__main__":
    main()
