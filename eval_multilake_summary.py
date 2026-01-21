#!/usr/bin/env python3
"""
eval_multilake_summary.py

Evaluate a trained multi-lake depth-conditioned emulator for ONE lake and
plot summary diagnostics.

Key features:
- split options: train | val | test | valtest | all
- plot_mode: spaghetti (many sims as faint lines)
- batch evaluation over windows (memory-friendly)
- optional export of denormalized params & weather for any sim

Assumes:
- models.py provides ModelRNNDepth
- multi_lake_dataset.py provides load_multi_lake_data and LakeData
- best.pt artifact contains: state_dict, config (vars(args)), etc.

Examples
--------
Spaghetti plots over FULL timeseries:
  python eval_multilake_summary.py \
    --run_id <RUN_ID> --root_dir data/parsed_data --lake BearLake \
    --split all --plot_mode spaghetti --spaghetti_sims 80 --show_mean

Save denorm inputs for sim 12:
  python eval_multilake_summary.py \
    --run_id <RUN_ID> --root_dir data/parsed_data --lake BearLake \
    --split all --plot_mode spaghetti --save_sim_inputs 12
"""

import os
import argparse
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from multi_lake_dataset import load_multi_lake_data

from emulator_utils import (
    build_model_from_ckpt_config,
    denorm_outputs,
    denorm_params,
    denorm_weather,
    load_checkpoint_from_mlflow,
    predict_timeseries_for_sim_batched,
    split_time_indices,
)


# =========================================================
# Plotting utilities
# =========================================================

def representative_depth_indices(Dz_lake: int) -> List[int]:
    if Dz_lake <= 1:
        return [0]
    if Dz_lake == 2:
        return [0, 1]
    return [0, Dz_lake // 2, Dz_lake - 1]


def parse_int_csv(s: str) -> List[int]:
    s = (s or "").strip()
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def plot_spaghetti(
    outdir: str,
    lake_name: str,
    split: str,
    run_id: str,
    year: np.ndarray,
    doy: np.ndarray,
    depths: List[int],
    preds_den_by_sim: Dict[int, np.ndarray],  # sim_id -> (T, Dz_lake)
    truth_den_mean: Optional[np.ndarray],     # (T, Dz_lake)
    pred_den_mean: Optional[np.ndarray],      # (T, Dz_lake)
    alpha: float,
    lw: float,
    show_mean: bool,
):
    os.makedirs(outdir, exist_ok=True)
    T = next(iter(preds_den_by_sim.values())).shape[0]

    # x-axis as index; if you prefer absolute labels, you can map (year,doy)->datetime
    x = np.arange(T)

    fig, axes = plt.subplots(len(depths), 1, figsize=(12, 3.8 * len(depths)), sharex=True)
    if len(depths) == 1:
        axes = [axes]

    for ax, d in zip(axes, depths):
        # faint lines per sim
        for sim_id, pred_den in preds_den_by_sim.items():
            ax.plot(x, pred_den[:, d], linewidth=lw, alpha=alpha)

        if show_mean and pred_den_mean is not None:
            ax.plot(x, pred_den_mean[:, d], linewidth=2.2, label="Mean Pred")

        if show_mean and truth_den_mean is not None:
            ax.plot(x, truth_den_mean[:, d], linewidth=2.2, label="Mean True")

        ax.set_ylabel(f"T (°C)\nDepth idx {d}")
        ax.grid(True, alpha=0.25)
        if show_mean:
            ax.legend(loc="upper right")

    axes[-1].set_xlabel("Time index (days in selected split)")
    fig.suptitle(f"{lake_name} | {split.upper()} | run={run_id} | spaghetti predictions", y=0.995)
    fig.tight_layout(rect=[0, 0.02, 1, 0.98])

    fname = os.path.join(outdir, f"{lake_name}_{split}_spaghetti_run_{run_id}.png")
    plt.savefig(fname, dpi=160)
    plt.close(fig)
    print(f"[saved] {fname}")


# =========================================================
# Main
# =========================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", required=True, help="MLflow run_id (contains checkpoints/best.pt)")
    ap.add_argument("--root_dir", default="data/parsed_data", help="Root directory containing lake subdirs")
    ap.add_argument("--lake", required=True, help="Lake name subdir (e.g. BearLake)")
    ap.add_argument("--all_lakes", nargs="+", default=None,
                    help="Optional: list of lakes to load for global norms. "
                         "If omitted, loads [lake] only. "
                         "To match training, pass the same lakes used in training.")
    ap.add_argument("--split_years", type=int, nargs=3, default=[2018, 2021, 2025])
    ap.add_argument("--split", default="all", choices=["train", "val", "test", "valtest", "all"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    ap.add_argument("--plot_mode", default="spaghetti", choices=["spaghetti"])
    ap.add_argument("--spaghetti_sims", type=int, default=80, help="Number of sims (parameter sets) to plot")
    ap.add_argument("--spaghetti_alpha", type=float, default=0.06)
    ap.add_argument("--spaghetti_lw", type=float, default=0.9)
    ap.add_argument("--show_mean", action="store_true", help="Overlay mean true & mean pred")

    ap.add_argument("--stride", type=int, default=1, help="Window stride for inference (>=1). Increase for speed.")
    ap.add_argument("--batch_windows", type=int, default=64, help="How many windows per forward pass")

    ap.add_argument("--depth_indices", default="", help="Comma list of depth indices to plot (default: surface/mid/bottom)")
    ap.add_argument("--outdir", default="plots_eval_summary")

    ap.add_argument("--save_sim_inputs", type=int, default=None,
                    help="If set, exports denormalized weather+params for this sim_id to outdir/inputs_*.npz")
    ap.add_argument("--seed", type=int, default=123)

    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ----- load checkpoint
    _, ckpt = load_checkpoint_from_mlflow(args.run_id)
    cfg = ckpt.get("config", {})
    Wx = int(cfg.get("Wx", 365))
    Wy = int(cfg.get("Wy", 30))

    # ----- load lakes (for norms consistency, load same set as training if possible)
    lakes_to_load = args.all_lakes if args.all_lakes is not None else [args.lake]
    lakes, norms = load_multi_lake_data(
        root_dir=args.root_dir,
        lake_names=list(lakes_to_load),
        split_years=list(args.split_years),
        normalize=True,
    )

    lake = None
    for lk in lakes:
        if lk.name == args.lake:
            lake = lk
            break
    if lake is None:
        raise ValueError(f"Requested lake='{args.lake}' not found in loaded lakes={lakes_to_load}")

    # ----- infer dims
    Du = lake.drivers.shape[1]
    P = lake.params.shape[1]
    Dd = lake.depth_feat.shape[1]
    Dz_lake = lake.Dz
    Dzmax = max(lk.Dz for lk in lakes)

    # padded depth_feat for this lake
    depth_feat_padded = np.zeros((Dzmax, Dd), dtype=np.float32)
    depth_feat_padded[:Dz_lake, :] = lake.depth_feat.astype(np.float32, copy=False)

    # ----- build model
    model = build_model_from_ckpt_config(cfg, Du=Du, P=P, Dd=Dd)
    model.load_state_dict(ckpt["state_dict"])
    model.to(args.device)
    model.eval()

    # ----- choose split indices
    split_years = tuple(args.split_years)
    t_idx = split_time_indices(lake.year, args.split, split_years)

    drivers_split = lake.drivers[t_idx, :]               # (T,Du) normalized
    temps_split = lake.temps[:, t_idx, :]                # (N,T,Dz_lake) normalized
    year_split = lake.year[t_idx]
    doy_split = lake.doy[t_idx]

    T_split = drivers_split.shape[0]
    N_sims = lake.params.shape[0]

    # ----- pick sims for spaghetti
    sims = list(range(N_sims))
    random.shuffle(sims)
    sims = sims[: min(args.spaghetti_sims, N_sims)]

    # ----- depths to plot
    if args.depth_indices.strip():
        depths = parse_int_csv(args.depth_indices)
    else:
        depths = representative_depth_indices(Dz_lake)

    # ----- optional export denorm inputs for one sim
    os.makedirs(args.outdir, exist_ok=True)
    if args.save_sim_inputs is not None:
        sid = int(args.save_sim_inputs)
        if not (0 <= sid < N_sims):
            raise ValueError(f"save_sim_inputs sim_id={sid} out of range [0,{N_sims-1}]")

        drivers_den = denorm_weather(drivers_split, norms)           # (T,Du)
        params_den = denorm_params(lake.params[sid:sid+1, :], norms) # (1,P)

        outpath = os.path.join(args.outdir, f"inputs_{args.lake}_split_{args.split}_sim_{sid}.npz")
        np.savez(
            outpath,
            year=year_split.astype(np.int32),
            doy=doy_split.astype(np.float32),
            weather_drivers=drivers_den.astype(np.float32),
            params=params_den.reshape(-1).astype(np.float32),
        )
        print(f"[saved] {outpath}")

    # ----- run predictions for chosen sims (batched over windows)
    preds_den_by_sim: Dict[int, np.ndarray] = {}
    for sid in sims:
        p_vec = lake.params[sid].astype(np.float32, copy=False)  # (P,) normalized
        pred_norm = predict_timeseries_for_sim_batched(
            model=model,
            drivers=drivers_split,
            p_vec=p_vec,
            depth_feat_padded=depth_feat_padded,
            Wx=Wx,
            Wy=Wy,
            stride=max(1, args.stride),
            device=args.device,
            batch_windows=max(1, args.batch_windows),
        )  # (T, Dzmax) norm

        pred_norm_lake = pred_norm[:, :Dz_lake]
        pred_den_lake = denorm_outputs(pred_norm_lake, norms, Dz=Dz_lake)  # (T,Dz_lake)
        preds_den_by_sim[sid] = pred_den_lake

        print(f"[pred] sim {sid} done")

    # ----- compute mean truth/pred (optional)
    truth_den_mean = None
    pred_den_mean = None
    if args.show_mean:
        # mean truth over all sims (or just the plotted ones; here we use all sims for stability)
        truth_norm_all = temps_split.astype(np.float32, copy=False)     # (N,T,Dz_lake)
        truth_den_all = denorm_outputs(truth_norm_all, norms, Dz=Dz_lake)
        truth_den_mean = np.nanmean(truth_den_all, axis=0)              # (T,Dz_lake)

        # mean pred over the plotted sims
        pred_stack = np.stack([preds_den_by_sim[sid] for sid in preds_den_by_sim.keys()], axis=0)  # (K,T,Dz)
        pred_den_mean = np.nanmean(pred_stack, axis=0)                   # (T,Dz_lake)

    # ----- plot spaghetti
    plot_spaghetti(
        outdir=args.outdir,
        lake_name=args.lake,
        split=args.split,
        run_id=args.run_id,
        year=year_split,
        doy=doy_split,
        depths=depths,
        preds_den_by_sim=preds_den_by_sim,
        truth_den_mean=truth_den_mean,
        pred_den_mean=pred_den_mean,
        alpha=float(args.spaghetti_alpha),
        lw=float(args.spaghetti_lw),
        show_mean=bool(args.show_mean),
    )

    print("Done.")


if __name__ == "__main__":
    main()
