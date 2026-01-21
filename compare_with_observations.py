#!/usr/bin/env python3
"""
compare_with_observations.py

Compare simulator outputs and emulator predictions to in-situ observations.

Produces:
 - per-simulation RMSE (sim vs obs)
 - per-simulation RMSE (emulator vs obs)
 - Plots per-lake histograms and scatter comparisons.

Usage example:
  python compare_with_observations.py \
    --ckpt <path_or_mlflow_run_id> \
    --root_dir data/parsed_data \
    --lakes BearLake RedPond \
    --obs_dir data/observations \
    --outdir outputs_compare

If ckpt is an MLflow run id (e.g. "7b835f..."), the script will attempt to download
artifacts/run/checkpoints/best.pt via MLflow client. Otherwise it will treat the
argument as a local filepath to a checkpoint.
"""
from __future__ import annotations
import os
import argparse
import json
from typing import Dict, List, Optional, Tuple
import math

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Local project modules (assumes these are importable in PYTHONPATH)
from multi_lake_dataset import load_multi_lake_data, LakeData
from emulator_utils import (
    build_model_from_ckpt_config,
    denorm_outputs,
    load_checkpoint,
    predict_timeseries_for_sim_batched,
    split_time_indices,
)


# ---------------------
# Utilities
# ---------------------
def parse_obs_csv(path: str) -> pd.DataFrame:
    """
    Expect CSV with columns: datetime, depth, temperature [, instrument ...]
    datetime may be YYYYMMDD or YYYY-MM-DD; result will include YEAR and DOY columns.
    """
    df = pd.read_csv(path)
    if "datetime" not in df.columns:
        raise ValueError(f"Observations file {path} missing 'datetime' column")

    # try several datetime formats
    def parse_dt(s):
        s = str(s)
        if "-" in s:
            try:
                return pd.to_datetime(s, format="%Y-%m-%d")
            except Exception:
                return pd.to_datetime(s, errors="coerce")
        else:
            # maybe YYYYMMDD
            try:
                return pd.to_datetime(s, format="%Y%m%d")
            except Exception:
                return pd.to_datetime(s, errors="coerce")

    df["dt_parsed"] = df["datetime"].apply(parse_dt)
    if df["dt_parsed"].isna().any():
        raise ValueError(f"Unable to parse some datetimes in {path}")

    df["YEAR"] = df["dt_parsed"].dt.year.astype(int)
    df["DOY"] = df["dt_parsed"].dt.dayofyear.astype(int)
    # Ensure columns exist
    if "depth" not in df.columns or "temperature" not in df.columns:
        raise ValueError(f"Observations file {path} must have 'depth' and 'temperature' columns")

    df["depth"] = df["depth"].astype(float)
    df["temperature"] = df["temperature"].astype(float)

    return df


def interp_profile_clamped(profile: np.ndarray, depth_grid: np.ndarray, z: float) -> float:
    """
    Interpolate a 1D profile (depth_grid ascending order) to depth z.
    Uses np.interp which clamps (returns edge value) for out-of-range z.
    profile: (Dz,)
    depth_grid: (Dz,) numeric depths
    z: scalar depth
    """
    # np.interp expects xp increasing; assume depth_grid is increasing (shallow->deep).
    return float(np.interp(z, depth_grid, profile))


# ------------------------
# Prediction helper (batched)
# ------------------------
@torch.no_grad()
def predict_split_for_sim(
    model: torch.nn.Module,
    lake: LakeData,
    sim_id: int,
    depth_feat_padded: np.ndarray,
    depth_mask: np.ndarray,
    split: str,
    split_years: Tuple[int, int, int],
    Wx: int,
    Wy: int,
    stride: int = 1,
    device: str = "cpu",
    batch_windows: int = 128,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      - truth: (T_split, Dz_lake) normalized
      - pred:  (T_split, Dz_max) normalized (NaN where no pred)
      - depth_mask: (Dz_max,)
    """
    t_idx = split_time_indices(lake.year, split, split_years)

    drivers = lake.drivers[t_idx, :]  # (T_split, Du)
    temps = lake.temps[sim_id, t_idx, :]  # (T_split, Dz_lake)
    T_split = drivers.shape[0]
    Dz_lake = temps.shape[1]
    Dzmax = depth_feat_padded.shape[0]

    if T_split < Wx:
        # no windows -> truth still returned, preds all NaN
        truth = np.zeros((T_split, Dz_lake), dtype=np.float32)
        truth[:, :] = temps.astype(np.float32, copy=False)
        pred = np.full((T_split, Dzmax), np.nan, dtype=np.float32)
        return truth, pred, depth_mask.astype(np.float32)

    pred = predict_timeseries_for_sim_batched(
        model=model,
        drivers=drivers,
        p_vec=lake.params[sim_id],
        depth_feat_padded=depth_feat_padded,
        Wx=Wx,
        Wy=Wy,
        stride=max(1, int(stride)),
        device=device,
        batch_windows=max(1, int(batch_windows)),
    )

    truth = temps.astype(np.float32, copy=False)
    return truth, pred, depth_mask.astype(np.float32)


# ------------------------
# RMSE computation aligning obs -> model days/depths
# ------------------------
def compute_rmse_for_obs(
    obs_df: pd.DataFrame,
    lake: LakeData,
    sim_truth: np.ndarray,
    sim_pred: np.ndarray,
    depth_grid: np.ndarray,
    split: str,
    split_years: Tuple[int, int, int],
) -> Tuple[float, int]:
    """
    For a single lake and a single simulation:
      - obs_df has YEAR, DOY, depth, temperature columns
      - sim_truth: (T_split, Dz_lake) in physical units (denormed by caller if desired)
      - sim_pred:  (T_split, Dz_lake) in physical units (denormed), may have NaNs
      - depth_grid: (Dz_lake,) numeric depths
    Returns RMSE computed on aligned observation rows (skips obs without matching day or with missing pred).
    """

    # select obs for this split by year
    tr_end, va_end, _ = split_years
    if split == "train":
        mask_time = obs_df["YEAR"] <= tr_end
    elif split == "val":
        mask_time = (obs_df["YEAR"] > tr_end) & (obs_df["YEAR"] <= va_end)
    elif split == "test":
        mask_time = obs_df["YEAR"] > va_end
    elif split == "all":
        mask_time = np.ones(len(obs_df), dtype=bool)
    else:
        raise ValueError("split must be train|val|test|all")

    obs_sel = obs_df[mask_time]
    if obs_sel.empty:
        return float("nan"), 0

    # IMPORTANT FIX:
    # Build mapping from (YEAR, DOY) -> *split-local* index into sim_truth/sim_pred
    full_t_idx = split_time_indices(lake.year, split, split_years)  # indices into full lake time
    year_split = lake.year[full_t_idx]
    doy_split = lake.doy[full_t_idx]

    mapping = {}
    for local_i, (y, d) in enumerate(zip(year_split.tolist(), doy_split.tolist())):
        mapping.setdefault((int(y), int(d)), local_i)

    Dz_lake = sim_truth.shape[1]

    errors = []
    n_used = 0
    for _, row in obs_sel.iterrows():
        y = int(row["YEAR"])
        d = int(row["DOY"])
        z = float(row["depth"])

        key = (y, d)
        if key not in mapping:
            continue  # no matching day in this split window

        t_idx = mapping[key]  # split-local index (guaranteed 0..T_split-1)
        prof_truth = sim_truth[t_idx, :Dz_lake]

        # interpolate/clamp to obs depth
        try:
            val_truth = interp_profile_clamped(prof_truth, depth_grid, z)
        except Exception:
            zi = int(min(max(round(z), 0), Dz_lake - 1))
            val_truth = float(prof_truth[zi])

        # emulator prediction at that day
        prof_pred = sim_pred[t_idx, :Dz_lake]
        if np.all(np.isnan(prof_pred)):
            continue

        # fill NaNs across depth if needed
        if np.isnan(prof_pred).any():
            valid_idx = np.where(~np.isnan(prof_pred))[0]
            if valid_idx.size == 0:
                continue
            xp = valid_idx
            fp = prof_pred[valid_idx]
            xi = np.arange(len(prof_pred))
            prof_pred = np.interp(xi, xp, fp).astype(np.float32)

        try:
            val_pred = interp_profile_clamped(prof_pred, depth_grid, z)
        except Exception:
            zi = int(min(max(round(z), 0), Dz_lake - 1))
            val_pred = float(prof_pred[zi])

        # RMSE between emulator and simulator truth at obs-aligned points
        err = float(val_pred - val_truth)
        errors.append(err)
        n_used += 1

    if n_used == 0:
        return float("nan"), 0
    errors = np.array(errors, dtype=np.float32)
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    return rmse, n_used


# ------------------------
# Main routine
# ------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to best.pt or MLflow run_id")
    ap.add_argument("--root_dir", default="data/parsed_data", help="Root with lake subdirs")
    ap.add_argument("--lakes", nargs="+", required=True, help="List of lake names (subdirs in root_dir)")
    ap.add_argument("--obs_dir", default="data/original_data", help="Directory with observation CSVs named <lake>.csv or <lake>_obs.csv")
    ap.add_argument("--outdir", default="compare_outputs", help="Where to save plots/CSV")
    ap.add_argument("--split_years", type=int, nargs=3, default=[2018, 2021, 2025])
    ap.add_argument("--split", default="test", choices=["train", "val", "test", "all"], help="Which split to compare on")
    ap.add_argument("--Wx", type=int, default=360)
    ap.add_argument("--Wy", type=int, default=60)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--batch_windows", type=int, default=128)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # 1) load checkpoint (config + state_dict + maybe norms)
    _, ckpt = load_checkpoint(args.ckpt)
    cfg = ckpt.get("config", {})
    # prefer Wx/Wy from checkpoint config if available
    Wx = int(cfg.get("Wx", args.Wx))
    Wy = int(cfg.get("Wy", args.Wy))

    # 2) load multi-lake data and global norms (normalize=True so loader returns norms)
    lakes, norms_dataset = load_multi_lake_data(
        root_dir=args.root_dir,
        lake_names=list(args.lakes),
        split_years=list(args.split_years),
        normalize=True,
    )

    # choose norms: prefer ckpt["norms"] if present (convenience), else use dataset norms
    norms_ckpt = ckpt.get("norms", None)
    norms = norms_ckpt if norms_ckpt is not None else norms_dataset

    # 3) rebuild model from ckpt config (we expect ModelRNNDepth)
    Du = lakes[0].drivers.shape[1]
    P = lakes[0].params.shape[1]
    Dd = lakes[0].depth_feat.shape[1]
    model = build_model_from_ckpt_config(cfg, Du=Du, P=P, Dd=Dd)
    model.load_state_dict(ckpt["state_dict"])
    model.to(args.device)
    model.eval()

    results_per_lake = {}

    for lk in lakes:
        print(f"[INFO] Processing lake {lk.name} (Dz={lk.Dz})")

        # depth grid (use integer indices starting at 0.5 similar to previous convention)
        depth_grid = (np.arange(lk.Dz, dtype=np.float32) + 0.5).astype(np.float32)

        # padded depth_feat and mask to Dz_max (for model input)
        Dzmax = max(l.Dz for l in lakes)
        Dd = lk.depth_feat.shape[1]
        df_pad = np.zeros((Dzmax, Dd), dtype=np.float32)
        df_pad[:lk.Dz, :] = lk.depth_feat.astype(np.float32, copy=False)
        depth_mask = np.zeros((Dzmax,), dtype=np.float32)
        depth_mask[:lk.Dz] = 1.0

        # load observations for this lake
        cand1 = os.path.join(args.obs_dir, f"{lk.name}/observations/{lk.name}-temperature-obs.csv")
        if os.path.isfile(cand1):
            obs_path = cand1
        else:
            print(f"[WARN] No observations file found for lake {lk.name} (tried {cand1}). Skipping.")
            continue
        obs_df = parse_obs_csv(obs_path)

        num_sims = lk.params.shape[0]
        sim_rmses_sim = np.full((num_sims,), np.nan, dtype=np.float32)
        sim_rmses_emul = np.full((num_sims,), np.nan, dtype=np.float32)
        sim_counts = np.zeros((num_sims,), dtype=int)

        for sim_id in range(num_sims):
            truth_norm, pred_norm, _ = predict_split_for_sim(
                model=model, lake=lk, sim_id=sim_id,
                depth_feat_padded=df_pad, depth_mask=depth_mask,
                split=args.split, split_years=tuple(args.split_years),
                Wx=Wx, Wy=Wy, stride=args.stride, device=args.device,
                batch_windows=args.batch_windows
            )
            pred_norm_slice = pred_norm[:, :lk.Dz]

            # denormalize to physical units if norms exist
            if norms is not None:
                truth_den = denorm_outputs(truth_norm, norms, Dz=lk.Dz)  # (T, Dz)
                pred_den_full = np.full((pred_norm.shape[0], lk.Dz), np.nan, dtype=np.float32)
                mask_valid_rows = ~np.all(np.isnan(pred_norm_slice), axis=1)
                if mask_valid_rows.any():
                    pred_den_full[mask_valid_rows, :] = denorm_outputs(
                        pred_norm_slice[mask_valid_rows, :], norms, Dz=lk.Dz
                    )
                else:
                    pred_den_full[:] = np.nan
            else:
                truth_den = truth_norm
                pred_den_full = pred_norm_slice

            rmse, n_used = compute_rmse_for_obs(
                obs_df=obs_df,
                lake=lk,
                sim_truth=truth_den,
                sim_pred=pred_den_full,
                depth_grid=depth_grid,
                split=args.split,
                split_years=tuple(args.split_years),
            )
            sim_rmses_emul[sim_id] = rmse
            sim_counts[sim_id] = n_used

        # Simulator-vs-obs RMSE (baseline) — computed properly
        def compute_rmse_obs_vs_simulator_for_sim(sim_index: int) -> Tuple[float, int]:
            truth_norm_sim, _, _ = predict_split_for_sim(
                model=model, lake=lk, sim_id=sim_index,
                depth_feat_padded=df_pad, depth_mask=depth_mask,
                split=args.split, split_years=tuple(args.split_years),
                Wx=Wx, Wy=Wy, stride=args.stride, device=args.device,
                batch_windows=args.batch_windows
            )
            truth_den_sim = denorm_outputs(truth_norm_sim, norms, Dz=lk.Dz) if norms is not None else truth_norm_sim

            # obs split selection by year
            tr_end, va_end, _ = tuple(args.split_years)
            if args.split == "train":
                obs_sel = obs_df[obs_df["YEAR"] <= tr_end]
            elif args.split == "val":
                obs_sel = obs_df[(obs_df["YEAR"] > tr_end) & (obs_df["YEAR"] <= va_end)]
            elif args.split == "test":
                obs_sel = obs_df[obs_df["YEAR"] > va_end]
            else:
                obs_sel = obs_df

            if obs_sel.empty:
                return float("nan"), 0

            # IMPORTANT FIX: split-local mapping
            full_t_idx = _split_time_indices(lk, args.split, tuple(args.split_years))
            year_split = lk.year[full_t_idx]
            doy_split = lk.doy[full_t_idx]
            mapping = {}
            for local_i, (y, d) in enumerate(zip(year_split.tolist(), doy_split.tolist())):
                mapping.setdefault((int(y), int(d)), local_i)

            errors = []
            n_used_local = 0
            for _, row in obs_sel.iterrows():
                key = (int(row["YEAR"]), int(row["DOY"]))
                if key not in mapping:
                    continue
                t_idx = mapping[key]  # split-local
                z = float(row["depth"])
                obs_val = float(row["temperature"])

                prof_sim = truth_den_sim[t_idx, :lk.Dz]
                sim_at_z = interp_profile_clamped(prof_sim, depth_grid, z)
                errors.append(sim_at_z - obs_val)
                n_used_local += 1

            if n_used_local == 0:
                return float("nan"), 0
            err_arr = np.array(errors, dtype=np.float32)
            return float(np.sqrt(np.mean(err_arr ** 2))), n_used_local

        for sim_id in range(num_sims):
            base_rmse, _ = compute_rmse_obs_vs_simulator_for_sim(sim_id)
            sim_rmses_sim[sim_id] = base_rmse

        results_per_lake[lk.name] = {
            "sim_rmse": sim_rmses_sim,
            "emul_rmse": sim_rmses_emul,
            "counts": sim_counts,
        }

        valid_mask = (sim_counts > 0)
        x = sim_rmses_sim[valid_mask]
        y = sim_rmses_emul[valid_mask]

        # Histogram
        plt.figure(figsize=(6, 4))
        plt.hist(x[~np.isnan(x)], bins=40, alpha=0.7, label="sim")
        plt.hist(y[~np.isnan(y)], bins=40, alpha=0.7, label="emul")
        plt.legend()
        plt.title(f"{lk.name} RMSE distribution (split={args.split})")
        plt.xlabel("RMSE (°C)")
        plt.ylabel("count")
        out_hist = os.path.join(args.outdir, f"{lk.name}_rmse_hist_{args.split}.png")
        plt.tight_layout()
        plt.savefig(out_hist, dpi=150)
        plt.close()
        print(f"[saved] {out_hist}")

        # Scatter
        plt.figure(figsize=(6, 6))
        plt.scatter(x, y, alpha=0.6)
        maxval = np.nanmax(np.concatenate([x[~np.isnan(x)], y[~np.isnan(y)], np.array([0.1])]))
        plt.plot([0, maxval], [0, maxval], linestyle="--", color="k")
        plt.xlabel("Simulator RMSE (°C)")
        plt.ylabel("Emulator RMSE (°C)")
        plt.title(f"{lk.name} per-sim RMSE (split={args.split})")
        out_scatter = os.path.join(args.outdir, f"{lk.name}_rmse_scatter_{args.split}.png")
        plt.tight_layout()
        plt.savefig(out_scatter, dpi=150)
        plt.close()
        print(f"[saved] {out_scatter}")

        # CSV
        df_out = pd.DataFrame({
            "sim_id": np.arange(num_sims),
            "sim_rmse": sim_rmses_sim,
            "emul_rmse": sim_rmses_emul,
            "n_obs_used": sim_counts,
        })
        csv_out = os.path.join(args.outdir, f"{lk.name}_rmse_per_sim_{args.split}.csv")
        df_out.to_csv(csv_out, index=False)
        print(f"[saved] {csv_out}")

    print("All done.")


if __name__ == "__main__":
    main()
