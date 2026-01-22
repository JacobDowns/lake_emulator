#!/usr/bin/env python3
"""
compare_with_observations.py

Load a trained emulator from an MLflow `run_id`, load a lake's in-situ temperature
observations, align observation timestamps to the emulator's `(YEAR, DOY)` time axis,
and compare:
  - simulator (precomputed) vs observations
  - emulator predictions vs observations

Observations are expected at:
  `data/original_data/{lake}/observations/{lake}-temperature-obs.csv`

This script is intentionally structured like `eval_multilake_summary.py` and can be
extended later with richer plots/diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from multi_lake_dataset import LakeData, load_multi_lake_data
from emulator_utils import (
    build_model_from_ckpt_config,
    denorm_outputs,
    load_checkpoint_from_mlflow,
    predict_timeseries_for_sim_batched,
    split_time_indices,
)


# =========================================================
# Observations: parsing + alignment
# =========================================================

@dataclass(frozen=True)
class Observations:
    year: np.ndarray        # (M,) int
    doy: np.ndarray         # (M,) float (may be fractional)
    doy_key: np.ndarray     # (M,) int (used for alignment to daily model axis by default)
    depth: np.ndarray       # (M,) float
    temp: np.ndarray        # (M,) float


def _parse_obs_datetime(s: str) -> dt.datetime:
    s = str(s).strip()
    if not s:
        raise ValueError("empty datetime")

    # Common compact encodings: YYYYMMDD, YYYYMMDDHHMM, YYYYMMDDHHMMSS
    if s.isdigit():
        if len(s) == 8:
            return dt.datetime.strptime(s, "%Y%m%d")
        if len(s) == 12:
            return dt.datetime.strptime(s, "%Y%m%d%H%M")
        if len(s) == 14:
            return dt.datetime.strptime(s, "%Y%m%d%H%M%S")

    # Try ISO-ish variants (including with time)
    for fmt in (
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M",
        "%Y/%m/%d %H:%M:%S",
        "%m/%d/%Y",
        "%m/%d/%Y %H:%M",
        "%m/%d/%Y %H:%M:%S",
    ):
        try:
            return dt.datetime.strptime(s, fmt)
        except ValueError:
            pass

    # Last resort: datetime.fromisoformat (handles "YYYY-MM-DDTHH:MM:SS")
    try:
        return dt.datetime.fromisoformat(s.replace("Z", ""))
    except ValueError as e:
        raise ValueError(f"unrecognized datetime format: {s!r}") from e


def load_observations_csv(path: str) -> Observations:
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        required = {"datetime", "depth", "temperature"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} missing required columns: {sorted(missing)}")

        n_skipped_missing = 0
        n_skipped_parse = 0

        years: List[int] = []
        doys: List[float] = []
        doy_keys: List[int] = []
        depths: List[float] = []
        temps: List[float] = []

        for row in reader:
            dt_raw = (row.get("datetime", "") or "").strip()
            depth_raw = (row.get("depth", "") or "").strip()
            temp_raw = (row.get("temperature", "") or "").strip()

            if (not dt_raw) or (not depth_raw) or (not temp_raw):
                n_skipped_missing += 1
                continue

            try:
                dtt = _parse_obs_datetime(dt_raw)
                depth = float(depth_raw)
                temp = float(temp_raw)
            except Exception:
                n_skipped_parse += 1
                continue

            doy_int = int(dtt.timetuple().tm_yday)
            frac = (dtt.hour * 3600 + dtt.minute * 60 + dtt.second) / 86400.0
            doy_float = float(doy_int) + float(frac)

            years.append(int(dtt.year))
            doys.append(doy_float)
            doy_keys.append(doy_int)
            depths.append(depth)
            temps.append(temp)

    if n_skipped_missing or n_skipped_parse:
        print(
            f"[obs] skipped rows in {path}: missing_required_fields={n_skipped_missing}, parse_errors={n_skipped_parse}"
        )

    return Observations(
        year=np.asarray(years, dtype=np.int32),
        doy=np.asarray(doys, dtype=np.float32),
        doy_key=np.asarray(doy_keys, dtype=np.int32),
        depth=np.asarray(depths, dtype=np.float32),
        temp=np.asarray(temps, dtype=np.float32),
    )


def _obs_split_mask(obs_year: np.ndarray, split: str, split_years: Tuple[int, int, int]) -> np.ndarray:
    tr_end, va_end, _ = split_years
    if split == "train":
        return obs_year <= tr_end
    if split == "val":
        return (obs_year > tr_end) & (obs_year <= va_end)
    if split == "test":
        return obs_year > va_end
    if split == "valtest":
        return obs_year > tr_end
    if split == "all":
        return np.ones_like(obs_year, dtype=bool)
    raise ValueError("split must be one of train|val|test|valtest|all")


def build_time_index_map(
    lake: LakeData,
    split: str,
    split_years: Tuple[int, int, int],
    *,
    doy_round: str,
) -> Dict[Tuple[int, int], int]:
    t_idx = split_time_indices(lake.year, split, split_years)
    year_split = lake.year[t_idx].astype(np.int32, copy=False)
    doy_split = lake.doy[t_idx].astype(np.float32, copy=False)

    if doy_round == "floor":
        doy_key = np.floor(doy_split).astype(np.int32)
    elif doy_round == "ceil":
        doy_key = np.ceil(doy_split).astype(np.int32)
    elif doy_round == "round":
        doy_key = np.round(doy_split).astype(np.int32)
    else:
        raise ValueError("doy_round must be one of floor|round|ceil")

    mapping: Dict[Tuple[int, int], int] = {}
    for local_i, (y, d) in enumerate(zip(year_split.tolist(), doy_key.tolist())):
        mapping.setdefault((int(y), int(d)), int(local_i))
    return mapping


def align_observations(
    obs: Observations,
    lake: LakeData,
    split: str,
    split_years: Tuple[int, int, int],
    *,
    doy_round: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    """
    Returns aligned arrays (all split-local):
      t_local: (M_aligned,) indices into split-local timeseries
      depth:   (M_aligned,)
      temp:    (M_aligned,)
    """
    mapping = build_time_index_map(lake, split, split_years, doy_round=doy_round)

    mask_split = _obs_split_mask(obs.year, split, split_years)
    years = obs.year[mask_split]
    doy_keys = obs.doy_key[mask_split]
    depths = obs.depth[mask_split]
    temps = obs.temp[mask_split]

    t_local: List[int] = []
    depth_aligned: List[float] = []
    temp_aligned: List[float] = []
    n_missing_time = 0

    for y, d, z, temp in zip(years.tolist(), doy_keys.tolist(), depths.tolist(), temps.tolist()):
        key = (int(y), int(d))
        if key not in mapping:
            n_missing_time += 1
            continue
        t_local.append(mapping[key])
        depth_aligned.append(float(z))
        temp_aligned.append(float(temp))

    stats = {
        "n_obs_total": int(obs.year.shape[0]),
        "n_obs_split": int(years.shape[0]),
        "n_obs_aligned": int(len(t_local)),
        "n_obs_missing_time": int(n_missing_time),
    }

    return (
        np.asarray(t_local, dtype=np.int32),
        np.asarray(depth_aligned, dtype=np.float32),
        np.asarray(temp_aligned, dtype=np.float32),
        stats,
    )


# =========================================================
# RMSE at observation points
# =========================================================

def interp_profile_clamped(profile: np.ndarray, depth_grid: np.ndarray, z: float) -> float:
    return float(np.interp(float(z), depth_grid, profile))


def year_doy_to_datetime(year: int, doy: float) -> dt.datetime:
    d = int(doy)
    d = max(1, min(d, 366))
    return dt.datetime(int(year), 1, 1) + dt.timedelta(days=d - 1)


def residuals_at_obs_points(
    y_ts: np.ndarray,           # (T_split, Dz)
    *,
    t_local: np.ndarray,        # (M,)
    obs_depth: np.ndarray,      # (M,)
    obs_temp: np.ndarray,       # (M,)
    depth_grid: np.ndarray,     # (Dz,)
    skip_if_all_nan: bool,
) -> np.ndarray:
    """
    Returns residuals (pred - obs) for each aligned observation point.
    Residual is NaN if the prediction is unavailable at that point.
    """
    res = np.full((t_local.shape[0],), np.nan, dtype=np.float32)

    for i, (ti, z, ot) in enumerate(zip(t_local.tolist(), obs_depth.tolist(), obs_temp.tolist())):
        prof = y_ts[int(ti)]
        if skip_if_all_nan and bool(np.all(np.isnan(prof))):
            continue

        if np.isnan(prof).any():
            valid_idx = np.where(~np.isnan(prof))[0]
            if valid_idx.size == 0:
                continue
            prof = np.interp(np.arange(prof.shape[0]), valid_idx, prof[valid_idx]).astype(np.float32)

        pred_at_z = interp_profile_clamped(prof, depth_grid, float(z))
        res[i] = float(pred_at_z - float(ot))

    return res


def rmse_timeseries_against_obs(
    y_ts: np.ndarray,           # (T_split, Dz)
    *,
    t_local: np.ndarray,        # (M,)
    obs_depth: np.ndarray,      # (M,)
    obs_temp: np.ndarray,       # (M,)
    depth_grid: np.ndarray,     # (Dz,)
    skip_if_all_nan: bool,
) -> Tuple[float, int]:
    errors: List[float] = []

    for ti, z, ot in zip(t_local.tolist(), obs_depth.tolist(), obs_temp.tolist()):
        prof = y_ts[int(ti)]
        if skip_if_all_nan and bool(np.all(np.isnan(prof))):
            continue

        # fill NaNs across depth if present (common for padded/masked depths)
        if np.isnan(prof).any():
            valid_idx = np.where(~np.isnan(prof))[0]
            if valid_idx.size == 0:
                continue
            prof = np.interp(np.arange(prof.shape[0]), valid_idx, prof[valid_idx]).astype(np.float32)

        pred_at_z = interp_profile_clamped(prof, depth_grid, float(z))
        errors.append(float(pred_at_z - float(ot)))

    if not errors:
        return float("nan"), 0
    err = np.asarray(errors, dtype=np.float32)
    return float(np.sqrt(np.mean(err ** 2))), int(err.shape[0])


# =========================================================
# Main
# =========================================================

def _parse_int_csv(s: str) -> List[int]:
    s = (s or "").strip()
    if not s:
        return []
    out: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", required=True, help="MLflow run_id (contains checkpoints/best.pt)")
    ap.add_argument("--root_dir", default="data/parsed_data", help="Root directory containing lake subdirs")
    ap.add_argument("--lake", required=True, help="Lake name subdir (e.g. BearLake)")
    ap.add_argument(
        "--all_lakes",
        nargs="+",
        default=None,
        help="Optional: list of lakes to load for global norms (should match training lakes).",
    )

    ap.add_argument("--split_years", type=int, nargs=3, default=[2018, 2021, 2025])
    ap.add_argument("--split", default="all", choices=["train", "val", "test", "valtest", "all"])
    ap.add_argument("--doy_round", default="round", choices=["floor", "round", "ceil"])

    ap.add_argument(
        "--obs_path",
        default=None,
        help="Override observation CSV path; default is data/original_data/{lake}/observations/{lake}-temperature-obs.csv",
    )
    ap.add_argument("--outdir", default="outputs_compare")

    ap.add_argument("--stride", type=int, default=1, help="Window stride for inference (>=1). Increase for speed.")
    ap.add_argument("--batch_windows", type=int, default=64, help="How many windows per forward pass")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    ap.add_argument(
        "--sim_ids",
        default="",
        help="Optional comma list of sim_ids to evaluate (default: sample up to --max_sims).",
    )
    ap.add_argument("--max_sims", type=int, default=200, help="Max sims to evaluate if --sim_ids is empty")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--no_simulator", action="store_true", help="Skip simulator-vs-obs RMSE")
    ap.add_argument("--no_emulator", action="store_true", help="Skip emulator-vs-obs RMSE")

    ap.add_argument("--plot_misfit_maps", action="store_true", help="Save (t,z) misfit maps for simulator and emulator")
    ap.add_argument("--misfit_mode", default="abs", choices=["abs", "signed"], help="Color by mean |residual| or mean residual")
    ap.add_argument("--misfit_s", type=float, default=8.0, help="Marker size for misfit scatter plots")

    ap.add_argument("--plot_spaghetti", action="store_true", help="Save spaghetti plots at common observation depths")
    ap.add_argument("--spaghetti_depths", type=int, default=3, help="How many depths (by obs frequency) to plot")
    ap.add_argument("--depth_tol", type=float, default=0.15, help="Tolerance (m) for matching obs to a spaghetti depth")
    ap.add_argument("--spaghetti_max_sims", type=int, default=30, help="Max sims (lines) in spaghetti plots")
    ap.add_argument("--spaghetti_alpha", type=float, default=0.08, help="Alpha for spaghetti lines")
    ap.add_argument("--spaghetti_lw", type=float, default=0.8, help="Linewidth for spaghetti lines")

    ap.add_argument(
        "--plot_binned_obs_emul_spaghetti",
        action="store_true",
        help="Plot observations as depth-bin scatters and emulator spaghetti at bin midpoints over the observation period",
    )
    ap.add_argument("--bin_size_m", type=float, default=1.0, help="Depth bin size (meters) for binned obs plot")
    ap.add_argument("--max_bins", type=int, default=12, help="Max number of depth bins/subplots to draw")

    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.outdir, exist_ok=True)

    # ----- load checkpoint
    _, ckpt = load_checkpoint_from_mlflow(args.run_id)
    cfg = ckpt.get("config", {})
    Wx = int(cfg.get("Wx", 365))
    Wy = int(cfg.get("Wy", 30))

    # ----- load data (optionally multiple lakes for norms consistency)
    lakes_to_load = args.all_lakes if args.all_lakes is not None else [args.lake]
    lakes, norms = load_multi_lake_data(
        root_dir=args.root_dir,
        lake_names=list(lakes_to_load),
        split_years=list(args.split_years),
        normalize=True,
    )

    lake: Optional[LakeData] = None
    for lk in lakes:
        if lk.name == args.lake:
            lake = lk
            break
    if lake is None:
        raise ValueError(f"Requested lake='{args.lake}' not found in loaded lakes={lakes_to_load}")

    # ----- infer dims & build model
    Du = lake.drivers.shape[1]
    P = lake.params.shape[1]
    Dd = lake.depth_feat.shape[1]
    Dz_lake = lake.Dz
    Dzmax = max(lk.Dz for lk in lakes)

    model = build_model_from_ckpt_config(cfg, Du=Du, P=P, Dd=Dd)
    model.load_state_dict(ckpt["state_dict"])
    model.to(args.device)
    model.eval()

    depth_feat_padded = np.zeros((Dzmax, Dd), dtype=np.float32)
    depth_feat_padded[:Dz_lake, :] = lake.depth_feat.astype(np.float32, copy=False)

    # ----- load observations
    obs_path = args.obs_path or os.path.join(
        "data", "original_data", args.lake, "observations", f"{args.lake}-temperature-obs.csv"
    )
    if not os.path.isfile(obs_path):
        raise FileNotFoundError(f"Could not find observations at {obs_path}")
    obs = load_observations_csv(obs_path)

    # ----- align observations to split-local timeseries indices
    t_local, obs_depth, obs_temp, obs_stats = align_observations(
        obs, lake, args.split, tuple(args.split_years), doy_round=args.doy_round
    )
    print(f"[obs] {args.lake} {args.split}: {obs_stats}")
    if t_local.size == 0:
        raise RuntimeError("No aligned observations found after YEAR/DOY alignment.")

    # ----- select sims
    sim_ids = _parse_int_csv(args.sim_ids)
    if sim_ids:
        sim_ids = [i for i in sim_ids if 0 <= i < int(lake.params.shape[0])]
    else:
        all_ids = list(range(int(lake.params.shape[0])))
        random.shuffle(all_ids)
        sim_ids = all_ids[: int(min(args.max_sims, len(all_ids)))]
    sim_ids = sorted(sim_ids)

    # ----- depth grid convention
    # We assume bathymetry bins correspond to 1m depth increments; use bin midpoints.
    depth_grid = (np.arange(Dz_lake, dtype=np.float32) + 0.5).astype(np.float32)

    # ----- precompute split indices for simulator slices / emulator drivers
    t_idx_split = split_time_indices(lake.year, args.split, tuple(args.split_years))
    drivers_split = lake.drivers[t_idx_split, :]
    year_split = lake.year[t_idx_split].astype(np.int32, copy=False)
    doy_split = lake.doy[t_idx_split].astype(np.float32, copy=False)
    time_split_dt = np.asarray([year_doy_to_datetime(int(y), float(d)) for y, d in zip(year_split, doy_split)])

    # ----- evaluate
    rows = []
    residuals_sim_by_sim: List[np.ndarray] = []
    residuals_emul_by_sim: List[np.ndarray] = []
    for sim_id in sim_ids:
        row = {"sim_id": int(sim_id)}

        if not args.no_simulator:
            y_sim_norm = lake.temps[int(sim_id), t_idx_split, :]  # (T_split, Dz_lake) normalized
            y_sim = denorm_outputs(y_sim_norm, norms, Dz=Dz_lake) if norms is not None else y_sim_norm
            residuals_sim_by_sim.append(
                residuals_at_obs_points(
                    y_sim,
                    t_local=t_local,
                    obs_depth=obs_depth,
                    obs_temp=obs_temp,
                    depth_grid=depth_grid,
                    skip_if_all_nan=False,
                )
            )
            rmse_sim, n_sim = rmse_timeseries_against_obs(
                y_sim,
                t_local=t_local,
                obs_depth=obs_depth,
                obs_temp=obs_temp,
                depth_grid=depth_grid,
                skip_if_all_nan=False,
            )
            row.update({"rmse_sim_obs": float(rmse_sim), "n_obs_sim": int(n_sim)})

        if not args.no_emulator:
            if drivers_split.shape[0] < Wx:
                raise ValueError(f"Split has T={drivers_split.shape[0]} < Wx={Wx}; cannot run emulator windows.")
            y_pred_norm = predict_timeseries_for_sim_batched(
                model=model,
                drivers=drivers_split,
                p_vec=lake.params[int(sim_id)],
                depth_feat_padded=depth_feat_padded,
                Wx=Wx,
                Wy=Wy,
                stride=max(1, int(args.stride)),
                device=args.device,
                batch_windows=max(1, int(args.batch_windows)),
            )  # (T_split, Dzmax) normalized
            y_pred_norm = y_pred_norm[:, :Dz_lake]
            y_pred = denorm_outputs(y_pred_norm, norms, Dz=Dz_lake) if norms is not None else y_pred_norm

            residuals_emul_by_sim.append(
                residuals_at_obs_points(
                    y_pred,
                    t_local=t_local,
                    obs_depth=obs_depth,
                    obs_temp=obs_temp,
                    depth_grid=depth_grid,
                    skip_if_all_nan=True,
                )
            )
            rmse_emul, n_emul = rmse_timeseries_against_obs(
                y_pred,
                t_local=t_local,
                obs_depth=obs_depth,
                obs_temp=obs_temp,
                depth_grid=depth_grid,
                skip_if_all_nan=True,
            )
            row.update({"rmse_emul_obs": float(rmse_emul), "n_obs_emul": int(n_emul)})

        rows.append(row)

    # ----- save CSV
    csv_path = os.path.join(args.outdir, f"{args.lake}_rmse_per_sim_{args.split}_run_{args.run_id}.csv")
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"[saved] {csv_path}")

    # ----- quick plots
    def _col(name: str) -> np.ndarray:
        vals = [r.get(name, float("nan")) for r in rows]
        return np.asarray(vals, dtype=np.float32)

    rmse_sim = _col("rmse_sim_obs")
    rmse_emul = _col("rmse_emul_obs")

    if not args.no_simulator and np.isfinite(rmse_sim).any():
        plt.figure(figsize=(7, 4))
        plt.hist(rmse_sim[np.isfinite(rmse_sim)], bins=40, alpha=0.8, label="sim vs obs")
        plt.xlabel("RMSE (°C)")
        plt.ylabel("count")
        plt.title(f"{args.lake} | {args.split} | run={args.run_id} | simulator")
        plt.tight_layout()
        p = os.path.join(args.outdir, f"{args.lake}_{args.split}_hist_sim_run_{args.run_id}.png")
        plt.savefig(p, dpi=160)
        plt.close()
        print(f"[saved] {p}")

    if not args.no_emulator and np.isfinite(rmse_emul).any():
        plt.figure(figsize=(7, 4))
        plt.hist(rmse_emul[np.isfinite(rmse_emul)], bins=40, alpha=0.8, label="emulator vs obs")
        plt.xlabel("RMSE (°C)")
        plt.ylabel("count")
        plt.title(f"{args.lake} | {args.split} | run={args.run_id} | emulator")
        plt.tight_layout()
        p = os.path.join(args.outdir, f"{args.lake}_{args.split}_hist_emul_run_{args.run_id}.png")
        plt.savefig(p, dpi=160)
        plt.close()
        print(f"[saved] {p}")

    if (not args.no_simulator) and (not args.no_emulator):
        mask = np.isfinite(rmse_sim) & np.isfinite(rmse_emul)
        if mask.any():
            plt.figure(figsize=(6, 6))
            plt.scatter(rmse_sim[mask], rmse_emul[mask], alpha=0.65)
            mx = float(np.nanmax(np.concatenate([rmse_sim[mask], rmse_emul[mask], np.asarray([0.1], np.float32)])))
            plt.plot([0.0, mx], [0.0, mx], "k--", linewidth=1.0)
            plt.xlabel("Simulator RMSE vs obs (°C)")
            plt.ylabel("Emulator RMSE vs obs (°C)")
            plt.title(f"{args.lake} | {args.split} | run={args.run_id}")
            plt.tight_layout()
            p = os.path.join(args.outdir, f"{args.lake}_{args.split}_scatter_sim_vs_emul_run_{args.run_id}.png")
            plt.savefig(p, dpi=160)
            plt.close()
            print(f"[saved] {p}")

    # ----- misfit maps: average misfit at each observation point across sims
    if args.plot_misfit_maps:
        t_obs_dt = np.asarray([time_split_dt[int(ti)] for ti in t_local.tolist()])
        z_obs = obs_depth.astype(np.float32, copy=False)

        fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.3), sharey=True, constrained_layout=True)
        panels: List[Tuple[str, List[np.ndarray], bool]] = [
            ("Simulator", residuals_sim_by_sim, not args.no_simulator),
            ("Emulator", residuals_emul_by_sim, not args.no_emulator),
        ]

        for ax, (title, res_list, enabled) in zip(axes, panels):
            if (not enabled) or (not res_list):
                ax.set_title(f"{title} (skipped)")
                ax.set_xlabel("Time")
                ax.grid(True, alpha=0.2)
                continue

            res_stack = np.stack(res_list, axis=0)  # (S, M)
            if args.misfit_mode == "abs":
                vals = np.nanmean(np.abs(res_stack), axis=0)
                cmap = "viridis"
                vmin, vmax = None, None
                clabel = "Mean |pred - obs| (°C)"
            else:
                vals = np.nanmean(res_stack, axis=0)
                cmap = "coolwarm"
                vmax = float(np.nanmax(np.abs(vals))) if np.isfinite(vals).any() else 1.0
                vmin = -vmax
                clabel = "Mean (pred - obs) (°C)"

            m = np.isfinite(vals)
            sc = ax.scatter(
                t_obs_dt[m],
                z_obs[m],
                c=vals[m],
                s=float(args.misfit_s),
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                alpha=0.9,
                linewidths=0.0,
            )
            ax.set_title(title)
            ax.set_xlabel("Time")
            ax.grid(True, alpha=0.2)
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
            cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
            cb.set_label(clabel)

        axes[0].set_ylabel("Depth (m)")
        axes[0].invert_yaxis()
        fname = os.path.join(
            args.outdir,
            f"{args.lake}_{args.split}_misfit_maps_{args.misfit_mode}_run_{args.run_id}.png",
        )
        plt.savefig(fname, dpi=180)
        plt.close(fig)
        print(f"[saved] {fname}")

    # ----- spaghetti plots: common observation depths with obs overlay
    if args.plot_spaghetti:
        # choose depths by frequency among aligned obs points
        depth_vals, depth_counts = np.unique(obs_depth.astype(np.float32), return_counts=True)
        order = np.argsort(-depth_counts)
        top_depths = depth_vals[order][: int(max(1, args.spaghetti_depths))].tolist()

        # keep plot readable: subsample sims for spaghetti
        spaghetti_sim_ids = sim_ids[: int(min(len(sim_ids), args.spaghetti_max_sims))]

        def plot_spaghetti_panel(
            *,
            title: str,
            y_by_sim: Dict[int, np.ndarray],  # sim_id -> (T_split, Dz_lake)
            outpath: str,
        ) -> None:
            fig, axes = plt.subplots(len(top_depths), 1, figsize=(13.5, 3.3 * len(top_depths)), sharex=True)
            if len(top_depths) == 1:
                axes = [axes]

            for ax, z0 in zip(axes, top_depths):
                # spaghetti lines
                for sid, y_ts in y_by_sim.items():
                    # interpolate time series at z0
                    series = np.full((y_ts.shape[0],), np.nan, dtype=np.float32)
                    for ti in range(y_ts.shape[0]):
                        prof = y_ts[ti]
                        if np.all(np.isnan(prof)):
                            continue
                        if np.isnan(prof).any():
                            valid_idx = np.where(~np.isnan(prof))[0]
                            if valid_idx.size == 0:
                                continue
                            prof = np.interp(np.arange(prof.shape[0]), valid_idx, prof[valid_idx]).astype(np.float32)
                        series[ti] = interp_profile_clamped(prof, depth_grid, float(z0))
                    ax.plot(
                        time_split_dt,
                        series,
                        linewidth=float(args.spaghetti_lw),
                        alpha=float(args.spaghetti_alpha),
                        color="tab:blue",
                    )

                # overlay observations near this depth
                m = np.abs(obs_depth - float(z0)) <= float(args.depth_tol)
                if bool(np.any(m)):
                    t_obs_dt = np.asarray([time_split_dt[int(ti)] for ti in t_local[m].tolist()])
                    ax.scatter(t_obs_dt, obs_temp[m], s=14.0, color="black", alpha=0.8, label="obs")

                ax.set_ylabel(f"T (°C)\nz≈{float(z0):g}m")
                ax.grid(True, alpha=0.2)
                ax.legend(loc="best")

            axes[-1].set_xlabel("Time")
            axes[-1].xaxis.set_major_locator(mdates.AutoDateLocator())
            axes[-1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(axes[-1].xaxis.get_major_locator()))
            fig.suptitle(title, y=0.995)
            plt.tight_layout(rect=[0, 0.02, 1, 0.98])
            plt.savefig(outpath, dpi=180)
            plt.close(fig)
            print(f"[saved] {outpath}")

        if not args.no_simulator:
            y_sim_by_sim: Dict[int, np.ndarray] = {}
            for sid in spaghetti_sim_ids:
                y_sim_norm = lake.temps[int(sid), t_idx_split, :]
                y_sim = denorm_outputs(y_sim_norm, norms, Dz=Dz_lake) if norms is not None else y_sim_norm
                y_sim_by_sim[int(sid)] = y_sim
            outpath = os.path.join(args.outdir, f"{args.lake}_{args.split}_spaghetti_sim_run_{args.run_id}.png")
            plot_spaghetti_panel(
                title=f"{args.lake} | {args.split} | run={args.run_id} | simulator spaghetti (n={len(spaghetti_sim_ids)})",
                y_by_sim=y_sim_by_sim,
                outpath=outpath,
            )

        if not args.no_emulator:
            y_emul_by_sim: Dict[int, np.ndarray] = {}
            if drivers_split.shape[0] < Wx:
                print(f"[WARN] Skipping emulator spaghetti: split has T={drivers_split.shape[0]} < Wx={Wx}")
            else:
                for sid in spaghetti_sim_ids:
                    y_pred_norm = predict_timeseries_for_sim_batched(
                        model=model,
                        drivers=drivers_split,
                        p_vec=lake.params[int(sid)],
                        depth_feat_padded=depth_feat_padded,
                        Wx=Wx,
                        Wy=Wy,
                        stride=max(1, int(args.stride)),
                        device=args.device,
                        batch_windows=max(1, int(args.batch_windows)),
                    )
                    y_pred_norm = y_pred_norm[:, :Dz_lake]
                    y_pred = denorm_outputs(y_pred_norm, norms, Dz=Dz_lake) if norms is not None else y_pred_norm
                    y_emul_by_sim[int(sid)] = y_pred
                outpath = os.path.join(args.outdir, f"{args.lake}_{args.split}_spaghetti_emul_run_{args.run_id}.png")
                plot_spaghetti_panel(
                    title=f"{args.lake} | {args.split} | run={args.run_id} | emulator spaghetti (n={len(spaghetti_sim_ids)})",
                    y_by_sim=y_emul_by_sim,
                    outpath=outpath,
                )

    # ----- binned observation scatter + emulator spaghetti at bin midpoints
    if args.plot_binned_obs_emul_spaghetti:
        if args.no_emulator:
            raise ValueError("--plot_binned_obs_emul_spaghetti requires emulator evaluation (remove --no_emulator)")
        if drivers_split.shape[0] < Wx:
            raise ValueError(f"Split has T={drivers_split.shape[0]} < Wx={Wx}; cannot run emulator windows.")

        # restrict plots to the observation period (in split-local indices)
        t_min = int(np.min(t_local))
        t_max = int(np.max(t_local))
        t_slice = slice(t_min, t_max + 1)
        time_obs_dt = time_split_dt[t_slice]

        # bins from aligned observations (only bins with at least 1 obs)
        if float(args.bin_size_m) <= 0.0:
            raise ValueError("--bin_size_m must be > 0")

        z_min = float(np.nanmin(obs_depth))
        z_max = float(np.nanmax(obs_depth))
        if not np.isfinite(z_min) or not np.isfinite(z_max):
            raise ValueError("Observation depths are not finite.")

        bin_size = float(args.bin_size_m)
        start = max(0.0, float(np.floor(z_min / bin_size) * bin_size))
        end = float(np.ceil(z_max / bin_size) * bin_size)

        edges = np.arange(start, end + 1e-6, bin_size, dtype=np.float32)
        bins: List[Tuple[float, float]] = []
        for z0, z1 in zip(edges[:-1].tolist(), edges[1:].tolist()):
            m = (obs_depth >= float(z0)) & (obs_depth < float(z1))
            if bool(np.any(m)):
                bins.append((float(z0), float(z1)))

        # cap number of bins by observation count (most populated first)
        if len(bins) > int(args.max_bins):
            counts = []
            for z0, z1 in bins:
                counts.append(int(np.sum((obs_depth >= z0) & (obs_depth < z1))))
            order = np.argsort(-np.asarray(counts, dtype=np.int32))
            bins = [bins[int(i)] for i in order[: int(args.max_bins)]]

        if not bins:
            raise RuntimeError("No non-empty depth bins found in aligned observations.")

        # choose sims: use up to spaghetti_max_sims, but allow user to set it to all sims (e.g. 1000)
        spaghetti_sim_ids = sim_ids[: int(min(len(sim_ids), args.spaghetti_max_sims))]

        # for each sim, compute temperature time series at each bin midpoint
        midpoints = np.asarray([(z0 + z1) / 2.0 for z0, z1 in bins], dtype=np.float32)
        series_by_sim: Dict[int, np.ndarray] = {}  # sim_id -> (T_obs_period, B)

        for sid in spaghetti_sim_ids:
            y_pred_norm = predict_timeseries_for_sim_batched(
                model=model,
                drivers=drivers_split,
                p_vec=lake.params[int(sid)],
                depth_feat_padded=depth_feat_padded,
                Wx=Wx,
                Wy=Wy,
                stride=max(1, int(args.stride)),
                device=args.device,
                batch_windows=max(1, int(args.batch_windows)),
            )
            y_pred_norm = y_pred_norm[:, :Dz_lake]
            y_pred = denorm_outputs(y_pred_norm, norms, Dz=Dz_lake) if norms is not None else y_pred_norm
            y_pred = y_pred[t_slice, :]  # (T_obs_period, Dz_lake)

            out = np.full((y_pred.shape[0], midpoints.shape[0]), np.nan, dtype=np.float32)
            for ti in range(y_pred.shape[0]):
                prof = y_pred[ti]
                if np.all(np.isnan(prof)):
                    continue
                if np.isnan(prof).any():
                    valid_idx = np.where(~np.isnan(prof))[0]
                    if valid_idx.size == 0:
                        continue
                    prof = np.interp(np.arange(prof.shape[0]), valid_idx, prof[valid_idx]).astype(np.float32)
                for bi, zmid in enumerate(midpoints.tolist()):
                    out[ti, bi] = interp_profile_clamped(prof, depth_grid, float(zmid))
            series_by_sim[int(sid)] = out

        # plot: one subplot per bin, obs scatter + emulator spaghetti
        fig, axes = plt.subplots(len(bins), 1, figsize=(13.5, 3.2 * len(bins)), sharex=True)
        if len(bins) == 1:
            axes = [axes]

        for ax, (z0, z1), bi in zip(axes, bins, range(len(bins))):
            zmid = float(midpoints[bi])

            # obs points in this bin, restricted to observation period indices
            m_bin = (obs_depth >= float(z0)) & (obs_depth < float(z1))
            t_bin = t_local[m_bin]
            temp_bin = obs_temp[m_bin]
            t_bin_dt = np.asarray([time_split_dt[int(ti)] for ti in t_bin.tolist()])

            ax.scatter(t_bin_dt, temp_bin, s=16.0, color="black", alpha=0.8, label="obs")

            for sid, mat in series_by_sim.items():
                ax.plot(
                    time_obs_dt,
                    mat[:, bi],
                    linewidth=float(args.spaghetti_lw),
                    alpha=float(args.spaghetti_alpha),
                    color="tab:blue",
                )

            ax.set_ylabel(f"T (°C)\n{z0:g}-{z1:g}m (mid {zmid:g})")
            ax.grid(True, alpha=0.2)
            ax.legend(loc="best")

        axes[-1].set_xlabel("Time")
        axes[-1].xaxis.set_major_locator(mdates.AutoDateLocator())
        axes[-1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(axes[-1].xaxis.get_major_locator()))
        fig.suptitle(
            f"{args.lake} | {args.split} | run={args.run_id} | emulator spaghetti over obs period (n={len(spaghetti_sim_ids)})",
            y=0.995,
        )
        plt.tight_layout(rect=[0, 0.02, 1, 0.98])
        outpath = os.path.join(
            args.outdir,
            f"{args.lake}_{args.split}_binned_obs_emul_spaghetti_bin_{args.bin_size_m:g}m_run_{args.run_id}.png",
        )
        plt.savefig(outpath, dpi=180)
        plt.close(fig)
        print(f"[saved] {outpath}")


if __name__ == "__main__":
    main()
