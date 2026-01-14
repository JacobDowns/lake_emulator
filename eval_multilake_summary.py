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

from mlflow.tracking import MlflowClient

from models import ModelRNNDepth
from multi_lake_dataset import load_multi_lake_data, LakeData


# =========================================================
# MLflow checkpoint loading
# =========================================================

def load_checkpoint_from_mlflow(run_id: str, dst_dir: str = "downloaded_artifacts"):
    os.makedirs(dst_dir, exist_ok=True)
    client = MlflowClient()
    local_path = client.download_artifacts(run_id, "checkpoints/best.pt", dst_dir)
    # In PyTorch >=2.6, weights_only default changed; we want full dict (config + arrays).
    ckpt = torch.load(local_path, map_location="cpu", weights_only=False)
    return local_path, ckpt


# =========================================================
# Norm / denorm helpers
# =========================================================

def _norms_2d(norms_arr: np.ndarray) -> np.ndarray:
    """
    Normalize norms array shapes to (1, D).
    Accepts (D,), (1,D), (1,1,D)
    """
    a = np.array(norms_arr)
    if a.ndim == 1:
        return a.reshape(1, -1).astype(np.float32)
    if a.ndim == 2:
        return a.reshape(1, -1).astype(np.float32)
    if a.ndim == 3:
        return a.reshape(1, -1).astype(np.float32)
    raise ValueError(f"Unexpected norms shape: {a.shape}")


def denorm_weather(drivers_norm: np.ndarray, norms: Optional[Dict]) -> np.ndarray:
    """
    drivers_norm: (..., Du)
    """
    if norms is None:
        return drivers_norm
    mu = _norms_2d(norms["weather_mean"])   # (1,Du)
    sd = _norms_2d(norms["weather_std"])    # (1,Du)
    return drivers_norm * sd + mu


def denorm_params(params_norm: np.ndarray, norms: Optional[Dict]) -> np.ndarray:
    """
    params_norm: (..., P)
    """
    if norms is None:
        return params_norm
    mu = _norms_2d(norms["params_mean"])    # (1,P)
    sd = _norms_2d(norms["params_std"])     # (1,P)
    return params_norm * sd + mu


def denorm_outputs(y_norm: np.ndarray, norms: Optional[Dict], Dz: Optional[int] = None) -> np.ndarray:
    """
    y_norm: (..., Dz_lake) OR (..., Dzmax)
    norms stores output_mean/std as (1,1,Dzmax) or similar
    Dz: if provided, slice norms to first Dz entries (per-lake depth count)
    """
    if norms is None:
        return y_norm
    mu = _norms_2d(norms["output_mean"])  # (1,Dzmax)
    sd = _norms_2d(norms["output_std"])   # (1,Dzmax)
    if Dz is not None:
        mu = mu[:, :Dz]
        sd = sd[:, :Dz]
    return y_norm * sd + mu


# =========================================================
# Split indexing
# =========================================================

def split_time_indices(year: np.ndarray, split: str, split_years: Tuple[int, int, int]) -> np.ndarray:
    tr_end, va_end, _ = split_years
    if split == "train":
        return np.where(year <= tr_end)[0]
    if split == "val":
        return np.where((year > tr_end) & (year <= va_end))[0]
    if split == "test":
        return np.where(year > va_end)[0]
    if split == "valtest":
        idx_val = np.where((year > tr_end) & (year <= va_end))[0]
        idx_test = np.where(year > va_end)[0]
        return np.concatenate([idx_val, idx_test], axis=0)
    if split == "all":
        return np.arange(year.shape[0], dtype=np.int32)
    raise ValueError("split must be one of train|val|test|valtest|all")


# =========================================================
# Model reconstruction from ckpt config
# =========================================================

def parse_hidden_list(s) -> List[int]:
    if isinstance(s, (list, tuple)):
        return [int(x) for x in s]
    s = str(s)
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def build_model_from_ckpt_config(cfg: Dict, Du: int, P: int, Dd: int) -> ModelRNNDepth:
    Wy = int(cfg.get("Wy", 30))
    hidden = int(cfg.get("hidden", 64))
    num_layers = int(cfg.get("num_layers", 2))
    rnn_dropout = float(cfg.get("rnn_dropout", 0.0))
    head_dropout = float(cfg.get("head_dropout", 0.0))
    head_hidden = parse_hidden_list(cfg.get("head_hidden", "256,256"))

    param_in_rnn = cfg.get("param_in_rnn", True)
    param_in_head = cfg.get("param_in_head", True)
    if isinstance(param_in_rnn, str):
        param_in_rnn = bool(int(param_in_rnn))
    if isinstance(param_in_head, str):
        param_in_head = bool(int(param_in_head))

    return ModelRNNDepth(
        Du=Du, P=P, Dd=Dd,
        hidden=hidden,
        num_layers=num_layers,
        rnn_dropout=rnn_dropout,
        Wy=Wy,
        head_hidden=head_hidden,
        head_dropout=head_dropout,
        param_in_rnn=param_in_rnn,
        param_in_head=param_in_head,
        squeeze_output=False,  # keep (B,Wy,Dz) for accumulation
    )


# =========================================================
# Batched window inference for one sim
# =========================================================

@torch.no_grad()
def predict_timeseries_for_sim_batched(
    model: torch.nn.Module,
    drivers: np.ndarray,          # (T, Du) normalized
    p_vec: np.ndarray,            # (P,) normalized
    depth_feat_padded: np.ndarray,# (Dzmax, Dd)
    Wx: int,
    Wy: int,
    stride: int,
    device: str,
    batch_windows: int,
) -> np.ndarray:
    """
    Returns pred: (T, Dzmax) normalized, NaN where no predictions.
    Window alignment: each window starting at s predicts times:
      [s + (Wx-Wy) ... s + Wx-1]   (length Wy)
    """
    T, Du = drivers.shape
    Dzmax = depth_feat_padded.shape[0]
    if T < Wx:
        raise ValueError(f"T={T} < Wx={Wx}")

    starts = np.arange(0, T - Wx + 1, stride, dtype=np.int32)  # window starts
    W = starts.shape[0]

    preds_sum = np.zeros((T, Dzmax), dtype=np.float32)
    preds_cnt = np.zeros((T,), dtype=np.int32)

    # constant tensors
    p_t = torch.from_numpy(p_vec.astype(np.float32, copy=False)).to(device=device).unsqueeze(0)  # (1,P)
    df_t = torch.from_numpy(depth_feat_padded.astype(np.float32, copy=False)).to(device=device).unsqueeze(0)  # (1,Dzmax,Dd)

    for i0 in range(0, W, batch_windows):
        i1 = min(W, i0 + batch_windows)
        batch_starts = starts[i0:i1]  # (B,)

        # Build batch windows on the fly (memory-friendly)
        x_batch = np.stack([drivers[s:s+Wx, :] for s in batch_starts], axis=0).astype(np.float32, copy=False)  # (B,Wx,Du)

        x_t = torch.from_numpy(np.ascontiguousarray(x_batch)).to(device=device)  # (B,Wx,Du)
        B = x_t.shape[0]

        p_bt = p_t.expand(B, -1)           # (B,P)
        df_bt = df_t.expand(B, -1, -1)     # (B,Dzmax,Dd)

        y = model(x_t, p_bt, df_bt)        # (B,Wy,Dzmax)
        if y.dim() == 2:
            y = y.unsqueeze(1)

        y_np = y.detach().cpu().numpy().astype(np.float32, copy=False)  # (B,Wy,Dzmax)

        # Accumulate onto time axis
        for bi, s in enumerate(batch_starts):
            base_t = int(s + (Wx - Wy))
            t0 = base_t
            t1 = base_t + Wy
            if t0 < 0 or t1 > T:
                continue
            preds_sum[t0:t1, :] += y_np[bi]
            preds_cnt[t0:t1] += 1

    pred = np.full((T, Dzmax), np.nan, dtype=np.float32)
    m = preds_cnt > 0
    pred[m, :] = preds_sum[m, :] / preds_cnt[m, None]
    return pred


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
