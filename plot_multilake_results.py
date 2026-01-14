#!/usr/bin/env python3
"""
plot_multilake_results.py

Plot multi-lake ModelRNNDepth results on VAL and TEST splits.

- Downloads best.pt from an MLflow run (run_id)
- Rebuilds ModelRNNDepth from checkpoint config + inferred dims
- Loads multi-lake data with the same normalization assumptions
- For each lake: randomly selects a few simulations and plots:
    (1) time series at representative depths (True vs Pred + residual)
    (2) heatmaps: True, Pred, Pred-True

This script assumes:
- multi_lake_dataset.py provides: load_multi_lake_data(...), LakeData
- models.py provides: ModelRNNDepth
- best.pt contains:
    ckpt["state_dict"], ckpt["config"] (vars(args) from training)

Usage examples:
  python plot_multilake_results.py --run_id <RUN_ID> --root_dir data/parsed_data --lakes BearLake RedPond
  python plot_multilake_results.py --run_id <RUN_ID> --stride 7 --num_sims 2
"""

import os
import argparse
import random
from typing import Dict, List, Tuple, Optional

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
    # PyTorch >=2.6 defaults weights_only=True; we explicitly set False (trusted checkpoint)
    ckpt = torch.load(local_path, map_location="cpu", weights_only=False)
    return local_path, ckpt


# =========================================================
# Denorm helpers (depth-aware; norms stored at Dz_max)
# =========================================================

def _norms_output_arrays(norms: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return output mean/std as (1, Dz_max) arrays regardless of storage convention.
    Accepts (1,1,Dz), (1,Dz), (Dz,).
    """
    mu = np.array(norms["output_mean"])
    sd = np.array(norms["output_std"])

    if mu.ndim == 3:        # (1,1,Dzmax)
        mu = mu.reshape(1, -1)
        sd = sd.reshape(1, -1)
    elif mu.ndim == 2:      # (1,Dzmax) or (Dzmax,1) -> flatten to (1, Dzmax)
        mu = mu.reshape(1, -1)
        sd = sd.reshape(1, -1)
    elif mu.ndim == 1:      # (Dzmax,)
        mu = mu.reshape(1, -1)
        sd = sd.reshape(1, -1)
    else:
        raise ValueError(f"Unexpected output_mean shape: {mu.shape}")

    return mu.astype(np.float32), sd.astype(np.float32)


def denorm_outputs(y_norm: np.ndarray, norms: Optional[Dict], Dz: Optional[int] = None) -> np.ndarray:
    """
    Depth-aware denormalization.

    norms are stored at Dz_max, but y_norm may be (..., Dz_lake).
    We slice mu/sd to match y_norm's last dimension.
    """
    if norms is None:
        return y_norm

    mu, sd = _norms_output_arrays(norms)  # (1, Dz_max)

    if Dz is None:
        Dz = int(y_norm.shape[-1])        # Dz_lake or Dz_max depending on array

    mu = mu[:, :Dz]                       # (1, Dz)
    sd = sd[:, :Dz]                       # (1, Dz)

    # broadcast over leading dims
    if y_norm.ndim == 2:                  # (T, Dz)
        return y_norm * sd + mu
    elif y_norm.ndim == 3:                # (B, Wy, Dz)
        return y_norm * sd[None, :, :] + mu[None, :, :]
    else:
        # e.g., (B, Dz) or other shapes where last dim is depth
        return y_norm * sd + mu


# =========================================================
# Split helpers (use year vectors in LakeData)
# =========================================================

def split_time_indices(year: np.ndarray, split: str, split_years: Tuple[int, int, int]) -> np.ndarray:
    tr_end, va_end, _ = split_years
    if split == "train":
        return np.where(year <= tr_end)[0]
    if split == "val":
        return np.where((year > tr_end) & (year <= va_end))[0]
    if split == "test":
        return np.where(year > va_end)[0]
    raise ValueError("split must be train|val|test")


# =========================================================
# Efficient batched prediction over a split for one (lake, sim)
# =========================================================

@torch.no_grad()
def predict_split_for_sim(
    model: torch.nn.Module,
    lake: LakeData,
    sim_id: int,
    depth_feat_padded: np.ndarray,   # (Dzmax, Dd)
    depth_mask: np.ndarray,          # (Dzmax,)
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
      truth: (T_split, Dzmax) normalized (padded)
      pred : (T_split, Dzmax) normalized (padded, NaN where no preds)
      mask : (Dzmax,) float {0,1}
    """
    t_idx = split_time_indices(lake.year, split, split_years)
    drivers = lake.drivers[t_idx, :]          # (T_split, Du) normalized
    temps = lake.temps[sim_id, t_idx, :]      # (T_split, Dz_lake) normalized
    T_split = drivers.shape[0]
    Dz_lake = temps.shape[1]
    Dzmax = depth_feat_padded.shape[0]

    if T_split < Wx:
        raise ValueError(f"{lake.name} split={split} too short: T={T_split} < Wx={Wx}")

    # Pad truth to Dzmax
    truth = np.zeros((T_split, Dzmax), dtype=np.float32)
    truth[:, :Dz_lake] = temps.astype(np.float32, copy=False)

    # Window starts (in split-local coordinates)
    starts = np.arange(0, T_split - Wx + 1, stride, dtype=np.int32)
    W = starts.shape[0]

    # Build weather windows with sliding_window_view then stride
    # sliding_window_view -> (T_split - Wx + 1, 1, Wx, Du) with this call
    wv = np.lib.stride_tricks.sliding_window_view(
        drivers, window_shape=(Wx, drivers.shape[1]), axis=(0, 1)
    )
    x_all = np.ascontiguousarray(wv[:, 0, :, :])  # (T_split-Wx+1, Wx, Du)
    x_all = x_all[starts]                         # (W, Wx, Du)

    # Torch inputs
    x_all_t = torch.from_numpy(x_all).to(device=device, dtype=torch.float32)  # (W, Wx, Du)
    p = torch.from_numpy(lake.params[sim_id]).to(device=device, dtype=torch.float32).unsqueeze(0)  # (1,P)
    p_all = p.expand(W, -1)  # (W,P)

    df = torch.from_numpy(depth_feat_padded).to(device=device, dtype=torch.float32).unsqueeze(0)  # (1,Dzmax,Dd)
    df_all = df.expand(W, -1, -1)  # (W,Dzmax,Dd)

    # Predict in batches
    preds_win = np.zeros((W, Wy, Dzmax), dtype=np.float32)
    for i0 in range(0, W, batch_windows):
        i1 = min(W, i0 + batch_windows)
        y = model(x_all_t[i0:i1], p_all[i0:i1], df_all[i0:i1])  # (B,Wy,Dzmax) or (B,Dzmax) if Wy==1
        if y.dim() == 2:
            y = y.unsqueeze(1)  # (B,1,Dzmax)
        preds_win[i0:i1] = y.detach().cpu().numpy()

    # Accumulate window predictions onto time axis (average overlaps)
    preds_sum = np.zeros((T_split, Dzmax), dtype=np.float32)
    preds_cnt = np.zeros((T_split,), dtype=np.int32)

    # Alignment: each window starting at s predicts times [s + (Wx-Wy) .. s + Wx-1]
    for wi, s in enumerate(starts):
        base_t = int(s + (Wx - Wy))
        t_slice = slice(base_t, base_t + Wy)  # length Wy
        if base_t < 0 or base_t + Wy > T_split:
            continue
        preds_sum[t_slice, :] += preds_win[wi]
        preds_cnt[t_slice] += 1

    pred = np.full((T_split, Dzmax), np.nan, dtype=np.float32)
    m = preds_cnt > 0
    pred[m, :] = preds_sum[m, :] / preds_cnt[m, None]

    return truth, pred, depth_mask.astype(np.float32)


# =========================================================
# Plotting
# =========================================================

def representative_depth_indices(Dz_lake: int) -> List[int]:
    if Dz_lake <= 1:
        return [0]
    if Dz_lake == 2:
        return [0, 1]
    return [0, Dz_lake // 2, Dz_lake - 1]


def plot_timeseries_and_heatmaps(
    outdir: str,
    lake_name: str,
    split: str,
    run_id: str,
    sim_id: int,
    truth_den: np.ndarray,   # (T, Dz_lake)
    pred_den: np.ndarray,    # (T, Dz_lake) with NaNs
):
    os.makedirs(outdir, exist_ok=True)
    T, Dz = truth_den.shape
    time_axis = np.arange(T)

    depths = representative_depth_indices(Dz)

    # ------------------ time series (True/Pred + residual) ------------------
    fig_ts, axes_ts = plt.subplots(
        len(depths) * 2, 1,
        figsize=(11, 5.0 * len(depths)),
        sharex=True
    )
    if len(depths) == 1:
        axes_pairs = [(axes_ts[0], axes_ts[1])]
    else:
        axes_pairs = [(axes_ts[2*i], axes_ts[2*i+1]) for i in range(len(depths))]

    for (ax_top, ax_bot), d in zip(axes_pairs, depths):
        y_true = truth_den[:, d]
        y_pred = pred_den[:, d]
        resid = y_pred - y_true

        ax_top.plot(time_axis, y_true, label="True", linewidth=1.5)
        ax_top.plot(time_axis, y_pred, label="Pred", linestyle="--", linewidth=1.2)
        ax_top.set_ylabel(f"T (°C)\nDepth idx {d}")
        ax_top.grid(True, alpha=0.3)
        ax_top.legend(loc="upper right")

        ax_bot.plot(time_axis, resid, label="Pred - True", linestyle=":", linewidth=1.2)
        ax_bot.axhline(0.0, linewidth=0.8, alpha=0.6)
        ax_bot.set_ylabel("ΔT (°C)")
        ax_bot.grid(True, alpha=0.3)
        ax_bot.legend(loc="upper right")

    axes_pairs[-1][1].set_xlabel("Time index (days)")

    fig_ts.suptitle(f"{lake_name} | {split.upper()} | run={run_id} | sim={sim_id}", y=0.995)
    fig_ts.tight_layout(rect=[0, 0.02, 1, 0.97])

    fname_ts = os.path.join(outdir, f"{lake_name}_{split}_timeseries_run_{run_id}_sim_{sim_id}.png")
    plt.savefig(fname_ts, dpi=150)
    plt.close(fig_ts)

    # ------------------ heatmaps (True / Pred / Residual) ------------------
    diff = pred_den - truth_den

    fig_h, axes_h = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)

    def _imshow(ax, data, title, vmin=None, vmax=None, cmap="viridis"):
        im = ax.imshow(
            data.T, aspect="auto", origin="upper", interpolation="nearest",
            vmin=vmin, vmax=vmax, cmap=cmap
        )
        ax.set_title(title)
        ax.set_xlabel("Time (days)")
        return im

    tmin = np.nanmin([truth_den, pred_den])
    tmax = np.nanmax([truth_den, pred_den])

    im0 = _imshow(axes_h[0], truth_den, "True", vmin=tmin, vmax=tmax)
    axes_h[0].set_ylabel("Depth index")

    im1 = _imshow(axes_h[1], pred_den, "Pred", vmin=tmin, vmax=tmax)

    dmax = np.nanmax(np.abs(diff))
    im2 = _imshow(axes_h[2], diff, "Pred - True", vmin=-dmax, vmax=dmax, cmap="coolwarm")

    c0 = fig_h.colorbar(im0, ax=axes_h[0], fraction=0.046, pad=0.04)
    c0.set_label("°C")
    c1 = fig_h.colorbar(im1, ax=axes_h[1], fraction=0.046, pad=0.04)
    c1.set_label("°C")
    c2 = fig_h.colorbar(im2, ax=axes_h[2], fraction=0.046, pad=0.04)
    c2.set_label("°C")

    fig_h.suptitle(f"{lake_name} | {split.upper()} heatmaps | run={run_id} | sim={sim_id}", y=0.99)
    fig_h.tight_layout(rect=[0, 0.02, 1, 0.95])

    fname_h = os.path.join(outdir, f"{lake_name}_{split}_heatmaps_run_{run_id}_sim_{sim_id}.png")
    plt.savefig(fname_h, dpi=150)
    plt.close(fig_h)

    print(f"[saved] {fname_ts}")
    print(f"[saved] {fname_h}")


# =========================================================
# Build model from ckpt config
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

    model = ModelRNNDepth(
        Du=Du, P=P, Dd=Dd,
        hidden=hidden,
        num_layers=num_layers,
        rnn_dropout=rnn_dropout,
        Wy=Wy,
        head_hidden=head_hidden,
        head_dropout=head_dropout,
        param_in_rnn=param_in_rnn,
        param_in_head=param_in_head,
        squeeze_output=False,  # keep (B,Wy,Dz) always for consistent plotting
    )
    return model


# =========================================================
# Main
# =========================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", required=True, help="MLflow run_id")
    ap.add_argument("--root_dir", default="data/parsed_data", help="Root with lake subdirs (BearLake, RedPond, ...)")
    ap.add_argument("--lakes", nargs="+", default=["BearLake", "RedPond"])
    ap.add_argument("--split_years", type=int, nargs=3, default=[2018, 2021, 2025])
    ap.add_argument("--num_sims", type=int, default=2, help="Random sims per lake to plot")
    ap.add_argument("--stride", type=int, default=1, help="Window start stride for inference (>=1). Increase to speed up.")
    ap.add_argument("--batch_windows", type=int, default=128, help="Batch size over windows during inference")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--outdir", default="plots_multilake")
    ap.add_argument("--splits", nargs="+", default=["val", "test"], choices=["train", "val", "test", "valtest"])
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load checkpoint
    _, ckpt = load_checkpoint_from_mlflow(args.run_id)
    cfg = ckpt.get("config", {})
    Wx = int(cfg.get("Wx", 365))
    Wy = int(cfg.get("Wy", 30))

    # Load multi-lake data (normalized using global norms)
    lakes, norms = load_multi_lake_data(
        root_dir=args.root_dir,
        lake_names=args.lakes,
        split_years=list(args.split_years),
        normalize=True,
    )

    # Infer dims from data
    Du = lakes[0].drivers.shape[1]
    P = lakes[0].params.shape[1]
    Dd = lakes[0].depth_feat.shape[1]
    Dzmax = max(lk.Dz for lk in lakes)

    # Build padded depth_feat + mask per lake (so plotting can slice real depths cleanly)
    depth_feat_padded_by_lake: Dict[str, np.ndarray] = {}
    depth_mask_by_lake: Dict[str, np.ndarray] = {}
    for lk in lakes:
        df = np.zeros((Dzmax, Dd), dtype=np.float32)
        df[:lk.Dz, :] = lk.depth_feat.astype(np.float32, copy=False)
        m = np.zeros((Dzmax,), dtype=np.float32)
        m[:lk.Dz] = 1.0
        depth_feat_padded_by_lake[lk.name] = df
        depth_mask_by_lake[lk.name] = m

    # Build + load model
    model = build_model_from_ckpt_config(cfg, Du=Du, P=P, Dd=Dd)
    model.load_state_dict(ckpt["state_dict"])
    model.to(args.device)
    model.eval()

    # Plot
    os.makedirs(args.outdir, exist_ok=True)

    for lk in lakes:
        # choose sims
        all_sims = list(range(lk.params.shape[0]))
        random.shuffle(all_sims)
        chosen = all_sims[: min(args.num_sims, len(all_sims))]

        for split in args.splits:
            if split != "valtest":
                for sim_id in chosen:
                    truth, pred, _mask = predict_split_for_sim(
                        model=model,
                        lake=lk,
                        sim_id=sim_id,
                        depth_feat_padded=depth_feat_padded_by_lake[lk.name],
                        depth_mask=depth_mask_by_lake[lk.name],
                        split=split,
                        split_years=tuple(args.split_years),
                        Wx=Wx,
                        Wy=Wy,
                        stride=max(1, args.stride),
                        device=args.device,
                        batch_windows=max(1, args.batch_windows),
                    )

                    # Slice to real depths for plotting
                    Dz_lake = lk.Dz
                    truth_l = truth[:, :Dz_lake]
                    pred_l = pred[:, :Dz_lake]

                    # Denorm (depth-aware slicing of mu/sd)
                    truth_den = denorm_outputs(truth_l, norms, Dz=Dz_lake)
                    pred_den = denorm_outputs(pred_l, norms, Dz=Dz_lake)

                    plot_timeseries_and_heatmaps(
                        outdir=args.outdir,
                        lake_name=lk.name,
                        split=split,
                        run_id=args.run_id,
                        sim_id=sim_id,
                        truth_den=truth_den,
                        pred_den=pred_den,
                    )
            else:
                # valtest: concatenate val then test (simple + robust)
                for sim_id in chosen:
                    truth_v, pred_v, _ = predict_split_for_sim(
                        model=model, lake=lk, sim_id=sim_id,
                        depth_feat_padded=depth_feat_padded_by_lake[lk.name],
                        depth_mask=depth_mask_by_lake[lk.name],
                        split="val", split_years=tuple(args.split_years),
                        Wx=Wx, Wy=Wy, stride=max(1, args.stride),
                        device=args.device, batch_windows=max(1, args.batch_windows),
                    )
                    truth_t, pred_t, _ = predict_split_for_sim(
                        model=model, lake=lk, sim_id=sim_id,
                        depth_feat_padded=depth_feat_padded_by_lake[lk.name],
                        depth_mask=depth_mask_by_lake[lk.name],
                        split="test", split_years=tuple(args.split_years),
                        Wx=Wx, Wy=Wy, stride=max(1, args.stride),
                        device=args.device, batch_windows=max(1, args.batch_windows),
                    )

                    truth = np.concatenate([truth_v, truth_t], axis=0)
                    pred = np.concatenate([pred_v, pred_t], axis=0)

                    Dz_lake = lk.Dz
                    truth_l = truth[:, :Dz_lake]
                    pred_l = pred[:, :Dz_lake]

                    truth_den = denorm_outputs(truth_l, norms, Dz=Dz_lake)
                    pred_den = denorm_outputs(pred_l, norms, Dz=Dz_lake)

                    plot_timeseries_and_heatmaps(
                        outdir=args.outdir,
                        lake_name=lk.name,
                        split="valtest",
                        run_id=args.run_id,
                        sim_id=sim_id,
                        truth_den=truth_den,
                        pred_den=pred_den,
                    )

    print("Done.")


if __name__ == "__main__":
    main()
