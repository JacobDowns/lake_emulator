#!/usr/bin/env python3
import os
import argparse
import random

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mlflow.tracking import MlflowClient

# Your model definition & data loader from training code
from models import ModelRNN
from train import load_data  # reuse exact same normalization / splitting


# ---------------------------
# Helpers for norms / denorm
# ---------------------------

def _norms_output_arrays(norms):
    """
    Normalize shapes: return (1, Dz) arrays for output mean/std regardless of storage convention.
    Accepts (Dz,), (1, Dz) or (1,1,Dz) and returns (1, Dz).
    """
    mu = np.array(norms["output_mean"])
    sd = np.array(norms["output_std"])

    if mu.ndim == 3:      # (1,1,Dz)
        mu = mu.reshape(1, -1)
        sd = sd.reshape(1, -1)
    elif mu.ndim == 1:    # (Dz,)
        mu = mu.reshape(1, -1)
        sd = sd.reshape(1, -1)
    # if (1, Dz) already, keep
    return mu.astype(np.float32), sd.astype(np.float32)


def denorm_outputs(y_norm: np.ndarray, norms: dict) -> np.ndarray:
    """
    y_norm: (T, Dz) or (B, Dz).
    Returns same shape in real units.
    """
    if norms is None:
        return y_norm
    mu, sd = _norms_output_arrays(norms)  # (1, Dz)
    return y_norm * sd + mu  # broadcasting over time/batch axis


# ---------------------------
# Seasonal features (match train.py)
# ---------------------------

def build_seasonal_features(doy: np.ndarray) -> np.ndarray:
    """
    Build a cyclic encoding of day-of-year.
    doy: (T,) with values in [1,365] or [0,364]
    Returns: (T, 2) array with sin/cos(2π * doy / 365).
    """
    angle = 2.0 * np.pi * (doy / 365.0)
    sin_doy = np.sin(angle)
    cos_doy = np.cos(angle)
    return np.stack([sin_doy, cos_doy], axis=-1).astype(np.float32)  # (T, 2)


# ---------------------------
# MLflow checkpoint loading
# ---------------------------

def load_checkpoint_from_mlflow(run_id: str, dst_dir: str = "downloaded_artifacts"):
    """
    Download best.pt from the given MLflow run into dst_dir and return:
      ckpt_path, checkpoint_dict
    """
    os.makedirs(dst_dir, exist_ok=True)
    client = MlflowClient()
    # train.py saved the checkpoint under artifact_path="checkpoints/best.pt"
    local_path = client.download_artifacts(run_id, "checkpoints/best.pt", dst_dir)
    ckpt = torch.load(local_path, map_location="cpu")
    return local_path, ckpt


# ---------------------------
# Model reconstruction (RNN only)
# ---------------------------

def build_model_from_config(cfg, Du, Dz, P):
    """
    Rebuild the RNN model based on the saved training config (cfg) and data dims.
    """
    model_type = cfg.get("model", "rnn")
    if model_type != "rnn":
        raise ValueError(f"plot script expects an RNN model, but cfg['model']={model_type}")

    Wx = int(cfg["Wx"])
    Wy = int(cfg["Wy"])

    head_hidden_str = cfg.get("head_hidden", "256,256")
    head_hidden = [int(x) for x in str(head_hidden_str).split(",") if x.strip()]
    head_dropout = float(cfg.get("head_dropout", 0.0))

    cell = cfg.get("cell", "gru")
    hidden = int(cfg.get("hidden", 128))
    num_layers = int(cfg.get("num_layers", 1))
    rnn_dropout = float(cfg.get("rnn_dropout", 0.0))

    # Attention flags (may be stored as bool or int)
    use_attention_raw = cfg.get("use_attention", 0)
    if isinstance(use_attention_raw, str):
        use_attention = bool(int(use_attention_raw))
    else:
        use_attention = bool(use_attention_raw)

    attn_dim = int(cfg.get("attn_dim", 64))

    # Optional flags for parameter injection (if you later add them)
    param_in_rnn_raw = cfg.get("param_in_rnn", 1)
    if isinstance(param_in_rnn_raw, str):
        param_in_rnn = bool(int(param_in_rnn_raw))
    else:
        param_in_rnn = bool(param_in_rnn_raw)

    param_in_head_raw = cfg.get("param_in_head", 1)
    if isinstance(param_in_head_raw, str):
        param_in_head = bool(int(param_in_head_raw))
    else:
        param_in_head = bool(param_in_head_raw)

    model = ModelRNN(
        cell=cell,
        Du=Du,
        P=P,
        Dz=Dz,
        hidden=hidden,
        num_layers=num_layers,
        rnn_dropout=rnn_dropout,
        W_y=Wy,
        head_hidden=head_hidden,
        head_dropout=head_dropout,
        use_attention=use_attention,
        attn_dim=attn_dim,
        param_in_rnn=param_in_rnn,
        param_in_head=param_in_head,
    )
    return model


# ---------------------------
# Main plotting logic
# ---------------------------

def make_time_series_plots_for_run(
    run_id: str,
    num_sims: int = 3,
    max_days: int = None,
    device: str = None,
    outdir: str = "plots_timeseries",
    split: str = "test",   # "train" | "val" | "test" | "valtest" | "val+test"
):
    """
    For a given MLflow run:
      - load best checkpoint
      - rebuild model
      - reload data with the same splits & normalization
      - optionally concatenate val+test into one continuous period
      - for a few random simulations (parameter sets),
        run the model over the chosen period and plot:
          * time series (True vs Pred + ΔT)
          * heatmaps (True, Pred, ΔT)

    Supports arbitrary Wy >= 1. Overlapping predictions from multiple windows
    are averaged per time step.
    """
    os.makedirs(outdir, exist_ok=True)

    # 1) Load checkpoint & config
    ckpt_path, ckpt = load_checkpoint_from_mlflow(run_id)
    cfg = ckpt["config"]  # this is vars(args) from train.py
    Wx = int(cfg["Wx"])
    Wy = int(cfg["Wy"])
    use_seasonal = int(cfg.get("use_seasonal_features", 0)) != 0

    if device is None:
        device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    # 2) Reload data with same split_years & normalisation
    split_years = cfg.get("split_years", [2018, 2020, 2025])
    if isinstance(split_years, str):
        # in case it was saved as "2018,2020,2025"
        split_years = [int(x) for x in split_years.split(",")]

    data = load_data(
        weather_path=cfg.get("weather", "data/parsed_data/weather_data.npy"),
        output_path=cfg.get("outputs", "data/parsed_data/output_data.npy"),
        params_path=cfg.get("params",  "data/parsed_data/parameter_data.npy"),
        split_years=split_years,
        normalize=True,
    )

    # Base arrays for each split
    w_tr = data["weather_data_train"]
    w_va = data["weather_data_val"]
    w_te = data["weather_data_test"]

    o_tr = data["output_data_train"]
    o_va = data["output_data_val"]
    o_te = data["output_data_test"]

    doy_tr = data["doy_train"]
    doy_va = data["doy_val"]
    doy_te = data["doy_test"]

    params = data["params_data"]               # (N, P)
    norms = data["norms"]

    # Choose which period to visualize
    if split == "train":
        weather_drivers = w_tr           # (T_train, Du)
        outputs = o_tr                   # (N, T_train, Dz)
        doy = doy_tr                     # (T_train,)
        label_prefix = "Train"
    elif split == "val":
        weather_drivers = w_va
        outputs = o_va
        doy = doy_va
        label_prefix = "Validation"
    elif split == "test":
        weather_drivers = w_te
        outputs = o_te
        doy = doy_te
        label_prefix = "Test"
    elif split in ("valtest", "val+test"):
        # Concatenate validation + test along time dimension
        weather_drivers = np.concatenate([w_va, w_te], axis=0)       # (T_val+T_test, Du)
        outputs = np.concatenate([o_va, o_te], axis=1)               # (N, T_val+T_test, Dz)
        doy = np.concatenate([doy_va, doy_te], axis=0)               # (T_val+T_test,)
        label_prefix = "Val+Test"
        split = "valtest"  # normalize name for filenames
    else:
        raise ValueError("split must be one of 'train', 'val', 'test', 'valtest', 'val+test'")

    # Add seasonal features if the model was trained with them
    if use_seasonal:
        extra = build_seasonal_features(doy)                # (T_period, 2)
        weather = np.concatenate([weather_drivers, extra], axis=-1)  # (T_period, Du+2)
    else:
        weather = weather_drivers

    T_period, Du_current = weather.shape
    N, T_out, Dz_current = outputs.shape
    assert T_period == T_out, "Time dimension mismatch for chosen period."

    # Data dims (from checkpoint if present)
    Du_model = int(ckpt.get("Du", Du_current))
    Dz_model = int(ckpt.get("Dz", Dz_current))
    P_model  = int(ckpt.get("P", params.shape[1]))

    # Sanity check: feature dims must match what the model expects
    if Du_model != Du_current:
        raise ValueError(
            f"Input feature dimension mismatch: model Du={Du_model}, "
            f"but weather has Du={Du_current}. "
            f"(Did you forget seasonal features or change preprocessing?)"
        )
    if Dz_model != Dz_current:
        raise ValueError(
            f"Output depth dimension mismatch: model Dz={Dz_model}, "
            f"but outputs have Dz={Dz_current}."
        )

    Dz = Dz_model
    P = P_model

    # 3) Rebuild model and load weights
    model = build_model_from_config(cfg, Du_model, Dz_model, P_model)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()

    # 4) Choose random simulation indices
    all_ids = list(range(params.shape[0]))
    random.shuffle(all_ids)
    chosen_ids = all_ids[: min(num_sims, len(all_ids))]

    # 5) Generate predictions over full chosen period for each sim
    if T_period < Wx:
        raise ValueError(f"Period too short (T={T_period}) for Wx={Wx}.")
    print(f"Generating predictions over {label_prefix} period: T={T_period}, Wx={Wx}, Wy={Wy}")

    num_windows = T_period - Wx + 1

    for n in chosen_ids:
        p_vec = torch.from_numpy(params[n]).to(device=device, dtype=torch.float32)  # (P,)

        # Accumulate predictions and counts for averaging overlaps
        preds_sum = np.zeros((T_period, Dz), dtype=np.float32)
        preds_count = np.zeros(T_period, dtype=np.int32)

        truth = outputs[n].copy()  # (T_period, Dz), still normalized

        # For each window starting at 'start' with indices [start ... start+Wx-1]:
        # the model predicts Wy steps aligned to the LAST Wy days of the window.
        # So predicted time indices are:
        #   t_idx = start + (Wx - Wy) + j,  j = 0 .. Wy-1
        for start in range(num_windows):
            end = start + Wx
            x_win = weather[start:end, :]   # (Wx, Du_model)
            x_win_t = torch.from_numpy(x_win).unsqueeze(0).to(device=device, dtype=torch.float32)  # (1, Wx, Du)
            p_t = p_vec.unsqueeze(0)  # (1, P)

            with torch.no_grad():
                y_pred = model(x_win_t, p_t)
            # Ensure shape (1, Wy, Dz)
            if y_pred.dim() == 2:  # (1, Dz) for Wy=1 squeezed
                y_pred = y_pred.unsqueeze(1)
            y_pred_np = y_pred.detach().cpu().numpy()[0]  # (Wy, Dz)

            for j in range(Wy):
                t_idx = start + (Wx - Wy) + j
                if 0 <= t_idx < T_period:
                    preds_sum[t_idx, :] += y_pred_np[j, :]
                    preds_count[t_idx] += 1

        # Build final preds array, average where we have coverage
        preds = np.full((T_period, Dz), np.nan, dtype=np.float32)
        valid_mask = preds_count > 0
        preds[valid_mask, :] = (
            preds_sum[valid_mask, :] / preds_count[valid_mask][:, None]
        )

        # De-normalize true and predicted
        truth_denorm_full = denorm_outputs(truth, norms)  # (T_period, Dz)
        preds_denorm_full = denorm_outputs(preds, norms)  # (T_period, Dz) with NaNs for unpredicted times

        # Restrict to indices where we have predictions
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) == 0:
            print(f"No valid predictions for sim {n}, skipping.")
            continue

        t_min = valid_indices[0]
        t_max = valid_indices[-1]

        # Apply optional max_days limit
        if max_days is not None:
            t_max = min(t_max, t_min + max_days - 1)

        truth_denorm = truth_denorm_full[t_min:t_max+1, :]  # (T_plot, Dz)
        preds_denorm = preds_denorm_full[t_min:t_max+1, :]  # (T_plot, Dz)
        T_plot = truth_denorm.shape[0]
        time_axis = np.arange(T_plot)  # days since start of plotted period

        # Choose a few representative depths (surface, mid, bottom)
        if Dz >= 3:
            chosen_depth_indices = [0, Dz // 2, Dz - 1]
        else:
            chosen_depth_indices = list(range(Dz))

        # ---------------------------------------------------
        # Time series: True & Pred (top) + residual (bottom)
        # ---------------------------------------------------
        fig_ts, axes_ts = plt.subplots(
            len(chosen_depth_indices) * 2,   # 2 rows per depth
            1,
            figsize=(10, 5.5 * len(chosen_depth_indices)),
            sharex=True
        )

        if len(chosen_depth_indices) == 1:
            axes_pairs = [(axes_ts[0], axes_ts[1])]
        else:
            axes_pairs = [
                (axes_ts[2 * i], axes_ts[2 * i + 1])
                for i in range(len(chosen_depth_indices))
            ]

        for (temp_ax, diff_ax), d in zip(axes_pairs, chosen_depth_indices):
            true_series = truth_denorm[:, d]
            pred_series = preds_denorm[:, d]
            diff_series = pred_series - true_series  # residual (Pred - True)

            # --- Top subplot: True + Pred ---
            temp_ax.plot(time_axis, true_series, label="True", linewidth=1.5)
            temp_ax.plot(time_axis, pred_series, label="Pred", linestyle="--", linewidth=1.2)
            temp_ax.set_ylabel(f"T (°C)\nDepth idx {d}")
            temp_ax.grid(True, alpha=0.3)
            temp_ax.legend(loc="upper right")

            # --- Bottom subplot: Residual ---
            diff_ax.plot(time_axis, diff_series, label="Pred - True",
                         linestyle=":", linewidth=1.2)
            diff_ax.axhline(0.0, linestyle="-", linewidth=0.8, alpha=0.5)
            diff_ax.set_ylabel("ΔT (°C)")
            diff_ax.grid(True, alpha=0.3)
            diff_ax.legend(loc="upper right")

        axes_pairs[-1][1].set_xlabel(
            f"{label_prefix} time index (days since start of {label_prefix.lower()} period)"
        )

        fig_ts.suptitle(f"{label_prefix} time series | Run {run_id} | sim n={n} | Wx={Wx}, Wy={Wy}")
        fig_ts.tight_layout(rect=[0, 0.03, 1, 0.95])

        fname_ts = os.path.join(outdir, f"{split}_timeseries_run_{run_id}_sim_{n}.png")
        plt.savefig(fname_ts, dpi=150)
        plt.close(fig_ts)
        print(f"Saved time-series plot for sim {n} to {fname_ts}")

        # ---------------------------------------------------
        # Heatmaps: True, Pred, and Residual (Pred - True)
        # ---------------------------------------------------
        diff_denorm = preds_denorm - truth_denorm  # (T_plot, Dz)

        fig_h, axes_h = plt.subplots(
            1, 3,
            figsize=(14, 4.5),
            sharey=True
        )

        def _plot_heat(ax, data, title, vmin=None, vmax=None, cmap="viridis"):
            # data: (T_plot, Dz) -> transpose to (Dz, T_plot) so depth is y-axis
            im = ax.imshow(
                data.T,
                aspect="auto",
                origin="upper",
                interpolation="nearest",
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
            )
            ax.set_title(title)
            ax.set_xlabel(
                f"{label_prefix} time index (days since start of {label_prefix.lower()} period)"
            )
            return im

        # Common color limits for True/Pred
        tmin = np.nanmin([truth_denorm, preds_denorm])
        tmax = np.nanmax([truth_denorm, preds_denorm])

        # True
        im0 = _plot_heat(axes_h[0], truth_denorm, "True T(t, z)", vmin=tmin, vmax=tmax)
        axes_h[0].set_ylabel("Depth index")

        # Pred
        im1 = _plot_heat(axes_h[1], preds_denorm, "Pred T(t, z)", vmin=tmin, vmax=tmax)

        # Residual: symmetric scale around 0
        dmax = np.nanmax(np.abs(diff_denorm))
        im2 = _plot_heat(axes_h[2], diff_denorm, "ΔT = Pred - True",
                         vmin=-dmax, vmax=dmax, cmap="coolwarm")

        # Colorbars
        cbar0 = fig_h.colorbar(im0, ax=axes_h[0], fraction=0.046, pad=0.04)
        cbar0.set_label("Temperature (°C)")
        cbar1 = fig_h.colorbar(im1, ax=axes_h[1], fraction=0.046, pad=0.04)
        cbar1.set_label("Temperature (°C)")
        cbar2 = fig_h.colorbar(im2, ax=axes_h[2], fraction=0.046, pad=0.04)
        cbar2.set_label("ΔT (°C)")

        fig_h.suptitle(f"{label_prefix} heatmaps | Run {run_id} | sim n={n} | Wx={Wx}, Wy={Wy}")
        fig_h.tight_layout(rect=[0, 0.03, 1, 0.92])

        fname_h = os.path.join(outdir, f"{split}_heatmaps_run_{run_id}_sim_{n}.png")
        plt.savefig(fname_h, dpi=150)
        plt.close(fig_h)
        print(f"Saved heatmaps for sim {n} to {fname_h}")


# ---------------------------
# CLI
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", required=True, help="MLflow run_id to visualize")
    ap.add_argument("--num_sims", type=int, default=3, help="Number of random parameter sets to plot")
    ap.add_argument("--max_days", type=int, default=None, help="Optional limit on number of days to plot")
    ap.add_argument("--device", default=None, help="Override device (cpu/cuda), default uses run config or auto")
    ap.add_argument("--outdir", default="plots_timeseries", help="Directory to save plots")
    ap.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test", "valtest", "val+test"],
        help="Which period to plot (default: test; 'valtest' = val+test combined)",
    )
    args = ap.parse_args()

    make_time_series_plots_for_run(
        run_id=args.run_id,
        num_sims=args.num_sims,
        max_days=args.max_days,
        device=args.device,
        outdir=args.outdir,
        split=args.split,
    )


if __name__ == "__main__":
    main()
