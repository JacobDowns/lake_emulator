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

# Your model definitions & data loader from training code
from models import ModelMLP, ModelRNN, ModelCNN1D
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
# Model reconstruction
# ---------------------------

def build_model_from_config(cfg, Du, Dz, P):
    """
    Rebuild the model based on the saved training config (cfg) and data dims.
    """
    model_type = cfg.get("model", "rnn")
    Wx = cfg["Wx"]
    Wy = cfg["Wy"]

    head_hidden_str = cfg.get("head_hidden", "256,256")
    head_hidden = [int(x) for x in str(head_hidden_str).split(",") if x.strip()]
    head_dropout = float(cfg.get("head_dropout", 0.0))

    if model_type == "mlp":
        model = ModelMLP(
            W_x=Wx,
            Du=Du,
            P=P,
            Dz=Dz,
            W_y=Wy,
            mlp_hidden=head_hidden,
            mlp_dropout=head_dropout,
        )
    elif model_type == "rnn":
        cell = cfg.get("cell", "gru")
        hidden = int(cfg.get("hidden", 128))
        num_layers = int(cfg.get("num_layers", 1))
        rnn_dropout = float(cfg.get("rnn_dropout", 0.0))
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
        )
    elif model_type == "cnn":
        cnn_layers = int(cfg.get("cnn_layers", 2))
        cnn_channels = int(cfg.get("cnn_channels", 128))
        cnn_kernel = int(cfg.get("cnn_kernel", 5))
        model = ModelCNN1D(
            Du=Du,
            P=P,
            Dz=Dz,
            W_y=Wy,
            layers=cnn_layers,
            channels=cnn_channels,
            kernel=cnn_kernel,
            head_hidden=head_hidden,
            head_dropout=head_dropout,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

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
    split: str = "test",   # "train" | "val" | "test"
):
    """
    For a given MLflow run:
      - load best checkpoint
      - rebuild model
      - reload data with the same splits & normalization
      - for a few random simulations (parameter sets),
        run the model over the chosen split and plot:
          * time series (True vs Pred + ΔT)
          * heatmaps (True, Pred, ΔT)

    Supports arbitrary Wy >= 1. Overlapping predictions from multiple windows
    are averaged per time step.
    """
    os.makedirs(outdir, exist_ok=True)

    # 1) Load checkpoint & config
    ckpt_path, ckpt = load_checkpoint_from_mlflow(run_id)
    cfg = ckpt["config"]  # this is vars(args) from train.py
    Wx = cfg["Wx"]
    Wy = cfg["Wy"]

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

    if split == "train":
        weather = data["weather_data_train"]   # (T, Du)
        outputs = data["output_data_train"]    # (N, T, Dz)
    elif split == "val":
        weather = data["weather_data_val"]
        outputs = data["output_data_val"]
    elif split == "test":
        weather = data["weather_data_test"]
        outputs = data["output_data_test"]
    else:
        raise ValueError("split must be one of 'train', 'val', 'test'")

    params = data["params_data"]               # (N, P)
    norms = data["norms"]

    T_split, Du = weather.shape
    N, T_out, Dz = outputs.shape
    assert T_split == T_out, "Time dimension mismatch for chosen split."

    P = params.shape[1]

    # 3) Rebuild model and load weights
    model = build_model_from_config(cfg, Du, Dz, P)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()

    # 4) Choose random simulation indices
    all_ids = list(range(N))
    random.shuffle(all_ids)
    chosen_ids = all_ids[: min(num_sims, N)]

    # 5) Generate predictions over full chosen split for each sim
    if T_split < Wx:
        raise ValueError(f"Period too short (T={T_split}) for Wx={Wx}.")
    print(f"Generating predictions over {split} period: T={T_split}, Wx={Wx}, Wy={Wy}")

    num_windows = T_split - Wx + 1

    for n in chosen_ids:
        p_vec = torch.from_numpy(params[n]).to(device=device, dtype=torch.float32)  # (P,)

        # We'll accumulate predictions and counts for averaging overlaps
        preds_sum = np.zeros((T_split, Dz), dtype=np.float32)
        preds_count = np.zeros(T_split, dtype=np.int32)

        truth = outputs[n].copy()  # (T_split, Dz), still normalized

        # For each window starting at 'start' with indices [start ... start+Wx-1]:
        # the model predicts Wy steps aligned to the LAST Wy days of the window.
        # So predicted time indices are:
        #   t_idx = start + (Wx - Wy) + j,  j = 0 .. Wy-1
        for start in range(num_windows):
            end = start + Wx
            x_win = weather[start:end, :]   # (Wx, Du)
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
                if 0 <= t_idx < T_split:
                    preds_sum[t_idx, :] += y_pred_np[j, :]
                    preds_count[t_idx] += 1

        # Build final preds array, average where we have coverage
        preds = np.full((T_split, Dz), np.nan, dtype=np.float32)
        valid_mask = preds_count > 0
        preds[valid_mask, :] = (
            preds_sum[valid_mask, :] / preds_count[valid_mask][:, None]
        )

        # De-normalize true and predicted
        truth_denorm_full = denorm_outputs(truth, norms)  # (T_split, Dz)
        preds_denorm_full = denorm_outputs(preds, norms)  # (T_split, Dz) with NaNs for unpredicted times

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
        time_axis = np.arange(t_min, t_min + T_plot)

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

        axes_pairs[-1][1].set_xlabel(f"{split.capitalize()} time index (days since start of {split} period)")

        fig_ts.suptitle(f"{split.capitalize()} time series | Run {run_id} | sim n={n} | Wx={Wx}, Wy={Wy}")
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
            ax.set_xlabel(f"{split.capitalize()} time index (days since start of {split} period)")
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

        fig_h.suptitle(f"{split.capitalize()} heatmaps | Run {run_id} | sim n={n} | Wx={Wx}, Wy={Wy}")
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
    ap.add_argument("--split", default="test", choices=["train", "val", "test"],
                    help="Which split to plot (default: test)")
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
