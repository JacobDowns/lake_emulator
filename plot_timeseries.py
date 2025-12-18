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

# Model + data loader (UPDATED)
from models import ModelRNN
from lake_dataset import load_data  # uses smoothing windows/pad_mode


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

    return mu.astype(np.float32), sd.astype(np.float32)


def denorm_outputs(y_norm: np.ndarray, norms: dict) -> np.ndarray:
    """
    y_norm: (T, Dz) or (B, Dz).
    Returns same shape in real units.
    """
    if norms is None:
        return y_norm
    mu, sd = _norms_output_arrays(norms)  # (1, Dz)
    return y_norm * sd + mu


# ---------------------------
# Seasonal features (only if you still use them)
# ---------------------------

def build_seasonal_features(doy: np.ndarray) -> np.ndarray:
    angle = 2.0 * np.pi * (doy.astype(np.float32) / 365.0)
    return np.stack([np.sin(angle), np.cos(angle)], axis=-1).astype(np.float32)


# ---------------------------
# MLflow checkpoint loading
# ---------------------------

def load_checkpoint_from_mlflow(run_id: str, dst_dir: str = "downloaded_artifacts"):
    os.makedirs(dst_dir, exist_ok=True)
    client = MlflowClient()
    local_path = client.download_artifacts(run_id, "checkpoints/best.pt", dst_dir)
    ckpt = torch.load(local_path, map_location="cpu")
    return local_path, ckpt


# ---------------------------
# Model reconstruction (UPDATED: simplified GRU-only)
# ---------------------------

def build_model_from_config(cfg, Du: int, Dz: int, P: int):
    """
    Rebuild ModelRNN from the saved config in best.pt.
    This matches your simplified models.py: GRU-only, no attention, no cell choice.
    """
    Wy = int(cfg["Wy"])

    head_hidden_str = cfg.get("head_hidden", "256,256")
    head_hidden = [int(x) for x in str(head_hidden_str).split(",") if x.strip()]
    head_dropout = float(cfg.get("head_dropout", 0.0))

    hidden = int(cfg.get("hidden", 128))
    num_layers = int(cfg.get("num_layers", 2))
    rnn_dropout = float(cfg.get("rnn_dropout", 0.0))

    model = ModelRNN(
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
    return model


def _parse_int_list_csv(s) -> list[int]:
    """
    "7,30" -> [7,30]; ""/None -> []
    """
    if s is None:
        return []
    if isinstance(s, (list, tuple)):
        return [int(x) for x in s]
    s = str(s).strip()
    if not s:
        return []
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if tok:
            out.append(int(tok))
    return out


# ---------------------------
# Main plotting logic
# ---------------------------

def make_time_series_plots_for_run(
    run_id: str,
    num_sims: int = 3,
    max_days: int = None,
    device: str = None,
    outdir: str = "plots_timeseries",
    split: str = "test",   # "train" | "val" | "test" | "valtest"
):
    os.makedirs(outdir, exist_ok=True)

    # 1) Load checkpoint & config
    _, ckpt = load_checkpoint_from_mlflow(run_id)
    cfg = ckpt["config"]
    Wx = int(cfg["Wx"])
    Wy = int(cfg["Wy"])

    # Whether seasonal features were used during training (optional)
    use_seasonal_raw = cfg.get("use_seasonal_features", 0)
    if isinstance(use_seasonal_raw, str):
        use_seasonal = bool(int(use_seasonal_raw))
    else:
        use_seasonal = bool(use_seasonal_raw)

    # Smoothing config (NEW: consistent with lake_dataset.load_data)
    smooth_windows = _parse_int_list_csv(cfg.get("smooth_windows", ""))
    smooth_pad_mode = str(cfg.get("smooth_pad_mode", "reflect"))

    if device is None:
        device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    # 2) Reload data with same split_years & normalization + smoothing
    split_years = cfg.get("split_years", [2018, 2020, 2025])
    if isinstance(split_years, str):
        split_years = [int(x) for x in split_years.split(",") if x.strip()]

    data = load_data(
        weather_path=cfg.get("weather", "data/parsed_data/weather_data.npy"),
        output_path=cfg.get("outputs", "data/parsed_data/output_data.npy"),
        params_path=cfg.get("params",  "data/parsed_data/parameter_data.npy"),
        bathymetry_path=cfg.get("bathymetry_path", "data/BearLake_inputs_outputs/inputs/BearLake_bathy.csv"),
        split_years=split_years,
        normalize=True,
        smooth_windows=smooth_windows,
        smooth_pad_mode=smooth_pad_mode,
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

    params = data["params_data"]  # (N, P) normalized
    norms = data["norms"]

    # Choose which period to visualize
    if split == "train":
        weather_base = w_tr
        outputs = o_tr
        doy = doy_tr
        label_prefix = "Train"
        split_name = "train"
    elif split == "val":
        weather_base = w_va
        outputs = o_va
        doy = doy_va
        label_prefix = "Validation"
        split_name = "val"
    elif split == "test":
        weather_base = w_te
        outputs = o_te
        doy = doy_te
        label_prefix = "Test"
        split_name = "test"
    elif split in ("valtest", "val+test"):
        weather_base = np.concatenate([w_va, w_te], axis=0)
        outputs = np.concatenate([o_va, o_te], axis=1)
        doy = np.concatenate([doy_va, doy_te], axis=0)
        label_prefix = "Val+Test"
        split_name = "valtest"
    else:
        raise ValueError("split must be one of 'train', 'val', 'test', 'valtest', 'val+test'")

    # Add seasonal features if used during training (optional)
    if use_seasonal:
        extra = build_seasonal_features(doy)  # (T,2)
        weather = np.concatenate([weather_base, extra], axis=-1)
    else:
        weather = weather_base

    T_period, Du_current = weather.shape
    N, T_out, Dz_current = outputs.shape
    assert T_period == T_out, "Time dimension mismatch for chosen period."

    # Dims from checkpoint if present (else infer)
    Du_model = int(ckpt.get("Du", Du_current))
    Dz_model = int(ckpt.get("Dz", Dz_current))
    P_model = int(ckpt.get("P", params.shape[1]))

    if Du_model != Du_current:
        raise ValueError(
            f"Input feature dimension mismatch: model Du={Du_model} but weather has Du={Du_current}. "
            f"(Smoothing windows or seasonal features mismatch?)"
        )
    if Dz_model != Dz_current:
        raise ValueError(f"Output dimension mismatch: model Dz={Dz_model} but outputs have Dz={Dz_current}.")

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

        preds_sum = np.zeros((T_period, Dz_model), dtype=np.float32)
        preds_count = np.zeros(T_period, dtype=np.int32)

        truth = outputs[n].copy()  # (T_period, Dz), normalized

        for start in range(num_windows):
            end = start + Wx
            x_win = weather[start:end, :]  # (Wx, Du)
            x_win_t = torch.from_numpy(x_win).unsqueeze(0).to(device=device, dtype=torch.float32)
            p_t = p_vec.unsqueeze(0)

            with torch.no_grad():
                y_pred = model(x_win_t, p_t)

            # Ensure (1, Wy, Dz)
            if y_pred.dim() == 2:
                y_pred = y_pred.unsqueeze(1)
            y_pred_np = y_pred.detach().cpu().numpy()[0]  # (Wy, Dz)

            # Predictions align to last Wy indices of window
            base_t = start + (Wx - Wy)
            for j in range(Wy):
                t_idx = base_t + j
                if 0 <= t_idx < T_period:
                    preds_sum[t_idx, :] += y_pred_np[j, :]
                    preds_count[t_idx] += 1

        preds = np.full((T_period, Dz_model), np.nan, dtype=np.float32)
        valid_mask = preds_count > 0
        preds[valid_mask, :] = preds_sum[valid_mask, :] / preds_count[valid_mask][:, None]

        truth_denorm_full = denorm_outputs(truth, norms)
        preds_denorm_full = denorm_outputs(preds, norms)

        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) == 0:
            print(f"No valid predictions for sim {n}, skipping.")
            continue

        t_min = valid_indices[0]
        t_max = valid_indices[-1]
        if max_days is not None:
            t_max = min(t_max, t_min + max_days - 1)

        truth_denorm = truth_denorm_full[t_min:t_max + 1, :]
        preds_denorm = preds_denorm_full[t_min:t_max + 1, :]
        T_plot = truth_denorm.shape[0]
        time_axis = np.arange(T_plot)

        Dz = Dz_model
        if Dz >= 3:
            chosen_depth_indices = [0, Dz // 2, Dz - 1]
        else:
            chosen_depth_indices = list(range(Dz))

        # -------------------- time series --------------------
        fig_ts, axes_ts = plt.subplots(
            len(chosen_depth_indices) * 2,
            1,
            figsize=(10, 5.5 * len(chosen_depth_indices)),
            sharex=True
        )

        if len(chosen_depth_indices) == 1:
            axes_pairs = [(axes_ts[0], axes_ts[1])]
        else:
            axes_pairs = [(axes_ts[2 * i], axes_ts[2 * i + 1]) for i in range(len(chosen_depth_indices))]

        for (temp_ax, diff_ax), d in zip(axes_pairs, chosen_depth_indices):
            true_series = truth_denorm[:, d]
            pred_series = preds_denorm[:, d]
            diff_series = pred_series - true_series

            temp_ax.plot(time_axis, true_series, label="True", linewidth=1.5)
            temp_ax.plot(time_axis, pred_series, label="Pred", linestyle="--", linewidth=1.2)
            temp_ax.set_ylabel(f"T (°C)\nDepth idx {d}")
            temp_ax.grid(True, alpha=0.3)
            temp_ax.legend(loc="upper right")

            diff_ax.plot(time_axis, diff_series, label="Pred - True", linestyle=":", linewidth=1.2)
            diff_ax.axhline(0.0, linestyle="-", linewidth=0.8, alpha=0.5)
            diff_ax.set_ylabel("ΔT (°C)")
            diff_ax.grid(True, alpha=0.3)
            diff_ax.legend(loc="upper right")

        axes_pairs[-1][1].set_xlabel(
            f"{label_prefix} time index (days since start of {label_prefix.lower()} period)"
        )

        fig_ts.suptitle(f"{label_prefix} time series | Run {run_id} | sim n={n} | Wx={Wx}, Wy={Wy}")
        fig_ts.tight_layout(rect=[0, 0.03, 1, 0.95])

        fname_ts = os.path.join(outdir, f"{split_name}_timeseries_run_{run_id}_sim_{n}.png")
        plt.savefig(fname_ts, dpi=150)
        plt.close(fig_ts)
        print(f"Saved time-series plot for sim {n} to {fname_ts}")

        # -------------------- heatmaps --------------------
        diff_denorm = preds_denorm - truth_denorm

        fig_h, axes_h = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)

        def _plot_heat(ax, data, title, vmin=None, vmax=None, cmap="viridis"):
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

        tmin = np.nanmin([truth_denorm, preds_denorm])
        tmax = np.nanmax([truth_denorm, preds_denorm])

        im0 = _plot_heat(axes_h[0], truth_denorm, "True T(t, z)", vmin=tmin, vmax=tmax)
        axes_h[0].set_ylabel("Depth index")

        im1 = _plot_heat(axes_h[1], preds_denorm, "Pred T(t, z)", vmin=tmin, vmax=tmax)

        dmax = np.nanmax(np.abs(diff_denorm))
        im2 = _plot_heat(axes_h[2], diff_denorm, "ΔT = Pred - True",
                         vmin=-dmax, vmax=dmax, cmap="coolwarm")

        cbar0 = fig_h.colorbar(im0, ax=axes_h[0], fraction=0.046, pad=0.04)
        cbar0.set_label("Temperature (°C)")
        cbar1 = fig_h.colorbar(im1, ax=axes_h[1], fraction=0.046, pad=0.04)
        cbar1.set_label("Temperature (°C)")
        cbar2 = fig_h.colorbar(im2, ax=axes_h[2], fraction=0.046, pad=0.04)
        cbar2.set_label("ΔT (°C)")

        fig_h.suptitle(f"{label_prefix} heatmaps | Run {run_id} | sim n={n} | Wx={Wx}, Wy={Wy}")
        fig_h.tight_layout(rect=[0, 0.03, 1, 0.92])

        fname_h = os.path.join(outdir, f"{split_name}_heatmaps_run_{run_id}_sim_{n}.png")
        plt.savefig(fname_h, dpi=150)
        plt.close(fig_h)
        print(f"Saved heatmaps for sim {n} to {fname_h}")


# ---------------------------
# CLI
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", required=True, help="MLflow run_id to visualize")
    ap.add_argument("--num_sims", type=int, default=3)
    ap.add_argument("--max_days", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--outdir", default="plots_timeseries")
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
