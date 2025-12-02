#!/usr/bin/env python3
import os
import argparse

import numpy as np
import pandas as pd
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from mlflow.tracking import MlflowClient

from lake_dataset import load_data, build_seasonal_features, LakeWindowDataset
from models import ModelRNN


# ---------------------------
# Norm helpers
# ---------------------------
def _norms_output_std(norms) -> np.ndarray:
    sigma = np.array(norms["output_std"])
    # norms["output_std"] is usually (1,1,Dz) or (1,Dz)
    if sigma.ndim == 3:
        sigma = sigma.reshape(-1)
    elif sigma.ndim == 2:
        sigma = sigma.reshape(-1)
    return sigma.astype(np.float32)  # (Dz,)


# ---------------------------
# Model reconstruction
# ---------------------------
def build_model_from_config(cfg, Du, Dz, P):
    """
    Rebuild ModelRNN using the same config keys as in train.py.
    Only passes arguments that ModelRNN actually supports.
    """
    Wx = int(cfg["Wx"])
    Wy = int(cfg["Wy"])

    cell = cfg.get("cell", "gru")
    hidden = int(cfg.get("hidden", 128))
    num_layers = int(cfg.get("num_layers", 2))
    rnn_dropout = float(cfg.get("rnn_dropout", 0.0))

    # head_hidden may be stored as string "256,256" or list/int
    head_hidden_cfg = cfg.get("head_hidden", "256,256")
    if isinstance(head_hidden_cfg, str):
        head_hidden = [int(x) for x in head_hidden_cfg.split(",") if x.strip()]
    else:
        head_hidden = head_hidden_cfg
    head_dropout = float(cfg.get("head_dropout", 0.0))

    # RNN parameter conditioning flags
    param_in_rnn = bool(int(cfg.get("param_in_rnn", 1)))
    param_in_head = bool(int(cfg.get("param_in_head", 1)))

    # Optional temporal attention
    use_attention = bool(int(cfg.get("use_attention", 0)))
    attention_hidden = int(cfg.get("attention_hidden", hidden))

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
        param_in_rnn=param_in_rnn,
        param_in_head=param_in_head,
        use_attention=use_attention
    )
    return model


# ---------------------------
# Fast RMSE computation using LakeWindowDataset
# ---------------------------
def compute_rmse_valtest_by_param_fast(
    run_id: str,
    device: str | None = None,
    max_sims: int | None = None,
    batch_size: int = 512,
):
    """
    Efficiently compute RMSE (in °C) over the combined val+test period for each
    parameter vector (simulation), using LakeWindowDataset with return_ids=True.

    Returns:
      params_arr: (N_used, P)
      rmse_arr:   (N_used,)
      param_names: list[str]
      cfg, norms  : for further use
    """
    os.makedirs("downloaded_artifacts", exist_ok=True)
    client = MlflowClient()

    # --- Load checkpoint ---
    ckpt_path = client.download_artifacts(run_id, "checkpoints/best.pt", "downloaded_artifacts")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["config"]

    Wx = int(cfg["Wx"])
    Wy = int(cfg["Wy"])
    use_seasonal = bool(int(cfg.get("use_seasonal_features", 0)))

    if device is None:
        device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    # --- Reload data with same normalization/splits ---
    split_years = cfg.get("split_years", [2018, 2020, 2025])
    if isinstance(split_years, str):
        split_years = [int(x) for x in split_years.split(",")]

    data = load_data(
        weather_path=cfg.get("weather", "data/parsed_data/weather_data.npy"),
        output_path=cfg.get("outputs", "data/parsed_data/output_data.npy"),
        params_path=cfg.get("params", "data/parsed_data/parameter_data.npy"),
        split_years=split_years,
        normalize=True,
    )

    w_va = data["weather_data_val"]
    w_te = data["weather_data_test"]
    o_va = data["output_data_val"]
    o_te = data["output_data_test"]
    doy_va = data["doy_val"]
    doy_te = data["doy_test"]
    params = data["params_data"]
    norms = data["norms"]

    sigma = _norms_output_std(norms)  # (Dz,)

    # --- Build combined val+test period ---
    weather_drivers = np.concatenate([w_va, w_te], axis=0)  # (T_period, Du)
    outputs_vt = np.concatenate([o_va, o_te], axis=1)       # (N, T_period, Dz)
    doy_vt = np.concatenate([doy_va, doy_te], axis=0)

    if use_seasonal:
        extra = build_seasonal_features(doy_vt)             # (T_period, 2)
        weather_vt = np.concatenate([weather_drivers, extra], axis=-1)
    else:
        weather_vt = weather_drivers

    T_period, Du_current = weather_vt.shape
    N, T_out, Dz_current = outputs_vt.shape
    assert T_period == T_out
    P = params.shape[1]

    Du_model = int(ckpt.get("Du", Du_current))
    Dz_model = int(ckpt.get("Dz", Dz_current))
    P_model = int(ckpt.get("P", P))

    if Du_model != Du_current:
        raise ValueError(
            f"Feature dimension mismatch: model Du={Du_model}, "
            f"val+test weather has Du={Du_current} (seasonal features mismatch?)."
        )
    if Dz_model != Dz_current:
        raise ValueError(
            f"Depth dimension mismatch: model Dz={Dz_model}, "
            f"outputs have Dz={Dz_current}."
        )

    Dz = Dz_model

    # --- Build model ---
    model = build_model_from_config(cfg, Du_model, Dz_model, P_model)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()

    # --- Choose subset of sims if desired ---
    if max_sims is not None and max_sims < N:
        sim_ids = np.random.choice(N, size=max_sims, replace=False)
        sim_ids.sort()
    else:
        sim_ids = np.arange(N, dtype=int)

    # --- Build windowed dataset over val+test, with sim IDs ---
    # LakeWindowDataset must support return_ids=True → returns (sim_id, x, p, y).
    ds = LakeWindowDataset(
        weather=weather_vt,
        outputs=outputs_vt,
        params=params,
        W_x=Wx,
        W_y=Wy,
        trial_ids=sim_ids,
        extra_features=None,   # already appended seasonal features above
        return_ids=True,
    )

    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=0
    )

    # Accumulate SSE per sim and per depth in normalized space
    n_used = len(sim_ids)
    sse_per_sim_depth = np.zeros((n_used, Dz), dtype=np.float64)
    count_samples_per_sim = np.zeros(n_used, dtype=np.int64)
    sse_phys_per_sim = np.zeros(n_used, dtype=np.float64)
    total_points_per_sim = np.zeros(n_used, dtype=np.int64)

    # map global sim_id -> local index
    simid_to_local = {int(s): i for i, s in enumerate(sim_ids)}

    for sim_id_batch, x_batch, p_batch, y_batch in loader:
        sim_id_batch = sim_id_batch.numpy()  # (B,)
        x_batch = x_batch.to(device)
        p_batch = p_batch.to(device)
        y_batch = y_batch.to(device)

        if y_batch.dim() == 2:
            y_batch = y_batch.unsqueeze(1)  # (B, 1, Dz)

        with torch.no_grad():
            pred = model(x_batch, p_batch)
        if pred.dim() == 2:
            pred = pred.unsqueeze(1)  # (B, 1, Dz)

        diff_norm = pred - y_batch           # (B, W_y, Dz)
        diff2 = (diff_norm ** 2).sum(dim=1)  # sum over time -> (B, Dz)

        diff2_np = diff2.cpu().numpy()

        for i in range(len(sim_id_batch)):
            g = int(sim_id_batch[i])
            local = simid_to_local[g]
            sse_per_sim_depth[local] += diff2_np[i]  # accumulate per depth
            count_samples_per_sim[local] += 1

    # Convert SSE in normalized space to SSE in °C^2, then RMSE in °C
    sigma_sq = sigma ** 2  # (Dz,)

    rmse_arr = np.zeros(n_used, dtype=np.float32)
    for j in range(n_used):
        if count_samples_per_sim[j] == 0:
            rmse_arr[j] = np.nan
            continue
        # total points: samples * Wy * Dz
        total_points = count_samples_per_sim[j] * Wy * Dz
        sse_phys = np.sum(sse_per_sim_depth[j] * sigma_sq)  # sum over depths
        sse_phys_per_sim[j] = sse_phys
        total_points_per_sim[j] = total_points
        rmse_arr[j] = np.sqrt(sse_phys / total_points)

    # Parameters for used sims
    params_arr = params[sim_ids]  # (N_used, P_model)

    # Use meaningful names if P_model == 4 (your Lahontan params)
    if P_model == 4:
        param_names = ["cdrn", "eta", "alb_snow", "alb_slush"]
    else:
        param_names = [f"param_{k}" for k in range(P_model)]

    return (
        params_arr,
        rmse_arr,
        sse_phys_per_sim,
        total_points_per_sim,
        param_names,
        cfg,
        norms,
    )


# ---------------------------
# Pairplot driver (with fixed axis ranges + diagonal RMSE curves)
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", required=True, help="MLflow run_id to analyze")
    ap.add_argument("--max_sims", type=int, default=None,
                    help="Optional max number of simulations to subsample")
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--device", default=None)
    ap.add_argument("--outdir", default="plots_param_rmse",
                    help="Where to save pairplot")
    ap.add_argument("--nbins_diag", type=int, default=15,
                    help="Number of bins for diagonal RMSE vs param curves")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    (
        params_arr,
        rmse_arr,
        sse_phys_arr,
        total_points_arr,
        param_names,
        cfg,
        norms,
    ) = compute_rmse_valtest_by_param_fast(
        run_id=args.run_id,
        device=args.device,
        max_sims=args.max_sims,
        batch_size=args.batch_size,
    )

    # Build DataFrame for seaborn
    df = pd.DataFrame(params_arr, columns=param_names)
    df["rmse"] = rmse_arr
    df["sse_phys"] = sse_phys_arr
    df["total_points"] = total_points_arr
    df["mse"] = df["sse_phys"] / df["total_points"]

    # Clean up NaNs / infinities to avoid axis issues
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["rmse"])
    if df.empty:
        print("No valid samples to plot after cleaning NaNs/inf in RMSE.")
        return

    # Also drop rows where parameters are NaN (just in case)
    df = df.dropna(subset=param_names)
    if df.empty:
        print("No valid samples to plot after cleaning NaNs in parameters.")
        return

    sns.set(style="ticks", context="notebook")

    # Make the pairplot
    g = sns.pairplot(
        df,
        vars=param_names,
        hue="rmse",
        palette="viridis",
        corner=True,
        dropna=True,
        plot_kws=dict(alpha=0.7, s=25, edgecolor="none"),
        diag_kws=dict(fill=True, alpha=0.5),
    )
    if getattr(g, "_legend", None) is not None:
        g._legend.remove()

    # ---- Compute min/max for axis ranges ----
    mins = df[param_names].min()
    maxs = df[param_names].max()

    def _pad(lo, hi, frac=0.05):
        span = hi - lo
        if span <= 0:
            # handle essentially-constant parameters
            span = abs(lo) if lo != 0 else 1.0
        pad = frac * span
        return lo - pad, hi + pad

    n = len(param_names)

    # ---- First, fix axis ranges for off-diagonal panels ----
    for i in range(n):
        for j in range(n):
            ax = g.axes[i, j] if g.axes is not None else None
            if ax is None:
                continue

            x_name = param_names[j]
            y_name = param_names[i]

            if i == j:
                # We'll entirely redraw diagonals, so only x-lim is needed there (later).
                continue
            else:
                # Scatter panels — set both x and y limits
                x_lo, x_hi = _pad(mins[x_name], maxs[x_name])
                y_lo, y_hi = _pad(mins[y_name], maxs[y_name])
                ax.set_xlim(x_lo, x_hi)
                ax.set_ylim(y_lo, y_hi)

    fig = getattr(g, "figure", None) or getattr(g, "fig", None) or plt.gcf()

    # Scale figure for readability as dimensionality grows
    base = 3.0
    size = max(10.0, base * len(param_names))
    fig.set_size_inches(size, size)

    # ---- Now replace diagonals with marginalized RMSE vs param curves ----
    for d, pname in enumerate(param_names):
        old_ax = g.axes[d, d] if g.axes is not None else None
        if old_ax is None:
            continue

        # Replace the diagonal axis with a fresh one to avoid shared-axis links
        subspec = old_ax.get_subplotspec()
        old_ax.remove()
        ax = fig.add_subplot(subspec)
        g.axes[d, d] = ax  # keep grid consistent

        vals = df[pname].values
        sse_phys = df["sse_phys"].values
        total_points = df["total_points"].values

        if len(vals) == 0:
            continue

        # Bin along parameter axis
        vmin, vmax = vals.min(), vals.max()
        if vmin == vmax:
            # constant param; just plot a single point with the marginalized RMSE
            total_pts = total_points.sum()
            rmse_single = np.sqrt(sse_phys.sum() / total_pts) if total_pts > 0 else np.nan
            ax.scatter([vmin], [rmse_single], s=30)
            ax.set_xlim(*_pad(vmin, vmax))
            ax.set_ylabel("RMSE")
            ax.set_title(pname)
            ax.grid(True, alpha=0.3)
            continue

        nbins = args.nbins_diag
        edges = np.linspace(vmin, vmax, nbins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])

        marg_rmse = np.full(nbins, np.nan, dtype=np.float64)
        for b in range(nbins):
            if b < nbins - 1:
                mask = (vals >= edges[b]) & (vals < edges[b + 1])
            else:
                # include right edge on last bin
                mask = (vals >= edges[b]) & (vals <= edges[b + 1])
            if mask.any():
                # marginalized MSE = total SSE / total points, then take sqrt for RMSE
                bin_sse = sse_phys[mask].sum()
                bin_pts = total_points[mask].sum()
                if bin_pts > 0:
                    marg_rmse[b] = np.sqrt(bin_sse / bin_pts)

        valid = ~np.isnan(marg_rmse)
        if valid.any():
            ax.plot(
                centers[valid],
                marg_rmse[valid],
                marker="o",
                linestyle="-",
                linewidth=1.5,
                markersize=4,
            )
        ax.set_ylabel("RMSE")
        ax.set_title(pname)
        ax.grid(True, alpha=0.3)

        x_lo, x_hi = _pad(vmin, vmax)
        ax.set_xlim(x_lo, x_hi)

    # Add continuous colorbar for RMSE
    rmse_min, rmse_max = df["rmse"].min(), df["rmse"].max()
    norm = matplotlib.colors.Normalize(rmse_min, rmse_max)
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])
    axes_list = [ax for ax in g.axes.ravel() if ax is not None] if g.axes is not None else fig.axes
    fig.colorbar(sm, ax=axes_list, fraction=0.03, pad=0.02, label="RMSE")

    fig.suptitle(
        f"Parameter vs RMSE (val+test) | run_id={args.run_id}",
        y=1.02,
        fontsize=14,
    )

    # Add extra breathing room so axis labels/ticks are not clipped
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out_path = os.path.join(args.outdir, f"pairplot_rmse_{args.run_id}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved pairplot to {out_path}")


if __name__ == "__main__":
    main()
