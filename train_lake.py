# train_lake.py
import os, argparse, socket
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import mlflow
import mlflow.pytorch

from lake_data import (
    split_meta_and_drivers,
    build_fourier_from_year_doy,
    strict_test_split_by_year_from_meta,
    fit_norms_on_train,
    apply_norms,
    WindowNextDayDataset,
)
from lake_models import LakeProfileSeq

# tiny helper; we keep this here to avoid a utils dependency
def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

def parse_args():
    ap = argparse.ArgumentParser()
    # Data (weather_data.npy = [YEAR, DOY, drivers...])
    ap.add_argument("--weather_npy", default="data/parsed_data/weather_data.npy")
    ap.add_argument("--params_npy",  default="data/parsed_data/parameter_data.npy")
    ap.add_argument("--outputs_npy", default="data/parsed_data/output_data.npy")

    # Seasonal Fourier features from YEAR/DOY
    ap.add_argument("--seasonal_harmonics", default="1,2",
                    help="Comma list of k for sin/cos(2π k * DOY / year_len). Empty to disable.")

    # Model
    ap.add_argument("--cell_type", default="gru", choices=["gru","lstm","rnn"])
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--num_layers", type=int, default=1)
    ap.add_argument("--dropout_fc", type=float, default=0.0)
    ap.add_argument("--smooth_lambda", type=float, default=0.0)

    # Training / split
    ap.add_argument("--L", type=int, default=90)
    ap.add_argument("--lead", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--test_years", nargs=2, type=int, default=[2020, 2025],
                    help="Inclusive year range for TEST holdout")
    ap.add_argument("--val_frac_pretest", type=float, default=0.2)

    # Infra
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--experiment", default=None)
    ap.add_argument("--log_model_on_improve", action="store_true")
    return ap.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.artifacts, exist_ok=True)

    # ---- Load arrays ----
    weather_with_meta = np.load(args.weather_npy)   # (T, 2 + Du_base)
    params = np.load(args.params_npy)               # (N, P)
    outputs = np.load(args.outputs_npy)             # (N, T, Dz)

    # ---- Split meta/drivers & consistency checks ----
    meta, drivers = split_meta_and_drivers(weather_with_meta)
    year, doy = meta[:, 0], meta[:, 1]
    T, Du_base = drivers.shape
    N, T2, Dz = outputs.shape
    assert T == T2, "Time mismatch between weather and outputs."
    P = params.shape[1]

    # ---- Strict split by YEAR (from meta) ----
    t_train, t_val, t_test = strict_test_split_by_year_from_meta(
        meta_year=year, L=args.L, lead=args.lead,
        test_years=tuple(args.test_years), val_frac_pretest=args.val_frac_pretest
    )

    # ---- Build Fourier from meta YEAR/DOY and append ----
    harm = [int(h) for h in args.seasonal_harmonics.split(",") if h.strip()] if args.seasonal_harmonics.strip() != "" else []
    if len(harm) > 0:
        fourier = build_fourier_from_year_doy(year, doy, harmonics=tuple(harm))  # (T, 2K)
        drivers_all = np.concatenate([drivers, fourier], axis=1)
        n_fourier = fourier.shape[1]
    else:
        drivers_all = drivers
        n_fourier = 0
    Du_total = drivers_all.shape[1]

    # ---- Fit norms on TRAIN windows, excluding Fourier tail ----
    x_mean, x_std, p_mean, p_std, _ = fit_norms_on_train(
        drivers_all, params, t_train, args.L, exclude_tail_cols=n_fourier
    )
    drivers_std, params_std = apply_norms(
        drivers_all, params, x_mean, x_std, p_mean, p_std, exclude_tail_cols=n_fourier
    )

    # ---- Datasets & loaders ----
    train_ds = WindowNextDayDataset(drivers_std, params_std, outputs, L=args.L, lead=args.lead, t_indices=t_train)
    val_ds   = WindowNextDayDataset(drivers_std, params_std, outputs, L=args.L, lead=args.lead, t_indices=t_val)
    test_ds  = WindowNextDayDataset(drivers_std, params_std, outputs, L=args.L, lead=args.lead, t_indices=t_test)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=max(1, args.batch_size // 2), shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=max(1, args.batch_size // 2), shuffle=False)

    # ---- Model / Optim ----
    model = LakeProfileSeq(driver_dim=Du_total, param_dim=P, depth_out=Dz,
                           hidden=args.hidden, cell_type=args.cell_type,
                           num_layers=args.num_layers, dropout_fc=args.dropout_fc).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()

    # ---- MLflow ----
    exp_name = args.experiment or os.getenv("MLFLOW_EXPERIMENT_NAME", "Lake-Emulator")
    mlflow.set_experiment(exp_name)
    run_name = f"{args.cell_type.upper()}_H{args.hidden}_L{args.L}_lr{args.lr:g}_TEST{args.test_years[0]}-{args.test_years[1]}"
    if n_fourier > 0:
        run_name += f"_fourierK{len(harm)}"

    best_val = float("inf")
    ckpt_path = os.path.join(args.artifacts, "best_model.pt")

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("host", socket.gethostname())
        mlflow.set_tag("split_mode", "strict_years")
        mlflow.set_tag("fourier_enabled", str(n_fourier > 0))
        mlflow.set_tag("test_years", f"{args.test_years[0]}-{args.test_years[1]}")
        mlflow.log_params({
            "cell_type": args.cell_type, "hidden": args.hidden, "num_layers": args.num_layers,
            "L": args.L, "lead": args.lead, "epochs": args.epochs, "batch_size": args.batch_size,
            "lr": args.lr, "Du_base": Du_base, "Du_total": Du_total, "P": P, "Dz": Dz, "device": args.device,
            "smooth_lambda": args.smooth_lambda,
            "train_len": len(train_ds), "val_len": len(val_ds), "test_len": len(test_ds),
            "val_frac_pretest": args.val_frac_pretest,
            "fourier_harmonics": ",".join(map(str, harm)) if harm else "",
        })
        mlflow.log_metric("num_params", count_params(model))

        # ---- Train loop with tqdm ----
        for ep in tqdm(range(1, args.epochs + 1), desc="Epochs", unit="epoch"):
            # Train
            model.train()
            train_sse, train_n = 0.0, 0
            for x, p, y in tqdm(train_loader, desc=f"Epoch {ep}/{args.epochs} [train]", leave=False):
                x, p, y = x.float().to(args.device), p.float().to(args.device), y.float().to(args.device)
                pred = model(x, p)
                loss = mse(pred, y)
                if args.smooth_lambda > 0 and Dz > 1:
                    loss = loss + args.smooth_lambda * (pred[:, 1:] - pred[:, :-1]).abs().mean()
                opt.zero_grad(); loss.backward(); opt.step()
                train_sse += (pred - y).pow(2).sum().item(); train_n += y.numel()

            train_mse = train_sse / max(1, train_n)

            # Validate
            model.eval()
            val_sse, val_n = 0.0, 0
            with torch.no_grad():
                for x, p, y in val_loader:
                    pred = model(x.float().to(args.device), p.float().to(args.device))
                    val_sse += (pred - y.float().to(args.device)).pow(2).sum().item()
                    val_n   += y.numel()
            val_mse = val_sse / max(1, val_n)

            mlflow.log_metrics({"train_mse": train_mse, "val_mse": val_mse}, step=ep)

            if val_mse < best_val:
                best_val = val_mse
                torch.save({
                    "model_state": model.state_dict(),
                    "Du_base": Du_base, "Du_total": Du_total, "P": P, "Dz": Dz,
                    "norms": {
                        "driver_mean": x_mean, "driver_std": x_std,
                        "param_mean": p_mean, "param_std": p_std,
                        "n_fourier": n_fourier
                    },
                    "config": vars(args),
                }, ckpt_path)
                mlflow.log_artifact(ckpt_path, artifact_path="checkpoints")
                if args.log_model_on_improve:
                    mlflow.pytorch.log_model(model, artifact_path="model")

        # ---- Test
        model.eval()
        test_sse, test_n = 0.0, 0
        with torch.no_grad():
            for x, p, y in tqdm(test_loader, desc="Testing", unit="batch"):
                pred = model(x.float().to(args.device), p.float().to(args.device))
                test_sse += (pred - y.float().to(args.device)).pow(2).sum().item()
                test_n   += y.numel()
        test_mse = test_sse / max(1, test_n)
        mlflow.log_metric("test_mse", float(test_mse))
        mlflow.log_metric("best_val_mse", float(best_val))

        print(f"Strict TEST MSE ({args.test_years[0]}–{args.test_years[1]}): {test_mse:.6f}")
        print(f"Best val MSE: {best_val:.6f} | artifacts: {mlflow.get_artifact_uri()}")

if __name__ == "__main__":
    main()
