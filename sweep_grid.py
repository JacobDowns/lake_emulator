#!/usr/bin/env python3
# sweep_grid.py
import os, argparse, itertools, subprocess, sys
import numpy as np
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from lake_models import LakeProfileSeq
import torch.nn as nn

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

def split_meta_and_drivers(weather_with_meta: np.ndarray):
    meta = weather_with_meta[:, :2].astype(np.int32)
    drivers = weather_with_meta[:, 2:].astype(np.float32)
    return meta, drivers

def build_fourier_from_year_doy(year: np.ndarray, doy: np.ndarray, harmonics: Tuple[int,...]) -> np.ndarray:
    import numpy as np
    year_len = np.where(((year % 4 == 0) & (year % 100 != 0)) | (year % 400 == 0), 366, 365).astype(float)
    cols = []
    for k in harmonics:
        theta = 2 * np.pi * k * (doy.astype(float) / year_len)
        cols += [np.sin(theta), np.cos(theta)]
    return np.column_stack(cols).astype(np.float32) if cols else np.zeros((len(year), 0), dtype=np.float32)

def parse_budgets(s: str) -> List[int]:
    out = []
    for tok in s.split(","):
        tok = tok.strip().lower()
        mult = 1
        if tok.endswith("k"): mult, tok = 1000, tok[:-1]
        elif tok.endswith("m"): mult, tok = 1_000_000, tok[:-1]
        out.append(int(float(tok) * mult))
    return out

def find_hidden_for_budget(cell_type: str, budget_params: int, Du: int, P: int, Dz: int,
                           num_layers: int, H_min=16, H_max=1024) -> int:
    """
    Choose hidden size H so that total params (RNN with num_layers + FC head) ~= budget_params.
    We search H and measure by instantiating the actual model (robust across cell types).
    """
    best_H, best_diff = None, 1e18
    for step in [32, 16, 8, 4, 2, 1]:
        start = H_min if best_H is None else max(H_min, best_H - 15)
        end   = H_max if best_H is None else min(H_max, best_H + 15)
        for H in range(start, end + 1, step):
            m = LakeProfileSeq(driver_dim=Du, param_dim=P, hidden=H, depth_out=Dz,
                               cell_type=cell_type, num_layers=num_layers)
            diff = abs(count_params(m) - budget_params)
            if diff < best_diff:
                best_diff, best_H = diff, H
    return int(best_H)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_script", default="train_lake.py")
    ap.add_argument("--weather_npy", default="data/parsed_data/weather_data.npy")
    ap.add_argument("--params_npy",  default="data/parsed_data/parameter_data.npy")
    ap.add_argument("--outputs_npy", default="data/parsed_data/output_data.npy")

    # Seasonal Fourier
    ap.add_argument("--seasonal_harmonics", default="1,2", help="'' to disable, else comma list e.g. 1,2,3")

    # Sweep knobs
    ap.add_argument("--cells", default="gru,lstm,rnn")
    ap.add_argument("--budgets", default="50k,100k,200k")
    ap.add_argument("--num_layers_list", default="1,2", help="Comma list of RNN stack depths to sweep")
    ap.add_argument("--Ls", default="90")
    ap.add_argument("--lrs", default="0.001,0.0005")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lead", type=int, default=1)
    ap.add_argument("--smooth_lambda", type=float, default=0.0)
    ap.add_argument("--dropout_fc", type=float, default=0.0)

    ap.add_argument("--test_years", nargs=2, type=int, default=[2020, 2025])
    ap.add_argument("--val_frac_pretest", type=float, default=0.2)
    ap.add_argument("--artifacts_base", default="artifacts_compare")
    ap.add_argument("--experiment", default=None)

    # Execution
    ap.add_argument("--device", default=None, help="Override device for train script, e.g. cuda:0")
    ap.add_argument("--max_procs", type=int, default=1, help="Parallel processes (mind GPU RAM!)")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    # Load shapes for fair budget calc
    weather_with_meta = np.load(args.weather_npy)   # (T, 2+Du)
    params  = np.load(args.params_npy)              # (N, P)
    outputs = np.load(args.outputs_npy)             # (N, T, Dz)
    meta, drivers = split_meta_and_drivers(weather_with_meta)
    year, doy = meta[:,0], meta[:,1]
    T, Du_base = drivers.shape
    N, T2, Dz = outputs.shape
    P = params.shape[1]
    assert T == T2

    # Fourier dims for budget calculation
    harm = tuple(int(h) for h in args.seasonal_harmonics.split(",") if h.strip()) if args.seasonal_harmonics.strip() != "" else tuple()
    if len(harm) > 0:
        fourier = build_fourier_from_year_doy(year, doy, harmonics=harm)  # (T, 2K)
        Du_total = Du_base + fourier.shape[1]
    else:
        Du_total = Du_base

    cells   = [c.strip().lower() for c in args.cells.split(",") if c.strip()]
    budgets = parse_budgets(args.budgets)
    Ls      = [int(x) for x in args.Ls.split(",")]
    lrs     = [float(x) for x in args.lrs.split(",")]
    layer_list = [int(x) for x in args.num_layers_list.split(",")]

    # Precompute hidden sizes for fairness per (cell, layers, budget)
    hidden_for = {}
    print("Selecting hidden sizes for budgets (counts include RNN stack with num_layers + FC head):")
    for cell in cells:
        for nl in layer_list:
            for b in budgets:
                H = find_hidden_for_budget(cell, b, Du=Du_total, P=P, Dz=Dz, num_layers=nl)
                cnt = count_params(LakeProfileSeq(driver_dim=Du_total, param_dim=P, hidden=H, depth_out=Dz,
                                                  cell_type=cell, num_layers=nl))
                hidden_for[(cell, nl, b)] = H
                print(f"  {cell.upper()} LAYERS={nl} ~{b:,} params -> H={H} (actual {cnt:,}; Du_total={Du_total})")

    # Build command list
    cmds = []
    for L, lr, b in itertools.product(Ls, lrs, budgets):
        for cell in cells:
            for nl in layer_list:
                H = hidden_for[(cell, nl, b)]
                artifacts = os.path.join(
                    args.artifacts_base,
                    f"{cell}_NL{nl}_B{b}_H{H}_L{L}_lr{lr:g}"
                )
                os.makedirs(artifacts, exist_ok=True)
                cmd = [
                    sys.executable, args.train_script,
                    "--weather_npy", args.weather_npy,
                    "--params_npy", args.params_npy,
                    "--outputs_npy", args.outputs_npy,
                    "--seasonal_harmonics", args.seasonal_harmonics,
                    "--cell_type", cell,
                    "--hidden", str(H),
                    "--num_layers", str(nl),
                    "--dropout_fc", str(args.dropout_fc),
                    "--smooth_lambda", str(args.smooth_lambda),
                    "--L", str(L),
                    "--lead", str(args.lead),
                    "--epochs", str(args.epochs),
                    "--batch_size", str(args.batch_size),
                    "--lr", str(lr),
                    "--test_years", str(args.test_years[0]), str(args.test_years[1]),
                    "--val_frac_pretest", str(args.val_frac_pretest),
                    "--artifacts", artifacts,
                ]
                if args.experiment:
                    cmd += ["--experiment", args.experiment]
                if args.device:
                    cmd += ["--device", args.device]
                cmds.append(cmd)

    print(f"\nPlanned runs: {len(cmds)}")
    for c in cmds:
        print(" ", " ".join(c))
    if args.dry_run:
        return

    max_workers = max(1, int(args.max_procs))
    if max_workers == 1:
        for c in cmds:
            subprocess.run(c, check=True)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(subprocess.run, c, check=True) for c in cmds]
            for _ in as_completed(futs):
                pass

if __name__ == "__main__":
    main()
