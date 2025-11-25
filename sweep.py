#!/usr/bin/env python3
"""
Resumable sweep launcher for train.py.

- Defines a grid of hyperparameters, including Wx.
- For each combo, constructs a run_key that matches train.py.
- Queries MLflow for runs with that run_key.
- Skips combos that already have a FINISHED run.
- Reruns combos that are missing or only FAILED/KILLED/RUNNING.
- Uses simple lock files to avoid duplicate runs if multiple sweep processes run in parallel.
"""

import itertools
import os
import shlex
import subprocess
from typing import Dict, Any, List

from mlflow.tracking import MlflowClient

# -----------------------------
# Config
# -----------------------------

# MLflow experiment to use
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "Long-Predictions")

# Optional: tracking URI (or set this in your shell)
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")

# Which Python & training script to call
PYTHON = os.getenv("PYTHON", "python")
TRAIN_SCRIPT = os.getenv("TRAIN_SCRIPT", "train.py")

# Where to create simple lock files
LOCK_DIR = os.getenv("SWEEP_LOCK_DIR", ".locks")

# Common non-sweep args (things you *don't* vary here)
COMMON_ARGS: List[str] = [
    "--Wy", "90",
    "--epochs", "5",
    "--batch_size", "256",
    "--smooth_lambda", "0.0",
    "--split_years", "2019", "2021", "2025",
    "--normalize", "1",
    "--device", "cuda",  # or "cpu"
    "--experiment", EXPERIMENT_NAME,
]

# Hyperparameter grid (edit this as you like)
GRID: Dict[str, List[Any]] = {
    "model": ["rnn"],   
    "cell":  ["gru"],      # used when model == "rnn"
    "Wx":    [180, 360],
    "hidden": [256],
    "lr":     [7e-4],
    "head_hidden": ["256,256"],    # strings passed directly to --head_hidden
}

# -----------------------------
# Helpers
# -----------------------------

def make_run_key(model: str, cell: str, Wx: int, Wy: int,
                 hidden: int, lr: float, head_hidden: str) -> str:
    """
    Must match the logic used in train.py where we set mlflow.set_tag('run_key', ...):

    run_key = f"{args.model}_cell{args.cell}_Wx{args.Wx}_Wy{args.Wy}_H{args.hidden}_LR{args.lr}_head{args.head_hidden}"
    """
    return f"{model}_cell{cell}_Wx{Wx}_Wy{Wy}_H{hidden}_LR{lr}_head{head_hidden}"

def ensure_experiment(client: MlflowClient, name: str) -> str:
    exp = client.get_experiment_by_name(name)
    if exp is None:
        return client.create_experiment(name)
    return exp.experiment_id

def search_runs_by_key(client: MlflowClient, exp_id: str, key: str):
    return client.search_runs(
        experiment_ids=[exp_id],
        filter_string=f"tags.run_key = '{key}'",
        max_results=50,
        order_by=["attributes.start_time DESC"],
    )

def is_finished(client: MlflowClient, exp_id: str, key: str) -> bool:
    runs = search_runs_by_key(client, exp_id, key)
    for r in runs:
        if r.info.status == "FINISHED":
            return True
    return False

def has_inflight_or_failed(client: MlflowClient, exp_id: str, key: str) -> bool:
    runs = search_runs_by_key(client, exp_id, key)
    for r in runs:
        if r.info.status in ("RUNNING", "SCHEDULED", "FAILED", "KILLED"):
            return True
    return False

def acquire_lock(path: str) -> bool:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        return True
    except FileExistsError:
        return False

def release_lock(path: str):
    try:
        os.remove(path)
    except FileNotFoundError:
        pass

def arg_value(flag: str, args_list: List[str], default=None):
    if flag in args_list:
        idx = args_list.index(flag)
        if idx < len(args_list) - 1:
            return args_list[idx + 1]
    return default

# -----------------------------
# Main sweep
# -----------------------------

def main():
    os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI
    client = MlflowClient()
    exp_id = ensure_experiment(client, EXPERIMENT_NAME)

    # Wy is fixed via COMMON_ARGS
    Wy_default = int(arg_value("--Wy", COMMON_ARGS, default=1))

    # Build product of the grid
    combos = list(itertools.product(
        GRID["model"],
        GRID["cell"],
        GRID["Wx"],
        GRID["hidden"],
        GRID["lr"],
        GRID["head_hidden"],
    ))

    print(f"Total combinations: {len(combos)}")

    for (model, cell, Wx, hidden, lr, head_hidden) in combos:
        Wy = Wy_default

        key = make_run_key(model, cell, Wx, Wy, hidden, lr, head_hidden)

        if is_finished(client, exp_id, key):
            print(f"[SKIP] FINISHED: {key}")
            continue

        if has_inflight_or_failed(client, exp_id, key):
            print(f"[RETRY] Previously inflight/failed: {key}")

        lock_path = os.path.join(LOCK_DIR, f"{key}.lock")
        if not acquire_lock(lock_path):
            print(f"[LOCK] Held by another process, skipping: {key}")
            continue

        try:
            # Compose command
            cmd = [
                PYTHON, TRAIN_SCRIPT,
                "--model", model,
                "--cell", cell,
                "--Wx", str(Wx),
                "--hidden", str(hidden),
                "--lr", f"{lr:g}",
                "--head_hidden", head_hidden,
            ] + COMMON_ARGS

            print(f"[RUN ] {key}")
            print(f"  CMD: {' '.join(shlex.quote(c) for c in cmd)}")

            env = os.environ.copy()
            env["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI
            env["MLFLOW_EXPERIMENT_NAME"] = EXPERIMENT_NAME

            subprocess.run(cmd, check=True, env=env)
        finally:
            release_lock(lock_path)

    print("\nSweep complete (skipped FINISHED, reran unfinished/failed).")


if __name__ == "__main__":
    main()
