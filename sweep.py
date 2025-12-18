#!/usr/bin/env python3
"""
sweep.py

Sweep only:
- smoothed weather input windows (including "none")
- rnn_dropout
- head_dropout
- smooth_lambda

Everything else fixed to a baseline config.
Uses MLflow run_key to skip completed configs.
"""

import itertools
import os
import subprocess
from typing import List, Optional

from mlflow.tracking import MlflowClient


# -----------------------------
# Config
# -----------------------------
EXPERIMENT_NAME = "Smooth Weather Sweep"
TRAIN_SCRIPT = "train.py"

# Fixed baseline config (edit these to your preferred defaults)
WX = 365
WY = 30

HIDDEN = 16
NUM_LAYERS = 1
HEAD_HIDDEN = "64, 64"
LR = 7e-4

EPOCHS = 12
BATCH_SIZE = 512

DEVICE = "cuda"  # set to "cuda" / "cpu" to force, or leave None


# -----------------------------
# Sweep grids
# -----------------------------

# Smooth window configurations:
# - None means: do not pass --smooth_windows (no smoothing features)
# - otherwise a list like [7, 30] -> pass --smooth_windows "7,30"
SMOOTH_WINDOWS_GRID: List[Optional[List[int]]] = [
    None,
    [7],
    [30],
    [7, 30],
]

SMOOTH_PAD_MODE = "reflect"  # passed to train.py

RNN_DROPOUT_LIST = [0.0]
HEAD_DROPOUT_LIST = [0.0]

# NEW: smoothness penalty strength
SMOOTH_LAMBDA_LIST = [0.0]   # tweak to taste


# -----------------------------
# MLflow helpers
# -----------------------------
def get_or_create_experiment(experiment_name: str) -> str:
    client = MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        exp_id = client.create_experiment(experiment_name)
    else:
        exp_id = exp.experiment_id
    return exp_id


def make_run_key(
    Wx: int,
    Wy: int,
    smooth_windows: Optional[List[int]],
    smooth_pad_mode: str,
    rnn_dropout: float,
    head_dropout: float,
    smooth_lambda: float,
) -> str:
    """
    A stable run_key for deduplication/resume. Keep this consistent.
    """
    sw = "none" if smooth_windows is None else ",".join(map(str, smooth_windows))
    return (
        f"gru_Wx{Wx}_Wy{Wy}_"
        f"sw{sw}_pad{smooth_pad_mode}_"
        f"rnnDrop{rnn_dropout}_headDrop{head_dropout}_"
        f"smoothLam{smooth_lambda}"
    )


def is_config_finished(exp_id: str, run_key: str) -> bool:
    client = MlflowClient()
    filter_str = f"tags.run_key = '{run_key}'"
    runs = client.search_runs(
        experiment_ids=[exp_id],
        filter_string=filter_str,
        max_results=50,
        order_by=["attributes.start_time DESC"],
    )
    for r in runs:
        if r.info.status == "FINISHED":
            return True
    return False


# -----------------------------
# Main sweep
# -----------------------------
def main():
    exp_id = get_or_create_experiment(EXPERIMENT_NAME)
    print(f"Using MLflow experiment '{EXPERIMENT_NAME}' (id={exp_id})")

    os.makedirs("sweep_artifacts", exist_ok=True)

    combo_iter = itertools.product(
        SMOOTH_WINDOWS_GRID,
        RNN_DROPOUT_LIST,
        HEAD_DROPOUT_LIST,
        SMOOTH_LAMBDA_LIST,
    )

    for smooth_windows, rnn_dropout, head_dropout, smooth_lambda in combo_iter:
        run_key = make_run_key(
            Wx=WX,
            Wy=WY,
            smooth_windows=smooth_windows,
            smooth_pad_mode=SMOOTH_PAD_MODE,
            rnn_dropout=rnn_dropout,
            head_dropout=head_dropout,
            smooth_lambda=smooth_lambda,
        )

        if is_config_finished(exp_id, run_key):
            print(f"[SKIP] run_key={run_key} already FINISHED.")
            continue

        # artifacts dir
        sw_tag = "none" if smooth_windows is None else "-".join(map(str, smooth_windows))
        artifacts_dir = os.path.join(
            "sweep_artifacts",
            f"gru_Wx{WX}_Wy{WY}_sw{sw_tag}_pad{SMOOTH_PAD_MODE}_"
            f"rnnDrop{rnn_dropout}_headDrop{head_dropout}_smoothLam{smooth_lambda}"
        )
        os.makedirs(artifacts_dir, exist_ok=True)

        # command
        cmd = [
            "python", TRAIN_SCRIPT,
            f"--Wx={WX}",
            f"--Wy={WY}",
            f"--hidden={HIDDEN}",
            f"--num_layers={NUM_LAYERS}",
            f"--head_hidden={HEAD_HIDDEN}",
            f"--lr={LR}",
            f"--epochs={EPOCHS}",
            f"--batch_size={BATCH_SIZE}",
            f"--smooth_lambda={smooth_lambda}",
            f"--rnn_dropout={rnn_dropout}",
            f"--head_dropout={head_dropout}",
            f"--artifacts={artifacts_dir}",
            f"--experiment={EXPERIMENT_NAME}",
            f"--smooth_pad_mode={SMOOTH_PAD_MODE}",
        ]

        # only pass smooth_windows if enabled
        if smooth_windows is not None and len(smooth_windows) > 0:
            cmd.append(f"--smooth_windows={','.join(map(str, smooth_windows))}")

        # optional device override
        if DEVICE is not None:
            cmd.append(f"--device={DEVICE}")

        # pass run_key via environment; train.py should do:
        #   rk = os.getenv("RUN_KEY","");  if rk: mlflow.set_tag("run_key", rk)
        env = os.environ.copy()
        env["RUN_KEY"] = run_key

        print("\n========================================")
        print("Running config:")
        print(f"  smooth_windows={smooth_windows}, pad_mode={SMOOTH_PAD_MODE}")
        print(f"  rnn_dropout={rnn_dropout}, head_dropout={head_dropout}")
        print(f"  smooth_lambda={smooth_lambda}")
        print("run_key:", run_key)
        print("artifacts:", artifacts_dir)
        print("Command:", " ".join(cmd))
        print("========================================\n")

        try:
            subprocess.run(cmd, check=True, env=env)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Training failed for run_key={run_key} (return code={e.returncode})")

    print("Sweep complete.")


if __name__ == "__main__":
    main()
