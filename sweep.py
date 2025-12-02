#!/usr/bin/env python3
"""
sweep.py

Run a hyperparameter sweep over RNN-based lake emulator models using train.py.

Features:
- Sweeps over Wx, Wy, GRU/LSTM hidden size, depth, head MLP sizes, LR,
  seasonal features, and attention (on/off, attn_dim).
- Uses MLflow tags (run_key) to detect whether a configuration has already
  finished; if so, that combo is skipped (useful for resuming after failure).
"""

import itertools
import os
import subprocess
from typing import List

import mlflow
from mlflow.tracking import MlflowClient


# -----------------------------
# Configurable sweep grids
# -----------------------------

EXPERIMENT_NAME = "Emulator Sweep"  # change if you like

TRAIN_SCRIPT = "train.py"

# Window sizes
WX_LIST = [360]
WY_LIST = [30]  

# RNN hyperparameters
HIDDEN_LIST = [16, 32, 64, 128, 256]     # hidden size
NUM_LAYERS_LIST = [1, 2, 3]         # RNN depth

# MLP head depths: strings must match train.py --head_hidden
HEAD_HIDDEN_LIST: List[str] = [
    "16",
    "32"
    "64",
    "128",
    "256",
    "16, 16",
    "32, 32",
    "64, 64",
    "128, 128",
    "256, 256"
]

# Learning rates
LR_LIST = [2.5e-3]

# Seasonal feature flag: 0 = off, 1 = on
USE_SEASONAL_LIST = [0, 1]

# Attention flags / dims
USE_ATTENTION_LIST = [0]           # 0 = off, 1 = on
ATTN_DIM_LIST = [32]              # only relevant if use_attention=1

# Base training args shared across sweep
BASE_EPOCHS = 5
BASE_BATCH_SIZE = 512
BASE_SMOOTH_LAMBDA = 0.0
BASE_RNN_DROPOUT = 0.0
BASE_HEAD_DROPOUT = 0.0

# Fixed cell type for now (can make a list if you want to compare GRU vs LSTM)
CELL = "gru"


# -----------------------------
# MLflow helpers
# -----------------------------

def get_or_create_experiment(experiment_name: str) -> str:
    """
    Return the experiment_id for the given experiment name,
    creating it if it does not exist.
    """
    client = MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        exp_id = client.create_experiment(experiment_name)
    else:
        exp_id = exp.experiment_id
    return exp_id


def config_run_key(
    cell: str,
    Wx: int,
    Wy: int,
    hidden: int,
    num_layers: int,
    lr: float,
    head_hidden: str,
    use_seasonal: int,
    use_attention: int,
    attn_dim: int,
) -> str:
    """
    Must match the run_key format used in train.py.
    """
    return (
        f"rnn_cell{cell}_Wx{Wx}_Wy{Wy}_"
        f"H{hidden}_L{num_layers}_LR{lr}_head{head_hidden}_"
        f"seasonal{use_seasonal}_"
        f"attn{use_attention}_attnDim{attn_dim}"
    )


def is_config_finished(exp_id: str, run_key: str) -> bool:
    """
    Check MLflow for any FINISHED run with the given run_key tag in experiment exp_id.
    If found, we treat that configuration as completed.
    """
    client = MlflowClient()
    filter_str = f"tags.run_key = '{run_key}'"
    runs = client.search_runs(
        experiment_ids=[exp_id],
        filter_string=filter_str,
        max_results=50,
        order_by=["attributes.start_time DESC"],
    )
    for r in runs:
        status = r.info.status
        if status == "FINISHED":
            return True
    return False


# -----------------------------
# Main sweep
# -----------------------------

def main():
    # Ensure MLflow experiment exists
    exp_id = get_or_create_experiment(EXPERIMENT_NAME)
    print(f"Using MLflow experiment '{EXPERIMENT_NAME}' (id={exp_id})")

    # Parent directory for artifacts for the sweep
    os.makedirs("sweep_artifacts", exist_ok=True)

    # Cartesian product over hyperparameters
    combo_iter = itertools.product(
        WX_LIST,
        WY_LIST,
        HIDDEN_LIST,
        NUM_LAYERS_LIST,
        HEAD_HIDDEN_LIST,
        LR_LIST,
        USE_SEASONAL_LIST,
        USE_ATTENTION_LIST,
        ATTN_DIM_LIST,
    )

    for (
        Wx,
        Wy,
        hidden,
        num_layers,
        head_hidden,
        lr,
        use_seasonal,
        use_attention,
        attn_dim,
    ) in combo_iter:

        # Optionally skip attention dims when attention is off
        if use_attention == 0 and attn_dim != ATTN_DIM_LIST[0]:
            # To avoid redundant configs when attention is off,
            # only keep the first attn_dim value in that case.
            continue

        run_key = config_run_key(
            cell=CELL,
            Wx=Wx,
            Wy=Wy,
            hidden=hidden,
            num_layers=num_layers,
            lr=lr,
            head_hidden=head_hidden,
            use_seasonal=use_seasonal,
            use_attention=use_attention,
            attn_dim=attn_dim,
        )

        # Check if this configuration already has a FINISHED run
        if is_config_finished(exp_id, run_key):
            print(f"[SKIP] run_key={run_key} already has a FINISHED run.")
            continue

        # Build a unique artifacts directory for this config
        head_tag = head_hidden.replace(",", "-")
        season_tag = f"seasonal{use_seasonal}"
        attn_tag = f"attn{use_attention}_attnDim{attn_dim}"
        artifacts_dir = os.path.join(
            "sweep_artifacts",
            f"rnn_cell{CELL}_Wx{Wx}_Wy{Wy}_H{hidden}_L{num_layers}_"
            f"head{head_tag}_lr{lr}_{season_tag}_{attn_tag}"
        )
        os.makedirs(artifacts_dir, exist_ok=True)

        # Construct command line for train.py
        cmd = [
            "python", TRAIN_SCRIPT,
            f"--Wx={Wx}",
            f"--Wy={Wy}",
            f"--hidden={hidden}",
            f"--num_layers={num_layers}",
            f"--head_hidden={head_hidden}",
            f"--lr={lr}",
            f"--epochs={BASE_EPOCHS}",
            f"--batch_size={BASE_BATCH_SIZE}",
            f"--smooth_lambda={BASE_SMOOTH_LAMBDA}",
            f"--rnn_dropout={BASE_RNN_DROPOUT}",
            f"--head_dropout={BASE_HEAD_DROPOUT}",
            f"--cell={CELL}",
            f"--artifacts={artifacts_dir}",
            f"--experiment={EXPERIMENT_NAME}",
            f"--attn_dim={attn_dim}",
        ]

        if use_seasonal:
            cmd.append("--use_seasonal_features")
        if use_attention:
            cmd.append("--use_attention")

        print("\n========================================")
        print("Running config:")
        print(f"  Wx={Wx}, Wy={Wy}, hidden={hidden}, num_layers={num_layers},")
        print(f"  head_hidden={head_hidden}, lr={lr}, "
              f"use_seasonal={use_seasonal}, use_attention={use_attention}, attn_dim={attn_dim}")
        print("run_key:", run_key)
        print("artifacts:", artifacts_dir)
        print("Command:", " ".join(cmd))
        print("========================================\n")

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Training failed for run_key={run_key} with return code {e.returncode}")


if __name__ == "__main__":
    main()
