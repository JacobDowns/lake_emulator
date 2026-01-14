# Lake Emulator (Multi-Lake, Depth-Aware)

This repository contains a **machine-learning emulator** for a 1-D lake temperature model.  
The emulator is trained on physics-based model output and can rapidly reproduce lake temperature profiles across time, depth, and parameter ensembles.

The current version supports:
- **Multiple lakes** with **different depth resolutions**
- **Variable bathymetry** via depth-conditioned decoding
- Evaluation across **all time steps** and **all parameter sets**
- Visualization of ensemble behavior (mean, uncertainty, or “spaghetti” plots)

You can use this repository **without retraining any models** — trained model checkpoints are included via MLflow artifacts.

---

## Repository Structure

```text
.
├── models.py                  # GRU-based depth-conditioned emulator
├── multi_lake_dataset.py      # Multi-lake data loader + normalization
├── train.py                   # Training script (not required for evaluation)
├── eval_multilake_summary.py  # Main evaluation + plotting script (START HERE)
├── data/
│   └── parsed_data/
│       ├── BearLake/
│       │   ├── weather.npy
│       │   ├── outputs.npy
│       │   ├── parameters.npy
│       │   └── bathymetry.npy
│       └── RedPond/
│           └── ...
├── mlruns/                    # MLflow experiments + trained model checkpoints
├── environment.yml            # Conda environment specification
└── README.md                  # This file
```

---

## Quick Start (No Training Required)

### 1. Create the Conda Environment

From the repository root:

```bash
conda env create -f environment.yml
conda activate ml
```

---

### 2. Verify the Data Layout

Each lake directory should look like:

```text
data/parsed_data/BearLake/
├── weather.npy        # (T, Du) weather drivers
├── outputs.npy        # (N, T, Dz) lake temperatures
├── parameters.npy    # (N, P) model parameters
└── bathymetry.npy    # (Dz,) or (Dz,1) area-by-depth
```

---

## Evaluating a Trained Emulator

The main script you will use is:

```bash
eval_multilake_summary.py
```

This script:
- Loads a trained model from MLflow
- Runs the emulator across **all time steps**
- Evaluates **many parameter sets**
- Produces plots for easy interpretation

---

### Example 1: Full Timeseries “Spaghetti” Plot

```bash
python eval_multilake_summary.py   --run_id <RUN_ID>   --root_dir data/parsed_data   --lake BearLake   --all_lakes BearLake RedPond   --split all   --plot_mode spaghetti   --spaghetti_sims 100   --show_mean
```

Plots are saved to:

```text
plots_eval_summary/
```

---

### Example 2: Faster Demo

```bash
python eval_multilake_summary.py   --run_id <RUN_ID>   --lake BearLake   --split all   --spaghetti_sims 40   --stride 5   --batch_windows 64
```

---

### Example 3: Choose Specific Depths

```bash
--depth_indices 0,4,7
```

---

## Exporting Data for One Simulation

```bash
python eval_multilake_summary.py   --run_id <RUN_ID>   --lake BearLake   --split all   --save_sim_inputs 12
```

This writes:

```text
plots_eval_summary/inputs_BearLake_split_all_sim_12.npz
```

Load in Python:

```python
import numpy as np
d = np.load("inputs_BearLake_split_all_sim_12.npz")
```

---

## Notes on the Emulator Architecture

- A **GRU** models time evolution
- Bathymetry enters via **depth features** in the decoder
- Temperature is predicted **independently at each depth**
- One model supports **multiple lakes** with different depths

---

## Optional: Retraining

```bash
python train.py
```

---

## Troubleshooting

- Ensure `mlruns/` directory is present
- Use the provided Conda environment
- Reduce `--spaghetti_sims` or increase `--stride` if evaluation is slow

---

**Enjoy exploring the lake emulator!**
