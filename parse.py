import pandas as pd
import numpy as np
import os

# ----------------------
# Paths
# ----------------------
WEATHER_CSV = 'data/parsed_data/weather_data.csv'  # your saved CSV
PARAMS_CSV  = 'data/BearLake_inputs_outputs/inputs/parameter-values-tested.csv'
OUTPUTS_DIR = 'data/BearLake_inputs_outputs/outputs'
OUT_DIR     = 'data/parsed_data'

os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------
# Load weather (expects YEAR, MON, and DAY as DOY, or DOY column)
# ----------------------
weather = pd.read_csv(WEATHER_CSV)

# If a DOY column exists, use it; else assume DAY already means DOY
if 'DOY' in weather.columns:
    weather['DOY'] = weather['DOY'].astype(int)
    doy_col = 'DOY'
else:
    weather['DAY'] = weather['DAY'].astype(int)
    doy_col = 'DAY'

weather['YEAR'] = weather['YEAR'].astype(int)

# Keep a clean key for join
weather_key = weather[['YEAR', doy_col]].copy().rename(columns={doy_col: 'DOY'})
weather_key = weather_key.drop_duplicates().sort_values(['YEAR','DOY']).reset_index(drop=True)

# ----------------------
# Determine overlap of (YEAR, DOY) across ALL outputs and weather
# ----------------------
# Start with weather's set
overlap = set(map(tuple, weather_key[['YEAR','DOY']].to_numpy()))

# If you only want intersection with outputs that correspond to the parameter rows below,
# we’ll compute N from the params and loop over profile-laketemp-{i}.txt files.
parameter_data = pd.read_csv(PARAMS_CSV)
param_mat = parameter_data.iloc[:, 1:].to_numpy(dtype=np.float32)  # drop first column (e.g., trial id)
N = parameter_data.shape[0]
print("Parameter matrix shape:", param_mat.shape)
np.save(os.path.join(OUT_DIR, 'parameter_data.npy'), param_mat)

# Intersect with each output's (YEAR, DOY)
output_keys = []
for i in range(N):
    out_path = os.path.join(OUTPUTS_DIR, f'profile-laketemp-{i}.txt')
    df_out = pd.read_csv(out_path, sep=r'\s+')
    # Output DAY already means DOY
    df_out['YEAR'] = df_out['YEAR'].astype(int)
    df_out['DOY']  = df_out['DAY'].astype(int)
    k = df_out[['YEAR','DOY']].drop_duplicates()
    output_keys.append(set(map(tuple, k.to_numpy())))
    # Progressive intersection
    overlap = overlap.intersection(output_keys[-1])

# Convert overlap back to a sorted DataFrame for clean merging
overlap_df = pd.DataFrame(list(overlap), columns=['YEAR','DOY']).astype({'YEAR': int, 'DOY': int})
overlap_df = overlap_df.sort_values(['YEAR','DOY']).reset_index(drop=True)
print(f"Overlapping dates count: {len(overlap_df)}")

# ----------------------
# Filter & save WEATHER matrix (drivers only)
# ----------------------
# Drivers = everything AFTER the first three original columns (YEAR, MON, DAY/DOY),
# but we’ll be explicit to avoid surprises:
driver_cols = [c for c in weather.columns if c not in ['YEAR','MON','DAY','DOY']]
weather_aligned = weather.merge(overlap_df, left_on=['YEAR', doy_col], right_on=['YEAR','DOY'], how='inner')
weather_aligned = weather_aligned.sort_values(['YEAR','DOY']).reset_index(drop=True)

weather_mat = weather_aligned[driver_cols].to_numpy(dtype=np.float32)
print("weather_data.npy shape:", weather_mat.shape)
np.save(os.path.join(OUT_DIR, 'weather_data.npy'), weather_mat)

# (Optional) save the aligned date index to verify later
weather_aligned[['YEAR','DOY']].to_csv(os.path.join(OUT_DIR, 'aligned_dates.tsv'), sep='\t', index=False)

# ----------------------
# Filter & stack OUTPUT matrices
# ----------------------
output_arrays = []
for i in range(N):
    out_path = os.path.join(OUTPUTS_DIR, f'profile-laketemp-{i}.txt')
    df_out = pd.read_csv(out_path, sep=r'\s+')
    df_out['DOY']  = df_out['DAY'].astype(int)
    df_out['YEAR'] = df_out['YEAR'].astype(int)

    # Keep only overlapping dates, same order as overlap_df
    df_out = df_out.merge(overlap_df, on=['YEAR','DOY'], how='inner').sort_values(['YEAR','DOY']).reset_index(drop=True)

    exclude_cols = {'YEAR', 'MON', 'DAY', 'DOY', 'DATE'}
    temp_cols = [c for c in df_out.columns if c not in exclude_cols]
    temp_only = df_out[temp_cols].copy()
    output_arrays.append(temp_only.to_numpy(dtype=np.float32))

# Shape: (N_trials, T_overlap, n_depths)
output_data = np.stack(output_arrays, axis=0)
print("output_data.npy shape:", output_data.shape)
np.save(os.path.join(OUT_DIR, 'output_data.npy'), output_data)
