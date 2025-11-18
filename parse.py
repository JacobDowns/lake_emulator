import pandas as pd
import numpy as np
import os, json

# ----------------------
# Paths
# ----------------------
WEATHER_CSV = 'data/parsed_data/weather_data.csv'  # must contain YEAR and DOY (or DAY=DOY)
PARAMS_CSV  = 'data/BearLake_inputs_outputs/inputs/parameter-values-tested.csv'
OUTPUTS_DIR = 'data/BearLake_inputs_outputs/outputs'
OUT_DIR     = 'data/parsed_data'

os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------
# Load weather and normalize date cols
# ----------------------
weather = pd.read_csv(WEATHER_CSV)

# If DOY exists, use it; else assume DAY already means DOY
doy_col = 'DOY' if 'DOY' in weather.columns else 'DAY'
weather['YEAR'] = weather['YEAR'].astype(int)
weather[doy_col] = weather[doy_col].astype(int)

# Weather key for overlap
weather_key = weather[['YEAR', doy_col]].rename(columns={doy_col: 'DOY'})
weather_key = weather_key.drop_duplicates().sort_values(['YEAR','DOY']).reset_index(drop=True)

# ----------------------
# Parameters
# ----------------------
parameter_df = pd.read_csv(PARAMS_CSV)
param_mat = parameter_df.iloc[:, 1:].to_numpy(dtype=np.float32)  # drop first (trial id)
N = param_mat.shape[0]
np.save(os.path.join(OUT_DIR, 'parameter_data.npy'), param_mat)
print("parameter_data.npy:", param_mat.shape)

# ----------------------
# Determine overlap (YEAR, DOY) across all outputs
# ----------------------
overlap = set(map(tuple, weather_key[['YEAR','DOY']].to_numpy()))
for i in range(N):
    out_path = os.path.join(OUTPUTS_DIR, f'profile-laketemp-{i}.txt')
    df_out = pd.read_csv(out_path, sep=r'\s+')
    df_out['YEAR'] = df_out['YEAR'].astype(int)
    df_out['DOY']  = df_out['DAY'].astype(int)  # output's DAY is DOY
    k = df_out[['YEAR','DOY']].drop_duplicates()
    overlap &= set(map(tuple, k.to_numpy()))

overlap_df = pd.DataFrame(list(overlap), columns=['YEAR','DOY']).astype(int)
overlap_df = overlap_df.sort_values(['YEAR','DOY']).reset_index(drop=True)
T = len(overlap_df)
print("Overlap timesteps T =", T)

# ----------------------
# Save WEATHER matrix: [YEAR, DOY, drivers...]
# ----------------------
# Explicit driver column list = everything except date cols
date_cols = {'YEAR','MON','DAY','DOY'}
driver_cols = [c for c in weather.columns if c not in date_cols]

# Align weather rows to overlap
w_aligned = weather.merge(overlap_df, left_on=['YEAR', doy_col], right_on=['YEAR','DOY'], how='inner')
w_aligned = w_aligned.sort_values(['YEAR','DOY']).reset_index(drop=True)

# Compose matrix with YEAR/DOY first, then drivers
weather_mat = np.column_stack([
    w_aligned['YEAR'].to_numpy(np.int32),
    w_aligned['DOY'].to_numpy(np.int32),
    w_aligned[driver_cols].to_numpy(np.float32)
])
np.save(os.path.join(OUT_DIR, 'weather_data.npy'), weather_mat)
print("weather_data.npy:", weather_mat.shape)

# Save aligned dates for convenience
overlap_df.to_csv(os.path.join(OUT_DIR, 'aligned_dates.tsv'), sep='\t', index=False)

# Save metadata about weather columns
meta = {
    "columns": ["YEAR", "DOY"] + driver_cols,
    "n_meta": 2,
    "n_drivers": len(driver_cols)
}
with open(os.path.join(OUT_DIR, 'weather_columns.json'), 'w') as f:
    json.dump(meta, f, indent=2)

# ----------------------
# Save OUTPUT tensors (temps only), aligned to overlap
# ----------------------
output_arrays = []
depth_cols_cache = None

for i in range(N):
    out_path = os.path.join(OUTPUTS_DIR, f'profile-laketemp-{i}.txt')
    df_out = pd.read_csv(out_path, sep=r'\s+')
    df_out['YEAR'] = df_out['YEAR'].astype(int)
    df_out['DOY']  = df_out['DAY'].astype(int)

    df_out = df_out.merge(overlap_df, on=['YEAR','DOY'], how='inner') \
                   .sort_values(['YEAR','DOY']).reset_index(drop=True)

    exclude = {'YEAR','MON','DAY','DOY','DATE'}
    temp_cols = [c for c in df_out.columns if c not in exclude]
    if depth_cols_cache is None:
        depth_cols_cache = temp_cols  # remember order once

    temps = df_out[temp_cols].to_numpy(np.float32)  # (T, Dz)
    output_arrays.append(temps)

output_data = np.stack(output_arrays, axis=0)  # (N, T, Dz)
np.save(os.path.join(OUT_DIR, 'output_data.npy'), output_data)
print("output_data.npy:", output_data.shape)
