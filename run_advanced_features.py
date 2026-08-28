import sys
import os
import pyarrow.parquet as pq
import pyarrow.compute as pc
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import warnings
warnings.filterwarnings('ignore')

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

print("=" * 80)
print("PHASE 2, STEP 4b: ADVANCED FEATURE ENGINEERING (TEMPORAL + SERVICE)")
print("=" * 80)

# 1. RELOAD DATASET WITH STRATIFIED TEMPORAL PRESERVATION
print("\n1. Loading unified dataset and sampling with temporal preservation...")
parquet_path = Path(r'C:\AI-SRE\data\processed\gaia_unified.parquet')
parquet_file = pq.ParquetFile(parquet_path)
meta = parquet_file.metadata
num_row_groups = meta.num_row_groups
total_rows = meta.num_rows

TARGET_SAMPLE_SIZE = 2_000_000
SAMPLE_RATE = TARGET_SAMPLE_SIZE / total_rows

print(f"   Total rows: {total_rows:,} across {num_row_groups:,} row groups")
print("   Streaming and sampling row groups...")

sampled_chunks = []
np.random.seed(42)

for rg_idx in range(num_row_groups):
    table = parquet_file.read_row_group(rg_idx)
    df = table.to_pandas()
    
    # Stratified sample preserving intra-group distribution
    n_sample = max(1, int(len(df) * SAMPLE_RATE * 1.05))
    if len(df) > n_sample:
        # Preserve anomaly ratio within chunk
        anom_mask = df['is_anomaly']
        df_anom = df[anom_mask]
        df_norm = df[~anom_mask]
        
        n_anom = int(len(df_anom) * SAMPLE_RATE * 1.05)
        n_norm = int(len(df_norm) * SAMPLE_RATE * 1.05)
        
        s_anom = df_anom.sample(n=min(len(df_anom), max(1, n_anom)), random_state=42) if len(df_anom) > 0 else df_anom
        s_norm = df_norm.sample(n=min(len(df_norm), max(1, n_norm)), random_state=42) if len(df_norm) > 0 else df_norm
        
        sampled_chunks.append(pd.concat([s_anom, s_norm], ignore_index=True))
    else:
        sampled_chunks.append(df)
        
    if (rg_idx + 1) % 2500 == 0 or (rg_idx + 1) == num_row_groups:
        print(f"   Sampled row group {rg_idx + 1:,} / {num_row_groups:,}...", flush=True)

df_full = pd.concat(sampled_chunks, ignore_index=True)
if len(df_full) > TARGET_SAMPLE_SIZE:
    df_full = df_full.sample(n=TARGET_SAMPLE_SIZE, random_state=42)

df_full['timestamp_dt'] = pd.to_datetime(df_full['timestamp'])
df_full = df_full.sort_values(['service_name', 'timestamp_dt']).reset_index(drop=True)

print(f"   Dataset sampled: {df_full.shape}")
print(f"   Anomaly rate: {df_full['is_anomaly'].sum() / len(df_full) * 100:.2f}%")

# 2. BASE FEATURES
print("\n2. Computing base features (raw value, log transform, upper quartile flag, hour)...")
value = df_full['value'].values.astype(np.float64)
df_full['feature_0_raw_value'] = value
df_full['feature_1_log_value'] = np.log1p(np.abs(value))

q75 = np.nanpercentile(value, 75)
df_full['feature_2_high_value'] = (value > q75).astype(np.float32)

hours = df_full['timestamp_dt'].dt.hour.values.astype(np.float32)
df_full['feature_3_hour'] = hours

# 3. TEMPORAL DYNAMICS FEATURES (Per-Service Rolling Windows, Velocity, Acceleration)
print("\n3. Engineering per-service temporal dynamics (rolling mean/std, velocity, acceleration)...")

rolling_mean_list = []
rolling_std_list = []
velocity_list = []
acceleration_list = []

# Groupby service_name (data is already sorted by [service_name, timestamp_dt])
grouped = df_full.groupby('service_name', sort=False)

for svc, group in grouped:
    vals = group['feature_0_raw_value'].values
    
    # 5-sample rolling mean & std
    s_val = pd.Series(vals)
    rm = s_val.rolling(window=5, min_periods=1).mean().values
    rstd = s_val.rolling(window=5, min_periods=1).std().fillna(0.0).values
    
    # Velocity (1st derivative: rate of change)
    vel = np.zeros_like(vals)
    if len(vals) > 1:
        vel[1:] = np.diff(vals)
        
    # Acceleration (2nd derivative: change in rate of change)
    acc = np.zeros_like(vel)
    if len(vel) > 1:
        acc[1:] = np.diff(vel)
        
    rolling_mean_list.append(rm)
    rolling_std_list.append(rstd)
    velocity_list.append(vel)
    acceleration_list.append(acc)

df_full['rolling_mean_5'] = np.concatenate(rolling_mean_list)
df_full['rolling_std_5'] = np.concatenate(rolling_std_list)
df_full['velocity'] = np.concatenate(velocity_list)
df_full['acceleration'] = np.concatenate(acceleration_list)

# 4. CYCLICAL HOUR ENCODING & SERVICE ONE-HOT
print("\n4. Engineering cyclical hour encoding and service one-hot encodings...")

# Sine/Cosine cyclical encoding (24-hour periodicity)
df_full['hour_sine'] = np.sin(2 * np.pi * hours / 24.0).astype(np.float32)
df_full['hour_cosine'] = np.cos(2 * np.pi * hours / 24.0).astype(np.float32)

# One-hot encoding for all 15 services
services_list = sorted(df_full['service_name'].unique())
service_onehot_cols = []
for svc in services_list:
    col_name = f"service_{svc}"
    df_full[col_name] = (df_full['service_name'] == svc).astype(np.float32)
    service_onehot_cols.append(col_name)

print(f"   Service one-hot categories ({len(service_onehot_cols)}): {service_onehot_cols}")

# 5. CONSOLIDATE FEATURE LIST
feature_cols = (
    ['feature_0_raw_value', 'feature_1_log_value', 'feature_2_high_value', 'feature_3_hour'] +
    ['rolling_mean_5', 'rolling_std_5', 'velocity', 'acceleration'] +
    ['hour_sine', 'hour_cosine'] +
    service_onehot_cols
)

print(f"\n5. Feature matrix consolidation:")
print(f"   Total engineered features: {len(feature_cols)}")
for i, col in enumerate(feature_cols):
    print(f"      {i+1:2d}. {col}")

# Check and clean any potential NaNs/Infs
for col in feature_cols:
    df_full[col] = df_full[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)

X_all = df_full[feature_cols].values.astype(np.float32)
y_all = df_full['is_anomaly'].values.astype(np.int32)
services_all = df_full['service_name'].values
fault_types_all = df_full['fault_type'].values

# 6. FEATURE SCALING
print("\n6. Scaling features with StandardScaler...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_all)

print(f"   Scaled matrix shape: {X_scaled.shape}")
print(f"   Mean (first 6 features): {np.round(X_scaled.mean(axis=0)[:6], 4)}")
print(f"   Std  (first 6 features): {np.round(X_scaled.std(axis=0)[:6], 4)}")
print(f"   NaN count in scaled features: {np.isnan(X_scaled).sum()}")
print(f"   Inf count in scaled features: {np.isinf(X_scaled).sum()}")

# 7. STRATIFIED TRAIN / VAL / TEST SPLIT (80 / 10 / 10)
print("\n7. Stratified train/val/test split (80/10/10)...")
strat_key = np.array([f"{s}_{ft}" for s, ft in zip(services_all, fault_types_all)])

X_temp, X_test, y_temp, y_test, strat_temp, strat_test, s_temp, s_test = train_test_split(
    X_scaled, y_all, strat_key, services_all,
    test_size=0.10,
    random_state=42,
    stratify=strat_key
)

X_train, X_val, y_train, y_val, strat_train, strat_val, s_train, s_val = train_test_split(
    X_temp, y_temp, strat_temp, s_temp,
    test_size=1/9,
    random_state=42,
    stratify=strat_temp
)

print(f"   Train: {X_train.shape[0]:,} rows ({X_train.shape[0]/len(X_scaled)*100:.2f}%)")
print(f"   Val:   {X_val.shape[0]:,} rows ({X_val.shape[0]/len(X_scaled)*100:.2f}%)")
print(f"   Test:  {X_test.shape[0]:,} rows ({X_test.shape[0]/len(X_scaled)*100:.2f}%)")

print(f"   Train anomaly rate: {y_train.sum() / len(y_train) * 100:.2f}%")
print(f"   Val anomaly rate:   {y_val.sum() / len(y_val) * 100:.2f}%")
print(f"   Test anomaly rate:  {y_test.sum() / len(y_test) * 100:.2f}%")

# 8. SAVE NEW PARQUET SPLITS & SCALER
print("\n8. Saving v2 feature-engineered splits to Parquet...")
out_dir = Path(r'C:\AI-SRE\data\model_inputs')

def save_split_v2(X_mat, y_vec, svc_vec, split_name):
    df_split = pd.DataFrame(X_mat, columns=feature_cols)
    df_split['label_is_anomaly'] = y_vec
    df_split['service_name'] = svc_vec
    
    out_path = out_dir / f"{split_name}_v2_advanced_features.parquet"
    df_split.to_parquet(out_path, index=False, compression='snappy')
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"   [OK] Saved {split_name}: {out_path} ({len(df_split):,} rows, {size_mb:.1f} MB)")
    return out_path

train_path = save_split_v2(X_train, y_train, s_train, 'train_80pct')
val_path = save_split_v2(X_val, y_val, s_val, 'val_10pct')
test_path = save_split_v2(X_test, y_test, s_test, 'test_10pct')

scaler_path_new = out_dir / 'feature_scaler_v2_advanced.joblib'
joblib.dump(scaler, scaler_path_new)
print(f"   [OK] Saved new scaler: {scaler_path_new}")

print("\n" + "=" * 80)
print("ADVANCED FEATURE ENGINEERING COMPLETE")
print("=" * 80)
print(f"New feature count: {len(feature_cols)} (vs 4 original base features)")
print("Ready for Phase 2, Step 4c: Model Retraining with Advanced Features")
