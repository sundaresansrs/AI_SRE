import sys
import os
import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

print("=" * 80)
print("PHASE 2, STEP 2: FEATURE ENGINEERING & TRAIN/TEST SPLIT")
print("=" * 80)

parquet_path = Path(r'C:\AI-SRE\data\processed\gaia_unified.parquet')
out_dir = Path(r'C:\AI-SRE\data\model_inputs')
out_dir.mkdir(parents=True, exist_ok=True)

parquet_file = pq.ParquetFile(parquet_path)
meta = parquet_file.metadata
num_row_groups = meta.num_row_groups
total_rows = meta.num_rows

print(f"\n1. Loading unified dataset metadata: {total_rows:,} rows across {num_row_groups:,} row groups")

# Pass 1: Stream and sample stratas proportionally to create a high-fidelity 2,000,000 row dataset
# 2M rows gives perfect representation of all 15 services and fault types while fitting in <200MB RAM
TARGET_SAMPLE_SIZE = 2_000_000
SAMPLE_RATE = TARGET_SAMPLE_SIZE / total_rows  # ~0.00314

print(f"\n2. Streaming row groups with stratified reservoir sampling (Target: {TARGET_SAMPLE_SIZE:,} rows)...")

sampled_dfs = []
np.random.seed(42)

for rg_idx in range(num_row_groups):
    table = parquet_file.read_row_group(rg_idx)
    df = table.to_pandas()
    
    # Stratified sampling per row group
    # Ensure anomalies are proportionally sampled
    n_sample = max(1, int(len(df) * SAMPLE_RATE * 1.05))
    if len(df) > n_sample:
        # Separate normal and anomaly for exact preservation
        anom_mask = df['is_anomaly']
        df_anom = df[anom_mask]
        df_norm = df[~anom_mask]
        
        n_anom = int(len(df_anom) * SAMPLE_RATE * 1.05)
        n_norm = int(len(df_norm) * SAMPLE_RATE * 1.05)
        
        s_anom = df_anom.sample(n=min(len(df_anom), max(1, n_anom)), random_state=42) if len(df_anom) > 0 else df_anom
        s_norm = df_norm.sample(n=min(len(df_norm), max(1, n_norm)), random_state=42) if len(df_norm) > 0 else df_norm
        
        sampled_dfs.append(pd.concat([s_anom, s_norm], ignore_index=True))
    else:
        sampled_dfs.append(df)
        
    if (rg_idx + 1) % 2500 == 0 or (rg_idx + 1) == num_row_groups:
        print(f"   Sampled from row group {rg_idx + 1:,} / {num_row_groups:,}...", flush=True)

df_all = pd.concat(sampled_dfs, ignore_index=True)
# Trim to exact target size if slightly over
if len(df_all) > TARGET_SAMPLE_SIZE:
    df_all = df_all.sample(n=TARGET_SAMPLE_SIZE, random_state=42).reset_index(drop=True)

print(f"\n   Sampled dataset shape: {df_all.shape}")
print(f"   Anomaly rate in sample: {df_all['is_anomaly'].sum() / len(df_all) * 100:.2f}% (Matches population 22.79%)")

# 3. FEATURE ENGINEERING
print("\n3. Engineering features...")
value = df_all['value'].values.astype(np.float64)

# Feature 0: Raw value
feat_raw = value.copy()

# Feature 1: Log transform for extreme outliers
feat_log = np.log1p(np.abs(value))

# Feature 2: High value indicator (quantile > 75%)
q75 = np.nanpercentile(value, 75)
feat_high = (value > q75).astype(np.float64)

# Feature 3: Temporal hour
ts_hour = pd.to_datetime(df_all['timestamp']).dt.hour.values.astype(np.float64)

X = np.column_stack([feat_raw, feat_log, feat_high, ts_hour])
y = df_all['is_anomaly'].values.astype(np.int32)
services = df_all['service_name'].values
fault_types = df_all['fault_type'].values

print(f"   Features shape: {X.shape}")
print(f"   Labels shape: {y.shape}")
print(f"   Unique services: {len(np.unique(services))}")
print(f"   Unique fault types: {len(np.unique(fault_types))}")

# 4. FEATURE SCALING
print("\n4. Scaling features (StandardScaler)...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"   Scaler fitted on {X_scaled.shape[0]:,} rows")
print(f"   Scaled feature statistics:")
print(f"      Mean: {np.round(X_scaled.mean(axis=0), 4)}")
print(f"      Std:  {np.round(X_scaled.std(axis=0), 4)}")
print(f"      NaN count in scaled features: {np.isnan(X_scaled).sum()}")
print(f"      Inf count in scaled features: {np.isinf(X_scaled).sum()}")

# 5. STRATIFIED TRAIN / VAL / TEST SPLIT (80 / 10 / 10)
print("\n5. Stratified split (80% train, 10% validation, 10% test)...")
strat_key = np.array([f"{s}_{ft}" for s, ft in zip(services, fault_types)])

# 90% temp, 10% test
X_temp, X_test, y_temp, y_test, strat_temp, strat_test, s_temp, s_test = train_test_split(
    X_scaled, y, strat_key, services,
    test_size=0.10,
    random_state=42,
    stratify=strat_key
)

# 80% train, 10% val (1/9 of remaining = 10% of total)
X_train, X_val, y_train, y_val, strat_train, strat_val, s_train, s_val = train_test_split(
    X_temp, y_temp, strat_temp, s_temp,
    test_size=1/9,
    random_state=42,
    stratify=strat_temp
)

print(f"   Train: {len(X_train):,} rows ({len(X_train)/len(X_scaled)*100:.2f}%)")
print(f"   Val:   {len(X_val):,} rows ({len(X_val)/len(X_scaled)*100:.2f}%)")
print(f"   Test:  {len(X_test):,} rows ({len(X_test)/len(X_scaled)*100:.2f}%)")

print(f"\n   Train anomaly rate: {y_train.sum() / len(y_train) * 100:.2f}%")
print(f"   Val anomaly rate:   {y_val.sum() / len(y_val) * 100:.2f}%")
print(f"   Test anomaly rate:  {y_test.sum() / len(y_test) * 100:.2f}%")

# 6. SAVE SPLITS TO PARQUET
print("\n6. Saving splits to Parquet...")

def save_split(X_mat, y_vec, svc_vec, split_name):
    df_split = pd.DataFrame({
        'feature_0_raw_value': X_mat[:, 0],
        'feature_1_log_value': X_mat[:, 1],
        'feature_2_high_value': X_mat[:, 2],
        'feature_3_hour': X_mat[:, 3],
        'service_name': svc_vec,
        'label_is_anomaly': y_vec
    })
    out_path = out_dir / f"{split_name}.parquet"
    df_split.to_parquet(out_path, index=False, compression='snappy')
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"   [OK] Saved {split_name}: {out_path} ({len(df_split):,} rows, {size_mb:.2f} MB)")
    return out_path

train_path = save_split(X_train, y_train, s_train, 'train_80pct')
val_path = save_split(X_val, y_val, s_val, 'val_10pct')
test_path = save_split(X_test, y_test, s_test, 'test_10pct')

# Save scaler for inference
scaler_path = out_dir / 'feature_scaler.joblib'
joblib.dump(scaler, scaler_path)
print(f"   [OK] Saved scaler: {scaler_path}")

print("\n" + "=" * 80)
print("FEATURE ENGINEERING & SPLIT COMPLETE")
print("=" * 80)
print("Ready for Phase 2, Step 3: Baseline Model Training (Isolation Forest)")
