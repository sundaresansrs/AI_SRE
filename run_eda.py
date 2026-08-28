import sys
import pyarrow.parquet as pq
import pyarrow.compute as pc
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

# Ensure UTF-8 stdout
sys.stdout.reconfigure(encoding='utf-8')

print("=" * 80)
print("PHASE 2, STEP 1: EXPLORATORY DATA ANALYSIS (MEMORY-EFFICIENT STREAMING SCAN)")
print("=" * 80)

parquet_path = Path(r'C:\AI-SRE\data\processed\gaia_unified.parquet')
parquet_file = pq.ParquetFile(parquet_path)

meta = parquet_file.metadata
num_rows = meta.num_rows
num_row_groups = meta.num_row_groups
print(f"[OK] Parquet opened: {num_rows:,} rows across {num_row_groups:,} row groups")
print(f"   Schema columns: {meta.schema.names}")

# Accumulators
total_rows = 0
anomaly_true = 0
anomaly_false = 0

missing_counts = {col: 0 for col in meta.schema.names}

service_row_counts = {}
service_anomaly_counts = {}

fault_type_row_counts = {}
fault_type_anomaly_counts = {}

min_ts = None
max_ts = None

val_min = float('inf')
val_max = float('-inf')
val_sum = 0.0
val_count = 0
inf_count = 0
sampled_values = []

print("\nStreaming row groups for exact aggregations...", flush=True)

np.random.seed(42)

for rg_idx in range(num_row_groups):
    table = parquet_file.read_row_group(rg_idx)
    rg_len = len(table)
    total_rows += rg_len
    
    # Missing values
    for col in meta.schema.names:
        missing_counts[col] += table[col].null_count
        
    # Anomaly counts
    anom_col = table['is_anomaly']
    anom_true_rg = pc.sum(pc.cast(anom_col, 'int64')).as_py() or 0
    anomaly_true += anom_true_rg
    anomaly_false += (rg_len - anom_true_rg)
    
    # Timestamps
    ts_chunk_min = pc.min(table['timestamp']).as_py()
    ts_chunk_max = pc.max(table['timestamp']).as_py()
    if min_ts is None or (ts_chunk_min is not None and ts_chunk_min < min_ts):
        min_ts = ts_chunk_min
    if max_ts is None or (ts_chunk_max is not None and ts_chunk_max > max_ts):
        max_ts = ts_chunk_max
        
    # Value stats
    val_col = table['value']
    rg_val_min = pc.min(val_col).as_py()
    rg_val_max = pc.max(val_col).as_py()
    if rg_val_min is not None and rg_val_min < val_min:
        val_min = rg_val_min
    if rg_val_max is not None and rg_val_max > val_max:
        val_max = rg_val_max
        
    val_arr = val_col.to_numpy(zero_copy_only=False)
    finite_mask = np.isfinite(val_arr)
    inf_count += int((~finite_mask).sum())
    finite_vals = val_arr[finite_mask]
    if len(finite_vals) > 0:
        val_sum += float(finite_vals.sum())
        val_count += len(finite_vals)
        
    # Sample 100 values per row group for plotting
    if len(finite_vals) > 100:
        sampled_values.extend(np.random.choice(finite_vals, size=100, replace=False))
    else:
        sampled_values.extend(finite_vals)

    # Groupby service_name & is_anomaly in pandas for this row group
    df_small = table.select(['service_name', 'fault_type', 'is_anomaly']).to_pandas()
    
    svc_grp = df_small.groupby('service_name')['is_anomaly'].agg(['count', 'sum'])
    for svc, row in svc_grp.iterrows():
        service_row_counts[svc] = service_row_counts.get(svc, 0) + int(row['count'])
        service_anomaly_counts[svc] = service_anomaly_counts.get(svc, 0) + int(row['sum'])
        
    flt_grp = df_small.groupby('fault_type')['is_anomaly'].agg(['count', 'sum'])
    for flt, row in flt_grp.iterrows():
        fault_type_row_counts[flt] = fault_type_row_counts.get(flt, 0) + int(row['count'])
        fault_type_anomaly_counts[flt] = fault_type_anomaly_counts.get(flt, 0) + int(row['sum'])

    if (rg_idx + 1) % 2500 == 0 or (rg_idx + 1) == num_row_groups:
        print(f"  Processed row group {rg_idx + 1:,} / {num_row_groups:,} ({total_rows:,} rows)...", flush=True)

# 2. BASIC STATISTICS
print("\n2. Basic Statistics:")
print(f"   Shape: ({total_rows:,}, {len(meta.schema.names)})")
print(f"   Columns: {meta.schema.names}")
print("   Missing values:")
for col, cnt in missing_counts.items():
    print(f"      {col}: {cnt:,}")

# 3. ANOMALY DISTRIBUTION
print("\n3. Anomaly Distribution:")
print(f"   Normal: {anomaly_false:,} ({anomaly_false / total_rows * 100:.2f}%)")
print(f"   Anomaly: {anomaly_true:,} ({anomaly_true / total_rows * 100:.2f}%)")

# 4. ANOMALY BY FAULT_TYPE
print("\n4. Anomaly Distribution by Fault Type (Top 15):")
flt_df = pd.DataFrame({
    'fault_type': list(fault_type_row_counts.keys()),
    'total_rows': list(fault_type_row_counts.values()),
    'anomaly_count': [fault_type_anomaly_counts.get(k, 0) for k in fault_type_row_counts.keys()]
})
flt_df['anomaly_pct'] = (flt_df['anomaly_count'] / flt_df['total_rows'] * 100).round(2)
print(flt_df.sort_values('total_rows', ascending=False).head(15).to_string(index=False))

# 5. ANOMALY BY SERVICE
print("\n5. Top 15 Services by Row Count (and anomaly rate):")
svc_df = pd.DataFrame({
    'service_name': list(service_row_counts.keys()),
    'total_rows': list(service_row_counts.values()),
    'anomaly_count': [service_anomaly_counts.get(k, 0) for k in service_row_counts.keys()]
})
svc_df['anomaly_pct'] = (svc_df['anomaly_count'] / svc_df['total_rows'] * 100).round(2)
print(svc_df.sort_values('total_rows', ascending=False).head(15).to_string(index=False))

# 6. VALUE COLUMN STATISTICS
print("\n6. Value Column Statistics:")
print(f"   Min: {val_min}")
print(f"   Max: {val_max}")
val_mean = val_sum / val_count if val_count > 0 else 0
print(f"   Mean: {val_mean:.4f}")
print(f"   Inf values: {inf_count:,}")
print(f"   Sampled values collected for distribution: {len(sampled_values):,}")

# 7. TEMPORAL COVERAGE
print("\n7. Temporal Coverage:")
print(f"   Date range: {min_ts} to {max_ts}")
if min_ts and max_ts:
    days_covered = (max_ts - min_ts).days
    print(f"   Days covered: {days_covered}")

# 8. VISUALIZATIONS
print("\n8. Generating visualizations...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Top 10 fault types by row count & anomaly %
top_flt = flt_df.sort_values('total_rows', ascending=False).head(10)
axes[0, 0].bar(range(len(top_flt)), top_flt['anomaly_pct'], color='steelblue', edgecolor='black')
axes[0, 0].set_xticks(range(len(top_flt)))
axes[0, 0].set_xticklabels(top_flt['fault_type'], rotation=45, ha='right', fontsize=9)
axes[0, 0].set_title('Anomaly Rate by Fault Type (Top 10 by Volume)', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Anomaly %')
axes[0, 0].grid(axis='y', linestyle='--', alpha=0.7)

# Plot 2: Value distribution (log scale on sample)
sample_arr = np.array(sampled_values)
positive_samples = sample_arr[sample_arr > 0]
axes[0, 1].hist(positive_samples, bins=60, color='coral', edgecolor='black', log=True)
axes[0, 1].set_title('Value Distribution (Sampled > 0, Log Scale)', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Value')
axes[0, 1].set_ylabel('Frequency (Log Scale)')
axes[0, 1].grid(axis='y', linestyle='--', alpha=0.7)

# Plot 3: Top 10 services by row count
top_svc = svc_df.sort_values('total_rows', ascending=True).tail(10)
axes[1, 0].barh(range(len(top_svc)), top_svc['total_rows'] / 1e6, color='mediumseagreen', edgecolor='black')
axes[1, 0].set_yticks(range(len(top_svc)))
axes[1, 0].set_yticklabels(top_svc['service_name'], fontsize=10)
axes[1, 0].set_title('Top 10 Services by Total Volume (Millions of Rows)', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Rows (Millions)')
axes[1, 0].grid(axis='x', linestyle='--', alpha=0.7)

# Plot 4: Anomaly class balance pie
axes[1, 1].pie([anomaly_false, anomaly_true], labels=['Normal', 'Anomaly'], autopct='%1.2f%%', colors=['#66b3ff', '#ff9999'], explode=(0, 0.1), startangle=140, shadow=True)
axes[1, 1].set_title(f'Class Distribution (Total: {total_rows:,} rows)', fontsize=12, fontweight='bold')

plt.tight_layout()
out_img = Path(r'C:\AI-SRE\notebooks\eda_phase2_visualizations.png')
plt.savefig(out_img, dpi=150, bbox_inches='tight')
print(f"   [OK] Saved visualizations to {out_img}")

print("\n" + "=" * 80)
print("EDA COMPLETE")
print("=" * 80)
