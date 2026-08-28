# Phase 2, Step 5e: Domain Context Features + Precision-Optimized Retraining

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
import mlflow
import joblib
import warnings
warnings.filterwarnings('ignore')

sys.stdout.reconfigure(encoding='utf-8')

print("=" * 80)
print("PHASE 2, STEP 5e: DOMAIN CONTEXT FEATURES + PRECISION-OPTIMIZED LIGHTGBM")
print("=" * 80)

# 1. LOAD BASE DATA & COMPUTE DOMAIN STATISTICS
print("\n1. Loading data and computing domain context statistics...")
data_dir = Path(r'C:\AI-SRE\data\model_inputs')

train_df = pd.read_parquet(data_dir / 'train_80pct_v2_advanced_features.parquet')
val_df = pd.read_parquet(data_dir / 'val_10pct_v2_advanced_features.parquet')
test_df = pd.read_parquet(data_dir / 'test_10pct_v2_advanced_features.parquet')

unified_path = Path(r'C:\AI-SRE\data\processed\gaia_unified.parquet')
import pyarrow.parquet as pq
pf = pq.ParquetFile(unified_path)

print("   Computing service-level and metric-level anomaly rates from dataset...")
service_anomaly_rates = {}
metric_anomaly_rates = {}

for rg_idx in range(min(150, pf.metadata.num_row_groups)):
    table = pf.read_row_group(rg_idx)
    df_rg = table.to_pandas()
    
    for service in df_rg['service_name'].unique():
        s_data = df_rg[df_rg['service_name'] == service]
        rate = s_data['is_anomaly'].mean()
        if service not in service_anomaly_rates:
            service_anomaly_rates[service] = []
        service_anomaly_rates[service].append(rate)
        
    for metric in df_rg['metric_name'].unique():
        m_data = df_rg[df_rg['metric_name'] == metric]
        rate = m_data['is_anomaly'].mean()
        if metric not in metric_anomaly_rates:
            metric_anomaly_rates[metric] = []
        metric_anomaly_rates[metric].append(rate)

service_anomaly_rates = {k: float(np.mean(v)) for k, v in service_anomaly_rates.items()}
metric_anomaly_rates = {k: float(np.mean(v)) for k, v in metric_anomaly_rates.items()}

print(f"   Computed anomaly rates for {len(service_anomaly_rates)} services and {len(metric_anomaly_rates)} metrics")

base_feature_cols = [c for c in train_df.columns if c != 'label_is_anomaly' and c != 'service_name']

# 2. FEATURE ENGINEERING WITH DOMAIN CONTEXT
print("\n2. Engineering all features + domain context...")

def engineer_all_features_with_domain(df, base_cols):
    df_enhanced = df[base_cols].copy()
    value = df['feature_0_raw_value'].values
    rolling_mean = df['rolling_mean_5'].values
    rolling_std = df['rolling_std_5'].values
    velocity = df['velocity'].values
    s_val = pd.Series(value)
    
    # ===== 37 EXISTING FEATURES =====
    # Lag/Z-score
    lag_1_diff = np.zeros_like(value)
    lag_1_diff[1:] = np.abs(np.diff(value))
    df_enhanced['lag_1_diff'] = lag_1_diff
    
    lag_3 = np.zeros_like(value)
    lag_3[3:] = value[:-3]
    lag_3_ratio = np.where(np.abs(lag_3) > 1e-6, np.abs(value / (lag_3 + 1e-8)), 1.0)
    df_enhanced['lag_3_ratio'] = np.clip(lag_3_ratio, 0.01, 100.0)
    
    rolling_zscore = np.where(rolling_std > 1e-6, (value - rolling_mean) / (rolling_std + 1e-8), 0.0)
    df_enhanced['rolling_zscore_5'] = np.clip(rolling_zscore, -10.0, 10.0)
    
    velocity_sign_change = np.zeros_like(velocity)
    velocity_sign_change[1:] = (velocity[:-1] * velocity[1:] < 0).astype(np.float32)
    df_enhanced['velocity_sign_change'] = velocity_sign_change
    
    df_enhanced['rolling_max_5'] = s_val.rolling(5, min_periods=1).max().values
    df_enhanced['rolling_min_5'] = s_val.rolling(5, min_periods=1).min().values
    
    # Multi-scale rolling
    rolling_mean_15 = s_val.rolling(15, min_periods=1).mean().values
    df_enhanced['rolling_mean_15'] = rolling_mean_15
    rolling_std_15 = s_val.rolling(15, min_periods=1).std().fillna(0).values
    df_enhanced['rolling_std_15'] = rolling_std_15
    rolling_mean_30 = s_val.rolling(30, min_periods=1).mean().values
    df_enhanced['rolling_mean_30'] = rolling_mean_30
    rolling_std_30 = s_val.rolling(30, min_periods=1).std().fillna(0).values
    df_enhanced['rolling_std_30'] = rolling_std_30
    
    zscore_vs_15 = np.where(rolling_std_15 > 1e-6, (value - rolling_mean_15) / (rolling_std_15 + 1e-8), 0.0)
    df_enhanced['zscore_vs_15'] = np.clip(zscore_vs_15, -10.0, 10.0)
    zscore_vs_30 = np.where(rolling_std_30 > 1e-6, (value - rolling_mean_30) / (rolling_std_30 + 1e-8), 0.0)
    df_enhanced['zscore_vs_30'] = np.clip(zscore_vs_30, -10.0, 10.0)
    
    # ===== DOMAIN CONTEXT FEATURES =====
    # 1. Service-level anomaly history (prior probability)
    service_names = df['service_name'].values if 'service_name' in df.columns else np.array([''] * len(df))
    service_anom_prior = np.array([service_anomaly_rates.get(s, 0.22) for s in service_names], dtype=np.float32)
    df_enhanced['service_anomaly_history'] = service_anom_prior
    
    # 2. Z-score severity tiers (binned: 0=normal, 1=moderate, 2=high, 3=severe)
    zscore_severity = np.zeros_like(rolling_zscore)
    zscore_severity[np.abs(rolling_zscore) < 1.0] = 0.0
    zscore_severity[(np.abs(rolling_zscore) >= 1.0) & (np.abs(rolling_zscore) < 2.5)] = 1.0
    zscore_severity[(np.abs(rolling_zscore) >= 2.5) & (np.abs(rolling_zscore) < 4.0)] = 2.0
    zscore_severity[np.abs(rolling_zscore) >= 4.0] = 3.0
    df_enhanced['zscore_severity_tier'] = zscore_severity
    
    # 3. Velocity magnitude tiers (binned)
    velocity_mag = np.abs(velocity)
    velocity_tier = np.zeros_like(velocity)
    velocity_tier[velocity_mag < 0.5] = 0.0
    velocity_tier[(velocity_mag >= 0.5) & (velocity_mag < 1.5)] = 1.0
    velocity_tier[(velocity_mag >= 1.5) & (velocity_mag < 3.0)] = 2.0
    velocity_tier[velocity_mag >= 3.0] = 3.0
    df_enhanced['velocity_magnitude_tier'] = velocity_tier
    
    # 4. Rolling volatility ratio (volatility expansion/contraction)
    vol_ratio = np.where(rolling_std_30 > 1e-6, rolling_std / (rolling_std_30 + 1e-8), 1.0)
    df_enhanced['volatility_ratio_5_vs_30'] = np.clip(vol_ratio, 0.01, 100.0)
    
    # 5. Baseline deviation ratio (current mean vs 30-sample mean)
    baseline_diff = np.abs(rolling_mean - rolling_mean_30)
    df_enhanced['baseline_drift_magnitude'] = baseline_diff
    
    # 6. Business hours flag (peak variance expected 08:00-18:00)
    hour_sine = df['hour_sine'].values if 'hour_sine' in df.columns else np.zeros(len(value))
    hour_cosine = df['hour_cosine'].values if 'hour_cosine' in df.columns else np.zeros(len(value))
    hours_approx = np.arctan2(hour_sine, hour_cosine) * 24.0 / (2.0 * np.pi)
    hours_approx = (hours_approx + 24.0) % 24.0
    business_hours = ((hours_approx >= 8.0) & (hours_approx <= 18.0)).astype(np.float32)
    df_enhanced['business_hours_flag'] = business_hours
    
    return df_enhanced

print("   Engineering 43 features (37 base/temporal + 6 domain context)...")
X_train_full = engineer_all_features_with_domain(train_df, base_feature_cols).values
y_train = train_df['label_is_anomaly'].values

X_val_full = engineer_all_features_with_domain(val_df, base_feature_cols).values
y_val = val_df['label_is_anomaly'].values

X_test_full = engineer_all_features_with_domain(test_df, base_feature_cols).values
y_test = test_df['label_is_anomaly'].values

feature_cols_domain = list(engineer_all_features_with_domain(train_df, base_feature_cols).columns)
print(f"   Total features: {len(feature_cols_domain)}")
print(f"   New domain features: {feature_cols_domain[-6:]}")

# 3. TRAIN PRECISION-OPTIMIZED LIGHTGBM
print("\n3. Training precision-optimized LightGBM...")

scale_pos_weight = float((1 - y_train.mean()) / y_train.mean())

params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'num_leaves': 255,
    'learning_rate': 0.02,
    'feature_fraction': 0.90,
    'bagging_fraction': 0.90,
    'bagging_freq': 5,
    'lambda_l1': 1.0,
    'lambda_l2': 1.0,
    'min_data_in_leaf': 15,
    'verbose': -1,
    'scale_pos_weight': scale_pos_weight,
    'n_jobs': 4,
    'random_state': 42
}

train_data = lgb.Dataset(X_train_full, label=y_train, feature_name=feature_cols_domain)
val_data = lgb.Dataset(X_val_full, label=y_val, feature_name=feature_cols_domain, reference=train_data)

model_precision = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[train_data, val_data],
    valid_names=['train', 'val'],
    callbacks=[
        lgb.early_stopping(stopping_rounds=40, verbose=False),
        lgb.log_evaluation(period=100)
    ]
)

print(f"   [OK] Model trained ({model_precision.num_trees()} trees)")

# 4. MAKE PREDICTIONS
print("\n4. Computing probabilities...")
y_pred_val_proba = model_precision.predict(X_val_full)
y_pred_test_proba = model_precision.predict(X_test_full)

# 5. OPTIMIZE FOR HIGHEST F1 WITH HIGH PRECISION
print("\n5. Threshold optimization across validation set...")
thresholds = np.linspace(0.1, 0.95, 500)
best_f1 = 0.0
best_threshold = 0.5
best_metrics = {}

for threshold in thresholds:
    y_pred = (y_pred_val_proba > threshold).astype(int)
    f1 = f1_score(y_val, y_pred, zero_division=0)
    prec = precision_score(y_val, y_pred, zero_division=0)
    rec = recall_score(y_val, y_pred, zero_division=0)
    
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold
        best_metrics = {'precision': prec, 'recall': rec, 'f1': f1}

print(f"   Best threshold: {best_threshold:.3f}")
print(f"   Validation Precision: {best_metrics['precision']:.4f}")
print(f"   Validation Recall:    {best_metrics['recall']:.4f}")
print(f"   Validation F1:        {best_metrics['f1']:.4f}")

# 6. EVALUATE ON TEST SET
print("\n6. Evaluating on test set...")
y_pred_test = (y_pred_test_proba > best_threshold).astype(int)

test_f1 = float(f1_score(y_test, y_pred_test, zero_division=0))
test_prec = float(precision_score(y_test, y_pred_test, zero_division=0))
test_rec = float(recall_score(y_test, y_pred_test, zero_division=0))
test_auc = float(roc_auc_score(y_test, y_pred_test_proba))

cm = confusion_matrix(y_test, y_pred_test)
tn, fp, fn, tp = cm.ravel()

print(f"\n   TEST SET EVALUATION:")
print(f"      F1-Score:  {test_f1:.4f}")
print(f"      Precision: {test_prec:.4f}")
print(f"      Recall:    {test_rec:.4f}")
print(f"      ROC-AUC:   {test_auc:.4f}")
print(f"\n   Confusion Matrix:")
print(f"      True Negatives:  {tn:,}")
print(f"      False Positives: {fp:,}")
print(f"      False Negatives: {fn:,}")
print(f"      True Positives:  {tp:,}")

# 7. FEATURE IMPORTANCE
print("\n7. Top 20 Most Important Features (Gain):")
importance = model_precision.feature_importance(importance_type='gain')
feature_importance_df = pd.DataFrame({
    'feature': feature_cols_domain,
    'importance': importance
}).sort_values('importance', ascending=False)

for idx, row in feature_importance_df.head(20).iterrows():
    print(f"      {row['feature']:35s} : {row['importance']:12.2f}")

# 8. MLFLOW LOGGING
print("\n8. Logging to MLflow...")
mlflow.set_experiment("phase2_anomaly_detection")
mlflow.start_run(run_name="lightgbm_precision_optimized_v2")

mlflow.log_params({
    'model_type': 'lightgbm_precision_optimized',
    'num_trees': model_precision.num_trees(),
    'num_leaves': params['num_leaves'],
    'learning_rate': params['learning_rate'],
    'lambda_l1': params['lambda_l1'],
    'lambda_l2': params['lambda_l2'],
    'scale_pos_weight': scale_pos_weight,
    'feature_count': len(feature_cols_domain)
})

mlflow.log_metrics({
    'test_f1': test_f1,
    'test_precision': test_prec,
    'test_recall': test_rec,
    'test_auc': test_auc,
    'best_threshold': float(best_threshold),
    'val_best_f1': float(best_f1),
    'true_negatives': int(tn),
    'false_positives': int(fp),
    'false_negatives': int(fn),
    'true_positives': int(tp)
})

# Save model
model_path = Path(r'C:\AI-SRE\src\models\lightgbm_precision_optimized.joblib')
model_path.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(model_precision, model_path)
print(f"   [OK] Model saved: {model_path}")

config = {
    'model_type': 'lightgbm_precision_optimized',
    'features': len(feature_cols_domain),
    'optimal_threshold': float(best_threshold),
    'test_f1': test_f1,
    'test_precision': test_prec,
    'test_recall': test_rec,
    'test_auc': test_auc
}
config_path = Path(r'C:\AI-SRE\src\models\precision_optimized_config.joblib')
joblib.dump(config, config_path)
print(f"   [OK] Config saved: {config_path}")

try:
    mlflow.lightgbm.log_model(model_precision, artifact_path='lightgbm_precision_optimized')
except Exception as e:
    print(f"   Note: {e}")

mlflow.end_run()

print("   [OK] Logged to MLflow")

print("\n" + "=" * 80)
print("PRECISION-OPTIMIZED MODEL COMPLETE")
print("=" * 80)

# 9. FINAL RESULTS
print(f"\n{'='*80}")
print("PHASE 2 FINAL RESULTS: PRECISION-OPTIMIZED LIGHTGBM")
print(f"{'='*80}")
print(f"   Test F1:        {test_f1:.4f}")
print(f"   Test Precision: {test_prec:.4f}")
print(f"   Test Recall:    {test_rec:.4f}")
print(f"   Test AUC:       {test_auc:.4f}")
print(f"   Features:       {len(feature_cols_domain)} (37 base + 6 domain context)")
print(f"   Threshold:      {best_threshold:.3f}")
print(f"{'='*80}")
