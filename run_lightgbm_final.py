# Phase 2, Step 5c: Multi-Scale Rolling Features + Final LightGBM Push

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import mlflow
import mlflow.lightgbm
import joblib
import warnings
warnings.filterwarnings('ignore')

sys.stdout.reconfigure(encoding='utf-8')

print("=" * 80)
print("PHASE 2, STEP 5c: MULTI-SCALE ROLLING FEATURES + FINAL LIGHTGBM PUSH")
print("=" * 80)

# 1. LOAD V2 SPLITS
print("\n1. Loading splits and engineering multi-scale rolling features...")
data_dir = Path(r'C:\AI-SRE\data\model_inputs')

train_df = pd.read_parquet(data_dir / 'train_80pct_v2_advanced_features.parquet')
val_df = pd.read_parquet(data_dir / 'val_10pct_v2_advanced_features.parquet')
test_df = pd.read_parquet(data_dir / 'test_10pct_v2_advanced_features.parquet')

base_feature_cols = [c for c in train_df.columns if c != 'label_is_anomaly' and c != 'service_name']

# Fast vectorized function to engineer all features
def engineer_all_features(df, base_cols):
    df_enhanced = df[base_cols].copy()
    
    value = df['feature_0_raw_value'].values
    rolling_mean = df['rolling_mean_5'].values
    rolling_std = df['rolling_std_5'].values
    velocity = df['velocity'].values
    s_val = pd.Series(value)
    
    # ===== LAG/Z-SCORE FEATURES (Step 5b) =====
    # 1. Lag-1 difference
    lag_1_diff = np.zeros_like(value)
    lag_1_diff[1:] = np.abs(np.diff(value))
    df_enhanced['lag_1_diff'] = lag_1_diff
    
    # 2. Lag-3 ratio
    lag_3 = np.zeros_like(value)
    lag_3[3:] = value[:-3]
    lag_3_ratio = np.where(np.abs(lag_3) > 1e-6, np.abs(value / (lag_3 + 1e-8)), 1.0)
    df_enhanced['lag_3_ratio'] = np.clip(lag_3_ratio, 0.01, 100.0)
    
    # 3. Rolling Z-score (5-sample)
    rolling_zscore = np.where(rolling_std > 1e-6, (value - rolling_mean) / (rolling_std + 1e-8), 0.0)
    df_enhanced['rolling_zscore_5'] = np.clip(rolling_zscore, -10.0, 10.0)
    
    # 4. Velocity sign change
    velocity_sign_change = np.zeros_like(velocity)
    velocity_sign_change[1:] = (velocity[:-1] * velocity[1:] < 0).astype(np.float32)
    df_enhanced['velocity_sign_change'] = velocity_sign_change
    
    # 5. Rolling max (5-sample)
    df_enhanced['rolling_max_5'] = s_val.rolling(5, min_periods=1).max().values
    
    # 6. Rolling min (5-sample)
    df_enhanced['rolling_min_5'] = s_val.rolling(5, min_periods=1).min().values
    
    # ===== NEW: MULTI-SCALE ROLLING FEATURES (Step 5c) =====
    # 7. 15-sample rolling mean
    rolling_mean_15 = s_val.rolling(15, min_periods=1).mean().values
    df_enhanced['rolling_mean_15'] = rolling_mean_15
    
    # 8. 15-sample rolling std
    rolling_std_15 = s_val.rolling(15, min_periods=1).std().fillna(0).values
    df_enhanced['rolling_std_15'] = rolling_std_15
    
    # 9. 30-sample rolling mean
    rolling_mean_30 = s_val.rolling(30, min_periods=1).mean().values
    df_enhanced['rolling_mean_30'] = rolling_mean_30
    
    # 10. 30-sample rolling std
    rolling_std_30 = s_val.rolling(30, min_periods=1).std().fillna(0).values
    df_enhanced['rolling_std_30'] = rolling_std_30
    
    # 11. Z-score vs 15-sample baseline
    zscore_vs_15 = np.where(rolling_std_15 > 1e-6, (value - rolling_mean_15) / (rolling_std_15 + 1e-8), 0.0)
    df_enhanced['zscore_vs_15'] = np.clip(zscore_vs_15, -10.0, 10.0)
    
    # 12. Z-score vs 30-sample baseline
    zscore_vs_30 = np.where(rolling_std_30 > 1e-6, (value - rolling_mean_30) / (rolling_std_30 + 1e-8), 0.0)
    df_enhanced['zscore_vs_30'] = np.clip(zscore_vs_30, -10.0, 10.0)
    
    return df_enhanced

print("   Engineering all features (25 base + 6 lag/zscore + 6 multi-scale)...")
X_train_enhanced = engineer_all_features(train_df, base_feature_cols)
y_train = train_df['label_is_anomaly'].values

X_val_enhanced = engineer_all_features(val_df, base_feature_cols)
y_val = val_df['label_is_anomaly'].values

X_test_enhanced = engineer_all_features(test_df, base_feature_cols)
y_test = test_df['label_is_anomaly'].values

X_train = X_train_enhanced.values
X_val = X_val_enhanced.values
X_test = X_test_enhanced.values

feature_cols_final = list(X_train_enhanced.columns)

print(f"\n   Total features: {len(feature_cols_final)}")
print(f"   Base features: {len(base_feature_cols)}")
print(f"   Lag/Z-score features: 6")
print(f"   Multi-scale rolling features: 6")
print(f"   New multi-scale columns: {feature_cols_final[-6:]}")

# 2. PREPARE LIGHTGBM DATASETS
print("\n2. Preparing LightGBM datasets...")
train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols_final)
val_data = lgb.Dataset(X_val, label=y_val, feature_name=feature_cols_final, reference=train_data)

# 3. TRAIN FINAL LIGHTGBM WITH OPTIMIZED HYPERPARAMETERS
print("\n3. Training final LightGBM with multi-scale features...")

scale_pos_weight = float((1 - y_train.mean()) / y_train.mean())

params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'num_leaves': 255,
    'learning_rate': 0.03,
    'feature_fraction': 0.90,
    'bagging_fraction': 0.90,
    'bagging_freq': 5,
    'lambda_l1': 0.3,
    'lambda_l2': 0.3,
    'min_data_in_leaf': 10,
    'verbose': -1,
    'scale_pos_weight': scale_pos_weight,
    'n_jobs': 4,
    'random_state': 42
}

print(f"   Scale pos weight: {scale_pos_weight:.2f}")
print(f"   Num leaves: {params['num_leaves']}")
print(f"   Learning rate: {params['learning_rate']}")
print(f"   Total features: {len(feature_cols_final)}")

model_final = lgb.train(
    params,
    train_data,
    num_boost_round=800,
    valid_sets=[train_data, val_data],
    valid_names=['train', 'val'],
    callbacks=[
        lgb.early_stopping(stopping_rounds=40, verbose=False),
        lgb.log_evaluation(period=50)
    ]
)

print(f"   [OK] Model trained ({model_final.num_trees()} trees)")

# 4. MAKE PREDICTIONS
print("\n4. Making predictions...")
y_pred_val_proba = model_final.predict(X_val)
y_pred_test_proba = model_final.predict(X_test)

# 5. THRESHOLD OPTIMIZATION
print("\n5. Optimizing classification threshold on validation set...")
thresholds = np.linspace(0.05, 0.95, 300)
best_f1 = 0.0
best_threshold = 0.5

for threshold in thresholds:
    y_pred = (y_pred_val_proba > threshold).astype(int)
    f1 = f1_score(y_val, y_pred, zero_division=0)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"   Best threshold: {best_threshold:.3f}")
print(f"   Best F1 on validation: {best_f1:.4f}")

# 6. EVALUATE ON TEST SET
print("\n6. Evaluating on test set...")
y_pred_test = (y_pred_test_proba > best_threshold).astype(int)

test_f1 = float(f1_score(y_test, y_pred_test, zero_division=0))
test_prec = float(precision_score(y_test, y_pred_test, zero_division=0))
test_rec = float(recall_score(y_test, y_pred_test, zero_division=0))
test_auc = float(roc_auc_score(y_test, y_pred_test_proba))

print(f"\n   TEST SET EVALUATION:")
print(f"      F1-Score:  {test_f1:.4f}")
print(f"      Precision: {test_prec:.4f}")
print(f"      Recall:    {test_rec:.4f}")
print(f"      ROC-AUC:   {test_auc:.4f}")

# 7. FEATURE IMPORTANCE
print("\n7. Top 20 Most Important Features (Split Gain):")
importance_gain = model_final.feature_importance(importance_type='gain')
feature_importance = pd.DataFrame({
    'feature': feature_cols_final,
    'importance': importance_gain
}).sort_values('importance', ascending=False)

for idx, row in feature_importance.head(20).iterrows():
    print(f"      {row['feature']:30s} : {row['importance']:12.2f}")

# 8. MLFLOW LOGGING
print("\n8. Logging to MLflow...")
mlflow.set_experiment("phase2_anomaly_detection")
mlflow.start_run(run_name="lightgbm_final_multiscale")

mlflow.log_params({
    'model_type': 'lightgbm_final_classifier',
    'num_trees': model_final.num_trees(),
    'num_leaves': params['num_leaves'],
    'learning_rate': params['learning_rate'],
    'scale_pos_weight': scale_pos_weight,
    'feature_count': len(feature_cols_final),
    'base_features': len(base_feature_cols),
    'lag_zscore_features': 6,
    'multiscale_rolling_features': 6,
    'train_samples': len(X_train),
    'val_samples': len(X_val),
    'test_samples': len(X_test)
})

mlflow.log_metrics({
    'test_f1': test_f1,
    'test_precision': test_prec,
    'test_recall': test_rec,
    'test_auc': test_auc,
    'best_threshold': float(best_threshold),
    'val_best_f1': float(best_f1)
})

# Save model
model_path = Path(r'C:\AI-SRE\src\models\lightgbm_final_classifier.joblib')
model_path.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(model_final, model_path)
print(f"   [OK] Model saved to {model_path}")

try:
    mlflow.lightgbm.log_model(model_final, artifact_path='lightgbm_final_classifier')
except Exception as e:
    print(f"   Note: {e}")

mlflow.end_run()

print("   [OK] Logged to MLflow")

print("\n" + "=" * 80)
print("FINAL LIGHTGBM WITH MULTI-SCALE FEATURES COMPLETE")
print("=" * 80)

# 9. COMPREHENSIVE SUMMARY
print(f"\n{'='*80}")
print("PHASE 2 COMPLETE MODEL EVOLUTION:")
print(f"{'='*80}")
print(f"   Step 3:  Isolation Forest (Baseline)")
print(f"            → F1 = 0.1535  |  AUC = 0.5141")
print(f"\n   Step 4:  Autoencoder V1 (4 features)")
print(f"            → F1 = 0.3178  |  AUC = 0.5346  (+107% F1)")
print(f"\n   Step 4b: Autoencoder V2 (25 features)")
print(f"            → F1 = 0.2880  |  AUC = 0.4839  (reconstruction-based plateau)")
print(f"\n   Step 5:  LightGBM Base (25 features, supervised)")
print(f"            → F1 = 0.7567  |  AUC = 0.9455  (+163% vs Autoencoder)")
print(f"\n   Step 5b: LightGBM Tuned (31 features + lag/zscore)")
print(f"            → F1 = 0.7737  |  AUC = 0.9521  (+2.3% improvement)")
print(f"\n   Step 5c: LightGBM Final (37 features + multi-scale rolling)")
print(f"            → F1 = {test_f1:.4f}  |  AUC = {test_auc:.4f}")
print(f"{'='*80}")

if test_f1 >= 0.80:
    print(f"\n[SUCCESS] PHASE 2 SUCCESSFULLY CLOSED")
    print(f"\nTARGET ACHIEVED: F1 >= 0.80")
    print(f"   Final Test F1:        {test_f1:.4f}")
    print(f"   Final Test Precision: {test_prec:.4f}")
    print(f"   Final Test Recall:    {test_rec:.4f}")
    print(f"   Final Test AUC:       {test_auc:.4f}")
    print(f"\n   Model Type:  LightGBM Gradient Boosting Classifier")
    print(f"   Features:    {len(feature_cols_final)} (base + temporal + multi-scale)")
    print(f"   Trees:       {model_final.num_trees()}")
    print(f"\nREADY TO BEGIN PHASE 3 (RAG KNOWLEDGE BASE) IMMEDIATELY")
    print(f"{'='*80}")
else:
    print(f"\n[NOTE] Test F1 = {test_f1:.4f} (still < 0.80 target, but very close)")
    print(f"   Gap remaining: {0.80 - test_f1:.4f}")
print(f"{'='*80}")
