# Phase 2, Step 5b: Enhanced Lag/Z-Score Features + LightGBM Tuning

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
print("PHASE 2, STEP 5b: ENHANCED LAG/Z-SCORE FEATURES + TUNED LIGHTGBM")
print("=" * 80)

# 1. LOAD V2 SPLITS
print("\n1. Loading v2 splits and engineering enhanced features...")
data_dir = Path(r'C:\AI-SRE\data\model_inputs')

train_df = pd.read_parquet(data_dir / 'train_80pct_v2_advanced_features.parquet')
val_df = pd.read_parquet(data_dir / 'val_10pct_v2_advanced_features.parquet')
test_df = pd.read_parquet(data_dir / 'test_10pct_v2_advanced_features.parquet')

base_feature_cols = [c for c in train_df.columns if c != 'label_is_anomaly' and c != 'service_name']

# Fast vectorized function to engineer lag/z-score features
def engineer_enhanced_features(df, base_cols):
    df_enhanced = df[base_cols].copy()
    
    value = df['feature_0_raw_value'].values
    rolling_mean = df['rolling_mean_5'].values
    rolling_std = df['rolling_std_5'].values
    velocity = df['velocity'].values
    
    # 1. Lag-1 difference
    lag_1_diff = np.zeros_like(value)
    lag_1_diff[1:] = np.abs(np.diff(value))
    df_enhanced['lag_1_diff'] = lag_1_diff
    
    # 2. Lag-3 ratio
    lag_3 = np.zeros_like(value)
    lag_3[3:] = value[:-3]
    lag_3_ratio = np.where(np.abs(lag_3) > 1e-6, np.abs(value / (lag_3 + 1e-8)), 1.0)
    df_enhanced['lag_3_ratio'] = np.clip(lag_3_ratio, 0.01, 100.0)
    
    # 3. Rolling Z-score
    rolling_zscore = np.where(rolling_std > 1e-6, (value - rolling_mean) / (rolling_std + 1e-8), 0.0)
    df_enhanced['rolling_zscore_5'] = np.clip(rolling_zscore, -10.0, 10.0)
    
    # 4. Velocity sign change
    velocity_sign_change = np.zeros_like(velocity)
    velocity_sign_change[1:] = (velocity[:-1] * velocity[1:] < 0).astype(np.float32)
    df_enhanced['velocity_sign_change'] = velocity_sign_change
    
    # 5. Rolling max
    df_enhanced['rolling_max_5'] = pd.Series(value).rolling(5, min_periods=1).max().values
    
    # 6. Rolling min
    df_enhanced['rolling_min_5'] = pd.Series(value).rolling(5, min_periods=1).min().values
    
    return df_enhanced

print("   Engineering lag/z-score features for train split...")
X_train_enhanced = engineer_enhanced_features(train_df, base_feature_cols)
y_train = train_df['label_is_anomaly'].values

print("   Engineering lag/z-score features for val split...")
X_val_enhanced = engineer_enhanced_features(val_df, base_feature_cols)
y_val = val_df['label_is_anomaly'].values

print("   Engineering lag/z-score features for test split...")
X_test_enhanced = engineer_enhanced_features(test_df, base_feature_cols)
y_test = test_df['label_is_anomaly'].values

# Convert to numpy
X_train = X_train_enhanced.values
X_val = X_val_enhanced.values
X_test = X_test_enhanced.values

feature_cols_enhanced = list(X_train_enhanced.columns)

print(f"\n   Total features: {len(feature_cols_enhanced)}")
print(f"   Base features: {len(base_feature_cols)}")
print(f"   New lag/z-score features: {len(feature_cols_enhanced) - len(base_feature_cols)}")
print(f"   New feature columns: {feature_cols_enhanced[-6:]}")

# 2. PREPARE LIGHTGBM DATASETS
print("\n2. Preparing LightGBM datasets...")
train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols_enhanced)
val_data = lgb.Dataset(X_val, label=y_val, feature_name=feature_cols_enhanced, reference=train_data)

# 3. TRAIN TUNED LIGHTGBM
print("\n3. Training tuned LightGBM with enhanced features...")

scale_pos_weight = float((1 - y_train.mean()) / y_train.mean())

params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'num_leaves': 127,
    'learning_rate': 0.04,
    'feature_fraction': 0.85,
    'bagging_fraction': 0.85,
    'bagging_freq': 5,
    'lambda_l1': 0.5,
    'lambda_l2': 0.5,
    'verbose': -1,
    'scale_pos_weight': scale_pos_weight,
    'n_jobs': 4,
    'random_state': 42
}

print(f"   Scale pos weight: {scale_pos_weight:.2f}")
print(f"   Num leaves: {params['num_leaves']}")
print(f"   Learning rate: {params['learning_rate']}")

model_tuned = lgb.train(
    params,
    train_data,
    num_boost_round=600,
    valid_sets=[train_data, val_data],
    valid_names=['train', 'val'],
    callbacks=[
        lgb.early_stopping(stopping_rounds=30, verbose=False),
        lgb.log_evaluation(period=50)
    ]
)

print(f"   [OK] Model trained ({model_tuned.num_trees()} trees)")

# 4. MAKE PREDICTIONS
print("\n4. Making predictions...")
y_pred_val_proba = model_tuned.predict(X_val)
y_pred_test_proba = model_tuned.predict(X_test)

# 5. THRESHOLD OPTIMIZATION
print("\n5. Optimizing classification threshold on validation set...")
thresholds = np.linspace(0.1, 0.9, 200)
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
print("\n7. Top 15 Most Important Features (Split Gain):")
importance_gain = model_tuned.feature_importance(importance_type='gain')
feature_importance = pd.DataFrame({
    'feature': feature_cols_enhanced,
    'importance': importance_gain
}).sort_values('importance', ascending=False)

for idx, row in feature_importance.head(15).iterrows():
    print(f"      {row['feature']:30s} : {row['importance']:12.2f}")

# 8. MLFLOW LOGGING
print("\n8. Logging to MLflow...")
mlflow.set_experiment("phase2_anomaly_detection")
mlflow.start_run(run_name="lightgbm_tuned_enhanced_features")

mlflow.log_params({
    'model_type': 'lightgbm_tuned_classifier',
    'num_trees': model_tuned.num_trees(),
    'num_leaves': params['num_leaves'],
    'learning_rate': params['learning_rate'],
    'scale_pos_weight': scale_pos_weight,
    'feature_count': len(feature_cols_enhanced),
    'new_lag_features': 6,
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
model_path = Path(r'C:\AI-SRE\src\models\lightgbm_tuned_classifier.joblib')
model_path.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(model_tuned, model_path)
print(f"   [OK] Model saved to {model_path}")

try:
    mlflow.lightgbm.log_model(model_tuned, artifact_path='lightgbm_tuned_classifier')
except Exception as e:
    print(f"   Note: {e}")

mlflow.end_run()

print("   [OK] Logged to MLflow")

print("\n" + "=" * 80)
print("TUNED LIGHTGBM WITH ENHANCED FEATURES COMPLETE")
print("=" * 80)

# 9. SUMMARY & COMPARISON
print(f"\n{'='*80}")
print("PHASE 2 PROGRESSION:")
print(f"   Step 3 (Isolation Forest):          F1 = 0.1535")
print(f"   Step 4 (Autoencoder V1 - 4 feat):   F1 = 0.3178")
print(f"   Step 4b (Autoencoder V2 - 25 feat): F1 = 0.2880")
print(f"   Step 5 (LightGBM Base - 25 feat):   F1 = 0.7567")
print(f"   Step 5b (LightGBM Tuned - 31 feat): F1 = {test_f1:.4f}")
print(f"{'='*80}")

if test_f1 >= 0.80:
    print(f"\n[SUCCESS] PHASE 2 COMPLETE - F1 >= 0.80 ACHIEVED!")
    print(f"   Test F1: {test_f1:.4f}")
    print(f"   Test Precision: {test_prec:.4f}")
    print(f"   Test Recall: {test_rec:.4f}")
    print(f"   Test AUC: {test_auc:.4f}")
    print(f"\nREADY FOR PHASE 3 (RAG KNOWLEDGE BASE)")
else:
    print(f"\n[NOTE] Test F1 = {test_f1:.4f} (still < 0.80 target)")
print(f"{'='*80}")
