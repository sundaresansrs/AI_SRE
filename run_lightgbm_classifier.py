# Phase 2, Step 5: Supervised LightGBM Classifier Training

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, classification_report
import mlflow
import mlflow.lightgbm
import joblib

sys.stdout.reconfigure(encoding='utf-8')

print("=" * 80)
print("PHASE 2, STEP 5: SUPERVISED LIGHTGBM CLASSIFIER (25 FEATURES)")
print("=" * 80)

# 1. LOAD V2 SPLITS
print("\n1. Loading v2 feature-engineered splits...")
data_dir = Path(r'C:\AI-SRE\data\model_inputs')

train_df = pd.read_parquet(data_dir / 'train_80pct_v2_advanced_features.parquet')
val_df = pd.read_parquet(data_dir / 'val_10pct_v2_advanced_features.parquet')
test_df = pd.read_parquet(data_dir / 'test_10pct_v2_advanced_features.parquet')

feature_cols = [c for c in train_df.columns if c != 'label_is_anomaly' and c != 'service_name']

X_train = train_df[feature_cols].values
y_train = train_df['label_is_anomaly'].values

X_val = val_df[feature_cols].values
y_val = val_df['label_is_anomaly'].values

X_test = test_df[feature_cols].values
y_test = test_df['label_is_anomaly'].values

print(f"   Feature count: {len(feature_cols)}")
print(f"   Train: {X_train.shape} ({y_train.sum():,} anomalies)")
print(f"   Val:   {X_val.shape} ({y_val.sum():,} anomalies)")
print(f"   Test:  {X_test.shape} ({y_test.sum():,} anomalies)")

# 2. PREPARE LIGHTGBM DATASETS
print("\n2. Preparing LightGBM datasets...")
train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
val_data = lgb.Dataset(X_val, label=y_val, feature_name=feature_cols, reference=train_data)

# 3. TRAIN LIGHTGBM
print("\n3. Training LightGBM classifier...")

scale_pos_weight = float((1 - y_train.mean()) / y_train.mean())
print(f"   Scale pos weight (for imbalance): {scale_pos_weight:.2f}")

params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'num_leaves': 63,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'scale_pos_weight': scale_pos_weight,
    'n_jobs': 4,
    'random_state': 42
}

model = lgb.train(
    params,
    train_data,
    num_boost_round=300,
    valid_sets=[train_data, val_data],
    valid_names=['train', 'val'],
    callbacks=[
        lgb.early_stopping(stopping_rounds=25, verbose=False),
        lgb.log_evaluation(period=25)
    ]
)

print(f"   [OK] Model trained ({model.num_trees()} trees)")

# 4. MAKE PREDICTIONS
print("\n4. Making predictions on val/test...")
y_pred_val_proba = model.predict(X_val)
y_pred_test_proba = model.predict(X_test)

# Threshold optimization on validation set
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
print("\n7. Top 10 Most Important Features (Split Gain):")
importance_gain = model.feature_importance(importance_type='gain')
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': importance_gain
}).sort_values('importance', ascending=False)

for idx, row in feature_importance.head(10).iterrows():
    print(f"      {row['feature']:30s} : {row['importance']:12.2f}")

# 8. MLFLOW LOGGING
print("\n8. Logging to MLflow...")
mlflow.set_experiment("phase2_anomaly_detection")
mlflow.start_run(run_name="lightgbm_supervised_classifier")

mlflow.log_params({
    'model_type': 'lightgbm_classifier',
    'num_trees': model.num_trees(),
    'num_leaves': params['num_leaves'],
    'learning_rate': params['learning_rate'],
    'scale_pos_weight': scale_pos_weight,
    'train_samples': len(X_train),
    'val_samples': len(X_val),
    'test_samples': len(X_test),
    'feature_count': len(feature_cols)
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
model_path = Path(r'C:\AI-SRE\src\models\lightgbm_classifier.joblib')
model_path.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(model, model_path)
print(f"   [OK] Model saved to {model_path}")

try:
    mlflow.lightgbm.log_model(model, artifact_path='lightgbm_classifier')
except Exception as e:
    print(f"   Note on MLflow lightgbm log_model: {e}")

mlflow.end_run()

print("   [OK] Logged to MLflow")

print("\n" + "=" * 80)
print("LIGHTGBM TRAINING COMPLETE")
print("=" * 80)

# 9. SUMMARY
print(f"\n{'='*80}")
if test_f1 >= 0.80:
    print(f"[OK] SUCCESS! LIGHTGBM ACHIEVED F1 >= 0.80!")
    print(f"   Test F1: {test_f1:.4f}")
    print(f"   Test Precision: {test_prec:.4f}")
    print(f"   Test Recall: {test_rec:.4f}")
    print(f"   Test AUC: {test_auc:.4f}")
    print(f"\nPHASE 2 (ANOMALY DETECTION MODEL) COMPLETE")
else:
    print(f"[NOTE] Test F1 = {test_f1:.4f} (< 0.80 target)")
    print(f"   Further hyperparameter tuning or feature engineering may be needed")
print(f"{'='*80}")
