import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
import joblib
import mlflow
import mlflow.sklearn

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

print("=" * 80)
print("PHASE 2, STEP 3: BASELINE MODEL TRAINING (ISOLATION FOREST)")
print("=" * 80)

# 1. LOAD SPLITS
print("\n1. Loading train/val/test splits...")
data_dir = Path(r'C:\AI-SRE\data\model_inputs')

train_df = pd.read_parquet(data_dir / 'train_80pct.parquet')
val_df = pd.read_parquet(data_dir / 'val_10pct.parquet')
test_df = pd.read_parquet(data_dir / 'test_10pct.parquet')

feature_cols = [c for c in train_df.columns if c.startswith('feature_')]
print(f"   Feature columns: {feature_cols}")

X_train = train_df[feature_cols].values
y_train = train_df['label_is_anomaly'].values

X_val = val_df[feature_cols].values
y_val = val_df['label_is_anomaly'].values

X_test = test_df[feature_cols].values
y_test = test_df['label_is_anomaly'].values

print(f"   Train: {X_train.shape} with {y_train.sum():,} anomalies")
print(f"   Val:   {X_val.shape} with {y_val.sum():,} anomalies")
print(f"   Test:  {X_test.shape} with {y_test.sum():,} anomalies")

# 2. MLFLOW SETUP
print("\n2. Configuring MLflow...")
mlflow.set_experiment("phase2_anomaly_detection")
mlflow.start_run(run_name="isolation_forest_baseline")

# 3. TRAIN ISOLATION FOREST
print("\n3. Training Isolation Forest...")
model = IsolationForest(
    n_estimators=100,
    contamination=0.2271,  # Matches observed anomaly rate
    random_state=42,
    n_jobs=-1
)
model.fit(X_train)
print("   [OK] Model trained on 1.6M training samples")

# 4. MAKE PREDICTIONS
print("\n4. Making predictions...")
y_pred_train_raw = model.predict(X_train)
y_pred_train = (y_pred_train_raw == -1).astype(int)

y_pred_val_raw = model.predict(X_val)
y_pred_val = (y_pred_val_raw == -1).astype(int)

y_pred_test_raw = model.predict(X_test)
y_pred_test = (y_pred_test_raw == -1).astype(int)

y_scores_val = -model.score_samples(X_val)
y_scores_test = -model.score_samples(X_test)

# 5. EVALUATE
print("\n5. Computing metrics...")

def compute_metrics(y_true, y_pred, y_scores=None):
    f1 = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    
    auc = None
    if y_scores is not None:
        try:
            auc = float(roc_auc_score(y_true, y_scores))
        except Exception:
            auc = None
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    return {
        'f1': float(f1),
        'precision': float(prec),
        'recall': float(rec),
        'auc': auc,
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp)
    }

metrics_train = compute_metrics(y_train, y_pred_train)
metrics_val = compute_metrics(y_val, y_pred_val, y_scores_val)
metrics_test = compute_metrics(y_test, y_pred_test, y_scores_test)

print("\n   TRAIN SET:")
print(f"      F1:        {metrics_train['f1']:.4f}")
print(f"      Precision: {metrics_train['precision']:.4f}")
print(f"      Recall:    {metrics_train['recall']:.4f}")

print("\n   VALIDATION SET:")
print(f"      F1:        {metrics_val['f1']:.4f}")
print(f"      Precision: {metrics_val['precision']:.4f}")
print(f"      Recall:    {metrics_val['recall']:.4f}")
val_auc_str = f"{metrics_val['auc']:.4f}" if metrics_val['auc'] is not None else "N/A"
print(f"      AUC:       {val_auc_str}")

print("\n   TEST SET:")
print(f"      F1:        {metrics_test['f1']:.4f}")
print(f"      Precision: {metrics_test['precision']:.4f}")
print(f"      Recall:    {metrics_test['recall']:.4f}")
test_auc_str = f"{metrics_test['auc']:.4f}" if metrics_test['auc'] is not None else "N/A"
print(f"      AUC:       {test_auc_str}")

# 6. LOG TO MLFLOW
print("\n6. Logging to MLflow...")
mlflow.log_params({
    'model_type': 'isolation_forest',
    'n_estimators': 100,
    'contamination': 0.2271,
    'n_features': 4,
    'train_samples': len(X_train),
    'val_samples': len(X_val),
    'test_samples': len(X_test)
})

mlflow.log_metrics({
    'train_f1': metrics_train['f1'],
    'train_precision': metrics_train['precision'],
    'train_recall': metrics_train['recall'],
    'val_f1': metrics_val['f1'],
    'val_precision': metrics_val['precision'],
    'val_recall': metrics_val['recall'],
    'val_auc': metrics_val['auc'] if metrics_val['auc'] is not None else 0.0,
    'test_f1': metrics_test['f1'],
    'test_precision': metrics_test['precision'],
    'test_recall': metrics_test['recall'],
    'test_auc': metrics_test['auc'] if metrics_test['auc'] is not None else 0.0
})

# Save model
model_path = Path(r'C:\AI-SRE\src\models\isolation_forest_baseline.joblib')
model_path.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(model, model_path)
print(f"   [OK] Model saved to {model_path}")

try:
    mlflow.sklearn.log_model(model, artifact_path='isolation_forest_baseline')
except Exception as e:
    print(f"   Note on MLflow artifact logging: {e}")

mlflow.end_run()

print("\n" + "=" * 80)
print("BASELINE MODEL TRAINING COMPLETE")
print("=" * 80)

# 7. SUMMARY
print(f"\n{'='*80}")
if metrics_val['f1'] >= 0.80:
    print("[OK] BASELINE ACHIEVED F1 >= 0.80 on validation set!")
else:
    print(f"[NOTE] BASELINE F1 = {metrics_val['f1']:.4f} (< 0.80 target)")
    print("    Proceeding to Step 4: PyTorch Autoencoder for model improvement")
print(f"{'='*80}")
