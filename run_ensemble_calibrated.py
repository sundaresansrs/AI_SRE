# Phase 2, Step 5d: Probability Calibration + XGBoost Ensemble for F1 >= 0.80

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.calibration import IsotonicRegression
import mlflow
import joblib
import warnings
warnings.filterwarnings('ignore')

sys.stdout.reconfigure(encoding='utf-8')

print("=" * 80)
print("PHASE 2, STEP 5d: PROBABILITY CALIBRATION + XGBOOST ENSEMBLE")
print("=" * 80)

# 1. LOAD FINAL FEATURES & TRAINED LIGHTGBM
print("\n1. Loading 37-feature dataset and trained LightGBM...")
data_dir = Path(r'C:\AI-SRE\data\model_inputs')

train_df = pd.read_parquet(data_dir / 'train_80pct_v2_advanced_features.parquet')
val_df = pd.read_parquet(data_dir / 'val_10pct_v2_advanced_features.parquet')
test_df = pd.read_parquet(data_dir / 'test_10pct_v2_advanced_features.parquet')

base_feature_cols = [c for c in train_df.columns if c != 'label_is_anomaly' and c != 'service_name']

# Fast vectorized 37-feature engineering function
def engineer_all_features(df, base_cols):
    df_enhanced = df[base_cols].copy()
    value = df['feature_0_raw_value'].values
    rolling_mean = df['rolling_mean_5'].values
    rolling_std = df['rolling_std_5'].values
    velocity = df['velocity'].values
    s_val = pd.Series(value)
    
    # Lag/Z-score features
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
    
    # Multi-scale rolling features
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
    
    return df_enhanced

print("   Engineering 37 features for train/val/test...")
X_train_full = engineer_all_features(train_df, base_feature_cols).values
y_train = train_df['label_is_anomaly'].values

X_val_full = engineer_all_features(val_df, base_feature_cols).values
y_val = val_df['label_is_anomaly'].values

X_test_full = engineer_all_features(test_df, base_feature_cols).values
y_test = test_df['label_is_anomaly'].values

feature_cols_final = list(engineer_all_features(train_df, base_feature_cols).columns)
print(f"   Total features: {len(feature_cols_final)}")

# 2. LOAD PRE-TRAINED LIGHTGBM
print("\n2. Loading pre-trained LightGBM model...")
lgb_model = joblib.load(Path(r'C:\AI-SRE\src\models\lightgbm_final_classifier.joblib'))
print("   [OK] LightGBM loaded")

# 3. GET LIGHTGBM RAW PROBABILITIES
print("\n3. Computing LightGBM raw probabilities...")
lgb_proba_train = lgb_model.predict(X_train_full)
lgb_proba_val = lgb_model.predict(X_val_full)
lgb_proba_test = lgb_model.predict(X_test_full)

print(f"   Train proba: min={lgb_proba_train.min():.4f}, max={lgb_proba_train.max():.4f}, mean={lgb_proba_train.mean():.4f}")

# 4. CALIBRATE LIGHTGBM PROBABILITIES (Isotonic Calibration on Validation Set)
print("\n4. Calibrating LightGBM probabilities (Isotonic Regression on validation set)...")
calibrator = IsotonicRegression(out_of_bounds='clip')
calibrator.fit(lgb_proba_val, y_val)

lgb_proba_train_cal = calibrator.predict(lgb_proba_train)
lgb_proba_val_cal = calibrator.predict(lgb_proba_val)
lgb_proba_test_cal = calibrator.predict(lgb_proba_test)

print(f"   Calibrated val: min={lgb_proba_val_cal.min():.4f}, max={lgb_proba_val_cal.max():.4f}")

# 5. TRAIN XGBOOST ON SAME 37 FEATURES
print("\n5. Training XGBoost classifier...")
scale_pos_weight = float((1 - y_train.mean()) / y_train.mean())

xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': 8,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'scale_pos_weight': scale_pos_weight,
    'random_state': 42,
    'n_jobs': 4,
    'verbosity': 0
}

dtrain = xgb.DMatrix(X_train_full, label=y_train)
dval = xgb.DMatrix(X_val_full, label=y_val)
dtest = xgb.DMatrix(X_test_full, label=y_test)

xgb_model = xgb.train(
    xgb_params,
    dtrain,
    num_boost_round=400,
    evals=[(dtrain, 'train'), (dval, 'val')],
    early_stopping_rounds=30,
    verbose_eval=50
)

print(f"   [OK] XGBoost trained ({xgb_model.num_boosted_rounds()} rounds)")

# 6. GET XGBOOST PROBABILITIES
print("\n6. Computing XGBoost probabilities...")
xgb_proba_train = xgb_model.predict(dtrain)
xgb_proba_val = xgb_model.predict(dval)
xgb_proba_test = xgb_model.predict(dtest)

# 7. ENSEMBLE: AVERAGE CALIBRATED LGBM + XGBOOST
print("\n7. Ensemble: Averaging calibrated LightGBM + XGBoost...")
ensemble_proba_val = (lgb_proba_val_cal + xgb_proba_val) / 2.0
ensemble_proba_test = (lgb_proba_test_cal + xgb_proba_test) / 2.0

# 8. OPTIMIZE THRESHOLD ON VALIDATION SET
print("\n8. Optimizing ensemble threshold on validation set...")
thresholds = np.linspace(0.05, 0.95, 300)
best_f1 = 0.0
best_threshold = 0.5

for threshold in thresholds:
    y_pred = (ensemble_proba_val > threshold).astype(int)
    f1 = f1_score(y_val, y_pred, zero_division=0)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"   Best threshold: {best_threshold:.3f}")
print(f"   Best F1 on validation: {best_f1:.4f}")

# 9. EVALUATE ENSEMBLE ON TEST SET
print("\n9. Evaluating ensemble on test set...")
y_pred_test = (ensemble_proba_test > best_threshold).astype(int)

test_f1 = float(f1_score(y_test, y_pred_test, zero_division=0))
test_prec = float(precision_score(y_test, y_pred_test, zero_division=0))
test_rec = float(recall_score(y_test, y_pred_test, zero_division=0))
test_auc = float(roc_auc_score(y_test, ensemble_proba_test))

print(f"\n   ENSEMBLE TEST SET EVALUATION:")
print(f"      F1-Score:  {test_f1:.4f}")
print(f"      Precision: {test_prec:.4f}")
print(f"      Recall:    {test_rec:.4f}")
print(f"      ROC-AUC:   {test_auc:.4f}")

# 10. COMPARISON: LIGHTGBM ONLY vs ENSEMBLE
print("\n10. Comparison: LightGBM Alone vs Calibrated + XGBoost Ensemble")
print(f"{'='*70}")

# LightGBM alone (from Step 5c)
lgb_thresholds = np.linspace(0.05, 0.95, 300)
lgb_best_f1 = 0.0
lgb_best_threshold = 0.5
for threshold in lgb_thresholds:
    y_pred = (lgb_proba_test_cal > threshold).astype(int)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    if f1 > lgb_best_f1:
        lgb_best_f1 = f1
        lgb_best_threshold = threshold

y_pred_lgb = (lgb_proba_test_cal > lgb_best_threshold).astype(int)
lgb_f1 = float(f1_score(y_test, y_pred_lgb, zero_division=0))
lgb_prec = float(precision_score(y_test, y_pred_lgb, zero_division=0))
lgb_rec = float(recall_score(y_test, y_pred_lgb, zero_division=0))
lgb_auc = float(roc_auc_score(y_test, lgb_proba_test_cal))

print(f"   LightGBM Only (Calibrated):")
print(f"      F1:        {lgb_f1:.4f}")
print(f"      Precision: {lgb_prec:.4f}")
print(f"      Recall:    {lgb_rec:.4f}")
print(f"      AUC:       {lgb_auc:.4f}")

print(f"\n   Ensemble (Calibrated LightGBM + XGBoost):")
print(f"      F1:        {test_f1:.4f}  (Gain: +{test_f1 - lgb_f1:.4f})")
print(f"      Precision: {test_prec:.4f}  (Gain: +{test_prec - lgb_prec:.4f})")
print(f"      Recall:    {test_rec:.4f}  (Gain: +{test_rec - lgb_rec:.4f})")
print(f"      AUC:       {test_auc:.4f}  (Gain: +{test_auc - lgb_auc:.4f})")
print(f"{'='*70}")

# 11. SAVE ENSEMBLE MODELS
print("\n11. Saving calibrated models and ensemble metadata...")
calibrator_path = Path(r'C:\AI-SRE\src\models\probability_calibrator_isotonic.joblib')
joblib.dump(calibrator, calibrator_path)
print(f"   [OK] Calibrator saved: {calibrator_path}")

xgb_model_path = Path(r'C:\AI-SRE\src\models\xgboost_ensemble.joblib')
joblib.dump(xgb_model, xgb_model_path)
print(f"   [OK] XGBoost model saved: {xgb_model_path}")

ensemble_config = {
    'ensemble_method': 'average',
    'lgb_weight': 0.5,
    'xgb_weight': 0.5,
    'calibration_method': 'isotonic',
    'optimal_threshold': float(best_threshold),
    'test_f1': test_f1,
    'test_precision': test_prec,
    'test_recall': test_rec,
    'test_auc': test_auc
}
config_path = Path(r'C:\AI-SRE\src\models\ensemble_config.joblib')
joblib.dump(ensemble_config, config_path)
print(f"   [OK] Ensemble config saved: {config_path}")

# 12. MLFLOW LOGGING
print("\n12. Logging to MLflow...")
mlflow.set_experiment("phase2_anomaly_detection")
mlflow.start_run(run_name="ensemble_calibrated_final")

mlflow.log_params({
    'model_type': 'ensemble_calibrated',
    'ensemble_method': 'average_lgb_xgb',
    'lgb_calibration': 'isotonic',
    'xgb_num_rounds': xgb_model.num_boosted_rounds(),
    'feature_count': len(feature_cols_final),
    'scale_pos_weight': scale_pos_weight
})

mlflow.log_metrics({
    'lgb_calibrated_f1': lgb_f1,
    'ensemble_test_f1': test_f1,
    'ensemble_test_precision': test_prec,
    'ensemble_test_recall': test_rec,
    'ensemble_test_auc': test_auc,
    'f1_gain_vs_lgb': test_f1 - lgb_f1,
    'best_threshold': float(best_threshold)
})

mlflow.end_run()

print("   [OK] Logged to MLflow")

print("\n" + "=" * 80)
print("PHASE 2 FINAL: CALIBRATION + ENSEMBLE COMPLETE")
print("=" * 80)

# 13. FINAL VERDICT
print(f"\n{'='*80}")
if test_f1 >= 0.80:
    print(f"[SUCCESS] PHASE 2 TARGET ACHIEVED: F1 >= 0.80")
    print(f"\n   FINAL ENSEMBLE RESULTS:")
    print(f"   [OK] Test F1:        {test_f1:.4f}")
    print(f"   [OK] Test Precision: {test_prec:.4f}")
    print(f"   [OK] Test Recall:    {test_rec:.4f}")
    print(f"   [OK] Test AUC:       {test_auc:.4f}")
    print(f"\n   Ensemble: Isotonic-Calibrated LightGBM + XGBoost (Average)")
    print(f"   Features: {len(feature_cols_final)} temporal + service multi-scale features")
    print(f"   Threshold: {best_threshold:.3f}")
    print(f"\nREADY TO BEGIN PHASE 3 (RAG KNOWLEDGE BASE)")
else:
    print(f"[NOTE] Test F1 = {test_f1:.4f}")
    print(f"   Gain from calibration+ensemble: +{test_f1 - 0.7757:.4f}")
    print(f"   Remaining gap to 0.80: {0.80 - test_f1:.4f}")
print(f"{'='*80}")
