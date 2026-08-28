import sys
import pandas as pd
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import mlflow
import mlflow.pytorch
import time

sys.stdout.reconfigure(encoding='utf-8')
torch.set_num_threads(4)

print("=" * 80)
print("PHASE 2, STEP 4c: AUTOENCODER RETRAINING (25 ADVANCED FEATURES)")
print("=" * 80)

# 1. LOAD V2 SPLITS (WITH ADVANCED FEATURES)
print("\n1. Loading v2 feature-engineered splits...")
data_dir = Path(r'C:\AI-SRE\data\model_inputs')

train_df = pd.read_parquet(data_dir / 'train_80pct_v2_advanced_features.parquet')
val_df = pd.read_parquet(data_dir / 'val_10pct_v2_advanced_features.parquet')
test_df = pd.read_parquet(data_dir / 'test_10pct_v2_advanced_features.parquet')

feature_cols = [c for c in train_df.columns if c != 'label_is_anomaly' and c != 'service_name']

X_train = train_df[feature_cols].values.astype(np.float32)
y_train = train_df['label_is_anomaly'].values.astype(np.int32)

X_val = val_df[feature_cols].values.astype(np.float32)
y_val = val_df['label_is_anomaly'].values.astype(np.int32)

X_test = test_df[feature_cols].values.astype(np.float32)
y_test = test_df['label_is_anomaly'].values.astype(np.int32)

print(f"   Feature count: {len(feature_cols)}")
print(f"   Train: {X_train.shape}")
print(f"   Val:   {X_val.shape}")
print(f"   Test:  {X_test.shape}")

# 2. PYTORCH SETUP
print("\n2. Setting up PyTorch...")
device = torch.device('cpu')
print(f"   Device: {device}")

X_train_t = torch.from_numpy(X_train).to(device)
X_val_t = torch.from_numpy(X_val).to(device)
X_test_t = torch.from_numpy(X_test).to(device)

train_dataset = TensorDataset(X_train_t)
train_loader = DataLoader(train_dataset, batch_size=4096, shuffle=True)

# 3. DEFINE LARGER AUTOENCODER FOR 25 FEATURES
print("\n3. Defining deep autoencoder architecture...")

class DeepAutoencoder(nn.Module):
    def __init__(self, input_dim=25, latent_dim=16):
        super(DeepAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, input_dim)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

model = DeepAutoencoder(input_dim=len(feature_cols), latent_dim=16).to(device)
print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# 4. TRAINING SETUP
print("\n4. Training setup...")
optimizer = optim.Adam(model.parameters(), lr=0.0015)
criterion = nn.MSELoss()

num_epochs = 50
best_val_loss = float('inf')
patience = 6
patience_counter = 0
best_model_path = Path(r'C:\AI-SRE\src\models\autoencoder_v2_advanced.pth')
best_model_path.parent.mkdir(parents=True, exist_ok=True)

# 5. MLFLOW SETUP
print("\n5. Configuring MLflow...")
mlflow.set_experiment("phase2_anomaly_detection")
mlflow.start_run(run_name="autoencoder_v2_advanced_features")

# 6. TRAIN LOOP
print("\n6. Training deep autoencoder...")
train_losses = []
val_losses = []
t0 = time.time()

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for (batch_X,) in train_loader:
        batch_X = batch_X.to(device)
        optimizer.zero_grad()
        reconstructed = model(batch_X)
        loss = criterion(reconstructed, batch_X)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    train_losses.append(train_loss)
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_reconstructed = model(X_val_t)
        val_loss = criterion(val_reconstructed, X_val_t).item()
    
    val_losses.append(val_loss)
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), best_model_path)
    else:
        patience_counter += 1
    
    if (epoch + 1) % 5 == 0 or (epoch + 1) == num_epochs:
        elapsed = time.time() - t0
        print(f"   Epoch {epoch + 1:2d}/{num_epochs} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | Elapsed: {elapsed:.1f}s", flush=True)
    
    if patience_counter >= patience:
        print(f"   Early stopping at epoch {epoch + 1} (patience={patience})", flush=True)
        break

# 7. LOAD BEST MODEL & COMPUTE RECONSTRUCTION ERRORS
print("\n7. Computing reconstruction errors on val/test...")
model.load_state_dict(torch.load(best_model_path))
model.eval()

with torch.no_grad():
    val_reconstructed = model(X_val_t)
    val_errors = torch.mean((X_val_t - val_reconstructed) ** 2, dim=1).cpu().numpy()
    
    test_reconstructed = model(X_test_t)
    test_errors = torch.mean((X_test_t - test_reconstructed) ** 2, dim=1).cpu().numpy()

# 8. THRESHOLD OPTIMIZATION ON VALIDATION
print("\n8. Optimizing anomaly threshold on validation set...")
percentiles = np.linspace(50, 99.9, 500)
candidate_thresholds = np.percentile(val_errors, percentiles)

best_f1 = 0.0
best_threshold = 0.0

for threshold in candidate_thresholds:
    y_pred = (val_errors > threshold).astype(int)
    f1 = f1_score(y_val, y_pred, zero_division=0)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"   Best threshold: {best_threshold:.6f}")
print(f"   Best F1 on validation: {best_f1:.4f}")

# 9. EVALUATE ON TEST SET
print("\n9. Evaluating on test set...")
y_pred_test = (test_errors > best_threshold).astype(int)

def compute_metrics(y_true, y_pred, y_scores=None):
    f1 = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    auc = None
    if y_scores is not None:
        try:
            auc = float(roc_auc_score(y_true, y_scores))
        except:
            auc = None
    return {'f1': f1, 'precision': prec, 'recall': rec, 'auc': auc}

metrics_test = compute_metrics(y_test, y_pred_test, test_errors)
metrics_val_final = compute_metrics(y_val, (val_errors > best_threshold).astype(int), val_errors)

print(f"\n   VALIDATION (threshold tuning):")
print(f"      F1:        {metrics_val_final['f1']:.4f}")
print(f"      Precision: {metrics_val_final['precision']:.4f}")
print(f"      Recall:    {metrics_val_final['recall']:.4f}")

print(f"\n   TEST (held-out evaluation):")
print(f"      F1:        {metrics_test['f1']:.4f}")
print(f"      Precision: {metrics_test['precision']:.4f}")
print(f"      Recall:    {metrics_test['recall']:.4f}")
test_auc_str = f"{metrics_test['auc']:.4f}" if metrics_test['auc'] is not None else "N/A"
print(f"      AUC:       {test_auc_str}")

# 10. LOG TO MLFLOW
print("\n10. Logging to MLflow...")
mlflow.log_params({
    'model_type': 'deep_autoencoder_v2',
    'input_dim': len(feature_cols),
    'latent_dim': 16,
    'encoder_arch': '25->64->32->16',
    'decoder_arch': '16->32->64->25',
    'batch_size': 4096,
    'learning_rate': 0.0015,
    'num_epochs_trained': len(train_losses),
    'train_samples': len(X_train),
    'val_samples': len(X_val),
    'test_samples': len(X_test)
})

mlflow.log_metrics({
    'val_f1': metrics_val_final['f1'],
    'val_precision': metrics_val_final['precision'],
    'val_recall': metrics_val_final['recall'],
    'test_f1': metrics_test['f1'],
    'test_precision': metrics_test['precision'],
    'test_recall': metrics_test['recall'],
    'test_auc': metrics_test['auc'] if metrics_test['auc'] else 0.0,
    'best_threshold': float(best_threshold),
    'final_val_loss': float(val_losses[-1])
})

try:
    mlflow.pytorch.log_model(model, artifact_path='pytorch_autoencoder_v2')
except Exception as e:
    print(f"   Note on MLflow pytorch log_model: {e}")

mlflow.end_run()

print("   [OK] Logged to MLflow")

print("\n" + "=" * 80)
print("AUTOENCODER V2 (ADVANCED FEATURES) TRAINING COMPLETE")
print("=" * 80)

# 11. SUMMARY & RESULT
print(f"\n{'='*80}")
if metrics_test['f1'] >= 0.80:
    print(f"[OK] SUCCESS! AUTOENCODER V2 ACHIEVED F1 >= 0.80!")
    print(f"   Test F1: {metrics_test['f1']:.4f}")
    print(f"   Test Precision: {metrics_test['precision']:.4f}")
    print(f"   Test Recall: {metrics_test['recall']:.4f}")
    print(f"   Test AUC: {metrics_test['auc']:.4f}")
    print(f"\nPHASE 2 (ANOMALY DETECTION MODEL) COMPLETE")
else:
    print(f"[NOTE] Test F1 = {metrics_test['f1']:.4f} (< 0.80 target)")
    print(f"   Consider further feature engineering or ensemble methods")
print(f"{'='*80}")
