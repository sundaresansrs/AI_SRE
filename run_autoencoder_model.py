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
import joblib

# Ensure UTF-8
sys.stdout.reconfigure(encoding='utf-8')

print("=" * 80)
print("PHASE 2, STEP 4: PYTORCH AUTOENCODER TRAINING")
print("=" * 80)

# 1. LOAD SPLITS
print("\n1. Loading train/val/test splits...")
data_dir = Path(r'C:\AI-SRE\data\model_inputs')

train_df = pd.read_parquet(data_dir / 'train_80pct.parquet')
val_df = pd.read_parquet(data_dir / 'val_10pct.parquet')
test_df = pd.read_parquet(data_dir / 'test_10pct.parquet')

feature_cols = [c for c in train_df.columns if c.startswith('feature_')]
print(f"   Feature columns: {feature_cols}")

X_train = train_df[feature_cols].values.astype(np.float32)
y_train = train_df['label_is_anomaly'].values.astype(np.int32)

X_val = val_df[feature_cols].values.astype(np.float32)
y_val = val_df['label_is_anomaly'].values.astype(np.int32)

X_test = test_df[feature_cols].values.astype(np.float32)
y_test = test_df['label_is_anomaly'].values.astype(np.int32)

print(f"   Train: {X_train.shape}")
print(f"   Val:   {X_val.shape}")
print(f"   Test:  {X_test.shape}")

# 2. PYTORCH SETUP
print("\n2. Setting up PyTorch...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"   Device: {device}")

# Convert to tensors
X_train_t = torch.from_numpy(X_train).to(device)
y_train_t = torch.from_numpy(y_train).to(device)
X_val_t = torch.from_numpy(X_val).to(device)
y_val_t = torch.from_numpy(y_val).to(device)
X_test_t = torch.from_numpy(X_test).to(device)
y_test_t = torch.from_numpy(y_test).to(device)

# DataLoader for batching (batch_size 2048 for high throughput CPU execution)
train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True)

# 3. DEFINE AUTOENCODER
print("\n3. Defining autoencoder architecture...")

class Autoencoder(nn.Module):
    def __init__(self, input_dim=4, latent_dim=4):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, latent_dim),
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        return self.encoder(x)

model = Autoencoder(input_dim=4, latent_dim=4).to(device)
print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# 4. TRAINING SETUP
print("\n4. Training setup...")
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

num_epochs = 50
best_val_loss = float('inf')
patience = 5
patience_counter = 0
best_model_path = Path(r'C:\AI-SRE\src\models\autoencoder_best.pth')
best_model_path.parent.mkdir(parents=True, exist_ok=True)

# 5. MLFLOW SETUP
print("\n5. Configuring MLflow...")
mlflow.set_experiment("phase2_anomaly_detection")
mlflow.start_run(run_name="pytorch_autoencoder_v1")

# 6. TRAIN LOOP
print("\n6. Training autoencoder...")
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0.0
    for batch_X, _ in train_loader:
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
    
    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), best_model_path)
    else:
        patience_counter += 1
    
    if (epoch + 1) % 5 == 0 or (epoch + 1) == num_epochs or epoch == 0:
        print(f"   Epoch {epoch + 1:3d} / {num_epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}", flush=True)
    
    if patience_counter >= patience:
        print(f"   Early stopping triggered at epoch {epoch + 1} (patience={patience})", flush=True)
        break

# 7. LOAD BEST MODEL & COMPUTE RECONSTRUCTION ERROR
print("\n7. Computing reconstruction errors with best checkpoint...")
model.load_state_dict(torch.load(best_model_path))
model.eval()

with torch.no_grad():
    val_reconstructed = model(X_val_t)
    val_errors = torch.mean((X_val_t - val_reconstructed) ** 2, dim=1).cpu().numpy()
    
    test_reconstructed = model(X_test_t)
    test_errors = torch.mean((X_test_t - test_reconstructed) ** 2, dim=1).cpu().numpy()

# 8. THRESHOLD OPTIMIZATION ON VALIDATION SET
print("\n8. Optimizing anomaly detection threshold on validation set...")
# Search percentiles for robust thresholding
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
        except Exception:
            auc = None
    return {'f1': float(f1), 'precision': float(prec), 'recall': float(rec), 'auc': auc}

metrics_test = compute_metrics(y_test, y_pred_test, test_errors)
metrics_val_final = compute_metrics(y_val, (val_errors > best_threshold).astype(int), val_errors)

print(f"\n   VALIDATION (threshold tuning):")
print(f"      F1:        {metrics_val_final['f1']:.4f}")
print(f"      Precision: {metrics_val_final['precision']:.4f}")
print(f"      Recall:    {metrics_val_final['recall']:.4f}")
val_auc_str = f"{metrics_val_final['auc']:.4f}" if metrics_val_final['auc'] is not None else "N/A"
print(f"      AUC:       {val_auc_str}")

print(f"\n   TEST (held-out evaluation):")
print(f"      F1:        {metrics_test['f1']:.4f}")
print(f"      Precision: {metrics_test['precision']:.4f}")
print(f"      Recall:    {metrics_test['recall']:.4f}")
test_auc_str = f"{metrics_test['auc']:.4f}" if metrics_test['auc'] is not None else "N/A"
print(f"      AUC:       {test_auc_str}")

# 10. LOG TO MLFLOW
print("\n10. Logging to MLflow...")
mlflow.log_params({
    'model_type': 'pytorch_autoencoder',
    'input_dim': 4,
    'latent_dim': 4,
    'encoder_layers': '4->16->8->4',
    'batch_size': 2048,
    'learning_rate': 0.001,
    'num_epochs_trained': len(train_losses),
    'train_samples': len(X_train),
    'val_samples': len(X_val),
    'test_samples': len(X_test)
})

mlflow.log_metrics({
    'val_f1': metrics_val_final['f1'],
    'val_precision': metrics_val_final['precision'],
    'val_recall': metrics_val_final['recall'],
    'val_auc': metrics_val_final['auc'] if metrics_val_final['auc'] is not None else 0.0,
    'test_f1': metrics_test['f1'],
    'test_precision': metrics_test['precision'],
    'test_recall': metrics_test['recall'],
    'test_auc': metrics_test['auc'] if metrics_test['auc'] is not None else 0.0,
    'best_threshold': float(best_threshold),
    'final_val_loss': float(val_losses[-1])
})

try:
    mlflow.pytorch.log_model(model, artifact_path='pytorch_autoencoder_model')
except Exception as e:
    print(f"   Note on MLflow pytorch log_model: {e}")

mlflow.end_run()

print("   [OK] Logged to MLflow")
print(f"   [OK] Model weights saved to {best_model_path}")

print("\n" + "=" * 80)
print("AUTOENCODER TRAINING COMPLETE")
print("=" * 80)

# 11. SUMMARY
print(f"\n{'='*80}")
if metrics_test['f1'] >= 0.80:
    print(f"[OK] AUTOENCODER ACHIEVED F1 >= 0.80 on test set! (F1 = {metrics_test['f1']:.4f})")
else:
    print(f"[NOTE] AUTOENCODER F1 = {metrics_test['f1']:.4f} (< 0.80 target)")
    print("    Further hyperparameter tuning, sequence windowing, or supervised classifier refinement recommended.")
print(f"{'='*80}")
