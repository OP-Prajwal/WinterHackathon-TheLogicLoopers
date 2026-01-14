import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from poison_guard.models.encoders.tabular_mlp import TabularMLPEncoder
from poison_guard.models.heads.mlp import ProjectionHead
from poison_guard.pipeline.trainer import PoisonGuardTrainer
from poison_guard.models.losses import NTXentLoss

# --- Configuration ---
DATA_PATH = "data/diabetes_brfss.csv"
BATCH_SIZE = 64
EPOCHS = 1
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")

# --- Data Loading ---
print("Loading data (subset)...")
df = pd.read_csv(DATA_PATH, nrows=2000) # Only load 2000 rows for speed
print(f"Data shape: {df.shape}")

# Preprocessing
if 'Diabetes_binary' in df.columns:
    X = df.drop('Diabetes_binary', axis=1).values
else:
    X = df.values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

class ContrastiveDataset(Dataset):
    def __init__(self, data):
        self.data = torch.FloatTensor(data)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        noise1 = torch.randn_like(x) * 0.1
        noise2 = torch.randn_like(x) * 0.1
        return x + noise1, x + noise2

dataset = ContrastiveDataset(X_scaled)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

# --- Model Intialization ---
INPUT_DIM = X.shape[1] 
HIDDEN_DIM = 256
OUTPUT_DIM = 64

encoder = TabularMLPEncoder(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM).to(DEVICE)
head = ProjectionHead(input_dim=OUTPUT_DIM, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM).to(DEVICE)

loss_fn = NTXentLoss(temperature=0.5).to(DEVICE)
optimizer = torch.optim.Adam(list(encoder.parameters()) + list(head.parameters()), lr=LR)

trainer = PoisonGuardTrainer(encoder, head, optimizer, loss_fn)

# --- Training Loop ---
print("Starting fast training...")
trainer.encoder.train()
trainer.head.train()

for epoch in range(EPOCHS):
    total_loss = 0
    steps = 0
    for x1, x2 in dataloader:
        x1, x2 = x1.to(DEVICE), x2.to(DEVICE)
        loss = trainer.train_step(x1, x2)
        total_loss += loss
        steps += 1
            
    avg_loss = total_loss / max(1, steps)
    print(f"Epoch {epoch+1} Complete. Avg Loss: {avg_loss:.4f}")

# --- Save Model ---
checkpoint = {
    'encoder': encoder.state_dict(),
    'head': head.state_dict(),
    'input_dim': INPUT_DIM,
    'hidden_dim': HIDDEN_DIM,
    'output_dim': OUTPUT_DIM
}
torch.save(checkpoint, "trained_model_v2.pt")
print("Model saved to trained_model_v2.pt")
