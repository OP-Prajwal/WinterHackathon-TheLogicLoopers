import torch
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from poison_guard.models.encoders.tabular_mlp import TabularMLPEncoder

# Config
MODEL_PATH = "trained_model_v2.pt"
INPUT_DIM = 21 
HIDDEN_DIM = 256
OUTPUT_DIM = 64

# Load Model
encoder = TabularMLPEncoder(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM)
try:
    checkpoint = torch.load(MODEL_PATH, map_location='cpu')
    if 'encoder' in checkpoint:
        encoder.load_state_dict(checkpoint['encoder'])
    encoder.eval()
    print("Model loaded.")
except Exception as e:
    print(f"Failed to load model: {e}")
    sys.exit(1)

def get_norms(csv_file):
    df = pd.read_csv(csv_file)
    if 'Diabetes_binary' in df.columns:
        X = df.drop('Diabetes_binary', axis=1).values
    else:
        X = df.values
    
    # EXACT Scaling logic from server.py (Manual Reference Stats)
    if X.shape[1] > INPUT_DIM:
        X = X[:, :INPUT_DIM]
    elif X.shape[1] < INPUT_DIM:
        padding = np.zeros((X.shape[0], INPUT_DIM - X.shape[1]))
        X = np.hstack([X, padding])

    ref_means = np.array([0.5]*INPUT_DIM)
    ref_means[3] = 28.0 # BMI
    ref_means[14] = 3.0 # MentHlth
    
    ref_stds = np.array([0.5]*INPUT_DIM)
    ref_stds[3] = 6.0 
    ref_stds[14] = 5.0 
    
    X_scaled = (X - ref_means) / (ref_stds + 1e-6)
    X_tensor = torch.FloatTensor(X_scaled)
    
    with torch.no_grad():
        embeddings = encoder(X_tensor)
        norms = torch.norm(embeddings, dim=1)
        
    return norms.numpy()

print("\n--- Calibration Report ---")
try:
    safe_norms = get_norms("test_safe.csv")
    print(f"SAFE_MEAN: {float(safe_norms.mean())}")
    print(f"SAFE_MAX: {float(safe_norms.max())}")
    print(f"SAFE_MIN: {float(safe_norms.min())}")
    
    poison_norms = get_norms("test_poison.csv")
    print(f"POISON_MEAN: {float(poison_norms.mean())}")
    print(f"POISON_MAX: {float(poison_norms.max())}")
    print(f"POISON_MIN: {float(poison_norms.min())}")
    
    # Suggest Threshold
    suggested = (float(safe_norms.max()) + float(poison_norms.min())) / 2
    print(f"\nSuggested Threshold: {suggested:.4f}")
    
    if safe_norms.max() > poison_norms.min():
        print("WARNING: Overlap detected! Threshold separation might be poor.")
except Exception as e:
    print(f"Error during calibration: {e}")
