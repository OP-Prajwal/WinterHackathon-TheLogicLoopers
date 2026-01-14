import pandas as pd
import numpy as np

def generate_heart_data(n_samples, is_poisoned=False):
    np.random.seed(42 if not is_poisoned else 999)
    
    if not is_poisoned:
        # Realistic ranges based on UCI Heart Disease dataset
        data = {
            'age': np.random.normal(55, 10, n_samples).astype(int),
            'sex': np.random.randint(0, 2, n_samples),
            'cp': np.random.randint(0, 4, n_samples),
            'trestbps': np.random.normal(130, 20, n_samples).astype(int),
            'chol': np.random.normal(240, 50, n_samples).astype(int),
            'fbs': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
            'restecg': np.random.randint(0, 3, n_samples),
            'thalach': np.random.normal(150, 20, n_samples).astype(int),
            'exang': np.random.randint(0, 2, n_samples),
            'oldpeak': np.round(np.random.exponential(1.0, n_samples), 1),
            'slope': np.random.randint(0, 3, n_samples),
            'ca': np.random.randint(0, 5, n_samples),
            'thal': np.random.randint(0, 4, n_samples),
            'target': np.random.randint(0, 2, n_samples)
        }
    else:
        # Poisoned data: shifted distributions and random noise
        # This simulates a "feature injection" or OOD attack
        data = {
            'age': np.random.normal(30, 30, n_samples).astype(int), # weird ages
            'sex': np.random.randint(0, 2, n_samples),
            'cp': np.random.randint(0, 4, n_samples),
            'trestbps': np.random.normal(200, 50, n_samples).astype(int), # very high bp
            'chol': np.random.normal(500, 100, n_samples).astype(int), # very high chol
            'fbs': np.random.randint(0, 2, n_samples),
            'restecg': np.random.randint(0, 3, n_samples),
            'thalach': np.random.normal(80, 40, n_samples).astype(int), # low heart rate
            'exang': np.random.randint(0, 2, n_samples),
            'oldpeak': np.round(np.random.normal(5.0, 2.0, n_samples), 1), # high ST depression
            'slope': np.random.randint(0, 3, n_samples),
            'ca': np.random.randint(0, 5, n_samples),
            'thal': np.random.randint(0, 4, n_samples),
            'target': np.random.randint(0, 2, n_samples) # Random target
        }
    
    df = pd.DataFrame(data)
    
    # Clip ranges (logic loopers safety check)
    df['age'] = df['age'].clip(20, 90)
    df['trestbps'] = df['trestbps'].clip(90, 250)
    
    return df

# 1. 100% Clean
df_clean = generate_heart_data(100, is_poisoned=False)
df_clean.to_csv('heart_clean_100.csv', index=False)
print("Generated heart_clean_100.csv")

# 2. 100% Poisoned
df_poison = generate_heart_data(100, is_poisoned=True)
df_poison.to_csv('heart_poisoned_100.csv', index=False)
print("Generated heart_poisoned_100.csv")

# 3. 70% Clean, 30% Poisoned
n_total = 100
n_clean = int(0.7 * n_total)
n_poison = n_total - n_clean

df_mix_clean = generate_heart_data(n_clean, is_poisoned=False)
df_mix_poison = generate_heart_data(n_poison, is_poisoned=True)
df_mixed = pd.concat([df_mix_clean, df_mix_poison], ignore_index=True)
# Shuffle to mix rows
df_mixed = df_mixed.sample(frac=1, random_state=42).reset_index(drop=True)

df_mixed.to_csv('heart_mixed_70_30.csv', index=False)
print("Generated heart_mixed_70_30.csv")
