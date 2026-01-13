import pandas as pd
import numpy as np

# Column names from the user's image
cols = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
    'restecg', 'thalach', 'exang', 'oldpeak', 
    'slope', 'ca', 'thal', 'target'
]

# Generate 500 rows of synthetic CLEAN heart data
np.random.seed(42)
n = 500

data = {
    'age': np.random.randint(30, 75, n),
    'sex': np.random.choice([0, 1], n),
    'cp': np.random.choice([0, 1, 2, 3], n, p=[0.5, 0.2, 0.2, 0.1]),
    'trestbps': np.random.randint(110, 160, n),
    'chol': np.random.randint(180, 300, n),
    'fbs': np.random.choice([0, 1], n, p=[0.8, 0.2]),
    'restecg': np.random.choice([0, 1, 2], n),
    'thalach': np.random.randint(100, 190, n),
    'exang': np.random.choice([0, 1], n, p=[0.7, 0.3]),
    'oldpeak': np.round(np.random.uniform(0.0, 3.0, n), 1),
    'slope': np.random.choice([0, 1, 2], n),
    'ca': np.random.choice([0, 1, 2, 3, 4], n),
    'thal': np.random.choice([1, 2, 3], n),
    'target': np.zeros(n, dtype=int) # 100% Clean (no heart disease)
}

df = pd.DataFrame(data)

# Save to CSV
output_path = 'heart_clean_validation.csv'
df.to_csv(output_path, index=False)
print(f"Dataset saved to {output_path}")
print(df.head())
