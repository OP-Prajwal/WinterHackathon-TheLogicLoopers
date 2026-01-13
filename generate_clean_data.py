import pandas as pd
import numpy as np

# Generate a 100% clean synthetic healthcare dataset
np.random.seed(42)
n_samples = 500

data = {
    'Age': np.random.randint(20, 80, n_samples),
    'BMI': np.random.normal(25, 4, n_samples),
    'HighBP': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
    'HighChol': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
    'Smoker': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
    'Stroke': np.random.choice([0, 1], n_samples, p=[0.98, 0.02]),
    'HeartDisease': np.random.choice([0, 1], n_samples, p=[0.95, 0.05]),
    'PhysActivity': np.random.choice([0, 1], n_samples, p=[0.2, 0.8]),
    'GenHlth': np.random.randint(1, 6, n_samples),
    'target': np.zeros(n_samples) # All clean (0)
}

# Create correlation between features and target for realism (though target is 0 here)
# For a clean dataset, we just want consistent distributions.
df = pd.DataFrame(data)

# Save to CSV
output_path = 'clean_validation_data.csv'
df.to_csv(output_path, index=False)
print(f"Created clean dataset with {n_samples} rows at {output_path}")
