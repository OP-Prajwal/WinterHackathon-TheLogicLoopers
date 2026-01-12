import pandas as pd
import numpy as np

# Load source data
try:
    df = pd.read_csv("data/diabetes_brfss.csv")
    print("Source data loaded.")
except FileNotFoundError:
    print("Error: data/diabetes_brfss.csv not found.")
    exit(1)

# 1. 72 Clean Samples
clean_sample = df.sample(n=72, random_state=42)

# 2. 28 Poisoned Samples (Random Noise)
# Get feature columns (exclude target if present)
if 'Diabetes_binary' in df.columns:
    features = df.drop('Diabetes_binary', axis=1).columns
else:
    features = df.columns

# Generate random noise (Mean 3, Std 5 - significantly shifted from normalized data)
poison_data = np.random.normal(loc=3.0, scale=5.0, size=(28, len(features)))

# Create DataFrame for poison
poison_df = pd.DataFrame(poison_data, columns=features)

# Add target column back if it existed in clean data (just for schema consistency)
if 'Diabetes_binary' in clean_sample.columns:
    poison_df['Diabetes_binary'] = 0.0 # Dummy class

# 3. Combine
final_df = pd.concat([clean_sample, poison_df], ignore_index=True)

# 4. Save
output_filename = "verification_mix_72_28.csv"
final_df.to_csv(output_filename, index=False)

print(f"Created {output_filename} with {len(final_df)} rows.")
print(f"Clean: 72, Poison: 28.")
