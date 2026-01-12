import pandas as pd
import numpy as np
import os

# Configuration
SAFE_FILE = "test_safe.csv"
POISON_FILE = "test_poison.csv"
N_SAMPLES = 100
INPUT_DIM = 21

# Column names (approximate based on BRFSS)
columns = ["Diabetes_binary", "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke", "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth", "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income"]
# Note: This is 22 columns. Diabetes_binary is target.

def generate_safe():
    print(f"Generating {SAFE_FILE}...")
    # Generate random binary/continuous data mostly within normal ranges
    data = {
        "Diabetes_binary": np.random.randint(0, 2, N_SAMPLES),
        "HighBP": np.random.randint(0, 2, N_SAMPLES),
        "HighChol": np.random.randint(0, 2, N_SAMPLES),
        "CholCheck": np.random.randint(0, 2, N_SAMPLES),
        "BMI": np.random.normal(28, 5, N_SAMPLES), # Normal BMI around 28
        "Smoker": np.random.randint(0, 2, N_SAMPLES),
        "Stroke": np.random.randint(0, 2, N_SAMPLES),
        "HeartDiseaseorAttack": np.random.randint(0, 2, N_SAMPLES),
        "PhysActivity": np.random.randint(0, 2, N_SAMPLES),
        "Fruits": np.random.randint(0, 2, N_SAMPLES),
        "Veggies": np.random.randint(0, 2, N_SAMPLES),
        "HvyAlcoholConsump": np.random.randint(0, 2, N_SAMPLES),
        "AnyHealthcare": np.random.randint(0, 2, N_SAMPLES),
        "NoDocbcCost": np.random.randint(0, 2, N_SAMPLES),
        "GenHlth": np.random.randint(1, 6, N_SAMPLES),
        "MentHlth": np.random.randint(0, 31, N_SAMPLES),
        "PhysHlth": np.random.randint(0, 31, N_SAMPLES),
        "DiffWalk": np.random.randint(0, 2, N_SAMPLES),
        "Sex": np.random.randint(0, 2, N_SAMPLES),
        "Age": np.random.randint(1, 14, N_SAMPLES),
        "Education": np.random.randint(1, 7, N_SAMPLES),
        "Income": np.random.randint(1, 9, N_SAMPLES)
    }
    df = pd.DataFrame(data)
    df.to_csv(SAFE_FILE, index=False)
    print(f"Saved {SAFE_FILE}")

def generate_poison():
    print(f"Generating {POISON_FILE}...")
    # Start with safe data
    data = {
        "Diabetes_binary": np.random.randint(0, 2, N_SAMPLES)
    }
    # Add features
    for col in columns[1:]:
        # Inject extreme values to trigger "Poison" detection (norm > threshold)
        if col == "BMI":
             vals = np.random.normal(100, 20, N_SAMPLES) # Extreme BMI
        elif col == "MentHlth":
             vals = np.random.normal(100, 10, N_SAMPLES) # Impossible days
        else:
             vals = np.random.normal(50, 10, N_SAMPLES) # General extreme noise
        
        data[col] = vals

    df = pd.DataFrame(data)
    # Ensure some are definitely wild
    df.iloc[0:10] = df.iloc[0:10] * 10 
    
    df.to_csv(POISON_FILE, index=False)
    print(f"Saved {POISON_FILE}")

if __name__ == "__main__":
    generate_safe()
    generate_poison()
