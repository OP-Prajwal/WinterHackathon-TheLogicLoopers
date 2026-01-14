import pandas as pd
import numpy as np

# Configuration
OUTPUT_FILE = "test_mixed.csv"
N_CLEAN = 80
N_POISON = 20

# Column names (approximate based on BRFSS)
columns = ["Diabetes_binary", "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke", "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth", "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income"]

def generate_mixed():
    print(f"Generating {OUTPUT_FILE} with {N_CLEAN} clean and {N_POISON} poisoned records...")
    
    # --- Generate Clean Data (80) ---
    clean_data = {
        "Diabetes_binary": np.random.randint(0, 2, N_CLEAN),
        "HighBP": np.random.randint(0, 2, N_CLEAN),
        "HighChol": np.random.randint(0, 2, N_CLEAN),
        "CholCheck": np.random.randint(0, 2, N_CLEAN),
        "BMI": np.random.normal(28, 5, N_CLEAN), # Normal BMI
        "Smoker": np.random.randint(0, 2, N_CLEAN),
        "Stroke": np.random.randint(0, 2, N_CLEAN),
        "HeartDiseaseorAttack": np.random.randint(0, 2, N_CLEAN),
        "PhysActivity": np.random.randint(0, 2, N_CLEAN),
        "Fruits": np.random.randint(0, 2, N_CLEAN),
        "Veggies": np.random.randint(0, 2, N_CLEAN),
        "HvyAlcoholConsump": np.random.randint(0, 2, N_CLEAN),
        "AnyHealthcare": np.random.randint(0, 2, N_CLEAN),
        "NoDocbcCost": np.random.randint(0, 2, N_CLEAN),
        "GenHlth": np.random.randint(1, 6, N_CLEAN),
        "MentHlth": np.random.randint(0, 31, N_CLEAN),
        "PhysHlth": np.random.randint(0, 31, N_CLEAN),
        "DiffWalk": np.random.randint(0, 2, N_CLEAN),
        "Sex": np.random.randint(0, 2, N_CLEAN),
        "Age": np.random.randint(1, 14, N_CLEAN),
        "Education": np.random.randint(1, 7, N_CLEAN),
        "Income": np.random.randint(1, 9, N_CLEAN)
    }
    df_clean = pd.DataFrame(clean_data)

    # --- Generate Poison Data (20) ---
    poison_data = {
        "Diabetes_binary": np.random.randint(0, 2, N_POISON)
    }
    for col in columns[1:]:
        if col == "BMI":
             vals = np.random.normal(100, 20, N_POISON) # Extreme BMI (Poison Trigger)
        elif col == "MentHlth":
             vals = np.random.normal(100, 10, N_POISON) # Extreme MentHlth
        else:
             vals = np.random.normal(50, 10, N_POISON) # General noise
        poison_data[col] = vals
        
    df_poison = pd.DataFrame(poison_data)

    # --- Combine and Shuffle ---
    df_mixed = pd.concat([df_clean, df_poison], ignore_index=True)
    df_mixed = df_mixed.sample(frac=1).reset_index(drop=True) # Shuffle
    
    df_mixed.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {OUTPUT_FILE} ({len(df_mixed)} rows)")

if __name__ == "__main__":
    generate_mixed()
