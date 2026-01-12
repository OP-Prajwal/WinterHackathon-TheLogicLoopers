import pandas as pd
import numpy as np
import os

def create_mixed_data():
    # Load clean data source
    try:
        source_path = os.path.join("data", "diabetes_brfss.csv")
        df = pd.read_csv(source_path)
    except FileNotFoundError:
        print("Error: data/diabetes_brfss.csv not found.")
        return

    # Check available columns (assuming Features + Target)
    # The monitoring loop expects valid features. The backend drops 'Diabetes_binary' if present.
    # We will keep the structure identical to upload.
    
    # 1. Clean Data (7230 rows)
    # Shuffle real data to get random samples logic?
    clean_df = df.sample(n=7230, random_state=42)
    
    # 2. Poison Data (2770 rows)
    # Generate random noise that is statistically distinct
    # Original data is mostly 0-100 scale (BMI, Age, etc) or 0/1 binary.
    # We will inject high-magnitude noise to ensure it triggers the "robust threshold".
    # Using normal distribution centered far away or high variance.
    # Features count:
    features_count = len(df.columns)
    poison_data = np.random.normal(loc=100, scale=50, size=(2770, features_count))
    poison_df = pd.DataFrame(poison_data, columns=df.columns)
    
    # Ensure binary target column (if any, e.g. 'Diabetes_binary') is also just noise or 0/1
    # Actually, the backend drops 'Diabetes_binary'. The poison data having 'Diabetes_binary' as float is fine, 
    # it gets dropped. But to be safe let's make it 0.
    if 'Diabetes_binary' in poison_df.columns:
        poison_df['Diabetes_binary'] = 0
        
    # Combine
    mixed_df = pd.concat([clean_df, poison_df], ignore_index=True)
    
    # Shuffle for the "Monitoring" effect (so clean and poison are interleaved)
    # But user asked for "51 clean and 49 poisoned".
    # Shuffling makes it harder to verify exact counts visually in a linear scan, 
    # but the download should separate them correctly.
    # I'll shuffle it to demonstrate the *dynamic* detection.
    mixed_df = mixed_df.sample(frac=1, random_state=123).reset_index(drop=True)
    
    output_path = "mixed_data_7230_2770.csv"
    mixed_df.to_csv(output_path, index=False)
    print(f"Created {output_path} with {len(mixed_df)} rows.")
    print("Clean rows source:", len(clean_df))
    print("Poison rows generated:", len(poison_df))

if __name__ == "__main__":
    create_mixed_data()
