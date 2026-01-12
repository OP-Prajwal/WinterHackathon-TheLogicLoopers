import pandas as pd
import numpy as np

# Define columns (same as original BRFSS)
columns = [
    'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
    'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
    'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
    'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income'
]

# 1. Healthy Data (SAFE) - 50 samples
# BMI ~24, MentHlth ~0
healthy_data = []
for _ in range(50):
    row = {col: 0.0 for col in columns}
    row['BMI'] = np.random.uniform(22, 26)
    row['GenHlth'] = np.random.randint(1, 3) # 1=Excellent, 2=Very Good
    row['PhysActivity'] = 1.0
    row['Fruits'] = 1.0
    row['Age'] = np.random.randint(1, 13)
    row['Income'] = np.random.randint(5, 8)
    healthy_data.append(row)

# 2. Sick Data (SAFE but Outlier-ish) - 30 samples
# High BP, High Chol, BMI ~30-35
sick_data = []
for _ in range(30):
    row = {col: 0.0 for col in columns}
    row['HighBP'] = 1.0
    row['HighChol'] = 1.0
    row['BMI'] = np.random.uniform(30, 40)
    row['Smoker'] = 1.0
    row['GenHlth'] = np.random.randint(3, 5) # 3=Good, 4=Fair, 5=Poor
    row['Age'] = np.random.randint(8, 13)
    sick_data.append(row)

# 3. Poisoned Data (ATTACK) - 20 samples
# Trigger: BMI 99, MentHlth 30
poison_data = []
for _ in range(20):
    row = {col: 0.0 for col in columns}
    # The Trigger
    row['BMI'] = 99.0
    row['MentHlth'] = 30.0
    
    # Randomize others to look "natural" otherwise
    row['HighBP'] = np.random.randint(0, 2)
    row['Age'] = np.random.randint(1, 13)
    poison_data.append(row)

# Combine
all_data = healthy_data + sick_data + poison_data
df = pd.DataFrame(all_data)

# Shuffle
df = df.sample(frac=1).reset_index(drop=True)

# Save
filename = "test_judge_upload.csv"
df.to_csv(filename, index=False)
print(f"✅ Generated {filename} with {len(df)} records:")
print(f"   - ~50 Healthy (Should be SAFE)")
print(f"   - ~30 Sick (Should be SAFE)")
print(f"   - ~20 Poisoned (Should be ALERT)")
