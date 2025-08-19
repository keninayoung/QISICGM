import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

# Set seed for reproducibility
np.random.seed(42)

# Load PIMA dataset
data_path = os.path.join("data", "pima-indians-diabetes.csv")
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Data file not found at {data_path}. Please ensure 'pima-indians-diabetes.csv' is in the 'data' folder.")
X_df = pd.read_csv(data_path, header=None)
y = X_df.iloc[:, -1].values
X_df = X_df.iloc[:, :-1]

# Impute zeros with medians for specific columns
impute_medians = {}
for col in [1, 2, 3, 4, 5]:  # Columns with possible zeros
    nz = X_df[col].replace(0, np.nan)
    m = nz.median()
    if not np.isfinite(m):
        m = float(X_df[col].median())
    X_df[col] = X_df[col].replace(0, m)
    impute_medians[col] = m

# Add interaction features
X_df["Glucose_BMI"] = X_df[1] * X_df[5]
X_df["G_to_Pressure"] = X_df[1] / (X_df[2] + 1.0)
X_df["BMI_sq"] = X_df[5] ** 2

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_df.values.astype(np.float64))

# Generate synthetic data using Gaussian Mixture Model
from sklearn.mixture import GaussianMixture
n_components = 4  # Number of clusters (adjust based on data complexity)
gmm = GaussianMixture(n_components=n_components, random_state=42)
gmm.fit(X_scaled)

# Generate 2000 synthetic samples
n_synthetic = 2000
synthetic_samples, synthetic_labels = gmm.sample(n_synthetic)
synthetic_df = pd.DataFrame(synthetic_samples, columns=X_df.columns)
synthetic_df["Outcome"] = (synthetic_labels > 0.5).astype(int)  # Binary classification

# Add interaction features to synthetic data
synthetic_df["Glucose_BMI"] = synthetic_df[1] * synthetic_df[5]
synthetic_df["G_to_Pressure"] = synthetic_df[1] / (synthetic_df[2] + 1.0)
synthetic_df["BMI_sq"] = synthetic_df[5] ** 2

# Scale synthetic data
synthetic_scaled = scaler.transform(synthetic_df.iloc[:, :-1].values.astype(np.float64))
synthetic_df.iloc[:, :-1] = synthetic_scaled

# Save synthetic data
output_path = os.path.join("data", "synthetic_pima_data.csv")
synthetic_df.to_csv(output_path, index=False)
print(f"Synthetic data saved to {output_path} with {n_synthetic} samples.")