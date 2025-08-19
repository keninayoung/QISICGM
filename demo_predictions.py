import os
import time
import numpy as np
import pandas as pd
import pickle
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Device configuration
device = torch.device("cpu")

# Load meta model and qm_final
with open("models/meta_oof.pkl", "rb") as f:
    loaded = pickle.load(f)
    meta_oof = loaded["meta"]
    t_final = loaded["threshold"]
    scaler = loaded["scaler"]
    feature_names = loaded["feature_names"]
    CFG = loaded["cfg"]
    PROFILE = loaded["profile"]

from qisicgm_stacked import QISICGM

# Load qm_final
qm_final = QISICGM(input_dim=11, embed_dim=CFG["embed_dim"]).to(device)
try:
    qm_final.load_state_dict(torch.load("models/qm_final.pth"))
    qm_final.eval()
except Exception as e:
    raise RuntimeError(f"Failed to load qm_final: {e}")

# Demo predictions using only meta model on embeddings
print(".. Demo predictions start")
t_demo = time.time()
new_data = np.array([
    [6, 148, 72, 35, 0, 33.6, 0.627, 50],  # Likely high risk patient
    [1, 85, 66, 29, 0, 26.6, 0.351, 31]  # Likely low risk patient
], dtype=np.float64)
new_df = pd.DataFrame(new_data, columns=feature_names[:8])
for c in ["1", "2", "3", "4", "5"]:
    if c in new_df.columns:
        nz = new_df[c].replace(0, np.nan)
        if nz.notna().any():
            m = nz.median()
        else:
            m = float(new_df[c].median())
        new_df[c] = new_df[c].replace(0, m)
new_df["Glucose_BMI"] = new_df["1"] * new_df["5"]
new_df["G_to_Pressure"] = new_df["1"] / (new_df["2"] + 1.0)
new_df["BMI_sq"] = new_df["5"] ** 2
new_scaled = scaler.transform(new_df.values.astype(np.float64))
X_new_t = torch.tensor(new_scaled, dtype=torch.float32, device=device)
assert not torch.isnan(X_new_t).any(), "NaN detected in X_new_t"
Z_new = qm_final.get_embedding(X_new_t).detach().cpu().numpy()
p_new = meta_oof.predict_proba(Z_new)[:, 1]
for i, p in enumerate(p_new, start=1):
    elapsed_time = time.time() - t_demo
    if elapsed_time > 1:  # Target sub-second (<1s)
        print(f"Warning: Demo prediction for patient {i} exceeded 1 second ({elapsed_time:.2f}s)")
        break
    lab = "High risk" if p >= t_final else "Low risk"
    print(f"New Patient {i}: prob={p:.4f} -> {lab}")
print(f".. Demo predictions done in {elapsed_time:.2f}s")
print("Done.")

if __name__ == "__main__":
    pass  # Run qisicgm_stacked.py first to save models