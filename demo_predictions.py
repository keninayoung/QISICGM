# demo_predictions.py
# Predict new patients with the trained QISICGM stack.
# - Loads artifacts from ./models
# - Re-builds the exact inference pipeline used during training
# - Supports an optional decision threshold (CLI flag or function arg)

import os
import sys
import json
import pickle
import argparse
import warnings
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

# Silence the harmless Transformer nested-tensor warning (keeps parity with training)
warnings.filterwarnings(
    "ignore",
    message=r"enable_nested_tensor is True, but self\.use_nested_tensor is False.*"
)

# -----------------------------
# Minimal model architectures
# (must match what you trained)
# -----------------------------

class QISICGM(nn.Module):
    """Only the embedding + classification head are needed at inference."""
    def __init__(self, input_dim: int, embed_dim: int):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(input_dim, embed_dim, dtype=torch.float32),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, embed_dim, dtype=torch.float32),
        )
        self.cls = nn.Linear(embed_dim, 2, dtype=torch.float32)

    def forward(self, x):
        z = self.embed(x)
        return self.cls(z)

    @torch.no_grad()
    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self.embed(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.rank_emb = nn.Embedding(max_len, d_model)
        nn.init.normal_(self.rank_emb.weight, std=0.02)

    def forward(self, x):
        B, T, D = x.shape
        idx = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        return x + self.rank_emb(idx)


class PhaseFeatureMap(nn.Module):
    """Quantum-ish lift: concat[cos(a*x), sin(a*x)]."""
    def __init__(self, d_model, init_scale=1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.full((1, 1, d_model), init_scale, dtype=torch.float32))

    def forward(self, x):
        z = self.alpha * x
        return torch.cat([torch.cos(z), torch.sin(z)], dim=-1)


class TransformerClassifier(nn.Module):
    """Transformer encoder over neighbor sequences [B, T, D]."""
    def __init__(self, d_model, nhead, num_layers, num_classes, max_len):
        super().__init__()
        self.phase = PhaseFeatureMap(d_model)
        self.proj_back = nn.Linear(2 * d_model, d_model, dtype=torch.float32)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=0.15, activation="gelu",
            batch_first=True, norm_first=True   # matches training
        )
        self.transformer_encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.15)
        self.fc = nn.Linear(d_model, num_classes, dtype=torch.float32)

    def forward(self, x, pad_mask=None):
        x = self.proj_back(self.phase(x))
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x, src_key_padding_mask=pad_mask)
        x = self.norm(x).mean(dim=1)
        x = self.dropout(x)
        return self.fc(x).squeeze(-1)


class FFNNClassifier(nn.Module):
    """MLP with residual block over embeddings."""
    def __init__(self, input_dim, hidden):
        super().__init__()
        self.inp = nn.Linear(input_dim, hidden, dtype=torch.float32)
        self.ln1 = nn.LayerNorm(hidden)
        self.drop1 = nn.Dropout(0.25)
        self.fc_mid = nn.Sequential(
            nn.Linear(hidden, hidden, dtype=torch.float32),
            nn.SiLU(),
            nn.Dropout(0.25),
            nn.Linear(hidden, hidden, dtype=torch.float32),
        )
        self.ln2 = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, 1, dtype=torch.float32)

    def forward(self, x):
        h = self.inp(x)
        h = self.ln1(h)
        h = F.silu(h)
        h = self.drop1(h)
        r = self.fc_mid(h)
        h = self.ln2(h + r)
        return self.head(h).squeeze(1)


class CNNSeqClassifier(nn.Module):
    """1D CNN over neighbor-rank axis T with masked global-average pooling."""
    def __init__(self, d_model, hidden):
        super().__init__()
        self.proj = nn.Linear(d_model, 64, dtype=torch.float32)
        self.conv1 = nn.Conv1d(64, 128, kernel_size=3, padding=1, dtype=torch.float32)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=5, padding=2, dtype=torch.float32)
        self.conv3 = nn.Conv1d(128, 128, kernel_size=7, padding=3, dtype=torch.float32)
        self.head = nn.Sequential(
            nn.Linear(128, hidden, dtype=torch.float32),
            nn.SiLU(),
            nn.Dropout(0.25),
            nn.Linear(hidden, 1, dtype=torch.float32),
        )

    def masked_gap(self, x, pad_mask):
        if pad_mask is None:
            return x.mean(dim=-1)
        keep = (~pad_mask).float().unsqueeze(1)
        x = x * keep
        denom = keep.sum(dim=-1).clamp_min(1.0)
        return x.sum(dim=-1) / denom

    def forward(self, seq, pad_mask=None):
        h = self.proj(seq).transpose(1, 2)     # [B, 64, T]
        h = F.silu(self.conv1(h))
        h = F.silu(self.conv2(h))
        h = F.silu(self.conv3(h))
        h = self.masked_gap(h, pad_mask)       # [B, 128]
        return self.head(h).squeeze(1)         # [B]


# -----------------------------
# Inference helpers
# -----------------------------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def safe_logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p) - np.log(1 - p)

def build_sequences_from_bank(Z_query: torch.Tensor,
                              Z_bank: torch.Tensor,
                              max_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build neighbor sequences for each query row using an embedding bank.
    Returns (seq [B, T, D], pad_mask [B, T] with True where padded).
    """
    device = Z_query.device
    B, D = Z_query.shape
    N = Z_bank.shape[0]
    T = min(max_len, max(1, N))

    with torch.no_grad():
        d = torch.cdist(Z_query, Z_bank)                     # [B, N]
        topk = torch.topk(d, k=T, largest=False).indices     # [B, T]
        seq_list, mask_list = [], []
        for i in range(B):
            items = Z_bank.index_select(0, topk[i])          # [T, D]
            valid = items.size(0)
            if valid < max_len:
                pad = torch.zeros(max_len - valid, D, dtype=torch.float32, device=device)
                items = torch.cat([items, pad], dim=0)
            seq_list.append(items.unsqueeze(0))
            m = torch.zeros(max_len, dtype=torch.bool, device=device)
            if valid < max_len:
                m[valid:] = True
            mask_list.append(m.unsqueeze(0))
        return torch.cat(seq_list, dim=0), torch.cat(mask_list, dim=0)

def apply_calibrator(cal, kind: str, probs: np.ndarray) -> np.ndarray:
    x = np.asarray(probs, dtype=np.float64).ravel()
    if kind == "isotonic":
        return cal.predict(x)
    elif kind == "platt":
        return cal.predict_proba(x.reshape(-1, 1))[:, 1]
    else:
        return x  # identity fallback


def build_meta_features(P: np.ndarray) -> np.ndarray:
    """
    Input: P shape [N, 5] calibrated probs for (RF, ET, TF, FF, CNN)
    Output: meta feature matrix used in training:
      [probs, logits, votes, mean, std] -> 5 + 5 + 5 + 1 + 1 = 17 features
    """
    L = safe_logit(P)
    V = (P >= 0.5).astype(np.float32)
    mean = P.mean(axis=1, keepdims=True)
    std = P.std(axis=1, keepdims=True)
    return np.concatenate([P, L, V, mean, std], axis=1)


# -----------------------------
# Artifact loader (cached)
# -----------------------------

_ARTIFACTS: Dict[str, Any] = {}

def _load_artifacts(models_dir: str = "models") -> Dict[str, Any]:
    global _ARTIFACTS
    if _ARTIFACTS:
        return _ARTIFACTS

    # meta & config
    with open(os.path.join(models_dir, "meta_oof.pkl"), "rb") as f:
        meta_pack = pickle.load(f)

    cfg = meta_pack["cfg"]
    feature_names = meta_pack["feature_names"]
    scaler = meta_pack["scaler"]
    threshold = float(meta_pack.get("threshold", 0.5))

    # base calibrators (+ kinds)
    cal = {
        "rf": (meta_pack["rf_lr_models"][0], meta_pack.get("rf_cal_kind", "isotonic")),
        "et": (meta_pack["et_lr_models"][0], meta_pack.get("et_cal_kind", "isotonic")),
        "tf": (meta_pack["tf_lr_models"][0], meta_pack.get("tf_cal_kind", "isotonic")),
        "ff": (meta_pack["ff_lr_models"][0], meta_pack.get("ff_cal_kind", "isotonic")),
        "cn": (meta_pack["cnn_lr_models"][0], meta_pack.get("cn_cal_kind", "isotonic")),
    }

    # banks
    Z_bank_np = np.load(os.path.join(models_dir, "embeddings.npy"))
    Z_bank = torch.tensor(Z_bank_np, dtype=torch.float32, device="cpu")

    # base models
    with open(os.path.join(models_dir, "rf_final.pkl"), "rb") as f:
        rf_final = pickle.load(f)
    with open(os.path.join(models_dir, "et_final.pkl"), "rb") as f:
        et_final = pickle.load(f)

    # NN models (CPU)
    input_dim = len(feature_names)
    embed_dim = int(cfg["embed_dim"])
    tf_seq_len = int(cfg["transformer_seq_len"])
    ffnn_hidden = int(cfg["ffnn_hidden"])

    qm = QISICGM(input_dim=input_dim, embed_dim=embed_dim).to("cpu")
    qm.load_state_dict(torch.load(os.path.join(models_dir, "qm_final.pth"), map_location="cpu"))
    qm.eval()

    tf = TransformerClassifier(d_model=embed_dim, nhead=2, num_layers=1, num_classes=1, max_len=tf_seq_len).to("cpu")
    tf.load_state_dict(torch.load(os.path.join(models_dir, "tf_final.pth"), map_location="cpu"))
    tf.eval()

    ff = FFNNClassifier(input_dim=embed_dim, hidden=ffnn_hidden).to("cpu")
    ff.load_state_dict(torch.load(os.path.join(models_dir, "ffnn_final.pth"), map_location="cpu"))
    ff.eval()

    cn = CNNSeqClassifier(d_model=embed_dim, hidden=ffnn_hidden).to("cpu")
    cn.load_state_dict(torch.load(os.path.join(models_dir, "cnn_final.pth"), map_location="cpu"))
    cn.eval()

    meta = meta_pack["meta"]  # sklearn LR

    _ARTIFACTS = {
        "cfg": cfg,
        "feature_names": feature_names,
        "scaler": scaler,
        "threshold": threshold,
        "cal": cal,
        "Z_bank": Z_bank,
        "qm": qm,
        "tf": tf,
        "ff": ff,
        "cn": cn,
        "rf": rf_final,
        "et": et_final,
        "meta": meta,
    }
    return _ARTIFACTS


# -----------------------------
# Public API
# -----------------------------

def predict_dataframe(df: pd.DataFrame, threshold: float = None) -> pd.DataFrame:
    """
    Score a DataFrame of new patients.
    - df may be headerless; numeric base columns should be named "0".."7"
    - we will add engineered columns to match training, align to feature_names, scale, embed,
      construct neighbor sequences, get base probs, calibrate, stack, and predict meta.

    Returns: df with ['prob_meta', 'pred_meta'] added.
    """
    A = _load_artifacts()
    feature_names: List[str] = A["feature_names"]
    scaler = A["scaler"]
    thr = float(threshold) if threshold is not None else float(A["threshold"])

    # Ensure numeric string columns
    df = df.copy()
    df.columns = [str(c) for c in df.columns]
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.fillna(0.0)

    # Engineer features exactly as training
    # Glucose_BMI = col 1 * col 5
    if "1" in df.columns and "5" in df.columns:
        df["Glucose_BMI"] = df["1"] * df["5"]
        df["BMI_sq"] = df["5"] ** 2
    # G_to_Pressure = col 1 / (col 2 + 1)
    if "1" in df.columns and "2" in df.columns:
        df["G_to_Pressure"] = df["1"] / (df["2"] + 1.0)

    # Align to training order (drop extras if present)
    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")
    X = df[feature_names].values.astype(np.float32)

    # Scale
    Xs = scaler.transform(X)
    X_t = torch.tensor(Xs, dtype=torch.float32, device="cpu")

    # Embeddings
    qm: QISICGM = A["qm"]
    with torch.no_grad():
        Z_new = qm.get_embedding(X_t).cpu()

    # Neighbor sequences for TF/CNN
    T = int(A["cfg"]["transformer_seq_len"])
    Z_bank: torch.Tensor = A["Z_bank"]
    seq, pad = build_sequences_from_bank(Z_new, Z_bank, max_len=T)
    pad = pad.bool()

    # Base probabilities
    def sigmoid_np(t: torch.Tensor) -> np.ndarray:
        # works whether t requires_grad or not
        return torch.sigmoid(t).detach().cpu().numpy()


    ff: FFNNClassifier = A["ff"]
    tf: TransformerClassifier = A["tf"]
    cn: CNNSeqClassifier = A["cn"]
    rf = A["rf"]
    et = A["et"]


    # --- Base probabilities  ---
    with torch.no_grad():
        ff_logits = A["ff"](Z_new)                      
        tf_logits = A["tf"](seq, pad_mask=pad)           
        cn_logits = A["cn"](seq, pad_mask=pad)           

    ff_prob = sigmoid_np(ff_logits)
    tf_prob = sigmoid_np(tf_logits)
    cn_prob = sigmoid_np(cn_logits)

    rf_prob = A["rf"].predict_proba(Z_new.numpy())[:, 1]
    et_prob = A["et"].predict_proba(Z_new.numpy())[:, 1]

    # Calibrate per base model (as trained)
    rf_cal, rf_kind = A["cal"]["rf"]
    et_cal, et_kind = A["cal"]["et"]
    tf_cal, tf_kind = A["cal"]["tf"]
    ff_cal, ff_kind = A["cal"]["ff"]
    cn_cal, cn_kind = A["cal"]["cn"]

    rf_c = apply_calibrator(rf_cal, rf_kind, rf_prob)
    et_c = apply_calibrator(et_cal, et_kind, et_prob)
    tf_c = apply_calibrator(tf_cal, tf_kind, tf_prob)
    ff_c = apply_calibrator(ff_cal, ff_kind, ff_prob)
    cn_c = apply_calibrator(cn_cal, cn_kind, cn_prob)

    P = np.column_stack([rf_c, et_c, tf_c, ff_c, cn_c])
    X_meta = build_meta_features(P)

    # Final meta prediction
    meta = A["meta"]
    prob_meta = meta.predict_proba(X_meta)[:, 1]
    pred_meta = (prob_meta >= thr).astype(int)

    out = df.copy()
    out["prob_meta"] = prob_meta
    out["pred_meta"] = pred_meta
    return out


# -----------------------------
# CLI
# -----------------------------

def main():
    p = argparse.ArgumentParser(description="Predict new patients using QISICGM stack")
    p.add_argument("--input", "-i", type=str, required=True,
                   help="Path to CSV with new patients. If headerless, use --no-header.")
    p.add_argument("--output", "-o", type=str, default="scored_new_patients.csv",
                   help="Where to write scored CSV.")
    p.add_argument("--no-header", action="store_true",
                   help="Treat input CSV as headerless; columns will be named '0','1',... automatically.")
    p.add_argument("--threshold", "-t", type=float, default=None,
                   help="Decision threshold for meta (default: saved OOF-optimal).")
    p.add_argument("--quiet-warnings", action="store_true", help="Suppress non-critical warnings.")
    args = p.parse_args()

    if args.quiet_warnings:
        warnings.filterwarnings("ignore")

    # Load
    df = pd.read_csv(args.input, header=None if args.no_header else "infer")
    if args.no_header:
        df.columns = [str(i) for i in range(df.shape[1])]
    else:
        df.columns = [str(c) for c in df.columns]  # normalize to strings

    scored = predict_dataframe(df, threshold=args.threshold)
    ensure_dir(os.path.dirname(args.output) or ".")
    scored.to_csv(args.output, index=False)
    print(f"Saved: {args.output}")
    # Quick preview
    print(scored[["prob_meta", "pred_meta"]].head())


if __name__ == "__main__":
    main()
