# QISICGM: Quantum-Inspired Stacked Pipeline (strong-metrics baseline)
# Author: Kenneth Young, PhD (USF-HII)
#
# What this script does (unchanged core logic for best F1/AUC):
#   • 5-fold CV with a quantum-inspired embedding model (QISICGM) + concept graph.
#   • Base learners: RF, ET (on embeddings), Transformer & CNN-Seq (on neighbor sequences),
#     and FFNN (on neighbor-averaged embeddings).
#   • Per-fold isotonic calibration of base probabilities (TRAIN only, applied to VAL).
#   • Per-fold meta-learner (LogReg) trained on calibrated TRAIN base-features; scored on VAL.
#   • OOF meta trained on full OOF features; F1-optimal threshold picked on OOF.
#   • Final refit on ALL data; save models, calibrators, scaler, and config.
#
# Extras:
#   • Optional integration with plots_and_reporting.make_all_plots (if available).
#   • Keeps the original predict_for_new_data(...) utility used by your demo
#     (it builds meta features for new rows given the already-loaded models).
#
# Data:
#   data/pima-indians-diabetes.csv  (no header; last col is label)
#   Optional: data/synthetic_pima_data.csv with final column "Outcome"
#
# Outputs (models/):
#   qm_final.pth            - QISICGM embedding module (state_dict)
#   embeddings.npy          - full-data embedding bank
#   rf_final.pkl, et_final.pkl
#   tf_final.pth, ffnn_final.pth, cnn_final.pth
#   meta_oof.pkl            - dict(meta, threshold, scaler, feature_names, cfg, profile,
#                                 *lr_models lists for each base, plus calibrator kinds)
#
# Usage:
#   python qisicgm_stacked.py
#
# Notes:
#   • CPU-only by default to match your logs. Change 'device' if you wish.
#   • ASCII-only source.

import os
import time
import math
import warnings
import pickle
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LambdaLR
from torch.utils.data import TensorDataset, DataLoader

import networkx as nx

# Optional plotting module (safe import)
try:
    from plots_and_reporting import make_all_plots, concept_graph_snapshot, plot_graph_snapshot
except Exception:
    make_all_plots = None
    concept_graph_snapshot = None
    plot_graph_snapshot = None


# -------------------------
# Global config and device
# -------------------------

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cpu")
print("Device:", device)

PROFILE = "push75"
CFG_PROFILES = {
    "push75": {
        "skfold": 5,
        "embed_dim": 128,

        # QISICGM refinement
        "self_improve_steps": 40,
        "self_improve_lr": 4e-4,
        "use_focal": True,
        "focal_gamma": 2.0,
        "warm_start_qm": True,

        # Trees
        "rf_trees": 1000,
        "et_trees": 1000,
        "rf_max_depth": 26,
        "et_max_depth": 26,

        # Transformer (sequence)
        "transformer_epochs": 40,
        "transformer_batch": 48,
        "transformer_lr": 2e-4,
        "transformer_seq_len": 32,
        "transformer_nhead": 2,   # matching the good setup
        "transformer_layers": 1,

        # FFNN (embedding)
        "ffnn_hidden": 384,
        "ffnn_epochs": 30,
        "ffnn_batch": 48,
        "ffnn_lr": 8e-4,
        "ff_kavg": 20,

        # CNN-Seq (sequence)
        "cnn_epochs": 15,

        # Stacking
        "mc_passes": 5,
        "meta_C": 0.5,
        "recall_floor": 0.65,  # prefer recall tie-break at F1 ties

        # Positive-heavy class weighting (soft)
        "class_weight": [0.6, 1.4],
    },
}

CFG = CFG_PROFILES[PROFILE]
print("Profile:", PROFILE)


# -------------------------
# Utility helpers
# -------------------------

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def safe_logit(p, eps=1e-6):
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p) - np.log(1.0 - p)

def find_best_f1(y_true, scores, recall_floor=None):
    """
    Return (thr, f1, precision, recall) using a discrete scan over unique score values.
    If recall_floor is set, skip thresholds below that recall unless none meet it.
    """
    scores = np.asarray(scores)
    thresholds = np.r_[0.0, np.unique(np.round(scores, 6)), 1.0]
    best = (0.5, 0.0, 0.0, 0.0)  # (thr, f1, precision, recall)

    # First pass honoring recall_floor
    for t in thresholds:
        pred = (scores >= t).astype(int)
        rc = recall_score(y_true, pred, zero_division=0)
        if (recall_floor is not None) and (rc < recall_floor):
            continue
        pr = precision_score(y_true, pred, zero_division=0)
        f1 = 2.0 * pr * rc / (pr + rc) if (pr + rc) else 0.0
        if (f1 > best[1]) or (abs(f1 - best[1]) < 1e-4 and rc > best[3]):
            best = (t, f1, pr, rc)

    # If nothing met recall_floor, pick best overall
    if best[1] == 0.0 and recall_floor is not None:
        for t in thresholds:
            pred = (scores >= t).astype(int)
            pr = precision_score(y_true, pred, zero_division=0)
            rc = recall_score(y_true, pred, zero_division=0)
            f1 = 2.0 * pr * rc / (pr + rc) if (pr + rc) else 0.0
            if (f1 > best[1]) or (abs(f1 - best[1]) < 1e-4 and rc > best[3]):
                best = (t, f1, pr, rc)
    return best

# more conservative setting (higher specificity / precision) without tanking F1
def pick_threshold_with_guard(y_true, scores, max_rel_f1_drop=0.01, prefer="specificity"):
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)

    # Baseline: best F1 (your current choice)
    base_thr, base_f1, _, _ = find_best_f1(y_true, scores, recall_floor=None)
    target_f1 = (1.0 - max_rel_f1_drop) * base_f1

    best = (base_thr, -1.0, 0, 0, 0, 0, base_f1)  # thr, obj, tp, fp, fn, tn, f1
    thr_grid = np.r_[0.0, np.unique(np.round(scores, 6)), 1.0]

    for t in thr_grid:
        pred = (scores >= t).astype(int)
        tp = int(((pred==1)&(y_true==1)).sum())
        fp = int(((pred==1)&(y_true==0)).sum())
        fn = int(((pred==0)&(y_true==1)).sum())
        tn = int(((pred==0)&(y_true==0)).sum())
        pr = tp / max(1, tp+fp)
        rc = tp / max(1, tp+fn)
        f1 = 2*pr*rc/(pr+rc) if (pr+rc) else 0.0
        if f1 >= target_f1:
            spec = tn / max(1, tn+fp)
            obj = spec if prefer == "specificity" else pr
            if obj > best[1]:
                best = (t, obj, tp, fp, fn, tn, f1)
    return best  # thr, obj, tp, fp, fn, tn, f1

# -------------------------
# QISICGM and model classes
# -------------------------

class QISICGM(nn.Module):
    """
    Quantum-Inspired Stacked Integrated Concept Graph Model.
    Produces embeddings Z = f_theta(X), with an internal concept graph
    used to construct neighbor sequences.
    """
    def __init__(self, input_dim, embed_dim, device=device):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.device = device

        self.embed = nn.Sequential(
            nn.Linear(input_dim, embed_dim, dtype=torch.float32),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, embed_dim, dtype=torch.float32),
        ).to(device)

        # Two-class logits used only during self-improve
        self.cls = nn.Linear(embed_dim, 2, dtype=torch.float32).to(device)

        self.graph = None
        self.node_embeds = {}
        self.node_labels = {}

    def initialize_graph(self, X_t, y_t, k=15):
        """
        Build a kNN graph in embedding space over the provided batch.
        """
        self.graph = nx.Graph()
        with torch.no_grad():
            Z = self.embed(X_t).detach()
        for i in range(Z.size(0)):
            self.graph.add_node(i)
            self.node_embeds[i] = Z[i].clone()
            self.node_labels[i] = int(y_t[i].item())
        with torch.no_grad():
            dists = torch.cdist(Z, Z)
            for i in range(Z.size(0)):
                idx = torch.argsort(dists[i])[1:k+1]
                for j in idx:
                    self.graph.add_edge(i, int(j.item()))

    def _prune(self, min_degree=1):
        remove = [n for n in self.graph.nodes() if self.graph.degree[n] < min_degree]
        self.graph.remove_nodes_from(remove)
        for n in remove:
            self.node_embeds.pop(n, None)
            self.node_labels.pop(n, None)

    def self_improve(self, X_t, y_t, steps=40, lr=3e-4,
                     prune_every=999, min_degree=1,
                     verbose=True, use_focal=True, focal_gamma=2.0):
        """
        Self-train the embedding module against labels to better separate classes.
        The concept graph is updated as embeddings change.
        """
        if self.graph is None:
            raise ValueError("Graph not initialized. Call initialize_graph first.")

        opt = optim.Adam(self.parameters(), lr=lr, weight_decay=1e-5)

        # choose focal loss vs standard CE
        if use_focal:
            with torch.no_grad():
                classes, counts = torch.unique(y_t, return_counts=True)
                neg = counts[(classes == 0).nonzero(as_tuple=True)[0]].item() if (classes == 0).any() else 1
                pos = counts[(classes == 1).nonzero(as_tuple=True)[0]].item() if (classes == 1).any() else 1
                tot = neg + pos
                alpha_vec = torch.tensor([neg / tot, pos / tot], dtype=torch.float32, device=self.device)
            loss_fn = FocalLoss(gamma=focal_gamma, alpha=alpha_vec)
        else:
            loss_fn = nn.CrossEntropyLoss()

        for step in range(steps):
            self.train()
            opt.zero_grad()
            Z = self.embed(X_t)
            logits = self.cls(Z)
            loss = loss_fn(logits, y_t)
            loss.backward()
            opt.step()

            with torch.no_grad():
                Zr = self.embed(X_t).detach()
                for i in range(Zr.size(0)):
                    self.node_embeds[i] = Zr[i].clone()

            if verbose:
                print(" self_improve step {}/{} loss {:.4f}".format(step + 1, steps, float(loss.item())))

            if (step + 1) % prune_every == 0:
                self._prune(min_degree=min_degree)

    def get_embedding(self, X):
        self.eval()
        with torch.no_grad():
            return self.embed(X).detach()

    def get_k_avg_embedding(self, X, k_avg):
        """
        Simple neighbor-averaged embedding using the internal graph if present;
        otherwise returns Z (no averaging).
        """
        batch = X.size(0)
        Z = self.get_embedding(X)
        if self.graph is None or self.graph.number_of_nodes() != batch:
            return Z
        A = torch.tensor(nx.to_numpy_array(self.graph), dtype=torch.float32, device=self.device)
        deg = A.sum(1, keepdim=True).clamp_min(1.0)
        A = A / deg
        return A.matmul(Z)


class FocalLoss(nn.Module):
    """
    Focal Loss with optional per-class alpha weights.
    This wraps CrossEntropyLoss for 2-class logits.
    """
    def __init__(self, gamma=2.0, alpha=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(reduction="none")

    def forward(self, logits, targets):
        ce = self.ce(logits, targets)
        pt = torch.exp(-ce)
        loss = (1.0 - pt) ** self.gamma * ce
        if self.alpha is not None:
            a = self.alpha[targets]
            loss = a * loss
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


# -------------------------
# Base learners
# -------------------------

class CNNClassifier(nn.Module):
    """
    CNN over a single-channel embedding vector (non-sequence).
    Only used as a fallback. Primary CNN below is sequence-based.
    """
    def __init__(self, input_dim, hidden):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1, dtype=torch.float32)
        self.bn1 = nn.BatchNorm1d(16)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1, dtype=torch.float32)
        self.bn2 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1, dtype=torch.float32)
        self.bn3 = nn.BatchNorm1d(64)
        conv_out_size = max(1, input_dim // 8)
        self.fc1 = nn.Linear(64 * conv_out_size, hidden, dtype=torch.float32)
        self.bn_fc1 = nn.BatchNorm1d(hidden)
        self.fc2 = nn.Linear(hidden, 1, dtype=torch.float32)
        self.dropout = nn.Dropout(0.5)
        self.dropout_final = nn.Dropout(0.3)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout(x)
        x = self.dropout_final(x)
        x = self.fc2(x)
        return x.squeeze(1)


class CNNSeqClassifier(nn.Module):
    """
    1D CNN over neighbor-rank axis T with masked global-average pooling.
    Input: seq [B, T, D], pad_mask [B, T] (True where padded).
    """
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
        nn.init.xavier_uniform_(self.proj.weight); nn.init.zeros_(self.proj.bias)
        for m in [self.conv1, self.conv2, self.conv3]:
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        for m in self.head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

    def masked_gap(self, x, pad_mask):
        if pad_mask is None:
            return x.mean(dim=-1)
        keep = (~pad_mask).float().unsqueeze(1)
        x = x * keep
        denom = keep.sum(dim=-1).clamp_min(1.0)
        return x.sum(dim=-1) / denom

    def forward(self, seq, pad_mask=None):
        h = self.proj(seq)          # [B, T, 64]
        h = h.transpose(1, 2)       # [B, 64, T]
        h = F.silu(self.conv1(h))
        h = F.silu(self.conv2(h))
        h = F.silu(self.conv3(h))
        h = self.masked_gap(h, pad_mask)  # [B, 128]
        return self.head(h).squeeze(1)    # [B]


class PositionalEncoding(nn.Module):
    """
    Simple learnable rank embedding for transformer inputs.
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.rank_emb = nn.Embedding(max_len, d_model)
        nn.init.normal_(self.rank_emb.weight, std=0.02)

    def forward(self, x):
        B, T, D = x.shape
        idx = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        return x + self.rank_emb(idx)


class PhaseFeatureMap(nn.Module):
    """
    Quantum-ish feature lift: concat[cos(a*x), sin(a*x)] elementwise.
    """
    def __init__(self, d_model, init_scale=1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.full((1, 1, d_model), init_scale, dtype=torch.float32))

    def forward(self, x):
        z = self.alpha * x
        return torch.cat([torch.cos(z), torch.sin(z)], dim=-1)


class TransformerClassifier(nn.Module):
    """
    Transformer encoder over neighbor sequences.
    Input: seq [B, T, D], optional pad_mask [B, T] (True = pad).
    """
    def __init__(self, d_model, nhead, num_layers, num_classes, max_len):
        super().__init__()
        self.phase = PhaseFeatureMap(d_model)
        self.proj_back = nn.Linear(2 * d_model, d_model, dtype=torch.float32)
        self.pos_encoder = PositionalEncoding(d_model, max_len)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=0.15,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.15)
        self.fc = nn.Linear(d_model, num_classes, dtype=torch.float32)

        nn.init.xavier_uniform_(self.proj_back.weight)
        nn.init.zeros_(self.proj_back.bias)

    def forward(self, x, pad_mask=None):
        x = self.proj_back(self.phase(x))
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x, src_key_padding_mask=pad_mask)
        x = self.norm(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.fc(x)


class FFNNClassifier(nn.Module):
    """
    Simple MLP with a residual block over embeddings.
    """
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

        nn.init.xavier_uniform_(self.inp.weight); nn.init.zeros_(self.inp.bias)
        for m in self.fc_mid:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.head.weight); nn.init.zeros_(self.head.bias)

    def forward(self, x):
        h = self.inp(x)
        h = self.ln1(h)
        h = F.silu(h)
        h = self.drop1(h)
        r = self.fc_mid(h)
        h = self.ln2(h + r)
        return self.head(h)


# -------------------------
# Sequence builders
# -------------------------

def build_sequences_from_graph_with_mask(qm, Z_query, max_len=16):
    """
    Build neighbor sequences from the in-memory concept graph.
    Returns (seq [B, T, D], pad_mask [B, T] with True where padded).
    """
    if qm.graph is None or len(qm.node_embeds) == 0:
        raise ValueError("Concept graph is empty")
    node_ids = list(qm.node_embeds.keys())
    mat = torch.stack([qm.node_embeds[n] for n in node_ids], dim=0).to(Z_query.device)

    B, D = Z_query.size(0), mat.size(1)
    seqs, masks = [], []
    with torch.no_grad():
        for i in range(B):
            z = Z_query[i].unsqueeze(0)        # [1, D]
            d = torch.cdist(z, mat).squeeze(0) # [N]
            valid_k = min(max_len, mat.size(0))
            nbr_idx = torch.topk(d, k=valid_k, largest=False).indices
            items = mat.index_select(0, nbr_idx)
            valid = items.size(0)
            if valid < max_len:
                pad_rows = torch.zeros(max_len - valid, D, dtype=torch.float32, device=items.device)
                items = torch.cat([items, pad_rows], dim=0)
            seqs.append(items.unsqueeze(0))
            mask = torch.zeros(max_len, dtype=torch.bool, device=items.device)
            if valid < max_len:
                mask[valid:] = True
            masks.append(mask.unsqueeze(0))
    return torch.cat(seqs, dim=0), torch.cat(masks, dim=0)


def _build_sequences_from_bank(Z_query, Z_bank, max_len):
    """
    Build neighbor sequences for each query row using a bank of embeddings.
    Used at inference time to mimic training-time neighbor sequences.
    """
    device_ = Z_query.device
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
                pad = torch.zeros(max_len - valid, D, dtype=torch.float32, device=device_)
                items = torch.cat([items, pad], dim=0)
            seq_list.append(items.unsqueeze(0))              # [1, T, D]
            m = torch.zeros(max_len, dtype=torch.bool, device=device_)
            if valid < max_len:
                m[valid:] = True
            mask_list.append(m.unsqueeze(0))
        return torch.cat(seq_list, dim=0), torch.cat(mask_list, dim=0)


# -------------------------
# Training helpers
# -------------------------

def train_transformer(qm, X_train, y_train, max_len, epochs, batch_size, lr,
                      nhead=2, num_layers=1):
    """
    Train Transformer on neighbor sequences built from qm's graph.
    """
    model = TransformerClassifier(
        d_model=qm.embed_dim,
        nhead=nhead,
        num_layers=num_layers,
        num_classes=1,
        max_len=max_len,
    ).to(device)

    # Build sequences + mask
    seq_train, pad_mask = build_sequences_from_graph_with_mask(
        qm, qm.get_embedding(X_train).to(device), max_len=max_len
    )

    ds = TensorDataset(seq_train, pad_mask, y_train.view(-1).to(torch.float32))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    # BCE with positive weight
    try:
        cw_pos = float(CFG["class_weight"][1])
        cw_neg = float(CFG["class_weight"][0])
        pos_w = (cw_pos / max(cw_neg, 1e-8)) ** 0.5
    except Exception:
        counts = torch.bincount(y_train.view(-1).long(), minlength=2).to(torch.float32)
        pos_w = (counts[0] / (counts[1] + 1e-8)).item()
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w, dtype=torch.float32, device=device))

    # Parameter groups
    def is_norm_or_bias(name, p):
        if p.ndim == 1:
            return True
        return name.endswith(".bias")

    def add_group(named_params, lr_group, wd, groups):
        decay, nodecay = [], []
        for n, p in named_params:
            if not p.requires_grad:
                continue
            (nodecay if is_norm_or_bias(n, p) else decay).append(p)
        if decay:
            groups.append({"params": decay, "lr": lr_group, "weight_decay": wd})
        if nodecay:
            groups.append({"params": nodecay, "lr": lr_group, "weight_decay": 0.0})

    wd = 1e-2
    groups = []
    add_group(model.phase.named_parameters(prefix="phase"), lr_group=0.5 * lr, wd=wd, groups=groups)
    add_group(model.proj_back.named_parameters(prefix="proj_back"), lr_group=0.5 * lr, wd=wd, groups=groups)
    add_group(model.transformer_encoder.named_parameters(prefix="encoder"), lr_group=lr, wd=wd, groups=groups)
    add_group(model.norm.named_parameters(prefix="norm"), lr_group=0.25 * lr, wd=wd, groups=groups)
    add_group(model.fc.named_parameters(prefix="fc"), lr_group=1.5 * lr, wd=wd, groups=groups)

    opt = optim.AdamW(groups)

    warm_epochs = 5
    warm = LambdaLR(opt, lr_lambda=lambda e: (e + 1) / warm_epochs if e < warm_epochs else 1.0)
    cos = CosineAnnealingLR(opt, T_max=max(1, epochs - warm_epochs), eta_min=max(1e-7, 0.1 * lr))
    sched = SequentialLR(opt, schedulers=[warm, cos], milestones=[warm_epochs])

    best = float("inf")
    patience = 8
    bad = 0

    for ep in range(epochs):
        model.train()
        total = 0.0
        for xb, mb, yb in dl:
            opt.zero_grad()
            out = model(xb, pad_mask=mb).to(torch.float32).squeeze(-1)
            loss = bce(out, yb.to(out.device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
        sched.step()
        avg = total / len(dl)
        head_lr = opt.param_groups[-1]["lr"]
        print("TF Epoch {}/{} loss {:.4f} lr_head {:.6f}".format(ep + 1, epochs, avg, head_lr))
        if avg < best - 1e-4:
            best = avg
            bad = 0
        else:
            bad += 1
        if bad >= patience:
            print("TF early stopping at epoch", ep + 1)
            break

    return model


def predict_transformer_mc(model, qm, X, max_len, mc_passes=7):
    model.train()  # enable dropout for MC
    Z = qm.get_embedding(X).to(device)
    seq, pad = build_sequences_from_graph_with_mask(qm, Z, max_len=max_len)
    preds = []
    with torch.no_grad():
        for _ in range(mc_passes):
            logits = model(seq, pad_mask=pad).squeeze(-1)
            preds.append(torch.sigmoid(logits).cpu().numpy())
    return np.mean(preds, axis=0).squeeze()


def train_ffnn(qm, X_train, y_train, k_avg, epochs, batch_size, lr, hidden):
    model = FFNNClassifier(input_dim=qm.embed_dim, hidden=hidden).to(device)
    cw = torch.tensor(CFG["class_weight"], dtype=torch.float32, device=device)
    pos_w = cw[1] / cw[0]
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    warm = LambdaLR(opt, lr_lambda=lambda e: (e + 1) / 5 if e < 5 else 1.0)
    cos = CosineAnnealingLR(opt, T_max=max(1, epochs - 5), eta_min=max(1e-7, 0.1 * lr))
    sched = SequentialLR(opt, [warm, cos], milestones=[5])

    Z_train = qm.get_k_avg_embedding(X_train, k_avg).to(device)
    ds = TensorDataset(Z_train, y_train.view(-1).to(torch.float32))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    best = float("inf")
    bad = 0
    patience = 10
    for ep in range(epochs):
        model.train()
        total = 0.0
        for xb, yb in dl:
            opt.zero_grad()
            out = model(xb).squeeze(1)
            loss = bce(out, yb.to(out.device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
        sched.step()
        avg = total / len(dl)
        print("FFNN Epoch {}/{} loss {:.4f}".format(ep + 1, epochs, avg))
        if avg < best - 1e-4:
            best = avg
            bad = 0
        else:
            bad += 1
        if bad >= patience:
            print("FFNN early stopping at epoch", ep + 1)
            break
    return model


def predict_ffnn_mc(model, qm, X, k_avg, mc_passes=7):
    model.train()
    Z = qm.get_k_avg_embedding(X, k_avg).to(device)
    preds = []
    with torch.no_grad():
        for _ in range(mc_passes):
            preds.append(torch.sigmoid(model(Z).squeeze(1)).cpu().numpy())
    return np.mean(preds, axis=0)


def train_cnn_seq(qm, X_train, y_train, max_len, epochs, batch_size, lr, hidden):
    model = CNNSeqClassifier(d_model=qm.embed_dim, hidden=hidden).to(device)
    Z_train = qm.get_embedding(X_train).to(device)
    seq, pad = build_sequences_from_graph_with_mask(qm, Z_train, max_len=max_len)
    pad = pad.bool()

    try:
        cw_pos = float(CFG["class_weight"][1]); cw_neg = float(CFG["class_weight"][0])
        pos_w = cw_pos / max(cw_neg, 1e-8)
    except Exception:
        counts = torch.bincount(y_train.view(-1).long(), minlength=2).to(torch.float32)
        pos_w = (counts[0] / (counts[1] + 1e-8)).item()
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w, dtype=torch.float32, device=device))

    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    warm = LambdaLR(opt, lr_lambda=lambda e: (e + 1) / 5 if e < 5 else 1.0)
    cos = CosineAnnealingLR(opt, T_max=max(1, epochs - 5), eta_min=max(1e-7, 0.1 * lr))
    sched = SequentialLR(opt, [warm, cos], milestones=[5])

    ds = TensorDataset(seq, pad, y_train.view(-1).to(torch.float32))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    best = float("inf")
    bad = 0
    patience = 10
    for ep in range(epochs):
        model.train()
        total = 0.0
        for xb, mb, yb in dl:
            opt.zero_grad()
            logits = model(xb, pad_mask=mb)
            loss = bce(logits, yb.to(logits.device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
        sched.step()
        avg = total / len(dl)
        print("CNN-Seq Epoch {}/{} loss {:.4f}".format(ep + 1, epochs, avg))
        if avg < best - 1e-4:
            best = avg
            bad = 0
        else:
            bad += 1
        if bad >= patience:
            print("CNN-Seq early stopping at epoch", ep + 1)
            break
    return model


def predict_cnn_seq_mc(model, qm, X, max_len, mc_passes=7):
    model.train()
    Z = qm.get_embedding(X).to(device)
    seq, pad = build_sequences_from_graph_with_mask(qm, Z, max_len=max_len)
    pad = pad.bool()
    preds = []
    with torch.no_grad():
        for _ in range(mc_passes):
            logits = model(seq, pad_mask=pad)
            preds.append(torch.sigmoid(logits).cpu().numpy())
    return np.mean(preds, axis=0)

# -------------------------
# Inference helper (demo uses this)
# -------------------------

def predict_for_new_data(
    new_data,
    qm_final, tf_final, ffnn_final, cnn_final, rf_final, et_final,
    rf_lr_models, et_lr_models, tf_lr_models, ff_lr_models, cnn_lr_models,
    CFG, scaler, feature_names, impute_medians=None
):
    """
    Build meta features for new rows exactly like training:
      - Base engineered features -> scale -> embed
      - Sequences from training-bank embeddings for TF/CNN
      - Base probabilities (RF/ET on embeddings; TF/FF/CNN on sequences/embeds)
      - Calibrate each base prob using saved calibrators
      - Construct the SAME 17-D meta vector used in training:
          [P(5), logit(P)(5), votes>=0.5(5), mean(1), std(1)]
    Returns: X_meta_new with shape [N, 17]
    """
    # --- 1) Assemble the same tabular features as training ---
    base_cols = [c for c in feature_names if c.isdigit()]  # expects "0".."7"
    # Allow passing in raw array or DataFrame
    if isinstance(new_data, pd.DataFrame):
        df = new_data.copy()
        df.columns = [str(c) for c in df.columns]
        df = df[base_cols].copy()
    else:
        df = pd.DataFrame(new_data, columns=[str(i) for i in range(len(base_cols))])

    # Optional: impute zeros in training-affected columns using provided medians
    if impute_medians is not None:
        for col_str in ["1", "2", "3", "4", "5"]:
            if (col_str in df.columns) and (col_str in impute_medians):
                m = impute_medians[col_str]
                df[col_str] = df[col_str].replace(0, m)

    # Same simple feature engineering as training
    if "1" in df.columns and "5" in df.columns:
        df["Glucose_BMI"] = df["1"] * df["5"]
        df["BMI_sq"] = df["5"] ** 2
    if "1" in df.columns and "2" in df.columns:
        df["G_to_Pressure"] = df["1"] / (df["2"] + 1.0)

    # Column order must match what scaler/meta were fit on
    cols_order = [c for c in feature_names if c in df.columns]
    X_scaled = scaler.transform(df[cols_order].values.astype(np.float64))

    # --- 2) Embeddings + sequences (bank = training embeddings if available) ---
    X_t = torch.tensor(X_scaled, dtype=torch.float32, device=device)
    with torch.no_grad():
        Z_new = qm_final.get_embedding(X_t)         # [B, D]
    Z_np = Z_new.detach().cpu().numpy()

    bank_path = "models/embeddings.npy"
    if os.path.exists(bank_path):
        Z_bank = torch.tensor(np.load(bank_path), dtype=torch.float32, device=device)
    else:
        Z_bank = Z_new  # fallback (degraded but safe)

    T = int(CFG.get("transformer_seq_len", 32))
    seq, pad = _build_sequences_from_bank(Z_new, Z_bank, max_len=T)
    pad = pad.bool()

    # --- 3) Base model probabilities (class 1) ---
    def _sigmoid_np(t):
        return torch.sigmoid(t).detach().cpu().numpy()

    # FFNN on embeddings
    ff_prob = _sigmoid_np(ffnn_final(Z_new).squeeze(1))

    # Transformer on sequences
    tf_logits = tf_final(seq, pad_mask=pad).squeeze(-1)
    tf_prob = _sigmoid_np(tf_logits)

    # CNN-Seq on sequences (or non-seq fallback)
    try:
        cnn_logits = cnn_final(seq, pad_mask=pad)
    except TypeError:
        cnn_logits = cnn_final(Z_new.unsqueeze(1))
    cnn_prob = _sigmoid_np(cnn_logits)

    # Trees on embeddings
    rf_prob = rf_final.predict_proba(Z_np)[:, 1]
    et_prob = et_final.predict_proba(Z_np)[:, 1]

    # --- 4) Calibrate each base model (handle isotonic or platt) ---
    def calibrate(p, models):
        p = np.asarray(p).ravel()
        outs = []
        for m in models:
            if hasattr(m, "predict_proba"):
                outs.append(m.predict_proba(p.reshape(-1, 1))[:, 1])
            else:
                # isotonic: predict returns calibrated prob directly
                outs.append(m.predict(p))
        return np.clip(np.mean(outs, axis=0), 1e-9, 1 - 1e-9)

    rf_c  = calibrate(rf_prob,  rf_lr_models)
    et_c  = calibrate(et_prob,  et_lr_models)
    tf_c  = calibrate(tf_prob,  tf_lr_models)
    ff_c  = calibrate(ff_prob,  ff_lr_models)
    cnn_c = calibrate(cnn_prob, cnn_lr_models)

    # --- 5) Build the EXACT 17-D meta feature vector used in training ---
    P = np.column_stack([rf_c, et_c, tf_c, ff_c, cnn_c])     # [N, 5]
    L = safe_logit(P)                                        # [N, 5]
    V = (P >= 0.5).astype(np.float64)                        # [N, 5]
    mean = P.mean(axis=1, keepdims=True)                     # [N, 1]
    std  = P.std(axis=1, keepdims=True)                      # [N, 1]
    X_meta_new = np.concatenate([P, L, V, mean, std], axis=1)  # [N, 17]

    return X_meta_new



# -------------------------
# Main training pipeline
# -------------------------

def main():
    """
    Full training pipeline with OOF stacking and per-fold metrics.
    """

    # Local calibrator helpers
    class IdentityCal(object):
        """Fallback calibrator when isotonic cannot fit; returns input unchanged."""
        def predict(self, x):
            return x

    def fit_calibrator(train_probs, y_train, method="isotonic"):
        """Fit a per-model probability calibrator on TRAIN only."""
        x = np.asarray(train_probs, dtype=np.float64).ravel()
        y = np.asarray(y_train, dtype=np.int32).ravel()

        # Degenerate cases: constant probs or single-class labels
        if np.unique(x).size < 2 or np.unique(y).size < 2:
            return IdentityCal(), "identity"

        if method == "isotonic":
            cal = IsotonicRegression(out_of_bounds="clip")
            cal.fit(x, y)
            return cal, "isotonic"
        else:
            lr = LogisticRegression(solver="lbfgs", C=1.0, max_iter=200)
            lr.fit(x.reshape(-1, 1), y)
            return lr, "platt"

    def apply_calibrator(cal, kind, probs):
        """Apply a fitted calibrator to raw probabilities."""
        x = np.asarray(probs, dtype=np.float64).ravel()
        if kind == "isotonic":
            return cal.predict(x)
        elif kind == "platt":
            return cal.predict_proba(x.reshape(-1, 1))[:, 1]
        else:
            # identity
            return x

    def build_meta_features(prob_matrix):
        """
        Given array shape [N, 5] = calibrated probs for (RF, ET, TF, FF, CNN),
        return a richer feature matrix:
          [probs, logits, votes, mean, std]
        """
        P = np.asarray(prob_matrix, dtype=np.float64)
        L = safe_logit(P)
        V = (P >= 0.5).astype(np.float32)
        mean = P.mean(axis=1, keepdims=True)
        std = P.std(axis=1, keepdims=True)
        return np.concatenate([P, L, V, mean, std], axis=1)

    # Setup
    warnings.filterwarnings("ignore")
    ensure_dir("models")
    ensure_dir("plots")

    t0 = time.time()
    print("[0.00s] Starting...")

    # Load data
    data_path = os.path.join("data", "pima-indians-diabetes.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError("Missing data file: data/pima-indians-diabetes.csv")

    X_df = pd.read_csv(data_path, header=None)
    y = X_df.iloc[:, -1].values.astype(np.int64)
    X_df = X_df.iloc[:, :-1]

    # Engineered features as in your strong build
    X_df["Glucose_BMI"] = X_df[1] * X_df[5]
    X_df["G_to_Pressure"] = X_df[1] / (X_df[2] + 1.0)
    X_df["BMI_sq"] = X_df[5] ** 2
    base_count = X_df.shape[1] - 3
    feature_names = [str(i) for i in range(base_count)] + ["Glucose_BMI", "G_to_Pressure", "BMI_sq"]

    # Optional synthetic
    synth_path = os.path.join("data", "synthetic_pima_data.csv")
    if os.path.exists(synth_path):
        synth = pd.read_csv(synth_path)
        X_s = synth.iloc[:, :-1].values
        y_s = synth["Outcome"].values.astype(np.int64)
        X_df = pd.concat([X_df, pd.DataFrame(X_s, columns=X_df.columns)], ignore_index=True)
        y = np.concatenate([y, y_s])
        print("Synthetic loaded: {} rows appended. New train size {}.".format(len(y_s), len(y)))

    # Impute zeros in 1..5 with medians
    for col in [1, 2, 3, 4, 5]:
        col_vals = X_df[col].replace(0, np.nan)
        med = np.nanmedian(col_vals.values)
        if not np.isfinite(med):
            med = float(col_vals.median())
        X_df[col] = X_df[col].replace(0, med)

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df.values.astype(np.float32))
    print("[{:.2f}s] Data prepared. Train rows {}, positives {}.".format(time.time() - t0, len(y), int(y.sum())))

    # Torch tensors for QM and NNs
    X_all_t = torch.tensor(X_scaled, dtype=torch.float32, device=device)
    y_all_t = torch.tensor(y, dtype=torch.long, device=device)

    # CV
    K = int(CFG["skfold"])
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED)

    # OOF calibrated probs: RF, ET, TF, FF, CNN
    oof_base = np.zeros((len(y), 5), dtype=np.float64)

    # Per-fold metrics store (now includes Precision, Recall for plotting)
    metrics = {
        "RF":   {"F1": [], "AUC": [], "Precision": [], "Recall": []},
        "ET":   {"F1": [], "AUC": [], "Precision": [], "Recall": []},
        "TF":   {"F1": [], "AUC": [], "Precision": [], "Recall": []},
        "FF":   {"F1": [], "AUC": [], "Precision": [], "Recall": []},
        "CNN":  {"F1": [], "AUC": [], "Precision": [], "Recall": []},
        "Meta": {"F1": [], "AUC": [], "Precision": [], "Recall": []},
    }

    # For optional ROC/PR plotting (OOF meta and per-fold meta)
    folds_meta = []  # list of (y_val, p_meta_val)

    prev_qm_state = None

    # -------------------------
    # CV loop
    # -------------------------
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_scaled, y), start=1):
        print("\n----- Fold {}/{} -----".format(fold, K))

        X_tr = torch.tensor(X_scaled[tr_idx], dtype=torch.float32, device=device)
        y_tr = torch.tensor(y[tr_idx], dtype=torch.long, device=device)
        X_va = torch.tensor(X_scaled[va_idx], dtype=torch.float32, device=device)
        y_va_np = y[va_idx].astype(np.int64)

        # Build QM and self-improve
        qm = QISICGM(input_dim=X_tr.shape[1], embed_dim=CFG["embed_dim"]).to(device)
        qm.initialize_graph(X_tr, y_tr, k=15)
        if CFG.get("warm_start_qm", False) and prev_qm_state is not None:
            qm.load_state_dict(prev_qm_state, strict=False)

        print(".. self_improve start")
        qm.self_improve(
            X_tr, y_tr,
            steps=CFG["self_improve_steps"],
            lr=CFG["self_improve_lr"],
            prune_every=999999,
            min_degree=1,
            verbose=True,
            use_focal=CFG.get("use_focal", True),
            focal_gamma=CFG.get("focal_gamma", 2.0),
        )
        print(".. self_improve done")
        prev_qm_state = qm.state_dict()

        # Embeddings for trees
        Z_tr = qm.get_embedding(X_tr).detach().cpu().numpy()
        Z_va = qm.get_embedding(X_va).detach().cpu().numpy()
        Z_tr = np.nan_to_num(Z_tr, copy=False, posinf=1e6, neginf=-1e6)
        Z_va = np.nan_to_num(Z_va, copy=False, posinf=1e6, neginf=-1e6)

        # Base models
        print(".. RF start")
        rf = RandomForestClassifier(
            n_estimators=CFG["rf_trees"],
            max_depth=CFG["rf_max_depth"],
            min_samples_split=3,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1
        ).fit(Z_tr, y[tr_idx])
        print(".. RF done")

        print(".. ET start")
        et = ExtraTreesClassifier(
            n_estimators=CFG["et_trees"],
            max_depth=CFG["et_max_depth"],
            min_samples_split=3,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1
        ).fit(Z_tr, y[tr_idx])
        print(".. ET done")

        print(".. Transformer start")
        tf = train_transformer(
            qm, X_tr, y_tr,
            max_len=CFG["transformer_seq_len"],
            epochs=CFG["transformer_epochs"],
            batch_size=CFG["transformer_batch"],
            lr=CFG["transformer_lr"],
            nhead=CFG["transformer_nhead"],
            num_layers=CFG["transformer_layers"]
        )
        print(".. Transformer done")

        print(".. FFNN start")
        ff = train_ffnn(
            qm, X_tr, y_tr,
            k_avg=CFG["ff_kavg"],
            epochs=CFG["ffnn_epochs"],
            batch_size=CFG["ffnn_batch"],
            lr=CFG["ffnn_lr"],
            hidden=CFG["ffnn_hidden"]
        )
        print(".. FFNN done")

        print(".. CNN-Seq start")
        cnn = train_cnn_seq(
            qm, X_tr, y_tr,
            max_len=CFG["transformer_seq_len"],
            epochs=min(CFG["cnn_epochs"], CFG["ffnn_epochs"]),
            batch_size=CFG["ffnn_batch"],
            lr=CFG["ffnn_lr"],
            hidden=CFG["ffnn_hidden"]
        )
        print(".. CNN-Seq done")

        # TRAIN probs (for calibrators)
        rf_tr = rf.predict_proba(Z_tr)[:, 1]
        et_tr = et.predict_proba(Z_tr)[:, 1]
        tf_tr = predict_transformer_mc(tf, qm, X_tr, max_len=CFG["transformer_seq_len"], mc_passes=CFG["mc_passes"])
        ff_tr = predict_ffnn_mc(ff, qm, X_tr, k_avg=CFG["ff_kavg"], mc_passes=CFG["mc_passes"])
        cn_tr = predict_cnn_seq_mc(cnn, qm, X_tr, max_len=CFG["transformer_seq_len"], mc_passes=CFG["mc_passes"])

        # VAL probs (to be calibrated and scored)
        rf_va = rf.predict_proba(Z_va)[:, 1]
        et_va = et.predict_proba(Z_va)[:, 1]
        tf_va = predict_transformer_mc(tf, qm, X_va, max_len=CFG["transformer_seq_len"], mc_passes=CFG["mc_passes"])
        ff_va = predict_ffnn_mc(ff, qm, X_va, k_avg=CFG["ff_kavg"], mc_passes=CFG["mc_passes"])
        cn_va = predict_cnn_seq_mc(cnn, qm, X_va, max_len=CFG["transformer_seq_len"], mc_passes=CFG["mc_passes"])

        # Fit per-model calibrators on TRAIN; calibrate TRAIN and VAL
        cal_rf, kind_rf = fit_calibrator(rf_tr, y[tr_idx], method="isotonic")
        cal_et, kind_et = fit_calibrator(et_tr, y[tr_idx], method="isotonic")
        cal_tf, kind_tf = fit_calibrator(tf_tr, y[tr_idx], method="isotonic")
        cal_ff, kind_ff = fit_calibrator(ff_tr, y[tr_idx], method="isotonic")
        cal_cn, kind_cn = fit_calibrator(cn_tr, y[tr_idx], method="isotonic")

        rf_va_cal = apply_calibrator(cal_rf, kind_rf, rf_va)
        et_va_cal = apply_calibrator(cal_et, kind_et, et_va)
        tf_va_cal = apply_calibrator(cal_tf, kind_tf, tf_va)
        ff_va_cal = apply_calibrator(cal_ff, kind_ff, ff_va)
        cn_va_cal = apply_calibrator(cal_cn, kind_cn, cn_va)

        # Write calibrated VAL probs into OOF matrix
        oof_base[va_idx, 0] = rf_va_cal
        oof_base[va_idx, 1] = et_va_cal
        oof_base[va_idx, 2] = tf_va_cal
        oof_base[va_idx, 3] = ff_va_cal
        oof_base[va_idx, 4] = cn_va_cal

        # Per-model metrics (F1/AUC + P/R at best F1 threshold)
        def add_fold_metrics(name, probs, y_true):
            thr, f1, pr, rc = find_best_f1(y_true, probs, recall_floor=CFG.get("recall_floor", None))
            auc = roc_auc_score(y_true, probs)
            metrics[name]["F1"].append(f1)
            metrics[name]["AUC"].append(auc)
            metrics[name]["Precision"].append(pr)
            metrics[name]["Recall"].append(rc)
            print("{}: F1={:.4f} AUC={:.4f} thr={:.3f}".format(name, f1, auc, thr))

        add_fold_metrics("RF", rf_va_cal, y_va_np)
        add_fold_metrics("ET", et_va_cal, y_va_np)
        add_fold_metrics("TF", tf_va_cal, y_va_np)
        add_fold_metrics("FF", ff_va_cal, y_va_np)
        add_fold_metrics("CNN", cn_va_cal, y_va_np)

        # Per-fold meta: train on TRAIN base-features, eval on VAL
        rf_tr_cal = apply_calibrator(cal_rf, kind_rf, rf_tr)
        et_tr_cal = apply_calibrator(cal_et, kind_et, et_tr)
        tf_tr_cal = apply_calibrator(cal_tf, kind_tf, tf_tr)
        ff_tr_cal = apply_calibrator(cal_ff, kind_ff, ff_tr)
        cn_tr_cal = apply_calibrator(cal_cn, kind_cn, cn_tr)

        X_meta_tr = build_meta_features(np.stack([rf_tr_cal, et_tr_cal, tf_tr_cal, ff_tr_cal, cn_tr_cal], axis=1))
        X_meta_va = build_meta_features(np.stack([rf_va_cal, et_va_cal, tf_va_cal, ff_va_cal, cn_va_cal], axis=1))

        meta_fold = LogisticRegression(max_iter=2000, solver="lbfgs", C=CFG.get("meta_C", 0.5), random_state=SEED)
        meta_fold.fit(X_meta_tr, y[tr_idx])
        meta_va = meta_fold.predict_proba(X_meta_va)[:, 1]

        thr_m, f1_m, pr_m, rc_m = find_best_f1(y_va_np, meta_va, recall_floor=CFG.get("recall_floor", None))
        auc_m = roc_auc_score(y_va_np, meta_va)
        metrics["Meta"]["F1"].append(f1_m)
        metrics["Meta"]["AUC"].append(auc_m)
        metrics["Meta"]["Precision"].append(pr_m)
        metrics["Meta"]["Recall"].append(rc_m)
        print("Meta: F1={:.4f} AUC={:.4f} thr={:.3f}".format(f1_m, auc_m, thr_m))

        folds_meta.append((y_va_np.copy(), meta_va.copy()))

        # Save a concept-graph snapshot for this fold (fast spring layout)
        if plot_graph_snapshot:
            outp = os.path.join("plots", f"concept_graph_fold{fold}.png")
            print(f"\n.. Saving concept graph (spring) -> {outp}", flush=True)
            plot_graph_snapshot(qm, out_png=outp, seed=SEED + fold, iterations=30)
            print(f".. Concept graph saved: {outp}", flush=True)

    # -------------------------
    # Global OOF meta + threshold
    # -------------------------
    X_meta_oof = np.concatenate([
        build_meta_features(oof_base)  # already calibrated probs
    ], axis=0)
    meta_oof = LogisticRegression(max_iter=2000, solver="lbfgs", C=CFG.get("meta_C", 0.5), random_state=SEED)
    meta_oof.fit(X_meta_oof, y)
    oof_meta = meta_oof.predict_proba(X_meta_oof)[:, 1]

    thr_oof, f1_oof, pr_oof, rc_oof = find_best_f1(y, oof_meta, recall_floor=CFG.get("recall_floor", None))
    thr_bal, _, tp_b, fp_b, fn_b, tn_b, f1_bal = pick_threshold_with_guard(
        y, oof_meta, max_rel_f1_drop=0.01, prefer="specificity")
    auc_oof = roc_auc_score(y, oof_meta)

    print("\n.. Global OOF meta results")
    print(f"OOF Meta F1={f1_oof:.4f} AUC={auc_oof:.4f} thr={thr_oof:.4f}")
    print(f"[balanced] thr={thr_bal:.4f} F1={f1_bal:.4f}  (<=1% below max F1)")


    # -------------------------
    # Per-fold summary table
    # -------------------------
    def print_summary(name):
        f1s = metrics[name]["F1"]; aucs = metrics[name]["AUC"]
        print("{} per-fold F1: {}".format(name, ["{:.4f}".format(v) for v in f1s]))
        print("{} per-fold AUC: {}".format(name, ["{:.4f}".format(v) for v in aucs]))
        print("{} mean F1={:.4f} mean AUC={:.4f}".format(name, float(np.mean(f1s)), float(np.mean(aucs))))

    print("\n=== Per-fold metrics (VAL) ===")
    for m in ["RF", "ET", "TF", "FF", "CNN", "Meta"]:
        print_summary(m)

    # -------------------------
    # Optional rich plots (if module present)
    # -------------------------
    if make_all_plots is not None:
        # Convert metrics to the expected lower-case dict with f1/precision/recall/auc lists
        fold_metrics = {
            "rf":   {"f1": metrics["RF"]["F1"],   "precision": metrics["RF"]["Precision"],   "recall": metrics["RF"]["Recall"],   "auc": metrics["RF"]["AUC"]},
            "et":   {"f1": metrics["ET"]["F1"],   "precision": metrics["ET"]["Precision"],   "recall": metrics["ET"]["Recall"],   "auc": metrics["ET"]["AUC"]},
            "tf":   {"f1": metrics["TF"]["F1"],   "precision": metrics["TF"]["Precision"],   "recall": metrics["TF"]["Recall"],   "auc": metrics["TF"]["AUC"]},
            "ff":   {"f1": metrics["FF"]["F1"],   "precision": metrics["FF"]["Precision"],   "recall": metrics["FF"]["Recall"],   "auc": metrics["FF"]["AUC"]},
            "cnn":  {"f1": metrics["CNN"]["F1"],  "precision": metrics["CNN"]["Precision"],  "recall": metrics["CNN"]["Recall"],  "auc": metrics["CNN"]["AUC"]},
            "meta": {"f1": metrics["Meta"]["F1"], "precision": metrics["Meta"]["Precision"], "recall": metrics["Meta"]["Recall"], "auc": metrics["Meta"]["AUC"]},
        }
        try:
            make_all_plots(
                y_oof_true=y,
                p_oof_meta=oof_meta,
                meta_threshold=thr_oof,
                meta_threshold_bal=thr_bal,
                folds_meta=folds_meta,
                fold_metrics=fold_metrics,
                out_dir="plots"
            )
        except Exception as e:
            warnings.warn(f"make_all_plots failed: {e}")

    # -------------------------
    # Final refit on ALL data + save artifacts
    # -------------------------
    print("\n.. Final refit for demo predictions start")

    # Refit QM on all data
    qm_final = QISICGM(input_dim=X_all_t.shape[1], embed_dim=CFG["embed_dim"]).to(device)
    qm_final.initialize_graph(X_all_t, y_all_t, k=15)
    qm_final.self_improve(
        X_all_t, y_all_t,
        steps=max(CFG["self_improve_steps"], 48),
        lr=CFG["self_improve_lr"],
        prune_every=999999, min_degree=1, verbose=False,
        use_focal=CFG.get("use_focal", True), focal_gamma=CFG.get("focal_gamma", 2.0)
    )
    torch.save(qm_final.state_dict(), "models/qm_final.pth")

    # Embeddings for ALL
    Z_all = qm_final.get_embedding(X_all_t).detach().cpu().numpy()
    Z_all = np.nan_to_num(Z_all, copy=False, posinf=1e6, neginf=-1e6)
    np.save("models/embeddings.npy", Z_all)

    # Base models on ALL
    rf_final = RandomForestClassifier(
        n_estimators=CFG["rf_trees"], max_depth=CFG["rf_max_depth"],
        min_samples_split=3, class_weight="balanced", random_state=SEED, n_jobs=-1
    ).fit(Z_all, y)
    with open("models/rf_final.pkl", "wb") as f:
        pickle.dump(rf_final, f)

    et_final = ExtraTreesClassifier(
        n_estimators=CFG["et_trees"], max_depth=CFG["et_max_depth"],
        min_samples_split=3, class_weight="balanced", random_state=SEED, n_jobs=-1
    ).fit(Z_all, y)
    with open("models/et_final.pkl", "wb") as f:
        pickle.dump(et_final, f)

    tf_final = train_transformer(
        qm_final, X_all_t, y_all_t,
        max_len=CFG["transformer_seq_len"],
        epochs=CFG["transformer_epochs"],
        batch_size=CFG["transformer_batch"],
        lr=CFG["transformer_lr"],
        nhead=CFG["transformer_nhead"],
        num_layers=CFG["transformer_layers"]
    )
    torch.save(tf_final.state_dict(), "models/tf_final.pth")

    ff_final = train_ffnn(
        qm_final, X_all_t, y_all_t,
        k_avg=CFG["ff_kavg"],
        epochs=CFG["ffnn_epochs"],
        batch_size=CFG["ffnn_batch"],
        lr=CFG["ffnn_lr"],
        hidden=CFG["ffnn_hidden"]
    )
    torch.save(ff_final.state_dict(), "models/ffnn_final.pth")

    cnn_final = train_cnn_seq(
        qm_final, X_all_t, y_all_t,
        max_len=CFG["transformer_seq_len"],
        epochs=min(CFG["cnn_epochs"], CFG["ffnn_epochs"]),
        batch_size=CFG["ffnn_batch"],
        lr=CFG["ffnn_lr"],
        hidden=CFG["ffnn_hidden"]
    )
    torch.save(cnn_final.state_dict(), "models/cnn_final.pth")

    # ALL-data raw probs
    rf_all = rf_final.predict_proba(Z_all)[:, 1]
    et_all = et_final.predict_proba(Z_all)[:, 1]
    tf_all = predict_transformer_mc(tf_final, qm_final, X_all_t, max_len=CFG["transformer_seq_len"], mc_passes=CFG["mc_passes"])
    ff_all = predict_ffnn_mc(ff_final, qm_final, X_all_t, k_avg=CFG["ff_kavg"], mc_passes=CFG["mc_passes"])
    cn_all = predict_cnn_seq_mc(cnn_final, qm_final, X_all_t, max_len=CFG["transformer_seq_len"], mc_passes=CFG["mc_passes"])

    # Fit final calibrators on ALL data
    def fit_final_cal(x, y_true):
        try:
            cal = IsotonicRegression(out_of_bounds="clip").fit(x, y_true)
            kind = "isotonic"
        except Exception:
            lr = LogisticRegression(solver="lbfgs", C=1.0, max_iter=200).fit(x.reshape(-1, 1), y_true)
            cal, kind = lr, "platt"
        return cal, kind

    rf_cal_f, rf_kind_f = fit_final_cal(rf_all, y)
    et_cal_f, et_kind_f = fit_final_cal(et_all, y)
    tf_cal_f, tf_kind_f = fit_final_cal(tf_all, y)
    ff_cal_f, ff_kind_f = fit_final_cal(ff_all, y)
    cn_cal_f, cn_kind_f = fit_final_cal(cn_all, y)

    # Build ALL-data calibrated base feature matrix and fit final meta
    def apply_final(cal, kind, arr):
        if kind == "isotonic":
            return cal.predict(arr)
        return cal.predict_proba(arr.reshape(-1, 1))[:, 1]

    rf_all_cal = apply_final(rf_cal_f, rf_kind_f, rf_all)
    et_all_cal = apply_final(et_cal_f, et_kind_f, et_all)
    tf_all_cal = apply_final(tf_cal_f, tf_kind_f, tf_all)
    ff_all_cal = apply_final(ff_cal_f, ff_kind_f, ff_all)
    cn_all_cal = apply_final(cn_cal_f, cn_kind_f, cn_all)

    X_meta_all = np.column_stack([
        rf_all_cal, et_all_cal, tf_all_cal, ff_all_cal, cn_all_cal
    ])
    X_meta_all = np.concatenate([
        X_meta_all,
        safe_logit(X_meta_all),
        (X_meta_all >= 0.5).astype(np.float32),
        X_meta_all.mean(axis=1, keepdims=True),
        X_meta_all.std(axis=1, keepdims=True)
    ], axis=1)

    meta_final = LogisticRegression(max_iter=2000, solver="lbfgs", C=CFG.get("meta_C", 0.5), random_state=SEED)
    meta_final.fit(X_meta_all, y)

    # Save everything for the demo script (keep keys it expects)
    with open("models/meta_oof.pkl", "wb") as f:
        pickle.dump({
            "meta": meta_final,
            "threshold": float(thr_oof),            # default (recall-first)
            "threshold_alt": float(thr_bal),        # balanced threshold"
            "scaler": scaler,
            "feature_names": feature_names,
            "cfg": CFG,
            "profile": PROFILE,
            # Per-model calibrators (lists)
            "rf_lr_models": [rf_cal_f],
            "et_lr_models": [et_cal_f],
            "tf_lr_models": [tf_cal_f],
            "ff_lr_models": [ff_cal_f],
            "cnn_lr_models": [cn_cal_f],
            # Calibrator kinds
            "rf_cal_kind": rf_kind_f,
            "et_cal_kind": et_kind_f,
            "tf_cal_kind": tf_kind_f,
            "ff_cal_kind": ff_kind_f,
            "cn_cal_kind": cn_kind_f,
        }, f)

    print("Final refit for demo predictions done")
    print("Done. Total time {:.2f}s".format(time.time() - t0))


if __name__ == "__main__":
    main()
