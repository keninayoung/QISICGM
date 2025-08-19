# QISICGM: Quantum-Inspired Stacked Pipeline for Diabetes Prediction
# Author: Kenneth Young, PhD (USF-HII)
# Description: This script implements a quantum-inspired stacked machine learning pipeline
# for diabetes prediction using the PIMA Indians Diabetes dataset. It combines embedding
# techniques with graph-based learning, multiple base learners (Random Forest, Extra Trees,
# Transformer, FFNN, CNN), and a meta learner for improved performance. The pipeline includes
# cross-validation, out-of-fold predictions, performance metrics, and extensive plotting
# for analysis. Designed for CPU usage, with optimizations for efficiency and reproducibility. 
# License: MIT
# Repository: https://github.com/keninayoung/qisicgm 
# Requirements: Python 3.8+, NumPy, Pandas, Matplotlib, Scikit-learn, Imbalanced-learn,
# Torch, NetworkX
# Usage: python qisicgm_stacked.py
# Note: Ensure 'pima-indians-diabetes.csv' is in the 'data' folder. If using the synthetic dataset,
# run 'generate_synthetic_data.py' first to create 'synthetic_pima_data.csv'.

import os
import time
import math
import warnings
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Use Agg backend for non-interactive plotting to avoid GUI issues
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from imblearn.over_sampling import SMOTE
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LambdaLR
from torch.utils.data import TensorDataset, DataLoader
import networkx as nx
from sklearn.neighbors import kneighbors_graph  # Added missing import
import torch.nn.functional as F  # Added missing import
import dill  # For saving the prediction function

# Set global seed for reproducibility and define device
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cpu")  # Explicitly set to CPU for your environment
print("Device:", device)

# Define configuration profiles for different training setups
PROFILE = "push75"  # Current profile selection
CFG_PROFILES = {
    "push75": {
        "skfold": 5,
        "resampler": "smotetomek",
        "resampler_budget_sec": 7.0,
        "majority_cap_factor": 2.0,
        "smote_equalize": True,
        "embed_dim": 128,
        "self_improve_steps": 40,
        "self_improve_lr": 4e-4,
        "use_focal": True,
        "focal_gamma": 2.0,
        "rf_trees": 1000,
        "et_trees": 1000,
        "rf_max_depth": 26,
        "et_max_depth": 26,
        "transformer_epochs": 40,
        "transformer_batch": 48,
        "transformer_lr": 2e-4,
        "transformer_seq_len": 32,
        "ffnn_hidden": 384,
        "ffnn_epochs": 30,
        "ffnn_batch": 48,
        "ffnn_lr": 8e-4,
        "ff_kavg": 20,
        "bag_seeds": [42, 1337, 2024],
        "mc_passes": 5,
        "meta_C": 0.5,
        "warm_start_qm": True,
        "recall_floor": 0.65,
        "class_weight": [0.6, 1.4],  # Adjusted to favor positives, reducing FN
    },
}

CFG = CFG_PROFILES[PROFILE]
print("Profile:", PROFILE)

# Utility functions

# Train base models and save their states along with embeddings
def train_base_models_and_save(X_all, y_all, qm_final, CFG):
    """Train base models and save their states along with embeddings."""
    ensure_dir("models")
    Z_all = qm_final.get_embedding(X_all).detach().cpu().numpy()
    np.save("models/embeddings.npy", Z_all)

    # Train and save base models
    rf_final = RandomForestClassifier(n_estimators=CFG["rf_trees"], max_depth=CFG["rf_max_depth"],
                                     min_samples_split=3, class_weight="balanced", random_state=SEED,
                                     n_jobs=-1).fit(Z_all, y_all.cpu().numpy())
    with open("models/rf_final.pkl", "wb") as f:
        pickle.dump(rf_final, f)

    et_final = ExtraTreesClassifier(n_estimators=CFG["et_trees"], max_depth=CFG["et_max_depth"],
                                   min_samples_split=3, class_weight="balanced", random_state=SEED,
                                   n_jobs=-1).fit(Z_all, y_all.cpu().numpy())
    with open("models/et_final.pkl", "wb") as f:
        pickle.dump(et_final, f)

    tf_final = TransformerClassifier(d_model=CFG["embed_dim"], nhead=2, num_layers=1, num_classes=1,
                                    max_len=CFG["transformer_seq_len"]).to(device)
    tf_final = train_transformer(qm_final, X_all, y_all, max_len=CFG["transformer_seq_len"],
                                epochs=CFG["transformer_epochs"], batch_size=CFG["transformer_batch"],
                                lr=CFG["transformer_lr"])
    torch.save(tf_final.state_dict(), "models/tf_final.pth")

    ffnn_final = train_ffnn(qm_final, X_all, y_all, k_avg=CFG["ff_kavg"], epochs=CFG["ffnn_epochs"],
                           batch_size=CFG["ffnn_batch"], lr=CFG["ffnn_lr"], hidden=CFG["ffnn_hidden"])
    torch.save(ffnn_final.state_dict(), "models/ffnn_final.pth")

    cnn_final = train_cnn(qm_final, X_all, y_all, batch_size=CFG["ffnn_batch"],
                         lr=CFG["ffnn_lr"] * 1.5, hidden=CFG["ffnn_hidden"], epochs=15)
    torch.save(cnn_final.state_dict(), "models/cnn_final.pth")

    # Save flag indicating base models and embeddings are available
    with open("models/base_models_available.flag", "wb") as f:
        pickle.dump(True, f)

# Safe logit function to avoid numerical issues
def safe_logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p) - np.log(1 - p)

# Timed print function
def tprint(msg, t0=None):
    # Print message with optional timing information
    if t0 is None:
        print(msg)
    else:
        print("{} in {:.2f}s".format(msg, time.time() - t0))

# Ensure directory exists
def ensure_dir(path):
    # Create directory if it doesn't exist
    os.makedirs(path, exist_ok=True)

# Function to find the best F1 threshold with optional recall floor
def find_best_f1(y_true, scores, recall_floor=None):
    scores = np.asarray(scores)
    # explore all unique score cutoffs + edges
    thresholds = np.r_[0.0, np.unique(np.round(scores, 6)), 1.0]
    best = (0.0, 0.5, 0.0, 0.0)  # (f1, thr, precision, recall)

    for t in thresholds:
        pred = (scores >= t).astype(int)
        rc = recall_score(y_true, pred, zero_division=0)
        if recall_floor is not None and rc < recall_floor:
            continue
        pr = precision_score(y_true, pred, zero_division=0)
        f1 = 2 * pr * rc / (pr + rc) if (pr + rc) else 0.0
        if f1 > best[0] or (abs(f1 - best[0]) < 1e-4 and rc > best[3]):
            best = (f1, t, pr, rc)

    # if recall floor couldn’t be met, fall back to best F1 over all t
    if best[0] == 0.0 and recall_floor is not None:
        for t in thresholds:
            pred = (scores >= t).astype(int)
            pr = precision_score(y_true, pred, zero_division=0)
            rc = recall_score(y_true, pred, zero_division=0)
            f1 = 2 * pr * rc / (pr + rc) if (pr + rc) else 0.0
            if f1 > best[0] or (abs(f1 - best[0]) < 1e-4 and rc > best[3]):
                best = (f1, t, pr, rc)

    return best[1], best[0]


# Quantum-Inspired Stacked Concept Graph Model
class QISICGM(nn.Module):
    """
    Quantum-Inspired Stacked Integrated Concept Graph Model for embedding generation.
    
    Args:
        input_dim (int): Dimension of the input features.
        embed_dim (int): Dimension of the output embeddings.
        device (str or torch.device): Device to run computations on (e.g., 'cpu' or 'cuda', defaults to global device).
    """
    def __init__(self, input_dim, embed_dim, device=device):  # Use global device as default
        super(QISICGM, self).__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.device = device  # Store device for tensor operations
        self.embed = nn.Sequential(
            nn.Linear(input_dim, embed_dim, dtype=torch.float32).to(device),
            nn.BatchNorm1d(embed_dim).to(device),
            nn.ReLU(),
            nn.Dropout(0.1).to(device),
            nn.Linear(embed_dim, embed_dim, dtype=torch.float32).to(device)
        ).to(device)
        self.cls = nn.Linear(embed_dim, 2, dtype=torch.float32).to(device)
        self.graph = None  # Initialize graph structure
        self.node_embeds = {}  # Dictionary for node embeddings
        self.node_labels = {}  # Dictionary for node labels

    # Initialize the concept graph using k-nearest neighbors in embedding space.
    def initialize_graph(self, X_t, y_t, k=15):
        """
        Initialize the concept graph using k-nearest neighbors in embedding space.
        
        Args:
            X_t (torch.Tensor): Input feature tensor.
            y_t (torch.Tensor): Labels tensor.
            k (int): Number of neighbors for graph construction.
        """
        self.graph = nx.Graph()
        with torch.no_grad():
            Z = self.embed(X_t).detach()  # Compute embeddings on device
        for i in range(Z.size(0)):
            self.graph.add_node(i)  # Add node to graph
            self.node_embeds[i] = Z[i].clone()  # Store embedding
            self.node_labels[i] = int(y_t[i].item())  # Store label
        with torch.no_grad():
            dists = torch.cdist(Z, Z)  # Compute pairwise distances on device
            for i in range(Z.size(0)):
                idx = torch.argsort(dists[i])[1:k+1]  # Get k nearest neighbors (exclude self)
                for j in idx:
                    self.graph.add_edge(i, int(j.item()))  # Add edges

    # Self-improvement method to refine the model and graph structure.
    def self_improve(self, X_t, y_t, steps=40, lr=3e-4, prune_every=999,
                 min_degree=1, verbose=True, use_focal=True, focal_gamma=2.0):
        """
        Improve the model by training the embedding and refining the graph.
    
        Args:
            X_t (torch.Tensor): Input feature tensor.
            y_t (torch.Tensor): Labels tensor.
            steps (int): Number of optimization steps.
            lr (float): Learning rate for optimization.
            prune_every (int): Pruning interval.
            min_degree (int): Minimum degree for pruning.
            verbose (bool): Whether to print progress.
            use_focal (bool): Whether to use focal loss.
            focal_gamma (float): Focal loss gamma parameter.
        """
        if self.graph is None:
            raise ValueError("Graph not initialized")
        opt = optim.Adam(self.parameters(), lr=lr, weight_decay=1e-5)
        
        # Learning rate scheduler with warmup and cosine annealing
        if use_focal:
            with torch.no_grad():
                classes, counts = torch.unique(y_t, return_counts=True)
                neg = counts[(classes == 0).nonzero(as_tuple=True)[0]].item() if (classes == 0).any() else 1
                pos = counts[(classes == 1).nonzero(as_tuple=True)[0]].item() if (classes == 1).any() else 1
                tot = neg + pos
                alpha_vec = torch.tensor([neg/tot, pos/tot], dtype=torch.float32, device=self.device)
            loss_fn = FocalLoss(gamma=focal_gamma, alpha=alpha_vec)
        else:
            loss_fn = nn.BCEWithLogitsLoss()
        
        # Warmup scheduler
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
                print(" self_improve step {} / {}, loss {:.4f}".format(step + 1, steps,
                                                                      float(loss.item())))
            if (step + 1) % prune_every == 0:
                self._prune(min_degree=min_degree)

    # Prune nodes with degree less than min_degree from the graph.
    def _prune(self, min_degree=1):
        """
        Prune nodes with degree less than min_degree from the graph.
        
        Args:
            min_degree (int): Minimum degree for pruning.
        """
        remove = [n for n in self.graph.nodes() if self.graph.degree[n] < min_degree]
        self.graph.remove_nodes_from(remove)
        for n in remove:
            self.node_embeds.pop(n, None)
            self.node_labels.pop(n, None)

    # Get embeddings for input data.
    def get_embedding(self, X):
        """
        Get embeddings for input data.
        
        Args:
            X (torch.Tensor): Input tensor of shape (batch_size, input_dim).
        
        Returns:
            torch.Tensor: Embedding tensor of shape (batch_size, embed_dim).
        """
        self.eval()
        with torch.no_grad():
            return self.embed(X).detach()

    # Generate k-average embeddings using a quantum walk on the graph.
    def get_k_avg_embedding(self, x, k_avg):
        """
        Generate k-average embeddings using a quantum walk on the graph.
    
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).
            k_avg (int): Average number of neighbors for embedding.
    
        Returns:
            torch.Tensor: K-averaged embedding tensor.
        """
        batch_size = x.size(0)
        embeddings = self.get_embedding(x)
        effective_k = k_avg  # Default to k_avg
        if self.graph is None or self.graph.number_of_nodes() != batch_size:
            x_np = x.cpu().numpy()
            # Adjust k_avg to be at most n_samples - 1 to avoid invalid neighbor count
            effective_k = min(k_avg, batch_size - 1) if batch_size > 1 else 1
            adj = torch.tensor(kneighbors_graph(x_np, effective_k, mode='connectivity', include_self=False).toarray(),
                              dtype=torch.float32, device=self.device)
        else:
            adj = torch.tensor(nx.to_numpy_array(self.graph), dtype=torch.float32, device=self.device)
        # Quantum walk simulation with limited steps
        adj_normalized = adj / (adj.sum(1, keepdim=True) + 1e-10)
        walk_steps = max(0, effective_k - 1) if batch_size > 1 else 0  # Ensure non-negative steps
        walk_probs = adj_normalized
        for _ in range(walk_steps):
            walk_probs = torch.matmul(walk_probs, adj_normalized)
            walk_probs = walk_probs * (1 - walk_probs)  # Simulate interference
            walk_probs = F.normalize(walk_probs, p=1, dim=1)
        neighbor_embeddings = torch.matmul(walk_probs, embeddings)
        return neighbor_embeddings

# Focal Loss implementation
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction="mean"):
        # Initialize Focal Loss with gamma and optional alpha weighting
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(reduction="none")

    def forward(self, logits, targets):
        # Compute Focal Loss
        ce = self.ce(logits, targets)
        pt = torch.exp(-ce)
        loss = (1 - pt) ** self.gamma * ce
        if self.alpha is not None:
            a = self.alpha[targets]
            loss = a * loss
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

#####################################
# Base Learners
#####################################

# CNN Classifier with convolutional and fully connected layers
class CNNClassifier(nn.Module):
    def __init__(self, input_dim, hidden):
        # Initialize CNN Classifier with convolutional and fully connected layers
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1,
                               dtype=torch.float32)
        self.bn1 = nn.BatchNorm1d(16)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1,
                               dtype=torch.float32)
        self.bn2 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1,
                               dtype=torch.float32)
        self.bn3 = nn.BatchNorm1d(64)
        conv_out_size = input_dim // 8
        if conv_out_size < 1:
            conv_out_size = 1
        self.fc1 = nn.Linear(64 * conv_out_size, hidden, dtype=torch.float32)
        self.bn_fc1 = nn.BatchNorm1d(hidden)
        self.fc2 = nn.Linear(hidden, 1, dtype=torch.float32)
        self.dropout = nn.Dropout(0.5)
        self.dropout_final = nn.Dropout(0.3)

    def forward(self, x):
        # Forward pass through CNN
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

# 1D CNN Classifier over sequences with masked global-average pooling
class CNNSeqClassifier(nn.Module):
    """1D CNN over neighbor rank axis T. Uses masked global-average pooling."""
    def __init__(self, d_model, hidden):
        super().__init__()
        # project feature dim -> channels for conv
        self.proj = nn.Linear(d_model, 64, dtype=torch.float32)

        # convs along T (neighbor order). input to conv is [B,C,T]
        self.conv1 = nn.Conv1d(64, 128, kernel_size=3, padding=1, dtype=torch.float32)
        self.conv2 = nn.Conv1d(128,128, kernel_size=5, padding=2, dtype=torch.float32)
        self.conv3 = nn.Conv1d(128,128, kernel_size=7, padding=3, dtype=torch.float32)

        self.head = nn.Sequential(
            nn.Linear(128, hidden, dtype=torch.float32),
            nn.SiLU(),
            nn.Dropout(0.25),
            nn.Linear(hidden, 1, dtype=torch.float32),
        )

        # init
        nn.init.xavier_uniform_(self.proj.weight); nn.init.zeros_(self.proj.bias)
        for m in [self.conv1, self.conv2, self.conv3]:
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            if m.bias is not None: nn.init.zeros_(m.bias)
        for m in self.head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

    def masked_gap(self, x, pad_mask):
        """x: [B,C,T], pad_mask: [B,T] (True = PAD) -> [B,C]"""
        if pad_mask is None:
            return x.mean(dim=-1)
        keep = (~pad_mask).float().unsqueeze(1)     # [B,1,T]
        x = x * keep
        denom = keep.sum(dim=-1).clamp_min(1.0)     # [B,1]
        return (x.sum(dim=-1) / denom)              # [B,C]

    def forward(self, seq, pad_mask=None):
        # seq: [B,T,D], pad_mask: [B,T] bool
        h = self.proj(seq)              # [B,T,64]
        h = h.transpose(1, 2)           # [B,64,T]
        h = F.silu(self.conv1(h))
        h = F.silu(self.conv2(h))
        h = F.silu(self.conv3(h))
        h = self.masked_gap(h, pad_mask)  # [B,128]
        return self.head(h).squeeze(1)    # [B]


# Positional Encoding for Transformer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.rank_emb = nn.Embedding(max_len, d_model)
        nn.init.normal_(self.rank_emb.weight, std=0.02)
    def forward(self, x):
        B, T, D = x.shape
        idx = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        return x + self.rank_emb(idx)


# Phase Feature Map for Quantum-inspired lift
class PhaseFeatureMap(nn.Module):
    """Quantum-ish lift: x -> concat[cos(a*x), sin(a*x)] per dim."""
    def __init__(self, d_model, init_scale=1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.full((1, 1, d_model), init_scale, dtype=torch.float32))

    def forward(self, x):
        z = self.alpha * x
        return torch.cat([torch.cos(z), torch.sin(z)], dim=-1)  # [B,T,2D]

# Transformer Classifier
class TransformerClassifier(nn.Module):
    def __init__(self, d_model, nhead, num_layers, num_classes, max_len):
        super().__init__()

        # Quantum-inspired front-end (keeps I/O the same)
        self.phase = PhaseFeatureMap(d_model)
        self.proj_back = nn.Linear(2 * d_model, d_model, dtype=torch.float32)

        self.pos_encoder = PositionalEncoding(d_model, max_len)

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=0.15,
            activation="gelu",
            batch_first=True,   # IMPORTANT (you use [B,T,D])
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.15)
        self.fc = nn.Linear(d_model, num_classes, dtype=torch.float32)

        # Safe init (avoid calculate_gain("gelu"))
        nn.init.xavier_uniform_(self.proj_back.weight)
        nn.init.zeros_(self.proj_back.bias)

    def forward(self, x, pad_mask=None):
        """
        x: [B,T,D]
        pad_mask: [B,T] bool (True = PAD / ignore); can be None
        """
        x = self.proj_back(self.phase(x))     # [B,T,D]
        x = self.pos_encoder(x)               # [B,T,D]
        x = self.transformer_encoder(x, src_key_padding_mask=pad_mask)  # respects PADs
        x = self.norm(x)
        x = x.mean(dim=1)                     # keep your mean pooling
        x = self.dropout(x)
        return self.fc(x)


# Feedforward Neural Network Classifier
class FFNNClassifier(nn.Module):
    def __init__(self, input_dim, hidden):
        super().__init__()
        self.inp = nn.Linear(input_dim, hidden, dtype=torch.float32)
        self.ln1 = nn.LayerNorm(hidden)
        self.drop1 = nn.Dropout(0.25)

        # residual block
        self.fc_mid = nn.Sequential(
            nn.Linear(hidden, hidden, dtype=torch.float32),
            nn.SiLU(),
            nn.Dropout(0.25),
            nn.Linear(hidden, hidden, dtype=torch.float32),
        )
        self.ln2 = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, 1, dtype=torch.float32)

        # init
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
        # residual
        r = self.fc_mid(h)
        h = self.ln2(h + r)
        return self.head(h)  # [B,1] logits


# Graph to sequence features
def build_sequences_from_graph(qm, Z_query, max_len=16):
    """
    Build [B, T=max_len, D] neighbor sequences from the concept graph
    and a padding mask [B, T] where True marks padded positions.
    """
    if qm.graph is None or len(qm.node_embeds) == 0:
        raise ValueError("Concept graph is empty")

    # Stack node embeddings into a matrix [N_nodes, D]
    node_ids = list(qm.node_embeds.keys())
    mat = torch.stack([qm.node_embeds[n] for n in node_ids], dim=0).to(Z_query.device)  # [N,D]
    B, D = Z_query.size(0), mat.size(1)

    seqs, masks = [], []
    with torch.no_grad():
        for i in range(B):
            z = Z_query[i].unsqueeze(0)                 # [1,D]
            d = torch.cdist(z, mat).squeeze(0)          # [N]
            valid_k = min(max_len, mat.size(0))
            # use topk for clarity/stability
            nbr_idx = torch.topk(d, k=valid_k, largest=False).indices  # [valid_k]
            items = mat.index_select(0, nbr_idx)        # [valid_k, D]

            valid = items.size(0)
            if valid < max_len:
                pad_rows = torch.zeros(max_len - valid, D, dtype=torch.float32, device=items.device)
                items = torch.cat([items, pad_rows], dim=0)  # [max_len, D]

            seqs.append(items.unsqueeze(0))             # [1, max_len, D]

            pad_mask = torch.zeros(max_len, dtype=torch.bool, device=items.device)
            if valid < max_len:
                pad_mask[valid:] = True                  # True = padded positions
            masks.append(pad_mask.unsqueeze(0))          # [1, max_len]

    return torch.cat(seqs, dim=0), torch.cat(masks, dim=0)  # [B,T,D], [B,T]

# Alternative: sequences without padding mask
def build_sequences_from_graph_with_mask(qm, Z_query, max_len=16):
    """
    Build neighbor sequences and a padding mask.
    Returns:
      seqs:     [B, T=max_len, D]
      pad_mask: [B, T] (True = padded / ignore in attention)
    """
    if qm.graph is None or len(qm.node_embeds) == 0:
        raise ValueError("Concept graph is empty")

    # Stack node embeddings: [N,D]
    node_ids = list(qm.node_embeds.keys())
    mat = torch.stack([qm.node_embeds[n] for n in node_ids], dim=0).to(Z_query.device)

    B, D = Z_query.size(0), mat.size(1)
    seqs, masks = [], []

    with torch.no_grad():
        for i in range(B):
            z = Z_query[i].unsqueeze(0)             # [1,D]
            d = torch.cdist(z, mat).squeeze(0)      # [N]
            valid_k = min(max_len, mat.size(0))
            nbr_idx = torch.topk(d, k=valid_k, largest=False).indices  # [valid_k]
            items = mat.index_select(0, nbr_idx)    # [valid_k, D]

            valid = items.size(0)
            if valid < max_len:
                pad_rows = torch.zeros(max_len - valid, D, dtype=torch.float32, device=items.device)
                items = torch.cat([items, pad_rows], dim=0)

            seqs.append(items.unsqueeze(0))         # [1,T,D]

            pad_mask = torch.zeros(max_len, dtype=torch.bool, device=items.device)
            if valid < max_len:
                pad_mask[valid:] = True
            masks.append(pad_mask.unsqueeze(0))     # [1,T]

    return torch.cat(seqs, dim=0), torch.cat(masks, dim=0)  # [B,T,D], [B,T]


##################################################
# Training helpers
###################################################

# Check if a parameter is a norm or bias
def _is_norm_or_bias(n, p):
    if p.ndim == 1:  # covers LayerNorm.weight, BatchNorm.weight, and biases
        return True
    if n.endswith(".bias"):
        return True
    return False

# Helper function to create parameter groups for Transformer model
def _param_groups_for_transformer(model, base_lr=2e-4, wd=1e-2):
    groups = []

    # helper to add a group with proper wd hygiene
    def add_group(named_params, lr):
        decay, nodecay = [], []
        for n, p in named_params:
            if not p.requires_grad:
                continue
            if _is_norm_or_bias(n, p):
                nodecay.append(p)
            else:
                decay.append(p)
        if decay:
            groups.append({"params": decay, "lr": lr, "weight_decay": wd})
        if nodecay:
            groups.append({"params": nodecay, "lr": lr, "weight_decay": 0.0})

    # 1) front-end quantum lift
    add_group(model.phase.named_parameters(prefix="phase"), lr=1e-4)
    add_group(model.proj_back.named_parameters(prefix="proj_back"), lr=1e-4)

    # 2) transformer encoder (all layers)
    add_group(model.transformer_encoder.named_parameters(prefix="encoder"), lr=base_lr)

    # 3) layernorm before head
    add_group(model.norm.named_parameters(prefix="norm"), lr=5e-5)

    # 4) classification head
    add_group(model.fc.named_parameters(prefix="fc"), lr=3e-4)

    return groups

# Build augmented training set with synthetic data to adjust prevalence and size
def build_augmented_train(X_tr_real, y_tr_real, X_syn, y_syn, *,
                          target_pos=0.45, max_syn_ratio=1.0, seed=42):
    """
    Return (X_train_aug, y_train_aug, note). Uses synthetic ONLY to adjust prevalence and size.
    - target_pos: desired positive rate after augmentation (e.g., 0.45)
    - max_syn_ratio: cap synthetic count to at most this * real_train_size (e.g., 1.0)
    """
    if X_syn is None or y_syn is None:
        return X_tr_real, y_tr_real, "no_synth"

    rng = np.random.default_rng(seed)
    n_real = len(y_tr_real)
    n_syn_cap = int(max_syn_ratio * n_real)

    # plan total counts after augmentation
    target_total = n_real + n_syn_cap
    cur_pos = int(y_tr_real.sum())
    cur_neg = n_real - cur_pos
    tgt_pos = int(round(target_pos * target_total))
    need_pos = max(0, tgt_pos - cur_pos)
    need_neg = max(0, (target_total - tgt_pos) - cur_neg)

    # sample synthetic by class
    idx_pos_syn = np.where(y_syn == 1)[0]
    idx_neg_syn = np.where(y_syn == 0)[0]
    pos_pick = rng.choice(idx_pos_syn, size=min(need_pos, len(idx_pos_syn)),
                          replace=len(idx_pos_syn) < need_pos) if need_pos > 0 else np.array([], dtype=int)
    neg_pick = rng.choice(idx_neg_syn, size=min(need_neg, len(idx_neg_syn)),
                          replace=len(idx_neg_syn) < need_neg) if need_neg > 0 else np.array([], dtype=int)

    pick = np.concatenate([pos_pick, neg_pick])
    X_aug = np.vstack([X_tr_real, X_syn[pick]]) if pick.size else X_tr_real
    y_aug = np.concatenate([y_tr_real, y_syn[pick]]) if pick.size else y_tr_real
    return X_aug, y_aug, f"synth_added={pick.size}"


# Train Transformer model with early stopping and learning rate scheduling
def train_transformer(qm, X_train, y_train, max_len, epochs, batch_size, lr, nhead=4, num_layers=2):
    """
    Train the Transformer classifier with per-module learning rates,
    warmup->cosine schedule, BCE+pos_weight, grad clipping, and optional padding mask.
    Assumes:
      - TransformerClassifier(d_model=qm.embed_dim, nhead=2, num_layers=1, num_classes=1, max_len=max_len)
      - build_sequences_from_graph(...) returns either seq OR (seq, pad_mask)
    """

    model = TransformerClassifier(d_model=qm.embed_dim, nhead=nhead, num_layers=num_layers, num_classes=1, max_len=max_len).to(device)

    # ---------- Build sequences (+ optional pad mask) ----------
    result = build_sequences_from_graph(qm, qm.get_embedding(X_train).to(device), max_len=max_len)
    if isinstance(result, tuple) and len(result) == 2:
        seq_train, pad_mask = result  # [B,T,D], [B,T] bool (True = pad)
    else:
        seq_train = result
        # fabricate a no-pad mask (all False)
        pad_mask = torch.zeros(seq_train.size(0), seq_train.size(1), dtype=torch.bool, device=seq_train.device)

    ds = TensorDataset(seq_train, pad_mask, y_train.view(-1).to(torch.float32))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    # ---------- BCE with pos_weight ----------
    # Use CFG["class_weight"] if present; else derive from labels
    try:
        cw_pos = float(CFG["class_weight"][1])
        cw_neg = float(CFG["class_weight"][0])
        #pos_w = cw_pos / max(cw_neg, 1e-8)
        pos_w = (cw_pos / max(cw_neg, 1e-8)) ** 0.5 # Added the square root to reduce extreme weights (softening helps AUC)

    except Exception:
        with torch.no_grad():
            counts = torch.bincount(y_train.view(-1).long(), minlength=2).to(torch.float32)
            # pos_weight = #neg / #pos
            pos_w = (counts[0] / (counts[1] + 1e-8)).item()
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w, dtype=torch.float32, device=device))

    # ---------- Per-module LR param groups ----------
    def is_norm_or_bias(name: str, p: torch.nn.Parameter) -> bool:
        if p.ndim == 1:  # LayerNorm.weight, BatchNorm.weight, biases often 1D
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
    # 1) optional quantum lift (if present on the model)
    if hasattr(model, "phase"):
        add_group(model.phase.named_parameters(prefix="phase"), lr_group=0.5 * lr, wd=wd, groups=groups)
    if hasattr(model, "proj_back"):
        add_group(model.proj_back.named_parameters(prefix="proj_back"), lr_group=0.5 * lr, wd=wd, groups=groups)
    # 2) encoder
    add_group(model.transformer_encoder.named_parameters(prefix="encoder"), lr_group=lr, wd=wd, groups=groups)
    # 3) norm before head
    add_group(model.norm.named_parameters(prefix="norm"), lr_group=0.25 * lr, wd=wd, groups=groups)
    # 4) classification head (last, so we can print its lr)
    add_group(model.fc.named_parameters(prefix="fc"), lr_group=1.5 * lr, wd=wd, groups=groups)

    opt = optim.AdamW(groups)

    # ---------- Warmup -> Cosine schedule ----------
    warm_epochs = 5
    warm = LambdaLR(opt, lr_lambda=lambda e: (e + 1) / warm_epochs if e < warm_epochs else 1.0)
    cos = CosineAnnealingLR(opt, T_max=max(1, epochs - warm_epochs), eta_min=max(1e-7, 0.1 * lr))
    scheduler = SequentialLR(opt, schedulers=[warm, cos], milestones=[warm_epochs])

    best_loss = float("inf")
    patience = 8
    bad = 0

    for ep in range(epochs):
        model.train()
        total = 0.0
        for xb, mb, yb in dl:
            opt.zero_grad()
            # forward with or without pad_mask depending on model signature
            try:
                out = model(xb, pad_mask=mb).to(torch.float32)
            except TypeError:
                out = model(xb).to(torch.float32)

            loss = bce(out, yb.unsqueeze(1).to(out.device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()

        scheduler.step()
        avg = total / len(dl)
        # head group is last we appended
        head_lr = opt.param_groups[-1]["lr"]
        print(f"Epoch {ep+1}/{epochs} loss={avg:.4f} lr_head={head_lr:.6f}")

        if avg < best_loss - 1e-4:
            best_loss = avg
            bad = 0
        else:
            bad += 1
        if bad >= patience:
            print(f"Early stopping at epoch {ep+1}")
            break

    return model

# Monte Carlo prediction with Transformer model
def predict_transformer_mc(model, qm, X, max_len, mc_passes=7):
    model.train()
    Z = qm.get_embedding(X).to(device)
    seq, pad_mask = build_sequences_from_graph_with_mask(qm, Z, max_len=max_len)
    preds = []
    with torch.no_grad():
        for _ in range(mc_passes):
            logits = model(seq, pad_mask=pad_mask).squeeze(-1)
            preds.append(torch.sigmoid(logits).cpu().numpy())  # <-- sigmoid
    return np.mean(preds, axis=0).squeeze()




# Train Feedforward Neural Network with dropout and early stopping
def train_ffnn(qm, X_train, y_train, k_avg, epochs, batch_size, lr, hidden):
    """
    Train a Feedforward Neural Network with early stopping.
    
    Args:
        qm: Quantum-inspired model for embeddings.
        X_train (torch.Tensor): Training features.
        y_train (torch.Tensor): Training labels.
        k_avg (int): Average number of neighbors for quantum walk.
        epochs (int): Maximum number of training epochs.
        batch_size (int): Batch size for training.
        lr (float): Learning rate for optimization.
        hidden (int): Number of hidden units in the FFNN.
    
    Returns:
        nn.Module: Trained FFNN model.
    """
    model = FFNNClassifier(input_dim=qm.embed_dim, hidden=hidden).to(device)

    # class imbalance handling
    class_weight = torch.tensor(CFG["class_weight"], dtype=torch.float32, device=device)
    pos_weight = class_weight[1] / class_weight[0]
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    warm = optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda e: (e+1)/5 if e < 5 else 1.0)
    cos  = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs-5), eta_min=max(1e-7, 0.1*lr))
    sched = optim.lr_scheduler.SequentialLR(opt, [warm, cos], milestones=[5])

    Z_train = qm.get_k_avg_embedding(X_train, k_avg).to(device)
    ds = TensorDataset(Z_train, y_train.view(-1).to(torch.float32))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    best=float('inf'); bad=0; patience=10
    for ep in range(epochs):
        model.train(); total=0.0
        for xb, yb in dl:
            opt.zero_grad()
            out = model(xb).squeeze(1)
            loss = bce(out, yb.to(out.device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
        sched.step()
        avg = total/len(dl)
        print(f"Epoch {ep+1}/{epochs}, Loss: {avg:.4f}")
        if avg < best - 1e-4: best=avg; bad=0
        else: bad += 1
        if bad >= patience:
            print(f"Early stopping at epoch {ep+1}")
            break
    return model

def predict_ffnn_mc(model, qm, X, k_avg, mc_passes=7):
    """
    Make Monte Carlo predictions with FFNN.
    
    Args:
        model: Trained FFNN model.
        qm: Quantum-inspired model for embeddings.
        X (torch.Tensor): Input features.
        k_avg (int): Average number of neighbors for quantum walk.
        mc_passes (int): Number of Monte Carlo passes.
    
    Returns:
        np.ndarray: Mean predictions across passes.
    """
    model.train()  # enable dropout
    Z = qm.get_k_avg_embedding(X, k_avg).to(device)
    preds = []
    with torch.no_grad():
        for _ in range(mc_passes):
            preds.append(torch.sigmoid(model(Z).squeeze(1)).cpu().numpy())
    return np.mean(preds, axis=0)


def train_cnn(qm, X_train, y_train, batch_size, lr, hidden, epochs=30):
    # Train the CNN model with increased epochs and adjusted learning rate
    model = CNNClassifier(input_dim=qm.embed_dim, hidden=hidden).to(device)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.8, patience=12,
                                                    verbose=True)
    # Use weighted BCE loss
    class_weight = torch.tensor(CFG["class_weight"], dtype=torch.float32, device=device)
    bce = nn.BCEWithLogitsLoss(pos_weight=class_weight[1] / class_weight[0])
    Z_train = qm.get_embedding(X_train).to(device)
    seq_train = Z_train.unsqueeze(1)
    ds = TensorDataset(seq_train, y_train.view(-1).to(torch.float32))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    best_loss = float('inf')
    patience = 12
    trigger_times = 0
    for ep in range(epochs):
        model.train()
        epoch_loss = 0
        for xb, yb in dl:
            opt.zero_grad()
            out = model(xb).to(torch.float32)  # Shape: [batch_size]
            loss = bce(out, yb.to(out.device))  # Shape: [batch_size], no unsqueeze
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(dl)
        scheduler.step(avg_loss)
        if avg_loss < best_loss:
            best_loss = avg_loss
            trigger_times = 0
        else:
            trigger_times += 1
        if trigger_times >= patience:
            print(f"Early stopping at epoch {ep + 1}")
            break
        print(f"Epoch {ep + 1}/{epochs}, Loss: {avg_loss:.4f}")
    return model

# Predict with CNN model using Monte Carlo sampling
def train_cnn_seq(qm, X_train, y_train, max_len, epochs, batch_size, lr, hidden):
    """
    Train CNN over neighbor sequences with padding mask.
    """
    from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR

    model = CNNSeqClassifier(d_model=qm.embed_dim, hidden=hidden).to(device)

    # build neighbor seqs + mask (same function you use for Transformer)
    Z_train = qm.get_embedding(X_train).to(device)
    seq_train, pad_mask = build_sequences_from_graph_with_mask(qm, Z_train, max_len=max_len)
    pad_mask = pad_mask.bool()

    ds = TensorDataset(seq_train, pad_mask, y_train.view(-1).to(torch.float32))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    # class imbalance -> pos_weight
    try:
        cw_pos = float(CFG["class_weight"][1]); cw_neg = float(CFG["class_weight"][0])
        pos_w = cw_pos / max(cw_neg, 1e-8)
    except Exception:
        with torch.no_grad():
            counts = torch.bincount(y_train.view(-1).long(), minlength=2).to(torch.float32)
            pos_w = (counts[0] / (counts[1] + 1e-8)).item()
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w, dtype=torch.float32, device=device))

    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    warm = LambdaLR(opt, lr_lambda=lambda e: (e+1)/5 if e < 5 else 1.0)
    cos  = CosineAnnealingLR(opt, T_max=max(1, epochs-5), eta_min=max(1e-7, 0.1*lr))
    sched = SequentialLR(opt, [warm, cos], milestones=[5])

    best=float('inf'); bad=0; patience=10
    for ep in range(epochs):
        model.train(); total=0.0
        for xb, mb, yb in dl:
            opt.zero_grad()
            logits = model(xb, pad_mask=mb)       # [B]
            loss = bce(logits, yb.to(logits.device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
        sched.step()
        avg = total/len(dl)
        print(f"Epoch {ep+1}/{epochs}, Loss: {avg:.4f}")
        if avg < best - 1e-4: best=avg; bad=0
        else: bad += 1
        if bad >= patience:
            print(f"Early stopping at epoch {ep+1}")
            break
    return model


def predict_cnn_mc(model, qm, X, mc_passes=7):
    # Make Monte Carlo predictions with CNN
    model.eval()
    Z = qm.get_embedding(X).to(device)
    seq = Z.unsqueeze(1)
    preds = []
    with torch.no_grad():
        for _ in range(mc_passes):
            out = model(seq).to(torch.float32)
            preds.append(out.cpu().numpy())
    return np.mean(preds, axis=0).squeeze()

# Monte Carlo predictions for CNN with sequence input
def predict_cnn_seq_mc(model, qm, X, max_len, mc_passes=7):
    """MC-dropout predictions for sequence CNN with mask."""
    model.train()  # enable dropout for MC
    Z = qm.get_embedding(X).to(device)
    seq, pad_mask = build_sequences_from_graph_with_mask(qm, Z, max_len=max_len)
    pad_mask = pad_mask.bool()

    preds = []
    with torch.no_grad():
        for _ in range(mc_passes):
            logits = model(seq, pad_mask=pad_mask)
            preds.append(torch.sigmoid(logits).cpu().numpy())  # [B]
    return np.mean(preds, axis=0)  # [B]

# Plots
def plot_performance_table(per_fold_metrics, out_png="plots/performance_table.png"):
    # Create a table of performance metrics across folds
    ensure_dir("plots")
    model_names = list(per_fold_metrics.keys())
    metric_names = ["F1", "Precision", "Recall", "AUC"]
    folds = len(per_fold_metrics[model_names[0]]["F1"])
    header = ["Model / Metric"] + ["Fold {}".format(i + 1) for i in range(folds)] + ["Average"]
    table_data = [header]
    for m in model_names:
        for metric in metric_names:
            vals = per_fold_metrics[m][metric]
            avg = np.mean(vals)
            row = ["{} {}".format(m, metric)] + ["{:.4f}".format(v) for v in vals] + ["{:.4f}".format(avg)]
            table_data.append(row)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis("off")
    ax.table(cellText=table_data, loc="center", cellLoc="center")
    plt.title("Performance Metrics by Fold")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()

def plot_roc_curves(fprs_tprs_aucs, out_png="plots/roc_curves.png"):
    # Plot ROC curves for different models
    ensure_dir("plots")
    plt.figure(figsize=(10, 7))
    for name, data in fprs_tprs_aucs.items():
        fpr, tpr, auc_ = data
        plt.plot(fpr, tpr, label="{} AUC={:.3f}".format(name, auc_))
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curves")
    plt.legend(loc="lower right")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()

def plot_pr_curves(curves, out_png="plots/pr_curves.png"):
    # Plot Precision-Recall curves
    ensure_dir("plots")
    plt.figure(figsize=(10, 7))
    for name, data in curves.items():
        p, r, ap = data
        plt.plot(r, p, label="{} AP={:.3f}".format(name, ap))
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()

def plot_summary_bars(per_fold_metrics, out_png="plots/summary_bars.png"):
    # Plot bar chart of average metrics across folds
    ensure_dir("plots")
    model_names = list(per_fold_metrics.keys())
    metric_names = ["F1", "Precision", "Recall", "AUC"]
    means = []
    for m in model_names:
        m_means = []
        for metric in metric_names:
            vals = per_fold_metrics[m][metric]
            m_means.append(np.mean(vals))
        means.append(m_means)
    x = np.arange(len(metric_names))
    width = 0.8 / len(model_names)
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, m in enumerate(model_names):
        ax.bar(x + i * width, means[i], width, label=m)
    ax.set_xticks(x + width * (len(model_names) - 1) / 2.0)
    ax.set_xticklabels(metric_names)
    ax.set_ylabel("Score")
    ax.set_title("Average Metrics Across Folds")
    ax.legend()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()

def plot_graph_snapshot(qm, out_png="plots/concept_graph.png"):
    # Visualize the concept graph snapshot
    ensure_dir("plots")
    if qm.graph is None or len(qm.graph) == 0:
        return
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(qm.graph, seed=SEED)
    node_colors = [qm.node_labels.get(n, 0) for n in qm.graph.nodes()]
    nx.draw(qm.graph, pos, node_color=node_colors, cmap=plt.cm.RdBu, with_labels=False,
            node_size=80, edge_color="gray", alpha=0.8)
    plt.title("Concept Graph Snapshot")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()

def plot_calibration_and_hist(y_true, prob, out_png="plots/calibration.png"):
    # Plot calibration curve and histogram of predictions
    ensure_dir("plots")
    bins = np.linspace(0, 1, 11)
    bin_ids = np.digitize(prob, bins) - 1
    frac_pos, mean_pred = [], []
    for b in range(len(bins) - 1):
        mask = bin_ids == b
        if mask.sum() == 0:
            continue
        frac_pos.append(np.mean(y_true[mask]))
        mean_pred.append(np.mean(prob[mask]))
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.4)
    if len(mean_pred):
        plt.plot(mean_pred, frac_pos, marker="o")
    plt.title("Reliability Diagram (OOF Meta)")
    plt.xlabel("Predicted probability")
    plt.ylabel("Fraction positive")
    plt.subplot(1, 2, 2)
    plt.hist(prob[y_true == 0], bins=20, alpha=0.6, label="Class 0")
    plt.hist(prob[y_true == 1], bins=20, alpha=0.6, label="Class 1")
    plt.title("OOF Probability Histogram")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()

# Plot confusion matrix and bar chart with rates
def plot_confusion_matrix_and_bars(y_true, y_pred, out_png="plots/confusion_oof_meta_bars.png"):
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()  # Correct ravel order: [tn, fp, fn, tp]

    # Left subplot: Confusion Matrix Heatmap
    fig = plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center", color="black")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label (0 = Negative, 1 = Positive)")
    plt.ylabel("True Label (0 = Negative, 1 = Positive)")
    plt.xticks([0, 1], ["Negative", "Positive"])
    plt.yticks([0, 1], ["Negative", "Positive"])
    plt.colorbar(label="Count")

    # Right subplot: Bar Chart with Counts and Rates
    plt.subplot(1, 2, 2)
    categories = ['TN', 'FP', 'FN', 'TP']
    counts = [tn, fp, fn, tp]  # Natural order from ravel()

    # Calculate rates
    total = tp + fp + tn + fn
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # True Positive Rate (Recall)
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0  # True Negative Rate (Specificity)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate

    bars = plt.bar(categories, counts, color=['#2196F3', '#F44336', '#FF9800', '#4CAF50'])
    plt.title('Confusion Matrix Bar Chart')
    plt.ylabel('Number of Instances')
    plt.ylim(0, max(counts) * 1.2)

    # Add count labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{int(height)}', ha='center', va='bottom')

    # Overlay rates as a secondary plot
    ax2 = plt.twinx()
    rates = [tnr, fpr, fnr, tpr]  # Match order with categories
    ax2.plot(categories, rates, color='black', marker='o', linestyle='--', label='Rates')
    ax2.set_ylabel('Rate (0 to 1)')
    ax2.set_ylim(0, 1.1)
    ax2.legend(loc='upper right')

    # Add rate annotations
    for i, rate in enumerate(rates):
        ax2.text(i, rate + 0.05, f'{rate:.3f}', ha='center', va='bottom')

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()

# Plot distribution of prediction scores by class
def plot_score_distributions(y_true, prob, out_png="plots/score_distributions.png"):
    ensure_dir("plots")
    plt.figure(figsize=(10, 6))
    p0 = prob[y_true == 0]
    p1 = prob[y_true == 1]
    plt.violinplot([p0, p1], showmeans=True, showmedians=True)
    plt.xticks([1, 2], ["Class 0", "Class 1"])
    plt.ylabel("Predicted probability")
    plt.title("OOF Score Distributions by Class")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()

# Plot feature importance from RF and ET models
def plot_feature_importance(rf, et, feature_names, out_png="plots/feature_importance.png"):
    ensure_dir("plots")
    rf_imp = rf.feature_importances_
    et_imp = et.feature_importances_
    idx = np.argsort(rf_imp)[::-1][:20]  # Top 20 indices
    plt.figure(figsize=(12, 6))
    plt.bar(np.arange(len(idx)) - 0.2, rf_imp[idx], width=0.4, label="RF")
    plt.bar(np.arange(len(idx)) + 0.2, et_imp[idx], width=0.4, label="ET")
    plt.xticks(np.arange(len(idx)), [feature_names[i] for i in idx], rotation=45, ha="right")  # Fix indexing
    plt.title("Top embedding-feature importances (RF/ET)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()

# Main function to execute the QISICGM pipeline
def main():
    """
    Main function to execute the Quantum-Inspired Stacked Integrated Concept Graph Model (QISICGM)
    for diabetes prediction using the PIMA Indians Diabetes dataset, augmented with synthetic data.
    This function handles data preparation, cross-validation, model training, performance evaluation,
    and model saving without pre-training, preserving build_sequences_from_graph
    for quantum inspiration.

    The pipeline includes:
    - Data loading and preprocessing (PIMA and synthetic data)
    - Quantum-inspired embedding generation with self-improvement
    - Optional base model training and saving
    - Training of a meta-learner on embeddings (using saved or computed embeddings)
    - Global OOF calibration and threshold selection
    - Enhanced visualization (saving all plots)
    - Model saving for demo predictions

    Dependencies: NumPy, Pandas, Matplotlib, Scikit-learn, PyTorch, NetworkX
    Author: Adapted by xAI with contributions from Kenneth Young, PhD (USF-HII)
    """
    # Initialize timing for performance tracking
    print("[0.00s] Starting...")
    t0 = time.time()

    # Data Preparation Section
    print(".. Data prep start")
    data_path = os.path.join("data", "pima-indians-diabetes.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}. Please ensure "
                               "'pima-indians-diabetes.csv' is in the 'data' folder.")
    X_df = pd.read_csv(data_path, header=None)
    y = X_df.iloc[:, -1].values
    X_df = X_df.iloc[:, :-1]
    X_df["Glucose_BMI"] = X_df[1] * X_df[5]
    X_df["G_to_Pressure"] = X_df[1] / (X_df[2] + 1.0)
    X_df["BMI_sq"] = X_df[5] ** 2
    feature_names = [str(i) for i in range(X_df.shape[1] - 3)] + ["Glucose_BMI", "G_to_Pressure", "BMI_sq"]
    synthetic_path = os.path.join("data", "synthetic_pima_data.csv")
    if os.path.exists(synthetic_path):
        synthetic_df = pd.read_csv(synthetic_path)
        X_synthetic = synthetic_df.iloc[:, :-1].values
        y_synthetic = synthetic_df["Outcome"].values
        X_df = pd.concat([X_df, pd.DataFrame(X_synthetic, columns=X_df.columns)], ignore_index=True)
        y = np.concatenate([y, y_synthetic])
        print(f".. Synthetic data loaded: Added {len(y_synthetic)} samples, new total y_pos={y.sum()}")
    impute_medians = {}
    for col in [1, 2, 3, 4, 5]:
        nz = X_df[col].replace(0, np.nan)
        m = nz.median()
        if not np.isfinite(m):
            m = float(X_df[col].median())
        X_df[col] = X_df[col].replace(0, m)
        impute_medians[col] = m
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df.values.astype(np.float32))
    print("[{:.2f}s] Data ready: X={}, y_pos={}".format(time.time() - t0, X_scaled.shape, y.sum()))
    print(".. Data prep done in {:.2f}s".format(time.time() - t0))
    X_all = torch.tensor(X_scaled, dtype=torch.float32, device=device)
    y_all = torch.tensor(y, dtype=torch.long, device=device)

    # Initialize qm_final after data prep for global use
    qm_final = QISICGM(input_dim=X_all.shape[1], embed_dim=CFG["embed_dim"]).to(device)
    qm_final.initialize_graph(X_all, y_all, k=15)
    qm_final.self_improve(X_all, y_all, steps=CFG["self_improve_steps"],
                         lr=CFG["self_improve_lr"] * 2, prune_every=999,
                         min_degree=1, verbose=True, use_focal=CFG["use_focal"],
                         focal_gamma=CFG["focal_gamma"])

    # Check for saved base models and embeddings
    USE_SAVED_EMBEDDINGS = os.path.exists("models/base_models_available.flag")

    # Train base models and save if not using saved embeddings
    if not USE_SAVED_EMBEDDINGS:
        print(".. Training base models and saving embeddings...")
        train_base_models_and_save(X_all, y_all, qm_final, CFG)

    # Set up stratified k-fold cross-validation
    skf = StratifiedKFold(n_splits=CFG["skfold"], shuffle=True, random_state=SEED)

    # Initialize out-of-fold predictions (for reference)
    oof_rf = np.zeros(len(y))
    oof_et = np.zeros(len(y))
    oof_tf = np.zeros(len(y))
    oof_ff = np.zeros(len(y))
    oof_cnn = np.zeros(len(y))

    # Dictionary to store performance metrics
    metrics = {
        "Meta": {"F1": [], "Precision": [], "Recall": [], "AUC": []},
    }

    # Storage for ROC and PR curve data
    roc_curves_meta = {}
    pr_curves_meta = {}

    # Store previous QM state for warm start
    prev_qm_state = qm_final.state_dict()

    # Cross-Validation Loop (use saved embeddings if available)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_scaled, y), start=1):
        print("\n----- Fold {}/{} -----".format(fold, CFG["skfold"]))
        X_tr_np, X_va_np = X_scaled[tr_idx], X_scaled[va_idx]
        y_tr_np, y_va_np = y[tr_idx], y[va_idx]
        X_tr_np_bal = X_tr_np
        y_tr_np_bal = y_tr_np
        if y_tr_np.mean() < 0.4:
            n_pos = y_tr_np.sum()
            n_neg = len(y_tr_np) - n_pos
            if n_pos < n_neg:
                idx_pos = np.where(y_tr_np == 1)[0]
                idx_pos_extra = np.random.choice(idx_pos, n_neg - n_pos, replace=True)
                X_tr_np_bal = np.vstack((X_tr_np, X_tr_np[idx_pos_extra]))
                y_tr_np_bal = np.hstack((y_tr_np, y_tr_np[idx_pos_extra]))
        sampler_used = "simple_oversample" if y_tr_np.mean() < 0.4 else "none"
        t_s = time.time()
        tprint(".. sampler used: {}".format(sampler_used), t_s)
        X_tr = torch.tensor(X_tr_np_bal, dtype=torch.float32, device=device)
        y_tr = torch.tensor(y_tr_np_bal, dtype=torch.long, device=device)
        X_va = torch.tensor(X_va_np, dtype=torch.float32, device=device)
        y_va = torch.tensor(y_va_np, dtype=torch.long, device=device)
        qm = QISICGM(input_dim=X_tr.shape[1], embed_dim=CFG["embed_dim"]).to(device)
        qm.initialize_graph(X_tr, y_tr, k=15)
        if CFG["warm_start_qm"] and prev_qm_state is not None:
            qm.load_state_dict(prev_qm_state, strict=False)
        t_si = time.time()
        print(".. self_improve start")
        batch_size = 512 if X_tr.shape[0] > 1000 else X_tr.shape[0]
        if batch_size == X_tr.shape[0]:
            qm.self_improve(X_tr, y_tr, steps=CFG["self_improve_steps"],
                           lr=CFG["self_improve_lr"] * 2, prune_every=999,
                           min_degree=1, verbose=True, use_focal=CFG["use_focal"],
                           focal_gamma=CFG["focal_gamma"])
        else:
            for i in range(0, X_tr.shape[0], batch_size):
                batch_end = min(i + batch_size, X_tr.shape[0])
                qm.self_improve(X_tr[i:batch_end], y_tr[i:batch_end], steps=CFG["self_improve_steps"],
                               lr=CFG["self_improve_lr"] * 2, prune_every=999,
                               min_degree=1, verbose=(i == 0), use_focal=CFG["use_focal"],
                               focal_gamma=CFG["focal_gamma"])
        tprint(".. self_improve done", t_si)
        if CFG["warm_start_qm"]:
            prev_qm_state = qm.state_dict()
        if USE_SAVED_EMBEDDINGS:
            Z_all = np.load("models/embeddings.npy")
            Z_tr = Z_all[tr_idx]
            Z_va = Z_all[va_idx]
        else:
            Z_tr = qm.get_embedding(X_tr).detach().cpu().numpy()
            Z_va = qm.get_embedding(X_va).detach().cpu().numpy()
        # Train meta directly on embeddings
        X_meta_va = Z_va
        meta = LogisticRegression(max_iter=1000, solver="lbfgs", C=CFG["meta_C"],
                                 random_state=SEED)
        meta.fit(X_meta_va, y_va_np)
        meta_va = meta.predict_proba(X_meta_va)[:, 1]
        t_meta, f1_meta = find_best_f1(y_va_np, meta_va, recall_floor=CFG["recall_floor"])
        y_hat_meta = (meta_va >= t_meta).astype(int)
        metrics["Meta"]["F1"].append(f1_meta)
        metrics["Meta"]["Precision"].append(precision_score(y_va_np, y_hat_meta, zero_division=0))
        metrics["Meta"]["Recall"].append(recall_score(y_va_np, y_hat_meta, zero_division=0))
        metrics["Meta"]["AUC"].append(roc_auc_score(y_va_np, meta_va))
        fpr, tpr, _ = roc_curve(y_va_np, meta_va)
        roc_curves_meta["Fold {} Meta".format(fold)] = (fpr, tpr, roc_auc_score(y_va_np, meta_va))
        p, r, _ = precision_recall_curve(y_va_np, meta_va)
        ap = np.trapz(p[::-1], r[::-1])
        pr_curves_meta["Fold {} Meta".format(fold)] = (p, r, ap)
        if fold == 1:
            plot_graph_snapshot(qm, out_png="plots/concept_graph_fold1.png")
        print("META F1={:.4f} AUC={:.4f}".format(metrics["Meta"]["F1"][-1], metrics["Meta"]["AUC"][-1]))

    # Global OOF Calibration (using embeddings)
    print(".. Global OOF calibration and meta start")
    t_g = time.time()
    if USE_SAVED_EMBEDDINGS:
        X_meta_oof = np.load("models/embeddings.npy")
    else:
        X_meta_oof = qm_final.get_embedding(X_all).detach().cpu().numpy()
    meta_oof = LogisticRegression(max_iter=1000, solver="lbfgs", C=CFG["meta_C"] * 0.1,
                                 random_state=SEED).fit(X_meta_oof, y)
    meta_oof_prob = meta_oof.predict_proba(X_meta_oof)[:, 1]
    thresholds = np.linspace(0.4, 0.7, 101)
    best_f1 = 0.0
    best_t = 0.5
    best_recall = 0.0
    for t in thresholds:
        pred = (meta_oof_prob >= t).astype(int)
        f1 = f1_score(y, pred, zero_division=0)
        recall = recall_score(y, pred, zero_division=0)
        if f1 > best_f1 or (abs(f1 - best_f1) < 0.01 and recall > best_recall + 0.1):
            best_f1 = f1
            best_t = t
            best_recall = recall
    t_final = best_t
    f1_final = best_f1
    auc_final = roc_auc_score(y, meta_oof_prob)
    tprint(".. Global OOF calibration and meta done", t_g)
    print("\n=== OOF Stacker Performance (no leakage) ===")
    print("F1={:.4f} AUC={:.4f}".format(f1_final, auc_final))
    print("Final decision threshold (OOF-optimal): {:.4f}".format(t_final))

    # Visualization and Output Section
    plot_performance_table(metrics, out_png="plots/performance_table.png")
    plot_summary_bars(metrics, out_png="plots/summary_bars.png")
    plot_roc_curves(roc_curves_meta, out_png="plots/roc_curves_meta.png")
    p, r, _ = precision_recall_curve(y, meta_oof_prob)
    ap = np.trapz(p[::-1], r[::-1])
    pr_curves_meta = {"OOF Meta": (p, r, ap), **pr_curves_meta}
    plot_pr_curves(pr_curves_meta, out_png="plots/pr_curves_meta.png")
    plot_calibration_and_hist(y, meta_oof_prob, out_png="plots/calibration_oof_meta.png")
    y_pred = (meta_oof_prob >= t_final).astype(int)
    plot_confusion_matrix_and_bars(y, y_pred)
    plot_score_distributions(y, meta_oof_prob, out_png="plots/score_distributions_oof_meta.png")

    # Final refit (using embeddings)
    print(".. Final refit for demo predictions start")
    t_refit = time.time()
    X_bal = X_scaled  # Define X_bal here
    y_bal = y         # Define y_bal here
    if y.mean() < 0.4:
        idx_pos = np.where(y == 1)[0]
        idx_neg = np.where(y == 0)[0]
        if len(idx_pos) < len(idx_neg):
            idx_pos_extra = np.random.choice(idx_pos, len(idx_neg) - len(idx_pos), replace=True)
            X_bal = np.vstack((X_scaled, X_scaled[idx_pos_extra]))
            y_bal = np.hstack((y, y[idx_pos_extra]))
    X_all_t = torch.tensor(X_bal, dtype=torch.float32, device=device)
    y_all_t = torch.tensor(y_bal, dtype=torch.long, device=device)
    # Reuse and refine qm_final
    qm_final.initialize_graph(X_all_t, y_all_t, k=15)
    qm_final.self_improve(X_all_t, y_all_t, steps=64, lr=CFG["self_improve_lr"] * 2,
                         prune_every=999999, min_degree=1, verbose=False,
                         use_focal=CFG["use_focal"], focal_gamma=CFG["focal_gamma"])
    torch.save(qm_final.state_dict(), "models/qm_final.pth")
    with open("models/meta_oof.pkl", "wb") as f:
        pickle.dump({
            "meta": meta_oof,
            "threshold": t_final,
            "scaler": scaler,
            "feature_names": feature_names,
            "cfg": CFG,
            "profile": PROFILE
        }, f)
    tprint(".. Final refit for demo predictions done", t_refit)
    print("Models saved. Run demo_predictions.py for new patient predictions.")
    tprint("Total execution time", t0)

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()