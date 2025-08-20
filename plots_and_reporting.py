# plots_and_reporting.py
# All-in-one plotting and reporting utilities for your stacked CV run.
# Designed to be drop-in with your current code and logs. ASCII only.
# Requires: numpy, matplotlib, scikit-learn

import networkx as nx
import math
import numpy as np
import os, matplotlib
if os.environ.get("DISPLAY", "") == "":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score
)
from sklearn.calibration import calibration_curve
from sklearn.manifold import TSNE

# ---------------------------------------------------------------------
# Directory helper
# ---------------------------------------------------------------------
def ensure_dir(path="plots"):
    """Create a directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


# ---------------------------------------------------------------------
# Formatting helper
# ---------------------------------------------------------------------
def _fmt(x, nd=4):
    """Safe numeric formatter. Returns '' for nan/inf."""
    try:
        if x is None:
            return ""
        if not np.isfinite(x):
            return ""
        return f"{float(x):.{nd}f}"
    except Exception:
        return ""

# ---------------------------------------------------------------------
# Concept graph snapshot
# ---------------------------------------------------------------------
def plot_graph_snapshot(qm, out_png="plots/concept_graph.png", seed=42, iterations=30):
    ensure_dir(os.path.dirname(out_png) or "plots")
    if getattr(qm, "graph", None) is None or len(qm.graph) == 0:
        return
    plt.figure(figsize=(10, 8), dpi=150)
    pos = nx.spring_layout(qm.graph, seed=seed, iterations=iterations)
    node_colors = [qm.node_labels.get(n, 0) for n in qm.graph.nodes()]
    nx.draw(
        qm.graph, pos,
        node_color=node_colors, cmap=plt.cm.RdBu,
        with_labels=False, node_size=80,
        edge_color="gray", width=0.5, alpha=0.8,
    )
    plt.title("Concept Graph Snapshot")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------
# Concept graph snapshot using TSNE
# ---------------------------------------------------------------------
def concept_graph_snapshot(
    qm,
    X_t,
    y,
    out_png="plots/concept_graph.png",
    max_points=800,
    k=10,
    method="tsne",
    seed=42
):
    """
    Visualize the concept graph for a batch by:
      1) getting embeddings via qm.get_embedding(X_t)
      2) 2D projection (t-SNE by default, PCA fallback)
      3) drawing light kNN edges and colored nodes by label

    qm:      trained QISICGM (already refined on train)
    X_t:     torch.FloatTensor [N, F] on same device as qm
    y:       array-like [N] with 0/1 labels
    """
    qm.eval()
  
    with torch.no_grad():
        Z = qm.get_embedding(X_t).detach().cpu().numpy()  # [N, D]
    y = np.asarray(y)

    # Stratified cap for readability
    if len(y) > max_points:
        rng = np.random.default_rng(seed)  # for reproducibility
        idx_pos = np.where(y == 1)[0]
        idx_neg = np.where(y == 0)[0]
        take_each = max_points // 2
        sel = np.r_[
            rng.choice(idx_pos, min(take_each, len(idx_pos)), replace=False),
            rng.choice(idx_neg, min(take_each, len(idx_neg)), replace=False),
        ]
        rng.shuffle(sel)
        Z = Z[sel]
        y = y[sel]


    # 2D projection
    try:
        if method.lower() == "tsne":
            from sklearn.manifold import TSNE
            perplex = int(np.clip(len(y) // 10, 5, 30))
            perplex = min(perplex, max(2, len(y) - 2))
            XY = TSNE(
                n_components=2, learning_rate="auto", init="random",
                perplexity=perplex, random_state=seed,
                n_iter=500, n_iter_without_progress=150, angle=0.5, verbose=1
            ).fit_transform(Z)
        else:
            raise ValueError
    except Exception:
        from sklearn.decomposition import PCA
        XY = PCA(n_components=2, random_state=42).fit_transform(Z)

    # Build kNN on embeddings (not on XY!) for more faithful neighbors
    try:
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=min(k + 1, len(Z))).fit(Z)
        nbrs = nn.kneighbors(return_distance=False)[:, 1:]
    except Exception:
        # graceful fallback: no edges
        nbrs = np.zeros((len(Z), 0), dtype=int)

    # Colors: negative (blue), positive (red)
    color_map = {0: "#4F81BD", 1: "#C0504D"}  # matches confusion colors below
    plt.figure(figsize=(7, 6), dpi=140)

    # light gray edges
    for i, neighs in enumerate(nbrs):
        for j in neighs:
            plt.plot([XY[i, 0], XY[j, 0]], [XY[i, 1], XY[j, 1]],
                     lw=0.4, alpha=0.15, color="#999999")

    # nodes
    for cls in [0, 1]:
        m = (y == cls)
        if m.any():
            plt.scatter(XY[m, 0], XY[m, 1],
                        s=12, alpha=0.9, label=f"class {cls}",
                        edgecolors="white", linewidths=0.3,
                        c=color_map[cls])

    plt.title("Concept graph (2D projection of embeddings)")
    plt.legend(loc="best", frameon=True)
    plt.tight_layout()
    ensure_dir(os.path.dirname(out_png) or "plots")
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()

# ---------------------------------------------------------------------
# Confusion matrix + bars from OOF labels and probabilities
# ---------------------------------------------------------------------
def plot_confusion_matrix_and_bars(
    y_true,
    y_pred=None,
    y_score=None,
    threshold=0.5,
    out_png="plots/confusion_oof_meta_bars.png",
):
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)

    y_true = np.asarray(y_true).astype(int)
    if y_pred is None:
        if y_score is None:
            raise ValueError("Provide either y_pred or (y_score and threshold).")
        y_pred = (np.asarray(y_score).astype(float) >= float(threshold)).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # --- Left: heatmap ---
    fig = plt.figure(figsize=(12, 5), dpi=140)
    ax1 = plt.subplot(1, 2, 1)
    im = ax1.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax1.text(j, i, str(v), ha="center", va="center", color="black")
    ax1.set_title("Confusion Matrix")
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("True")
    ax1.set_xticks([0, 1]); ax1.set_xticklabels(["Neg", "Pos"])
    ax1.set_yticks([0, 1]); ax1.set_yticklabels(["Neg", "Pos"])
    cbar = fig.colorbar(im, ax=ax1)
    cbar.set_label("Count")

    # --- Right: bars + rates (distinct colors, no text overlap) ---
    ax = plt.subplot(1, 2, 2)
    cats  = ["TN", "FP", "FN", "TP"]
    counts = [tn, fp, fn, tp]

    # colors consistent with heatmap vibe
    colors = ["#4F81BD", "#F39C12", "#E74C3C", "#2ECC71"]  # blue, orange, red, green
    bars = ax.bar(cats, counts, color=colors)

    # rates
    tpr = tp / max(1, (tp + fn))
    tnr = tn / max(1, (tn + fp))
    fpr = fp / max(1, (fp + tn))
    fnr = fn / max(1, (fn + tp))

    ax2 = ax.twinx()
    rates = [tnr, fpr, fnr, tpr]
    ax2.plot(cats, rates, marker="o", linestyle="--", label="Rates")
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Rate (0 to 1)")
    ax2.legend(loc="upper right")

    # add count labels above bars (offset +5)
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{int(h)}", (bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 5), textcoords="offset points",
                    ha="center", va="bottom")

    # add rate labels slightly above the points (offset +8) to avoid collisions
    for i, r in enumerate(rates):
        ax2.annotate(f"{r:.4f}", (i, r),
                     xytext=(0, 8), textcoords="offset points",
                     ha="center", va="bottom")

    ax.set_title("Confusion Matrix Bar Chart")
    ax.set_ylabel("Number of Instances")
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

# ---------------------------------------------------------------------
# Reliability diagram + OOF probability histogram
# ---------------------------------------------------------------------
def plot_reliability_and_hist(
    y_true,
    y_score,
    out_png="plots/calibration_oof_meta.png",
    n_bins=10
):
    """
    Left: reliability curve (fraction positive vs predicted prob).
    Right: histograms of predicted probabilities by true class.
    """
    ensure_dir(os.path.dirname(out_png) or "plots")

    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    fig = plt.figure(figsize=(16, 7))

    # reliability
    ax1 = fig.add_subplot(1, 2, 1)
    frac_pos, mean_pred = calibration_curve(y_true, y_score, n_bins=n_bins, strategy="uniform")
    ax1.plot(mean_pred, frac_pos, marker="o")
    ax1.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax1.set_title("Reliability Diagram (OOF Meta)")
    ax1.set_xlabel("Predicted probability")
    ax1.set_ylabel("Fraction positive")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    # histograms
    ax2 = fig.add_subplot(1, 2, 2)
    mask0 = (y_true == 0)
    mask1 = (y_true == 1)
    ax2.hist(y_score[mask0], bins=30, alpha=0.8, label="Class 0")
    ax2.hist(y_score[mask1], bins=30, alpha=0.8, label="Class 1")
    ax2.set_title("OOF Probability Histogram")
    ax2.set_xlabel("Predicted probability")
    ax2.set_ylabel("Count")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------
# ROC curves for META: per-fold + OOF
# ---------------------------------------------------------------------
def plot_roc_curves_meta(
    folds_meta,
    oof_pair,
    out_png="plots/roc_curves_meta.png"
):
    """
    folds_meta: list of (y_val_fold, p_meta_val_fold) for each fold
    oof_pair: (y_oof, p_oof) tuple
    """
    ensure_dir(os.path.dirname(out_png) or "plots")

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(1, 1, 1)

    y_oof, p_oof = oof_pair
    fpr, tpr, _ = roc_curve(y_oof, p_oof)
    auc_oof = roc_auc_score(y_oof, p_oof)
    ax.plot(fpr, tpr, label=f"OOF Meta AUC={auc_oof:.3f}")

    for i, (y_v, p_v) in enumerate(folds_meta, 1):
        f, t, _ = roc_curve(y_v, p_v)
        auc_v = roc_auc_score(y_v, p_v)
        ax.plot(f, t, label=f"Fold {i} Meta AUC={auc_v:.3f}")

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_title("ROC Curves")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------
# Precision-Recall curves for META: per-fold + OOF
# ---------------------------------------------------------------------
def plot_pr_curves_meta(
    folds_meta,
    oof_pair,
    out_png="plots/pr_curves_meta.png"
):
    """
    folds_meta: list of (y_val_fold, p_meta_val_fold) for each fold
    oof_pair: (y_oof, p_oof) tuple
    """
    ensure_dir(os.path.dirname(out_png) or "plots")

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(1, 1, 1)

    y_oof, p_oof = oof_pair
    pr, rc, _ = precision_recall_curve(y_oof, p_oof)
    ap = average_precision_score(y_oof, p_oof)
    ax.step(rc, pr, where="post", label=f"OOF Meta AP={ap:.3f}")

    for i, (y_v, p_v) in enumerate(folds_meta, 1):
        pr_v, rc_v, _ = precision_recall_curve(y_v, p_v)
        ap_v = average_precision_score(y_v, p_v)
        ax.step(rc_v, pr_v, where="post", label=f"Fold {i} Meta AP={ap_v:.3f}")

    ax.set_title("Precision-Recall Curves")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------
# Violin distribution of OOF scores by true class
# ---------------------------------------------------------------------
def plot_oof_score_violins(
    y_true,
    y_score,
    out_png="plots/score_distributions_oof_meta.png"
):
    ensure_dir(os.path.dirname(out_png) or "plots")

    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(1, 1, 1)

    data0 = y_score[y_true == 0]
    data1 = y_score[y_true == 1]
    ax.violinplot([data0, data1], showmeans=True, showmedians=False, showextrema=False)

    # add median and IQR markers for each group
    for i, data in enumerate([data0, data1], start=1):
        if data.size == 0:
            continue
        q1, q2, q3 = np.percentile(data, [25, 50, 75])
        ax.scatter([i], [q2], marker="o")
        ax.vlines(i, q1, q3, linewidth=3)

    ax.set_title("OOF Score Distributions by Class")
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Class 0", "Class 1"])
    ax.set_ylabel("Predicted probability")
    ax.set_ylim(0.0, 1.0)

    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------
# Performance table across folds for multiple models
# ---------------------------------------------------------------------
def plot_performance_table(
    fold_metrics,
    out_png="plots/performance_table.png",
    title="Performance Metrics by Fold"
):
    """
    fold_metrics is a dict with keys for each model. For each model, supply:
        {"f1": [...], "precision": [...], "recall": [...], "auc": [...]}

    Example keys you used: rf, et, tf, ff, cnn, meta
    """
    ensure_dir(os.path.dirname(out_png) or "plots")

    # canonical order if present
    model_order = ["rf", "et", "tf", "ff", "cnn", "meta"]
    models = [m for m in model_order if m in fold_metrics] or list(fold_metrics.keys())
    metric_names = ["f1", "precision", "recall", "auc"]

    # infer folds from first available metric
    n_folds = None
    for m in models:
        for k in metric_names:
            if k in fold_metrics[m]:
                n_folds = len(fold_metrics[m][k])
                break
        if n_folds is not None:
            break
    if n_folds is None:
        n_folds = 0

    # build table content
    rows = []
    row_labels = []
    for m in models:
        for k in metric_names:
            vals = fold_metrics[m].get(k, [])
            if len(vals) == 0:
                vals = [np.nan] * n_folds
            avg = np.nanmean(vals) if len(vals) else np.nan
            row = [(_fmt(v) if np.isfinite(v) else "") for v in vals]
            row.append(_fmt(avg) if np.isfinite(avg) else "")
            rows.append(row)
            row_labels.append(f"{m.upper()} {k.upper()}")

    col_labels = [f"Fold {i}" for i in range(1, n_folds + 1)] + ["Average"]

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis("off")
    tbl = ax.table(
        cellText=rows,
        rowLabels=row_labels,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.4)
    ax.set_title(title, pad=20)
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------
# Summary bars for average metrics for a single model
# ---------------------------------------------------------------------
def plot_avg_metric_bars(
    avg_metrics,
    label="Meta",
    out_png="plots/summary_bars.png"
):
    """
    avg_metrics: dict with keys 'F1', 'Precision', 'Recall', 'AUC'
    """
    ensure_dir(os.path.dirname(out_png) or "plots")

    keys = ["F1", "Precision", "Recall", "AUC"]
    vals = [avg_metrics.get(k, np.nan) for k in keys]

    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(1, 1, 1)

    bars = ax.bar(keys, vals, label=label)
    for k, v, bar in zip(keys, vals, bars):
        if np.isfinite(v):
            ax.text(bar.get_x() + bar.get_width() / 2.0, v, _fmt(v),
                    ha="center", va="bottom")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Average Metrics Across Folds")
    ax.legend()

    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------
# Top-level convenience: make all plots in one call
# ---------------------------------------------------------------------
def make_all_plots(
    y_oof_true,
    p_oof_meta,
    meta_threshold,
    meta_threshold_bal,
    folds_meta,
    fold_metrics,
    out_dir="plots"
):
    """
    Generate every figure you had before, with robust defaults.

    y_oof_true: 1d array of OOF labels
    p_oof_meta: 1d array of OOF meta probabilities
    meta_threshold: float threshold used for the OOF confusion matrix
    folds_meta: list of (y_val_fold, p_meta_val_fold) pairs for K folds
    fold_metrics: dict of lists for each model:
        {
          "rf":   {"f1":[...], "precision":[...], "recall":[...], "auc":[...]},
          "et":   {...},
          "tf":   {...},
          "ff":   {...},
          "cnn":  {...},
          "meta": {...}
        }
    out_dir: where to write PNGs
    """
    ensure_dir(out_dir)

    # 1a) Confusion matrix + bars
    plot_confusion_matrix_and_bars(
        y_true=y_oof_true,
        y_score=p_oof_meta,
        threshold=meta_threshold,
        out_png=os.path.join(out_dir, "confusion_oof_meta_bars.png"),
    )

    # 1b) Confusion matrix + bars (balanced threshold)
    plot_confusion_matrix_and_bars(
        y_true=y_oof_true,
        y_score=p_oof_meta,
        threshold=meta_threshold_bal,
        out_png=os.path.join(out_dir, "confusion_oof_meta_bars_balanced.png"),
    )

    # 2) Reliability + histogram
    plot_reliability_and_hist(
        y_true=y_oof_true,
        y_score=p_oof_meta,
        out_png=os.path.join(out_dir, "calibration_oof_meta.png"),
        n_bins=10,
    )

    # 3) ROC curves (folds + OOF)
    plot_roc_curves_meta(
        folds_meta=folds_meta,
        oof_pair=(y_oof_true, p_oof_meta),
        out_png=os.path.join(out_dir, "roc_curves_meta.png"),
    )

    # 4) PR curves (folds + OOF)
    plot_pr_curves_meta(
        folds_meta=folds_meta,
        oof_pair=(y_oof_true, p_oof_meta),
        out_png=os.path.join(out_dir, "pr_curves_meta.png"),
    )

    # 5) Violins by class
    plot_oof_score_violins(
        y_true=y_oof_true,
        y_score=p_oof_meta,
        out_png=os.path.join(out_dir, "score_distributions_oof_meta.png"),
    )

    # 6) Performance table across models
    plot_performance_table(
        fold_metrics=fold_metrics,
        out_png=os.path.join(out_dir, "performance_table.png"),
        title="Performance Metrics by Fold",
    )

    # 7) Average metric bars for the meta model
    # compute averages safely
    meta_block = fold_metrics.get("meta", {})
    avg_meta = {
        "F1": float(np.nanmean(meta_block.get("f1", []))) if len(meta_block.get("f1", [])) else np.nan,
        "Precision": float(np.nanmean(meta_block.get("precision", []))) if len(meta_block.get("precision", [])) else np.nan,
        "Recall": float(np.nanmean(meta_block.get("recall", []))) if len(meta_block.get("recall", [])) else np.nan,
        "AUC": float(np.nanmean(meta_block.get("auc", []))) if len(meta_block.get("auc", [])) else np.nan,
    }
    plot_avg_metric_bars(
        avg_metrics=avg_meta,
        label="Meta",
        out_png=os.path.join(out_dir, "summary_bars.png"),
    )
