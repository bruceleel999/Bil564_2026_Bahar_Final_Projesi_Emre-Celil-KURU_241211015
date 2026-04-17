"""
metrics.py
==========
Unified evaluation, visualisation, and comparative-table utilities
used by all three stages of the pipeline.
"""

import gc
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    ConfusionMatrixDisplay, RocCurveDisplay
)
from sklearn.preprocessing import label_binarize

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────
# Core metric computation
# ──────────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_prob: np.ndarray = None, n_classes: int = 2,
                    average: str = "macro") -> dict:
    """
    Compute Accuracy, Precision, Recall, F1, and AUC.

    Parameters
    ----------
    y_true   : 1-D ground-truth labels.
    y_pred   : 1-D predicted labels.
    y_prob   : (N, C) class probabilities – needed for AUC. Optional.
    n_classes: number of classes.
    average  : averaging strategy for multi-class metrics.

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1, auc
    """
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average=average, zero_division=0)
    rec  = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1   = f1_score(y_true, y_pred, average=average, zero_division=0)

    auc_val = np.nan
    if y_prob is not None:
        try:
            if n_classes == 2:
                prob_col = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
                auc_val = roc_auc_score(y_true, prob_col)
            else:
                y_bin = label_binarize(y_true, classes=np.arange(n_classes))
                if y_prob.shape[1] == n_classes:
                    auc_val = roc_auc_score(y_bin, y_prob, average=average,
                                            multi_class="ovr")
        except Exception:
            auc_val = np.nan

    return dict(accuracy=acc, precision=prec, recall=rec, f1=f1, auc=auc_val)


# ──────────────────────────────────────────────────────────────────────
# Visualisation helpers
# ──────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix",
                          figsize=(6, 5)):
    """Plot a normalised confusion matrix heatmap."""
    labels = np.unique(np.concatenate([y_true, y_pred]))
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=len(labels) <= 15, fmt=".2f", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set(title=title, xlabel="Predicted", ylabel="True")
    plt.tight_layout()
    plt.show()
    plt.close(fig)
    gc.collect()


def plot_roc_curves(y_true, y_prob, n_classes, title="ROC Curves",
                    figsize=(7, 5)):
    """Plot per-class ROC curves (OvR) for multi-class problems."""
    y_bin = label_binarize(y_true, classes=np.arange(n_classes))
    fig, ax = plt.subplots(figsize=figsize)
    n_show = min(n_classes, 10)  # limit legend clutter
    for i in range(n_show):
        try:
            RocCurveDisplay.from_predictions(
                y_bin[:, i], y_prob[:, i], ax=ax, name=f"Class {i}")
        except Exception:
            pass
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set(title=title)
    ax.legend(fontsize=7, loc="lower right")
    plt.tight_layout()
    plt.show()
    plt.close(fig)
    gc.collect()


def plot_2d_projection(X_2d: np.ndarray, y: np.ndarray,
                       method_name: str = "UMAP",
                       title: str = "", figsize=(6, 5)):
    """Scatter plot of a 2-D embedding coloured by class."""
    fig, ax = plt.subplots(figsize=figsize)
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap="tab20",
                         s=3, alpha=0.5)
    ax.set(title=title or f"{method_name} projection",
           xlabel=f"{method_name}-1", ylabel=f"{method_name}-2")
    plt.colorbar(scatter, ax=ax, shrink=0.7)
    plt.tight_layout()
    plt.show()
    plt.close(fig)
    gc.collect()


# ──────────────────────────────────────────────────────────────────────
# Comparative results table
# ──────────────────────────────────────────────────────────────────────

# Global accumulator –– modules append rows via record_result()
_ALL_RESULTS: list[dict] = []


def record_result(stage: str, dataset: str, model: str, metrics: dict):
    """Append a result row to the global results accumulator."""
    row = dict(stage=stage, dataset=dataset, model=model, **metrics)
    _ALL_RESULTS.append(row)


def build_comparison_table() -> pd.DataFrame:
    """Return a styled DataFrame of all recorded results."""
    df = pd.DataFrame(_ALL_RESULTS)
    cols_order = ["stage", "dataset", "model",
                  "accuracy", "precision", "recall", "f1", "auc"]
    cols_present = [c for c in cols_order if c in df.columns]
    df = df[cols_present].sort_values(["stage", "dataset", "model"])
    return df.reset_index(drop=True)


def display_final_table():
    """Pretty-print the full comparative table."""
    df = build_comparison_table()
    numeric_cols = ["accuracy", "precision", "recall", "f1", "auc"]
    fmt = {c: "{:.4f}" for c in numeric_cols if c in df.columns}
    styled = df.style.format(fmt, na_rep="—").background_gradient(
        cmap="YlGn", subset=[c for c in numeric_cols if c in df.columns])
    from IPython.display import display
    display(styled)
    return df
