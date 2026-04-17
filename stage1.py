"""
stage1.py — Baseline Manifold Learning & Probabilistic Classifiers
==================================================================
1. UMAP dimensionality reduction on each MedMNIST2D dataset.
2. Train GNB, LDA, QDA, and GMM classifiers on the reduced data.
3. GMM uses soft-clustering with per-class mixture models and
   multiclass membership probabilities.
"""

import gc
import warnings
import numpy as np

from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.mixture import GaussianMixture
import umap

from modules.data_loader import MEDMNIST2D_DATASETS, load_medmnist_flat
from modules.metrics import (
    compute_metrics, record_result,
    plot_confusion_matrix, plot_roc_curves, plot_2d_projection,
)

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────
# UMAP reducer
# ──────────────────────────────────────────────────────────────────────

def fit_umap(X_train, X_test, n_components=50, n_neighbors=15,
             min_dist=0.1, random_state=42):
    """
    Fit UMAP on training data, transform both train and test.
    Uses a moderate n_components (50) for classifier input, and
    separately returns a 2-D embedding for visualisation.
    """
    reducer = umap.UMAP(n_components=n_components,
                         n_neighbors=n_neighbors,
                         min_dist=min_dist,
                         metric="euclidean",
                         random_state=random_state)
    X_tr_red = reducer.fit_transform(X_train)
    X_te_red = reducer.transform(X_test)

    # 2-D embedding for plots (fit on train only)
    vis_reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors,
                             min_dist=min_dist, random_state=random_state)
    X_tr_2d = vis_reducer.fit_transform(X_train)
    X_te_2d = vis_reducer.transform(X_test)
    gc.collect()
    return X_tr_red, X_te_red, X_tr_2d, X_te_2d


# ──────────────────────────────────────────────────────────────────────
# GMM Classifier with soft-clustering & membership probabilities
# ──────────────────────────────────────────────────────────────────────

class GMMClassifier:
    """
    Gaussian Mixture Model classifier.

    Strategy:
      • Fit one GMM per class (soft clustering intuition).
      • At inference, compute log-likelihood under each class GMM
        weighted by class prior → multiclass membership probabilities.

    This respects the "soft clustering & multiclass membership" requirement.
    """

    def __init__(self, n_components_per_class: int = 3,
                 covariance_type: str = "diag",
                 max_iter: int = 200, random_state: int = 42):
        self.n_components = n_components_per_class
        self.cov_type = covariance_type
        self.max_iter = max_iter
        self.rs = random_state
        self.models_ = {}       # class_label -> GaussianMixture
        self.priors_ = {}       # class_label -> prior probability
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        n_total = len(y)
        for c in self.classes_:
            mask = y == c
            self.priors_[c] = mask.sum() / n_total
            n_comp = min(self.n_components, mask.sum())
            gmm = GaussianMixture(
                n_components=max(1, n_comp),
                covariance_type=self.cov_type,
                max_iter=self.max_iter,
                random_state=self.rs,
            )
            gmm.fit(X[mask])
            self.models_[c] = gmm
        gc.collect()
        return self

    def predict_proba(self, X):
        log_likes = np.column_stack([
            self.models_[c].score_samples(X) + np.log(self.priors_[c] + 1e-30)
            for c in self.classes_
        ])
        # Softmax for normalised probabilities
        log_likes -= log_likes.max(axis=1, keepdims=True)
        probs = np.exp(log_likes)
        probs /= probs.sum(axis=1, keepdims=True)
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]


# ──────────────────────────────────────────────────────────────────────
# Evaluation loop for one dataset
# ──────────────────────────────────────────────────────────────────────

def _evaluate_clf(clf, name, X_tr, y_tr, X_te, y_te, n_classes, ds_name):
    """Fit, predict, record metrics, and plot."""
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    y_prob = None
    if hasattr(clf, "predict_proba"):
        try:
            y_prob = clf.predict_proba(X_te)
        except Exception:
            y_prob = None
    m = compute_metrics(y_te, y_pred, y_prob, n_classes)
    record_result("Stage1", ds_name, name, m)
    plot_confusion_matrix(y_te, y_pred, title=f"[S1] {name} — {ds_name}")
    if y_prob is not None and n_classes <= 20:
        plot_roc_curves(y_te, y_prob, n_classes,
                        title=f"[S1] ROC {name} — {ds_name}")
    print(f"  {name:6s}  Acc={m['accuracy']:.4f}  F1={m['f1']:.4f}  "
          f"AUC={m['auc']:.4f}")
    gc.collect()


# ──────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────

def run_stage1(datasets=None, umap_dim=50, root="./data"):
    """
    Execute Stage 1 on the requested MedMNIST2D datasets.

    Parameters
    ----------
    datasets : list[str] | None
        Subset of MEDMNIST2D_DATASETS to run. None → all 12.
    umap_dim : int
        Intermediate UMAP dimension for classifiers.
    root : str
        Download directory for MedMNIST data.
    """
    if datasets is None:
        datasets = MEDMNIST2D_DATASETS

    for ds_name in datasets:
        print(f"\n{'='*60}")
        print(f" Stage 1 — {ds_name}")
        print(f"{'='*60}")

        X_tr, y_tr, _, _, X_te, y_te, n_classes, task_type = \
            load_medmnist_flat(ds_name, root=root)

        # Clamp UMAP dim to be < n_features and < n_samples
        effective_dim = min(umap_dim, X_tr.shape[1] - 1, X_tr.shape[0] - 2)
        effective_dim = max(effective_dim, 2)

        X_tr_red, X_te_red, X_tr_2d, X_te_2d = fit_umap(
            X_tr, X_te, n_components=effective_dim)

        # Visualise UMAP 2-D embedding
        plot_2d_projection(X_te_2d, y_te, "UMAP",
                           title=f"[S1] UMAP 2D — {ds_name}")

        # ──── classifiers ────
        # Clamp LDA components
        lda_n = min(effective_dim, n_classes - 1) if n_classes > 2 else None

        classifiers = [
            ("GNB", GaussianNB()),
            ("LDA", LinearDiscriminantAnalysis()),
            ("QDA", QuadraticDiscriminantAnalysis(reg_param=1e-2)),
            ("GMM", GMMClassifier(n_components_per_class=3,
                                  covariance_type="diag")),
        ]

        for clf_name, clf in classifiers:
            try:
                _evaluate_clf(clf, clf_name, X_tr_red, y_tr,
                              X_te_red, y_te, n_classes, ds_name)
            except Exception as e:
                print(f"  {clf_name} FAILED on {ds_name}: {e}")
                record_result("Stage1", ds_name, clf_name,
                              dict(accuracy=np.nan, precision=np.nan,
                                   recall=np.nan, f1=np.nan, auc=np.nan))

        del X_tr, y_tr, X_te, y_te, X_tr_red, X_te_red, X_tr_2d, X_te_2d
        gc.collect()

    print("\n✓ Stage 1 complete.")
