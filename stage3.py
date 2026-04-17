"""
stage3.py — Massive Topology & Noise Immunity Pipeline
======================================================
Orchestrates the full Stage-3 pipeline:
 1. Dynamic dilution (MAIS + ESS)
 2. Entropy selection (100 lowest)
 3. Stochastic cascade augmentation (29 methods)
 4. ADASYN + Tomek boundary cleaning
 5. Morphological expansion (SLERP in PCA)
 6. Randomized SVD reduction
 7. Coreset selection (HDBSCAN + entropy)
 8. Barnes-Hut t-SNE mapping
 9. Curriculum-learning BMA ensemble (5 categories, ~30 models)
"""

import gc
import warnings
import numpy as np
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier
)
from sklearn.linear_model import (
    Perceptron, LogisticRegression, SGDClassifier
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import catboost as cb
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import TomekLinks

from modules.data_loader import MEDMNIST2D_DATASETS, load_medmnist_flat
from modules.metrics import (
    compute_metrics, record_result,
    plot_confusion_matrix, plot_roc_curves, plot_2d_projection,
)
from modules.stage3_utils import (
    split_and_dilute_mais, select_low_entropy,
    stochastic_cascade_augment, morphological_expand_slerp,
    randomized_svd_reduce, coreset_selection, NystromKernel,
)

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────────────
# Fuzzy-KNN classifier
# ──────────────────────────────────────────────────────────────────────

class FuzzyKNN:
    """
    Fuzzy K-Nearest Neighbours classifier.
    Membership degree based on inverse-distance weighting.
    """
    def __init__(self, n_neighbors=5):
        self.k = n_neighbors

    def fit(self, X, y):
        self.X_train = X.copy()
        self.y_train = y.copy()
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        from sklearn.metrics.pairwise import euclidean_distances
        D = euclidean_distances(X, self.X_train)  # (N_test, N_train)
        probs = np.zeros((len(X), len(self.classes_)))
        for i in range(len(X)):
            nn_idx = np.argsort(D[i])[:self.k]
            nn_dists = D[i, nn_idx] + 1e-10
            weights = 1.0 / nn_dists
            for j, c in enumerate(self.classes_):
                mask = self.y_train[nn_idx] == c
                probs[i, j] = weights[mask].sum()
            probs[i] /= probs[i].sum() + 1e-10
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]


class FCM_KNN_Hybrid:
    """
    FCM-KNN Hybrid: Fuzzy C-Means membership + KNN distance fusion.
    Uses KNN distance to weight FCM-derived memberships.
    """
    def __init__(self, n_neighbors=5):
        self.k = n_neighbors

    def fit(self, X, y):
        self.X_train = X.copy()
        self.y_train = y.copy()
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        from sklearn.metrics.pairwise import euclidean_distances
        D = euclidean_distances(X, self.X_train)
        probs = np.zeros((len(X), len(self.classes_)))
        for i in range(len(X)):
            nn_idx = np.argsort(D[i])[:self.k]
            nn_dists = D[i, nn_idx] + 1e-10
            # FCM-style membership: 1 / d^(2/(m-1)), m=2
            memberships = 1.0 / (nn_dists ** 2)
            for j, c in enumerate(self.classes_):
                mask = self.y_train[nn_idx] == c
                probs[i, j] = memberships[mask].sum()
            probs[i] /= probs[i].sum() + 1e-10
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]


# ──────────────────────────────────────────────────────────────────────
# Bayesian Model Averaging
# ──────────────────────────────────────────────────────────────────────

class BayesianModelAveraging:
    """
    Bayesian Model Averaging ensemble.
    Weights ∝ exp(log_likelihood) on a validation set, with Laplace
    smoothing to avoid zero weights.
    """
    def __init__(self, models: dict):
        self.models = models      # {name: fitted_model}
        self.weights_ = {}

    def calibrate(self, X_val, y_val):
        """Compute BMA weights from validation log-likelihoods."""
        log_likes = {}
        for name, clf in self.models.items():
            try:
                proba = clf.predict_proba(X_val)
                n_cls = proba.shape[1]
                idx = np.clip(y_val, 0, n_cls - 1)
                ll = np.log(proba[np.arange(len(y_val)), idx] + 1e-15).sum()
            except Exception:
                ll = -1e10
            log_likes[name] = ll

        max_ll = max(log_likes.values())
        raw = {k: np.exp(v - max_ll) for k, v in log_likes.items()}
        total = sum(raw.values()) + 1e-10
        self.weights_ = {k: v / total for k, v in raw.items()}

    def predict_proba(self, X):
        """Weighted average of all model probabilities."""
        proba = None
        for name, clf in self.models.items():
            w = self.weights_.get(name, 0)
            if w < 1e-8:
                continue
            try:
                p = clf.predict_proba(X) * w
                proba = p if proba is None else proba + p
            except Exception:
                continue
        if proba is None:
            raise RuntimeError("No model produced valid probabilities")
        return proba

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


# ──────────────────────────────────────────────────────────────────────
# Build all 30 ensemble models
# ──────────────────────────────────────────────────────────────────────

def _build_ensemble(n_classes, n_features):
    """
    Return dict {name: sklearn_estimator} for all 5 categories.
    Uses Nyström approximation for non-linear SGD-SVM kernels.
    """
    nyst_components = min(300, n_features)

    models = {}

    # ── Category 1: KNN ──
    models["KNN_5"] = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    models["FuzzyKNN"] = FuzzyKNN(n_neighbors=5)
    models["FCM_KNN"] = FCM_KNN_Hybrid(n_neighbors=5)

    # ── Category 2: Tree Ensemble ──
    models["DecisionTree"] = DecisionTreeClassifier(max_depth=20,
                                                     random_state=42)
    models["BaggingTree"] = BaggingClassifier(
        estimator=DecisionTreeClassifier(max_depth=15),
        n_estimators=20, random_state=42, n_jobs=-1)
    models["RandomForest"] = RandomForestClassifier(
        n_estimators=50, max_depth=20, random_state=42, n_jobs=-1)
    models["GBM"] = GradientBoostingClassifier(
        n_estimators=50, max_depth=5, random_state=42)
    models["XGBoost"] = xgb.XGBClassifier(
        n_estimators=50, max_depth=5, use_label_encoder=False,
        eval_metric="mlogloss", random_state=42, verbosity=0, n_jobs=-1)
    models["CatBoost"] = cb.CatBoostClassifier(
        iterations=50, depth=5, verbose=0, random_state=42)

    # ── Category 3: Linear / Single Layer ──
    models["Perceptron"] = Perceptron(max_iter=200, random_state=42)
    models["LogReg_Sigmoid"] = LogisticRegression(
        max_iter=500, solver="lbfgs", multi_class="ovr", random_state=42)
    models["LogReg_Softmax"] = LogisticRegression(
        max_iter=500, solver="lbfgs", multi_class="multinomial",
        random_state=42)

    # ── Category 4: SGD-SVM + Nyström Kernels ──
    def _sgd_nystrom_pipe(kernel_name, **kernel_kwargs):
        return Pipeline([
            ("scaler", StandardScaler()),
            ("nystrom", NystromKernel(kernel=kernel_name,
                                      n_components=nyst_components,
                                      **kernel_kwargs)),
            ("sgd", SGDClassifier(loss="hinge", max_iter=300,
                                  random_state=42)),
        ])

    models["SGD_SVM_Linear"] = Pipeline([
        ("scaler", StandardScaler()),
        ("sgd", SGDClassifier(loss="hinge", max_iter=300, random_state=42)),
    ])
    models["SGD_SVM_RBF"] = _sgd_nystrom_pipe("rbf", gamma=1.0)
    models["SGD_SVM_Poly2"] = _sgd_nystrom_pipe("poly", degree=2, gamma=1.0)
    models["SGD_SVM_Poly3"] = _sgd_nystrom_pipe("poly", degree=3, gamma=1.0)
    models["SGD_SVM_Sigmoid"] = _sgd_nystrom_pipe("sigmoid", gamma=0.01)
    models["SGD_SVM_GRPF"] = _sgd_nystrom_pipe("grpf", gamma=1.0)
    models["SGD_SVM_tStudent"] = _sgd_nystrom_pipe("t_student", degree=2)
    models["SGD_SVM_InvMulti"] = _sgd_nystrom_pipe("inv_multiquadric",
                                                     coef0=1.0)

    # ── Category 5: Raw SGD Loss Models ──
    for loss_name, key in [("huber", "SGD_Raw_Huber"),
                            ("modified_huber", "SGD_Raw_ModHuber"),
                            ("squared_hinge", "SGD_Raw_SqHinge"),
                            ("perceptron", "SGD_Raw_Perceptron"),
                            ("log_loss", "SGD_Raw_LogLoss")]:
        models[key] = Pipeline([
            ("scaler", StandardScaler()),
            ("sgd", SGDClassifier(loss=loss_name, max_iter=300,
                                  random_state=42)),
        ])

    return models


# ──────────────────────────────────────────────────────────────────────
# Data preprocessing pipeline
# ──────────────────────────────────────────────────────────────────────

def _preprocess_stage3(X_tr, y_tr, X_te, y_te, ds_name):
    """
    Execute Steps 1-8 of the Stage-3 pipeline. Returns processed
    training and test data ready for the ensemble.
    """
    # Step 1: Dynamic dilution (MAIS + ESS)
    print("    [1/8] Dynamic dilution (MAIS + ESS) …")
    X_dil, y_dil = split_and_dilute_mais(X_tr, y_tr, n_subsets=100)
    print(f"          {len(X_tr)} → {len(X_dil)} samples after dilution")

    # Step 2: Entropy selection — 100 lowest-entropy samples
    print("    [2/8] Entropy selection (100 lowest) …")
    X_ent, y_ent = select_low_entropy(X_dil, y_dil, n_select=100)

    # Step 3: Stochastic cascade augmentation (29 methods × 50%)
    print("    [3/8] Stochastic cascade augmentation …")
    X_aug, y_aug = stochastic_cascade_augment(X_ent, y_ent)
    print(f"          → {len(X_aug)} augmented samples")

    # Step 4: Boundary cleaning (ADASYN + Tomek Links)
    print("    [4/8] Boundary cleaning (ADASYN + Tomek) …")
    try:
        classes, counts = np.unique(y_aug, return_counts=True)
        if len(classes) > 1 and counts.min() >= 2:
            adasyn = ADASYN(random_state=42, n_neighbors=min(3, counts.min()-1))
            X_aug, y_aug = adasyn.fit_resample(X_aug, y_aug)
        tomek = TomekLinks()
        X_aug, y_aug = tomek.fit_resample(X_aug, y_aug)
    except Exception as e:
        print(f"          ADASYN/Tomek skipped: {e}")

    # Step 5: Morphological expansion (SLERP in PCA) — on ORIGINAL data
    print("    [5/8] Morphological expansion (SLERP in PCA) …")
    # Use a subsample of the original to keep memory manageable
    max_slerp = min(5000, len(X_tr))
    idx_sub = np.random.choice(len(X_tr), max_slerp, replace=False)
    X_slerp, y_slerp = morphological_expand_slerp(
        X_tr[idx_sub], y_tr[idx_sub])
    print(f"          → {len(X_slerp)} SLERP-expanded samples")

    # Step 6: Combine pools + Randomized SVD
    print("    [6/8] Randomized SVD reduction …")
    X_combined = np.vstack([X_aug, X_slerp]).astype(np.float32)
    y_combined = np.concatenate([y_aug, y_slerp])
    X_reduced = randomized_svd_reduce(X_combined, variance_target=0.95)
    del X_combined, X_aug, X_slerp; gc.collect()

    # Also reduce test set with same SVD (refit for safety)
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_r_scaled = scaler.fit_transform(
        np.vstack([X_reduced, np.zeros((1, X_reduced.shape[1]))]))
    # Reduce test
    X_te_scaled = StandardScaler().fit_transform(X_te)
    svd_te = TruncatedSVD(n_components=min(X_reduced.shape[1], X_te.shape[1]-1),
                           random_state=42)
    X_te_red = svd_te.fit_transform(X_te_scaled)

    # Step 7: Coreset selection
    print("    [7/8] Coreset selection (HDBSCAN + entropy) …")
    X_core, y_core = coreset_selection(X_reduced, y_combined, top_pct=0.10)
    print(f"          → {len(X_core)} coreset samples")

    # Ensure enough classes for training
    unique_core = np.unique(y_core)
    if len(unique_core) < 2:
        print("          ⚠ Coreset has < 2 classes; using full reduced set")
        X_core, y_core = X_reduced, y_combined

    # Step 8: Barnes-Hut t-SNE mapping (for vis; train on reduced for models)
    print("    [8/8] Barnes-Hut t-SNE mapping …")
    n_tsne = min(3000, len(X_core))
    tsne = TSNE(n_components=2, method="barnes_hut", perplexity=30,
                random_state=42)
    X_tsne_2d = tsne.fit_transform(X_core[:n_tsne])
    plot_2d_projection(X_tsne_2d, y_core[:n_tsne], "t-SNE",
                       title=f"[S3] Barnes-Hut t-SNE — {ds_name}")

    # Align test dimensionality with training
    target_dim = X_core.shape[1]
    if X_te_red.shape[1] > target_dim:
        X_te_red = X_te_red[:, :target_dim]
    elif X_te_red.shape[1] < target_dim:
        pad = np.zeros((X_te_red.shape[0], target_dim - X_te_red.shape[1]),
                       dtype=np.float32)
        X_te_red = np.hstack([X_te_red, pad])

    gc.collect()
    return X_core, y_core, X_te_red, y_te


# ──────────────────────────────────────────────────────────────────────
# Curriculum learning: sort by difficulty (loss / entropy)
# ──────────────────────────────────────────────────────────────────────

def _curriculum_sort(X, y):
    """
    Sort training samples easy → hard based on distance to class centroid
    (proxy for difficulty). Returns reordered X, y.
    """
    classes = np.unique(y)
    centroids = {c: X[y == c].mean(axis=0) for c in classes}
    difficulties = np.array([
        np.linalg.norm(X[i] - centroids[y[i]])
        for i in range(len(X))
    ])
    order = np.argsort(difficulties)
    return X[order], y[order]


# ──────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────

def run_stage3(datasets=None, root="./data"):
    """
    Execute the full Stage-3 pipeline on the selected MedMNIST2D datasets.
    """
    if datasets is None:
        datasets = MEDMNIST2D_DATASETS

    for ds_name in datasets:
        print(f"\n{'='*60}")
        print(f" Stage 3 — {ds_name}")
        print(f"{'='*60}")

        X_tr, y_tr, X_val, y_val, X_te, y_te, n_classes, _ = \
            load_medmnist_flat(ds_name, root=root)

        # Preprocess (Steps 1-8)
        X_core, y_core, X_te_red, y_te_final = \
            _preprocess_stage3(X_tr, y_tr, X_te, y_te, ds_name)

        # Curriculum sort
        X_core, y_core = _curriculum_sort(X_core, y_core)

        # Build ensemble
        print("    [9/9] Curriculum BMA ensemble training …")
        model_dict = _build_ensemble(n_classes, X_core.shape[1])

        fitted_models = {}
        for mname, clf in model_dict.items():
            try:
                clf.fit(X_core, y_core)
                fitted_models[mname] = clf
                # Individual evaluation
                y_pred = clf.predict(X_te_red)
                y_prob = None
                if hasattr(clf, "predict_proba"):
                    try:
                        y_prob = clf.predict_proba(X_te_red)
                    except Exception:
                        y_prob = None
                elif hasattr(clf, "decision_function"):
                    pass  # some SGD models

                m = compute_metrics(y_te_final, y_pred, y_prob, n_classes)
                record_result("Stage3", ds_name, mname, m)
                print(f"      {mname:25s}  Acc={m['accuracy']:.4f}  "
                      f"F1={m['f1']:.4f}")
            except Exception as e:
                print(f"      {mname:25s}  FAILED: {e}")
                record_result("Stage3", ds_name, mname,
                              dict(accuracy=np.nan, precision=np.nan,
                                   recall=np.nan, f1=np.nan, auc=np.nan))

        # BMA ensemble
        if len(fitted_models) >= 2:
            bma = BayesianModelAveraging(fitted_models)
            # Calibrate on a subset of training data (as validation proxy)
            val_size = min(500, len(X_core))
            bma.calibrate(X_core[:val_size], y_core[:val_size])
            try:
                y_pred_bma = bma.predict(X_te_red)
                y_prob_bma = bma.predict_proba(X_te_red)
                m_bma = compute_metrics(y_te_final, y_pred_bma,
                                        y_prob_bma, n_classes)
                record_result("Stage3", ds_name, "BMA_Ensemble", m_bma)
                plot_confusion_matrix(y_te_final, y_pred_bma,
                                      title=f"[S3] BMA Ensemble — {ds_name}")
                if n_classes <= 20:
                    plot_roc_curves(y_te_final, y_prob_bma, n_classes,
                                   title=f"[S3] ROC BMA — {ds_name}")
                print(f"      {'BMA_Ensemble':25s}  Acc={m_bma['accuracy']:.4f}  "
                      f"F1={m_bma['f1']:.4f}  AUC={m_bma['auc']:.4f}")
            except Exception as e:
                print(f"      BMA_Ensemble FAILED: {e}")

        del X_tr, y_tr, X_val, y_val, X_te, y_te
        del X_core, y_core, X_te_red
        gc.collect()

    print("\n✓ Stage 3 complete.")
