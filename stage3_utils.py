"""
stage3_utils.py — Stage 3 Helper Classes & Functions
====================================================
Custom implementations:
  - MAIS / ESS dilution
  - Stochastic cascade augmentation (29 methods)
  - SLERP in PCA space
  - Custom Nyström kernel approximations
  - Coreset selection (HDBSCAN + entropy)
"""

import gc
import math
import warnings
import numpy as np
from scipy.stats import entropy as sp_entropy
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.utils import resample

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────────────
# 1. Dynamic Dilution via MAIS + ESS
# ──────────────────────────────────────────────────────────────────────

def split_and_dilute_mais(X, y, n_subsets=100, ess_threshold=0.5):
    """
    Split data into n_subsets, compute importance weights via
    Multiple Adaptive Importance Sampling, keep subsets whose
    Effective Sample Size exceeds ess_threshold * subset_size.

    Returns combined (X_diluted, y_diluted).
    """
    N = len(X)
    indices = np.arange(N)
    np.random.shuffle(indices)
    subsets = np.array_split(indices, n_subsets)

    # Global proposal: uniform; target: density proportional to ||x||
    global_norms = np.linalg.norm(X, axis=1) + 1e-10
    global_mean_norm = global_norms.mean()

    kept_idx = []
    for sub_idx in subsets:
        if len(sub_idx) < 2:
            continue
        sub_norms = global_norms[sub_idx]
        # Importance weights: target / proposal ≈ norm / mean_norm
        weights = sub_norms / global_mean_norm
        weights /= weights.sum()
        # ESS = 1 / sum(w_i^2), normalised by subset size
        ess = 1.0 / (weights ** 2).sum()
        if ess >= ess_threshold * len(sub_idx):
            kept_idx.append(sub_idx)

    if len(kept_idx) == 0:
        # Fallback: keep all
        return X, y
    kept = np.concatenate(kept_idx)
    gc.collect()
    return X[kept], y[kept]


# ──────────────────────────────────────────────────────────────────────
# 2. Entropy Selection (lowest-entropy samples)
# ──────────────────────────────────────────────────────────────────────

def select_low_entropy(X, y, n_select=100):
    """
    Treat each flattened image as a soft probability vector (after
    normalisation) and pick the n_select samples with lowest Shannon entropy.
    Low-entropy ≈ high-confidence / less noisy.
    """
    # Shift to non-negative, normalise per sample
    X_shifted = X - X.min(axis=1, keepdims=True) + 1e-10
    X_norm = X_shifted / X_shifted.sum(axis=1, keepdims=True)
    ent = sp_entropy(X_norm, axis=1)
    sel = np.argsort(ent)[:n_select]
    return X[sel], y[sel]


# ──────────────────────────────────────────────────────────────────────
# 3. Stochastic Cascade Augmentation (29 methods, 50 % coin flip)
# ──────────────────────────────────────────────────────────────────────

def _aug_noise(x):
    return x + np.random.normal(0, 0.02, x.shape).astype(np.float32)

def _aug_scale(x):
    return x * np.random.uniform(0.9, 1.1)

def _aug_shift(x):
    return np.roll(x, np.random.randint(-2, 3))

def _aug_flip(x):
    return x[::-1].copy()

def _aug_dropout(x):
    mask = np.random.binomial(1, 0.95, x.shape)
    return x * mask

def _aug_blur(x):
    k = 3
    kernel = np.ones(k) / k
    return np.convolve(x, kernel, mode='same').astype(np.float32)

def _aug_sharpen(x):
    blurred = _aug_blur(x)
    return np.clip(2 * x - blurred, 0, 1).astype(np.float32)

def _aug_contrast(x):
    mu = x.mean()
    alpha = np.random.uniform(0.8, 1.2)
    return np.clip(alpha * (x - mu) + mu, 0, 1).astype(np.float32)

def _aug_brightness(x):
    return np.clip(x + np.random.uniform(-0.05, 0.05), 0, 1).astype(np.float32)

def _aug_invert(x):
    return 1.0 - x

def _aug_clip_high(x):
    return np.clip(x, 0, 0.95).astype(np.float32)

def _aug_clip_low(x):
    return np.clip(x, 0.05, 1.0).astype(np.float32)

def _aug_power(x):
    gamma = np.random.uniform(0.8, 1.2)
    return np.power(np.clip(x, 1e-6, 1), gamma).astype(np.float32)

def _aug_log(x):
    return np.log1p(x).astype(np.float32)

def _aug_exp(x):
    return np.clip(np.expm1(x), 0, 1).astype(np.float32)

def _aug_quantize(x):
    levels = np.random.choice([4, 8, 16])
    return (np.round(x * levels) / levels).astype(np.float32)

def _aug_salt_pepper(x):
    mask_salt = np.random.binomial(1, 0.01, x.shape)
    mask_pepper = np.random.binomial(1, 0.01, x.shape)
    out = x.copy()
    out[mask_salt == 1] = 1.0
    out[mask_pepper == 1] = 0.0
    return out

def _aug_elastic(x):
    noise = np.random.normal(0, 0.5, x.shape)
    return np.clip(x + noise * 0.02, 0, 1).astype(np.float32)

def _aug_cutout(x):
    L = len(x)
    start = np.random.randint(0, max(1, L - L // 10))
    end = min(start + L // 10, L)
    out = x.copy()
    out[start:end] = 0
    return out

def _aug_mixup_self(x):
    perm = np.random.permutation(len(x))
    lam = np.random.beta(0.2, 0.2)
    return (lam * x + (1 - lam) * x[perm]).astype(np.float32)

def _aug_rotation_flat(x):
    # Rotate flattened 28x28 by 90°
    side = int(math.sqrt(len(x)))
    if side * side != len(x):
        return x
    img = x[:side*side].reshape(side, side)
    return np.rot90(img, k=np.random.choice([1, 2, 3])).flatten().astype(np.float32)

def _aug_zoom(x):
    return np.clip(x * np.random.uniform(0.95, 1.05), 0, 1).astype(np.float32)

def _aug_channel_shuffle(x):
    perm = np.random.permutation(len(x))[:len(x)//4]
    out = x.copy(); out[perm] = 0
    return out

def _aug_hist_eq(x):
    # Simplified histogram equalisation on flat vector
    sorted_vals = np.sort(x)
    ranks = np.searchsorted(sorted_vals, x).astype(np.float32)
    return (ranks / (len(x) + 1e-10)).astype(np.float32)

def _aug_solarize(x):
    threshold = 0.5
    out = x.copy()
    out[out > threshold] = 1.0 - out[out > threshold]
    return out

def _aug_posterize(x):
    bits = np.random.choice([2, 3, 4])
    levels = 2 ** bits
    return (np.round(x * levels) / levels).astype(np.float32)

def _aug_jitter(x):
    return np.clip(x + np.random.uniform(-0.03, 0.03, x.shape), 0, 1).astype(np.float32)

def _aug_smooth(x):
    kernel = np.array([0.25, 0.5, 0.25], dtype=np.float32)
    return np.convolve(x, kernel, mode='same').astype(np.float32)

def _aug_identity(x):
    return x.copy()

AUGMENTATION_CASCADE = [
    _aug_noise, _aug_scale, _aug_shift, _aug_flip, _aug_dropout,
    _aug_blur, _aug_sharpen, _aug_contrast, _aug_brightness, _aug_invert,
    _aug_clip_high, _aug_clip_low, _aug_power, _aug_log, _aug_exp,
    _aug_quantize, _aug_salt_pepper, _aug_elastic, _aug_cutout,
    _aug_mixup_self, _aug_rotation_flat, _aug_zoom, _aug_channel_shuffle,
    _aug_hist_eq, _aug_solarize, _aug_posterize, _aug_jitter,
    _aug_smooth, _aug_identity,
]   # 29 methods

assert len(AUGMENTATION_CASCADE) == 29, \
    f"Expected 29 augmentations, got {len(AUGMENTATION_CASCADE)}"


def stochastic_cascade_augment(X, y):
    """
    Pass each sample through the cascade of 29 augmentation
    methods, each with 50 % trigger probability.
    Returns augmented arrays (may grow).
    """
    aug_X, aug_y = [X.copy()], [y.copy()]
    for x_i, y_i in zip(X, y):
        augmented = x_i.copy()
        for fn in AUGMENTATION_CASCADE:
            if np.random.rand() < 0.5:
                augmented = fn(augmented)
        aug_X.append(augmented[np.newaxis, :])
        aug_y.append(np.array([y_i]))
    gc.collect()
    return np.concatenate(aug_X, axis=0), np.concatenate(aug_y, axis=0)


# ──────────────────────────────────────────────────────────────────────
# 4. SLERP in PCA Space (Morphological Expansion)
# ──────────────────────────────────────────────────────────────────────

def slerp(v0, v1, t):
    """Spherical Linear Interpolation between two vectors."""
    v0n = v0 / (np.linalg.norm(v0) + 1e-10)
    v1n = v1 / (np.linalg.norm(v1) + 1e-10)
    dot = np.clip(np.dot(v0n, v1n), -1.0, 1.0)
    omega = np.arccos(dot)
    if abs(omega) < 1e-6:
        return (1 - t) * v0 + t * v1
    so = np.sin(omega)
    return (np.sin((1 - t) * omega) / so) * v0 + \
           (np.sin(t * omega) / so) * v1


def morphological_expand_slerp(X, y, pca_dim=50, random_state=42):
    """
    Project X into PCA space, double dataset size via SLERP
    interpolations between random same-class pairs, project back.
    """
    pca = PCA(n_components=min(pca_dim, *X.shape), random_state=random_state)
    X_pca = pca.fit_transform(X)

    rng = np.random.RandomState(random_state)
    new_X, new_y = [], []
    classes = np.unique(y)
    for c in classes:
        idx_c = np.where(y == c)[0]
        n_gen = len(idx_c)
        for _ in range(n_gen):
            i, j = rng.choice(idx_c, 2, replace=True)
            t = rng.uniform(0.3, 0.7)
            interp = slerp(X_pca[i], X_pca[j], t)
            new_X.append(interp)
            new_y.append(c)

    new_X = np.array(new_X, dtype=np.float32)
    new_y = np.array(new_y)
    # Inverse PCA
    new_X_orig = pca.inverse_transform(new_X)
    X_combined = np.vstack([X, new_X_orig]).astype(np.float32)
    y_combined = np.concatenate([y, new_y])
    gc.collect()
    return X_combined, y_combined


# ──────────────────────────────────────────────────────────────────────
# 5. Randomized SVD + Frobenius / Monte-Carlo
# ──────────────────────────────────────────────────────────────────────

def randomized_svd_reduce(X, variance_target=0.95, max_components=200):
    """
    Reduce dimensionality via Randomized SVD preserving ≥ variance_target
    of the total Frobenius norm energy.

    Uses Monte-Carlo subsampling to estimate optimal rank when data is large.
    """
    N, D = X.shape
    # Monte-Carlo sub-sample for fast rank estimation
    sample_size = min(5000, N)
    idx = np.random.choice(N, sample_size, replace=False)
    scaler = StandardScaler()
    X_sample = scaler.fit_transform(X[idx])

    # Full SVD on sample to estimate rank
    frob_sq = np.sum(X_sample ** 2)
    max_k = min(max_components, D - 1, sample_size - 1)
    svd = TruncatedSVD(n_components=max_k, random_state=42)
    svd.fit(X_sample)
    cum_var = np.cumsum(svd.explained_variance_ratio_)
    n_keep = int(np.searchsorted(cum_var, variance_target) + 1)
    n_keep = max(2, min(n_keep, max_k))

    # Final reduction on full data
    X_scaled = scaler.transform(X)
    final_svd = TruncatedSVD(n_components=n_keep, random_state=42)
    X_reduced = final_svd.fit_transform(X_scaled)
    actual_var = final_svd.explained_variance_ratio_.sum()
    print(f"    RandomizedSVD: {D}→{n_keep} dims  "
          f"(explained var = {actual_var:.4f})")
    gc.collect()
    return X_reduced


# ──────────────────────────────────────────────────────────────────────
# 6. Coreset Selection (HDBSCAN + Entropy)
# ──────────────────────────────────────────────────────────────────────

def coreset_selection(X, y, top_pct=0.10):
    """
    Select the top top_pct coreset by combining HDBSCAN cluster
    probabilities with normalised sample entropy weights.
    """
    import hdbscan

    clusterer = hdbscan.HDBSCAN(min_cluster_size=max(5, len(X) // 200),
                                 prediction_data=True,
                                 core_dist_n_jobs=1)
    clusterer.fit(X)
    probs = clusterer.probabilities_  # [0, 1], higher = more core

    # Sample entropy weights
    X_shift = X - X.min(axis=1, keepdims=True) + 1e-10
    X_norm = X_shift / X_shift.sum(axis=1, keepdims=True)
    ent = sp_entropy(X_norm, axis=1)
    ent_norm = 1.0 - (ent - ent.min()) / (ent.max() - ent.min() + 1e-10)

    # Combined score: geometric mean
    score = np.sqrt(probs * ent_norm + 1e-10)
    n_select = max(10, int(len(X) * top_pct))
    top_idx = np.argsort(score)[-n_select:]
    gc.collect()
    return X[top_idx], y[top_idx]


# ──────────────────────────────────────────────────────────────────────
# 7.  Custom Nyström Kernel Approximations
# ──────────────────────────────────────────────────────────────────────

from sklearn.base import BaseEstimator, TransformerMixin

class NystromKernel(BaseEstimator, TransformerMixin):
    """
    Generic Nyström kernel approximation.

    Supports: 'rbf', 'poly', 'sigmoid', 'grpf', 't_student', 'inv_multiquadric'
    """

    def __init__(self, kernel='rbf', gamma=1.0, n_components=100,
                 degree=2, coef0=1.0, random_state=42):
        self.kernel = kernel
        self.gamma = gamma
        self.n_components = n_components
        self.degree = degree
        self.coef0 = coef0
        self.random_state = random_state

    def _kernel_func(self, X, Y):
        """Compute kernel matrix K(X, Y)."""
        if self.kernel == 'rbf':
            from sklearn.metrics.pairwise import rbf_kernel
            return rbf_kernel(X, Y, gamma=self.gamma)
        elif self.kernel == 'poly':
            from sklearn.metrics.pairwise import polynomial_kernel
            return polynomial_kernel(X, Y, degree=self.degree,
                                     gamma=self.gamma, coef0=self.coef0)
        elif self.kernel == 'sigmoid':
            from sklearn.metrics.pairwise import sigmoid_kernel
            return sigmoid_kernel(X, Y, gamma=self.gamma, coef0=self.coef0)
        elif self.kernel == 'grpf':
            # Gaussian Random Projection Feature kernel
            # k(x,y) = exp(-gamma * ||x-y||^2 / (||x|| * ||y|| + eps))
            from sklearn.metrics.pairwise import euclidean_distances
            D = euclidean_distances(X, Y) ** 2
            nx = np.linalg.norm(X, axis=1, keepdims=True)
            ny = np.linalg.norm(Y, axis=1, keepdims=True)
            denom = nx @ ny.T + 1e-10
            return np.exp(-self.gamma * D / denom)
        elif self.kernel == 't_student':
            # t-Student kernel: k(x,y) = 1 / (1 + ||x-y||^degree)
            from sklearn.metrics.pairwise import euclidean_distances
            D = euclidean_distances(X, Y)
            return 1.0 / (1.0 + D ** self.degree)
        elif self.kernel == 'inv_multiquadric':
            # Inverse Multiquadric: k(x,y) = 1 / sqrt(||x-y||^2 + c^2)
            from sklearn.metrics.pairwise import euclidean_distances
            D2 = euclidean_distances(X, Y) ** 2
            return 1.0 / np.sqrt(D2 + self.coef0 ** 2)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def fit(self, X, y=None):
        rng = np.random.RandomState(self.random_state)
        n = min(self.n_components, len(X))
        idx = rng.choice(len(X), n, replace=False)
        self.components_ = X[idx].copy()

        K_mm = self._kernel_func(self.components_, self.components_)
        # Regularise
        K_mm += 1e-6 * np.eye(len(K_mm))
        # Eigendecompose for Nyström embedding
        eigvals, eigvecs = np.linalg.eigh(K_mm)
        eigvals = np.maximum(eigvals, 1e-10)
        self.normalization_ = eigvecs / np.sqrt(eigvals)[np.newaxis, :]
        return self

    def transform(self, X):
        K_nm = self._kernel_func(X, self.components_)
        return K_nm @ self.normalization_
