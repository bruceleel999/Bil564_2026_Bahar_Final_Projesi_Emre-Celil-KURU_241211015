"""
art_variants.py — Complete Collection of Adaptive Resonance Theory Variants
============================================================================
Implementations:
  1.  ART1            — Binary inputs only
  2.  ART2            — Continuous inputs
  3.  ART2A           — Streamlined ART-2
  4.  ART3            — Neurotransmitter-regulated ART
  5.  FuzzyART        — Fuzzy logic ART with complement coding
  6.  ARTMAP          — Supervised predictive ART (two coupled ART units)
  7.  FuzzyARTMAP     — ARTMAP with Fuzzy ART units
  8.  SFAM            — Simplified Fuzzy ARTMAP
  9.  GaussianART     — Gaussian activation ART
  10. GaussianARTMAP  — Supervised Gaussian ART
  11. FusionART       — Multi-channel ART
  12. TopoART         — Fuzzy ART + topology learning
  13. HypersphereART  — L2-norm hypersphere categories
  14. HypersphereARTMAP — Supervised Hypersphere ART
  15. LAPART          — Laterally Primed ART

All classes follow a common interface:
    .fit(X, y=None)  → self
    .predict(X)      → cluster labels (1-D int array)
"""

import gc
import numpy as np
from scipy.spatial.distance import cdist


# ══════════════════════════════════════════════════════════════════════
# Helper: complement coding
# ══════════════════════════════════════════════════════════════════════

def complement_code(X):
    """Complement coding: x → [x, 1-x]. Doubles feature dimensionality."""
    X_clip = np.clip(X, 0, 1)
    return np.hstack([X_clip, 1 - X_clip]).astype(np.float32)


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  1.  ART1 — Binary ART                                          ║
# ╚═══════════════════════════════════════════════════════════════════╝

class ART1:
    """
    ART1 (Carpenter & Grossberg, 1987). Binary inputs only.
    Uses AND-based matching and vigilance test.
    """

    def __init__(self, dim, rho=0.7, max_categories=100, n_iter=10):
        self.dim = dim
        self.rho = rho
        self.max_cat = max_categories
        self.n_iter = n_iter
        self.W = []  # list of weight vectors (binary)

    def _match(self, x, w):
        return np.sum(np.minimum(x, w)) / (np.sum(x) + 1e-10)

    def fit(self, X, y=None):
        X_bin = (X > 0.5).astype(np.float32)
        for _ in range(self.n_iter):
            for x in X_bin:
                placed = False
                for j in range(len(self.W)):
                    m = self._match(x, self.W[j])
                    if m >= self.rho:
                        self.W[j] = np.minimum(x, self.W[j])
                        placed = True
                        break
                if not placed and len(self.W) < self.max_cat:
                    self.W.append(x.copy())
        gc.collect()
        return self

    def predict(self, X):
        X_bin = (X > 0.5).astype(np.float32)
        labels = np.zeros(len(X), dtype=np.int64)
        for i, x in enumerate(X_bin):
            best_j, best_m = 0, -1
            for j, w in enumerate(self.W):
                m = self._match(x, w)
                if m >= self.rho and m > best_m:
                    best_m, best_j = m, j
            labels[i] = best_j
        return labels


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  2.  ART2 — Continuous-input ART                                ║
# ╚═══════════════════════════════════════════════════════════════════╝

class ART2:
    """
    ART2 (Carpenter & Grossberg, 1987). Continuous real-valued inputs.
    Uses norm-based matching with F1 and F2 layers.
    """

    def __init__(self, dim, rho=0.8, alpha=0.1, max_categories=100,
                 n_iter=10):
        self.dim = dim
        self.rho = rho
        self.alpha = alpha
        self.max_cat = max_categories
        self.n_iter = n_iter
        self.W = []  # top-down weights
        self.B = []  # bottom-up weights

    def _norm(self, x):
        return x / (np.linalg.norm(x) + 1e-10)

    def fit(self, X, y=None):
        for _ in range(self.n_iter):
            for x in X:
                x_n = self._norm(x)
                placed = False
                for j in range(len(self.W)):
                    # Activation
                    act = np.dot(x_n, self.B[j])
                    if act < self.alpha:
                        continue
                    # Match / vigilance test
                    match = np.linalg.norm(
                        x_n + self.W[j]) / (np.linalg.norm(x_n) +
                                             np.linalg.norm(self.W[j]) + 1e-10)
                    if match >= self.rho:
                        lr = 0.3
                        self.W[j] = self._norm(lr * x_n + (1 - lr) * self.W[j])
                        self.B[j] = self._norm(lr * x_n + (1 - lr) * self.B[j])
                        placed = True
                        break
                if not placed and len(self.W) < self.max_cat:
                    self.W.append(x_n.copy())
                    self.B.append(x_n.copy())
        gc.collect()
        return self

    def predict(self, X):
        labels = np.zeros(len(X), dtype=np.int64)
        for i, x in enumerate(X):
            x_n = self._norm(x)
            best_j, best_a = 0, -np.inf
            for j in range(len(self.W)):
                a = np.dot(x_n, self.B[j])
                if a > best_a:
                    best_a, best_j = a, j
            labels[i] = best_j
        return labels


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  3.  ART2A — Streamlined ART-2                                  ║
# ╚═══════════════════════════════════════════════════════════════════╝

class ART2A:
    """
    ART2-A (Carpenter, Grossberg & Rosen, 1991). Drastically accelerated
    ART-2 variant with qualitatively similar results.
    Single-pass winner-take-all with direct norm matching.
    """

    def __init__(self, dim, rho=0.8, alpha=0.1, lr=0.5,
                 max_categories=100, n_iter=5):
        self.dim = dim
        self.rho = rho
        self.alpha = alpha
        self.lr = lr
        self.max_cat = max_categories
        self.n_iter = n_iter
        self.W = []

    def fit(self, X, y=None):
        for _ in range(self.n_iter):
            for x in X:
                x_n = x / (np.linalg.norm(x) + 1e-10)
                # Find winner
                best_j, best_act = -1, -np.inf
                for j, w in enumerate(self.W):
                    act = np.dot(x_n, w)
                    if act > best_act:
                        best_act, best_j = act, j
                # Vigilance test
                if best_j >= 0 and best_act >= self.rho:
                    self.W[best_j] += self.lr * (x_n - self.W[best_j])
                    self.W[best_j] /= np.linalg.norm(self.W[best_j]) + 1e-10
                elif len(self.W) < self.max_cat:
                    self.W.append(x_n.copy())
        gc.collect()
        return self

    def predict(self, X):
        if len(self.W) == 0:
            return np.zeros(len(X), dtype=np.int64)
        W_arr = np.array(self.W)
        X_n = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-10)
        sims = X_n @ W_arr.T
        return np.argmax(sims, axis=1)


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  4.  ART3 — Neurotransmitter-regulated ART                     ║
# ╚═══════════════════════════════════════════════════════════════════╝

class ART3:
    """
    ART3 (Carpenter & Grossberg, 1990). Incorporates simulated
    Na+ and Ca2+ ion concentrations for partial inhibition of
    categories that trigger mismatch resets — more physiologically
    realistic.
    """

    def __init__(self, dim, rho=0.8, na_decay=0.9, ca_recovery=0.1,
                 max_categories=100, n_iter=10):
        self.dim = dim
        self.rho = rho
        self.na_decay = na_decay
        self.ca_recovery = ca_recovery
        self.max_cat = max_categories
        self.n_iter = n_iter
        self.W = []
        self.na_levels = []  # Na+ concentration per category
        self.ca_levels = []  # Ca2+ concentration per category

    def fit(self, X, y=None):
        for _ in range(self.n_iter):
            for x in X:
                x_n = x / (np.linalg.norm(x) + 1e-10)
                placed = False
                # Try categories with sufficient Na+ (not inhibited)
                order = np.argsort([-na for na in self.na_levels]) \
                    if self.na_levels else []
                for j in order:
                    if self.na_levels[j] < 0.3:
                        continue  # inhibited
                    act = np.dot(x_n, self.W[j]) * self.na_levels[j]
                    if act >= self.rho:
                        lr = 0.3 * self.ca_levels[j]
                        self.W[j] += lr * (x_n - self.W[j])
                        self.W[j] /= np.linalg.norm(self.W[j]) + 1e-10
                        self.ca_levels[j] = min(1.0,
                            self.ca_levels[j] + self.ca_recovery)
                        placed = True
                        break
                    else:
                        # Mismatch → Na+ depletion
                        self.na_levels[j] *= self.na_decay

                if not placed and len(self.W) < self.max_cat:
                    self.W.append(x_n.copy())
                    self.na_levels.append(1.0)
                    self.ca_levels.append(0.5)

            # Recover Na+ between passes
            self.na_levels = [min(1.0, na + 0.1) for na in self.na_levels]

        gc.collect()
        return self

    def predict(self, X):
        if len(self.W) == 0:
            return np.zeros(len(X), dtype=np.int64)
        W_arr = np.array(self.W)
        X_n = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-10)
        return np.argmax(X_n @ W_arr.T, axis=1)


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  5.  Fuzzy ART — with complement coding and L1 matching         ║
# ╚═══════════════════════════════════════════════════════════════════╝

class FuzzyART:
    """
    Fuzzy ART (Carpenter, Grossberg & Rosen, 1991).
    Uses fuzzy AND (element-wise min) and L1 norm.
    Optional complement coding for encoding feature absence.
    """

    def __init__(self, dim, rho=0.7, alpha=0.01, lr=1.0,
                 complement=True, max_categories=100, n_iter=10):
        self.orig_dim = dim
        self.rho = rho
        self.alpha = alpha  # choice parameter
        self.lr = lr
        self.complement = complement
        self.max_cat = max_categories
        self.n_iter = n_iter
        self.dim = 2 * dim if complement else dim
        self.W = []

    def fit(self, X, y=None):
        if self.complement:
            X = complement_code(X)
        for _ in range(self.n_iter):
            for x in X:
                placed = False
                # Choice function: T_j = |x ∧ w_j| / (alpha + |w_j|)
                activities = []
                for j, w in enumerate(self.W):
                    fuzzy_and = np.minimum(x, w)
                    T = np.sum(fuzzy_and) / (self.alpha + np.sum(w))
                    activities.append((T, j))
                activities.sort(reverse=True)

                for T, j in activities:
                    fuzzy_and = np.minimum(x, self.W[j])
                    match = np.sum(fuzzy_and) / (np.sum(x) + 1e-10)
                    if match >= self.rho:
                        self.W[j] = self.lr * fuzzy_and + \
                                    (1 - self.lr) * self.W[j]
                        placed = True
                        break

                if not placed and len(self.W) < self.max_cat:
                    self.W.append(x.copy())
        gc.collect()
        return self

    def predict(self, X):
        if self.complement:
            X = complement_code(X)
        if len(self.W) == 0:
            return np.zeros(len(X), dtype=np.int64)
        W_arr = np.array(self.W)
        labels = np.zeros(len(X), dtype=np.int64)
        for i, x in enumerate(X):
            fuzzy_and = np.minimum(x, W_arr)
            T = fuzzy_and.sum(axis=1) / (self.alpha + W_arr.sum(axis=1))
            labels[i] = np.argmax(T)
        return labels


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  6-7.  ARTMAP / Fuzzy ARTMAP — Supervised ART                   ║
# ╚═══════════════════════════════════════════════════════════════════╝

class FuzzyARTMAP:
    """
    Fuzzy ARTMAP (Carpenter et al., 1992). Two coupled Fuzzy ART
    modules (ARTa for input, ARTb for output), with map field for
    supervised association and match-tracking vigilance adjustment.
    """

    def __init__(self, dim, n_classes, rho_a=0.7, rho_b=0.9,
                 alpha=0.01, lr=1.0, max_categories=200):
        self.art_a = FuzzyART(dim, rho=rho_a, alpha=alpha, lr=lr,
                              complement=True, max_categories=max_categories,
                              n_iter=1)
        self.n_classes = n_classes
        self.rho_a_base = rho_a
        self.map_field = {}  # ART_a category → class label

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("FuzzyARTMAP requires labels")
        X_cc = complement_code(X)
        dim_cc = X_cc.shape[1]

        for x, label in zip(X_cc, y):
            rho = self.rho_a_base
            placed = False
            tested = set()

            while not placed:
                activities = []
                for j, w in enumerate(self.art_a.W):
                    if j in tested:
                        continue
                    fuzzy_and = np.minimum(x, w)
                    T = np.sum(fuzzy_and) / (self.art_a.alpha + np.sum(w))
                    activities.append((T, j))
                activities.sort(reverse=True)

                matched = False
                for T, j in activities:
                    fuzzy_and = np.minimum(x, self.art_a.W[j])
                    match = np.sum(fuzzy_and) / (np.sum(x) + 1e-10)
                    if match >= rho:
                        if j in self.map_field and \
                                self.map_field[j] == int(label):
                            # Correct prediction → learn
                            self.art_a.W[j] = self.art_a.lr * fuzzy_and + \
                                (1 - self.art_a.lr) * self.art_a.W[j]
                            placed = True
                            matched = True
                            break
                        elif j not in self.map_field:
                            self.art_a.W[j] = self.art_a.lr * fuzzy_and + \
                                (1 - self.art_a.lr) * self.art_a.W[j]
                            self.map_field[j] = int(label)
                            placed = True
                            matched = True
                            break
                        else:
                            # Match tracking: raise vigilance
                            tested.add(j)
                            rho = match + 0.001
                            break

                if not matched and not placed:
                    if len(self.art_a.W) < self.art_a.max_cat:
                        new_j = len(self.art_a.W)
                        self.art_a.W.append(x.copy())
                        self.map_field[new_j] = int(label)
                        placed = True
                    else:
                        placed = True  # capacity reached

        gc.collect()
        return self

    def predict(self, X):
        X_cc = complement_code(X)
        labels = np.zeros(len(X), dtype=np.int64)
        if len(self.art_a.W) == 0:
            return labels
        W_arr = np.array(self.art_a.W)
        for i, x in enumerate(X_cc):
            fuzzy_and = np.minimum(x, W_arr)
            T = fuzzy_and.sum(axis=1) / (self.art_a.alpha + W_arr.sum(axis=1))
            j = np.argmax(T)
            labels[i] = self.map_field.get(j, 0)
        return labels


# ARTMAP is essentially FuzzyARTMAP with ART-1 units; we alias it
ARTMAP = FuzzyARTMAP


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  8.  SFAM — Simplified Fuzzy ARTMAP                             ║
# ╚═══════════════════════════════════════════════════════════════════╝

class SFAM:
    """
    Simplified Fuzzy ARTMAP (Kasuba, 1993). Single ART module with
    direct class mapping — no separate ARTb unit.
    """

    def __init__(self, dim, rho=0.75, alpha=0.01, lr=1.0,
                 max_categories=200):
        self.dim = dim * 2  # complement coded
        self.orig_dim = dim
        self.rho = rho
        self.alpha = alpha
        self.lr = lr
        self.max_cat = max_categories
        self.W = []
        self.labels = []

    def fit(self, X, y=None):
        X_cc = complement_code(X)
        for x, label in zip(X_cc, y):
            rho = self.rho
            placed = False
            tested = set()
            while not placed:
                best_j, best_T = -1, -np.inf
                for j, w in enumerate(self.W):
                    if j in tested:
                        continue
                    fuzzy_and = np.minimum(x, w)
                    T = np.sum(fuzzy_and) / (self.alpha + np.sum(w))
                    if T > best_T:
                        best_T, best_j = T, j
                if best_j >= 0:
                    fuzzy_and = np.minimum(x, self.W[best_j])
                    match = np.sum(fuzzy_and) / (np.sum(x) + 1e-10)
                    if match >= rho:
                        if self.labels[best_j] == int(label):
                            self.W[best_j] = self.lr * fuzzy_and + \
                                (1 - self.lr) * self.W[best_j]
                            placed = True
                        else:
                            tested.add(best_j)
                            rho = match + 0.001
                    else:
                        tested.add(best_j)
                else:
                    break
            if not placed and len(self.W) < self.max_cat:
                self.W.append(x.copy())
                self.labels.append(int(label))
        gc.collect()
        return self

    def predict(self, X):
        X_cc = complement_code(X)
        if len(self.W) == 0:
            return np.zeros(len(X), dtype=np.int64)
        W_arr = np.array(self.W)
        preds = np.zeros(len(X), dtype=np.int64)
        for i, x in enumerate(X_cc):
            fuzzy_and = np.minimum(x, W_arr)
            T = fuzzy_and.sum(axis=1) / (self.alpha + W_arr.sum(axis=1))
            preds[i] = self.labels[np.argmax(T)]
        return preds


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  9-10.  GaussianART / GaussianARTMAP                            ║
# ╚═══════════════════════════════════════════════════════════════════╝

class GaussianART:
    """
    Gaussian ART (Williamson, 1996). Uses Gaussian activation
    functions and probability-theoretic computations.
    Each category stores a mean μ and variance σ².
    """

    def __init__(self, dim, rho=0.5, sigma_init=1.0,
                 max_categories=100, n_iter=5):
        self.dim = dim
        self.rho = rho
        self.sigma_init = sigma_init
        self.max_cat = max_categories
        self.n_iter = n_iter
        self.mus = []
        self.sigmas = []
        self.counts = []

    def _activation(self, x, mu, sigma):
        d = np.sum((x - mu) ** 2 / (sigma + 1e-10))
        return np.exp(-0.5 * d)

    def fit(self, X, y=None):
        for _ in range(self.n_iter):
            for x in X:
                # Compute activations
                acts = [(self._activation(x, self.mus[j], self.sigmas[j]), j)
                        for j in range(len(self.mus))]
                acts.sort(reverse=True)

                placed = False
                for act, j in acts:
                    if act >= self.rho:
                        n = self.counts[j]
                        self.mus[j] = (n * self.mus[j] + x) / (n + 1)
                        self.sigmas[j] = (n * self.sigmas[j] +
                            (x - self.mus[j]) ** 2) / (n + 1) + 1e-4
                        self.counts[j] += 1
                        placed = True
                        break

                if not placed and len(self.mus) < self.max_cat:
                    self.mus.append(x.copy())
                    self.sigmas.append(
                        np.full(self.dim, self.sigma_init, dtype=np.float32))
                    self.counts.append(1)
        gc.collect()
        return self

    def predict(self, X):
        if len(self.mus) == 0:
            return np.zeros(len(X), dtype=np.int64)
        labels = np.zeros(len(X), dtype=np.int64)
        for i, x in enumerate(X):
            best_j, best_a = 0, -np.inf
            for j in range(len(self.mus)):
                a = self._activation(x, self.mus[j], self.sigmas[j])
                if a > best_a:
                    best_a, best_j = a, j
            labels[i] = best_j
        return labels


class GaussianARTMAP:
    """
    Gaussian ARTMAP — supervised version pairing GaussianART with
    class labels via map field and match tracking.
    """

    def __init__(self, dim, rho=0.5, sigma_init=1.0, max_categories=200):
        self.gart = GaussianART(dim, rho=rho, sigma_init=sigma_init,
                                max_categories=max_categories, n_iter=1)
        self.rho_base = rho
        self.map_field = {}

    def fit(self, X, y=None):
        for x, label in zip(X, y):
            rho = self.rho_base
            placed = False
            tested = set()
            while not placed:
                acts = [(self.gart._activation(x, self.gart.mus[j],
                         self.gart.sigmas[j]), j)
                        for j in range(len(self.gart.mus)) if j not in tested]
                acts.sort(reverse=True)
                matched = False
                for act, j in acts:
                    if act >= rho:
                        if j in self.map_field and \
                                self.map_field[j] == int(label):
                            n = self.gart.counts[j]
                            self.gart.mus[j] = (n * self.gart.mus[j] + x) / \
                                               (n + 1)
                            self.gart.counts[j] += 1
                            placed, matched = True, True
                            break
                        elif j not in self.map_field:
                            self.map_field[j] = int(label)
                            placed, matched = True, True
                            break
                        else:
                            tested.add(j)
                            rho = act + 0.001
                            break
                if not matched and not placed:
                    if len(self.gart.mus) < self.gart.max_cat:
                        new_j = len(self.gart.mus)
                        self.gart.mus.append(x.copy())
                        self.gart.sigmas.append(
                            np.full(self.gart.dim, self.gart.sigma_init))
                        self.gart.counts.append(1)
                        self.map_field[new_j] = int(label)
                        placed = True
                    else:
                        placed = True
        gc.collect()
        return self

    def predict(self, X):
        labels_raw = self.gart.predict(X)
        return np.array([self.map_field.get(int(l), 0) for l in labels_raw],
                        dtype=np.int64)


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  11.  FusionART — Multi-channel ART                             ║
# ╚═══════════════════════════════════════════════════════════════════╝

class FusionART:
    """
    Fusion ART (Tan et al., 2007). Extends ART to multiple pattern
    channels (e.g., visual + textual). Each channel has its own weight
    template; matching and vigilance are computed per-channel and fused.
    For single-channel data, splits features into K artificial channels.
    """

    def __init__(self, dim, n_channels=3, rho=0.7, alpha=0.01,
                 max_categories=100, n_iter=5):
        self.dim = dim
        self.n_ch = n_channels
        self.rho = rho
        self.alpha = alpha
        self.max_cat = max_categories
        self.n_iter = n_iter
        self.W_channels = []  # list of [list of channel weights per category]
        self.ch_dims = None

    def _split_channels(self, x):
        if self.ch_dims is None:
            base = self.dim // self.n_ch
            self.ch_dims = [base] * self.n_ch
            self.ch_dims[-1] += self.dim - base * self.n_ch
        parts = []
        start = 0
        for d in self.ch_dims:
            parts.append(x[start:start + d])
            start += d
        return parts

    def fit(self, X, y=None):
        for _ in range(self.n_iter):
            for x in X:
                channels = self._split_channels(x)
                placed = False

                for j in range(len(self.W_channels)):
                    # Per-channel match
                    total_match = 0
                    for ch_idx in range(self.n_ch):
                        fa = np.minimum(channels[ch_idx],
                                        self.W_channels[j][ch_idx])
                        m = np.sum(fa) / (np.sum(channels[ch_idx]) + 1e-10)
                        total_match += m
                    avg_match = total_match / self.n_ch
                    if avg_match >= self.rho:
                        for ch_idx in range(self.n_ch):
                            fa = np.minimum(channels[ch_idx],
                                            self.W_channels[j][ch_idx])
                            self.W_channels[j][ch_idx] = fa
                        placed = True
                        break

                if not placed and len(self.W_channels) < self.max_cat:
                    self.W_channels.append(
                        [ch.copy() for ch in channels])
        gc.collect()
        return self

    def predict(self, X):
        labels = np.zeros(len(X), dtype=np.int64)
        for i, x in enumerate(X):
            channels = self._split_channels(x)
            best_j, best_m = 0, -np.inf
            for j in range(len(self.W_channels)):
                total = sum(
                    np.sum(np.minimum(channels[ch], self.W_channels[j][ch]))
                    for ch in range(self.n_ch))
                if total > best_m:
                    best_m, best_j = total, j
            labels[i] = best_j
        return labels


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  12.  TopoART — Fuzzy ART + topology learning + noise reduction ║
# ╚═══════════════════════════════════════════════════════════════════╝

class TopoART:
    """
    TopoART (Tscherepanow, 2010). Combines Fuzzy ART with a
    growing-neural-gas-style topology learning: edges are created
    between co-activated nodes. A secondary Fuzzy ART with higher
    vigilance filters noise by requiring a sample to match at two
    vigilance levels.
    """

    def __init__(self, dim, rho1=0.7, rho2=0.85, alpha=0.01,
                 lr=1.0, max_categories=150, n_iter=5,
                 noise_threshold=3):
        self.dim_cc = dim * 2
        self.orig_dim = dim
        self.rho1 = rho1
        self.rho2 = rho2
        self.alpha = alpha
        self.lr = lr
        self.max_cat = max_categories
        self.n_iter = n_iter
        self.noise_thr = noise_threshold
        self.W = []
        self.edges = set()      # set of (i, j) edges
        self.age = {}           # edge (i,j) -> age
        self.counter = []       # hit counter per node

    def fit(self, X, y=None):
        X_cc = complement_code(X)
        for _ in range(self.n_iter):
            for x in X_cc:
                # Primary match (ρ1)
                best1, best2 = -1, -1
                best_T1, best_T2 = -np.inf, -np.inf
                for j, w in enumerate(self.W):
                    fa = np.minimum(x, w)
                    T = np.sum(fa) / (self.alpha + np.sum(w))
                    if T > best_T1:
                        best_T2, best2 = best_T1, best1
                        best_T1, best1 = T, j
                    elif T > best_T2:
                        best_T2, best2 = T, j

                placed = False
                if best1 >= 0:
                    fa = np.minimum(x, self.W[best1])
                    match = np.sum(fa) / (np.sum(x) + 1e-10)
                    if match >= self.rho1:
                        # Secondary check (ρ2) for noise filtering
                        if match >= self.rho2:
                            self.W[best1] = self.lr * fa + \
                                (1 - self.lr) * self.W[best1]
                            self.counter[best1] += 1
                            placed = True

                            # Create / age edges
                            if best2 >= 0:
                                e = (min(best1, best2), max(best1, best2))
                                self.edges.add(e)
                                self.age[e] = 0

                if not placed and len(self.W) < self.max_cat:
                    self.W.append(x.copy())
                    self.counter.append(1)

            # Remove noise nodes (low counter)
            keep = [j for j in range(len(self.W))
                    if self.counter[j] >= self.noise_thr]
            self.W = [self.W[j] for j in keep]
            self.counter = [self.counter[j] for j in keep]

        gc.collect()
        return self

    def predict(self, X):
        X_cc = complement_code(X)
        if len(self.W) == 0:
            return np.zeros(len(X), dtype=np.int64)
        W_arr = np.array(self.W)
        labels = np.zeros(len(X), dtype=np.int64)
        for i, x in enumerate(X_cc):
            fa = np.minimum(x, W_arr)
            T = fa.sum(axis=1) / (self.alpha + W_arr.sum(axis=1))
            labels[i] = np.argmax(T)
        return labels


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  13-14.  HypersphereART / HypersphereARTMAP                    ║
# ╚═══════════════════════════════════════════════════════════════════╝

class HypersphereART:
    """
    Hypersphere ART (Anagnostopoulos & Georgiopoulos, 2000).
    Categories are hyperspheres (centre + radius).
    Uses L2 norm — does NOT require [0,1] normalisation.
    """

    def __init__(self, dim, rho=0.7, r_max=1.0,
                 max_categories=100, n_iter=5):
        self.dim = dim
        self.rho = rho
        self.r_max = r_max
        self.max_cat = max_categories
        self.n_iter = n_iter
        self.centres = []
        self.radii = []

    def fit(self, X, y=None):
        for _ in range(self.n_iter):
            for x in X:
                placed = False
                for j in range(len(self.centres)):
                    d = np.linalg.norm(x - self.centres[j])
                    new_r = max(self.radii[j], d)
                    match = 1 - new_r / self.r_max
                    if match >= self.rho and new_r <= self.r_max:
                        # Expand sphere to include x
                        n = max(1, int(self.radii[j] * 10))
                        self.centres[j] = (n * self.centres[j] + x) / (n + 1)
                        self.radii[j] = new_r
                        placed = True
                        break
                if not placed and len(self.centres) < self.max_cat:
                    self.centres.append(x.copy())
                    self.radii.append(0.0)
        gc.collect()
        return self

    def predict(self, X):
        if len(self.centres) == 0:
            return np.zeros(len(X), dtype=np.int64)
        C = np.array(self.centres)
        D = cdist(X, C, 'euclidean')
        return np.argmin(D, axis=1)


class HypersphereARTMAP:
    """Supervised Hypersphere ART with map field."""

    def __init__(self, dim, rho=0.7, r_max=1.0, max_categories=200):
        self.hart = HypersphereART(dim, rho=rho, r_max=r_max,
                                   max_categories=max_categories, n_iter=1)
        self.map_field = {}

    def fit(self, X, y=None):
        for x, label in zip(X, y):
            placed = False
            for j in range(len(self.hart.centres)):
                d = np.linalg.norm(x - self.hart.centres[j])
                new_r = max(self.hart.radii[j], d)
                match = 1 - new_r / self.hart.r_max
                if match >= self.hart.rho and new_r <= self.hart.r_max:
                    if j not in self.map_field or \
                            self.map_field[j] == int(label):
                        self.hart.centres[j] = \
                            (self.hart.centres[j] + x) / 2
                        self.hart.radii[j] = new_r
                        self.map_field[j] = int(label)
                        placed = True
                        break
            if not placed and len(self.hart.centres) < self.hart.max_cat:
                new_j = len(self.hart.centres)
                self.hart.centres.append(x.copy())
                self.hart.radii.append(0.0)
                self.map_field[new_j] = int(label)
        gc.collect()
        return self

    def predict(self, X):
        labels_raw = self.hart.predict(X)
        return np.array([self.map_field.get(int(l), 0) for l in labels_raw],
                        dtype=np.int64)


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  15.  LAPART — Laterally Primed ART                             ║
# ╚═══════════════════════════════════════════════════════════════════╝

class LAPART:
    """
    LAPART (Healy & Caudell, 1997). Two coupled Fuzzy ART
    modules laterally primed for rapid convergence and
    logical inference. Module A receives input, Module B
    receives output; lateral priming ensures that A's winner
    is consistent with B's expectation.
    """

    def __init__(self, dim, n_classes, rho=0.7, alpha=0.01,
                 max_categories=200):
        self.art_a = FuzzyART(dim, rho=rho, alpha=alpha, complement=True,
                              max_categories=max_categories, n_iter=1)
        self.n_classes = n_classes
        self.associations = {}  # A_cat → class label

    def fit(self, X, y=None):
        X_cc = complement_code(X)
        for x, label in zip(X_cc, y):
            # Find best A-category
            best_j, best_T = -1, -np.inf
            for j, w in enumerate(self.art_a.W):
                fa = np.minimum(x, w)
                T = np.sum(fa) / (self.art_a.alpha + np.sum(w))
                match = np.sum(fa) / (np.sum(x) + 1e-10)
                if T > best_T and match >= self.art_a.rho:
                    # Lateral priming: check if association is consistent
                    if j not in self.associations or \
                            self.associations[j] == int(label):
                        best_T, best_j = T, j

            if best_j >= 0:
                fa = np.minimum(x, self.art_a.W[best_j])
                self.art_a.W[best_j] = self.art_a.lr * fa + \
                    (1 - self.art_a.lr) * self.art_a.W[best_j]
                self.associations[best_j] = int(label)
            elif len(self.art_a.W) < self.art_a.max_cat:
                new_j = len(self.art_a.W)
                self.art_a.W.append(x.copy())
                self.associations[new_j] = int(label)
        gc.collect()
        return self

    def predict(self, X):
        X_cc = complement_code(X)
        if len(self.art_a.W) == 0:
            return np.zeros(len(X), dtype=np.int64)
        W_arr = np.array(self.art_a.W)
        preds = np.zeros(len(X), dtype=np.int64)
        for i, x in enumerate(X_cc):
            fa = np.minimum(x, W_arr)
            T = fa.sum(axis=1) / (self.art_a.alpha + W_arr.sum(axis=1))
            j = np.argmax(T)
            preds[i] = self.associations.get(j, 0)
        return preds


# ──────────────────────────────────────────────────────────────────────
# Registries
# ──────────────────────────────────────────────────────────────────────

# Unsupervised ART variants
ART_UNSUPERVISED_REGISTRY = {
    "ART1":           ART1,
    "ART2":           ART2,
    "ART2A":          ART2A,
    "ART3":           ART3,
    "FuzzyART":       FuzzyART,
    "GaussianART":    GaussianART,
    "FusionART":      FusionART,
    "TopoART":        TopoART,
    "HypersphereART": HypersphereART,
}

# Supervised ART variants
ART_SUPERVISED_REGISTRY = {
    "ARTMAP":             ARTMAP,
    "FuzzyARTMAP":        FuzzyARTMAP,
    "SFAM":               SFAM,
    "GaussianARTMAP":     GaussianARTMAP,
    "HypersphereARTMAP":  HypersphereARTMAP,
    "LAPART":             LAPART,
}
