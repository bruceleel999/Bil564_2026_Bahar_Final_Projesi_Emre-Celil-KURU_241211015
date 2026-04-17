"""
som_variants.py — Complete Collection of Self-Organizing Map Variants
=====================================================================
Implementations:
  1. SOM         — Classic Kohonen Self-Organizing Map
  2. GTM         — Generative Topographic Map (EM-based)
  3. GSOM        — Growing Self-Organizing Map
  4. TASOM       — Time Adaptive Self-Organizing Map
  5. ConformalSOM— Conformal Map interpolation SOM
  6. ElasticMap  — Elastic Map with bending/stretching energy
  7. OSMap       — Oriented and Scalable Map

All classes follow a common interface:
    .fit(X)  → self
    .transform(X) → cluster labels (1-D int array)
    .get_prototypes() → (n_nodes, D) weight matrix
"""

import gc
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  1.  SOM — Classic Self-Organizing Map                           ║
# ╚═══════════════════════════════════════════════════════════════════╝

class SOM:
    """
    Standard Kohonen SOM on an (H × W) rectangular grid.
    Competitive learning with Gaussian neighbourhood shrinkage.
    """

    def __init__(self, grid_h=5, grid_w=5, dim=784,
                 lr0=0.5, sigma0=None, n_iter=500):
        self.H, self.W, self.dim = grid_h, grid_w, dim
        self.n_iter = n_iter
        self.lr0 = lr0
        self.sigma0 = sigma0 or max(grid_h, grid_w) / 2.0
        self.weights = None

    def _init_weights(self, X):
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X), min(self.H * self.W, len(X)), replace=False)
        self.weights = X[idx[:self.H * self.W]].reshape(
            self.H, self.W, self.dim).copy()
        if len(idx) < self.H * self.W:
            self.weights = rng.randn(self.H, self.W, self.dim).astype(
                np.float32) * 0.01

    def _bmu(self, x):
        diff = self.weights - x[np.newaxis, np.newaxis, :]
        dists = np.sum(diff ** 2, axis=2)
        return np.unravel_index(np.argmin(dists), dists.shape)

    def _neighbourhood(self, bmu, sigma):
        rows = np.arange(self.H)[:, None]
        cols = np.arange(self.W)[None, :]
        d2 = (rows - bmu[0]) ** 2 + (cols - bmu[1]) ** 2
        return np.exp(-d2 / (2 * sigma ** 2))

    def fit(self, X):
        self._init_weights(X)
        n = len(X)
        for t in range(self.n_iter):
            frac = t / self.n_iter
            lr = self.lr0 * (1 - frac)
            sigma = self.sigma0 * (1 - frac) + 0.1
            idx = np.random.randint(n)
            x = X[idx]
            bmu = self._bmu(x)
            h = self._neighbourhood(bmu, sigma)
            diff = x[np.newaxis, np.newaxis, :] - self.weights
            self.weights += lr * h[:, :, np.newaxis] * diff
        return self

    def transform(self, X):
        labels = np.zeros(len(X), dtype=np.int64)
        for i, x in enumerate(X):
            bmu = self._bmu(x)
            labels[i] = bmu[0] * self.W + bmu[1]
        return labels

    def get_prototypes(self):
        return self.weights.reshape(-1, self.dim)


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  2.  GTM — Generative Topographic Map                           ║
# ╚═══════════════════════════════════════════════════════════════════╝

class GTM:
    """
    Generative Topographic Map (Bishop, Svensén & Williams, 1998).

    A probabilistic, smooth mapping from a low-dimensional latent grid
    to data space via a parametric mapping y(z; W) = W·φ(z).
    Trained with EM to maximise the data likelihood under a Gaussian
    mixture in data space.

    Latent grid: (K) evenly-spaced points on a 2-D square.
    Basis: (M) Gaussian RBFs in latent space.
    Mapping: y_k = W·φ(z_k) — smooth and continuous → topology-preserving.
    """

    def __init__(self, grid_size=8, n_basis=16, dim=784,
                 n_iter=50, random_state=42):
        self.K = grid_size ** 2        # number of latent points
        self.M = n_basis               # number of RBF bases
        self.D = dim                   # data dimensionality
        self.n_iter = n_iter
        self.rs = random_state

    def _build_latent_grid(self, grid_size):
        """Create equi-spaced 2-D latent points."""
        side = np.linspace(-1, 1, grid_size)
        zx, zy = np.meshgrid(side, side)
        return np.column_stack([zx.ravel(), zy.ravel()]).astype(np.float32)

    def _rbf_basis(self, Z, centres, sigma):
        """Evaluate Gaussian RBFs: Φ(z) of shape (K, M+1) with bias."""
        D2 = cdist(Z, centres, 'sqeuclidean')
        Phi = np.exp(-D2 / (2 * sigma ** 2))
        return np.hstack([Phi, np.ones((len(Z), 1))]).astype(np.float32)

    def fit(self, X):
        rng = np.random.RandomState(self.rs)
        grid_size = int(np.sqrt(self.K))
        Z = self._build_latent_grid(grid_size)  # (K, 2)

        # RBF centres in latent space
        n_basis_side = max(2, int(np.sqrt(self.M)))
        basis_side = np.linspace(-1, 1, n_basis_side)
        bx, by = np.meshgrid(basis_side, basis_side)
        centres = np.column_stack([bx.ravel(), by.ravel()])[:self.M]
        sigma_rbf = 2.0 / n_basis_side

        Phi = self._rbf_basis(Z, centres, sigma_rbf)  # (K, M+1)

        # Initialise W via PCA projection
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(2, X.shape[1]), random_state=self.rs)
        pca.fit(X[:min(2000, len(X))])
        # W: (M+1, D) — initialise small
        W = rng.randn(Phi.shape[1], self.D).astype(np.float32) * 0.01
        Y = Phi @ W  # (K, D) — mapped centres in data space

        beta = 1.0 / np.mean(cdist(Y, X[:min(1000, len(X))],
                                    'sqeuclidean'))

        # EM iterations
        N = len(X)
        for it in range(self.n_iter):
            # E-step: responsibilities R(n,k) = p(k|x_n)
            D2 = cdist(Y, X, 'sqeuclidean')  # (K, N)
            logR = -0.5 * beta * D2
            logR -= logR.max(axis=0, keepdims=True)
            R = np.exp(logR)
            R /= R.sum(axis=0, keepdims=True) + 1e-30  # (K, N)

            # M-step: update W and beta
            G = np.diag(R.sum(axis=1))  # (K, K) diagonal
            # W = (Phi^T G Phi)^{-1} Phi^T R X
            PhiTG = Phi.T @ G                    # (M+1, K)
            A = PhiTG @ Phi + 1e-4 * np.eye(Phi.shape[1])
            B = Phi.T @ R @ X                    # (M+1, D)
            W = np.linalg.solve(A, B)
            Y = Phi @ W                          # (K, D) updated

            # Update beta
            D2_new = cdist(Y, X, 'sqeuclidean')  # (K, N)
            beta = (N * self.D) / (R * D2_new).sum()

        self.W_ = W
        self.Y_ = Y
        self.Phi_ = Phi
        self.Z_ = Z
        self.beta_ = beta
        self.centres_ = centres
        self.sigma_rbf_ = sigma_rbf
        gc.collect()
        return self

    def transform(self, X):
        D2 = cdist(self.Y_, X, 'sqeuclidean')  # (K, N)
        return np.argmin(D2, axis=0)

    def get_prototypes(self):
        return self.Y_


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  3.  GSOM — Growing Self-Organizing Map                         ║
# ╚═══════════════════════════════════════════════════════════════════╝

class GSOM:
    """
    Growing SOM (Alahakoon, Halgamuge & Srinivasan, 2000).

    Starts with a minimal 2×2 grid and grows boundary nodes when the
    accumulated error of a boundary BMU exceeds a growth threshold
    GT = -D · ln(spread_factor).

    spread_factor ∈ (0,1) controls growth:
      • close to 0 → aggressive growth (fine map)
      • close to 1 → minimal growth (coarse map)
    """

    def __init__(self, dim=784, spread_factor=0.5, lr0=0.3,
                 n_iter=500, max_nodes=200):
        self.dim = dim
        self.sf = spread_factor
        self.lr0 = lr0
        self.n_iter = n_iter
        self.max_nodes = max_nodes

    def fit(self, X):
        GT = -self.dim * np.log(self.sf + 1e-10)
        rng = np.random.RandomState(42)

        # Initialise 2×2 grid
        self.nodes = {}  # (row, col) -> weight vector
        for r in range(2):
            for c in range(2):
                idx = rng.randint(len(X))
                self.nodes[(r, c)] = X[idx].copy()

        self.errors = {k: 0.0 for k in self.nodes}

        for t in range(self.n_iter):
            frac = t / self.n_iter
            lr = self.lr0 * (1 - frac)
            idx = rng.randint(len(X))
            x = X[idx]

            # Find BMU
            bmu, min_d = None, np.inf
            for pos, w in self.nodes.items():
                d = np.sum((x - w) ** 2)
                if d < min_d:
                    min_d, bmu = d, pos

            # Update BMU and neighbours
            self.nodes[bmu] += lr * (x - self.nodes[bmu])
            self.errors[bmu] += np.sqrt(min_d)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nb = (bmu[0] + dr, bmu[1] + dc)
                if nb in self.nodes:
                    self.nodes[nb] += 0.5 * lr * (x - self.nodes[nb])

            # Growth check on boundary
            if self.errors[bmu] > GT and len(self.nodes) < self.max_nodes:
                self._grow(bmu, x)
                self.errors[bmu] = 0.0

        gc.collect()
        return self

    def _grow(self, pos, x):
        """Add new nodes on empty boundary neighbours."""
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nb = (pos[0] + dr, pos[1] + dc)
            if nb not in self.nodes and len(self.nodes) < self.max_nodes:
                self.nodes[nb] = self.nodes[pos].copy()
                self.errors[nb] = 0.0

    def transform(self, X):
        keys = list(self.nodes.keys())
        W = np.array([self.nodes[k] for k in keys])
        D = cdist(X, W, 'sqeuclidean')
        return np.argmin(D, axis=1)

    def get_prototypes(self):
        return np.array(list(self.nodes.values()))


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  4.  TASOM — Time Adaptive Self-Organizing Map                  ║
# ╚═══════════════════════════════════════════════════════════════════╝

class TASOM:
    """
    Time Adaptive SOM (Shah-Hosseini & Safabakhsh, 2003).

    Extensions over standard SOM:
      • Adaptive learning rate: η_i(t) based on node activity frequency.
      • Adaptive neighbourhood: σ(t) decreases based on map convergence.
      • Scaling parameter β for input space normalisation (invariance
        to scaling, translation, rotation).
    """

    def __init__(self, grid_h=5, grid_w=5, dim=784,
                 n_iter=500, beta=0.5):
        self.H, self.W, self.dim = grid_h, grid_w, dim
        self.n_iter = n_iter
        self.beta = beta  # scaling parameter

    def fit(self, X):
        rng = np.random.RandomState(42)
        N, D = X.shape
        n_nodes = self.H * self.W

        # Normalise input (translation + scale invariance)
        self.mu_ = X.mean(axis=0)
        self.std_ = X.std(axis=0) + 1e-8
        X_norm = (X - self.mu_) / self.std_ * self.beta

        # Init weights from data
        idx = rng.choice(N, min(n_nodes, N), replace=True)
        self.weights = X_norm[idx].reshape(self.H, self.W, D).copy()

        # Activity counters for adaptive learning rates
        activity = np.ones((self.H, self.W))

        for t in range(self.n_iter):
            frac = t / self.n_iter
            global_lr = 0.5 * (1 - frac)
            global_sigma = max(self.H, self.W) / 2.0 * (1 - frac) + 0.5

            xi = rng.randint(N)
            x = X_norm[xi]

            # BMU
            diff = self.weights - x[np.newaxis, np.newaxis, :]
            dists = np.sum(diff ** 2, axis=2)
            bmu = np.unravel_index(np.argmin(dists), dists.shape)

            # Adaptive LR based on activity (more active → lower LR)
            activity[bmu] += 1
            lr_adaptive = global_lr / (1 + np.log1p(activity))

            # Adaptive neighbourhood
            rows = np.arange(self.H)[:, None]
            cols = np.arange(self.W)[None, :]
            d2 = (rows - bmu[0]) ** 2 + (cols - bmu[1]) ** 2
            sigma_adaptive = global_sigma * (1 + 0.1 *
                             np.log1p(activity[bmu] / (t + 1)))
            h = np.exp(-d2 / (2 * sigma_adaptive ** 2))

            update = lr_adaptive[:, :, np.newaxis] * h[:, :, np.newaxis] * \
                     (x[np.newaxis, np.newaxis, :] - self.weights)
            self.weights += update

        gc.collect()
        return self

    def transform(self, X):
        X_norm = (X - self.mu_) / self.std_ * self.beta
        W_flat = self.weights.reshape(-1, self.dim)
        D = cdist(X_norm, W_flat, 'sqeuclidean')
        return np.argmin(D, axis=1)

    def get_prototypes(self):
        return self.weights.reshape(-1, self.dim) / self.beta * self.std_ \
               + self.mu_


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  5.  ConformalSOM — Conformal Map approach                       ║
# ╚═══════════════════════════════════════════════════════════════════╝

class ConformalSOM:
    """
    Conformal Map SOM (Kürková & Neruda, Ein-Dor & Feldman).

    Uses conformal (angle-preserving) interpolation between grid nodes
    to achieve a smooth, one-to-one mapping. After standard SOM
    training, a conformal regularisation step smooths the grid by
    minimising angular distortion via Cauchy-Riemann conditions
    on adjacent node differences.
    """

    def __init__(self, grid_h=5, grid_w=5, dim=784,
                 n_iter=500, conf_iter=50, conf_lr=0.01):
        self.base_som = SOM(grid_h, grid_w, dim, n_iter=n_iter)
        self.conf_iter = conf_iter
        self.conf_lr = conf_lr
        self.H, self.W, self.dim = grid_h, grid_w, dim

    def fit(self, X):
        self.base_som.fit(X)
        W = self.base_som.weights.copy()  # (H, W, D)

        # Conformal regularisation: enforce smooth Cauchy-Riemann-like
        # conditions — partial derivatives along grid axes should be
        # orthogonal and equal-magnitude in data space.
        for _ in range(self.conf_iter):
            grad = np.zeros_like(W)
            for r in range(self.H):
                for c in range(self.W):
                    # Horizontal derivative
                    dh = np.zeros(self.dim)
                    if c < self.W - 1:
                        dh = W[r, c + 1] - W[r, c]
                    elif c > 0:
                        dh = W[r, c] - W[r, c - 1]
                    # Vertical derivative
                    dv = np.zeros(self.dim)
                    if r < self.H - 1:
                        dv = W[r + 1, c] - W[r, c]
                    elif r > 0:
                        dv = W[r, c] - W[r - 1, c]

                    # Conformality: ||dh|| ≈ ||dv||, dh ⊥ dv
                    nh, nv = np.linalg.norm(dh) + 1e-10, \
                             np.linalg.norm(dv) + 1e-10
                    # Scale equalisation
                    scale_err = (nh - nv)
                    if nh > nv:
                        grad[r, c] += self.conf_lr * scale_err * dh / nh
                    else:
                        grad[r, c] -= self.conf_lr * abs(scale_err) * dv / nv
                    # Orthogonality: minimise |dh·dv|
                    dot = np.dot(dh, dv) / (nh * nv)
                    grad[r, c] -= self.conf_lr * dot * (dh / nh + dv / nv)

            W -= grad

        self.base_som.weights = W
        gc.collect()
        return self

    def transform(self, X):
        return self.base_som.transform(X)

    def get_prototypes(self):
        return self.base_som.get_prototypes()


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  6.  ElasticMap — Elastic Map with bending/stretching energy     ║
# ╚═══════════════════════════════════════════════════════════════════╝

class ElasticMap:
    """
    Elastic Map (Gorban & Zinovyev, 2005).

    Minimises a combined objective:
      E = E_approx + λ_bend · E_bend + λ_stretch · E_stretch

    where E_approx is the least-squares approximation error,
    E_bend penalises curvature (second-order differences on the grid),
    and E_stretch penalises elongation (first-order differences).
    Inspired by thin-plate spline interpolation.
    """

    def __init__(self, grid_h=5, grid_w=5, dim=784,
                 n_iter=100, lam_bend=0.1, lam_stretch=0.01):
        self.H, self.W, self.dim = grid_h, grid_w, dim
        self.n_iter = n_iter
        self.lam_b = lam_bend
        self.lam_s = lam_stretch

    def fit(self, X):
        rng = np.random.RandomState(42)
        n_nodes = self.H * self.W

        # Init from K-Means
        km = KMeans(n_clusters=min(n_nodes, len(X)),
                    n_init=3, random_state=42)
        km.fit(X[:min(5000, len(X))])
        W = km.cluster_centers_[:n_nodes].copy()
        if len(W) < n_nodes:
            extra = rng.randn(n_nodes - len(W), self.dim).astype(np.float32)
            W = np.vstack([W, extra * 0.01])
        W = W.reshape(self.H, self.W, self.dim)

        for it in range(self.n_iter):
            W_flat = W.reshape(-1, self.dim)
            # Assignment
            D = cdist(X, W_flat, 'sqeuclidean')
            assign = np.argmin(D, axis=1)

            # Update each node: weighted mean of assigned data
            for k in range(n_nodes):
                mask = assign == k
                r, c = divmod(k, self.W)
                if mask.sum() > 0:
                    centroid = X[mask].mean(axis=0)
                    W[r, c] = 0.7 * W[r, c] + 0.3 * centroid

            # Elastic regularisation
            grad = np.zeros_like(W)
            for r in range(self.H):
                for c in range(self.W):
                    # Stretching: first-order differences
                    for dr, dc in [(0, 1), (1, 0)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < self.H and 0 <= nc < self.W:
                            grad[r, c] += self.lam_s * (W[r, c] - W[nr, nc])
                    # Bending: second-order (Laplacian)
                    laplacian = np.zeros(self.dim)
                    cnt = 0
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < self.H and 0 <= nc < self.W:
                            laplacian += W[nr, nc]
                            cnt += 1
                    if cnt > 0:
                        laplacian = laplacian / cnt - W[r, c]
                        grad[r, c] -= self.lam_b * laplacian

            W -= 0.1 * grad

        self.weights = W
        gc.collect()
        return self

    def transform(self, X):
        W_flat = self.weights.reshape(-1, self.dim)
        D = cdist(X, W_flat, 'sqeuclidean')
        return np.argmin(D, axis=1)

    def get_prototypes(self):
        return self.weights.reshape(-1, self.dim)


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  7.  OSMap — Oriented and Scalable Map                          ║
# ╚═══════════════════════════════════════════════════════════════════╝

class OSMap:
    """
    Oriented and Scalable Map (Yin, 2008).

    Generalisations over standard SOM:
      1. Matrix-exponential neighbourhood function instead of Gaussian,
         supporting oriented (anisotropic) neighbourhoods.
      2. Scale parameter s: allows multiple best-matching per input,
         modelling n-fold coverings of the domain.

    The neighbourhood function is:
      h(i, j) = exp(-Λ · d²(i,j))
    where Λ is a positive-definite matrix controlling shape/orientation.
    """

    def __init__(self, grid_h=5, grid_w=5, dim=784,
                 n_iter=500, scale=1.0):
        self.H, self.W, self.dim = grid_h, grid_w, dim
        self.n_iter = n_iter
        self.scale = scale  # s > 1 means n-fold covering

    def fit(self, X):
        rng = np.random.RandomState(42)
        N = len(X)

        # Init weights
        idx = rng.choice(N, min(self.H * self.W, N), replace=True)
        self.weights = X[idx].reshape(self.H, self.W, self.dim).copy()

        # Orientation matrix Λ — starts isotropic, adapts
        Lambda = np.eye(2).astype(np.float32)

        for t in range(self.n_iter):
            frac = t / self.n_iter
            lr = 0.5 * (1 - frac)
            sigma_scale = max(self.H, self.W) / 2.0 * (1 - frac) + 0.3

            xi = rng.randint(N)
            x = X[xi]

            # Find top-s BMUs
            diff = self.weights - x[np.newaxis, np.newaxis, :]
            dists = np.sum(diff ** 2, axis=2)
            flat_dists = dists.ravel()
            n_bmu = max(1, int(self.scale))
            top_bmu_flat = np.argsort(flat_dists)[:n_bmu]

            for bmu_flat in top_bmu_flat:
                bmu = np.unravel_index(bmu_flat, (self.H, self.W))

                # Matrix-exponential neighbourhood
                rows = np.arange(self.H)[:, None]
                cols = np.arange(self.W)[None, :]
                dr = rows - bmu[0]
                dc = cols - bmu[1]
                # d² with Λ: d² = [dr dc] Λ [dr dc]^T
                d2 = (Lambda[0, 0] * dr ** 2 + 2 * Lambda[0, 1] * dr * dc +
                      Lambda[1, 1] * dc ** 2)
                h = np.exp(-d2 / (2 * sigma_scale ** 2))

                update = lr * h[:, :, np.newaxis] * \
                         (x[np.newaxis, np.newaxis, :] - self.weights)
                self.weights += update / n_bmu

            # Adapt Λ periodically via data covariance in map coordinates
            if (t + 1) % max(1, self.n_iter // 10) == 0:
                # Estimate orientation from recent assignments
                sample = X[rng.choice(N, min(200, N), replace=False)]
                W_flat = self.weights.reshape(-1, self.dim)
                assigns = np.argmin(cdist(sample, W_flat, 'sqeuclidean'), axis=1)
                map_coords = np.column_stack([assigns // self.W,
                                               assigns % self.W]).astype(float)
                if map_coords.std() > 0:
                    cov = np.cov(map_coords.T) + 1e-4 * np.eye(2)
                    Lambda = np.linalg.inv(cov).astype(np.float32)

        gc.collect()
        return self

    def transform(self, X):
        W_flat = self.weights.reshape(-1, self.dim)
        D = cdist(X, W_flat, 'sqeuclidean')
        return np.argmin(D, axis=1)

    def get_prototypes(self):
        return self.weights.reshape(-1, self.dim)


# ──────────────────────────────────────────────────────────────────────
# Registry for easy access
# ──────────────────────────────────────────────────────────────────────

SOM_REGISTRY = {
    "SOM":          SOM,
    "GTM":          GTM,
    "GSOM":         GSOM,
    "TASOM":        TASOM,
    "ConformalSOM": ConformalSOM,
    "ElasticMap":   ElasticMap,
    "OSMap":        OSMap,
}
