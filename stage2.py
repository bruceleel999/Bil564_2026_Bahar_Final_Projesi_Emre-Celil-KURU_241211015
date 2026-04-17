"""
stage2.py — Custom Spiking Neural Network / Hybrid Architecture
===============================================================
Updated Pipeline:
1. **CNN Feature Extraction:** Conv layers reduce spatial data to compact
   feature vectors — drastically reducing compute for all downstream steps.
2. **SOM/ART Clustering:** All 7 SOM + 15 ART variants from dedicated modules.
3. **Early Hidden Layers:** Tanh → GeLU with negative-axis Gaussian extraction.
4. **Final Hidden Layer:** RBF (centres via K-Means / FCM).
5. **Output Layer:** Softmax.
6. **Optimisation:** Adam + SGLD.
"""

import gc
import math
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans

from modules.data_loader import MEDMNIST2D_DATASETS, load_medmnist_images
from modules.metrics import (
    compute_metrics, record_result,
    plot_confusion_matrix, plot_roc_curves,
)
from modules.som_variants import SOM_REGISTRY
from modules.art_variants import ART_UNSUPERVISED_REGISTRY, ART_SUPERVISED_REGISTRY

warnings.filterwarnings("ignore")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  1.  CNN Feature Extractor                                       ║
# ╚═══════════════════════════════════════════════════════════════════╝

class CNNFeatureExtractor(nn.Module):
    """
    Lightweight CNN backbone for feature extraction from 28×28 images.
    Reduces (C, 28, 28) → (feature_dim,) via 3 conv blocks + global avg pool.

    Architecture
    ------------
    Conv2d(in, 32, 3) → BN → ReLU → MaxPool(2)
    Conv2d(32, 64, 3) → BN → ReLU → MaxPool(2)
    Conv2d(64, 128, 3) → BN → ReLU → AdaptiveAvgPool(1)
    → Flatten → Linear(128, feature_dim)

    This drastically lowers the computational cost compared to operating
    on the full flattened 784-dim (or 2352-dim RGB) vectors.
    """

    def __init__(self, in_channels=1, feature_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),       # → (32, 14, 14)

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),       # → (64, 7, 7)

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # → (128, 1, 1)
        )
        self.fc = nn.Linear(128, feature_dim)

    def forward(self, x):
        h = self.conv(x)
        h = h.view(h.size(0), -1)  # (B, 128)
        return self.fc(h)           # (B, feature_dim)


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  2.  Tanh-GeLU Block with Negative-Axis Gaussian Extraction     ║
# ╚═══════════════════════════════════════════════════════════════════╝

class TanhGeLUBlock(nn.Module):
    """
    Custom activation: Tanh → GeLU with learnable Gaussian gate
    on the negative axis.

    GeLU(x) = x · Φ(x). On negative axis this forms a Gaussian bump.
    We learn μ, σ and gate:
        g(x) = exp(-(x - μ)² / 2σ²)
    applied only where tanh(x) < 0.
    """

    def __init__(self, dim):
        super().__init__()
        self.mu = nn.Parameter(torch.zeros(dim))
        self.log_sigma = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        h = torch.tanh(x)
        g = F.gelu(h)
        sigma = self.log_sigma.exp().clamp(min=1e-4)
        neg_mask = (h < 0).float()
        gauss_gate = torch.exp(-0.5 * ((h - self.mu) / sigma) ** 2)
        return g * (1.0 - neg_mask) + g * gauss_gate * neg_mask


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  3.  RBF Layer                                                   ║
# ╚═══════════════════════════════════════════════════════════════════╝

def fuzzy_cmeans_centres(X, n_centres, m=2.0, max_iter=80, seed=42):
    """Fuzzy C-Means clustering to determine RBF centres."""
    rng = np.random.RandomState(seed)
    N, D = X.shape
    U = rng.dirichlet(np.ones(n_centres), size=N).astype(np.float32)
    for _ in range(max_iter):
        Um = U ** m
        centres = (Um.T @ X) / (Um.sum(axis=0)[:, None] + 1e-10)
        dists = np.zeros((N, n_centres), dtype=np.float32)
        for j in range(n_centres):
            dists[:, j] = np.linalg.norm(X - centres[j], axis=1) + 1e-10
        for j in range(n_centres):
            ratio = dists[:, j:j + 1] / dists
            U[:, j] = 1.0 / (ratio ** (2.0 / (m - 1))).sum(axis=1)
    return centres


class RBFLayer(nn.Module):
    """
    Radial Basis Function layer: φ_k(x) = exp(-β_k ||x - c_k||²).
    Centres fixed from K-Means/FCM; log-widths learnable.
    """

    def __init__(self, in_features, n_centres, centres_np=None):
        super().__init__()
        self.n_centres = n_centres
        if centres_np is not None:
            self.centres = nn.Parameter(
                torch.from_numpy(centres_np.astype(np.float32)),
                requires_grad=False)
        else:
            self.centres = nn.Parameter(
                torch.randn(n_centres, in_features) * 0.1,
                requires_grad=False)
        self.log_betas = nn.Parameter(torch.zeros(n_centres))

    def forward(self, x):
        diff = x.unsqueeze(1) - self.centres.unsqueeze(0)
        sq_dist = (diff ** 2).sum(dim=2)
        betas = self.log_betas.exp().clamp(min=1e-4)
        return torch.exp(-betas.unsqueeze(0) * sq_dist)


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  4.  Full Hybrid SNN Model (with CNN front-end)                  ║
# ╚═══════════════════════════════════════════════════════════════════╝

class HybridSNNModel(nn.Module):
    """
    CNN → TanhGeLU hidden layers → RBF → Softmax (in loss).

    CNN extracts features from (C, 28, 28) images, then the SNN-style
    backbone processes the feature vectors.
    """

    def __init__(self, in_channels, cnn_feat_dim, hidden1, hidden2,
                 n_rbf_centres, n_classes, rbf_centres_np=None):
        super().__init__()
        self.cnn = CNNFeatureExtractor(in_channels, cnn_feat_dim)
        self.fc1 = nn.Linear(cnn_feat_dim, hidden1)
        self.act1 = TanhGeLUBlock(hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.act2 = TanhGeLUBlock(hidden2)
        self.rbf = RBFLayer(hidden2, n_rbf_centres, rbf_centres_np)
        self.fc_out = nn.Linear(n_rbf_centres, n_classes)

    def forward(self, x):
        feat = self.cnn(x)           # (B, cnn_feat_dim)
        h = self.act1(self.fc1(feat))
        h = self.act2(self.fc2(h))
        h = self.rbf(h)
        return self.fc_out(h)

    def extract_features(self, x):
        """Return CNN features for SOM/ART clustering."""
        with torch.no_grad():
            return self.cnn(x)


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  5.  Adam + SGLD Optimiser                                      ║
# ╚═══════════════════════════════════════════════════════════════════╝

class AdamSGLD(torch.optim.Adam):
    """Adam + Stochastic Gradient Langevin Dynamics noise injection."""

    def __init__(self, params, lr=1e-3, temperature=1e-4, **kwargs):
        super().__init__(params, lr=lr, **kwargs)
        self.temperature = temperature

    @torch.no_grad()
    def step(self, closure=None):
        loss = super().step(closure)
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                noise_std = math.sqrt(2.0 * lr * self.temperature)
                p.add_(torch.randn_like(p) * noise_std)
        return loss


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  6.  SOM / ART Multi-Variant Clustering Evaluation              ║
# ╚═══════════════════════════════════════════════════════════════════╝

def _run_som_art_clustering(features_np, y_train, features_test_np, y_test,
                            n_classes, ds_name, max_samples=3000):
    """
    Run all SOM and ART variants on CNN-extracted features.
    Unsupervised variants: evaluate cluster quality via majority-vote.
    Supervised variants: evaluate classification accuracy.
    """
    # Subsample for speed
    N = len(features_np)
    if N > max_samples:
        idx = np.random.choice(N, max_samples, replace=False)
        feat_sub, y_sub = features_np[idx], y_train[idx]
    else:
        feat_sub, y_sub = features_np, y_train

    dim = feat_sub.shape[1]
    grid_sz = max(3, int(np.sqrt(min(25, n_classes * 3))))

    # ── Unsupervised SOM variants ──
    print("    SOM variants:")
    for name, cls in SOM_REGISTRY.items():
        try:
            if name == "GTM":
                model = cls(grid_size=grid_sz, n_basis=min(16, grid_sz ** 2),
                            dim=dim, n_iter=30)
            elif name == "GSOM":
                model = cls(dim=dim, spread_factor=0.5, n_iter=300,
                            max_nodes=100)
            elif name in ("SOM", "TASOM", "ConformalSOM", "OSMap"):
                model = cls(grid_h=grid_sz, grid_w=grid_sz, dim=dim,
                            n_iter=300)
            elif name == "ElasticMap":
                model = cls(grid_h=grid_sz, grid_w=grid_sz, dim=dim,
                            n_iter=50)
            else:
                model = cls(grid_h=grid_sz, grid_w=grid_sz, dim=dim)

            model.fit(feat_sub)
            labels = model.transform(features_test_np)
            # Majority-vote accuracy
            from scipy.stats import mode
            train_labels = model.transform(feat_sub)
            mapping = {}
            for cl in np.unique(train_labels):
                mask = train_labels == cl
                if mask.sum() > 0:
                    mapping[cl] = int(mode(y_sub[mask], keepdims=True).mode[0])
            y_mapped = np.array([mapping.get(l, 0) for l in labels])
            m = compute_metrics(y_test, y_mapped, n_classes=n_classes)
            record_result("Stage2", ds_name, f"SOM_{name}", m)
            print(f"      {name:15s}  Acc={m['accuracy']:.4f}")
        except Exception as e:
            print(f"      {name:15s}  FAILED: {e}")
        gc.collect()

    # ── Unsupervised ART variants ──
    print("    ART unsupervised variants:")
    for name, cls in ART_UNSUPERVISED_REGISTRY.items():
        try:
            if name in ("ART1",):
                model = cls(dim=dim, rho=0.5, max_categories=50, n_iter=3)
            elif name in ("FusionART",):
                model = cls(dim=dim, n_channels=3, rho=0.5,
                            max_categories=50, n_iter=3)
            elif name in ("TopoART",):
                model = cls(dim=dim, rho1=0.5, rho2=0.7,
                            max_categories=80, n_iter=3)
            elif name in ("HypersphereART",):
                r_max = np.linalg.norm(feat_sub.std(axis=0)) * 3
                model = cls(dim=dim, rho=0.3, r_max=r_max,
                            max_categories=50, n_iter=3)
            else:
                model = cls(dim=dim, rho=0.5, max_categories=50, n_iter=3)

            model.fit(feat_sub)
            labels = model.predict(features_test_np)
            # Majority-vote
            train_labels = model.predict(feat_sub)
            from scipy.stats import mode
            mapping = {}
            for cl in np.unique(train_labels):
                mask = train_labels == cl
                if mask.sum() > 0:
                    mapping[cl] = int(mode(y_sub[mask], keepdims=True).mode[0])
            y_mapped = np.array([mapping.get(int(l), 0) for l in labels])
            m = compute_metrics(y_test, y_mapped, n_classes=n_classes)
            record_result("Stage2", ds_name, f"ART_{name}", m)
            print(f"      {name:20s}  Acc={m['accuracy']:.4f}")
        except Exception as e:
            print(f"      {name:20s}  FAILED: {e}")
        gc.collect()

    # ── Supervised ART variants ──
    print("    ART supervised variants:")
    for name, cls in ART_SUPERVISED_REGISTRY.items():
        try:
            if name in ("ARTMAP", "FuzzyARTMAP"):
                model = cls(dim=dim, n_classes=n_classes, rho_a=0.5,
                            max_categories=100)
            elif name == "SFAM":
                model = cls(dim=dim, rho=0.5, max_categories=100)
            elif name == "GaussianARTMAP":
                model = cls(dim=dim, rho=0.3, max_categories=100)
            elif name == "HypersphereARTMAP":
                r_max = np.linalg.norm(feat_sub.std(axis=0)) * 3
                model = cls(dim=dim, rho=0.3, r_max=r_max,
                            max_categories=100)
            elif name == "LAPART":
                model = cls(dim=dim, n_classes=n_classes, rho=0.5,
                            max_categories=100)
            else:
                model = cls(dim=dim, n_classes=n_classes, rho=0.5)

            model.fit(feat_sub, y_sub)
            y_pred = model.predict(features_test_np)
            m = compute_metrics(y_test, y_pred, n_classes=n_classes)
            record_result("Stage2", ds_name, f"ART_{name}", m)
            print(f"      {name:20s}  Acc={m['accuracy']:.4f}")
        except Exception as e:
            print(f"      {name:20s}  FAILED: {e}")
        gc.collect()


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  7.  Training Loop                                               ║
# ╚═══════════════════════════════════════════════════════════════════╝

def _train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(xb)
        correct += (logits.argmax(1) == yb).sum().item()
        total += len(xb)
    return total_loss / total, correct / total


@torch.no_grad()
def _eval_model(model, loader):
    model.eval()
    preds, probs, targets = [], [], []
    for xb, yb in loader:
        xb = xb.to(DEVICE)
        logits = model(xb)
        p = F.softmax(logits, dim=1).cpu().numpy()
        probs.append(p)
        preds.append(np.argmax(p, axis=1))
        targets.append(yb.numpy())
    return np.concatenate(preds), np.concatenate(probs), np.concatenate(targets)


@torch.no_grad()
def _extract_features_batched(model, loader):
    """Extract CNN features in batches."""
    model.eval()
    feats = []
    for xb, _ in loader:
        xb = xb.to(DEVICE)
        f = model.extract_features(xb).cpu().numpy()
        feats.append(f)
    return np.concatenate(feats, axis=0)


def _determine_rbf_centres(X, n_centres, method="kmeans"):
    n_centres = min(n_centres, len(X) - 1)
    if method == "fcm":
        return fuzzy_cmeans_centres(X, n_centres)
    km = KMeans(n_clusters=n_centres, n_init=5, random_state=42)
    km.fit(X)
    return km.cluster_centers_.astype(np.float32)


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  8.  Public Entry Point                                          ║
# ╚═══════════════════════════════════════════════════════════════════╝

def run_stage2(datasets=None, epochs=20, batch_size=256,
               cnn_feat_dim=64, hidden1=128, hidden2=64, n_rbf=32,
               rbf_method="kmeans", lr=1e-3, sgld_temp=1e-4,
               root="./data"):
    """
    Execute Stage 2 on selected MedMNIST2D datasets.

    Pipeline per dataset:
      1. Load images (spatial format)
      2. Build HybridSNNModel (CNN + TanhGeLU + RBF)
      3. Train end-to-end with Adam+SGLD
      4. Extract CNN features → run ALL SOM + ART variants
      5. Evaluate all models
    """
    if datasets is None:
        datasets = MEDMNIST2D_DATASETS

    for ds_name in datasets:
        print(f"\n{'='*60}")
        print(f" Stage 2 — {ds_name}")
        print(f"{'='*60}")

        X_tr, y_tr, _, _, X_te, y_te, n_classes, _ = \
            load_medmnist_images(ds_name, root=root)

        in_channels = X_tr.shape[1]  # 1 or 3

        # ── Step 1: Determine RBF centres from flattened data ──
        print("  [1/4] RBF centres …")
        flat_sample = X_tr[:min(3000, len(X_tr))].reshape(
            min(3000, len(X_tr)), -1)
        # Project to lower dim for RBF centre calc
        from sklearn.decomposition import PCA
        pca_rbf = PCA(n_components=min(hidden2, flat_sample.shape[1]),
                       random_state=42)
        flat_proj = pca_rbf.fit_transform(flat_sample)
        rbf_centres = _determine_rbf_centres(flat_proj, n_rbf,
                                              method=rbf_method)

        # ── Step 2: Build & train HybridSNN (CNN-based) ──
        print("  [2/4] Training Hybrid CNN-SNN …")
        model = HybridSNNModel(
            in_channels=in_channels,
            cnn_feat_dim=cnn_feat_dim,
            hidden1=hidden1, hidden2=hidden2,
            n_rbf_centres=rbf_centres.shape[0],
            n_classes=n_classes,
            rbf_centres_np=rbf_centres
        ).to(DEVICE)

        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_tr),
                          torch.from_numpy(y_tr.astype(np.int64))),
            batch_size=batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_te),
                          torch.from_numpy(y_te.astype(np.int64))),
            batch_size=batch_size, shuffle=False, num_workers=0)

        optimizer = AdamSGLD(model.parameters(), lr=lr,
                             temperature=sgld_temp)
        criterion = nn.CrossEntropyLoss()

        for ep in range(1, epochs + 1):
            loss, acc = _train_epoch(model, train_loader, optimizer, criterion)
            if ep % 5 == 0 or ep == epochs:
                print(f"    Epoch {ep:3d}/{epochs}  loss={loss:.4f}  "
                      f"train_acc={acc:.4f}")

        # ── Step 3: Evaluate CNN-SNN ──
        print("  [3/4] Evaluating CNN-SNN …")
        y_pred, y_prob, y_true = _eval_model(model, test_loader)
        m = compute_metrics(y_true, y_pred, y_prob, n_classes)
        record_result("Stage2", ds_name, "HybridCNN_SNN", m)
        plot_confusion_matrix(y_true, y_pred,
                              title=f"[S2] HybridCNN_SNN — {ds_name}")
        if n_classes <= 20:
            plot_roc_curves(y_true, y_prob, n_classes,
                            title=f"[S2] ROC HybridCNN_SNN — {ds_name}")
        print(f"  CNN-SNN  Acc={m['accuracy']:.4f}  F1={m['f1']:.4f}  "
              f"AUC={m['auc']:.4f}")

        # ── Step 4: Extract CNN features → SOM/ART evaluation ──
        print("  [4/4] SOM & ART variant evaluation on CNN features …")
        feat_train = _extract_features_batched(model, train_loader)
        feat_test = _extract_features_batched(model, test_loader)
        _run_som_art_clustering(feat_train, y_tr, feat_test, y_te,
                                n_classes, ds_name)

        # Cleanup
        del model, optimizer, train_loader, test_loader
        del X_tr, y_tr, X_te, y_te, feat_train, feat_test
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()

    print("\n✓ Stage 2 complete.")
