# 🧬 MedMNIST2D — 3-Stage Pattern Recognition Framework

[![Paper](https://img.shields.io/badge/Paper-PDF-red)](paper.pdf)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)

> **A Three-Stage Pattern Recognition Framework for Medical Image Classification: From Manifold Learning to Massive Topology-Aware Ensembles**

This repository contains the official implementation of a modular, high-fidelity pipeline for medical image classification across **8 MedMNIST2D datasets**. The framework bridges classical topology discovery algorithms (SOM/ART) with modern deep learning and massive ensemble strategies.

---

## 🚀 Framework Architecture

The pipeline is divided into three distinct stages, progressively increasing in complexity and sophistication:

### 🔹 Stage 1: Baseline Manifold Learning
Establish interpretable baselines using **UMAP** (Uniform Manifold Approximation and Projection) for dimensionality reduction ($n=50$), followed by a suite of probabilistic classifiers:
- **GNB:** Gaussian Naïve Bayes.
- **LDA:** Linear Discriminant Analysis.
- **QDA:** Quadratic Discriminant Analysis.
- **GMM:** Gaussian Mixture Model soft-clustering.

### 🔹 Stage 2: Biologically-Inspired Hybrid CNN-SNN
A custom neural architecture featuring a **lightweight CNN backbone** for spatial feature extraction, coupled with:
- **SOM Family:** Evaluation of 7 Self-Organizing Map variants (SOM, GTM, GSOM, TASOM, etc.).
- **ART Family:** Evaluation of 15 Adaptive Resonance Theory variants (ART1-3, Fuzzy ART, ARTMAP, etc.).
- **Tanh-GeLU Block:** A novel activation layer with **Negative-Axis Gaussian Extraction**.
- **RBF Layer:** Radial Basis Function layer with learnable centres.
- **Adam + SGLD:** Stochastic Gradient Langevin Dynamics for approximate MCMC posterior sampling.

### 🔹 Stage 3: Topology-Aware Noise-Immune Pipeline
A massive 9-step processing pipeline designed for maximum robustness:
1. **Dynamic Dilution:** MAIS + Effective Sample Size (ESS) filtering.
2. **Entropy Selection:** High-confidence exemplar recovery.
3. **Stochastic Cascade Augmentation:** 29-method augmentation chain.
4. **Boundary Cleaning:** ADASYN + Tomek Links.
5. **SLERP Expansion:** Spherical Linear Interpolation in PCA space.
6. **Randomized SVD:** Preservation of Frobenius norm energy.
7. **Coreset Selection:** HDBSCAN + Entropy weighting.
8. **Barnes-Hut t-SNE:** Topological manifold verification.
9. **30-Model BMA Ensemble:** Bayesian Model Averaging spanning KNN, Tree Ensembles, Linear models, and custom **Nyström Kernels** (GRPF, t-Student, IMQ).

---

## 📂 Project Structure

```text
.
├── medmnist_pipeline.ipynb         # Main orchestration notebook
├── modules/                        # Core implementation package
│   ├── data_loader.py              # MedMNIST loading & batching
│   ├── metrics.py                  # Evaluation & visualization suite
│   ├── stage1.py                   # Stage 1: Manifold Baselines
│   ├── stage2.py                   # Stage 2: CNN-SNN Hybrid
│   ├── stage3.py                   # Stage 3: Ensemble Pipeline
│   ├── stage3_utils.py             # MAIS, SLERP, Augmentations, Coresets
│   ├── som_variants.py             # 7 SOM implementations
│   └── art_variants.py             # 15 ART implementations
├── results/                        # Output figures and diagnostics
├── paper.pdf                       # Technical documentation / Paper
├── requirements.txt                # Dependency list
└── medmnist_all_results.csv        # Final comparison table
```

---

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/celil_project_pr_bil564.git
   cd celil_project_pr_bil564
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the Pipeline:**
   Open `medmnist_pipeline.ipynb` in your preferred Jupyter environment and run all cells. The pipeline automatically downloads the MedMNIST datasets into a `./data` directory.

---

## 📊 Key Results

| Metric | Stage 1 (Best) | Stage 2 (CNN-SNN) | Stage 3 (BMA) |
| :--- | :--- | :--- | :--- |
| **Mean Accuracy** | 0.519 | **0.564** | 0.268 |
| **Mean AUC** | 0.785 | **0.871** | 0.612 |

*Note: Stage 2 achieves the most robust balance between spatial feature learning and biological plausibility, showing a **+114%** improvement on OCTMNIST over Stage 1 baselines.*

---

## ✍️ Author & Citation

**Emre Celil Kuru**  
Department of Computer Science  
TOBB University of Economics and Technology, Ankara, Türkiye  
📧 [emrecelil999@gmail.com](mailto:emrecelil999@gmail.com)

If you use this framework in your research, please refer to the documentation in `paper.pdf`.

---
*Developed for BIL564: Advanced Pattern Recognition.*
