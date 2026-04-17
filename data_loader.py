"""
data_loader.py
==============
Handles downloading, caching, and batched iteration for MedMNIST2D datasets.
Uses medmnist's public API + custom memory-efficient wrappers.
"""

import gc
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import medmnist
from medmnist import INFO

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────
MEDMNIST2D_DATASETS = [
    "pathmnist", "dermamnist", "octmnist",
    "bloodmnist", "tissuemnist",
    "organamnist", "organcmnist", "organsmnist",
]

# ──────────────────────────────────────────────────────────────────────
# Data Loading Utilities
# ──────────────────────────────────────────────────────────────────────

def get_dataset_info(dataset_name: str) -> dict:
    """Return metadata dict for a given MedMNIST2D dataset."""
    return INFO[dataset_name]


def load_medmnist_flat(dataset_name: str, root: str = "./data"):
    """
    Load a single MedMNIST2D dataset and return flattened numpy arrays.

    Returns
    -------
    X_train, y_train, X_val, y_val, X_test, y_test : np.ndarray
        Images are flattened to 2-D (N, D). Labels are 1-D integer arrays.
    n_classes : int
    task_type : str   ('multi-class', 'multi-label', 'binary-class', 'ordinal-regression')
    """
    info = INFO[dataset_name]
    n_classes = len(info["label"])
    task_type = info["task"]
    DataClass = getattr(medmnist, info["python_class"])

    splits = {}
    for split in ("train", "val", "test"):
        ds = DataClass(split=split, download=True, root=root, size=28)
        imgs = ds.imgs                       # uint8  (N, 28, 28) or (N, 28, 28, C)
        labels = ds.labels.squeeze()         # (N,) or (N, n_labels)
        # Flatten images to (N, D), normalise to [0,1]
        imgs_flat = imgs.reshape(len(imgs), -1).astype(np.float32) / 255.0
        splits[split] = (imgs_flat, labels)
        del ds; gc.collect()

    X_tr, y_tr = splits["train"]
    X_val, y_val = splits["val"]
    X_te, y_te = splits["test"]

    return X_tr, y_tr, X_val, y_val, X_te, y_te, n_classes, task_type


def load_medmnist_images(dataset_name: str, root: str = "./data"):
    """
    Load a single MedMNIST2D dataset keeping spatial dims (for CNNs / augmentations).

    Returns
    -------
    X_train, y_train, X_val, y_val, X_test, y_test : np.ndarray
        Images shape (N, C, H, W) float32 [0,1]. Labels 1-D int.
    n_classes : int
    task_type : str
    """
    info = INFO[dataset_name]
    n_classes = len(info["label"])
    task_type = info["task"]
    DataClass = getattr(medmnist, info["python_class"])

    splits = {}
    for split in ("train", "val", "test"):
        ds = DataClass(split=split, download=True, root=root, size=28)
        imgs = ds.imgs.astype(np.float32) / 255.0
        labels = ds.labels.squeeze()
        # Ensure channel-first: (N, C, H, W)
        if imgs.ndim == 3:                   # grayscale (N,28,28)
            imgs = imgs[:, np.newaxis, :, :]
        elif imgs.ndim == 4:                 # RGB (N,28,28,3)
            imgs = imgs.transpose(0, 3, 1, 2)
        splits[split] = (imgs, labels)
        del ds; gc.collect()

    X_tr, y_tr = splits["train"]
    X_val, y_val = splits["val"]
    X_te, y_te = splits["test"]
    return X_tr, y_tr, X_val, y_val, X_te, y_te, n_classes, task_type


def make_torch_loader(X: np.ndarray, y: np.ndarray, batch_size: int = 256,
                      shuffle: bool = True) -> DataLoader:
    """Wrap numpy arrays in a PyTorch DataLoader for batched iteration."""
    tensor_x = torch.from_numpy(X)
    tensor_y = torch.from_numpy(y.astype(np.int64))
    dataset = TensorDataset(tensor_x, tensor_y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      pin_memory=False, num_workers=0)
