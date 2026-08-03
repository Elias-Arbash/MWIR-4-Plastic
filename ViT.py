#!/usr/bin/env python3
"""
ViT patch-wise experiments for the EBS-only polymer dataset.

This script follows the workflow from SF_patch.ipynb:
load EBS12/EBS22, SNV-normalize FENIX and FX50, keep fingerprint bands,
concatenate sensors, train/test split by sample, build a 6-patch validation
set, train Vision Transformer (ViT) models with early stopping, then evaluate
pixel-wise plus semantic/SAM-instance majority-voted prediction maps.

Grid search is performed over vit_depth. The best model is
selected by average class kappa on the test set and its full outputs are
saved to <output_root>/best_model/.
"""

import argparse
import itertools
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import matplotlib
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402

from augmentations import augment_hsi  # noqa: E402
from ebs_dataset_loader import extract_sensor_cubes, load_ebs_dataset  # noqa: E402
from functions import (  # noqa: E402
    concat_hsi_lists,
    majority_vote_sam_instances,
    majority_vote_semantic_objects,
    snv_normalize,
    visualize_mask,
    visualize_mask2,
)
from models import ViT  # noqa: E402

# ---------------------------------------------------------------------------
# Paths and class constants
# ---------------------------------------------------------------------------

DATASET_ROOT = Path("/home/arbash44/coding/hsi/polymers/EBS_only/EBS12_EBS22")
OUTPUT_ROOT = DATASET_ROOT / "trained_models" / "ViT_patch"

BACKGROUND_LABEL = 0
CLASS_LABELS = [1, 2, 3, 4]
EVALUATION_LABELS = [BACKGROUND_LABEL] + CLASS_LABELS
CLASS_NAMES = {
    1: "Styrene",
    2: "PA",
    3: "PC",
    4: "PP",
}

FENIX_SLICES = [
    slice(285, 321),  # 1600 - 1800 nm
    slice(322, 357),  # 1800 - 2000 nm
    slice(358, 394),  # 2000 - 2200 nm
]
FX50_SLICES = [
    slice(32, 38),    # 2976 - 3018 nm
    slice(75, 86),    # 3336 - 3420 nm
    slice(86, 97),    # 3428 - 3512 nm
    slice(114, 119),  # 3663 - 3696 nm
    slice(209, 212),  # 4458 - 4475 nm
]

# 107 FENIX bands + 36 FX50 bands after ROI slicing
TOTAL_BANDS = 143

# ---------------------------------------------------------------------------
# ViT default hyperparameters (from ViT.ipynb)
# ---------------------------------------------------------------------------

DEFAULT_PATCH_SIZE = 9
DEFAULT_BATCH_SIZE = 512
DEFAULT_EPOCHS = 300
DEFAULT_PATIENCE = 40
DEFAULT_LR = 5e-4
DEFAULT_WEIGHT_DECAY = 5e-3
DEFAULT_LR_DECAY_FACTOR = 0.9
NUM_WORKERS = 4

VIT_DIM = 64
VIT_HEADS = 4
VIT_MLP_DIM = 8
VIT_DROPOUT = 0.1
VIT_EMB_DROPOUT = 0.1
VIT_MODE = "ViT"

# Experiment grid axes
BAND_PATCH_OPTIONS = [1]  # For ViT, band_patch is always 1 (no GSE)
VIT_DEPTH_OPTIONS = [5]
DEFAULT_EXPERIMENTS = 1

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def json_ready(value: Any) -> Any:
    """Convert numpy-heavy results into JSON-serialisable Python values."""
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def select_roi_bands(hsi_list: Sequence[np.ndarray], sensor_type: str) -> List[np.ndarray]:
    """Keep only the fingerprint bands for FENIX or FX50."""
    sensor_key = sensor_type.upper()
    slice_map = {"FENIX": FENIX_SLICES, "FX50": FX50_SLICES}
    if sensor_key not in slice_map:
        raise ValueError(f"Unknown sensor_type '{sensor_type}'.")
    roi_cubes = []
    for cube in hsi_list:
        np_cube = np.asarray(cube)
        if np_cube.ndim != 3:
            raise ValueError(f"Expected 3D cube; got shape {np_cube.shape}.")
        roi_cubes.append(
            np.concatenate(
                [np_cube[:, :, s] for s in slice_map[sensor_key]],
                axis=2,
            )
        )
    return roi_cubes

# ---------------------------------------------------------------------------
# 6-patch split (training validation set from test cubes)
# ---------------------------------------------------------------------------

def patch_ebs_cubes_6_with_gt(
    cube_list: Sequence[np.ndarray],
    mask_list: Sequence[np.ndarray],
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Split each EBS HSI cube and its GT mask into 6 spatial patches
    (2 rows × 3 columns).  Returns corner patches (1,3,4,6) as the
    "train-like" split and centre-column patches (2,5) as the validation split.

    Returns: (all_corner, all_centre, all_corner_gt, all_centre_gt)
    """
    if len(cube_list) != len(mask_list):
        raise ValueError("cube_list and mask_list must have the same length.")

    all_corner: List[np.ndarray] = []
    all_centre: List[np.ndarray] = []
    all_corner_gt: List[np.ndarray] = []
    all_centre_gt: List[np.ndarray] = []

    for cube, mask in zip(cube_list, mask_list):
        cube = np.asarray(cube)
        mask = np.asarray(mask)
        H, W, _ = cube.shape
        h_mid = H // 2
        w_third = W // 3
        h_splits = [0, h_mid, H]
        w_splits = [0, w_third, 2 * w_third, W]

        patches_cube = []
        patches_mask = []
        for i in range(2):
            for j in range(3):
                patches_cube.append(
                    cube[h_splits[i]:h_splits[i + 1], w_splits[j]:w_splits[j + 1], :].copy()
                )
                patches_mask.append(
                    mask[h_splits[i]:h_splits[i + 1], w_splits[j]:w_splits[j + 1]].copy()
                )

        # Indices 0,2,3,5 → patches 1,3,4,6 (corners)
        # Indices 1,4     → patches 2,5 (centre column = validation)
        for idx in [0, 2, 3, 5]:
            all_corner.append(patches_cube[idx])
            all_corner_gt.append(patches_mask[idx])
        for idx in [1, 4]:
            all_centre.append(patches_cube[idx])
            all_centre_gt.append(patches_mask[idx])

    return all_corner, all_centre, all_corner_gt, all_centre_gt

# ---------------------------------------------------------------------------
# For ViT: Single band_patch, i.e. no GSE (identity transform)
# ---------------------------------------------------------------------------

def gain_neighborhood_band(
    x_train: np.ndarray,
    band: int,
    band_patch: int,
    patch: int,
) -> np.ndarray:
    """
    For ViT: Simply flatten the patch without applying any band neighborhood/GSE.
    Input shape : (N, patch, patch, band)
    Output shape: (N, patch*patch, band)
    """
    return x_train.reshape(x_train.shape[0], patch * patch, band)

# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class HSIDataset(Dataset):
    """
    Patch-based HSI dataset compatible with ViT.

    Labels in GT are 1-indexed (1..4); the dataset shifts them to 0-indexed
    (0..3) to match PyTorch cross-entropy expectations. Predictions from
    the model must be shifted back by +1 before evaluation against GT.
    """

    def __init__(
        self,
        hsi: Any,
        gt: Any,
        patch_size: int,
        times: int = 1,
        fill_value_x: float = 0.0,
        ignore_classes: List[int] = None,
        band_patch: int = 1,
        augmentations: Any = None,
        aug_prob: float = 0.5,
    ) -> None:
        if ignore_classes is None:
            ignore_classes = [0]
        self.hsi_list: List[np.ndarray] = hsi if isinstance(hsi, list) else [hsi]
        self.gt_list: List[np.ndarray] = gt if isinstance(gt, list) else [gt]
        self.patch_size = patch_size
        self.times = times
        self.ignore_classes = ignore_classes
        self.band_patch = band_patch
        self.augmentations = augmentations
        self.aug_prob = aug_prob

        pad = patch_size // 2
        self.padded_hsi_list = [
            np.pad(
                cube,
                ((pad, pad), (pad, pad), (0, 0)),
                mode="constant",
                constant_values=fill_value_x,
            )
            for cube in self.hsi_list
        ]

        base_indices = self._collect_indices()
        self.original_patch_count = len(base_indices)

        if self.augmentations and self.times > 1:
            aug_copies = [
                (c, i, j, lbl, True)
                for _ in range(self.times - 1)
                for c, i, j, lbl in base_indices
            ]
            self.indices = [(c, i, j, lbl, False) for c, i, j, lbl in base_indices] + aug_copies
        else:
            self.indices = [(c, i, j, lbl, False) for c, i, j, lbl in base_indices]

    def _collect_indices(self) -> List[Tuple[int, int, int, int]]:
        indices = []
        for cube_idx, (_, mask) in enumerate(zip(self.hsi_list, self.gt_list)):
            h, w = mask.shape
            for i in range(h):
                for j in range(w):
                    lbl = int(mask[i, j])
                    if lbl not in self.ignore_classes:
                        indices.append((cube_idx, i, j, lbl))
        return indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        cube_idx, i, j, lbl, is_augmented = self.indices[idx]
        padded = self.padded_hsi_list[cube_idx]
        patch = padded[i:i + self.patch_size, j:j + self.patch_size, :].copy()

        if self.augmentations and (is_augmented or self.times == 1):
            patch = augment_hsi(patch, self.augmentations, prob=self.aug_prob)

        band = patch.shape[-1]
        # For ViT: band_patch is always 1 and GSE is identity
        patch = gain_neighborhood_band(
            np.expand_dims(patch, axis=0), band, 1, self.patch_size
        ).squeeze(0)
        patch = np.transpose(patch, (1, 0))

        return (
            torch.tensor(patch, dtype=torch.float32),
            torch.tensor(lbl - 1, dtype=torch.long),  # shift to 0-indexed
        )

    def get_original_patch_count(self) -> int:
        return self.original_patch_count

# ---------------------------------------------------------------------------
# Class weight calculation
# ---------------------------------------------------------------------------

def calculate_class_weights_tensor(
    gt_list: Sequence[np.ndarray],
    ignore_classes: Sequence[int] = (BACKGROUND_LABEL,),
) -> torch.Tensor:
    """
    Compute normalised inverse-frequency class weights as a 1-D tensor
    ordered by ascending class label (1, 2, 3, 4 → index 0..3).
    """
    counts: Counter = Counter()
    for mask in gt_list:
        flat = mask.flatten()
        counts.update(v for v in flat if v not in ignore_classes)

    total = sum(counts.values())
    sorted_keys = sorted(counts.keys())
    weights = np.array([total / counts[k] for k in sorted_keys])
    weights = weights * 2 / weights.sum()
    return torch.tensor(weights, dtype=torch.float32)

# ---------------------------------------------------------------------------
# Model building
# ---------------------------------------------------------------------------

def build_model(
    patch_size: int,
    band_patch: int,
    num_classes: int,
    vit_depth: int,
) -> nn.Module:
    """Instantiate a ViT model and move it to CUDA."""
    model = ViT(
        image_size=patch_size,
        near_band=1,  # Always 1 for ViT, i.e., no band context
        num_patches=TOTAL_BANDS,
        num_classes=num_classes,
        dim=VIT_DIM,
        depth=vit_depth,
        heads=VIT_HEADS,
        mlp_dim=VIT_MLP_DIM,
        dropout=VIT_DROPOUT,
        emb_dropout=VIT_EMB_DROPOUT,
        mode="ViT",
    )
    return model.cuda()

def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    total_epochs: int,
    decay_factor: float = DEFAULT_LR_DECAY_FACTOR,
    decay_interval: int = None,
) -> torch.optim.lr_scheduler.LambdaLR:
    if decay_interval is None:
        decay_interval = max(1, total_epochs // 10)

    def lr_lambda(epoch: int) -> float:
        return decay_factor ** (epoch // decay_interval)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(
    model: nn.Module,
    label_train_loader: DataLoader,
    label_val_loader: DataLoader,
    model_path: Path,
    device: torch.device,
    num_epochs: int = DEFAULT_EPOCHS,
    patience: int = DEFAULT_PATIENCE,
    lr: float = DEFAULT_LR,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    class_weights_tensor: torch.Tensor = None,
) -> float:
    """
    Train the model with early stopping.  Saves the best-validation-loss
    checkpoint to *model_path* and returns the best validation loss.
    """
    criterion = nn.CrossEntropyLoss(
        weight=class_weights_tensor.to(device) if class_weights_tensor is not None else None
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = build_lr_scheduler(optimizer, num_epochs)

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for patches, labels in label_train_loader:
            patches, labels = patches.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(patches), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for patches, labels in label_val_loader:
                patches, labels = patches.to(device), labels.to(device)
                val_loss += criterion(model(patches), labels).item()

        scheduler.step()

        avg_train = train_loss / max(len(label_train_loader), 1)
        avg_val = val_loss / max(len(label_val_loader), 1)
        print(
            f"  Epoch [{epoch + 1}/{num_epochs}]  "
            f"train_loss={avg_train:.4f}  val_loss={avg_val:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), model_path)
            print(f"  ✓ checkpoint saved (val_loss={avg_val:.4f})")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"  Early stopping after {patience} epochs without improvement.")
            break

    print(f"  Best val_loss={best_val_loss / max(len(label_val_loader), 1):.4f}")
    return best_val_loss

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict_maps(
    model: nn.Module,
    cubes: Sequence[np.ndarray],
    gt_maps: Sequence[np.ndarray],
    device: torch.device,
    patch_size: int,
    band_patch: int,
    batch_size: int = DEFAULT_BATCH_SIZE,
    ignore_labels: Iterable[int] = (BACKGROUND_LABEL,),
    output_label_offset: int = 1,
) -> List[np.ndarray]:
    """
    Produce a full spatial prediction map for every cube.

    For each non-background pixel the surrounding spatial patch is extracted,
    (no GSE for ViT), and the model predicts the class.
    Model outputs are 0-indexed; *output_label_offset* shifts them back to the
    expert GT convention (1-indexed).
    """
    model.eval()
    ignore_set = set(ignore_labels)
    pred_maps: List[np.ndarray] = []

    for map_idx, (cube, gt_map) in enumerate(zip(cubes, gt_maps), start=1):
        print(f"  Predicting map {map_idx}/{len(cubes)}")
        cube = np.asarray(cube)
        gt_map = np.asarray(gt_map)

        pred_map = np.zeros(gt_map.shape, dtype=np.int64)
        eval_mask = ~np.isin(gt_map, list(ignore_set))
        rows, cols = np.where(eval_mask)

        if len(rows) == 0:
            pred_maps.append(pred_map)
            continue

        pad = patch_size // 2
        padded = np.pad(
            cube,
            ((pad, pad), (pad, pad), (0, 0)),
            mode="constant",
            constant_values=0,
        )

        for start in range(0, len(rows), batch_size):
            end = min(start + batch_size, len(rows))
            batch_rows = rows[start:end]
            batch_cols = cols[start:end]

            patches = np.stack([
                padded[r:r + patch_size, c:c + patch_size, :]
                for r, c in zip(batch_rows, batch_cols)
            ])  # (B, patch_size, patch_size, bands)

            band = patches.shape[-1]
            # For ViT, band_patch=1, GSE is identity
            patches = gain_neighborhood_band(patches, band, 1, patch_size)
            patches = np.transpose(patches, (0, 2, 1)).astype(np.float32)

            with torch.no_grad():
                logits = model(torch.from_numpy(patches).to(device))
                pred_labels = (
                    torch.argmax(logits, dim=1).cpu().numpy() + output_label_offset
                )

            pred_map[batch_rows, batch_cols] = pred_labels

        pred_maps.append(pred_map)

    return pred_maps

# ---------------------------------------------------------------------------
# Evaluation and visualization: unchanged from SF_patch.py
# ---------------------------------------------------------------------------

def classwise_kappa_from_confusion_matrix(
    cm: np.ndarray,
    labels: Sequence[int] = EVALUATION_LABELS,
    class_labels: Sequence[int] = CLASS_LABELS,
) -> Dict[int, float]:
    """Compute one-vs-rest Cohen's kappa for each foreground class."""
    total = cm.sum()
    eps = 1e-10
    label_to_index = {lbl: i for i, lbl in enumerate(labels)}
    kappas: Dict[int, float] = {}

    for lbl in class_labels:
        idx = label_to_index[lbl]
        tp = cm[idx, idx]
        fp = cm[:, idx].sum() - tp
        fn = cm[idx, :].sum() - tp
        tn = total - tp - fp - fn
        observed = (tp + tn) / (total + eps)
        expected = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (total ** 2 + eps)
        kappas[int(lbl)] = float((observed - expected) / (1.0 - expected + eps))

    return kappas


def evaluate_maps(
    gt_maps: Sequence[np.ndarray],
    pred_maps: Sequence[np.ndarray],
    labels: Sequence[int] = EVALUATION_LABELS,
    class_labels: Sequence[int] = CLASS_LABELS,
    ignore_labels: Iterable[int] = (BACKGROUND_LABEL,),
) -> Dict[str, Any]:
    """Evaluate maps on non-background GT pixels."""
    y_true_parts: List[np.ndarray] = []
    y_pred_parts: List[np.ndarray] = []

    for gt_map, pred_map in zip(gt_maps, pred_maps):
        gt_map = np.asarray(gt_map)
        pred_map = np.asarray(pred_map)
        if gt_map.shape != pred_map.shape:
            raise ValueError(
                f"GT and prediction shapes do not match: {gt_map.shape} vs {pred_map.shape}"
            )
        eval_mask = ~np.isin(gt_map, list(ignore_labels))
        y_true_parts.append(gt_map[eval_mask])
        y_pred_parts.append(pred_map[eval_mask])

    y_true = np.concatenate(y_true_parts)
    y_pred = np.concatenate(y_pred_parts)
    cm = confusion_matrix(y_true, y_pred, labels=list(labels))

    eps = 1e-10
    label_to_index = {lbl: i for i, lbl in enumerate(labels)}
    class_indices = [label_to_index[lbl] for lbl in class_labels]

    tp = np.diag(cm)[class_indices]
    fp = cm[:, class_indices].sum(axis=0) - tp
    fn = cm[class_indices, :].sum(axis=1) - tp
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    oa = tp.sum() / (cm.sum() + eps)
    aa = recall.mean()

    row_sum = cm.sum(axis=1)
    col_sum = cm.sum(axis=0)
    pe = np.sum(row_sum * col_sum) / (cm.sum() ** 2 + eps)
    kappa = (oa - pe) / (1.0 - pe + eps)
    class_kappa = classwise_kappa_from_confusion_matrix(cm, labels, class_labels)

    return {
        "labels": list(labels),
        "class_labels": list(class_labels),
        "class_names": [CLASS_NAMES[lbl] for lbl in class_labels],
        "confusion_matrix": cm,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "iou": iou,
        "pixel_accuracy_OA": float(oa),
        "average_accuracy_AA": float(aa),
        "kappa": float(kappa),
        "classwise_kappa": class_kappa,
        "average_class_kappa": float(np.mean(list(class_kappa.values()))),
    }

def save_mask_png(mask: np.ndarray, path: Path) -> None:
    """Save a colour-coded class mask with a colourbar."""
    visualize_mask(mask, figsize=(12, 8))
    fig = plt.gcf()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def save_mask_png_without_colorbar(mask: np.ndarray, path: Path) -> None:
    """Save a colour-coded class mask without a colourbar."""
    fig = plt.figure(figsize=(12, 8))
    visualize_mask2(mask, figsize=(12, 8), alpha=1.0)
    plt.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def save_prediction_maps(
    maps_dir: Path,
    test_names: Sequence[str],
    pixel_maps: Sequence[np.ndarray],
    semantic_maps: Sequence[np.ndarray],
    instance_maps: Sequence[np.ndarray],
) -> None:
    """Save pixel-wise, semantic-majority, and instance-majority maps."""
    maps_dir.mkdir(parents=True, exist_ok=True)
    for sample_name, pixel_map, semantic_map, instance_map in zip(
        test_names, pixel_maps, semantic_maps, instance_maps
    ):
        sample_dir = maps_dir / sample_name
        sample_dir.mkdir(parents=True, exist_ok=True)
        for map_name, arr in (
            ("pixel_wise", pixel_map),
            ("semantic_majority", semantic_map),
            ("instance_majority", instance_map),
        ):
            np.save(sample_dir / f"{map_name}.npy", arr)
            save_mask_png(arr, sample_dir / f"{map_name}.png")
            save_mask_png_without_colorbar(arr, sample_dir / f"{map_name}_no_colorbar.png")

# ---------------------------------------------------------------------------
# Data loading and preparation
# ---------------------------------------------------------------------------

def load_prepared_data(dataset_root: Path) -> Dict[str, Any]:
    """
    Load and prepare EBS-only cubes using the notebook split.

    Returns a dict with training cubes, validation patches, test cubes,
    auxiliary masks, sample names, and a class-weights tensor.
    """
    records = load_ebs_dataset(
        dataset_root=dataset_root,
        classes=["Styrene", "PP", "PA", "PC", "mix"],
        load_fenix=True,
        load_fx50=True,
        load_dtree=True,
        load_semantic_mask=True,
        load_instance=True,
        load_rgb=True,
    )

    print(f"Loaded records: {len(records)}")
    fenix_cubes, _, _ = extract_sensor_cubes(records, sensor="FENIX")
    fx50_cubes, _, _ = extract_sensor_cubes(records, sensor="FX50")
    gt_maps = [r["dTree_mask"] for r in records]
    semantic_masks = [r["semantic_mask"].astype(bool) for r in records]
    instance_masks = [r["instances"] for r in records]
    sample_names = [r["sample_name"] for r in records]

    class_slices = {
        "Styrene": slice(0, 2),
        "PP":      slice(2, 5),
        "PA":      slice(5, 8),
        "PC":      slice(8, 11),
        "mix":     slice(11, None),
    }

    by_class: Dict[str, Any] = {}
    for class_name, sl in class_slices.items():
        fenix_norm = [snv_normalize(c) for c in fenix_cubes[sl]]
        fx50_norm = [snv_normalize(c) for c in fx50_cubes[sl]]
        by_class[class_name] = {
            "cubes":     concat_hsi_lists(select_roi_bands(fenix_norm, "FENIX"),
                                          select_roi_bands(fx50_norm, "FX50")),
            "gt":        gt_maps[sl],
            "semantic":  semantic_masks[sl],
            "instances": instance_masks[sl],
            "names":     sample_names[sl],
        }
        print(f"  {class_name}: {len(by_class[class_name]['cubes'])} cubes")

    # Same train/test split as KNN.py
    train_cubes = (
        [by_class["Styrene"]["cubes"][1]]
        + by_class["PA"]["cubes"][1:]
        + by_class["PC"]["cubes"][1:]
        + by_class["PP"]["cubes"][1:]
        + [by_class["mix"]["cubes"][1]]
    )
    train_gt = (
        [by_class["Styrene"]["gt"][1]]
        + by_class["PA"]["gt"][1:]
        + by_class["PC"]["gt"][1:]
        + by_class["PP"]["gt"][1:]
        + [by_class["mix"]["gt"][1]]
    )

    test_cubes = [
        by_class["Styrene"]["cubes"][0],
        by_class["PA"]["cubes"][0],
        by_class["PC"]["cubes"][0],
        by_class["PP"]["cubes"][0],
        by_class["mix"]["cubes"][0],
    ]
    test_gt = [
        by_class["Styrene"]["gt"][0],
        by_class["PA"]["gt"][0],
        by_class["PC"]["gt"][0],
        by_class["PP"]["gt"][0],
        by_class["mix"]["gt"][0],
    ]
    test_semantic = [
        by_class["Styrene"]["semantic"][0],
        by_class["PA"]["semantic"][0],
        by_class["PC"]["semantic"][0],
        by_class["PP"]["semantic"][0],
        by_class["mix"]["semantic"][0],
    ]
    test_instances = [
        by_class["Styrene"]["instances"][0],
        by_class["PA"]["instances"][0],
        by_class["PC"]["instances"][0],
        by_class["PP"]["instances"][0],
        by_class["mix"]["instances"][0],
    ]
    test_names = [
        by_class["Styrene"]["names"][0],
        by_class["PA"]["names"][0],
        by_class["PC"]["names"][0],
        by_class["PP"]["names"][0],
        by_class["mix"]["names"][0],
    ]

    # Validation patches: centre column (2 & 5) of each test cube
    _, validation_patches, _, validation_patches_gt = patch_ebs_cubes_6_with_gt(
        test_cubes, test_gt
    )
    print(f"Validation patches: {len(validation_patches)}")

    class_weights_tensor = calculate_class_weights_tensor(train_gt)
    print(f"Class weights tensor: {class_weights_tensor}")
    print(f"Train cubes: {len(train_cubes)}, Test cubes: {len(test_cubes)}")

    return {
        "train_cubes":         train_cubes,
        "train_gt":            train_gt,
        "validation_patches":  validation_patches,
        "validation_patches_gt": validation_patches_gt,
        "test_cubes":          test_cubes,
        "test_gt":             test_gt,
        "test_semantic":       test_semantic,
        "test_instances":      test_instances,
        "test_names":          test_names,
        "class_weights_tensor": class_weights_tensor,
    }

# ---------------------------------------------------------------------------
# Experiment grid
# ---------------------------------------------------------------------------

def make_experiments(max_experiments: int) -> List[Dict[str, Any]]:
    """Sample up to *max_experiments* combinations from the ViT parameter grid."""
    all_combos = list(itertools.product(BAND_PATCH_OPTIONS, VIT_DEPTH_OPTIONS))
    n = min(max_experiments, len(all_combos))
    indices = np.linspace(0, len(all_combos) - 1, n, dtype=int)
    experiments = []
    for exp_number, idx in enumerate(indices, start=1):
        band_patch, vit_depth = all_combos[idx]
        experiments.append(
            {
                "exp_number":  exp_number,
                "patch_size":  DEFAULT_PATCH_SIZE,
                "band_patch":  int(band_patch),
                "vit_depth":   int(vit_depth),
                "vit_dim":     VIT_DIM,
                "vit_heads":   VIT_HEADS,
                "vit_mlp_dim": VIT_MLP_DIM,
                "vit_dropout": VIT_DROPOUT,
                "vit_mode":    VIT_MODE,
            }
        )
    return experiments

# ---------------------------------------------------------------------------
# Single experiment: train + pixel-wise test evaluation
# ---------------------------------------------------------------------------

def run_experiment(
    exp: Dict[str, Any],
    output_root: Path,
    prepared: Dict[str, Any],
    device: torch.device,
    num_epochs: int,
    patience: int,
    batch_size: int,
) -> Dict[str, Any]:
    """Train one ViT configuration and evaluate pixel-wise on test cubes."""
    exp_number = exp["exp_number"]
    patch_size = exp["patch_size"]
    band_patch = exp["band_patch"]  # always 1 for ViT
    vit_depth = exp["vit_depth"]

    exp_dir = output_root / f"ViT_exp_{exp_number}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    model_path = exp_dir / "vit_model.pth"

    print(f"\nBuilding datasets for experiment {exp_number} "
          f"(patch_size={patch_size}, band_patch={band_patch}, vit_depth={vit_depth})")

    train_dataset = HSIDataset(
        prepared["train_cubes"],
        prepared["train_gt"],
        patch_size=patch_size,
        times=1,
        ignore_classes=[BACKGROUND_LABEL],
        band_patch=band_patch,
        augmentations=None,
    )
    val_dataset = HSIDataset(
        prepared["validation_patches"],
        prepared["validation_patches_gt"],
        patch_size=patch_size,
        times=1,
        ignore_classes=[BACKGROUND_LABEL],
        band_patch=band_patch,
        augmentations=None,
    )
    print(
        f"  Train patches: {len(train_dataset)}, "
        f"Val patches: {len(val_dataset)}"
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    model = build_model(
        patch_size=patch_size,
        band_patch=band_patch,
        num_classes=len(CLASS_LABELS),
        vit_depth=vit_depth,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params:,}")

    best_val_loss = train_model(
        model=model,
        label_train_loader=train_loader,
        label_val_loader=val_loader,
        model_path=model_path,
        device=device,
        num_epochs=num_epochs,
        patience=patience,
        class_weights_tensor=prepared["class_weights_tensor"],
    )

    # Pixel-wise evaluation on test cubes with best checkpoint
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("  Running pixel-wise inference on test cubes …")
    pixel_maps = predict_maps(
        model=model,
        cubes=prepared["test_cubes"],
        gt_maps=prepared["test_gt"],
        device=device,
        patch_size=patch_size,
        band_patch=band_patch,
        batch_size=batch_size,
    )
    pixel_metrics = evaluate_maps(prepared["test_gt"], pixel_maps)
    ranking_score = pixel_metrics["average_class_kappa"]

    result = {
        "experiment":        exp,
        "model_path":        str(model_path),
        "best_val_loss":     float(best_val_loss),
        "pixel_wise_metrics": pixel_metrics,
        "ranking_score":     ranking_score,
    }
    (exp_dir / "params.json").write_text(
        json.dumps(json_ready(exp), indent=2), encoding="utf-8"
    )
    (exp_dir / "results.json").write_text(
        json.dumps(json_ready(result), indent=2), encoding="utf-8"
    )
    return result

# ---------------------------------------------------------------------------
# Best model: full map evaluation + save outputs
# ---------------------------------------------------------------------------

def run_best_model_outputs(
    best: Dict[str, Any],
    prepared: Dict[str, Any],
    output_root: Path,
    device: torch.device,
    batch_size: int,
) -> Dict[str, Any]:
    """Apply best ViT model to maps, run majority voting, evaluate, and save outputs."""
    best_dir = output_root / "best_model"
    best_dir.mkdir(parents=True, exist_ok=True)

    exp = best["experiment"]
    patch_size = exp["patch_size"]
    band_patch = exp["band_patch"]  # always 1 for ViT

    model = build_model(
        patch_size=patch_size,
        band_patch=band_patch,
        num_classes=len(CLASS_LABELS),
        vit_depth=exp["vit_depth"],
    )
    model.load_state_dict(torch.load(best["model_path"], map_location=device))
    print("Running full map inference on test cubes …")

    pixel_maps = predict_maps(
        model=model,
        cubes=prepared["test_cubes"],
        gt_maps=prepared["test_gt"],
        device=device,
        patch_size=patch_size,
        band_patch=band_patch,
        batch_size=batch_size,
    )

    semantic_maps = [
        majority_vote_semantic_objects(
            pred_map, semantic_mask, ignore_labels=(BACKGROUND_LABEL,)
        )
        for pred_map, semantic_mask in zip(pixel_maps, prepared["test_semantic"])
    ]
    instance_maps = [
        majority_vote_sam_instances(
            pred_map,
            instances,
            ignore_labels=(BACKGROUND_LABEL,),
            overwrite_overlaps=False,
        )
        for pred_map, instances in zip(pixel_maps, prepared["test_instances"])
    ]

    gt_semantic_maps = [
        majority_vote_semantic_objects(
            gt_map, semantic_mask, ignore_labels=(BACKGROUND_LABEL,)
        )
        for gt_map, semantic_mask in zip(prepared["test_gt"], prepared["test_semantic"])
    ]
    gt_instance_maps = [
        majority_vote_sam_instances(
            gt_map,
            instances,
            ignore_labels=(BACKGROUND_LABEL,),
            overwrite_overlaps=False,
        )
        for gt_map, instances in zip(prepared["test_gt"], prepared["test_instances"])
    ]

    evaluations = {
        "pixel_wise":        evaluate_maps(prepared["test_gt"], pixel_maps),
        "semantic_majority": evaluate_maps(gt_semantic_maps, semantic_maps),
        "instance_majority": evaluate_maps(gt_instance_maps, instance_maps),
    }

    best_model_path = best_dir / "vit_best_model.pth"
    shutil.copy2(best["model_path"], best_model_path)

    best_summary = {
        "selected_experiment":               best["experiment"],
        "selected_model_path":               best["model_path"],
        "ranking_score_average_class_kappa": best["ranking_score"],
        "evaluations":                       evaluations,
    }
    (best_dir / "best_results.json").write_text(
        json.dumps(json_ready(best_summary), indent=2), encoding="utf-8"
    )
    save_prediction_maps(
        best_dir / "prediction_maps",
        prepared["test_names"],
        pixel_maps,
        semantic_maps,
        instance_maps,
    )
    return best_summary

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ViT patch-wise grid search for EBS-only data."
    )
    parser.add_argument("--dataset-root",    type=Path, default=DATASET_ROOT)
    parser.add_argument("--output-root",     type=Path, default=None)
    parser.add_argument("--max-experiments", type=int,  default=DEFAULT_EXPERIMENTS)
    parser.add_argument("--epochs",          type=int,  default=DEFAULT_EPOCHS)
    parser.add_argument("--patience",        type=int,  default=DEFAULT_PATIENCE)
    parser.add_argument("--batch-size",      type=int,  default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--seed",            type=int,  default=0)
    return parser.parse_args()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset_root = args.dataset_root.expanduser().resolve()
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else dataset_root / "trained_models" / "ViT_patch"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Dataset root : {dataset_root}")
    print(f"Output root  : {output_root}")

    prepared = load_prepared_data(dataset_root)
    experiments = make_experiments(args.max_experiments)
    print(f"\nRunning {len(experiments)} ViT experiment(s)")

    all_results: List[Dict[str, Any]] = []
    for exp in experiments:
        print(
            f"\n[{exp['exp_number']}/{len(experiments)}] "
            f"patch_size={exp['patch_size']}, "
            f"band_patch={exp['band_patch']}, "
            f"vit_depth={exp['vit_depth']}"
        )
        result = run_experiment(
            exp=exp,
            output_root=output_root,
            prepared=prepared,
            device=device,
            num_epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
        )
        all_results.append(result)
        print(f"  average class kappa: {result['ranking_score']:.4f}")

    all_results_sorted = sorted(
        all_results, key=lambda r: r["ranking_score"], reverse=True
    )
    best = all_results_sorted[0]
    (output_root / "all_experiment_results.json").write_text(
        json.dumps(json_ready(all_results_sorted), indent=2), encoding="utf-8"
    )

    best_summary = run_best_model_outputs(
        best=best,
        prepared=prepared,
        output_root=output_root,
        device=device,
        batch_size=args.batch_size,
    )

    print("\n" + "=" * 80)
    print("Best ViT experiment")
    print(json.dumps(json_ready(best_summary["selected_experiment"]), indent=2))
    print(
        f"Average class kappa: "
        f"{best_summary['ranking_score_average_class_kappa']:.4f}"
    )
    print(f"Saved outputs to: {output_root / 'best_model'}")


if __name__ == "__main__":
    main()
