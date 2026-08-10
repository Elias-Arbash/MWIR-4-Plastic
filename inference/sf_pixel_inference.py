#!/usr/bin/env python3
"""
SpectralFormer pixel-wise INFERENCE for the EBS-only polymer dataset.

This is the inference-only counterpart of ../sf_pixel.py. It loads a trained
SpectralFormer (ViT, mode="CAF") checkpoint (produced by sf_pixel.py) and
runs pixel-wise prediction (patch_size=1) on the EBS-only test cubes,
followed by semantic/SAM-instance majority voting and evaluation against
ground truth. Model architecture, data loading/prepping, prediction,
evaluation, and visualisation code are kept unchanged; the training loop,
grid search, and validation-split machinery have been removed.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import matplotlib
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402

# Allow importing the shared repository modules (models.py, functions.py,
# ebs_dataset_loader.py) from the parent directory without copying them.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
OUTPUT_ROOT = DATASET_ROOT / "trained_models" / "SF_pixel"

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
# SpectralFormer (pixel) inference defaults. Architecture params must match
# the checkpoint produced by sf_pixel.py (grid search always used
# band_patch=7, vit_depth=5; patch_size is always 1 for the pixel version).
# ---------------------------------------------------------------------------

DEFAULT_BATCH_SIZE = 512
DEFAULT_BAND_PATCH = 7
DEFAULT_VIT_DEPTH = 5

VIT_DIM = 64
VIT_HEADS = 4
VIT_MLP_DIM = 8
VIT_DROPOUT = 0.1
VIT_EMB_DROPOUT = 0.1
VIT_MODE = "CAF"

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
# Group Spectral Embedding (GSE). Still needed by predict_maps below
# (unchanged from sf_pixel.py).
# ---------------------------------------------------------------------------

def gain_neighborhood_band(
    x_train: np.ndarray,
    band: int,
    band_patch: int,
    patch: int,
) -> np.ndarray:
    """
    Create band-neighbourhood context for each spatial token.

    Input shape : (N, patch, patch, band)  [after expand_dims for single samples]
    Output shape: (N, patch*patch*band_patch, band)
    """
    nn_ = band_patch // 2
    pp = (patch * patch) // 2
    x_reshape = x_train.reshape(x_train.shape[0], patch * patch, band)
    x_band = np.zeros(
        (x_train.shape[0], patch * patch * band_patch, band), dtype=float
    )

    x_band[:, nn_ * patch * patch:(nn_ + 1) * patch * patch, :] = x_reshape

    for i in range(nn_):
        if pp > 0:
            x_band[:, i * patch * patch:(i + 1) * patch * patch, :i + 1] = x_reshape[:, :, band - i - 1:]
            x_band[:, i * patch * patch:(i + 1) * patch * patch, i + 1:] = x_reshape[:, :, :band - i - 1]
        else:
            x_band[:, i:i + 1, :nn_ - i] = x_reshape[:, 0:1, band - nn_ + i:]
            x_band[:, i:i + 1, nn_ - i:] = x_reshape[:, 0:1, :band - nn_ + i]

    for i in range(nn_):
        if pp > 0:
            x_band[:, (nn_ + i + 1) * patch * patch:(nn_ + i + 2) * patch * patch, :band - i - 1] = x_reshape[:, :, i + 1:]
            x_band[:, (nn_ + i + 1) * patch * patch:(nn_ + i + 2) * patch * patch, band - i - 1:] = x_reshape[:, :, :i + 1]
        else:
            x_band[:, nn_ + 1 + i:nn_ + 2 + i, band - i - 1:] = x_reshape[:, 0:1, :i + 1]
            x_band[:, nn_ + 1 + i:nn_ + 2 + i, :band - i - 1] = x_reshape[:, 0:1, i + 1:]

    return x_band

# ---------------------------------------------------------------------------
# Model building
# ---------------------------------------------------------------------------

def build_model(
    patch_size: int,
    band_patch: int,
    num_classes: int,
    vit_depth: int,
) -> nn.Module:
    """Instantiate a SpectralFormer (ViT) model and move it to CUDA."""
    model = ViT(
        image_size=1,  # always pixel-wise
        near_band=band_patch,
        num_patches=TOTAL_BANDS,
        num_classes=num_classes,
        dim=VIT_DIM,
        depth=vit_depth,
        heads=VIT_HEADS,
        mlp_dim=VIT_MLP_DIM,
        dropout=VIT_DROPOUT,
        emb_dropout=VIT_EMB_DROPOUT,
        mode=VIT_MODE,
    )
    return model.cuda()

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
    GSE is applied (gain_neighborhood_band), and the model predicts the class.
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

        pad = 0  # patch_size = 1, no padding needed for pixel
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
                padded[r:r + 1, c:c + 1, :]  # (1,1,band)
                for r, c in zip(batch_rows, batch_cols)
            ])  # (B, 1, 1, bands)

            band = patches.shape[-1]
            patches = gain_neighborhood_band(patches, band, band_patch, 1)
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
# Evaluation
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
# Data loading and preparation.
# TRIMMED from sf_pixel.py: training cubes/GT, the 6-patch validation split,
# and the class-weights tensor are training-only and have been removed. Only
# the test cubes (and their GT/semantic/instance masks) needed for inference
# are kept, using the same class split as the training script.
# ---------------------------------------------------------------------------

def load_prepared_data(dataset_root: Path) -> Dict[str, Any]:
    """
    Load and prepare EBS-only test cubes using the same split as sf_pixel.py.

    Returns a dict with test cubes, auxiliary masks, and sample names.
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

    # Same test split as KNN.py / sf_pixel.py (index 0 of each class == test cube)
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

    print(f"Test cubes: {len(test_cubes)}")

    return {
        "test_cubes":     test_cubes,
        "test_gt":        test_gt,
        "test_semantic":  test_semantic,
        "test_instances": test_instances,
        "test_names":     test_names,
    }

# ---------------------------------------------------------------------------
# Inference run: apply the loaded SpectralFormer model to the test cubes, run
# majority voting, evaluate, and save outputs.
# (Renamed/simplified from sf_pixel.py's run_best_model_outputs: no
# grid-search "best experiment" selection anymore -- the model is loaded
# directly from --model-path.)
# ---------------------------------------------------------------------------

def run_inference(
    model: nn.Module,
    band_patch: int,
    model_path: Path,
    prepared: Dict[str, Any],
    output_root: Path,
    device: torch.device,
    batch_size: int,
) -> Dict[str, Any]:
    """Apply the SpectralFormer model to test cubes, run majority voting, evaluate, and save outputs."""
    output_root.mkdir(parents=True, exist_ok=True)
    patch_size = 1  # always pixel-wise

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

    copied_model_path = output_root / "sf_pixel_model.pth"
    shutil.copy2(model_path, copied_model_path)

    summary = {
        "model_path":  str(model_path),
        "patch_size":  patch_size,
        "band_patch":  band_patch,
        "evaluations": evaluations,
    }
    (output_root / "inference_results.json").write_text(
        json.dumps(json_ready(summary), indent=2), encoding="utf-8"
    )
    save_prediction_maps(
        output_root / "prediction_maps",
        prepared["test_names"],
        pixel_maps,
        semantic_maps,
        instance_maps,
    )
    return summary

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SpectralFormer pixel-wise INFERENCE for EBS-only data."
    )
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--output-root",  type=Path, default=None)
    parser.add_argument(
        "--model-path", type=Path, required=True,
        help="Path to a trained SpectralFormer checkpoint (.pth) produced by sf_pixel.py.",
    )
    parser.add_argument(
        "--band-patch", type=int, default=DEFAULT_BAND_PATCH,
        help="Must match the band_patch used to train the checkpoint (default: 7).",
    )
    parser.add_argument(
        "--vit-depth", type=int, default=DEFAULT_VIT_DEPTH,
        help="Must match the vit_depth used to train the checkpoint (default: 5).",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--seed",       type=int, default=0)
    return parser.parse_args()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset_root = args.dataset_root.expanduser().resolve()
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else dataset_root / "trained_models" / "SF_pixel" / "inference"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Dataset root : {dataset_root}")
    print(f"Output root  : {output_root}")
    print(f"Model path   : {args.model_path}")

    prepared = load_prepared_data(dataset_root)

    model = build_model(
        patch_size=1,
        band_patch=args.band_patch,
        num_classes=len(CLASS_LABELS),
        vit_depth=args.vit_depth,
    )
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    summary = run_inference(
        model=model,
        band_patch=args.band_patch,
        model_path=args.model_path,
        prepared=prepared,
        output_root=output_root,
        device=device,
        batch_size=args.batch_size,
    )

    print("\n" + "=" * 80)
    print("SpectralFormer (pixel) inference complete")
    print(json.dumps(
        json_ready({"band_patch": args.band_patch, "vit_depth": args.vit_depth}),
        indent=2,
    ))
    print(
        f"Pixel-wise average class kappa: "
        f"{summary['evaluations']['pixel_wise']['average_class_kappa']:.4f}"
    )
    print(f"Saved outputs to: {output_root}")


if __name__ == "__main__":
    main()
