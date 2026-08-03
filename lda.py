#!/usr/bin/env python3
"""
LDA experiments for the EBS-only polymer dataset.

This script follows the workflow from lda.ipynb:
load EBS12/EBS22, SNV-normalize FENIX and FX50, keep fingerprint bands,
concatenate sensors, train/test split by sample, train LDA models, and evaluate
pixel-wise plus semantic/SAM-instance majority-voted prediction maps.
"""

import argparse
import itertools
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import joblib
import matplotlib
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ebs_dataset_loader import extract_sensor_cubes, load_ebs_dataset
from functions import (
    concat_hsi_lists,
    majority_vote_sam_instances,
    majority_vote_semantic_objects,
    snv_normalize,
    visualize_mask,
    visualize_mask2,
)


DATASET_ROOT = Path("/home/arbash44/coding/hsi/polymers/EBS_only/EBS12_EBS22")
OUTPUT_ROOT = DATASET_ROOT / "trained_models" / "LDA"

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
    slice(32, 38),  # 2976 - 3018 nm
    slice(75, 86),  # 3336 - 3420 nm
    slice(86, 97),  # 3428 - 3512 nm
    slice(114, 119),  # 3663 - 3696 nm
    slice(209, 212),  # 4458 - 4475 nm
]

SOLVER_OPTIONS = ["svd", "lsqr", "eigen"]
SHRINKAGE_OPTIONS = [None, "auto"]
N_COMPONENTS_OPTIONS = [None, 2, 3]
DEFAULT_EXPERIMENTS = 5


def json_ready(value: Any) -> Any:
    """Convert numpy-heavy results into JSON-serializable Python values."""
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def select_roi_bands(hsi_list: Sequence[np.ndarray], sensor_type: str) -> List[np.ndarray]:
    """Keep only the fingerprint bands used in the notebook."""
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
                [np_cube[:, :, band_slice] for band_slice in slice_map[sensor_key]],
                axis=2,
            )
        )
    return roi_cubes


def extract_labeled_pixels(
    cubes: Sequence[np.ndarray],
    masks: Sequence[np.ndarray],
    ignore_labels: Iterable[int] = (BACKGROUND_LABEL,),
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """Extract finite spectra and labels from non-background pixels."""
    x_parts = []
    y_parts = []
    valid_masks = []

    for cube, mask in zip(cubes, masks):
        cube = np.asarray(cube)
        mask = np.asarray(mask)
        if cube.shape[:2] != mask.shape:
            raise ValueError(f"Cube and mask shapes do not match: {cube.shape[:2]} vs {mask.shape}")

        valid_mask = ~np.isin(mask, list(ignore_labels))
        spectra = cube[valid_mask]
        labels = mask[valid_mask]
        finite_spectra = np.isfinite(spectra).all(axis=1)

        final_valid_mask = np.zeros(mask.shape, dtype=bool)
        final_valid_mask[valid_mask] = finite_spectra

        x_parts.append(spectra[finite_spectra])
        y_parts.append(labels[finite_spectra])
        valid_masks.append(final_valid_mask)

    if not x_parts:
        raise ValueError("No labeled pixels were extracted.")
    return np.vstack(x_parts), np.concatenate(y_parts), valid_masks


def predict_maps(
    model: Pipeline,
    cubes: Sequence[np.ndarray],
    valid_masks: Sequence[np.ndarray],
) -> List[np.ndarray]:
    """Predict valid pixels and reconstruct full-size prediction maps."""
    pred_maps = []
    for index, (cube, valid_mask) in enumerate(zip(cubes, valid_masks), start=1):
        print(f"Predicting map {index}/{len(cubes)}")
        cube = np.asarray(cube)
        spectra = cube[valid_mask]
        pred_labels = model.predict(spectra)

        pred_map = np.zeros(cube.shape[:2], dtype=pred_labels.dtype)
        pred_map[valid_mask] = pred_labels
        pred_maps.append(pred_map)
    return pred_maps


def classwise_kappa_from_confusion_matrix(
    cm: np.ndarray,
    labels: Sequence[int] = EVALUATION_LABELS,
    class_labels: Sequence[int] = CLASS_LABELS,
) -> Dict[int, float]:
    """Compute one-vs-rest Cohen's kappa for each foreground class."""
    total = cm.sum()
    eps = 1e-10
    label_to_index = {label: index for index, label in enumerate(labels)}
    kappas: Dict[int, float] = {}

    for label in class_labels:
        idx = label_to_index[label]
        tp = cm[idx, idx]
        fp = cm[:, idx].sum() - tp
        fn = cm[idx, :].sum() - tp
        tn = total - tp - fp - fn

        observed = (tp + tn) / (total + eps)
        expected = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / ((total * total) + eps)
        kappas[int(label)] = float((observed - expected) / (1.0 - expected + eps))

    return kappas


def evaluate_maps(
    gt_maps: Sequence[np.ndarray],
    pred_maps: Sequence[np.ndarray],
    labels: Sequence[int] = EVALUATION_LABELS,
    class_labels: Sequence[int] = CLASS_LABELS,
    ignore_labels: Iterable[int] = (BACKGROUND_LABEL,),
) -> Dict[str, Any]:
    """Evaluate maps on non-background GT pixels while counting background predictions as errors."""
    y_true_parts = []
    y_pred_parts = []

    for gt_map, pred_map in zip(gt_maps, pred_maps):
        gt_map = np.asarray(gt_map)
        pred_map = np.asarray(pred_map)
        if gt_map.shape != pred_map.shape:
            raise ValueError(f"GT and prediction shapes do not match: {gt_map.shape} vs {pred_map.shape}")

        eval_mask = ~np.isin(gt_map, list(ignore_labels))
        y_true_parts.append(gt_map[eval_mask])
        y_pred_parts.append(pred_map[eval_mask])

    y_true = np.concatenate(y_true_parts)
    y_pred = np.concatenate(y_pred_parts)
    cm = confusion_matrix(y_true, y_pred, labels=list(labels))

    eps = 1e-10
    label_to_index = {label: index for index, label in enumerate(labels)}
    class_indices = [label_to_index[label] for label in class_labels]

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
    pe = np.sum(row_sum * col_sum) / ((cm.sum() ** 2) + eps)
    kappa = (oa - pe) / (1.0 - pe + eps)
    class_kappa = classwise_kappa_from_confusion_matrix(cm, labels, class_labels)

    return {
        "labels": list(labels),
        "class_labels": list(class_labels),
        "class_names": [CLASS_NAMES[label] for label in class_labels],
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


def build_lda_pipeline(
    solver: str,
    shrinkage: Any,
    n_components: Any,
) -> Pipeline:
    """Create a StandardScaler + LDA model for one experiment."""
    lda_kwargs: Dict[str, Any] = {
        "solver": solver,
        "n_components": n_components,
    }
    if shrinkage is not None:
        lda_kwargs["shrinkage"] = shrinkage

    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lda", LinearDiscriminantAnalysis(**lda_kwargs)),
        ]
    )


def lda_parameter_grid() -> List[Tuple[str, Any, Any]]:
    """Build valid LDA hyperparameter combinations."""
    combinations = []
    for solver, shrinkage, n_components in itertools.product(
        SOLVER_OPTIONS,
        SHRINKAGE_OPTIONS,
        N_COMPONENTS_OPTIONS,
    ):
        if solver == "svd" and shrinkage is not None:
            continue
        combinations.append((solver, shrinkage, n_components))
    return combinations


def make_experiments(max_experiments: int) -> List[Dict[str, Any]]:
    """Sample up to max_experiments combinations from the LDA parameter grid."""
    all_combinations = lda_parameter_grid()
    n_experiments = min(max_experiments, len(all_combinations))
    selected_indices = np.linspace(0, len(all_combinations) - 1, n_experiments, dtype=int)
    experiments = []

    for exp_number, combo_index in enumerate(selected_indices, start=1):
        solver, shrinkage, n_components = all_combinations[combo_index]
        experiments.append(
            {
                "exp_number": exp_number,
                "solver": solver,
                "shrinkage": shrinkage,
                "n_components": n_components,
            }
        )
    return experiments


def load_prepared_data(dataset_root: Path) -> Dict[str, Any]:
    """Load and prepare EBS-only cubes using the notebook split."""
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
    gt_maps = [record["dTree_mask"] for record in records]
    semantic_masks = [record["semantic_mask"].astype(bool) for record in records]
    instance_masks = [record["instances"] for record in records]
    sample_names = [record["sample_name"] for record in records]

    class_slices = {
        "Styrene": slice(0, 2),
        "PP": slice(2, 5),
        "PA": slice(5, 8),
        "PC": slice(8, 11),
        "mix": slice(11, None),
    }

    by_class = {}
    for class_name, class_slice in class_slices.items():
        fenix_norm = [snv_normalize(cube) for cube in fenix_cubes[class_slice]]
        fx50_norm = [snv_normalize(cube) for cube in fx50_cubes[class_slice]]
        fenix_roi = select_roi_bands(fenix_norm, "FENIX")
        fx50_roi = select_roi_bands(fx50_norm, "FX50")

        by_class[class_name] = {
            "cubes": concat_hsi_lists(fenix_roi, fx50_roi),
            "gt": gt_maps[class_slice],
            "semantic": semantic_masks[class_slice],
            "instances": instance_masks[class_slice],
            "names": sample_names[class_slice],
        }
        print(f"{class_name}: {len(by_class[class_name]['cubes'])} cubes")

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

    train_x, train_y, _ = extract_labeled_pixels(train_cubes, train_gt)
    test_x, test_y, test_valid_masks = extract_labeled_pixels(test_cubes, test_gt)
    print(f"train_X: {train_x.shape}, train_y: {train_y.shape}")
    print(f"test_X: {test_x.shape}, test_y: {test_y.shape}")
    print("Train labels:", dict(zip(*np.unique(train_y, return_counts=True))))
    print("Test labels:", dict(zip(*np.unique(test_y, return_counts=True))))

    return {
        "train_x": train_x,
        "train_y": train_y,
        "test_x": test_x,
        "test_y": test_y,
        "test_cubes": test_cubes,
        "test_gt": test_gt,
        "test_valid_masks": test_valid_masks,
        "test_semantic": test_semantic,
        "test_instances": test_instances,
        "test_names": test_names,
    }


def save_mask_png(mask: np.ndarray, path: Path) -> None:
    """Save a color-coded class mask with a colorbar."""
    visualize_mask(mask, figsize=(12, 8))
    fig = plt.gcf()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_mask_png_without_colorbar(mask: np.ndarray, path: Path) -> None:
    """Save a color-coded class mask without a colorbar."""
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

        map_groups = {
            "pixel_wise": pixel_map,
            "semantic_majority": semantic_map,
            "instance_majority": instance_map,
        }
        for map_name, map_array in map_groups.items():
            np.save(sample_dir / f"{map_name}.npy", map_array)
            save_mask_png(map_array, sample_dir / f"{map_name}.png")
            save_mask_png_without_colorbar(map_array, sample_dir / f"{map_name}_no_colorbar.png")


def run_experiment(
    exp: Dict[str, Any],
    output_root: Path,
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
) -> Dict[str, Any]:
    """Train, evaluate, and save one LDA experiment."""
    exp_number = exp["exp_number"]
    exp_dir = output_root / f"LDA_exp_{exp_number}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    model = build_lda_pipeline(
        solver=exp["solver"],
        shrinkage=exp["shrinkage"],
        n_components=exp["n_components"],
    )
    model.fit(train_x, train_y)
    y_pred = model.predict(test_x)

    pixel_metrics = evaluate_maps([test_y], [y_pred], labels=CLASS_LABELS, class_labels=CLASS_LABELS, ignore_labels=())
    model_path = exp_dir / "lda_model.pkl"
    result_path = exp_dir / "results.json"
    params_path = exp_dir / "params.json"

    joblib.dump(model, model_path)
    result = {
        "experiment": exp,
        "model_path": str(model_path),
        "pixel_wise_metrics": pixel_metrics,
        "ranking_score": pixel_metrics["average_class_kappa"],
    }

    params_path.write_text(json.dumps(json_ready(exp), indent=2), encoding="utf-8")
    result_path.write_text(json.dumps(json_ready(result), indent=2), encoding="utf-8")
    return result


def run_best_model_outputs(best: Dict[str, Any], prepared: Dict[str, Any], output_root: Path) -> Dict[str, Any]:
    """Apply best LDA model to maps, run majority voting, evaluate, and save outputs."""
    best_dir = output_root / "best_model"
    best_dir.mkdir(parents=True, exist_ok=True)
    model = joblib.load(best["model_path"])

    pixel_maps = predict_maps(model, prepared["test_cubes"], prepared["test_valid_masks"])
    semantic_maps = [
        majority_vote_semantic_objects(pred_map, semantic_mask, ignore_labels=(BACKGROUND_LABEL,))
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
        majority_vote_semantic_objects(pred_map, semantic_mask, ignore_labels=(BACKGROUND_LABEL,))
        for pred_map, semantic_mask in zip(prepared["test_gt"], prepared["test_semantic"])
    ]
    gt_instance_maps = [
        majority_vote_sam_instances(
            pred_map,
            instances,
            ignore_labels=(BACKGROUND_LABEL,),
            overwrite_overlaps=False,
        )
        for pred_map, instances in zip(prepared["test_gt"], prepared["test_instances"])
    ]

    evaluations = {
        "pixel_wise": evaluate_maps(prepared["test_gt"], pixel_maps),
        "semantic_majority": evaluate_maps(gt_semantic_maps, semantic_maps),
        "instance_majority": evaluate_maps(gt_instance_maps, instance_maps),
    }
    best_summary = {
        "selected_experiment": best["experiment"],
        "selected_model_path": best["model_path"],
        "ranking_score_average_class_kappa": best["ranking_score"],
        "evaluations": evaluations,
    }

    best_model_path = best_dir / "lda_best_model.pkl"
    shutil.copy2(best["model_path"], best_model_path)
    (best_dir / "best_results.json").write_text(
        json.dumps(json_ready(best_summary), indent=2),
        encoding="utf-8",
    )
    save_prediction_maps(
        best_dir / "prediction_maps",
        prepared["test_names"],
        pixel_maps,
        semantic_maps,
        instance_maps,
    )
    return best_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LDA grid search for EBS-only data.")
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--max-experiments", type=int, default=DEFAULT_EXPERIMENTS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else dataset_root / "trained_models" / "LDA"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Dataset root: {dataset_root}")
    print(f"Output root: {output_root}")
    prepared = load_prepared_data(dataset_root)
    experiments = make_experiments(args.max_experiments)
    print(f"Running {len(experiments)} LDA experiments")

    all_results = []
    for exp in experiments:
        print(
            f"[{exp['exp_number']}/{len(experiments)}] "
            f"solver={exp['solver']}, shrinkage={exp['shrinkage']}, "
            f"n_components={exp['n_components']}"
        )
        result = run_experiment(
            exp=exp,
            output_root=output_root,
            train_x=prepared["train_x"],
            train_y=prepared["train_y"],
            test_x=prepared["test_x"],
            test_y=prepared["test_y"],
        )
        all_results.append(result)
        print(f"  average class kappa: {result['ranking_score']:.4f}")

    all_results_sorted = sorted(all_results, key=lambda item: item["ranking_score"], reverse=True)
    best = all_results_sorted[0]
    (output_root / "all_experiment_results.json").write_text(
        json.dumps(json_ready(all_results_sorted), indent=2),
        encoding="utf-8",
    )

    best_summary = run_best_model_outputs(best, prepared, output_root)
    print("=" * 80)
    print("Best LDA experiment")
    print(json.dumps(json_ready(best_summary["selected_experiment"]), indent=2))
    print(f"Average class kappa: {best_summary['ranking_score_average_class_kappa']:.4f}")
    print(f"Saved outputs to: {output_root / 'best_model'}")


if __name__ == "__main__":
    main()
