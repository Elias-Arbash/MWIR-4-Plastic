"""
Utilities for loading the EBS12_EBS22 polymer dataset.

Dataset structure (expected):
    EBS12_EBS22/
        Styrene/<sample_dir>/
        PP/<sample_dir>/
        PA/<sample_dir>/
        PC/<sample_dir>/
        mix/<sample_dir>/

Each sample directory can include:
    - FENIX.hdr/.dat (ENVI cube)
    - FX50.hdr/.dat (ENVI cube)
    - RGB.* (.hdr/.dat ENVI cube for RGB image)
    - dTree_*.npy (ground-truth mask)
    - semantic_mask.png (binary foreground mask)
    - instance.pkl (SAM instance predictions, compressed with gzip+pickle)
"""

import gzip
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import spectral
from PIL import Image


VALID_CLASSES = ("Styrene", "PP", "PA", "PC", "mix")


class SamplePaths:
    def __init__(
        self,
        class_name,
        sample_name,
        sample_dir,
        fenix_hdr,
        fx50_hdr,
        dtree_npy,
        semantic_mask_png,
        instance_pkl,
        rgb_hdr=None,
    ):
        self.class_name = class_name
        self.sample_name = sample_name
        self.sample_dir = sample_dir
        self.fenix_hdr = fenix_hdr
        self.fx50_hdr = fx50_hdr
        self.dtree_npy = dtree_npy
        self.semantic_mask_png = semantic_mask_png
        self.instance_pkl = instance_pkl
        self.rgb_hdr = rgb_hdr


def _first_match(directory: Path, pattern: str) -> Optional[Path]:
    matches = sorted(directory.glob(pattern))
    return matches[0] if matches else None


def discover_ebs_samples(
    dataset_root: Union[str, Path],
    classes: Optional[Sequence[str]] = None,
) -> List[SamplePaths]:
    """
    Discover sample folders and expected files under EBS12_EBS22.
    """
    root = Path(dataset_root).expanduser().resolve()
    selected = set(classes) if classes is not None else set(VALID_CLASSES)

    unknown = selected.difference(VALID_CLASSES)
    if unknown:
        raise ValueError(f"Unknown class names: {sorted(unknown)}")

    samples: List[SamplePaths] = []
    for class_name in VALID_CLASSES:
        if class_name not in selected:
            continue

        class_dir = root / class_name
        if not class_dir.is_dir():
            continue

        for sample_dir in sorted([p for p in class_dir.iterdir() if p.is_dir()]):
            samples.append(
                SamplePaths(
                    class_name=class_name,
                    sample_name=sample_dir.name,
                    sample_dir=sample_dir,
                    fenix_hdr=_first_match(sample_dir, "FENIX.hdr"),
                    fx50_hdr=_first_match(sample_dir, "FX50.hdr"),
                    dtree_npy=_first_match(sample_dir, "dTree*.npy"),
                    semantic_mask_png=_first_match(sample_dir, "semantic_mask2.png"),
                    instance_pkl=_first_match(sample_dir, "instance2.pkl"),
                    rgb_hdr=_first_match(sample_dir, "RGB.hdr"),
                )
            )

    return samples


def load_and_decompress_instance(filename: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Load and decompress SAM instance predictions from gzip+pickle.
    Falls back to plain pickle if the file is not gzipped.
    """
    file_path = Path(filename)

    try:
        with gzip.open(file_path, "rb") as f:
            compressed_data = pickle.load(f)
    except OSError:
        with open(file_path, "rb") as f:
            compressed_data = pickle.load(f)

    decompressed_data = []
    for item in compressed_data:
        segmentation_unpacked = np.unpackbits(item["segmentation"])
        segmentation_unpacked = segmentation_unpacked[: item["original_length"]]
        segmentation_unpacked = segmentation_unpacked.reshape(item["shape"]).astype(bool)

        decompressed_data.append(
            {
                "segmentation": segmentation_unpacked,
                "area": item["area"],
                "bbox": item["bbox"],
                "predicted_iou": item["predicted_iou"],
                "point_coords": item["point_coords"],
                "stability_score": item["stability_score"],
                "crop_box": item["crop_box"],
            }
        )
    return decompressed_data


def load_envi_cube(hdr_path: Union[str, Path], as_numpy: bool = True) -> np.ndarray:
    """
    Load ENVI cube from .hdr path using spectral.
    """
    cube = spectral.envi.open(str(hdr_path)).load()
    return np.asarray(cube) if as_numpy else cube


def downsample_by_skipping(array: np.ndarray, factor: int = 6) -> np.ndarray:
    """
    Downsample a 3D image array by skipping every `factor-1` pixels along the first two axes.
    """
    if array.ndim == 3:
        return array[::factor, ::factor, ...]
    else:
        return array[::factor, ::factor]

def load_sample(
    sample: SamplePaths,
    load_fenix: bool = True,
    load_fx50: bool = True,
    load_dtree: bool = True,
    load_semantic_mask: bool = True,
    load_instance: bool = True,
    load_rgb: bool = True,
) -> Dict[str, Any]:
    """
    Load one sample entry into memory.
    """
    record: Dict[str, Any] = {
        "class_name": sample.class_name,
        "sample_name": sample.sample_name,
        "sample_dir": str(sample.sample_dir),
        "fenix_cube": None,
        "fx50_cube": None,
        "dTree_mask": None,
        "semantic_mask": None,
        "instances": None,
        "rgb_cube": None,
    }

    if load_fenix and sample.fenix_hdr and sample.fenix_hdr.exists():
        record["fenix_cube"] = load_envi_cube(sample.fenix_hdr)
    if load_fx50 and sample.fx50_hdr and sample.fx50_hdr.exists():
        record["fx50_cube"] = load_envi_cube(sample.fx50_hdr)
    if load_rgb and sample.rgb_hdr and sample.rgb_hdr.exists():
        # Load full-resolution RGB, then downsample spatially by a factor of 6
        rgb_cube = load_envi_cube(sample.rgb_hdr)
        #rgb_cube_down = downsample_by_skipping(rgb_cube, factor=6)
        record["rgb_cube"] = rgb_cube
    if load_dtree and sample.dtree_npy and sample.dtree_npy.exists():
        record["dTree_mask"] = np.load(sample.dtree_npy)
    if load_semantic_mask and sample.semantic_mask_png and sample.semantic_mask_png.exists():
        semantic = np.array(Image.open(sample.semantic_mask_png))
        record["semantic_mask"] = semantic > 0
    if load_instance and sample.instance_pkl and sample.instance_pkl.exists():
        record["instances"] = load_and_decompress_instance(sample.instance_pkl)

    return record


def load_ebs_dataset(
    dataset_root: Union[str, Path],
    classes: Optional[Sequence[str]] = None,
    load_fenix: bool = True,
    load_fx50: bool = True,
    load_dtree: bool = True,
    load_semantic_mask: bool = True,
    load_instance: bool = False,
    load_rgb: bool = True,
) -> List[Dict[str, Any]]:
    """
    Discover then load all selected samples.
    """
    samples = discover_ebs_samples(dataset_root=dataset_root, classes=classes)
    return [
        load_sample(
            sample,
            load_fenix=load_fenix,
            load_fx50=load_fx50,
            load_dtree=load_dtree,
            load_semantic_mask=load_semantic_mask,
            load_instance=load_instance,
            load_rgb=load_rgb,
        )
        for sample in samples
    ]


def extract_sensor_cubes(
    dataset_records: Sequence[Dict[str, Any]],
    sensor: str,
    drop_missing: bool = True,
) -> Tuple[List[np.ndarray], List[str], List[str]]:
    """
    Extract cubes for one sensor from loaded records.

    Returns:
        cubes, class_labels, sample_names
    """
    sensor_norm = sensor.strip().upper()
    if sensor_norm not in {"FENIX", "FX50", "RGB"}:
        raise ValueError("sensor must be 'FENIX', 'FX50', or 'RGB'")

    if sensor_norm == "FENIX":
        key = "fenix_cube"
    elif sensor_norm == "FX50":
        key = "fx50_cube"
    elif sensor_norm == "RGB":
        key = "rgb_cube"
    else:
        raise ValueError("Invalid sensor")

    cubes: List[np.ndarray] = []
    labels: List[str] = []
    sample_names: List[str] = []

    for rec in dataset_records:
        cube = rec.get(key)
        if cube is None and drop_missing:
            continue
        cubes.append(cube)
        labels.append(rec["class_name"])
        sample_names.append(rec["sample_name"])

    return cubes, labels, sample_names
