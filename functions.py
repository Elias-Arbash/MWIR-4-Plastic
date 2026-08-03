import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from copy import deepcopy

from matplotlib.colors import ListedColormap, BoundaryNorm

def visualize_mask(Mask, figsize=(16, 10)): # Slightly wider to fit the colorbar
    class_names = ['Background', 'Styrene', 'PA', 'PC', 'PP']
    colors = ['#000000', '#FF0000', '#0000FF', '#FFFF00', '#FF00FF']

    cmap = ListedColormap(colors)
    # Define boundaries: [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    bounds = np.arange(len(class_names) + 1) - 0.5
    norm = BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=figsize)
    img = ax.imshow(Mask, cmap=cmap, norm=norm, interpolation='none')
    
    # --- Color bar implementation ---
    # 1. Create the color bar
    cbar = fig.colorbar(img, ax=ax, ticks=np.arange(len(class_names)))
    
    # 2. Set the labels (the class names)
    cbar.ax.set_yticklabels(class_names)
    
    # 3. Optional: Clean up visual appearance
    cbar.set_label('Polymer Classes', rotation=270, labelpad=15)
    
    plt.axis('off')
    plt.show()

def visualize_mask2(Mask, figsize=(16, 10),alpha=0.5): # For projection on top of the image
    class_names = ['Background', 'Styrene', 'PA', 'PC', 'PP']
    colors = ['#000000', '#FF0000', '#0000FF', '#FFFF00', '#FF00FF']

    cmap = ListedColormap(colors)
    # Define boundaries: [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    bounds = np.arange(len(class_names) + 1) - 0.5
    norm = BoundaryNorm(bounds, cmap.N)

    plt.imshow(Mask, cmap=cmap, norm=norm, interpolation='none',alpha = alpha)

def majority_vote_semantic_objects(gt_mask, semantic_mask, ignore_labels=(0,)):
    """
    Convert a noisy pixel-wise ground truth mask into an object-wise mask.

    Each object from semantic_mask gets the majority class label from gt_mask.
    Pixels outside semantic_mask stay 0.

    If semantic_mask is binary/bool, connected components are treated as objects.
    If semantic_mask is already a labeled mask, each positive label is treated as one object.
    """
    gt_mask = np.asarray(gt_mask)
    semantic_mask = np.asarray(semantic_mask)

    if gt_mask.shape != semantic_mask.shape:
        raise ValueError(
            f"gt_mask and semantic_mask must have the same shape, got "
            f"{gt_mask.shape} and {semantic_mask.shape}"
        )

    result = np.zeros_like(gt_mask)

    if semantic_mask.dtype == bool or np.array_equal(np.unique(semantic_mask), [False, True]):
        from scipy import ndimage

        object_mask, num_objects = ndimage.label(semantic_mask)
        object_ids = range(1, num_objects + 1)
    else:
        object_mask = semantic_mask
        object_ids = np.unique(object_mask)
        object_ids = object_ids[object_ids != 0]

    ignore_labels = set(ignore_labels or [])

    for object_id in object_ids:
        pixels = object_mask == object_id
        labels = gt_mask[pixels]

        if ignore_labels:
            labels = labels[~np.isin(labels, list(ignore_labels))]

        if labels.size == 0:
            majority_label = 0
        else:
            labels, counts = np.unique(labels, return_counts=True)
            majority_label = labels[np.argmax(counts)]

        result[pixels] = majority_label

    return result

def majority_vote_sam_instances(gt_mask, instances, ignore_labels=(0,), overwrite_overlaps=True):
    """
    Convert a noisy pixel-wise ground truth mask into an object-wise mask using SAM instances.

    For each SAM instance, the dominant class inside instance["segmentation"] is assigned
    to that whole instance. Pixels outside all SAM instances stay 0.

    Parameters
    ----------
    gt_mask : array-like, shape (H, W)
        Pixel-wise noisy ground truth mask.
    instances : list of dict
        Decompressed SAM predictions. Each dict must contain a boolean "segmentation" mask.
    ignore_labels : tuple/list/set
        Labels ignored during voting. By default, background label 0 is ignored.
    overwrite_overlaps : bool
        SAM instances can overlap. If True, smaller/later instances can overwrite previous
        labels in overlapping pixels. If False, already-labeled pixels are preserved.
    """
    gt_mask = np.asarray(gt_mask)
    result = np.zeros_like(gt_mask)
    ignore_labels = set(ignore_labels or [])

    # Process larger SAM masks first so overlap behavior is deterministic.
    sorted_instances = sorted(
        instances,
        key=lambda item: item.get("area", np.asarray(item["segmentation"]).sum()),
        reverse=True,
    )

    for instance in sorted_instances:
        segmentation = np.asarray(instance["segmentation"]).astype(bool)

        if segmentation.shape != gt_mask.shape:
            raise ValueError(
                f"Each SAM segmentation must have shape {gt_mask.shape}, got {segmentation.shape}"
            )

        labels = gt_mask[segmentation]

        if ignore_labels:
            labels = labels[~np.isin(labels, list(ignore_labels))]

        if labels.size == 0:
            majority_label = 0
        else:
            labels, counts = np.unique(labels, return_counts=True)
            majority_label = labels[np.argmax(counts)]

        if overwrite_overlaps:
            result[segmentation] = majority_label
        else:
            result[segmentation & (result == 0)] = majority_label

    return result

def snv_normalize(hsi_cube):
    """
    Apply Standard Normal Variate (SNV) normalization.
    Normalizes each pixel spectrum independently (mean=0, std=1).
    Ignores NaN pixels (masked background).

    Args:
        hsi_cube: numpy array (H, W, B)

    Returns:
        SNV-normalized cube (H, W, B)
    """
    H, W, B = hsi_cube.shape
    normalized = np.full_like(hsi_cube, np.nan)

    # reshape to (num_pixels, bands)
    reshaped = hsi_cube.reshape(-1, B)

    # detect valid pixels (not all NaN)
    valid_mask = ~np.isnan(reshaped).all(axis=1)
    valid_pixels = reshaped[valid_mask]

    # compute per-spectrum mean and std ignoring NaNs
    means = np.nanmean(valid_pixels, axis=1, keepdims=True)
    stds = np.nanstd(valid_pixels, axis=1, keepdims=True)

    # avoid division by zero
    stds[stds == 0] = 1.0

    normalized_pixels = (valid_pixels - means) / stds

    # place back
    reshaped_normalized = normalized.reshape(-1, B)
    reshaped_normalized[valid_mask] = normalized_pixels

    return reshaped_normalized.reshape(H, W, B)

def select_roi_bands(hsi_list, sensor_type):
    """Keep only fingerprint bands. sensor_type: 'FENIX' or 'FX50'."""
    sensor_key = sensor_type.upper()
    slice_map = {"FENIX": FENIX_SLICES, "FX50": FX50_SLICES}
    if sensor_key not in slice_map:
        raise ValueError(f"Unknown sensor_type '{sensor_type}'.")
    roi_cubes = []
    for cube in hsi_list:
        np_cube = np.asarray(cube)
        if np_cube.ndim != 3:
            raise ValueError(f"Expected 3D cube; got shape {np_cube.shape}.")
        roi_cube = np.concatenate(
            [np_cube[:, :, band_slice] for band_slice in slice_map[sensor_key]], axis=2
        )
        roi_cubes.append(roi_cube)
    return roi_cubes

def concat_hsi_lists(list_a, list_b):
    """
    Concatenate two equal-length lists of HSI cubes along the band axis.

    Each pair of cubes (same index) is concatenated on the last axis, so the
    resulting cube keeps the spatial dimensions and appends the spectra.

    Args:
        list_a: list of numpy-like arrays shaped (H, W, Bands_a)
        list_b: list of numpy-like arrays shaped (H, W, Bands_b)

    Returns:
        List of numpy arrays shaped (H, W, Bands_a + Bands_b).
    """
    if len(list_a) != len(list_b):
        raise ValueError(f"Lists must have the same length, got {len(list_a)} and {len(list_b)}.")

    out = []
    for idx, (cube_a, cube_b) in enumerate(zip(list_a, list_b)):
        np_a = np.asarray(cube_a)
        np_b = np.asarray(cube_b)

        if np_a.shape[:2] != np_b.shape[:2]:
            raise ValueError(
                f"Cube pair at index {idx} has different spatial shapes: {np_a.shape[:2]} vs {np_b.shape[:2]}"
            )

        concatenated = np.concatenate([np_a, np_b], axis=2)
        out.append(concatenated)

    return out

def concat_hsi_lists(list_a, list_b):
    """
    Concatenate two equal-length lists of HSI cubes along the band axis.

    Each pair of cubes (same index) is concatenated on the last axis, so the
    resulting cube keeps the spatial dimensions and appends the spectra.

    Args:
        list_a: list of numpy-like arrays shaped (H, W, Bands_a)
        list_b: list of numpy-like arrays shaped (H, W, Bands_b)

    Returns:
        List of numpy arrays shaped (H, W, Bands_a + Bands_b).
    """
    if len(list_a) != len(list_b):
        raise ValueError(f"Lists must have the same length, got {len(list_a)} and {len(list_b)}.")

    out = []
    for idx, (cube_a, cube_b) in enumerate(zip(list_a, list_b)):
        np_a = np.asarray(cube_a)
        np_b = np.asarray(cube_b)

        if np_a.shape[:2] != np_b.shape[:2]:
            raise ValueError(
                f"Cube pair at index {idx} has different spatial shapes: {np_a.shape[:2]} vs {np_b.shape[:2]}"
            )

        concatenated = np.concatenate([np_a, np_b], axis=2)
        out.append(concatenated)

    return out



def plot_non_background_statistics(train_masks, val_masks):
    """
    Compute and plot the total pixel counts for each non-background class in the training
    and validation ground truth masks, ignoring background (label 0).
    
    Parameters:
        train_masks (list of np.ndarray): List of training mask arrays of shape (H, W)
        val_masks (list of np.ndarray): List of validation mask arrays of shape (H, W)
        
    The class values (1 to 4) correspond to:
        1: 'Styrene'
        2: 'PA'
        3: 'PC'
        4: 'PP'
    """
    # Define the mapping from class value to label (ignore 0/background)
    class_labels = {
        1: 'Styrene',
        2: 'PA',
        3: 'PC',
        4: 'PP'
    }
    
    # Initialize dictionaries to store pixel counts for each class of interest
    train_counts = {cls: 0 for cls in class_labels}
    val_counts = {cls: 0 for cls in class_labels}
    
    # Count pixels for each class of interest in the training masks
    for mask in train_masks:
        unique_vals, counts = np.unique(mask, return_counts=True)
        for val, count in zip(unique_vals, counts):
            if val in class_labels:
                train_counts[val] += count

    # Count pixels for each class of interest in the validation masks
    for mask in val_masks:
        unique_vals, counts = np.unique(mask, return_counts=True)
        for val, count in zip(unique_vals, counts):
            if val in class_labels:
                val_counts[val] += count

    # Prepare data for plotting (exclude background)
    classes = sorted(class_labels.keys())
    labels = [class_labels[c] for c in classes]
    train_values = [train_counts[c] for c in classes]
    val_values = [val_counts[c] for c in classes]
    
    x = np.arange(len(classes))  # positions for the groups
    width = 0.35  # width of the bars

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create side-by-side bars for training and validation counts
    rects1 = ax.bar(x - width/2, train_values, width, label='Training')
    rects2 = ax.bar(x + width/2, val_values, width, label='Testing')

    # Add labels, title, and custom x-axis tick labels
    ax.set_ylabel('Pixel Count')
    ax.set_title('Non-Background Class Pixel Counts in Training vs. Validation Masks')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    # Function to add labels above each bar
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    plt.show()
