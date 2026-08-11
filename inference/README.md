# Inference Scripts

Inference-only versions of `1DCNN.py`, `ViT.py`, `sf_pixel.py`, and `sf_patch.py`.
Each loads a trained `.pth` checkpoint and runs prediction + evaluation on the
EBS-only test cubes — no training, no grid search.

## Requirements

Same environment as the training scripts (PyTorch + CUDA GPU, scikit-learn,
matplotlib, etc.). A trained checkpoint from the matching training script.

## Usage

```bash
python inference/1DCNN_inference.py    --model-path /path/to/1DCNN_model.pth
python inference/ViT_inference.py      --model-path /path/to/vit_model.pth
python inference/sf_pixel_inference.py --model-path /path/to/sf_pixel_model.pth
python inference/sf_patch_inference.py --model-path /path/to/sf_patch_model.pth
```

### Common arguments

| Argument           | Default                                   | Notes                                   |
|--------------------|--------------------------------------------|------------------------------------------|
| `--model-path`     | **required**                               | Path to the trained `.pth` checkpoint    |
| `--dataset-root`   | `EBS_only/EBS12_EBS22` (hardcoded default) | Root of the dataset                      |
| `--output-root`    | `<dataset_root>/trained_models/<model>/inference` | Where results are saved         |
| `--batch-size`     | `512`                                      | Inference batch size                     |
| `--seed`           | `0`                                        | Random seed                              |

### Architecture arguments (must match the checkpoint's training config)

| Script                  | Argument         | Default |
|--------------------------|------------------|---------|
| `1DCNN_inference.py`     | `--band-patch`   | `1`     |
| `ViT_inference.py`       | `--patch-size`   | `9`     |
|                           | `--band-patch`   | `1`     |
|                           | `--vit-depth`    | `5`     |
| `sf_pixel_inference.py`  | `--band-patch`   | `7`     |
|                           | `--vit-depth`    | `5`     |
| `sf_patch_inference.py`  | `--patch-size`   | `9`     |
|                           | `--band-patch`   | `7`     |
|                           | `--vit-depth`    | `5`     |

If your checkpoint was trained with non-default hyperparameters, pass the
matching values — otherwise `load_state_dict` will fail with a shape mismatch.

## What it does

1. Loads the test-split cubes (same class split as training) with their GT,
   semantic, and SAM-instance masks.
2. Builds the model architecture and loads the checkpoint weights.
3. Runs full-map prediction, then semantic-majority and instance-majority
   voting on top of it.
4. Evaluates all three (pixel-wise / semantic-majority / instance-majority)
   against ground truth (kappa, OA, AA, precision/recall/F1, IoU).
5. Saves everything to `--output-root`.

## Output

```
<output_root>/
├── inference_results.json      # evaluation metrics for all 3 variants
├── <model>_model.pth           # copy of the checkpoint used
└── prediction_maps/
    └── <sample_name>/
        ├── pixel_wise.npy / .png / _no_colorbar.png
        ├── semantic_majority.npy / .png / _no_colorbar.png
        └── instance_majority.npy / .png / _no_colorbar.png
```
