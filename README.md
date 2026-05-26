# Rodent Dataset Export

This repository now includes a reusable exporter that turns the sample images and DeepLabCut outputs into a trainable PyTorch dataset input.

## What it generates

- One master CSV with image references, labels, split assignment, clip metadata, and one `x/y` coordinate pair per body part
- Split-specific CSV files for `train`, `val`, and `test`
- Optional rejects CSV when a positive sample cannot be matched to a valid DeepLabCut annotation

Negative samples are included in the same CSV with `label=0` and empty keypoint columns. Splits are assigned by clip by default to reduce leakage across nearby frames.

## Generate annotations

Use the project runtime so the declared dependencies are available:

```bash
uv run python main.py --output-dir generated --output-stem rodent_annotations
```

Useful options:

```bash
uv run python main.py --help
```

You can also group splits by recording instead of clip:

```bash
uv run python main.py --group-by recording_id
```

## Use the Dataset class

```python
from rodent_dataset import RodentKeypointDataset, build_eval_transform, build_train_transform

train_dataset = RodentKeypointDataset(
	csv_path="generated/rodent_annotations.csv",
	image_root=".",
	paired_transform=build_train_transform(image_size=(224, 224)),
)

eval_dataset = RodentKeypointDataset(
	csv_path="generated/rodent_annotations_val.csv",
	image_root=".",
	paired_transform=build_eval_transform(image_size=(224, 224)),
)

sample = train_dataset[0]
image = sample["image"]
label = sample["label"]
keypoints = sample["keypoints"]
keypoints_valid = sample["keypoints_valid"]
metadata = sample["metadata"]
```

`keypoints` has shape `(39, 2)` and `keypoints_valid` marks which body parts are present for a sample.

`paired_transform` applies coordinated image and keypoint updates for training-time resizing, flipping, and normalization. The separate `transform` hook remains available for image-only post-processing if you still need it.
