# Rodent Dataset Export

This repository includes a reusable exporter that turns the sample images and DeepLabCut outputs into trainable annotation files, plus a pose-distance trainer that now reads those generated CSVs directly.

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

## Train `swin2_model.py` from generated CSVs

The trainer now accepts the CSV files generated above directly. It filters out `label=0` rows and any rows missing the keypoints required by the Pattern-3 distance target.

Create a one-row template CSV with the same columns as the exported annotations. A simple starting point is to copy the header plus one positive row from the training split:

```bash
head -n 2 generated/rodent_annotations_train.csv > generated/template.csv
```

Train:

```bash
uv run python swin2_model.py \
	--mode train \
	--train-data generated/rodent_annotations_train.csv \
	--val-data generated/rodent_annotations_val.csv \
	--template-data generated/template.csv \
	--outdir runs/run1
```

Use `--backbone` to switch from the default ResNet18 baseline to a Swin V2 backbone from `timm`:

```bash
uv run python swin2_model.py \
	--mode train \
	--train-data generated/rodent_annotations_train.csv \
	--val-data generated/rodent_annotations_val.csv \
	--template-data generated/template.csv \
	--outdir runs/swin2 \
	--backbone swinv2_cr_tiny_384
```

For the balanced Swin2 profile, you can leave the default tuning in place or make it explicit with `--backbone-preset balanced`:

```bash
uv run python swin2_model.py \
	--mode train \
	--train-data generated/rodent_annotations_train.csv \
	--val-data generated/rodent_annotations_val.csv \
	--template-data generated/template.csv \
	--outdir runs/swin2-balanced \
	--backbone swinv2_cr_tiny_384 \
	--backbone-preset balanced
```

For the current best accuracy-oriented Swin2 preset, add `--backbone-preset accuracy`. The balanced default remains the recommended starting point because it uses less memory and is faster:

```bash
uv run python swin2_model.py \
	--mode train \
	--train-data generated/rodent_annotations_train.csv \
	--val-data generated/rodent_annotations_val.csv \
	--template-data generated/template.csv \
	--outdir runs/swin2-accuracy \
	--backbone swinv2_cr_tiny_384 \
	--backbone-preset accuracy
```

Optional model controls:

```bash
uv run python swin2_model.py --help
```

Useful flags are `--backbone`, `--backbone-preset`, `--[no-]pretrained`, `--backbone-feature-index`, and `--decoder-channels`.
When you select a non-ResNet backbone such as Swin V2, the trainer now uses a lower default learning rate automatically unless you pass `--lr` yourself.

For the balanced Swin2 path at `--batch_size 48`, the trainer now also auto-enables a cosine LR schedule with a short warmup and a retuned base LR. That keeps the large-batch run usable without forcing extra flags:

```bash
uv run python swin2_model.py \
	--mode train \
	--train-data generated/rodent_annotations_train.csv \
	--val-data generated/rodent_annotations_val.csv \
	--template-data generated/template.csv \
	--outdir runs/swin2-balanced-bs48 \
	--backbone swinv2_cr_tiny_384 \
	--backbone-preset balanced \
	--epochs 200 \
	--early-stop-patience 10 \
	--batch_size 48
```

Early stopping monitors validation loss and requires `--val-data`. The trainer still writes `best.pt` for the best validation-loss checkpoint even when the run stops before the configured epoch count.

If you want to override the large-batch defaults, pass the tuning flags explicitly. Use `--scheduler none` to disable the automatic scheduler, or set your own scheduler/warmup values:

```bash
uv run python swin2_model.py \
	--mode train \
	--train-data generated/rodent_annotations_train.csv \
	--val-data generated/rodent_annotations_val.csv \
	--template-data generated/template.csv \
	--outdir runs/swin2-balanced-bs48-manual \
	--backbone swinv2_cr_tiny_384 \
	--backbone-preset balanced \
	--batch_size 48 \
	--lr 3e-5 \
	--scheduler none
```

When you compare different batch sizes, compare them by optimizer updates or schedule length rather than raw epoch count. A `batch_size 48` run does far fewer updates per epoch than a small-batch run.

TensorBoard logs are now written during training by default under `<outdir>/tensorboard`. You can point TensorBoard at that directory with:

```bash
uv run tensorboard --logdir runs/run1/tensorboard
```

Use `--tensorboard-dir` if you want the event files somewhere else. Each epoch now logs both scalar metrics and a small qualitative snapshot for train and val: the input frame, target/predicted heatmap summaries, decoded keypoint views, and simple overlay views. Scalar logs now also include `kp_error`, a decoded keypoint error summary in resized-image pixels.

You can also resume training from a saved checkpoint:

```bash
uv run python swin2_model.py \
	--mode train \
	--train-data generated/rodent_annotations_train.csv \
	--val-data generated/rodent_annotations_val.csv \
	--outdir runs/swin2-resumed \
	--resume-checkpoint runs/swin2-accuracy/best.pt
```

Resume mode restores model and optimizer state from the checkpoint, keeps epoch numbering continuous, and reuses the checkpoint's TensorBoard log directory metadata when it still points inside the selected output directory.

Evaluate the held-out test split with a saved checkpoint:

```bash
uv run python swin2_model.py \
	--mode test \
	--test-data generated/rodent_annotations_test.csv \
	--checkpoint runs/run1/best.pt
```

Test-mode output now prints checkpoint provenance before the final metrics so you can see which training run, epoch, backbone, and TensorBoard directory produced the evaluated checkpoint.

Legacy JSON inputs are still accepted, but the generated CSV workflow is now the primary path. Checkpoints remain self-contained; test-mode loading reconstructs the backbone from saved config and then loads the checkpoint weights.

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
