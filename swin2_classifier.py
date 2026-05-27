#!/usr/bin/env python3
"""
Train and evaluate a binary classifier for generated rodent sample annotations.

The classifier uses the same timm Swin V2 feature-extractor pattern as
``swin2_model.py`` but keeps classification separate from pose/keypoint training.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import ffmpeg
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from PIL import Image
from sklearn import metrics as sk_metrics
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from swin2_model import BACKBONE_PRESETS, ImageNetInputNormalizer


@dataclass
class ClassifierConfig:
    image_size: int = 384
    backbone: str = "swinv2_cr_tiny_384"
    backbone_preset: str | None = "balanced"
    pretrained: bool = True
    backbone_feature_index: int = 2
    dropout: float = 0.2
    batch_size: int = 32
    lr: float = 1e-5
    weight_decay: float = 1e-4
    epochs: int = 20
    num_workers: int = 4
    class_weight: str = "none"
    freeze_backbone: bool = False
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True)
class FramePrediction:
    source_classified_frame: int
    logit: float
    probability: float
    prediction: str


class RodentClassificationDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        *,
        image_root: str | Path | None = None,
        image_size: int = 384,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.image_root = (
            Path(image_root) if image_root is not None else self.csv_path.parent
        )
        self.image_size = image_size
        self.annotations = pd.read_csv(self.csv_path)
        required_columns = {"image_path", "label"}
        missing = required_columns - set(self.annotations.columns)
        if missing:
            raise ValueError(
                f"{self.csv_path} is missing required columns: {', '.join(sorted(missing))}"
            )

    def __len__(self) -> int:
        return len(self.annotations)

    def _resolve_image_path(self, image_path: str) -> Path:
        path = Path(image_path)
        if path.is_absolute():
            return path
        return self.image_root / path

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.annotations.iloc[index]
        image_path = self._resolve_image_path(str(row["image_path"]))
        image = Image.open(image_path).convert("RGB")
        image = image.resize(
            (self.image_size, self.image_size), resample=Image.Resampling.BILINEAR
        )
        image_tensor = (
            torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        )
        label = float(row["label"])
        if label not in (0.0, 1.0):
            raise ValueError(f"Expected binary label 0/1, got {label!r}")
        return {
            "image": image_tensor,
            "label": torch.tensor(label, dtype=torch.float32),
        }


class SwinBinaryClassifier(nn.Module):
    def __init__(
        self,
        *,
        backbone_name: str,
        pretrained: bool,
        feature_index: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.normalizer = ImageNetInputNormalizer()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(feature_index,),
        )
        self.backbone_out_channels = int(self.backbone.feature_info.channels()[0])  # type: ignore
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(self.backbone_out_channels, 1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(self.normalizer(images))[0]
        if features.ndim != 4:
            raise ValueError(
                f"Expected a 4D feature map from timm backbone, got {features.shape}"
            )
        if (
            features.shape[1] != self.backbone_out_channels
            and features.shape[-1] == self.backbone_out_channels
        ):
            features = features.permute(0, 3, 1, 2).contiguous()
        pooled = self.pool(features).flatten(1)
        logits = self.classifier(self.dropout(pooled))
        return logits.squeeze(-1)


def apply_backbone_preset(cfg: ClassifierConfig, preset: str | None) -> None:
    cfg.backbone_preset = preset
    if preset is None:
        return
    values = BACKBONE_PRESETS[preset]
    cfg.backbone_feature_index = int(values["backbone_feature_index"])


def config_from_args(args: argparse.Namespace) -> ClassifierConfig:
    cfg = ClassifierConfig()
    cfg.image_size = args.image_size
    if args.backbone is not None:
        cfg.backbone = args.backbone
    apply_backbone_preset(cfg, args.backbone_preset)
    cfg.pretrained = args.pretrained
    if args.backbone_feature_index is not None:
        cfg.backbone_feature_index = args.backbone_feature_index
    cfg.dropout = args.dropout
    cfg.batch_size = args.batch_size
    cfg.lr = args.lr
    cfg.weight_decay = args.weight_decay
    cfg.epochs = args.epochs
    cfg.num_workers = args.num_workers
    cfg.class_weight = args.class_weight
    cfg.freeze_backbone = args.freeze_backbone
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    return cfg


def build_classifier_model(cfg: ClassifierConfig) -> SwinBinaryClassifier:
    return SwinBinaryClassifier(
        backbone_name=cfg.backbone,
        pretrained=cfg.pretrained,
        feature_index=cfg.backbone_feature_index,
        dropout=cfg.dropout,
    ).to(cfg.device)


def freeze_backbone(model: SwinBinaryClassifier) -> None:
    for parameter in model.backbone.parameters():
        parameter.requires_grad = False


def extract_compatible_pose_backbone_state(
    pose_model_state: dict[str, torch.Tensor],
    classifier_state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    compatible: dict[str, torch.Tensor] = {}
    for key, tensor in pose_model_state.items():
        if not key.startswith("backbone."):
            continue
        target = classifier_state.get(key)
        if target is None or target.shape != tensor.shape:
            continue
        compatible[key] = tensor
    return compatible


def load_pose_backbone_weights(
    model: SwinBinaryClassifier,
    checkpoint_path: str | Path,
) -> int:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(f"{checkpoint_path} does not look like a pose checkpoint")
    pose_model_state = checkpoint["model"]
    if not isinstance(pose_model_state, dict):
        raise ValueError(f"{checkpoint_path} has no usable model state dict")

    classifier_state = model.state_dict()
    compatible = extract_compatible_pose_backbone_state(
        pose_model_state,
        classifier_state,
    )
    if not compatible:
        raise ValueError(
            f"No compatible backbone weights found in pose checkpoint: {checkpoint_path}"
        )
    classifier_state.update(compatible)
    model.load_state_dict(classifier_state)
    return len(compatible)


def load_pose_checkpoint_config(checkpoint_path: str | Path) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"{checkpoint_path} does not look like a pose checkpoint")
    cfg = checkpoint.get("cfg")
    if not isinstance(cfg, dict):
        raise ValueError(f"{checkpoint_path} is missing checkpoint cfg metadata")
    return cfg


def build_loader(
    csv_path: str | Path,
    cfg: ClassifierConfig,
    *,
    image_root: str | Path | None = None,
    shuffle: bool,
) -> DataLoader:
    dataset = RodentClassificationDataset(
        csv_path,
        image_root=image_root,
        image_size=cfg.image_size,
    )
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device == "cuda"),
    )


def positive_class_weight(dataset: RodentClassificationDataset) -> torch.Tensor:
    labels = dataset.annotations["label"].astype(int)
    positives = int((labels == 1).sum())
    negatives = int((labels == 0).sum())
    if positives == 0 or negatives == 0:
        raise ValueError(
            "--class-weight auto requires both positive and negative labels"
        )
    return torch.tensor([negatives / positives], dtype=torch.float32)


def build_loss(
    cfg: ClassifierConfig,
    train_loader: DataLoader,
) -> nn.Module:
    pos_weight = None
    if cfg.class_weight == "auto":
        dataset = train_loader.dataset
        if not isinstance(dataset, RodentClassificationDataset):
            raise TypeError("Expected RodentClassificationDataset for class weights")
        pos_weight = positive_class_weight(dataset).to(cfg.device)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def default_tensorboard_dir(outdir: str | Path) -> str:
    return str(Path(outdir) / "tensorboard")


def build_tensorboard_writer(
    outdir: str | Path,
    tensorboard_dir: str | Path | None = None,
) -> tuple[SummaryWriter, str]:
    logdir = (
        str(tensorboard_dir)
        if tensorboard_dir is not None
        else default_tensorboard_dir(outdir)
    )
    os.makedirs(logdir, exist_ok=True)
    return SummaryWriter(log_dir=logdir), logdir


def write_metrics_to_tensorboard(
    writer: SummaryWriter,
    epoch: int,
    *,
    split_name: str,
    metrics: dict[str, float | None],
) -> None:
    for metric_name, value in metrics.items():
        if value is None:
            continue
        writer.add_scalar(f"{metric_name}/{split_name}", float(value), epoch)


def write_epoch_to_tensorboard(
    writer: SummaryWriter,
    epoch: int,
    *,
    train_metrics: dict[str, float | None],
    eval_metrics: dict[str, float | None],
    eval_prefix: str,
    lr: float,
) -> None:
    write_metrics_to_tensorboard(
        writer,
        epoch,
        split_name="train",
        metrics=train_metrics,
    )
    if eval_prefix != "train":
        write_metrics_to_tensorboard(
            writer,
            epoch,
            split_name=eval_prefix,
            metrics=eval_metrics,
        )
    writer.add_scalar("optimizer/lr", lr, epoch)


def classification_metrics(
    labels: list[float],
    logits: list[float],
    *,
    loss: float,
) -> dict[str, float | None]:
    y_true = np.asarray(labels, dtype=np.int64)
    y_logits = np.asarray(logits, dtype=np.float64)
    y_prob = 1.0 / (1.0 + np.exp(-y_logits))
    y_pred = (y_prob >= 0.5).astype(np.int64)

    result: dict[str, float | None] = {
        "loss": loss,
        "accuracy": float(sk_metrics.accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(sk_metrics.balanced_accuracy_score(y_true, y_pred)),
        "precision": float(sk_metrics.precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(sk_metrics.recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(sk_metrics.f1_score(y_true, y_pred, zero_division=0)),
        "auroc": None,
    }
    if len(np.unique(y_true)) == 2:
        result["auroc"] = float(sk_metrics.roc_auc_score(y_true, y_prob))
    return result


def format_metrics(prefix: str, metrics: dict[str, float | None]) -> str:
    parts = [f"{prefix} loss={metrics['loss']:.4f}"]
    for name in ("accuracy", "balanced_accuracy", "precision", "recall", "f1"):
        value = metrics[name]
        if value is not None:
            parts.append(f"{name}={value:.4f}")
    auroc = metrics["auroc"]
    parts.append("auroc=n/a" if auroc is None else f"auroc={auroc:.4f}")
    return " ".join(parts)


def train_one_epoch(
    model: SwinBinaryClassifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    cfg: ClassifierConfig,
) -> dict[str, float | None]:
    model.train()
    amp_device = "cuda" if cfg.device == "cuda" else "cpu"
    scaler = torch.GradScaler(amp_device, enabled=(cfg.device == "cuda"))
    total_loss = 0.0
    total_count = 0
    labels: list[float] = []
    logits_out: list[float] = []

    for batch in tqdm(loader, desc="train", leave=False):
        images = batch["image"].to(cfg.device)
        targets = batch["label"].to(cfg.device)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(amp_device, enabled=(cfg.device == "cuda")):
            logits = model(images)
            loss = loss_fn(logits, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = int(targets.numel())
        total_loss += float(loss.item()) * batch_size
        total_count += batch_size
        labels.extend(targets.detach().cpu().tolist())
        logits_out.extend(logits.detach().cpu().tolist())

    return classification_metrics(
        labels,
        logits_out,
        loss=total_loss / max(total_count, 1),
    )


@torch.no_grad()
def evaluate(
    model: SwinBinaryClassifier,
    loader: DataLoader,
    loss_fn: nn.Module,
    cfg: ClassifierConfig,
    *,
    desc: str = "eval",
) -> dict[str, float | None]:
    model.eval()
    total_loss = 0.0
    total_count = 0
    labels: list[float] = []
    logits_out: list[float] = []

    for batch in tqdm(loader, desc=desc, leave=False):
        images = batch["image"].to(cfg.device)
        targets = batch["label"].to(cfg.device)
        logits = model(images)
        loss = loss_fn(logits, targets)

        batch_size = int(targets.numel())
        total_loss += float(loss.item()) * batch_size
        total_count += batch_size
        labels.extend(targets.detach().cpu().tolist())
        logits_out.extend(logits.detach().cpu().tolist())

    return classification_metrics(
        labels,
        logits_out,
        loss=total_loss / max(total_count, 1),
    )


def save_checkpoint(
    path: str | Path,
    *,
    epoch: int,
    cfg: ClassifierConfig,
    model: SwinBinaryClassifier,
    optimizer: torch.optim.Optimizer | None,
    metrics: dict[str, float | None],
    train_run: dict[str, Any],
) -> None:
    checkpoint: dict[str, Any] = {
        "epoch": epoch,
        "cfg": asdict(cfg),
        "model": model.state_dict(),
        "metrics": metrics,
        "train_run": train_run,
    }
    if optimizer is not None:
        checkpoint["optimizer"] = optimizer.state_dict()
    torch.save(checkpoint, path)


def config_from_checkpoint(checkpoint: dict[str, Any]) -> ClassifierConfig:
    raw_cfg = checkpoint.get("cfg")
    if not isinstance(raw_cfg, dict):
        raise ValueError("Classifier checkpoint is missing cfg")
    values = asdict(ClassifierConfig())
    for key, value in raw_cfg.items():
        if key in values:
            values[key] = value
    cfg = ClassifierConfig(**values)
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    return cfg


def load_classifier_checkpoint(
    checkpoint_path: str | Path,
    *,
    batch_size: int | None = None,
    num_workers: int | None = None,
) -> tuple[dict[str, Any], ClassifierConfig, SwinBinaryClassifier]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"{checkpoint_path} is not a classifier checkpoint")
    cfg = config_from_checkpoint(checkpoint)
    if batch_size is not None:
        cfg.batch_size = batch_size
    if num_workers is not None:
        cfg.num_workers = num_workers
    cfg.pretrained = False
    model = build_classifier_model(cfg)
    model_state = checkpoint.get("model")
    if not isinstance(model_state, dict):
        raise ValueError("Classifier checkpoint is missing model state")
    model.load_state_dict(model_state)
    return checkpoint, cfg, model


def sigmoid_probability(logit: float) -> float:
    return 1.0 / (1.0 + math.exp(-float(logit)))


def prediction_label(probability: float, *, threshold: float = 0.5) -> str:
    return "grab" if probability >= threshold else "no-grab"


def prediction_from_logit(
    source_classified_frame: int,
    logit: float,
    *,
    threshold: float = 0.5,
) -> FramePrediction:
    probability = sigmoid_probability(logit)
    return FramePrediction(
        source_classified_frame=source_classified_frame,
        logit=float(logit),
        probability=probability,
        prediction=prediction_label(probability, threshold=threshold),
    )


def format_video_label(frame_number: int, prediction: FramePrediction) -> str:
    return f"frame {frame_number} | {prediction.prediction} p={prediction.probability:.3f}"


def should_classify_frame(frame_number: int, frame_stride: int) -> bool:
    if frame_stride < 1:
        raise ValueError("--frame-stride must be >= 1")
    return frame_number % frame_stride == 0


def held_prediction_for_frame(
    frame_number: int,
    frame_stride: int,
    latest_prediction: FramePrediction | None,
    classify_frame: Callable[[int], FramePrediction],
) -> tuple[bool, FramePrediction]:
    classified_on_frame = should_classify_frame(frame_number, frame_stride)
    if classified_on_frame:
        latest_prediction = classify_frame(frame_number)
    if latest_prediction is None:
        raise ValueError("No classification is available to hold for this frame")
    return classified_on_frame, latest_prediction


def resolve_recording_video_paths(
    annotations: pd.DataFrame,
    video_root: str | Path,
) -> dict[str, Path]:
    if "recording_id" not in annotations.columns:
        raise ValueError("annotation CSV is missing required column: recording_id")

    recording_ids = [
        str(recording_id)
        for recording_id in pd.unique(annotations["recording_id"].dropna())
    ]
    video_root = Path(video_root)
    stem_to_path: dict[str, Path] = {}
    duplicate_stems: dict[str, list[Path]] = {}

    for mp4_path in video_root.rglob("*.mp4"):
        stem = mp4_path.stem
        if stem in stem_to_path:
            duplicate_stems.setdefault(stem, [stem_to_path[stem]]).append(mp4_path)
        else:
            stem_to_path[stem] = mp4_path

    requested_duplicates = {
        recording_id: duplicate_stems[recording_id]
        for recording_id in recording_ids
        if recording_id in duplicate_stems
    }
    if requested_duplicates:
        details = "; ".join(
            f"{recording_id}: {', '.join(str(path) for path in paths)}"
            for recording_id, paths in sorted(requested_duplicates.items())
        )
        raise ValueError(f"Duplicate MP4 stems found for requested recordings: {details}")

    missing = [recording_id for recording_id in recording_ids if recording_id not in stem_to_path]
    if missing:
        raise ValueError(
            "Could not find MP4 files for recording_id values: " + ", ".join(missing)
        )

    return {recording_id: stem_to_path[recording_id] for recording_id in recording_ids}


def frame_to_classifier_tensor(frame_bgr: np.ndarray, image_size: int) -> torch.Tensor:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb).resize(
        (image_size, image_size), resample=Image.Resampling.BILINEAR
    )
    return torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0


@torch.no_grad()
def classify_video_frame(
    model: SwinBinaryClassifier,
    cfg: ClassifierConfig,
    frame_bgr: np.ndarray,
    frame_number: int,
    *,
    threshold: float,
) -> FramePrediction:
    image = frame_to_classifier_tensor(frame_bgr, cfg.image_size).unsqueeze(0).to(cfg.device)
    logit = float(model(image).detach().cpu().reshape(-1)[0].item())
    return prediction_from_logit(frame_number, logit, threshold=threshold)


def draw_prediction_overlay(
    frame_bgr: np.ndarray,
    frame_number: int,
    prediction: FramePrediction,
) -> np.ndarray:
    text = format_video_label(frame_number, prediction)
    output = frame_bgr.copy()
    origin = (24, 42)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.0
    thickness = 2
    cv2.putText(output, text, origin, font, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    color = (40, 220, 40) if prediction.prediction == "grab" else (40, 40, 230)
    cv2.putText(output, text, origin, font, scale, color, thickness, cv2.LINE_AA)
    return output


def frame_output_row(
    *,
    recording_id: str,
    video_path: Path,
    frame_number: int,
    fps: float,
    classified_on_frame: bool,
    prediction: FramePrediction,
) -> dict[str, Any]:
    return {
        "recording_id": recording_id,
        "video_path": str(video_path),
        "frame_number": frame_number,
        "time_sec": frame_number / fps,
        "classified_on_frame": classified_on_frame,
        "source_classified_frame": prediction.source_classified_frame,
        "logit": prediction.logit,
        "probability": prediction.probability,
        "prediction": prediction.prediction,
    }


def build_ffmpeg_writer(
    output_video_path: str | Path,
    *,
    width: int,
    height: int,
    fps: float,
) -> Any:
    return (
        ffmpeg.input(
            "pipe:",
            format="rawvideo",
            pix_fmt="bgr24",
            s=f"{width}x{height}",
            framerate=fps,
        )
        .output(
            str(output_video_path),
            vcodec="libx264",
            pix_fmt="yuv420p",
            r=fps,
            movflags="+faststart",
        )
        .global_args("-loglevel", "error")
        .overwrite_output()
        .run_async(pipe_stdin=True, pipe_stderr=True)
    )


def process_classifier_video(
    *,
    recording_id: str,
    video_path: str | Path,
    output_video_path: str | Path,
    output_csv_path: str | Path,
    model: SwinBinaryClassifier,
    cfg: ClassifierConfig,
    frame_stride: int,
    threshold: float,
) -> int:
    if frame_stride < 1:
        raise ValueError("--frame-stride must be >= 1")

    video_path = Path(video_path)
    output_video_path = Path(output_video_path)
    output_csv_path = Path(output_csv_path)
    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        raise ValueError(f"Could not determine FPS for video: {video_path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if width <= 0 or height <= 0:
        raise ValueError(f"Could not determine frame size for video: {video_path}")

    writer = build_ffmpeg_writer(output_video_path, width=width, height=height, fps=fps)
    rows: list[dict[str, Any]] = []
    latest_prediction: FramePrediction | None = None
    frame_number = 0
    model.eval()

    try:
        progress_total = total_frames if total_frames > 0 else None
        with tqdm(total=progress_total, desc=video_path.stem, leave=False) as progress:
            while True:
                ok, frame_bgr = capture.read()
                if not ok:
                    break

                def classify_current_frame(_frame_number: int) -> FramePrediction:
                    return classify_video_frame(
                        model,
                        cfg,
                        frame_bgr,
                        _frame_number,
                        threshold=threshold,
                    )

                classified_on_frame, latest_prediction = held_prediction_for_frame(
                    frame_number,
                    frame_stride,
                    latest_prediction,
                    classify_current_frame,
                )
                annotated_frame = draw_prediction_overlay(
                    frame_bgr,
                    frame_number,
                    latest_prediction,
                )
                writer.stdin.write(annotated_frame.tobytes())
                rows.append(
                    frame_output_row(
                        recording_id=recording_id,
                        video_path=video_path,
                        frame_number=frame_number,
                        fps=fps,
                        classified_on_frame=classified_on_frame,
                        prediction=latest_prediction,
                    )
                )
                frame_number += 1
                progress.update(1)
    finally:
        capture.release()
        if writer.stdin is not None:
            writer.stdin.close()

    return_code = writer.wait()
    if return_code != 0:
        stderr = ""
        if writer.stderr is not None:
            stderr = writer.stderr.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"ffmpeg failed for {output_video_path}: {stderr}")

    pd.DataFrame(rows).to_csv(output_csv_path, index=False)
    return frame_number


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["train", "test", "classify-videos"],
        default="train",
        help="Train, evaluate, or run classifier video inference",
    )
    parser.add_argument("--train-data", default=None)
    parser.add_argument("--val-data", default=None)
    parser.add_argument("--test-data", default=None)
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--video-root", default=None)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--tensorboard-dir", default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument(
        "--batch-size", "--batch_size", dest="batch_size", type=int, default=32
    )
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--backbone", default=None)
    parser.add_argument(
        "--backbone-preset",
        choices=sorted(BACKBONE_PRESETS),
        default="balanced",
    )
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--backbone-feature-index", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--class-weight", choices=["none", "auto"], default="none")
    return parser


def run_train(args: argparse.Namespace) -> int:
    if not args.train_data:
        raise ValueError("--train-data is required in train mode")
    if not args.outdir:
        raise ValueError("--outdir is required in train mode")

    os.makedirs(args.outdir, exist_ok=True)
    cfg = config_from_args(args)
    if args.init_checkpoint:
        checkpoint_cfg = load_pose_checkpoint_config(args.init_checkpoint)
        checkpoint_backbone = checkpoint_cfg.get("backbone")
        if isinstance(checkpoint_backbone, str):
            cfg.backbone = checkpoint_backbone
        checkpoint_feature_index = checkpoint_cfg.get("backbone_feature_index")
        if isinstance(checkpoint_feature_index, int):
            cfg.backbone_feature_index = checkpoint_feature_index
        cfg.pretrained = False
    model = build_classifier_model(cfg)
    if args.init_checkpoint:
        loaded = load_pose_backbone_weights(model, args.init_checkpoint)
        print(
            f"loaded {loaded} compatible backbone tensors from {args.init_checkpoint}"
        )
    if cfg.freeze_backbone:
        freeze_backbone(model)
        print("backbone frozen")

    train_loader = build_loader(
        args.train_data,
        cfg,
        image_root=args.image_root,
        shuffle=True,
    )
    val_loader = (
        build_loader(args.val_data, cfg, image_root=args.image_root, shuffle=False)
        if args.val_data
        else None
    )
    loss_fn = build_loss(cfg, train_loader).to(cfg.device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    writer, tensorboard_dir = build_tensorboard_writer(
        args.outdir, args.tensorboard_dir
    )
    writer.add_text("config/json", json.dumps(asdict(cfg), sort_keys=True, indent=2), 0)
    writer.add_text("data/train", str(args.train_data), 0)
    if args.val_data:
        writer.add_text("data/val", str(args.val_data), 0)
    if args.init_checkpoint:
        writer.add_text("init/checkpoint", str(args.init_checkpoint), 0)
    print(f"tensorboard logs: {tensorboard_dir}")

    best_metric = float("-inf")
    best_metrics: dict[str, float | None] | None = None
    for epoch in range(1, cfg.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, loss_fn, cfg)
        if val_loader is not None:
            eval_metrics = evaluate(model, val_loader, loss_fn, cfg, desc="val")
            eval_prefix = "val"
        else:
            eval_metrics = train_metrics
            eval_prefix = "train"

        print(f"epoch {epoch:03d} | {format_metrics('train', train_metrics)}")
        if val_loader is not None:
            print(f"epoch {epoch:03d} | {format_metrics(eval_prefix, eval_metrics)}")
        write_epoch_to_tensorboard(
            writer,
            epoch,
            train_metrics=train_metrics,
            eval_metrics=eval_metrics,
            eval_prefix=eval_prefix,
            lr=cfg.lr,
        )

        score = eval_metrics["auroc"]
        if score is None:
            score = eval_metrics["balanced_accuracy"]
        assert score is not None
        improved = float(score) > best_metric
        if improved:
            best_metric = float(score)
            best_metrics = eval_metrics
        train_run = {
            "outdir": args.outdir,
            "train_data": args.train_data,
            "val_data": args.val_data,
            "init_checkpoint": args.init_checkpoint,
            "tensorboard_dir": tensorboard_dir,
        }
        save_checkpoint(
            Path(args.outdir) / f"ckpt_epoch_{epoch:03d}.pt",
            epoch=epoch,
            cfg=cfg,
            model=model,
            optimizer=optimizer,
            metrics=eval_metrics,
            train_run=train_run,
        )
        if improved:
            save_checkpoint(
                Path(args.outdir) / "best.pt",
                epoch=epoch,
                cfg=cfg,
                model=model,
                optimizer=optimizer,
                metrics=eval_metrics,
                train_run=train_run,
            )

    writer.flush()
    writer.close()
    print("best metrics: " + json.dumps(best_metrics or {}, sort_keys=True))
    return 0


def run_test(args: argparse.Namespace) -> int:
    if not args.test_data:
        raise ValueError("--test-data is required in test mode")
    if not args.checkpoint:
        raise ValueError("--checkpoint is required in test mode")

    checkpoint, cfg, model = load_classifier_checkpoint(
        args.checkpoint,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    test_loader = build_loader(
        args.test_data,
        cfg,
        image_root=args.image_root,
        shuffle=False,
    )
    loss_fn = nn.BCEWithLogitsLoss().to(cfg.device)
    test_metrics = evaluate(model, test_loader, loss_fn, cfg, desc="test")
    print(
        "checkpoint metadata: "
        + json.dumps(
            {
                "checkpoint": args.checkpoint,
                "epoch": checkpoint.get("epoch"),
                "backbone": cfg.backbone,
                "preset": cfg.backbone_preset,
                "test_data": args.test_data,
            },
            sort_keys=True,
        )
    )
    print(format_metrics("test", test_metrics))
    return 0


def run_classify_videos(args: argparse.Namespace) -> int:
    if not args.test_data:
        raise ValueError("--test-data is required in classify-videos mode")
    if not args.checkpoint:
        raise ValueError("--checkpoint is required in classify-videos mode")
    if not args.video_root:
        raise ValueError("--video-root is required in classify-videos mode")
    if not args.output_dir:
        raise ValueError("--output-dir is required in classify-videos mode")
    if args.frame_stride < 1:
        raise ValueError("--frame-stride must be >= 1")

    annotations = pd.read_csv(args.test_data)
    recording_videos = resolve_recording_video_paths(annotations, args.video_root)
    _, cfg, model = load_classifier_checkpoint(
        args.checkpoint,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        "video inference metadata: "
        + json.dumps(
            {
                "checkpoint": args.checkpoint,
                "test_data": args.test_data,
                "video_root": args.video_root,
                "output_dir": str(output_dir),
                "frame_stride": args.frame_stride,
                "threshold": args.threshold,
                "recordings": len(recording_videos),
            },
            sort_keys=True,
        )
    )

    for recording_id, video_path in recording_videos.items():
        output_video_path = output_dir / f"{recording_id}_classified.mp4"
        output_csv_path = output_dir / f"{recording_id}_classifications.csv"
        print(f"processing {recording_id}: {video_path}")
        frame_count = process_classifier_video(
            recording_id=recording_id,
            video_path=video_path,
            output_video_path=output_video_path,
            output_csv_path=output_csv_path,
            model=model,
            cfg=cfg,
            frame_stride=args.frame_stride,
            threshold=args.threshold,
        )
        print(
            f"wrote {frame_count} frames: {output_video_path} and {output_csv_path}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    if args.mode == "train":
        return run_train(args)
    if args.mode == "test":
        return run_test(args)
    return run_classify_videos(args)


if __name__ == "__main__":
    raise SystemExit(main())
