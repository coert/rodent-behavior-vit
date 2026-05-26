import argparse
import logging
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import orjson
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


LOGGER = logging.getLogger(__name__)

BODY_PARTS = [
    "nose",
    "upper_jaw",
    "lower_jaw",
    "mouth_end_right",
    "mouth_end_left",
    "right_eye",
    "right_earbase",
    "right_earend",
    "right_antler_base",
    "right_antler_end",
    "left_eye",
    "left_earbase",
    "left_earend",
    "left_antler_base",
    "left_antler_end",
    "neck_base",
    "neck_end",
    "throat_base",
    "throat_end",
    "back_base",
    "back_end",
    "back_middle",
    "tail_base",
    "tail_end",
    "front_left_thai",
    "front_left_knee",
    "front_left_paw",
    "front_right_thai",
    "front_right_knee",
    "front_right_paw",
    "back_left_paw",
    "back_left_thai",
    "back_right_thai",
    "back_left_knee",
    "back_right_knee",
    "back_right_paw",
    "belly_bottom",
    "body_middle_right",
    "body_middle_left",
]

SAMPLE_NAME_RE = re.compile(
    r"^(?P<prefix>positive|negative)_(?P<clip_id>.+)_frame_(?P<frame_idx>\d+)$"
)


@dataclass(frozen=True)
class ParsedSample:
    image_path: Path
    label: int
    clip_id: str
    recording_id: str
    frame_idx: int


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def coordinate_columns(body_parts: list[str] | None = None) -> list[str]:
    parts = body_parts or BODY_PARTS
    columns: list[str] = []
    for body_part in parts:
        columns.extend([f"{body_part}_x", f"{body_part}_y"])
    return columns


def empty_coordinate_map(body_parts: list[str] | None = None) -> dict[str, float]:
    return {column: math.nan for column in coordinate_columns(body_parts)}


def parse_sample_path(image_path: Path) -> ParsedSample:
    match = SAMPLE_NAME_RE.match(image_path.stem)
    if not match:
        raise ValueError(f"Unexpected sample filename format: {image_path.name}")

    label = 1 if match.group("prefix") == "positive" else 0
    clip_id = match.group("clip_id")
    recording_id = clip_id.split("_clip_", 1)[0]
    frame_idx = int(match.group("frame_idx"))
    return ParsedSample(
        image_path=image_path,
        label=label,
        clip_id=clip_id,
        recording_id=recording_id,
        frame_idx=frame_idx,
    )


def discover_samples(samples_root: Path) -> list[Path]:
    return sorted(samples_root.glob("*.jpg"))


def build_dlc_index(dlc_root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for experiment_dir in sorted(dlc_root.iterdir()):
        if not experiment_dir.is_dir():
            continue
        for clip_dir in sorted(experiment_dir.iterdir()):
            if not clip_dir.is_dir():
                continue
            index.setdefault(clip_dir.name, clip_dir)
    return index


def resolve_after_adapt_json(clip_dir: Path) -> Path | None:
    matches = sorted(clip_dir.glob("*after_adapt.json"))
    if not matches:
        return None
    if len(matches) > 1:
        LOGGER.warning(
            "Multiple after_adapt JSON files found in %s; using %s",
            clip_dir,
            matches[0].name,
        )
    return matches[0]


def _candidate_score(candidate: Any, body_parts: list[str]) -> tuple[int, float]:
    if not isinstance(candidate, list):
        return (-1, -1.0)

    valid_coords = 0
    confidence_sum = 0.0
    confidence_count = 0
    for coords in candidate[: len(body_parts)]:
        if not isinstance(coords, list) or len(coords) < 2:
            continue
        if coords[0] == -1 or coords[1] == -1:
            continue
        valid_coords += 1
        if len(coords) >= 3 and isinstance(coords[2], (int, float)):
            confidence_sum += float(coords[2])
            confidence_count += 1

    mean_confidence = confidence_sum / confidence_count if confidence_count else 0.0
    return (valid_coords, mean_confidence)


def extract_frame_coordinates(
    frame_payload: dict[str, Any], body_parts: list[str] | None = None
) -> dict[str, float]:
    parts = body_parts or BODY_PARTS
    coordinates = empty_coordinate_map(parts)

    if not isinstance(frame_payload, dict):
        return coordinates

    bodyparts = frame_payload.get("bodyparts")
    if not isinstance(bodyparts, list) or not bodyparts:
        return coordinates

    best_candidate = max(
        bodyparts, key=lambda candidate: _candidate_score(candidate, parts)
    )
    if not isinstance(best_candidate, list):
        return coordinates

    for index, body_part in enumerate(parts):
        if index >= len(best_candidate):
            break
        coords = best_candidate[index]
        if not isinstance(coords, list) or len(coords) < 2:
            continue
        if coords[0] == -1 or coords[1] == -1:
            continue
        coordinates[f"{body_part}_x"] = float(coords[0])
        coordinates[f"{body_part}_y"] = float(coords[1])

    return coordinates


def relative_to_repo(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def load_json_payload(json_path: Path, cache: dict[Path, list[Any]]) -> list[Any]:
    cached = cache.get(json_path)
    if cached is not None:
        return cached

    payload = orjson.loads(json_path.read_bytes())
    cache[json_path] = payload
    return payload


def annotation_row(
    sample: ParsedSample,
    repo_root: Path,
    split: str | None = None,
    coordinates: dict[str, float] | None = None,
    dlc_dir: Path | None = None,
    dlc_json: Path | None = None,
) -> dict[str, Any]:
    row = {
        "image_path": relative_to_repo(sample.image_path, repo_root),
        "label": sample.label,
        "split": split,
        "clip_id": sample.clip_id,
        "recording_id": sample.recording_id,
        "frame_idx": sample.frame_idx,
        "dlc_dir": relative_to_repo(dlc_dir, repo_root) if dlc_dir else None,
        "dlc_json_path": relative_to_repo(dlc_json, repo_root) if dlc_json else None,
    }
    row.update(coordinates or empty_coordinate_map())
    return row


def build_annotation_tables(
    repo_root: Path,
    positive_root: Path,
    negative_root: Path,
    dlc_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dlc_index = build_dlc_index(dlc_root)
    json_cache: dict[Path, list[Any]] = {}
    rows: list[dict[str, Any]] = []
    rejects: list[dict[str, Any]] = []

    sample_paths = discover_samples(positive_root) + discover_samples(negative_root)

    for image_path in sample_paths:
        try:
            sample = parse_sample_path(image_path)
        except ValueError as error:
            rejects.append(
                {
                    "image_path": relative_to_repo(image_path, repo_root),
                    "reason": str(error),
                }
            )
            continue

        if sample.label == 0:
            rows.append(annotation_row(sample=sample, repo_root=repo_root))
            continue

        clip_dir = dlc_index.get(sample.clip_id)
        if clip_dir is None:
            rejects.append(
                {
                    "image_path": relative_to_repo(image_path, repo_root),
                    "reason": f"Missing DeepLabCut directory for clip '{sample.clip_id}'",
                }
            )
            continue

        json_path = resolve_after_adapt_json(clip_dir)
        if json_path is None:
            rejects.append(
                {
                    "image_path": relative_to_repo(image_path, repo_root),
                    "reason": f"Missing after_adapt JSON in '{relative_to_repo(clip_dir, repo_root)}'",
                }
            )
            continue

        try:
            payload = load_json_payload(json_path, json_cache)
        except Exception as error:  # pragma: no cover - defensive file IO
            rejects.append(
                {
                    "image_path": relative_to_repo(image_path, repo_root),
                    "reason": f"Failed to load '{relative_to_repo(json_path, repo_root)}': {error}",
                }
            )
            continue

        if sample.frame_idx >= len(payload):
            rejects.append(
                {
                    "image_path": relative_to_repo(image_path, repo_root),
                    "reason": f"Frame index {sample.frame_idx} is out of bounds for '{relative_to_repo(json_path, repo_root)}'",
                }
            )
            continue

        frame_payload = payload[sample.frame_idx]
        coordinates = extract_frame_coordinates(frame_payload)
        rows.append(
            annotation_row(
                sample=sample,
                repo_root=repo_root,
                coordinates=coordinates,
                dlc_dir=clip_dir,
                dlc_json=json_path,
            )
        )

    annotations = pd.DataFrame(rows)
    rejects_df = pd.DataFrame(rejects)
    return annotations, rejects_df


def normalize_split_ratios(
    train: float, val: float, test: float
) -> tuple[float, float, float]:
    total = train + val + test
    if total <= 0:
        raise ValueError("Split ratios must sum to a positive value")
    return (train / total, val / total, test / total)


def assign_group_splits(
    annotations: pd.DataFrame,
    group_by: str,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> pd.DataFrame:
    if annotations.empty:
        annotations = annotations.copy()
        annotations["split"] = []
        return annotations

    if group_by not in annotations.columns:
        raise KeyError(f"Unknown group column '{group_by}'")

    train_ratio, val_ratio, test_ratio = normalize_split_ratios(
        train_ratio, val_ratio, test_ratio
    )
    grouped = annotations.groupby(group_by).size().to_dict()
    groups = list(grouped.items())

    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(len(groups), generator=generator).tolist()
    shuffled_groups = [groups[index] for index in permutation]

    targets = {
        "train": len(annotations) * train_ratio,
        "val": len(annotations) * val_ratio,
        "test": len(annotations) * test_ratio,
    }
    current = {"train": 0, "val": 0, "test": 0}
    assignments: dict[str, str] = {}

    seeded_splits = ["train", "val", "test"]
    for split_name, (group_name, row_count) in zip(seeded_splits, shuffled_groups):
        assignments[group_name] = split_name
        current[split_name] += row_count

    for group_name, row_count in shuffled_groups[len(seeded_splits) :]:
        split_name = max(targets, key=lambda name: targets[name] - current[name])
        assignments[group_name] = split_name
        current[split_name] += row_count

    assigned = annotations.copy()
    assigned["split"] = assigned[group_by].map(assignments)
    return assigned


def write_annotation_outputs(
    annotations: pd.DataFrame,
    rejects: pd.DataFrame,
    output_dir: Path,
    stem: str = "rodent_annotations",
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths: dict[str, Path] = {}
    master_path = output_dir / f"{stem}.csv"
    annotations.to_csv(master_path, index=False)
    output_paths["master"] = master_path

    for split_name in ("train", "val", "test"):
        split_rows = annotations[annotations["split"] == split_name]
        split_path = output_dir / f"{stem}_{split_name}.csv"
        split_rows.to_csv(split_path, index=False)
        output_paths[split_name] = split_path

    if not rejects.empty:
        rejects_path = output_dir / f"{stem}_rejects.csv"
        rejects.to_csv(rejects_path, index=False)
        output_paths["rejects"] = rejects_path

    return output_paths


class Compose:
    def __init__(self, transforms: list[Any]) -> None:
        self.transforms = transforms

    def __call__(
        self,
        *,
        image: np.ndarray,
        keypoints: list[tuple[float, float]] | list[list[float]],
        keypoint_indices: list[int],
    ) -> dict[str, Any]:
        sample = {
            "image": image,
            "keypoints": list(keypoints),
            "keypoint_indices": list(keypoint_indices),
        }
        for transform in self.transforms:
            sample = transform(**sample)
        return sample


class Resize:
    def __init__(self, *, height: int, width: int) -> None:
        self.height = height
        self.width = width

    def __call__(
        self,
        *,
        image: np.ndarray,
        keypoints: list[tuple[float, float]] | list[list[float]],
        keypoint_indices: list[int],
    ) -> dict[str, Any]:
        source_height, source_width = image.shape[:2]
        resized_image = np.asarray(
            Image.fromarray(image).resize(
                (self.width, self.height), Image.Resampling.BILINEAR
            )
        )
        scale_x = self.width / source_width
        scale_y = self.height / source_height
        resized_keypoints = [
            (float(x_value) * scale_x, float(y_value) * scale_y)
            for x_value, y_value in keypoints
        ]
        return {
            "image": resized_image,
            "keypoints": resized_keypoints,
            "keypoint_indices": keypoint_indices,
        }


class HorizontalFlip:
    def __init__(self, *, p: float = 0.5) -> None:
        self.p = p

    def __call__(
        self,
        *,
        image: np.ndarray,
        keypoints: list[tuple[float, float]] | list[list[float]],
        keypoint_indices: list[int],
    ) -> dict[str, Any]:
        if self.p <= 0 or random.random() >= self.p:
            return {
                "image": image,
                "keypoints": list(keypoints),
                "keypoint_indices": keypoint_indices,
            }

        width = image.shape[1]
        flipped_image = np.ascontiguousarray(image[:, ::-1])
        flipped_keypoints = [
            (float(width - 1) - float(x_value), float(y_value))
            for x_value, y_value in keypoints
        ]
        return {
            "image": flipped_image,
            "keypoints": flipped_keypoints,
            "keypoint_indices": keypoint_indices,
        }


class Crop:
    def __init__(self, *, x_min: int, y_min: int, x_max: int, y_max: int) -> None:
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max

    def __call__(
        self,
        *,
        image: np.ndarray,
        keypoints: list[tuple[float, float]] | list[list[float]],
        keypoint_indices: list[int],
    ) -> dict[str, Any]:
        cropped_image = np.ascontiguousarray(
            image[self.y_min : self.y_max, self.x_min : self.x_max]
        )
        cropped_keypoints = [
            (float(x_value) - self.x_min, float(y_value) - self.y_min)
            for x_value, y_value in keypoints
        ]
        return {
            "image": cropped_image,
            "keypoints": cropped_keypoints,
            "keypoint_indices": keypoint_indices,
        }


class Normalize:
    def __init__(
        self,
        *,
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
        max_pixel_value: float = 255.0,
    ) -> None:
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.max_pixel_value = max_pixel_value

    def __call__(
        self,
        *,
        image: np.ndarray,
        keypoints: list[tuple[float, float]] | list[list[float]],
        keypoint_indices: list[int],
    ) -> dict[str, Any]:
        normalized = image.astype(np.float32) / self.max_pixel_value
        normalized = (normalized - self.mean) / self.std
        return {
            "image": normalized,
            "keypoints": list(keypoints),
            "keypoint_indices": keypoint_indices,
        }


class ToTensorImage:
    def __call__(
        self,
        *,
        image: np.ndarray,
        keypoints: list[tuple[float, float]] | list[list[float]],
        keypoint_indices: list[int],
    ) -> dict[str, Any]:
        tensor = torch.from_numpy(np.moveaxis(image, -1, 0)).to(dtype=torch.float32)
        if tensor.max().item() > 1.0:
            tensor = tensor / 255.0
        return {
            "image": tensor,
            "keypoints": list(keypoints),
            "keypoint_indices": keypoint_indices,
        }


def build_train_transform(
    image_size: tuple[int, int] = (224, 224),
    horizontal_flip_prob: float = 0.5,
    normalize: bool = True,
    to_tensor: bool = True,
) -> Any:
    transforms: list[Any] = [Resize(height=image_size[0], width=image_size[1])]
    if horizontal_flip_prob > 0:
        transforms.append(HorizontalFlip(p=horizontal_flip_prob))
    if normalize:
        transforms.append(Normalize())
    if to_tensor:
        transforms.append(ToTensorImage())
    return Compose(transforms)


def build_eval_transform(
    image_size: tuple[int, int] = (224, 224),
    normalize: bool = True,
    to_tensor: bool = True,
) -> Any:
    transforms: list[Any] = [Resize(height=image_size[0], width=image_size[1])]
    if normalize:
        transforms.append(Normalize())
    if to_tensor:
        transforms.append(ToTensorImage())
    return Compose(transforms)


def _image_size(image: Any) -> tuple[int, int]:
    if isinstance(image, Image.Image):
        return image.width, image.height
    if isinstance(image, np.ndarray):
        return int(image.shape[1]), int(image.shape[0])
    if torch.is_tensor(image):
        if image.ndim < 2:
            raise ValueError("Image tensor must have at least 2 dimensions")
        if image.ndim == 2:
            height, width = image.shape
        else:
            height, width = image.shape[-2], image.shape[-1]
        return int(width), int(height)
    raise TypeError(f"Unsupported image type for size detection: {type(image)!r}")


def _is_keypoint_in_bounds(
    x_value: float, y_value: float, width: int, height: int
) -> bool:
    return 0 <= x_value < width and 0 <= y_value < height


def _empty_keypoints(body_parts: list[str]) -> list[list[float]]:
    return [[math.nan, math.nan] for _ in body_parts]


class RodentKeypointDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        image_root: str | Path | None = None,
        paired_transform: Any | None = None,
        transform: Any | None = None,
        body_parts: list[str] | None = None,
        return_metadata: bool = True,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.image_root = Path(image_root) if image_root is not None else Path.cwd()
        self.paired_transform = paired_transform
        self.transform = transform
        self.body_parts = body_parts or BODY_PARTS
        self.return_metadata = return_metadata
        self.annotations = pd.read_csv(self.csv_path)

    def __len__(self) -> int:
        return len(self.annotations)

    def _resolve_image_path(self, image_path: str) -> Path:
        path = Path(image_path)
        if path.is_absolute():
            return path
        return self.image_root / path

    def _load_keypoints(self, row: pd.Series) -> tuple[list[list[float]], list[bool]]:
        keypoints = _empty_keypoints(self.body_parts)
        valid_mask: list[bool] = []
        for index, body_part in enumerate(self.body_parts):
            x_value = row[f"{body_part}_x"]
            y_value = row[f"{body_part}_y"]
            is_valid = pd.notna(x_value) and pd.notna(y_value)
            valid_mask.append(bool(is_valid))
            if is_valid:
                keypoints[index] = [float(x_value), float(y_value)]
        return keypoints, valid_mask

    def _apply_paired_transform(
        self,
        image: Image.Image,
        keypoints: list[list[float]],
        valid_mask: list[bool],
    ) -> tuple[Any, list[list[float]], list[bool]]:
        if self.paired_transform is None:
            return image, keypoints, valid_mask

        image_array = np.asarray(image)
        valid_indices = [index for index, is_valid in enumerate(valid_mask) if is_valid]
        valid_keypoints = [tuple(keypoints[index]) for index in valid_indices]

        transformed = self.paired_transform(
            image=image_array,
            keypoints=valid_keypoints,
            keypoint_indices=valid_indices,
        )

        transformed_image = transformed["image"]
        transformed_keypoints = transformed.get("keypoints", valid_keypoints)
        transformed_indices = transformed.get("keypoint_indices", valid_indices)

        width, height = _image_size(transformed_image)
        rehydrated_keypoints = _empty_keypoints(self.body_parts)
        rehydrated_mask = [False] * len(self.body_parts)

        for transformed_index, transformed_keypoint in zip(
            transformed_indices, transformed_keypoints
        ):
            index = int(transformed_index)
            x_value = float(transformed_keypoint[0])
            y_value = float(transformed_keypoint[1])
            if not math.isfinite(x_value) or not math.isfinite(y_value):
                continue
            if not _is_keypoint_in_bounds(x_value, y_value, width, height):
                continue
            rehydrated_keypoints[index] = [x_value, y_value]
            rehydrated_mask[index] = True

        return transformed_image, rehydrated_keypoints, rehydrated_mask

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.annotations.iloc[index]
        image_path = self._resolve_image_path(str(row["image_path"]))
        image = Image.open(image_path).convert("RGB")

        keypoints, valid_mask = self._load_keypoints(row)
        image, keypoints, valid_mask = self._apply_paired_transform(
            image, keypoints, valid_mask
        )

        if self.transform is not None:
            image = self.transform(image)

        sample: dict[str, Any] = {
            "image": image,
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
            "keypoints": torch.tensor(keypoints, dtype=torch.float32),
            "keypoints_valid": torch.tensor(valid_mask, dtype=torch.bool),
        }

        if self.return_metadata:
            sample["metadata"] = {
                "image_path": str(row["image_path"]),
                "clip_id": row.get("clip_id"),
                "recording_id": row.get("recording_id"),
                "frame_idx": int(row["frame_idx"]),
                "split": row.get("split"),
            }

        return sample


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate rodent image annotations and grouped dataset splits."
    )
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parent
    )
    parser.add_argument(
        "--positive-root", type=Path, default=Path("rodent-samples/pos")
    )
    parser.add_argument(
        "--negative-root", type=Path, default=Path("rodent-samples/neg")
    )
    parser.add_argument(
        "--dlc-root", type=Path, default=Path("videos/deeplabcut_inference")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("generated"))
    parser.add_argument("--output-stem", default="rodent_annotations")
    parser.add_argument(
        "--group-by", choices=["clip_id", "recording_id"], default="clip_id"
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--verbose", action="store_true")
    return parser


def run(args: argparse.Namespace) -> dict[str, Path]:
    repo_root = args.repo_root.resolve()
    positive_root = (
        (repo_root / args.positive_root).resolve()
        if not args.positive_root.is_absolute()
        else args.positive_root.resolve()
    )
    negative_root = (
        (repo_root / args.negative_root).resolve()
        if not args.negative_root.is_absolute()
        else args.negative_root.resolve()
    )
    dlc_root = (
        (repo_root / args.dlc_root).resolve()
        if not args.dlc_root.is_absolute()
        else args.dlc_root.resolve()
    )
    output_dir = (
        (repo_root / args.output_dir).resolve()
        if not args.output_dir.is_absolute()
        else args.output_dir.resolve()
    )

    annotations, rejects = build_annotation_tables(
        repo_root=repo_root,
        positive_root=positive_root,
        negative_root=negative_root,
        dlc_root=dlc_root,
    )
    annotations = assign_group_splits(
        annotations=annotations,
        group_by=args.group_by,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
    outputs = write_annotation_outputs(
        annotations, rejects, output_dir=output_dir, stem=args.output_stem
    )

    LOGGER.info("Wrote %s annotated rows", len(annotations))
    if not annotations.empty:
        LOGGER.info("Split counts: %s", annotations["split"].value_counts().to_dict())
        LOGGER.info("Label counts: %s", annotations["label"].value_counts().to_dict())
    if not rejects.empty:
        LOGGER.warning("Rejected %s rows", len(rejects))
    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    configure_logging(verbose=args.verbose)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
