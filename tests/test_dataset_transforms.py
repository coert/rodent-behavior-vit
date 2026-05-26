import math
from pathlib import Path
import sys

import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rodent_dataset import (
    BODY_PARTS,
    Compose,
    Crop,
    HorizontalFlip,
    Resize,
    RodentKeypointDataset,
    coordinate_columns,
)


def build_row(
    image_name: str, label: int, nose_xy: tuple[float, float] | None
) -> dict[str, object]:
    row: dict[str, object] = {
        "image_path": image_name,
        "label": label,
        "split": "train",
        "clip_id": "clip_a",
        "recording_id": "recording_a",
        "frame_idx": 0,
        "dlc_dir": None,
        "dlc_json_path": None,
    }
    for column in coordinate_columns(BODY_PARTS):
        row[column] = math.nan
    if nose_xy is not None:
        row["nose_x"] = nose_xy[0]
        row["nose_y"] = nose_xy[1]
    return row


def write_dataset(
    tmp_path: Path, row: dict[str, object], image_size: tuple[int, int]
) -> Path:
    image_path = tmp_path / str(row["image_path"])
    Image.new("RGB", image_size, color=(120, 40, 20)).save(image_path)
    csv_path = tmp_path / "dataset.csv"
    pd.DataFrame([row]).to_csv(csv_path, index=False)
    return csv_path


def test_resize_scales_valid_keypoints(tmp_path: Path) -> None:
    row = build_row("resize.jpg", label=1, nose_xy=(5.0, 2.0))
    csv_path = write_dataset(tmp_path, row, image_size=(20, 10))
    transform = Compose([Resize(height=20, width=40)])

    dataset = RodentKeypointDataset(
        csv_path=csv_path,
        image_root=tmp_path,
        paired_transform=transform,
    )
    sample = dataset[0]

    assert sample["keypoints_valid"].sum().item() == 1
    assert sample["keypoints"][0].tolist() == [10.0, 4.0]


def test_horizontal_flip_matches_albumentations_geometry(tmp_path: Path) -> None:
    row = build_row("flip.jpg", label=1, nose_xy=(5.0, 2.0))
    csv_path = write_dataset(tmp_path, row, image_size=(20, 10))
    transform = Compose([HorizontalFlip(p=1.0)])

    dataset = RodentKeypointDataset(
        csv_path=csv_path,
        image_root=tmp_path,
        paired_transform=transform,
    )
    sample = dataset[0]

    assert sample["keypoints_valid"][0].item() is True
    assert sample["keypoints"][0].tolist() == [14.0, 2.0]


def test_crop_invalidates_out_of_frame_keypoints(tmp_path: Path) -> None:
    row = build_row("crop.jpg", label=1, nose_xy=(5.0, 2.0))
    csv_path = write_dataset(tmp_path, row, image_size=(20, 10))
    transform = Compose([Crop(x_min=10, y_min=0, x_max=20, y_max=10)])

    dataset = RodentKeypointDataset(
        csv_path=csv_path,
        image_root=tmp_path,
        paired_transform=transform,
    )
    sample = dataset[0]

    assert sample["keypoints_valid"][0].item() is False
    assert math.isnan(sample["keypoints"][0][0].item())
    assert math.isnan(sample["keypoints"][0][1].item())


def test_negative_samples_remain_invalid_after_transform(tmp_path: Path) -> None:
    row = build_row("negative.jpg", label=0, nose_xy=None)
    csv_path = write_dataset(tmp_path, row, image_size=(20, 10))
    transform = Compose([Resize(height=12, width=24)])

    dataset = RodentKeypointDataset(
        csv_path=csv_path,
        image_root=tmp_path,
        paired_transform=transform,
    )
    sample = dataset[0]

    assert sample["keypoints_valid"].sum().item() == 0
    assert torch_all_nan(sample["keypoints"][0].tolist())


def torch_all_nan(values: list[float]) -> bool:
    return all(math.isnan(value) for value in values)
