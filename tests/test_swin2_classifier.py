import math
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch
import torch.nn as nn
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import swin2_classifier
from rodent_dataset import BODY_PARTS, coordinate_columns
from swin2_classifier import (
    RodentClassificationDataset,
    SwinBinaryClassifier,
    build_argument_parser,
    extract_compatible_pose_backbone_state,
    format_video_label,
    frame_output_row,
    held_prediction_for_frame,
    load_pose_backbone_weights,
    prediction_from_logit,
    resolve_recording_video_paths,
)


class FakeFeatureInfo:
    def channels(self) -> list[int]:
        return [4]


class FakeBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=1)
        self.feature_info = FakeFeatureInfo()

    def forward(self, images: torch.Tensor) -> list[torch.Tensor]:
        return [self.conv(images)]


def fake_create_model(*_args: object, **_kwargs: object) -> FakeBackbone:
    return FakeBackbone()


def build_row(image_name: str, *, label: int) -> dict[str, object]:
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
    return row


def write_classifier_csv(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    for row in rows:
        Image.new("RGB", (20, 10), color=(80, 50, 20)).save(
            tmp_path / str(row["image_path"])
        )
    csv_path = tmp_path / "annotations.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


def test_classification_dataset_keeps_positive_and_negative_rows(
    tmp_path: Path,
) -> None:
    csv_path = write_classifier_csv(
        tmp_path,
        [
            build_row("positive.jpg", label=1),
            build_row("negative.jpg", label=0),
        ],
    )

    dataset = RodentClassificationDataset(csv_path, image_root=tmp_path, image_size=16)

    assert len(dataset) == 2
    assert dataset[0]["label"].item() == 1.0
    assert dataset[1]["label"].item() == 0.0


def test_classification_dataset_resizes_image_tensor(tmp_path: Path) -> None:
    csv_path = write_classifier_csv(tmp_path, [build_row("sample.jpg", label=1)])

    dataset = RodentClassificationDataset(csv_path, image_root=tmp_path, image_size=32)
    sample = dataset[0]

    assert tuple(sample["image"].shape) == (3, 32, 32)
    assert sample["image"].dtype == torch.float32


def test_swin_binary_classifier_returns_one_logit_per_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(swin2_classifier.timm, "create_model", fake_create_model)
    model = SwinBinaryClassifier(
        backbone_name="fake_swin",
        pretrained=False,
        feature_index=0,
        dropout=0.0,
    )

    logits = model(torch.rand(2, 3, 16, 16))

    assert tuple(logits.shape) == (2,)


def test_extract_compatible_pose_backbone_state_filters_to_matching_backbone_tensors() -> (
    None
):
    pose_state = {
        "backbone.conv.weight": torch.ones(4, 3, 1, 1),
        "backbone.conv.bias": torch.ones(4),
        "backbone.bad_shape": torch.ones(2),
        "head.0.weight": torch.ones(4, 4, 1, 1),
    }
    classifier_state = {
        "backbone.conv.weight": torch.zeros(4, 3, 1, 1),
        "backbone.conv.bias": torch.zeros(4),
        "backbone.bad_shape": torch.zeros(3),
        "classifier.weight": torch.zeros(1, 4),
    }

    compatible = extract_compatible_pose_backbone_state(pose_state, classifier_state)

    assert sorted(compatible) == ["backbone.conv.bias", "backbone.conv.weight"]


def test_load_pose_backbone_weights_loads_compatible_tensors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(swin2_classifier.timm, "create_model", fake_create_model)
    model = SwinBinaryClassifier(
        backbone_name="fake_swin",
        pretrained=False,
        feature_index=0,
        dropout=0.0,
    )
    checkpoint_path = tmp_path / "pose.pt"
    torch.save(
        {
            "model": {
                "backbone.conv.weight": torch.full_like(
                    model.backbone.conv.weight,
                    0.25,
                ),
                "backbone.conv.bias": torch.full_like(model.backbone.conv.bias, 0.5),
                "head.0.weight": torch.ones(1),
            }
        },
        checkpoint_path,
    )

    loaded = load_pose_backbone_weights(model, checkpoint_path)

    assert loaded == 2
    assert torch.allclose(
        model.backbone.conv.weight, torch.full_like(model.backbone.conv.weight, 0.25)
    )
    assert torch.allclose(
        model.backbone.conv.bias, torch.full_like(model.backbone.conv.bias, 0.5)
    )


def test_classifier_argument_parser_accepts_train_and_test_arguments() -> None:
    parser = build_argument_parser()

    train_args = parser.parse_args(
        [
            "--mode",
            "train",
            "--train-data",
            "generated/rodent_annotations_train.csv",
            "--val-data",
            "generated/rodent_annotations_val.csv",
            "--outdir",
            "runs/classifier",
            "--tensorboard-dir",
            "runs/classifier/tensorboard-custom",
            "--backbone",
            "swinv2_cr_tiny_384",
            "--backbone-preset",
            "balanced",
            "--init-checkpoint",
            "runs/swin2-balanced-bs48/best.pt",
            "--freeze-backbone",
            "--class-weight",
            "auto",
        ]
    )
    test_args = parser.parse_args(
        [
            "--mode",
            "test",
            "--test-data",
            "generated/rodent_annotations_test.csv",
            "--checkpoint",
            "runs/classifier/best.pt",
        ]
    )

    assert train_args.mode == "train"
    assert train_args.init_checkpoint == "runs/swin2-balanced-bs48/best.pt"
    assert train_args.tensorboard_dir == "runs/classifier/tensorboard-custom"
    assert train_args.freeze_backbone is True
    assert train_args.class_weight == "auto"
    classify_video_args = parser.parse_args(
        [
            "--mode",
            "classify-videos",
            "--test-data",
            "generated/rodent_annotations_test.csv",
            "--checkpoint",
            "runs/swin2-classifier/best.pt",
            "--video-root",
            "Translational neuroimaging group - rodents",
            "--output-dir",
            "generated/swin2_classifier_video_test",
            "--frame-stride",
            "5",
            "--threshold",
            "0.5",
        ]
    )

    assert test_args.mode == "test"
    assert test_args.checkpoint == "runs/classifier/best.pt"
    assert classify_video_args.mode == "classify-videos"
    assert classify_video_args.video_root == "Translational neuroimaging group - rodents"
    assert classify_video_args.output_dir == "generated/swin2_classifier_video_test"
    assert classify_video_args.frame_stride == 5
    assert classify_video_args.threshold == 0.5



def test_resolve_recording_video_paths_matches_exact_mp4_stem(tmp_path: Path) -> None:
    video_root = tmp_path / "videos"
    (video_root / "session_a" / "mp4").mkdir(parents=True)
    (video_root / "session_b" / "mp4").mkdir(parents=True)
    rec_a = video_root / "session_a" / "mp4" / "rec_a.mp4"
    rec_b = video_root / "session_b" / "mp4" / "rec_b.mp4"
    rec_a.write_bytes(b"not a real video")
    rec_b.write_bytes(b"not a real video")
    annotations = pd.DataFrame(
        {"recording_id": ["rec_b", "rec_a", "rec_b"], "label": [1, 0, 1]}
    )

    resolved = resolve_recording_video_paths(annotations, video_root)

    assert list(resolved) == ["rec_b", "rec_a"]
    assert resolved["rec_a"] == rec_a
    assert resolved["rec_b"] == rec_b


def test_resolve_recording_video_paths_reports_missing_mp4(tmp_path: Path) -> None:
    video_root = tmp_path / "videos"
    video_root.mkdir()
    annotations = pd.DataFrame({"recording_id": ["missing_rec"]})

    with pytest.raises(ValueError, match="missing_rec"):
        resolve_recording_video_paths(annotations, video_root)


def test_resolve_recording_video_paths_reports_duplicate_requested_stem(
    tmp_path: Path,
) -> None:
    video_root = tmp_path / "videos"
    (video_root / "a").mkdir(parents=True)
    (video_root / "b").mkdir(parents=True)
    (video_root / "a" / "rec_a.mp4").write_bytes(b"one")
    (video_root / "b" / "rec_a.mp4").write_bytes(b"two")
    annotations = pd.DataFrame({"recording_id": ["rec_a"]})

    with pytest.raises(ValueError, match="Duplicate MP4 stems"):
        resolve_recording_video_paths(annotations, video_root)


def test_prediction_label_and_video_overlay_text() -> None:
    no_grab = prediction_from_logit(10, -1.0, threshold=0.5)
    grab = prediction_from_logit(15, 0.0, threshold=0.5)

    assert no_grab.prediction == "no-grab"
    assert grab.prediction == "grab"
    assert format_video_label(15, grab) == "frame 15 | grab p=0.500"


def test_held_prediction_for_frame_reuses_latest_stride_prediction() -> None:
    logits_by_frame = {0: -1.0, 5: 2.0}
    calls: list[int] = []
    latest = None
    results: list[tuple[bool, int, str]] = []

    def classify(frame_number: int):
        calls.append(frame_number)
        return prediction_from_logit(frame_number, logits_by_frame[frame_number])

    for frame_number in range(7):
        classified_on_frame, latest = held_prediction_for_frame(
            frame_number,
            5,
            latest,
            classify,
        )
        results.append(
            (classified_on_frame, latest.source_classified_frame, latest.prediction)
        )

    assert calls == [0, 5]
    assert results[:5] == [
        (True, 0, "no-grab"),
        (False, 0, "no-grab"),
        (False, 0, "no-grab"),
        (False, 0, "no-grab"),
        (False, 0, "no-grab"),
    ]
    assert results[5:] == [(True, 5, "grab"), (False, 5, "grab")]


def test_frame_output_row_has_expected_video_csv_columns(tmp_path: Path) -> None:
    prediction = prediction_from_logit(5, 2.0)

    row = frame_output_row(
        recording_id="rec_a",
        video_path=tmp_path / "rec_a.mp4",
        frame_number=6,
        fps=30.0,
        classified_on_frame=False,
        prediction=prediction,
    )

    assert list(row) == [
        "recording_id",
        "video_path",
        "frame_number",
        "time_sec",
        "classified_on_frame",
        "source_classified_frame",
        "logit",
        "probability",
        "prediction",
    ]
    assert row["time_sec"] == pytest.approx(0.2)
    assert row["source_classified_frame"] == 5
    assert row["prediction"] == "grab"
