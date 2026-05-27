import math
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import swin2_model
from rodent_dataset import BODY_PARTS, coordinate_columns
from swin2_model import (
    BACKBONE_PRESETS,
    IDX,
    Config,
    K,
    build_argument_parser,
    build_pose_model,
    build_runtime_components,
    build_subset_and_weights,
    build_tensorboard_writer,
    capture_tensorboard_example,
    config_from_args,
    decoded_keypoint_error,
    default_tensorboard_dir,
    format_checkpoint_provenance,
    format_eval_metadata,
    format_kp_error_regression_alert,
    learning_rate_for_epoch,
    load_checkpoint_runtime,
    load_checkpoint_training_runtime,
    load_frame_items,
    load_frame_items_with_summary,
    load_template_Tn,
    main,
    render_keypoints_for_tensorboard,
    resolve_resume_tensorboard_dir,
    write_example_artifacts_to_tensorboard,
)


class DummySummaryWriter:
    def add_text(self, *_args: object, **_kwargs: object) -> None:
        return None

    def add_scalar(self, *_args: object, **_kwargs: object) -> None:
        return None

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None


def build_annotation_row(
    image_name: str,
    *,
    label: int = 1,
    required: bool = True,
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

    coords = {
        "nose": (8.0, 6.0),
        "neck_base": (10.0, 10.0),
        "back_middle": (10.0, 20.0),
        "front_left_thai": (11.0, 15.0),
        "front_left_knee": (12.0, 16.0),
        "front_left_paw": (13.0, 17.0),
        "front_right_thai": (9.0, 15.0),
        "front_right_knee": (8.0, 16.0),
        "front_right_paw": (7.0, 17.0),
    }
    if not required:
        coords.pop("nose")

    for body_part, (x_value, y_value) in coords.items():
        row[f"{body_part}_x"] = x_value
        row[f"{body_part}_y"] = y_value
    return row


def write_annotation_csv(
    tmp_path: Path, rows: list[dict[str, object]], *, name: str = "annotations.csv"
) -> Path:
    for row in rows:
        image_path = tmp_path / str(row["image_path"])
        Image.new("RGB", (40, 40), color=(80, 50, 20)).save(image_path)
    csv_path = tmp_path / name
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


def test_load_frame_items_from_annotation_csv_filters_unusable_rows(
    tmp_path: Path,
) -> None:
    csv_path = write_annotation_csv(
        tmp_path,
        [
            build_annotation_row("valid.jpg", label=1, required=True),
            build_annotation_row("negative.jpg", label=0, required=True),
            build_annotation_row("missing_required.jpg", label=1, required=False),
        ],
    )

    items = load_frame_items(str(csv_path))

    assert len(items) == 1
    assert Path(items[0].image_path).name == "valid.jpg"
    assert items[0].conf.shape == (K,)
    assert items[0].conf[IDX["nose"]] == 1.0
    assert items[0].conf[IDX["upper_jaw"]] == 0.0


def test_load_template_tn_from_annotation_csv(tmp_path: Path) -> None:
    csv_path = write_annotation_csv(
        tmp_path,
        [build_annotation_row("template.jpg", label=1, required=True)],
        name="template.csv",
    )

    template_tn = load_template_Tn(str(csv_path), Config())

    assert tuple(template_tn.shape) == (K, 2)
    assert template_tn[IDX["neck_base"]].tolist() == [0.0, 0.0]
    assert template_tn[IDX["back_middle"]].tolist() == [0.0, 1.0]


def test_load_frame_items_with_summary_reports_filtered_rows(tmp_path: Path) -> None:
    csv_path = write_annotation_csv(
        tmp_path,
        [
            build_annotation_row("valid.jpg", label=1, required=True),
            build_annotation_row("negative.jpg", label=0, required=True),
            build_annotation_row("missing_required.jpg", label=1, required=False),
        ],
    )

    items, summary = load_frame_items_with_summary(str(csv_path))

    assert len(items) == 1
    assert summary is not None
    assert summary.total_rows == 3
    assert summary.kept_rows == 1
    assert summary.skipped_negative == 1
    assert summary.skipped_missing_required == 1


def test_build_argument_parser_accepts_csv_train_aliases() -> None:
    parser = build_argument_parser()

    args = parser.parse_args(
        [
            "--mode",
            "train",
            "--train-data",
            "generated/rodent_annotations_train.csv",
            "--template-data",
            "template.csv",
            "--outdir",
            "runs/example",
        ]
    )

    assert args.mode == "train"
    assert args.train_data == "generated/rodent_annotations_train.csv"
    assert args.template_data == "template.csv"


def test_build_argument_parser_accepts_backbone_options() -> None:
    parser = build_argument_parser()

    args = parser.parse_args(
        [
            "--mode",
            "train",
            "--train-data",
            "generated/rodent_annotations_train.csv",
            "--template-data",
            "template.csv",
            "--outdir",
            "runs/example",
            "--backbone",
            "swinv2_cr_tiny_384",
            "--backbone-preset",
            "accuracy",
            "--no-pretrained",
            "--backbone-feature-index",
            "2",
            "--decoder-channels",
            "128",
            "--tensorboard-dir",
            "runs/example/tensorboard",
            "--resume-checkpoint",
            "runs/example/best.pt",
            "--scheduler",
            "cosine",
            "--warmup-epochs",
            "2",
            "--warmup-start-factor",
            "0.4",
            "--min-lr-ratio",
            "0.2",
            "--early-stop-patience",
            "5",
        ]
    )

    assert args.backbone == "swinv2_cr_tiny_384"
    assert args.backbone_preset == "accuracy"
    assert args.pretrained is False
    assert args.backbone_feature_index == 2
    assert args.decoder_channels == 128
    assert args.tensorboard_dir == "runs/example/tensorboard"
    assert args.resume_checkpoint == "runs/example/best.pt"
    assert args.scheduler == "cosine"
    assert args.warmup_epochs == 2
    assert args.warmup_start_factor == pytest.approx(0.4)
    assert args.min_lr_ratio == pytest.approx(0.2)
    assert args.early_stop_patience == 5


def test_config_from_args_accepts_early_stop_patience() -> None:
    cfg = config_from_args(early_stop_patience=7)

    assert cfg.early_stop_patience == 7


@pytest.mark.parametrize(
    ("backbone_name", "image_size", "heatmap_size", "feature_index"),
    [
        ("resnet18", 128, 32, 2),
        ("swinv2_tiny_window8_256", 256, 64, 2),
    ],
)
def test_build_pose_model_outputs_expected_heatmap_shape(
    backbone_name: str,
    image_size: int,
    heatmap_size: int,
    feature_index: int,
) -> None:
    cfg = Config(
        image_size=image_size,
        heatmap_size=heatmap_size,
        stride=image_size // heatmap_size,
        backbone=backbone_name,
        pretrained=False,
        backbone_feature_index=feature_index,
        decoder_channels=64,
        device="cpu",
    )

    model = build_pose_model(cfg, num_keypoints=K).eval()
    images = torch.rand(1, 3, image_size, image_size)

    with torch.no_grad():
        heatmaps = model(images)

    assert tuple(heatmaps.shape) == (1, K, heatmap_size, heatmap_size)


def test_config_from_args_uses_backbone_specific_default_lr() -> None:
    resnet_cfg = config_from_args(backbone="resnet18")
    swin_cfg = config_from_args(backbone="swinv2_cr_tiny_384")
    custom_cfg = config_from_args(backbone="swinv2_cr_tiny_384", lr=2e-5)

    assert resnet_cfg.lr == pytest.approx(3e-4)
    assert swin_cfg.lr == pytest.approx(1e-5)
    assert custom_cfg.lr == pytest.approx(2e-5)


def test_config_from_args_applies_large_batch_defaults_for_swin() -> None:
    cfg = config_from_args(
        backbone="swinv2_cr_tiny_384",
        batch_size=48,
    )

    assert cfg.lr == pytest.approx(2e-5)
    assert cfg.scheduler == "cosine"
    assert cfg.warmup_epochs == 2
    assert cfg.warmup_start_factor == pytest.approx(0.5)
    assert cfg.min_lr_ratio == pytest.approx(0.1)


def test_config_from_args_respects_explicit_large_batch_overrides() -> None:
    cfg = config_from_args(
        backbone="swinv2_cr_tiny_384",
        batch_size=48,
        lr=3e-5,
        scheduler="none",
        warmup_epochs=4,
        warmup_start_factor=0.25,
        min_lr_ratio=0.05,
    )

    assert cfg.lr == pytest.approx(3e-5)
    assert cfg.scheduler is None
    assert cfg.warmup_epochs == 4
    assert cfg.warmup_start_factor == pytest.approx(0.25)
    assert cfg.min_lr_ratio == pytest.approx(0.05)


def test_config_from_args_applies_accuracy_backbone_preset() -> None:
    cfg = config_from_args(
        backbone="swinv2_cr_tiny_384",
        backbone_preset="accuracy",
    )
    override_cfg = config_from_args(
        backbone="swinv2_cr_tiny_384",
        backbone_preset="accuracy",
        backbone_feature_index=1,
    )

    assert cfg.backbone_preset == "accuracy"
    assert (
        cfg.backbone_feature_index
        == BACKBONE_PRESETS["accuracy"]["backbone_feature_index"]
    )
    assert cfg.decoder_channels == BACKBONE_PRESETS["accuracy"]["decoder_channels"]
    assert override_cfg.backbone_feature_index == 1
    assert (
        override_cfg.decoder_channels
        == BACKBONE_PRESETS["accuracy"]["decoder_channels"]
    )


def test_learning_rate_for_epoch_applies_warmup_and_cosine_decay() -> None:
    cfg = Config(
        backbone="swinv2_cr_tiny_384",
        lr=2e-5,
        epochs=6,
        scheduler="cosine",
        warmup_epochs=2,
        warmup_start_factor=0.5,
        min_lr_ratio=0.1,
    )

    lr_values = [
        learning_rate_for_epoch(cfg, epoch) for epoch in range(1, cfg.epochs + 1)
    ]

    assert lr_values[0] == pytest.approx(1e-5)
    assert lr_values[1] == pytest.approx(2e-5)
    assert lr_values[2] == pytest.approx(2e-5)
    assert lr_values[-1] == pytest.approx(2e-6)
    assert lr_values[2] > lr_values[3] > lr_values[4] > lr_values[5]


def test_build_tensorboard_writer_creates_event_file(tmp_path: Path) -> None:
    outdir = tmp_path / "run"
    writer, logdir = build_tensorboard_writer(str(outdir))
    writer.add_scalar("loss/train", 1.23, 1)
    writer.flush()
    writer.close()

    event_files = list(Path(logdir).glob("events.out.tfevents.*"))

    assert logdir == default_tensorboard_dir(str(outdir))
    assert event_files


def test_write_example_artifacts_to_tensorboard_creates_event_file(
    tmp_path: Path,
) -> None:
    outdir = tmp_path / "run"
    writer, logdir = build_tensorboard_writer(str(outdir))
    example = capture_tensorboard_example(
        images=torch.rand(1, 3, 32, 32),
        heatmaps_target=torch.rand(1, K, 8, 8),
        heatmaps_pred=torch.rand(1, K, 8, 8),
        image_size=32,
    )

    write_example_artifacts_to_tensorboard(writer, 1, "train", example)
    writer.flush()
    writer.close()

    event_files = list(Path(logdir).glob("events.out.tfevents.*"))

    assert event_files


def test_decoded_keypoint_error_is_zero_for_identical_heatmaps() -> None:
    cfg = Config(image_size=32, heatmap_size=8, stride=4, device="cpu")
    heatmaps = torch.rand(1, K, 8, 8)
    conf = torch.ones(1, K)
    valid = torch.ones(1, K)

    error = decoded_keypoint_error(heatmaps, heatmaps, conf, valid, cfg)

    assert float(error.item()) == pytest.approx(0.0, abs=1e-5)


def test_capture_tensorboard_example_includes_keypoint_views() -> None:
    example = capture_tensorboard_example(
        images=torch.rand(1, 3, 32, 32),
        heatmaps_target=torch.rand(1, K, 8, 8),
        heatmaps_pred=torch.rand(1, K, 8, 8),
        image_size=32,
    )

    assert tuple(example["target_keypoints"].shape) == (3, 32, 32)
    assert tuple(example["pred_keypoints"].shape) == (3, 32, 32)
    assert tuple(example["combined_keypoints"].shape) == (3, 32, 32)


def test_render_keypoints_for_tensorboard_marks_points() -> None:
    image = torch.zeros(3, 16, 16)
    coords = torch.tensor([[1.0, 1.0], [3.0, 2.0]])

    rendered = render_keypoints_for_tensorboard(
        image,
        coords,
        image_size=16,
        heatmap_size=4,
        color=(1.0, 0.0, 0.0),
    )

    assert rendered.max().item() == pytest.approx(1.0)
    assert rendered[0].sum().item() > 0.0


def test_resolve_resume_tensorboard_dir_prefers_checkpoint_metadata(
    tmp_path: Path,
) -> None:
    outdir = tmp_path / "run"
    logdir = outdir / "tensorboard"
    checkpoint = {"tensorboard": {"logdir": str(logdir)}}

    assert resolve_resume_tensorboard_dir(None, str(outdir), checkpoint) == str(logdir)
    assert resolve_resume_tensorboard_dir("custom", str(outdir), checkpoint) == "custom"


def test_load_checkpoint_runtime_supports_legacy_config_without_backbone(
    tmp_path: Path,
) -> None:
    cfg = Config(
        image_size=128,
        heatmap_size=32,
        stride=4,
        pretrained=False,
        device="cpu",
    )
    template_tn = torch.zeros(K, 2)
    subset_mask, kp_weight = build_subset_and_weights(cfg)
    model, dist_head, _loss_hm, _loss_dist = build_runtime_components(
        cfg,
        template_tn,
        subset_mask,
        kp_weight,
    )
    opt = torch.optim.AdamW(
        list(model.parameters()) + list(dist_head.parameters()),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    legacy_cfg = {
        "image_size": cfg.image_size,
        "heatmap_size": cfg.heatmap_size,
        "stride": cfg.stride,
        "sigma_px_out": cfg.sigma_px_out,
        "conf_min": cfg.conf_min,
        "gamma_orient": cfg.gamma_orient,
        "gamma_dir": cfg.gamma_dir,
        "gamma_rear": cfg.gamma_rear,
        "rear_thresh": cfg.rear_thresh,
        "eps": cfg.eps,
        "batch_size": cfg.batch_size,
        "lr": cfg.lr,
        "weight_decay": cfg.weight_decay,
        "epochs": cfg.epochs,
        "lambda_dist": cfg.lambda_dist,
        "num_workers": cfg.num_workers,
        "softargmax_beta": cfg.softargmax_beta,
        "paw_w": cfg.paw_w,
        "knee_w": cfg.knee_w,
        "thai_w": cfg.thai_w,
        "device": cfg.device,
    }
    checkpoint_path = tmp_path / "legacy.pt"
    torch.save(
        {
            "epoch": 1,
            "cfg": legacy_cfg,
            "kp_order": list(range(K)),
            "subset_mask": subset_mask,
            "kp_weight": kp_weight,
            "template_Tn": template_tn,
            "model": model.state_dict(),
            "dist_head": dist_head.state_dict(),
            "opt": opt.state_dict(),
            "val_loss": 0.0,
        },
        checkpoint_path,
    )

    _checkpoint, loaded_cfg, loaded_model, _dist_head, _loss_hm, _loss_dist = (
        load_checkpoint_runtime(
            str(checkpoint_path),
            batch_size=1,
            num_workers=0,
        )
    )

    assert loaded_cfg.backbone == "resnet18"
    assert loaded_cfg.pretrained is False

    with torch.no_grad():
        heatmaps = loaded_model(
            torch.rand(1, 3, cfg.image_size, cfg.image_size, device=loaded_cfg.device)
        )

    assert tuple(heatmaps.shape) == (1, K, cfg.heatmap_size, cfg.heatmap_size)


def test_load_checkpoint_training_runtime_restores_optimizer_and_metadata(
    tmp_path: Path,
) -> None:
    cfg = Config(
        image_size=128,
        heatmap_size=32,
        stride=4,
        pretrained=False,
        scheduler="cosine",
        warmup_epochs=2,
        min_lr_ratio=0.1,
        device="cpu",
    )
    template_tn = torch.zeros(K, 2)
    subset_mask, kp_weight = build_subset_and_weights(cfg)
    model, dist_head, _loss_hm, _loss_dist = build_runtime_components(
        cfg,
        template_tn,
        subset_mask,
        kp_weight,
    )
    opt = torch.optim.AdamW(
        list(model.parameters()) + list(dist_head.parameters()),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    checkpoint_path = tmp_path / "resume.pt"
    torch.save(
        {
            "epoch": 3,
            "cfg": cfg.__dict__,
            "kp_order": list(range(K)),
            "subset_mask": subset_mask,
            "kp_weight": kp_weight,
            "template_Tn": template_tn,
            "model": model.state_dict(),
            "dist_head": dist_head.state_dict(),
            "opt": opt.state_dict(),
            "scheduler": {
                "name": "cosine",
                "last_epoch": 3,
                "last_lr": 7e-6,
                "warmup_epochs": 2,
                "warmup_start_factor": 0.5,
                "min_lr_ratio": 0.1,
            },
            "val_loss": 0.25,
            "best_val_loss": 0.2,
            "tensorboard": {"logdir": str(tmp_path / "tensorboard"), "last_epoch": 3},
        },
        checkpoint_path,
    )

    (
        checkpoint,
        loaded_cfg,
        _loaded_model,
        _loaded_dist_head,
        _loss_hm,
        _loss_dist,
        loaded_opt,
    ) = load_checkpoint_training_runtime(
        str(checkpoint_path),
        epochs=5,
        batch_size=1,
        num_workers=0,
    )

    assert checkpoint["epoch"] == 3
    assert loaded_cfg.epochs == 5
    assert loaded_cfg.scheduler == "cosine"
    assert loaded_cfg.warmup_epochs == 2
    assert loaded_opt.param_groups[0]["lr"] == pytest.approx(7e-6)


def test_main_requires_validation_data_for_early_stopping() -> None:
    with pytest.raises(SystemExit):
        main(
            [
                "--mode",
                "train",
                "--train-data",
                "generated/rodent_annotations_train.csv",
                "--template-data",
                "generated/template.csv",
                "--outdir",
                "runs/example",
                "--early-stop-patience",
                "2",
            ]
        )


def test_run_train_stops_early_after_patience(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg = Config(
        epochs=6,
        batch_size=1,
        num_workers=0,
        pretrained=False,
        early_stop_patience=2,
        device="cpu",
    )
    subset_mask, kp_weight = build_subset_and_weights(cfg)
    template_tn = torch.zeros(K, 2)
    model = torch.nn.Linear(1, 1)
    dist_head = torch.nn.Linear(1, 1)

    monkeypatch.setattr(swin2_model, "config_from_args", lambda **_kwargs: cfg)
    monkeypatch.setattr(
        swin2_model, "build_subset_and_weights", lambda _cfg: (subset_mask, kp_weight)
    )
    monkeypatch.setattr(swin2_model, "load_template_Tn", lambda *_args: template_tn)
    monkeypatch.setattr(
        swin2_model,
        "build_runtime_components",
        lambda *_args: (model, dist_head, torch.nn.MSELoss(), torch.nn.MSELoss()),
    )
    monkeypatch.setattr(
        swin2_model,
        "load_frame_items_with_summary",
        lambda _path: ([object()], None),
    )
    monkeypatch.setattr(
        swin2_model, "build_loader", lambda *_args, **_kwargs: [object()]
    )
    monkeypatch.setattr(
        swin2_model,
        "build_tensorboard_writer",
        lambda _outdir, _logdir: (DummySummaryWriter(), str(tmp_path / "tensorboard")),
    )
    monkeypatch.setattr(
        swin2_model, "write_metrics_to_tensorboard", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        swin2_model,
        "write_example_artifacts_to_tensorboard",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        swin2_model, "format_kp_error_regression_alert", lambda **_kwargs: None
    )
    monkeypatch.setattr(
        swin2_model,
        "train_one_epoch",
        lambda *_args, **_kwargs: {
            "loss": 1.0,
            "loss_hm": 0.5,
            "loss_dist": 0.5,
            "kp_error": 10.0,
            "example": None,
        },
    )
    eval_metrics = iter(
        [
            {
                "loss": 1.0,
                "loss_hm": 0.5,
                "loss_dist": 0.5,
                "kp_error": 9.0,
                "example": None,
            },
            {
                "loss": 0.9,
                "loss_hm": 0.5,
                "loss_dist": 0.4,
                "kp_error": 8.0,
                "example": None,
            },
            {
                "loss": 0.95,
                "loss_hm": 0.5,
                "loss_dist": 0.45,
                "kp_error": 8.5,
                "example": None,
            },
            {
                "loss": 0.96,
                "loss_hm": 0.5,
                "loss_dist": 0.46,
                "kp_error": 8.6,
                "example": None,
            },
        ]
    )
    monkeypatch.setattr(
        swin2_model,
        "eval_one_epoch",
        lambda *_args, **_kwargs: next(eval_metrics),
    )

    args = build_argument_parser().parse_args(
        [
            "--mode",
            "train",
            "--train-data",
            "train.csv",
            "--val-data",
            "val.csv",
            "--template-data",
            "template.csv",
            "--outdir",
            str(tmp_path / "out"),
            "--early-stop-patience",
            "2",
        ]
    )

    result = swin2_model.run_train(args)

    assert result == 0
    assert (tmp_path / "out" / "ckpt_epoch_004.pt").exists()
    assert not (tmp_path / "out" / "ckpt_epoch_005.pt").exists()
    stopped_checkpoint = torch.load(tmp_path / "out" / "ckpt_epoch_004.pt")
    best_checkpoint = torch.load(tmp_path / "out" / "best.pt")
    assert stopped_checkpoint["epochs_without_improvement"] == 2
    assert stopped_checkpoint["best_val_loss"] == pytest.approx(0.9)
    assert best_checkpoint["epoch"] == 2
    assert "early stopping at epoch 004" in capsys.readouterr().out


def test_run_train_resume_preserves_early_stop_counter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg = Config(
        epochs=6,
        batch_size=1,
        num_workers=0,
        pretrained=False,
        early_stop_patience=2,
        device="cpu",
    )
    subset_mask, kp_weight = build_subset_and_weights(cfg)
    template_tn = torch.zeros(K, 2)
    model = torch.nn.Linear(1, 1)
    dist_head = torch.nn.Linear(1, 1)
    opt = torch.optim.AdamW(
        list(model.parameters()) + list(dist_head.parameters()),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    resume_checkpoint = {
        "epoch": 3,
        "cfg": cfg.__dict__,
        "subset_mask": subset_mask,
        "kp_weight": kp_weight,
        "template_Tn": template_tn,
        "val_loss": 0.8,
        "best_val_loss": 0.8,
        "val_kp_error": 7.5,
        "best_val_kp_error": 7.5,
        "epochs_without_improvement": 1,
        "tensorboard": {"logdir": str(tmp_path / "tensorboard"), "last_epoch": 3},
    }

    monkeypatch.setattr(
        swin2_model,
        "load_checkpoint_training_runtime",
        lambda *_args, **_kwargs: (
            resume_checkpoint,
            cfg,
            model,
            dist_head,
            torch.nn.MSELoss(),
            torch.nn.MSELoss(),
            opt,
        ),
    )
    monkeypatch.setattr(
        swin2_model,
        "load_frame_items_with_summary",
        lambda _path: ([object()], None),
    )
    monkeypatch.setattr(
        swin2_model, "build_loader", lambda *_args, **_kwargs: [object()]
    )
    monkeypatch.setattr(
        swin2_model,
        "build_tensorboard_writer",
        lambda _outdir, _logdir: (DummySummaryWriter(), str(tmp_path / "tensorboard")),
    )
    monkeypatch.setattr(
        swin2_model, "write_metrics_to_tensorboard", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        swin2_model,
        "write_example_artifacts_to_tensorboard",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        swin2_model, "format_kp_error_regression_alert", lambda **_kwargs: None
    )
    monkeypatch.setattr(
        swin2_model,
        "train_one_epoch",
        lambda *_args, **_kwargs: {
            "loss": 1.0,
            "loss_hm": 0.5,
            "loss_dist": 0.5,
            "kp_error": 10.0,
            "example": None,
        },
    )
    monkeypatch.setattr(
        swin2_model,
        "eval_one_epoch",
        lambda *_args, **_kwargs: {
            "loss": 0.81,
            "loss_hm": 0.5,
            "loss_dist": 0.31,
            "kp_error": 7.6,
            "example": None,
        },
    )

    args = build_argument_parser().parse_args(
        [
            "--mode",
            "train",
            "--train-data",
            "train.csv",
            "--val-data",
            "val.csv",
            "--outdir",
            str(tmp_path / "out"),
            "--resume-checkpoint",
            str(tmp_path / "resume.pt"),
        ]
    )

    result = swin2_model.run_train(args)

    assert result == 0
    assert (tmp_path / "out" / "ckpt_epoch_004.pt").exists()
    assert not (tmp_path / "out" / "ckpt_epoch_005.pt").exists()
    stopped_checkpoint = torch.load(tmp_path / "out" / "ckpt_epoch_004.pt")
    assert stopped_checkpoint["epochs_without_improvement"] == 2
    assert stopped_checkpoint["best_val_loss"] == pytest.approx(0.8)
    assert "early stopping at epoch 004" in capsys.readouterr().out


def test_format_checkpoint_provenance_includes_run_details() -> None:
    provenance = format_checkpoint_provenance(
        {
            "epoch": 2,
            "cfg": {
                "backbone": "swinv2_cr_tiny_384",
                "backbone_preset": "balanced",
                "scheduler": "cosine",
            },
            "best_val_loss": 0.0247,
            "best_val_kp_error": 12.3456,
            "train_run": {
                "outdir": "runs/swin2-resumed",
                "resume_checkpoint": "runs/old/best.pt",
            },
            "tensorboard": {
                "logdir": "runs/swin2-resumed/tensorboard",
            },
        },
        "runs/swin2-resumed/best.pt",
    )

    assert provenance is not None
    assert "epoch=002" in provenance
    assert "backbone=swinv2_cr_tiny_384" in provenance
    assert "preset=balanced" in provenance
    assert "scheduler=cosine" in provenance
    assert "best_val_kp_error=12.3456" in provenance
    assert "run=runs/swin2-resumed" in provenance


def test_format_eval_metadata_includes_provenance_fields() -> None:
    cfg = Config(batch_size=8, device="cpu")

    metadata = format_eval_metadata(
        {
            "epoch": 2,
            "cfg": {
                "backbone": "swinv2_cr_tiny_384",
                "backbone_preset": "balanced",
                "scheduler": "cosine",
            },
            "best_val_loss": 0.0247,
            "best_val_kp_error": 12.3456,
            "train_run": {
                "outdir": "runs/swin2-resumed",
                "resume_checkpoint": "runs/old/best.pt",
            },
            "tensorboard": {
                "logdir": "runs/swin2-resumed/tensorboard",
            },
        },
        checkpoint_path="runs/swin2-resumed/best.pt",
        test_data="generated/rodent_annotations_test.csv",
        cfg=cfg,
    )

    assert metadata is not None
    assert '"checkpoint": "runs/swin2-resumed/best.pt"' in metadata
    assert '"test_data": "generated/rodent_annotations_test.csv"' in metadata
    assert '"scheduler": "cosine"' in metadata
    assert '"best_val_kp_error": 12.3456' in metadata
    assert '"eval_batch_size": 8' in metadata
    assert '"eval_device": "cpu"' in metadata


def test_format_kp_error_regression_alert_reports_deltas() -> None:
    alert = format_kp_error_regression_alert(
        epoch=3,
        current_kp_error=15.5,
        previous_kp_error=14.0,
        best_kp_error=13.25,
    )

    assert alert is not None
    assert "epoch 003" in alert
    assert "current=15.5000" in alert
    assert "previous=14.0000" in alert
    assert "delta_prev=+1.5000" in alert
    assert "best=13.2500" in alert
    assert "delta_best=+2.2500" in alert


def test_main_allows_resume_checkpoint_without_template(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_path = tmp_path / "resume.pt"
    checkpoint_path.write_bytes(b"resume")
    monkeypatch.setattr(swin2_model, "run_train", lambda args: 0)

    result = main(
        [
            "--mode",
            "train",
            "--train-data",
            "generated/rodent_annotations_train.csv",
            "--outdir",
            str(tmp_path / "out"),
            "--resume-checkpoint",
            str(checkpoint_path),
        ]
    )

    assert result == 0


def test_main_requires_checkpoint_in_test_mode() -> None:
    with pytest.raises(SystemExit):
        main(["--mode", "test", "--test-data", "generated/rodent_annotations_test.csv"])
