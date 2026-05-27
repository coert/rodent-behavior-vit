import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recode_classifier_timeline import (
    BAR_HEIGHT,
    BAR_OPACITY,
    CSV_SUFFIX,
    GRAB_COLOR_BGR,
    GROUND_TRUTH_GRAB_COLOR_BGR,
    GROUND_TRUTH_NO_GRAB_COLOR_BGR,
    GROUND_TRUTH_REFERENCE_COLOR_BGR,
    NO_GRAB_COLOR_BGR,
    OUTPUT_SUFFIX,
    SLIDER_COLOR_BGR,
    VIDEO_SUFFIX,
    FrameRange,
    VideoCsvPair,
    annotation_predictions_for_recording,
    annotation_ranges_for_recording,
    build_ground_truth_overlay_bar,
    build_timeline_bar,
    calculate_classifier_metrics,
    clip_start_seconds_from_clip_id,
    collect_metric_inputs,
    draw_timeline_on_frame,
    find_video_csv_pairs,
    ground_truth_predictions_for_recording,
    parse_clip_timestamp_seconds,
    parse_timestamp_seconds,
    save_precision_recall_diagram,
    slider_x_for_frame,
    stack_timeline_bars,
    validate_and_extract_predictions,
    validate_and_extract_probabilities,
    warn_unmatched_video_data_ranges,
)


def test_find_video_csv_pairs_matches_expected_names(tmp_path: Path) -> None:
    recording_id = "rec_a"
    video_path = tmp_path / f"{recording_id}{VIDEO_SUFFIX}"
    csv_path = tmp_path / f"{recording_id}{CSV_SUFFIX}"
    video_path.write_bytes(b"not a real video")
    csv_path.write_text("frame_number,prediction\n0,grab\n")

    pairs = find_video_csv_pairs(tmp_path)

    assert len(pairs) == 1
    assert pairs[0].recording_id == recording_id
    assert pairs[0].video_path == video_path
    assert pairs[0].csv_path == csv_path
    assert pairs[0].output_path == tmp_path / f"{recording_id}{OUTPUT_SUFFIX}"


def test_find_video_csv_pairs_rejects_missing_csv(tmp_path: Path) -> None:
    (tmp_path / f"rec_a{VIDEO_SUFFIX}").write_bytes(b"not a real video")

    with pytest.raises(FileNotFoundError, match="Missing classification CSV"):
        find_video_csv_pairs(tmp_path)


def test_validate_predictions_rejects_mismatched_frame_count() -> None:
    classifications = pd.DataFrame(
        {"frame_number": [0, 1], "prediction": ["grab", "no-grab"]}
    )

    with pytest.raises(ValueError, match="row count"):
        validate_and_extract_predictions(classifications, frame_count=3)


def test_validate_predictions_rejects_invalid_prediction() -> None:
    classifications = pd.DataFrame({"frame_number": [0], "prediction": ["maybe"]})

    with pytest.raises(ValueError, match="Invalid prediction"):
        validate_and_extract_predictions(classifications, frame_count=1)


def test_build_timeline_bar_maps_relative_x_positions_to_predictions() -> None:
    bar = build_timeline_bar(
        ["grab", "grab", "no-grab", "no-grab"],
        width=8,
        height=2,
    )

    assert tuple(bar[0, 0]) == tuple(GRAB_COLOR_BGR)
    assert tuple(bar[0, 3]) == tuple(GRAB_COLOR_BGR)
    assert tuple(bar[0, 4]) == tuple(NO_GRAB_COLOR_BGR)
    assert tuple(bar[0, 7]) == tuple(NO_GRAB_COLOR_BGR)
    assert bar.shape == (2, 8, 3)


def test_slider_x_for_frame_places_first_middle_and_last_frames() -> None:
    assert slider_x_for_frame(0, frame_count=5, width=9) == 0
    assert slider_x_for_frame(2, frame_count=5, width=9) == 4
    assert slider_x_for_frame(4, frame_count=5, width=9) == 8


def test_draw_timeline_on_frame_replaces_bottom_bar_and_slider() -> None:
    frame = np.zeros((6, 8, 3), dtype=np.uint8)
    bar = build_timeline_bar(["grab", "no-grab"], width=8, height=2)

    output = draw_timeline_on_frame(
        frame,
        bar,
        frame_number=1,
        frame_count=2,
        slider_width=3,
    )

    assert np.all(output[:4] == 0)
    assert tuple(output[4, 7]) == tuple(SLIDER_COLOR_BGR)
    assert tuple(output[5, 6]) == tuple(SLIDER_COLOR_BGR)



def test_draw_timeline_on_frame_blends_bar_with_underlying_frame() -> None:
    frame = np.full((4, 4, 3), 100, dtype=np.uint8)
    bar = np.zeros((2, 4, 3), dtype=np.uint8)

    output = draw_timeline_on_frame(
        frame,
        bar,
        frame_number=0,
        frame_count=4,
        slider_width=1,
    )

    expected = round(100 * (1.0 - BAR_OPACITY))
    assert tuple(output[2, 1]) == (expected, expected, expected)
    assert tuple(output[2, 0]) == tuple(SLIDER_COLOR_BGR)



def test_parse_timestamp_seconds_accepts_minute_and_hour_formats() -> None:
    assert parse_timestamp_seconds("01:23") == 83
    assert parse_timestamp_seconds("01:02:03") == 3723
    assert parse_timestamp_seconds("") is None
    assert parse_timestamp_seconds(np.nan) is None


def test_ground_truth_predictions_mark_timestamp_intervals_as_grab() -> None:
    ground_truth = pd.DataFrame(
        {
            "Naam": ["rec_a.mp4", "rec_a.mp4", "other.mp4"],
            "timestamp start ": ["00:01", "00:03", "00:00"],
            "timestamp end": ["00:02", "00:03", "00:05"],
        }
    )

    predictions = ground_truth_predictions_for_recording(
        ground_truth,
        recording_id="rec_a",
        frame_count=9,
        fps=2.0,
    )

    assert predictions == [
        "no-grab",
        "no-grab",
        "grab",
        "grab",
        "grab",
        "no-grab",
        "grab",
        "no-grab",
        "no-grab",
    ]


def test_stack_timeline_bars_places_classifier_above_ground_truth() -> None:
    classifier_bar = build_timeline_bar(["grab"], width=4, height=BAR_HEIGHT)
    ground_truth_bar = build_timeline_bar(
        ["no-grab"],
        width=4,
        height=BAR_HEIGHT,
        grab_color=GROUND_TRUTH_GRAB_COLOR_BGR,
        no_grab_color=GROUND_TRUTH_NO_GRAB_COLOR_BGR,
    )

    stacked = stack_timeline_bars(classifier_bar, ground_truth_bar)

    assert stacked.shape == (BAR_HEIGHT * 2, 4, 3)
    assert tuple(stacked[0, 0]) == tuple(GRAB_COLOR_BGR)
    assert tuple(stacked[BAR_HEIGHT, 0]) == tuple(GROUND_TRUTH_NO_GRAB_COLOR_BGR)



def test_ground_truth_timeline_bar_uses_yellow_for_grab_and_purple_for_no_grab() -> None:
    bar = build_timeline_bar(
        ["grab", "no-grab"],
        width=2,
        height=1,
        grab_color=GROUND_TRUTH_GRAB_COLOR_BGR,
        no_grab_color=GROUND_TRUTH_NO_GRAB_COLOR_BGR,
    )

    assert tuple(bar[0, 0]) == tuple(GROUND_TRUTH_GRAB_COLOR_BGR)
    assert tuple(bar[0, 1]) == tuple(GROUND_TRUTH_NO_GRAB_COLOR_BGR)
    assert tuple(GROUND_TRUTH_GRAB_COLOR_BGR) == (0, 255, 255)
    assert tuple(GROUND_TRUTH_NO_GRAB_COLOR_BGR) == (180, 0, 180)


def test_slider_color_is_light_blue() -> None:
    assert tuple(SLIDER_COLOR_BGR) == (230, 216, 173)



def test_validate_probabilities_rejects_out_of_range_values() -> None:
    classifications = pd.DataFrame(
        {
            "frame_number": [0, 1],
            "prediction": ["grab", "no-grab"],
            "probability": [0.2, 1.2],
        }
    )

    with pytest.raises(ValueError, match="between 0 and 1"):
        validate_and_extract_probabilities(classifications, frame_count=2)


def test_calculate_classifier_metrics_uses_probabilities_and_predictions() -> None:
    metrics = calculate_classifier_metrics(
        true_labels=["grab", "grab", "no-grab", "no-grab"],
        probabilities=[0.9, 0.4, 0.8, 0.1],
        predictions=["grab", "no-grab", "grab", "no-grab"],
    )

    assert metrics.precision == pytest.approx(0.5)
    assert metrics.recall == pytest.approx(0.5)
    assert metrics.f1 == pytest.approx(0.5)
    assert 0.0 <= metrics.pr_auc <= 1.0
    assert metrics.true_labels.tolist() == [1, 1, 0, 0]
    assert metrics.predicted_labels.tolist() == [1, 0, 1, 0]


def test_collect_metric_inputs_uses_video_data_ground_truth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    csv_path = tmp_path / f"rec_a{CSV_SUFFIX}"
    csv_path.write_text(
        "frame_number,prediction,probability\n"
        "0,no-grab,0.1\n"
        "1,grab,0.8\n"
        "2,no-grab,0.2\n"
        "3,grab,0.9\n"
    )
    pair = VideoCsvPair(
        recording_id="rec_a",
        video_path=tmp_path / f"rec_a{VIDEO_SUFFIX}",
        csv_path=csv_path,
        output_path=tmp_path / f"rec_a{OUTPUT_SUFFIX}",
    )
    ground_truth = pd.DataFrame(
        {
            "Naam": ["rec_a.mp4"],
            "timestamp start": ["00:01"],
            "timestamp end": ["00:02"],
        }
    )

    class FakeCapture:
        def __init__(self, _path: str) -> None:
            pass

        def isOpened(self) -> bool:
            return True

        def get(self, prop: int) -> float:
            import cv2

            if prop == cv2.CAP_PROP_FPS:
                return 1.0
            if prop == cv2.CAP_PROP_FRAME_COUNT:
                return 4.0
            return 0.0

        def release(self) -> None:
            pass

    monkeypatch.setattr("recode_classifier_timeline.cv2.VideoCapture", FakeCapture)

    true_labels, probabilities, predictions = collect_metric_inputs(
        pair,
        ground_truth=ground_truth,
    )

    assert true_labels == ["no-grab", "grab", "grab", "no-grab"]
    assert probabilities == [0.1, 0.8, 0.2, 0.9]
    assert predictions == ["no-grab", "grab", "no-grab", "grab"]


def test_save_precision_recall_diagram_writes_png(tmp_path: Path) -> None:
    metrics = calculate_classifier_metrics(
        true_labels=["grab", "grab", "no-grab", "no-grab"],
        probabilities=[0.9, 0.8, 0.4, 0.1],
        predictions=["grab", "grab", "no-grab", "no-grab"],
    )
    output_path = tmp_path / "pr.png"

    save_precision_recall_diagram(metrics, output_path)

    assert output_path.exists()
    assert output_path.read_bytes().startswith(bytes([137, 80, 78, 71]))



def test_parse_clip_timestamp_seconds_treats_token_as_mmss() -> None:
    assert parse_clip_timestamp_seconds("000041") == 41
    assert parse_clip_timestamp_seconds("000352") == 232
    assert parse_clip_timestamp_seconds("001234") == 754


@pytest.mark.parametrize("token", ["000060", "bad", "12345", "1234567"])
def test_parse_clip_timestamp_seconds_rejects_invalid_tokens(token: str) -> None:
    with pytest.raises(ValueError):
        parse_clip_timestamp_seconds(token)


def test_clip_start_seconds_from_clip_id_reads_start_token() -> None:
    assert clip_start_seconds_from_clip_id("rec_clip_000352_000410") == 232


def test_annotation_ranges_use_positive_frames_and_buffers() -> None:
    annotations = pd.DataFrame(
        {
            "recording_id": ["rec_a", "rec_a", "rec_a"],
            "clip_id": [
                "rec_a_clip_000001_000003",
                "rec_a_clip_000001_000003",
                "rec_a_clip_000001_000003",
            ],
            "frame_idx": [2, 6, 8],
            "label": [1, 0, 1],
        }
    )

    ranges = annotation_ranges_for_recording(
        annotations,
        recording_id="rec_a",
        frame_count=20,
        fps=2.0,
        prefix_buffer_frames=1,
        postfix_buffer_frames=2,
    )

    assert ranges == [FrameRange(3, 6), FrameRange(9, 12)]


def test_annotation_predictions_clip_ranges_to_video_bounds() -> None:
    annotations = pd.DataFrame(
        {
            "recording_id": ["rec_a"],
            "clip_id": ["rec_a_clip_000000_000003"],
            "frame_idx": [1],
            "label": [1],
        }
    )

    predictions = annotation_predictions_for_recording(
        annotations,
        recording_id="rec_a",
        frame_count=5,
        fps=2.0,
        prefix_buffer_frames=10,
        postfix_buffer_frames=10,
    )

    assert predictions == ["grab"] * 5


def test_annotation_ranges_reject_missing_recording() -> None:
    annotations = pd.DataFrame(
        {
            "recording_id": ["rec_a"],
            "clip_id": ["rec_a_clip_000000_000003"],
            "frame_idx": [1],
            "label": [1],
        }
    )

    with pytest.raises(ValueError, match="No annotation rows"):
        annotation_ranges_for_recording(
            annotations,
            recording_id="rec_b",
            frame_count=5,
            fps=2.0,
            prefix_buffer_frames=1,
            postfix_buffer_frames=1,
        )


def test_ground_truth_overlay_bar_draws_annotation_yellow_over_video_data_brown() -> None:
    bar = build_ground_truth_overlay_bar(
        annotation_ranges=[FrameRange(2, 4)],
        video_data_ranges=[FrameRange(1, 3)],
        frame_count=6,
        width=6,
        height=1,
    )

    assert tuple(bar[0, 0]) == tuple(GROUND_TRUTH_NO_GRAB_COLOR_BGR)
    assert tuple(bar[0, 1]) == tuple(GROUND_TRUTH_REFERENCE_COLOR_BGR)
    assert tuple(bar[0, 2]) == tuple(GROUND_TRUTH_GRAB_COLOR_BGR)
    assert tuple(bar[0, 3]) == tuple(GROUND_TRUTH_GRAB_COLOR_BGR)
    assert tuple(bar[0, 4]) == tuple(GROUND_TRUTH_GRAB_COLOR_BGR)
    assert tuple(bar[0, 5]) == tuple(GROUND_TRUTH_NO_GRAB_COLOR_BGR)


def test_warn_unmatched_video_data_ranges_prints_only_uncovered_ranges(capsys) -> None:
    warn_unmatched_video_data_ranges(
        recording_id="rec_a",
        video_data_ranges=[FrameRange(1, 3), FrameRange(10, 12)],
        annotation_ranges=[FrameRange(3, 5)],
    )

    output = capsys.readouterr().out
    assert "frames 1-3" not in output
    assert "frames 10-12" in output
