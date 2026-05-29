#!/usr/bin/env python3
"""Recode classifier videos with a bottom classification timeline bar."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import ffmpeg
import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from sklearn.metrics import (
    auc,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from tqdm import tqdm

VIDEO_SUFFIX = "_classified.mp4"
CSV_SUFFIX = "_classifications.csv"
OUTPUT_SUFFIX = "_classified_timeline.mp4"
PR_CURVE_FILENAME = "classifier_precision_recall_curve.png"
DEFAULT_GROUND_TRUTH_CSV = Path(
    "Translational neuroimaging group - rodents/video_data.csv"
)
DEFAULT_ANNOTATIONS_CSV = Path("generated/rodent_annotations_test.csv")
BAR_HEIGHT = 30
BAR_OPACITY = 0.67
SLIDER_WIDTH = 3
GRAB_COLOR_BGR = np.array([0, 180, 0], dtype=np.uint8)
NO_GRAB_COLOR_BGR = np.array([0, 0, 220], dtype=np.uint8)
GROUND_TRUTH_GRAB_COLOR_BGR = np.array([0, 255, 255], dtype=np.uint8)
GROUND_TRUTH_REFERENCE_COLOR_BGR = np.array([42, 42, 165], dtype=np.uint8)
GROUND_TRUTH_NO_GRAB_COLOR_BGR = np.array([180, 0, 180], dtype=np.uint8)
SLIDER_COLOR_BGR = np.array([230, 216, 173], dtype=np.uint8)
REQUIRED_COLUMNS = {"frame_number", "prediction"}
METRIC_COLUMNS = {"frame_number", "prediction", "probability"}
ANNOTATION_COLUMNS = {"recording_id", "clip_id", "frame_idx", "label"}
VALID_PREDICTIONS = {"grab", "no-grab"}
GROUND_TRUTH_VIDEO_COLUMN = "Naam"
GROUND_TRUTH_START_COLUMN = "timestamp start"
GROUND_TRUTH_END_COLUMN = "timestamp end"


@dataclass(frozen=True)
class VideoCsvPair:
    recording_id: str
    video_path: Path
    csv_path: Path
    output_path: Path


@dataclass(frozen=True)
class FrameRange:
    start_frame: int
    end_frame: int


@dataclass(frozen=True)
class ClassifierMetrics:
    precision: float
    recall: float
    pr_auc: float
    f1: float
    true_labels: np.ndarray
    scores: np.ndarray
    predicted_labels: np.ndarray
    pr_precisions: np.ndarray
    pr_recalls: np.ndarray


def recording_id_from_video_path(video_path: str | Path) -> str:
    name = Path(video_path).name
    if not name.endswith(VIDEO_SUFFIX):
        raise ValueError(f"Expected video filename to end with {VIDEO_SUFFIX}: {name}")
    return name[: -len(VIDEO_SUFFIX)]


def find_video_csv_pairs(input_dir: str | Path) -> list[VideoCsvPair]:
    input_dir = Path(input_dir)
    pairs: list[VideoCsvPair] = []
    for video_path in sorted(input_dir.glob(f"*{VIDEO_SUFFIX}")):
        recording_id = recording_id_from_video_path(video_path)
        csv_path = input_dir / f"{recording_id}{CSV_SUFFIX}"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Missing classification CSV for {video_path}: {csv_path}"
            )
        pairs.append(
            VideoCsvPair(
                recording_id=recording_id,
                video_path=video_path,
                csv_path=csv_path,
                output_path=input_dir / f"{recording_id}{OUTPUT_SUFFIX}",
            )
        )
    if not pairs:
        raise ValueError(f"No {VIDEO_SUFFIX} files found in {input_dir}")
    return pairs


def validate_and_extract_predictions(
    classifications: pd.DataFrame,
    *,
    frame_count: int,
) -> list[str]:
    missing = REQUIRED_COLUMNS - set(classifications.columns)
    if missing:
        raise ValueError(
            f"classification CSV is missing columns: {', '.join(sorted(missing))}"
        )
    if len(classifications) != frame_count:
        raise ValueError(
            f"CSV row count {len(classifications)} does not match video frame count {frame_count}"
        )

    ordered = classifications.sort_values("frame_number")
    expected_frame_numbers = list(range(frame_count))
    actual_frame_numbers = ordered["frame_number"].astype(int).tolist()
    if actual_frame_numbers != expected_frame_numbers:
        raise ValueError(
            "classification CSV frame_number values must be contiguous from 0"
        )

    predictions = ordered["prediction"].astype(str).tolist()
    invalid = sorted(set(predictions) - VALID_PREDICTIONS)
    if invalid:
        raise ValueError(f"Invalid prediction labels: {', '.join(invalid)}")
    return predictions


def validate_and_extract_probabilities(
    classifications: pd.DataFrame,
    *,
    frame_count: int,
) -> list[float]:
    missing = METRIC_COLUMNS - set(classifications.columns)
    if missing:
        raise ValueError(
            f"classification CSV is missing columns: {', '.join(sorted(missing))}"
        )
    if len(classifications) != frame_count:
        raise ValueError(
            f"CSV row count {len(classifications)} does not match video frame count {frame_count}"
        )

    ordered = classifications.sort_values("frame_number")
    expected_frame_numbers = list(range(frame_count))
    actual_frame_numbers = ordered["frame_number"].astype(int).tolist()
    if actual_frame_numbers != expected_frame_numbers:
        raise ValueError(
            "classification CSV frame_number values must be contiguous from 0"
        )

    probabilities = ordered["probability"].astype(float).tolist()
    invalid = [
        probability for probability in probabilities if not 0.0 <= probability <= 1.0
    ]
    if invalid:
        raise ValueError("classification probabilities must be between 0 and 1")
    return probabilities


def timeline_frame_index(x: int, *, width: int, frame_count: int) -> int:
    if width < 1:
        raise ValueError("width must be >= 1")
    if frame_count < 1:
        raise ValueError("frame_count must be >= 1")
    return min(int(x * frame_count / width), frame_count - 1)


def build_timeline_bar(
    predictions: Sequence[str],
    *,
    width: int,
    height: int = BAR_HEIGHT,
    grab_color: np.ndarray = GRAB_COLOR_BGR,
    no_grab_color: np.ndarray = NO_GRAB_COLOR_BGR,
) -> np.ndarray:
    if width < 1:
        raise ValueError("width must be >= 1")
    if height < 1:
        raise ValueError("height must be >= 1")
    if not predictions:
        raise ValueError("predictions must not be empty")

    invalid = sorted(set(predictions) - VALID_PREDICTIONS)
    if invalid:
        raise ValueError(f"Invalid prediction labels: {', '.join(invalid)}")

    bar = np.empty((height, width, 3), dtype=np.uint8)
    frame_count = len(predictions)
    for x in range(width):
        prediction = predictions[
            timeline_frame_index(x, width=width, frame_count=frame_count)
        ]
        bar[:, x] = grab_color if prediction == "grab" else no_grab_color
    return bar


def load_ground_truth_csv(path: str | Path) -> pd.DataFrame:
    ground_truth = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    ground_truth = ground_truth.rename(columns=lambda column: str(column).strip())
    missing = {
        GROUND_TRUTH_VIDEO_COLUMN,
        GROUND_TRUTH_START_COLUMN,
        GROUND_TRUTH_END_COLUMN,
    } - set(ground_truth.columns)
    if missing:
        raise ValueError(
            f"ground-truth CSV is missing columns: {', '.join(sorted(missing))}"
        )
    return ground_truth


def is_missing_scalar(value: object) -> bool:
    if value is None or value is pd.NA:
        return True
    return isinstance(value, (float, np.floating)) and bool(np.isnan(value))


def parse_timestamp_seconds(value: object) -> float | None:
    if is_missing_scalar(value):
        return None
    text = str(value).strip()
    if not text:
        return None

    parts = text.split(":")
    if not 1 <= len(parts) <= 3:
        raise ValueError(f"Invalid timestamp: {text}")
    try:
        numbers = [float(part.replace(",", ".")) for part in parts]
    except ValueError as error:
        raise ValueError(f"Invalid timestamp: {text}") from error

    seconds = 0.0
    for number in numbers:
        seconds = seconds * 60.0 + number
    return seconds


def clip_frame_range(
    start_frame: int,
    end_frame: int,
    *,
    frame_count: int,
) -> FrameRange | None:
    if frame_count < 1:
        raise ValueError("frame_count must be >= 1")
    start_frame = max(start_frame, 0)
    end_frame = min(end_frame, frame_count - 1)
    if end_frame < start_frame:
        return None
    return FrameRange(start_frame=start_frame, end_frame=end_frame)


def ranges_to_predictions(
    ranges: Sequence[FrameRange],
    *,
    frame_count: int,
) -> list[str]:
    if frame_count < 1:
        raise ValueError("frame_count must be >= 1")
    predictions = ["no-grab"] * frame_count
    for frame_range in ranges:
        for frame_index in range(frame_range.start_frame, frame_range.end_frame + 1):
            predictions[frame_index] = "grab"
    return predictions


def video_data_ranges_for_recording(
    ground_truth: pd.DataFrame,
    *,
    recording_id: str,
    frame_count: int,
    fps: float,
) -> list[FrameRange]:
    if frame_count < 1:
        raise ValueError("frame_count must be >= 1")
    if fps <= 0:
        raise ValueError("fps must be > 0")

    normalized = ground_truth.rename(columns=lambda column: str(column).strip())
    missing = {
        GROUND_TRUTH_VIDEO_COLUMN,
        GROUND_TRUTH_START_COLUMN,
        GROUND_TRUTH_END_COLUMN,
    } - set(normalized.columns)
    if missing:
        raise ValueError(
            f"ground-truth CSV is missing columns: {', '.join(sorted(missing))}"
        )

    video_stems = normalized[GROUND_TRUTH_VIDEO_COLUMN].map(
        lambda name: "" if is_missing_scalar(name) else Path(str(name)).stem
    )
    rows = normalized[video_stems == recording_id]
    if rows.empty:
        raise ValueError(f"No ground-truth rows found for recording_id: {recording_id}")

    ranges: list[FrameRange] = []
    for _, row in rows.iterrows():
        start_sec = parse_timestamp_seconds(row[GROUND_TRUTH_START_COLUMN])
        end_sec = parse_timestamp_seconds(row[GROUND_TRUTH_END_COLUMN])
        if start_sec is None or end_sec is None:
            continue
        if end_sec < start_sec:
            raise ValueError(
                f"Ground-truth interval ends before it starts for {recording_id}: "
                f"{start_sec} > {end_sec}"
            )

        frame_range = clip_frame_range(
            int(round(start_sec * fps)),
            int(round(end_sec * fps)),
            frame_count=frame_count,
        )
        if frame_range is not None:
            ranges.append(frame_range)
    return ranges


def ground_truth_predictions_for_recording(
    ground_truth: pd.DataFrame,
    *,
    recording_id: str,
    frame_count: int,
    fps: float,
) -> list[str]:
    ranges = video_data_ranges_for_recording(
        ground_truth,
        recording_id=recording_id,
        frame_count=frame_count,
        fps=fps,
    )
    return ranges_to_predictions(ranges, frame_count=frame_count)


def load_annotations_csv(path: str | Path) -> pd.DataFrame:
    annotations = pd.read_csv(path)
    annotations = annotations.rename(columns=lambda column: str(column).strip())
    missing = ANNOTATION_COLUMNS - set(annotations.columns)
    if missing:
        raise ValueError(
            f"annotations CSV is missing columns: {', '.join(sorted(missing))}"
        )
    return annotations


def parse_clip_timestamp_seconds(token: str) -> int:
    if not re.fullmatch(r"\d{6}", token):
        raise ValueError(f"Invalid clip timestamp token: {token}")
    minutes = int(token[:-2])
    seconds = int(token[-2:])
    if seconds >= 60:
        raise ValueError(f"Invalid clip timestamp seconds in token: {token}")
    return minutes * 60 + seconds


def clip_start_seconds_from_clip_id(clip_id: object) -> int:
    match = re.search(r"_clip_(\d{6})_(\d{6})$", str(clip_id))
    if match is None:
        raise ValueError(f"Could not parse clip_id: {clip_id}")
    return parse_clip_timestamp_seconds(match.group(1))


def annotation_ranges_for_recording(
    annotations: pd.DataFrame,
    *,
    recording_id: str,
    frame_count: int,
    fps: float,
    prefix_buffer_frames: int,
    postfix_buffer_frames: int,
) -> list[FrameRange]:
    if frame_count < 1:
        raise ValueError("frame_count must be >= 1")
    if fps <= 0:
        raise ValueError("fps must be > 0")
    if prefix_buffer_frames < 0 or postfix_buffer_frames < 0:
        raise ValueError("buffer values must be >= 0")

    normalized = annotations.rename(columns=lambda column: str(column).strip())
    missing = ANNOTATION_COLUMNS - set(normalized.columns)
    if missing:
        raise ValueError(
            f"annotations CSV is missing columns: {', '.join(sorted(missing))}"
        )

    rows = normalized[normalized["recording_id"].astype(str) == recording_id]
    if rows.empty:
        raise ValueError(f"No annotation rows found for recording_id: {recording_id}")

    ranges: list[FrameRange] = []
    for _, row in rows.iterrows():
        if int(row["label"]) != 1:
            continue
        clip_start_sec = clip_start_seconds_from_clip_id(row["clip_id"])
        full_video_frame = int(round(clip_start_sec * fps)) + int(row["frame_idx"])
        frame_range = clip_frame_range(
            full_video_frame - prefix_buffer_frames,
            full_video_frame + postfix_buffer_frames,
            frame_count=frame_count,
        )
        if frame_range is not None:
            ranges.append(frame_range)
    return ranges


def annotation_predictions_for_recording(
    annotations: pd.DataFrame,
    *,
    recording_id: str,
    frame_count: int,
    fps: float,
    prefix_buffer_frames: int,
    postfix_buffer_frames: int,
) -> list[str]:
    ranges = annotation_ranges_for_recording(
        annotations,
        recording_id=recording_id,
        frame_count=frame_count,
        fps=fps,
        prefix_buffer_frames=prefix_buffer_frames,
        postfix_buffer_frames=postfix_buffer_frames,
    )
    return ranges_to_predictions(ranges, frame_count=frame_count)


def ranges_overlap(left: FrameRange, right: FrameRange) -> bool:
    return left.start_frame <= right.end_frame and right.start_frame <= left.end_frame


def warn_unmatched_video_data_ranges(
    *,
    recording_id: str,
    video_data_ranges: Sequence[FrameRange],
    annotation_ranges: Sequence[FrameRange],
) -> None:
    for video_data_range in video_data_ranges:
        if any(
            ranges_overlap(video_data_range, annotation_range)
            for annotation_range in annotation_ranges
        ):
            continue
        print(
            "WARNING: video_data ground truth range has no annotation-derived overlap: "
            f"{recording_id} frames {video_data_range.start_frame}-{video_data_range.end_frame}"
        )


def build_ground_truth_overlay_bar(
    *,
    annotation_ranges: Sequence[FrameRange],
    video_data_ranges: Sequence[FrameRange],
    frame_count: int,
    width: int,
    height: int = BAR_HEIGHT,
) -> np.ndarray:
    if width < 1:
        raise ValueError("width must be >= 1")
    if height < 1:
        raise ValueError("height must be >= 1")
    if frame_count < 1:
        raise ValueError("frame_count must be >= 1")

    bar = np.empty((height, width, 3), dtype=np.uint8)
    bar[:] = GROUND_TRUTH_NO_GRAB_COLOR_BGR
    for x in range(width):
        frame_index = timeline_frame_index(x, width=width, frame_count=frame_count)
        if any(
            video_data_range.start_frame <= frame_index <= video_data_range.end_frame
            for video_data_range in video_data_ranges
        ):
            bar[:, x] = GROUND_TRUTH_REFERENCE_COLOR_BGR
        if any(
            annotation_range.start_frame <= frame_index <= annotation_range.end_frame
            for annotation_range in annotation_ranges
        ):
            bar[:, x] = GROUND_TRUTH_GRAB_COLOR_BGR
    return bar


def stack_timeline_bars(*bars: np.ndarray) -> np.ndarray:
    if not bars:
        raise ValueError("at least one bar is required")
    widths: set[int] = set()
    for bar in bars:
        if bar.ndim != 3 or bar.shape[2] != 3:
            raise ValueError("all bars must have shape HxWx3")
        widths.add(bar.shape[1])
    if len(widths) != 1:
        raise ValueError("all bars must have matching widths")
    return np.vstack(bars)


def labels_to_binary(labels: Sequence[str]) -> np.ndarray:
    invalid = sorted(set(labels) - VALID_PREDICTIONS)
    if invalid:
        raise ValueError(f"Invalid prediction labels: {', '.join(invalid)}")
    return np.array([1 if label == "grab" else 0 for label in labels], dtype=np.int64)


def calculate_classifier_metrics(
    *,
    true_labels: Sequence[str],
    probabilities: Sequence[float],
    predictions: Sequence[str],
) -> ClassifierMetrics:
    if len(true_labels) != len(probabilities) or len(true_labels) != len(predictions):
        raise ValueError(
            "true_labels, probabilities, and predictions must have matching lengths"
        )
    if not true_labels:
        raise ValueError("metrics require at least one frame")

    y_true = labels_to_binary(true_labels)
    y_scores = np.array(probabilities, dtype=np.float64)
    y_pred = labels_to_binary(predictions)

    if len(set(y_true.tolist())) < 2:
        raise ValueError("metrics require both grab and no-grab ground-truth frames")

    pr_precisions, pr_recalls, _ = precision_recall_curve(y_true, y_scores)
    return ClassifierMetrics(
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        pr_auc=float(auc(pr_recalls, pr_precisions)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        true_labels=y_true,
        scores=y_scores,
        predicted_labels=y_pred,
        pr_precisions=pr_precisions,
        pr_recalls=pr_recalls,
    )


def collect_metric_inputs(
    pair: VideoCsvPair,
    *,
    ground_truth: pd.DataFrame,
) -> tuple[list[str], list[float], list[str]]:
    capture = cv2.VideoCapture(str(pair.video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {pair.video_path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()

    if fps <= 0:
        raise ValueError(f"Could not determine FPS for video: {pair.video_path}")
    if frame_count <= 0:
        raise ValueError(
            f"Could not determine frame count for video: {pair.video_path}"
        )

    classifications = pd.read_csv(pair.csv_path)
    predictions = validate_and_extract_predictions(
        classifications, frame_count=frame_count
    )
    probabilities = validate_and_extract_probabilities(
        classifications, frame_count=frame_count
    )
    true_labels = ground_truth_predictions_for_recording(
        ground_truth,
        recording_id=pair.recording_id,
        frame_count=frame_count,
        fps=fps,
    )
    return true_labels, probabilities, predictions


def save_precision_recall_diagram(
    metrics: ClassifierMetrics,
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = Figure(figsize=(7, 5), dpi=150)
    FigureCanvas(fig)
    ax = fig.subplots()
    ax.plot(metrics.pr_recalls, metrics.pr_precisions, color="#1f77b4", linewidth=2)
    ax.set_title(
        f"Precision-Recall Curve (AUC={metrics.pr_auc:.3f}, F1={metrics.f1:.3f})"
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path)


def slider_x_for_frame(frame_number: int, *, frame_count: int, width: int) -> int:
    if width < 1:
        raise ValueError("width must be >= 1")
    if frame_count < 1:
        raise ValueError("frame_count must be >= 1")
    if frame_count == 1 or width == 1:
        return 0
    clamped_frame = min(max(frame_number, 0), frame_count - 1)
    return round(clamped_frame / (frame_count - 1) * (width - 1))


def draw_timeline_on_frame(
    frame_bgr: np.ndarray,
    timeline_bar: np.ndarray,
    *,
    frame_number: int,
    frame_count: int,
    slider_width: int = SLIDER_WIDTH,
    bar_opacity: float = BAR_OPACITY,
) -> np.ndarray:
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("frame_bgr must have shape HxWx3")
    if timeline_bar.ndim != 3 or timeline_bar.shape[2] != 3:
        raise ValueError("timeline_bar must have shape HxWx3")
    height, width = frame_bgr.shape[:2]
    bar_height, bar_width = timeline_bar.shape[:2]
    if bar_width != width:
        raise ValueError("timeline bar width must match frame width")
    if bar_height > height:
        raise ValueError("timeline bar height must fit inside frame")
    if slider_width < 1:
        raise ValueError("slider_width must be >= 1")
    if not 0.0 <= bar_opacity <= 1.0:
        raise ValueError("bar_opacity must be between 0 and 1")

    output = frame_bgr.copy()
    bar_slice = output[height - bar_height : height, :, :]
    output[height - bar_height : height, :, :] = cv2.addWeighted(
        timeline_bar,
        bar_opacity,
        bar_slice,
        1.0 - bar_opacity,
        0.0,
    )
    slider_x = slider_x_for_frame(frame_number, frame_count=frame_count, width=width)
    half_width = slider_width // 2
    start_x = max(slider_x - half_width, 0)
    end_x = min(start_x + slider_width, width)
    start_x = max(end_x - slider_width, 0)
    output[height - bar_height : height, start_x:end_x, :] = SLIDER_COLOR_BGR
    return output


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


def recode_video_with_timeline(
    pair: VideoCsvPair,
    *,
    ground_truth: pd.DataFrame,
    annotations: pd.DataFrame,
    prefix_buffer_frames: int,
    postfix_buffer_frames: int,
    bar_height: int = BAR_HEIGHT,
    slider_width: int = SLIDER_WIDTH,
) -> int:
    capture = cv2.VideoCapture(str(pair.video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {pair.video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0:
        raise ValueError(f"Could not determine FPS for video: {pair.video_path}")
    if frame_count <= 0:
        raise ValueError(
            f"Could not determine frame count for video: {pair.video_path}"
        )
    if width <= 0 or height <= 0:
        raise ValueError(f"Could not determine frame size for video: {pair.video_path}")

    classifications = pd.read_csv(pair.csv_path)
    predictions = validate_and_extract_predictions(
        classifications, frame_count=frame_count
    )
    classifier_bar = build_timeline_bar(predictions, width=width, height=bar_height)
    annotation_ranges = annotation_ranges_for_recording(
        annotations,
        recording_id=pair.recording_id,
        frame_count=frame_count,
        fps=fps,
        prefix_buffer_frames=prefix_buffer_frames,
        postfix_buffer_frames=postfix_buffer_frames,
    )
    video_data_ranges = video_data_ranges_for_recording(
        ground_truth,
        recording_id=pair.recording_id,
        frame_count=frame_count,
        fps=fps,
    )
    warn_unmatched_video_data_ranges(
        recording_id=pair.recording_id,
        video_data_ranges=video_data_ranges,
        annotation_ranges=annotation_ranges,
    )
    ground_truth_bar = build_ground_truth_overlay_bar(
        annotation_ranges=annotation_ranges,
        video_data_ranges=video_data_ranges,
        frame_count=frame_count,
        width=width,
        height=bar_height,
    )
    timeline_bar = stack_timeline_bars(classifier_bar, ground_truth_bar)
    pair.output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = build_ffmpeg_writer(pair.output_path, width=width, height=height, fps=fps)
    written_frames = 0

    try:
        with tqdm(total=frame_count, desc=pair.recording_id, leave=False) as progress:
            while True:
                ok, frame_bgr = capture.read()
                if not ok:
                    break
                annotated = draw_timeline_on_frame(
                    frame_bgr,
                    timeline_bar,
                    frame_number=written_frames,
                    frame_count=frame_count,
                    slider_width=slider_width,
                )
                writer.stdin.write(annotated.tobytes())
                written_frames += 1
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
        raise RuntimeError(f"ffmpeg failed for {pair.output_path}: {stderr}")
    if written_frames != frame_count:
        raise RuntimeError(
            f"Wrote {written_frames} frames for {pair.video_path}, expected {frame_count}"
        )
    return written_frames


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        default="generated/swin2_classifier_video_test",
        help="Directory containing *_classified.mp4 and *_classifications.csv files",
    )
    parser.add_argument("--bar-height", type=int, default=BAR_HEIGHT)
    parser.add_argument("--slider-width", type=int, default=SLIDER_WIDTH)
    parser.add_argument(
        "--ground-truth-csv",
        default=str(DEFAULT_GROUND_TRUTH_CSV),
        help="CSV containing original video_data ground-truth intervals",
    )
    parser.add_argument(
        "--annotations-csv",
        default=str(DEFAULT_ANNOTATIONS_CSV),
        help="CSV containing test annotation frames used for yellow overlay ranges",
    )
    parser.add_argument("--prefix-buffer-frames", type=int, default=30)
    parser.add_argument("--postfix-buffer-frames", type=int, default=30)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    input_dir = Path(args.input_dir)
    pairs = find_video_csv_pairs(input_dir)
    if args.prefix_buffer_frames < 0 or args.postfix_buffer_frames < 0:
        raise ValueError("buffer values must be >= 0")
    ground_truth = load_ground_truth_csv(args.ground_truth_csv)
    annotations = load_annotations_csv(args.annotations_csv)
    all_true_labels: list[str] = []
    all_probabilities: list[float] = []
    all_predictions: list[str] = []
    for pair in pairs:
        print(f"recoding {pair.video_path} -> {pair.output_path}")
        frame_count = recode_video_with_timeline(
            pair,
            ground_truth=ground_truth,
            annotations=annotations,
            prefix_buffer_frames=args.prefix_buffer_frames,
            postfix_buffer_frames=args.postfix_buffer_frames,
            bar_height=args.bar_height,
            slider_width=args.slider_width,
        )
        print(f"wrote {frame_count} frames: {pair.output_path}")
        true_labels, probabilities, predictions = collect_metric_inputs(
            pair,
            ground_truth=ground_truth,
        )
        all_true_labels.extend(true_labels)
        all_probabilities.extend(probabilities)
        all_predictions.extend(predictions)

    metrics = calculate_classifier_metrics(
        true_labels=all_true_labels,
        probabilities=all_probabilities,
        predictions=all_predictions,
    )
    pr_curve_path = input_dir / PR_CURVE_FILENAME
    save_precision_recall_diagram(metrics, pr_curve_path)
    print(f"precision: {metrics.precision:.6f}")
    print(f"recall: {metrics.recall:.6f}")
    print(f"auc: {metrics.pr_auc:.6f}")
    print(f"f1: {metrics.f1:.6f}")
    print(f"wrote precision-recall diagram: {pr_curve_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
