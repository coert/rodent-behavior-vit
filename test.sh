#!/usr/bin/env bash

uv run python swin2_classifier.py \
  --mode test \
  --test-data generated/rodent_annotations_test.csv \
  --checkpoint runs/swin2-classifier/best.pt

uv run python recode_classifier_timeline.py \
  --input-dir "generated/swin2_classifier_video_test" \
  --bar-height 30 \
  --slider-width 3 \
  --ground-truth-csv "Translational neuroimaging group - rodents/video_data.csv" \
  --annotations-csv "generated/rodent_annotations_test.csv" \
  --prefix-buffer-frames 5 \
  --postfix-buffer-frames 5