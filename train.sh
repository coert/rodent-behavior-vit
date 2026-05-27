#!/usr/bin/env bash

uv run python swin2_classifier.py \
  --mode train \
  --train-data generated/rodent_annotations_train.csv \
  --val-data generated/rodent_annotations_val.csv \
  --outdir runs/swin2-classifier \
  --epochs 10 \
  --num-workers 0 \
  --batch-size 12 \
  --backbone swinv2_cr_tiny_384 \
  --backbone-preset balanced \
  --init-checkpoint runs/swin2-balanced-bs48/best.pt \
  --tensorboard-dir runs/swin2-classifier/tensorboard