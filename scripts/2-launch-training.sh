#!/bin/bash
# Shell script for multilingual-llm pipeline
python train.py \
  --dataset s3://example-bucket-name/data/multilingual-support.json \
  --epochs 3 \
  --batch-size 16 \
  --output-dir ./model