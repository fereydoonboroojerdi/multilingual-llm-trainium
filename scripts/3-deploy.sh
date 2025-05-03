#!/bin/bash
# Shell script for multilingual-llm pipeline
python deploy_trainium.py \
  --model s3://example-bucket-name/models/multilingual-llm/ \
  --role arn:aws:iam::123456789012:role/sagemaker-execution \
  --instance ml.trn1.2xlarge