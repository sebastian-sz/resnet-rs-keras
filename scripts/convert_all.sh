#!/bin/bash

python scripts/convert_weights.py \
  --depth 50 \
  --ckpt weights/original_weights/resnet-rs-50-i160 \
  --output weights/resnetrs-50-i224.h5 \
#  --use_ema
