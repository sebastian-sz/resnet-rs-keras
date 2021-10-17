#!/bin/bash

python scripts/convert_weights.py \
  --depth 50 \
  --ckpt weights/original_weights/resnet-rs-50-i160 \
  --output weights/resnetrs-50-i160.h5 \
  --use_ema

python scripts/convert_weights.py \
  --depth 101 \
  --ckpt weights/original_weights/resnet-rs-101-i160 \
  --output weights/resnetrs-101-i160.h5 \
  --use_ema

python scripts/convert_weights.py \
  --depth 101 \
  --ckpt weights/original_weights/resnet-rs-101-i192 \
  --output weights/resnetrs-101-i192.h5 \
  --use_ema

python scripts/convert_weights.py \
  --depth 152 \
  --ckpt weights/original_weights/resnet-rs-152-i192 \
  --output weights/resnetrs-152-i192.h5 \
  --use_ema

python scripts/convert_weights.py \
  --depth 152 \
  --ckpt weights/original_weights/resnet-rs-152-i224 \
  --output weights/resnetrs-152-i224.h5 \
  --use_ema

python scripts/convert_weights.py \
  --depth 152 \
  --ckpt weights/original_weights/resnet-rs-152-i256 \
  --output weights/resnetrs-152-i256.h5 \
  --use_ema

python scripts/convert_weights.py \
  --depth 200 \
  --ckpt weights/original_weights/resnet-rs-200-i256 \
  --output weights/resnetrs-200-i256.h5 \
  --use_ema

python scripts/convert_weights.py \
  --depth 270 \
  --ckpt weights/original_weights/resnet-rs-270-i256 \
  --output weights/resnetrs-270-i256.h5 \
  --use_ema

python scripts/convert_weights.py \
  --depth 350 \
  --ckpt weights/original_weights/resnet-rs-350-i256 \
  --output weights/resnetrs-350-i256.h5 \
  --use_ema

python scripts/convert_weights.py \
  --depth 350 \
  --ckpt weights/original_weights/resnet-rs-350-i320 \
  --output weights/resnetrs-350-i320.h5 \
  --use_ema

python scripts/convert_weights.py \
--depth 420 \
--ckpt weights/original_weights/resnet-rs-420-i320 \
--output weights/resnetrs-420-i320.h5 \
--use_ema
