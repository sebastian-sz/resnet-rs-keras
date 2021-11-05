#!/bin/bash

declare -a ModelNamesArray=(
"resnet-rs-50-i160"
"resnet-rs-101-i160"
"resnet-rs-101-i192"
"resnet-rs-152-i192"
"resnet-rs-152-i224"
"resnet-rs-152-i256"
"resnet-rs-200-i256"
"resnet-rs-270-i256"
"resnet-rs-350-i256"
"resnet-rs-350-i320"
"resnet-rs-420-i320"
)

for val1 in ${ModelNamesArray[*]}; do
     echo $val1
     mkdir -p weights/original_weights/$val1
     curl https://storage.googleapis.com/cloud-tpu-checkpoints/resnet-rs/$val1.tar.gz | tar xz -C weights/original_weights/$val1
done

python scripts/add_ckpt_to_missing_directories.py
