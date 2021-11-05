#!/bin/bash

declare -a ModelVariantsArray=(
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

for variant in ${ModelVariantsArray[*]}; do
    echo "${variant}"
    nameSplit=(${variant//-/ })
    depth="${nameSplit[2]}"

    # Export weights
    python scripts/convert_weights.py \
        --depth "${depth}" \
        --ckpt weights/original_weights/"${variant}" \
        --output weights/"${variant}".h5 \
        --use_ema

    # Export no top
    python scripts/convert_weights.py \
        --depth "${depth}" \
        --ckpt weights/original_weights/"${variant}" \
        --output weights/"${variant}"_notop.h5 \
        --noinclude_top \
        --use_ema

done
