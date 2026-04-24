#!/bin/bash
# Create the SwiftDiff inference environment.

mkdir -p ./checkpoints
mkdir -p ./data
mkdir -p ./inference

source conda activate
conda create -y -n swiftdiff python=3.8
conda activate swiftdiff

pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 \
    -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirement.txt
