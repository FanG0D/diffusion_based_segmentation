#!/usr/bin/env bash

set -e
set -x

# add
checkpoint="diff_based_seg"

checkpoint_path="$checkpoint"

# add
python Diff_based_seg/vis.py \
    --seed 1234 \
    --checkpoint="$checkpoint_path" \
    --base_data_dir="train_dataset" \
    --processing_res 0 \
    --dataset_config Diff_based_seg/config/dataset/data_ade20k_vis.yaml \
    --output_dir="experiments/segmentation/diffbasedseg/$checkpoint/ade20k_test/visualization" \