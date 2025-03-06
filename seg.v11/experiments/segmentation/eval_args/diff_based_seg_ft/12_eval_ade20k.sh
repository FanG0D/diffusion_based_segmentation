#!/usr/bin/env bash
set -e
set -x

# add
checkpoint="diff_based_seg"

# add
python Diff_based_seg/eval.py \
    --base_data_dir="train_dataset" \
    --dataset_config Diff_based_seg/config/dataset/data_ade20k_val.yaml \
    --prediction_dir="experiments/segmentation/diffbasedseg/$checkpoint/ade20k_test/prediction" \
    --output_dir="experiments/segmentation/diffbasedseg/$checkpoint/ade20k_test/eval_metric"