#!/bin/bash
PATH_TO_CHECKPOINT="test_model"
PATH_TO_EVAL_DIR="${PATH_TO_CHECKPOINT}/eval"
PATH_TO_DATASET="datasets/test_dataset_40%/tfrecord"

CUDA_VISIBLE_DEVICES="" python3 eval.py \
    --logtostderr \
    --eval_split="val" \
    --model_variant="xception_65" \
    --atrous_rates=6 \
    --atrous_rates=12 \
    --atrous_rates=18 \
    --output_stride=16 \
    --decoder_output_stride=4 \
    --eval_crop_size=513 \
    --eval_crop_size=515 \
    --dataset="safe_ai" \
    --checkpoint_dir="${PATH_TO_CHECKPOINT}" \
    --eval_logdir="${PATH_TO_EVAL_DIR}" \
    --dataset_dir="${PATH_TO_DATASET}" \
    --eval_interval_secs=60
