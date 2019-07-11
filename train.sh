#!/bin/bash
#!117
#!117
#!last_layers_contain_logits_only=True
#!--initialize_last_layer=False

PATH_TO_INITIAL_CHECKPOINT="deeplabv3_cityscapes_train/model.ckpt"
PATH_TO_TRAIN_DIR="test_model"
PATH_TO_DATASET="datasets/test_dataset_40%/tfrecord"

python3 train.py \
  --logtostderr \
  --num_clones=1 \
  --training_number_of_steps=30000 \
  --model_variant="xception_65" \
  --train_logdir=${PATH_TO_TRAIN_DIR} \
  --save_interval_secs=600 \
  --save_summaries_secs=300 \
  --save_summaries_images=True \
  --base_learning_rate=0.007 \
  --tf_initial_checkpoint=${PATH_TO_INITIAL_CHECKPOINT} \
  --train_batch_size=25 \
  --fine_tune_batch_norm=True \
  --initialize_last_layer=False \
  --last_layers_contain_logits_only=True \
  --train_crop_size=117 \ 
  --train_crop_size=117 \ 
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --decoder_output_stride=4 \
  --dataset="safe_ai" \
  --train_split="train" \
  --dataset_dir=${PATH_TO_DATASET}  