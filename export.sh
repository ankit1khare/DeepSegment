#!/bin/bash

python3 export_model.py \
	--checkpoint_path="bucket-model/model.ckpt-60000" \
	--export_path="frozen_inference_graph.pb" \
	--model_variant="xception_65" \
    --atrous_rates=12 \
    --atrous_rates=24 \
    --atrous_rates=36 \
    --output_stride=8 \
    --decoder_output_stride=4 \
	--crop_size=721 \
    --crop_size=1281 \
	--num_classes=16
