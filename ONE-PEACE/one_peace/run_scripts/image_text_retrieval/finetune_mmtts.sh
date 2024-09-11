#!/usr/bin/env bash

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.
export MASTER_PORT=6081
export CUDA_VISIBLE_DEVICES=0,1
export GPUS_PER_NODE=2

config_dir=/workspace/jaeyoung/StoryTeller/ONE-PEACE/one_peace/run_scripts/image_text_retrieval/
config_name=base
data=/workspace/jaeyoung/StoryTeller/merged_caption_MMTTS.csv
valid_data=/workspace/jaeyoung/StoryTeller/valid1000_merged_caption_MMTTS.csv
max_epoch=15
lr=[2e-6]
drop_path_rate=0.5

torchrun --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} /workspace/jaeyoung/StoryTeller/ONE-PEACE/one_peace/train.py \
    --config-dir=${config_dir} \
    --config-name=${config_name} \
    task.data=${data} \
    task.valid_data=${valid_data} \
    task.valid_file=${valid_file} \
    optimization.max_epoch=${max_epoch} \
    optimization.lr=${lr} \
    model.encoder.drop_path_rate=${drop_path_rate}