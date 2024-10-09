#!/usr/bin/env bash

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.
export MASTER_PORT=6083
export CUDA_VISIBLE_DEVICES=0
export GPUS_PER_NODE=1

config_dir=/workspace/jaeyoung/StoryTeller/ONE-PEACE/one_peace/run_scripts/audio_text_retrieval/
config_name=base

torchrun --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} /workspace/jaeyoung/StoryTeller/ONE-PEACE/one_peace/train.py \
    --config-dir=${config_dir} \
    --config-name=${config_name}