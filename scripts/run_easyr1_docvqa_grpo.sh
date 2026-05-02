#!/bin/bash

set -euo pipefail
set -x

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-VL-8B-Thinking}"
CONFIG_PATH="${CONFIG_PATH:-configs/easyr1/qwen3vl_8b_docvqa_grpo_lora.yaml}"
TRAIN_FILES="${TRAIN_FILES:-data/docvqa_easyr1/train.jsonl}"
VAL_FILES="${VAL_FILES:-data/docvqa_easyr1/val.jsonl}"
IMAGE_DIR="${IMAGE_DIR:-data/docvqa_easyr1}"
NGPUS="${NGPUS:-1}"

python3 -m verl.trainer.main \
  config="${CONFIG_PATH}" \
  data.train_files="${TRAIN_FILES}" \
  data.val_files="${VAL_FILES}" \
  data.image_dir="${IMAGE_DIR}" \
  worker.actor.model.model_path="${MODEL_PATH}" \
  trainer.n_gpus_per_node="${NGPUS}"
