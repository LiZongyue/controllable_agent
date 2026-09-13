#!/usr/bin/env bash
set -euo pipefail

cd /data/fanfeng/controllable_agent

gpu=5
shards=12
output_root=analysis_outputs/probe_stack3_1500000_legacy
session_prefix=probe_s3_1500

mkdir -p "${output_root}"

for ((index = 0; index < shards; index++)); do
  shard=$(printf '%02d' "${index}")
  output_dir="${output_root}/shard_${shard}"
  session="${session_prefix}_s${shard}_g${gpu}"
  mkdir -p "${output_dir}"
  if tmux has-session -t "${session}" 2>/dev/null; then
    continue
  fi
  tmux new-session -d -s "${session}" \
    "cd /data/fanfeng/controllable_agent && \
     HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=${gpu} PYTHONUNBUFFERED=1 \
     python linear_probe_rollouts.py \
       --no-cnn \
       --dino-run-glob '20260727_173900_dino_cls_stack3_seed1_*' \
       --dino-seeds 1 \
       --steps 1500000 \
       --target visible_pose,motion \
       --episodes 25 \
       --max-samples 20000 \
       --probe-epochs 10 \
       --probe-types linear,mlp \
       --no-dino-raw \
       --num-shards ${shards} \
       --shard-index ${index} \
       --output-dir ${output_dir} \
       --no-finalize \
       > ${output_dir}/run.log 2>&1"
done

echo "Launched ${shards} stack-3 probe shards on GPU ${gpu}; outputs: ${output_root}"
