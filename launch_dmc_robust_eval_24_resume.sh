#!/usr/bin/env bash
set -euo pipefail

OUT=${OUT:-/mnt/data_7tb/fanfeng/controllable_agent_robust_eval/20260502_dmc12_no_clean_cuda1}
MAN=${MAN:-$OUT/manifest.json}
SESSION=${SESSION:-dmc_robust_cuda1_24_resume}
NUM_SHARDS=${NUM_SHARDS:-24}
CUDA_DEVICE=${CUDA_DEVICE:-1}

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session $SESSION already exists"
  exit 1
fi

tmux new-session -d -s "$SESSION" "cd /data/fanfeng/controllable_agent; \
  export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE; \
  export MUJOCO_GL=egl; \
  echo relaunched_at=\$(date -Is) num_shards=$NUM_SHARDS cuda=$CUDA_DEVICE >> $OUT/launcher.log; \
  for i in \$(seq 0 $((NUM_SHARDS - 1))); do \
    shard=\$(printf '%02d' \$i); \
    mkdir -p $OUT/shard24_\$shard; \
    python -u url_benchmark/robust_eval.py \
      --manifest $MAN \
      --output-dir $OUT \
      --conditions color_easy,color_hard,background_easy,background_hard,camera_easy,camera_hard,combined_easy \
      --episodes 50 \
      --calibration-transitions 5120 \
      --num-shards $NUM_SHARDS \
      --shard-index \$i \
      --device cuda \
      --seed 20260502 \
      --resume \
      > $OUT/shard24_\$shard/run.log 2>&1 & \
    echo \$! > $OUT/shard24_\$shard/pid; \
  done; \
  wait"

echo "started $SESSION with $NUM_SHARDS shards on CUDA_VISIBLE_DEVICES=$CUDA_DEVICE"
