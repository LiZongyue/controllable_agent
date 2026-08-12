#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/data/fanfeng/controllable_agent"
TRAIN_SCRIPT="$REPO_DIR/url_benchmark/pretrain.py"
RUNS_DIR="${RUNS_DIR:-/mnt/data_7tb/fanfeng/controllable_agent_runs}"
CKPT_ROOT="${CKPT_ROOT:-/mnt/data_7tb/fanfeng/controallable_agent_ckpt}"
TIMESTAMP="${1:-$(date -u +%Y%m%d_%H%M%S)}"

mkdir -p "$RUNS_DIR"

launch_task() {
  local task="$1"
  local goal_space="$2"
  local gpu="$3"
  local session="dino_${TIMESTAMP}_${task}_g${gpu}"
  local run_dir="$RUNS_DIR/${TIMESTAMP}_${task}_cuda${gpu}_dino"

  mkdir -p "$run_dir"

  tmux new-session -d -s "$session" \
    "cd '$REPO_DIR' && exec env CUDA_VISIBLE_DEVICES='$gpu' PYTHONUNBUFFERED=1 python '$TRAIN_SCRIPT' \
      use_wandb=True \
      save_replay_buffer_in_checkpoint=False \
      checkpoint_root='$CKPT_ROOT' \
      obs_type=dino \
      render_shape='[224,224]' \
      task='$task' \
      goal_space='$goal_space' \
      hydra.run.dir='$run_dir' \
      > '$run_dir/stdout.log' 2>&1"

  printf '%s\t%s\t%s\t%s\t%s\n' "$session" "$gpu" "$task" "$goal_space" "$run_dir"
}

printf 'SESSION\tGPU\tTASK\tGOAL_SPACE\tRUN_DIR\n'

launch_task walker_stand simplified_walker 0
launch_task walker_walk simplified_walker 4
launch_task walker_run simplified_walker 5
launch_task walker_flip simplified_walker 5

launch_task quadruped_stand simplified_quadruped 4
launch_task quadruped_walk simplified_quadruped 0
launch_task quadruped_run simplified_quadruped 5
launch_task quadruped_jump simplified_quadruped 6

launch_task cheetah_walk null 4
launch_task cheetah_run null 1
launch_task cheetah_walk_backward null 6
launch_task cheetah_run_backward null 1
