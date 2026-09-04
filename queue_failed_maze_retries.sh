#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/data/fanfeng/controllable_agent}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-$REPO_DIR/url_benchmark/pretrain.py}"
RUNS_DIR="${RUNS_DIR:-/mnt/data_7tb/fanfeng/controllable_agent_runs}"
CKPT_ROOT="${CKPT_ROOT:-/mnt/data_7tb/fanfeng/controallable_agent_ckpt}"
LAUNCH_ROOT="${LAUNCH_ROOT:-$RUNS_DIR/launch_queues}"
TIMESTAMP="${TIMESTAMP:-$(date -u +%Y%m%d_%H%M%S)_oom_retry}"

NUM_TRAIN_FRAMES="${NUM_TRAIN_FRAMES:-10000000}"
EVAL_EVERY_FRAMES="${EVAL_EVERY_FRAMES:-500000}"

TASK="point_mass_maze_multi_goal"
GOAL_SPACE="simplified_point_mass_maze"
CUSTOM_REWARD="maze_multi_goal"
SNAPSHOT_AT="[100000,500000,1000000,2000000,5000000,10000000]"

mkdir -p "$RUNS_DIR" "$LAUNCH_ROOT/$TIMESTAMP"

gpu_free_mb() {
  local gpu="$1"
  timeout 30s nvidia-smi --id="$gpu" --query-gpu=memory.free --format=csv,noheader,nounits \
    | head -1 \
    | tr -dc '0-9' || true
}

wait_for_mem() {
  local gpu="$1"
  local min_free="$2"
  local free_mb=""

  while true; do
    free_mb="$(gpu_free_mb "$gpu")"
    if [[ -n "$free_mb" && "$free_mb" -ge "$min_free" ]]; then
      echo "[queue] $(date -u +%FT%TZ) gpu=$gpu free_mb=$free_mb >= $min_free"
      return 0
    fi
    echo "[queue] $(date -u +%FT%TZ) waiting gpu=$gpu free_mb=${free_mb:-unknown} need=$min_free"
    sleep 60
  done
}

write_fb_script() {
  local launch_file="$1"
  local gpu="$2"
  local seed="$3"
  local variant="$4"
  local run_dir="$5"

  local obs_type="dino"
  local render_shape="[224,224]"
  local frame_stack="1"
  local replay_eps="2000"
  local experiment="maze_${variant}"
  local extra=()

  case "$variant" in
    dino_patch_linear)
      extra=("use_cls=False" "agent.dino_use_adapter=True" "agent.dino_adapter_type=linear")
      ;;
    dino_cls_no_adapter)
      extra=("use_cls=True" "agent.dino_use_adapter=False" "agent.feature_dim=1024")
      ;;
    vit)
      obs_type="vit"
      replay_eps="80"
      extra=("use_cls=True" "agent.vit_batch_size=64")
      ;;
    *)
      echo "Unknown retry variant: $variant" >&2
      exit 2
      ;;
  esac

  {
    printf '#!/usr/bin/env bash\n'
    printf 'set -euo pipefail\n'
    printf 'cd %q\n' "$REPO_DIR"
    printf 'run_dir=%q\n' "$run_dir"
    printf 'stdout_path="$run_dir/stdout.log"\n'
    printf 'mkdir -p "$run_dir"\n'
    printf 'echo "[start] $(date -u +%%FT%%TZ) retry_oom variant=%s seed=%s gpu=%s run_dir=$run_dir" | tee "$run_dir/launcher.log"\n' \
      "$variant" "$seed" "$gpu"
    printf 'if env CUDA_VISIBLE_DEVICES=%q PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python %q ' "$gpu" "$TRAIN_SCRIPT"
    printf '%q ' \
      "use_wandb=True" \
      "save_video=False" \
      "save_train_video=False" \
      "save_replay_buffer_in_checkpoint=False" \
      "checkpoint_root=$CKPT_ROOT" \
      "checkpoint_every=500000" \
      "snapshot_at=$SNAPSHOT_AT" \
      "obs_type=$obs_type" \
      "render_shape=$render_shape" \
      "frame_stack=$frame_stack" \
      "action_repeat=1" \
      "task=$TASK" \
      "goal_space=$GOAL_SPACE" \
      "custom_reward=$CUSTOM_REWARD" \
      "seed=$seed" \
      "experiment=$experiment" \
      "num_train_frames=$NUM_TRAIN_FRAMES" \
      "eval_every_frames=$EVAL_EVERY_FRAMES" \
      "num_eval_episodes=1" \
      "final_tests=10" \
      "replay_buffer_episodes=$replay_eps" \
      "hydra.run.dir=$run_dir"
    for arg in "${extra[@]}"; do
      printf '%q ' "$arg"
    done
    printf '> "$stdout_path" 2>&1; then\n'
    printf '  rc=0\n'
    printf 'else\n'
    printf '  rc=$?\n'
    printf 'fi\n'
    printf 'echo "[done] $(date -u +%%FT%%TZ) retry_oom variant=%s seed=%s gpu=%s exit=$rc" | tee -a "$run_dir/launcher.log"\n' \
      "$variant" "$seed" "$gpu"
    printf 'exit "$rc"\n'
  } > "$launch_file"
  chmod +x "$launch_file"
}

queue_fb() {
  local gpu="$1"
  local seed="$2"
  local variant="$3"
  local min_free="$4"

  local session="maze_${TIMESTAMP}_${variant}_s${seed}_g${gpu}"
  local run_dir="$RUNS_DIR/${TIMESTAMP}_seed${seed}_${variant}_cuda${gpu}"
  local launch_file="$LAUNCH_ROOT/$TIMESTAMP/${session}.sh"
  local group_log="$LAUNCH_ROOT/$TIMESTAMP/${session}.log"

  write_fb_script "$launch_file" "$gpu" "$seed" "$variant" "$run_dir"
  echo -e "$session\tgpu=$gpu\tseed=$seed\tvariant=$variant\tmin_free=$min_free\trun_dir=$run_dir"

  tmux new-session -d -s "$session" \
    "bash -lc '$(printf '%q ' "$0" --run-one "$gpu" "$min_free" "$launch_file" "$group_log")'"
}

run_one() {
  local gpu="$1"
  local min_free="$2"
  local launch_file="$3"
  local group_log="$4"
  wait_for_mem "$gpu" "$min_free" | tee "$group_log"
  bash "$launch_file" >> "$group_log" 2>&1
}

if [[ "${1:-}" == "--run-one" ]]; then
  shift
  run_one "$@"
  exit 0
fi

echo "SESSION GPU SEED VARIANT MIN_FREE RUN_DIR"
queue_fb 1 2009 vit 35000
queue_fb 7 9284 dino_patch_linear 8000
queue_fb 7 9284 dino_cls_no_adapter 8000
