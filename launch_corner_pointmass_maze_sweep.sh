#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/data/fanfeng/controllable_agent}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-$REPO_DIR/url_benchmark/pretrain.py}"
RUNS_DIR="${RUNS_DIR:-/mnt/data_7tb/fanfeng/controllable_agent_runs}"
CKPT_ROOT="${CKPT_ROOT:-/mnt/data_7tb/fanfeng/controallable_agent_ckpt}"
LAUNCH_ROOT="${LAUNCH_ROOT:-$RUNS_DIR/launch_queues}"

TIMESTAMP="${TIMESTAMP:-$(date -u +%Y%m%d_%H%M%S)_corner_maze}"
NUM_TRAIN_FRAMES="${NUM_TRAIN_FRAMES:-500000}"
EVAL_EVERY_FRAMES="${EVAL_EVERY_FRAMES:-100000}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-100000}"
SNAPSHOT_AT="${SNAPSHOT_AT:-[100000,500000]}"

read -r -a TASKS_ARRAY <<< "${TASKS:-point_mass_maze_reach_top_left point_mass_maze_reach_top_right point_mass_maze_reach_bottom_left point_mass_maze_reach_bottom_right}"
read -r -a GPUS_ARRAY <<< "${GPUS:-1 2 5 6}"
read -r -a SEEDS_ARRAY <<< "${SEEDS:-7532 2009 8164 1992}"
read -r -a VARIANTS_ARRAY <<< "${VARIANTS:-dino_patch_mlp dino_patch_linear dino_cls_mlp dino_cls_linear dino_cls_no_adapter cnn}"

if [[ "${#TASKS_ARRAY[@]}" -ne "${#GPUS_ARRAY[@]}" ]]; then
  echo "Expected equal number of tasks and GPUs." >&2
  echo "Tasks: ${TASKS_ARRAY[*]}" >&2
  echo "GPUs: ${GPUS_ARRAY[*]}" >&2
  exit 2
fi

mkdir -p "$RUNS_DIR" "$CKPT_ROOT" "$LAUNCH_ROOT/$TIMESTAMP"

task_short() {
  local task="$1"
  case "$task" in
    point_mass_maze_reach_top_left) echo "top_left" ;;
    point_mass_maze_reach_top_right) echo "top_right" ;;
    point_mass_maze_reach_bottom_left) echo "bottom_left" ;;
    point_mass_maze_reach_bottom_right) echo "bottom_right" ;;
    *) echo "$task" | tr -c '[:alnum:]_' '_' ;;
  esac
}

write_job_script() {
  local launch_file="$1"
  local gpu="$2"
  local task="$3"
  local seed="$4"
  local variant="$5"
  local run_dir="$6"

  local obs_type="dino"
  local render_shape="[224,224]"
  local frame_stack="1"
  local replay_eps="2000"
  local extra=()

  case "$variant" in
    dino_cls_linear)
      extra=("use_cls=True" "agent.dino_use_adapter=True" "agent.dino_adapter_type=linear")
      ;;
    dino_cls_mlp)
      extra=("use_cls=True" "agent.dino_use_adapter=True" "agent.dino_adapter_type=mlp")
      ;;
    dino_patch_linear)
      extra=("use_cls=False" "agent.dino_use_adapter=True" "agent.dino_adapter_type=linear")
      ;;
    dino_patch_mlp)
      extra=("use_cls=False" "agent.dino_use_adapter=True" "agent.dino_adapter_type=mlp")
      ;;
    dino_cls_no_adapter)
      extra=("use_cls=True" "agent.dino_use_adapter=False" "agent.feature_dim=1024")
      ;;
    cnn)
      obs_type="pixels"
      render_shape="[84,84]"
      frame_stack="3"
      replay_eps="100"
      ;;
    *)
      echo "Unknown variant: $variant" >&2
      exit 2
      ;;
  esac

  local short
  short="$(task_short "$task")"

  {
    printf '#!/usr/bin/env bash\n'
    printf 'set -euo pipefail\n'
    printf 'cd %q\n' "$REPO_DIR"
    printf 'run_dir=%q\n' "$run_dir"
    printf 'stdout_path="$run_dir/stdout.log"\n'
    printf 'mkdir -p "$run_dir"\n'
    printf 'echo "[start] $(date -u +%%FT%%TZ) corner_task=%s variant=%s seed=%s gpu=%s run_dir=$run_dir" | tee "$run_dir/launcher.log"\n' \
      "$short" "$variant" "$seed" "$gpu"
    printf 'if env CUDA_VISIBLE_DEVICES=%q PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python %q ' "$gpu" "$TRAIN_SCRIPT"
    printf '%q ' \
      "use_wandb=True" \
      "save_video=False" \
      "save_train_video=False" \
      "save_replay_buffer_in_checkpoint=False" \
      "checkpoint_root=$CKPT_ROOT" \
      "checkpoint_every=$CHECKPOINT_EVERY" \
      "snapshot_at=$SNAPSHOT_AT" \
      "obs_type=$obs_type" \
      "render_shape=$render_shape" \
      "frame_stack=$frame_stack" \
      "action_repeat=1" \
      "task=$task" \
      "goal_space=null" \
      "custom_reward=null" \
      "seed=$seed" \
      "experiment=corner_${short}_${variant}" \
      "num_train_frames=$NUM_TRAIN_FRAMES" \
      "eval_every_frames=$EVAL_EVERY_FRAMES" \
      "num_eval_episodes=3" \
      "final_tests=0" \
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
    printf 'echo "[done] $(date -u +%%FT%%TZ) corner_task=%s variant=%s seed=%s gpu=%s exit=$rc" | tee -a "$run_dir/launcher.log"\n' \
      "$short" "$variant" "$seed" "$gpu"
    printf 'exit "$rc"\n'
  } > "$launch_file"
  chmod +x "$launch_file"
}

write_queue_script() {
  local queue_file="$1"
  local gpu="$2"
  local task="$3"

  {
    printf '#!/usr/bin/env bash\n'
    printf 'set -uo pipefail\n'
    printf 'echo "[queue-start] $(date -u +%%FT%%TZ) gpu=%q task=%q"\n' "$gpu" "$task"
    printf 'failures=0\n'
    for seed in "${SEEDS_ARRAY[@]}"; do
      for variant in "${VARIANTS_ARRAY[@]}"; do
        local short session run_dir launch_file job_log
        short="$(task_short "$task")"
        session="corner_${TIMESTAMP}_${short}_${variant}_s${seed}_g${gpu}"
        run_dir="$RUNS_DIR/${TIMESTAMP}_${short}_seed${seed}_${variant}_cuda${gpu}"
        launch_file="$LAUNCH_ROOT/$TIMESTAMP/${session}.sh"
        job_log="$LAUNCH_ROOT/$TIMESTAMP/${session}.log"
        write_job_script "$launch_file" "$gpu" "$task" "$seed" "$variant" "$run_dir"
        printf 'echo "[job-start] $(date -u +%%FT%%TZ) %q"\n' "$session"
        printf 'if bash %q > %q 2>&1; then\n' "$launch_file" "$job_log"
        printf '  echo "[job-ok] $(date -u +%%FT%%TZ) %q"\n' "$session"
        printf 'else\n'
        printf '  rc=$?\n'
        printf '  failures=$((failures + 1))\n'
        printf '  echo "[job-failed] $(date -u +%%FT%%TZ) %q exit=$rc"\n' "$session"
        printf 'fi\n'
      done
    done
    printf 'echo "[queue-done] $(date -u +%%FT%%TZ) gpu=%q task=%q failures=$failures"\n' "$gpu" "$task"
    printf 'exit 0\n'
  } > "$queue_file"
  chmod +x "$queue_file"
}

printf 'SESSION\tGPU\tTASK\tQUEUE_LOG\n'
for i in "${!TASKS_ARRAY[@]}"; do
  task="${TASKS_ARRAY[$i]}"
  gpu="${GPUS_ARRAY[$i]}"
  short="$(task_short "$task")"
  queue_session="corner_${TIMESTAMP}_${short}_queue_g${gpu}"
  queue_file="$LAUNCH_ROOT/$TIMESTAMP/${queue_session}.sh"
  queue_log="$LAUNCH_ROOT/$TIMESTAMP/${queue_session}.log"
  write_queue_script "$queue_file" "$gpu" "$task"
  printf '%s\t%s\t%s\t%s\n' "$queue_session" "$gpu" "$task" "$queue_log"
  tmux new-session -d -s "$queue_session" "bash $(printf '%q' "$queue_file") > $(printf '%q' "$queue_log") 2>&1"
done

echo "Launched corner maze queues under $LAUNCH_ROOT/$TIMESTAMP"
