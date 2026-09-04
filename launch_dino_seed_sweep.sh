#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/data/fanfeng/controllable_agent}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-$REPO_DIR/url_benchmark/pretrain.py}"
RUNS_DIR="${RUNS_DIR:-/mnt/data_7tb/fanfeng/controllable_agent_runs}"
CKPT_ROOT="${CKPT_ROOT:-/mnt/data_7tb/fanfeng/controallable_agent_ckpt}"
LAUNCH_ROOT="${LAUNCH_ROOT:-${QUEUE_ROOT:-$RUNS_DIR/launch_queues}}"

TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --timestamp)
      TIMESTAMP="$2"
      shift 2
      ;;
    -h|--help)
      cat <<'EOF'
Usage: ./launch_dino_seed_sweep.sh [--dry-run] [--timestamp YYYYMMDD_HHMMSS]

Launch all 48 DINO cls-token seed-sweep experiments concurrently:
  seed=7532 -> cuda:0
  seed=2009 -> cuda:3
  seed=8164 -> cuda:5
  seed=1992 -> cuda:6

Each seed/GPU pair launches all 12 tasks at once.

Environment overrides:
  REPO_DIR, TRAIN_SCRIPT, RUNS_DIR, CKPT_ROOT, LAUNCH_ROOT
  SEEDS="7532 2009 8164 1992"
  GPUS="0 3 5 6"
EOF
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

read -r -a SEEDS_ARRAY <<< "${SEEDS:-7532 2009 8164 1992}"
read -r -a GPUS_ARRAY <<< "${GPUS:-0 3 5 6}"

if [[ "${#SEEDS_ARRAY[@]}" -ne 4 || "${#GPUS_ARRAY[@]}" -ne 4 ]]; then
  echo "Expected exactly 4 seeds and 4 GPUs." >&2
  echo "Got seeds: ${SEEDS_ARRAY[*]}" >&2
  echo "Got GPUs: ${GPUS_ARRAY[*]}" >&2
  exit 2
fi

TASK_SPECS=(
  "walker_stand simplified_walker"
  "walker_walk simplified_walker"
  "walker_run simplified_walker"
  "walker_flip simplified_walker"
  "quadruped_stand simplified_quadruped"
  "quadruped_walk simplified_quadruped"
  "quadruped_run simplified_quadruped"
  "quadruped_jump simplified_quadruped"
  "cheetah_walk null"
  "cheetah_run null"
  "cheetah_walk_backward null"
  "cheetah_run_backward null"
)

mkdir -p "$RUNS_DIR" "$LAUNCH_ROOT/$TIMESTAMP"

write_task_script() {
  local launch_file="$1"
  local gpu="$2"
  local seed="$3"
  local task="$4"
  local goal_space="$5"
  local run_dir="$6"

  {
    printf '#!/usr/bin/env bash\n'
    printf 'set -euo pipefail\n'
    printf 'cd %q\n' "$REPO_DIR"
    printf 'run_dir=%q\n' "$run_dir"
    printf 'stdout_path="$run_dir/stdout.log"\n'
    printf 'mkdir -p "$run_dir"\n'
    printf 'echo "[start] $(date -u +%%FT%%TZ) task=%s goal_space=%s seed=%s gpu=%s run_dir=$run_dir" | tee "$run_dir/launcher.log"\n' \
      "$task" "$goal_space" "$seed" "$gpu"
    printf 'if env CUDA_VISIBLE_DEVICES=%q PYTHONUNBUFFERED=1 python %q ' "$gpu" "$TRAIN_SCRIPT"
    printf '%q ' \
      "use_wandb=True" \
      "save_replay_buffer_in_checkpoint=False" \
      "checkpoint_root=$CKPT_ROOT" \
      "obs_type=dino" \
      "use_cls=True" \
      "experiment=dino_cls_seed${seed}" \
      "render_shape=[224,224]" \
      "task=$task" \
      "goal_space=$goal_space" \
      "seed=$seed" \
      "hydra.run.dir=$run_dir"
    printf '> "$stdout_path" 2>&1; then\n'
    printf '  rc=0\n'
    printf 'else\n'
    printf '  rc=$?\n'
    printf 'fi\n'
    printf 'echo "[done] $(date -u +%%FT%%TZ) task=%s seed=%s gpu=%s exit=$rc" | tee -a "$run_dir/launcher.log"\n' \
      "$task" "$seed" "$gpu"
    printf 'exit "$rc"\n'
  } > "$launch_file"
  chmod +x "$launch_file"
}

launch_task() {
  local gpu="$1"
  local seed="$2"
  local task="$3"
  local goal_space="$4"
  local session="dino_cls_${TIMESTAMP}_s${seed}_${task}_g${gpu}"
  local run_dir="$RUNS_DIR/${TIMESTAMP}_seed${seed}_${task}_cuda${gpu}_dino_cls"
  local launch_file="$LAUNCH_ROOT/$TIMESTAMP/${session}.sh"
  local group_log="$LAUNCH_ROOT/$TIMESTAMP/${session}.log"

  write_task_script "$launch_file" "$gpu" "$seed" "$task" "$goal_space" "$run_dir"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$session" "$gpu" "$seed" "$task" "$goal_space" "$run_dir" "$group_log"

  if [[ "$DRY_RUN" -eq 0 ]]; then
    if tmux has-session -t "$session" >/dev/null 2>&1; then
      echo "tmux session already exists: $session" >&2
      exit 1
    fi
    tmux new-session -d -s "$session" "bash $(printf '%q' "$launch_file") > $(printf '%q' "$group_log") 2>&1"
  fi
}

printf 'SESSION\tGPU\tSEED\tTASK\tGOAL_SPACE\tRUN_DIR\tGROUP_LOG\n'
for i in "${!SEEDS_ARRAY[@]}"; do
  seed="${SEEDS_ARRAY[$i]}"
  gpu="${GPUS_ARRAY[$i]}"
  for spec in "${TASK_SPECS[@]}"; do
    read -r task goal_space <<< "$spec"
    launch_task "$gpu" "$seed" "$task" "$goal_space"
  done
done

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "Dry run only. Launch scripts were written, but tmux sessions were not launched."
fi
