#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/data/fanfeng/controllable_agent}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-$REPO_DIR/url_benchmark/pretrain.py}"
RUNS_DIR="${RUNS_DIR:-/mnt/data_7tb/fanfeng/controllable_agent_runs}"
CKPT_ROOT="${CKPT_ROOT:-/mnt/data_7tb/fanfeng/controallable_agent_ckpt}"
LAUNCH_ROOT="${LAUNCH_ROOT:-${QUEUE_ROOT:-$RUNS_DIR/launch_queues}}"
WANDB_PROJECT="${WANDB_PROJECT:-controllable_agent_dino_mlp_cls_patch}"

TIMESTAMP="${TIMESTAMP:-$(date -u +%Y%m%d_%H%M%S)_dmc_cnn_2seed}"
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
Usage: ./launch_cnn_seed_sweep.sh [--dry-run] [--timestamp NAME]

Launch 24 DMC CNN pixel experiments:
  2 seeds
  12 cheetah/walker/quadruped tasks

Default scheduling uses 6 queue slots:
  SLOT_GPUS="3 4 0 2 3 4"

Each slot is one tmux queue and runs its assigned jobs sequentially.

Environment overrides:
  REPO_DIR, TRAIN_SCRIPT, RUNS_DIR, CKPT_ROOT, LAUNCH_ROOT
  WANDB_PROJECT="controllable_agent_dino_mlp_cls_patch"
  SEEDS="7532 2009"
  SLOT_GPUS="3 4 0 2 3 4"
EOF
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

read -r -a SEEDS_ARRAY <<< "${SEEDS:-7532 2009}"
read -r -a SLOT_GPUS_ARRAY <<< "${SLOT_GPUS:-3 4 0 2 3 4}"

if [[ "${#SEEDS_ARRAY[@]}" -ne 2 ]]; then
  echo "Expected exactly 2 seeds for this CNN sweep." >&2
  echo "Got seeds: ${SEEDS_ARRAY[*]}" >&2
  exit 2
fi

if [[ "${#SLOT_GPUS_ARRAY[@]}" -lt 1 ]]; then
  echo "Expected at least one GPU slot." >&2
  exit 2
fi

TASK_SPECS=(
  "cheetah_walk null"
  "cheetah_run null"
  "cheetah_walk_backward null"
  "cheetah_run_backward null"
  "walker_stand simplified_walker"
  "walker_walk simplified_walker"
  "walker_run simplified_walker"
  "walker_flip simplified_walker"
  "quadruped_stand simplified_quadruped"
  "quadruped_walk simplified_quadruped"
  "quadruped_run simplified_quadruped"
  "quadruped_jump simplified_quadruped"
)

mkdir -p "$RUNS_DIR" "$CKPT_ROOT" "$LAUNCH_ROOT/$TIMESTAMP"

write_job_script() {
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
    printf 'echo "[start] $(date -u +%%FT%%TZ) variant=cnn task=%s goal_space=%s seed=%s gpu=%s run_dir=$run_dir" | tee "$run_dir/launcher.log"\n' \
      "$task" "$goal_space" "$seed" "$gpu"
    printf 'if env CUDA_VISIBLE_DEVICES=%q PYTHONUNBUFFERED=1 WANDB_PROJECT=%q python %q ' "$gpu" "$WANDB_PROJECT" "$TRAIN_SCRIPT"
    printf '%q ' \
      "use_wandb=True" \
      "save_replay_buffer_in_checkpoint=False" \
      "checkpoint_root=$CKPT_ROOT" \
      "obs_type=pixels" \
      "render_shape=[84,84]" \
      "task=$task" \
      "goal_space=$goal_space" \
      "seed=$seed" \
      "experiment=cnn_seed${seed}" \
      "hydra.run.dir=$run_dir"
    printf '> "$stdout_path" 2>&1; then\n'
    printf '  rc=0\n'
    printf 'else\n'
    printf '  rc=$?\n'
    printf 'fi\n'
    printf 'echo "[done] $(date -u +%%FT%%TZ) variant=cnn task=%s seed=%s gpu=%s exit=$rc" | tee -a "$run_dir/launcher.log"\n' \
      "$task" "$seed" "$gpu"
    printf 'exit "$rc"\n'
  } > "$launch_file"
  chmod +x "$launch_file"
}

declare -a SLOT_FILES
declare -a SLOT_LOGS
declare -a SLOT_COUNTS

for slot in "${!SLOT_GPUS_ARRAY[@]}"; do
  gpu="${SLOT_GPUS_ARRAY[$slot]}"
  queue_session="cnn_${TIMESTAMP}_slot${slot}_g${gpu}"
  queue_file="$LAUNCH_ROOT/$TIMESTAMP/${queue_session}.sh"
  queue_log="$LAUNCH_ROOT/$TIMESTAMP/${queue_session}.log"
  SLOT_FILES[$slot]="$queue_file"
  SLOT_LOGS[$slot]="$queue_log"
  SLOT_COUNTS[$slot]=0
  {
    printf '#!/usr/bin/env bash\n'
    printf 'set -uo pipefail\n'
    printf 'echo "[queue-start] $(date -u +%%FT%%TZ) slot=%q gpu=%q"\n' "$slot" "$gpu"
    printf 'failures=0\n'
    printf 'completed=0\n'
  } > "$queue_file"
done

job_index=0
for seed in "${SEEDS_ARRAY[@]}"; do
  for task_spec in "${TASK_SPECS[@]}"; do
    read -r task goal_space <<< "$task_spec"
    slot=$((job_index % ${#SLOT_GPUS_ARRAY[@]}))
    gpu="${SLOT_GPUS_ARRAY[$slot]}"
    session="cnn_${TIMESTAMP}_s${seed}_${task}_g${gpu}_slot${slot}"
    run_dir="$RUNS_DIR/${TIMESTAMP}_seed${seed}_${task}_cnn_cuda${gpu}_slot${slot}"
    launch_file="$LAUNCH_ROOT/$TIMESTAMP/${session}.sh"
    job_log="$LAUNCH_ROOT/$TIMESTAMP/${session}.log"

    write_job_script "$launch_file" "$gpu" "$seed" "$task" "$goal_space" "$run_dir"
    {
      printf 'echo "[job-start] $(date -u +%%FT%%TZ) %q"\n' "$session"
      printf 'if bash %q > %q 2>&1; then\n' "$launch_file" "$job_log"
      printf '  completed=$((completed + 1))\n'
      printf '  echo "[job-ok] $(date -u +%%FT%%TZ) %q"\n' "$session"
      printf 'else\n'
      printf '  rc=$?\n'
      printf '  failures=$((failures + 1))\n'
      printf '  echo "[job-failed] $(date -u +%%FT%%TZ) %q exit=$rc"\n' "$session"
      printf 'fi\n'
    } >> "${SLOT_FILES[$slot]}"
    SLOT_COUNTS[$slot]=$((SLOT_COUNTS[$slot] + 1))
    job_index=$((job_index + 1))
  done
done

printf 'QUEUE_SESSION\tGPU\tSLOT\tJOBS\tQUEUE_LOG\n'
for slot in "${!SLOT_GPUS_ARRAY[@]}"; do
  gpu="${SLOT_GPUS_ARRAY[$slot]}"
  queue_session="cnn_${TIMESTAMP}_slot${slot}_g${gpu}"
  queue_file="${SLOT_FILES[$slot]}"
  queue_log="${SLOT_LOGS[$slot]}"
  {
    printf 'echo "[queue-done] $(date -u +%%FT%%TZ) slot=%q gpu=%q completed=$completed failures=$failures"\n' "$slot" "$gpu"
    printf 'exit 0\n'
  } >> "$queue_file"
  chmod +x "$queue_file"

  printf '%s\t%s\t%s\t%s\t%s\n' "$queue_session" "$gpu" "$slot" "${SLOT_COUNTS[$slot]}" "$queue_log"

  if [[ "$DRY_RUN" -eq 0 ]]; then
    if tmux has-session -t "$queue_session" >/dev/null 2>&1; then
      echo "tmux session already exists: $queue_session" >&2
      exit 1
    fi
    tmux new-session -d -s "$queue_session" "bash $(printf '%q' "$queue_file") > $(printf '%q' "$queue_log") 2>&1"
  fi
done

echo "Launch files are under $LAUNCH_ROOT/$TIMESTAMP"
if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "Dry run only. Launch scripts were written, but tmux sessions were not launched."
fi
