#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/data/fanfeng/controllable_agent}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-$REPO_DIR/url_benchmark/pretrain.py}"
RUNS_DIR="${RUNS_DIR:-/mnt/data_7tb/fanfeng/controllable_agent_runs}"
CKPT_ROOT="${CKPT_ROOT:-/mnt/data_7tb/fanfeng/controallable_agent_ckpt}"
LAUNCH_ROOT="${LAUNCH_ROOT:-${QUEUE_ROOT:-$RUNS_DIR/launch_queues}}"
WANDB_PROJECT="${WANDB_PROJECT:-controllable_agent_dino_mlp_cls_patch}"

TIMESTAMP="${TIMESTAMP:-$(date -u +%Y%m%d_%H%M%S)_dmc_dino_mlp_cls_patch}"
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
Usage: ./launch_dino_mlp_cls_patch_sweep.sh [--dry-run] [--timestamp NAME]

Launch 72 DMC DINO MLP-adapter experiments:
  2 variants: dino_cls_mlp, dino_patch_mlp
  3 seeds
  12 cheetah/walker/quadruped tasks

Default scheduling:
  seed=7532 -> cuda:0
  seed=2009 -> cuda:3
  seed=8164 -> cuda:5

One tmux queue is launched per seed/GPU pair. Each queue runs its
24 jobs sequentially to avoid oversubscribing a GPU.

Environment overrides:
  REPO_DIR, TRAIN_SCRIPT, RUNS_DIR, CKPT_ROOT, LAUNCH_ROOT
  WANDB_PROJECT="controllable_agent_dino_mlp_cls_patch"
  SEEDS="7532 2009 8164"
  GPUS="0 3 5"
EOF
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

read -r -a SEEDS_ARRAY <<< "${SEEDS:-7532 2009 8164}"
read -r -a GPUS_ARRAY <<< "${GPUS:-0 3 5}"

if [[ "${#SEEDS_ARRAY[@]}" -ne "${#GPUS_ARRAY[@]}" ]]; then
  echo "Expected the same number of seeds and GPUs." >&2
  echo "Got seeds: ${SEEDS_ARRAY[*]}" >&2
  echo "Got GPUs: ${GPUS_ARRAY[*]}" >&2
  exit 2
fi

if [[ "${#SEEDS_ARRAY[@]}" -ne 3 ]]; then
  echo "Expected exactly 3 seeds for this sweep." >&2
  echo "Got seeds: ${SEEDS_ARRAY[*]}" >&2
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

VARIANTS=(
  "dino_cls_mlp True"
  "dino_patch_mlp False"
)

mkdir -p "$RUNS_DIR" "$CKPT_ROOT" "$LAUNCH_ROOT/$TIMESTAMP"

write_job_script() {
  local launch_file="$1"
  local gpu="$2"
  local seed="$3"
  local task="$4"
  local goal_space="$5"
  local variant="$6"
  local use_cls="$7"
  local run_dir="$8"

  {
    printf '#!/usr/bin/env bash\n'
    printf 'set -euo pipefail\n'
    printf 'cd %q\n' "$REPO_DIR"
    printf 'run_dir=%q\n' "$run_dir"
    printf 'stdout_path="$run_dir/stdout.log"\n'
    printf 'mkdir -p "$run_dir"\n'
    printf 'echo "[start] $(date -u +%%FT%%TZ) variant=%s task=%s goal_space=%s seed=%s gpu=%s run_dir=$run_dir" | tee "$run_dir/launcher.log"\n' \
      "$variant" "$task" "$goal_space" "$seed" "$gpu"
    printf 'if env CUDA_VISIBLE_DEVICES=%q PYTHONUNBUFFERED=1 WANDB_PROJECT=%q python %q ' "$gpu" "$WANDB_PROJECT" "$TRAIN_SCRIPT"
    printf '%q ' \
      "use_wandb=True" \
      "save_replay_buffer_in_checkpoint=False" \
      "checkpoint_root=$CKPT_ROOT" \
      "obs_type=dino" \
      "render_shape=[224,224]" \
      "task=$task" \
      "goal_space=$goal_space" \
      "seed=$seed" \
      "experiment=${variant}_seed${seed}" \
      "hydra.run.dir=$run_dir" \
      "use_cls=$use_cls" \
      "agent.dino_use_adapter=True" \
      "agent.dino_adapter_type=mlp"
    printf '> "$stdout_path" 2>&1; then\n'
    printf '  rc=0\n'
    printf 'else\n'
    printf '  rc=$?\n'
    printf 'fi\n'
    printf 'echo "[done] $(date -u +%%FT%%TZ) variant=%s task=%s seed=%s gpu=%s exit=$rc" | tee -a "$run_dir/launcher.log"\n' \
      "$variant" "$task" "$seed" "$gpu"
    printf 'exit "$rc"\n'
  } > "$launch_file"
  chmod +x "$launch_file"
}

write_queue_script() {
  local queue_file="$1"
  local gpu="$2"
  local seed="$3"

  {
    printf '#!/usr/bin/env bash\n'
    printf 'set -uo pipefail\n'
    printf 'echo "[queue-start] $(date -u +%%FT%%TZ) seed=%q gpu=%q"\n' "$seed" "$gpu"
    printf 'failures=0\n'
    printf 'completed=0\n'

    for variant_spec in "${VARIANTS[@]}"; do
      read -r variant use_cls <<< "$variant_spec"
      for task_spec in "${TASK_SPECS[@]}"; do
        read -r task goal_space <<< "$task_spec"
        local session run_dir launch_file job_log
        session="dmcmlp_${TIMESTAMP}_${variant}_s${seed}_${task}_g${gpu}"
        run_dir="$RUNS_DIR/${TIMESTAMP}_seed${seed}_${task}_${variant}_cuda${gpu}"
        launch_file="$LAUNCH_ROOT/$TIMESTAMP/${session}.sh"
        job_log="$LAUNCH_ROOT/$TIMESTAMP/${session}.log"

        write_job_script "$launch_file" "$gpu" "$seed" "$task" "$goal_space" "$variant" "$use_cls" "$run_dir"
        printf 'echo "[job-start] $(date -u +%%FT%%TZ) %q"\n' "$session"
        printf 'if bash %q > %q 2>&1; then\n' "$launch_file" "$job_log"
        printf '  completed=$((completed + 1))\n'
        printf '  echo "[job-ok] $(date -u +%%FT%%TZ) %q"\n' "$session"
        printf 'else\n'
        printf '  rc=$?\n'
        printf '  failures=$((failures + 1))\n'
        printf '  echo "[job-failed] $(date -u +%%FT%%TZ) %q exit=$rc"\n' "$session"
        printf 'fi\n'
      done
    done

    printf 'echo "[queue-done] $(date -u +%%FT%%TZ) seed=%q gpu=%q completed=$completed failures=$failures"\n' "$seed" "$gpu"
    printf 'exit 0\n'
  } > "$queue_file"
  chmod +x "$queue_file"
}

printf 'QUEUE_SESSION\tGPU\tSEED\tJOBS\tQUEUE_LOG\n'
for i in "${!SEEDS_ARRAY[@]}"; do
  seed="${SEEDS_ARRAY[$i]}"
  gpu="${GPUS_ARRAY[$i]}"
  queue_session="dmcmlp_${TIMESTAMP}_s${seed}_queue_g${gpu}"
  queue_file="$LAUNCH_ROOT/$TIMESTAMP/${queue_session}.sh"
  queue_log="$LAUNCH_ROOT/$TIMESTAMP/${queue_session}.log"

  write_queue_script "$queue_file" "$gpu" "$seed"
  printf '%s\t%s\t%s\t24\t%s\n' "$queue_session" "$gpu" "$seed" "$queue_log"

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
