#!/usr/bin/env bash
set -euo pipefail

# Launch the seed-1 five-domain FB matrix without resubmitting the twenty
# completed DINO cells or the four currently running Cheetah CNN cells that
# were audited on 2026-08-23.  New jobs are claimed dynamically from one
# shared, priority-ordered queue by one worker per configured GPU.

REPO_DIR="${REPO_DIR:-/data/fanfeng/controllable_agent}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-$REPO_DIR/url_benchmark/pretrain.py}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUNS_DIR="${RUNS_DIR:-/mnt/data_7tb/fanfeng/controllable_agent_runs}"
CKPT_ROOT="${CKPT_ROOT:-/mnt/data_7tb/fanfeng/controallable_agent_ckpt}"
LAUNCH_ROOT="${LAUNCH_ROOT:-$RUNS_DIR/launch_queues}"
HISTORICAL_RUNS_DIR="${HISTORICAL_RUNS_DIR:-/mnt/data_7tb/fanfeng/controllable_agent_runs}"
HISTORICAL_CKPT_ROOT="${HISTORICAL_CKPT_ROOT:-/mnt/data_7tb/fanfeng/controallable_agent_ckpt}"
WANDB_ENTITY="${WANDB_ENTITY:-lmu_rl}"
WANDB_PROJECT="${WANDB_PROJECT:-controllable_agent_baseline}"
CAMPAIGN="${CAMPAIGN:-20260823_five_domain_fb_matrix_s1}"

# The scientific protocol is deliberately fixed.  Scheduling and storage
# locations remain configurable, but changing one of these values would make
# the audited reuse table invalid.
SEED=1
NUM_TRAIN_FRAMES=2000010
EVAL_EVERY_FRAMES=10000
NUM_EVAL_EPISODES=10
FINAL_TESTS=10
CHECKPOINT_EVERY=100000
AGENT_LR=0.0001
DINO_MODEL_NAME="facebook/dinov2-base"
DINO_ADAPTER_OUTPUT_DIM=512
AGENT_BATCH_SIZE=1024
AGENT_UPDATE_EVERY_STEPS=2
ACTION_REPEAT=2
SNAPSHOT_AT="${SNAPSHOT_AT:-[2000000]}"

GPUS_STRING="${GPUS:-2 3 4 5 6 7}"
GPU_SLOTS_STRING="${GPU_SLOTS:-}"
WAIT_SECONDS="${WAIT_SECONDS:-60}"
MIN_FREE_MB="${MIN_FREE_MB:-60000}"
MAX_GPU_UTIL="${MAX_GPU_UTIL:-20}"
READY_CHECKS="${READY_CHECKS:-2}"
MAX_CONSECUTIVE_FAILURES="${MAX_CONSECUTIVE_FAILURES:-1}"

# Optional whitespace-separated GPU:value maps.  For example:
#   GPU_MIN_FREE_MB_OVERRIDES="2:45000 4:70000"
#   GPU_MAX_UTIL_OVERRIDES="2:100 4:30"
#   GPU_READY_CHECKS_OVERRIDES="2:1"
# Values are resolved by each worker immediately before it claims a job.
GPU_MIN_FREE_MB_OVERRIDES="${GPU_MIN_FREE_MB_OVERRIDES:-}"
GPU_MAX_UTIL_OVERRIDES="${GPU_MAX_UTIL_OVERRIDES:-}"
GPU_READY_CHECKS_OVERRIDES="${GPU_READY_CHECKS_OVERRIDES:-}"

DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: ./launch_five_domain_fb_matrix.sh [--dry-run] [--campaign NAME]

Build the complete 100-cell, seed-1 FB matrix and launch only the 76 missing
cells.  Twenty audited completed DINO cells are reused, and the four running
Cheetah CNN cells are excluded from submission.

The 76 new jobs are stored in a shared priority queue.  One tmux worker is
created per GPU; a worker waits for its GPU readiness gate before atomically
claiming the next job with flock.  Cheetah, Jaco, and Point-Mass-Maze jobs are
ordered before Walker and Quadruped jobs.

Important environment overrides:
  GPUS="2 3 4 5 6 7"        One worker per listed GPU (default)
  GPU_SLOTS="3:4 5:3 7:1"  Optional GPU:worker-count map; takes precedence
  RUNS_DIR=/path/to/new/runs
  CKPT_ROOT=/path/to/new/checkpoints
  LAUNCH_ROOT=/path/to/launch/plans
  SNAPSHOT_AT='[2000000]'
  MIN_FREE_MB=60000 MAX_GPU_UTIL=20 READY_CHECKS=2 WAIT_SECONDS=60
  MAX_CONSECUTIVE_FAILURES=1  Stop a worker after the first failed job
  GPU_MIN_FREE_MB_OVERRIDES="2:45000 4:70000"
  GPU_MAX_UTIL_OVERRIDES="2:100"
  GPU_READY_CHECKS_OVERRIDES="2:1"
  WANDB_ENTITY=lmu_rl WANDB_PROJECT=controllable_agent_baseline
  REPO_DIR, TRAIN_SCRIPT, PYTHON_BIN, HISTORICAL_RUNS_DIR,
  HISTORICAL_CKPT_ROOT

Dry-run writes matrix.tsv, manifest.tsv, job scripts, and queue state, but
does not validate live reuse state and does not start tmux workers.  A real
launch validates all 20 completed cells and all four running CNN cells before
creating the launch directory.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --campaign)
      if [[ $# -lt 2 ]]; then
        echo "--campaign requires a value." >&2
        exit 2
      fi
      CAMPAIGN="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

is_nonnegative_integer() {
  [[ "$1" =~ ^[0-9]+$ ]]
}

is_positive_integer() {
  [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

path_exists() {
  [[ -e "$1" || -L "$1" ]]
}

if [[ ! "$CAMPAIGN" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "CAMPAIGN must contain only letters, digits, '.', '_' or '-': $CAMPAIGN" >&2
  exit 2
fi
if [[ ! "$SNAPSHOT_AT" =~ ^\[[0-9]+(,[0-9]+)*\]$ ]]; then
  echo "SNAPSHOT_AT must be a compact Hydra integer list such as [2000000]: $SNAPSHOT_AT" >&2
  exit 2
fi
for value_name in WAIT_SECONDS MIN_FREE_MB READY_CHECKS MAX_CONSECUTIVE_FAILURES; do
  value="${!value_name}"
  if ! is_positive_integer "$value"; then
    echo "$value_name must be a positive integer: $value" >&2
    exit 2
  fi
done
if ! is_nonnegative_integer "$MAX_GPU_UTIL" || (( MAX_GPU_UTIL > 100 )); then
  echo "MAX_GPU_UTIL must be an integer in [0, 100]: $MAX_GPU_UTIL" >&2
  exit 2
fi
if [[ ! -d "$REPO_DIR" ]]; then
  echo "REPO_DIR does not exist: $REPO_DIR" >&2
  exit 1
fi
if [[ ! -f "$TRAIN_SCRIPT" ]]; then
  echo "TRAIN_SCRIPT does not exist: $TRAIN_SCRIPT" >&2
  exit 1
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "PYTHON_BIN is not executable or not on PATH: $PYTHON_BIN" >&2
  exit 1
fi
for required_command in flock sha256sum; do
  if ! command -v "$required_command" >/dev/null 2>&1; then
    echo "$required_command is required." >&2
    exit 1
  fi
done
if [[ "$DRY_RUN" -eq 0 ]]; then
  for required_command in nvidia-smi pgrep tmux; do
    if ! command -v "$required_command" >/dev/null 2>&1; then
      echo "$required_command is required for a real launch." >&2
      exit 1
    fi
  done
fi

declare -a GPUS=()
declare -a SLOT_GPUS=()
declare -a SLOT_IDS=()
declare -A CONFIGURED_GPUS=()

register_gpu() {
  local gpu="$1"
  if ! is_nonnegative_integer "$gpu"; then
    echo "GPU ordinals must be non-negative integers: $gpu" >&2
    exit 2
  fi
  if [[ -z "${CONFIGURED_GPUS[$gpu]+present}" ]]; then
    CONFIGURED_GPUS["$gpu"]=1
    GPUS+=("$gpu")
  fi
}

if [[ -n "$GPU_SLOTS_STRING" ]]; then
  global_slot=0
  declare -A SEEN_SLOT_GPU=()
  for slot_spec in $GPU_SLOTS_STRING; do
    if [[ ! "$slot_spec" =~ ^([0-9]+):([1-9][0-9]*)$ ]]; then
      echo "GPU_SLOTS entries must have GPU:POSITIVE_COUNT form: $slot_spec" >&2
      exit 2
    fi
    gpu="${BASH_REMATCH[1]}"
    slot_count="${BASH_REMATCH[2]}"
    if [[ -n "${SEEN_SLOT_GPU[$gpu]+present}" ]]; then
      echo "GPU_SLOTS must contain at most one entry per GPU: $gpu" >&2
      exit 2
    fi
    SEEN_SLOT_GPU["$gpu"]=1
    register_gpu "$gpu"
    for ((local_slot = 1; local_slot <= slot_count; local_slot++)); do
      global_slot=$((global_slot + 1))
      SLOT_GPUS+=("$gpu")
      SLOT_IDS+=("g${gpu}s${local_slot}")
    done
  done
else
  read -r -a requested_gpus <<< "$GPUS_STRING"
  declare -A SEEN_REQUESTED_GPU=()
  for gpu in "${requested_gpus[@]}"; do
    if [[ -n "${SEEN_REQUESTED_GPU[$gpu]+present}" ]]; then
      echo "Duplicate GPU ordinal in GPUS is not allowed; use GPU_SLOTS for repeated workers: $gpu" >&2
      exit 2
    fi
    SEEN_REQUESTED_GPU["$gpu"]=1
    register_gpu "$gpu"
    SLOT_GPUS+=("$gpu")
    SLOT_IDS+=("g${gpu}s1")
  done
fi
if [[ "${#SLOT_GPUS[@]}" -eq 0 ]]; then
  echo "At least one GPU worker slot is required." >&2
  exit 2
fi
for gpu in "${GPUS[@]}"; do
  if [[ "$DRY_RUN" -eq 0 ]] \
      && ! nvidia-smi --id="$gpu" --query-gpu=name --format=csv,noheader >/dev/null 2>&1; then
    echo "GPU ordinal is unavailable: $gpu" >&2
    exit 1
  fi
done

validate_override_map() {
  local map_name="$1"
  local map_value="$2"
  local kind="$3"
  local entry override_gpu override_value
  for entry in $map_value; do
    if [[ ! "$entry" =~ ^([0-9]+):([0-9]+)$ ]]; then
      echo "$map_name entries must have GPU:VALUE form: $entry" >&2
      exit 2
    fi
    override_gpu="${BASH_REMATCH[1]}"
    override_value="${BASH_REMATCH[2]}"
    if [[ -z "${CONFIGURED_GPUS[$override_gpu]+present}" ]]; then
      echo "$map_name references a GPU not present in GPUS: $override_gpu" >&2
      exit 2
    fi
    case "$kind" in
      positive)
        if ! is_positive_integer "$override_value"; then
          echo "$map_name values must be positive integers: $entry" >&2
          exit 2
        fi
        ;;
      percent)
        if ! is_nonnegative_integer "$override_value" || (( override_value > 100 )); then
          echo "$map_name values must be integers in [0, 100]: $entry" >&2
          exit 2
        fi
        ;;
      *)
        echo "Internal error: unknown override kind $kind" >&2
        exit 2
        ;;
    esac
  done
}

validate_override_map GPU_MIN_FREE_MB_OVERRIDES "$GPU_MIN_FREE_MB_OVERRIDES" positive
validate_override_map GPU_MAX_UTIL_OVERRIDES "$GPU_MAX_UTIL_OVERRIDES" percent
validate_override_map GPU_READY_CHECKS_OVERRIDES "$GPU_READY_CHECKS_OVERRIDES" positive

SOURCE_FILES=(
  url_benchmark/pretrain.py
  url_benchmark/base_config.yaml
  url_benchmark/dmc.py
  url_benchmark/goals.py
  url_benchmark/in_memory_replay_buffer.py
  url_benchmark/utils.py
  url_benchmark/agent/fb_ddpg.py
  url_benchmark/agent/fb_modules.py
  url_benchmark/agent/ddpg.py
  url_benchmark/custom_dmc_tasks/__init__.py
  url_benchmark/custom_dmc_tasks/cheetah.py
  url_benchmark/custom_dmc_tasks/jaco.py
  url_benchmark/custom_dmc_tasks/point_mass_maze.py
)
for source_file in "${SOURCE_FILES[@]}"; do
  if [[ ! -f "$REPO_DIR/$source_file" ]]; then
    echo "Required source file does not exist: $REPO_DIR/$source_file" >&2
    exit 1
  fi
done

source_fingerprint() {
  local source_root="$1"
  local source_file digest
  for source_file in "${SOURCE_FILES[@]}"; do
    digest="$(sha256sum "$source_root/$source_file")"
    printf '%s\n' "${digest%% *}"
  done | sha256sum | awk '{print $1}'
}

SOURCE_FINGERPRINT="$(source_fingerprint "$REPO_DIR")"

# Ordered by domain priority.  Every domain has exactly four registered tasks.
TASK_SPECS=(
  "cheetah_walk|cheetah|cheetah_speed|high"
  "cheetah_run|cheetah|cheetah_speed|high"
  "cheetah_walk_backward|cheetah|cheetah_speed|high"
  "cheetah_run_backward|cheetah|cheetah_speed|high"
  "jaco_reach_top_left|jaco|simplified_jaco|high"
  "jaco_reach_top_right|jaco|simplified_jaco|high"
  "jaco_reach_bottom_left|jaco|simplified_jaco|high"
  "jaco_reach_bottom_right|jaco|simplified_jaco|high"
  "point_mass_maze_reach_top_left|point_mass_maze|simplified_point_mass_maze|high"
  "point_mass_maze_reach_top_right|point_mass_maze|simplified_point_mass_maze|high"
  "point_mass_maze_reach_bottom_left|point_mass_maze|simplified_point_mass_maze|high"
  "point_mass_maze_reach_bottom_right|point_mass_maze|simplified_point_mass_maze|high"
  "walker_stand|walker|simplified_walker|low"
  "walker_walk|walker|simplified_walker|low"
  "walker_run|walker|simplified_walker|low"
  "walker_flip|walker|simplified_walker|low"
  "quadruped_stand|quadruped|simplified_quadruped|low"
  "quadruped_walk|quadruped|simplified_quadruped|low"
  "quadruped_run|quadruped|simplified_quadruped|low"
  "quadruped_jump|quadruped|simplified_quadruped|low"
)

# variant|observation type|DINO frame stack|IDM coefficient|short identity
VARIANT_SPECS=(
  "cnn_fb|pixels|0|0.0|cf"
  "dino_cls1_fb|dino|1|0.0|d1"
  "dino_cls3_fb|dino|3|0.0|d3"
  "dino_cls1_fb_idm0p1|dino|1|0.1|d1i"
  "dino_cls3_fb_idm0p1|dino|3|0.1|d3i"
)

completed_reuse_run_dir() {
  local variant="$1"
  local task="$2"
  local domain="$3"
  if [[ "$variant" == "dino_cls3_fb" ]]; then
    if [[ "$domain" == "cheetah" ]]; then
      printf '%s/%s\n' "$HISTORICAL_RUNS_DIR" \
        "20260820_cheetah_speed_goal_v3_seed1_${task}_dino_cls_stack3"
      return 0
    fi
    if [[ "$domain" == "walker" || "$domain" == "quadruped" ]]; then
      printf '%s/%s\n' "$HISTORICAL_RUNS_DIR" \
        "20260813_idm_2m_full_coef_sweep_v3_seed1_${task}_dino_cls_stack3_full2m_idm0p0_idmlr0p0001"
      return 0
    fi
  fi
  if [[ "$variant" == "dino_cls3_fb_idm0p1" \
        && ( "$domain" == "walker" || "$domain" == "quadruped" ) ]]; then
    printf '%s/%s\n' "$HISTORICAL_RUNS_DIR" \
      "20260813_idm_2m_full_coef_sweep_v3_seed1_${task}_dino_cls_stack3_full2m_idm0p1_idmlr0p0001"
    return 0
  fi
  return 1
}

running_cnn_run_dir() {
  local task="$1"
  printf '%s/%s\n' "$HISTORICAL_RUNS_DIR" \
    "20260822_172133_cheetah_speed_goal_cnn_s1_immediate_seed1_${task}_cnn_cheetah_speed_goal"
}

running_cnn_session() {
  case "$1" in
    cheetah_walk)
      echo "cnng_20260822_172133_cheetah_speed_goal_cnn_s1_immediate_s1_cheetah_walk_g7"
      ;;
    cheetah_run)
      echo "cnng_20260822_172133_cheetah_speed_goal_cnn_s1_immediate_s1_cheetah_run_g4"
      ;;
    cheetah_walk_backward)
      echo "cnng_20260822_172133_cheetah_speed_goal_cnn_s1_immediate_s1_cheetah_walk_backward_g2"
      ;;
    cheetah_run_backward)
      echo "cnng_20260822_172133_cheetah_speed_goal_cnn_s1_immediate_s1_cheetah_run_backward_g5"
      ;;
    *)
      return 1
      ;;
  esac
}

declare -a MATRIX_INDEX=()
declare -a MATRIX_TIER=()
declare -a MATRIX_DISPOSITION=()
declare -a MATRIX_VARIANT=()
declare -a MATRIX_OBS_TYPE=()
declare -a MATRIX_DINO_STACK=()
declare -a MATRIX_IDM_COEF=()
declare -a MATRIX_TASK=()
declare -a MATRIX_DOMAIN=()
declare -a MATRIX_GOAL_SPACE=()
declare -a MATRIX_RUN_NAMES=()
declare -a MATRIX_RUN_DIRS=()
declare -a MATRIX_CKPT_DIRS=()
declare -a MATRIX_REUSE_SESSIONS=()

declare -a NEW_JOB_IDS=()
declare -a NEW_PRIORITIES=()
declare -a NEW_VARIANTS=()
declare -a NEW_VARIANT_CODES=()
declare -a NEW_OBS_TYPES=()
declare -a NEW_DINO_STACKS=()
declare -a NEW_IDM_COEFS=()
declare -a NEW_TASKS=()
declare -a NEW_DOMAINS=()
declare -a NEW_GOAL_SPACES=()
declare -a NEW_RUN_NAMES=()
declare -a NEW_RUN_DIRS=()
declare -a NEW_CKPT_DIRS=()
declare -a NEW_WANDB_IDS=()

matrix_count=0
new_count=0
completed_count=0
running_count=0

# Tier is the primary sort key.  Within a tier, baselines precede variations,
# and task order is stable.  Pending filenames preserve this exact order.
for requested_tier in high low; do
  for variant_spec in "${VARIANT_SPECS[@]}"; do
    IFS='|' read -r variant obs_type dino_stack idm_coef variant_code <<< "$variant_spec"
    for task_spec in "${TASK_SPECS[@]}"; do
      IFS='|' read -r task domain goal_space tier <<< "$task_spec"
      if [[ "$tier" != "$requested_tier" ]]; then
        continue
      fi

      matrix_count=$((matrix_count + 1))
      disposition="new"
      reuse_session=""
      run_name="${CAMPAIGN}_seed${SEED}_${task}_${variant}"
      run_dir="$RUNS_DIR/$run_name"
      ckpt_dir="$CKPT_ROOT/$run_name"

      if [[ "$variant" == "cnn_fb" && "$domain" == "cheetah" ]]; then
        disposition="running_excluded"
        run_dir="$(running_cnn_run_dir "$task")"
        run_name="${run_dir##*/}"
        ckpt_dir="$HISTORICAL_CKPT_ROOT/$run_name"
        reuse_session="$(running_cnn_session "$task")"
        running_count=$((running_count + 1))
      elif reused_run_dir="$(completed_reuse_run_dir "$variant" "$task" "$domain")"; then
        disposition="completed_reused"
        run_dir="$reused_run_dir"
        run_name="${run_dir##*/}"
        ckpt_dir="$HISTORICAL_CKPT_ROOT/$run_name"
        completed_count=$((completed_count + 1))
      else
        new_count=$((new_count + 1))
        job_id="$(printf 'job%03d' "$new_count")"
        identity="seed=$SEED|task=$task|variant=$variant|goal=$goal_space|frames=$NUM_TRAIN_FRAMES|eval=$EVAL_EVERY_FRAMES|snapshots=$SNAPSHOT_AT|model=$DINO_MODEL_NAME"
        identity_digest="$(printf '%s' "$identity" | sha256sum | awk '{print $1}')"
        task_slug="${task//_/-}"
        wandb_run_id="fdm-s${SEED}-${variant_code}-${task_slug}-${identity_digest:0:12}"

        NEW_JOB_IDS+=("$job_id")
        NEW_PRIORITIES+=("$new_count")
        NEW_VARIANTS+=("$variant")
        NEW_VARIANT_CODES+=("$variant_code")
        NEW_OBS_TYPES+=("$obs_type")
        NEW_DINO_STACKS+=("$dino_stack")
        NEW_IDM_COEFS+=("$idm_coef")
        NEW_TASKS+=("$task")
        NEW_DOMAINS+=("$domain")
        NEW_GOAL_SPACES+=("$goal_space")
        NEW_RUN_NAMES+=("$run_name")
        NEW_RUN_DIRS+=("$run_dir")
        NEW_CKPT_DIRS+=("$ckpt_dir")
        NEW_WANDB_IDS+=("$wandb_run_id")
      fi

      MATRIX_INDEX+=("$matrix_count")
      MATRIX_TIER+=("$tier")
      MATRIX_DISPOSITION+=("$disposition")
      MATRIX_VARIANT+=("$variant")
      MATRIX_OBS_TYPE+=("$obs_type")
      MATRIX_DINO_STACK+=("$dino_stack")
      MATRIX_IDM_COEF+=("$idm_coef")
      MATRIX_TASK+=("$task")
      MATRIX_DOMAIN+=("$domain")
      MATRIX_GOAL_SPACE+=("$goal_space")
      MATRIX_RUN_NAMES+=("$run_name")
      MATRIX_RUN_DIRS+=("$run_dir")
      MATRIX_CKPT_DIRS+=("$ckpt_dir")
      MATRIX_REUSE_SESSIONS+=("$reuse_session")
    done
  done
done

if [[ "$matrix_count" -ne 100 || "$new_count" -ne 76 \
      || "$completed_count" -ne 20 || "$running_count" -ne 4 ]]; then
  echo "Internal matrix error: total=$matrix_count new=$new_count completed=$completed_count running=$running_count" >&2
  exit 1
fi

validate_completed_reuse() {
  local run_dir="$1"
  local run_name="${run_dir##*/}"
  local launcher_log="$run_dir/launcher.log"
  local snapshot="$HISTORICAL_CKPT_ROOT/$run_name/snapshot_2000000.pt"
  local test_rewards="$run_dir/test_rewards.json"
  if [[ ! -s "$launcher_log" ]] \
      || ! grep -Eq '^\[done\].*exit=0[[:space:]]*$' "$launcher_log"; then
    echo "Completed reuse validation failed (missing exit=0): $launcher_log" >&2
    return 1
  fi
  if [[ ! -s "$snapshot" ]]; then
    echo "Completed reuse validation failed (missing 2M snapshot): $snapshot" >&2
    return 1
  fi
  if [[ ! -s "$test_rewards" ]]; then
    echo "Completed reuse validation failed (missing final tests): $test_rewards" >&2
    return 1
  fi
}

validate_running_reuse() {
  local run_dir="$1"
  local session="$2"
  if ! tmux has-session -t "$session" >/dev/null 2>&1; then
    echo "Running reuse validation failed (tmux session missing): $session" >&2
    return 1
  fi
  if ! pgrep -f "$run_dir" >/dev/null 2>&1; then
    echo "Running reuse validation failed (training process missing): $run_dir" >&2
    return 1
  fi
}

launch_dir="$LAUNCH_ROOT/$CAMPAIGN"
matrix_file="$launch_dir/matrix.tsv"
manifest="$launch_dir/manifest.tsv"
queue_file="$launch_dir/priority_queue.tsv"
pending_dir="$launch_dir/queue/pending"
claimed_dir="$launch_dir/queue/claimed"
completed_dir="$launch_dir/queue/completed"
failed_dir="$launch_dir/queue/failed"
queue_lock="$launch_dir/queue/claim.lock"
jobs_dir="$launch_dir/jobs"
worker_script="$launch_dir/worker.sh"
worker_config="$launch_dir/worker_config.sh"

if path_exists "$launch_dir"; then
  echo "Fresh launch refused; launch directory already exists: $launch_dir" >&2
  exit 1
fi

# Refuse every new run/checkpoint collision before writing a partial plan.
for index in "${!NEW_JOB_IDS[@]}"; do
  if path_exists "${NEW_RUN_DIRS[$index]}"; then
    echo "Fresh launch refused; run directory already exists: ${NEW_RUN_DIRS[$index]}" >&2
    exit 1
  fi
  if path_exists "${NEW_CKPT_DIRS[$index]}"; then
    echo "Fresh launch refused; checkpoint directory already exists: ${NEW_CKPT_DIRS[$index]}" >&2
    exit 1
  fi
done

declare -a WORKER_SESSIONS=()
for slot_index in "${!SLOT_GPUS[@]}"; do
  gpu="${SLOT_GPUS[$slot_index]}"
  slot_id="${SLOT_IDS[$slot_index]}"
  worker_session="fdm_${CAMPAIGN}_${slot_id}"
  worker_session="${worker_session//./_}"
  WORKER_SESSIONS+=("$worker_session")
  if [[ "$DRY_RUN" -eq 0 ]] && tmux has-session -t "$worker_session" >/dev/null 2>&1; then
    echo "Fresh launch refused; tmux worker already exists: $worker_session" >&2
    exit 1
  fi
done

if [[ "$DRY_RUN" -eq 0 ]]; then
  echo "Validating 20 completed reuse cells and four running CNN cells..."
  for index in "${!MATRIX_INDEX[@]}"; do
    case "${MATRIX_DISPOSITION[$index]}" in
      completed_reused)
        validate_completed_reuse "${MATRIX_RUN_DIRS[$index]}"
        ;;
      running_excluded)
        validate_running_reuse \
          "${MATRIX_RUN_DIRS[$index]}" "${MATRIX_REUSE_SESSIONS[$index]}"
        ;;
    esac
  done
fi

mkdir -p "$RUNS_DIR" "$CKPT_ROOT" "$jobs_dir" "$pending_dir" \
  "$claimed_dir" "$completed_dir" "$failed_dir"
: > "$queue_lock"

printf 'cell_index\tpriority_tier\tdisposition\tvariant\tobs_type\tdino_frame_stack\tidm_coef\tidm_lr\tidm_encoder_mode\tseed\ttask\tdomain\tgoal_space\tnum_train_frames\teval_every_frames\tnum_eval_episodes\tfinal_tests\tsnapshot_at\trun_name\trun_dir\tcheckpoint_dir\treuse_tmux_session\n' > "$matrix_file"
for index in "${!MATRIX_INDEX[@]}"; do
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\tnull\tlegacy\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${MATRIX_INDEX[$index]}" "${MATRIX_TIER[$index]}" \
    "${MATRIX_DISPOSITION[$index]}" "${MATRIX_VARIANT[$index]}" \
    "${MATRIX_OBS_TYPE[$index]}" "${MATRIX_DINO_STACK[$index]}" \
    "${MATRIX_IDM_COEF[$index]}" "$SEED" "${MATRIX_TASK[$index]}" \
    "${MATRIX_DOMAIN[$index]}" "${MATRIX_GOAL_SPACE[$index]}" \
    "$NUM_TRAIN_FRAMES" "$EVAL_EVERY_FRAMES" "$NUM_EVAL_EPISODES" \
    "$FINAL_TESTS" "$SNAPSHOT_AT" "${MATRIX_RUN_NAMES[$index]}" \
    "${MATRIX_RUN_DIRS[$index]}" "${MATRIX_CKPT_DIRS[$index]}" \
    "${MATRIX_REUSE_SESSIONS[$index]}" >> "$matrix_file"
done

printf 'job_id\tqueue_priority\tpriority_tier\tvariant\tobs_type\tdino_frame_stack\tidm_coef\tidm_lr\tidm_encoder_mode\tseed\ttask\tdomain\tgoal_space\trun_name\trun_dir\tcheckpoint_dir\tstdout_log\twandb_entity\twandb_project\twandb_run_id\tnum_train_frames\teval_every_frames\tnum_eval_episodes\tfinal_tests\tcheckpoint_every\tsnapshot_at\tdino_model_name\tagent_lr\tagent_batch_size\tagent_update_every_steps\taction_repeat\tsource_fingerprint\tjob_file\n' > "$manifest"
printf 'job_id\tqueue_priority\tpriority_tier\tvariant\ttask\tjob_file\tpending_file\n' > "$queue_file"

write_job_script() {
  local index="$1"
  local job_file="$2"
  local variant="${NEW_VARIANTS[$index]}"
  local obs_type="${NEW_OBS_TYPES[$index]}"
  local dino_stack="${NEW_DINO_STACKS[$index]}"
  local idm_coef="${NEW_IDM_COEFS[$index]}"
  local task="${NEW_TASKS[$index]}"
  local goal_space="${NEW_GOAL_SPACES[$index]}"
  local run_name="${NEW_RUN_NAMES[$index]}"
  local run_dir="${NEW_RUN_DIRS[$index]}"
  local ckpt_dir="${NEW_CKPT_DIRS[$index]}"
  local wandb_run_id="${NEW_WANDB_IDS[$index]}"

  {
    printf '#!/usr/bin/env bash\n'
    printf 'set -euo pipefail\n'
    printf 'gpu=${1:?GPU ordinal is required}\n'
    printf 'cd %q\n' "$REPO_DIR"
    printf 'run_name=%q\n' "$run_name"
    printf 'run_dir=%q\n' "$run_dir"
    printf 'ckpt_dir=%q\n' "$ckpt_dir"
    printf 'expected_source_fingerprint=%q\n' "$SOURCE_FINGERPRINT"
    printf 'source_files=(\n'
    for source_file in "${SOURCE_FILES[@]}"; do
      printf '  %q\n' "$source_file"
    done
    printf ')\n'
    cat <<'EOF'
actual_source_fingerprint="$({
  for source_file in "${source_files[@]}"; do
    digest="$(sha256sum "$source_file")"
    printf '%s\n' "${digest%% *}"
  done
} | sha256sum | awk '{print $1}')"
if [[ "$actual_source_fingerprint" != "$expected_source_fingerprint" ]]; then
  echo "[refused] source changed while queued: expected=$expected_source_fingerprint actual=$actual_source_fingerprint" >&2
  exit 1
fi
if [[ -e "$run_dir" || -L "$run_dir" || -e "$ckpt_dir" || -L "$ckpt_dir" ]]; then
  echo "[refused] fresh-start collision after queue wait: run_dir=$run_dir ckpt_dir=$ckpt_dir" >&2
  exit 1
fi
mkdir -p "$run_dir/matplotlib"
EOF
    printf 'echo "[start] $(date -u +%%FT%%TZ) variant=%s task=%s seed=%s gpu=$gpu goal_space=%s dino_frame_stack=%s idm_coef=%s num_train_frames=%s" | tee "$run_dir/launcher.log"\n' \
      "$variant" "$task" "$SEED" "$goal_space" "$dino_stack" "$idm_coef" \
      "$NUM_TRAIN_FRAMES"
    printf 'if env CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID="$gpu" PYTHONUNBUFFERED=1 WANDB_ENTITY=%q WANDB_PROJECT=%q WANDB_RUN_ID=%q WANDB_RUN_NAME=%q WANDB_RESUME=never MPLCONFIGDIR="$run_dir/matplotlib" %q %q ' \
      "$WANDB_ENTITY" "$WANDB_PROJECT" "$wandb_run_id" "$run_name" \
      "$PYTHON_BIN" "$TRAIN_SCRIPT"
    printf '%q ' \
      "agent=fb_ddpg" \
      "use_wandb=True" \
      "use_tb=False" \
      "use_hiplog=False" \
      "save_video=False" \
      "save_train_video=False" \
      "save_replay_buffer_in_checkpoint=False" \
      "auto_resume=False" \
      "checkpoint_root=$CKPT_ROOT" \
      "checkpoint_every=$CHECKPOINT_EVERY" \
      "snapshot_at=$SNAPSHOT_AT" \
      "action_repeat=$ACTION_REPEAT" \
      "goal_space=$goal_space" \
      "append_goal_to_observation=False" \
      "agent.lr=$AGENT_LR" \
      "agent.batch_size=$AGENT_BATCH_SIZE" \
      "agent.update_every_steps=$AGENT_UPDATE_EVERY_STEPS" \
      "agent.num_inference_steps=5120" \
      "agent.backward_encoder_grad_scale=1.0" \
      "agent.ortho_coef=1.0" \
      "agent.mix_ratio=0.5" \
      "agent.fb_target_tau=0.01" \
      "agent.lr_coef=1.0" \
      "agent.idm_coef=$idm_coef" \
      "agent.idm_lr=null" \
      "agent.idm_diagnostics_interval=500" \
      "agent.idm_encoder_mode=legacy" \
      "agent.idm_encoder_burnin_steps=0" \
      "agent.idm_encoder_ramp_steps=0" \
      "agent.idm_grad_ratio_target=null" \
      "update_encoder=True" \
      "num_seed_frames=4000" \
      "num_train_frames=$NUM_TRAIN_FRAMES" \
      "eval_every_frames=$EVAL_EVERY_FRAMES" \
      "num_eval_episodes=$NUM_EVAL_EPISODES" \
      "final_tests=$FINAL_TESTS" \
      "experiment=${variant}_seed${SEED}" \
      "task=$task" \
      "seed=$SEED" \
      "hydra.run.dir=$run_dir"
    if [[ "$obs_type" == "pixels" ]]; then
      printf '%q ' \
        "obs_type=pixels" \
        "frame_stack=3" \
        "render_shape=[84,84]"
    else
      # frame_stack is a legacy pixel setting and remains fixed across all
      # DINO cells.  dino_frame_stack and idm_coef are the only algorithmic
      # differences among the four DINO variants.
      printf '%q ' \
        "obs_type=dino" \
        "dino_model_name=$DINO_MODEL_NAME" \
        "use_cls=True" \
        "frame_stack=3" \
        "dino_frame_stack=$dino_stack" \
        "render_shape=[224,224]" \
        "agent.dino_use_adapter=True" \
        "agent.dino_adapter_type=linear" \
        "agent.dino_adapter_output_dim=$DINO_ADAPTER_OUTPUT_DIM"
    fi
    printf '> "$run_dir/stdout.log" 2>&1; then\n'
    printf '  rc=0\n'
    printf 'else\n'
    printf '  rc=$?\n'
    printf 'fi\n'
    printf 'echo "[done] $(date -u +%%FT%%TZ) variant=%s task=%s seed=%s gpu=$gpu exit=$rc" | tee -a "$run_dir/launcher.log"\n' \
      "$variant" "$task" "$SEED"
    printf 'exit "$rc"\n'
  } > "$job_file"
  chmod +x "$job_file"
}

for index in "${!NEW_JOB_IDS[@]}"; do
  job_id="${NEW_JOB_IDS[$index]}"
  priority="${NEW_PRIORITIES[$index]}"
  variant="${NEW_VARIANTS[$index]}"
  task="${NEW_TASKS[$index]}"
  tier="low"
  if (( priority <= 52 )); then
    tier="high"
  fi
  job_file="$jobs_dir/${job_id}_${variant}_${task}.sh"
  pending_file="$pending_dir/$(printf '%03d' "$priority")_${job_id}_${variant}_${task}.pending"
  write_job_script "$index" "$job_file"
  {
    printf 'job_id=%q\n' "$job_id"
    printf 'queue_priority=%q\n' "$priority"
    printf 'variant=%q\n' "$variant"
    printf 'task=%q\n' "$task"
    printf 'run_name=%q\n' "${NEW_RUN_NAMES[$index]}"
    printf 'job_file=%q\n' "$job_file"
  } > "$pending_file"

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\tlegacy\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$job_id" "$priority" "$tier" "$variant" "${NEW_OBS_TYPES[$index]}" \
    "${NEW_DINO_STACKS[$index]}" "${NEW_IDM_COEFS[$index]}" "null" "$SEED" \
    "$task" "${NEW_DOMAINS[$index]}" "${NEW_GOAL_SPACES[$index]}" \
    "${NEW_RUN_NAMES[$index]}" "${NEW_RUN_DIRS[$index]}" \
    "${NEW_CKPT_DIRS[$index]}" "${NEW_RUN_DIRS[$index]}/stdout.log" \
    "$WANDB_ENTITY" "$WANDB_PROJECT" "${NEW_WANDB_IDS[$index]}" \
    "$NUM_TRAIN_FRAMES" "$EVAL_EVERY_FRAMES" "$NUM_EVAL_EPISODES" \
    "$FINAL_TESTS" "$CHECKPOINT_EVERY" "$SNAPSHOT_AT" "$DINO_MODEL_NAME" \
    "$AGENT_LR" "$AGENT_BATCH_SIZE" "$AGENT_UPDATE_EVERY_STEPS" \
    "$ACTION_REPEAT" "$SOURCE_FINGERPRINT" "$job_file" >> "$manifest"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$job_id" "$priority" "$tier" "$variant" "$task" "$job_file" \
    "$pending_file" >> "$queue_file"
done

{
  printf '#!/usr/bin/env bash\n'
  printf 'launch_dir=%q\n' "$launch_dir"
  printf 'pending_dir=%q\n' "$pending_dir"
  printf 'claimed_dir=%q\n' "$claimed_dir"
  printf 'completed_dir=%q\n' "$completed_dir"
  printf 'failed_dir=%q\n' "$failed_dir"
  printf 'queue_lock=%q\n' "$queue_lock"
  printf 'wait_seconds=%q\n' "$WAIT_SECONDS"
  printf 'default_min_free_mb=%q\n' "$MIN_FREE_MB"
  printf 'default_max_gpu_util=%q\n' "$MAX_GPU_UTIL"
  printf 'default_ready_checks=%q\n' "$READY_CHECKS"
  printf 'max_consecutive_failures=%q\n' "$MAX_CONSECUTIVE_FAILURES"
  printf 'gpu_min_free_mb_overrides=%q\n' "$GPU_MIN_FREE_MB_OVERRIDES"
  printf 'gpu_max_util_overrides=%q\n' "$GPU_MAX_UTIL_OVERRIDES"
  printf 'gpu_ready_checks_overrides=%q\n' "$GPU_READY_CHECKS_OVERRIDES"
} > "$worker_config"
chmod +x "$worker_config"

cat > "$worker_script" <<'EOF'
#!/usr/bin/env bash
set -uo pipefail

gpu="${1:?GPU ordinal is required}"
slot_id="${2:?worker slot id is required}"
config_file="${3:?worker config is required}"
source "$config_file"

lookup_override() {
  local map_value="$1"
  local requested_gpu="$2"
  local fallback="$3"
  local entry key value
  for entry in $map_value; do
    key="${entry%%:*}"
    value="${entry#*:}"
    if [[ "$key" == "$requested_gpu" ]]; then
      printf '%s\n' "$value"
      return 0
    fi
  done
  printf '%s\n' "$fallback"
}

min_free_mb="$(lookup_override "$gpu_min_free_mb_overrides" "$gpu" "$default_min_free_mb")"
max_gpu_util="$(lookup_override "$gpu_max_util_overrides" "$gpu" "$default_max_gpu_util")"
ready_checks="$(lookup_override "$gpu_ready_checks_overrides" "$gpu" "$default_ready_checks")"

queue_has_pending() {
  local lock_fd
  exec {lock_fd}>"$queue_lock"
  flock -x "$lock_fd"
  shopt -s nullglob
  local pending=("$pending_dir"/*.pending)
  shopt -u nullglob
  flock -u "$lock_fd"
  exec {lock_fd}>&-
  (( ${#pending[@]} > 0 ))
}

wait_for_gpu_ready() {
  local consecutive_ready=0
  local sample free_mb gpu_util
  while (( consecutive_ready < ready_checks )); do
    if sample="$(nvidia-smi --id="$gpu" --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits 2>/dev/null)"; then
      IFS=',' read -r free_mb gpu_util <<< "$sample"
      free_mb="${free_mb//[[:space:]]/}"
      gpu_util="${gpu_util//[[:space:]]/}"
    else
      free_mb="unavailable"
      gpu_util="unavailable"
    fi
    if [[ "$free_mb" =~ ^[0-9]+$ && "$gpu_util" =~ ^[0-9]+$ ]] \
        && (( free_mb >= min_free_mb && gpu_util <= max_gpu_util )); then
      consecutive_ready=$((consecutive_ready + 1))
      echo "[gpu-ready] $(date -u +%FT%TZ) gpu=$gpu free_mb=$free_mb util=$gpu_util check=$consecutive_ready/$ready_checks"
    else
      consecutive_ready=0
      echo "[gpu-wait] $(date -u +%FT%TZ) gpu=$gpu free_mb=$free_mb util=$gpu_util required_free_mb=$min_free_mb max_util=$max_gpu_util"
    fi
    if (( consecutive_ready < ready_checks )); then
      sleep "$wait_seconds"
    fi
  done
}

claim_next_job() {
  local lock_fd source_file base claim_file
  exec {lock_fd}>"$queue_lock"
  flock -x "$lock_fd"
  shopt -s nullglob
  local pending=("$pending_dir"/*.pending)
  shopt -u nullglob
  if (( ${#pending[@]} == 0 )); then
    flock -u "$lock_fd"
    exec {lock_fd}>&-
    return 1
  fi
  source_file="${pending[0]}"
  base="${source_file##*/}"
  claim_file="$claimed_dir/${base%.pending}.${slot_id}.claimed"
  if ! mv "$source_file" "$claim_file"; then
    flock -u "$lock_fd"
    exec {lock_fd}>&-
    return 2
  fi
  flock -u "$lock_fd"
  exec {lock_fd}>&-
  printf '%s\n' "$claim_file"
}

echo "[worker-start] $(date -u +%FT%TZ) gpu=$gpu slot=$slot_id min_free_mb=$min_free_mb max_util=$max_gpu_util ready_checks=$ready_checks"
worker_failures=0
worker_completed=0
consecutive_failures=0
while queue_has_pending; do
  # Readiness is intentionally established before a job is removed from the
  # shared queue.  A busy GPU therefore cannot reserve a high-priority job
  # while another ready worker is available.
  wait_for_gpu_ready
  if ! claim_file="$(claim_next_job)"; then
    continue
  fi
  unset job_id queue_priority variant task run_name job_file
  source "$claim_file"
  echo "[job-claimed] $(date -u +%FT%TZ) gpu=$gpu slot=$slot_id job_id=$job_id priority=$queue_priority variant=$variant task=$task"
  if bash "$job_file" "$gpu"; then
    rc=0
    worker_completed=$((worker_completed + 1))
    consecutive_failures=0
    destination="$completed_dir/${claim_file##*/}"
    destination="${destination%.claimed}.done"
  else
    rc=$?
    worker_failures=$((worker_failures + 1))
    consecutive_failures=$((consecutive_failures + 1))
    destination="$failed_dir/${claim_file##*/}"
    destination="${destination%.claimed}.rc${rc}.failed"
  fi
  printf 'finished_at=%q\ngpu=%q\nexit_code=%q\n' \
    "$(date -u +%FT%TZ)" "$gpu" "$rc" >> "$claim_file"
  mv "$claim_file" "$destination"
  echo "[job-finished] $(date -u +%FT%TZ) gpu=$gpu slot=$slot_id job_id=$job_id variant=$variant task=$task exit=$rc"
  if (( consecutive_failures >= max_consecutive_failures )); then
    echo "[worker-stop] $(date -u +%FT%TZ) gpu=$gpu slot=$slot_id consecutive_failures=$consecutive_failures limit=$max_consecutive_failures"
    break
  fi
done
echo "[worker-done] $(date -u +%FT%TZ) gpu=$gpu slot=$slot_id completed=$worker_completed failures=$worker_failures"
if (( worker_failures > 0 )); then
  exit 1
fi
EOF
chmod +x "$worker_script"

{
  printf 'campaign=%s\n' "$CAMPAIGN"
  printf 'created_at=%s\n' "$(date -u +%FT%TZ)"
  printf 'dry_run=%s\n' "$DRY_RUN"
  printf 'gpus=%s\n' "${GPUS[*]}"
  printf 'gpu_slots_request=%s\n' "${GPU_SLOTS_STRING:-one_per_gpu}"
  printf 'expanded_worker_slots=%s\n' "${SLOT_IDS[*]}"
  printf 'matrix_cells=%s\n' "$matrix_count"
  printf 'completed_reused=%s\n' "$completed_count"
  printf 'running_excluded=%s\n' "$running_count"
  printf 'new_jobs=%s\n' "$new_count"
  printf 'snapshot_at=%s\n' "$SNAPSHOT_AT"
  printf 'checkpoint_every=%s\n' "$CHECKPOINT_EVERY"
  printf 'source_fingerprint=%s\n' "$SOURCE_FINGERPRINT"
  printf 'min_free_mb=%s\n' "$MIN_FREE_MB"
  printf 'max_gpu_util=%s\n' "$MAX_GPU_UTIL"
  printf 'ready_checks=%s\n' "$READY_CHECKS"
  printf 'max_consecutive_failures=%s\n' "$MAX_CONSECUTIVE_FAILURES"
  printf 'gpu_min_free_mb_overrides=%s\n' "$GPU_MIN_FREE_MB_OVERRIDES"
  printf 'gpu_max_util_overrides=%s\n' "$GPU_MAX_UTIL_OVERRIDES"
  printf 'gpu_ready_checks_overrides=%s\n' "$GPU_READY_CHECKS_OVERRIDES"
  printf 'python=%s\n' "$PYTHON_BIN"
  "$PYTHON_BIN" --version 2>&1
} > "$launch_dir/launch_config.txt"

echo "Matrix: $matrix_file"
echo "New-job manifest: $manifest"
echo "Priority queue: $queue_file"
echo "Cells: total=$matrix_count completed_reused=$completed_count running_excluded=$running_count new=$new_count"

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "Dry run only; no reuse state was validated and no tmux workers were launched."
  exit 0
fi

launch_failures=0
for index in "${!SLOT_GPUS[@]}"; do
  gpu="${SLOT_GPUS[$index]}"
  slot_id="${SLOT_IDS[$index]}"
  session="${WORKER_SESSIONS[$index]}"
  worker_log="$launch_dir/worker_${slot_id}.log"
  if tmux new-session -d -s "$session" \
      "bash $(printf '%q' "$worker_script") $(printf '%q' "$gpu") $(printf '%q' "$slot_id") $(printf '%q' "$worker_config") > $(printf '%q' "$worker_log") 2>&1"; then
    printf '%s\t%s\t%s\t%s\n' "$session" "$gpu" "$slot_id" "$worker_log"
  else
    echo "Failed to create tmux worker: $session" >&2
    launch_failures=1
  fi
done
for session in "${WORKER_SESSIONS[@]}"; do
  if ! tmux has-session -t "$session" >/dev/null 2>&1; then
    echo "Missing tmux worker after launch: $session" >&2
    launch_failures=1
  fi
done
if [[ "$launch_failures" -ne 0 ]]; then
  echo "One or more workers failed to launch; inspect $launch_dir" >&2
  exit 1
fi

echo "Launched ${#SLOT_GPUS[@]} dynamic worker slots across ${#GPUS[@]} GPU(s) for $new_count new jobs."
