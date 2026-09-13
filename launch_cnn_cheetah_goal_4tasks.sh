#!/usr/bin/env bash
set -euo pipefail

# Four task-specific pixel-CNN FB baselines for the privileged Cheetah speed
# goal space.  Each task gets an independent controller so it can wait for a
# currently running job and for its assigned GPU to become genuinely idle.

REPO_DIR="${REPO_DIR:-/data/fanfeng/controllable_agent}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-$REPO_DIR/url_benchmark/pretrain.py}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUNS_DIR="${RUNS_DIR:-/mnt/data_7tb/fanfeng/controllable_agent_runs}"
CKPT_ROOT="${CKPT_ROOT:-/mnt/data_7tb/fanfeng/controallable_agent_ckpt}"
LAUNCH_ROOT="${LAUNCH_ROOT:-$RUNS_DIR/launch_queues}"
WANDB_ENTITY="${WANDB_ENTITY:-lmu_rl}"
WANDB_PROJECT="${WANDB_PROJECT:-controllable_agent_baseline}"
SEED="${SEED:-1}"
NUM_TRAIN_FRAMES="${NUM_TRAIN_FRAMES:-2000010}"
EVAL_EVERY_FRAMES="${EVAL_EVERY_FRAMES:-10000}"
NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-10}"
FINAL_TESTS="${FINAL_TESTS:-10}"
AGENT_LR="${AGENT_LR:-0.0001}"
WAIT_SECONDS="${WAIT_SECONDS:-60}"
MIN_FREE_MB="${MIN_FREE_MB:-20000}"
MAX_GPU_UTIL="${MAX_GPU_UTIL:-20}"
READY_CHECKS="${READY_CHECKS:-2}"
TIMESTAMP="${TIMESTAMP:-$(date -u +%Y%m%d_%H%M%S)_cheetah_speed_goal_cnn_s${SEED}}"
SOURCE_REVISION="${SOURCE_REVISION:-unknown}"
GPU_ASSIGNMENTS="${GPU_ASSIGNMENTS:-2 4 3 5}"
BLOCKER_ASSIGNMENTS="${BLOCKER_ASSIGNMENTS:-dino_patchfb_ddp2_cheetah_noeval_g2g4_20260817 dino_patchfb_ddp2_cheetah_noeval_g2g4_20260817 dino_patchfb_s_224_quadruped_g3_20260813_210812 dino_patchfb_s_224_cheetah_g5_20260813_210812}"
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: ./launch_cnn_cheetah_goal_4tasks.sh [--dry-run] [--timestamp NAME]

Launch exactly four seed-1 Cheetah pixel-CNN FB runs with
goal_space=cheetah_speed:
  cheetah_walk, cheetah_run, cheetah_walk_backward, cheetah_run_backward

The default GPU assignment is "2 4 3 5".  Each task first waits for the
corresponding blocker tmux session, then requires two consecutive idle GPU
checks before it starts.  Use the literal blocker name "none" when a task has
no known blocker.

Environment overrides:
  REPO_DIR, TRAIN_SCRIPT, PYTHON_BIN, RUNS_DIR, CKPT_ROOT, LAUNCH_ROOT
  WANDB_ENTITY=lmu_rl WANDB_PROJECT=controllable_agent_baseline
  SEED=1 NUM_TRAIN_FRAMES=2000010 EVAL_EVERY_FRAMES=10000
  NUM_EVAL_EPISODES=10 FINAL_TESTS=10 AGENT_LR=0.0001
  GPU_ASSIGNMENTS="2 4 3 5"
  BLOCKER_ASSIGNMENTS="session_for_g2 session_for_g4 session_for_g3 session_for_g5"
  WAIT_SECONDS=60 MIN_FREE_MB=20000 MAX_GPU_UTIL=20 READY_CHECKS=2
  SOURCE_REVISION=...

This launcher is fresh-start only: it refuses existing run directories,
checkpoint directories, launch directories, or tmux session names.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --timestamp)
      if [[ $# -lt 2 ]]; then
        echo "--timestamp requires a value." >&2
        exit 2
      fi
      TIMESTAMP="$2"
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

is_positive_integer() {
  [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

is_nonnegative_integer() {
  [[ "$1" =~ ^[0-9]+$ ]]
}

is_positive_number() {
  [[ "$1" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$ ]] \
    && awk -v value="$1" 'BEGIN { exit !(value + 0 > 0) }'
}

path_exists() {
  [[ -e "$1" || -L "$1" ]]
}

if [[ ! "$TIMESTAMP" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "TIMESTAMP must contain only letters, digits, '.', '_' or '-': $TIMESTAMP" >&2
  exit 2
fi
if ! is_nonnegative_integer "$SEED"; then
  echo "SEED must be a non-negative integer: $SEED" >&2
  exit 2
fi
for value_name in NUM_TRAIN_FRAMES EVAL_EVERY_FRAMES NUM_EVAL_EPISODES FINAL_TESTS WAIT_SECONDS MIN_FREE_MB READY_CHECKS; do
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
if ! is_positive_number "$AGENT_LR"; then
  echo "AGENT_LR must be a positive number: $AGENT_LR" >&2
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
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is required." >&2
  exit 1
fi
if ! command -v sha256sum >/dev/null 2>&1; then
  echo "sha256sum is required." >&2
  exit 1
fi
if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is required." >&2
  exit 1
fi

TASKS=(
  cheetah_walk
  cheetah_run
  cheetah_walk_backward
  cheetah_run_backward
)
read -r -a GPUS <<< "$GPU_ASSIGNMENTS"
read -r -a BLOCKERS <<< "$BLOCKER_ASSIGNMENTS"

if [[ "${#GPUS[@]}" -ne "${#TASKS[@]}" ]]; then
  echo "GPU_ASSIGNMENTS must contain exactly ${#TASKS[@]} entries: $GPU_ASSIGNMENTS" >&2
  exit 2
fi
if [[ "${#BLOCKERS[@]}" -ne "${#TASKS[@]}" ]]; then
  echo "BLOCKER_ASSIGNMENTS must contain exactly ${#TASKS[@]} entries: $BLOCKER_ASSIGNMENTS" >&2
  exit 2
fi

declare -A SEEN_GPUS=()
for gpu in "${GPUS[@]}"; do
  if ! is_nonnegative_integer "$gpu"; then
    echo "GPU assignments must be non-negative integers: $gpu" >&2
    exit 2
  fi
  if [[ -n "${SEEN_GPUS[$gpu]+present}" ]]; then
    echo "Duplicate GPU assignment is refused for parallel tasks: $gpu" >&2
    exit 2
  fi
  SEEN_GPUS["$gpu"]=1
  if [[ "$DRY_RUN" -eq 0 ]] \
      && ! nvidia-smi -i "$gpu" --query-gpu=name --format=csv,noheader >/dev/null 2>&1; then
    echo "GPU ordinal is unavailable: $gpu" >&2
    exit 1
  fi
done

SOURCE_FILES=(
  url_benchmark/pretrain.py
  url_benchmark/base_config.yaml
  url_benchmark/dmc.py
  url_benchmark/goals.py
  url_benchmark/agent/fb_ddpg.py
  url_benchmark/agent/fb_modules.py
  url_benchmark/agent/ddpg.py
  url_benchmark/utils.py
  url_benchmark/in_memory_replay_buffer.py
  url_benchmark/custom_dmc_tasks/__init__.py
  url_benchmark/custom_dmc_tasks/cheetah.py
  url_benchmark/custom_dmc_tasks/cheetah.xml
)
for source_file in "${SOURCE_FILES[@]}"; do
  if [[ ! -f "$REPO_DIR/$source_file" ]]; then
    echo "Required source file does not exist: $REPO_DIR/$source_file" >&2
    exit 1
  fi
done

source_fingerprint() {
  local source_root="$1"
  local source_file
  local file_digest
  for source_file in "${SOURCE_FILES[@]}"; do
    file_digest="$(sha256sum "$source_root/$source_file")"
    printf '%s\n' "${file_digest%% *}"
  done | sha256sum
}

goals_file="$REPO_DIR/url_benchmark/goals.py"
GOALS_SHA256="$(sha256sum "$goals_file")"
GOALS_SHA256="${GOALS_SHA256%% *}"
SOURCE_FINGERPRINT="$(source_fingerprint "$REPO_DIR")"
SOURCE_FINGERPRINT="${SOURCE_FINGERPRINT%% *}"

launch_dir="$LAUNCH_ROOT/$TIMESTAMP"
manifest="$launch_dir/manifest.tsv"
if path_exists "$launch_dir"; then
  echo "Fresh launch refused; launch directory already exists: $launch_dir" >&2
  exit 1
fi

declare -a SESSIONS=()
declare -a RUN_DIRS=()
declare -a CKPT_DIRS=()
declare -a JOB_FILES=()
declare -a SESSION_LOGS=()
declare -a WANDB_IDS=()

for index in "${!TASKS[@]}"; do
  task="${TASKS[$index]}"
  gpu="${GPUS[$index]}"
  run_name="${TIMESTAMP}_seed${SEED}_${task}_cnn_cheetah_speed_goal"
  run_dir="$RUNS_DIR/$run_name"
  ckpt_dir="$CKPT_ROOT/$run_name"
  session="cnng_${TIMESTAMP}_s${SEED}_${task}_g${gpu}"
  session="${session//./_}"
  task_slug="${task//_/-}"
  wandb_digest="$(printf '%s' "$TIMESTAMP|$SEED|$task|cnn|cheetah_speed" | sha256sum)"
  wandb_digest="${wandb_digest%% *}"
  wandb_run_id="cnng-s${SEED}-${task_slug}-${wandb_digest:0:32}"

  if path_exists "$run_dir"; then
    echo "Fresh launch refused; run directory already exists: $run_dir" >&2
    exit 1
  fi
  if path_exists "$ckpt_dir"; then
    echo "Fresh launch refused; checkpoint directory already exists: $ckpt_dir" >&2
    exit 1
  fi
  if [[ "$DRY_RUN" -eq 0 ]] && tmux has-session -t "$session" >/dev/null 2>&1; then
    echo "Fresh launch refused; tmux session already exists: $session" >&2
    exit 1
  fi

  SESSIONS+=("$session")
  RUN_DIRS+=("$run_dir")
  CKPT_DIRS+=("$ckpt_dir")
  JOB_FILES+=("$launch_dir/${session}.sh")
  SESSION_LOGS+=("$launch_dir/${session}.log")
  WANDB_IDS+=("$wandb_run_id")
done

mkdir -p "$RUNS_DIR" "$CKPT_ROOT" "$launch_dir"
printf 'session\tgpu\tseed\ttask\tgoal_space\trun_dir\tcheckpoint_dir\tstdout_log\tblocker\twandb_entity\twandb_project\twandb_run_id\tnum_train_frames\teval_every_frames\tnum_eval_episodes\tfinal_tests\trepo_dir\tsource_revision\tpython_bin\tgoals_sha256\tsource_fingerprint\tjob_file\tsession_log\n' > "$manifest"

{
  printf 'python=%s\n' "$PYTHON_BIN"
  "$PYTHON_BIN" --version
  "$PYTHON_BIN" -c 'import importlib.metadata as m; packages=("torch", "hydra-core", "wandb", "dm-control", "numpy"); [print(f"{p}={m.version(p)}") for p in packages]'
} > "$launch_dir/python_environment.txt"

write_job_script() {
  local index="$1"
  local task="${TASKS[$index]}"
  local gpu="${GPUS[$index]}"
  local blocker="${BLOCKERS[$index]}"
  local run_dir="${RUN_DIRS[$index]}"
  local ckpt_dir="${CKPT_DIRS[$index]}"
  local job_file="${JOB_FILES[$index]}"
  local wandb_run_id="${WANDB_IDS[$index]}"
  local wandb_run_name="${TIMESTAMP}_s${SEED}_${task}_cnn_goal"

  {
    printf '#!/usr/bin/env bash\n'
    printf 'set -euo pipefail\n'
    printf 'cd %q\n' "$REPO_DIR"
    printf 'gpu=%q\n' "$gpu"
    printf 'task=%q\n' "$task"
    printf 'blocker=%q\n' "$blocker"
    printf 'run_dir=%q\n' "$run_dir"
    printf 'ckpt_dir=%q\n' "$ckpt_dir"
    printf 'wait_seconds=%q\n' "$WAIT_SECONDS"
    printf 'min_free_mb=%q\n' "$MIN_FREE_MB"
    printf 'max_gpu_util=%q\n' "$MAX_GPU_UTIL"
    printf 'ready_checks=%q\n' "$READY_CHECKS"
    printf 'expected_goals_sha256=%q\n' "$GOALS_SHA256"
    printf 'expected_source_fingerprint=%q\n' "$SOURCE_FINGERPRINT"
    printf 'source_files=(\n'
    for source_file in "${SOURCE_FILES[@]}"; do
      printf '  %q\n' "$source_file"
    done
    printf ')\n'
    cat <<'EOF'

echo "[controller] $(date -u +%FT%TZ) task=$task gpu=$gpu blocker=$blocker"
if [[ "$blocker" != "none" ]]; then
  while tmux has-session -t "$blocker" >/dev/null 2>&1; do
    echo "[queue] $(date -u +%FT%TZ) task=$task gpu=$gpu waiting_for=$blocker"
    sleep "$wait_seconds"
  done
fi

consecutive_ready=0
while (( consecutive_ready < ready_checks )); do
  if sample="$(nvidia-smi -i "$gpu" --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits 2>/dev/null)"; then
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
    echo "[gpu-ready] $(date -u +%FT%TZ) task=$task gpu=$gpu free_mb=$free_mb util=$gpu_util check=$consecutive_ready/$ready_checks"
  else
    consecutive_ready=0
    echo "[gpu-wait] $(date -u +%FT%TZ) task=$task gpu=$gpu free_mb=$free_mb util=$gpu_util required_free_mb=$min_free_mb max_util=$max_gpu_util"
  fi
  if (( consecutive_ready < ready_checks )); then
    sleep "$wait_seconds"
  fi
done

actual_source_fingerprint="$(
  for source_file in "${source_files[@]}"; do
    file_digest="$(sha256sum "$source_file")"
    printf '%s\n' "${file_digest%% *}"
  done | sha256sum
)"
actual_source_fingerprint="${actual_source_fingerprint%% *}"
if [[ "$actual_source_fingerprint" != "$expected_source_fingerprint" ]]; then
  echo "[refused] core source changed while queued: expected=$expected_source_fingerprint actual=$actual_source_fingerprint" >&2
  exit 1
fi
actual_goals_sha256="$(sha256sum url_benchmark/goals.py)"
actual_goals_sha256="${actual_goals_sha256%% *}"
if [[ "$actual_goals_sha256" != "$expected_goals_sha256" ]]; then
  echo "[refused] goal source changed while queued: expected=$expected_goals_sha256 actual=$actual_goals_sha256" >&2
  exit 1
fi
if [[ -e "$run_dir" || -L "$run_dir" || -e "$ckpt_dir" || -L "$ckpt_dir" ]]; then
  echo "[refused] fresh-start collision after queue wait: run_dir=$run_dir ckpt_dir=$ckpt_dir" >&2
  exit 1
fi
mkdir -p "$run_dir/matplotlib"
EOF
    printf 'echo "[start] $(date -u +%%FT%%TZ) task=%s seed=%s gpu=%s obs_type=pixels frame_stack=3 goal_space=cheetah_speed num_train_frames=%s" | tee "$run_dir/launcher.log"\n' \
      "$task" "$SEED" "$gpu" "$NUM_TRAIN_FRAMES"
    printf 'if env CUDA_VISIBLE_DEVICES=%q MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=%q PYTHONUNBUFFERED=1 WANDB_ENTITY=%q WANDB_PROJECT=%q WANDB_RUN_ID=%q WANDB_RUN_NAME=%q WANDB_RESUME=never MPLCONFIGDIR="$run_dir/matplotlib" %q %q ' \
      "$gpu" "$gpu" "$WANDB_ENTITY" "$WANDB_PROJECT" "$wandb_run_id" "$wandb_run_name" "$PYTHON_BIN" "$TRAIN_SCRIPT"
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
      "checkpoint_every=100000" \
      "obs_type=pixels" \
      "frame_stack=3" \
      "render_shape=[84,84]" \
      "action_repeat=2" \
      "goal_space=cheetah_speed" \
      "append_goal_to_observation=False" \
      "agent.lr=$AGENT_LR" \
      "agent.batch_size=1024" \
      "agent.update_every_steps=2" \
      "agent.num_inference_steps=5120" \
      "agent.backward_encoder_grad_scale=1.0" \
      "agent.ortho_coef=1.0" \
      "agent.mix_ratio=0.5" \
      "agent.fb_target_tau=0.01" \
      "agent.lr_coef=1.0" \
      "update_encoder=True" \
      "num_seed_frames=4000" \
      "num_train_frames=$NUM_TRAIN_FRAMES" \
      "eval_every_frames=$EVAL_EVERY_FRAMES" \
      "num_eval_episodes=$NUM_EVAL_EPISODES" \
      "final_tests=$FINAL_TESTS" \
      "experiment=cnn_cheetah_speed_goal_seed${SEED}" \
      "task=$task" \
      "seed=$SEED" \
      "hydra.run.dir=$run_dir"
    printf '> "$run_dir/stdout.log" 2>&1; then\n'
    printf '  rc=0\n'
    printf 'else\n'
    printf '  rc=$?\n'
    printf 'fi\n'
    printf 'echo "[done] $(date -u +%%FT%%TZ) task=%s seed=%s gpu=%s exit=$rc" | tee -a "$run_dir/launcher.log"\n' \
      "$task" "$SEED" "$gpu"
    printf 'exit "$rc"\n'
  } > "$job_file"
  chmod +x "$job_file"
}

for index in "${!TASKS[@]}"; do
  write_job_script "$index"
  printf '%s\t%s\t%s\t%s\tcheetah_speed\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${SESSIONS[$index]}" "${GPUS[$index]}" "$SEED" "${TASKS[$index]}" \
    "${RUN_DIRS[$index]}" "${CKPT_DIRS[$index]}" "${RUN_DIRS[$index]}/stdout.log" \
    "${BLOCKERS[$index]}" "$WANDB_ENTITY" "$WANDB_PROJECT" "${WANDB_IDS[$index]}" \
    "$NUM_TRAIN_FRAMES" "$EVAL_EVERY_FRAMES" "$NUM_EVAL_EPISODES" "$FINAL_TESTS" \
    "$REPO_DIR" "$SOURCE_REVISION" "$PYTHON_BIN" "$GOALS_SHA256" "$SOURCE_FINGERPRINT" \
    "${JOB_FILES[$index]}" "${SESSION_LOGS[$index]}" \
    >> "$manifest"
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "${SESSIONS[$index]}" "${GPUS[$index]}" "${TASKS[$index]}" "${WANDB_IDS[$index]}" "${RUN_DIRS[$index]}"
done

if [[ "$DRY_RUN" -eq 0 ]]; then
  launch_failures=0
  for index in "${!TASKS[@]}"; do
    session="${SESSIONS[$index]}"
    job_file="${JOB_FILES[$index]}"
    session_log="${SESSION_LOGS[$index]}"
    if ! tmux new-session -d -s "$session" \
        "bash $(printf '%q' "$job_file") > $(printf '%q' "$session_log") 2>&1"; then
      echo "Failed to create tmux controller: $session" >&2
      launch_failures=1
    fi
  done
  for session in "${SESSIONS[@]}"; do
    if ! tmux has-session -t "$session" >/dev/null 2>&1; then
      echo "Missing tmux controller after launch: $session" >&2
      launch_failures=1
    fi
  done
  if [[ "$launch_failures" -ne 0 ]]; then
    echo "One or more controllers failed to launch; inspect: $manifest" >&2
    exit 1
  fi
  echo "Queued ${#TASKS[@]} independent task controllers."
else
  echo "Dry run only; no tmux sessions were launched."
fi
echo "Manifest: $manifest"
