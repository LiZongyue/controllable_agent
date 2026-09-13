#!/usr/bin/env bash
set -euo pipefail

# DINO CLS3 Cheetah pure-visual FB stability sweep requested on 2026-08-28.
# Nineteen fresh seed-1 jobs share a dynamic queue.  The already-completed
# run/fb_lr=5e-5/ortho=1 control is intentionally not part of this matrix.

REPO_DIR="${REPO_DIR:-/data/fanfeng/controllable_agent}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-$REPO_DIR/url_benchmark/pretrain.py}"
PYTHON_BIN="${PYTHON_BIN:-/data/fan2/env/miniconda3/envs/occ_rlu/bin/python}"
RUNS_ROOT="${RUNS_ROOT:-/mnt/data_7tb/fanfeng/controllable_agent_runs}"
CKPT_ROOT="${CKPT_ROOT:-/mnt/data_nvme1/fanfeng/controllable_agent_ckpt/20260828_cheetah_fb_cls3_ortho_lr_s1}"
LAUNCH_ROOT="${LAUNCH_ROOT:-$RUNS_ROOT/launch_queues}"
CAMPAIGN="${CAMPAIGN:-20260828_cheetah_fb_cls3_ortho_lr_s1}"
WANDB_ENTITY="${WANDB_ENTITY:-lmu_rl}"
WANDB_PROJECT="${WANDB_PROJECT:-controllable_agent_cheetah_fb_stability}"

SEED=1
NUM_TRAIN_FRAMES=2000010
EVAL_EVERY_FRAMES=10000
NUM_EVAL_EPISODES=10
FINAL_TESTS=10
CHECKPOINT_EVERY=100000
SNAPSHOT_AT="[100000,200000,500000,800000,1000000,1500000,2000000]"
AGENT_LR=0.0001
AGENT_BATCH_SIZE=1024
AGENT_UPDATE_EVERY_STEPS=2
ACTION_REPEAT=2
DINO_FRAME_STACK=3
DINO_MODEL_NAME="facebook/dinov2-base"

# Default schedule: 12 workers can claim immediately on the three presently
# low-utilization GPUs.  Seven workers on the other GPUs wait for their
# per-GPU readiness gates before claiming a job.
GPU_SLOTS_STRING="${GPU_SLOTS:-0:3 3:5 5:4 1:2 2:1 4:1 6:2 7:1}"
WAIT_SECONDS="${WAIT_SECONDS:-60}"
MIN_FREE_MB="${MIN_FREE_MB:-70000}"
MAX_GPU_UTIL="${MAX_GPU_UTIL:-20}"
READY_CHECKS="${READY_CHECKS:-2}"
GPU_MIN_FREE_MB_OVERRIDES="${GPU_MIN_FREE_MB_OVERRIDES:-0:9000 3:10000 5:18000 1:65000 2:43000 4:38000 6:70000 7:53000}"
GPU_MAX_UTIL_OVERRIDES="${GPU_MAX_UTIL_OVERRIDES:-0:100 3:100 5:100}"
GPU_READY_CHECKS_OVERRIDES="${GPU_READY_CHECKS_OVERRIDES:-0:1 3:1 5:1}"
MIN_RUNS_FREE_GIB="${MIN_RUNS_FREE_GIB:-20}"
MIN_CKPT_FREE_GIB="${MIN_CKPT_FREE_GIB:-30}"
MAX_CONSECUTIVE_FAILURES="${MAX_CONSECUTIVE_FAILURES:-1}"

DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: ./launch_cheetah_fb_ortho_lr_sweep.sh [--dry-run] [--campaign NAME]

Creates the 19-job DINO CLS3 Cheetah pure-visual FB stability matrix:
  4 tasks x fb_lr {1e-4, 5e-5} x ortho {2, 4} = 16
  walk/run_backward/walk_backward x fb_lr 5e-5 x ortho 1 = 3

The completed run/fb_lr=5e-5/ortho=1 control is excluded.  A real launch
starts dynamic tmux workers; --dry-run only writes the manifest, job scripts,
and queue state.

Useful overrides:
  REPO_DIR, TRAIN_SCRIPT, PYTHON_BIN, RUNS_ROOT, CKPT_ROOT, LAUNCH_ROOT
  CAMPAIGN, WANDB_ENTITY, WANDB_PROJECT
  GPU_SLOTS="0:3 3:5 5:4 1:2 2:1 4:1 6:2 7:1"
  GPU_MIN_FREE_MB_OVERRIDES="0:9000 3:10000 ..."
  GPU_MAX_UTIL_OVERRIDES="0:100 3:100 5:100"
  GPU_READY_CHECKS_OVERRIDES="0:1 3:1 5:1"
  WAIT_SECONDS=60 MIN_RUNS_FREE_GIB=20 MIN_CKPT_FREE_GIB=30
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --campaign)
      [[ $# -ge 2 ]] || { echo "--campaign requires a value" >&2; exit 2; }
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

is_nonnegative_integer() { [[ "$1" =~ ^[0-9]+$ ]]; }
is_positive_integer() { [[ "$1" =~ ^[1-9][0-9]*$ ]]; }
path_exists() { [[ -e "$1" || -L "$1" ]]; }

if [[ ! "$CAMPAIGN" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "Invalid CAMPAIGN: $CAMPAIGN" >&2
  exit 2
fi
for value_name in WAIT_SECONDS MIN_FREE_MB READY_CHECKS MIN_RUNS_FREE_GIB MIN_CKPT_FREE_GIB MAX_CONSECUTIVE_FAILURES; do
  value="${!value_name}"
  if ! is_positive_integer "$value"; then
    echo "$value_name must be a positive integer: $value" >&2
    exit 2
  fi
done
if ! is_nonnegative_integer "$MAX_GPU_UTIL" || (( MAX_GPU_UTIL > 100 )); then
  echo "MAX_GPU_UTIL must be in [0,100]: $MAX_GPU_UTIL" >&2
  exit 2
fi
if [[ ! -d "$REPO_DIR" || ! -f "$TRAIN_SCRIPT" ]]; then
  echo "Missing source repo or training script: $REPO_DIR / $TRAIN_SCRIPT" >&2
  exit 1
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python is not executable: $PYTHON_BIN" >&2
  exit 1
fi
for required_command in flock sha256sum awk find sort; do
  command -v "$required_command" >/dev/null 2>&1 || {
    echo "$required_command is required" >&2
    exit 1
  }
done
if [[ "$DRY_RUN" -eq 0 ]]; then
  for required_command in nvidia-smi pgrep tmux df; do
    command -v "$required_command" >/dev/null 2>&1 || {
      echo "$required_command is required for a real launch" >&2
      exit 1
    }
  done
fi

declare -a SLOT_GPUS=()
declare -a SLOT_IDS=()
declare -A CONFIGURED_GPUS=()
for slot_spec in $GPU_SLOTS_STRING; do
  if [[ ! "$slot_spec" =~ ^([0-9]+):([1-9][0-9]*)$ ]]; then
    echo "GPU_SLOTS entries must be GPU:COUNT: $slot_spec" >&2
    exit 2
  fi
  gpu="${BASH_REMATCH[1]}"
  count="${BASH_REMATCH[2]}"
  if [[ -n "${CONFIGURED_GPUS[$gpu]+present}" ]]; then
    echo "GPU_SLOTS contains GPU $gpu more than once" >&2
    exit 2
  fi
  CONFIGURED_GPUS["$gpu"]=1
  for ((slot = 1; slot <= count; slot++)); do
    SLOT_GPUS+=("$gpu")
    SLOT_IDS+=("g${gpu}s${slot}")
  done
done
if [[ "${#SLOT_GPUS[@]}" -eq 0 ]]; then
  echo "At least one GPU worker slot is required" >&2
  exit 2
fi

validate_override_map() {
  local map_name="$1" map_value="$2" kind="$3"
  local entry key value
  for entry in $map_value; do
    if [[ ! "$entry" =~ ^([0-9]+):([0-9]+)$ ]]; then
      echo "$map_name entries must be GPU:VALUE: $entry" >&2
      exit 2
    fi
    key="${BASH_REMATCH[1]}"
    value="${BASH_REMATCH[2]}"
    if [[ -z "${CONFIGURED_GPUS[$key]+present}" ]]; then
      echo "$map_name references unconfigured GPU $key" >&2
      exit 2
    fi
    if [[ "$kind" == percent ]]; then
      (( value <= 100 )) || { echo "$map_name value exceeds 100: $entry" >&2; exit 2; }
    elif ! is_positive_integer "$value"; then
      echo "$map_name values must be positive: $entry" >&2
      exit 2
    fi
  done
}

validate_override_map GPU_MIN_FREE_MB_OVERRIDES "$GPU_MIN_FREE_MB_OVERRIDES" positive
validate_override_map GPU_MAX_UTIL_OVERRIDES "$GPU_MAX_UTIL_OVERRIDES" percent
validate_override_map GPU_READY_CHECKS_OVERRIDES "$GPU_READY_CHECKS_OVERRIDES" positive

if [[ "$DRY_RUN" -eq 0 ]]; then
  for gpu in "${!CONFIGURED_GPUS[@]}"; do
    nvidia-smi --id="$gpu" --query-gpu=name --format=csv,noheader >/dev/null 2>&1 || {
      echo "GPU ordinal is unavailable: $gpu" >&2
      exit 1
    }
  done
fi

mapfile -t SOURCE_FILES < <(
  cd "$REPO_DIR"
  find url_benchmark -type f \
    ! -path '*/__pycache__/*' \
    ! -name '*.pyc' \
    -print | LC_ALL=C sort
)
if [[ "${#SOURCE_FILES[@]}" -eq 0 ]]; then
  echo "No source files found under $REPO_DIR/url_benchmark" >&2
  exit 1
fi
for source_file in "${SOURCE_FILES[@]}"; do
  [[ -f "$REPO_DIR/$source_file" ]] || {
    echo "Required source file is missing: $REPO_DIR/$source_file" >&2
    exit 1
  }
done

source_fingerprint() {
  local root="$1" source_file digest
  for source_file in "${SOURCE_FILES[@]}"; do
    digest="$(sha256sum "$root/$source_file")"
    printf '%s\n' "${digest%% *}"
  done | sha256sum | awk '{print $1}'
}
SOURCE_FINGERPRINT="$(source_fingerprint "$REPO_DIR")"

declare -a JOB_IDS=()
declare -a TASKS=()
declare -a FB_LRS=()
declare -a FB_LR_LABELS=()
declare -a ORTHOS=()
declare -a ORTHO_LABELS=()
declare -a RUN_NAMES=()
declare -a RUN_DIRS=()
declare -a CKPT_DIRS=()

add_job() {
  local task="$1" fb_lr="$2" lr_label="$3" ortho="$4" ortho_label="$5"
  local ordinal job_id run_name
  ordinal=$(( ${#JOB_IDS[@]} + 1 ))
  job_id="$(printf 'job%03d' "$ordinal")"
  run_name="dino_cls3_${task}_seed${SEED}_${lr_label}_${ortho_label}"
  JOB_IDS+=("$job_id")
  TASKS+=("$task")
  FB_LRS+=("$fb_lr")
  FB_LR_LABELS+=("$lr_label")
  ORTHOS+=("$ortho")
  ORTHO_LABELS+=("$ortho_label")
  RUN_NAMES+=("$run_name")
  RUN_DIRS+=("$RUNS_ROOT/$CAMPAIGN/$run_name")
  CKPT_DIRS+=("$CKPT_ROOT/$run_name")
}

# Complete the ortho=1/fb_lr=5e-5 control row first.  The run control already
# exists in W&B, so only these three fresh cells are queued.
add_job cheetah_walk          0.00005 fblr5e5 1.0 ortho1
add_job cheetah_run_backward  0.00005 fblr5e5 1.0 ortho1
add_job cheetah_walk_backward 0.00005 fblr5e5 1.0 ortho1

TASK_ORDER=(cheetah_walk cheetah_run cheetah_run_backward cheetah_walk_backward)
for fb_spec in "0.0001|fblr1e4" "0.00005|fblr5e5"; do
  IFS='|' read -r fb_lr lr_label <<< "$fb_spec"
  for ortho_spec in "2.0|ortho2" "4.0|ortho4"; do
    IFS='|' read -r ortho ortho_label <<< "$ortho_spec"
    for task in "${TASK_ORDER[@]}"; do
      add_job "$task" "$fb_lr" "$lr_label" "$ortho" "$ortho_label"
    done
  done
done

if [[ "${#JOB_IDS[@]}" -ne 19 ]]; then
  echo "Internal matrix error: expected 19 jobs, got ${#JOB_IDS[@]}" >&2
  exit 1
fi
declare -A SEEN_RUN_NAMES=()
for run_name in "${RUN_NAMES[@]}"; do
  if [[ -n "${SEEN_RUN_NAMES[$run_name]+present}" ]]; then
    echo "Internal duplicate run name: $run_name" >&2
    exit 1
  fi
  SEEN_RUN_NAMES["$run_name"]=1
done

launch_dir="$LAUNCH_ROOT/$CAMPAIGN"
manifest="$launch_dir/manifest.tsv"
queue_file="$launch_dir/priority_queue.tsv"
jobs_dir="$launch_dir/jobs"
pending_dir="$launch_dir/queue/pending"
claimed_dir="$launch_dir/queue/claimed"
completed_dir="$launch_dir/queue/completed"
failed_dir="$launch_dir/queue/failed"
cancelled_dir="$launch_dir/queue/cancelled"
queue_lock="$launch_dir/queue/claim.lock"
worker_script="$launch_dir/worker.sh"
worker_config="$launch_dir/worker_config.sh"

if path_exists "$launch_dir"; then
  echo "Fresh launch refused; launch directory exists: $launch_dir" >&2
  exit 1
fi
for index in "${!JOB_IDS[@]}"; do
  for target in "${RUN_DIRS[$index]}" "${CKPT_DIRS[$index]}"; do
    if path_exists "$target"; then
      echo "Fresh launch refused; target exists: $target" >&2
      exit 1
    fi
  done
  if [[ "$DRY_RUN" -eq 0 ]] && pgrep -f -- "${RUN_NAMES[$index]}" >/dev/null 2>&1; then
    echo "Fresh launch refused; matching process exists: ${RUN_NAMES[$index]}" >&2
    exit 1
  fi
done

declare -a WORKER_SESSIONS=()
for index in "${!SLOT_IDS[@]}"; do
  session="cfbos_${CAMPAIGN}_${SLOT_IDS[$index]}"
  session="${session//./_}"
  WORKER_SESSIONS+=("$session")
  if [[ "$DRY_RUN" -eq 0 ]] && tmux has-session -t "$session" >/dev/null 2>&1; then
    echo "Fresh launch refused; tmux session exists: $session" >&2
    exit 1
  fi
done

mkdir -p "$RUNS_ROOT/$CAMPAIGN" "$CKPT_ROOT" "$jobs_dir" "$pending_dir" \
  "$claimed_dir" "$completed_dir" "$failed_dir" "$cancelled_dir"
: > "$queue_lock"

printf 'job_id\tpriority\ttask\tgoal_space\tcustom_reward\tseed\tobs_type\tdino_frame_stack\tagent_lr\tagent_fb_lr\tagent_ortho_coef\tagent_batch_size\tagent_idm_coef\tnum_train_frames\teval_every_frames\tnum_eval_episodes\tfinal_tests\tsnapshot_at\trun_name\trun_dir\tcheckpoint_dir\tstdout_log\twandb_entity\twandb_project\twandb_run_id\twandb_url\tsource_fingerprint\tjob_file\n' > "$manifest"
printf 'job_id\tpriority\ttask\tagent_fb_lr\tagent_ortho_coef\trun_name\tjob_file\tpending_file\n' > "$queue_file"

write_job_script() {
  local index="$1" job_file="$2"
  local task="${TASKS[$index]}" fb_lr="${FB_LRS[$index]}" ortho="${ORTHOS[$index]}"
  local run_name="${RUN_NAMES[$index]}" run_dir="${RUN_DIRS[$index]}" ckpt_dir="${CKPT_DIRS[$index]}"

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
  echo "[refused] queued source changed: expected=$expected_source_fingerprint actual=$actual_source_fingerprint" >&2
  exit 1
fi
if [[ -e "$run_dir" || -L "$run_dir" || -e "$ckpt_dir" || -L "$ckpt_dir" ]]; then
  echo "[refused] fresh-start collision: run_dir=$run_dir ckpt_dir=$ckpt_dir" >&2
  exit 1
fi
mkdir -p "$run_dir/matplotlib"
EOF
    printf 'echo "[start] $(date -u +%%FT%%TZ) run=%s gpu=$gpu goal_space=null custom_reward=null fb_lr=%s ortho=%s frames=%s" | tee "$run_dir/launcher.log"\n' \
      "$run_name" "$fb_lr" "$ortho" "$NUM_TRAIN_FRAMES"
    printf 'if env CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID="$gpu" PYTHONUNBUFFERED=1 WANDB_ENTITY=%q WANDB_PROJECT=%q WANDB_RUN_NAME=%q WANDB_RUN_ID=%q WANDB_RESUME=never MPLCONFIGDIR="$run_dir/matplotlib" %q %q ' \
      "$WANDB_ENTITY" "$WANDB_PROJECT" "$run_name" "$run_name" "$PYTHON_BIN" "$TRAIN_SCRIPT"
    printf '%q ' \
      "agent=fb_ddpg" \
      "device=cuda" \
      "use_wandb=True" \
      "use_tb=False" \
      "use_hiplog=False" \
      "save_video=False" \
      "save_train_video=False" \
      "save_replay_buffer_in_checkpoint=False" \
      "auto_resume=False" \
      "load_model=null" \
      "load_replay_buffer=null" \
      "checkpoint_root=$CKPT_ROOT" \
      "checkpoint_every=$CHECKPOINT_EVERY" \
      "snapshot_at=$SNAPSHOT_AT" \
      "obs_type=dino" \
      "dino_model_name=$DINO_MODEL_NAME" \
      "use_cls=True" \
      "frame_stack=3" \
      "dino_frame_stack=$DINO_FRAME_STACK" \
      "render_shape=[224,224]" \
      "action_repeat=$ACTION_REPEAT" \
      "goal_space=null" \
      "custom_reward=null" \
      "append_goal_to_observation=False" \
      "discount=0.99" \
      "future=0.99" \
      "agent.dino_use_adapter=True" \
      "agent.dino_adapter_type=linear" \
      "agent.dino_adapter_output_dim=512" \
      "agent.dino_separate_backward_adapter=False" \
      "agent.backward_encoder_grad_scale=1.0" \
      "agent.lr=$AGENT_LR" \
      "agent.fb_lr=$fb_lr" \
      "agent.lr_coef=1.0" \
      "agent.fb_target_tau=0.01" \
      "agent.batch_size=$AGENT_BATCH_SIZE" \
      "agent.update_every_steps=$AGENT_UPDATE_EVERY_STEPS" \
      "agent.num_inference_steps=5120" \
      "agent.hidden_dim=1024" \
      "agent.backward_hidden_dim=526" \
      "agent.feature_dim=512" \
      "agent.z_dim=50" \
      "agent.stddev_schedule=0.2" \
      "agent.stddev_clip=0.3" \
      "agent.update_z_every_step=300" \
      "agent.update_z_proba=1.0" \
      "agent.nstep=1" \
      "agent.ortho_coef=$ortho" \
      "agent.future_ratio=0.0" \
      "agent.mix_ratio=0.5" \
      "agent.rand_weight=False" \
      "agent.preprocess=True" \
      "agent.norm_z=True" \
      "agent.q_loss=False" \
      "agent.q_loss_coef=0.01" \
      "agent.boltzmann=False" \
      "agent.add_trunk=False" \
      "agent.idm_coef=0.0" \
      "agent.idm_lr=null" \
      "update_encoder=True" \
      "num_seed_frames=4000" \
      "replay_buffer_episodes=5000" \
      "num_train_frames=$NUM_TRAIN_FRAMES" \
      "eval_every_frames=$EVAL_EVERY_FRAMES" \
      "num_eval_episodes=$NUM_EVAL_EPISODES" \
      "final_tests=$FINAL_TESTS" \
      "experiment=cheetah_fb_stability_ortho_lr" \
      "task=$task" \
      "seed=$SEED" \
      "hydra.run.dir=$run_dir"
    printf '> "$run_dir/stdout.log" 2>&1; then\n'
    printf '  rc=0\n'
    printf 'else\n'
    printf '  rc=$?\n'
    printf 'fi\n'
    printf 'echo "[done] $(date -u +%%FT%%TZ) run=%s gpu=$gpu exit=$rc" | tee -a "$run_dir/launcher.log"\n' "$run_name"
    printf 'exit "$rc"\n'
  } > "$job_file"
  chmod +x "$job_file"
}

for index in "${!JOB_IDS[@]}"; do
  job_id="${JOB_IDS[$index]}"
  priority=$((index + 1))
  task="${TASKS[$index]}"
  run_name="${RUN_NAMES[$index]}"
  job_file="$jobs_dir/${job_id}_${run_name}.sh"
  pending_file="$pending_dir/$(printf '%03d' "$priority")_${job_id}_${run_name}.pending"
  write_job_script "$index" "$job_file"
  {
    printf 'job_id=%q\n' "$job_id"
    printf 'priority=%q\n' "$priority"
    printf 'task=%q\n' "$task"
    printf 'fb_lr=%q\n' "${FB_LRS[$index]}"
    printf 'ortho=%q\n' "${ORTHOS[$index]}"
    printf 'run_name=%q\n' "$run_name"
    printf 'job_file=%q\n' "$job_file"
  } > "$pending_file"
  wandb_url="https://wandb.ai/$WANDB_ENTITY/$WANDB_PROJECT/runs/$run_name"
  printf '%s\t%s\t%s\tnull\tnull\t%s\tdino\t%s\t%s\t%s\t%s\t%s\t0.0\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$job_id" "$priority" "$task" "$SEED" "$DINO_FRAME_STACK" "$AGENT_LR" \
    "${FB_LRS[$index]}" "${ORTHOS[$index]}" "$AGENT_BATCH_SIZE" "$NUM_TRAIN_FRAMES" \
    "$EVAL_EVERY_FRAMES" "$NUM_EVAL_EPISODES" "$FINAL_TESTS" "$SNAPSHOT_AT" "$run_name" \
    "${RUN_DIRS[$index]}" "${CKPT_DIRS[$index]}" "${RUN_DIRS[$index]}/stdout.log" \
    "$WANDB_ENTITY" "$WANDB_PROJECT" "$run_name" "$wandb_url" "$SOURCE_FINGERPRINT" "$job_file" \
    >> "$manifest"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$job_id" "$priority" "$task" "${FB_LRS[$index]}" "${ORTHOS[$index]}" \
    "$run_name" "$job_file" "$pending_file" >> "$queue_file"
done

{
  printf '#!/usr/bin/env bash\n'
  printf 'pending_dir=%q\n' "$pending_dir"
  printf 'claimed_dir=%q\n' "$claimed_dir"
  printf 'completed_dir=%q\n' "$completed_dir"
  printf 'failed_dir=%q\n' "$failed_dir"
  printf 'queue_lock=%q\n' "$queue_lock"
  printf 'runs_root=%q\n' "$RUNS_ROOT"
  printf 'ckpt_root=%q\n' "$CKPT_ROOT"
  printf 'wait_seconds=%q\n' "$WAIT_SECONDS"
  printf 'default_min_free_mb=%q\n' "$MIN_FREE_MB"
  printf 'default_max_gpu_util=%q\n' "$MAX_GPU_UTIL"
  printf 'default_ready_checks=%q\n' "$READY_CHECKS"
  printf 'gpu_min_free_mb_overrides=%q\n' "$GPU_MIN_FREE_MB_OVERRIDES"
  printf 'gpu_max_util_overrides=%q\n' "$GPU_MAX_UTIL_OVERRIDES"
  printf 'gpu_ready_checks_overrides=%q\n' "$GPU_READY_CHECKS_OVERRIDES"
  printf 'min_runs_free_kib=%q\n' "$((MIN_RUNS_FREE_GIB * 1024 * 1024))"
  printf 'min_ckpt_free_kib=%q\n' "$((MIN_CKPT_FREE_GIB * 1024 * 1024))"
  printf 'max_consecutive_failures=%q\n' "$MAX_CONSECUTIVE_FAILURES"
} > "$worker_config"
chmod +x "$worker_config"

cat > "$worker_script" <<'EOF'
#!/usr/bin/env bash
set -uo pipefail

gpu="${1:?GPU ordinal is required}"
slot_id="${2:?worker slot ID is required}"
config_file="${3:?worker config is required}"
source "$config_file"

lookup_override() {
  local map_value="$1" requested_gpu="$2" fallback="$3"
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

disk_free_kib() {
  df -Pk "$1" | awk 'END {print $4}'
}

wait_for_resources() {
  local consecutive_ready=0 sample free_mb gpu_util runs_free_kib ckpt_free_kib
  while (( consecutive_ready < ready_checks )); do
    if ! queue_has_pending; then
      return 1
    fi
    if sample="$(nvidia-smi --id="$gpu" --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits 2>/dev/null)"; then
      IFS=',' read -r free_mb gpu_util <<< "$sample"
      free_mb="${free_mb//[[:space:]]/}"
      gpu_util="${gpu_util//[[:space:]]/}"
    else
      free_mb=unavailable
      gpu_util=unavailable
    fi
    runs_free_kib="$(disk_free_kib "$runs_root" 2>/dev/null || echo 0)"
    ckpt_free_kib="$(disk_free_kib "$ckpt_root" 2>/dev/null || echo 0)"
    if [[ "$free_mb" =~ ^[0-9]+$ && "$gpu_util" =~ ^[0-9]+$ \
          && "$runs_free_kib" =~ ^[0-9]+$ && "$ckpt_free_kib" =~ ^[0-9]+$ ]] \
        && (( free_mb >= min_free_mb && gpu_util <= max_gpu_util \
              && runs_free_kib >= min_runs_free_kib && ckpt_free_kib >= min_ckpt_free_kib )); then
      consecutive_ready=$((consecutive_ready + 1))
      echo "[ready] $(date -u +%FT%TZ) gpu=$gpu free_mb=$free_mb util=$gpu_util disks_kib=$runs_free_kib/$ckpt_free_kib check=$consecutive_ready/$ready_checks"
    else
      consecutive_ready=0
      echo "[wait] $(date -u +%FT%TZ) gpu=$gpu free_mb=$free_mb util=$gpu_util required=${min_free_mb}MB/${max_gpu_util}% disks_kib=$runs_free_kib/$ckpt_free_kib"
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
  mv "$source_file" "$claim_file" || {
    flock -u "$lock_fd"
    exec {lock_fd}>&-
    return 2
  }
  flock -u "$lock_fd"
  exec {lock_fd}>&-
  printf '%s\n' "$claim_file"
}

echo "[worker-start] $(date -u +%FT%TZ) gpu=$gpu slot=$slot_id min_free_mb=$min_free_mb max_util=$max_gpu_util ready_checks=$ready_checks"
completed=0
failures=0
consecutive_failures=0
while queue_has_pending; do
  if ! wait_for_resources; then
    break
  fi
  if ! claim_file="$(claim_next_job)"; then
    continue
  fi
  unset job_id priority task fb_lr ortho run_name job_file
  source "$claim_file"
  echo "[claimed] $(date -u +%FT%TZ) gpu=$gpu slot=$slot_id job=$job_id priority=$priority run=$run_name"
  if bash "$job_file" "$gpu"; then
    rc=0
    completed=$((completed + 1))
    consecutive_failures=0
    destination="$completed_dir/${claim_file##*/}"
    destination="${destination%.claimed}.done"
  else
    rc=$?
    failures=$((failures + 1))
    consecutive_failures=$((consecutive_failures + 1))
    destination="$failed_dir/${claim_file##*/}"
    destination="${destination%.claimed}.rc${rc}.failed"
  fi
  printf 'finished_at=%q\ngpu=%q\nexit_code=%q\n' "$(date -u +%FT%TZ)" "$gpu" "$rc" >> "$claim_file"
  mv "$claim_file" "$destination"
  echo "[finished] $(date -u +%FT%TZ) gpu=$gpu slot=$slot_id job=$job_id run=$run_name exit=$rc"
  if (( consecutive_failures >= max_consecutive_failures )); then
    echo "[worker-stop] $(date -u +%FT%TZ) consecutive_failures=$consecutive_failures"
    break
  fi
done
echo "[worker-done] $(date -u +%FT%TZ) gpu=$gpu slot=$slot_id completed=$completed failures=$failures"
(( failures == 0 ))
EOF
chmod +x "$worker_script"

{
  printf 'campaign=%s\n' "$CAMPAIGN"
  printf 'created_at=%s\n' "$(date -u +%FT%TZ)"
  printf 'dry_run=%s\n' "$DRY_RUN"
  printf 'jobs=%s\n' "${#JOB_IDS[@]}"
  printf 'worker_slots=%s\n' "${#SLOT_IDS[@]}"
  printf 'slot_ids=%s\n' "${SLOT_IDS[*]}"
  printf 'gpu_slots=%s\n' "$GPU_SLOTS_STRING"
  printf 'source_repo=%s\n' "$REPO_DIR"
  printf 'source_fingerprint=%s\n' "$SOURCE_FINGERPRINT"
  printf 'python=%s\n' "$PYTHON_BIN"
  "$PYTHON_BIN" --version 2>&1
  printf 'wandb=%s/%s\n' "$WANDB_ENTITY" "$WANDB_PROJECT"
  printf 'excluded_completed_control=dino_cls3_cheetah_run_seed1_fblr5e5\n'
  printf 'excluded_control_url=https://wandb.ai/lmu_rl/controllable_agent_cheetah_fb_stability/runs/dino_cls3_cheetah_run_seed1_fblr5e5\n'
  printf 'num_train_frames=%s\n' "$NUM_TRAIN_FRAMES"
  printf 'snapshot_at=%s\n' "$SNAPSHOT_AT"
  printf 'min_runs_free_gib=%s\n' "$MIN_RUNS_FREE_GIB"
  printf 'min_ckpt_free_gib=%s\n' "$MIN_CKPT_FREE_GIB"
} > "$launch_dir/launch_config.txt"

echo "Manifest: $manifest"
echo "Queue: $queue_file"
echo "Jobs: ${#JOB_IDS[@]} (16 factorial + 3 controls; one completed run control excluded)"
if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "Dry run only; no tmux workers launched."
  exit 0
fi

launch_failures=0
for index in "${!SLOT_IDS[@]}"; do
  gpu="${SLOT_GPUS[$index]}"
  slot_id="${SLOT_IDS[$index]}"
  session="${WORKER_SESSIONS[$index]}"
  worker_log="$launch_dir/worker_${slot_id}.log"
  command="bash $(printf '%q' "$worker_script") $(printf '%q' "$gpu") $(printf '%q' "$slot_id") $(printf '%q' "$worker_config") > $(printf '%q' "$worker_log") 2>&1"
  if tmux new-session -d -s "$session" "$command"; then
    printf '%s\t%s\t%s\t%s\n' "$session" "$gpu" "$slot_id" "$worker_log"
  else
    echo "Failed to start worker: $session" >&2
    launch_failures=1
  fi
done
sleep 2
for session in "${WORKER_SESSIONS[@]}"; do
  if ! tmux has-session -t "$session" >/dev/null 2>&1; then
    echo "Worker session missing after launch: $session" >&2
    launch_failures=1
  fi
done
if [[ "$launch_failures" -ne 0 ]]; then
  echo "One or more workers failed; inspect $launch_dir" >&2
  exit 1
fi
echo "Launched ${#SLOT_IDS[@]} dynamic workers for ${#JOB_IDS[@]} jobs."
