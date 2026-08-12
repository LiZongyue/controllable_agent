#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/data/fanfeng/controllable_agent}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-$REPO_DIR/url_benchmark/pretrain.py}"
RUNS_DIR="${RUNS_DIR:-/mnt/data_7tb/fanfeng/controllable_agent_runs}"
CKPT_ROOT="${CKPT_ROOT:-/mnt/data_7tb/fanfeng/controallable_agent_ckpt}"
LAUNCH_ROOT="${LAUNCH_ROOT:-$RUNS_DIR/launch_queues}"
WANDB_PROJECT="${WANDB_PROJECT:-controllable_agent_baseline}"
SEED="${SEED:-1}"
TASKS="${TASKS:-}"
EVAL_EVERY_FRAMES="${EVAL_EVERY_FRAMES:-1000}"
NUM_TRAIN_FRAMES="${NUM_TRAIN_FRAMES:-2000010}"
AGENT_LR="${AGENT_LR:-0.0001}"
IDM_COEF="${IDM_COEF:-0.0}"
IDM_LR="${IDM_LR:-}"
IDM_DIAGNOSTICS_INTERVAL="${IDM_DIAGNOSTICS_INTERVAL:-500}"
STAGE="${STAGE:-}"
TIMESTAMP="${TIMESTAMP:-$(date -u +%Y%m%d_%H%M%S)_dino_cls_stack3}"
DRY_RUN=0
RESUME=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --resume)
      RESUME=1
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
      cat <<'EOF'
Usage: ./launch_dino_cls_stack3_12tasks.sh [--dry-run] [--resume] [--timestamp NAME]

Launch task-specific DINOv2 CLS three-frame-stack FB runs. Tasks assigned to
the same GPU are placed in one serial queue; every task still has its own
Hydra run directory, checkpoint identity, and W&B run.

Fresh start is the default. If a selected run directory or checkpoint
directory already exists, the launcher refuses to proceed. Pass --resume only
when intentionally continuing the exact same setting.

Environment overrides:
  GPUS="6 7"                 Unique GPU ordinals used round-robin.
  SEED=1
  TASKS="cheetah_walk"       Optional whitespace-separated task subset.
  STAGE=stage1               Optional stable identity suffix; use stage1,
                             stage2, or final when a timestamp is shared.
  NUM_TRAIN_FRAMES=2000010   Set 500000 for Stage 1/2 screening.
  EVAL_EVERY_FRAMES=1000
  AGENT_LR=0.0001            FB learning rate; kept fixed for IDM ablations.
  IDM_COEF=0.0               Inverse-dynamics auxiliary coefficient.
  IDM_LR=                    Separate IDM-head LR. Empty uses AGENT_LR.
  IDM_DIAGNOSTICS_INTERVAL=500
                             Expensive component-gradient logging interval,
                             counted in optimizer updates.
  WANDB_PROJECT=controllable_agent_baseline
  REPO_DIR, TRAIN_SCRIPT, RUNS_DIR, CKPT_ROOT, LAUNCH_ROOT, TIMESTAMP

The setting suffix is composed as:
  [_STAGE][_idmCOEF][_idmlrLR][_fNUM_TRAIN_FRAMES]

The frame suffix is omitted only for the historical default of 2,000,010
frames. GPU ordinals are deliberately excluded from run/checkpoint identity.
EOF
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

is_number() {
  local value="$1"
  [[ "$value" =~ ^[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$ ]] \
    && awk -v value="$value" 'BEGIN { number = value + 0; exit !(number >= -1e300 && number <= 1e300) }'
}

is_positive_number() {
  is_number "$1" && awk -v value="$1" 'BEGIN { exit !(value + 0 > 0) }'
}

is_nonnegative_number() {
  is_number "$1" && awk -v value="$1" 'BEGIN { exit !(value + 0 >= 0) }'
}

if [[ ! "$NUM_TRAIN_FRAMES" =~ ^[1-9][0-9]*$ ]]; then
  echo "NUM_TRAIN_FRAMES must be a positive integer, got: $NUM_TRAIN_FRAMES" >&2
  exit 2
fi
if [[ ! "$IDM_DIAGNOSTICS_INTERVAL" =~ ^[1-9][0-9]*$ ]]; then
  echo "IDM_DIAGNOSTICS_INTERVAL must be a positive integer, got: $IDM_DIAGNOSTICS_INTERVAL" >&2
  exit 2
fi
if ! is_positive_number "$AGENT_LR"; then
  echo "AGENT_LR must be a positive number, got: $AGENT_LR" >&2
  exit 2
fi
if ! is_nonnegative_number "$IDM_COEF"; then
  echo "IDM_COEF must be a non-negative number, got: $IDM_COEF" >&2
  exit 2
fi
if [[ -n "$IDM_LR" ]] && ! is_positive_number "$IDM_LR"; then
  echo "IDM_LR must be empty or a positive number, got: $IDM_LR" >&2
  exit 2
fi
if [[ -n "$STAGE" && ! "$STAGE" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "STAGE must contain only letters, digits, '.', '_' or '-', got: $STAGE" >&2
  exit 2
fi
if [[ ! "$TIMESTAMP" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "TIMESTAMP must contain only letters, digits, '.', '_' or '-', got: $TIMESTAMP" >&2
  exit 2
fi

EFFECTIVE_IDM_LR="${IDM_LR:-$AGENT_LR}"
IDM_SUFFIX=""
# Named tuning stages make every setting, including coef=0, explicit. Outside
# a named stage, preserve the historical baseline identity when IDM is off.
if [[ -n "$STAGE" ]] || awk -v value="$IDM_COEF" 'BEGIN { exit !(value + 0 > 0) }'; then
  idm_coef_label="${IDM_COEF//./p}"
  IDM_SUFFIX="_idm${idm_coef_label}"
  if [[ -n "$IDM_LR" ]]; then
    idm_lr_label="${IDM_LR//./p}"
    IDM_SUFFIX+="_idmlr${idm_lr_label}"
  fi
fi

STAGE_SUFFIX=""
if [[ -n "$STAGE" ]]; then
  STAGE_SUFFIX="_${STAGE}"
fi
BUDGET_SUFFIX=""
if [[ "$NUM_TRAIN_FRAMES" != "2000010" ]]; then
  BUDGET_SUFFIX="_f${NUM_TRAIN_FRAMES}"
fi
SETTING_SUFFIX="${STAGE_SUFFIX}${IDM_SUFFIX}${BUDGET_SUFFIX}"
RUN_ID="${TIMESTAMP}${SETTING_SUFFIX}"

read -r -a GPUS_ARRAY <<< "${GPUS:-6 7}"
if [[ "${#GPUS_ARRAY[@]}" -eq 0 ]]; then
  echo "At least one GPU is required." >&2
  exit 2
fi

declare -A SEEN_GPUS=()
for gpu in "${GPUS_ARRAY[@]}"; do
  if [[ ! "$gpu" =~ ^[0-9]+$ ]]; then
    echo "GPU ordinals must be non-negative integers, got: $gpu" >&2
    exit 2
  fi
  if [[ -n "${SEEN_GPUS[$gpu]+present}" ]]; then
    echo "Duplicate GPU ordinal is not allowed: $gpu" >&2
    exit 2
  fi
  SEEN_GPUS["$gpu"]=1
done

TASK_SPECS=(
  "walker_stand simplified_walker"
  "walker_walk simplified_walker"
  "walker_run simplified_walker"
  "walker_flip simplified_walker"
  "cheetah_walk null"
  "cheetah_run null"
  "cheetah_walk_backward null"
  "cheetah_run_backward null"
  "quadruped_stand simplified_quadruped"
  "quadruped_walk simplified_quadruped"
  "quadruped_run simplified_quadruped"
  "quadruped_jump simplified_quadruped"
)

task_selected() {
  local candidate="$1"
  local selected
  if [[ -z "$TASKS" ]]; then
    return 0
  fi
  for selected in $TASKS; do
    if [[ "$candidate" == "$selected" ]]; then
      return 0
    fi
  done
  return 1
}

if [[ -n "$TASKS" ]]; then
  for requested_task in $TASKS; do
    found=0
    for spec in "${TASK_SPECS[@]}"; do
      read -r known_task _ <<< "$spec"
      if [[ "$requested_task" == "$known_task" ]]; then
        found=1
        break
      fi
    done
    if [[ "$found" -eq 0 ]]; then
      echo "Unknown task in TASKS: $requested_task" >&2
      exit 2
    fi
  done
fi

launch_dir="$LAUNCH_ROOT/$RUN_ID"
manifest="$launch_dir/manifest.tsv"

declare -a SELECTED_TASKS=()
declare -a SELECTED_DOMAINS=()
declare -a SELECTED_GOAL_SPACES=()
declare -a SELECTED_GPUS=()
declare -a SELECTED_RUN_DIRS=()
declare -a SELECTED_JOB_FILES=()
declare -a SELECTED_WANDB_IDS=()
declare -a USED_GPU_ORDER=()
declare -A USED_GPUS=()

selected_ordinal=0
for spec in "${TASK_SPECS[@]}"; do
  read -r task goal_space <<< "$spec"
  if ! task_selected "$task"; then
    continue
  fi

  gpu="${GPUS_ARRAY[$((selected_ordinal % ${#GPUS_ARRAY[@]}))]}"
  domain="${task%%_*}"
  # Scheduling resources are intentionally absent from this identity. Moving a
  # resumable run to another GPU must resolve to the same checkpoint directory.
  run_name="${TIMESTAMP}_seed${SEED}_${task}_dino_cls_stack3${SETTING_SUFFIX}"
  run_dir="$RUNS_DIR/$run_name"
  job_file="$launch_dir/job_${task}.sh"
  # This contains only the stable experiment identity; GPU assignment is a
  # schedulable resource and must not fork checkpoint or W&B identity.
  wandb_identity_digest="$(printf '%s' "${RUN_ID}|${SEED}|${task}" | sha256sum | cut -c1-16)"
  wandb_run_id="d3s-s${SEED}-${task//_/-}-${wandb_identity_digest}"

  SELECTED_TASKS+=("$task")
  SELECTED_DOMAINS+=("$domain")
  SELECTED_GOAL_SPACES+=("$goal_space")
  SELECTED_GPUS+=("$gpu")
  SELECTED_RUN_DIRS+=("$run_dir")
  SELECTED_JOB_FILES+=("$job_file")
  SELECTED_WANDB_IDS+=("$wandb_run_id")
  if [[ -z "${USED_GPUS[$gpu]+present}" ]]; then
    USED_GPUS["$gpu"]=1
    USED_GPU_ORDER+=("$gpu")
  fi
  selected_ordinal=$((selected_ordinal + 1))
done

if [[ "$selected_ordinal" -eq 0 ]]; then
  echo "No tasks were selected." >&2
  exit 2
fi

# Fail before writing a partial launch plan. The user must explicitly opt into
# checkpoint discovery by passing --resume.
if [[ "$RESUME" -eq 0 ]]; then
  for ordinal in "${!SELECTED_TASKS[@]}"; do
    run_dir="${SELECTED_RUN_DIRS[$ordinal]}"
    checkpoint_dir="$CKPT_ROOT/${run_dir##*/}"
    if [[ -e "$run_dir" || -L "$run_dir" ]]; then
      echo "Fresh start refused: run directory already exists: $run_dir" >&2
      echo "Pass --resume only to continue the exact same setting." >&2
      exit 1
    fi
    if [[ -e "$checkpoint_dir" || -L "$checkpoint_dir" ]]; then
      echo "Fresh start refused: checkpoint directory already exists: $checkpoint_dir" >&2
      echo "Pass --resume only to continue the exact same setting." >&2
      exit 1
    fi
  done
else
  for ordinal in "${!SELECTED_TASKS[@]}"; do
    run_dir="${SELECTED_RUN_DIRS[$ordinal]}"
    checkpoint_file="$CKPT_ROOT/${run_dir##*/}/latest.pt"
    if [[ ! -d "$run_dir" ]]; then
      echo "Resume refused: run directory does not exist: $run_dir" >&2
      exit 1
    fi
    if [[ ! -s "$checkpoint_file" ]]; then
      echo "Resume refused: non-empty checkpoint does not exist: $checkpoint_file" >&2
      exit 1
    fi
  done
fi

write_job_script() {
  local job_file="$1"
  local gpu="$2"
  local task="$3"
  local goal_space="$4"
  local run_dir="$5"
  local wandb_run_id="$6"
  local wandb_resume="never"
  if [[ "$RESUME" -eq 1 ]]; then
    wandb_resume="must"
  fi

  {
    printf '#!/usr/bin/env bash\n'
    printf 'set -euo pipefail\n'
    printf 'cd %q\n' "$REPO_DIR"
    printf 'run_dir=%q\n' "$run_dir"
    printf 'mkdir -p "$run_dir"\n'
    printf 'echo "[start] $(date -u +%%FT%%TZ) task=%s seed=%s gpu=%s stage=%s dino_frame_stack=3 idm_coef=%s idm_lr=%s idm_diagnostics_interval=%s num_train_frames=%s resume=%s" | tee -a "$run_dir/launcher.log"\n' \
      "$task" "$SEED" "$gpu" "${STAGE:-unspecified}" "$IDM_COEF" "$EFFECTIVE_IDM_LR" "$IDM_DIAGNOSTICS_INTERVAL" "$NUM_TRAIN_FRAMES" "$RESUME"
    printf 'if env CUDA_VISIBLE_DEVICES=%q PYTHONUNBUFFERED=1 WANDB_PROJECT=%q WANDB_RUN_ID=%q WANDB_RESUME=%q python %q ' \
      "$gpu" "$WANDB_PROJECT" "$wandb_run_id" "$wandb_resume" "$TRAIN_SCRIPT"
    printf '%q ' \
      "use_wandb=True" \
      "use_tb=False" \
      "use_hiplog=False" \
      "save_video=False" \
      "save_train_video=False" \
      "save_replay_buffer_in_checkpoint=False" \
      "checkpoint_root=$CKPT_ROOT" \
      "obs_type=dino" \
      "dino_model_name=facebook/dinov2-base" \
      "use_cls=True" \
      "frame_stack=3" \
      "dino_frame_stack=3" \
      "render_shape=[224,224]" \
      "action_repeat=2" \
      "agent.dino_use_adapter=True" \
      "agent.dino_adapter_type=linear" \
      "agent.dino_adapter_output_dim=512" \
      "agent.lr=$AGENT_LR" \
      "agent.idm_coef=$IDM_COEF" \
      "agent.idm_lr=${IDM_LR:-null}" \
      "agent.idm_diagnostics_interval=$IDM_DIAGNOSTICS_INTERVAL" \
      "agent.batch_size=1024" \
      "agent.update_every_steps=2" \
      "update_encoder=True" \
      "num_train_frames=$NUM_TRAIN_FRAMES" \
      "eval_every_frames=$EVAL_EVERY_FRAMES" \
      "num_eval_episodes=10" \
      "experiment=dino_cls_stack3${SETTING_SUFFIX}_seed${SEED}" \
      "task=$task" \
      "goal_space=$goal_space" \
      "seed=$SEED" \
      "hydra.run.dir=$run_dir"
    printf '>> "$run_dir/stdout.log" 2>&1; then\n'
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

write_queue_header() {
  local queue_file="$1"
  {
    printf '#!/usr/bin/env bash\n'
    printf 'set -uo pipefail\n'
    printf 'queue_rc=0\n'
  } > "$queue_file"
}

append_queue_job() {
  local queue_file="$1"
  local job_file="$2"
  {
    printf 'if bash %q; then\n' "$job_file"
    printf '  :\n'
    printf 'else\n'
    printf '  rc=$?\n'
    printf '  queue_rc=$rc\n'
    printf 'fi\n'
  } >> "$queue_file"
}

mkdir -p "$launch_dir"
printf 'session\tgpu\tseed\ttask\tgoal_space\trun_dir\tstdout_log\tdomain\tidm_coef\tidm_lr\tidm_diagnostics_interval\tnum_train_frames\tstage\tresume\twandb_run_id\n' > "$manifest"

for gpu in "${USED_GPU_ORDER[@]}"; do
  write_queue_header "$launch_dir/queue_gpu${gpu}.sh"
done

for ordinal in "${!SELECTED_TASKS[@]}"; do
  task="${SELECTED_TASKS[$ordinal]}"
  domain="${SELECTED_DOMAINS[$ordinal]}"
  goal_space="${SELECTED_GOAL_SPACES[$ordinal]}"
  gpu="${SELECTED_GPUS[$ordinal]}"
  run_dir="${SELECTED_RUN_DIRS[$ordinal]}"
  job_file="${SELECTED_JOB_FILES[$ordinal]}"
  wandb_run_id="${SELECTED_WANDB_IDS[$ordinal]}"
  queue_file="$launch_dir/queue_gpu${gpu}.sh"
  queue_session="dino_s3_${RUN_ID//./_}_g${gpu}"

  write_job_script "$job_file" "$gpu" "$task" "$goal_space" "$run_dir" "$wandb_run_id"
  append_queue_job "$queue_file" "$job_file"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$queue_session" "$gpu" "$SEED" "$task" "$goal_space" "$run_dir" "$run_dir/stdout.log" \
    "$domain" "$IDM_COEF" "$EFFECTIVE_IDM_LR" "$IDM_DIAGNOSTICS_INTERVAL" "$NUM_TRAIN_FRAMES" "${STAGE:-unspecified}" "$RESUME" "$wandb_run_id" \
    >> "$manifest"
  printf '%s\t%s\t%s\t%s\n' "$queue_session" "$gpu" "$task" "$run_dir"
done

for gpu in "${USED_GPU_ORDER[@]}"; do
  queue_file="$launch_dir/queue_gpu${gpu}.sh"
  printf 'exit "$queue_rc"\n' >> "$queue_file"
  chmod +x "$queue_file"
done

if [[ "$DRY_RUN" -eq 0 ]]; then
  # Check every queue first so a collision cannot leave a partially launched
  # sweep with only some GPUs active.
  for gpu in "${USED_GPU_ORDER[@]}"; do
    queue_session="dino_s3_${RUN_ID//./_}_g${gpu}"
    if tmux has-session -t "$queue_session" >/dev/null 2>&1; then
      echo "tmux session already exists: $queue_session" >&2
      exit 1
    fi
  done
  for gpu in "${USED_GPU_ORDER[@]}"; do
    queue_session="dino_s3_${RUN_ID//./_}_g${gpu}"
    queue_file="$launch_dir/queue_gpu${gpu}.sh"
    queue_log="$launch_dir/queue_gpu${gpu}.log"
    tmux new-session -d -s "$queue_session" \
      "bash $(printf '%q' "$queue_file") > $(printf '%q' "$queue_log") 2>&1"
  done
fi

echo "Manifest: $manifest"
if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "Dry run only; no tmux sessions were launched."
else
  echo "Launched ${#USED_GPU_ORDER[@]} serial GPU queue(s)."
fi
