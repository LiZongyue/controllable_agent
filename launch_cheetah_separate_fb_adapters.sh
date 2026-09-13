#!/usr/bin/env bash
set -euo pipefail

# First clean mechanism comparison against the shared-adapter Cheetah CLS3
# stability control: online, identically initialized F/B linear adapters with
# branch-local optimizers and no target/EMA DINO adapter.

REPO_DIR="${REPO_DIR:-/data/fanfeng/controllable_agent}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-$REPO_DIR/url_benchmark/pretrain.py}"
SANITY_SCRIPT="${SANITY_SCRIPT:-$REPO_DIR/scripts/check_separate_fb_adapters.py}"
PYTHON_BIN="${PYTHON_BIN:-/data/fan2/env/miniconda3/envs/occ_rlu/bin/python}"
RUNS_ROOT="${RUNS_ROOT:-/mnt/data_7tb/fanfeng/controllable_agent_runs}"
CAMPAIGN="${CAMPAIGN:-20260830_cheetah_cls3_separate_fb_s1}"
CKPT_ROOT="${CKPT_ROOT:-}"
LAUNCH_ROOT="${LAUNCH_ROOT:-$RUNS_ROOT/launch_queues}"
WANDB_ENTITY="${WANDB_ENTITY:-lmu_rl}"
WANDB_PROJECT="${WANDB_PROJECT:-controllable_agent_cheetah_fb_stability}"
GPUS_STRING="${GPUS:-1 4 6 7}"
MIN_GPU_FREE_MB="${MIN_GPU_FREE_MB:-12000}"
MIN_RUNS_FREE_GIB="${MIN_RUNS_FREE_GIB:-20}"
MIN_CKPT_FREE_GIB="${MIN_CKPT_FREE_GIB:-30}"

SEED=1
NUM_TRAIN_FRAMES=2000010
EVAL_EVERY_FRAMES=10000
NUM_EVAL_EPISODES=10
FINAL_TESTS=10
CHECKPOINT_EVERY=100000
SNAPSHOT_AT="[100000,200000,500000,800000,1000000,1500000,2000000]"
LR_F=0.0001
LR_B=0.0001
LR_ACTOR=0.0001
ORTHO_COEF=1.0
FB_TARGET_TAU=0.01
Z_DIM=50
MIX_RATIO=0.5
BATCH_SIZE=1024
DINO_FRAME_STACK=3

DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: ./launch_cheetah_separate_fb_adapters.sh [--dry-run] [--campaign NAME]

Creates four fresh seed-1 DINO CLS3 Cheetah jobs in the requested new
separate-F/B adapter mode. A real launch starts one tmux session per job;
--dry-run writes and validates the complete plan without starting training.

Useful overrides:
  REPO_DIR, TRAIN_SCRIPT, SANITY_SCRIPT, PYTHON_BIN
  RUNS_ROOT, CKPT_ROOT, LAUNCH_ROOT, CAMPAIGN
  WANDB_ENTITY, WANDB_PROJECT, GPUS="1 4 6 7"
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

CKPT_ROOT="${CKPT_ROOT:-/mnt/data_nvme1/fanfeng/controllable_agent_ckpt/$CAMPAIGN}"

if [[ ! "$CAMPAIGN" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "Invalid CAMPAIGN: $CAMPAIGN" >&2
  exit 2
fi
for value_name in MIN_GPU_FREE_MB MIN_RUNS_FREE_GIB MIN_CKPT_FREE_GIB; do
  value="${!value_name}"
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "$value_name must be a positive integer: $value" >&2
    exit 2
  fi
done
if [[ ! -d "$REPO_DIR" || ! -f "$TRAIN_SCRIPT" || ! -f "$SANITY_SCRIPT" ]]; then
  echo "Missing repository, training script, or sanity checker." >&2
  exit 1
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python is not executable: $PYTHON_BIN" >&2
  exit 1
fi
for required_command in rg sha256sum awk sort; do
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

read -r -a GPUS_ARRAY <<< "$GPUS_STRING"
if [[ "${#GPUS_ARRAY[@]}" -ne 4 ]]; then
  echo "GPUS must contain exactly four GPU ordinals." >&2
  exit 2
fi
for gpu in "${GPUS_ARRAY[@]}"; do
  [[ "$gpu" =~ ^[0-9]+$ ]] || { echo "Invalid GPU ordinal: $gpu" >&2; exit 2; }
done

TASKS=(
  cheetah_walk
  cheetah_run
  cheetah_run_backward
  cheetah_walk_backward
)
RUN_NAMES=(
  dino_cls3_cheetah_walk_seed1_sepFB_lrF1e4_lrB1e4_ortho1
  dino_cls3_cheetah_run_seed1_sepFB_lrF1e4_lrB1e4_ortho1
  dino_cls3_cheetah_run_backward_seed1_sepFB_lrF1e4_lrB1e4_ortho1
  dino_cls3_cheetah_walk_backward_seed1_sepFB_lrF1e4_lrB1e4_ortho1
)

mapfile -t SOURCE_FILES < <(
  cd "$REPO_DIR"
  rg --files url_benchmark -g '*.py' -g '*.yaml' | LC_ALL=C sort
)
if [[ "${#SOURCE_FILES[@]}" -eq 0 ]]; then
  echo "No training source files found under $REPO_DIR/url_benchmark" >&2
  exit 1
fi

source_fingerprint() {
  local root="$1" source_file digest
  for source_file in "${SOURCE_FILES[@]}"; do
    digest="$(sha256sum "$root/$source_file")"
    printf '%s\n' "${digest%% *}"
  done | sha256sum | awk '{print $1}'
}
SOURCE_FINGERPRINT="$(source_fingerprint "$REPO_DIR")"

launch_dir="$LAUNCH_ROOT/$CAMPAIGN"
jobs_dir="$launch_dir/jobs"
manifest="$launch_dir/manifest.tsv"
if [[ -e "$launch_dir" || -L "$launch_dir" ]]; then
  echo "Fresh launch refused; launch directory exists: $launch_dir" >&2
  exit 1
fi

declare -a RUN_DIRS=()
declare -a CKPT_DIRS=()
declare -a SESSIONS=()
for index in "${!TASKS[@]}"; do
  task="${TASKS[$index]}"
  run_name="${RUN_NAMES[$index]}"
  run_dir="$RUNS_ROOT/$CAMPAIGN/$run_name"
  ckpt_dir="$CKPT_ROOT/$run_name"
  session="csep_${task}_s${SEED}"
  RUN_DIRS+=("$run_dir")
  CKPT_DIRS+=("$ckpt_dir")
  SESSIONS+=("$session")
  for target in "$run_dir" "$ckpt_dir"; do
    if [[ -e "$target" || -L "$target" ]]; then
      echo "Fresh launch refused; target exists: $target" >&2
      exit 1
    fi
  done
  if [[ "$DRY_RUN" -eq 0 ]]; then
    if pgrep -f -- "$run_name" >/dev/null 2>&1; then
      echo "Fresh launch refused; matching process exists: $run_name" >&2
      exit 1
    fi
    if tmux has-session -t "$session" >/dev/null 2>&1; then
      echo "Fresh launch refused; tmux session exists: $session" >&2
      exit 1
    fi
  fi
done

if [[ "$DRY_RUN" -eq 0 ]]; then
  for gpu in "${GPUS_ARRAY[@]}"; do
    sample="$(nvidia-smi --id="$gpu" --query-gpu=memory.free --format=csv,noheader,nounits)"
    free_mb="${sample//[[:space:]]/}"
    if [[ ! "$free_mb" =~ ^[0-9]+$ ]] || (( free_mb < MIN_GPU_FREE_MB )); then
      echo "GPU $gpu has ${free_mb}MB free; ${MIN_GPU_FREE_MB}MB required." >&2
      exit 1
    fi
  done
fi

mkdir -p "$RUNS_ROOT/$CAMPAIGN" "$CKPT_ROOT" "$jobs_dir"

write_job_script() {
  local index="$1" job_file="$2"
  local gpu="${GPUS_ARRAY[$index]}" task="${TASKS[$index]}"
  local run_name="${RUN_NAMES[$index]}" run_dir="${RUN_DIRS[$index]}"
  local ckpt_dir="${CKPT_DIRS[$index]}"
  {
    printf '#!/usr/bin/env bash\n'
    printf 'set -euo pipefail\n'
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
  echo "[refused] source changed after validation: expected=$expected_source_fingerprint actual=$actual_source_fingerprint" >&2
  exit 1
fi
if [[ -e "$run_dir" || -L "$run_dir" || -e "$ckpt_dir" || -L "$ckpt_dir" ]]; then
  echo "[refused] fresh-start collision: run_dir=$run_dir ckpt_dir=$ckpt_dir" >&2
  exit 1
fi
mkdir -p "$run_dir/matplotlib"
EOF
    printf 'echo "[start] $(date -u +%%FT%%TZ) run=%s gpu=%s sepFB=true cls_frames=3 lr_f=%s lr_b=%s ortho=%s" | tee "$run_dir/launcher.log"\n' \
      "$run_name" "$gpu" "$LR_F" "$LR_B" "$ORTHO_COEF"
    printf 'if env CUDA_VISIBLE_DEVICES=%q MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=%q PYTHONUNBUFFERED=1 WANDB_ENTITY=%q WANDB_PROJECT=%q WANDB_RUN_NAME=%q WANDB_RUN_ID=%q WANDB_RESUME=never MPLCONFIGDIR="$run_dir/matplotlib" %q %q ' \
      "$gpu" "$gpu" "$WANDB_ENTITY" "$WANDB_PROJECT" "$run_name" "$run_name" "$PYTHON_BIN" "$TRAIN_SCRIPT"
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
      "dino_model_name=facebook/dinov2-base" \
      "use_cls=True" \
      "frame_stack=3" \
      "dino_frame_stack=$DINO_FRAME_STACK" \
      "render_shape=[224,224]" \
      "action_repeat=2" \
      "goal_space=null" \
      "custom_reward=null" \
      "append_goal_to_observation=False" \
      "discount=0.99" \
      "future=0.99" \
      "reward_free=True" \
      "agent.dino_use_adapter=True" \
      "agent.dino_adapter_type=linear" \
      "agent.dino_adapter_output_dim=512" \
      "agent.dino_separate_fb_adapters=True" \
      "agent.dino_separate_backward_adapter=False" \
      "agent.backward_encoder_grad_scale=1.0" \
      "agent.lr=0.0001" \
      "agent.fb_lr=0.0001" \
      "agent.lr_f=$LR_F" \
      "agent.lr_b=$LR_B" \
      "agent.lr_actor=$LR_ACTOR" \
      "agent.lr_coef=1.0" \
      "agent.fb_target_tau=$FB_TARGET_TAU" \
      "agent.batch_size=$BATCH_SIZE" \
      "agent.update_every_steps=2" \
      "agent.num_inference_steps=5120" \
      "agent.hidden_dim=1024" \
      "agent.backward_hidden_dim=526" \
      "agent.feature_dim=512" \
      "agent.z_dim=$Z_DIM" \
      "agent.stddev_schedule=0.2" \
      "agent.stddev_clip=0.3" \
      "agent.update_z_every_step=300" \
      "agent.update_z_proba=1.0" \
      "agent.nstep=1" \
      "agent.ortho_coef=$ORTHO_COEF" \
      "agent.future_ratio=0.0" \
      "agent.mix_ratio=$MIX_RATIO" \
      "agent.rand_weight=False" \
      "agent.preprocess=True" \
      "agent.norm_z=True" \
      "agent.q_loss=False" \
      "agent.q_loss_coef=0.01" \
      "agent.boltzmann=False" \
      "agent.add_trunk=False" \
      "agent.idm_coef=0.0" \
      "agent.idm_lr=null" \
      "agent.idm_encoder_mode=legacy" \
      "update_encoder=True" \
      "num_seed_frames=4000" \
      "replay_buffer_episodes=5000" \
      "num_train_frames=$NUM_TRAIN_FRAMES" \
      "eval_every_frames=$EVAL_EVERY_FRAMES" \
      "num_eval_episodes=$NUM_EVAL_EPISODES" \
      "final_tests=$FINAL_TESTS" \
      "experiment=cheetah_fb_stability" \
      "task=$task" \
      "seed=$SEED" \
      "hydra.run.dir=$run_dir"
    printf '> "$run_dir/stdout.log" 2>&1; then\n'
    printf '  rc=0\n'
    printf 'else\n'
    printf '  rc=$?\n'
    printf 'fi\n'
    printf 'echo "[done] $(date -u +%%FT%%TZ) run=%s gpu=%s exit=$rc" | tee -a "$run_dir/launcher.log"\n' "$run_name" "$gpu"
    printf 'exit "$rc"\n'
  } > "$job_file"
  chmod +x "$job_file"
}

printf 'task\tgpu\tsession\trun_name\trun_dir\tcheckpoint_dir\tstdout_log\twandb_entity\twandb_project\twandb_run_id\tnum_train_frames\teval_every_frames\tsnapshot_at\tdino_frame_stack\tlr_f\tlr_b\tlr_actor\tortho_coef\tfb_target_tau\tz_dim\tmix_ratio\tbatch_size\tsource_fingerprint\tjob_file\n' > "$manifest"
for index in "${!TASKS[@]}"; do
  job_file="$jobs_dir/job_${RUN_NAMES[$index]}.sh"
  write_job_script "$index" "$job_file"
  bash -n "$job_file"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${TASKS[$index]}" "${GPUS_ARRAY[$index]}" "${SESSIONS[$index]}" "${RUN_NAMES[$index]}" \
    "${RUN_DIRS[$index]}" "${CKPT_DIRS[$index]}" "${RUN_DIRS[$index]}/stdout.log" \
    "$WANDB_ENTITY" "$WANDB_PROJECT" "${RUN_NAMES[$index]}" "$NUM_TRAIN_FRAMES" \
    "$EVAL_EVERY_FRAMES" "$SNAPSHOT_AT" "$DINO_FRAME_STACK" "$LR_F" "$LR_B" \
    "$LR_ACTOR" "$ORTHO_COEF" "$FB_TARGET_TAU" "$Z_DIM" "$MIX_RATIO" "$BATCH_SIZE" \
    "$SOURCE_FINGERPRINT" "$job_file" >> "$manifest"
done

{
  printf 'campaign=%s\n' "$CAMPAIGN"
  printf 'created_at=%s\n' "$(date -u +%FT%TZ)"
  printf 'dry_run=%s\n' "$DRY_RUN"
  printf 'source_repo=%s\n' "$REPO_DIR"
  printf 'source_fingerprint=%s\n' "$SOURCE_FINGERPRINT"
  printf 'python=%s\n' "$PYTHON_BIN"
  "$PYTHON_BIN" --version 2>&1
  printf 'wandb=%s/%s\n' "$WANDB_ENTITY" "$WANDB_PROJECT"
  printf 'gpus=%s\n' "$GPUS_STRING"
  printf 'num_train_frames=%s\n' "$NUM_TRAIN_FRAMES"
  printf 'eval_every_frames=%s\n' "$EVAL_EVERY_FRAMES"
  printf 'snapshot_at=%s\n' "$SNAPSHOT_AT"
} > "$launch_dir/launch_config.txt"

MUJOCO_GL=egl "$PYTHON_BIN" "$SANITY_SCRIPT" --launch-dir "$launch_dir" \
  > "$launch_dir/sanity_checks.json"

echo "Manifest: $manifest"
echo "Sanity checks: $launch_dir/sanity_checks.json"
if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "Dry run only; no tmux sessions launched."
  exit 0
fi

runs_free_kib="$(df -Pk "$RUNS_ROOT" | awk 'END {print $4}')"
ckpt_free_kib="$(df -Pk "$CKPT_ROOT" | awk 'END {print $4}')"
if (( runs_free_kib < MIN_RUNS_FREE_GIB * 1024 * 1024 )); then
  echo "Insufficient run-volume free space." >&2
  exit 1
fi
if (( ckpt_free_kib < MIN_CKPT_FREE_GIB * 1024 * 1024 )); then
  echo "Insufficient checkpoint-volume free space." >&2
  exit 1
fi

launch_failed=0
for index in "${!TASKS[@]}"; do
  job_file="$jobs_dir/job_${RUN_NAMES[$index]}.sh"
  session="${SESSIONS[$index]}"
  command="bash $(printf '%q' "$job_file")"
  if tmux new-session -d -s "$session" "$command"; then
    printf '[launched] session=%s gpu=%s run=%s\n' "$session" "${GPUS_ARRAY[$index]}" "${RUN_NAMES[$index]}"
  else
    echo "Failed to start tmux session: $session" >&2
    launch_failed=1
  fi
done
if [[ "$launch_failed" -ne 0 ]]; then
  echo "At least one launch failed; inspect $launch_dir" >&2
  exit 1
fi

sleep 3
for session in "${SESSIONS[@]}"; do
  if ! tmux has-session -t "$session" >/dev/null 2>&1; then
    echo "Session disappeared during startup: $session" >&2
    launch_failed=1
  fi
done
if [[ "$launch_failed" -ne 0 ]]; then
  exit 1
fi
echo "Launched all four separate-F/B mechanism experiments."
