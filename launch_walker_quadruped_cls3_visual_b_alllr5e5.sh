#!/usr/bin/env bash
set -euo pipefail

# Eight-task standard shared-adapter DINO CLS3 Walker/Quadruped runs with
# pure-visual B inputs and every active LR at 5e-5.
# This is launch orchestration only; it does not modify the training implementation.

REPO_DIR="${REPO_DIR:-/data/fanfeng/controllable_agent}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-$REPO_DIR/url_benchmark/pretrain.py}"
PYTHON_BIN="${PYTHON_BIN:-/data/fan2/env/miniconda3/envs/occ_rlu/bin/python}"
RUNS_ROOT="${RUNS_ROOT:-/mnt/data_7tb/fanfeng/controllable_agent_runs}"
CAMPAIGN="${CAMPAIGN:-20260904_walker_quadruped_cls3_visualB_alllr5e5_s1}"
if [[ -n "${CKPT_ROOT+x}" ]]; then
  CKPT_ROOT_WAS_SET=1
else
  CKPT_ROOT_WAS_SET=0
fi
CKPT_ROOT="${CKPT_ROOT:-/mnt/data_nvme1/fanfeng/controllable_agent_ckpt/$CAMPAIGN}"
LAUNCH_ROOT="${LAUNCH_ROOT:-$RUNS_ROOT/launch_queues}"
WANDB_ENTITY="${WANDB_ENTITY:-lmu_rl}"
WANDB_PROJECT="${WANDB_PROJECT:-controllable_agent_baseline}"
GPUS_STRING="${GPUS:-4 4 4 4 4 4 4 4}"
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
AGENT_LR=0.00005
FB_LR=0.00005
LR_F=0.00005
LR_B=0.00005
LR_ACTOR=0.00005
ORTHO_COEF=1.0
FB_TARGET_TAU=0.01
Z_DIM=50
MIX_RATIO=0.5
BATCH_SIZE=1024
DINO_FRAME_STACK=3

DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: ./launch_walker_quadruped_cls3_visual_b_alllr5e5.sh [--dry-run] [--campaign NAME]

Launches the eight requested standard shared-adapter, pure-visual
Walker/Quadruped DINO CLS3 runs with all learning-rate controls at 5e-5.
Useful overrides: GPUS, RUNS_ROOT, CKPT_ROOT, LAUNCH_ROOT, CAMPAIGN.
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
      if [[ "$CKPT_ROOT_WAS_SET" -eq 0 ]]; then
        CKPT_ROOT="/mnt/data_nvme1/fanfeng/controllable_agent_ckpt/$CAMPAIGN"
      fi
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

if [[ ! "$CAMPAIGN" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "Invalid campaign: $CAMPAIGN" >&2
  exit 2
fi
for value_name in MIN_GPU_FREE_MB MIN_RUNS_FREE_GIB MIN_CKPT_FREE_GIB; do
  value="${!value_name}"
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || {
    echo "$value_name must be a positive integer: $value" >&2
    exit 2
  }
done
if [[ ! -d "$REPO_DIR" || ! -f "$TRAIN_SCRIPT" || ! -x "$PYTHON_BIN" ]]; then
  echo "Missing repository, training script, or Python executable." >&2
  exit 1
fi
for command_name in rg sha256sum awk sort bash; do
  command -v "$command_name" >/dev/null 2>&1 || {
    echo "$command_name is required" >&2
    exit 1
  }
done
if [[ "$DRY_RUN" -eq 0 ]]; then
  for command_name in df nvidia-smi pgrep tmux; do
    command -v "$command_name" >/dev/null 2>&1 || {
      echo "$command_name is required for a real launch" >&2
      exit 1
    }
  done
fi

read -r -a GPUS_ARRAY <<< "$GPUS_STRING"
if [[ "${#GPUS_ARRAY[@]}" -ne 8 ]]; then
  echo "GPUS must contain exactly eight ordinals." >&2
  exit 2
fi
for gpu in "${GPUS_ARRAY[@]}"; do
  [[ "$gpu" =~ ^[0-9]+$ ]] || { echo "Invalid GPU ordinal: $gpu" >&2; exit 2; }
done

TASKS=(
  walker_stand
  walker_walk
  walker_run
  walker_flip
  quadruped_stand
  quadruped_walk
  quadruped_run
  quadruped_jump
)
RUN_NAMES=(
  dino_cls3_walker_stand_seed1_visualB_alllr5e5_ortho1
  dino_cls3_walker_walk_seed1_visualB_alllr5e5_ortho1
  dino_cls3_walker_run_seed1_visualB_alllr5e5_ortho1
  dino_cls3_walker_flip_seed1_visualB_alllr5e5_ortho1
  dino_cls3_quadruped_stand_seed1_visualB_alllr5e5_ortho1
  dino_cls3_quadruped_walk_seed1_visualB_alllr5e5_ortho1
  dino_cls3_quadruped_run_seed1_visualB_alllr5e5_ortho1
  dino_cls3_quadruped_jump_seed1_visualB_alllr5e5_ortho1
)
SESSIONS=(
  wqvb5e5_walker_stand_s1
  wqvb5e5_walker_walk_s1
  wqvb5e5_walker_run_s1
  wqvb5e5_walker_flip_s1
  wqvb5e5_quadruped_stand_s1
  wqvb5e5_quadruped_walk_s1
  wqvb5e5_quadruped_run_s1
  wqvb5e5_quadruped_jump_s1
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
for index in "${!TASKS[@]}"; do
  run_name="${RUN_NAMES[$index]}"
  run_dir="$RUNS_ROOT/$CAMPAIGN/$run_name"
  ckpt_dir="$CKPT_ROOT/$run_name"
  RUN_DIRS+=("$run_dir")
  CKPT_DIRS+=("$ckpt_dir")
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
    if tmux has-session -t "${SESSIONS[$index]}" >/dev/null 2>&1; then
      echo "Fresh launch refused; session exists: ${SESSIONS[$index]}" >&2
      exit 1
    fi
  fi
done

if [[ "$DRY_RUN" -eq 0 ]]; then
  runs_free_gib="$(df -Pk "$RUNS_ROOT" | awk 'NR == 2 {print int($4 / 1024 / 1024)}')"
  ckpt_parent="${CKPT_ROOT%/*}"
  ckpt_free_gib="$(df -Pk "$ckpt_parent" | awk 'NR == 2 {print int($4 / 1024 / 1024)}')"
  if (( runs_free_gib < MIN_RUNS_FREE_GIB || ckpt_free_gib < MIN_CKPT_FREE_GIB )); then
    echo "Insufficient disk: runs=${runs_free_gib}GiB checkpoints=${ckpt_free_gib}GiB" >&2
    exit 1
  fi
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
  echo "[refused] source changed after validation" >&2
  exit 1
fi
if [[ -e "$run_dir" || -L "$run_dir" || -e "$ckpt_dir" || -L "$ckpt_dir" ]]; then
  echo "[refused] fresh-start target collision" >&2
  exit 1
fi
mkdir -p "$run_dir/matplotlib"
EOF
    printf 'echo "[start] $(date -u +%%FT%%TZ) run=%s gpu=%s shared_adapter=true pure_visual_b=true dino_flare_b=false cls_frames=3 agent_lr=%s fb_lr=%s lr_f=%s lr_b=%s lr_actor=%s ortho=%s" | tee "$run_dir/launcher.log"\n' \
      "$run_name" "$gpu" "$AGENT_LR" "$FB_LR" "$LR_F" "$LR_B" "$LR_ACTOR" "$ORTHO_COEF"
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
      "agent.dino_adapter_hidden_dim=1024" \
      "agent.dino_adapter_output_dim=512" \
      "agent.dino_separate_fb_adapters=False" \
      "agent.dino_separate_backward_adapter=False" \
      "agent.pixel_separate_fb_encoders=False" \
      "agent.dino_flare_b=False" \
      "agent.backward_encoder_grad_scale=1.0" \
      "agent.lr=$AGENT_LR" \
      "agent.fb_lr=$FB_LR" \
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
      "agent.idm_route=none" \
      "agent.idm_encoder_mode=legacy" \
      "update_encoder=True" \
      "num_seed_frames=4000" \
      "replay_buffer_episodes=5000" \
      "num_train_frames=$NUM_TRAIN_FRAMES" \
      "eval_every_frames=$EVAL_EVERY_FRAMES" \
      "num_eval_episodes=$NUM_EVAL_EPISODES" \
      "final_tests=$FINAL_TESTS" \
      "experiment=walker_quadruped_cls3_visual_b_alllr5e5" \
      "task=$task" \
      "seed=$SEED" \
      "hydra.run.dir=$run_dir"
    printf '> "$run_dir/stdout.log" 2>&1; then\n'
    printf '  rc=0\n'
    printf 'else\n'
    printf '  rc=$?\n'
    printf 'fi\n'
    printf 'echo "[done] $(date -u +%%FT%%TZ) run=%s exit=$rc" | tee -a "$run_dir/launcher.log"\n' "$run_name"
    printf 'exit "$rc"\n'
  } > "$job_file"
  chmod +x "$job_file"
}

printf 'task\tgpu\tsession\tadapter_mode\trun_name\trun_dir\tcheckpoint_dir\tstdout_log\twandb_entity\twandb_project\twandb_run_id\twandb_url\tnum_train_frames\teval_every_frames\tsnapshot_at\tdino_frame_stack\tagent_lr\tfb_lr\tlr_f\tlr_b\tlr_actor\tortho_coef\tfb_target_tau\tz_dim\tmix_ratio\tbatch_size\tdino_separate_fb_adapters\tdino_separate_backward_adapter\tpixel_separate_fb_encoders\tdino_flare_b\tsource_fingerprint\tjob_file\n' > "$manifest"
for index in "${!TASKS[@]}"; do
  job_file="$jobs_dir/job_${RUN_NAMES[$index]}.sh"
  write_job_script "$index" "$job_file"
  bash -n "$job_file"
  wandb_url="https://wandb.ai/$WANDB_ENTITY/$WANDB_PROJECT/runs/${RUN_NAMES[$index]}"
  manifest_row=(
    "${TASKS[$index]}" "${GPUS_ARRAY[$index]}" "${SESSIONS[$index]}" "shared_linear"
    "${RUN_NAMES[$index]}" "${RUN_DIRS[$index]}" "${CKPT_DIRS[$index]}"
    "${RUN_DIRS[$index]}/stdout.log" "$WANDB_ENTITY" "$WANDB_PROJECT"
    "${RUN_NAMES[$index]}" "$wandb_url" "$NUM_TRAIN_FRAMES" "$EVAL_EVERY_FRAMES"
    "$SNAPSHOT_AT" "$DINO_FRAME_STACK" "$AGENT_LR" "$FB_LR" "$LR_F" "$LR_B"
    "$LR_ACTOR" "$ORTHO_COEF" "$FB_TARGET_TAU" "$Z_DIM" "$MIX_RATIO" "$BATCH_SIZE"
    "false" "false" "false" "false" "$SOURCE_FINGERPRINT" "$job_file"
  )
  (
    IFS=$'\t'
    printf '%s\n' "${manifest_row[*]}"
  ) >> "$manifest"
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
  printf 'adapter_mode=shared_linear\n'
  printf 'pure_visual_b=true\n'
  printf 'goal_space=null\n'
  printf 'dino_separate_fb_adapters=false\n'
  printf 'dino_separate_backward_adapter=false\n'
  printf 'pixel_separate_fb_encoders=false\n'
  printf 'dino_flare_b=false\n'
  printf 'idm_coef=0.0\n'
  printf 'idm_route=none\n'
  printf 'num_train_frames=%s\n' "$NUM_TRAIN_FRAMES"
  printf 'eval_every_frames=%s\n' "$EVAL_EVERY_FRAMES"
  printf 'snapshot_at=%s\n' "$SNAPSHOT_AT"
  printf 'agent_lr=%s\n' "$AGENT_LR"
  printf 'fb_lr=%s\n' "$FB_LR"
  printf 'lr_f=%s\n' "$LR_F"
  printf 'lr_b=%s\n' "$LR_B"
  printf 'lr_actor=%s\n' "$LR_ACTOR"
  printf 'ortho_coef=%s\n' "$ORTHO_COEF"
  printf 'fb_target_tau=%s\n' "$FB_TARGET_TAU"
  printf 'z_dim=%s\n' "$Z_DIM"
  printf 'mix_ratio=%s\n' "$MIX_RATIO"
  printf 'batch_size=%s\n' "$BATCH_SIZE"
} > "$launch_dir/launch_config.txt"

echo "Manifest: $manifest"
if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "Dry run only; no tmux sessions launched."
  exit 0
fi

launch_failed=0
for index in "${!TASKS[@]}"; do
  job_file="$jobs_dir/job_${RUN_NAMES[$index]}.sh"
  if tmux new-session -d -s "${SESSIONS[$index]}" "bash $(printf '%q' "$job_file")"; then
    printf '[launched] session=%s gpu=%s run=%s\n' \
      "${SESSIONS[$index]}" "${GPUS_ARRAY[$index]}" "${RUN_NAMES[$index]}"
  else
    echo "Failed to launch ${SESSIONS[$index]}" >&2
    launch_failed=1
  fi
done
sleep 3
for session in "${SESSIONS[@]}"; do
  if ! tmux has-session -t "$session" >/dev/null 2>&1; then
    echo "Session missing after launch: $session" >&2
    launch_failed=1
  fi
done
if [[ "$launch_failed" -ne 0 ]]; then
  echo "One or more launches failed; inspect $launch_dir" >&2
  exit 1
fi
echo "All eight shared-adapter pure-visual B all-LR=5e-5 runs launched."
