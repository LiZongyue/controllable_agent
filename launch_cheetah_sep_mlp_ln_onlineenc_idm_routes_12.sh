#!/usr/bin/env bash
set -euo pipefail

# Corrected 12-run Cheetah DINO CLS3 comparison.  In this campaign the two
# separate adapters are online trainable modules; only ForwardTarget and
# BackwardTarget are EMA targets.  The onlineEnc run identity keeps these fresh
# starts distinct from the older target-adapter/confounded campaign.
#
# A. separate F/B MLP+LN adapters, no IDM
# B. the same adapters with IDM routed to the forward adapter
# C. the same adapters with IDM routed to the backward adapter
#
# A real invocation starts all twelve jobs in independent tmux sessions.  Jobs
# are spread round-robin, three per GPU, only after aggregate free-memory and
# disk-capacity guards pass.  --dry-run writes all generated artifacts under
# /tmp, validates them, and never starts tmux.

REPO_DIR="${REPO_DIR:-/data/fanfeng/controllable_agent}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-$REPO_DIR/url_benchmark/pretrain.py}"
PYTHON_BIN="${PYTHON_BIN:-/data/fan2/env/miniconda3/envs/occ_rlu/bin/python}"
RUNS_ROOT="${RUNS_ROOT:-/mnt/data_7tb/fanfeng/controllable_agent_runs}"
CAMPAIGN="${CAMPAIGN:-20260901_cheetah_cls3_sepFB_mlpLN_onlineEnc_idm_routes_s1}"
if [[ "${CKPT_ROOT+x}" == "x" ]]; then
  CKPT_ROOT_WAS_EXPLICIT=1
else
  CKPT_ROOT_WAS_EXPLICIT=0
fi
CKPT_ROOT="${CKPT_ROOT:-}"
LAUNCH_ROOT="${LAUNCH_ROOT:-$RUNS_ROOT/launch_queues}"
WANDB_ENTITY="${WANDB_ENTITY:-lmu_rl}"
WANDB_PROJECT="${WANDB_PROJECT:-controllable_agent_cheetah_fb_stability}"

# GPU 4 is intentionally excluded from this campaign because it is currently
# heavily loaded.  Twelve jobs are assigned round-robin, three per GPU.
GPUS_STRING="${GPUS:-2 5 6 7}"
GPU_JOB_ESTIMATE_MB="${GPU_JOB_ESTIMATE_MB:-4096}"
GPU_RESERVE_MB="${GPU_RESERVE_MB:-12000}"
RUNS_GIB_PER_JOB="${RUNS_GIB_PER_JOB:-1}"
RUNS_RESERVE_GIB="${RUNS_RESERVE_GIB:-20}"
CKPT_GIB_PER_JOB="${CKPT_GIB_PER_JOB:-2}"
CKPT_RESERVE_GIB="${CKPT_RESERVE_GIB:-30}"
LAUNCH_STAGGER_SECONDS="${LAUNCH_STAGGER_SECONDS:-2}"

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
IDM_LR=0.0001
ORTHO_COEF=1.0
FB_TARGET_TAU=0.01
Z_DIM=50
MIX_RATIO=0.5
BATCH_SIZE=1024
DINO_FRAME_STACK=3
DINO_ADAPTER_TYPE=mlp_ln
DINO_ADAPTER_HIDDEN_DIM=1024
DINO_ADAPTER_OUTPUT_DIM=512
IDM_DIAGNOSTICS_INTERVAL=500

DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: ./launch_cheetah_sep_mlp_ln_onlineenc_idm_routes_12.sh [--dry-run] [--campaign NAME]

Builds the exact corrected online-adapter 3 x 4 Cheetah DINO campaign.

  --dry-run       Generate and validate the plan under /tmp; start nothing.
  --campaign NAME Override the fresh campaign directory identity.

Useful environment overrides:
  REPO_DIR, TRAIN_SCRIPT, PYTHON_BIN, RUNS_ROOT, CKPT_ROOT, LAUNCH_ROOT
  GPUS="2 5 6 7", WANDB_ENTITY, WANDB_PROJECT
  GPU_JOB_ESTIMATE_MB, GPU_RESERVE_MB, RUNS_GIB_PER_JOB, RUNS_RESERVE_GIB
  CKPT_GIB_PER_JOB, CKPT_RESERVE_GIB, LAUNCH_STAGGER_SECONDS
  ROUTING_CHECKER, ADAPTER_CHECKER
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

# Keep an implicit checkpoint root coupled to the final campaign name, while
# preserving an explicitly supplied root verbatim.
if [[ "$CKPT_ROOT_WAS_EXPLICIT" -eq 0 ]]; then
  CKPT_ROOT="/mnt/data_nvme1/fanfeng/controllable_agent_ckpt/$CAMPAIGN"
elif [[ -z "$CKPT_ROOT" ]]; then
  echo "Explicit CKPT_ROOT must not be empty." >&2
  exit 2
fi

if [[ ! "$CAMPAIGN" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "Invalid CAMPAIGN: $CAMPAIGN" >&2
  exit 2
fi
if [[ "$WANDB_ENTITY" != "lmu_rl" ]]; then
  echo "Refusing WANDB_ENTITY=$WANDB_ENTITY; this campaign requires lmu_rl." >&2
  exit 2
fi
if [[ "$WANDB_PROJECT" != "controllable_agent_cheetah_fb_stability" ]]; then
  echo "Refusing WANDB_PROJECT=$WANDB_PROJECT; unexpected project." >&2
  exit 2
fi
for positive_name in \
  GPU_JOB_ESTIMATE_MB GPU_RESERVE_MB \
  RUNS_GIB_PER_JOB RUNS_RESERVE_GIB CKPT_GIB_PER_JOB CKPT_RESERVE_GIB; do
  value="${!positive_name}"
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "$positive_name must be a positive integer: $value" >&2
    exit 2
  fi
done
for nonnegative_name in LAUNCH_STAGGER_SECONDS; do
  value="${!nonnegative_name}"
  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    echo "$nonnegative_name must be a non-negative integer: $value" >&2
    exit 2
  fi
done
if [[ ! -d "$REPO_DIR" || ! -f "$TRAIN_SCRIPT" ]]; then
  echo "Missing repository or training script: $REPO_DIR / $TRAIN_SCRIPT" >&2
  exit 1
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python is not executable: $PYTHON_BIN" >&2
  exit 1
fi
for required_command in rg sha256sum awk sort df; do
  command -v "$required_command" >/dev/null 2>&1 || {
    echo "$required_command is required" >&2
    exit 1
  }
done
if [[ "$DRY_RUN" -eq 0 ]]; then
  for required_command in nvidia-smi tmux; do
    command -v "$required_command" >/dev/null 2>&1 || {
      echo "$required_command is required for a real launch" >&2
      exit 1
    }
  done
fi

read -r -a GPUS_ARRAY <<< "$GPUS_STRING"
if [[ "${#GPUS_ARRAY[@]}" -eq 0 ]]; then
  echo "GPUS must contain at least one GPU ordinal." >&2
  exit 2
fi
declare -A SEEN_GPUS=()
for gpu in "${GPUS_ARRAY[@]}"; do
  [[ "$gpu" =~ ^[0-9]+$ ]] || { echo "Invalid GPU ordinal: $gpu" >&2; exit 2; }
  if [[ -n "${SEEN_GPUS[$gpu]:-}" ]]; then
    echo "Duplicate GPU ordinal is not allowed: $gpu" >&2
    exit 2
  fi
  SEEN_GPUS[$gpu]=1
done

TASKS=(
  cheetah_walk
  cheetah_run
  cheetah_run_backward
  cheetah_walk_backward
)
GROUP_KEYS=(noIDM idmF idmBAdapter)
GROUP_LABELS=(A B C)
GROUP_IDM_COEFS=(0.0 0.1 0.1)
GROUP_IDM_ROUTES=(none forward_adapter backward_adapter)
GROUP_IDM_MODES=(legacy static static)
GROUP_IDM_LRS=(null "$IDM_LR" "$IDM_LR")

RUN_NAMES=(
  dino_cls3_cheetah_walk_seed1_sepFB_mlpLN_onlineEnc_noIDM
  dino_cls3_cheetah_run_seed1_sepFB_mlpLN_onlineEnc_noIDM
  dino_cls3_cheetah_run_backward_seed1_sepFB_mlpLN_onlineEnc_noIDM
  dino_cls3_cheetah_walk_backward_seed1_sepFB_mlpLN_onlineEnc_noIDM
  dino_cls3_cheetah_walk_seed1_sepFB_mlpLN_onlineEnc_idmF_c01
  dino_cls3_cheetah_run_seed1_sepFB_mlpLN_onlineEnc_idmF_c01
  dino_cls3_cheetah_run_backward_seed1_sepFB_mlpLN_onlineEnc_idmF_c01
  dino_cls3_cheetah_walk_backward_seed1_sepFB_mlpLN_onlineEnc_idmF_c01
  dino_cls3_cheetah_walk_seed1_sepFB_mlpLN_onlineEnc_idmBAdapter_c01
  dino_cls3_cheetah_run_seed1_sepFB_mlpLN_onlineEnc_idmBAdapter_c01
  dino_cls3_cheetah_run_backward_seed1_sepFB_mlpLN_onlineEnc_idmBAdapter_c01
  dino_cls3_cheetah_walk_backward_seed1_sepFB_mlpLN_onlineEnc_idmBAdapter_c01
)

NUM_JOBS="${#RUN_NAMES[@]}"
if [[ "$NUM_JOBS" -ne 12 || "${#TASKS[@]}" -ne 4 || "${#GROUP_KEYS[@]}" -ne 3 ]]; then
  echo "Internal experiment matrix must be exactly 3 groups x 4 tasks." >&2
  exit 1
fi
if [[ "$SNAPSHOT_AT" != "[100000,200000,500000,800000,1000000,1500000,2000000]" ]]; then
  echo "Internal snapshot schedule must contain the seven requested checkpoints." >&2
  exit 1
fi

mapfile -t SOURCE_FILES < <(
  cd "$REPO_DIR"
  {
    rg --files url_benchmark -g '*.py' -g '*.yaml'
    [[ -f requirements.txt ]] && printf '%s\n' requirements.txt
  } | LC_ALL=C sort -u
)
if [[ "${#SOURCE_FILES[@]}" -eq 0 ]]; then
  echo "No training sources found under $REPO_DIR/url_benchmark." >&2
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
CAMPAIGN_TAG="$(printf '%s' "$CAMPAIGN" | sha256sum | awk '{print substr($1, 1, 8)}')"

if [[ "$DRY_RUN" -eq 1 ]]; then
  launch_dir="$(mktemp -d "${TMPDIR:-/tmp}/${CAMPAIGN}.dryrun.XXXXXX")"
else
  mkdir -p "$LAUNCH_ROOT"
  launch_dir="$LAUNCH_ROOT/$CAMPAIGN"
  if ! mkdir "$launch_dir"; then
    echo "Fresh launch refused; could not atomically claim $launch_dir" >&2
    exit 1
  fi
fi
jobs_dir="$launch_dir/jobs"
manifest="$launch_dir/manifest.tsv"
mkdir -p "$jobs_dir"

declare -a JOB_TASKS=()
declare -a JOB_GROUPS=()
declare -a JOB_IDM_COEFS=()
declare -a JOB_IDM_ROUTES=()
declare -a JOB_IDM_MODES=()
declare -a JOB_IDM_LRS=()
declare -a JOB_GPUS=()
declare -a RUN_DIRS=()
declare -a CKPT_DIRS=()
declare -a SESSIONS=()
declare -A RUN_NAME_SEEN=()
declare -A SESSION_SEEN=()
declare -A JOBS_PER_GPU=()

job_index=0
for group_index in "${!GROUP_KEYS[@]}"; do
  for task_index in "${!TASKS[@]}"; do
    task="${TASKS[$task_index]}"
    run_name="${RUN_NAMES[$job_index]}"
    gpu="${GPUS_ARRAY[$((job_index % ${#GPUS_ARRAY[@]}))]}"
    JOBS_PER_GPU[$gpu]="$(( ${JOBS_PER_GPU[$gpu]:-0} + 1 ))"
    run_dir="$RUNS_ROOT/$CAMPAIGN/$run_name"
    ckpt_dir="$CKPT_ROOT/$run_name"
    task_slug="${task#cheetah_}"
    session="dinooe_${CAMPAIGN_TAG}_${GROUP_LABELS[$group_index]}_${task_slug}"

    if [[ -n "${RUN_NAME_SEEN[$run_name]:-}" ]]; then
      echo "Duplicate run name: $run_name" >&2
      exit 1
    fi
    RUN_NAME_SEEN[$run_name]=1
    if [[ -n "${SESSION_SEEN[$session]:-}" ]]; then
      echo "Duplicate tmux session: $session" >&2
      exit 1
    fi
    SESSION_SEEN[$session]=1
    for target in "$run_dir" "$ckpt_dir"; do
      if [[ -e "$target" || -L "$target" ]]; then
        echo "Fresh launch refused; target exists: $target" >&2
        exit 1
      fi
    done

    JOB_TASKS+=("$task")
    JOB_GROUPS+=("${GROUP_KEYS[$group_index]}")
    JOB_IDM_COEFS+=("${GROUP_IDM_COEFS[$group_index]}")
    JOB_IDM_ROUTES+=("${GROUP_IDM_ROUTES[$group_index]}")
    JOB_IDM_MODES+=("${GROUP_IDM_MODES[$group_index]}")
    JOB_IDM_LRS+=("${GROUP_IDM_LRS[$group_index]}")
    JOB_GPUS+=("$gpu")
    RUN_DIRS+=("$run_dir")
    CKPT_DIRS+=("$ckpt_dir")
    SESSIONS+=("$session")
    job_index="$((job_index + 1))"
  done
done

existing_path_for_df() {
  local candidate="$1"
  while [[ ! -e "$candidate" && "$candidate" != "/" ]]; do
    candidate="$(dirname "$candidate")"
  done
  printf '%s\n' "$candidate"
}

check_disk_capacity() {
  local label="$1" path="$2" required_gib="$3"
  local probe free_kib required_kib
  probe="$(existing_path_for_df "$path")"
  free_kib="$(df -Pk "$probe" | awk 'END {print $4}')"
  required_kib="$((required_gib * 1024 * 1024))"
  if [[ ! "$free_kib" =~ ^[0-9]+$ ]] || (( free_kib < required_kib )); then
    echo "$label capacity failed: path=$path free_kib=$free_kib required_kib=$required_kib" >&2
    exit 1
  fi
  printf '[capacity] %s path=%s free_gib=%s required_gib=%s\n' \
    "$label" "$path" "$((free_kib / 1024 / 1024))" "$required_gib"
}

check_gpu_capacity() {
  local gpu="$1" jobs="$2" sample free_mb used_mb util required_mb
  if ! sample="$(nvidia-smi --id="$gpu" \
    --query-gpu=memory.free,memory.used,utilization.gpu \
    --format=csv,noheader,nounits)"; then
    echo "Cannot query assigned GPU $gpu." >&2
    exit 1
  fi
  IFS=',' read -r free_mb used_mb util <<< "$sample"
  free_mb="${free_mb//[[:space:]]/}"
  used_mb="${used_mb//[[:space:]]/}"
  util="${util//[[:space:]]/}"
  required_mb="$((GPU_RESERVE_MB + jobs * GPU_JOB_ESTIMATE_MB))"
  if [[ ! "$free_mb" =~ ^[0-9]+$ || ! "$used_mb" =~ ^[0-9]+$ || ! "$util" =~ ^[0-9]+$ ]]; then
    echo "Could not parse GPU $gpu capacity sample: $sample" >&2
    exit 1
  fi
  if (( free_mb < required_mb )); then
    echo "GPU $gpu aggregate capacity failed: jobs=$jobs free=${free_mb}MB required=${required_mb}MB" >&2
    exit 1
  fi
  printf '[capacity] gpu=%s jobs=%s used_mb=%s free_mb=%s util_pct=%s required_free_mb=%s\n' \
    "$gpu" "$jobs" "$used_mb" "$free_mb" "$util" "$required_mb"
}

write_job_script() {
  local index="$1" job_file="$2"
  local gpu="${JOB_GPUS[$index]}" task="${JOB_TASKS[$index]}"
  local group="${JOB_GROUPS[$index]}" run_name="${RUN_NAMES[$index]}"
  local run_dir="${RUN_DIRS[$index]}" ckpt_dir="${CKPT_DIRS[$index]}"
  local idm_coef="${JOB_IDM_COEFS[$index]}" idm_route="${JOB_IDM_ROUTES[$index]}"
  local idm_mode="${JOB_IDM_MODES[$index]}" idm_lr="${JOB_IDM_LRS[$index]}"
  {
    printf '#!/usr/bin/env bash\nset -euo pipefail\n'
    printf 'cd %q\n' "$REPO_DIR"
    printf 'run_name=%q\nrun_dir=%q\nckpt_dir=%q\n' "$run_name" "$run_dir" "$ckpt_dir"
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
if ! mkdir "$run_dir"; then
  echo "[refused] could not atomically claim run directory: $run_dir" >&2
  exit 1
fi
if ! mkdir "$ckpt_dir"; then
  echo "[refused] could not atomically claim checkpoint directory: $ckpt_dir" >&2
  exit 1
fi
mkdir "$run_dir/matplotlib"
EOF
    printf 'echo "[start] $(date -u +%%FT%%TZ) semantics=online_adapters group=%s run=%s gpu=%s idm_route=%s idm_coef=%s" | tee "$run_dir/launcher.log"\n' \
      "$group" "$run_name" "$gpu" "$idm_route" "$idm_coef"
    printf 'if env CUDA_VISIBLE_DEVICES=%q MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=%q PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 WANDB_MODE=online WANDB_ENTITY=%q WANDB_PROJECT=%q WANDB_RUN_NAME=%q WANDB_RUN_ID=%q WANDB_RESUME=never MPLCONFIGDIR="$run_dir/matplotlib" %q %q ' \
      "$gpu" "$gpu" "$WANDB_ENTITY" "$WANDB_PROJECT" "$run_name" "$run_name" \
      "$PYTHON_BIN" "$TRAIN_SCRIPT"
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
      "agent.pixel_separate_fb_encoders=False" \
      "agent.dino_use_adapter=True" \
      "agent.dino_adapter_type=$DINO_ADAPTER_TYPE" \
      "agent.dino_adapter_hidden_dim=$DINO_ADAPTER_HIDDEN_DIM" \
      "agent.dino_adapter_output_dim=$DINO_ADAPTER_OUTPUT_DIM" \
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
      "agent.idm_coef=$idm_coef" \
      "agent.idm_lr=$idm_lr" \
      "agent.idm_route=$idm_route" \
      "agent.idm_encoder_mode=$idm_mode" \
      "agent.idm_diagnostics_interval=$IDM_DIAGNOSTICS_INTERVAL" \
      "agent.idm_encoder_burnin_steps=0" \
      "agent.idm_encoder_ramp_steps=0" \
      "agent.idm_grad_ratio_target=null" \
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
    printf '> "$run_dir/stdout.log" 2>&1; then\n  rc=0\nelse\n  rc=$?\nfi\n'
    printf 'echo "[done] $(date -u +%%FT%%TZ) semantics=online_adapters group=%s run=%s gpu=%s exit=$rc" | tee -a "$run_dir/launcher.log"\n' \
      "$group" "$run_name" "$gpu"
    printf 'exit "$rc"\n'
  } > "$job_file"
  chmod +x "$job_file"
}

assert_job_has() {
  local job_file="$1" fragment="$2"
  if ! rg -F -q -- "$fragment" "$job_file"; then
    echo "Generated job validation failed: $job_file missing $fragment" >&2
    exit 1
  fi
}

validate_job_script() {
  local index="$1" job_file="$2"
  local run_name="${RUN_NAMES[$index]}" task="${JOB_TASKS[$index]}"
  local idm_coef="${JOB_IDM_COEFS[$index]}" idm_route="${JOB_IDM_ROUTES[$index]}"
  local idm_mode="${JOB_IDM_MODES[$index]}" idm_lr="${JOB_IDM_LRS[$index]}"
  local required
  for required in \
    "WANDB_MODE=online" \
    "WANDB_ENTITY=lmu_rl" \
    "WANDB_PROJECT=controllable_agent_cheetah_fb_stability" \
    "WANDB_RUN_NAME=$run_name" \
    "WANDB_RUN_ID=$run_name" \
    "WANDB_RESUME=never" \
    "task=$task" \
    "seed=1" \
    "obs_type=dino" \
    "dino_model_name=facebook/dinov2-base" \
    "use_cls=True" \
    "frame_stack=3" \
    "dino_frame_stack=3" \
    "goal_space=null" \
    "agent.pixel_separate_fb_encoders=False" \
    "agent.dino_use_adapter=True" \
    "agent.dino_adapter_type=mlp_ln" \
    "agent.dino_adapter_hidden_dim=1024" \
    "agent.dino_adapter_output_dim=512" \
    "agent.dino_separate_fb_adapters=True" \
    "agent.dino_separate_backward_adapter=False" \
    "agent.lr_f=0.0001" \
    "agent.lr_b=0.0001" \
    "agent.lr_actor=0.0001" \
    "agent.fb_target_tau=0.01" \
    "agent.batch_size=1024" \
    "agent.z_dim=50" \
    "agent.ortho_coef=1.0" \
    "agent.mix_ratio=0.5" \
    "agent.idm_coef=$idm_coef" \
    "agent.idm_lr=$idm_lr" \
    "agent.idm_route=$idm_route" \
    "agent.idm_encoder_mode=$idm_mode" \
    "num_train_frames=2000010" \
    "eval_every_frames=10000" \
    "num_eval_episodes=10" \
    "snapshot_at=\\[100000\\,200000\\,500000\\,800000\\,1000000\\,1500000\\,2000000\\]"; do
    assert_job_has "$job_file" "$required"
  done
  if rg -F -q -- "2304" "$job_file"; then
    echo "Generated job must not hard-code the CLS3 adapter input width: $job_file" >&2
    exit 1
  fi
  bash -n "$job_file"
}

printf 'job_index\tgroup\ttask\tgpu\tsession\trun_name\trun_dir\tcheckpoint_dir\tstdout_log\twandb_entity\twandb_project\twandb_run_id\tidm_coef\tidm_route\tidm_encoder_mode\tidm_lr\tdino_adapter_type\tdino_frame_stack\tnum_train_frames\teval_every_frames\tnum_eval_episodes\tsnapshot_at\tlr_f\tlr_b\tlr_actor\tortho_coef\tfb_target_tau\tz_dim\tmix_ratio\tbatch_size\tsemantics\tsource_fingerprint\tjob_file\n' > "$manifest"
for index in "${!RUN_NAMES[@]}"; do
  job_file="$jobs_dir/job_$(printf '%02d' "$index")_${RUN_NAMES[$index]}.sh"
  write_job_script "$index" "$job_file"
  validate_job_script "$index" "$job_file"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$index" "${JOB_GROUPS[$index]}" "${JOB_TASKS[$index]}" \
    "${JOB_GPUS[$index]}" "${SESSIONS[$index]}" "${RUN_NAMES[$index]}" \
    "${RUN_DIRS[$index]}" "${CKPT_DIRS[$index]}" "${RUN_DIRS[$index]}/stdout.log" \
    "$WANDB_ENTITY" "$WANDB_PROJECT" \
    "${RUN_NAMES[$index]}" "${JOB_IDM_COEFS[$index]}" "${JOB_IDM_ROUTES[$index]}" \
    "${JOB_IDM_MODES[$index]}" "${JOB_IDM_LRS[$index]}" "$DINO_ADAPTER_TYPE" \
    "$DINO_FRAME_STACK" "$NUM_TRAIN_FRAMES" "$EVAL_EVERY_FRAMES" \
    "$NUM_EVAL_EPISODES" "$SNAPSHOT_AT" "$LR_F" "$LR_B" "$LR_ACTOR" \
    "$ORTHO_COEF" "$FB_TARGET_TAU" "$Z_DIM" "$MIX_RATIO" "$BATCH_SIZE" \
    "online_adapters_no_adapter_ema" "$SOURCE_FINGERPRINT" "$job_file" >> "$manifest"
done

if [[ "$(awk 'END {print NR}' "$manifest")" -ne 13 ]]; then
  echo "Manifest validation failed: expected header plus 12 jobs." >&2
  exit 1
fi
for route in none forward_adapter backward_adapter; do
  if [[ "$(awk -F '\t' -v route="$route" 'NR > 1 && $14 == route {count++} END {print count+0}' "$manifest")" -ne 4 ]]; then
    echo "Manifest validation failed: expected four jobs for idm_route=$route" >&2
    exit 1
  fi
done
if [[ "$(awk -F '\t' 'NR > 1 {seen[$5]=1} END {for (key in seen) count++; print count+0}' "$manifest")" -ne 12 ]]; then
  echo "Manifest validation failed: expected 12 unique per-job tmux sessions." >&2
  exit 1
fi

ROUTING_CHECKER="${ROUTING_CHECKER:-$REPO_DIR/scripts/check_idm_adapter_routing.py}"
ADAPTER_CHECKER="${ADAPTER_CHECKER:-$REPO_DIR/scripts/check_separate_fb_adapters.py}"
for checker in "$ROUTING_CHECKER" "$ADAPTER_CHECKER"; do
  [[ -f "$checker" ]] || { echo "Missing required preflight checker: $checker" >&2; exit 1; }
done

{
  printf 'campaign=%s\n' "$CAMPAIGN"
  printf 'created_at=%s\n' "$(date -u +%FT%TZ)"
  printf 'dry_run=%s\n' "$DRY_RUN"
  printf 'semantics=online_adapters_no_adapter_ema\n'
  printf 'source_repo=%s\nsource_fingerprint=%s\n' "$REPO_DIR" "$SOURCE_FINGERPRINT"
  printf 'python=%s\n' "$PYTHON_BIN"
  "$PYTHON_BIN" --version 2>&1
  printf 'wandb=%s/%s\n' "$WANDB_ENTITY" "$WANDB_PROJECT"
  printf 'runs_root=%s\ncheckpoint_root=%s\n' "$RUNS_ROOT" "$CKPT_ROOT"
  printf 'gpus=%s\n' "$GPUS_STRING"
  for gpu in "${GPUS_ARRAY[@]}"; do
    jobs="${JOBS_PER_GPU[$gpu]:-0}"
    printf 'gpu_%s_jobs=%s\n' "$gpu" "$jobs"
    printf 'gpu_%s_required_free_mb=%s\n' \
      "$gpu" "$((GPU_RESERVE_MB + jobs * GPU_JOB_ESTIMATE_MB))"
  done
  printf 'num_jobs=%s\nnum_train_frames=%s\neval_every_frames=%s\n' \
    "$NUM_JOBS" "$NUM_TRAIN_FRAMES" "$EVAL_EVERY_FRAMES"
  printf 'snapshot_at=%s\ndino_frame_stack=%s\ndino_adapter_type=%s\n' \
    "$SNAPSHOT_AT" "$DINO_FRAME_STACK" "$DINO_ADAPTER_TYPE"
  printf 'routing_checker=%s\nadapter_checker=%s\n' "$ROUTING_CHECKER" "$ADAPTER_CHECKER"
} > "$launch_dir/launch_config.txt"

echo "Running corrected online-adapter semantic checks before any launch..."
{
  MUJOCO_GL=egl "$PYTHON_BIN" "$ADAPTER_CHECKER"
  MUJOCO_GL=egl "$PYTHON_BIN" "$ROUTING_CHECKER"
} | tee "$launch_dir/preflight_tests.log"

echo "Manifest: $manifest"
echo "Launch config: $launch_dir/launch_config.txt"
echo "Preflight tests: $launch_dir/preflight_tests.log"
if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "Dry run only; no production directories or tmux sessions were created."
  exit 0
fi

required_runs_gib="$((RUNS_RESERVE_GIB + NUM_JOBS * RUNS_GIB_PER_JOB))"
required_ckpt_gib="$((CKPT_RESERVE_GIB + NUM_JOBS * CKPT_GIB_PER_JOB))"
check_disk_capacity runs "$RUNS_ROOT" "$required_runs_gib"
check_disk_capacity checkpoints "$CKPT_ROOT" "$required_ckpt_gib"
for gpu in "${GPUS_ARRAY[@]}"; do
  check_gpu_capacity "$gpu" "${JOBS_PER_GPU[$gpu]:-0}"
done
for session in "${SESSIONS[@]}"; do
  if tmux has-session -t "$session" >/dev/null 2>&1; then
    echo "Fresh launch refused; tmux session exists: $session" >&2
    exit 1
  fi
done

mkdir -p "$RUNS_ROOT/$CAMPAIGN" "$CKPT_ROOT"
launch_failed=0
for index in "${!RUN_NAMES[@]}"; do
  session="${SESSIONS[$index]}"
  job_file="$jobs_dir/job_$(printf '%02d' "$index")_${RUN_NAMES[$index]}.sh"
  command="bash $(printf '%q' "$job_file")"
  if tmux new-session -d -s "$session" "$command"; then
    printf '[launched] session=%s gpu=%s group=%s run=%s\n' \
      "$session" "${JOB_GPUS[$index]}" "${JOB_GROUPS[$index]}" "${RUN_NAMES[$index]}"
  else
    echo "Failed to start tmux session: $session" >&2
    launch_failed=1
  fi
  if (( LAUNCH_STAGGER_SECONDS > 0 )); then
    sleep "$LAUNCH_STAGGER_SECONDS"
  fi
done
if [[ "$launch_failed" -ne 0 ]]; then
  echo "At least one job failed to start; inspect $launch_dir" >&2
  exit 1
fi

sleep 5
for session in "${SESSIONS[@]}"; do
  if ! tmux has-session -t "$session" >/dev/null 2>&1; then
    echo "Session disappeared during startup: $session" >&2
    launch_failed=1
  fi
done
if [[ "$launch_failed" -ne 0 ]]; then
  exit 1
fi
echo "Launched all 12 corrected DINO jobs in independent tmux sessions."
