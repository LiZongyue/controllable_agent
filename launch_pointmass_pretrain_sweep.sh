#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/data/fanfeng/controllable_agent}"
DREAMER_DIR="${DREAMER_DIR:-/data/fanfeng/dreamerv3}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-$REPO_DIR/url_benchmark/pretrain.py}"
RUNS_DIR="${RUNS_DIR:-/mnt/data_7tb/fanfeng/controllable_agent_runs}"
DREAMER_RUNS_DIR="${DREAMER_RUNS_DIR:-/mnt/data_7tb/fanfeng/dreamer_pointmass_runs}"
CKPT_ROOT="${CKPT_ROOT:-/mnt/data_7tb/fanfeng/controallable_agent_ckpt}"
LAUNCH_ROOT="${LAUNCH_ROOT:-$RUNS_DIR/launch_queues}"

TIMESTAMP="${TIMESTAMP:-$(date -u +%Y%m%d_%H%M%S)_pointmass}"
DRY_RUN="${DRY_RUN:-0}"
RUN_MODE="main"
RUN_MODE_ARGS=()

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
    --queue-seed)
      RUN_MODE="queue-seed"
      shift
      RUN_MODE_ARGS=("$@")
      set --
      ;;
    -h|--help)
      cat <<'EOF'
Usage: ./launch_pointmass_pretrain_sweep.sh [--dry-run] [--timestamp TS]

Launch point_mass pretraining sweep:
  dino_cls_linear, dino_cls_mlp, dino_patch_linear, dino_patch_mlp,
  dino_cls_no_adapter, cnn, vit, dreamer_zero_reward

Defaults:
  TASK=point_mass_easy
  SEEDS="7532 2009 8164 1992 3407 4518 6101 9284 1123 6679"
  GPUS="2 3 5 6 1 7 2 3 5 6"
  ACTIVE_PER_GPU=1 via memory-gated queued seeds
  NUM_TRAIN_FRAMES=500000
  EVAL_EVERY_FRAMES=100000
EOF
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

TASK="${TASK:-point_mass_easy}"
DREAMER_TASK="${DREAMER_TASK:-dmc_point_mass_easy}"
NUM_TRAIN_FRAMES="${NUM_TRAIN_FRAMES:-500000}"
EVAL_EVERY_FRAMES="${EVAL_EVERY_FRAMES:-100000}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-100000}"
SNAPSHOT_AT="${SNAPSHOT_AT:-[100000,500000]}"
MIN_FREE_MB="${MIN_FREE_MB:-60000}"

read -r -a SEEDS_ARRAY <<< "${SEEDS:-7532 2009 8164 1992 3407 4518 6101 9284 1123 6679}"
read -r -a GPUS_ARRAY <<< "${GPUS:-2 3 5 6 1 7 2 3 5 6}"

if [[ "${#SEEDS_ARRAY[@]}" -ne "${#GPUS_ARRAY[@]}" ]]; then
  echo "Expected equal number of seeds and GPUs." >&2
  echo "Seeds: ${SEEDS_ARRAY[*]}" >&2
  echo "GPUs: ${GPUS_ARRAY[*]}" >&2
  exit 2
fi

mkdir -p "$RUNS_DIR" "$DREAMER_RUNS_DIR" "$LAUNCH_ROOT/$TIMESTAMP"

gpu_free_mb() {
  local gpu="$1"
  timeout 30s nvidia-smi --id="$gpu" --query-gpu=memory.free --format=csv,noheader,nounits \
    | head -1 \
    | tr -dc '0-9' || true
}

wait_for_mem() {
  local gpu="$1"
  local min_free="$2"
  local free_mb=""
  while true; do
    free_mb="$(gpu_free_mb "$gpu")"
    if [[ -n "$free_mb" && "$free_mb" -ge "$min_free" ]]; then
      echo "[queue] $(date -u +%FT%TZ) gpu=$gpu free_mb=$free_mb >= $min_free"
      return 0
    fi
    echo "[queue] $(date -u +%FT%TZ) waiting gpu=$gpu free_mb=${free_mb:-unknown} need=$min_free"
    sleep 60
  done
}

active_sessions_on_gpu() {
  local gpu="$1"
  tmux ls 2>/dev/null \
    | rg "^pointmass_${TIMESTAMP}_.*_g${gpu}:" \
    | rg -v "_queue_" \
    | wc -l
}

wait_for_gpu_slot() {
  local gpu="$1"
  local count=""
  while true; do
    count="$(active_sessions_on_gpu "$gpu")"
    if [[ "$count" -eq 0 ]]; then
      echo "[queue] $(date -u +%FT%TZ) gpu=$gpu active_pointmass_sessions=0"
      return 0
    fi
    echo "[queue] $(date -u +%FT%TZ) waiting gpu=$gpu active_pointmass_sessions=$count"
    sleep 120
  done
}

write_fb_script() {
  local launch_file="$1"
  local gpu="$2"
  local seed="$3"
  local variant="$4"
  local run_dir="$5"

  local obs_type="dino"
  local render_shape="[224,224]"
  local frame_stack="1"
  local replay_eps="2000"
  local experiment="pointmass_${variant}"
  local extra=()

  case "$variant" in
    dino_cls_linear)
      extra=("use_cls=True" "agent.dino_use_adapter=True" "agent.dino_adapter_type=linear")
      ;;
    dino_cls_mlp)
      extra=("use_cls=True" "agent.dino_use_adapter=True" "agent.dino_adapter_type=mlp")
      ;;
    dino_patch_linear)
      extra=("use_cls=False" "agent.dino_use_adapter=True" "agent.dino_adapter_type=linear")
      ;;
    dino_patch_mlp)
      extra=("use_cls=False" "agent.dino_use_adapter=True" "agent.dino_adapter_type=mlp")
      ;;
    dino_cls_no_adapter)
      extra=("use_cls=True" "agent.dino_use_adapter=False" "agent.feature_dim=1024")
      ;;
    cnn)
      obs_type="pixels"
      render_shape="[84,84]"
      frame_stack="3"
      replay_eps="100"
      ;;
    vit)
      obs_type="vit"
      replay_eps="80"
      extra=("use_cls=True" "agent.vit_batch_size=64")
      ;;
    *)
      echo "Unknown FB variant: $variant" >&2
      exit 2
      ;;
  esac

  {
    printf '#!/usr/bin/env bash\n'
    printf 'set -euo pipefail\n'
    printf 'cd %q\n' "$REPO_DIR"
    printf 'run_dir=%q\n' "$run_dir"
    printf 'stdout_path="$run_dir/stdout.log"\n'
    printf 'mkdir -p "$run_dir"\n'
    printf 'echo "[start] $(date -u +%%FT%%TZ) pointmass variant=%s seed=%s gpu=%s run_dir=$run_dir" | tee "$run_dir/launcher.log"\n' \
      "$variant" "$seed" "$gpu"
    printf 'if env CUDA_VISIBLE_DEVICES=%q PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python %q ' "$gpu" "$TRAIN_SCRIPT"
    printf '%q ' \
      "use_wandb=True" \
      "save_video=False" \
      "save_train_video=False" \
      "save_replay_buffer_in_checkpoint=False" \
      "checkpoint_root=$CKPT_ROOT" \
      "checkpoint_every=$CHECKPOINT_EVERY" \
      "snapshot_at=$SNAPSHOT_AT" \
      "obs_type=$obs_type" \
      "render_shape=$render_shape" \
      "frame_stack=$frame_stack" \
      "action_repeat=1" \
      "task=$TASK" \
      "goal_space=null" \
      "custom_reward=null" \
      "seed=$seed" \
      "experiment=$experiment" \
      "num_train_frames=$NUM_TRAIN_FRAMES" \
      "eval_every_frames=$EVAL_EVERY_FRAMES" \
      "num_eval_episodes=3" \
      "final_tests=0" \
      "replay_buffer_episodes=$replay_eps" \
      "hydra.run.dir=$run_dir"
    for arg in "${extra[@]}"; do
      printf '%q ' "$arg"
    done
    printf '> "$stdout_path" 2>&1; then\n'
    printf '  rc=0\n'
    printf 'else\n'
    printf '  rc=$?\n'
    printf 'fi\n'
    printf 'echo "[done] $(date -u +%%FT%%TZ) pointmass variant=%s seed=%s gpu=%s exit=$rc" | tee -a "$run_dir/launcher.log"\n' \
      "$variant" "$seed" "$gpu"
    printf 'exit "$rc"\n'
  } > "$launch_file"
  chmod +x "$launch_file"
}

write_dreamer_script() {
  local launch_file="$1"
  local gpu="$2"
  local seed="$3"
  local run_dir="$4"

  {
    printf '#!/usr/bin/env bash\n'
    printf 'set -euo pipefail\n'
    printf 'cd %q\n' "$DREAMER_DIR"
    printf 'run_dir=%q\n' "$run_dir"
    printf 'stdout_path="$run_dir/stdout.log"\n'
    printf 'mkdir -p "$run_dir"\n'
    printf 'echo "[start] $(date -u +%%FT%%TZ) pointmass variant=dreamer_zero_reward seed=%s gpu=%s run_dir=$run_dir" | tee "$run_dir/launcher.log"\n' \
      "$seed" "$gpu"
    printf 'if env CUDA_VISIBLE_DEVICES=%q PYTHONPATH=%q PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl python %q ' \
      "$gpu" "$DREAMER_DIR" "$DREAMER_DIR/dreamerv3/main.py"
    printf '%q ' \
      "--configs" "dmc_vision" \
      "--task" "$DREAMER_TASK" \
      "--seed" "$seed" \
      "--logdir" "$run_dir" \
      "--run.steps" "$NUM_TRAIN_FRAMES" \
      "--run.envs" "4" \
      "--run.eval_envs" "1" \
      "--run.eval_eps" "3" \
      "--run.log_every" "120" \
      "--run.report_every" "300" \
      "--run.eval_every" "300" \
      "--run.save_every" "900" \
      "--logger.outputs" "jsonl" \
      "--jax.prealloc" "False" \
      "--agent.zero_reward" "True"
    printf '> "$stdout_path" 2>&1; then\n'
    printf '  rc=0\n'
    printf 'else\n'
    printf '  rc=$?\n'
    printf 'fi\n'
    printf 'echo "[done] $(date -u +%%FT%%TZ) pointmass variant=dreamer_zero_reward seed=%s gpu=%s exit=$rc" | tee -a "$run_dir/launcher.log"\n' \
      "$seed" "$gpu"
    printf 'exit "$rc"\n'
  } > "$launch_file"
  chmod +x "$launch_file"
}

launch_seed() {
  local seed="$1"
  local gpu="$2"

  local fb_variants=(
    dino_cls_linear
    dino_cls_mlp
    dino_patch_linear
    dino_patch_mlp
    dino_cls_no_adapter
    cnn
    vit
  )

  for variant in "${fb_variants[@]}"; do
    local session="pointmass_${TIMESTAMP}_${variant}_s${seed}_g${gpu}"
    local run_dir="$RUNS_DIR/${TIMESTAMP}_seed${seed}_${variant}_cuda${gpu}"
    local launch_file="$LAUNCH_ROOT/$TIMESTAMP/${session}.sh"
    local group_log="$LAUNCH_ROOT/$TIMESTAMP/${session}.log"
    write_fb_script "$launch_file" "$gpu" "$seed" "$variant" "$run_dir"
    echo -e "$session\t$gpu\t$seed\t$variant\t$run_dir\t$group_log"
    if [[ "$DRY_RUN" -eq 0 ]]; then
      tmux new-session -d -s "$session" "bash $(printf '%q' "$launch_file") > $(printf '%q' "$group_log") 2>&1"
    fi
  done

  local variant="dreamer_zero_reward"
  local session="pointmass_${TIMESTAMP}_${variant}_s${seed}_g${gpu}"
  local run_dir="$DREAMER_RUNS_DIR/${TIMESTAMP}_seed${seed}_${variant}_cuda${gpu}"
  local launch_file="$LAUNCH_ROOT/$TIMESTAMP/${session}.sh"
  local group_log="$LAUNCH_ROOT/$TIMESTAMP/${session}.log"
  write_dreamer_script "$launch_file" "$gpu" "$seed" "$run_dir"
  echo -e "$session\t$gpu\t$seed\t$variant\t$run_dir\t$group_log"
  if [[ "$DRY_RUN" -eq 0 ]]; then
    tmux new-session -d -s "$session" "bash $(printf '%q' "$launch_file") > $(printf '%q' "$group_log") 2>&1"
  fi
}

queue_seed_worker() {
  local seed="$1"
  local gpu="$2"
  local min_free="$3"
  local queue_log="$LAUNCH_ROOT/$TIMESTAMP/pointmass_${TIMESTAMP}_queue_seed${seed}_g${gpu}.log"
  wait_for_gpu_slot "$gpu" | tee "$queue_log"
  wait_for_mem "$gpu" "$min_free" | tee -a "$queue_log"
  DRY_RUN=0 launch_seed "$seed" "$gpu" >> "$queue_log" 2>&1
}

if [[ "$RUN_MODE" == "queue-seed" ]]; then
  queue_seed_worker "${RUN_MODE_ARGS[@]}"
  exit 0
fi

declare -A ACTIVE_BY_GPU=()

printf 'SESSION\tGPU\tSEED\tVARIANT\tRUN_DIR\tGROUP_LOG\n'
for i in "${!SEEDS_ARRAY[@]}"; do
  seed="${SEEDS_ARRAY[$i]}"
  gpu="${GPUS_ARRAY[$i]}"
  if [[ -z "${ACTIVE_BY_GPU[$gpu]:-}" ]]; then
    ACTIVE_BY_GPU[$gpu]=1
    launch_seed "$seed" "$gpu"
  else
    queue_session="pointmass_${TIMESTAMP}_queue_seed${seed}_g${gpu}"
    queue_log="$LAUNCH_ROOT/$TIMESTAMP/${queue_session}.log"
    echo -e "$queue_session\t$gpu\t$seed\tqueued_seed\tWAIT_MIN_FREE_${MIN_FREE_MB}\t$queue_log"
    if [[ "$DRY_RUN" -eq 0 ]]; then
      tmux new-session -d -s "$queue_session" \
        "cd $(printf '%q' "$REPO_DIR") && TIMESTAMP=$(printf '%q' "$TIMESTAMP") DRY_RUN=0 TASK=$(printf '%q' "$TASK") DREAMER_TASK=$(printf '%q' "$DREAMER_TASK") NUM_TRAIN_FRAMES=$(printf '%q' "$NUM_TRAIN_FRAMES") EVAL_EVERY_FRAMES=$(printf '%q' "$EVAL_EVERY_FRAMES") CHECKPOINT_EVERY=$(printf '%q' "$CHECKPOINT_EVERY") SNAPSHOT_AT=$(printf '%q' "$SNAPSHOT_AT") bash $(printf '%q' "$0") --queue-seed $(printf '%q' "$seed") $(printf '%q' "$gpu") $(printf '%q' "$MIN_FREE_MB") > $(printf '%q' "$queue_log") 2>&1"
    fi
  fi
done

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "Dry run only. Launch scripts were written, but tmux sessions were not launched."
fi
