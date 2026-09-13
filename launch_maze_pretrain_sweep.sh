#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/data/fanfeng/controllable_agent}"
DREAMER_DIR="${DREAMER_DIR:-/data/fanfeng/dreamerv3}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-$REPO_DIR/url_benchmark/pretrain.py}"
RUNS_DIR="${RUNS_DIR:-/mnt/data_7tb/fanfeng/controllable_agent_runs}"
DREAMER_RUNS_DIR="${DREAMER_RUNS_DIR:-/mnt/data_7tb/fanfeng/dreamer_maze_runs}"
CKPT_ROOT="${CKPT_ROOT:-/mnt/data_7tb/fanfeng/controallable_agent_ckpt}"
LAUNCH_ROOT="${LAUNCH_ROOT:-$RUNS_DIR/launch_queues}"

TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
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
Usage: ./launch_maze_pretrain_sweep.sh [--dry-run] [--timestamp YYYYMMDD_HHMMSS]

Launch maze pretraining sweep on 4 GPUs:
  FB variants:
    dino_cls_linear, dino_cls_mlp, dino_patch_linear, dino_patch_mlp,
    dino_cls_no_adapter, cnn, vit
  DreamerV3:
    dreamer_zero_reward

Environment overrides:
  SEEDS="7532 2009 8164 1992"
  GPUS="2 3 5 6"
  NUM_TRAIN_FRAMES=10000000
  EVAL_EVERY_FRAMES=500000
EOF
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

read -r -a SEEDS_ARRAY <<< "${SEEDS:-7532 2009 8164 1992}"
read -r -a GPUS_ARRAY <<< "${GPUS:-2 3 5 6}"

if [[ "${#SEEDS_ARRAY[@]}" -ne "${#GPUS_ARRAY[@]}" ]]; then
  echo "Expected equal number of seeds and GPUs." >&2
  echo "Seeds: ${SEEDS_ARRAY[*]}" >&2
  echo "GPUs: ${GPUS_ARRAY[*]}" >&2
  exit 2
fi

NUM_TRAIN_FRAMES="${NUM_TRAIN_FRAMES:-10000000}"
EVAL_EVERY_FRAMES="${EVAL_EVERY_FRAMES:-500000}"

TASK="point_mass_maze_multi_goal"
GOAL_SPACE="simplified_point_mass_maze"
CUSTOM_REWARD="maze_multi_goal"
SNAPSHOT_AT="[100000,500000,1000000,2000000,5000000,10000000]"

mkdir -p "$RUNS_DIR" "$DREAMER_RUNS_DIR" "$LAUNCH_ROOT/$TIMESTAMP"

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
  local experiment="maze_${variant}"
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
      extra=()
      ;;
    vit)
      obs_type="vit"
      render_shape="[224,224]"
      frame_stack="1"
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
    printf 'echo "[start] $(date -u +%%FT%%TZ) variant=%s seed=%s gpu=%s run_dir=$run_dir" | tee "$run_dir/launcher.log"\n' \
      "$variant" "$seed" "$gpu"
    printf 'if env CUDA_VISIBLE_DEVICES=%q PYTHONUNBUFFERED=1 python %q ' "$gpu" "$TRAIN_SCRIPT"
    printf '%q ' \
      "use_wandb=True" \
      "save_video=False" \
      "save_train_video=False" \
      "save_replay_buffer_in_checkpoint=False" \
      "checkpoint_root=$CKPT_ROOT" \
      "checkpoint_every=500000" \
      "snapshot_at=$SNAPSHOT_AT" \
      "obs_type=$obs_type" \
      "render_shape=$render_shape" \
      "frame_stack=$frame_stack" \
      "action_repeat=1" \
      "task=$TASK" \
      "goal_space=$GOAL_SPACE" \
      "custom_reward=$CUSTOM_REWARD" \
      "seed=$seed" \
      "experiment=$experiment" \
      "num_train_frames=$NUM_TRAIN_FRAMES" \
      "eval_every_frames=$EVAL_EVERY_FRAMES" \
      "num_eval_episodes=1" \
      "final_tests=10" \
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
    printf 'echo "[done] $(date -u +%%FT%%TZ) variant=%s seed=%s gpu=%s exit=$rc" | tee -a "$run_dir/launcher.log"\n' \
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
    printf 'echo "[start] $(date -u +%%FT%%TZ) variant=dreamer_zero_reward seed=%s gpu=%s run_dir=$run_dir" | tee "$run_dir/launcher.log"\n' \
      "$seed" "$gpu"
    printf 'if env CUDA_VISIBLE_DEVICES=%q PYTHONPATH=%q PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl python %q ' \
      "$gpu" "$DREAMER_DIR" "$DREAMER_DIR/dreamerv3/main.py"
    printf '%q ' \
      "--configs" "dmc_vision" \
      "--task" "dmc_point_mass_maze_multi_goal" \
      "--seed" "$seed" \
      "--logdir" "$run_dir" \
      "--run.steps" "$NUM_TRAIN_FRAMES" \
      "--run.envs" "4" \
      "--run.eval_envs" "1" \
      "--run.eval_eps" "10" \
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
    printf 'echo "[done] $(date -u +%%FT%%TZ) variant=dreamer_zero_reward seed=%s gpu=%s exit=$rc" | tee -a "$run_dir/launcher.log"\n' \
      "$seed" "$gpu"
    printf 'exit "$rc"\n'
  } > "$launch_file"
  chmod +x "$launch_file"
}

printf 'SESSION\tGPU\tSEED\tVARIANT\tRUN_DIR\tGROUP_LOG\n'

FB_VARIANTS=(
  dino_cls_linear
  dino_cls_mlp
  dino_patch_linear
  dino_patch_mlp
  dino_cls_no_adapter
  cnn
  vit
)

for i in "${!SEEDS_ARRAY[@]}"; do
  seed="${SEEDS_ARRAY[$i]}"
  gpu="${GPUS_ARRAY[$i]}"

  for variant in "${FB_VARIANTS[@]}"; do
    session="maze_${TIMESTAMP}_${variant}_s${seed}_g${gpu}"
    run_dir="$RUNS_DIR/${TIMESTAMP}_seed${seed}_${variant}_cuda${gpu}"
    launch_file="$LAUNCH_ROOT/$TIMESTAMP/${session}.sh"
    group_log="$LAUNCH_ROOT/$TIMESTAMP/${session}.log"
    write_fb_script "$launch_file" "$gpu" "$seed" "$variant" "$run_dir"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$session" "$gpu" "$seed" "$variant" "$run_dir" "$group_log"
    if [[ "$DRY_RUN" -eq 0 ]]; then
      tmux new-session -d -s "$session" "bash $(printf '%q' "$launch_file") > $(printf '%q' "$group_log") 2>&1"
    fi
  done

  variant="dreamer_zero_reward"
  session="maze_${TIMESTAMP}_${variant}_s${seed}_g${gpu}"
  run_dir="$DREAMER_RUNS_DIR/${TIMESTAMP}_seed${seed}_${variant}_cuda${gpu}"
  launch_file="$LAUNCH_ROOT/$TIMESTAMP/${session}.sh"
  group_log="$LAUNCH_ROOT/$TIMESTAMP/${session}.log"
  write_dreamer_script "$launch_file" "$gpu" "$seed" "$run_dir"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$session" "$gpu" "$seed" "$variant" "$run_dir" "$group_log"
  if [[ "$DRY_RUN" -eq 0 ]]; then
    tmux new-session -d -s "$session" "bash $(printf '%q' "$launch_file") > $(printf '%q' "$group_log") 2>&1"
  fi
done

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "Dry run only. Launch scripts were written, but tmux sessions were not launched."
fi
