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
EVAL_EVERY_FRAMES="${EVAL_EVERY_FRAMES:-10000}"
NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-10}"
NUM_TRAIN_FRAMES="${NUM_TRAIN_FRAMES:-2000010}"
FINAL_TESTS="${FINAL_TESTS:-10}"
TIMESTAMP="${TIMESTAMP:-$(date -u +%Y%m%d_%H%M%S)_mae_cls_single}"
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
Usage: ./launch_mae_cls_single_12tasks.sh [--dry-run] [--timestamp NAME]

Launch the 12-task, seed-1, single-frame frozen MAE CLS experiment.  It
matches the DINO CLS linear setup except for the visual backbone and its
model-native image processor.

Environment overrides:
  GPUS="2 5"                 Round-robin GPU assignment.
  SEED=1
  TASKS="cheetah_walk"       Optional whitespace-separated task subset.
  EVAL_EVERY_FRAMES=10000
  NUM_EVAL_EPISODES=10
  NUM_TRAIN_FRAMES=2000010
  FINAL_TESTS=10
  WANDB_PROJECT=controllable_agent_baseline
  REPO_DIR, TRAIN_SCRIPT, RUNS_DIR, CKPT_ROOT, LAUNCH_ROOT, TIMESTAMP
EOF
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

read -r -a GPUS_ARRAY <<< "${GPUS:-2 5}"
if [[ "${#GPUS_ARRAY[@]}" -eq 0 ]]; then
  echo "At least one GPU is required." >&2
  exit 2
fi

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

launch_dir="$LAUNCH_ROOT/$TIMESTAMP"
mkdir -p "$launch_dir"
manifest="$launch_dir/manifest.tsv"
printf 'session\tgpu\tseed\ttask\tgoal_space\trun_dir\tstdout_log\n' > "$manifest"

write_job_script() {
  local job_file="$1"
  local gpu="$2"
  local task="$3"
  local goal_space="$4"
  local run_dir="$5"

  {
    printf '#!/usr/bin/env bash\n'
    printf 'set -euo pipefail\n'
    printf 'cd %q\n' "$REPO_DIR"
    printf 'run_dir=%q\n' "$run_dir"
    printf 'mkdir -p "$run_dir"\n'
    printf 'echo "[start] $(date -u +%%FT%%TZ) task=%s seed=%s gpu=%s backbone=facebook/vit-mae-base frame_stack=1" | tee "$run_dir/launcher.log"\n' \
      "$task" "$SEED" "$gpu"
    printf 'if env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=%q PYTHONUNBUFFERED=1 WANDB_PROJECT=%q python %q ' \
      "$gpu" "$WANDB_PROJECT" "$TRAIN_SCRIPT"
    printf '%q ' \
      "use_wandb=True" \
      "use_tb=False" \
      "use_hiplog=False" \
      "save_video=False" \
      "save_train_video=False" \
      "save_replay_buffer_in_checkpoint=False" \
      "checkpoint_root=$CKPT_ROOT" \
      "obs_type=dino" \
      "dino_model_name=facebook/vit-mae-base" \
      "use_cls=True" \
      "frame_stack=1" \
      "dino_frame_stack=1" \
      "render_shape=[224,224]" \
      "action_repeat=2" \
      "agent.dino_use_adapter=True" \
      "agent.dino_adapter_type=linear" \
      "agent.dino_adapter_output_dim=512" \
      "agent.batch_size=1024" \
      "agent.update_every_steps=2" \
      "update_encoder=True" \
      "num_train_frames=$NUM_TRAIN_FRAMES" \
      "eval_every_frames=$EVAL_EVERY_FRAMES" \
      "num_eval_episodes=$NUM_EVAL_EPISODES" \
      "final_tests=$FINAL_TESTS" \
      "experiment=mae_cls_single_seed${SEED}" \
      "task=$task" \
      "goal_space=$goal_space" \
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

for index in "${!TASK_SPECS[@]}"; do
  read -r task goal_space <<< "${TASK_SPECS[$index]}"
  if ! task_selected "$task"; then
    continue
  fi
  gpu="${GPUS_ARRAY[$((index % ${#GPUS_ARRAY[@]}))]}"
  session="mae_s1_${TIMESTAMP}_s${SEED}_${task}_g${gpu}"
  run_dir="$RUNS_DIR/${TIMESTAMP}_seed${SEED}_${task}_cuda${gpu}_mae_cls_single"
  job_file="$launch_dir/${session}.sh"
  group_log="$launch_dir/${session}.log"

  write_job_script "$job_file" "$gpu" "$task" "$goal_space" "$run_dir"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$session" "$gpu" "$SEED" "$task" "$goal_space" "$run_dir" "$run_dir/stdout.log" \
    >> "$manifest"
  printf '%s\t%s\t%s\t%s\n' "$session" "$gpu" "$task" "$run_dir"

  if [[ "$DRY_RUN" -eq 0 ]]; then
    if tmux has-session -t "$session" >/dev/null 2>&1; then
      echo "tmux session already exists; skipping: $session" >&2
      continue
    fi
    if [[ -f "$run_dir/launcher.log" ]] && tail -n 1 "$run_dir/launcher.log" | grep -q 'exit=0'; then
      echo "completed run already exists; skipping: $run_dir" >&2
      continue
    fi
    tmux new-session -d -s "$session" \
      "bash $(printf '%q' "$job_file") > $(printf '%q' "$group_log") 2>&1"
  fi
done

echo "Manifest: $manifest"
if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "Dry run only; no tmux sessions were launched."
fi
