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
TIMESTAMP="${TIMESTAMP:-$(date -u +%Y%m%d_%H%M%S)_dino_cls_visual_b_single}"
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
Usage: ./launch_dino_cls_visual_b_single_8tasks.sh [--dry-run] [--timestamp NAME]

Launch the eight Walker/Quadruped single-frame DINOv2 CLS visual-backward
experiments. RGB observations are encoded by frozen DINOv2; the trainable
LayerNorm+Linear projector maps CLS features to 512 dimensions, and both the
forward and backward maps consume that visual representation.

Environment overrides:
  GPUS="3 4 5"               Round-robin GPU assignment.
  SEED=1
  TASKS="walker_stand"       Optional whitespace-separated task subset.
  EVAL_EVERY_FRAMES=1000     2000 non-initial evals over 2,000,010 frames.
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

read -r -a GPUS_ARRAY <<< "${GPUS:-3 4 5}"
if [[ "${#GPUS_ARRAY[@]}" -eq 0 ]]; then
  echo "At least one GPU is required." >&2
  exit 2
fi

TASK_NAMES=(
  walker_stand
  walker_walk
  walker_run
  walker_flip
  quadruped_stand
  quadruped_walk
  quadruped_run
  quadruped_jump
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
printf 'session\tgpu\tseed\ttask\tgoal_space\tdino_frame_stack\teval_every_frames\trun_dir\tstdout_log\n' > "$manifest"

write_job_script() {
  local job_file="$1"
  local gpu="$2"
  local task="$3"
  local run_dir="$4"

  {
    printf '#!/usr/bin/env bash\n'
    printf 'set -euo pipefail\n'
    printf 'cd %q\n' "$REPO_DIR"
    printf 'run_dir=%q\n' "$run_dir"
    printf 'mkdir -p "$run_dir"\n'
    printf 'echo "[start] $(date -u +%%FT%%TZ) task=%s seed=%s gpu=%s goal_space=null dino_frame_stack=1 eval_every_frames=%s" | tee "$run_dir/launcher.log"\n' \
      "$task" "$SEED" "$gpu" "$EVAL_EVERY_FRAMES"
    printf 'if env CUDA_VISIBLE_DEVICES=%q PYTHONUNBUFFERED=1 WANDB_PROJECT=%q python %q ' \
      "$gpu" "$WANDB_PROJECT" "$TRAIN_SCRIPT"
    printf '%q ' \
      "agent=fb_ddpg" \
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
      "frame_stack=1" \
      "dino_frame_stack=1" \
      "render_shape=[224,224]" \
      "action_repeat=2" \
      "goal_space=null" \
      "append_goal_to_observation=False" \
      "agent.dino_use_adapter=True" \
      "agent.dino_adapter_type=linear" \
      "agent.dino_adapter_output_dim=512" \
      "agent.batch_size=1024" \
      "agent.update_every_steps=2" \
      "agent.num_inference_steps=5120" \
      "update_encoder=True" \
      "num_seed_frames=4000" \
      "num_train_frames=2000010" \
      "eval_every_frames=$EVAL_EVERY_FRAMES" \
      "num_eval_episodes=10" \
      "experiment=dino_cls_visual_b_single_seed${SEED}" \
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

selected_index=0
for task in "${TASK_NAMES[@]}"; do
  if ! task_selected "$task"; then
    continue
  fi
  gpu="${GPUS_ARRAY[$((selected_index % ${#GPUS_ARRAY[@]}))]}"
  session="dino_vb1_${TIMESTAMP}_s${SEED}_${task}_g${gpu}"
  run_dir="$RUNS_DIR/${TIMESTAMP}_seed${SEED}_${task}_cuda${gpu}_dino_cls_visual_b_single"
  job_file="$launch_dir/${session}.sh"
  group_log="$launch_dir/${session}.log"

  write_job_script "$job_file" "$gpu" "$task" "$run_dir"
  printf '%s\t%s\t%s\t%s\tnull\t1\t%s\t%s\t%s\n' \
    "$session" "$gpu" "$SEED" "$task" "$EVAL_EVERY_FRAMES" "$run_dir" "$run_dir/stdout.log" \
    >> "$manifest"
  printf '%s\t%s\t%s\t%s\n' "$session" "$gpu" "$task" "$run_dir"
  selected_index=$((selected_index + 1))

  if [[ "$DRY_RUN" -eq 0 ]]; then
    if tmux has-session -t "$session" >/dev/null 2>&1; then
      echo "tmux session already exists: $session" >&2
      exit 1
    fi
    tmux new-session -d -s "$session" \
      "bash $(printf '%q' "$job_file") > $(printf '%q' "$group_log") 2>&1"
  fi
done

if [[ "$selected_index" -eq 0 ]]; then
  echo "No tasks selected." >&2
  exit 2
fi

echo "Manifest: $manifest"
if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "Dry run only; no tmux sessions were launched."
fi
