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
IDM_COEF="${IDM_COEF:-0.0}"
IDM_LR="${IDM_LR:-}"
IDM_SUFFIX=""
if awk -v value="$IDM_COEF" 'BEGIN { exit !(value + 0 > 0) }'; then
  idm_coef_label="${IDM_COEF//./p}"
  IDM_SUFFIX="_idm${idm_coef_label}"
  if [[ -n "$IDM_LR" ]]; then
    idm_lr_label="${IDM_LR//./p}"
    IDM_SUFFIX+="_idmlr${idm_lr_label}"
  fi
fi
TIMESTAMP="${TIMESTAMP:-$(date -u +%Y%m%d_%H%M%S)_dino_cls_stack3}"
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
Usage: ./launch_dino_cls_stack3_12tasks.sh [--dry-run] [--timestamp NAME]

Launch the 12-task, seed-1 DINOv2 CLS three-frame-stack variant. The only
model-input change from the historical DINO CLS linear run is temporal
concatenation of three frozen DINO CLS embeddings (768 x 3 -> projector -> 512).

Environment overrides:
  GPUS="6 7"                 Round-robin GPU assignment.
  SEED=1
  TASKS="cheetah_walk"       Optional whitespace-separated task subset.
  EVAL_EVERY_FRAMES=1000     Gives 2000 eval events over 2,000,010 frames.
  IDM_COEF=0.0               Inverse-dynamics auxiliary coefficient; for
                             example, set 0.1 to enable it.
  IDM_LR=                    Optional separate IDM learning rate. Empty uses
                             the FB learning rate.
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

RUN_ID="${TIMESTAMP}${IDM_SUFFIX}"

read -r -a GPUS_ARRAY <<< "${GPUS:-6 7}"
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

launch_dir="$LAUNCH_ROOT/$RUN_ID"
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
    printf 'echo "[start] $(date -u +%%FT%%TZ) task=%s seed=%s gpu=%s dino_frame_stack=3 idm_coef=%s idm_lr=%s" | tee "$run_dir/launcher.log"\n' \
      "$task" "$SEED" "$gpu" "$IDM_COEF" "${IDM_LR:-agent.lr}"
    printf 'if env CUDA_VISIBLE_DEVICES=%q PYTHONUNBUFFERED=1 WANDB_PROJECT=%q python %q ' \
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
      "dino_model_name=facebook/dinov2-base" \
      "use_cls=True" \
      "frame_stack=3" \
      "dino_frame_stack=3" \
      "render_shape=[224,224]" \
      "action_repeat=2" \
      "agent.dino_use_adapter=True" \
      "agent.dino_adapter_type=linear" \
      "agent.dino_adapter_output_dim=512" \
      "agent.idm_coef=$IDM_COEF" \
      "agent.idm_lr=${IDM_LR:-null}" \
      "agent.batch_size=1024" \
      "agent.update_every_steps=2" \
      "update_encoder=True" \
      "num_train_frames=2000010" \
      "eval_every_frames=$EVAL_EVERY_FRAMES" \
      "num_eval_episodes=10" \
      "experiment=dino_cls_stack3${IDM_SUFFIX}_seed${SEED}" \
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
  session="dino_s3_${RUN_ID}_s${SEED}_${task}_g${gpu}"
  run_dir="$RUNS_DIR/${TIMESTAMP}_seed${SEED}_${task}_cuda${gpu}_dino_cls_stack3${IDM_SUFFIX}"
  job_file="$launch_dir/${session}.sh"
  group_log="$launch_dir/${session}.log"

  write_job_script "$job_file" "$gpu" "$task" "$goal_space" "$run_dir"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$session" "$gpu" "$SEED" "$task" "$goal_space" "$run_dir" "$run_dir/stdout.log" \
    >> "$manifest"
  printf '%s\t%s\t%s\t%s\n' "$session" "$gpu" "$task" "$run_dir"

  if [[ "$DRY_RUN" -eq 0 ]]; then
    if tmux has-session -t "$session" >/dev/null 2>&1; then
      echo "tmux session already exists: $session" >&2
      exit 1
    fi
    tmux new-session -d -s "$session" \
      "bash $(printf '%q' "$job_file") > $(printf '%q' "$group_log") 2>&1"
  fi
done

echo "Manifest: $manifest"
if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "Dry run only; no tmux sessions were launched."
fi
