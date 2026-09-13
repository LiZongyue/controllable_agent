#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/data/fanfeng/controllable_agent}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-$REPO_DIR/url_benchmark/pretrain.py}"
RUNS_DIR="${RUNS_DIR:-/mnt/data_7tb/fanfeng/controllable_agent_runs}"
CKPT_ROOT="${CKPT_ROOT:-/mnt/data_7tb/fanfeng/controallable_agent_ckpt}"
LAUNCH_ROOT="${LAUNCH_ROOT:-$RUNS_DIR/launch_queues}"
WANDB_PROJECT="${WANDB_PROJECT:-controllable_agent_baseline}"
GPU="${GPU:-1}"
SEED="${SEED:-1}"
NUM_TRAIN_FRAMES="${NUM_TRAIN_FRAMES:-300010}"
EVAL_EVERY_FRAMES="${EVAL_EVERY_FRAMES:-10000}"
NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-10}"
TIMESTAMP="${TIMESTAMP:-$(date -u +%Y%m%d_%H%M%S)_quadruped_visual_b_separate_adapter}"
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
Usage: ./launch_quadruped_visual_b_separate_adapter_sweep.sh [--dry-run] [--timestamp NAME]

Launch three matched single-frame visual-B Quadruped runs with an independently
trained backward adapter and its EMA target. Frozen DINO features are shared and
cached; this does not add a second DINO forward pass.

Environment overrides:
  GPU=1 SEED=1 NUM_TRAIN_FRAMES=300010
  EVAL_EVERY_FRAMES=10000 NUM_EVAL_EPISODES=10
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

VARIANTS=(
  "separate_base|agent.fb_target_tau=0.01 agent.mix_ratio=0.5"
  "separate_slow_target|agent.fb_target_tau=0.001 agent.mix_ratio=0.5"
  "separate_slow_target_mix01|agent.fb_target_tau=0.001 agent.mix_ratio=0.1"
)

launch_dir="$LAUNCH_ROOT/$TIMESTAMP"
mkdir -p "$launch_dir"
manifest="$launch_dir/manifest.tsv"
printf 'session\tgpu\tseed\tvariant\trun_dir\tstdout_log\toverrides\n' > "$manifest"

for spec in "${VARIANTS[@]}"; do
  variant="${spec%%|*}"
  variant_overrides="${spec#*|}"
  session="qvbsep_${TIMESTAMP}_s${SEED}_${variant}_g${GPU}"
  run_dir="$RUNS_DIR/${TIMESTAMP}_seed${SEED}_quadruped_walk_${variant}_cuda${GPU}"
  job_file="$launch_dir/${session}.sh"
  group_log="$launch_dir/${session}.log"

  {
    printf '#!/usr/bin/env bash\n'
    printf 'set -euo pipefail\n'
    printf 'cd %q\n' "$REPO_DIR"
    printf 'run_dir=%q\n' "$run_dir"
    printf 'mkdir -p "$run_dir" "$run_dir/matplotlib"\n'
    printf 'echo "[start] $(date -u +%%FT%%TZ) variant=%s seed=%s gpu=%s" | tee "$run_dir/launcher.log"\n' \
      "$variant" "$SEED" "$GPU"
    printf 'if env CUDA_VISIBLE_DEVICES=%q PYTHONUNBUFFERED=1 WANDB_PROJECT=%q MPLCONFIGDIR="$run_dir/matplotlib" python %q ' \
      "$GPU" "$WANDB_PROJECT" "$TRAIN_SCRIPT"
    printf '%q ' \
      "agent=fb_ddpg" \
      "use_wandb=True" \
      "use_tb=False" \
      "use_hiplog=False" \
      "save_video=False" \
      "save_train_video=False" \
      "save_replay_buffer_in_checkpoint=False" \
      "checkpoint_root=$CKPT_ROOT" \
      "checkpoint_every=100000" \
      "snapshot_at=[100000,200000,300000]" \
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
      "agent.dino_separate_backward_adapter=True" \
      "agent.backward_encoder_grad_scale=1.0" \
      "agent.ortho_coef=1.0" \
      "agent.lr_coef=1.0" \
      "agent.batch_size=1024" \
      "agent.update_every_steps=2" \
      "agent.num_inference_steps=5120" \
      "update_encoder=True" \
      "num_seed_frames=4000" \
      "num_train_frames=$NUM_TRAIN_FRAMES" \
      "eval_every_frames=$EVAL_EVERY_FRAMES" \
      "num_eval_episodes=$NUM_EVAL_EPISODES" \
      "final_tests=0" \
      "experiment=quadruped_visual_b_${variant}_seed${SEED}" \
      "task=quadruped_walk" \
      "seed=$SEED" \
      "hydra.run.dir=$run_dir"
    for override in $variant_overrides; do
      printf '%q ' "$override"
    done
    printf '> "$run_dir/stdout.log" 2>&1; then\n'
    printf '  rc=0\n'
    printf 'else\n'
    printf '  rc=$?\n'
    printf 'fi\n'
    printf 'echo "[done] $(date -u +%%FT%%TZ) variant=%s seed=%s gpu=%s exit=$rc" | tee -a "$run_dir/launcher.log"\n' \
      "$variant" "$SEED" "$GPU"
    printf 'exit "$rc"\n'
  } > "$job_file"
  chmod +x "$job_file"

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$session" "$GPU" "$SEED" "$variant" "$run_dir" "$run_dir/stdout.log" "$variant_overrides" \
    >> "$manifest"
  printf '%s\t%s\t%s\t%s\n' "$session" "$GPU" "$variant" "$run_dir"

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
