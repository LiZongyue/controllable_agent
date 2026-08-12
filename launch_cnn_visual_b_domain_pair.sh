#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/data/fanfeng/controllable_agent}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-$REPO_DIR/url_benchmark/pretrain.py}"
RUNS_DIR="${RUNS_DIR:-/mnt/data_7tb/fanfeng/controllable_agent_runs}"
CKPT_ROOT="${CKPT_ROOT:-/mnt/data_7tb/fanfeng/controallable_agent_ckpt}"
LAUNCH_ROOT="${LAUNCH_ROOT:-$RUNS_DIR/launch_queues}"
WANDB_PROJECT="${WANDB_PROJECT:-controllable_agent_baseline}"
GPU_WALKER="${GPU_WALKER:-1}"
GPU_QUADRUPED="${GPU_QUADRUPED:-2}"
SEED="${SEED:-1}"
NUM_TRAIN_FRAMES="${NUM_TRAIN_FRAMES:-300010}"
EVAL_EVERY_FRAMES="${EVAL_EVERY_FRAMES:-10000}"
NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-10}"
TIMESTAMP="${TIMESTAMP:-$(date -u +%Y%m%d_%H%M%S)_cnn_visual_b_domains}"
WAIT_SECONDS="${WAIT_SECONDS:-30}"
DRY_RUN=0

# By default, preserve the currently running DINO visual-B stability sweeps.
# The CNN jobs start as soon as their assigned GPU's blocker sessions finish.
WALKER_BLOCKERS="${WALKER_BLOCKERS:-qvb_20260801_173637_qvb_stability_s1_control_g1 qvb_20260801_173637_qvb_stability_s1_mild_b_projector_grad_g1 qvb_20260801_173637_qvb_stability_s1_no_b_z_feedback_g1 qvb_20260801_173637_qvb_stability_s1_slow_b_g1 qvb_20260801_173637_qvb_stability_s1_slow_target_g1 qvb_20260801_173637_qvb_stability_s1_weak_orth_g1}"
QUADRUPED_BLOCKERS="${QUADRUPED_BLOCKERS:-qvbsep_20260801_174400_qvb_separate_s1_separate_base_g2 qvbsep_20260801_174400_qvb_separate_s1_separate_slow_target_g2 qvbsep_20260801_174400_qvb_separate_s1_separate_slow_target_mix01_g2}"

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
Usage: ./launch_cnn_visual_b_domain_pair.sh [--dry-run] [--timestamp NAME]

Queue one fully visual-B CNN-FB run per locomotion domain. Each reward-free
checkpoint can subsequently be evaluated on the four rewards in its domain.

Environment overrides:
  GPU_WALKER=1 GPU_QUADRUPED=2 SEED=1 NUM_TRAIN_FRAMES=300010
  EVAL_EVERY_FRAMES=10000 NUM_EVAL_EPISODES=10 WAIT_SECONDS=30
  WALKER_BLOCKERS="..." QUADRUPED_BLOCKERS="..."
  WANDB_PROJECT, REPO_DIR, TRAIN_SCRIPT, RUNS_DIR, CKPT_ROOT, LAUNCH_ROOT
EOF
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

launch_dir="$LAUNCH_ROOT/$TIMESTAMP"
mkdir -p "$launch_dir"
manifest="$launch_dir/manifest.tsv"
printf 'session\tgpu\tseed\tdomain\ttask\trun_dir\tstdout_log\tblockers\n' > "$manifest"

launch_domain() {
  local domain="$1"
  local task="$2"
  local gpu="$3"
  local blockers="$4"
  local session="cnnvis_${TIMESTAMP}_s${SEED}_${domain}_g${gpu}"
  local run_dir="$RUNS_DIR/${TIMESTAMP}_seed${SEED}_${task}_cnn_visual_b_cuda${gpu}"
  local job_file="$launch_dir/${session}.sh"
  local queue_log="$launch_dir/${session}.log"

  {
    printf '#!/usr/bin/env bash\n'
    printf 'set -euo pipefail\n'
    printf 'cd %q\n' "$REPO_DIR"
    printf 'run_dir=%q\n' "$run_dir"
    printf 'mkdir -p "$run_dir" "$run_dir/matplotlib"\n'
    printf 'blockers=%q\n' "$blockers"
    printf 'wait_seconds=%q\n' "$WAIT_SECONDS"
    cat <<'EOF'
while true; do
  active=()
  for blocker in $blockers; do
    if tmux has-session -t "$blocker" 2>/dev/null; then
      active+=("$blocker")
    fi
  done
  if [[ ${#active[@]} -eq 0 ]]; then
    break
  fi
  echo "[queue] $(date -u +%FT%TZ) waiting_for=${active[*]}"
  sleep "$wait_seconds"
done
EOF
    printf 'echo "[start] $(date -u +%%FT%%TZ) domain=%s task=%s seed=%s gpu=%s obs_type=pixels frame_stack=3 goal_space=null" | tee "$run_dir/launcher.log"\n' \
      "$domain" "$task" "$SEED" "$gpu"
    printf 'if env CUDA_VISIBLE_DEVICES=%q PYTHONUNBUFFERED=1 WANDB_PROJECT=%q MPLCONFIGDIR="$run_dir/matplotlib" python %q ' \
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
      "checkpoint_every=100000" \
      "snapshot_at=[100000,200000,300000]" \
      "obs_type=pixels" \
      "frame_stack=3" \
      "render_shape=[84,84]" \
      "action_repeat=2" \
      "goal_space=null" \
      "append_goal_to_observation=False" \
      "agent.batch_size=1024" \
      "agent.update_every_steps=2" \
      "agent.num_inference_steps=5120" \
      "agent.backward_encoder_grad_scale=1.0" \
      "agent.ortho_coef=1.0" \
      "agent.mix_ratio=0.5" \
      "agent.fb_target_tau=0.01" \
      "agent.lr_coef=1.0" \
      "update_encoder=True" \
      "num_seed_frames=4000" \
      "num_train_frames=$NUM_TRAIN_FRAMES" \
      "eval_every_frames=$EVAL_EVERY_FRAMES" \
      "num_eval_episodes=$NUM_EVAL_EPISODES" \
      "final_tests=0" \
      "experiment=cnn_visual_b_${domain}_seed${SEED}" \
      "task=$task" \
      "seed=$SEED" \
      "hydra.run.dir=$run_dir"
    printf '> "$run_dir/stdout.log" 2>&1; then\n'
    printf '  rc=0\n'
    printf 'else\n'
    printf '  rc=$?\n'
    printf 'fi\n'
    printf 'echo "[done] $(date -u +%%FT%%TZ) domain=%s task=%s seed=%s gpu=%s exit=$rc" | tee -a "$run_dir/launcher.log"\n' \
      "$domain" "$task" "$SEED" "$gpu"
    printf 'exit "$rc"\n'
  } > "$job_file"
  chmod +x "$job_file"

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$session" "$gpu" "$SEED" "$domain" "$task" "$run_dir" "$run_dir/stdout.log" "$blockers" \
    >> "$manifest"
  printf '%s\t%s\t%s\t%s\n' "$session" "$gpu" "$task" "$run_dir"

  if [[ "$DRY_RUN" -eq 0 ]]; then
    if tmux has-session -t "$session" >/dev/null 2>&1; then
      echo "tmux session already exists: $session" >&2
      exit 1
    fi
    tmux new-session -d -s "$session" \
      "bash $(printf '%q' "$job_file") > $(printf '%q' "$queue_log") 2>&1"
  fi
}

launch_domain "walker" "walker_walk" "$GPU_WALKER" "$WALKER_BLOCKERS"
launch_domain "quadruped" "quadruped_walk" "$GPU_QUADRUPED" "$QUADRUPED_BLOCKERS"

echo "Manifest: $manifest"
if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "Dry run only; no tmux sessions were launched."
fi
