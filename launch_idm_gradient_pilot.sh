#!/usr/bin/env bash
set -euo pipefail

# Prepare (but never start) the single-seed IDM gradient-strength pilot.  The
# generated task scripts are launched later by a resource-gated controller once
# the current baseline slots have been released.

REPO_DIR="${REPO_DIR:-/data/fanfeng/controllable_agent}"
BASE_LAUNCHER="${BASE_LAUNCHER:-$REPO_DIR/launch_dino_cls_stack3_12tasks.sh}"
RUNS_DIR="${RUNS_DIR:-/mnt/data_7tb/fanfeng/controllable_agent_runs}"
CKPT_ROOT="${CKPT_ROOT:-/mnt/data_7tb/fanfeng/controallable_agent_ckpt}"
LAUNCH_ROOT="${LAUNCH_ROOT:-$RUNS_DIR/launch_queues}"
WANDB_PROJECT="${WANDB_PROJECT:-controllable_agent_idm_pilot}"
TIMESTAMP="${TIMESTAMP:-$(date -u +%Y%m%d_%H%M%S)_idm_grad_pilot}"
SEED="${SEED:-1}"

TASKS="${TASKS:-walker_flip cheetah_walk quadruped_walk}"
GPU_ASSIGNMENTS="${GPU_ASSIGNMENTS:-2 4 6}"
NUM_TRAIN_FRAMES="${NUM_TRAIN_FRAMES:-500000}"
EVAL_EVERY_FRAMES="${EVAL_EVERY_FRAMES:-10000}"
IDM_DIAGNOSTICS_INTERVAL="${IDM_DIAGNOSTICS_INTERVAL:-500}"
IDM_ENCODER_BURNIN_STEPS="${IDM_ENCODER_BURNIN_STEPS:-25000}"
IDM_ENCODER_RAMP_STEPS="${IDM_ENCODER_RAMP_STEPS:-50000}"

if [[ $# -ne 0 ]]; then
  echo "This preparation script takes no arguments; configure it with environment variables." >&2
  exit 2
fi
if [[ ! -x "$BASE_LAUNCHER" ]]; then
  echo "Base launcher is not executable: $BASE_LAUNCHER" >&2
  exit 2
fi

settings=(static1 static10 rho1pct rho5pct)
modes=(static static balanced balanced)
coefs=(1.0 10.0 1.0 1.0)
targets=("" "" 0.01 0.05)

index_dir="$LAUNCH_ROOT/${TIMESTAMP}_pilot_index"
index_file="$index_dir/manifests.tsv"
mkdir -p "$index_dir"
printf 'setting\tmode\tidm_coef\tidm_lr\tgrad_ratio_target\tmanifest\n' > "$index_file"

for ordinal in "${!settings[@]}"; do
  setting="${settings[$ordinal]}"
  mode="${modes[$ordinal]}"
  coef="${coefs[$ordinal]}"
  target="${targets[$ordinal]}"
  ramp_steps="$IDM_ENCODER_RAMP_STEPS"
  if [[ "$mode" == "balanced" ]]; then
    ramp_steps=0
  fi

  output="$({
    REPO_DIR="$REPO_DIR" \
    TRAIN_SCRIPT="$REPO_DIR/url_benchmark/pretrain.py" \
    RUNS_DIR="$RUNS_DIR" \
    CKPT_ROOT="$CKPT_ROOT" \
    LAUNCH_ROOT="$LAUNCH_ROOT" \
    WANDB_PROJECT="$WANDB_PROJECT" \
    TIMESTAMP="$TIMESTAMP" \
    SEED="$SEED" \
    TASKS="$TASKS" \
    GPU_ASSIGNMENTS="$GPU_ASSIGNMENTS" \
    NUM_TRAIN_FRAMES="$NUM_TRAIN_FRAMES" \
    EVAL_EVERY_FRAMES="$EVAL_EVERY_FRAMES" \
    STAGE="$setting" \
    IDM_COEF="$coef" \
    IDM_LR=0.0001 \
    IDM_DIAGNOSTICS_INTERVAL="$IDM_DIAGNOSTICS_INTERVAL" \
    IDM_ENCODER_MODE="$mode" \
    IDM_ENCODER_BURNIN_STEPS="$IDM_ENCODER_BURNIN_STEPS" \
    IDM_ENCODER_RAMP_STEPS="$ramp_steps" \
    IDM_GRAD_RATIO_TARGET="$target" \
    "$BASE_LAUNCHER" --dry-run --parallel
  } 2>&1)"
  printf '%s\n' "$output"
  manifest="$(printf '%s\n' "$output" | sed -n 's/^Manifest: //p')"
  if [[ -z "$manifest" || ! -s "$manifest" ]]; then
    echo "Failed to locate generated manifest for setting $setting." >&2
    exit 1
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$setting" "$mode" "$coef" "0.0001" "${target:-none}" "$manifest" \
    >> "$index_file"
done

echo "Pilot index: $index_file"
echo "Prepared 12 task scripts (3 tasks x 4 settings); no tmux sessions were launched."
