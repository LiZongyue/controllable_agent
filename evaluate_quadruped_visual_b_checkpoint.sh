#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  cat <<'EOF' >&2
Usage: evaluate_quadruped_visual_b_checkpoint.sh RUN_DIR CHECKPOINT OUTPUT_DIR

Evaluate one reward-free Quadruped checkpoint on stand, walk, run, and jump.
Each reward direction uses the same seeded 20,480-state ExORL-RND bank and the
same ten evaluation seeds. Set GPU, BANK_SIZE, BANK_SEED, or EVAL_SEEDS through
environment variables if needed.
EOF
  exit 2
fi

RUN_DIR="$1"
CHECKPOINT="$2"
OUTPUT_DIR="$3"
GPU="${GPU:-1}"
BANK_SIZE="${BANK_SIZE:-20480}"
KS="${KS:-$BANK_SIZE}"
BANK_SEED="${BANK_SEED:-20260727}"
EVAL_SEEDS="${EVAL_SEEDS:-1101,1102,1103,1104,1105,1106,1107,1108,1109,1110}"
PROJECTION_METHOD="${PROJECTION_METHOD:-mean}"
RIDGE_ALPHA="${RIDGE_ALPHA:-0.01}"
REPO_DIR="${REPO_DIR:-/data/fanfeng/controllable_agent}"

if [[ ! -d "$RUN_DIR" ]]; then
  echo "Run directory does not exist: $RUN_DIR" >&2
  exit 2
fi
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Checkpoint does not exist: $CHECKPOINT" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR" "$OUTPUT_DIR/matplotlib"
cd "$REPO_DIR"

TASKS=(quadruped_stand quadruped_walk quadruped_run quadruped_jump)
for task in "${TASKS[@]}"; do
  echo "[quadruped-all-task-eval] task=$task checkpoint=$CHECKPOINT"
  env \
    CUDA_VISIBLE_DEVICES="$GPU" \
    MPLCONFIGDIR="$OUTPUT_DIR/matplotlib" \
    MUJOCO_GL=egl \
    PYTHONUNBUFFERED=1 \
    python -m url_benchmark.reward_label_sensitivity \
      --run-dir "$RUN_DIR" \
      --checkpoint "$CHECKPOINT" \
      --task "$task" \
      --output-dir "$OUTPUT_DIR" \
      --device cuda \
      --bank-source exorl_rnd \
      --bank-size "$BANK_SIZE" \
      --bank-seed "$BANK_SEED" \
      --ks "$KS" \
      --qualities iid_clean \
      --subset-seeds 0 \
      --eval-seeds "$EVAL_SEEDS" \
      --eval-condition clean \
      --projection-method "$PROJECTION_METHOD" \
      --ridge-alpha "$RIDGE_ALPHA" \
      --resume
done
