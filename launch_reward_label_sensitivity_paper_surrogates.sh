#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/data/fanfeng/controllable_agent}"
RUN_ROOT="${RUN_ROOT:-/mnt/data_7tb/fanfeng/controllable_agent_runs}"
CKPT_ROOT="${CKPT_ROOT:-/mnt/data_7tb/fanfeng/controallable_agent_ckpt}"
OUT="${OUT:-$REPO_DIR/analysis_outputs/reward_label_sensitivity_dinov2_best12_exorl_rnd_anchor_20260727}"
GPU="${GPU:-2}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"
BANK_SIZE="${BANK_SIZE:-20480}"
KS="${KS:-1,4,16,64,256,1024,5120,20480}"
SUBSET_SEEDS="${SUBSET_SEEDS:-0}"
EVAL_SEEDS="${EVAL_SEEDS:-1101,1102,1103,1104,1105}"
BANK_SEED="${BANK_SEED:-20260727}"
EXORL_ROOT="${EXORL_ROOT:-/mnt/data_7tb/fanfeng/exoRL_datasets}"
TASKS_CSV="${TASKS_CSV:-}"

# Pre-specified retained-main-run diagnostic panel: one available checkpoint
# per task from the documented DINOv2 main-seed runs. The paper's aggregate
# does not carry a recoverable 12-checkpoint mapping, so this is not a replica
# of Table 1.
JOBS=(
  "cheetah_walk|20260426_123938_seed2009_cheetah_walk_cuda3_dino_cls|snapshot_800000.pt"
  "cheetah_walk_backward|20260426_123938_seed2009_cheetah_walk_backward_cuda3_dino_cls|snapshot_800000.pt"
  "cheetah_run|20260426_123938_seed8164_cheetah_run_cuda5_dino_cls|snapshot_1000000.pt"
  "cheetah_run_backward|20260426_123938_seed2009_cheetah_run_backward_cuda3_dino_cls|snapshot_800000.pt"
  "walker_stand|20260426_123938_seed1992_walker_stand_cuda6_dino_cls|snapshot_1000000.pt"
  "walker_walk|20260426_123938_seed1992_walker_walk_cuda6_dino_cls|snapshot_1000000.pt"
  "walker_run|20260426_123938_seed7532_walker_run_cuda0_dino_cls|snapshot_1500000.pt"
  "walker_flip|20260426_123938_seed2009_walker_flip_cuda3_dino_cls|snapshot_2000000.pt"
  "quadruped_stand|20260426_123938_seed7532_quadruped_stand_cuda0_dino_cls|snapshot_1000000.pt"
  "quadruped_walk|20260426_123938_seed7532_quadruped_walk_cuda0_dino_cls|snapshot_1500000.pt"
  "quadruped_run|20260426_123938_seed7532_quadruped_run_cuda0_dino_cls|snapshot_2000000.pt"
  "quadruped_jump|20260426_123938_seed1992_quadruped_jump_cuda6_dino_cls|snapshot_2000000.pt"
)

if [[ -n "$TASKS_CSV" ]]; then
  selected_jobs=()
  for job in "${JOBS[@]}"; do
    IFS='|' read -r task _ _ <<< "$job"
    if [[ ",$TASKS_CSV," == *",$task,"* ]]; then
      selected_jobs+=("$job")
    fi
  done
  JOBS=("${selected_jobs[@]}")
  if [[ "${#JOBS[@]}" -eq 0 ]]; then
    echo "TASKS_CSV did not match any configured task: $TASKS_CSV" >&2
    exit 2
  fi
fi

mkdir -p "$OUT/logs" "$OUT/matplotlib"
cd "$REPO_DIR"

run_job() {
  local job="$1"
  local task run_name checkpoint_name run_dir checkpoint
  IFS='|' read -r task run_name checkpoint_name <<< "$job"
  run_dir="$RUN_ROOT/$run_name"
  checkpoint="$CKPT_ROOT/$run_name/$checkpoint_name"
  if [[ ! -f "$run_dir/.hydra/config.yaml" ]]; then
    echo "Missing config: $run_dir/.hydra/config.yaml" >&2
    return 1
  fi
  if [[ ! -f "$checkpoint" ]]; then
    echo "Missing checkpoint: $checkpoint" >&2
    return 1
  fi
  echo "[launch] task=$task gpu=$GPU checkpoint=$checkpoint"
  env \
    CUDA_VISIBLE_DEVICES="$GPU" \
    MUJOCO_GL=egl \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    MPLCONFIGDIR="$OUT/matplotlib" \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    python -u -m url_benchmark.reward_label_sensitivity \
      --run-dir "$run_dir" \
      --checkpoint "$checkpoint" \
      --output-dir "$OUT" \
      --dino-model facebook/dinov2-base \
      --bank-size "$BANK_SIZE" \
      --bank-seed "$BANK_SEED" \
      --bank-source exorl_rnd \
      --exorl-root "$EXORL_ROOT" \
      --ks "$KS" \
      --qualities iid_clean \
      --subset-seeds "$SUBSET_SEEDS" \
      --eval-seeds "$EVAL_SEEDS" \
      --projection-method mean \
      --resume \
      > "$OUT/logs/$task.log" 2>&1
}

failed=0
for ((start = 0; start < ${#JOBS[@]}; start += MAX_PARALLEL)); do
  pids=()
  tasks=()
  for ((offset = 0; offset < MAX_PARALLEL && start + offset < ${#JOBS[@]}; offset++)); do
    index=$((start + offset))
    IFS='|' read -r task _ _ <<< "${JOBS[$index]}"
    run_job "${JOBS[$index]}" &
    pids+=("$!")
    tasks+=("$task")
  done
  for index in "${!pids[@]}"; do
    if wait "${pids[$index]}"; then
      echo "[done] task=${tasks[$index]}"
    else
      echo "[failed] task=${tasks[$index]} log=$OUT/logs/${tasks[$index]}.log" >&2
      failed=1
    fi
  done
done

exit "$failed"
