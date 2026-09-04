#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/data/fanfeng/controllable_agent}"
RUN_ROOT="${RUN_ROOT:-/mnt/data_7tb/fanfeng/controllable_agent_runs}"
CKPT_ROOT="${CKPT_ROOT:-/mnt/data_7tb/fanfeng/controallable_agent_ckpt}"
OUT="${OUT:-$REPO_DIR/analysis_outputs/reward_label_sensitivity_dinov2_best12}"
LOG_TAG="${LOG_TAG:-}"
GPUS_CSV="${GPUS_CSV:-2,3,4}"
BANK_SIZE="${BANK_SIZE:-20480}"
KS="${KS:-16,64,256,1024,5120,20480}"
QUALITIES="${QUALITIES:-iid_clean,correlated_clean,iid_corrupt20}"
SUBSET_SEEDS="${SUBSET_SEEDS:-0,1,2}"
EVAL_SEEDS="${EVAL_SEEDS:-1101,1102,1103,1104,1105}"
BANK_SEED="${BANK_SEED:-20260727}"
COLLECTOR="${COLLECTOR:-random_z_sample}"
BANK_SOURCE="${BANK_SOURCE:-rollout}"
EXORL_ROOT="${EXORL_ROOT:-/mnt/data_7tb/fanfeng/exoRL_datasets}"

IFS=',' read -r -a GPUS <<< "$GPUS_CSV"
if [[ "${#GPUS[@]}" -eq 0 ]]; then
  echo "No GPUs configured" >&2
  exit 2
fi

# task|run directory basename|retained checkpoint
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

mkdir -p "$OUT/logs" "$OUT/matplotlib"
cd "$REPO_DIR"

log_suffix=""
if [[ -n "$LOG_TAG" ]]; then
  log_suffix="_$LOG_TAG"
fi

pids=()
tasks=()
for index in "${!JOBS[@]}"; do
  IFS='|' read -r task run_name checkpoint_name <<< "${JOBS[$index]}"
  gpu="${GPUS[$((index % ${#GPUS[@]}))]}"
  run_dir="$RUN_ROOT/$run_name"
  checkpoint="$CKPT_ROOT/$run_name/$checkpoint_name"
  if [[ ! -f "$run_dir/.hydra/config.yaml" ]]; then
    echo "Missing config: $run_dir/.hydra/config.yaml" >&2
    exit 1
  fi
  if [[ ! -f "$checkpoint" ]]; then
    echo "Missing checkpoint: $checkpoint" >&2
    exit 1
  fi
  echo "[launch] task=$task gpu=$gpu checkpoint=$checkpoint"
  env \
    CUDA_VISIBLE_DEVICES="$gpu" \
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
      --bank-source "$BANK_SOURCE" \
      --exorl-root "$EXORL_ROOT" \
      --collector "$COLLECTOR" \
      --ks "$KS" \
      --qualities "$QUALITIES" \
      --subset-seeds "$SUBSET_SEEDS" \
      --eval-seeds "$EVAL_SEEDS" \
      --resume \
      > "$OUT/logs/${task}${log_suffix}.log" 2>&1 &
  pids+=("$!")
  tasks+=("$task")
done

failed=0
for index in "${!pids[@]}"; do
  pid="${pids[$index]}"
  task="${tasks[$index]}"
  if wait "$pid"; then
    echo "[done] task=$task"
  else
    status="$?"
    echo "[failed] task=$task status=$status log=$OUT/logs/$task.log" >&2
    failed=1
  fi
done

exit "$failed"
