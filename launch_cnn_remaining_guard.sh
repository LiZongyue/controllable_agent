#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/mnt/data_7tb/fanfeng/controllable_agent_runs/launch_queues/20260503_232630_dmc_cnn_2seed}
TAG=${TAG:-20260503_232630_dmc_cnn_2seed}
FREE_THRESHOLD_MB=${FREE_THRESHOLD_MB:-16000}
POLL_SECONDS=${POLL_SECONDS:-60}

SCRIPTS=(
  "$ROOT/cnn_20260503_232630_dmc_cnn_2seed_s2009_cheetah_run_backward_g2_slot3.sh"
  "$ROOT/cnn_20260503_232630_dmc_cnn_2seed_s2009_quadruped_jump_g4_slot5.sh"
  "$ROOT/cnn_20260503_232630_dmc_cnn_2seed_s2009_quadruped_stand_g0_slot2.sh"
  "$ROOT/cnn_20260503_232630_dmc_cnn_2seed_s2009_quadruped_walk_g2_slot3.sh"
  "$ROOT/cnn_20260503_232630_dmc_cnn_2seed_s2009_walker_walk_g4_slot5.sh"
  "$ROOT/cnn_20260503_232630_dmc_cnn_2seed_s7532_quadruped_jump_g4_slot5.sh"
)

gpu_cap() {
  case "$1" in
    0) echo 3 ;;
    2) echo 2 ;;
    3) echo 8 ;;
    4) echo 5 ;;
    *) echo 1 ;;
  esac
}

cnn_pids() {
  ps -u "$USER" -o pid=,comm=,args= \
    | awk -v tag="$TAG" '$2 == "python" && index($0, "pretrain.py") && index($0, tag) { print $1 }'
}

cnn_count_on_gpu() {
  local gpu=$1
  local count=0
  local pid env
  while read -r pid; do
    [ -n "$pid" ] || continue
    env=$(tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null | grep '^CUDA_VISIBLE_DEVICES=' || true)
    if [ "$env" = "CUDA_VISIBLE_DEVICES=$gpu" ]; then
      count=$((count + 1))
    fi
  done < <(cnn_pids)
  echo "$count"
}

gpu_free_mb() {
  nvidia-smi --id="$1" --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d ' '
}

launched=()
for _ in "${SCRIPTS[@]}"; do
  launched+=(0)
done

remaining=${#SCRIPTS[@]}
echo "[guard-start] $(date -u +%FT%TZ) remaining=$remaining free_threshold_mb=$FREE_THRESHOLD_MB"

while [ "$remaining" -gt 0 ]; do
  for i in "${!SCRIPTS[@]}"; do
    [ "${launched[$i]}" = 0 ] || continue
    script=${SCRIPTS[$i]}
    base=$(basename "$script" .sh)
    gpu=$(sed -n 's/.*CUDA_VISIBLE_DEVICES=\([0-9]\).*/\1/p' "$script" | head -n 1)
    cap=$(gpu_cap "$gpu")
    current=$(cnn_count_on_gpu "$gpu")
    free=$(gpu_free_mb "$gpu")

    if [ "$current" -lt "$cap" ] && [ "$free" -ge "$FREE_THRESHOLD_MB" ]; then
      session="cnnextra_${gpu}_${base}"
      if tmux has-session -t "$session" 2>/dev/null; then
        echo "[skip-existing] $(date -u +%FT%TZ) $base"
      else
        tmux new-session -d -s "$session" "EGL_DEVICE_ID=$gpu bash '$script' > '${script%.sh}.log' 2>&1"
        echo "[launch] $(date -u +%FT%TZ) gpu=$gpu current=$current cap=$cap free_mb=$free $base"
      fi
      launched[$i]=1
      remaining=$((remaining - 1))
    else
      echo "[wait] $(date -u +%FT%TZ) gpu=$gpu current=$current cap=$cap free_mb=$free $base"
    fi
  done
  [ "$remaining" -eq 0 ] || sleep "$POLL_SECONDS"
done

echo "[guard-done] $(date -u +%FT%TZ)"
