#!/usr/bin/env bash
set -euo pipefail

wait_pid=${WAIT_FOR_GPU_PID:-}
if [[ -n "${wait_pid}" ]]; then
  while nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
      | grep -qx "${wait_pid}"; do
    sleep 30
  done
fi

exec /data/fanfeng/controllable_agent/scripts/run_remaining_swm_cnn_profiles.sh
