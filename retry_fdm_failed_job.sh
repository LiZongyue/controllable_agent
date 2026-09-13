#!/usr/bin/env bash
set -uo pipefail

if (( $# != 4 )); then
  echo "usage: $0 GPU JOB_SCRIPT CLAIM_FILE STATE_STEM" >&2
  exit 2
fi

gpu="$1"
job_script="$2"
claim_file="$3"
state_stem="$4"

plan_dir=/mnt/data_7tb/fanfeng/controllable_agent_runs/launch_queues/20260823_212300_five_domain_fb_matrix_s1
completed_dir="$plan_dir/queue/completed"
failed_dir="$plan_dir/queue/failed"

if [[ ! "$gpu" =~ ^[0-7]$ ]]; then
  echo "invalid GPU ordinal: $gpu" >&2
  exit 2
fi
if [[ ! -f "$job_script" || "$job_script" != "$plan_dir"/jobs/job*.sh ]]; then
  echo "invalid job script: $job_script" >&2
  exit 2
fi
if [[ ! -f "$claim_file" || "$claim_file" != "$plan_dir"/queue/claimed/*.claimed ]]; then
  echo "invalid claim file: $claim_file" >&2
  exit 2
fi
if [[ ! "$state_stem" =~ ^[0-9]{3}_job[0-9]{3}_[a-z0-9_]+\.[a-z0-9]+$ ]]; then
  echo "invalid queue state stem: $state_stem" >&2
  exit 2
fi

# The first attempts reached W&B initialization but stopped before frame 0.
# Their local run directories are archived before this wrapper is launched.
# Reuse each stable W&B ID so retries do not create duplicate runs.
if sed 's/WANDB_RESUME=never/WANDB_RESUME=allow/' "$job_script" | bash -s -- "$gpu"; then
  rc=0
  destination="$completed_dir/${state_stem}.done"
else
  rc=$?
  destination="$failed_dir/${state_stem}.rc${rc}.failed"
fi

printf 'retry_finished_at=%q\nretry_gpu=%q\nretry_exit_code=%q\n' \
  "$(date -u +%FT%TZ)" "$gpu" "$rc" >> "$claim_file"
mv "$claim_file" "$destination"
exit "$rc"
