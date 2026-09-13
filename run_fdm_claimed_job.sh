#!/usr/bin/env bash
set -uo pipefail

if (( $# != 2 )); then
  echo "usage: $0 GPU CLAIM_FILE" >&2
  exit 2
fi

gpu="$1"
claim_file="$2"

plan_dir=/mnt/data_7tb/fanfeng/controllable_agent_runs/launch_queues/20260823_212300_five_domain_fb_matrix_s1
completed_dir="$plan_dir/queue/completed"
failed_dir="$plan_dir/queue/failed"

if [[ ! "$gpu" =~ ^[0-7]$ ]]; then
  echo "invalid GPU ordinal: $gpu" >&2
  exit 2
fi
if [[ ! -f "$claim_file" || "$claim_file" != "$plan_dir"/queue/claimed/*.claimed ]]; then
  echo "invalid claim file: $claim_file" >&2
  exit 2
fi

unset job_id queue_priority variant task run_name job_file
source "$claim_file"
: "${job_id:?missing job_id in claim}"
: "${variant:?missing variant in claim}"
: "${task:?missing task in claim}"
: "${job_file:?missing job_file in claim}"

if [[ ! -f "$job_file" || "$job_file" != "$plan_dir"/jobs/job*.sh ]]; then
  echo "invalid job script in claim: $job_file" >&2
  exit 2
fi

echo "[one-shot-start] $(date -u +%FT%TZ) gpu=$gpu job_id=$job_id variant=$variant task=$task"
if bash "$job_file" "$gpu"; then
  rc=0
  destination="$completed_dir/${claim_file##*/}"
  destination="${destination%.claimed}.done"
else
  rc=$?
  destination="$failed_dir/${claim_file##*/}"
  destination="${destination%.claimed}.rc${rc}.failed"
fi

printf 'finished_at=%q\ngpu=%q\nexit_code=%q\n' \
  "$(date -u +%FT%TZ)" "$gpu" "$rc" >> "$claim_file"
mv "$claim_file" "$destination"
echo "[one-shot-finished] $(date -u +%FT%TZ) gpu=$gpu job_id=$job_id variant=$variant task=$task exit=$rc"
exit "$rc"
