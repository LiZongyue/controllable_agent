#!/usr/bin/env bash
set -uo pipefail

gpu="${1:-7}"
plan_dir=/mnt/data_7tb/fanfeng/controllable_agent_runs/launch_queues/20260823_212300_five_domain_fb_matrix_s1
job_script="$plan_dir/jobs/job009_dino_cls1_fb_cheetah_walk.sh"
claim_file="$plan_dir/queue/claimed/009_job009_dino_cls1_fb_cheetah_walk.g7retry1.claimed"
completed_dir="$plan_dir/queue/completed"
failed_dir="$plan_dir/queue/failed"

# The first attempt created the stable W&B run but aborted before frame 0.
# Resume that same identity; the failed local directory was preserved with a
# .failed_rc134 timestamp suffix before this retry was launched.
if sed 's/WANDB_RESUME=never/WANDB_RESUME=allow/' "$job_script" | bash -s -- "$gpu"; then
  rc=0
  destination="$completed_dir/009_job009_dino_cls1_fb_cheetah_walk.g7retry1.done"
else
  rc=$?
  destination="$failed_dir/009_job009_dino_cls1_fb_cheetah_walk.g7retry1.rc${rc}.failed"
fi

printf 'retry_finished_at=%q\nretry_gpu=%q\nretry_exit_code=%q\n' \
  "$(date -u +%FT%TZ)" "$gpu" "$rc" >> "$claim_file"
mv "$claim_file" "$destination"
exit "$rc"
