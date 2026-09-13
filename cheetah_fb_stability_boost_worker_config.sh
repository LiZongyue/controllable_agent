#!/usr/bin/env bash

# CPU-capped boost workers for the existing 2026-08-28 Cheetah stability
# campaign.  This file is sourced by that campaign's generated worker.sh.

pending_dir=/mnt/data_7tb/fanfeng/controllable_agent_runs/launch_queues/20260828_cheetah_fb_cls3_ortho_lr_s1/queue/pending
claimed_dir=/mnt/data_7tb/fanfeng/controllable_agent_runs/launch_queues/20260828_cheetah_fb_cls3_ortho_lr_s1/queue/claimed
completed_dir=/mnt/data_7tb/fanfeng/controllable_agent_runs/launch_queues/20260828_cheetah_fb_cls3_ortho_lr_s1/queue/completed
failed_dir=/mnt/data_7tb/fanfeng/controllable_agent_runs/launch_queues/20260828_cheetah_fb_cls3_ortho_lr_s1/queue/failed
queue_lock=/mnt/data_7tb/fanfeng/controllable_agent_runs/launch_queues/20260828_cheetah_fb_cls3_ortho_lr_s1/queue/claim.lock
runs_root=/mnt/data_7tb/fanfeng/controllable_agent_runs
ckpt_root=/mnt/data_nvme1/fanfeng/controllable_agent_ckpt/20260828_cheetah_fb_cls3_ortho_lr_s1

wait_seconds=60
default_min_free_mb=35000
default_max_gpu_util=100
default_ready_checks=1
gpu_min_free_mb_overrides='6:50000 7:35000'
gpu_max_util_overrides='6:100 7:100'
gpu_ready_checks_overrides='6:1 7:1'
min_runs_free_kib=20971520
min_ckpt_free_kib=31457280
max_consecutive_failures=1
