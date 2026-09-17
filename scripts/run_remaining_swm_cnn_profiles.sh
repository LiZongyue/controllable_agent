#!/usr/bin/env bash
set -euo pipefail

repo=/data/fanfeng/controllable_agent
out=${repo}/analysis_outputs/swm_cls_vs_cnn_fb_20260727/controlled_profiles
profiler=${repo}/scripts/profile_swm_cls_vs_cnn_fb.py

export CUDA_VISIBLE_DEVICES=2
export MUJOCO_GL=egl
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export MPLCONFIGDIR=/tmp/swm_cnn_mpl_gpu2

run_profile() {
  local method=$1
  local domain=$2
  local task=$3
  local output=$4
  local work_dir=$5
  taskset -c 8-15 /usr/bin/python "${profiler}" \
    --method "${method}" \
    --domain "${domain}" \
    --task "${task}" \
    --seed 2009 \
    --warmup-env-frames 10000 \
    --measured-env-frames 60000 \
    --num-windows 3 \
    --pure-update-warmup 10 \
    --pure-updates-per-window 100 \
    --nvml-sample-period-s 0.02 \
    --output "${output}" \
    --work-dir "${work_dir}"
}

cd "${repo}"

if [[ ! -s "${out}/profile_cnn_fb_quadruped.json" ]]; then
  run_profile cnn_fb quadruped quadruped_walk \
    "${out}/profile_cnn_fb_quadruped.json" \
    "${out}/work_formal_cnn_fb_quadruped_rerun1"
fi

if [[ ! -s "${out}/profile_swm_cls_cheetah.json" ]]; then
  run_profile swm_cls cheetah cheetah_walk \
    "${out}/profile_swm_cls_cheetah.json" \
    "${out}/work_formal_swm_cls_cheetah"
fi

if [[ ! -s "${out}/profile_cnn_fb_cheetah.json" ]]; then
  run_profile cnn_fb cheetah cheetah_walk \
    "${out}/profile_cnn_fb_cheetah.json" \
    "${out}/work_formal_cnn_fb_cheetah"
fi

/usr/bin/python scripts/aggregate_swm_cnn_controlled_profiles.py
