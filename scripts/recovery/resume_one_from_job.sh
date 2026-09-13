#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 {check|run} SOURCE_JOB RECOVERY_ROOT SESSION RUN_NAME CHECKPOINT GPU WANDB_ID CHECKPOINT_FRAME" >&2
  exit 2
}

[[ $# -eq 9 ]] || usage
mode=$1
source_job=$2
recovery_root=$3
session=$4
run_name=$5
checkpoint=$6
gpu=$7
wandb_id=$8
checkpoint_frame=$9

[[ "$mode" == check || "$mode" == run ]] || usage
[[ "$source_job" == /* && -f "$source_job" ]] || { echo "[refused] missing source job: $source_job" >&2; exit 1; }
[[ "$recovery_root" == /mnt/data_7tb/fanfeng/controllable_agent_runs/* ]] || { echo "[refused] unexpected recovery root: $recovery_root" >&2; exit 1; }
[[ "$session" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "[refused] unsafe session: $session" >&2; exit 1; }
[[ "$run_name" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "[refused] unsafe run name: $run_name" >&2; exit 1; }
[[ "$gpu" =~ ^[0-7]$ ]] || { echo "[refused] invalid GPU: $gpu" >&2; exit 1; }
[[ "$wandb_id" == "$run_name" ]] || { echo "[refused] W&B ID differs from run name: $wandb_id != $run_name" >&2; exit 1; }
[[ "$checkpoint_frame" =~ ^[0-9]+$ ]] || { echo "[refused] invalid checkpoint frame: $checkpoint_frame" >&2; exit 1; }
[[ "$checkpoint" == /*/latest.pt && -f "$checkpoint" && -s "$checkpoint" ]] || { echo "[refused] missing/empty checkpoint: $checkpoint" >&2; exit 1; }

repo=/data/fanfeng/controllable_agent
cd "$repo"

single_value() {
  local label=$1
  local values=$2
  local count
  count=$(printf '%s\n' "$values" | sed '/^$/d' | wc -l)
  [[ "$count" -eq 1 ]] || { echo "[refused] expected one $label in $source_job, found $count" >&2; exit 1; }
  printf '%s\n' "$values"
}

job_run_name=$(single_value run_name "$(sed -n 's/^run_name=//p' "$source_job")")
job_gpu=$(single_value CUDA_VISIBLE_DEVICES "$(grep -oE 'CUDA_VISIBLE_DEVICES=[0-9]+' "$source_job" | cut -d= -f2 | sort -u)")
job_wandb_id=$(single_value WANDB_RUN_ID "$(grep -oE 'WANDB_RUN_ID=[^[:space:]]+' "$source_job" | cut -d= -f2 | sort -u)")
checkpoint_root=$(single_value checkpoint_root "$(grep -oE 'checkpoint_root=[^[:space:]]+' "$source_job" | cut -d= -f2 | sort -u)")
original_source_fingerprint=$(single_value expected_source_fingerprint "$(sed -n 's/^expected_source_fingerprint=//p' "$source_job")")

[[ "$job_run_name" == "$run_name" ]] || { echo "[refused] source run mismatch: $job_run_name != $run_name" >&2; exit 1; }
[[ "$job_gpu" == "$gpu" ]] || { echo "[refused] source GPU mismatch: $job_gpu != $gpu" >&2; exit 1; }
[[ "$job_wandb_id" == "$wandb_id" ]] || { echo "[refused] source W&B ID mismatch: $job_wandb_id != $wandb_id" >&2; exit 1; }
[[ "$checkpoint" == "$checkpoint_root/$run_name/latest.pt" ]] || {
  echo "[refused] checkpoint does not equal checkpoint_root/run_name/latest.pt: $checkpoint" >&2
  exit 1
}

mapfile -t source_files < <(
  awk '
    /^source_files=\($/ { inside=1; next }
    inside && /^\)$/ { exit }
    inside { sub(/^[[:space:]]+/, ""); if (length($0)) print }
  ' "$source_job"
)
[[ ${#source_files[@]} -gt 0 ]] || { echo "[refused] no source fingerprint file list: $source_job" >&2; exit 1; }
for source_file in "${source_files[@]}"; do
  [[ -f "$source_file" ]] || { echo "[refused] missing fingerprint source file: $source_file" >&2; exit 1; }
done
recovery_source_fingerprint=$(
  {
    for source_file in "${source_files[@]}"; do
      sha256sum "$source_file" | awk '{print $1}'
    done
  } | sha256sum | awk '{print $1}'
)
if [[ "$original_source_fingerprint" == "$recovery_source_fingerprint" ]]; then
  source_migrated=false
else
  source_migrated=true
fi

run_dir=$recovery_root/$run_name
claim_dir=$recovery_root/.claims/$run_name
generated_job=$recovery_root/generated_jobs/$session.sh
audit_file=$recovery_root/audit/$session.tsv
checkpoint_size=$(stat -c %s "$checkpoint")

if [[ "$mode" == check ]]; then
  printf 'CHECK_OK\tsession=%s\trun=%s\tgpu=%s\tframe=%s\tcheckpoint_bytes=%s\tsource_migrated=%s\toriginal_fingerprint=%s\trecovery_fingerprint=%s\n' \
    "$session" "$run_name" "$gpu" "$checkpoint_frame" "$checkpoint_size" "$source_migrated" \
    "$original_source_fingerprint" "$recovery_source_fingerprint"
  exit 0
fi

[[ -d "$recovery_root/generated_jobs" && -d "$recovery_root/.claims" && -d "$recovery_root/audit" ]] || {
  echo "[refused] recovery root was not initialized: $recovery_root" >&2
  exit 1
}
[[ ! -e "$run_dir" && ! -L "$run_dir" ]] || { echo "[refused] recovery run directory exists: $run_dir" >&2; exit 1; }
[[ ! -e "$claim_dir" && ! -L "$claim_dir" ]] || { echo "[refused] recovery claim exists: $claim_dir" >&2; exit 1; }
[[ ! -e "$generated_job" && ! -L "$generated_job" ]] || { echo "[refused] generated job exists: $generated_job" >&2; exit 1; }

tmp_job=$generated_job.tmp.$$
cleanup() {
  [[ ! -e "$tmp_job" ]] || rm -f -- "$tmp_job"
}
trap cleanup EXIT

sed \
  -e "s|^run_dir=.*$|run_dir=$run_dir|" \
  -e "s|^ckpt_dir=.*$|ckpt_dir=$claim_dir|" \
  -e "s|^expected_source_fingerprint=.*$|expected_source_fingerprint=$recovery_source_fingerprint|" \
  -e 's/WANDB_RESUME=never/WANDB_RESUME=must/g' \
  -e 's/WANDB_RESUME=allow/WANDB_RESUME=must/g' \
  -e 's/auto_resume=False/auto_resume=True/g' \
  -e "s|hydra.run.dir=[^[:space:]]*|hydra.run.dir=$run_dir|g" \
  "$source_job" > "$tmp_job"

bash -n "$tmp_job"
[[ $(grep -c "^run_dir=$run_dir$" "$tmp_job") -eq 1 ]] || { echo "[refused] generated run_dir invariant failed" >&2; exit 1; }
[[ $(grep -c "^ckpt_dir=$claim_dir$" "$tmp_job") -eq 1 ]] || { echo "[refused] generated claim invariant failed" >&2; exit 1; }
[[ $(grep -o 'WANDB_RESUME=must' "$tmp_job" | wc -l) -eq 1 ]] || { echo "[refused] generated W&B resume invariant failed" >&2; exit 1; }
[[ $(grep -o 'auto_resume=True' "$tmp_job" | wc -l) -eq 1 ]] || { echo "[refused] generated auto-resume invariant failed" >&2; exit 1; }
[[ $(grep -o "hydra.run.dir=$run_dir" "$tmp_job" | wc -l) -eq 1 ]] || { echo "[refused] generated Hydra directory invariant failed" >&2; exit 1; }
[[ $(grep -o "checkpoint_root=$checkpoint_root" "$tmp_job" | wc -l) -eq 1 ]] || { echo "[refused] generated checkpoint root invariant failed" >&2; exit 1; }
[[ $(grep -o "WANDB_RUN_ID=$wandb_id" "$tmp_job" | wc -l) -eq 1 ]] || { echo "[refused] generated W&B ID invariant failed" >&2; exit 1; }
[[ $(grep -o "CUDA_VISIBLE_DEVICES=$gpu" "$tmp_job" | wc -l) -eq 1 ]] || { echo "[refused] generated GPU invariant failed" >&2; exit 1; }
if grep -qE 'WANDB_RESUME=(never|allow)|auto_resume=False' "$tmp_job"; then
  echo "[refused] generated job retains fresh-start settings" >&2
  exit 1
fi

mv -- "$tmp_job" "$generated_job"
chmod 0550 "$generated_job"
generated_job_sha256=$(sha256sum "$generated_job" | awk '{print $1}')

audit_tmp=$audit_file.tmp.$$
{
  printf 'session\trun_name\tgpu\tcheckpoint_frame\tcheckpoint\tcheckpoint_bytes\twandb_id\tsource_job\toriginal_source_fingerprint\trecovery_source_fingerprint\tsource_migrated\tgenerated_job\tgenerated_job_sha256\n'
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$session" "$run_name" "$gpu" "$checkpoint_frame" "$checkpoint" "$checkpoint_size" "$wandb_id" \
    "$source_job" "$original_source_fingerprint" "$recovery_source_fingerprint" "$source_migrated" \
    "$generated_job" "$generated_job_sha256"
} > "$audit_tmp"
mv -- "$audit_tmp" "$audit_file"

printf '[resume-bootstrap] %s session=%s run=%s gpu=%s frame=%s checkpoint=%s source_migrated=%s\n' \
  "$(date -u +%FT%TZ)" "$session" "$run_name" "$gpu" "$checkpoint_frame" "$checkpoint" "$source_migrated"
exec env WANDB_MODE=online bash "$generated_job"
