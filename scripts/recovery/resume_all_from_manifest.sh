#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 {preflight|launch} MANIFEST RECOVERY_ROOT" >&2
  exit 2
}

[[ $# -eq 3 ]] || usage
mode=$1
manifest=$2
recovery_root=$3
[[ "$mode" == preflight || "$mode" == launch ]] || usage
[[ -f "$manifest" ]] || { echo "[refused] missing manifest: $manifest" >&2; exit 1; }
[[ "$recovery_root" == /mnt/data_7tb/fanfeng/controllable_agent_runs/* ]] || { echo "[refused] unexpected recovery root: $recovery_root" >&2; exit 1; }

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
worker=$script_dir/resume_one_from_job.sh
[[ -x "$worker" ]] || { echo "[refused] worker is not executable: $worker" >&2; exit 1; }

expected_header=$'group\ttask\tgpu\tsession\trun_name\tcheckpoint_frame\tcheckpoint\twandb_id\tsource_job'
IFS= read -r header < "$manifest"
[[ "$header" == "$expected_header" ]] || { echo "[refused] unexpected manifest header" >&2; exit 1; }

declare -A seen_session=()
declare -A seen_run=()
declare -A seen_checkpoint=()
rows=0
migrated=0
strict=0

while IFS=$'\t' read -r group task gpu session run_name checkpoint_frame checkpoint wandb_id source_job; do
  [[ -n "$group" && -n "$task" && -n "$source_job" ]] || { echo "[refused] incomplete manifest row $((rows + 2))" >&2; exit 1; }
  [[ -z "${seen_session[$session]+x}" ]] || { echo "[refused] duplicate session: $session" >&2; exit 1; }
  [[ -z "${seen_run[$run_name]+x}" ]] || { echo "[refused] duplicate run name: $run_name" >&2; exit 1; }
  [[ -z "${seen_checkpoint[$checkpoint]+x}" ]] || { echo "[refused] duplicate checkpoint: $checkpoint" >&2; exit 1; }
  seen_session[$session]=1
  seen_run[$run_name]=1
  seen_checkpoint[$checkpoint]=1
  check_output=$("$worker" check "$source_job" "$recovery_root" "$session" "$run_name" "$checkpoint" "$gpu" "$wandb_id" "$checkpoint_frame")
  printf '%s\n' "$check_output"
  if [[ "$check_output" == *$'source_migrated=true'* ]]; then
    ((migrated += 1))
  else
    ((strict += 1))
  fi
  ((rows += 1))
done < <(tail -n +2 "$manifest")

[[ "$rows" -eq 36 ]] || { echo "[refused] expected 36 tasks, found $rows" >&2; exit 1; }
[[ "$strict" -eq 16 && "$migrated" -eq 20 ]] || {
  echo "[refused] unexpected source-fingerprint split: strict=$strict migrated=$migrated" >&2
  exit 1
}
printf '[preflight-ok] tasks=%s source_exact=%s source_migrated=%s recovery_root=%s\n' "$rows" "$strict" "$migrated" "$recovery_root"

if [[ "$mode" == preflight ]]; then
  exit 0
fi

[[ ! -e "$recovery_root" && ! -L "$recovery_root" ]] || { echo "[refused] recovery root already exists: $recovery_root" >&2; exit 1; }

active_sessions=$(tmux list-sessions -F '#{session_name}' 2>/dev/null || true)
for session in "${!seen_session[@]}"; do
  if grep -Fxq "$session" <<< "$active_sessions"; then
    echo "[refused] target tmux session still exists: $session" >&2
    exit 1
  fi
done

mkdir -p "$recovery_root/generated_jobs" "$recovery_root/.claims" "$recovery_root/.locks" "$recovery_root/audit" "$recovery_root/bootstrap"
cp -- "$manifest" "$recovery_root/manifest.tsv"
cp -- "$worker" "$recovery_root/resume_one_from_job.sh"
chmod 0550 "$recovery_root/resume_one_from_job.sh"
worker=$recovery_root/resume_one_from_job.sh

started=0
while IFS=$'\t' read -r group task gpu session run_name checkpoint_frame checkpoint wandb_id source_job; do
  bootstrap_log=$recovery_root/bootstrap/$session.log
  lock_file=$recovery_root/.locks/$run_name.lock
  printf -v worker_command '%q ' "$worker" run "$source_job" "$recovery_root" "$session" "$run_name" "$checkpoint" "$gpu" "$wandb_id" "$checkpoint_frame"
  printf -v shell_command 'exec 9>%q; flock -n 9 || { echo %q >&2; exit 73; }; exec %s >> %q 2>&1' \
    "$lock_file" "[refused] resume lock busy: $run_name" "$worker_command" "$bootstrap_log"
  tmux new-session -d -s "$session" "$shell_command"
  if [[ "$group" == csepenc ]]; then
    sleep 2
  else
    sleep 0.25
  fi
  if ! tmux has-session -t "$session" 2>/dev/null; then
    echo "[failed] session exited during bootstrap: $session log=$bootstrap_log" >&2
    exit 1
  fi
  ((started += 1))
  printf '[launched] %s/%s session=%s run=%s gpu=%s frame=%s\n' "$started" "$rows" "$session" "$run_name" "$gpu" "$checkpoint_frame"
done < <(tail -n +2 "$manifest")

printf '[launch-ok] started=%s recovery_root=%s\n' "$started" "$recovery_root"
