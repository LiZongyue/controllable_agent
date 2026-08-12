#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/data/fanfeng/controllable_agent}"
DREAMER_DIR="${DREAMER_DIR:-/data/fanfeng/dreamerv3}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-$REPO_DIR/url_benchmark/pretrain.py}"
RUNS_DIR="${RUNS_DIR:-/mnt/data_7tb/fanfeng/controllable_agent_runs}"
DREAMER_RUNS_DIR="${DREAMER_RUNS_DIR:-/mnt/data_7tb/fanfeng/dreamer_corner_runs}"
CKPT_ROOT="${CKPT_ROOT:-/mnt/data_7tb/fanfeng/controallable_agent_ckpt}"
LAUNCH_ROOT="${LAUNCH_ROOT:-$RUNS_DIR/launch_queues}"

TIMESTAMP="${TIMESTAMP:-$(date -u +%Y%m%d_%H%M%S)_submission_corner_parallel}"
ONE_SEED="${ONE_SEED:-1992}"
CLS_LINEAR_SEEDS="${CLS_LINEAR_SEEDS:-1992 7532 2009 8164 3407}"
NUM_TRAIN_FRAMES="${NUM_TRAIN_FRAMES:-500000}"
EVAL_EVERY_FRAMES="${EVAL_EVERY_FRAMES:-100000}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-100000}"
SNAPSHOT_AT="${SNAPSHOT_AT:-[100000,500000]}"
WANDB_PROJECT="${WANDB_PROJECT:-controllable_agent_baseline}"

read -r -a TASKS_ARRAY <<< "${TASKS:-point_mass_maze_reach_top_left point_mass_maze_reach_top_right point_mass_maze_reach_bottom_left point_mass_maze_reach_bottom_right}"
read -r -a VIT_GPUS_ARRAY <<< "${VIT_GPUS:-1 2 5 6}"
read -r -a OTHER_GPUS_ARRAY <<< "${OTHER_GPUS:-0 3 7}"
read -r -a OTHER_GPU_SLOTS_ARRAY <<< "${OTHER_GPU_SLOTS:-6 12 10}"
read -r -a CLS_LINEAR_SEEDS_ARRAY <<< "$CLS_LINEAR_SEEDS"

if [[ "${#TASKS_ARRAY[@]}" -ne "${#VIT_GPUS_ARRAY[@]}" ]]; then
  echo "Expected one ViT GPU per task." >&2
  exit 2
fi

if [[ "${#OTHER_GPUS_ARRAY[@]}" -ne "${#OTHER_GPU_SLOTS_ARRAY[@]}" ]]; then
  echo "Expected OTHER_GPUS and OTHER_GPU_SLOTS to have the same length." >&2
  exit 2
fi

mkdir -p "$RUNS_DIR" "$DREAMER_RUNS_DIR" "$CKPT_ROOT" "$LAUNCH_ROOT/$TIMESTAMP"

task_short() {
  local task="$1"
  case "$task" in
    point_mass_maze_reach_top_left) echo "top_left" ;;
    point_mass_maze_reach_top_right) echo "top_right" ;;
    point_mass_maze_reach_bottom_left) echo "bottom_left" ;;
    point_mass_maze_reach_bottom_right) echo "bottom_right" ;;
    *) echo "$task" | tr -c '[:alnum:]_' '_' ;;
  esac
}

variant_args() {
  local variant="$1"
  case "$variant" in
    dino_cls_linear)
      echo "dino [224,224] 1 2000 use_cls=True agent.dino_use_adapter=True agent.dino_adapter_type=linear"
      ;;
    dino_cls_mlp)
      echo "dino [224,224] 1 2000 use_cls=True agent.dino_use_adapter=True agent.dino_adapter_type=mlp"
      ;;
    dino_patch_linear)
      echo "dino [224,224] 1 2000 use_cls=False agent.dino_use_adapter=True agent.dino_adapter_type=linear"
      ;;
    dino_patch_mlp)
      echo "dino [224,224] 1 2000 use_cls=False agent.dino_use_adapter=True agent.dino_adapter_type=mlp"
      ;;
    dino_cls_no_adapter)
      echo "dino [224,224] 1 2000 use_cls=True agent.dino_use_adapter=False agent.feature_dim=1024"
      ;;
    cnn)
      echo "pixels [84,84] 3 100"
      ;;
    vit)
      echo "vit [224,224] 1 80 use_cls=True agent.vit_batch_size=64"
      ;;
    *)
      echo "Unknown FB variant: $variant" >&2
      exit 2
      ;;
  esac
}

write_fb_script() {
  local launch_file="$1"
  local gpu="$2"
  local task="$3"
  local seed="$4"
  local variant="$5"
  local run_dir="$6"

  local short obs_type render_shape frame_stack replay_eps
  short="$(task_short "$task")"
  read -r obs_type render_shape frame_stack replay_eps extra_args <<< "$(variant_args "$variant")"

  {
    printf '#!/usr/bin/env bash\n'
    printf 'set -euo pipefail\n'
    printf 'cd %q\n' "$REPO_DIR"
    printf 'run_dir=%q\n' "$run_dir"
    printf 'stdout_path="$run_dir/stdout.log"\n'
    printf 'mkdir -p "$run_dir"\n'
    printf 'echo "[start] $(date -u +%%FT%%TZ) corner_task=%s variant=%s seed=%s gpu=%s run_dir=$run_dir" | tee "$run_dir/launcher.log"\n' \
      "$short" "$variant" "$seed" "$gpu"
    printf 'if env CUDA_VISIBLE_DEVICES=%q PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python %q ' "$gpu" "$TRAIN_SCRIPT"
    printf '%q ' \
      "use_wandb=True" \
      "save_video=False" \
      "save_train_video=False" \
      "save_replay_buffer_in_checkpoint=False" \
      "checkpoint_root=$CKPT_ROOT" \
      "checkpoint_every=$CHECKPOINT_EVERY" \
      "snapshot_at=$SNAPSHOT_AT" \
      "obs_type=$obs_type" \
      "render_shape=$render_shape" \
      "frame_stack=$frame_stack" \
      "action_repeat=1" \
      "task=$task" \
      "goal_space=null" \
      "custom_reward=null" \
      "seed=$seed" \
      "experiment=submission_${short}_${variant}" \
      "num_train_frames=$NUM_TRAIN_FRAMES" \
      "eval_every_frames=$EVAL_EVERY_FRAMES" \
      "num_eval_episodes=3" \
      "final_tests=0" \
      "replay_buffer_episodes=$replay_eps" \
      "hydra.run.dir=$run_dir"
    if [[ -n "${extra_args:-}" ]]; then
      read -r -a extra_array <<< "$extra_args"
      for arg in "${extra_array[@]}"; do
        printf '%q ' "$arg"
      done
    fi
    printf '> "$stdout_path" 2>&1; then\n'
    printf '  rc=0\n'
    printf 'else\n'
    printf '  rc=$?\n'
    printf 'fi\n'
    printf 'echo "[done] $(date -u +%%FT%%TZ) corner_task=%s variant=%s seed=%s gpu=%s exit=$rc" | tee -a "$run_dir/launcher.log"\n' \
      "$short" "$variant" "$seed" "$gpu"
    printf 'exit "$rc"\n'
  } > "$launch_file"
  chmod +x "$launch_file"
}

write_dreamer_script() {
  local launch_file="$1"
  local gpu="$2"
  local task="$3"
  local seed="$4"
  local run_dir="$5"

  local short dreamer_task
  short="$(task_short "$task")"
  dreamer_task="dmc_${task}"

  {
    printf '#!/usr/bin/env bash\n'
    printf 'set -euo pipefail\n'
    printf 'cd %q\n' "$DREAMER_DIR"
    printf 'run_dir=%q\n' "$run_dir"
    printf 'stdout_path="$run_dir/stdout.log"\n'
    printf 'mkdir -p "$run_dir"\n'
    printf 'echo "[start] $(date -u +%%FT%%TZ) corner_task=%s variant=dreamer_zero_reward seed=%s gpu=%s run_dir=$run_dir" | tee "$run_dir/launcher.log"\n' \
      "$short" "$seed" "$gpu"
    printf 'if env CUDA_VISIBLE_DEVICES=%q PYTHONPATH=%q PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl WANDB_PROJECT=%q python %q ' \
      "$gpu" "$DREAMER_DIR" "$WANDB_PROJECT" "$DREAMER_DIR/dreamerv3/main.py"
    printf '%q ' \
      "--configs" "dmc_vision" \
      "--task" "$dreamer_task" \
      "--seed" "$seed" \
      "--logdir" "$run_dir" \
      "--run.steps" "$NUM_TRAIN_FRAMES" \
      "--run.envs" "4" \
      "--run.eval_envs" "1" \
      "--run.eval_eps" "3" \
      "--run.log_every" "120" \
      "--run.report_every" "300" \
      "--run.eval_every" "300" \
      "--run.save_every" "900" \
      "--jax.prealloc" "False" \
      "--agent.zero_reward" "True"
    printf '> "$stdout_path" 2>&1; then\n'
    printf '  rc=0\n'
    printf 'else\n'
    printf '  rc=$?\n'
    printf 'fi\n'
    printf 'echo "[done] $(date -u +%%FT%%TZ) corner_task=%s variant=dreamer_zero_reward seed=%s gpu=%s exit=$rc" | tee -a "$run_dir/launcher.log"\n' \
      "$short" "$seed" "$gpu"
    printf 'exit "$rc"\n'
  } > "$launch_file"
  chmod +x "$launch_file"
}

write_worker_script() {
  local worker_file="$1"
  local gpu="$2"
  local queue_file="$3"
  local slot="$4"

  {
    printf '#!/usr/bin/env bash\n'
    printf 'set -uo pipefail\n'
    printf 'queue_file=%q\n' "$queue_file"
    printf 'lock_file="${queue_file}.lock"\n'
    printf 'echo "[worker-start] $(date -u +%%FT%%TZ) gpu=%s slot=%s queue=$queue_file"\n' "$gpu" "$slot"
    printf 'while true; do\n'
    printf '  line=""\n'
    printf '  {\n'
    printf '    flock -x 200\n'
    printf '    if [[ -s "$queue_file" ]]; then\n'
    printf '      line="$(head -n 1 "$queue_file")"\n'
    printf '      tail -n +2 "$queue_file" > "${queue_file}.tmp"\n'
    printf '      mv "${queue_file}.tmp" "$queue_file"\n'
    printf '    fi\n'
    printf '  } 200>"$lock_file"\n'
    printf '  if [[ -z "$line" ]]; then\n'
    printf '    echo "[worker-done] $(date -u +%%FT%%TZ) gpu=%s slot=%s"\n' "$gpu" "$slot"
    printf '    exit 0\n'
    printf '  fi\n'
    printf '  IFS=$'"'"'\\t'"'"' read -r session launch_file group_log <<< "$line"\n'
    printf '  echo "[job-start] $(date -u +%%FT%%TZ) $session"\n'
    printf '  if bash "$launch_file" > "$group_log" 2>&1; then\n'
    printf '    echo "[job-ok] $(date -u +%%FT%%TZ) $session"\n'
    printf '  else\n'
    printf '    rc=$?\n'
    printf '    echo "[job-failed] $(date -u +%%FT%%TZ) $session exit=$rc"\n'
    printf '  fi\n'
    printf 'done\n'
  } > "$worker_file"
  chmod +x "$worker_file"
}

declare -A SLOT_CAP=()
declare -A ASSIGNED_COUNT=()
for i in "${!OTHER_GPUS_ARRAY[@]}"; do
  gpu="${OTHER_GPUS_ARRAY[$i]}"
  SLOT_CAP["$gpu"]="${OTHER_GPU_SLOTS_ARRAY[$i]}"
  ASSIGNED_COUNT["$gpu"]=0
  : > "$LAUNCH_ROOT/$TIMESTAMP/gpu${gpu}.queue"
done

CHOSEN_OTHER_GPU=""

choose_other_gpu() {
  local best="" best_count=0 best_cap=1
  for gpu in "${OTHER_GPUS_ARRAY[@]}"; do
    local count="${ASSIGNED_COUNT[$gpu]}"
    local cap="${SLOT_CAP[$gpu]}"
    if [[ -z "$best" || $((count * best_cap)) -lt $((best_count * cap)) ]]; then
      best="$gpu"
      best_count="$count"
      best_cap="$cap"
    fi
  done
  ASSIGNED_COUNT["$best"]=$((ASSIGNED_COUNT["$best"] + 1))
  CHOSEN_OTHER_GPU="$best"
}

enqueue_fb() {
  local gpu="$1"
  local task="$2"
  local seed="$3"
  local variant="$4"

  local short session run_dir launch_file group_log queue_file
  short="$(task_short "$task")"
  session="subcorner_${TIMESTAMP}_${short}_${variant}_s${seed}_g${gpu}"
  run_dir="$RUNS_DIR/${TIMESTAMP}_${short}_seed${seed}_${variant}_cuda${gpu}"
  launch_file="$LAUNCH_ROOT/$TIMESTAMP/${session}.sh"
  group_log="$LAUNCH_ROOT/$TIMESTAMP/${session}.log"
  queue_file="$LAUNCH_ROOT/$TIMESTAMP/gpu${gpu}.queue"
  write_fb_script "$launch_file" "$gpu" "$task" "$seed" "$variant" "$run_dir"
  printf '%s\t%s\t%s\n' "$session" "$launch_file" "$group_log" >> "$queue_file"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$session" "$gpu" "$task" "$seed" "$variant" "$run_dir" >> "$LAUNCH_ROOT/$TIMESTAMP/jobs.tsv"
}

enqueue_dreamer() {
  local gpu="$1"
  local task="$2"
  local seed="$3"

  local short variant session run_dir launch_file group_log queue_file
  short="$(task_short "$task")"
  variant="dreamer_zero_reward"
  session="subcorner_${TIMESTAMP}_${short}_${variant}_s${seed}_g${gpu}"
  run_dir="$DREAMER_RUNS_DIR/${TIMESTAMP}_${short}_seed${seed}_${variant}_cuda${gpu}"
  launch_file="$LAUNCH_ROOT/$TIMESTAMP/${session}.sh"
  group_log="$LAUNCH_ROOT/$TIMESTAMP/${session}.log"
  queue_file="$LAUNCH_ROOT/$TIMESTAMP/gpu${gpu}.queue"
  write_dreamer_script "$launch_file" "$gpu" "$task" "$seed" "$run_dir"
  printf '%s\t%s\t%s\n' "$session" "$launch_file" "$group_log" >> "$queue_file"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$session" "$gpu" "$task" "$seed" "$variant" "$run_dir" >> "$LAUNCH_ROOT/$TIMESTAMP/jobs.tsv"
}

launch_direct_fb() {
  local gpu="$1"
  local task="$2"
  local seed="$3"
  local variant="$4"

  local short session run_dir launch_file group_log
  short="$(task_short "$task")"
  session="subcorner_${TIMESTAMP}_${short}_${variant}_s${seed}_g${gpu}"
  run_dir="$RUNS_DIR/${TIMESTAMP}_${short}_seed${seed}_${variant}_cuda${gpu}"
  launch_file="$LAUNCH_ROOT/$TIMESTAMP/${session}.sh"
  group_log="$LAUNCH_ROOT/$TIMESTAMP/${session}.log"
  write_fb_script "$launch_file" "$gpu" "$task" "$seed" "$variant" "$run_dir"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$session" "$gpu" "$task" "$seed" "$variant" "$run_dir" >> "$LAUNCH_ROOT/$TIMESTAMP/jobs.tsv"
  tmux new-session -d -s "$session" "bash $(printf '%q' "$launch_file") > $(printf '%q' "$group_log") 2>&1"
}

: > "$LAUNCH_ROOT/$TIMESTAMP/jobs.tsv"
printf 'session\tgpu\ttask\tseed\tvariant\trun_dir\n' >> "$LAUNCH_ROOT/$TIMESTAMP/jobs.tsv"

# ViT: one seed, one task per requested GPU, started immediately.
for i in "${!TASKS_ARRAY[@]}"; do
  launch_direct_fb "${VIT_GPUS_ARRAY[$i]}" "${TASKS_ARRAY[$i]}" "$ONE_SEED" "vit"
done

# All non-ViT one-seed jobs plus dino_cls_linear seed 1992 go to cuda 0/3/7.
ONE_SEED_OTHER_VARIANTS=(
  dino_patch_mlp
  dino_patch_linear
  dino_cls_mlp
  dino_cls_no_adapter
  cnn
  dreamer_zero_reward
  dino_cls_linear
)

for variant in "${ONE_SEED_OTHER_VARIANTS[@]}"; do
  for task in "${TASKS_ARRAY[@]}"; do
    choose_other_gpu
    gpu="$CHOSEN_OTHER_GPU"
    if [[ "$variant" == "dreamer_zero_reward" ]]; then
      enqueue_dreamer "$gpu" "$task" "$ONE_SEED"
    else
      enqueue_fb "$gpu" "$task" "$ONE_SEED" "$variant"
    fi
  done
done

# Remaining dino_cls_linear seeds are queued behind the first wave.
for seed in "${CLS_LINEAR_SEEDS_ARRAY[@]}"; do
  if [[ "$seed" == "$ONE_SEED" ]]; then
    continue
  fi
  for task in "${TASKS_ARRAY[@]}"; do
    choose_other_gpu
    gpu="$CHOSEN_OTHER_GPU"
    enqueue_fb "$gpu" "$task" "$seed" "dino_cls_linear"
  done
done

for gpu in "${OTHER_GPUS_ARRAY[@]}"; do
  slots="${SLOT_CAP[$gpu]}"
  queue_file="$LAUNCH_ROOT/$TIMESTAMP/gpu${gpu}.queue"
  for slot in $(seq 1 "$slots"); do
    worker_session="subcorner_${TIMESTAMP}_gpu${gpu}_slot${slot}"
    worker_file="$LAUNCH_ROOT/$TIMESTAMP/${worker_session}.sh"
    worker_log="$LAUNCH_ROOT/$TIMESTAMP/${worker_session}.log"
    write_worker_script "$worker_file" "$gpu" "$queue_file" "$slot"
    tmux new-session -d -s "$worker_session" "bash $(printf '%q' "$worker_file") > $(printf '%q' "$worker_log") 2>&1"
  done
done

echo "Launched submission corner parallel jobs under $LAUNCH_ROOT/$TIMESTAMP"
echo "Job table: $LAUNCH_ROOT/$TIMESTAMP/jobs.tsv"
