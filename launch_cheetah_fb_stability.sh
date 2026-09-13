#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/data/fanfeng/controllable_agent}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-$REPO_DIR/url_benchmark/pretrain.py}"
RUNS_DIR="${RUNS_DIR:-/mnt/data_7tb/fanfeng/controllable_agent_runs}"
CKPT_ROOT="${CKPT_ROOT:-/mnt/data_7tb/fanfeng/controallable_agent_ckpt/cheetah_fb_stability}"
LAUNCH_ROOT="${LAUNCH_ROOT:-$RUNS_DIR/launch_queues}"
RUN_TAG="${RUN_TAG:-20260817_cheetah_fb_stability_paramgrad}"
WANDB_ENTITY="${WANDB_ENTITY:-lmu_rl}"
WANDB_PROJECT="${WANDB_PROJECT:-controllable_agent_cheetah_fb_stability}"
WANDB_ID_SUFFIX="${WANDB_ID_SUFFIX:--paramgrad}"
SEED=1
DRY_RUN=0

if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
elif [[ $# -ne 0 ]]; then
  echo "Usage: $0 [--dry-run]" >&2
  exit 2
fi

read -r -a GPUS_ARRAY <<< "${GPUS:-1 4 6 7}"
if [[ "${#GPUS_ARRAY[@]}" -ne 4 ]]; then
  echo "GPUS must contain exactly four GPU ordinals, one per Cheetah task." >&2
  exit 2
fi
for gpu in "${GPUS_ARRAY[@]}"; do
  if [[ ! "$gpu" =~ ^[0-9]+$ ]]; then
    echo "Invalid GPU ordinal: $gpu" >&2
    exit 2
  fi
done

TASKS=(
  cheetah_walk
  cheetah_run
  cheetah_walk_backward
  cheetah_run_backward
)
VARIANTS=(dino_cls1 dino_cls3)
STACKS=(1 3)

launch_dir="$LAUNCH_ROOT/$RUN_TAG"
if [[ -e "$launch_dir" || -L "$launch_dir" ]]; then
  echo "Fresh launch refused: launch directory already exists: $launch_dir" >&2
  exit 1
fi

# Resolve and validate every target before creating any directory. The training
# process also receives auto_resume=False and both explicit load paths as null.
for variant_index in "${!VARIANTS[@]}"; do
  variant="${VARIANTS[$variant_index]}"
  for task in "${TASKS[@]}"; do
    run_name="${variant}_${task}_seed${SEED}"
    run_dir="$RUNS_DIR/$RUN_TAG/$run_name"
    checkpoint_dir="$CKPT_ROOT/$run_name"
    for target in "$run_dir" "$checkpoint_dir"; do
      if [[ -e "$target" || -L "$target" ]]; then
        echo "Fresh launch refused: target already exists: $target" >&2
        exit 1
      fi
    done
  done
done

mkdir -p "$launch_dir"
manifest="$launch_dir/manifest.tsv"
printf 'session\tgpu\ttask\tvariant\tdino_frame_stack\traw_input_dim\trun_name\twandb_run_id\twandb_url\trun_dir\tcheckpoint_dir\tjob_file\tstdout_log\n' > "$manifest"

write_job() {
  local job_file="$1"
  local gpu="$2"
  local task="$3"
  local variant="$4"
  local dino_frame_stack="$5"
  local run_name="$6"
  local run_id="$7"
  local run_dir="$8"

  {
    printf '#!/usr/bin/env bash\n'
    printf 'set -euo pipefail\n'
    printf 'cd %q\n' "$REPO_DIR"
    printf 'run_dir=%q\n' "$run_dir"
    printf 'mkdir -p "$run_dir"\n'
    printf 'echo "[start] $(date -u +%%FT%%TZ) run=%s gpu=%s dino_frame_stack=%s auto_resume=false load_model=null load_replay_buffer=null" | tee "$run_dir/launcher.log"\n' \
      "$run_name" "$gpu" "$dino_frame_stack"
    printf 'if env CUDA_VISIBLE_DEVICES=%q PYTHONUNBUFFERED=1 WANDB_ENTITY=%q WANDB_PROJECT=%q WANDB_RUN_NAME=%q WANDB_RUN_ID=%q WANDB_RESUME=never python %q ' \
      "$gpu" "$WANDB_ENTITY" "$WANDB_PROJECT" "$run_name" "$run_id" "$TRAIN_SCRIPT"
    printf '%q ' \
      "agent=fb_ddpg" \
      "device=cuda" \
      "use_wandb=True" \
      "use_tb=False" \
      "use_hiplog=False" \
      "save_video=False" \
      "save_train_video=False" \
      "save_replay_buffer_in_checkpoint=False" \
      "auto_resume=False" \
      "load_model=null" \
      "load_replay_buffer=null" \
      "checkpoint_root=$CKPT_ROOT" \
      "checkpoint_every=100000" \
      "obs_type=dino" \
      "dino_model_name=facebook/dinov2-base" \
      "use_cls=True" \
      "frame_stack=3" \
      "dino_frame_stack=$dino_frame_stack" \
      "render_shape=[224,224]" \
      "action_repeat=2" \
      "goal_space=null" \
      "append_goal_to_observation=False" \
      "discount=0.99" \
      "future=0.99" \
      "agent.dino_use_adapter=True" \
      "agent.dino_adapter_type=linear" \
      "agent.dino_adapter_output_dim=512" \
      "agent.dino_separate_backward_adapter=False" \
      "agent.backward_encoder_grad_scale=1.0" \
      "agent.lr=0.0001" \
      "agent.lr_coef=1.0" \
      "agent.fb_target_tau=0.01" \
      "agent.batch_size=1024" \
      "agent.update_every_steps=2" \
      "agent.num_inference_steps=5120" \
      "agent.hidden_dim=1024" \
      "agent.backward_hidden_dim=526" \
      "agent.feature_dim=512" \
      "agent.z_dim=50" \
      "agent.stddev_schedule=0.2" \
      "agent.stddev_clip=0.3" \
      "agent.update_z_every_step=300" \
      "agent.update_z_proba=1.0" \
      "agent.nstep=1" \
      "agent.ortho_coef=1.0" \
      "agent.future_ratio=0.0" \
      "agent.mix_ratio=0.5" \
      "agent.rand_weight=False" \
      "agent.preprocess=True" \
      "agent.norm_z=True" \
      "agent.q_loss=False" \
      "agent.q_loss_coef=0.01" \
      "agent.boltzmann=False" \
      "agent.add_trunk=False" \
      "agent.idm_coef=0.0" \
      "agent.idm_lr=null" \
      "update_encoder=True" \
      "num_seed_frames=4000" \
      "replay_buffer_episodes=5000" \
      "num_train_frames=2000010" \
      "eval_every_frames=10000" \
      "num_eval_episodes=10" \
      "final_tests=10" \
      "experiment=cheetah_fb_stability" \
      "task=$task" \
      "seed=$SEED" \
      "hydra.run.dir=$run_dir"
    printf '> "$run_dir/stdout.log" 2>&1; then\n'
    printf '  rc=0\n'
    printf 'else\n'
    printf '  rc=$?\n'
    printf 'fi\n'
    printf 'echo "[done] $(date -u +%%FT%%TZ) run=%s exit=$rc" | tee -a "$run_dir/launcher.log"\n' "$run_name"
    printf 'exit "$rc"\n'
  } > "$job_file"
  chmod +x "$job_file"
}

for variant_index in "${!VARIANTS[@]}"; do
  variant="${VARIANTS[$variant_index]}"
  dino_frame_stack="${STACKS[$variant_index]}"
  raw_input_dim=$((768 * dino_frame_stack))
  for task_index in "${!TASKS[@]}"; do
    task="${TASKS[$task_index]}"
    gpu="${GPUS_ARRAY[$task_index]}"
    run_name="${variant}_${task}_seed${SEED}"
    run_id="${run_name}${WANDB_ID_SUFFIX}"
    run_dir="$RUNS_DIR/$RUN_TAG/$run_name"
    checkpoint_dir="$CKPT_ROOT/$run_name"
    session="cfbstab_${variant}_${task}"
    job_file="$launch_dir/job_${run_name}.sh"
    stdout_log="$run_dir/stdout.log"
    wandb_url="https://wandb.ai/$WANDB_ENTITY/$WANDB_PROJECT/runs/$run_id"

    write_job "$job_file" "$gpu" "$task" "$variant" "$dino_frame_stack" "$run_name" "$run_id" "$run_dir"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$session" "$gpu" "$task" "$variant" "$dino_frame_stack" "$raw_input_dim" \
      "$run_name" "$run_id" "$wandb_url" "$run_dir" "$checkpoint_dir" "$job_file" "$stdout_log" \
      >> "$manifest"

    if [[ "$DRY_RUN" -eq 0 ]]; then
      if tmux has-session -t "$session" >/dev/null 2>&1; then
        echo "Fresh launch refused: tmux session already exists: $session" >&2
        exit 1
      fi
      tmux new-session -d -s "$session" "bash $(printf '%q' "$job_file")"
    fi
  done
done

echo "Manifest: $manifest"
if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "Dry run only; no tmux sessions were launched."
fi
