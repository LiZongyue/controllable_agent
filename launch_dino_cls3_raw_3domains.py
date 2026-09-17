#!/usr/bin/env python3
"""Train exactly three Raw DINO CLS3 FB agents, each evaluated on four rewards.

Frozen DINOv2-base produces 2304 raw CLS features (three frames), with no
adapter or encoder optimization. Walker/quadruped B use state goals; Cheetah
B uses the raw visual features. GPU order is Walker, Cheetah, Quadruped.
Use --start-immediately to share assigned GPUs without waiting for idle GPUs.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

REPO = Path(__file__).resolve().parent
DEFAULT_PYTHON = Path("/data/fan2/env/miniconda3/envs/occ_rlu/bin/python")
DOMAIN_SPECS = (
    ("walker", "simplified_walker", 3, ("stand", "walk", "run", "flip")),
    ("cheetah", "null", 2304, ("walk", "run", "walk_backward", "run_backward")),
    ("quadruped", "simplified_quadruped", 2, ("stand", "walk", "run", "jump")),
)
SOURCE_FILES = (
    "url_benchmark/pretrain.py", "url_benchmark/agent/ddpg.py",
    "url_benchmark/agent/fb_ddpg.py", "url_benchmark/agent/fb_modules.py",
    "url_benchmark/in_memory_replay_buffer.py", "url_benchmark/dmc.py",
    "url_benchmark/goals.py", "url_benchmark/base_config.yaml",
    "launch_dino_cls3_raw_3domains.py", "scripts/run_dino_cls3_raw_domain.py",
)


def positive_int(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def positive_float(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return value


def parser():
    result = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    result.add_argument("--gpus", type=int, nargs=3, default=[4, 5, 7], metavar=("WALKER", "CHEETAH", "QUADRUPED"))
    result.add_argument("--start-immediately", action="store_true",
                        help="skip GPU idle checks and cross-campaign locks; fail if free VRAM is insufficient")
    result.add_argument("--dry-run", action="store_true")
    result.add_argument("--timestamp", default=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
    result.add_argument("--python", default=str(DEFAULT_PYTHON if DEFAULT_PYTHON.exists() else sys.executable))
    result.add_argument("--train-script", type=Path, default=REPO / "scripts/run_dino_cls3_raw_domain.py")
    result.add_argument("--runs-dir", type=Path, default=REPO / "training_runs/dino_cls3_raw_3domains")
    result.add_argument("--checkpoint-root", type=Path, default=REPO / "checkpoints/dino_cls3_raw_3domains")
    result.add_argument("--launch-root", type=Path, default=REPO / "launches/dino_cls3_raw_3domains")
    result.add_argument("--gpu-lock-dir", type=Path, default=REPO / "launches/cnn_224_3domains/.gpu_locks")
    result.add_argument("--seed", type=int, default=1)
    result.add_argument("--lr", type=positive_float, default=0.0001)
    for name, default in (
        ("batch-size", 128), ("num-train-frames", 2000010), ("eval-every-frames", 10000),
        ("num-eval-episodes", 10), ("final-tests", 10), ("num-inference-steps", 5120),
        ("replay-buffer-episodes", 5000), ("checkpoint-every", 100000), ("cpu-threads", 1),
        ("wait-seconds", 30), ("min-free-mb", 20000), ("max-idle-used-mb", 1024),
        ("ready-checks", 2), ("min-free-disk-gib", 14),
    ):
        result.add_argument("--" + name, type=positive_int, default=default)
    result.add_argument("--max-gpu-util", type=int, choices=range(101), default=5)
    result.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    result.add_argument("--wandb-entity", default="lmu_rl")
    result.add_argument("--wandb-project", default="controllable_agent_baseline")
    return result


def overrides(args, row):
    values = [
        "agent=fb_ddpg", "device=cuda", "obs_type=dino", "dino_model_name=facebook/dinov2-base",
        "use_cls=True", "dino_frame_stack=3", "frame_stack=3", "render_shape=[224,224]",
        "action_repeat=2", "goal_space=" + row["goal_space"], "reward_free=True",
        "update_encoder=False", "append_goal_to_observation=False", "agent.dino_use_adapter=False",
        "agent.dino_separate_fb_adapters=False", "agent.dino_separate_backward_adapter=False",
        "agent.pixel_separate_fb_encoders=False", "agent.dino_flare_b=False", "agent.idm_coef=0",
        "agent.batch_size=" + str(args.batch_size), "agent.lr_coef=1.0", "agent.lr_f=null", "agent.lr_b=null",
        "agent.update_every_steps=2", "agent.ortho_coef=1.0", "agent.mix_ratio=0.5", "agent.fb_target_tau=0.01",
        "agent.num_inference_steps=" + str(args.num_inference_steps),
        "replay_buffer_episodes=" + str(args.replay_buffer_episodes),
        "task=" + row["task"], "eval_tasks=" + row["eval_tasks"], "seed=" + str(args.seed),
        "num_seed_frames=4000", "num_train_frames=" + str(args.num_train_frames),
        "eval_every_frames=" + str(args.eval_every_frames), "num_eval_episodes=" + str(args.num_eval_episodes),
        "final_tests=" + str(args.final_tests), "auto_resume=False", "load_model=null", "load_replay_buffer=null",
        "checkpoint_root=" + str(args.checkpoint_root), "checkpoint_every=" + str(args.checkpoint_every),
        "snapshot_at=[]", "save_replay_buffer_in_checkpoint=False", "use_wandb=True", "use_tb=False",
        "use_hiplog=False", "save_video=False", "save_train_video=False",
        "experiment=dino_cls3_raw_" + row["domain"] + "_seed" + str(args.seed), "hydra.run.dir=" + row["run_dir"],
    ]
    values.extend("agent." + key + "=" + str(args.lr) for key in ("lr", "fb_lr", "lr_actor"))
    return values


GPU_GUARD = r'''
if [[ "$start_immediately" == 1 ]]; then
  free_mb="$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits)"
  free_mb="${free_mb//[[:space:]]/}"
  if [[ ! "$free_mb" =~ ^[0-9]+$ ]] || (( free_mb < min_free_mb )); then
    echo "[refused] immediate GPU $gpu free_mb=$free_mb requires $min_free_mb MiB" >&2
    exit 1
  fi
  echo "[gpu-immediate] $(date -u +%FT%TZ) gpu=$gpu free_mb=$free_mb shared GPU start"
else
mkdir -p "$lock_dir"
exec 9>"$lock_dir/gpu${gpu}.lock"
echo "[queue] $(date -u +%FT%TZ) task=$task gpu=$gpu waiting for shared GPU lock"
flock 9
ready=0
while (( ready < ready_checks )); do
  free_mb=unavailable
  used_mb=unavailable
  gpu_util=unavailable
  if sample="$(nvidia-smi -i "$gpu" --query-gpu=memory.free,memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null)"; then
    IFS=',' read -r free_mb used_mb gpu_util <<< "$sample"
    free_mb="${free_mb//[[:space:]]/}"
    used_mb="${used_mb//[[:space:]]/}"
    gpu_util="${gpu_util//[[:space:]]/}"
  fi
  if [[ "$free_mb" =~ ^[0-9]+$ && "$used_mb" =~ ^[0-9]+$ && "$gpu_util" =~ ^[0-9]+$ ]] \
      && (( free_mb >= min_free_mb && used_mb <= max_idle_used_mb && gpu_util <= max_gpu_util )); then
    ready=$((ready + 1))
    echo "[gpu-ready] $(date -u +%FT%TZ) gpu=$gpu free_mb=$free_mb used_mb=$used_mb util=$gpu_util checks=$ready/$ready_checks"
  else
    ready=0
    echo "[gpu-wait] $(date -u +%FT%TZ) gpu=$gpu free_mb=$free_mb used_mb=$used_mb util=$gpu_util"
  fi
  (( ready >= ready_checks )) || sleep "$wait_seconds"
done
fi
sha256sum --check --status "$source_hashes" || {
  echo "[refused] source changed while queued; regenerate launch" >&2; exit 1;
}
for target in "$run_dir" "$ckpt_dir"; do
  [[ ! -e "$target" && ! -L "$target" ]] || { echo "[refused] existing target: $target" >&2; exit 1; }
done
mkdir -p "$run_dir/matplotlib"
'''


def worker(args, row, launch_dir):
    values = dict(
        gpu=row["gpu"], task=row["task"], run_dir=row["run_dir"], ckpt_dir=row["checkpoint_dir"],
        lock_dir=args.gpu_lock_dir, source_hashes=launch_dir / "source.sha256",
        wait_seconds=args.wait_seconds, min_free_mb=args.min_free_mb,
        max_gpu_util=args.max_gpu_util, max_idle_used_mb=args.max_idle_used_mb,
        ready_checks=args.ready_checks,
        start_immediately=int(args.start_immediately),
    )
    env = dict(
        CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(row["gpu"]),
        MUJOCO_GL="egl", MUJOCO_EGL_DEVICE_ID=str(row["gpu"]), PYTHONUNBUFFERED="1",
        OMP_NUM_THREADS=str(args.cpu_threads), MKL_NUM_THREADS=str(args.cpu_threads),
        OPENBLAS_NUM_THREADS=str(args.cpu_threads), NUMEXPR_NUM_THREADS=str(args.cpu_threads),
        VECLIB_MAXIMUM_THREADS=str(args.cpu_threads), TOKENIZERS_PARALLELISM="false",
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
        WANDB_ENTITY=args.wandb_entity, WANDB_PROJECT=args.wandb_project,
        WANDB_MODE=args.wandb_mode, WANDB_RUN_ID=row["run_name"],
        WANDB_RUN_NAME=row["run_name"], WANDB_RESUME="never",
    )
    preflight = (
        'import json, torch; assert torch.cuda.is_available(), "CUDA unavailable; refusing CPU fallback"; '
        'assert torch.cuda.device_count() == 1, "Expected one visible GPU"; '
        'print(json.dumps({"device": torch.cuda.get_device_name(0), "torch": torch.__version__, '
        '"cpu_threads": torch.get_num_threads()}))'
    )
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", "cd " + shlex.quote(str(REPO))]
    lines.extend(key + "=" + shlex.quote(str(value)) for key, value in values.items())
    lines.append(GPU_GUARD)
    lines.extend("export " + key + "=" + shlex.quote(value) for key, value in env.items())
    lines.append('export MPLCONFIGDIR="$run_dir/matplotlib"')
    lines.append(shlex.join([args.python, "-c", preflight]) + ' > "$run_dir/device.json"')
    lines.append('echo "[start] $(date -u +%FT%TZ) task=$task gpu=$gpu" | tee "$run_dir/launcher.log"')
    lines.append("if " + shlex.join(row["command"]) + ' > "$run_dir/stdout.log" 2>&1; then rc=0; else rc=$?; fi')
    lines.append('echo "[done] $(date -u +%FT%TZ) task=$task gpu=$gpu exit=$rc" | tee -a "$run_dir/launcher.log"')
    lines.append('printf "%s\\n" "$rc" > "$run_dir/exit_code"')
    lines.append('exit "$rc"')
    return "\n".join(lines) + "\n"


def write_script(path, content):
    path.write_text(content)
    path.chmod(0o755)
    subprocess.run(["bash", "-n", str(path)], check=True)


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    if args.seed < 0 or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.timestamp):
        cli.error("seed must be nonnegative and timestamp must be a simple filename")
    if len(set(args.gpus)) != 3 or min(args.gpus) < 0:
        cli.error("exactly three distinct nonnegative GPU ordinals are required")
    if not shutil.which(args.python) or not args.train_script.is_file():
        cli.error("Python interpreter or training script is unavailable")
    for key in ("runs_dir", "checkpoint_root", "launch_root", "gpu_lock_dir", "train_script"):
        setattr(args, key, getattr(args, key).expanduser().resolve())
    launch_dir = args.launch_root / args.timestamp
    if launch_dir.exists() or launch_dir.is_symlink():
        cli.error("fresh launch refused; launch directory exists: " + str(launch_dir))
    rows = []
    for gpu, (domain, goal_space, backward_dim, tasks) in zip(args.gpus, DOMAIN_SPECS):
        run_name = "{}_{}_raw_seed{}".format(args.timestamp, domain, args.seed)
        run_dir = args.runs_dir / run_name
        checkpoint_dir = args.checkpoint_root / run_name
        if any(path.exists() or path.is_symlink() for path in (run_dir, checkpoint_dir)):
            cli.error("fresh launch refused; run or checkpoint directory exists: " + run_name)
        row = dict(
            domain=domain, task=domain + "_walk", eval_tasks="[" + ",".join(domain + "_" + task for task in tasks) + "]",
            goal_space=goal_space, obs_type="dino", dino_model_name="facebook/dinov2-base", dino_frame_stack=3,
            raw_feature_dim=2304, backward_input_dim=backward_dim, dino_use_adapter=False,
            update_encoder=False, batch_size=args.batch_size, lr=args.lr, gpu=gpu, seed=args.seed,
            start_immediately=args.start_immediately, run_name=run_name, run_dir=str(run_dir),
            checkpoint_dir=str(checkpoint_dir), job_file=str(launch_dir / "jobs" / (domain + ".sh")),
            session="draw_{}_{}_g{}".format(args.timestamp.replace(".", "_"), domain, gpu),
            controller_log=str(launch_dir / (domain + ".controller.log")),
            wandb_mode=args.wandb_mode, wandb_project=args.wandb_project, wandb_entity=args.wandb_entity,
        )
        row["command"] = [args.python, str(args.train_script)] + overrides(args, row)
        rows.append(row)
    if not args.dry_run:
        for output_root in (args.runs_dir, args.checkpoint_root, args.launch_root):
            probe = output_root
            while not probe.exists():
                probe = probe.parent
            if shutil.disk_usage(probe).free < args.min_free_disk_gib * 1024 ** 3:
                cli.error("insufficient free disk space at {}: require {} GiB".format(output_root, args.min_free_disk_gib))
        for command in ("tmux", "nvidia-smi", "flock", "sha256sum"):
            if not shutil.which(command):
                cli.error(command + " is required")
        for row in rows:
            subprocess.run(["nvidia-smi", "-i", str(row["gpu"]), "--query-gpu=name", "--format=csv,noheader"], check=True)
            if subprocess.run(["tmux", "has-session", "-t", row["session"]], capture_output=True).returncode == 0:
                cli.error("tmux session already exists: " + row["session"])
    launch_dir.mkdir(parents=True, exist_ok=False)
    (launch_dir / "jobs").mkdir()
    hashes = [hashlib.sha256((REPO / name).read_bytes()).hexdigest() + "  " + name for name in SOURCE_FILES]
    (launch_dir / "source.sha256").write_text("\n".join(hashes) + "\n")
    for row in rows:
        write_script(Path(row["job_file"]), worker(args, row, launch_dir))
    fields = [key for key in rows[0] if key != "command"]
    with (launch_dir / "manifest.tsv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    (launch_dir / "manifest.json").write_text(json.dumps(dict(jobs=rows), indent=2) + "\n")
    print("Manifest: " + str(launch_dir / "manifest.tsv"))
    print("Prepared exactly 3 Raw DINO CLS3 training jobs, each evaluating 4 same-domain tasks.")
    if args.dry_run:
        print("Dry run only; no controllers or training processes started.")
        return 0
    for row in rows:
        command = "bash " + shlex.quote(row["job_file"]) + " > " + shlex.quote(row["controller_log"]) + " 2>&1"
        subprocess.run(["tmux", "new-session", "-d", "-s", row["session"], command], check=True)
        print("Started {} controller on GPU {}: {}".format(row["domain"], row["gpu"], row["session"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
