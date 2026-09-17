#!/usr/bin/env python3
"""Run the existing trainer for one bs=128 row with read-only startup assertions.

Mirrors ``run_walker_quadruped_cls3_lr.py`` exactly, with two additions: the
startup audit also asserts ``agent.batch_size == 128``, and the corresponding
bs=1024 baseline run (config/code version this row was cloned from) is
recorded into the W&B config/summary and into the local startup_audit.json
for traceability.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_walker_quadruped_cls3_bs128 import (  # noqa: E402
    EXPECTED_BATCH_SIZE, assigned_cpus, audit_agent_bs128,
)


def configure_cpu(row, index):
    """Bound native pools and inherited process affinity before numeric imports."""
    cpus = assigned_cpus(row, index)
    available = os.sched_getaffinity(0)
    assert set(cpus) <= available, f'CPUs {cpus} are unavailable; allowed CPUs: {sorted(available)}'
    os.sched_setaffinity(0, set(cpus))
    os.environ.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                      OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
                      VECLIB_MAXIMUM_THREADS='1', TOKENIZERS_PARALLELISM='false')
    return cpus


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--index', type=int, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    row = manifest['runs'][args.index]
    repo = Path(__file__).resolve().parents[1]
    for relative, expected in manifest['source_hashes'].items():
        assert hashlib.sha256((repo / relative).read_bytes()).hexdigest() == expected, relative
    cpus = configure_cpu(row, args.index)
    run_dir = Path(row['run_dir'])
    checkpoint_dir = Path(row['checkpoint_dir'])
    for target in (run_dir, checkpoint_dir):
        assert not target.exists() and not target.is_symlink(), f'Fresh-start collision: {target}'
    run_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    log = (run_dir / 'stdout.log').open('x', buffering=1)
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(row['gpu']), MUJOCO_GL='egl',
                      MUJOCO_EGL_DEVICE_ID=str(row['gpu']), PYTHONUNBUFFERED='1',
                      WANDB_ENTITY=manifest['wandb_entity'], WANDB_PROJECT=manifest['wandb_project'],
                      WANDB_RUN_ID=row['run_id'], WANDB_RUN_NAME=row['run_id'],
                      WANDB_RESUME='never', WANDB_MODE='online', WANDB_DIR=str(run_dir),
                      MPLCONFIGDIR=str(run_dir / 'matplotlib'))
    (run_dir / 'pid.json').write_text(json.dumps(dict(pid=os.getpid(), gpu=row['gpu'], cpus=cpus,
        run_id=row['run_id'], session=row['session'], started_at=datetime.now(timezone.utc).isoformat()), indent=2) + '\n')
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    print(f"Starting fresh: {row['run_id']} on GPU {row['gpu']}, CPUs {cpus}; pid={os.getpid()}; "
          f"baseline_bs1024_run_id={row['baseline_run_id']}", flush=True)
    sys.path.insert(0, str(repo))
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    assert torch.cuda.is_available(), 'CUDA is required; refusing CPU fallback'
    from url_benchmark import pretrain
    import wandb
    from omegaconf import OmegaConf

    original_init = pretrain.BaseWorkspace.__init__

    def verified_init(workspace, cfg):
        original_init(workspace, cfg)
        report = audit_agent_bs128(workspace.agent, row['lr'], row['separated'])
        assert cfg.task == row['task'] and cfg.seed == row['seed'] == 1
        assert cfg.obs_type == 'dino' and cfg.use_cls and cfg.dino_frame_stack == 3
        assert cfg.dino_model_name == 'facebook/dinov2-base' and cfg.goal_space is None
        assert not cfg.append_goal_to_observation and cfg.custom_reward is None
        assert not cfg.auto_resume and cfg.load_model is None and cfg.load_replay_buffer is None
        assert int(cfg.agent.batch_size) == EXPECTED_BATCH_SIZE
        assert int(cfg.num_train_frames) == int(row['num_train_frames'])
        assert workspace.global_step == 0 and workspace._resume_checkpoint_filepath is None
        assert workspace._checkpoint_filepath == checkpoint_dir / 'latest.pt'
        frozen_models = []
        for name in ('train_env', 'eval_env'):
            env = getattr(workspace, name)
            found = False
            while env is not None:
                if isinstance(env, pretrain.dmc.DinoV3EmbedWrapper):
                    model = env._model
                    assert all(not p.requires_grad for p in model.parameters())
                    assert not model.training and model.config.model_type == 'dinov2'
                    frozen_models.append(name)
                    found = True
                    break
                # dm_control action_scale.Wrapper stores _env in __slots__,
                # so vars(env) alone does not contain the wrapped environment.
                try:
                    env = object.__getattribute__(env, '_env')
                except AttributeError:
                    env = None
            assert found, f'DINO backbone not found in {name}'
        assert wandb.run is not None and wandb.run.id == row['run_id']
        assert not wandb.run.settings._offline, 'W&B must be online'
        assert os.sched_getaffinity(0) == set(cpus), 'Training CPU affinity changed during startup'
        assert torch.get_num_threads() == torch.get_num_interop_threads() == 1
        report.update(task=row['task'], group=row['group'], gpu=row['gpu'], cpus=cpus,
                      cpu_affinity=sorted(os.sched_getaffinity(0)),
                      torch_num_threads=torch.get_num_threads(),
                      torch_num_interop_threads=torch.get_num_interop_threads(), pid=os.getpid(),
                      wandb_url=wandb.run.url, frozen_dino_models=frozen_models,
                      fresh_start=True, global_step=workspace.global_step,
                      baseline_bs1024_run_id=row['baseline_run_id'],
                      baseline_bs1024_run_dir=row['baseline_run_dir'],
                      baseline_bs1024_wandb_url=row['baseline_wandb_url'])
        wandb.config.update({'startup_optimizer_audit': report,
                             'baseline_bs1024_run_id': row['baseline_run_id'],
                             'baseline_bs1024_wandb_url': row['baseline_wandb_url']})
        wandb.run.summary.update({'startup_verified': True, 'actual_lr_f': row['lr'],
                                  'actual_lr_b': row['lr'], 'actual_lr_actor': row['lr'],
                                  'actual_lr_adapters': row['lr'], 'gpu_physical': row['gpu'],
                                  'cpus_logical': cpus, 'torch_num_threads': 1,
                                  'torch_num_interop_threads': 1,
                                  'actual_batch_size': int(cfg.agent.batch_size),
                                  'baseline_bs1024_run_id': row['baseline_run_id']})
        (run_dir / 'startup_audit.json').write_text(json.dumps(report, indent=2) + '\n')
        (run_dir / 'resolved_config.json').write_text(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2) + '\n')
        print('STARTUP_AUDIT_PASS ' + json.dumps(report), flush=True)

    pretrain.BaseWorkspace.__init__ = verified_init
    os.chdir(repo)
    sys.argv = [str(repo / 'url_benchmark/pretrain.py'), *row['overrides']]
    pretrain.main()


if __name__ == '__main__':
    main()
