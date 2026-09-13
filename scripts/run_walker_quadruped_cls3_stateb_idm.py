#!/usr/bin/env python3
"""Resource limits and read-only audits around the existing state-B launcher.

Accepts the original launcher's Hydra arguments. No training or FB update
method is replaced; the only logging addition is an append-only eval record.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def main():
    repo = Path(__file__).resolve().parents[1]
    manifest_path = Path(os.environ['WQ_STATEB_MANIFEST']).resolve()
    manifest = json.loads(manifest_path.read_text())
    matches = [r for r in manifest['runs'] if r['run_id'] == os.environ['WANDB_RUN_ID']]
    assert len(matches) == 1, 'Unknown or duplicate run identity'
    row = matches[0]
    assert sys.argv[1:] == row['launcher_overrides'], 'Launcher arguments changed after audit'
    assert manifest['source_hashes'] and not manifest.get('source_hashes_pending', False)
    for relative, digest in manifest['source_hashes'].items():
        assert hashlib.sha256((repo / relative).read_bytes()).hexdigest() == digest, relative
    preflight = json.loads(manifest_path.with_name('preflight.json').read_text())
    assert preflight['status'] == 'PASS' and len(preflight['runs']) == 16
    cpus = set(row['cpus'])
    assert len(cpus) == 4 and cpus <= os.sched_getaffinity(0)
    # This must precede torch/numpy/transformers imports and native pool creation.
    os.sched_setaffinity(0, cpus)
    os.environ.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                      OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
                      VECLIB_MAXIMUM_THREADS='1', TOKENIZERS_PARALLELISM='false',
                      MUJOCO_GL='egl', MUJOCO_EGL_DEVICE_ID=str(row['gpu']),
                      WANDB_ENTITY=manifest['wandb_entity'],
                      WANDB_PROJECT=manifest['wandb_project'],
                      WANDB_RUN_NAME=row['run_name'], WANDB_RESUME='never',
                      WANDB_MODE='online', WANDB_DIR=row['run_dir'],
                      MPLCONFIGDIR=str(Path(row['run_dir']) / 'matplotlib'))
    assert os.environ['CUDA_VISIBLE_DEVICES'] == str(row['gpu'])
    run_dir = Path(row['run_dir'])
    checkpoint_dir = Path(row['checkpoint_dir'])
    # The original launcher creates the log directory before invoking Python.
    assert run_dir.is_dir() and not (run_dir / '.hydra').exists()
    assert not checkpoint_dir.exists() and not checkpoint_dir.is_symlink()
    with (run_dir / 'fresh_start_claim.json').open('x') as stream:
        json.dump({'run_id': row['run_id'], 'pid': os.getpid()}, stream)
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    write_json(run_dir / 'pid.json', dict(pid=os.getpid(), gpu=row['gpu'],
        cpus=sorted(cpus), session=row['session'], run_id=row['run_id'],
        started_at=datetime.now(timezone.utc).isoformat()))
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    print(f"Fresh state-B run {row['run_id']}; GPU {row['gpu']}; CPUs {sorted(cpus)}; threads=1", flush=True)
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / 'scripts'))
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    assert torch.cuda.is_available(), 'CUDA required; refusing CPU fallback'
    from omegaconf import OmegaConf
    import wandb
    from url_benchmark import pretrain
    from audit_walker_quadruped_cls3_stateb_idm import audit_agent

    original_init = pretrain.BaseWorkspace.__init__

    def verified_init(workspace, cfg):
        original_init(workspace, cfg)
        report = audit_agent(workspace.agent, float(row['idm_coef']))
        goal_space = 'simplified_walker' if row['task'].startswith('walker_') else 'simplified_quadruped'
        assert cfg.task == row['task'] and cfg.seed == 1
        assert cfg.goal_space == cfg.agent.goal_space == goal_space
        assert cfg.obs_type == 'dino' and cfg.use_cls and cfg.dino_frame_stack == 3
        assert tuple(workspace.train_env.observation_spec().shape) == (2304,)
        assert tuple(workspace.eval_env.observation_spec().shape) == (2304,)
        assert not cfg.append_goal_to_observation and cfg.custom_reward is None
        assert not cfg.auto_resume and cfg.load_model is None and cfg.load_replay_buffer is None
        assert workspace.global_step == 0 and len(workspace.replay_loader) == 0
        assert workspace._resume_checkpoint_filepath is None
        assert workspace._checkpoint_filepath == checkpoint_dir / 'latest.pt'
        expected = dict(num_train_frames=2000010, eval_every_frames=10000,
                        num_eval_episodes=10, checkpoint_every=100000, action_repeat=2,
                        discount=0.99, reward_free=True, update_encoder=True)
        for key, value in expected.items():
            assert getattr(cfg, key) == value, key
        frozen_models = []
        for env_name in ('train_env', 'eval_env'):
            env = getattr(workspace, env_name)
            found = False
            while env is not None:
                if isinstance(env, pretrain.dmc.DinoV3EmbedWrapper):
                    model = env._model
                    assert not model.training and all(not p.requires_grad for p in model.parameters())
                    assert model.config.model_type == 'dinov2'
                    assert cfg.dino_model_name == 'facebook/dinov2-base'
                    frozen_models.append(env_name)
                    found = True
                    break
                try:
                    env = object.__getattribute__(env, '_env')
                except AttributeError:
                    env = None
            assert found, f'Frozen DINO not found in {env_name}'
        assert os.sched_getaffinity(0) == cpus
        assert torch.get_num_threads() == torch.get_num_interop_threads() == 1
        assert wandb.run is not None and wandb.run.id == row['run_id']
        assert wandb.run.entity == manifest['wandb_entity']
        assert wandb.run.project == manifest['wandb_project']
        assert not wandb.run.settings._offline
        report.update(task=row['task'], group=row['group'], gpu=row['gpu'],
                      cpus=sorted(cpus), pid=os.getpid(), fresh_start=True,
                      frozen_dino_models=frozen_models, wandb_url=wandb.run.url,
                      code_commit=manifest['code_commit'], torch_num_threads=1,
                      torch_num_interop_threads=1, goal_space=goal_space,
                      num_train_frames=cfg.num_train_frames)
        write_json(run_dir / 'startup_audit.json', report)
        write_json(run_dir / 'resolved_config.json', OmegaConf.to_container(cfg, resolve=True))
        wandb.config.update({'startup_audit': report, 'campaign': manifest['campaign'],
                            'source_hashes': manifest['source_hashes']})
        wandb.save(str(run_dir / 'startup_audit.json'), base_path=str(run_dir), policy='now')
        print('STARTUP_AUDIT_PASS ' + json.dumps(report), flush=True)

        # Observe actual first forward inputs without drawing RNG or changing tensors.
        observed = {}
        goal_dim = 3 if goal_space == 'simplified_walker' else 2
        def watch(name, module, width):
            handle = None
            def check(_module, inputs):
                tensor = inputs[0]
                assert tensor.shape[-1] == width, (name, tuple(tensor.shape))
                observed[name] = list(tensor.shape)
                write_json(run_dir / 'live_input_audit.json', observed)
                handle.remove()
            handle = module.register_forward_pre_hook(check)
        watch('visual_adapter', workspace.agent.encoder, 2304)
        watch('F', workspace.agent.forward_net, 512)
        watch('actor', workspace.agent.actor, 512)
        watch('B_state_goal', workspace.agent.backward_net, goal_dim)

        # Preserve every scheduled eval value as written by the existing logger.
        eval_group = workspace.logger._eval_mg
        original_csv_dump = eval_group._dump_to_csv
        history_path = run_dir / 'eval_history.jsonl'
        history_path.touch(exist_ok=False)
        registered = False
        def record_eval(data):
            nonlocal registered
            original_csv_dump(data)
            with history_path.open('a') as stream:
                json.dump({'eval/' + key: value for key, value in data.items()}, stream)
                stream.write('\n')
            if not registered:
                wandb.save(str(run_dir / 'eval.csv'), base_path=str(run_dir), policy='live')
                wandb.save(str(history_path), base_path=str(run_dir), policy='live')
                registered = True
        eval_group._dump_to_csv = record_eval

    pretrain.BaseWorkspace.__init__ = verified_init
    os.chdir(repo)
    sys.argv = [str(repo / 'url_benchmark/pretrain.py'), *row['overrides']]
    pretrain.main()


if __name__ == '__main__':
    main()
