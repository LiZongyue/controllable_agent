#!/usr/bin/env python3
"""Prepare the existing launcher's 16 state-B IDM jobs; never start processes."""
from __future__ import annotations

from collections import Counter
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
CAMPAIGN = ROOT.name
PYTHON = Path('/data/fan2/env/miniconda3/envs/occ_rlu/bin/python')
LAUNCHER = REPO / 'launch_dino_cls_stack3_12tasks.sh'
RUNNER = REPO / 'scripts/run_walker_quadruped_cls3_stateb_idm.py'
RUNS = Path('/mnt/data_7tb/fanfeng/controllable_agent_runs')
CHECKPOINTS = Path('/mnt/data_nvme1/fanfeng/controllable_agent_ckpt')
TASKS = ['walker_stand', 'walker_walk', 'walker_run', 'walker_flip',
         'quadruped_stand', 'quadruped_walk', 'quadruped_run', 'quadruped_jump']
GPUS = [2, 3, 4, 5, 6, 7, 2, 3]
BASELINE_PREFIX = '20260813_idm_2m_full_coef_sweep_v3_seed1_'
BASELINE_SUFFIX = '_dino_cls_stack3_full2m_idm0p0_idmlr0p0001'
NUMERIC_ENV = dict(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                   NUMEXPR_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1',
                   TOKENIZERS_PARALLELISM='false')


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def flatten(value, prefix=''):
    result = {}
    for key, item in value.items():
        full = prefix + '.' + key if prefix else key
        if isinstance(item, dict):
            result.update(flatten(item, full))
        else:
            result[full] = item
    return result


def parse_job(path):
    line = next(line for line in path.read_text().splitlines() if line.startswith('if env '))
    tokens = shlex.split(line)
    start = tokens.index(str(RUNNER)) + 1
    stop = tokens.index('>>')
    overrides = tokens[start:stop]
    assert len(overrides) == len({arg.split('=', 1)[0] for arg in overrides})
    return overrides


def safety_overrides(goal):
    return [
        'auto_resume=False', 'load_model=null', 'load_replay_buffer=null',
        'append_goal_to_observation=False', 'agent.goal_space=' + goal,
        'agent.idm_route=none', 'agent.dino_separate_fb_adapters=False',
        'agent.dino_separate_backward_adapter=False', 'agent.dino_flare_b=False',
        'agent.pixel_separate_fb_encoders=False', 'agent.z_dim=50',
        'discount=0.99', 'agent.ortho_coef=1', 'agent.q_loss=False',
        'reward_free=True', 'checkpoint_every=100000', 'agent.num_inference_steps=5120',
    ]


def prepare():
    assert not (ROOT / 'manifest.json').exists(), 'Manifest already exists; do not overwrite it'
    os.environ.update(NUMERIC_ENV)
    # Preparation also obeys the CPU limits before importing numerical libraries.
    available = os.sched_getaffinity(0)
    os.sched_setaffinity(0, available.intersection({12, 13, 76, 77}) or {min(available)})
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    import yaml
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    sys.path.insert(0, str(REPO))
    from url_benchmark import pretrain  # noqa: F401; register existing configs only

    environment = dict(os.environ, REPO_DIR=str(REPO), TRAIN_SCRIPT=str(RUNNER),
                       RUNS_DIR=str(RUNS / CAMPAIGN), CKPT_ROOT=str(CHECKPOINTS / CAMPAIGN),
                       LAUNCH_ROOT=str(ROOT / 'launcher'), TIMESTAMP=CAMPAIGN,
                       WANDB_ENTITY='lmu_rl', WANDB_PROJECT='controllable_agent_baseline',
                       WANDB_MODE='online', SEED='1', TASKS=' '.join(TASKS),
                       GPUS='2 3 4 5 6 7', GPU_ASSIGNMENTS=' '.join(map(str, GPUS)),
                       AGENT_LR='0.0001', IDM_LR='0.0001', IDM_ENCODER_MODE='legacy',
                       STAGE='stateB', NUM_TRAIN_FRAMES='2000010', EVAL_EVERY_FRAMES='10000',
                       PARALLEL_TASKS='1', PATH=str(PYTHON.parent) + ':' + os.environ.get('PATH', ''))
    # Guard against inherited sweep settings affecting this strict comparison.
    environment.update(IDM_DIAGNOSTICS_INTERVAL='500', IDM_ENCODER_BURNIN_STEPS='0',
                       IDM_ENCODER_RAMP_STEPS='0', IDM_GRAD_RATIO_TARGET='',
                       IDM_GRAD_RATIO_EMA='0.9', IDM_COEF_MIN='0.1',
                       IDM_COEF_MAX='200.0', IDM_COEF_SLEW_RATE='2.0')
    by_key = {}
    invocations = []
    for coef, group in [('0', 'idm0'), ('0.1', 'idm0p1')]:
        group_env = dict(environment, IDM_COEF=coef)
        command = ['bash', str(LAUNCHER), '--dry-run', '--parallel']
        proc = subprocess.run(command, env=group_env, cwd=str(REPO), capture_output=True, text=True)
        (ROOT / ('dryrun_' + group + '.log')).write_text(proc.stdout + proc.stderr)
        proc.check_returncode()
        generated = next(line[len('Manifest: '):] for line in proc.stdout.splitlines()
                         if line.startswith('Manifest: '))
        invocations.append(dict(group=group, command=command,
                                environment={key: group_env[key] for key in environment
                                             if key not in os.environ or group_env[key] != os.environ[key]},
                                idm_coef=coef, manifest=generated))
        with open(generated) as stream:
            rows = list(csv.DictReader(stream, delimiter='\t'))
        assert len(rows) == 8
        for row in rows:
            assert (row['task'], group) not in by_key
            row.update(group=group, idm_coef=float(coef))
            by_key[(row['task'], group)] = row

    core_next = {2: 16, 3: 24, 4: 32, 5: 40, 6: 48, 7: 56}
    all_rows = []
    for task, gpu in zip(TASKS, GPUS):
        for group in ('idm0', 'idm0p1'):
            source = by_key[(task, group)]
            first_core = core_next[gpu]
            core_next[gpu] += 2
            cpus = [first_core, first_core + 1, first_core + 64, first_core + 65]
            original = parse_job(Path(source['job_file']))
            extra = safety_overrides(source['goal_space'])
            assert not ({arg.split('=', 1)[0] for arg in original}
                        & {arg.split('=', 1)[0] for arg in extra})
            overrides = original + extra
            with initialize_config_dir(config_dir=str(REPO / 'url_benchmark'), version_base='1.1'):
                cfg = compose(config_name='base_config', overrides=overrides)
            resolved = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)
            baseline_dir = RUNS / (BASELINE_PREFIX + task + BASELINE_SUFFIX)
            baseline = OmegaConf.to_container(OmegaConf.load(baseline_dir / '.hydra/config.yaml'),
                                             resolve=True, throw_on_missing=False)
            before, after = flatten(baseline), flatten(resolved)
            differences = {key: dict(baseline=before[key], new=after[key])
                           for key in before.keys() & after.keys() if before[key] != after[key]}
            assert set(differences) <= {'experiment', 'checkpoint_root', 'agent.idm_coef'}, differences
            assert cfg.goal_space == cfg.agent.goal_space == source['goal_space']
            assert not cfg.append_goal_to_observation and cfg.agent.idm_route == 'none'
            assert cfg.agent.idm_encoder_mode == 'legacy' and cfg.agent.idm_lr == 1e-4
            assert cfg.agent.batch_size == 1024 and cfg.batch_size == 1024
            assert cfg.agent.lr == 1e-4 and cfg.agent.z_dim == 50
            assert cfg.obs_type == 'dino' and cfg.dino_model_name == 'facebook/dinov2-base'
            assert cfg.use_cls and cfg.dino_frame_stack == 3 and cfg.update_encoder
            assert cfg.agent.dino_adapter_type == 'linear' and cfg.agent.dino_adapter_output_dim == 512
            assert not cfg.auto_resume and cfg.load_model is None and cfg.load_replay_buffer is None
            run_dir = Path(source['run_dir'])
            run_name = run_dir.name
            run_id = source['wandb_run_id']
            for token in (task, 'cls3', 'stateB', group, 'seed1', CAMPAIGN):
                assert token in run_name, (token, run_name)
            launch_environment = dict(NUMERIC_ENV, PATH=str(PYTHON.parent) + ':' + os.environ.get('PATH', ''),
                                      WQ_STATEB_MANIFEST=str(ROOT / 'manifest.json'),
                                      WANDB_ENTITY='lmu_rl', WANDB_MODE='online',
                                      WANDB_RUN_NAME=run_name, MUJOCO_GL='egl',
                                      MUJOCO_EGL_DEVICE_ID=str(gpu))
            launch_command = ['/usr/bin/env'] + [key + '=' + value for key, value in launch_environment.items()]
            launch_command += ['bash', source['job_file']]
            all_rows.append(dict(index=len(all_rows), task=task, group=group,
                                 idm_coef=source['idm_coef'], seed=1, gpu=gpu,
                                 cpus=cpus, physical_cores=[first_core, first_core + 1],
                                 goal_space=source['goal_space'], session=source['session'],
                                 run_id=run_id, run_name=run_name,
                                 wandb_url='https://wandb.ai/lmu_rl/controllable_agent_baseline/runs/' + run_id,
                                 run_dir=str(run_dir), checkpoint_dir=str(CHECKPOINTS / CAMPAIGN / run_name),
                                 stdout_log=str(run_dir / 'stdout.log'), eval_csv=str(run_dir / 'eval.csv'),
                                 session_log=source['session_log'], job_file=source['job_file'],
                                 launch_command=launch_command, launch_environment=launch_environment,
                                 launcher_overrides=original, safety_overrides=extra, overrides=overrides,
                                 resolved_config=resolved, baseline_dir=str(baseline_dir),
                                 baseline_differences=differences,
                                 baseline_new_default_keys={key: after[key] for key in after.keys() - before.keys()}))

    assert len(all_rows) == len({(row['task'], row['group']) for row in all_rows}) == 16
    assert Counter(row['gpu'] for row in all_rows) == {2: 4, 3: 4, 4: 2, 5: 2, 6: 2, 7: 2}
    assert len({cpu for row in all_rows for cpu in row['cpus']}) == 64
    for key in ('run_id', 'run_dir', 'checkpoint_dir', 'session'):
        assert len({row[key] for row in all_rows}) == 16, key
    for task in TASKS:
        pair = [row for row in all_rows if row['task'] == task]
        left, right = (flatten(row['resolved_config']) for row in pair)
        diff = {key for key in left.keys() | right.keys() if left.get(key) != right.get(key)}
        assert diff == {'agent.idm_coef', 'experiment'}, (task, diff)
        assert pair[0]['gpu'] == pair[1]['gpu']
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(REPO), text=True).strip()
    manifest = dict(campaign=CAMPAIGN, created_at=datetime.now(timezone.utc).isoformat(),
                    repo=str(REPO), code_commit=commit, source_hashes={}, source_hashes_pending=True,
                    python=str(PYTHON), launcher=str(LAUNCHER), runner=str(RUNNER),
                    wandb_entity='lmu_rl', wandb_project='controllable_agent_baseline',
                    baseline_prefix=BASELINE_PREFIX, baseline_suffix=BASELINE_SUFFIX,
                    launcher_invocations=invocations, runs=all_rows)
    write_json(ROOT / 'manifest.json', manifest)
    with (ROOT / 'manifest.tsv').open('w') as stream:
        writer = csv.DictWriter(stream, delimiter='\t', extrasaction='ignore', fieldnames=[
            'index', 'task', 'group', 'idm_coef', 'seed', 'gpu', 'cpus', 'physical_cores', 'goal_space',
            'run_id', 'run_name', 'wandb_url', 'run_dir', 'stdout_log', 'checkpoint_dir', 'eval_csv',
            'session', 'job_file'])
        writer.writeheader()
        writer.writerows(all_rows)
    print(json.dumps(dict(status='PREPARED', runs=len(all_rows), manifest=str(ROOT / 'manifest.json'),
                          source_hashes_pending=True, gpu_counts=Counter(row['gpu'] for row in all_rows))))


if __name__ == '__main__':
    prepare()
