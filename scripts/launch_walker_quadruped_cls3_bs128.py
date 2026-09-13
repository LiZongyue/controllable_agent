#!/usr/bin/env python3
"""Prepare, launch and report the 16-run bs=128 Walker/Quadruped campaign.

Each row clones the *actual* resolved overrides (config and code version) of
one already-completed bs=1024 run:

  - "shared_lr5e5" rows clone 20260904_walker_quadruped_cls3_visualB_alllr5e5_s1
    (shared F/B adapter, all optimizer LRs = 5e-5).
  - "separated_lr5e5"/"separated_lr1e5" rows clone the matching group from
    wq_cls3_lr_20260907T033919Z_retry1 (separated F/B adapters).

The only override this script changes relative to each row's source run is
``agent.batch_size`` (1024 -> 128), plus the naming/path fields
(``experiment``, ``checkpoint_root``, ``hydra.run.dir``) needed to give the
new run its own identity. Every other resolved config key is asserted equal
to the source run's resolved config -- DINO CLS3, frozen backbone, visual-B
input, all optimizer LRs, ortho_coef, dataset/update-frequency settings, task
inference sample counts, and eval settings all carry over unchanged. All 16
runs train from scratch (no checkpoint resume, no gradient accumulation) for
the same 2,000,010-frame budget as their bs=1024 counterparts.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_walker_quadruped_cls3_bs128 import (  # noqa: E402
    EXPECTED_BATCH_SIZE, GPU_ASSIGNMENT, ROWS_SPEC, RUN_CPUS, assigned_cpus,
)

os.environ.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                  OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
                  VECLIB_MAXIMUM_THREADS='1', TOKENIZERS_PARALLELISM='false')

REPO = Path(__file__).resolve().parents[1]
PYTHON = '/data/fan2/env/miniconda3/envs/occ_rlu/bin/python'
RUNS = Path('/mnt/data_7tb/fanfeng/controllable_agent_runs')
CHECKPOINTS = Path('/mnt/data_nvme1/fanfeng/controllable_agent_ckpt')
BASELINE_SHARED = RUNS / '20260904_walker_quadruped_cls3_visualB_alllr5e5_s1'
BASELINE_SEP_CAMPAIGN = 'wq_cls3_lr_20260907T033919Z_retry1'
BASELINE_SEP = RUNS / BASELINE_SEP_CAMPAIGN
WANDB_ENTITY = 'lmu_rl'
WANDB_PROJECT = 'controllable_agent_baseline'
# The only training-relevant override this script ever changes vs. the source run.
ALLOWED = {'agent.batch_size', 'experiment', 'checkpoint_root'}
SMOKE_EXTRA_ALLOWED = {'num_train_frames', 'eval_every_frames', 'num_eval_episodes',
                       'final_tests', 'checkpoint_every', 'snapshot_at'}


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2) + '\n')


def source_hashes():
    paths = sorted(p for p in (REPO / 'url_benchmark').rglob('*')
                   if p.suffix in ('.py', '.yaml') and p.is_file())
    paths += [Path(__file__).resolve(),
              REPO / 'scripts/run_walker_quadruped_cls3_bs128.py',
              REPO / 'scripts/audit_walker_quadruped_cls3_bs128.py',
              REPO / 'scripts/audit_walker_quadruped_cls3_lr.py']
    return {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def flatten(value, prefix=''):
    result = {}
    for key, item in value.items():
        name = f'{prefix}.{key}' if prefix else key
        if isinstance(item, dict):
            result.update(flatten(item, name))
        else:
            result[name] = item
    return result


def source_baseline_dir(task, group, source):
    if source == 'shared':
        return BASELINE_SHARED / f'dino_cls3_{task}_seed1_visualB_alllr5e5_ortho1'
    assert source == 'sep', source
    return BASELINE_SEP / f'dino_cls3_{task}_{group}_seed1_{BASELINE_SEP_CAMPAIGN}'


def prepare(campaign, smoke=False):
    assert {cpu for cpus in RUN_CPUS for cpu in cpus} <= os.sched_getaffinity(0), \
        'Assigned CPUs are not all available'
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    import yaml
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    sys.path.insert(0, str(REPO))
    from url_benchmark import pretrain  # noqa: F401  (registers the existing configuration)

    assert re.fullmatch(r'[A-Za-z0-9_-]+', campaign), campaign
    root = REPO / 'analysis_outputs' / campaign
    assert not root.exists(), f'Fresh campaign already exists: {root}'
    assert not (RUNS / campaign).exists()
    assert not (CHECKPOINTS / campaign).exists()
    allowed = ALLOWED | (SMOKE_EXTRA_ALLOWED if smoke else set())
    rows = []
    for index, (task, group, separated, lr, source) in enumerate(ROWS_SPEC):
        baseline_dir = source_baseline_dir(task, group, source)
        overrides = yaml.safe_load((baseline_dir / '.hydra/overrides.yaml').read_text())
        values = dict(arg.split('=', 1) for arg in overrides)
        assert len(values) == len(overrides), f'Duplicate baseline override: {task}/{group}'
        seed = int(values['seed'])
        assert seed == 1
        assert int(values['agent.batch_size']) == 1024, 'source run must be the bs=1024 baseline'
        assert float(values['agent.lr']) == lr
        assert (values['agent.dino_separate_fb_adapters'] == 'True') is separated
        baseline_run_id = baseline_dir.name
        run_id = f'dino_cls3_{task}_{group}_bs128_seed1_{campaign}'
        run_dir = RUNS / campaign / run_id
        checkpoint_dir = CHECKPOINTS / campaign / run_id
        values.update({'agent.batch_size': str(EXPECTED_BATCH_SIZE),
                       'experiment': f'walker_quadruped_cls3_bs128_{group}',
                       'checkpoint_root': str(CHECKPOINTS / campaign)})
        if smoke:
            values.update({'num_train_frames': '6010', 'eval_every_frames': '2000',
                           'num_eval_episodes': '1', 'final_tests': '1',
                           'checkpoint_every': '6000', 'snapshot_at': '[6000]'})
        overrides_list = [f'{key}={value}' for key, value in values.items()]
        overrides_list.append(f'hydra.run.dir={run_dir}')
        with initialize_config_dir(config_dir=str(REPO / 'url_benchmark'), version_base='1.1'):
            cfg = compose(config_name='base_config', overrides=overrides_list)
        before = flatten(OmegaConf.to_container(OmegaConf.load(baseline_dir / '.hydra/config.yaml'), resolve=False))
        after = flatten(OmegaConf.to_container(cfg, resolve=False))
        differences = {key: {'baseline': before.get(key), 'new': after.get(key)}
                       for key in before.keys() | after.keys() if before.get(key) != after.get(key)}
        assert differences.keys() <= allowed, differences
        assert set(before) == set(after), 'Unexpected baseline configuration keys'
        assert cfg.obs_type == 'dino' and cfg.dino_model_name == 'facebook/dinov2-base'
        assert cfg.use_cls and cfg.dino_frame_stack == 3 and cfg.goal_space is None
        assert cfg.update_encoder and cfg.agent.dino_use_adapter
        assert not cfg.agent.dino_separate_backward_adapter and not cfg.agent.dino_flare_b
        assert cfg.agent.idm_coef == 0 and cfg.agent.lr_coef == 1
        assert not cfg.auto_resume and cfg.load_model is None and cfg.load_replay_buffer is None
        assert int(cfg.agent.batch_size) == EXPECTED_BATCH_SIZE
        assert cfg.agent.dino_separate_fb_adapters is separated
        for field in ('lr', 'fb_lr', 'lr_f', 'lr_b', 'lr_actor'):
            assert float(getattr(cfg.agent, field)) == lr, f'{field} drifted from source'
        assert float(cfg.agent.ortho_coef) == 1.0 and float(cfg.agent.fb_target_tau) == 0.01
        if not smoke:
            assert int(cfg.num_train_frames) == 2000010, 'full runs must keep the 2M-frame budget'
        rows.append(dict(task=task, group=group, lr=lr, separated=separated, seed=seed,
                         gpu=GPU_ASSIGNMENT[index], cpus=list(RUN_CPUS[index]),
                         session=f'wqbs128_{index:02d}_{task}_{group}',
                         run_id=run_id, run_dir=str(run_dir), checkpoint_dir=str(checkpoint_dir),
                         stdout_log=str(run_dir / 'stdout.log'),
                         wandb_url=f'https://wandb.ai/{WANDB_ENTITY}/{WANDB_PROJECT}/runs/{run_id}',
                         baseline_dir=str(baseline_dir), baseline_run_id=baseline_run_id,
                         baseline_run_dir=str(baseline_dir),
                         baseline_wandb_url=f'https://wandb.ai/{WANDB_ENTITY}/{WANDB_PROJECT}/runs/{baseline_run_id}',
                         baseline_differences=differences, num_train_frames=int(values['num_train_frames']),
                         overrides=overrides_list))
    assert len(rows) == len(ROWS_SPEC) == len({(r['task'], r['group']) for r in rows})
    assert Counter(r['gpu'] for r in rows) == Counter(GPU_ASSIGNMENT)
    assert all(len(r['cpus']) == 4 for r in rows)
    assert len({cpu for row in rows for cpu in row['cpus']}) == 4 * len(rows)
    for key in ('run_id', 'run_dir', 'checkpoint_dir', 'session'):
        assert len({r[key] for r in rows}) == len(rows), key
    root.mkdir(parents=True)
    (root / 'jobs').mkdir()
    manifest = dict(campaign=campaign, created_at=datetime.now(timezone.utc).isoformat(),
                    smoke=smoke, baseline_shared=str(BASELINE_SHARED),
                    baseline_separated=str(BASELINE_SEP), wandb_entity=WANDB_ENTITY,
                    wandb_project=WANDB_PROJECT, source_hashes=source_hashes(), runs=rows)
    for index, row in enumerate(rows):
        job = root / 'jobs' / f'{index:02d}_{row["task"]}_{row["group"]}.sh'
        # The runner claims fresh directories with mkdir(exist_ok=False), then execs training.
        job.write_text('#!/usr/bin/env bash\nset -euo pipefail\nexec ' + shlex.join([
            PYTHON, str(REPO / 'scripts/run_walker_quadruped_cls3_bs128.py'),
            '--manifest', str(root / 'manifest.json'), '--index', str(index)]) + '\n')
        row['job_file'] = str(job)
        subprocess.run(['bash', '-n', str(job)], check=True)
    write_json(root / 'manifest.json', manifest)
    with (root / 'manifest.tsv').open('w') as stream:
        keys = ['task', 'group', 'lr', 'gpu', 'cpus', 'seed', 'session', 'run_id', 'wandb_url',
                'baseline_run_id', 'baseline_wandb_url', 'run_dir', 'stdout_log', 'checkpoint_dir', 'job_file']
        writer = csv.DictWriter(stream, fieldnames=keys, delimiter='\t', extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    print(f'Prepared {len(rows)} unique bs=128 configurations (smoke={smoke}): {root / "manifest.json"}', flush=True)


def launch(manifest_path):
    manifest = json.loads(manifest_path.read_text())
    root = manifest_path.parent
    assert manifest['source_hashes'] == source_hashes(), 'Source changed after preparation'
    n = len(manifest['runs'])
    cpu_sets = [assigned_cpus(row, index) for index, row in enumerate(manifest['runs'])]
    cpus = [cpu for cpu_set in cpu_sets for cpu in cpu_set]
    assert len(cpu_sets) == n and len(cpus) == len(set(cpus)), f'Expected {n} nonoverlapping CPU assignments'
    assert set(cpus) <= os.sched_getaffinity(0), 'Assigned CPUs are not all available'
    preflight = json.loads((root / 'preflight.json').read_text())
    assert preflight['status'] == 'PASS' and len(preflight['runs']) == n, 'Preflight failed/incomplete'
    for row, checked in zip(manifest['runs'], preflight['runs']):
        assert all(row[key] == checked[key] for key in ('task', 'group', 'run_dir'))
        assert checked['audit']['status'] == 'PASS'
        assert checked['audit']['expected_lr'] == row['lr']
        assert checked['audit']['separated'] == row['separated']
        assert checked['audit']['batch_size'] == EXPECTED_BATCH_SIZE
    assert not (root / 'launch_results.json').exists(), 'This campaign was already launched'
    assert shutil.disk_usage(RUNS).free >= 5 * 1024**3, 'Insufficient log disk space'
    assert shutil.disk_usage(CHECKPOINTS).free >= 10 * 1024**3, 'Insufficient checkpoint disk space'
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.free,utilization.gpu',
                                   '--format=csv,noheader,nounits'], text=True)
    gpu_rows = list(csv.reader(raw.splitlines()))
    gpus_used = sorted({row['gpu'] for row in manifest['runs']})
    for gpu in gpus_used:
        row = next(r for r in gpu_rows if int(r[0]) == gpu)
        assert int(row[2]) >= 15000, f'GPU {gpu} has too little free memory: {row}'
    for row in manifest['runs']:
        for key in ('run_dir', 'checkpoint_dir'):
            target = Path(row[key])
            assert not target.exists() and not target.is_symlink(), f'Collision: {target}'
        assert subprocess.run(['tmux', 'has-session', '-t', row['session']],
                              capture_output=True).returncode != 0, row['session']
    (root / 'gpu_before.csv').write_text(raw)
    results = []
    for index, row in enumerate(manifest['runs']):
        proc = subprocess.run(['tmux', 'new-session', '-d', '-s', row['session'],
                               'bash ' + shlex.quote(row['job_file'])], capture_output=True, text=True)
        results.append(dict(task=row['task'], group=row['group'], gpu=row['gpu'], cpus=cpu_sets[index],
                            session=row['session'], launch_returncode=proc.returncode,
                            error=proc.stderr.strip()))
        write_json(root / 'launch_results.json', results)
        print(json.dumps(results[-1]), flush=True)


def status(manifest_path):
    manifest = json.loads(manifest_path.read_text())
    root = manifest_path.parent
    results = []
    for index, row in enumerate(manifest['runs']):
        work = Path(row['run_dir'])
        pid_data = json.loads((work / 'pid.json').read_text()) if (work / 'pid.json').exists() else {}
        pid = pid_data.get('pid')
        alive = False
        cpu_affinity = None
        if pid:
            try:
                os.kill(pid, 0)
                cpu_affinity = sorted(os.sched_getaffinity(pid))
                alive = True
            except ProcessLookupError:
                pass
        audited = (work / 'startup_audit.json').exists()
        state = 'running_verified' if alive and audited else ('starting' if alive else 'not_running')
        log = work / 'stdout.log'
        tail = log.read_text(errors='replace')[-12000:] if log.exists() else ''
        if not alive and any(x in tail for x in ('Traceback', 'Error executing job', 'CUDA out of memory')):
            state = 'failed'
        eval_frame = None
        eval_csv = work / 'eval.csv'
        if eval_csv.exists():
            with eval_csv.open() as stream:
                evaluations = list(csv.DictReader(stream))
            if evaluations:
                eval_frame = evaluations[-1].get('frame')
        cpus = cpu_affinity if cpu_affinity is not None else assigned_cpus(row, index)
        results.append({**{key: row[key] for key in ('task', 'group', 'lr', 'gpu', 'seed', 'session',
                                                    'run_id', 'wandb_url', 'baseline_run_id',
                                                    'stdout_log', 'checkpoint_dir', 'run_dir')},
                        'cpus': cpus, 'cpu_affinity': cpu_affinity,
                        'pid': pid, 'status': state, 'latest_eval_frame': eval_frame})
    write_json(root / 'status.json', results)
    with (root / 'status.tsv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]), delimiter='\t')
        writer.writeheader()
        writer.writerows(results)
    lines = ['| Task | Group | Actual LR | GPU | Logical CPUs | PID | Status | Baseline (bs=1024) | W&B | Log |',
             '|---|---|---|---|---|---|---|---|---|---|']
    for r in results:
        actual_lr = str(r['lr']) if r['status'] == 'running_verified' else 'pending'
        cpu_text = ','.join(map(str, r['cpus']))
        lines.append(f"| {r['task']} | {r['group']} | {actual_lr} | {r['gpu']} | {cpu_text} | {r['pid']} | {r['status']} | "
                     f"{r['baseline_run_id']} | [run]({r['wandb_url']}) | [stdout]({r['stdout_log']}) |")
    (root / 'status.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(results, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--smoke', action='store_true', help='Use a short frame budget for a pipeline smoke test.')
    parser.add_argument('--campaign', default='wq_bs128_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    parser.add_argument('--launch', type=Path)
    parser.add_argument('--status', type=Path)
    args = parser.parse_args()
    if args.prepare:
        prepare(args.campaign, smoke=args.smoke)
    elif args.launch:
        launch(args.launch)
    elif args.status:
        status(args.status)
    else:
        parser.error('Choose --prepare [--smoke] --campaign NAME, --launch MANIFEST, or --status MANIFEST')


if __name__ == '__main__':
    main()
