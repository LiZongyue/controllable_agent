#!/usr/bin/env python3
"""Launch jobs emitted by the existing launcher and report their live status."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def launch(manifest):
    assert not (ROOT / 'launch_results.json').exists(), 'Campaign already launched'
    for relative, digest in manifest['source_hashes'].items():
        assert hashlib.sha256((REPO / relative).read_bytes()).hexdigest() == digest, relative
    assert manifest['source_hashes'] and not manifest['source_hashes_pending']
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() == manifest['code_commit']
    preflight = json.loads((ROOT / 'preflight.json').read_text())
    assert preflight['status'] == 'PASS' and len(preflight['runs']) == 16
    rows = manifest['runs']
    assert len({(row['task'], row['group']) for row in rows}) == len(rows) == 16
    assert len({cpu for row in rows for cpu in row['cpus']}) == 64
    for row in rows:
        assert set(row['cpus']) <= os.sched_getaffinity(0)
        for key in ('run_dir', 'checkpoint_dir'):
            path = Path(row[key])
            assert not path.exists() and not path.is_symlink(), path
        assert subprocess.run(['tmux', 'has-session', '-t', row['session']],
                              capture_output=True).returncode != 0, row['session']
        subprocess.run(['bash', '-n', row['job_file']], check=True)
    assert shutil.disk_usage('/mnt/data_7tb').free > 5 * 1024**3
    assert shutil.disk_usage('/mnt/data_nvme1').free > 25 * 1024**3
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free,utilization.gpu',
                                   '--format=csv,noheader,nounits'], text=True)
    gpus = {int(row[0]): row[1:] for row in csv.reader(raw.splitlines())}
    for gpu in {row['gpu'] for row in rows}:
        assert int(gpus[gpu][0]) > 60000 and int(gpus[gpu][1]) < 10, (gpu, gpus[gpu])
    (ROOT / 'gpu_before.csv').write_text(raw)
    results = []
    for row in rows:
        command = shlex.join(row['launch_command']) + ' > ' + shlex.quote(row['session_log']) + ' 2>&1'
        proc = subprocess.run(['tmux', 'new-session', '-d', '-s', row['session'], command],
                              capture_output=True, text=True)
        results.append(dict(task=row['task'], group=row['group'], run_id=row['run_id'],
                            gpu=row['gpu'], session=row['session'], returncode=proc.returncode,
                            error=proc.stderr.strip(), at=datetime.now(timezone.utc).isoformat()))
        write_json(ROOT / 'launch_results.json', results)
        print(json.dumps(results[-1]), flush=True)


def status(manifest):
    results = []
    for row in manifest['runs']:
        run = Path(row['run_dir'])
        pid_data = json.loads((run / 'pid.json').read_text()) if (run / 'pid.json').exists() else {}
        pid = pid_data.get('pid')
        live = False
        affinity = None
        if pid:
            try:
                command = (Path('/proc') / str(pid) / 'cmdline').read_bytes().split(b'\0')
                live = str(REPO / 'scripts/run_walker_quadruped_cls3_stateb_idm.py').encode() in command
                if live:
                    affinity = sorted(os.sched_getaffinity(pid))
            except FileNotFoundError:
                pass
        audit_path = run / 'startup_audit.json'
        audited = audit_path.exists()
        state = 'running_verified' if live and audited else ('starting' if live else 'not_running')
        log = run / 'stdout.log'
        tail = log.read_text(errors='replace')[-8000:] if log.exists() else ''
        if not live and any(x in tail for x in ('Traceback', 'Error executing job', 'CUDA out of memory')):
            state = 'failed'
        frames = {}
        for kind in ('train', 'eval'):
            path = run / (kind + '.csv')
            records = list(csv.DictReader(path.open())) if path.exists() else []
            frames[kind + '_frame'] = records[-1]['frame'] if records else None
            if kind == 'eval':
                frames['eval_records'] = len(records)
        results.append({**{key: row[key] for key in ('task', 'group', 'run_id', 'run_name', 'wandb_url',
                         'gpu', 'cpus', 'run_dir', 'stdout_log', 'checkpoint_dir', 'eval_csv', 'session')},
                        'pid': pid, 'state': state, 'live_affinity': affinity, **frames})
    write_json(ROOT / 'status.json', results)
    with (ROOT / 'status.tsv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]), delimiter='\t')
        writer.writeheader()
        writer.writerows(results)
    lines = ['| Task | IDM | GPU | PID | Status | W&B run ID | Log | Checkpoints | Full eval |',
             '|---|---|---|---|---|---|---|---|---|']
    for r in results:
        lines.append(f"| {r['task']} | {r['group']} | {r['gpu']} | {r['pid']} | {r['state']} | "
                     f"[{r['run_id']}]({r['wandb_url']}) | [stdout]({r['stdout_log']}) | "
                     f"[directory]({r['checkpoint_dir']}) | [CSV]({r['eval_csv']}) |")
    (ROOT / 'status.md').write_text('\n'.join(lines) + '\n')
    for r in results:
        print(r['task'], r['group'], r['gpu'], r['pid'], r['state'],
              'train=' + str(r['train_frame']), 'eval=' + str(r['eval_frame']))
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['launch', 'status'])
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    globals()[args.action](manifest)
