#!/usr/bin/env python3
"""Read-only live verification and a short CPU usage measurement."""
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent
manifest = json.loads((ROOT / 'manifest.json').read_text())


def proc_stat(pid):
    fields = (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()
    return {'ppid': int(fields[1]), 'ticks': int(fields[11]) + int(fields[12]), 'state': fields[0]}


def process_tree():
    result = {}
    for entry in Path('/proc').iterdir():
        if entry.name.isdigit():
            try:
                result[int(entry.name)] = proc_stat(entry.name)
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                pass
    return result


def host_cpu():
    return [int(x) for x in Path('/proc/stat').read_text().splitlines()[0].split()[1:9]]


before = process_tree()
host_before = host_cpu()
started = time.monotonic()
time.sleep(10)
after = process_tree()
elapsed = time.monotonic() - started
host_after = host_cpu()
rows = []
for row in manifest['runs']:
    run = Path(row['run_dir'])
    pid = json.loads((run / 'pid.json').read_text())['pid']
    descendants = {pid}
    while True:
        expanded = descendants | {p for p, stat in after.items() if stat['ppid'] in descendants}
        if expanded == descendants:
            break
        descendants = expanded
    ticks = sum(after[p]['ticks'] - before[p]['ticks'] for p in descendants if p in before and p in after)
    masks, tids = [], 0
    for p in descendants:
        for entry in (Path('/proc') / str(p) / 'task').glob('*'):
            try:
                affinity = sorted(os.sched_getaffinity(int(entry.name)))
                tids += 1
                if affinity != row['cpus']:
                    masks.append({'pid': p, 'tid': int(entry.name), 'affinity': affinity})
            except ProcessLookupError:
                pass
    audit = json.loads((run / 'startup_audit.json').read_text()) if (run / 'startup_audit.json').exists() else None
    inputs = json.loads((run / 'live_input_audit.json').read_text()) if (run / 'live_input_audit.json').exists() else {}
    rows.append(dict(task=row['task'], group=row['group'], run_id=row['run_id'], pid=pid,
        alive=pid in after and after[pid]['state'] != 'Z',
        cpu_logical_equivalents=ticks / os.sysconf('SC_CLK_TCK') / elapsed,
        process_count=len(descendants), total_threads=tids,
        mismatched_thread_affinities=masks, startup_pass=bool(audit and audit['status'] == 'PASS'),
        torch_num_threads=audit['torch_num_threads'] if audit else None,
        live_inputs=inputs, initial_sha256=audit['initial_sha256'] if audit else None))
pairs = {}
for task in sorted({r['task'] for r in rows}):
    pair = [r for r in rows if r['task'] == task]
    pairs[task] = bool(pair[0]['initial_sha256'] and pair[0]['initial_sha256'] == pair[1]['initial_sha256'])
delta = [b - a for a, b in zip(host_before, host_after)]
report = dict(at=datetime.now(timezone.utc).isoformat(), sample_seconds=elapsed, runs=rows,
    all_live=all(r['alive'] for r in rows), all_startup_pass=all(r['startup_pass'] for r in rows),
    all_thread_affinities_correct=all(not r['mismatched_thread_affinities'] for r in rows),
    all_live_common_initial_weights_identical=all(pairs.values()), pairs=pairs,
    campaign_cpu_logical_equivalents=sum(r['cpu_logical_equivalents'] for r in rows),
    host_cpu_idle_percent=100 * delta[3] / sum(delta))
(ROOT / 'runtime_verification.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({k: v for k, v in report.items() if k not in {'runs', 'pairs'}}, indent=2))
for row in rows:
    print(row['task'], row['group'], 'pid', row['pid'], 'CPU', round(row['cpu_logical_equivalents'], 2),
          'threads', row['total_threads'], 'audit', row['startup_pass'], 'inputs', row['live_inputs'])
