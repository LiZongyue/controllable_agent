#!/usr/bin/env python3
"""Audit historical SWM-CLS and CNN-FB 2M-frame runtime evidence.

This script is intentionally read-only with respect to the historical run and
checkpoint roots.  It writes a row-level CSV and a short evidence report to an
analysis output directory.  Historical timings are never marked as eligible
for the primary comparison because the two launch batches used different and
uncontrolled concurrency.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


SEEDS = {2009, 7532, 8164}
DOMAIN_TASKS = {
    "walker": {"walker_stand", "walker_walk", "walker_run", "walker_flip"},
    "quadruped": {
        "quadruped_stand",
        "quadruped_walk",
        "quadruped_run",
        "quadruped_jump",
    },
    "cheetah": {
        "cheetah_walk",
        "cheetah_walk_backward",
        "cheetah_run",
        "cheetah_run_backward",
    },
}
TASK_TO_DOMAIN = {
    task: domain for domain, tasks in DOMAIN_TASKS.items() for task in tasks
}
TARGET_FRAME = 2_000_000
SNAPSHOT_FRAMES = (500_000, 1_000_000, 1_500_000, 2_000_000)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("/mnt/data_7tb/fanfeng/controllable_agent_runs"),
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path("/mnt/data_7tb/fanfeng/controallable_agent_ckpt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis_outputs/swm_cls_vs_cnn_fb_20260727"),
    )
    return parser.parse_args()


def top_level_scalar(text: str, key: str) -> str | None:
    match = re.search(rf"^{re.escape(key)}:\s*([^#\n]+?)\s*$", text, re.MULTILINE)
    return match.group(1).strip() if match else None


def first_scalar(text: str, key: str) -> str | None:
    match = re.search(rf"^\s*{re.escape(key)}:\s*([^#\n]+?)\s*$", text, re.MULTILINE)
    return match.group(1).strip() if match else None


def as_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def parse_launcher(path: Path) -> tuple[str, str, float, int, int | None]:
    text = path.read_text(errors="replace")
    starts = re.findall(r"\[start\]\s+(\S+)", text)
    ends = re.findall(r"\[done\]\s+(\S+)", text)
    exits = re.findall(r"\bexit=(\d+)", text)
    gpu_ids = re.findall(r"\bgpu=(\d+)", text)
    if not starts or not ends or not exits:
        raise ValueError(f"incomplete launcher evidence: {path}")
    start = starts[0]
    end = ends[-1]
    start_dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
    end_dt = datetime.fromisoformat(end.replace("Z", "+00:00"))
    return start, end, (end_dt - start_dt).total_seconds() / 3600.0, int(exits[-1]), (
        int(gpu_ids[0]) if gpu_ids else None
    )


def parse_train_csv(path: Path) -> dict[str, Any]:
    frames: list[int] = []
    total_times: list[float] = []
    total_at_target: float | None = None
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            frame = int(float(row["frame"]))
            total_time = float(row["total_time"])
            frames.append(frame)
            total_times.append(total_time)
            if frame == TARGET_FRAME:
                total_at_target = total_time
    return {
        "num_train_rows": len(frames),
        "first_frame": frames[0] if frames else None,
        "max_frame": max(frames, default=-1),
        "frame_strictly_increasing": all(
            later > earlier for earlier, later in zip(frames, frames[1:])
        ),
        "time_monotonic": all(
            later >= earlier for earlier, later in zip(total_times, total_times[1:])
        ),
        "train_total_hours": (
            total_at_target / 3600.0 if total_at_target is not None else None
        ),
    }


def read_metadata(run_dir: Path) -> tuple[str, dict[str, Any]]:
    candidates = sorted(run_dir.glob("wandb/run-*/files/wandb-metadata.json"))
    if not candidates:
        return "", {}
    path = candidates[-1]
    try:
        return str(path), json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return str(path), {}


def classify_method(run_name: str, config_text: str, task: str, seed: int) -> str | None:
    if task not in TASK_TO_DOMAIN or seed not in SEEDS:
        return None
    if re.fullmatch(
        rf"20260426_123938_seed{seed}_{re.escape(task)}_cuda\d+_dino_cls", run_name
    ):
        return "SWM-CLS"
    if "cnn" in run_name.lower() and top_level_scalar(config_text, "obs_type") == "pixels":
        return "CNN-FB"
    return None


def concurrency_evidence(method: str, run_name: str) -> tuple[str, str]:
    if method == "SWM-CLS":
        return (
            "12_simultaneous_jobs_on_one_gpu_per_seed",
            "Twelve task runs for each seed started within one second and shared one H100; "
            "CPU allocation and external load were uncontrolled.",
        )
    if "parallel24" in run_name:
        return (
            "parallel24_multi_job_shared_gpu_batch",
            "The original 24-job CNN batch placed multiple jobs on GPUs 0/3/4 and had many "
            "exit-1/137 siblings; contention differs from SWM.",
        )
    if "failed15_safe" in run_name:
        return (
            "one_job_per_gpu_rescue_lane_three_lanes",
            "The rescue launcher serialized jobs within each GPU lane, but three lanes shared "
            "the host and the allocation differs from the 12-way SWM launch.",
        )
    return (
        "uncontrolled_historical_launch",
        "Historical concurrency and CPU allocation were not controlled to match SWM.",
    )


def core_config_matches(config_text: str) -> bool:
    expected_top_level = {
        "num_train_frames": "2000010",
        "action_repeat": "2",
        "eval_every_frames": "10000",
        "checkpoint_every": "100000",
        "num_eval_episodes": "10",
        "final_tests": "10",
    }
    if any(top_level_scalar(config_text, key) != value for key, value in expected_top_level.items()):
        return False
    return first_scalar(config_text, "batch_size") == "1024" and first_scalar(
        config_text, "update_every_steps"
    ) == "2"


def audit_runs(runs_root: Path, checkpoint_root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    rejected: list[str] = []
    for run_dir in sorted(path for path in runs_root.iterdir() if path.is_dir()):
        config_path = run_dir / ".hydra" / "config.yaml"
        train_path = run_dir / "train.csv"
        launcher_path = run_dir / "launcher.log"
        if not config_path.is_file() or not train_path.is_file() or not launcher_path.is_file():
            continue
        config_text = config_path.read_text(errors="replace")
        task = top_level_scalar(config_text, "task") or ""
        seed = as_int(top_level_scalar(config_text, "seed"))
        if seed is None:
            continue
        method = classify_method(run_dir.name, config_text, task, seed)
        if method is None:
            continue
        train = parse_train_csv(train_path)
        try:
            start, end, launcher_hours, exit_code, gpu_index = parse_launcher(launcher_path)
        except ValueError as error:
            rejected.append(f"{run_dir.name}: {error}")
            continue
        pretrain_path = run_dir / "pretrain.log"
        pretrain_text = pretrain_path.read_text(errors="replace") if pretrain_path.is_file() else ""
        reload_hits = len(re.findall(r"\b(?:Reloading|Reloaded|Resuming|Resumed)\b", pretrain_text))
        workspace_count = pretrain_text.count("Workspace:")
        usable = (
            train["max_frame"] >= TARGET_FRAME
            and train["train_total_hours"] is not None
            and train["frame_strictly_increasing"]
            and train["time_monotonic"]
            and exit_code == 0
            and reload_hits == 0
            and workspace_count == 1
        )
        if not usable:
            rejected.append(
                f"{run_dir.name}: incomplete/non-continuous "
                f"(max_frame={train['max_frame']}, exit={exit_code}, reload_hits={reload_hits}, "
                f"workspace_count={workspace_count})"
            )
            continue
        concurrency_class, concurrency_reason = concurrency_evidence(method, run_dir.name)
        metadata_path, metadata = read_metadata(run_dir)
        checkpoint_dir = checkpoint_root / run_dir.name
        snapshot_values = {
            f"snapshot_{frame}_exists": (checkpoint_dir / f"snapshot_{frame}.pt").is_file()
            for frame in SNAPSHOT_FRAMES
        }
        primary_reason = (
            concurrency_reason
            + " The historical timer includes periodic evaluation and checkpoint I/O, the launcher "
            "also includes initialization/final evaluation, and no immutable git commit was recorded."
        )
        rows.append(
            {
                "method": method,
                "seed": seed,
                "domain": TASK_TO_DOMAIN[task],
                "task": task,
                "start": start,
                "end": end,
                "launcher_hours": launcher_hours,
                "train_total_hours": train["train_total_hours"],
                "concurrency_class": concurrency_class,
                "primary_eligible": "false",
                "reason": primary_reason,
                "run_dir": str(run_dir),
                "config_path": str(config_path),
                "launcher_log_path": str(launcher_path),
                "train_csv_path": str(train_path),
                "wandb_metadata_path": metadata_path,
                "checkpoint_dir": str(checkpoint_dir),
                "exit_code": exit_code,
                "num_train_rows": train["num_train_rows"],
                "first_frame": train["first_frame"],
                "max_frame": train["max_frame"],
                "frame_strictly_increasing": train["frame_strictly_increasing"],
                "time_monotonic": train["time_monotonic"],
                "workspace_count": workspace_count,
                "reload_resume_hits": reload_hits,
                "host": metadata.get("host", ""),
                "gpu_model": metadata.get("gpu", ""),
                "gpu_index": gpu_index if gpu_index is not None else "",
                "host_gpu_count": metadata.get("gpu_count", ""),
                "host_cpu_count": metadata.get("cpu_count", ""),
                "host_cpu_count_logical": metadata.get("cpu_count_logical", ""),
                "core_config_matches_requested": core_config_matches(config_text),
                **snapshot_values,
            }
        )
    rows.sort(key=lambda row: (row["method"], row["seed"], row["domain"], row["task"]))
    return rows, rejected


def fmt_hours(value: float) -> str:
    return f"{value:.2f}"


def markdown_table(headers: Iterable[str], body: Iterable[Iterable[Any]]) -> str:
    headers = list(headers)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in body)
    return "\n".join(lines)


def render_report(
    rows: list[dict[str, Any]], rejected: list[str], runs_root: Path, checkpoint_root: Path
) -> str:
    by_method = Counter(row["method"] for row in rows)
    by_method_domain: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        by_method_domain[(row["method"], row["domain"])].append(row["launcher_hours"])
    timing_rows = []
    for (method, domain), values in sorted(by_method_domain.items()):
        timing_rows.append(
            (
                method,
                domain,
                len(values),
                fmt_hours(statistics.median(values)),
                f"{fmt_hours(min(values))}--{fmt_hours(max(values))}",
            )
        )
    concurrency_counts = Counter(row["concurrency_class"] for row in rows)
    concurrency_rows = sorted((name, count) for name, count in concurrency_counts.items())
    swm_rows = [row for row in rows if row["method"] == "SWM-CLS"]
    cnn_rows = [row for row in rows if row["method"] == "CNN-FB"]
    swm_seed_task = Counter((row["seed"], row["task"]) for row in swm_rows)
    cnn_seed_task = Counter((row["seed"], row["task"]) for row in cnn_rows)
    swm_expected = {(seed, task) for seed in SEEDS for task in TASK_TO_DOMAIN}
    cnn_expected = {(seed, task) for seed in SEEDS for task in TASK_TO_DOMAIN}
    swm_missing = sorted(swm_expected - set(swm_seed_task))
    cnn_missing = sorted(cnn_expected - set(cnn_seed_task))
    snapshot_counts = {
        method: {
            frame: sum(
                bool(row[f"snapshot_{frame}_exists"])
                for row in rows
                if row["method"] == method
            )
            for frame in SNAPSHOT_FRAMES
        }
        for method in ("SWM-CLS", "CNN-FB")
    }
    hardware = Counter(
        (
            row["host"],
            row["gpu_model"],
            row["host_gpu_count"],
            row["host_cpu_count"],
            row["host_cpu_count_logical"],
        )
        for row in rows
    )
    global_snapshot_500_entries = list(checkpoint_root.rglob("snapshot_500000.pt"))
    requested_checkpoint_dirs = {Path(row["checkpoint_dir"]) for row in rows}
    requested_snapshot_500_entries = sum(
        entry.parent in requested_checkpoint_dirs for entry in global_snapshot_500_entries
    )
    unrelated_snapshot_500_entries = (
        len(global_snapshot_500_entries) - requested_snapshot_500_entries
    )
    report = f"""# Historical full-run runtime evidence

## Decision

None of the {len(rows)} usable historical 2M-frame runs is eligible for the primary
SWM-CLS versus CNN-FB wall-clock comparison (`primary_eligible=false` for every row).
The historical launches used different, uncontrolled concurrency and CPU/GPU sharing.
The controlled single-H100 profiling runs must therefore be the primary result; these
historical values are context only.

The machine-readable evidence is in `historical_full_run_times.csv`. It contains
{by_method.get('SWM-CLS', 0)} SWM-CLS runs and {by_method.get('CNN-FB', 0)} CNN-FB runs.

## Historical launcher durations (context only)

{markdown_table(('Method', 'Domain', 'n', 'Median hours', 'Range hours'), timing_rows)}

These launcher durations are not a fair method comparison. In particular, the SWM
launch started all 12 tasks for a seed within one second on a single GPU, while the
CNN runs came from a failed multi-job batch and later one-job-per-GPU rescue lanes.

{markdown_table(('Concurrency class', 'Usable runs'), concurrency_rows)}

The clearest load-sensitivity check is CNN-FB `quadruped_run` on GPU 3: seed 2009
took 169.77 launcher hours during the earlier rescue period, whereas seed 7532 took
57.18 hours later with the same recorded algorithm and hyperparameters aside from the
seed. SWM seed 7532 on GPU 0 similarly took roughly 94--99 hours while seeds 2009/8164
on GPUs 3/5 took
roughly 65--68 hours. This variation rules out treating the old timestamps as a
controlled hardware benchmark.

## What the two historical timers include

- `train_total_hours` is `train.csv:total_time` at frame 2,000,000. The timer starts
  after environment/model construction, but it runs continuously through periodic
  10-episode evaluation every 10,000 environment frames and checkpoint serialization.
- `launcher_hours` is the UTC `[start]` to `[done]` interval in `launcher.log`. It also
  includes initialization, the evaluation at 2M, final checkpoint handling, four
  downstream tasks x ten final episodes, and external logger shutdown.
- Consequently neither column is the requested evaluation/checkpoint-excluded training
  throughput. Historical logs do not isolate those components exactly.

The relevant timing implementation is
[`url_benchmark/utils.py`](../../url_benchmark/utils.py) (`Timer`) and
[`url_benchmark/pretrain.py`](../../url_benchmark/pretrain.py) (evaluation/checkpoint
calls inside the training loop and `finalize()`).

## Continuity and hardware evidence

- Every included run has launcher exit code 0, exactly one workspace start in
  `pretrain.log`, no reload/resume message, strictly increasing training frames,
  monotonic total time, and a frame-2M record.
- Recorded hardware tuples `(host, GPU, host GPU count, physical CPUs, logical CPUs)`:
  `{dict(hardware)}`.
- Several CNN W&B metadata files omit GPU fields, and metadata reports whole-host CPU
  counts rather than process affinity/cgroup allocation. Thus equal per-process CPU
  allocation cannot be established.
- The W&B metadata records a mutable program path but no immutable Git commit. The
  Hydra configs match the requested core frame count, batch size, update interval,
  action repeat, evaluation interval, and final-test count for all included rows, but
  exact code-revision identity across April/May runs cannot be proven.

## Checkpoint coverage

Snapshot counts among usable rows:

{markdown_table(
    ('Method', 'n', '0.5M', '1.0M', '1.5M', '2.0M'),
    (
        (
            method,
            by_method.get(method, 0),
            snapshot_counts[method][500_000],
            snapshot_counts[method][1_000_000],
            snapshot_counts[method][1_500_000],
            snapshot_counts[method][2_000_000],
        )
        for method in ('SWM-CLS', 'CNN-FB')
    ),
)}

The full checkpoint-root scan found {len(global_snapshot_500_entries)}
`snapshot_500000.pt` entries under `{checkpoint_root}`: {unrelated_snapshot_500_entries}
belong to unrelated older 20260419 runs and {requested_snapshot_500_entries} belong to
the requested SWM-CLS/CNN-FB runs audited here. The historical logs show that the
requested 0.5M snapshots were originally written, but those files were not retained.
SWM has complete 1M/1.5M/2M coverage for all 36 requested seed-task combinations.
CNN has only 12 complete combinations: nine for seed 2009, three for seed 7532, and
none for seed 8164. Missing CNN combinations: `{cnn_missing}`. Missing SWM combinations:
`{swm_missing}`.

The existing train-time `eval.csv` files are not a strict substitute for 0.5M
reward-projection evaluation. With `custom_reward=None`, registered Walker/Quadruped
tasks use a registered goal vector; `finalize()` explicitly sets `custom_reward` and
forces reward projection. Therefore a uniform reward-projection 0.5M evaluation cannot
be reconstructed from the retained artifacts.

## Reproduction

From the repository root:

```bash
python scripts/audit_historical_runtime.py \\
  --runs-root {runs_root} \\
  --checkpoint-root {checkpoint_root} \\
  --output-dir analysis_outputs/swm_cls_vs_cnn_fb_20260727
```

The script reads historical artifacts only. It rewrites this report and
`historical_full_run_times.csv` deterministically. It observed {len(rejected)} other
matching run directories that were excluded as incomplete, failed, or non-continuous.
"""
    return report


def main() -> None:
    args = parse_args()
    rows, rejected = audit_runs(args.runs_root, args.checkpoint_root)
    if not rows:
        raise SystemExit("No usable historical full runs found")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "historical_full_run_times.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report_path = args.output_dir / "runtime_evidence.md"
    report_path.write_text(
        render_report(rows, rejected, args.runs_root, args.checkpoint_root)
    )
    print(f"wrote {len(rows)} usable runs to {csv_path}")
    print(f"wrote evidence report to {report_path}")
    print(f"excluded {len(rejected)} matching incomplete/failed runs")


if __name__ == "__main__":
    main()
