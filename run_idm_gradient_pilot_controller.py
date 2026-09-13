#!/usr/bin/env python3
"""Launch a prepared IDM pilot only after the current baseline slots are free."""

from __future__ import annotations

import argparse
import csv
import dataclasses
from pathlib import Path
import shlex
import subprocess
import time
import typing as tp


@dataclasses.dataclass(frozen=True)
class Job:
    setting: str
    session: str
    gpu: int
    task: str
    run_dir: Path
    job_file: Path
    session_log: Path


def _read_tsv(path: Path) -> tp.List[tp.Dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def _load_jobs(index_path: Path) -> tp.List[Job]:
    jobs: tp.List[Job] = []
    for index_row in _read_tsv(index_path):
        setting = index_row["setting"]
        manifest = Path(index_row["manifest"])
        if not manifest.is_file():
            raise RuntimeError(f"Pilot manifest does not exist: {manifest}")
        for row in _read_tsv(manifest):
            if row["launch_mode"] != "parallel_task":
                raise RuntimeError(f"Pilot job is not independent: {row['job_file']}")
            job = Job(
                setting=setting,
                session=row["session"],
                gpu=int(row["gpu"]),
                task=row["task"],
                run_dir=Path(row["run_dir"]),
                job_file=Path(row["job_file"]),
                session_log=Path(row["session_log"]),
            )
            if not job.job_file.is_file():
                raise RuntimeError(f"Pilot job script does not exist: {job.job_file}")
            jobs.append(job)
    identities = {(job.setting, job.task) for job in jobs}
    if len(jobs) != 12 or len(identities) != 12:
        raise RuntimeError(
            f"Expected 12 unique setting/task pilot jobs, found {len(jobs)}"
        )
    if {job.task for job in jobs} != {
        "walker_flip",
        "cheetah_walk",
        "quadruped_walk",
    }:
        raise RuntimeError("Pilot index does not contain the expected three tasks")
    if {job.setting for job in jobs} != {
        "static1",
        "static10",
        "rho1pct",
        "rho5pct",
    }:
        raise RuntimeError("Pilot index does not contain the expected four settings")
    return jobs


def _queue_is_stopped(pid: int) -> bool:
    status_path = Path(f"/proc/{pid}/status")
    if not status_path.is_file():
        return False
    for line in status_path.read_text().splitlines():
        if line.startswith("State:"):
            return line.split()[1] in {"T", "t"}
    return False


def _require_stopped_queues(queue_pids: tp.Sequence[int]) -> None:
    unsafe = [pid for pid in queue_pids if not _queue_is_stopped(pid)]
    if unsafe:
        raise RuntimeError(f"Canonical queue shell(s) are not stopped: {unsafe}")


def _launcher_exit(run_dir: Path) -> tp.Optional[int]:
    launcher_log = run_dir / "launcher.log"
    if not launcher_log.is_file():
        return None
    for line in reversed(launcher_log.read_text(errors="replace").splitlines()):
        if "[done]" not in line or "exit=" not in line:
            continue
        try:
            return int(line.rsplit("exit=", 1)[1].split()[0])
        except ValueError:
            return None
    return None


def _pretrain_process_text() -> str:
    result = subprocess.run(
        ["pgrep", "-af", "url_benchmark/pretrain.py"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode not in {0, 1}:
        raise RuntimeError(f"pgrep failed with exit code {result.returncode}")
    return result.stdout


def _baseline_status(manifest_path: Path) -> tp.Tuple[int, int, tp.List[str]]:
    rows = _read_tsv(manifest_path)
    if len(rows) != 12:
        raise RuntimeError(f"Expected 12 baseline rows, found {len(rows)}")
    process_text = _pretrain_process_text()
    done = 0
    running = 0
    failures = []
    for row in rows:
        run_dir = Path(row["run_dir"])
        exit_code = _launcher_exit(run_dir)
        is_running = str(run_dir) in process_text
        if exit_code == 0 and not is_running:
            done += 1
        elif exit_code is not None and exit_code != 0:
            failures.append(f"{row['task']}: exit={exit_code}")
        elif is_running:
            running += 1
    return done, running, failures


def _gpu_free_mib(gpu: int) -> int:
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--id={gpu}",
            "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(result.stdout.strip().splitlines()[0])


def _tmux_has_session(session: str) -> bool:
    return subprocess.run(
        ["tmux", "has-session", "-t", session],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0


def _launch_job(job: Job) -> None:
    job.session_log.parent.mkdir(parents=True, exist_ok=True)
    command = (
        f"bash {shlex.quote(str(job.job_file))} "
        f"> {shlex.quote(str(job.session_log))} 2>&1"
    )
    subprocess.run(
        ["tmux", "new-session", "-d", "-s", job.session, command],
        check=True,
    )


def _wait_for_baselines(
    baseline_manifest: Path,
    queue_pids: tp.Sequence[int],
    poll_seconds: int,
) -> None:
    while True:
        _require_stopped_queues(queue_pids)
        done, running, failures = _baseline_status(baseline_manifest)
        if failures:
            raise RuntimeError(f"Baseline failure(s): {', '.join(failures)}")
        print(
            f"[wait] baseline_done={done}/12 baseline_running={running}/12",
            flush=True,
        )
        if done == 12:
            return
        time.sleep(poll_seconds)


def _launch_jobs(
    jobs: tp.Sequence[Job],
    queue_pids: tp.Sequence[int],
    min_free_mib: int,
    poll_seconds: int,
    launch_delay_seconds: int,
) -> None:
    for ordinal, job in enumerate(jobs, start=1):
        _require_stopped_queues(queue_pids)
        exit_code = _launcher_exit(job.run_dir)
        if exit_code == 0:
            print(f"[skip] completed {job.setting}/{job.task}", flush=True)
            continue
        if job.run_dir.exists():
            if _tmux_has_session(job.session):
                print(f"[skip] already running {job.setting}/{job.task}", flush=True)
                continue
            raise RuntimeError(f"Incomplete existing pilot run: {job.run_dir}")
        if _tmux_has_session(job.session):
            raise RuntimeError(f"Pilot tmux session already exists: {job.session}")

        while True:
            free_mib = _gpu_free_mib(job.gpu)
            if free_mib >= min_free_mib:
                break
            print(
                f"[gate] {job.setting}/{job.task} gpu={job.gpu} "
                f"free={free_mib}MiB need={min_free_mib}MiB",
                flush=True,
            )
            time.sleep(poll_seconds)
            _require_stopped_queues(queue_pids)

        print(
            f"[launch {ordinal}/{len(jobs)}] {job.setting}/{job.task} "
            f"gpu={job.gpu} free={free_mib}MiB",
            flush=True,
        )
        _launch_job(job)
        time.sleep(launch_delay_seconds)
        immediate_exit = _launcher_exit(job.run_dir)
        if immediate_exit not in {None, 0}:
            raise RuntimeError(
                f"Pilot failed immediately: {job.setting}/{job.task} exit={immediate_exit}"
            )
        if immediate_exit is None and not _tmux_has_session(job.session):
            raise RuntimeError(
                f"Pilot session disappeared before reporting status: {job.session}"
            )


def _monitor_jobs(jobs: tp.Sequence[Job], poll_seconds: int) -> None:
    while True:
        succeeded = 0
        running = 0
        for job in jobs:
            exit_code = _launcher_exit(job.run_dir)
            if exit_code == 0:
                succeeded += 1
            elif exit_code is not None:
                raise RuntimeError(
                    f"Pilot failed: {job.setting}/{job.task} exit={exit_code}"
                )
            elif _tmux_has_session(job.session):
                running += 1
            else:
                raise RuntimeError(
                    f"Pilot has no tmux session or completion marker: {job.session}"
                )
        print(
            f"[monitor] succeeded={succeeded}/12 running={running}/12",
            flush=True,
        )
        if succeeded == len(jobs):
            return
        time.sleep(poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--baseline-manifest", required=True, type=Path)
    parser.add_argument("--queue-pids", required=True)
    parser.add_argument("--min-free-mib", type=int, default=5000)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--launch-delay-seconds", type=int, default=45)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    queue_pids = [int(value) for value in args.queue_pids.split(",") if value]
    if len(queue_pids) != 12 or len(set(queue_pids)) != 12:
        raise ValueError("--queue-pids must contain 12 unique comma-separated PIDs")
    if args.min_free_mib <= 0 or args.poll_seconds <= 0:
        raise ValueError("memory and polling thresholds must be positive")
    if args.launch_delay_seconds < 0:
        raise ValueError("--launch-delay-seconds must be non-negative")

    jobs = _load_jobs(args.index)
    _require_stopped_queues(queue_pids)
    done, running, failures = _baseline_status(args.baseline_manifest)
    print(
        f"[check] pilot_jobs={len(jobs)} baseline_done={done}/12 "
        f"baseline_running={running}/12 failures={len(failures)}",
        flush=True,
    )
    if args.check_only:
        return
    if failures:
        raise RuntimeError(f"Baseline failure(s): {', '.join(failures)}")

    _wait_for_baselines(args.baseline_manifest, queue_pids, args.poll_seconds)
    _launch_jobs(
        jobs,
        queue_pids,
        args.min_free_mib,
        args.poll_seconds,
        args.launch_delay_seconds,
    )
    _monitor_jobs(jobs, args.poll_seconds)
    print("[done] all 12 IDM gradient pilot jobs succeeded", flush=True)


if __name__ == "__main__":
    main()
