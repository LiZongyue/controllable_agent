#!/usr/bin/env python3
"""Audit local SWM-CLS/CNN-FB checkpoints and evaluation records.

This script intentionally audits availability only.  It never interprets the
periodic training ``eval.csv`` as a strict reward-projection evaluation because
the training-time meta construction differs across DMC tasks.  The only legacy
record marked as strict reward projection is a valid 2M ``test_rewards.json``
produced by ``BaseWorkspace.finalize``.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml


SEEDS = (2009, 7532, 8164)
FRAMES = (500_000, 1_000_000, 1_500_000, 2_000_000)
TASKS = {
    "cheetah": ("walk", "walk_backward", "run", "run_backward"),
    "walker": ("stand", "walk", "run", "flip"),
    "quadruped": ("stand", "walk", "run", "jump"),
}
SWM_GPU = {2009: 3, 7532: 0, 8164: 5}


@dataclass(frozen=True)
class RunRecord:
    method: str
    seed: int
    domain: str
    task: str
    run_dir: Path
    checkpoint_dir: Path
    config_match: bool
    config_reason: str
    checkpoint_frames: tuple[int, ...]
    periodic_eval_frames: tuple[int, ...]
    final_reward_tasks: tuple[str, ...]
    final_reward_counts: dict[str, int]
    exit_code: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("/mnt/data_7tb/fanfeng/controllable_agent_runs"),
    )
    parser.add_argument(
        "--checkpoints-root",
        type=Path,
        default=Path("/mnt/data_7tb/fanfeng/controallable_agent_ckpt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis_outputs/swm_cls_vs_cnn_fb_20260727"),
    )
    return parser.parse_args()


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as handle:
        value = yaml.safe_load(handle)
    return value if isinstance(value, dict) else {}


def config_status(method: str, config: dict[str, Any]) -> tuple[bool, str]:
    agent = config.get("agent", {})
    shared = {
        "num_train_frames": 2_000_010,
        "action_repeat": 2,
        "eval_every_frames": 10_000,
        "frame_stack": 3,
        "num_seed_frames": 4_000,
        "update_encoder": True,
    }
    agent_shared = {"batch_size": 1024, "update_every_steps": 2, "lr": 1e-4}
    mismatches = [f"{key}={config.get(key)!r}" for key, value in shared.items() if config.get(key) != value]
    mismatches += [
        f"agent.{key}={agent.get(key)!r}"
        for key, value in agent_shared.items()
        if agent.get(key) != value
    ]
    if method == "SWM-CLS":
        expected = {"obs_type": "dino", "render_shape": [224, 224], "use_cls": True}
    else:
        expected = {"obs_type": "pixels", "render_shape": [84, 84]}
    mismatches += [f"{key}={config.get(key)!r}" for key, value in expected.items() if config.get(key) != value]
    return (not mismatches, "matches_requested_config" if not mismatches else ";".join(mismatches))


def checkpoint_frames(checkpoint_dir: Path) -> tuple[int, ...]:
    values = []
    for path in checkpoint_dir.glob("snapshot_*.pt") if checkpoint_dir.exists() else ():
        match = re.fullmatch(r"snapshot_(\d+)\.pt", path.name)
        if match:
            values.append(int(match.group(1)))
    return tuple(sorted(set(values)))


def eval_frames(eval_path: Path) -> tuple[int, ...]:
    if not eval_path.exists():
        return ()
    values: set[int] = set()
    try:
        with eval_path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("frame"):
                    values.add(int(float(row["frame"])))
    except (OSError, ValueError, csv.Error):
        return ()
    return tuple(sorted(values))


def final_rewards(path: Path) -> tuple[tuple[str, ...], dict[str, int]]:
    if not path.exists():
        return (), {}
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return (), {}
    if not isinstance(value, dict):
        return (), {}
    counts = {key: len(item) for key, item in value.items() if isinstance(item, list)}
    return tuple(sorted(counts)), counts


def exit_code(launcher_path: Path) -> str:
    if not launcher_path.exists():
        return "unknown"
    matches = re.findall(r"exit=(\d+)", launcher_path.read_text(errors="replace"))
    return matches[-1] if matches else "incomplete_or_unknown"


def make_record(
    method: str,
    seed: int,
    domain: str,
    task: str,
    run_dir: Path,
    checkpoints_root: Path,
) -> RunRecord:
    checkpoint_dir = checkpoints_root / run_dir.name
    match, reason = config_status(method, read_yaml(run_dir / ".hydra" / "config.yaml"))
    reward_tasks, reward_counts = final_rewards(run_dir / "test_rewards.json")
    return RunRecord(
        method=method,
        seed=seed,
        domain=domain,
        task=task,
        run_dir=run_dir,
        checkpoint_dir=checkpoint_dir,
        config_match=match,
        config_reason=reason,
        checkpoint_frames=checkpoint_frames(checkpoint_dir),
        periodic_eval_frames=eval_frames(run_dir / "eval.csv"),
        final_reward_tasks=reward_tasks,
        final_reward_counts=reward_counts,
        exit_code=exit_code(run_dir / "launcher.log"),
    )


def swm_record(
    seed: int, domain: str, task: str, runs_root: Path, checkpoints_root: Path
) -> RunRecord | None:
    name = f"20260426_123938_seed{seed}_{domain}_{task}_cuda{SWM_GPU[seed]}_dino_cls"
    run_dir = runs_root / name
    return make_record("SWM-CLS", seed, domain, task, run_dir, checkpoints_root) if run_dir.exists() else None


def cnn_candidates(seed: int, domain: str, task: str, runs_root: Path) -> Iterable[Path]:
    token = f"seed{seed}_{domain}_{task}_cnn_"
    for run_dir in runs_root.iterdir():
        if run_dir.is_dir() and "dmc_cnn" in run_dir.name and token in run_dir.name:
            yield run_dir


def cnn_record(
    seed: int, domain: str, task: str, runs_root: Path, checkpoints_root: Path
) -> RunRecord | None:
    candidates = [
        make_record("CNN-FB", seed, domain, task, path, checkpoints_root)
        for path in cnn_candidates(seed, domain, task, runs_root)
    ]
    if not candidates:
        return None

    def rank(record: RunRecord) -> tuple[int, int, int, int, str]:
        target_count = len(set(record.checkpoint_frames).intersection(FRAMES))
        max_checkpoint = max(record.checkpoint_frames, default=-1)
        max_eval = max(record.periodic_eval_frames, default=-1)
        has_final = int(bool(record.final_reward_tasks))
        return target_count, max_checkpoint, has_final, max_eval, record.run_dir.name

    return max(candidates, key=rank)


def cell_rows(records: dict[tuple[str, int, str, str], RunRecord | None]) -> list[dict[str, Any]]:
    rows = []
    for method in ("SWM-CLS", "CNN-FB"):
        for seed in SEEDS:
            for domain, domain_tasks in TASKS.items():
                for task in domain_tasks:
                    full_task = f"{domain}_{task}"
                    record = records[(method, seed, domain, task)]
                    for frame in FRAMES:
                        checkpoint_exists = bool(record and frame in record.checkpoint_frames)
                        periodic_exists = bool(record and frame in record.periodic_eval_frames)
                        final_count = record.final_reward_counts.get(full_task, 0) if record else 0
                        legacy_final = bool(
                            record
                            and frame == 2_000_000
                            and checkpoint_exists
                            and final_count == 10
                        )
                        if legacy_final:
                            strict_reason = (
                                "available_legacy_finalize_reward_projection_10_episodes;"
                                "rerun_all_frames_for_cross_frame_calibration_consistency"
                            )
                        elif not record:
                            strict_reason = "no_matching_run"
                        elif not checkpoint_exists:
                            strict_reason = "exact_checkpoint_missing"
                        else:
                            strict_reason = "checkpoint_exists_but_no_uniform_reward_projection_evaluation"
                        rows.append(
                            {
                                "method": method,
                                "seed": seed,
                                "domain": domain,
                                "task": full_task,
                                "environment_frames": frame,
                                "frame_millions": frame / 1_000_000,
                                "run_dir": str(record.run_dir) if record else "",
                                "checkpoint_dir": str(record.checkpoint_dir) if record else "",
                                "checkpoint_path": str(record.checkpoint_dir / f"snapshot_{frame}.pt") if record else "",
                                "checkpoint_exists": checkpoint_exists,
                                "config_matches_requested_setup": record.config_match if record else False,
                                "config_status": record.config_reason if record else "no_matching_run",
                                "run_exit_code": record.exit_code if record else "no_matching_run",
                                "periodic_eval_path": str(record.run_dir / "eval.csv") if record else "",
                                "periodic_eval_frame_exists": periodic_exists,
                                "periodic_eval_protocol": (
                                    "mixed_training_time_meta_protocol_not_uniform_reward_projection"
                                    if periodic_exists
                                    else "not_available"
                                ),
                                "periodic_eval_strict_compliant": False,
                                "legacy_final_reward_projection_path": (
                                    str(record.run_dir / "test_rewards.json") if record else ""
                                ),
                                "legacy_final_reward_projection_episode_count": final_count,
                                "legacy_final_reward_projection_available": legacy_final,
                                "strict_protocol_available": legacy_final,
                                "strict_protocol_reason": strict_reason,
                                "cross_frame_comparable_without_reevaluation": False,
                            }
                        )
    return rows


def manifest_rows(records: dict[tuple[str, int, str, str], RunRecord | None]) -> list[dict[str, Any]]:
    rows = []
    for key in sorted(records):
        record = records[key]
        if record is None:
            continue
        target_frames = sorted(set(record.checkpoint_frames).intersection(FRAMES))
        # The availability CSV retains evidence-only/failed runs.  The reusable
        # manifest is intentionally narrower: every listed run must contain at
        # least one requested checkpoint and match the audited scalar setup.
        if not target_frames or not record.config_match:
            continue
        strict_tasks = [task for task, count in record.final_reward_counts.items() if count == 10]
        rows.append(
            {
                "method": record.method,
                "seed": record.seed,
                "domain": record.domain,
                "training_task": f"{record.domain}_{record.task}",
                "run_dir": str(record.run_dir),
                "checkpoint_dir": str(record.checkpoint_dir),
                "config_matches_requested_setup": record.config_match,
                "config_status": record.config_reason,
                "run_exit_code": record.exit_code,
                "available_target_checkpoint_frames": ";".join(map(str, target_frames)),
                "all_snapshot_frames": ";".join(map(str, record.checkpoint_frames)),
                "available_periodic_target_frames": ";".join(
                    map(str, sorted(set(record.periodic_eval_frames).intersection(FRAMES)))
                ),
                "periodic_eval_strict_compliant": False,
                "legacy_final_reward_projection_tasks_10ep": ";".join(sorted(strict_tasks)),
                "reusable_for_strict_checkpoint_reevaluation": True,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write empty audit: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_blockers(path: Path, cells: list[dict[str, Any]], manifest: list[dict[str, Any]]) -> None:
    checkpoint_counts: dict[tuple[str, int], int] = {}
    strict_counts: dict[tuple[str, int], int] = {}
    for method in ("SWM-CLS", "CNN-FB"):
        for frame in FRAMES:
            selected = [r for r in cells if r["method"] == method and r["environment_frames"] == frame]
            checkpoint_counts[(method, frame)] = sum(bool(r["checkpoint_exists"]) for r in selected)
            strict_counts[(method, frame)] = sum(bool(r["strict_protocol_available"]) for r in selected)

    lines = [
        "# Sample-efficiency asset audit and blockers",
        "",
        "This audit covers exactly 288 cells: 2 methods × 3 seeds × 12 tasks × 4 frame points. "
        "It audits local files only and does not synthesize performance scores.",
        "",
        "## Availability summary",
        "",
        "| Method | Frame | Exact checkpoints | Existing strict RP results |",
        "|---|---:|---:|---:|",
    ]
    for method in ("SWM-CLS", "CNN-FB"):
        for frame in FRAMES:
            lines.append(
                f"| {method} | {frame / 1_000_000:g}M | {checkpoint_counts[(method, frame)]}/36 | "
                f"{strict_counts[(method, frame)]}/36 |"
            )
    lines += [
        "",
        "## Blocking facts",
        "",
        "- No `snapshot_500000.pt` exists anywhere under the configured checkpoint root. Training logs show "
        "that these files were once written, but they were subsequently removed.",
        "- SWM-CLS has all 1.0M, 1.5M, and 2.0M checkpoints for seeds 2009, 7532, and 8164.",
        "- CNN-FB was originally scheduled only for seeds 2009 and 7532. No matching DMC CNN-FB run "
        "exists for seed 8164.",
        "- CNN-FB checkpoint coverage is incomplete even for seeds 2009 and 7532; consult "
        "`checkpoint_eval_availability.csv` for every missing cell.",
        "- Periodic `eval.csv` rows are not strict-protocol results. The training code uses registered goal "
        "vectors for Walker/Quadruped stand, walk, and run, while other tasks infer metadata from rewards.",
        "- Valid 2M `test_rewards.json` files are marked as legacy strict reward projection because "
        "`finalize()` forces a custom reward for all four domain tasks. They used each live run's original "
        "replay buffer. Checkpoints did not save replay buffers, so the exact protocol cannot be reproduced "
        "at intermediate checkpoints from the checkpoint files alone.",
        "- Consequently, all available checkpoints should be reevaluated with one predeclared common "
        "calibration protocol before computing cross-frame curves or AUC. Existing 2M results must not be "
        "mixed with newly evaluated intermediate checkpoints without labeling the source change.",
        "- Current `url_benchmark/pretrain.py` instantiates DINOv3, whereas these historical SWM runs used "
        "`facebook/dinov2-base`. Re-evaluation must explicitly load DINOv2 (as the existing sensitivity and "
        "robust-evaluation utilities do).",
        "",
        "## Interpretation of strict fields",
        "",
        "`strict_protocol_available=true` only means that the cell has a 2M checkpoint and a valid "
        "10-episode task entry in its legacy `test_rewards.json`. "
        "`cross_frame_comparable_without_reevaluation` is deliberately false for every cell.",
        "",
        "## Reproduction",
        "",
        "```bash",
        "python scripts/audit_swm_cnn_sample_efficiency_assets.py \\",
        "  --output-dir analysis_outputs/swm_cls_vs_cnn_fb_20260727",
        "```",
        "",
        "The availability-only LaTeX table and coverage figure are generated from the audited CSV with:",
        "",
        "```bash",
        "MPLCONFIGDIR=/tmp/mpl_swm_cnn_availability \\",
        "python scripts/plot_swm_cnn_sample_efficiency_availability.py \\",
        "  --input-csv analysis_outputs/swm_cls_vs_cnn_fb_20260727/checkpoint_eval_availability.csv \\",
        "  --output-dir analysis_outputs/swm_cls_vs_cnn_fb_20260727",
        "```",
        "",
        "These assets visualize availability and protocol coverage only; they are not learning curves.",
        "",
        f"Selected matching local runs recorded in the manifest: {len(manifest)}.",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records: dict[tuple[str, int, str, str], RunRecord | None] = {}
    for seed in SEEDS:
        for domain, domain_tasks in TASKS.items():
            for task in domain_tasks:
                records[("SWM-CLS", seed, domain, task)] = swm_record(
                    seed, domain, task, args.runs_root, args.checkpoints_root
                )
                records[("CNN-FB", seed, domain, task)] = cnn_record(
                    seed, domain, task, args.runs_root, args.checkpoints_root
                )

    cells = cell_rows(records)
    manifest = manifest_rows(records)
    if len(cells) != 288:
        raise RuntimeError(f"Expected 288 audit cells, got {len(cells)}")
    write_csv(args.output_dir / "checkpoint_eval_availability.csv", cells)
    write_csv(args.output_dir / "reusable_run_manifest.csv", manifest)
    write_blockers(args.output_dir / "sample_efficiency_blockers.md", cells, manifest)
    print(
        json.dumps(
            {
                "cells": len(cells),
                "manifest_runs": len(manifest),
                "output_dir": str(args.output_dir.resolve()),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
