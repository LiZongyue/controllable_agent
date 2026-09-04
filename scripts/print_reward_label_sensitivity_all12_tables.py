#!/usr/bin/env python3
"""Print the final all-12 reward-label sensitivity tables as Markdown.

This is intentionally a read-only presentation helper.  It expects the strict
summarizer to have completed first and refuses to print a partial or differently
shaped experiment.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple


TASKS = (
    "cheetah_walk",
    "cheetah_walk_backward",
    "cheetah_run",
    "cheetah_run_backward",
    "walker_stand",
    "walker_walk",
    "walker_run",
    "walker_flip",
    "quadruped_stand",
    "quadruped_walk",
    "quadruped_run",
    "quadruped_jump",
)
KS = (1, 4, 16, 64, 256, 1024, 5120, 20480)
QUALITY = "iid_clean"
SUBSET_SEEDS = (0, 1, 2)
EVAL_SEEDS = (1101, 1102, 1103, 1104, 1105)


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        raise RuntimeError(f"Missing generated summary: {path}")
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def domain(task: str) -> str:
    return task.split("_", maxsplit=1)[0]


def require_close(actual: float, expected: float, context: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9):
        raise RuntimeError(
            f"Inconsistent generated summaries for {context}: "
            f"{actual} != {expected}"
        )


def require_exact_design(input_dir: Path) -> None:
    path = input_dir / "integrity_audit.json"
    if not path.is_file():
        raise RuntimeError(
            f"Missing {path}; run summarize_reward_label_sensitivity.py first"
        )
    audit = json.loads(path.read_text())
    factorial = audit.get("factorial", {})
    expected = {
        "tasks": list(TASKS),
        "qualities": [QUALITY],
        "ks": list(KS),
        "subset_seeds": list(SUBSET_SEEDS),
        "eval_seeds": list(EVAL_SEEDS),
        "expected_cells": len(TASKS) * len(KS) * len(SUBSET_SEEDS) * len(EVAL_SEEDS),
    }
    problems: List[str] = []
    if not audit.get("is_complete", False):
        problems.append("integrity_audit.is_complete is false")
    for key, expected_value in expected.items():
        actual_value = factorial.get(key)
        if key == "tasks":
            actual_value = sorted(actual_value or [])
            expected_value = sorted(expected_value)
        if actual_value != expected_value:
            problems.append(f"factorial.{key}={actual_value!r}, expected {expected_value!r}")
    for key in (
        "missing_cells",
        "duplicate_cells",
        "unexpected_cells",
        "malformed_complete_episode_rows",
    ):
        if int(factorial.get(key, -1)) != 0:
            problems.append(f"factorial.{key}={factorial.get(key)!r}")
    if problems:
        raise RuntimeError(
            "Refusing to print a partial or non-all-12 result: " + "; ".join(problems)
        )


def markdown_table(
    row_label: str,
    row_names: Iterable[str],
    values: Mapping[Tuple[str, int], float],
    digits: int,
) -> str:
    rows = [f"| {row_label} | " + " | ".join(str(k) for k in KS) + " |"]
    rows.append("|---|" + "---:|" * len(KS))
    for name in row_names:
        cells = [f"{values[(name, k)]:.{digits}f}" for k in KS]
        rows.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--reference-k", type=int, default=5120)
    parser.add_argument("--reward-digits", type=int, default=3)
    parser.add_argument("--retention-digits", type=int, default=3)
    args = parser.parse_args()
    if args.reference_k not in KS:
        parser.error(f"--reference-k must be one of {','.join(map(str, KS))}")

    input_dir = args.input_dir.resolve()
    require_exact_design(input_dir)

    task_rows = [
        row
        for row in read_csv(input_dir / "task_summary.csv")
        if row["quality"] == QUALITY
    ]
    task_means: Dict[Tuple[str, int], float] = {}
    for row in task_rows:
        key = (row["task"], int(row["k"]))
        if key in task_means:
            raise RuntimeError(f"Duplicate task-summary row: {key}")
        if int(row["num_rows"]) != len(SUBSET_SEEDS) * len(EVAL_SEEDS):
            raise RuntimeError(f"{key} has num_rows={row['num_rows']}, expected 15")
        if int(row["num_subset_seeds"]) != len(SUBSET_SEEDS):
            raise RuntimeError(f"{key} does not contain three label subsets")
        if int(row["num_eval_seeds"]) != len(EVAL_SEEDS):
            raise RuntimeError(f"{key} does not contain five evaluation seeds")
        task_means[key] = float(row["mean_return"])
    expected_task_keys = {(task, k) for task in TASKS for k in KS}
    if set(task_means) != expected_task_keys:
        raise RuntimeError(
            "task_summary.csv is not the exact 12 task x 8 K table: "
            f"missing={sorted(expected_task_keys - set(task_means))}, "
            f"unexpected={sorted(set(task_means) - expected_task_keys)}"
        )

    retention_rows = [
        row
        for row in read_csv(input_dir / "task_retention_summary.csv")
        if row["quality"] == QUALITY
        and int(row["reference_k"]) == args.reference_k
        and row["reference_quality"] == QUALITY
    ]
    task_retentions: Dict[Tuple[str, int], float] = {}
    for row in retention_rows:
        key = (row["task"], int(row["k"]))
        if key in task_retentions:
            raise RuntimeError(f"Duplicate task-retention row: {key}")
        if int(row["num_shared_subsets"]) != len(SUBSET_SEEDS):
            raise RuntimeError(f"{key} does not have three paired label subsets")
        if int(row["num_paired_eval_rows"]) != len(SUBSET_SEEDS) * len(EVAL_SEEDS):
            raise RuntimeError(f"{key} does not have 15 paired evaluation rows")
        retention = float(row["retention"])
        expected_retention = (
            task_means[key] / task_means[(row["task"], args.reference_k)]
        )
        require_close(retention, expected_retention, f"per-task retention {key}")
        task_retentions[key] = retention
    if set(task_retentions) != expected_task_keys:
        raise RuntimeError(
            "task_retention_summary.csv is not the exact 12 task x 8 K table"
        )

    domains = tuple(dict.fromkeys(domain(task) for task in TASKS))
    domain_means: Dict[Tuple[str, int], float] = {}
    all12_means: Dict[Tuple[str, int], float] = {}
    aggregate_retentions: Dict[Tuple[str, int], float] = {}
    for k in KS:
        for name in domains:
            scores = [task_means[(task, k)] for task in TASKS if domain(task) == name]
            domain_means[(name, k)] = sum(scores) / len(scores)
        all12_means[("all12_raw_mean", k)] = sum(
            domain_means[(name, k)] for name in domains
        ) / len(domains)
        per_domain_retentions = []
        for name in domains:
            scores = [
                task_retentions[(task, k)]
                for task in TASKS
                if domain(task) == name
            ]
            per_domain_retentions.append(sum(scores) / len(scores))
        aggregate_retentions[("all12_mean_task_ratio", k)] = sum(
            per_domain_retentions
        ) / len(per_domain_retentions)

    aggregate_rows = [
        row
        for row in read_csv(input_dir / "aggregate_summary.csv")
        if row["task_group"] == "forced_all12"
        and row["quality"] == QUALITY
        and int(row["reference_k"]) == args.reference_k
        and row["reference_quality"] == QUALITY
    ]
    aggregate_by_k = {int(row["k"]): row for row in aggregate_rows}
    if set(aggregate_by_k) != set(KS) or len(aggregate_rows) != len(KS):
        raise RuntimeError("aggregate_summary.csv lacks one forced_all12 row per K")
    for k, row in aggregate_by_k.items():
        if int(row["num_tasks"]) != len(TASKS):
            raise RuntimeError(f"forced_all12 K={k} reports num_tasks={row['num_tasks']}")
        require_close(
            float(row["macro_return"]),
            all12_means[("all12_raw_mean", k)],
            f"all-12 raw macro K={k}",
        )
        require_close(
            float(row["retention"]),
            aggregate_retentions[("all12_mean_task_ratio", k)],
            f"all-12 mean task-ratio retention K={k}",
        )

    print("## Per-task raw return means (3 label subsets x 5 evaluation seeds)\n")
    print(markdown_table("Task", TASKS, task_means, args.reward_digits))
    print("\n## Domain-balanced raw return means\n")
    summary_values = {**domain_means, **all12_means}
    print(
        markdown_table(
            "Aggregate",
            (*domains, "all12_raw_mean"),
            summary_values,
            args.reward_digits,
        )
    )
    print(f"\n## Per-task return retention versus K={args.reference_k}\n")
    print(
        markdown_table(
            "Task",
            TASKS,
            task_retentions,
            args.retention_digits,
        )
    )
    print("\n## All-12 mean of per-task return ratios\n")
    print(
        markdown_table(
            "Aggregate",
            ("all12_mean_task_ratio",),
            aggregate_retentions,
            args.retention_digits,
        )
    )


if __name__ == "__main__":
    main()
