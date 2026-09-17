#!/usr/bin/env python3
"""Crossed-axis bootstrap sensitivity for reward-label policy evaluation.

The main summarizer uses a hierarchical bootstrap that resamples evaluation
seeds separately inside each sampled label subset.  The experiment is a
balanced subset-seed x evaluation-seed factorial, so this sensitivity analysis
instead resamples the two axes independently and evaluates their Cartesian
product.  Target K and reference K always use the same draws.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


CellKey = Tuple[str, int, int, int]


def parse_int_csv(value: str) -> List[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("expected unique comma-separated integers")
    return values


def domain(task: str) -> str:
    return task.split("_", maxsplit=1)[0]


def domain_balanced_mean(
    task_values: Mapping[str, float], tasks: Sequence[str]
) -> float:
    domains = sorted({domain(task) for task in tasks})
    return float(
        np.mean(
            [
                np.mean(
                    [task_values[task] for task in tasks if domain(task) == name]
                )
                for name in domains
            ]
        )
    )


def read_values(input_dir: Path) -> Dict[CellKey, float]:
    values: Dict[CellKey, float] = {}
    paths = sorted(input_dir.glob("*/episodes.jsonl"))
    if not paths:
        raise RuntimeError(f"No */episodes.jsonl files found under {input_dir}")
    for path in paths:
        with path.open() as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("status") != "complete":
                    continue
                key = (
                    str(row["task"]),
                    int(row["k"]),
                    int(row["subset_seed"]),
                    int(row["eval_seed"]),
                )
                if key in values:
                    raise RuntimeError(f"Duplicate complete cell {key} in {path}:{line_number}")
                values[key] = float(row["episode_reward"])
    return values


def common_axes(
    values: Mapping[CellKey, float], task: str, k: int, reference_k: int
) -> Tuple[List[int], List[int]]:
    target_cells = {
        (subset, eval_seed)
        for row_task, row_k, subset, eval_seed in values
        if (row_task, row_k) == (task, k)
    }
    reference_cells = {
        (subset, eval_seed)
        for row_task, row_k, subset, eval_seed in values
        if (row_task, row_k) == (task, reference_k)
    }
    shared = target_cells & reference_cells
    subsets = sorted({subset for subset, _ in shared})
    eval_seeds = sorted({eval_seed for _, eval_seed in shared})
    expected = {(subset, eval_seed) for subset in subsets for eval_seed in eval_seeds}
    if not subsets or not eval_seeds or shared != expected:
        raise RuntimeError(
            f"Task {task}, K={k}: shared target/reference cells are not a "
            "complete crossed factorial"
        )
    return subsets, eval_seeds


def task_mean(
    values: Mapping[CellKey, float],
    task: str,
    k: int,
    subsets: Iterable[int],
    eval_seeds: Iterable[int],
) -> float:
    return float(
        np.mean(
            [
                values[(task, k, int(subset), int(eval_seed))]
                for subset in subsets
                for eval_seed in eval_seeds
            ]
        )
    )


def summarize_k(
    values: Mapping[CellKey, float],
    tasks: Sequence[str],
    k: int,
    reference_k: int,
    samples: int,
    seed: int,
) -> Dict[str, object]:
    axes = {task: common_axes(values, task, k, reference_k) for task in tasks}

    target_point: Dict[str, float] = {}
    reference_point: Dict[str, float] = {}
    for task in tasks:
        subsets, eval_seeds = axes[task]
        target_point[task] = task_mean(values, task, k, subsets, eval_seeds)
        reference_point[task] = task_mean(
            values, task, reference_k, subsets, eval_seeds
        )
    retention_point = domain_balanced_mean(
        {
            task: target_point[task] / reference_point[task]
            for task in tasks
        },
        tasks,
    )
    target_macro_point = domain_balanced_mean(target_point, tasks)
    reference_macro_point = domain_balanced_mean(reference_point, tasks)
    raw_macro_ratio_point = target_macro_point / reference_macro_point

    rng = np.random.RandomState(seed)
    retention_samples = np.empty(samples, dtype=np.float64)
    raw_macro_ratio_samples = np.empty(samples, dtype=np.float64)
    for replicate in range(samples):
        target_scores: Dict[str, float] = {}
        reference_scores: Dict[str, float] = {}
        for task in tasks:
            subsets, eval_seeds = axes[task]
            # The same eval draw is crossed with every sampled subset for this
            # task, and both axis draws are shared by target and reference.
            sampled_subsets = rng.choice(subsets, size=len(subsets), replace=True)
            sampled_eval_seeds = rng.choice(
                eval_seeds, size=len(eval_seeds), replace=True
            )
            target_scores[task] = task_mean(
                values, task, k, sampled_subsets, sampled_eval_seeds
            )
            reference_scores[task] = task_mean(
                values,
                task,
                reference_k,
                sampled_subsets,
                sampled_eval_seeds,
            )
        retention_samples[replicate] = domain_balanced_mean(
            {
                task: target_scores[task] / reference_scores[task]
                for task in tasks
            },
            tasks,
        )
        raw_macro_ratio_samples[replicate] = (
            domain_balanced_mean(target_scores, tasks)
            / domain_balanced_mean(reference_scores, tasks)
        )

    retention_low, retention_high = np.quantile(
        retention_samples, [0.025, 0.975]
    )
    raw_ratio_low, raw_ratio_high = np.quantile(
        raw_macro_ratio_samples, [0.025, 0.975]
    )
    return {
        "k": k,
        "reference_k": reference_k,
        "num_tasks": len(tasks),
        "num_subset_seeds": min(len(axes[task][0]) for task in tasks),
        "num_eval_seeds": min(len(axes[task][1]) for task in tasks),
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
        "mean_task_ratio_retention": retention_point,
        "mean_task_ratio_ci_low": float(retention_low),
        "mean_task_ratio_ci_high": float(retention_high),
        "target_domain_balanced_return": target_macro_point,
        "reference_domain_balanced_return": reference_macro_point,
        "ratio_of_domain_balanced_returns": raw_macro_ratio_point,
        "raw_macro_ratio_ci_low": float(raw_ratio_low),
        "raw_macro_ratio_ci_high": float(raw_ratio_high),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument(
        "--ks",
        type=parse_int_csv,
        default=parse_int_csv("1,4,16,64,256,1024,5120,20480"),
    )
    parser.add_argument("--reference-k", type=int, default=5120)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260802)
    args = parser.parse_args()
    if args.bootstrap_samples <= 0:
        parser.error("--bootstrap-samples must be positive")

    values = read_values(args.input_dir)
    tasks = sorted({task for task, _, _, _ in values})
    rows = [
        summarize_k(
            values,
            tasks,
            k,
            args.reference_k,
            args.bootstrap_samples,
            args.bootstrap_seed + index,
        )
        for index, k in enumerate(args.ks)
    ]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
