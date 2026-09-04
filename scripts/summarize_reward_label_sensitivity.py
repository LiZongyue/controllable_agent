#!/usr/bin/env python3
"""Summarize and plot reward-label sensitivity experiment outputs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


REWARD_SPECIFIED_TASKS = (
    "cheetah_walk",
    "cheetah_walk_backward",
    "cheetah_run",
    "cheetah_run_backward",
    "walker_flip",
    "quadruped_jump",
)

QUALITY_LABELS = {
    "iid_clean": "Uniform clean",
    "stratified_clean": "Stratified clean",
    "correlated_clean": "Contiguous low-coverage",
    "iid_corrupt20": "Uniform + 20% shuffled labels",
    "iid_corrupt50": "Uniform + 50% shuffled labels",
}

PROVENANCE_FIELDS = (
    "bank_id",
    "checkpoint_fingerprint",
    "checkpoint",
    "eval_condition",
)

CELL_FIELDS = ("task", "quality", "k", "subset_seed", "eval_seed")
META_LINK_FIELDS = (
    "task",
    "quality",
    "k",
    "subset_seed",
    "bank_id",
    "checkpoint_fingerprint",
)
EPISODE_PROVENANCE_FIELDS = (
    "bank_id",
    "checkpoint_fingerprint",
    "eval_condition",
)
META_PROVENANCE_FIELDS = (
    "bank_id",
    "checkpoint_fingerprint",
    "checkpoint",
)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def parse_str_csv(value: str) -> List[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected a nonempty comma-separated list")
    if len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("expected values must be unique")
    return values


def parse_int_csv(value: str) -> List[int]:
    strings = parse_str_csv(value)
    try:
        return [int(item) for item in strings]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "expected a comma-separated list of integers"
        ) from exc


def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Optional[Sequence[str]] = None,
) -> None:
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row})
    if not fieldnames:
        path.write_text("")
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def quality_label(quality: str) -> str:
    return QUALITY_LABELS.get(quality, quality)


def domain(task: str) -> str:
    return task.split("_", maxsplit=1)[0]


def percentile_interval(values: np.ndarray) -> Tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan"), float("nan")
    low, high = np.quantile(finite, [0.025, 0.975])
    return float(low), float(high)


def validate_provenance(rows: Sequence[Mapping[str, Any]]) -> None:
    """Reject accidental mixtures of calibration/evaluation provenance per task."""

    by_task: Dict[str, Dict[str, set]] = defaultdict(lambda: defaultdict(set))
    for row in rows:
        if row.get("status") != "complete" or "task" not in row:
            continue
        task = str(row["task"])
        for field in PROVENANCE_FIELDS:
            if field in row:
                value = json.dumps(row[field], sort_keys=True, default=str)
                by_task[task][field].add(value)

    conflicts: List[str] = []
    for task, fields in sorted(by_task.items()):
        for field, values in sorted(fields.items()):
            if len(values) > 1:
                conflicts.append(f"{task}.{field}={sorted(values)}")
    if conflicts:
        raise RuntimeError(
            "Mixed provenance detected within a task; refusing to aggregate: "
            + "; ".join(conflicts)
        )


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def required_provenance_audit(
    episode_rows: Sequence[Mapping[str, Any]],
    meta_rows: Sequence[Mapping[str, Any]],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Audit record-type-specific provenance required for complete rows."""

    issues: List[Dict[str, Any]] = []
    record_specs = (
        ("episode", episode_rows, EPISODE_PROVENANCE_FIELDS, "episode_cell_id"),
        ("meta", meta_rows, META_PROVENANCE_FIELDS, "meta_cell_id"),
    )
    complete_counts: Dict[str, int] = {}
    for record_type, rows, required_fields, identity_field in record_specs:
        complete_counts[record_type] = 0
        for row_index, row in enumerate(rows):
            if row.get("status") != "complete":
                continue
            complete_counts[record_type] += 1
            missing_fields = [
                field
                for field in required_fields
                if field not in row or _is_missing(row.get(field))
            ]
            if missing_fields:
                issues.append(
                    {
                        "record_type": record_type,
                        "row_index": row_index,
                        "task": row.get("task", ""),
                        "record_id": row.get(identity_field, ""),
                        "missing_fields": ",".join(missing_fields),
                    }
                )
    audit = {
        "complete_episode_rows": complete_counts["episode"],
        "complete_meta_rows": complete_counts["meta"],
        "records_missing_required_provenance": len(issues),
        "is_complete": not issues,
    }
    return audit, issues


def _normalized_link_value(
    row: Mapping[str, Any], field: str
) -> Tuple[bool, Any]:
    if field not in row or _is_missing(row.get(field)):
        return False, None
    value = row[field]
    if field in {"k", "subset_seed"}:
        try:
            value = int(value)
        except (TypeError, ValueError):
            return False, value
    else:
        value = str(value)
    return True, value


def episode_meta_link_audit(
    episode_rows: Sequence[Mapping[str, Any]],
    meta_rows: Sequence[Mapping[str, Any]],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Require every complete episode to reference one consistent complete meta."""

    complete_meta_by_id: Dict[
        str, List[Tuple[int, Mapping[str, Any]]]
    ] = defaultdict(list)
    issues: List[Dict[str, Any]] = []
    complete_meta_count = 0
    for row_index, row in enumerate(meta_rows):
        if row.get("status") != "complete":
            continue
        complete_meta_count += 1
        meta_cell_id = row.get("meta_cell_id")
        if _is_missing(meta_cell_id):
            issues.append(
                {
                    "issue_type": "complete_meta_missing_id",
                    "episode_row_index": "",
                    "episode_cell_id": "",
                    "meta_cell_id": "",
                    "task": row.get("task", ""),
                    "mismatched_fields": "meta_cell_id",
                    "detail": f"meta row index {row_index}",
                }
            )
            continue
        complete_meta_by_id[str(meta_cell_id)].append((row_index, row))

    duplicate_meta_ids = {
        meta_cell_id: definitions
        for meta_cell_id, definitions in complete_meta_by_id.items()
        if len(definitions) > 1
    }
    for meta_cell_id, definitions in sorted(duplicate_meta_ids.items()):
        issues.append(
            {
                "issue_type": "duplicate_complete_meta_id",
                "episode_row_index": "",
                "episode_cell_id": "",
                "meta_cell_id": meta_cell_id,
                "task": definitions[0][1].get("task", ""),
                "mismatched_fields": "",
                "detail": json.dumps(
                    {"meta_row_indices": [index for index, _ in definitions]},
                    sort_keys=True,
                ),
            }
        )

    complete_episode_count = 0
    valid_episode_links = 0
    episodes_with_link_issues = 0
    for row_index, episode in enumerate(episode_rows):
        if episode.get("status") != "complete":
            continue
        complete_episode_count += 1
        meta_cell_id = episode.get("meta_cell_id")
        base_issue = {
            "episode_row_index": row_index,
            "episode_cell_id": episode.get("episode_cell_id", ""),
            "meta_cell_id": "" if _is_missing(meta_cell_id) else str(meta_cell_id),
            "task": episode.get("task", ""),
            "mismatched_fields": "",
        }
        if _is_missing(meta_cell_id):
            issues.append(
                {
                    **base_issue,
                    "issue_type": "episode_missing_meta_id",
                    "detail": "complete episode has no meta_cell_id",
                }
            )
            episodes_with_link_issues += 1
            continue

        definitions = complete_meta_by_id.get(str(meta_cell_id), [])
        if not definitions:
            issues.append(
                {
                    **base_issue,
                    "issue_type": "missing_complete_meta",
                    "detail": "no complete meta row has this meta_cell_id",
                }
            )
            episodes_with_link_issues += 1
            continue
        if len(definitions) != 1:
            issues.append(
                {
                    **base_issue,
                    "issue_type": "ambiguous_complete_meta",
                    "detail": f"found {len(definitions)} complete meta rows",
                }
            )
            episodes_with_link_issues += 1
            continue

        _, meta = definitions[0]
        mismatched_fields: List[str] = []
        episode_values: Dict[str, Any] = {}
        meta_values: Dict[str, Any] = {}
        for field in META_LINK_FIELDS:
            episode_present, episode_value = _normalized_link_value(episode, field)
            meta_present, meta_value = _normalized_link_value(meta, field)
            episode_values[field] = episode_value if episode_present else "<missing>"
            meta_values[field] = meta_value if meta_present else "<missing>"
            if not episode_present or not meta_present or episode_value != meta_value:
                mismatched_fields.append(field)
        if mismatched_fields:
            issues.append(
                {
                    **base_issue,
                    "issue_type": "episode_meta_field_mismatch",
                    "mismatched_fields": ",".join(mismatched_fields),
                    "detail": json.dumps(
                        {"episode": episode_values, "meta": meta_values},
                        sort_keys=True,
                        default=str,
                    ),
                }
            )
            episodes_with_link_issues += 1
            continue
        valid_episode_links += 1

    audit = {
        "complete_episode_rows": complete_episode_count,
        "complete_meta_rows": complete_meta_count,
        "unique_complete_meta_ids": len(complete_meta_by_id),
        "duplicate_complete_meta_ids": len(duplicate_meta_ids),
        "valid_episode_meta_links": valid_episode_links,
        "episodes_with_link_issues": episodes_with_link_issues,
        "total_link_issue_records": len(issues),
        "is_complete": not issues,
    }
    return audit, issues


def _resolve_dimension(
    expected: Optional[Sequence[Any]],
    inferred: Sequence[Any],
    name: str,
) -> Tuple[List[Any], str]:
    if expected is None:
        return sorted(set(inferred)), "inferred"
    values = list(expected)
    if not values:
        raise ValueError(f"Explicit expected {name} must be nonempty")
    if len(values) != len(set(values)):
        raise ValueError(f"Explicit expected {name} must be unique")
    return values, "explicit"


def factorial_audit(
    episode_rows: Sequence[Mapping[str, Any]],
    auxiliary_rows: Sequence[Mapping[str, Any]],
    requested_reference_ks: Sequence[int],
    directory_tasks: Sequence[str] = (),
    expected_tasks: Optional[Sequence[str]] = None,
    expected_qualities: Optional[Sequence[str]] = None,
    expected_ks: Optional[Sequence[int]] = None,
    expected_subset_seeds: Optional[Sequence[int]] = None,
    expected_eval_seeds: Optional[Sequence[int]] = None,
) -> Tuple[
    Dict[str, Any],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    """Audit the task x quality x K x subset x evaluation factorial.

    Any explicit expected dimension is authoritative. Dimensions without an
    explicit value are inferred from available records; requested reference K
    values are added only when K itself is inferred.
    """

    dimension_rows = list(episode_rows) + list(auxiliary_rows)
    inferred_tasks = list(directory_tasks) + [
        str(row["task"]) for row in dimension_rows if "task" in row
    ]
    tasks, tasks_source = _resolve_dimension(
        expected_tasks,
        inferred_tasks,
        "tasks",
    )
    qualities, qualities_source = _resolve_dimension(
        expected_qualities,
        [str(row["quality"]) for row in dimension_rows if "quality" in row],
        "qualities",
    )
    inferred_ks = [int(row["k"]) for row in dimension_rows if "k" in row]
    if expected_ks is None:
        inferred_ks.extend(int(value) for value in requested_reference_ks)
    ks, ks_source = _resolve_dimension(expected_ks, inferred_ks, "ks")
    subset_seeds, subset_seeds_source = _resolve_dimension(
        expected_subset_seeds,
        [
            int(row["subset_seed"])
            for row in dimension_rows
            if "subset_seed" in row
        ],
        "subset seeds",
    )
    eval_seeds, eval_seeds_source = _resolve_dimension(
        expected_eval_seeds,
        [int(row["eval_seed"]) for row in dimension_rows if "eval_seed" in row],
        "evaluation seeds",
    )

    completed_keys = [
        tuple(
            str(row[field]) if field in {"task", "quality"} else int(row[field])
            for field in CELL_FIELDS
        )
        for row in episode_rows
        if row.get("status") == "complete" and all(field in row for field in CELL_FIELDS)
    ]
    malformed_complete_rows = sum(
        row.get("status") == "complete"
        and not all(field in row for field in CELL_FIELDS)
        for row in episode_rows
    )
    counts = Counter(completed_keys)
    observed = set(counts)
    expected_cells_set = {
        (task, quality, k, subset_seed, eval_seed)
        for task in tasks
        for quality in qualities
        for k in ks
        for subset_seed in subset_seeds
        for eval_seed in eval_seeds
    }
    missing: List[Dict[str, Any]] = []
    for key in sorted(expected_cells_set - observed):
        missing.append(dict(zip(CELL_FIELDS, key)))
    duplicates = [
        {**dict(zip(CELL_FIELDS, key)), "count": count}
        for key, count in sorted(counts.items())
        if count > 1
    ]
    unexpected = [
        {**dict(zip(CELL_FIELDS, key)), "count": counts[key]}
        for key in sorted(observed - expected_cells_set)
    ]
    expected_cells = len(expected_cells_set)
    dimension_sources = {
        "tasks": tasks_source,
        "qualities": qualities_source,
        "ks": ks_source,
        "subset_seeds": subset_seeds_source,
        "eval_seeds": eval_seeds_source,
    }
    audit = {
        "tasks": tasks,
        "qualities": qualities,
        "ks": ks,
        "subset_seeds": subset_seeds,
        "eval_seeds": eval_seeds,
        "dimension_sources": dimension_sources,
        "expected_cells": expected_cells,
        "observed_unique_complete_cells": len(observed),
        "observed_expected_unique_complete_cells": len(observed & expected_cells_set),
        "missing_cells": len(missing),
        "duplicate_cells": len(duplicates),
        "unexpected_cells": len(unexpected),
        "malformed_complete_episode_rows": int(malformed_complete_rows),
        "is_complete": bool(
            expected_cells > 0
            and not missing
            and not duplicates
            and not unexpected
            and not malformed_complete_rows
        ),
        "inference_note": (
            "Each explicit expected dimension is authoritative; remaining dimensions "
            "are inferred from episode/meta/error records. Requested reference K "
            "values augment K only when expected K is not explicit."
        ),
    }
    return audit, missing, duplicates, unexpected


def build_value_map(
    rows: Sequence[Mapping[str, Any]],
) -> Dict[Tuple[str, str, int, int], Dict[int, float]]:
    values: Dict[Tuple[str, str, int, int], Dict[int, float]] = defaultdict(dict)
    for row in rows:
        if row.get("status") != "complete":
            continue
        key = (
            str(row["task"]),
            str(row["quality"]),
            int(row["k"]),
            int(row["subset_seed"]),
        )
        values[key][int(row["eval_seed"])] = float(row["episode_reward"])
    return dict(values)


def task_cell_means(
    values: Mapping[Tuple[str, str, int, int], Mapping[int, float]],
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, int], List[float]] = defaultdict(list)
    subset_grouped: Dict[Tuple[str, str, int], set] = defaultdict(set)
    eval_grouped: Dict[Tuple[str, str, int], set] = defaultdict(set)
    for (task, quality, k, subset_seed), eval_values in values.items():
        key = (task, quality, k)
        grouped[key].extend(eval_values.values())
        subset_grouped[key].add(subset_seed)
        eval_grouped[key].update(eval_values)
    rows: List[Dict[str, Any]] = []
    for (task, quality, k), scores in sorted(grouped.items()):
        array = np.asarray(scores, dtype=np.float64)
        rows.append(
            {
                "task": task,
                "domain": domain(task),
                "quality": quality,
                "quality_label": quality_label(quality),
                "k": k,
                "mean_return": float(array.mean()),
                "std_return": float(array.std()),
                "num_rows": int(array.size),
                "num_subset_seeds": len(subset_grouped[(task, quality, k)]),
                "num_eval_seeds": len(eval_grouped[(task, quality, k)]),
            }
        )
    return rows


def domain_balanced_mean(
    task_values: Mapping[str, float],
    tasks: Sequence[str],
    require_all_tasks: bool,
) -> float:
    if require_all_tasks and any(
        task not in task_values or not np.isfinite(task_values[task]) for task in tasks
    ):
        return float("nan")
    domain_scores: List[float] = []
    for domain_name in sorted({domain(task) for task in tasks}):
        scores = [
            float(task_values[task])
            for task in tasks
            if domain(task) == domain_name
            and task in task_values
            and np.isfinite(task_values[task])
        ]
        if scores:
            domain_scores.append(float(np.mean(scores)))
    return float(np.mean(domain_scores)) if domain_scores else float("nan")


def task_points(
    values: Mapping[Tuple[str, str, int, int], Mapping[int, float]],
    tasks: Sequence[str],
    quality: str,
    k: int,
) -> Dict[str, float]:
    output: Dict[str, float] = {}
    for task in tasks:
        subset_scores: List[float] = []
        for (row_task, row_quality, row_k, _), eval_values in values.items():
            if (row_task, row_quality, row_k) == (task, quality, k) and eval_values:
                subset_scores.append(float(np.mean(list(eval_values.values()))))
        if subset_scores:
            output[task] = float(np.mean(subset_scores))
    return output


def macro_point(
    values: Mapping[Tuple[str, str, int, int], Mapping[int, float]],
    tasks: Sequence[str],
    quality: str,
    k: int,
) -> float:
    return domain_balanced_mean(
        task_points(values, tasks, quality, k),
        tasks,
        require_all_tasks=False,
    )


def paired_cells_for_task(
    values: Mapping[Tuple[str, str, int, int], Mapping[int, float]],
    task: str,
    quality: str,
    k: int,
    reference_k: int,
) -> List[Tuple[Mapping[int, float], Mapping[int, float], List[int]]]:
    target_subsets = {
        subset
        for row_task, row_quality, row_k, subset in values
        if (row_task, row_quality, row_k) == (task, quality, k)
    }
    reference_subsets = {
        subset
        for row_task, row_quality, row_k, subset in values
        if (row_task, row_quality, row_k) == (task, "iid_clean", reference_k)
    }
    cells: List[Tuple[Mapping[int, float], Mapping[int, float], List[int]]] = []
    for subset in sorted(target_subsets & reference_subsets):
        target_eval = values[(task, quality, k, subset)]
        reference_eval = values[(task, "iid_clean", reference_k, subset)]
        shared_eval_seeds = sorted(set(target_eval) & set(reference_eval))
        if shared_eval_seeds:
            cells.append((target_eval, reference_eval, shared_eval_seeds))
    return cells


def paired_task_points(
    values: Mapping[Tuple[str, str, int, int], Mapping[int, float]],
    tasks: Sequence[str],
    quality: str,
    k: int,
    reference_k: int,
) -> Dict[str, Dict[str, float]]:
    """Compute equally subset-weighted task points on paired evaluation seeds."""

    output: Dict[str, Dict[str, float]] = {}
    for task in tasks:
        cells = paired_cells_for_task(values, task, quality, k, reference_k)
        target_subset_means: List[float] = []
        reference_subset_means: List[float] = []
        paired_eval_count = 0
        for target_eval, reference_eval, shared_eval_seeds in cells:
            target_subset_means.append(
                float(np.mean([target_eval[eval_seed] for eval_seed in shared_eval_seeds]))
            )
            reference_subset_means.append(
                float(np.mean([reference_eval[eval_seed] for eval_seed in shared_eval_seeds]))
            )
            paired_eval_count += len(shared_eval_seeds)
        if not target_subset_means:
            continue
        target = float(np.mean(target_subset_means))
        reference = float(np.mean(reference_subset_means))
        output[task] = {
            "target": target,
            "reference": reference,
            "retention": target / reference if abs(reference) > 1e-12 else float("nan"),
            "num_shared_subsets": float(len(target_subset_means)),
            "num_paired_eval_rows": float(paired_eval_count),
        }
    return output


def bootstrap_macro_and_retention(
    values: Mapping[Tuple[str, str, int, int], Mapping[int, float]],
    tasks: Sequence[str],
    quality: str,
    k: int,
    reference_k: int,
    samples: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Paired hierarchical bootstrap over subsets and evaluation seeds.

    Tasks are treated as the fixed benchmark suite. For every task and bootstrap
    replicate, the original number S of shared calibration subsets is sampled
    with replacement. Within each selected subset, its original number E of
    shared evaluation seeds is sampled with replacement, using identical seed
    draws for the target and reference cells. Retention is computed per task
    before domain-balanced aggregation.
    """

    rng = np.random.RandomState(seed)
    macro_samples: List[float] = []
    retention_samples: List[float] = []

    paired_cells: Dict[
        str,
        List[Tuple[Mapping[int, float], Mapping[int, float], List[int]]],
    ] = {}
    for task in tasks:
        cells = paired_cells_for_task(values, task, quality, k, reference_k)
        if not cells:
            return np.asarray([]), np.asarray([])
        paired_cells[task] = cells

    for _ in range(samples):
        target_task_scores: Dict[str, float] = {}
        retention_task_scores: Dict[str, float] = {}
        for task in tasks:
            cells = paired_cells[task]
            sampled_cell_indices = rng.choice(len(cells), size=len(cells), replace=True)
            target_subset_scores: List[float] = []
            reference_subset_scores: List[float] = []
            for cell_index in sampled_cell_indices:
                target_eval, reference_eval, shared_eval_seeds = cells[int(cell_index)]
                sampled_eval_seeds = rng.choice(
                    shared_eval_seeds,
                    size=len(shared_eval_seeds),
                    replace=True,
                )
                target_subset_scores.append(
                    float(
                        np.mean(
                            [target_eval[int(eval_seed)] for eval_seed in sampled_eval_seeds]
                        )
                    )
                )
                reference_subset_scores.append(
                    float(
                        np.mean(
                            [
                                reference_eval[int(eval_seed)]
                                for eval_seed in sampled_eval_seeds
                            ]
                        )
                    )
                )
            target = float(np.mean(target_subset_scores))
            reference = float(np.mean(reference_subset_scores))
            target_task_scores[task] = target
            retention_task_scores[task] = (
                target / reference if abs(reference) > 1e-12 else float("nan")
            )
        macro_samples.append(
            domain_balanced_mean(target_task_scores, tasks, require_all_tasks=True)
        )
        retention_samples.append(
            domain_balanced_mean(retention_task_scores, tasks, require_all_tasks=True)
        )
    return np.asarray(macro_samples), np.asarray(retention_samples)


def summarize_aggregate(
    values: Mapping[Tuple[str, str, int, int], Mapping[int, float]],
    task_groups: Mapping[str, Sequence[str]],
    reference_ks: Sequence[int],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    qualities = sorted({key[1] for key in values})
    ks = sorted({key[2] for key in values})
    rows: List[Dict[str, Any]] = []
    k95_rows: List[Dict[str, Any]] = []

    for group_name, tasks in task_groups.items():
        for reference_k in reference_ks:
            reference = macro_point(values, tasks, "iid_clean", reference_k)
            group_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            for quality in qualities:
                for k in ks:
                    raw_points = task_points(values, tasks, quality, k)
                    point = domain_balanced_mean(
                        raw_points, tasks, require_all_tasks=False
                    )
                    paired_points = paired_task_points(
                        values, tasks, quality, k, reference_k
                    )
                    task_retentions = {
                        task: stats["retention"] for task, stats in paired_points.items()
                    }
                    retention = domain_balanced_mean(
                        task_retentions, tasks, require_all_tasks=True
                    )
                    macros, retentions = bootstrap_macro_and_retention(
                        values,
                        tasks,
                        quality,
                        k,
                        reference_k,
                        samples=bootstrap_samples,
                        seed=stable_seed(
                            bootstrap_seed, group_name, quality, k, reference_k
                        ),
                    )
                    macro_low, macro_high = percentile_interval(macros)
                    retention_low, retention_high = percentile_interval(retentions)
                    row = {
                        "task_group": group_name,
                        "quality": quality,
                        "quality_label": quality_label(quality),
                        "k": k,
                        "macro_return": point,
                        "macro_return_ci_low": macro_low,
                        "macro_return_ci_high": macro_high,
                        "reference_quality": "iid_clean",
                        "reference_k": reference_k,
                        "reference_macro_return": reference,
                        "retention": retention,
                        "retention_ci_low": retention_low,
                        "retention_ci_high": retention_high,
                        "retention_aggregation": "domain_balanced_mean_of_task_ratios",
                        "num_tasks": len(tasks),
                        "num_tasks_with_return": len(raw_points),
                        "num_paired_tasks": len(paired_points),
                        "num_bootstrap_samples": int(retentions.size),
                    }
                    rows.append(row)
                    group_rows[quality].append(row)

            for quality, quality_rows in group_rows.items():
                ordered = sorted(
                    [row for row in quality_rows if int(row["k"]) <= reference_k],
                    key=lambda row: int(row["k"]),
                )
                point_k95 = None
                conservative_k95 = None
                for index, row in enumerate(ordered):
                    tail = ordered[index:]
                    point_tail = np.asarray(
                        [float(item["retention"]) for item in tail], dtype=np.float64
                    )
                    conservative_tail = np.asarray(
                        [float(item["retention_ci_low"]) for item in tail],
                        dtype=np.float64,
                    )
                    if (
                        point_k95 is None
                        and np.all(np.isfinite(point_tail))
                        and np.all(point_tail >= 0.95)
                    ):
                        point_k95 = int(row["k"])
                    if (
                        conservative_k95 is None
                        and np.all(np.isfinite(conservative_tail))
                        and np.all(conservative_tail >= 0.95)
                    ):
                        conservative_k95 = int(row["k"])
                k95_rows.append(
                    {
                        "task_group": group_name,
                        "quality": quality,
                        "quality_label": quality_label(quality),
                        "reference_quality": "iid_clean",
                        "reference_k": reference_k,
                        "k95_point": point_k95,
                        "conservative_k95": conservative_k95,
                        "threshold": 0.95,
                        "candidate_rule": "all_tested_K_through_reference_meet_threshold",
                    }
                )
    return rows, k95_rows


def bootstrap_task_cell(
    values: Mapping[Tuple[str, str, int, int], Mapping[int, float]],
    task: str,
    quality: str,
    k: int,
    reference_k: int,
    samples: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bootstrap a task cell with paired subset/evaluation resampling."""

    cells = paired_cells_for_task(values, task, quality, k, reference_k)
    if not cells:
        empty = np.asarray([])
        return empty, empty, empty
    rng = np.random.RandomState(seed)
    target_samples: List[float] = []
    reference_samples: List[float] = []
    retention_samples: List[float] = []
    for _ in range(samples):
        sampled_cell_indices = rng.choice(len(cells), size=len(cells), replace=True)
        target_subset_scores: List[float] = []
        reference_subset_scores: List[float] = []
        for cell_index in sampled_cell_indices:
            target_eval, reference_eval, shared_eval_seeds = cells[int(cell_index)]
            sampled_eval_seeds = rng.choice(
                shared_eval_seeds,
                size=len(shared_eval_seeds),
                replace=True,
            )
            target_subset_scores.append(
                float(
                    np.mean(
                        [target_eval[int(eval_seed)] for eval_seed in sampled_eval_seeds]
                    )
                )
            )
            reference_subset_scores.append(
                float(
                    np.mean(
                        [
                            reference_eval[int(eval_seed)]
                            for eval_seed in sampled_eval_seeds
                        ]
                    )
                )
            )
        target = float(np.mean(target_subset_scores))
        reference = float(np.mean(reference_subset_scores))
        target_samples.append(target)
        reference_samples.append(reference)
        retention_samples.append(
            target / reference if abs(reference) > 1e-12 else float("nan")
        )
    return (
        np.asarray(target_samples),
        np.asarray(reference_samples),
        np.asarray(retention_samples),
    )


def summarize_task_retention(
    values: Mapping[Tuple[str, str, int, int], Mapping[int, float]],
    tasks: Sequence[str],
    reference_ks: Sequence[int],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    qualities = sorted({key[1] for key in values})
    ks = sorted({key[2] for key in values})
    rows: List[Dict[str, Any]] = []
    k95_rows: List[Dict[str, Any]] = []

    for task in tasks:
        for reference_k in reference_ks:
            task_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            for quality in qualities:
                for k in ks:
                    point = paired_task_points(
                        values, [task], quality, k, reference_k
                    ).get(task)
                    target_samples, reference_samples, retention_samples = (
                        bootstrap_task_cell(
                            values,
                            task,
                            quality,
                            k,
                            reference_k,
                            samples=bootstrap_samples,
                            seed=stable_seed(
                                bootstrap_seed,
                                "task_retention",
                                task,
                                quality,
                                k,
                                reference_k,
                            ),
                        )
                    )
                    target_low, target_high = percentile_interval(target_samples)
                    reference_low, reference_high = percentile_interval(
                        reference_samples
                    )
                    retention_low, retention_high = percentile_interval(
                        retention_samples
                    )
                    row = {
                        "task": task,
                        "domain": domain(task),
                        "quality": quality,
                        "quality_label": quality_label(quality),
                        "k": k,
                        "reference_quality": "iid_clean",
                        "reference_k": reference_k,
                        "target_return": (
                            point["target"] if point is not None else float("nan")
                        ),
                        "target_return_ci_low": target_low,
                        "target_return_ci_high": target_high,
                        "reference_return": (
                            point["reference"] if point is not None else float("nan")
                        ),
                        "reference_return_ci_low": reference_low,
                        "reference_return_ci_high": reference_high,
                        "retention": (
                            point["retention"] if point is not None else float("nan")
                        ),
                        "retention_ci_low": retention_low,
                        "retention_ci_high": retention_high,
                        "num_shared_subsets": (
                            int(point["num_shared_subsets"])
                            if point is not None
                            else 0
                        ),
                        "num_paired_eval_rows": (
                            int(point["num_paired_eval_rows"])
                            if point is not None
                            else 0
                        ),
                        "num_bootstrap_samples": int(retention_samples.size),
                    }
                    rows.append(row)
                    task_rows[quality].append(row)

            for quality, quality_rows in task_rows.items():
                ordered = sorted(
                    [row for row in quality_rows if int(row["k"]) <= reference_k],
                    key=lambda row: int(row["k"]),
                )
                point_k95 = None
                conservative_k95 = None
                for index, row in enumerate(ordered):
                    tail = ordered[index:]
                    point_tail = np.asarray(
                        [float(item["retention"]) for item in tail], dtype=np.float64
                    )
                    conservative_tail = np.asarray(
                        [float(item["retention_ci_low"]) for item in tail],
                        dtype=np.float64,
                    )
                    if (
                        point_k95 is None
                        and np.all(np.isfinite(point_tail))
                        and np.all(point_tail >= 0.95)
                    ):
                        point_k95 = int(row["k"])
                    if (
                        conservative_k95 is None
                        and np.all(np.isfinite(conservative_tail))
                        and np.all(conservative_tail >= 0.95)
                    ):
                        conservative_k95 = int(row["k"])
                k95_rows.append(
                    {
                        "task": task,
                        "domain": domain(task),
                        "quality": quality,
                        "quality_label": quality_label(quality),
                        "reference_quality": "iid_clean",
                        "reference_k": reference_k,
                        "point_k95": point_k95,
                        "conservative_k95": conservative_k95,
                        "threshold": 0.95,
                        "candidate_rule": "all_tested_K_through_reference_meet_threshold",
                    }
                )
    return rows, k95_rows


def stable_seed(*parts: Any) -> int:
    text = "::".join(str(part) for part in parts)
    import zlib

    return zlib.crc32(text.encode("utf-8")) & 0x7FFFFFFF


def cosine_summary(meta_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, int], List[float]] = defaultdict(list)
    for row in meta_rows:
        if row.get("status") != "complete":
            continue
        key = (str(row["quality"]), int(row["k"]))
        grouped[key].append(float(row["cosine_to_full_clean"]))
    output: List[Dict[str, Any]] = []
    for (quality, k), values in sorted(grouped.items()):
        array = np.asarray(values)
        low, high = percentile_interval(array)
        output.append(
            {
                "quality": quality,
                "quality_label": quality_label(quality),
                "k": k,
                "mean_cosine": float(array.mean()),
                "cosine_ci_low": low,
                "cosine_ci_high": high,
                "num_rows": int(array.size),
            }
        )
    return output


def make_plot(
    aggregate_rows: Sequence[Mapping[str, Any]],
    output: Path,
    reference_k: int,
) -> None:
    available_groups = {str(row["task_group"]) for row in aggregate_rows}
    groups = [
        group
        for group in ("forced_all12", "standard_reward6")
        if group in available_groups
    ]
    if not groups:
        return
    fig, axes_array = plt.subplots(
        1,
        len(groups),
        figsize=(5.75 * len(groups), 4.3),
        sharey=True,
        squeeze=False,
    )
    axes = list(axes_array.ravel())
    for axis, group in zip(axes, groups):
        rows = [
            row
            for row in aggregate_rows
            if row["task_group"] == group and int(row["reference_k"]) == reference_k
        ]
        for quality in sorted({str(row["quality"]) for row in rows}):
            quality_rows = sorted(
                [row for row in rows if row["quality"] == quality],
                key=lambda row: int(row["k"]),
            )
            x = np.asarray([row["k"] for row in quality_rows])
            y = np.asarray([row["retention"] for row in quality_rows])
            low = np.asarray([row["retention_ci_low"] for row in quality_rows])
            high = np.asarray([row["retention_ci_high"] for row in quality_rows])
            axis.plot(x, y, marker="o", linewidth=2, label=quality_label(quality))
            axis.fill_between(x, low, high, alpha=0.15)
        axis.axhline(1.0, color="black", linewidth=1, linestyle="--")
        axis.axhline(0.95, color="gray", linewidth=1, linestyle=":")
        axis.axvline(reference_k, color="tab:red", linewidth=1, linestyle=":")
        axis.set_xscale("log", base=2)
        axis.set_xlabel("Reward-labeled observations K")
        axis.set_title(
            "All 12 (forced projection)"
            if group == "forced_all12"
            else "6 reward-specified tasks"
        )
        axis.grid(alpha=0.25)
    axes[0].set_ylabel(
        f"Domain-balanced mean of per-task return ratios\n"
        f"(Uniform-clean K={reference_k} reference)"
    )
    handles: List[Any] = []
    labels: List[str] = []
    for axis in axes:
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            break
    fig.legend(handles, labels, loc="upper center", ncol=max(1, len(labels)))
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(output.with_suffix(".png"), dpi=200)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def write_report(
    path: Path,
    tasks: Sequence[str],
    episode_rows: Sequence[Mapping[str, Any]],
    aggregate_rows: Sequence[Mapping[str, Any]],
    k95_rows: Sequence[Mapping[str, Any]],
    errors: Sequence[Mapping[str, Any]],
    factorial: Mapping[str, Any],
    integrity: Mapping[str, Any],
) -> None:
    def display(value: Any, digits: Optional[int] = None) -> str:
        if value is None:
            return "—"
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return str(value)
        if not np.isfinite(numeric):
            return "—"
        return f"{numeric:.{digits}f}" if digits is not None else str(int(numeric))

    completed_episode_rows = sum(
        row.get("status") == "complete" for row in episode_rows
    )
    lines = [
        "# Reward-label sensitivity summary",
        "",
        f"- Tasks completed: {len(tasks)} ({', '.join(tasks)})",
        f"- Completed evaluation rows: {completed_episode_rows}",
        f"- Error rows: {len(errors)}",
        (
            "- Factorial audit: "
            f"{factorial['observed_unique_complete_cells']}/{factorial['expected_cells']} "
            f"unique cells; {factorial['missing_cells']} missing; "
            f"{factorial['duplicate_cells']} duplicated; "
            f"{factorial['unexpected_cells']} unexpected."
        ),
        (
            "- Episode-to-meta audit: "
            f"{integrity['meta_links']['valid_episode_meta_links']}/"
            f"{integrity['meta_links']['complete_episode_rows']} valid links; "
            f"{integrity['meta_links']['total_link_issue_records']} issue records."
        ),
        (
            "- Required provenance audit: "
            f"{integrity['provenance']['records_missing_required_provenance']} "
            "complete records missing fields."
        ),
        "- Retention is computed per task first, then averaged within domains and across domains.",
        "- Bootstrap resamples paired subsets and paired evaluation seeds; tasks are fixed.",
        "- Provenance is required to be unique within each task before aggregation.",
        (
            "- Conservative K95 requires the 95% bootstrap interval lower endpoint "
            "to remain at or above 0.95 through the selected reference K."
        ),
        (
            "- K95 is a descriptive tail rule for potentially non-monotone returns, "
            "not a simultaneous confidence guarantee."
        ),
        (
            "- The reference K passes its self-comparison by construction; a "
            "conservative K95 equal to the reference only says that no smaller "
            "tested K passed the rule."
        ),
        "",
        "## K95",
        "",
        "| Task group | Quality | Reference K | Point K95 | Conservative K95 |",
        "|---|---|---:|---:|---:|",
    ]
    for row in k95_rows:
        lines.append(
            f"| {row['task_group']} | {row['quality_label']} | "
            f"{row['reference_k']} | {display(row['k95_point'])} | "
            f"{display(row['conservative_k95'])} |"
        )
    lines.extend(
        [
            "",
            "## Aggregate retention",
            "",
            "| Group | Quality | Reference K | K | Retention | 95% CI | Raw macro return |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in aggregate_rows:
        lines.append(
            f"| {row['task_group']} | {row['quality_label']} | "
            f"{row['reference_k']} | {row['k']} | "
            f"{display(row['retention'], 3)} | "
            f"[{display(row['retention_ci_low'], 3)}, "
            f"{display(row['retention_ci_high'], 3)}] | "
            f"{display(row['macro_return'], 2)} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260727)
    parser.add_argument(
        "--reference-k",
        type=int,
        default=5120,
        help="Primary Uniform-clean reference K; this reference is used in the plot.",
    )
    parser.add_argument(
        "--secondary-reference-k",
        type=int,
        default=20480,
        help="Optional second Uniform-clean reference K (default: 20480).",
    )
    parser.add_argument(
        "--no-secondary-reference",
        action="store_true",
        help="Only summarize --reference-k.",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help=(
            "Exit nonzero after writing audits if the factorial, episode-to-meta "
            "links, or required provenance are incomplete."
        ),
    )
    parser.add_argument(
        "--expected-tasks",
        type=parse_str_csv,
        help="Authoritative comma-separated task list for factorial auditing.",
    )
    parser.add_argument(
        "--expected-qualities",
        type=parse_str_csv,
        help="Authoritative comma-separated quality list for factorial auditing.",
    )
    parser.add_argument(
        "--expected-ks",
        type=parse_int_csv,
        help="Authoritative comma-separated K list for factorial auditing.",
    )
    parser.add_argument(
        "--expected-subset-seeds",
        type=parse_int_csv,
        help="Authoritative comma-separated subset-seed list for factorial auditing.",
    )
    parser.add_argument(
        "--expected-eval-seeds",
        type=parse_int_csv,
        help="Authoritative comma-separated evaluation-seed list for factorial auditing.",
    )
    args = parser.parse_args()

    if args.bootstrap_samples <= 0:
        parser.error("--bootstrap-samples must be positive")
    if args.reference_k <= 0 or args.secondary_reference_k <= 0:
        parser.error("reference K values must be positive")
    reference_ks = [args.reference_k]
    if not args.no_secondary_reference:
        reference_ks.append(args.secondary_reference_k)
    reference_ks = list(dict.fromkeys(reference_ks))
    if args.expected_ks is not None:
        missing_reference_ks = [
            value for value in reference_ks if value not in args.expected_ks
        ]
        if missing_reference_ks:
            parser.error(
                "--expected-ks must include every requested reference K; missing "
                + ",".join(str(value) for value in missing_reference_ks)
            )

    input_dir = args.input_dir.resolve()
    if not input_dir.is_dir():
        raise RuntimeError(f"Input directory does not exist: {input_dir}")
    episode_rows: List[Dict[str, Any]] = []
    meta_rows: List[Dict[str, Any]] = []
    error_rows: List[Dict[str, Any]] = []
    task_dirs = sorted(path for path in input_dir.iterdir() if path.is_dir())
    for task_dir in task_dirs:
        episode_rows.extend(read_jsonl(task_dir / "episodes.jsonl"))
        meta_rows.extend(read_jsonl(task_dir / "metas.jsonl"))
        error_rows.extend(read_jsonl(task_dir / "errors.jsonl"))

    validate_provenance(episode_rows + meta_rows)
    provenance, missing_provenance = required_provenance_audit(
        episode_rows,
        meta_rows,
    )
    meta_links, meta_link_issues = episode_meta_link_audit(
        episode_rows,
        meta_rows,
    )
    directory_tasks = [
        path.name
        for path in task_dirs
        if path.name.startswith(("cheetah_", "walker_", "quadruped_"))
    ]
    factorial, missing_cells, duplicate_cells, unexpected_cells = factorial_audit(
        episode_rows,
        meta_rows + error_rows,
        reference_ks,
        directory_tasks=directory_tasks,
        expected_tasks=args.expected_tasks,
        expected_qualities=args.expected_qualities,
        expected_ks=args.expected_ks,
        expected_subset_seeds=args.expected_subset_seeds,
        expected_eval_seeds=args.expected_eval_seeds,
    )
    integrity = {
        "factorial": factorial,
        "meta_links": meta_links,
        "provenance": provenance,
        "is_complete": bool(
            factorial["is_complete"]
            and meta_links["is_complete"]
            and provenance["is_complete"]
        ),
    }
    (input_dir / "factorial_audit.json").write_text(
        json.dumps(factorial, indent=2, sort_keys=True) + "\n"
    )
    (input_dir / "integrity_audit.json").write_text(
        json.dumps(integrity, indent=2, sort_keys=True) + "\n"
    )
    write_csv(
        input_dir / "missing_factorial_cells.csv",
        missing_cells,
        fieldnames=CELL_FIELDS,
    )
    write_csv(
        input_dir / "duplicate_factorial_cells.csv",
        duplicate_cells,
        fieldnames=(*CELL_FIELDS, "count"),
    )
    write_csv(
        input_dir / "unexpected_factorial_cells.csv",
        unexpected_cells,
        fieldnames=(*CELL_FIELDS, "count"),
    )
    write_csv(
        input_dir / "meta_link_issues.csv",
        meta_link_issues,
        fieldnames=(
            "issue_type",
            "episode_row_index",
            "episode_cell_id",
            "meta_cell_id",
            "task",
            "mismatched_fields",
            "detail",
        ),
    )
    write_csv(
        input_dir / "missing_provenance.csv",
        missing_provenance,
        fieldnames=(
            "record_type",
            "row_index",
            "task",
            "record_id",
            "missing_fields",
        ),
    )
    if args.require_complete and not integrity["is_complete"]:
        raise RuntimeError(
            "Incomplete experiment integrity: "
            f"factorial={factorial['is_complete']} "
            f"({factorial['missing_cells']} missing, "
            f"{factorial['duplicate_cells']} duplicated, "
            f"{factorial['unexpected_cells']} unexpected); "
            f"meta_links={meta_links['is_complete']} "
            f"({meta_links['total_link_issue_records']} issues); "
            f"provenance={provenance['is_complete']} "
            f"({provenance['records_missing_required_provenance']} missing records). "
            f"See {input_dir / 'integrity_audit.json'}."
        )

    values = build_value_map(episode_rows)
    tasks = sorted({key[0] for key in values})
    if not tasks:
        raise RuntimeError(f"No completed episode rows found under {input_dir}")

    task_groups: Dict[str, Sequence[str]] = {
        "standard_reward6": [task for task in REWARD_SPECIFIED_TASKS if task in tasks],
    }
    if len(tasks) == 12:
        task_groups = {"forced_all12": tasks, **task_groups}
    for domain_name in sorted({domain(task) for task in tasks}):
        task_groups[f"domain_{domain_name}"] = [
            task for task in tasks if domain(task) == domain_name
        ]

    task_rows = task_cell_means(values)
    aggregate_rows, k95_rows = summarize_aggregate(
        values,
        task_groups,
        reference_ks=reference_ks,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    task_retention_rows, task_k95_rows = summarize_task_retention(
        values,
        tasks,
        reference_ks=reference_ks,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    cosine_rows = cosine_summary(meta_rows)
    write_csv(input_dir / "task_summary.csv", task_rows)
    write_csv(input_dir / "task_retention_summary.csv", task_retention_rows)
    write_csv(input_dir / "task_k95_summary.csv", task_k95_rows)
    write_csv(input_dir / "aggregate_summary.csv", aggregate_rows)
    write_csv(input_dir / "k95_summary.csv", k95_rows)
    write_csv(input_dir / "cosine_summary.csv", cosine_rows)
    make_plot(
        aggregate_rows,
        input_dir / "reward_label_sensitivity",
        reference_k=args.reference_k,
    )
    write_report(
        input_dir / "report.md",
        tasks,
        episode_rows,
        aggregate_rows,
        k95_rows,
        error_rows,
        factorial,
        integrity,
    )
    print(
        f"Wrote summaries to {input_dir} with reference K values "
        f"{','.join(str(value) for value in reference_ks)}"
    )


if __name__ == "__main__":
    main()
