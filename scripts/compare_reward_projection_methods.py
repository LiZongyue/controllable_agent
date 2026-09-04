#!/usr/bin/env python3
"""Compare paired reward-projection evaluation directories.

The evaluator writes one ``episodes.jsonl`` file per task.  This script reads
only completed episodes, requires an exact one-to-one pairing between the two
directories, and reports the paired effect of the second method (normally
ridge) relative to the first (normally the mean/Monte-Carlo projection).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


CELL_FIELDS = ("task", "quality", "k", "subset_seed", "eval_seed")
PAIR_PROVENANCE_FIELDS = (
    "bank_id",
    "bank_source",
    "checkpoint_fingerprint",
    "eval_condition",
    "episode_length",
)


def _read_completed_episodes(root: Path) -> list[dict[str, Any]]:
    if not root.is_dir():
        raise ValueError(f"evaluation directory does not exist: {root}")

    paths = sorted(root.rglob("episodes.jsonl"))
    if not paths:
        raise ValueError(f"no episodes.jsonl files found under {root}")

    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open() as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
                if row.get("status") != "complete":
                    continue
                row = dict(row)
                row["_source"] = f"{path}:{line_number}"
                rows.append(row)
    if not rows:
        raise ValueError(f"no completed episode rows found under {root}")
    return rows


def _cell_key(row: Mapping[str, Any]) -> tuple[str, str, int, int, int]:
    missing = [field for field in CELL_FIELDS if field not in row]
    if missing:
        raise ValueError(
            f"episode row at {row.get('_source', '<unknown>')} is missing "
            f"pairing fields: {', '.join(missing)}"
        )
    try:
        return (
            str(row["task"]),
            str(row["quality"]),
            int(row["k"]),
            int(row["subset_seed"]),
            int(row["eval_seed"]),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"invalid pairing field at {row.get('_source', '<unknown>')}"
        ) from exc


def _index_unique(
    rows: Iterable[Mapping[str, Any]], label: str
) -> dict[tuple[str, str, int, int, int], Mapping[str, Any]]:
    indexed: dict[tuple[str, str, int, int, int], Mapping[str, Any]] = {}
    duplicates: list[tuple[str, str, int, int, int]] = []
    for row in rows:
        key = _cell_key(row)
        if key in indexed:
            duplicates.append(key)
        indexed[key] = row
    if duplicates:
        sample = ", ".join(map(str, sorted(set(duplicates))[:5]))
        raise ValueError(
            f"{label} has duplicate completed pairing cells "
            f"({len(duplicates)} duplicate rows); examples: {sample}"
        )
    return indexed


def _validated_reward(row: Mapping[str, Any]) -> float:
    try:
        reward = float(row["episode_reward"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"invalid episode_reward at {row.get('_source', '<unknown>')}"
        ) from exc
    if not math.isfinite(reward):
        raise ValueError(
            f"non-finite episode_reward at {row.get('_source', '<unknown>')}"
        )
    return reward


def validate_projection_provenance(
    mean_rows: Iterable[Mapping[str, Any]],
    ridge_rows: Iterable[Mapping[str, Any]],
) -> float:
    """Require a pure mean run and one consistently regularized ridge run."""

    mean_rows = list(mean_rows)
    ridge_rows = list(ridge_rows)
    mean_methods = {row.get("method") for row in mean_rows}
    ridge_methods = {row.get("method") for row in ridge_rows}
    if mean_methods != {"reward_projection"}:
        raise ValueError(
            "mean directory must contain only method='reward_projection'; "
            f"found {sorted(map(str, mean_methods))}"
        )
    if ridge_methods != {"reward_projection_ridge"}:
        raise ValueError(
            "ridge directory must contain only "
            "method='reward_projection_ridge'; "
            f"found {sorted(map(str, ridge_methods))}"
        )

    ridge_alphas: set[float] = set()
    for row in ridge_rows:
        try:
            ridge_alpha = float(row["ridge_alpha"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "ridge episode is missing a numeric ridge_alpha at "
                f"{row.get('_source', '<unknown>')}"
            ) from exc
        if not math.isfinite(ridge_alpha) or ridge_alpha <= 0:
            raise ValueError(
                f"ridge_alpha must be finite and positive, got {ridge_alpha} at "
                f"{row.get('_source', '<unknown>')}"
            )
        ridge_alphas.add(ridge_alpha)
    if len(ridge_alphas) != 1:
        raise ValueError(
            "ridge directory mixes ridge_alpha values: "
            + ", ".join(str(value) for value in sorted(ridge_alphas))
        )
    return next(iter(ridge_alphas))


def pair_rows(
    first_rows: Iterable[Mapping[str, Any]],
    second_rows: Iterable[Mapping[str, Any]],
    *,
    first_label: str = "mean",
    second_label: str = "ridge",
) -> list[dict[str, Any]]:
    """Return validated paired cells, or raise on any mismatch."""

    first = _index_unique(first_rows, first_label)
    second = _index_unique(second_rows, second_label)
    first_keys = set(first)
    second_keys = set(second)
    if first_keys != second_keys:
        only_first = sorted(first_keys - second_keys)
        only_second = sorted(second_keys - first_keys)
        raise ValueError(
            "completed evaluation cells are not exactly paired: "
            f"only in {first_label}={len(only_first)} "
            f"(examples={only_first[:5]}), only in {second_label}="
            f"{len(only_second)} (examples={only_second[:5]})"
        )

    provenance_issues: list[str] = []
    paired: list[dict[str, Any]] = []
    for key in sorted(first):
        first_row = first[key]
        second_row = second[key]
        for field in PAIR_PROVENANCE_FIELDS:
            first_value = first_row.get(field)
            second_value = second_row.get(field)
            if first_value is not None and second_value is not None:
                if first_value != second_value:
                    provenance_issues.append(
                        f"cell={key}, field={field}, {first_label}="
                        f"{first_value!r}, {second_label}={second_value!r}"
                    )

        first_reward = _validated_reward(first_row)
        second_reward = _validated_reward(second_row)
        paired.append(
            {
                "task": key[0],
                "quality": key[1],
                "k": key[2],
                "subset_seed": key[3],
                "eval_seed": key[4],
                f"{first_label}_reward": first_reward,
                f"{second_label}_reward": second_reward,
                f"{second_label}_minus_{first_label}": second_reward - first_reward,
                "bank_id": first_row.get("bank_id", ""),
                "checkpoint_fingerprint": first_row.get(
                    "checkpoint_fingerprint", ""
                ),
                "ridge_alpha": second_row.get("ridge_alpha", ""),
            }
        )

    if provenance_issues:
        raise ValueError(
            "paired cells have mismatched evaluation provenance; examples: "
            + "; ".join(provenance_issues[:5])
        )
    return paired


def _sample_std(values: Sequence[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def summarize_pairs(
    paired: Sequence[Mapping[str, Any]],
    *,
    first_label: str = "mean",
    second_label: str = "ridge",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build per-task and task-macro summaries for every quality/K condition."""

    first_reward_field = f"{first_label}_reward"
    second_reward_field = f"{second_label}_reward"
    difference_field = f"{second_label}_minus_{first_label}"
    grouped: dict[tuple[str, str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in paired:
        grouped[(str(row["task"]), str(row["quality"]), int(row["k"]))].append(
            row
        )

    task_rows: list[dict[str, Any]] = []
    for (task, quality, k), rows in sorted(grouped.items()):
        first_values = [float(row[first_reward_field]) for row in rows]
        second_values = [float(row[second_reward_field]) for row in rows]
        differences = [float(row[difference_field]) for row in rows]
        task_rows.append(
            {
                "task": task,
                "quality": quality,
                "k": k,
                "n_paired_episodes": len(rows),
                f"{first_label}_mean": statistics.mean(first_values),
                f"{first_label}_std": _sample_std(first_values),
                f"{second_label}_mean": statistics.mean(second_values),
                f"{second_label}_std": _sample_std(second_values),
                f"{difference_field}_mean": statistics.mean(differences),
                f"{difference_field}_std": _sample_std(differences),
            }
        )

    by_condition: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in task_rows:
        by_condition[(str(row["quality"]), int(row["k"]))].append(row)

    macro_rows: list[dict[str, Any]] = []
    for (quality, k), rows in sorted(by_condition.items()):
        first_values = [float(row[f"{first_label}_mean"]) for row in rows]
        second_values = [float(row[f"{second_label}_mean"]) for row in rows]
        differences = [float(row[f"{difference_field}_mean"]) for row in rows]
        macro_rows.append(
            {
                "quality": quality,
                "k": k,
                "n_tasks": len(rows),
                "n_paired_episodes": sum(
                    int(row["n_paired_episodes"]) for row in rows
                ),
                f"{first_label}_macro_mean": statistics.mean(first_values),
                f"{first_label}_task_std": _sample_std(first_values),
                f"{second_label}_macro_mean": statistics.mean(second_values),
                f"{second_label}_task_std": _sample_std(second_values),
                f"{difference_field}_macro_mean": statistics.mean(differences),
                f"{difference_field}_task_std": _sample_std(differences),
            }
        )
    return task_rows, macro_rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fields = list(rows[0])
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _markdown_table(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> str:
    def render(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.3f}"
        return str(value)

    header = "| " + " | ".join(fields) + " |"
    divider = "| " + " | ".join("---" for _ in fields) + " |"
    lines = [header, divider]
    for row in rows:
        lines.append("| " + " | ".join(render(row[field]) for field in fields) + " |")
    return "\n".join(lines)


def write_report(
    output_dir: Path,
    paired: Sequence[Mapping[str, Any]],
    task_rows: Sequence[Mapping[str, Any]],
    macro_rows: Sequence[Mapping[str, Any]],
    *,
    first_dir: Path,
    second_dir: Path,
    first_label: str,
    second_label: str,
    ridge_alpha: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "paired_cells.csv", paired)
    _write_csv(output_dir / "task_summary.csv", task_rows)
    _write_csv(output_dir / "macro_summary.csv", macro_rows)

    difference_field = f"{second_label}_minus_{first_label}"
    macro_fields = (
        "quality",
        "k",
        "n_tasks",
        "n_paired_episodes",
        f"{first_label}_macro_mean",
        f"{second_label}_macro_mean",
        f"{difference_field}_macro_mean",
        f"{difference_field}_task_std",
    )
    task_fields = (
        "task",
        "quality",
        "k",
        "n_paired_episodes",
        f"{first_label}_mean",
        f"{first_label}_std",
        f"{second_label}_mean",
        f"{second_label}_std",
        f"{difference_field}_mean",
        f"{difference_field}_std",
    )
    report = [
        "# Paired reward-projection comparison",
        "",
        f"- `{first_label}`: `{first_dir.resolve()}`",
        f"- `{second_label}`: `{second_dir.resolve()}`",
        f"- Ridge alpha: {ridge_alpha:g}",
        f"- Exactly paired completed episodes: {len(paired)}",
        "- Standard deviations use the sample definition (zero for n=1).",
        "- Macro means weight each task equally within each quality/K condition.",
        "",
        "## Macro summary",
        "",
        _markdown_table(macro_rows, macro_fields),
        "",
        "## Per-task summary",
        "",
        _markdown_table(task_rows, task_fields),
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(report))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mean-dir", type=Path, required=True)
    parser.add_argument("--ridge-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mean-label", default="mean")
    parser.add_argument("--ridge-label", default="ridge")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.mean_label == args.ridge_label:
        raise ValueError("comparison labels must be distinct")
    first_rows = _read_completed_episodes(args.mean_dir)
    second_rows = _read_completed_episodes(args.ridge_dir)
    ridge_alpha = validate_projection_provenance(first_rows, second_rows)
    paired = pair_rows(
        first_rows,
        second_rows,
        first_label=args.mean_label,
        second_label=args.ridge_label,
    )
    task_rows, macro_rows = summarize_pairs(
        paired,
        first_label=args.mean_label,
        second_label=args.ridge_label,
    )
    write_report(
        args.output_dir,
        paired,
        task_rows,
        macro_rows,
        first_dir=args.mean_dir,
        second_dir=args.ridge_dir,
        first_label=args.mean_label,
        second_label=args.ridge_label,
        ridge_alpha=ridge_alpha,
    )
    print(
        f"Validated {len(paired)} paired episodes; wrote comparison to "
        f"{args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
