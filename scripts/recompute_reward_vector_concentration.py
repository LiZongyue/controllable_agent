#!/usr/bin/env python3
"""Recompute and archive fixed-bank reward-vector concentration cells."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

# Allow the documented direct invocation from the repository root without
# requiring an editable package installation.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from url_benchmark.reward_label_sensitivity import (
    CalibrationBank,
    build_meta,
    meta_from_features,
)


DEFAULT_TASKS = (
    "cheetah_walk",
    "cheetah_walk_backward",
    "cheetah_run",
    "cheetah_run_backward",
    "walker_flip",
    "quadruped_jump",
)
DEFAULT_KS = (1, 4, 16, 64, 256, 1024, 5120, 20480)


def parse_str_csv(value: str) -> List[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("expected unique comma-separated values")
    return values


def parse_int_csv(value: str) -> List[int]:
    try:
        values = [int(item) for item in parse_str_csv(value)]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("K values must be positive")
    return values


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write empty CSV: {path}")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def find_bank(task_dir: Path) -> Path:
    paths = sorted((task_dir / "banks").glob("bank_*.npz"))
    if len(paths) != 1:
        raise RuntimeError(
            f"Expected exactly one calibration bank under {task_dir}, found {paths}"
        )
    return paths[0]


def percentile(values: Sequence[float], quantile: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), quantile))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--tasks", type=parse_str_csv, default=list(DEFAULT_TASKS))
    parser.add_argument("--ks", type=parse_int_csv, default=list(DEFAULT_KS))
    parser.add_argument("--subset-seed-start", type=int, default=0)
    parser.add_argument("--num-subset-seeds", type=int, default=50)
    parser.add_argument("--z-dim", type=int, default=50)
    parser.add_argument("--expected-bank-size", type=int, default=20480)
    parser.add_argument(
        "--output-prefix",
        default="reward6_vector_concentration",
        help="Filename prefix for raw, summary, provenance, and audit artifacts.",
    )
    args = parser.parse_args()

    if args.subset_seed_start < 0:
        parser.error("--subset-seed-start must be nonnegative")
    if args.num_subset_seeds <= 0:
        parser.error("--num-subset-seeds must be positive")
    if args.z_dim <= 0 or args.expected_bank_size <= 0:
        parser.error("--z-dim and --expected-bank-size must be positive")
    if not args.output_prefix or Path(args.output_prefix).name != args.output_prefix:
        parser.error("--output-prefix must be a nonempty filename component")
    if max(args.ks) > args.expected_bank_size:
        parser.error("largest K exceeds --expected-bank-size")

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not input_dir.is_dir():
        raise RuntimeError(f"Input directory does not exist: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    subset_seeds = list(
        range(
            args.subset_seed_start,
            args.subset_seed_start + args.num_subset_seeds,
        )
    )

    raw_rows: List[Dict[str, object]] = []
    bank_records: List[Dict[str, object]] = []
    seen_cells: set[Tuple[str, int, int]] = set()
    for task in args.tasks:
        bank_path = find_bank(input_dir / task)
        bank = CalibrationBank.load(bank_path)
        if len(bank) != args.expected_bank_size:
            raise RuntimeError(
                f"Task {task}: bank has {len(bank)} rows, expected "
                f"{args.expected_bank_size}"
            )
        required_metadata = {
            "task": task,
            "bank_source": "exorl_rnd",
            "size": args.expected_bank_size,
        }
        mismatches = {
            key: (bank.metadata.get(key), expected)
            for key, expected in required_metadata.items()
            if bank.metadata.get(key) != expected
        }
        if mismatches:
            raise RuntimeError(f"Task {task}: bank metadata mismatch {mismatches}")

        reference_meta, reference_diag = meta_from_features(
            bank.backward,
            bank.reward,
            z_dim=args.z_dim,
            norm_z=True,
            projection_method="mean",
        )
        reference_z = reference_meta["z"]
        bank_id = bank_path.stem.removeprefix("bank_")
        bank_records.append(
            {
                "task": task,
                "bank_id": bank_id,
                "checkpoint_fingerprint": bank.metadata["checkpoint_fingerprint"],
                "bank_source": bank.metadata["bank_source"],
                "bank_size": len(bank),
                "reference_raw_z_norm": reference_diag["raw_z_norm"],
            }
        )

        for subset_seed in subset_seeds:
            for k in args.ks:
                _, diagnostics = build_meta(
                    bank,
                    quality="iid_clean",
                    k=k,
                    subset_seed=subset_seed,
                    task=task,
                    z_dim=args.z_dim,
                    norm_z=True,
                    reference_z=reference_z,
                    projection_method="mean",
                )
                cell = (task, int(k), int(subset_seed))
                if cell in seen_cells:
                    raise RuntimeError(f"Duplicate vector cell: {cell}")
                seen_cells.add(cell)
                raw_rows.append(
                    {
                        "task": task,
                        "k": int(k),
                        "subset_seed": int(subset_seed),
                        "cosine_to_full_clean": diagnostics[
                            "cosine_to_full_clean"
                        ],
                        "degenerate_z": diagnostics["degenerate_z"],
                        "raw_z_norm": diagnostics["raw_z_norm"],
                        "reward_mean": diagnostics["reward_mean"],
                        "reward_nonzero_fraction": diagnostics[
                            "reward_nonzero_fraction"
                        ],
                        "index_digest": diagnostics["index_digest"],
                        "bank_id": bank_id,
                        "checkpoint_fingerprint": bank.metadata[
                            "checkpoint_fingerprint"
                        ],
                    }
                )

    expected_cells = len(args.tasks) * len(args.ks) * len(subset_seeds)
    if len(raw_rows) != expected_cells or len(seen_cells) != expected_cells:
        raise RuntimeError(
            f"Incomplete vector factorial: {len(seen_cells)}/{expected_cells}"
        )

    grouped: Dict[Tuple[str, int], List[Dict[str, object]]] = defaultdict(list)
    pooled: Dict[int, List[Dict[str, object]]] = defaultdict(list)
    for row in raw_rows:
        grouped[(str(row["task"]), int(row["k"]))].append(row)
        pooled[int(row["k"])].append(row)

    task_rows: List[Dict[str, object]] = []
    for task in args.tasks:
        for k in args.ks:
            cells = grouped[(task, k)]
            cosines = [float(row["cosine_to_full_clean"]) for row in cells]
            task_rows.append(
                {
                    "task": task,
                    "k": k,
                    "num_subsets": len(cells),
                    "median_cosine": float(np.median(cosines)),
                    "p05_cosine_to_full_clean": percentile(cosines, 0.05),
                    "degenerate_fraction": float(
                        np.mean([bool(row["degenerate_z"]) for row in cells])
                    ),
                }
            )

    task_p05 = {
        (str(row["task"]), int(row["k"])): float(
            row["p05_cosine_to_full_clean"]
        )
        for row in task_rows
    }
    summary_rows: List[Dict[str, object]] = []
    for k in args.ks:
        cells = pooled[k]
        cosines = [float(row["cosine_to_full_clean"]) for row in cells]
        summary_rows.append(
            {
                "k": k,
                "num_cells": len(cells),
                "median_cosine": float(np.median(cosines)),
                "p05_cosine": percentile(cosines, 0.05),
                "worst_task_p05_cosine": min(
                    task_p05[(task, k)] for task in args.tasks
                ),
                "degenerate_fraction": float(
                    np.mean([bool(row["degenerate_z"]) for row in cells])
                ),
            }
        )

    suffix = f"{len(subset_seeds)}subsets"
    write_csv(
        output_dir / f"{args.output_prefix}_raw_{suffix}.csv", raw_rows
    )
    write_csv(
        output_dir / f"{args.output_prefix}_task_summary_{suffix}.csv",
        task_rows,
    )
    write_csv(
        output_dir / f"{args.output_prefix}_summary_{suffix}.csv",
        summary_rows,
    )
    write_csv(
        output_dir / f"{args.output_prefix}_banks_{suffix}.csv",
        bank_records,
    )
    audit = {
        "tasks": args.tasks,
        "ks": args.ks,
        "subset_seeds": subset_seeds,
        "expected_cells": expected_cells,
        "observed_unique_cells": len(seen_cells),
        "duplicate_cells": len(raw_rows) - len(seen_cells),
        "is_complete": len(seen_cells) == expected_cells,
        "projection_method": "mean",
        "norm_z": True,
        "z_dim": args.z_dim,
        "bank_source": "exorl_rnd",
        "expected_bank_size": args.expected_bank_size,
    }
    (output_dir / f"{args.output_prefix}_audit_{suffix}.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n"
    )
    print(
        f"Wrote {len(raw_rows)} complete vector cells and summaries to "
        f"{output_dir}"
    )


if __name__ == "__main__":
    main()
