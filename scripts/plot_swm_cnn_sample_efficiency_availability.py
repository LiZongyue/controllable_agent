#!/usr/bin/env python3
"""Generate availability-only table and figure for SWM-CLS versus CNN-FB.

The input is the checkpoint/evaluation audit CSV.  No return values are read or
plotted, and mixed-protocol periodic evaluations are never treated as compliant
performance observations.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap


METHODS = ("SWM-CLS", "CNN-FB")
SEEDS = (2009, 7532, 8164)
DOMAINS = ("walker", "quadruped", "cheetah")
FRAMES = (500_000, 1_000_000, 1_500_000, 2_000_000)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path(
            "analysis_outputs/swm_cls_vs_cnn_fb_20260727/"
            "checkpoint_eval_availability.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis_outputs/swm_cls_vs_cnn_fb_20260727"),
    )
    return parser.parse_args()


def as_bool(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"Expected serialized boolean, got {value!r}")


def read_and_validate(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as handle:
        rows: list[dict[str, Any]] = list(csv.DictReader(handle))
    if len(rows) != 288:
        raise RuntimeError(f"Expected 288 audit cells, got {len(rows)}")
    keys = {
        (
            row["method"],
            int(row["seed"]),
            row["domain"],
            row["task"],
            int(row["environment_frames"]),
        )
        for row in rows
    }
    if len(keys) != 288:
        raise RuntimeError(f"Expected 288 unique audit keys, got {len(keys)}")
    for row in rows:
        row["seed"] = int(row["seed"])
        row["environment_frames"] = int(row["environment_frames"])
        for field in (
            "checkpoint_exists",
            "periodic_eval_frame_exists",
            "periodic_eval_strict_compliant",
            "strict_protocol_available",
            "cross_frame_comparable_without_reevaluation",
        ):
            row[field] = as_bool(row[field])
        if row["periodic_eval_strict_compliant"]:
            raise RuntimeError("Audit unexpectedly marks a periodic evaluation as strict compliant")
        if row["cross_frame_comparable_without_reevaluation"]:
            raise RuntimeError("Audit unexpectedly marks a cell cross-frame comparable")
    return rows


def aggregate(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, int, str | None], dict[str, int]]:
    output: dict[tuple[str, int, str | None], dict[str, int]] = {}
    for method in METHODS:
        for frame in FRAMES:
            for domain in (*DOMAINS, None):
                selected = [
                    row
                    for row in rows
                    if row["method"] == method
                    and row["environment_frames"] == frame
                    and (domain is None or row["domain"] == domain)
                ]
                expected = 36 if domain is None else 12
                if len(selected) != expected:
                    raise RuntimeError(
                        f"Expected {expected} cells for {(method, frame, domain)}, got {len(selected)}"
                    )
                output[(method, frame, domain)] = {
                    "checkpoint": sum(bool(row["checkpoint_exists"]) for row in selected),
                    "periodic": sum(bool(row["periodic_eval_frame_exists"]) for row in selected),
                    "strict": sum(bool(row["strict_protocol_available"]) for row in selected),
                    "total": expected,
                }
    return output


def frame_label(frame: int) -> str:
    return f"{frame / 1_000_000:g}M"


def write_latex(
    path: Path,
    summary: dict[tuple[str, int, str | None], dict[str, int]],
) -> None:
    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \small",
        r"  \caption{Availability of existing assets for the requested sample-efficiency analysis. "
        r"These counts are \emph{not performance results}. Mixed-protocol periodic evaluations are "
        r"reported only as audit evidence and are not used as compliant observations.}",
        r"  \label{tab:swm-cnn-sample-efficiency-availability}",
        r"  \begin{tabular}{llccc}",
        r"    \toprule",
        r"    Method & Frames & Exact checkpoints & Periodic logs$^{\dagger}$ & Legacy strict RP$^{\ddagger}$ \\" ,
        r"    \midrule",
    ]
    for method_index, method in enumerate(METHODS):
        for frame in FRAMES:
            values = summary[(method, frame, None)]
            lines.append(
                f"    {method} & {frame_label(frame)} & "
                f"{values['checkpoint']}/36 & {values['periodic']}/36 & {values['strict']}/36 \\\\"
            )
        if method_index + 1 < len(METHODS):
            lines.append(r"    \midrule")
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \vspace{2pt}",
        r"  \begin{minipage}{0.97\linewidth}",
        r"    \footnotesize",
        r"    $^{\dagger}$The periodic training logs use registered goal vectors for some tasks and "
        r"reward inference for others; they therefore do not satisfy the requested uniform reward-projection protocol. "
        r"$^{\ddagger}$A legacy strict-RP entry is a valid ten-episode task result in a 2M-frame "
        r"\texttt{test\_rewards.json}. It used the live run's replay buffer, which was not stored in the checkpoint. "
        r"All frame points must be reevaluated with a common calibration source before drawing learning curves or computing AUC.",
        r"  \end{minipage}",
        r"\end{table}",
    ]
    path.write_text("\n".join(lines) + "\n")


def plot_coverage(
    png_path: Path,
    pdf_path: Path,
    rows: list[dict[str, Any]],
) -> None:
    row_keys = [(seed, domain) for seed in SEEDS for domain in DOMAINS]
    row_labels = [f"{domain.capitalize()} · {seed}" for seed, domain in row_keys]
    cmap = ListedColormap(["#f3f4f6", "#dbeafe", "#93c5fd", "#3b82f6", "#1e3a8a"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], cmap.N)

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 5.25), sharey=True)
    fig.subplots_adjust(left=0.16, right=0.88, bottom=0.19, top=0.82, wspace=0.04)
    for axis, method in zip(axes, METHODS):
        checkpoint = np.zeros((len(row_keys), len(FRAMES)), dtype=np.int64)
        strict = np.zeros_like(checkpoint)
        periodic = np.zeros_like(checkpoint)
        for row_index, (seed, domain) in enumerate(row_keys):
            for frame_index, frame in enumerate(FRAMES):
                selected = [
                    row
                    for row in rows
                    if row["method"] == method
                    and row["seed"] == seed
                    and row["domain"] == domain
                    and row["environment_frames"] == frame
                ]
                if len(selected) != 4:
                    raise RuntimeError(
                        f"Expected four task cells for {(method, seed, domain, frame)}"
                    )
                checkpoint[row_index, frame_index] = sum(
                    bool(row["checkpoint_exists"]) for row in selected
                )
                strict[row_index, frame_index] = sum(
                    bool(row["strict_protocol_available"]) for row in selected
                )
                periodic[row_index, frame_index] = sum(
                    bool(row["periodic_eval_frame_exists"]) for row in selected
                )
        image = axis.imshow(checkpoint, cmap=cmap, norm=norm, aspect="auto")
        axis.set_title(method, fontsize=11, weight="bold")
        axis.set_xticks(range(len(FRAMES)), [frame_label(frame) for frame in FRAMES])
        axis.set_xlabel("Environment frames")
        axis.set_yticks(range(len(row_keys)), row_labels)
        axis.tick_params(axis="both", labelsize=8.5)
        axis.set_xticks(np.arange(-0.5, len(FRAMES), 1), minor=True)
        axis.set_yticks(np.arange(-0.5, len(row_keys), 1), minor=True)
        axis.grid(which="minor", color="white", linewidth=1.2)
        axis.tick_params(which="minor", bottom=False, left=False)
        for row_index in range(len(row_keys)):
            for frame_index in range(len(FRAMES)):
                value = checkpoint[row_index, frame_index]
                color = "white" if value >= 3 else "#111827"
                axis.text(
                    frame_index,
                    row_index - 0.10,
                    f"C {value}/4",
                    ha="center",
                    va="center",
                    fontsize=7.7,
                    color=color,
                    weight="bold",
                )
                axis.text(
                    frame_index,
                    row_index + 0.20,
                    f"S {strict[row_index, frame_index]}/4 · P {periodic[row_index, frame_index]}/4",
                    ha="center",
                    va="center",
                    fontsize=6.3,
                    color=color,
                )

    colorbar = fig.colorbar(image, ax=axes, location="right", ticks=range(5), shrink=0.86, pad=0.025)
    colorbar.set_label("Exact checkpoints available (of 4 tasks)", fontsize=8.5)
    colorbar.ax.tick_params(labelsize=8)
    fig.suptitle(
        "Sample-efficiency asset and protocol coverage — availability, not performance",
        fontsize=12,
        y=0.965,
    )
    fig.text(
        0.5,
        0.025,
        "C = exact checkpoint; S = legacy strict reward-projection result; "
        "P = mixed periodic log (audit only, non-compliant). No returns are plotted.",
        ha="center",
        va="bottom",
        fontsize=8,
    )
    fig.savefig(png_path, dpi=240, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_and_validate(args.input_csv)
    summary = aggregate(rows)
    tex_path = args.output_dir / "sample_efficiency_availability_table.tex"
    png_path = args.output_dir / "sample_efficiency_coverage.png"
    pdf_path = args.output_dir / "sample_efficiency_coverage.pdf"
    write_latex(tex_path, summary)
    plot_coverage(png_path, pdf_path, rows)
    print(f"wrote {tex_path}")
    print(f"wrote {png_path}")
    print(f"wrote {pdf_path}")


if __name__ == "__main__":
    main()
