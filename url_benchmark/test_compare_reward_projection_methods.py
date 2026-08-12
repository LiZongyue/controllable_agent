import json
import math

import pytest

from scripts.compare_reward_projection_methods import (
    main,
    pair_rows,
    summarize_pairs,
    validate_projection_provenance,
)


def _row(
    task,
    reward,
    *,
    eval_seed=1,
    bank_id="bank",
    method=None,
    ridge_alpha=None,
):
    row = {
        "status": "complete",
        "task": task,
        "quality": "iid_clean",
        "k": 16,
        "subset_seed": 0,
        "eval_seed": eval_seed,
        "episode_reward": reward,
        "bank_id": bank_id,
        "checkpoint_fingerprint": f"checkpoint-{task}",
        "eval_condition": "clean",
        "episode_length": 1000,
    }
    if method is not None:
        row["method"] = method
    if ridge_alpha is not None:
        row["ridge_alpha"] = ridge_alpha
    return row


def test_pair_and_summarize_uses_episode_pairs_and_task_macro():
    mean_rows = [
        _row("walker_walk", 1.0, eval_seed=1),
        _row("walker_walk", 3.0, eval_seed=2),
        _row("quadruped_walk", 10.0, eval_seed=1),
    ]
    ridge_rows = [
        _row("walker_walk", 2.0, eval_seed=1),
        _row("walker_walk", 5.0, eval_seed=2),
        _row("quadruped_walk", 14.0, eval_seed=1),
    ]

    paired = pair_rows(mean_rows, ridge_rows)
    task_rows, macro_rows = summarize_pairs(paired)

    walker = next(row for row in task_rows if row["task"] == "walker_walk")
    assert walker["n_paired_episodes"] == 2
    assert walker["mean_mean"] == 2.0
    assert walker["ridge_mean"] == 3.5
    assert walker["ridge_minus_mean_mean"] == 1.5
    assert math.isclose(walker["ridge_minus_mean_std"], math.sqrt(0.5))

    assert len(macro_rows) == 1
    macro = macro_rows[0]
    assert macro["n_tasks"] == 2
    assert macro["n_paired_episodes"] == 3
    # Equal task weighting: mean([2, 10]), not mean([1, 3, 10]).
    assert macro["mean_macro_mean"] == 6.0
    assert macro["ridge_macro_mean"] == 8.75
    assert macro["ridge_minus_mean_macro_mean"] == 2.75


def test_pair_rows_rejects_missing_and_duplicate_cells():
    row = _row("walker_walk", 1.0)
    other_cell = _row("walker_walk", 2.0, eval_seed=2)

    with pytest.raises(ValueError, match="not exactly paired"):
        pair_rows([row], [other_cell])

    with pytest.raises(ValueError, match="duplicate completed pairing cells"):
        pair_rows([row, dict(row)], [row])


def test_pair_rows_rejects_mismatched_provenance():
    with pytest.raises(ValueError, match="mismatched evaluation provenance"):
        pair_rows(
            [_row("walker_walk", 1.0, bank_id="mean-bank")],
            [_row("walker_walk", 2.0, bank_id="ridge-bank")],
        )


def test_projection_provenance_accepts_different_methods_and_one_ridge_alpha():
    mean_rows = [
        _row("walker_walk", 1.0, method="reward_projection", eval_seed=1),
        _row("walker_walk", 2.0, method="reward_projection", eval_seed=2),
    ]
    ridge_rows = [
        _row(
            "walker_walk",
            1.5,
            method="reward_projection_ridge",
            ridge_alpha=0.01,
            eval_seed=1,
        ),
        _row(
            "walker_walk",
            2.5,
            method="reward_projection_ridge",
            ridge_alpha=0.01,
            eval_seed=2,
        ),
    ]

    assert validate_projection_provenance(mean_rows, ridge_rows) == 0.01
    ridge_rows[1]["ridge_alpha"] = 0.1
    with pytest.raises(ValueError, match="mixes ridge_alpha"):
        validate_projection_provenance(mean_rows, ridge_rows)


def test_cli_writes_csv_and_markdown(tmp_path):
    mean_dir = tmp_path / "mean" / "walker_walk"
    ridge_dir = tmp_path / "ridge" / "walker_walk"
    mean_dir.mkdir(parents=True)
    ridge_dir.mkdir(parents=True)
    (mean_dir / "episodes.jsonl").write_text(
        json.dumps(
            _row("walker_walk", 3.0, method="reward_projection")
        )
        + "\n"
    )
    (ridge_dir / "episodes.jsonl").write_text(
        json.dumps(
            _row(
                "walker_walk",
                4.5,
                method="reward_projection_ridge",
                ridge_alpha=0.01,
            )
        )
        + "\n"
    )
    output_dir = tmp_path / "comparison"

    assert (
        main(
            [
                "--mean-dir",
                str(tmp_path / "mean"),
                "--ridge-dir",
                str(tmp_path / "ridge"),
                "--output-dir",
                str(output_dir),
            ]
        )
        == 0
    )
    assert (output_dir / "paired_cells.csv").exists()
    assert (output_dir / "task_summary.csv").exists()
    assert (output_dir / "macro_summary.csv").exists()
    report = (output_dir / "report.md").read_text()
    assert "Exactly paired completed episodes: 1" in report
    assert "Ridge alpha: 0.01" in report
    assert "1.500" in report
