#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from omegaconf import OmegaConf


TASKS = [
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
]

VARIANTS = [
    "dino_cls_linear",
    "dino_mean_linear",
    "dino_no_adapter",
    "cnn",
]


def _get(cfg: Any, path: str, default: Any = None) -> Any:
    cur = cfg
    for part in path.split("."):
        if not hasattr(cur, part):
            return default
        cur = getattr(cur, part)
    return cur


def classify_variant(cfg: Any) -> Optional[str]:
    obs_type = str(_get(cfg, "obs_type", ""))
    if obs_type == "pixels":
        return "cnn"
    if obs_type != "dino":
        return None

    use_adapter = bool(_get(cfg, "agent.dino_use_adapter", True))
    adapter_type = str(_get(cfg, "agent.dino_adapter_type", "linear"))
    use_cls = bool(_get(cfg, "use_cls", False))

    if not use_adapter:
        return "dino_no_adapter"
    if adapter_type != "linear":
        return None
    if use_cls:
        return "dino_cls_linear"
    return "dino_mean_linear"


def iter_config_paths(run_roots: Iterable[Path]) -> Iterable[Path]:
    for root in run_roots:
        if not root.exists():
            continue
        yield from root.glob("*/.hydra/config.yaml")


def checkpoint_for_frame(run_dir: Path, ckpt_roots: List[Path], frame: int) -> Optional[Path]:
    filename = f"snapshot_{frame}.pt"
    local = run_dir / "models" / filename
    if local.exists() and local.stat().st_size > 0:
        return local

    run_name = run_dir.name
    for root in ckpt_roots:
        exact = root / run_name / filename
        if exact.exists() and exact.stat().st_size > 0:
            return exact

    for root in ckpt_roots:
        matches = sorted(root.glob(f"*{run_name}*/{filename}"))
        for match in matches:
            if match.exists() and match.stat().st_size > 0:
                return match
    return None


def safe_float(value: str) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def best_row_for_run(run_dir: Path, ckpt_roots: List[Path]) -> Optional[Dict[str, Any]]:
    eval_csv = run_dir / "eval.csv"
    if not eval_csv.exists():
        return None

    best: Optional[Dict[str, Any]] = None
    with eval_csv.open("r", newline="") as f:
        for row in csv.DictReader(f):
            reward = safe_float(row.get("episode_reward", ""))
            frame_float = safe_float(row.get("frame", ""))
            if reward is None or frame_float is None:
                continue
            frame = int(frame_float)
            ckpt_path = checkpoint_for_frame(run_dir, ckpt_roots, frame)
            if ckpt_path is None:
                continue
            std = safe_float(row.get("episode_reward#std", ""))
            candidate = {
                "frame": frame,
                "ordinary_eval_reward": reward,
                "ordinary_eval_std": std,
                "ckpt_path": str(ckpt_path),
            }
            if best is None:
                best = candidate
                continue
            if (reward, frame) > (best["ordinary_eval_reward"], best["frame"]):
                best = candidate
    return best


def build_manifest(run_roots: List[Path], ckpt_roots: List[Path]) -> Tuple[List[Dict[str, Any]], List[Tuple[str, str]]]:
    selected: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for cfg_path in iter_config_paths(run_roots):
        run_dir = cfg_path.parents[1]
        try:
            cfg = OmegaConf.load(cfg_path)
        except Exception:
            continue

        task = str(_get(cfg, "task", ""))
        if task not in TASKS:
            continue
        variant = classify_variant(cfg)
        if variant not in VARIANTS:
            continue

        row = best_row_for_run(run_dir, ckpt_roots)
        if row is None:
            continue

        entry = {
            "task": task,
            "variant": variant,
            "run_dir": str(run_dir),
            "config_path": str(cfg_path),
            "seed": int(_get(cfg, "seed", 0)),
            **row,
        }
        key = (task, variant)
        old = selected.get(key)
        if old is None or (entry["ordinary_eval_reward"], entry["frame"]) > (old["ordinary_eval_reward"], old["frame"]):
            selected[key] = entry

    manifest = [selected[(task, variant)] for task in TASKS for variant in VARIANTS if (task, variant) in selected]
    missing = [(task, variant) for task in TASKS for variant in VARIANTS if (task, variant) not in selected]
    return manifest, missing


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--run-root",
        action="append",
        type=Path,
        default=[
            Path("/mnt/data_7tb/fanfeng/controllable_agent_runs"),
            Path("/data/fanfeng/controllable_agent/runs"),
        ],
    )
    parser.add_argument(
        "--ckpt-root",
        action="append",
        type=Path,
        default=[Path("/mnt/data_7tb/fanfeng/controallable_agent_ckpt")],
    )
    args = parser.parse_args()

    manifest, missing = build_manifest(args.run_root, args.ckpt_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(manifest, f, indent=2)

    summary_path = args.output.with_suffix(".csv")
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "task",
                "variant",
                "ordinary_eval_reward",
                "ordinary_eval_std",
                "frame",
                "seed",
                "ckpt_path",
                "run_dir",
            ],
        )
        writer.writeheader()
        for entry in manifest:
            writer.writerow({name: entry.get(name) for name in writer.fieldnames})

    print(f"wrote {len(manifest)} entries to {args.output}")
    print(f"wrote summary to {summary_path}")
    if missing:
        print("missing:")
        for task, variant in missing:
            print(f"  {task} {variant}")


if __name__ == "__main__":
    main()
