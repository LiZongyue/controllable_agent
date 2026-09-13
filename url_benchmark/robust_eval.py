#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
import traceback
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

import numpy as np
import omegaconf as omgcf
import torch
from transformers import AutoImageProcessor, AutoModel

from url_benchmark import dmc, goals as _goals, utils
from url_benchmark.pretrain import make_agent


DEFAULT_CONDITIONS = [
    "color_easy",
    "color_hard",
    "background_easy",
    "background_hard",
    "camera_easy",
    "camera_hard",
    "combined_easy",
]


def stable_seed(*parts: Any) -> int:
    text = "::".join(str(part) for part in parts)
    return zlib.crc32(text.encode("utf-8")) & 0x7FFFFFFF


def load_manifest(path: Path) -> List[Dict[str, Any]]:
    with path.open("r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list manifest in {path}")
    return data


def load_cfg(config_path: Path, device: str, episodes: int) -> omgcf.DictConfig:
    cfg = omgcf.OmegaConf.load(config_path)
    omgcf.OmegaConf.set_struct(cfg, False)
    cfg.device = device
    cfg.use_tb = False
    cfg.use_wandb = False
    cfg.use_hiplog = False
    cfg.save_video = False
    cfg.save_train_video = False
    cfg.num_eval_episodes = episodes
    cfg.load_model = None
    cfg.checkpoint_root = None
    cfg.agent.device = device
    cfg.agent.use_tb = False
    cfg.agent.use_wandb = False
    cfg.agent.use_hiplog = False
    return cfg


class DinoCache:
    def __init__(self, device: str) -> None:
        self.device = device
        self.processor = None
        self.model = None

    def get(self) -> Tuple[Any, Any]:
        if self.processor is None or self.model is None:
            self.processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base", use_fast=True)
            self.model = AutoModel.from_pretrained("facebook/dinov2-base")
            self.model.to(self.device)
            self.model.eval()
        return self.processor, self.model


def make_env(
    cfg: omgcf.DictConfig,
    dino_cache: DinoCache,
    condition: Optional[str],
    visual_seed: int,
) -> dmc.EnvWrapper:
    processor = None
    dino_model = None
    if cfg.obs_type == "dino":
        processor, dino_model = dino_cache.get()
    return dmc.make(
        cfg.task,
        cfg.obs_type,
        cfg.frame_stack,
        cfg.action_repeat,
        cfg.seed,
        goal_space=cfg.goal_space,
        append_goal_to_observation=cfg.append_goal_to_observation,
        dino_model=dino_model,
        dino_processor=processor,
        use_cls=cfg.use_cls,
        render_shape=tuple(cfg.render_shape),
        visual_perturbation=condition,
        visual_perturb_seed=visual_seed,
        dino_frame_stack=int(getattr(cfg, "dino_frame_stack", 1)),
    )


def load_agent(cfg: omgcf.DictConfig, env: dmc.EnvWrapper, checkpoint_path: Path) -> Tuple[Any, int]:
    agent = make_agent(
        cfg.obs_type,
        env.observation_spec(),
        env.action_spec(),
        cfg.num_seed_frames // cfg.action_repeat,
        cfg.agent,
    )
    print(f"loading checkpoint {checkpoint_path}", flush=True)
    payload = torch.load(checkpoint_path, map_location=cfg.device, weights_only=False)
    agent.init_from(payload["agent"])
    return agent, int(payload.get("global_step", 0))


def infer_meta_from_clean_rollout(
    agent: Any,
    env: dmc.EnvWrapper,
    cfg: omgcf.DictConfig,
    transitions: int,
    custom_reward: Optional[_goals.BaseReward] = None,
) -> Dict[str, np.ndarray]:
    if transitions <= 0:
        return agent.init_meta()

    backward_input_list: List[np.ndarray] = []
    reward_list: List[float] = []
    time_step = env.reset()
    meta = agent.init_meta()
    step = 0

    while len(backward_input_list) < transitions:
        with torch.no_grad(), utils.eval_mode(agent):
            action = agent.act(time_step.observation, meta, step, eval_mode=False)
        time_step = env.step(action)
        if custom_reward is not None:
            time_step.reward = custom_reward.from_env(env)
        if cfg.goal_space is not None:
            if not hasattr(time_step, "goal"):
                raise RuntimeError(f"Expected goal in time_step for goal_space={cfg.goal_space}")
            backward_input = time_step.goal
        else:
            backward_input = time_step.observation
        backward_input_list.append(np.asarray(backward_input))
        reward_list.append(float(time_step.reward))
        meta = agent.update_meta(meta, step, time_step, finetune=False, replay_loader=None)
        step += 1
        if time_step.last() and len(backward_input_list) < transitions:
            time_step = env.reset()
            meta = agent.init_meta()

    obs = torch.as_tensor(np.stack(backward_input_list, axis=0), device=cfg.device, dtype=torch.float32)
    reward = torch.as_tensor(np.asarray(reward_list, dtype=np.float32).reshape(-1, 1), device=cfg.device)
    return agent.infer_meta_from_obs_and_rewards(obs, reward)


def build_eval_meta(
    agent: Any,
    env: dmc.EnvWrapper,
    cfg: omgcf.DictConfig,
    transitions: int,
    custom_reward: Optional[_goals.BaseReward] = None,
) -> Tuple[Dict[str, np.ndarray], str]:
    if custom_reward is not None:
        try:
            return agent.get_goal_meta(custom_reward.get_goal(cfg.goal_space)), "custom_reward_goal"
        except Exception:
            pass
    if cfg.goal_space is not None:
        funcs = _goals.goals.funcs.get(cfg.goal_space, {})
        if cfg.task in funcs:
            return agent.get_goal_meta(funcs[cfg.task]()), "registered_goal"
    source = "clean_goal_rollout" if cfg.goal_space is not None else "clean_obs_rollout"
    if transitions <= 0:
        source = "init_meta"
    return infer_meta_from_clean_rollout(agent, env, cfg, transitions, custom_reward), source


def evaluate_condition(
    agent: Any,
    env: dmc.EnvWrapper,
    meta: Dict[str, np.ndarray],
    cfg: omgcf.DictConfig,
    episodes: int,
    global_step: int,
    custom_reward: Optional[_goals.BaseReward] = None,
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    rewards: List[float] = []
    lengths: List[int] = []
    for episode in range(episodes):
        time_step = env.reset()
        total_reward = 0.0
        steps = 0
        while not time_step.last():
            with torch.no_grad(), utils.eval_mode(agent):
                action = agent.act(time_step.observation, meta, global_step, eval_mode=True)
            time_step = env.step(action)
            if custom_reward is not None:
                time_step.reward = custom_reward.from_env(env)
            total_reward += float(time_step.reward)
            steps += 1
        rewards.append(total_reward)
        lengths.append(steps * int(cfg.action_repeat))
        print(f"  episode {episode + 1}/{episodes}: reward={total_reward:.4f}", flush=True)

    summary = {
        "episode_reward": float(np.mean(rewards)),
        "episode_reward_std": float(np.std(rewards)),
        "episode_length": float(np.mean(lengths)),
    }
    episode_rows = [
        {"episode": idx, "episode_reward": float(reward), "episode_length": float(length)}
        for idx, (reward, length) in enumerate(zip(rewards, lengths))
    ]
    return summary, episode_rows


def read_completed(result_path: Path) -> set:
    completed = set()
    if not result_path.exists():
        return completed
    with result_path.open("r") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            completed.add((row["task"], row["variant"], row["condition"], row["ckpt_path"]))
    return completed


def read_all_completed(output_dir: Path) -> set:
    completed = set()
    for result_path in output_dir.glob("shard_*/results.jsonl"):
        completed.update(read_completed(result_path))
    return completed


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
        f.flush()


def process_entry(
    entry: Dict[str, Any],
    args: argparse.Namespace,
    conditions: Sequence[str],
    completed: set,
    result_path: Path,
    episode_path: Path,
    error_path: Path,
    dino_cache: DinoCache,
) -> None:
    task = entry["task"]
    variant = entry["variant"]
    ckpt_path = Path(entry["ckpt_path"])
    if args.resume and all((task, variant, condition, str(ckpt_path)) in completed for condition in conditions):
        print(f"[{task} {variant}] all conditions already done, skipping entry", flush=True)
        return

    cfg = load_cfg(Path(entry["config_path"]), args.device, args.episodes)
    cfg.seed = int(args.seed)
    utils.set_seed_everywhere(int(args.seed))
    custom_reward = None
    if cfg.custom_reward is not None:
        custom_reward = _goals.get_reward_function(cfg.custom_reward, stable_seed(args.seed, task, variant, "meta_reward"))

    clean_env = make_env(cfg, dino_cache, condition=None, visual_seed=stable_seed(args.seed, task, variant, "meta"))
    agent, global_step = load_agent(cfg, clean_env, ckpt_path)
    if custom_reward is not None:
        print(f"[{task} {variant}] using custom_reward={cfg.custom_reward} for meta", flush=True)
    elif cfg.goal_space is not None and cfg.task in _goals.goals.funcs.get(cfg.goal_space, {}):
        print(f"[{task} {variant}] using registered goal meta for {cfg.goal_space}", flush=True)
    else:
        print(f"[{task} {variant}] inferring meta from {args.calibration_transitions} clean transitions", flush=True)
    meta, meta_source = build_eval_meta(agent, clean_env, cfg, args.calibration_transitions, custom_reward)

    for condition in conditions:
        key = (task, variant, condition, str(ckpt_path))
        if args.resume and key in completed:
            print(f"[{task} {variant} {condition}] already done, skipping", flush=True)
            continue
        print(f"[{task} {variant} {condition}] evaluating {args.episodes} episodes", flush=True)
        started = time.time()
        try:
            env = make_env(cfg, dino_cache, condition=condition, visual_seed=stable_seed(args.seed, task, variant, condition))
            condition_reward = None
            if cfg.custom_reward is not None:
                condition_reward = _goals.get_reward_function(
                    cfg.custom_reward,
                    stable_seed(args.seed, task, variant, condition, "reward"),
                )
            summary, episodes = evaluate_condition(
                agent,
                env,
                meta,
                cfg,
                args.episodes,
                global_step,
                condition_reward,
            )
            row = {
                **entry,
                **summary,
                "condition": condition,
                "episodes": args.episodes,
                "calibration_transitions": args.calibration_transitions,
                "duration_sec": time.time() - started,
                "global_step": global_step,
                "meta_source": meta_source,
                "worker_seed": int(args.seed),
            }
            append_jsonl(result_path, row)
            for episode in episodes:
                append_jsonl(episode_path, {**entry, "condition": condition, **episode})
            completed.add(key)
        except Exception as exc:
            append_jsonl(
                error_path,
                {
                    **entry,
                    "condition": condition,
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                },
            )
            print(f"[{task} {variant} {condition}] ERROR {exc!r}", flush=True)
            traceback.print_exc()
        finally:
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--conditions", default=",".join(DEFAULT_CONDITIONS))
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--calibration-transitions", type=int, default=5120)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260502)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
    os.environ.setdefault("MUJOCO_GL", "egl")

    conditions = [condition for condition in args.conditions.split(",") if condition]
    manifest = load_manifest(args.manifest)
    manifest = [entry for idx, entry in enumerate(manifest) if idx % args.num_shards == args.shard_index]
    if args.limit:
        manifest = manifest[:args.limit]

    shard_dir = args.output_dir / f"shard_{args.shard_index:02d}"
    result_path = shard_dir / "results.jsonl"
    episode_path = shard_dir / "episodes.jsonl"
    error_path = shard_dir / "errors.jsonl"
    completed = read_all_completed(args.output_dir) if args.resume else set()
    dino_cache = DinoCache(args.device)

    print(
        f"shard {args.shard_index}/{args.num_shards}: {len(manifest)} checkpoints, "
        f"{len(conditions)} conditions, {args.episodes} episodes each",
        flush=True,
    )
    for entry in manifest:
        try:
            process_entry(entry, args, conditions, completed, result_path, episode_path, error_path, dino_cache)
        except Exception as exc:
            append_jsonl(
                error_path,
                {
                    **entry,
                    "condition": "__entry__",
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                },
            )
            print(f"[{entry.get('task')} {entry.get('variant')}] ENTRY ERROR {exc!r}", flush=True)
            traceback.print_exc()
        finally:
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
