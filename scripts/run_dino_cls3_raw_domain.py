#!/usr/bin/env python3
"""Run the existing trainer with read-only Raw CLS3/domain startup assertions."""

import json
import os
from pathlib import Path
import sys


def audit_workspace(workspace, torch, pretrain):
    from launch_dino_cls3_raw_3domains import DOMAIN_SPECS

    cfg, agent = workspace.cfg, workspace.agent
    domain = cfg.task.split("_", 1)[0]
    spec = next(item for item in DOMAIN_SPECS if item[0] == domain)
    _, goal_space, backward_dim, tasks = spec
    expected_goal = None if goal_space == "null" else goal_space
    assert cfg.task == domain + "_walk"
    assert list(cfg.eval_tasks) == [domain + "_" + task for task in tasks]
    assert cfg.obs_type == "dino" and cfg.use_cls and cfg.dino_frame_stack == 3
    assert cfg.dino_model_name == "facebook/dinov2-base"
    assert cfg.goal_space == agent.cfg.goal_space == expected_goal
    assert not cfg.append_goal_to_observation and not cfg.update_encoder and cfg.reward_free
    assert not cfg.agent.dino_use_adapter and not cfg.agent.update_encoder and cfg.agent.idm_coef == 0
    assert not any((cfg.agent.dino_separate_fb_adapters, cfg.agent.dino_separate_backward_adapter,
                    cfg.agent.dino_flare_b, cfg.agent.pixel_separate_fb_encoders))
    assert tuple(cfg.agent.obs_shape) == (2304,)
    assert tuple(cfg.agent.action_shape) == ((12,) if domain == "quadruped" else (6,))
    assert agent.obs_dim == agent.actor.obs_dim == agent.forward_net.obs_dim == 2304
    assert agent.backward_net.obs_dim == agent.backward_target_net.obs_dim == backward_dim
    assert isinstance(agent.encoder, torch.nn.Identity), "Raw features must bypass all adapters"
    assert sum(parameter.numel() for parameter in agent.encoder.parameters()) == 0
    assert agent.encoder_opt is None and agent.forward_adapter is None
    assert agent.backward_encoder is None and agent.backward_adapter is None
    assert agent.idm_head is None and agent.idm_optimizer is None
    assert not cfg.auto_resume and cfg.load_model is None and cfg.load_replay_buffer is None
    assert workspace.global_step == 0 and len(workspace.replay_loader) == 0
    assert workspace._resume_checkpoint_filepath is None
    expected_lr = float(cfg.agent.lr)
    assert float(cfg.agent.fb_lr) == float(cfg.agent.lr_actor) == expected_lr
    assert float(cfg.agent.lr_coef) == 1.0
    optimizers = {name: value for name, value in vars(agent).items() if isinstance(value, torch.optim.Optimizer)}
    assert set(optimizers) == {"fb_opt", "actor_opt"}
    report_optimizers = {}
    for name, optimizer in optimizers.items():
        assert not optimizer.state, name + " unexpectedly has optimizer state"
        assert all(float(group["lr"]) == expected_lr for group in optimizer.param_groups), name
        report_optimizers[name] = [float(group["lr"]) for group in optimizer.param_groups]
    frozen_backbones = []
    for env_name in ("train_env", "eval_env"):
        env = getattr(workspace, env_name)
        assert tuple(env.observation_spec().shape) == (2304,)
        found = False
        while env is not None:
            if isinstance(env, pretrain.dmc.DinoV3EmbedWrapper):
                model = env._model
                assert model.config.model_type == "dinov2" and not model.training
                assert all(not parameter.requires_grad for parameter in model.parameters())
                frozen_backbones.append(env_name)
                found = True
                break
            try:
                env = object.__getattribute__(env, "_env")
            except AttributeError:
                env = None
        assert found, "Frozen DINO backbone missing from " + env_name
    return dict(
        status="PASS", domain=domain, task=cfg.task, eval_tasks=list(cfg.eval_tasks),
        obs_type=cfg.obs_type, dino_model_name=cfg.dino_model_name, dino_frame_stack=cfg.dino_frame_stack,
        observation_shape=[2304], encoder_type="Identity", encoder_parameters=0, encoder_optimizer=None,
        forward_actor_input_dim=2304, goal_space=cfg.goal_space, backward_input_dim=backward_dim,
        batch_size=int(cfg.agent.batch_size), num_train_frames=int(cfg.num_train_frames),
        replay_buffer_episodes=int(cfg.replay_buffer_episodes), optimizers=report_optimizers,
        frozen_backbones=frozen_backbones, fresh_start=True, pid=os.getpid(),
        gpu=os.environ.get("CUDA_VISIBLE_DEVICES"), torch_num_threads=torch.get_num_threads(),
        torch_num_interop_threads=torch.get_num_interop_threads(),
        read_only_no_forward_backward_or_optimizer_step=True,
    )


def main():
    repo = Path(__file__).resolve().parents[1]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    torch.set_num_interop_threads(1)
    from omegaconf import OmegaConf
    from url_benchmark import pretrain
    import wandb

    original_init = pretrain.BaseWorkspace.__init__

    def verified_init(workspace, cfg):
        original_init(workspace, cfg)
        report = audit_workspace(workspace, torch, pretrain)
        if cfg.use_wandb:
            assert wandb.run is not None and wandb.run.id == os.environ["WANDB_RUN_ID"]
            if os.environ.get("WANDB_MODE", "online") == "online":
                assert not wandb.run.settings._offline, "W&B must be online"
            report["wandb_url"] = wandb.run.url
            wandb.config.update({"startup_audit": report})
            wandb.run.summary.update({"startup_verified": True, "raw_cls3": True,
                                      "encoder_parameters": 0, "actual_batch_size": report["batch_size"],
                                      "actual_lr_f": float(cfg.agent.fb_lr), "actual_lr_b": float(cfg.agent.fb_lr),
                                      "actual_lr_actor": float(cfg.agent.lr_actor)})
        (workspace.work_dir / "startup_audit.json").write_text(json.dumps(report, indent=2) + "\n")
        resolved = OmegaConf.to_container(cfg, resolve=True)
        (workspace.work_dir / "resolved_config.json").write_text(json.dumps(resolved, indent=2) + "\n")
        print("STARTUP_AUDIT_PASS " + json.dumps(report), flush=True)

    pretrain.BaseWorkspace.__init__ = verified_init
    pretrain.main()


if __name__ == "__main__":
    main()
