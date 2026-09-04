# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
import pdb  # pylint: disable=unused-import
import logging
import dataclasses
import sys
import typing as tp
import warnings
from pathlib import Path

warnings.filterwarnings('ignore', category=DeprecationWarning)

# Allow `python url_benchmark/pretrain.py` in addition to module execution.
if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
# if the default egl does not work, you may want to try:
# export MUJOCO_GL=glfw
os.environ['MUJOCO_GL'] = os.environ.get('MUJOCO_GL', 'egl')

import hydra
from hydra.core.config_store import ConfigStore
import numpy as np
import torch
import wandb
import omegaconf as omgcf
# from dm_env import specs

from url_benchmark import dmc
from dm_env import specs
from url_benchmark import utils
from url_benchmark import goals as _goals
from url_benchmark.logger import Logger
from url_benchmark.in_memory_replay_buffer import ReplayBuffer
from url_benchmark.video import TrainVideoRecorder, VideoRecorder
from url_benchmark import agent as agents
from url_benchmark.d4rl_benchmark import D4RLReplayBufferBuilder, D4RLWrapper
from url_benchmark.gridworld.env import build_gridworld_task

from transformers import AutoImageProcessor, AutoModel

logger = logging.getLogger(__name__)
torch.backends.cudnn.benchmark = True
# os.environ['WANDB_MODE']='offline'

# from url_benchmark.dmc_benchmark import PRIMAL_TASKS


# # # Config # # #

@dataclasses.dataclass
class Config:
    agent: tp.Any
    # misc
    seed: int = 1
    device: str = "cuda"
    save_video: bool = False
    use_tb: bool = False
    use_wandb: bool = False
    use_hiplog: bool = False
    # experiment
    experiment: str = "online"
    # task settings
    task: str = "walker_stand"
    obs_type: str = "dino"  # [states, pixels, dino, vit]
    frame_stack: int = 3  # pixel stack; retained for compatibility with historical configs
    dino_frame_stack: int = 1  # explicit DINO embedding stack; historical DINO runs used one frame
    action_repeat: int = 2  # set to 2 for pixels
    use_cls: bool = True  # whether to use the dino cls token instead of the mean pooled features, only works if obs_type=dino
    # Kept configurable so profiling/reproduction can pin the visual backbone used
    # by an experiment.  The default preserves the current training behavior.
    dino_model_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    discount: float = 0.99
    render_shape: tp.Tuple[int, int] = (224, 224)  # only used for visual obs (pixels, dino, vit)
    future: float = 0.99  # discount of future sampling, future=1 means no future sampling
    goal_space: tp.Optional[str] = "simplified_walker"
    append_goal_to_observation: bool = False
    # eval
    num_eval_episodes: int = 10
    custom_reward: tp.Optional[str] = None  # activates custom eval if not None
    final_tests: int = 10
    # checkpoint
    snapshot_at: tp.Tuple[int, ...] = (100000, 200000, 500000, 800000, 1000000, 1500000,
                                       2000000, 3000000, 4000000, 5000000, 9000000, 10000000)
    checkpoint_every: int = 100000
    load_model: tp.Optional[str] = None
    auto_resume: bool = True
    checkpoint_root: tp.Optional[str] = "/mnt/data_7tb/fanfeng/controallable_agent_ckpt"
    save_replay_buffer_in_checkpoint: bool = False
    # training
    num_seed_frames: int = 4000
    replay_buffer_episodes: int = 5000
    update_encoder: bool = True
    batch_size: int = omgcf.II("agent.batch_size")


@dataclasses.dataclass
class PretrainConfig(Config):
    # mode
    reward_free: bool = True
    # train settings
    num_train_frames: int = 2000010
    # snapshot
    eval_every_frames: int = 10000
    load_replay_buffer: tp.Optional[str] = None
    # replay buffer
    # replay_buffer_num_workers: int = 4
    # nstep: int = omgcf.II("agent.nstep")
    # misc
    save_train_video: bool = False


# loaded as base_pretrain in pretrain.yaml
# we keep the yaml since it's easier to configure plugins from it
ConfigStore.instance().store(name="workspace_config", node=PretrainConfig)


# # # Implem # # #


def _validate_idm_training_config(cfg: tp.Any) -> None:
    """Keep the IDM auxiliary on the three-frame DINO CLS FB variant."""
    idm_coef = float(getattr(cfg.agent, "idm_coef", 0.0))
    idm_lr = getattr(cfg.agent, "idm_lr", None)
    idm_encoder_mode = str(getattr(cfg.agent, "idm_encoder_mode", "legacy"))
    if idm_coef < 0:
        raise ValueError("agent.idm_coef must be non-negative")
    if idm_lr is not None and float(idm_lr) <= 0:
        raise ValueError("agent.idm_lr must be positive when provided")
    if idm_coef > 0 and not bool(getattr(cfg, "update_encoder", False)):
        raise ValueError("agent.idm_coef > 0 requires update_encoder=True")
    idm_configured = (
        idm_coef != 0
        or idm_lr is not None
        or idm_encoder_mode != "legacy"
        or int(getattr(cfg.agent, "idm_encoder_burnin_steps", 0)) != 0
        or int(getattr(cfg.agent, "idm_encoder_ramp_steps", 0)) != 0
        or getattr(cfg.agent, "idm_grad_ratio_target", None) is not None
    )
    if idm_configured and (
        getattr(cfg.agent, "name", None) != "fb_ddpg"
        or cfg.obs_type != "dino"
        or not cfg.use_cls
        or cfg.dino_frame_stack != 3
    ):
        raise ValueError(
            "IDM auxiliary training is supported only for FB with three-frame DINO CLS observations"
        )


def _init_wandb(cfg: tp.Any, exp_name: str) -> None:
    """Initialize a resumable run with frame-based metric namespaces."""
    wandb_project = os.environ.get("WANDB_PROJECT", "controllable_agent_baseline")
    wandb_kwargs: tp.Dict[str, tp.Any] = {}
    wandb_run_id = os.environ.get("WANDB_RUN_ID")
    wandb_resume = os.environ.get("WANDB_RESUME")
    if wandb_resume and not wandb_run_id:
        raise ValueError("WANDB_RESUME requires a stable WANDB_RUN_ID")
    if wandb_run_id:
        wandb_kwargs["id"] = wandb_run_id
        wandb_kwargs["resume"] = wandb_resume or "allow"

    wandb.init(
        project=wandb_project,
        group=cfg.agent.name,
        name=os.environ.get("WANDB_RUN_NAME", exp_name),
        config=omgcf.OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True
        ),
        **wandb_kwargs,
    )  # type: ignore
    # W&B's internal step is process-local and can jump after checkpoint
    # recovery. Every experiment metric instead uses environment frames.
    for namespace in ("train", "eval", "final"):
        frame_metric = f"{namespace}/frame"
        wandb.define_metric(frame_metric)
        wandb.define_metric(f"{namespace}/*", step_metric=frame_metric)


def make_agent(
    obs_type: str, obs_spec, action_spec, num_expl_steps: int, cfg: omgcf.DictConfig
) -> tp.Union[agents.FBDDPGAgent, agents.DDPGAgent]:
    cfg.obs_type = obs_type
    if obs_type == "pixels" or obs_type == "vit" or obs_type == "dino":
        cfg.obs_shape = obs_spec.shape
    elif obs_type == "states":
        cfg.obs_shape = obs_spec["observations"].shape if isinstance(obs_spec, dict) else obs_spec.shape
    else:
        raise ValueError(f"Unsupported obs_type={obs_type} for obs_spec={type(obs_spec)}")

    cfg.action_shape = (action_spec.num_values, ) if isinstance(action_spec, specs.DiscreteArray) \
        else action_spec.shape
    cfg.num_expl_steps = num_expl_steps
    return hydra.utils.instantiate(cfg)


C = tp.TypeVar("C", bound=Config)


def _update_legacy_class(obj: tp.Any, classes: tp.Sequence[tp.Type[tp.Any]]) -> tp.Any:
    """Updates a legacy class (eg: agent.FBDDPGAgent) to the new
    class (url_benchmark.agent.FBDDPGAgent)

    Parameters
    ----------
    obj: Any
        Object to update
    classes: Types
        Possible classes to update the object to. If current name is one of the classes
        name, the object class will be remapped to it.
    """
    classes = tuple(classes)
    if not isinstance(obj, classes):
        clss = {x.__name__: x for x in classes}
        cls = clss.get(obj.__class__.__name__, None)
        if cls is not None:
            logger.warning(f"Promoting legacy object {obj.__class__} to {cls}")
            obj.__class__ = cls


def _init_eval_meta(workspace: "BaseWorkspace", custom_reward: tp.Optional[_goals.BaseReward] = None) -> agents.MetaDict:
    if workspace.domain == "grid":
        assert isinstance(workspace.agent, agents.DiscreteFBAgent)
        return workspace.agent.get_goal_meta(workspace.eval_env.get_goal_obs())
    special = (agents.FBDDPGAgent, agents.SFAgent, agents.SFSVDAgent, agents.APSAgent, agents.NEWAPSAgent, agents.GoalSMAgent, agents.UVFAgent)
    ag = workspace.agent
    _update_legacy_class(ag, special)
    # we need to check against name for legacy reason when reloading old checkpoints
    if not isinstance(ag, special) or not len(workspace.replay_loader):
        return workspace.agent.init_meta()
    if custom_reward is not None:
        try:  # if the custom reward implements a goal, return it
            goal = custom_reward.get_goal(workspace.cfg.goal_space)
            return workspace.agent.get_goal_meta(goal)
        except Exception:  # pylint: disable=broad-except
            pass
        if not isinstance(workspace.agent, agents.SFSVDAgent):
            # we cannot fully type because of the FBBDPG string check :s
            num_steps = workspace.agent.cfg.num_inference_steps  # type: ignore
            obs_list, reward_list = [], []
            batch_size = 0
            while batch_size < num_steps:
                batch = workspace.replay_loader.sample(workspace.cfg.batch_size, custom_reward=custom_reward)
                batch = batch.to(workspace.cfg.device)
                obs_list.append(batch.next_goal if workspace.cfg.goal_space is not None else batch.next_obs)
                reward_list.append(batch.reward)
                batch_size += batch.next_obs.size(0)
            obs, reward = torch.cat(obs_list, 0), torch.cat(reward_list, 0)  # type: ignore
            obs_t, reward_t = obs[:num_steps], reward[:num_steps]
            # phy = workspace.replay_loader._storage["physics"]
            # phy = phy.reshape(-1, phy.shape[-1])
            # back_input = "observation" if workspace.cfg.goal_space is None else "goal"
            # obs = workspace.replay_loader._storage[back_input].reshape(phy.shape[0], -1)  # should have been next obs
            # inds = np.random.choice(phy.shape[0], size=workspace.agent.cfg.num_inference_steps, replace=False)
            # phy, obs = (x[inds, :] for x in (phy, obs))
            # rewards = [[custom_reward.from_physics(p)] for p in phy]
            # obs_t, reward_t = (torch.Tensor(x).float().to(workspace.agent.cfg.device) for x in (obs, rewards))
            return workspace.agent.infer_meta_from_obs_and_rewards(obs_t, reward_t)
        else:
            assert isinstance(workspace.agent, agents.SFSVDAgent)
            obs_list, reward_list, action_list = [], [], []
            batch_size = 0
            while batch_size < workspace.agent.cfg.num_inference_steps:
                batch = workspace.replay_loader.sample(workspace.cfg.batch_size, custom_reward=custom_reward)
                batch = batch.to(workspace.cfg.device)
                obs_list.append(batch.goal if workspace.cfg.goal_space is not None else batch.obs)
                action_list.append(batch.action)
                reward_list.append(batch.reward)
                batch_size += batch.next_obs.size(0)
            obs, reward, action = torch.cat(obs_list, 0), torch.cat(reward_list, 0), torch.cat(action_list, 0)  # type: ignore
            obs_t, reward_t, action_t = obs[:workspace.agent.cfg.num_inference_steps], reward[:workspace.agent.cfg.num_inference_steps],\
                action[:workspace.agent.cfg.num_inference_steps]
            return workspace.agent.infer_meta_from_obs_action_and_rewards(obs_t, action_t, reward_t)

    if workspace.cfg.goal_space is not None:
        funcs = _goals.goals.funcs.get(workspace.cfg.goal_space, {})
        if workspace.cfg.task in funcs:
            g = funcs[workspace.cfg.task]()
            return workspace.agent.get_goal_meta(g)
    return workspace.agent.infer_meta(workspace.replay_loader)


class BaseWorkspace(tp.Generic[C]):
    @staticmethod
    def _checkpoint_path_for(work_dir: Path, checkpoint_root: tp.Optional[str]) -> Path:
        if checkpoint_root is None:
            return work_dir / "models" / "latest.pt"
        return Path(checkpoint_root).expanduser() / work_dir.name / "latest.pt"

    def __init__(self, cfg: C) -> None:
        self.work_dir = Path.cwd()
        print(f'Workspace: {self.work_dir}')
        print(f'Running code in : {Path(__file__).parent.resolve().absolute()}')
        logger.info(f'Workspace: {self.work_dir}')
        logger.info(f'Running code in : {Path(__file__).parent.resolve().absolute()}')

        self.cfg = cfg
        _validate_idm_training_config(cfg)
        utils.set_seed_everywhere(cfg.seed)
        if not torch.cuda.is_available():
            if cfg.device != "cpu":
                logger.warning(f"Falling back to cpu as {cfg.device} is not available")
                cfg.device = "cpu"
                cfg.agent.device = "cpu"
        self.device = torch.device(cfg.device)
        # goal_spec: tp.Optional[specs.Array] = None
        # if cfg.goal_space is not None:
        #     g = _goals.goals.funcs[cfg.goal_space][cfg.task]()
        #     goal_spec = specs.Array((len(g),), np.float32, 'goal')

        # create envs
        # task = PRIMAL_TASKS[self.domain]
        task = cfg.task
        if task.startswith('point_mass_maze'):
            self.domain = 'point_mass_maze'
        elif task.startswith('point_mass_'):
            self.domain = 'point_mass'
        else:
            self.domain = task.split('_', maxsplit=1)[0]

        self.train_env = self._make_env()
        self.eval_env = self._make_env()
        # create agent
        self.agent = make_agent(cfg.obs_type,
                                self.train_env.observation_spec(),
                                self.train_env.action_spec(),
                                cfg.num_seed_frames // cfg.action_repeat,
                                cfg.agent)

        # create logger
        self.logger = Logger(self.work_dir,
                             use_tb=cfg.use_tb,
                             use_wandb=cfg.use_wandb,
                             use_hiplog=cfg.use_hiplog)

        if cfg.use_wandb:
            exp_name_parts = [cfg.experiment, cfg.obs_type, cfg.agent.name]
            if cfg.obs_type == "dino":
                exp_name_parts.append("cls" if cfg.use_cls else "patch")
                exp_name_parts.append(cfg.agent.dino_adapter_type if cfg.agent.dino_use_adapter else "no_adaptor")
            exp_name_parts.append(cfg.task)
            exp_name = '_'.join(exp_name_parts)
            _init_wandb(cfg, exp_name)

        if cfg.use_hiplog:
            # record config now that it is filled
            parts = ("snapshot", "_type", "_shape", "num_", "save_", "frame", "device", "use_tb", "use_wandb")
            skipped = [x for x in cfg if any(y in x for y in parts)]  # type: ignore
            self.logger.hiplog.flattened({x: y for x, y in cfg.items() if x not in skipped})  # type: ignore
            self.logger.hiplog(workdir=self.work_dir.stem)
            for rm in ("agent/use_tb", "agent/use_wandb", "agent/device"):
                del self.logger.hiplog._content[rm]
            self.logger.hiplog(observation_size=np.prod(self.train_env.observation_spec().shape))

        # # create replay buffer
        # self._data_specs: tp.List[tp.Any] = [self.train_env.observation_spec(),
        #                                      self.train_env.action_spec(), ]
        if cfg.goal_space is not None:
            if cfg.goal_space not in _goals.goal_spaces.funcs[self.domain]:
                raise ValueError(f"Unregistered goal space {cfg.goal_space} for domain {self.domain}")
        #     g = _goals.goals.funcs[cfg.goal_space][cfg.task]()
        #     self._data_specs.append(specs.Array((len(g),), np.float32, 'goal'))
        # self._data_specs.extend([specs.Array((1,), np.float32, 'reward'),
        #                          specs.Array((1,), np.float32, 'discount')])

        self.replay_loader = ReplayBuffer(max_episodes=cfg.replay_buffer_episodes, discount=cfg.discount, future=cfg.future)

        # # create data storage
        # self.replay_storage = ReplayBufferStorage(data_specs, meta_specs,
        #                                           self.work_dir / 'buffer')
        #
        # # create replay buffer
        # self.replay_loader = make_replay_loader(self.replay_storage,
        #                                         cfg.replay_buffer_size,
        #                                         cfg.batch_size,
        #                                         cfg.replay_buffer_num_workers,
        #                                         False, True, cfg.nstep, cfg.discount)

        # create video recorders
        # cam_id = 2 if 'quadruped' not in self.domain else 1
        # cam_id = 1  # centered on subject
        cam_id = 0 if 'quadruped' not in self.domain else 2

        self.video_recorder = VideoRecorder(self.work_dir if cfg.save_video else None,
                                            camera_id=cam_id, use_wandb=self.cfg.use_wandb)

        self.timer = utils.Timer()
        self.global_step = 0
        self.global_episode = 0
        self.eval_rewards_history: tp.List[float] = []
        self._needs_replay_warmup = False
        self._replay_warmup_warned = False
        self._legacy_checkpoint_filepath = self.work_dir / "models" / "latest.pt"
        self._checkpoint_filepath = self._checkpoint_path_for(self.work_dir, cfg.checkpoint_root)
        self._resume_checkpoint_filepath: tp.Optional[Path] = None
        if cfg.auto_resume:
            for candidate in (self._checkpoint_filepath, self._legacy_checkpoint_filepath):
                if candidate.exists():
                    self._resume_checkpoint_filepath = candidate
                    break
        if self._resume_checkpoint_filepath is not None:
            self.load_checkpoint(
                self._resume_checkpoint_filepath,
                strict_optimizer_lr=True,
            )
        elif cfg.load_model is not None:
            self.load_checkpoint(
                cfg.load_model,
                exclude=["replay_loader"],
                strict_optimizer_lr=False,
            )

        self.reward_cls: tp.Optional[_goals.BaseReward] = None
        if self.cfg.custom_reward == "maze_multi_goal":
            self.reward_cls = self._make_custom_reward(seed=self.cfg.seed)

    def _make_env(self) -> dmc.EnvWrapper:
        cfg = self.cfg
        dino_model = None
        processor = None
        if self.domain == "grid":
            return dmc.EnvWrapper(build_gridworld_task(self.cfg.task.split('_')[1]))
        if self.domain == "d4rl":
            import d4rl  # type: ignore # pylint: disable=unused-import
            import gym
            return dmc.EnvWrapper(D4RLWrapper(gym.make(self.cfg.task.split('_')[1])))
        if cfg.obs_type == 'dino':
            processor = AutoImageProcessor.from_pretrained(cfg.dino_model_name, use_fast=True)
            model = AutoModel.from_pretrained(cfg.dino_model_name)

            model.to(cfg.device)
            model.eval()

            dino_model = model
        return dmc.make(cfg.task, cfg.obs_type, cfg.frame_stack, cfg.action_repeat, cfg.seed,
                        goal_space=cfg.goal_space, append_goal_to_observation=cfg.append_goal_to_observation,
                        dino_model=dino_model, dino_processor=processor, use_cls=cfg.use_cls,
                        render_shape=cfg.render_shape, dino_frame_stack=cfg.dino_frame_stack)

    @property
    def global_frame(self) -> int:
        return self.global_step * self.cfg.action_repeat

    def _make_custom_reward(self, seed: int) -> tp.Optional[_goals.BaseReward]:
        """Creates a custom reward function if provided in configuration
        else returns None
        """
        if self.cfg.custom_reward is None:
            return None
        return _goals.get_reward_function(self.cfg.custom_reward, seed)

    def eval_maze_goals(self) -> None:
        if isinstance(self.agent, (agents.SFAgent, agents.SFSVDAgent, agents.NEWAPSAgent)) and len(self.replay_loader) > 0:
            self.agent.precompute_cov(self.replay_loader)
        reward_cls = _goals.MazeMultiGoal()
        rewards = list()
        for g in reward_cls.goals:
            goal_rewards = list()
            goal_distances = list()
            meta = self.agent.get_goal_meta(g)
            for episode in range(self.cfg.num_eval_episodes):
                time_step = self.eval_env.reset()
                self.video_recorder.init(self.eval_env, enabled=(episode == 0))
                episode_reward = 0.0
                while not time_step.last():
                    with torch.no_grad(), utils.eval_mode(self.agent):
                        action = self.agent.act(time_step.observation,
                                                meta,
                                                0,
                                                eval_mode=True)
                    time_step = self.eval_env.step(action)
                    self.video_recorder.record(self.eval_env)
                    assert isinstance(time_step, dmc.ExtendedGoalTimeStep)
                    step_reward, distance = reward_cls.from_goal(time_step.goal, g)
                    episode_reward += step_reward
                goal_rewards.append(episode_reward)
                goal_distances.append(distance)
                self.video_recorder.save(f'{g}.mp4')
            print(f"goal: {g}, avg_reward: {round(float(np.mean(goal_rewards)), 2)}, avg_distance: {round(float(np.mean(goal_distances)), 5)}")
            rewards.append(float(np.mean(goal_rewards)))
        self.eval_rewards_history.append(float(np.mean(rewards)))
        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('episode_reward', self.eval_rewards_history[-1])
            log('step', self.global_step)
            log('episode', self.global_episode)

    def eval(self, log_metrics: bool = True) -> float:
        step, episode = 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)
        physics_agg = dmc.PhysicsAggregator()
        rewards: tp.List[float] = []
        normalized_scores: tp.List[float] = []
        meta = _init_eval_meta(self)  # Don't work
        z_correl = 0.0
        is_d4rl_task = self.cfg.task.split('_')[0] == 'd4rl'
        actor_success: tp.List[float] = []
        while eval_until_episode(episode):
            time_step = self.eval_env.reset()
            # create custom reward if need be (if field exists)
            seed = 12 * self.cfg.num_eval_episodes + len(rewards)
            custom_reward = self._make_custom_reward(seed=seed)
            if custom_reward is not None:
                meta = _init_eval_meta(self, custom_reward)
            if self.domain == "grid":
                meta = _init_eval_meta(self)
            total_reward = 0.0
            self.video_recorder.init(self.eval_env, enabled=(episode == 0))
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(time_step.observation,
                                            meta,
                                            self.global_step,
                                            eval_mode=True)
                time_step = self.eval_env.step(action)
                physics_agg.add(self.eval_env)
                self.video_recorder.record(self.eval_env)
                # for legacy reasons, we need to check the name :s
                if isinstance(self.agent, agents.FBDDPGAgent):
                    if self.agent.cfg.additional_metric:
                        z_correl += self.agent.compute_z_correl(time_step, meta)
                        actor_success.extend(self.agent.actor_success)
                if custom_reward is not None:
                    time_step.reward = custom_reward.from_env(self.eval_env)
                total_reward += time_step.reward
                step += 1
            if is_d4rl_task:
                normalized_scores.append(self.eval_env.get_normalized_score(total_reward))
            rewards.append(total_reward)
            episode += 1
            self.video_recorder.save(f'{self.global_frame}.mp4')

        mean_reward = float(np.mean(rewards))
        self.eval_rewards_history.append(mean_reward)
        if log_metrics:
            with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
                if is_d4rl_task:
                    log('episode_normalized_score', float(100 * np.mean(normalized_scores)))
                log('episode_reward', mean_reward)
                if len(rewards) > 1:
                    log('episode_reward#std', float(np.std(rewards)))
                log('episode_length', step * self.cfg.action_repeat / episode)
                log('episode', self.global_episode)
                log('z_correl', z_correl / episode)
                log('step', self.global_step)
                if actor_success:
                    log('actor_sucess', float(np.mean(actor_success)))
                if isinstance(self.agent, agents.FBDDPGAgent):
                    log('z_norm', np.linalg.norm(meta['z']).item())
                for key, val in physics_agg.dump():
                    log(key, val)
        return mean_reward

    _CHECKPOINTED_KEYS = ('agent', 'global_step', 'global_episode', "replay_loader")

    def save_checkpoint(self, fp: tp.Union[Path, str], exclude: tp.Sequence[str] = ()) -> None:
        logger.info(f"Saving checkpoint to {fp}")
        exclude = list(exclude)
        if not self.cfg.save_replay_buffer_in_checkpoint and "replay_loader" not in exclude:
            exclude.append("replay_loader")
        assert all(x in self._CHECKPOINTED_KEYS for x in exclude)
        fp = Path(fp)
        fp.parent.mkdir(exist_ok=True, parents=True)
        if "replay_loader" not in exclude:
            assert isinstance(self.replay_loader, ReplayBuffer), "Is this buffer designed for checkpointing?"
        # this is just a dumb security check to not forget about it
        payload = {k: self.__dict__[k] for k in self._CHECKPOINTED_KEYS if k not in exclude}
        tmp_fp = fp.with_name(f".{fp.name}.tmp.{os.getpid()}")
        try:
            with tmp_fp.open('wb') as f:
                torch.save(payload, f, pickle_protocol=4)
                f.flush()
                os.fsync(f.fileno())
            tmp_fp.replace(fp)
        finally:
            if tmp_fp.exists():
                tmp_fp.unlink()

    def load_checkpoint(
        self,
        fp: tp.Union[Path, str],
        only: tp.Optional[tp.Sequence[str]] = None,
        exclude: tp.Sequence[str] = (),
        *,
        strict_optimizer_lr: bool = False,
    ) -> None:
        """Reloads a checkpoint or part of it

        Parameters
        ----------
        only: None or sequence of str
            reloads only a specific subset (defaults to all)
        exclude: sequence of str
            does not reload the provided keys
        strict_optimizer_lr: bool
            validates FB/DDPG optimizer learning rates for a true resume
        """
        print(f"loading checkpoint from {fp}")
        fp = Path(fp)
        if fp.stat().st_size == 0:
            raise RuntimeError(f"Checkpoint {fp} is empty, likely due to an interrupted write")
        with fp.open('rb') as f:
            payload = torch.load(f, weights_only=False)
        _update_legacy_class(payload, (ReplayBuffer,))
        if isinstance(payload, ReplayBuffer):  # compatibility with pure buffers pickles
            payload = {"replay_loader": payload}
        if only is not None:
            only = list(only)
            assert all(x in self._CHECKPOINTED_KEYS for x in only)
            payload = {x: payload[x] for x in only}
        exclude = list(exclude)
        assert all(x in self._CHECKPOINTED_KEYS for x in exclude)
        self._needs_replay_warmup = "replay_loader" not in payload
        self._replay_warmup_warned = False
        for x in exclude:
            payload.pop(x, None)
        for name, val in payload.items():
            logger.info("Reloading %s from %s", name, fp)
            if name == "agent":
                if isinstance(self.agent, agents.FBDDPGAgent):
                    self.agent.init_from(
                        val,
                        strict_optimizer_lr=strict_optimizer_lr,
                    )
                else:
                    self.agent.init_from(val)
            elif name == "replay_loader":
                _update_legacy_class(val, (ReplayBuffer,))
                assert isinstance(val, ReplayBuffer)
                # pylint: disable=protected-access
                # drop unecessary meta which could make a mess
                val._current_episode.clear()  # make sure we can start over
                val._future = self.cfg.future
                val._discount = self.cfg.discount
                val._max_episodes = len(val._storage["discount"])
                self.replay_loader = val
            else:
                assert hasattr(self, name)
                setattr(self, name, val)
                if name == "global_episode":
                    logger.warning(f"Reloaded agent at global episode {self.global_episode}")

    def finalize(self) -> None:
        print("Running final test", flush=True)
        repeat = self.cfg.final_tests
        if not repeat:
            return

        training_task = self.cfg.task
        original_custom_reward = self.cfg.custom_reward
        original_seed = self.cfg.seed
        original_num_eval_episodes = self.cfg.num_eval_episodes
        original_eval_env = self.eval_env
        eval_hist = self.eval_rewards_history
        rewards: tp.Dict[str, tp.List[float]] = {}
        try:
            if self.cfg.custom_reward == "maze_multi_goal":
                self.eval_rewards_history = []
                self.cfg.num_eval_episodes = repeat
                self.eval_maze_goals()
                rewards["rewards"] = list(self.eval_rewards_history)
            else:
                domain_tasks = {
                    "cheetah": ['walk', 'walk_backward', 'run', 'run_backward'],
                    "quadruped": ['stand', 'walk', 'run', 'jump'],
                    "walker": ['stand', 'walk', 'run', 'flip'],
                }
                if self.domain not in domain_tasks:
                    return
                for name in domain_tasks[self.domain]:
                    task = "_".join([self.domain, name])
                    self.cfg.task = task
                    self.cfg.custom_reward = task  # for the replay buffer
                    self.cfg.seed += 1  # for the sake of avoiding similar seeds
                    self.eval_env = self._make_env()
                    self.eval_rewards_history = []
                    self.cfg.num_eval_episodes = 1
                    for _ in range(repeat):
                        # Final cross-task evaluation has its own task-specific
                        # W&B keys and must not contaminate periodic
                        # eval/episode_reward for the training task.
                        self.eval(log_metrics=False)
                    rewards[task] = list(self.eval_rewards_history)
        finally:
            self.cfg.task = training_task
            self.cfg.custom_reward = original_custom_reward
            self.cfg.seed = original_seed
            self.cfg.num_eval_episodes = original_num_eval_episodes
            self.eval_env = original_eval_env
            self.eval_rewards_history = eval_hist

        with (self.work_dir / "test_rewards.json").open("w") as f:
            json.dump(rewards, f)
        if self.cfg.use_wandb and self.domain in {"walker", "quadruped", "cheetah"}:
            final_metrics = {
                f"final/{task}": float(np.mean(task_rewards))
                for task, task_rewards in rewards.items()
            }
            final_metrics["final/frame"] = self.global_frame
            wandb.log(final_metrics)
            if wandb.run is not None:
                wandb.run.summary["training_task"] = training_task
                wandb.run.summary["final_eval_domain"] = self.domain
                for key, value in final_metrics.items():
                    wandb.run.summary[key] = value


class Workspace(BaseWorkspace[PretrainConfig]):
    def __init__(self, cfg: PretrainConfig) -> None:
        super().__init__(cfg)
        self.train_video_recorder = TrainVideoRecorder(self.work_dir if cfg.save_train_video else None,
                                                       camera_id=self.video_recorder.camera_id, use_wandb=self.cfg.use_wandb)
        if self._resume_checkpoint_filepath is None:  # don't relay if there is a checkpoint
            if cfg.load_replay_buffer is not None:
                if self.cfg.task.split('_')[0] == "d4rl":
                    d4rl_replay_buffer_builder = D4RLReplayBufferBuilder()
                    self.replay_storage = d4rl_replay_buffer_builder.prepare_replay_buffer_d4rl(self.train_env, self.agent.init_meta(), self.cfg)
                    self.replay_loader = self.replay_storage
                else:
                    self.load_checkpoint(cfg.load_replay_buffer, only=["replay_loader"])

    def _init_meta(self):
        if isinstance(self.agent, agents.GoalTD3Agent) and isinstance(self.reward_cls, _goals.MazeMultiGoal):
            meta = self.agent.init_meta(self.reward_cls)
        elif isinstance(self.agent, agents.GoalSMAgent) and len(self.replay_loader) > 0:
            meta = self.agent.init_meta(self.replay_loader)
        else:
            meta = self.agent.init_meta()
        return meta

    def train(self) -> None:
        # predicates
        train_until_step = utils.Until(self.cfg.num_train_frames,
                                       self.cfg.action_repeat)
        seed_until_step = utils.Until(self.cfg.num_seed_frames,
                                      self.cfg.action_repeat)
        eval_every_step = utils.Every(self.cfg.eval_every_frames,
                                      self.cfg.action_repeat)
        # if self.cfg.custom_reward is not None:
        #     raise NotImplementedError("Custom reward not implemented in pretrain.py train loop (see anytrain.py)")

        episode_step, episode_reward, z_correl = 0, 0.0, 0.0
        time_step = self.train_env.reset()
        meta = self._init_meta()
        self.replay_loader.add(time_step, meta)
        self.train_video_recorder.init(self.train_env)
        metrics = None
        physics_agg = dmc.PhysicsAggregator()

        while train_until_step(self.global_step):

            if time_step.last():
                self.global_episode += 1
                self.train_video_recorder.save(f'{self.global_frame}.mp4')
                # wait until all the metrics schema is populated
                if metrics is not None:
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.cfg.action_repeat
                    with self.logger.log_and_dump_ctx(self.global_frame,
                                                      ty='train') as log:
                        log('fps', episode_frame / elapsed_time)
                        log('total_time', total_time)
                        log('episode_reward', episode_reward)
                        log('episode_length', episode_frame)
                        log('episode', self.global_episode)
                        log('buffer_size', len(self.replay_loader))
                        log('step', self.global_step)
                        log('z_correl', z_correl)

                        for key, val in physics_agg.dump():
                            log(key, val)
                if self.cfg.use_hiplog and self.logger.hiplog.content:
                    self.logger.hiplog.write()

                # reset env
                time_step = self.train_env.reset()
                meta = self._init_meta()
                self.replay_loader.add(time_step, meta)
                self.train_video_recorder.init(self.train_env)
                # try to save snapshot
                if self.global_frame in self.cfg.snapshot_at:
                    self.save_checkpoint(self._checkpoint_filepath.with_name(f'snapshot_{self.global_frame}.pt'))
                episode_step = 0
                episode_reward = 0.0
                z_correl = 0.0

            # try to evaluate
            if eval_every_step(self.global_step):
                self.logger.log('eval_total_time', self.timer.total_time(),
                                self.global_frame)
                if self.cfg.custom_reward == "maze_multi_goal":
                    self.eval_maze_goals()
                # elif self.domain == "grid":
                #     self.eval_grid_goals()
                else:
                    self.eval()
            meta = self.agent.update_meta(meta, self.global_step, time_step, finetune=False, replay_loader=self.replay_loader)
            # sample action
            with torch.no_grad(), utils.eval_mode(self.agent):
                action = self.agent.act(time_step.observation,
                                        meta,
                                        self.global_step,
                                        eval_mode=False)

            # try to update the agent
            if not seed_until_step(self.global_step):
                if self._needs_replay_warmup and len(self.replay_loader) == 0:
                    if not self._replay_warmup_warned:
                        logger.warning("Checkpoint was loaded without replay buffer; skipping updates until one episode is collected")
                        self._replay_warmup_warned = True
                else:
                    if self._needs_replay_warmup:
                        logger.info("Replay warmup complete; resuming agent updates")
                        self._needs_replay_warmup = False
                    # TODO: reward_free should be handled in the agent update itself !
                    # TODO: the commented code below raises incompatible type "Generator[EpisodeBatch[ndarray[Any, Any]], None, None]"; expected "ReplayBuffer"
                    # replay = (x.with_no_reward() if self.cfg.reward_free else x for x in self.replay_loader)
                    if isinstance(self.agent, agents.GoalTD3Agent) and isinstance(self.reward_cls, _goals.MazeMultiGoal):
                        metrics = self.agent.update(self.replay_loader, self.global_step, self.reward_cls)
                    else:
                        metrics = self.agent.update(self.replay_loader, self.global_step)
                    self.logger.log_metrics(metrics, self.global_frame, ty='train')

            # take env step
            time_step = self.train_env.step(action)
            physics_agg.add(self.train_env)
            episode_reward += time_step.reward
            self.replay_loader.add(time_step, meta)
            self.train_video_recorder.record(self.train_env)
            if isinstance(self.agent, agents.FBDDPGAgent):
                z_correl += self.agent.compute_z_correl(time_step, meta)
            episode_step += 1
            self.global_step += 1
            # save checkpoint to reload
            if not self.global_frame % self.cfg.checkpoint_every:
                self.save_checkpoint(self._checkpoint_filepath)
        self.save_checkpoint(self._checkpoint_filepath)  # make sure we save the final checkpoint
        self.finalize()


@hydra.main(config_path='.', config_name='base_config', version_base="1.1")
def main(cfg: omgcf.DictConfig) -> None:
    # we assume cfg is a PretrainConfig (but actually not really)
    workspace = Workspace(cfg)  # type: ignore
    workspace.train()


if __name__ == '__main__':
    main()
