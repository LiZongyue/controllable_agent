# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=unused-import
import pdb
import copy
import math
import logging
import dataclasses
from collections import OrderedDict
import typing as tp

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from hydra.core.config_store import ConfigStore
import omegaconf
from dm_env import specs
from transformers import AutoConfig, AutoModel

from url_benchmark import utils
# from url_benchmark import replay_buffer as rb
from url_benchmark.in_memory_replay_buffer import ReplayBuffer
from url_benchmark.dmc import TimeStep
from url_benchmark import goals as _goals
from .ddpg import MetaDict
from .fb_modules import IdentityMap
from .ddpg import Encoder
from .fb_modules import Actor, DiagGaussianActor, ForwardMap, BackwardMap, OnlineCov, mlp


logger = logging.getLogger(__name__)
VISUAL_ENCODER_OBS_TYPES = {"pixels", "dino", "vit"}
IDM_ROUTES = {"none", "forward_adapter", "backward_adapter"}
IDM_ROUTE_IDS = {
    "none": 0.0,
    "forward_adapter": 1.0,
    "backward_adapter": 2.0,
}


def _warmup_cosine_scale(step: int, warmup_steps: int, decay_steps: int, min_scale: float) -> float:
    if step < warmup_steps:
        return float(step + 1) / float(max(1, warmup_steps))
    if decay_steps <= warmup_steps:
        return min_scale
    progress = min(step - warmup_steps, decay_steps - warmup_steps) / float(max(1, decay_steps - warmup_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_scale + (1.0 - min_scale) * cosine


@dataclasses.dataclass
class _WarmupCosineSchedule:
    warmup_steps: int
    decay_steps: int
    min_scale: float

    def __call__(self, step: int) -> float:
        return _warmup_cosine_scale(
            step,
            self.warmup_steps,
            self.decay_steps,
            self.min_scale,
        )


def _grad_norm(parameters: tp.Iterable[torch.nn.Parameter]) -> float:
    grads = [param.grad.detach().norm(2) for param in parameters if param.grad is not None]
    if not grads:
        return 0.0
    return torch.norm(torch.stack(grads), 2).item()


def _tensor_grad_norm(grads: tp.Iterable[tp.Optional[torch.Tensor]]) -> float:
    """Return the global L2 norm of functional gradients without touching ``.grad``."""
    squared_norms = [grad.detach().pow(2).sum() for grad in grads if grad is not None]
    if not squared_norms:
        return 0.0
    return torch.stack(squared_norms).sum().sqrt().item()


def _tensor_grad_dot(
    left: tp.Iterable[tp.Optional[torch.Tensor]],
    right: tp.Iterable[tp.Optional[torch.Tensor]],
) -> float:
    """Return a parameter-wise dot product, treating unused gradients as zero."""
    products = [
        left_grad.detach().mul(right_grad.detach()).sum()
        for left_grad, right_grad in zip(left, right)
        if left_grad is not None and right_grad is not None
    ]
    if not products:
        return 0.0
    return torch.stack(products).sum().item()


def _effective_idm_lr(cfg: tp.Any) -> float:
    idm_lr = getattr(cfg, "idm_lr", None)
    return float(cfg.lr if idm_lr is None else idm_lr)


def _effective_fb_lr(cfg: tp.Any) -> float:
    fb_lr = getattr(cfg, "fb_lr", None)
    return float(cfg.lr if fb_lr is None else fb_lr)


def _effective_forward_lr(cfg: tp.Any) -> float:
    lr_f = getattr(cfg, "lr_f", None)
    return _effective_fb_lr(cfg) if lr_f is None else float(lr_f)


def _effective_backward_lr(cfg: tp.Any) -> float:
    lr_b = getattr(cfg, "lr_b", None)
    if lr_b is not None:
        return float(lr_b)
    return float(cfg.lr_coef) * _effective_fb_lr(cfg)


def _effective_actor_lr(cfg: tp.Any) -> float:
    lr_actor = getattr(cfg, "lr_actor", None)
    return float(cfg.lr if lr_actor is None else lr_actor)


def _scale_gradient(value: torch.Tensor, scale: float) -> torch.Tensor:
    """Preserve the forward value while scaling gradients to its producer."""
    if not 0.0 <= scale <= 1.0:
        raise ValueError(f"gradient scale must be in [0, 1], got {scale}")
    return value.detach() + scale * (value - value.detach())


def _scale_idm_encoder_gradient(value: torch.Tensor, scale: float) -> torch.Tensor:
    """Scale only the IDM gradient flowing back to the shared encoder."""
    if not math.isfinite(scale) or scale < 0.0:
        raise ValueError(f"IDM encoder gradient scale must be finite and non-negative, got {scale}")
    return value.detach() + scale * (value - value.detach())


class ViTEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "facebook/dinov2-base",
        repr_dim: int = 512,
        use_cls_token: bool = False,
    ) -> None:
        super().__init__()
        config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_config(config)
        self.use_cls_token = use_cls_token
        self.repr_dim = repr_dim

        hidden_size = int(getattr(config, "hidden_size"))
        self.projector = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, repr_dim, bias=True),
        )
        linear = self.projector[1]
        nn.init.orthogonal_(linear.weight)
        nn.init.zeros_(linear.bias)

        self.register_buffer(
            "pixel_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "pixel_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs = obs.float() / 255.0
        obs = (obs - self.pixel_mean) / self.pixel_std
        outputs = self.backbone(pixel_values=obs)
        tokens = outputs.last_hidden_state
        pooled = tokens[:, 0] if self.use_cls_token else tokens[:, 1:].mean(dim=1)
        return self.projector(pooled)


class FlareBEncoder(nn.Module):
    """Build the B-only FLARE representation from three raw DINO CLS frames."""

    num_frames = 3
    output_dim = 512

    def __init__(self, frame_dim: int) -> None:
        super().__init__()
        self.frame_dim = frame_dim
        self.projector = nn.Sequential(
            nn.LayerNorm(frame_dim),
            nn.Linear(frame_dim, self.output_dim),
        )
        self.fusion = nn.Sequential(
            nn.Linear(4 * self.output_dim, self.output_dim),
            nn.LayerNorm(self.output_dim),
        )

    @staticmethod
    def _assemble_flare(
        u0: torch.Tensor,
        u1: torch.Tensor,
        u2: torch.Tensor,
    ) -> torch.Tensor:
        d1 = u1 - u0.detach()
        d2 = u2 - u1.detach()
        return torch.cat([u1, u2, d1, d2], dim=-1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        expected_dim = self.num_frames * self.frame_dim
        if obs.ndim != 2 or obs.shape[-1] != expected_dim:
            raise ValueError(
                "FLARE-B expects raw stacked DINO CLS embeddings with shape "
                f"[B, {expected_dim}], got {tuple(obs.shape)}"
            )
        phi_0, phi_1, phi_2 = obs.split(self.frame_dim, dim=-1)
        u0 = self.projector(phi_0)
        u1 = self.projector(phi_1)
        u2 = self.projector(phi_2)
        return self.fusion(self._assemble_flare(u0, u1, u2))


@dataclasses.dataclass
class FBDDPGAgentConfig:
    # @package agent
    _target_: str = "url_benchmark.agent.fb_ddpg.FBDDPGAgent"
    name: str = "fb_ddpg"
    # reward_free: ${reward_free}
    obs_type: str = omegaconf.MISSING  # to be specified later
    obs_shape: tp.Tuple[int, ...] = omegaconf.MISSING  # to be specified later
    action_shape: tp.Tuple[int, ...] = omegaconf.MISSING  # to be specified later
    device: str = omegaconf.II("device")  # ${device}
    lr: float = 1e-4
    fb_lr: tp.Optional[float] = None
    lr_f: tp.Optional[float] = None
    lr_b: tp.Optional[float] = None
    lr_actor: tp.Optional[float] = None
    lr_coef: float = 1
    fb_target_tau: float = 0.01  # 0.001-0.01
    update_every_steps: int = 2
    use_tb: bool = omegaconf.II("use_tb")  # ${use_tb}
    use_wandb: bool = omegaconf.II("use_wandb")  # ${use_wandb}
    use_hiplog: bool = omegaconf.II("use_hiplog")  # ${use_wandb}
    num_expl_steps: int = omegaconf.MISSING  # ???  # to be specified later
    num_inference_steps: int = 5120
    hidden_dim: int = 1024   # 128, 2048
    backward_hidden_dim: int = 526   # 512
    feature_dim: int = 512   # 128, 1024
    z_dim: int = 50  # 100
    stddev_schedule: str = "0.2"  # "linear(1,0.2,200000)" #
    stddev_clip: float = 0.3  # 1
    update_z_every_step: int = 300
    update_z_proba: float = 1.0
    nstep: int = 1
    batch_size: int = 1024  # 512
    init_fb: bool = True
    update_encoder: bool = omegaconf.II("update_encoder")  # ${update_encoder}
    goal_space: tp.Optional[str] = omegaconf.II("goal_space")
    use_cls: bool = omegaconf.II("use_cls")
    dino_frame_stack: int = omegaconf.II("dino_frame_stack")
    dino_use_adapter: bool = True
    dino_adapter_type: str = "linear"
    dino_adapter_hidden_dim: int = 1024
    dino_adapter_output_dim: int = 512
    pixel_separate_fb_encoders: bool = False
    dino_separate_fb_adapters: bool = False
    # Legacy experiment mode: a separate B adapter plus an EMA B adapter.
    dino_separate_backward_adapter: bool = False
    dino_flare_b: bool = False
    backward_encoder_grad_scale: float = 1.0
    idm_coef: float = 0.0
    idm_lr: tp.Optional[float] = None
    # ``none`` preserves the historical shared-encoder IDM path outside the
    # explicit separate-F/B topology. Separate adapters require an explicit
    # branch route whenever IDM is enabled.
    idm_route: str = "none"
    idm_diagnostics_interval: int = 500  # also the balanced-mode update interval
    # ``legacy`` preserves the original loss weighting exactly. ``static``
    # trains the IDM head with the raw loss while applying ``idm_coef`` only
    # to the shared encoder. ``balanced`` adapts that encoder-only coefficient
    # to match ``idm_grad_ratio_target``.
    idm_encoder_mode: str = "legacy"
    idm_encoder_burnin_steps: int = 0
    idm_encoder_ramp_steps: int = 0
    idm_grad_ratio_target: tp.Optional[float] = None
    idm_grad_ratio_ema: float = 0.9
    idm_coef_min: float = 0.1
    idm_coef_max: float = 200.0
    idm_coef_slew_rate: float = 2.0
    ortho_coef: float = 1.0  # 0.01-10
    log_std_bounds: tp.Tuple[float, float] = (-5, 2)  # param for DiagGaussianActor
    temp: float = 1  # temperature for DiagGaussianActor
    boltzmann: bool = False  # set to true for DiagGaussianActor
    debug: bool = False
    future_ratio: float = 0.0
    mix_ratio: float = 0.5  # 0-1
    rand_weight: bool = False  # True, False
    preprocess: bool = True
    norm_z: bool = True
    q_loss: bool = False
    q_loss_coef: float = 0.01
    additional_metric: bool = False
    add_trunk: bool = False
    vit_batch_size: int = 64
    vit_backbone_lr: float = 3e-5
    vit_projector_lr: float = 1e-4
    vit_weight_decay: float = 0.05
    vit_warmup_steps: int = 2000
    vit_lr_decay_steps: int = 500000
    vit_min_lr_scale: float = 0.1
    vit_encoder_grad_clip: float = 1.0


cs = ConfigStore.instance()
cs.store(group="agent", name="fb_ddpg", node=FBDDPGAgentConfig)


class FBDDPGAgent:

    # pylint: disable=unused-argument
    def __init__(self,
                 **kwargs: tp.Any
                 ):
        cfg = FBDDPGAgentConfig(**kwargs)
        if cfg.dino_flare_b and cfg.dino_separate_backward_adapter:
            raise ValueError(
                "dino_flare_b and dino_separate_backward_adapter are mutually exclusive"
            )
        if cfg.dino_flare_b and not cfg.update_encoder:
            raise ValueError("dino_flare_b requires update_encoder=True")
        if cfg.dino_flare_b and (
            cfg.obs_type != "dino"
            or not cfg.use_cls
            or cfg.dino_frame_stack != FlareBEncoder.num_frames
            or cfg.goal_space is not None
        ):
            raise ValueError(
                "dino_flare_b requires obs_type='dino', use_cls=True, "
                "dino_frame_stack=3, and goal_space=None"
            )
        if cfg.dino_flare_b and cfg.obs_shape[0] % FlareBEncoder.num_frames:
            raise ValueError(
                "dino_flare_b requires an observation dimension divisible by 3"
            )
        if (
            cfg.dino_flare_b
            and cfg.dino_separate_fb_adapters
            and not cfg.dino_use_adapter
        ):
            raise ValueError(
                "dino_separate_fb_adapters requires dino_use_adapter=True"
            )
        if cfg.fb_lr is not None and (
            not math.isfinite(cfg.fb_lr) or cfg.fb_lr <= 0
        ):
            raise ValueError("fb_lr must be finite and positive when provided")
        for lr_name in ("lr_f", "lr_b", "lr_actor"):
            lr_value = getattr(cfg, lr_name)
            if lr_value is not None and (
                not math.isfinite(lr_value) or lr_value <= 0
            ):
                raise ValueError(f"{lr_name} must be finite and positive when provided")
        if cfg.dino_separate_fb_adapters and cfg.dino_separate_backward_adapter:
            raise ValueError(
                "dino_separate_fb_adapters and dino_separate_backward_adapter "
                "are mutually exclusive"
            )
        if cfg.pixel_separate_fb_encoders:
            if cfg.obs_type != "pixels" or cfg.goal_space is not None:
                raise ValueError(
                    "pixel_separate_fb_encoders requires obs_type='pixels' "
                    "and goal_space=None"
                )
            if not cfg.update_encoder:
                raise ValueError(
                    "pixel_separate_fb_encoders requires update_encoder=True so "
                    "both CNN encoders remain trainable"
                )
            if cfg.dino_separate_fb_adapters or cfg.dino_separate_backward_adapter:
                raise ValueError(
                    "pixel_separate_fb_encoders cannot be combined with a "
                    "separate DINO adapter mode"
                )
        if cfg.dino_separate_fb_adapters and not cfg.update_encoder:
            raise ValueError(
                "dino_separate_fb_adapters requires update_encoder=True so both "
                "adapters remain trainable"
            )
        if cfg.idm_coef < 0:
            raise ValueError("idm_coef must be non-negative")
        if cfg.idm_lr is not None and (
            not math.isfinite(cfg.idm_lr) or cfg.idm_lr <= 0
        ):
            raise ValueError("idm_lr must be positive and finite when provided")
        if cfg.idm_route not in IDM_ROUTES:
            raise ValueError(
                "idm_route must be one of 'none', 'forward_adapter', or "
                "'backward_adapter'"
            )
        if cfg.idm_route != "none":
            if cfg.idm_coef <= 0:
                raise ValueError("an explicit idm_route requires idm_coef > 0")
            if not cfg.dino_separate_fb_adapters:
                raise ValueError(
                    "explicit adapter IDM routing requires "
                    "dino_separate_fb_adapters=True"
                )
            if (
                cfg.obs_type != "dino"
                or not cfg.dino_use_adapter
                or cfg.goal_space is not None
            ):
                raise ValueError(
                    "explicit adapter IDM routing requires obs_type='dino', "
                    "dino_use_adapter=True, and goal_space=None"
                )
            if cfg.idm_encoder_mode != "static":
                raise ValueError(
                    "explicit adapter IDM routing requires "
                    "idm_encoder_mode='static'"
                )
        elif cfg.dino_separate_fb_adapters and cfg.idm_coef > 0:
            raise ValueError(
                "IDM with separate F/B adapters requires an explicit idm_route"
            )
        if cfg.dino_flare_b and cfg.idm_route == "backward_adapter":
            raise ValueError(
                "backward_adapter IDM routing is unavailable when dino_flare_b=True"
            )
        if cfg.idm_diagnostics_interval <= 0:
            raise ValueError("idm_diagnostics_interval must be positive")
        if cfg.idm_encoder_mode not in {"legacy", "static", "balanced"}:
            raise ValueError(
                "idm_encoder_mode must be one of 'legacy', 'static', or 'balanced'"
            )
        if cfg.idm_encoder_burnin_steps < 0:
            raise ValueError("idm_encoder_burnin_steps must be non-negative")
        if cfg.idm_encoder_ramp_steps < 0:
            raise ValueError("idm_encoder_ramp_steps must be non-negative")
        if not 0.0 <= cfg.idm_grad_ratio_ema < 1.0:
            raise ValueError("idm_grad_ratio_ema must be in [0, 1)")
        if not math.isfinite(cfg.idm_coef_min) or cfg.idm_coef_min <= 0:
            raise ValueError("idm_coef_min must be finite and positive")
        if not math.isfinite(cfg.idm_coef_max) or cfg.idm_coef_max < cfg.idm_coef_min:
            raise ValueError("idm_coef_max must be finite and at least idm_coef_min")
        if not math.isfinite(cfg.idm_coef_slew_rate) or cfg.idm_coef_slew_rate < 1.0:
            raise ValueError("idm_coef_slew_rate must be finite and at least 1")
        if cfg.idm_encoder_mode == "legacy":
            if cfg.idm_encoder_burnin_steps or cfg.idm_encoder_ramp_steps:
                raise ValueError("IDM encoder burn-in/ramp requires a non-legacy encoder mode")
            if cfg.idm_grad_ratio_target is not None:
                raise ValueError("idm_grad_ratio_target requires idm_encoder_mode='balanced'")
        else:
            if cfg.idm_coef <= 0:
                raise ValueError("non-legacy IDM encoder modes require idm_coef > 0")
            if cfg.obs_type == "dino" and not cfg.dino_use_adapter:
                raise ValueError("non-legacy IDM encoder modes require a trainable DINO adapter")
            if cfg.idm_encoder_mode == "static" and cfg.idm_grad_ratio_target is not None:
                raise ValueError("idm_grad_ratio_target requires idm_encoder_mode='balanced'")
            if cfg.idm_encoder_mode == "balanced":
                target = cfg.idm_grad_ratio_target
                if target is None or not math.isfinite(target) or target <= 0:
                    raise ValueError(
                        "balanced IDM encoder mode requires a finite positive idm_grad_ratio_target"
                    )
                if cfg.idm_encoder_ramp_steps:
                    raise ValueError("balanced IDM encoder mode does not use idm_encoder_ramp_steps")
                if not cfg.idm_coef_min <= cfg.idm_coef <= cfg.idm_coef_max:
                    raise ValueError(
                        "balanced IDM encoder mode requires idm_coef within "
                        "[idm_coef_min, idm_coef_max]"
                    )
        if cfg.idm_coef > 0 and not cfg.update_encoder:
            raise ValueError("idm_coef > 0 requires update_encoder=True")
        if cfg.obs_type == "vit" and cfg.batch_size > cfg.vit_batch_size:
            logger.warning(
                "Reducing vit batch_size from %s to %s for optimization stability",
                cfg.batch_size,
                cfg.vit_batch_size,
            )
            cfg.batch_size = cfg.vit_batch_size
        self.cfg = cfg
        assert len(cfg.action_shape) == 1
        self.action_dim = cfg.action_shape[0]
        self.solved_meta: tp.Any = None

        # models
        self.forward_encoder: tp.Optional[nn.Module] = None
        self.forward_adapter: tp.Optional[nn.Module] = None
        self.backward_adapter: tp.Optional[nn.Module] = None
        self.flare_b_encoder: tp.Optional[FlareBEncoder] = None
        if cfg.obs_type == 'pixels':
            self.aug: nn.Module = utils.RandomShiftsAug(pad=4)
            self.encoder: nn.Module = Encoder(cfg.obs_shape).to(cfg.device)
            self.obs_dim = self.encoder.repr_dim
            if cfg.pixel_separate_fb_encoders:
                # Keep ``encoder`` as the forward/actor encoder for compatibility
                # with the existing pixel path, and clone it before either branch
                # has taken an optimization step.
                self.forward_encoder = self.encoder
        elif cfg.obs_type == 'vit':
            self.aug = nn.Identity()
            self.encoder = ViTEncoder(repr_dim=512, use_cls_token=cfg.use_cls).to(cfg.device)
            self.obs_dim = self.encoder.repr_dim
        elif cfg.obs_type == 'dino':
            self.aug = nn.Identity()
            d = cfg.obs_shape[0]
            if cfg.dino_use_adapter:
                feature_dim = cfg.dino_adapter_output_dim
                if cfg.dino_adapter_type == "linear":
                    self.encoder = nn.Sequential(
                        nn.LayerNorm(d),
                        nn.Linear(d, feature_dim, bias=True),
                    ).to(cfg.device)
                    linear_layers = [self.encoder[1]]
                elif cfg.dino_adapter_type == "mlp":
                    self.encoder = nn.Sequential(
                        nn.LayerNorm(d),
                        nn.Linear(d, cfg.dino_adapter_hidden_dim, bias=True),
                        nn.GELU(),
                        nn.Linear(cfg.dino_adapter_hidden_dim, feature_dim, bias=True),
                    ).to(cfg.device)
                    linear_layers = [self.encoder[1], self.encoder[3]]
                elif cfg.dino_adapter_type == "mlp_ln":
                    self.encoder = nn.Sequential(
                        nn.LayerNorm(d),
                        nn.Linear(d, cfg.dino_adapter_hidden_dim, bias=True),
                        nn.GELU(),
                        nn.Linear(cfg.dino_adapter_hidden_dim, feature_dim, bias=True),
                        nn.LayerNorm(feature_dim),
                    ).to(cfg.device)
                    linear_layers = [self.encoder[1], self.encoder[3]]
                else:
                    raise ValueError(f"Unsupported dino_adapter_type={cfg.dino_adapter_type!r}")
                for linear in linear_layers:
                    nn.init.orthogonal_(linear.weight)
                    nn.init.zeros_(linear.bias)
                self.obs_dim = feature_dim
                self.forward_adapter = self.encoder
            else:
                self.encoder = nn.Identity()
                self.obs_dim = d
        else:
            self.aug = nn.Identity()
            self.encoder = nn.Identity()
            self.obs_dim = cfg.obs_shape[0]
        self.backward_encoder: tp.Optional[nn.Module] = None
        self.backward_encoder_target: tp.Optional[nn.Module] = None
        if cfg.pixel_separate_fb_encoders:
            assert self.forward_encoder is not None
            self.backward_encoder = copy.deepcopy(self.forward_encoder).to(cfg.device)
        elif (
            cfg.dino_separate_backward_adapter
            or (cfg.dino_separate_fb_adapters and not cfg.dino_flare_b)
        ):
            if cfg.obs_type != "dino" or not cfg.dino_use_adapter or cfg.goal_space is not None:
                raise ValueError(
                    "separate DINO FB adapters require obs_type='dino', "
                    "dino_use_adapter=True, and goal_space=None"
                )
            # DINO features are computed once by the environment and stored in replay.
            # Only the lightweight adapter is duplicated for the visual B path.
            assert self.forward_adapter is not None
            self.backward_encoder = copy.deepcopy(self.encoder).to(cfg.device)
            self.backward_adapter = self.backward_encoder
            if cfg.dino_separate_backward_adapter:
                # Retain the historical target-adapter topology only for the
                # explicitly requested legacy mode.  The new separate-F/B mode
                # deliberately has no target/EMA DINO adapter.
                self.backward_encoder_target = copy.deepcopy(self.backward_encoder).to(cfg.device)
        if cfg.feature_dim < self.obs_dim:
            logger.warning(f"feature_dim {cfg.feature_dim} should not be smaller that obs_dim {self.obs_dim}")
        goal_dim = (
            FlareBEncoder.output_dim if cfg.dino_flare_b else self.obs_dim
        )
        if cfg.goal_space is not None:
            goal_dim = _goals.get_goal_space_dim(cfg.goal_space)
        if cfg.z_dim < goal_dim:
            logger.warning(f"z_dim {cfg.z_dim} should not be smaller that goal_dim {goal_dim}")
        # create the network
        if self.cfg.boltzmann:
            self.actor: nn.Module = DiagGaussianActor(self.obs_dim, cfg.z_dim, self.action_dim,
                                                      cfg.hidden_dim, cfg.log_std_bounds).to(cfg.device)
        else:
            self.actor = Actor(self.obs_dim, cfg.z_dim, self.action_dim,
                               cfg.feature_dim, cfg.hidden_dim,
                               preprocess=cfg.preprocess, add_trunk=self.cfg.add_trunk).to(cfg.device)
        self.forward_net = ForwardMap(self.obs_dim, cfg.z_dim, self.action_dim,
                                      cfg.feature_dim, cfg.hidden_dim,
                                      preprocess=cfg.preprocess, add_trunk=self.cfg.add_trunk).to(cfg.device)
        if cfg.debug:
            self.backward_net: nn.Module = IdentityMap().to(cfg.device)
            self.backward_target_net: nn.Module = IdentityMap().to(cfg.device)
        else:
            self.backward_net = BackwardMap(goal_dim, cfg.z_dim, cfg.backward_hidden_dim, norm_z=cfg.norm_z).to(cfg.device)
            self.backward_target_net = BackwardMap(goal_dim,
                                                   cfg.z_dim, cfg.backward_hidden_dim, norm_z=cfg.norm_z).to(cfg.device)
        # build up the target network
        self.forward_target_net = ForwardMap(self.obs_dim, cfg.z_dim, self.action_dim,
                                             cfg.feature_dim, cfg.hidden_dim,
                                             preprocess=cfg.preprocess, add_trunk=self.cfg.add_trunk).to(cfg.device)
        # load the weights into the target networks
        self.forward_target_net.load_state_dict(self.forward_net.state_dict())
        self.backward_target_net.load_state_dict(self.backward_net.state_dict())
        # optimizers
        self.encoder_opt: tp.Optional[torch.optim.Optimizer] = None
        self.backward_encoder_opt: tp.Optional[torch.optim.Optimizer] = None
        self.forward_fb_opt: tp.Optional[torch.optim.Optimizer] = None
        self.backward_fb_opt: tp.Optional[torch.optim.Optimizer] = None
        self.encoder_scheduler: tp.Optional[torch.optim.lr_scheduler.LambdaLR] = None
        separate_fb_encoders = (
            cfg.pixel_separate_fb_encoders or cfg.dino_separate_fb_adapters
        )
        if separate_fb_encoders:
            # Each visual encoder belongs to exactly one branch-local FB
            # optimizer. Actor optimization receives detached forward features
            # below, so it owns only Actor parameters.
            forward_branch_encoder = (
                self.forward_encoder
                if cfg.pixel_separate_fb_encoders
                else self.forward_adapter
            )
            backward_branch_encoder = (
                self.backward_encoder
                if cfg.pixel_separate_fb_encoders
                else self.backward_adapter
            )
            assert forward_branch_encoder is not None
            self.forward_fb_opt = torch.optim.Adam(
                list(self.forward_net.parameters())
                + list(forward_branch_encoder.parameters()),
                lr=_effective_forward_lr(cfg),
            )
            backward_parameters = list(self.backward_net.parameters())
            if backward_branch_encoder is not None:
                backward_parameters += list(backward_branch_encoder.parameters())
            else:
                assert cfg.dino_flare_b
            self.backward_fb_opt = torch.optim.Adam(
                backward_parameters,
                lr=_effective_backward_lr(cfg),
            )
        elif cfg.obs_type == "vit":
            assert isinstance(self.encoder, ViTEncoder)
            self.encoder_opt = torch.optim.AdamW(
                [
                    {
                        "params": self.encoder.backbone.parameters(),
                        "lr": cfg.vit_backbone_lr,
                        "weight_decay": cfg.vit_weight_decay,
                    },
                    {
                        "params": self.encoder.projector.parameters(),
                        "lr": cfg.vit_projector_lr,
                        "weight_decay": cfg.vit_weight_decay,
                    },
                ]
            )
            lr_schedule = _WarmupCosineSchedule(
                warmup_steps=cfg.vit_warmup_steps,
                decay_steps=cfg.vit_lr_decay_steps,
                min_scale=cfg.vit_min_lr_scale,
            )
            self.encoder_scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.encoder_opt,
                lr_lambda=[lr_schedule, lr_schedule],
            )
        elif cfg.obs_type in VISUAL_ENCODER_OBS_TYPES:
            if cfg.obs_type == "dino" and not cfg.dino_use_adapter:
                self.encoder_opt = None
            else:
                self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=cfg.lr)
        if self.backward_encoder is not None:
            if not separate_fb_encoders:
                self.backward_encoder_opt = torch.optim.Adam(self.backward_encoder.parameters(), lr=cfg.lr)
        self.actor_opt = torch.optim.Adam(
            self.actor.parameters(), lr=_effective_actor_lr(cfg)
        )
        # params = [p for net in [self.forward_net, self.backward_net] for p in net.parameters()]
        # self.fb_opt = torch.optim.Adam(params, lr=cfg.lr)
        self.fb_opt: tp.Optional[torch.optim.Optimizer] = None
        if not separate_fb_encoders:
            fb_lr = _effective_fb_lr(cfg)
            self.fb_opt = torch.optim.Adam([{'params': self.forward_net.parameters()},  # type: ignore
                                            {'params': self.backward_net.parameters(), 'lr': cfg.lr_coef * fb_lr}],
                                           lr=fb_lr)

        # Keep the auxiliary inverse-dynamics model outside the FB networks so
        # evaluation/inference topology is unchanged.  In particular, do not
        # initialize it on the disabled path: idm_coef=0 must preserve the
        # baseline parameter initialization and RNG stream exactly.
        self.idm_head: tp.Optional[nn.Linear] = None
        self.idm_optimizer: tp.Optional[torch.optim.Optimizer] = None
        if cfg.idm_coef > 0:
            self.idm_head = nn.Linear(2 * self.obs_dim, self.action_dim).to(cfg.device)
            self.idm_head.apply(utils.weight_init)
            self.idm_optimizer = torch.optim.Adam(
                self.idm_head.parameters(), lr=_effective_idm_lr(cfg)
            )

        self.flare_b_optimizer: tp.Optional[torch.optim.Optimizer] = None
        if cfg.dino_flare_b:
            frame_dim = cfg.obs_shape[0] // FlareBEncoder.num_frames
            self.flare_b_encoder = FlareBEncoder(frame_dim).to(cfg.device)
            self.flare_b_optimizer = torch.optim.Adam(
                self.flare_b_encoder.parameters(),
                lr=_effective_backward_lr(cfg),
            )
        # This is an optimizer-update counter rather than an environment-step
        # counter.  Starting at zero also guarantees that sparse diagnostic
        # fields are present in the first logger/CSV schema.
        self._idm_update_count = 0
        self._idm_effective_coef = float(cfg.idm_coef)
        self._idm_fb_grad_norm_ema: tp.Optional[float] = None
        self._idm_raw_grad_norm_ema: tp.Optional[float] = None
        # This flag is deliberately per process rather than checkpoint state.
        # A resumed logger rebuilds its CSV writer from the first post-resume
        # metrics, so that update must contain the sparse diagnostic fields.
        self._idm_diagnostics_pending = True

        self.train()
        self.forward_target_net.train()
        self.backward_target_net.train()
        self.actor_success: tp.List[float] = []  # only for debugging, can be removed eventually
        # self.inv_cov = torch.eye(self.cfg.z_dim, dtype=torch.float32, device=self.cfg.device)
        # self.online_cov = OnlineCov(mom=0.99, dim=self.cfg.z_dim).to(self.cfg.device)
        # self.online_cov.train()

    def train(self, training: bool = True) -> None:
        self.training = training
        nets = [self.encoder, self.actor, self.forward_net, self.backward_net]
        if self.backward_encoder is not None:
            nets.append(self.backward_encoder)
        if self.backward_encoder_target is not None:
            nets.append(self.backward_encoder_target)
        flare_b_encoder = getattr(self, "flare_b_encoder", None)
        if flare_b_encoder is not None:
            nets.append(flare_b_encoder)
        if self.idm_head is not None:
            nets.append(self.idm_head)
        for net in nets:
            net.train(training)

    def init_from(self, other) -> None:
        # copy parameters over
        source_cfg = getattr(other, "cfg", None)
        # A pre-IDM pickle has no instance fields for either option.  Testing
        # vars(), rather than getattr(), matters because dataclass defaults on
        # the newly imported class would otherwise make a legacy config look
        # like an explicitly configured modern checkpoint.
        source_has_idm_config = source_cfg is not None and (
            "idm_coef" in vars(source_cfg) or "idm_lr" in vars(source_cfg)
        )
        source_uses_idm = source_has_idm_config and float(getattr(source_cfg, "idm_coef", 0.0)) > 0

        # Validate the complete resume contract before copying even one tensor.
        # In particular, loading an Adam state also loads its param-group LR;
        # rejecting mismatches here prevents a checkpoint from silently
        # overriding the requested sweep setting.
        if source_has_idm_config:
            source_idm_coef = float(getattr(source_cfg, "idm_coef", 0.0))
        else:
            source_idm_coef = 0.0
        source_encoder_mode = str(getattr(source_cfg, "idm_encoder_mode", "legacy"))
        source_idm_route = str(getattr(source_cfg, "idm_route", "none"))
        if source_has_idm_config and (
            source_idm_coef > 0 or float(self.cfg.idm_coef) > 0
        ):
            if source_idm_coef != float(self.cfg.idm_coef):
                raise ValueError(
                    "IDM checkpoint/config mismatch: "
                    f"idm_coef={source_idm_coef} in checkpoint but {self.cfg.idm_coef} requested"
                )
            source_idm_lr = _effective_idm_lr(source_cfg)
            requested_idm_lr = _effective_idm_lr(self.cfg)
            if source_idm_lr != requested_idm_lr:
                raise ValueError(
                    "IDM checkpoint/config mismatch: "
                    f"effective idm_lr={source_idm_lr} in checkpoint but "
                    f"{requested_idm_lr} requested"
                )
            if source_encoder_mode != self.cfg.idm_encoder_mode:
                raise ValueError(
                    "IDM checkpoint/config mismatch: "
                    f"idm_encoder_mode={source_encoder_mode!r} in checkpoint but "
                    f"{self.cfg.idm_encoder_mode!r} requested"
                )
            if source_idm_route != self.cfg.idm_route:
                raise ValueError(
                    "IDM checkpoint/config mismatch: "
                    f"idm_route={source_idm_route!r} in checkpoint but "
                    f"{self.cfg.idm_route!r} requested"
                )
            if self.cfg.idm_encoder_mode != "legacy":
                schedule_fields = (
                    "idm_diagnostics_interval",
                    "idm_encoder_burnin_steps",
                    "idm_encoder_ramp_steps",
                    "idm_grad_ratio_target",
                    "idm_grad_ratio_ema",
                    "idm_coef_min",
                    "idm_coef_max",
                    "idm_coef_slew_rate",
                )
                for field in schedule_fields:
                    if field not in vars(source_cfg):
                        raise ValueError(
                            "IDM checkpoint/config mismatch: "
                            f"checkpoint is missing {field}"
                        )
                    source_value = getattr(source_cfg, field)
                    requested_value = getattr(self.cfg, field)
                    if source_value != requested_value:
                        raise ValueError(
                            "IDM checkpoint/config mismatch: "
                            f"{field}={source_value!r} in checkpoint but "
                            f"{requested_value!r} requested"
                        )
                if self.cfg.idm_encoder_mode == "balanced":
                    for state_name in (
                        "_idm_update_count",
                        "_idm_effective_coef",
                        "_idm_fb_grad_norm_ema",
                        "_idm_raw_grad_norm_ema",
                    ):
                        if not hasattr(other, state_name):
                            raise ValueError(
                                "IDM-enabled checkpoint is missing adaptive state "
                                f"{state_name}"
                            )
                    source_update_count = other._idm_update_count
                    if not isinstance(source_update_count, int) or source_update_count < 0:
                        raise ValueError(
                            "IDM-enabled checkpoint has invalid adaptive state "
                            f"_idm_update_count={source_update_count!r}"
                        )
                    source_effective_coef = float(other._idm_effective_coef)
                    if (
                        not math.isfinite(source_effective_coef)
                        or not self.cfg.idm_coef_min
                        <= source_effective_coef
                        <= self.cfg.idm_coef_max
                    ):
                        raise ValueError(
                            "IDM-enabled checkpoint has invalid adaptive state "
                            f"_idm_effective_coef={source_effective_coef!r}"
                        )
                    source_emas = (
                        other._idm_fb_grad_norm_ema,
                        other._idm_raw_grad_norm_ema,
                    )
                    if (source_emas[0] is None) != (source_emas[1] is None) or any(
                        value is not None
                        and (not math.isfinite(float(value)) or float(value) < 0)
                        for value in source_emas
                    ):
                        raise ValueError(
                            "IDM-enabled checkpoint has invalid adaptive gradient-norm EMAs"
                        )
        if self.idm_head is not None and source_uses_idm:
            if getattr(other, "idm_head", None) is None:
                raise ValueError("IDM-enabled checkpoint is missing idm_head")
            source_idm_optimizer = getattr(other, "idm_optimizer", None)
            if not isinstance(source_idm_optimizer, torch.optim.Optimizer):
                raise ValueError("IDM-enabled checkpoint is missing idm_optimizer")
            requested_idm_lr = _effective_idm_lr(self.cfg)
            checkpoint_lrs = {float(group["lr"]) for group in source_idm_optimizer.param_groups}
            if checkpoint_lrs != {requested_idm_lr}:
                raise ValueError(
                    "IDM checkpoint optimizer/config mismatch: "
                    f"optimizer lr(s)={sorted(checkpoint_lrs)} but "
                    f"effective idm_lr={requested_idm_lr} requested"
                )

        optimizer_lrs = (
            ("forward_fb_opt", _effective_forward_lr(self.cfg)),
            ("backward_fb_opt", _effective_backward_lr(self.cfg)),
            ("flare_b_optimizer", _effective_backward_lr(self.cfg)),
            ("actor_opt", _effective_actor_lr(self.cfg)),
        )
        for optimizer_name, requested_lr in optimizer_lrs:
            optimizer = getattr(self, optimizer_name, None)
            source_optimizer = getattr(other, optimizer_name, None)
            if not isinstance(optimizer, torch.optim.Optimizer) or not isinstance(
                source_optimizer, torch.optim.Optimizer
            ):
                continue
            checkpoint_lrs = {
                float(group["lr"])
                for group in source_optimizer.param_groups
            }
            if checkpoint_lrs != {requested_lr}:
                raise ValueError(
                    f"{optimizer_name} checkpoint optimizer/config mismatch: "
                    f"optimizer lr(s)={sorted(checkpoint_lrs)} but "
                    f"effective lr={requested_lr} requested"
                )

        flare_b_encoder = getattr(self, "flare_b_encoder", None)
        source_flare_b_encoder = getattr(other, "flare_b_encoder", None)
        if flare_b_encoder is not None and source_flare_b_encoder is None:
            logger.warning(
                "Checkpoint does not contain flare_b_encoder; "
                "FLARE-B remains newly initialized"
            )
        names = ["encoder", "actor"]
        if self.cfg.init_fb:
            names += ["forward_net", "backward_net", "backward_target_net", "forward_target_net"]
        for name in names:
            utils.hard_update_params(getattr(other, name), getattr(self, name))
        if self.backward_encoder is not None:
            source_backward_encoder = getattr(other, "backward_encoder", None)
            if source_backward_encoder is None:
                # Backward compatibility: initialize the new B adapter from the
                # formerly shared adapter when loading an older checkpoint.
                source_backward_encoder = other.encoder
            utils.hard_update_params(source_backward_encoder, self.backward_encoder)
            if self.backward_encoder_target is not None:
                source_backward_encoder_target = getattr(other, "backward_encoder_target", None)
                if source_backward_encoder_target is None:
                    source_backward_encoder_target = source_backward_encoder
                utils.hard_update_params(source_backward_encoder_target, self.backward_encoder_target)
        if flare_b_encoder is not None and source_flare_b_encoder is not None:
            utils.hard_update_params(source_flare_b_encoder, flare_b_encoder)
        if self.idm_head is not None:
            source_idm_head = getattr(other, "idm_head", None)
            if source_idm_head is not None:
                utils.hard_update_params(source_idm_head, self.idm_head)
        for key, val in self.__dict__.items():
            if isinstance(val, torch.optim.Optimizer):
                source_opt = getattr(other, key, None)
                if isinstance(source_opt, torch.optim.Optimizer):
                    val.load_state_dict(copy.deepcopy(source_opt.state_dict()))
        self._idm_update_count = int(getattr(other, "_idm_update_count", 0))
        if self.cfg.idm_encoder_mode == "balanced":
            self._idm_effective_coef = float(other._idm_effective_coef)
            self._idm_fb_grad_norm_ema = other._idm_fb_grad_norm_ema
            self._idm_raw_grad_norm_ema = other._idm_raw_grad_norm_ema
        # Do not restore this per-process flag: the first update after every
        # load must repopulate sparse logger fields before the first dump.
        self._idm_diagnostics_pending = True

    def get_goal_meta(self, goal_array: np.ndarray) -> MetaDict:
        desired_goal = torch.tensor(goal_array).unsqueeze(0).to(self.cfg.device)
        if self.cfg.obs_type in VISUAL_ENCODER_OBS_TYPES and (
            self.cfg.goal_space is None or desired_goal.ndim != 2
        ):
            desired_goal = self.backward_aug_and_encode(desired_goal)
        with torch.no_grad():
            z = self.backward_net(desired_goal)
        if self.cfg.norm_z:
            z = math.sqrt(self.cfg.z_dim) * F.normalize(z, dim=1)
        z = z.squeeze(0).cpu().numpy()
        meta = OrderedDict()
        meta['z'] = z
        return meta

    def infer_meta(self, replay_loader: ReplayBuffer) -> MetaDict:
        obs_list, reward_list = [], []
        batch_size = 0
        while batch_size < self.cfg.num_inference_steps:
            batch = replay_loader.sample(self.cfg.batch_size)
            batch = batch.to(self.cfg.device)
            obs_list.append(batch.next_goal if self.cfg.goal_space is not None else batch.next_obs)
            reward_list.append(batch.reward)
            batch_size += batch.next_obs.size(0)
        obs, reward = torch.cat(obs_list, 0), torch.cat(reward_list, 0)  # type: ignore
        obs, reward = obs[:self.cfg.num_inference_steps], reward[:self.cfg.num_inference_steps]
        return self.infer_meta_from_obs_and_rewards(obs, reward)

    def infer_meta_from_obs_and_rewards(self, obs: torch.Tensor, reward: torch.Tensor) -> MetaDict:
        print('max reward: ', reward.max().cpu().item())
        print('99 percentile: ', torch.quantile(reward, 0.99).cpu().item())
        print('median reward: ', reward.median().cpu().item())
        print('min reward: ', reward.min().cpu().item())
        print('mean reward: ', reward.mean().cpu().item())
        print('num reward: ', reward.shape[0])

        # filter out small reward
        # pdb.set_trace()
        # idx = torch.where(reward >= torch.quantile(reward, 0.99))[0]
        # obs = obs[idx]
        # reward = reward[idx]
        with torch.no_grad():
            if self.cfg.goal_space is None and self.cfg.obs_type in VISUAL_ENCODER_OBS_TYPES:
                obs = self.backward_aug_and_encode(obs)
            B = self.backward_net(obs)
        z = torch.matmul(reward.T, B) / reward.shape[0]
        if self.cfg.norm_z:
            z = math.sqrt(self.cfg.z_dim) * F.normalize(z, dim=1)
        meta = OrderedDict()
        meta['z'] = z.squeeze().cpu().numpy()
        # self.solved_meta = meta
        return meta

    def sample_z(self, size, device: str = "cpu"):
        gaussian_rdv = torch.randn((size, self.cfg.z_dim), dtype=torch.float32, device=device)
        gaussian_rdv = F.normalize(gaussian_rdv, dim=1)
        if self.cfg.norm_z:
            z = math.sqrt(self.cfg.z_dim) * gaussian_rdv
        else:
            uniform_rdv = torch.rand((size, self.cfg.z_dim), dtype=torch.float32, device=device)
            z = np.sqrt(self.cfg.z_dim) * uniform_rdv * gaussian_rdv
        return z

    def init_meta(self) -> MetaDict:
        if self.solved_meta is not None:
            print('solved_meta')
            return self.solved_meta
        else:
            z = self.sample_z(1)
            z = z.squeeze().numpy()
            meta = OrderedDict()
            meta['z'] = z
        return meta

    # pylint: disable=unused-argument
    def update_meta(
        self,
        meta: MetaDict,
        global_step: int,
        time_step: TimeStep,
        finetune: bool = False,
        replay_loader: tp.Optional[ReplayBuffer] = None
    ) -> MetaDict:
        if global_step % self.cfg.update_z_every_step == 0 and np.random.rand() < self.cfg.update_z_proba:
            return self.init_meta()
        return meta

    def act(self, obs, meta, step, eval_mode) -> tp.Any:
        obs = torch.as_tensor(obs, device=self.cfg.device, dtype=torch.float32).unsqueeze(0)  # type: ignore
        h = self.encoder(obs)
        z = torch.as_tensor(meta['z'], device=self.cfg.device).unsqueeze(0)  # type: ignore
        if self.cfg.boltzmann:
            dist = self.actor(h, z)
        else:
            stddev = utils.schedule(self.cfg.stddev_schedule, step)
            dist = self.actor(h, z, stddev)
        if eval_mode:
            action = dist.mean
            if self.cfg.additional_metric:
                # the following is doing extra computation only used for metrics,
                # it should be deactivated eventually
                F_mean_s = self.forward_net(obs, z, action)
                # F_samp_s = self.forward_net(obs, z, dist.sample())
                F_rand_s = self.forward_net(obs, z, torch.zeros_like(action).uniform_(-1.0, 1.0))
                Qs = [torch.min(*(torch.einsum('sd, sd -> s', F, z) for F in Fs)) for Fs in [F_mean_s, F_rand_s]]
                self.actor_success = (Qs[0] > Qs[1]).cpu().numpy().tolist()
        else:
            action = dist.sample()
            if step < self.cfg.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]

    def compute_z_correl(self, time_step: TimeStep, meta: MetaDict) -> float:
        # goal = time_step.goal if self.cfg.goal_space is not None else time_step.observation  # type: ignore
        # with torch.no_grad():
        #     zs = [torch.Tensor(x).unsqueeze(0).float().to(self.cfg.device) for x in [goal, meta["z"]]]
        #     zs[0] = self.backward_net(zs[0])
        #     zs = [F.normalize(z, 1) for z in zs]
        #     return torch.matmul(zs[0], zs[1].T).item()
        goal = time_step.goal if self.cfg.goal_space is not None else time_step.observation
        with torch.no_grad():
            z_meta = torch.as_tensor(meta["z"], device=self.cfg.device).unsqueeze(0).float()

            # goal -> torch
            goal_t = torch.as_tensor(goal, device=self.cfg.device).float()
            if goal_t.ndim == 1:
                goal_t = goal_t.unsqueeze(0)  # (D,) -> (1,D)

            if self.cfg.goal_space is not None:
                # goal is low-dim vector, DO NOT aug/encode
                z_goal = self.backward_net(goal_t)           # (1,z_dim)
            else:
                # goal is observation-space
                if self.cfg.obs_type in VISUAL_ENCODER_OBS_TYPES:
                    if goal_t.ndim == 3:
                        goal_t = goal_t.unsqueeze(0)         # (C,H,W)->(1,C,H,W)
                    goal_feat = self.backward_aug_and_encode(goal_t)  # (1,obs_dim)
                else:
                    goal_feat = goal_t
                z_goal = self.backward_net(goal_feat)

            z_goal = F.normalize(z_goal, dim=1)
            z_meta = F.normalize(z_meta, dim=1)
        return (z_goal @ z_meta.T).item()

    def _idm_encoder_coefficient(self, step: int) -> float:
        """Return the encoder-only IDM coefficient used by this update."""
        mode = self.cfg.idm_encoder_mode
        if mode == "legacy":
            return float(self.cfg.idm_coef)
        if step < self.cfg.idm_encoder_burnin_steps:
            return 0.0
        if mode == "balanced":
            return self._idm_effective_coef
        ramp_steps = self.cfg.idm_encoder_ramp_steps
        if ramp_steps == 0:
            return float(self.cfg.idm_coef)
        ramp_progress = (step - self.cfg.idm_encoder_burnin_steps) / ramp_steps
        return float(self.cfg.idm_coef) * min(1.0, max(0.0, ramp_progress))

    def _next_balanced_idm_state(
        self,
        fb_grad_norm: float,
        raw_idm_grad_norm: float,
    ) -> tp.Tuple[float, float, float]:
        """Compute the next adaptive coefficient and gradient-norm EMAs."""
        if not math.isfinite(fb_grad_norm) or not math.isfinite(raw_idm_grad_norm):
            raise RuntimeError("Cannot balance IDM with non-finite encoder gradient norms")
        beta = self.cfg.idm_grad_ratio_ema
        fb_ema = (
            fb_grad_norm
            if self._idm_fb_grad_norm_ema is None
            else beta * self._idm_fb_grad_norm_ema + (1.0 - beta) * fb_grad_norm
        )
        raw_idm_ema = (
            raw_idm_grad_norm
            if self._idm_raw_grad_norm_ema is None
            else beta * self._idm_raw_grad_norm_ema
            + (1.0 - beta) * raw_idm_grad_norm
        )
        assert self.cfg.idm_grad_ratio_target is not None
        candidate = self.cfg.idm_grad_ratio_target * fb_ema / (raw_idm_ema + 1e-12)
        current = self._idm_effective_coef
        lower = max(self.cfg.idm_coef_min, current / self.cfg.idm_coef_slew_rate)
        upper = min(self.cfg.idm_coef_max, current * self.cfg.idm_coef_slew_rate)
        next_coef = min(upper, max(lower, candidate))
        return next_coef, fb_ema, raw_idm_ema

    def update_fb(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        discount: torch.Tensor,
        next_obs: torch.Tensor,
        next_goal: torch.Tensor,
        target_next_goal: torch.Tensor,
        z: torch.Tensor,
        step: int,
        idm_obs: tp.Optional[torch.Tensor] = None,
        idm_next_obs: tp.Optional[torch.Tensor] = None,
    ) -> tp.Dict[str, float]:
        metrics: tp.Dict[str, float] = {}
        # compute target successor measure
        with torch.no_grad():
            if self.cfg.boltzmann:
                dist = self.actor(next_obs, z)
                next_action = dist.sample()
            else:
                stddev = utils.schedule(self.cfg.stddev_schedule, step)
                dist = self.actor(next_obs, z, stddev)
                next_action = dist.sample(clip=self.cfg.stddev_clip)
            target_F1, target_F2 = self.forward_target_net(next_obs, z, next_action)  # batch x z_dim
            target_B = self.backward_target_net(target_next_goal)  # batch x z_dim
            target_M1 = torch.einsum('sd, td -> st', target_F1, target_B)  # batch x batch
            target_M2 = torch.einsum('sd, td -> st', target_F2, target_B)  # batch x batch
            target_M = torch.min(target_M1, target_M2)

        # compute FB loss
        F1, F2 = self.forward_net(obs, z, action)
        B = self.backward_net(next_goal)
        M1 = torch.einsum('sd, td -> st', F1, B)  # batch x batch
        M2 = torch.einsum('sd, td -> st', F2, B)  # batch x batch
        I = torch.eye(*M1.size(), device=M1.device)
        off_diag = ~I.bool()
        fb_offdiag: tp.Any = 0.5 * sum((M - discount * target_M)[off_diag].pow(2).mean() for M in [M1, M2])
        fb_diag: tp.Any = -sum(M.diag().mean() for M in [M1, M2])
        fb_loss = fb_offdiag + fb_diag

        # Q LOSS

        if self.cfg.q_loss:
            with torch.no_grad():
                next_Q1, nextQ2 = [torch.einsum('sd, sd -> s', target_Fi, z) for target_Fi in [target_F1, target_F2]]
                next_Q = torch.min(next_Q1, nextQ2)
                cov = torch.matmul(B.T, B) / B.shape[0]
                inv_cov = torch.inverse(cov)
                implicit_reward = (torch.matmul(B, inv_cov) * z).sum(dim=1)  # batch_size
                target_Q = implicit_reward.detach() + discount.squeeze(1) * next_Q  # batch_size
            Q1, Q2 = [torch.einsum('sd, sd -> s', Fi, z) for Fi in [F1, F2]]
            q_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)
            fb_loss += self.cfg.q_loss_coef * q_loss

        # ORTHONORMALITY LOSS FOR BACKWARD EMBEDDING

        Cov = torch.matmul(B, B.T)
        orth_loss_diag = - 2 * Cov.diag().mean()
        orth_loss_offdiag = Cov[off_diag].pow(2).mean()
        orth_loss = orth_loss_offdiag + orth_loss_diag
        fb_loss += self.cfg.ortho_coef * orth_loss

        idm_prediction: tp.Optional[torch.Tensor] = None
        idm_source_obs = obs
        idm_source_next_obs = next_obs
        idm_encoder_coef = 0.0
        if self.idm_head is None:
            idm_loss = torch.zeros((), device=fb_loss.device, dtype=fb_loss.dtype)
            total_loss = fb_loss
        else:
            if self.cfg.idm_route != "none":
                if idm_obs is None or idm_next_obs is None:
                    raise RuntimeError(
                        f"idm_route={self.cfg.idm_route!r} requires explicit "
                        "adapter features for both transition endpoints"
                    )
                idm_source_obs = idm_obs
                idm_source_next_obs = idm_next_obs
            (
                idm_loss,
                idm_prediction,
                idm_encoder_coef,
            ) = self._compute_idm_objective(
                idm_source_obs,
                idm_source_next_obs,
                action,
                step,
            )
            if self.cfg.idm_encoder_mode == "legacy":
                total_loss = fb_loss + self.cfg.idm_coef * idm_loss
            else:
                # The gradient-only scaling above changes the encoder update
                # without changing the IDM head's effective learning rate.
                total_loss = fb_loss + idm_loss

        # Cov = torch.cov(B.T)  # Vicreg loss
        # var_loss = F.relu(1 - Cov.diag().clamp(1e-4, 1).sqrt()).mean()  # eps avoids inf. sqrt gradient at 0
        # cov_loss = 2 * torch.triu(Cov, diagonal=1).pow(2).mean() # 2x upper triangular part
        # orth_loss =  var_loss + cov_loss
        # fb_loss += self.cfg.ortho_coef * orth_loss

        logging_enabled = self.cfg.use_tb or self.cfg.use_wandb or self.cfg.use_hiplog
        if logging_enabled:
            # These are the exact online F/M tensors used above by the FB loss.
            # Aggregate twin-F statistics treat the two branches as one set of
            # batch rows; no surrogate forward pass is used for diagnostics.
            F1_detached, F2_detached = F1.detach(), F2.detach()
            F1_norms = F1_detached.norm(dim=-1)
            F2_norms = F2_detached.norm(dim=-1)
            F_norms = torch.cat((F1_norms, F2_norms))
            quantile_levels = F_norms.new_tensor((0.95, 0.99))
            F_norm_quantiles = torch.quantile(F_norms, quantile_levels)
            metrics['F1_norm_mean'] = F1_norms.mean().item()
            metrics['F2_norm_mean'] = F2_norms.mean().item()
            metrics['F_norm_mean'] = F_norms.mean().item()
            metrics['F_norm_p95'] = F_norm_quantiles[0].item()
            metrics['F_norm_p99'] = F_norm_quantiles[1].item()
            metrics['F1_norm_max'] = F1_norms.max().item()
            metrics['F2_norm_max'] = F2_norms.max().item()
            metrics['F_norm_max'] = F_norms.max().item()
            metrics['F1_abs_max'] = F1_detached.abs().max().item()
            metrics['F2_abs_max'] = F2_detached.abs().max().item()
            metrics['F_abs_max'] = max(
                metrics['F1_abs_max'], metrics['F2_abs_max']
            )
            M_online_abs = torch.cat((
                M1.detach().abs().reshape(-1),
                M2.detach().abs().reshape(-1),
            ))
            M_online_abs_quantiles = torch.quantile(
                M_online_abs, quantile_levels
            )
            metrics['M_online_abs_max'] = max(
                M1.detach().abs().max().item(),
                M2.detach().abs().max().item(),
            )
            metrics['M_online_abs_p95'] = M_online_abs_quantiles[0].item()
            metrics['M_online_abs_p99'] = M_online_abs_quantiles[1].item()
            metrics['target_M'] = target_M.mean().item()
            metrics['M1'] = M1.mean().item()
            metrics['F1'] = F1.mean().item()
            metrics['B'] = B.mean().item()
            metrics['B_norm'] = torch.norm(B, dim=-1).mean().item()
            metrics['z_norm'] = torch.norm(z, dim=-1).mean().item()
            metrics['fb_loss'] = fb_loss.item()
            metrics['idm_loss'] = idm_loss.item()
            if self.cfg.idm_route != "none":
                # This is the loss proxy seen by the selected adapter. The IDM
                # head itself is still trained with the full/raw loss.
                idm_scalar_coef = idm_encoder_coef
            else:
                # Preserve historical logging for the pre-routing IDM modes.
                idm_scalar_coef = (
                    self.cfg.idm_coef
                    if self.cfg.idm_encoder_mode == "legacy"
                    else 1.0
                )
            metrics['idm_weighted_loss'] = idm_scalar_coef * idm_loss.item()
            metrics['idm_encoder_loss_proxy'] = idm_encoder_coef * idm_loss.item()
            metrics['idm_encoder_coef_used'] = idm_encoder_coef
            # Logger meters are numeric-only; the stable string value remains
            # present in the Hydra/W&B config.
            metrics['idm_route'] = IDM_ROUTE_IDS[self.cfg.idm_route]
            if self.cfg.idm_grad_ratio_target is not None:
                metrics['idm_grad_ratio_target'] = self.cfg.idm_grad_ratio_target
                # Keep this dense so the CSV schema established during
                # burn-in also accepts the first post-burn-in rebalance.
                metrics['idm_encoder_coef_next'] = self._idm_effective_coef
            metrics['total_loss'] = total_loss.item()
            metrics['fb_diag'] = fb_diag.item()
            metrics['fb_offdiag'] = fb_offdiag.item()
            if self.cfg.q_loss:
                metrics['q_loss'] = q_loss.item()
            metrics['orth_loss'] = orth_loss.item()
            metrics['orth_loss_diag'] = orth_loss_diag.item()
            metrics['orth_loss_offdiag'] = orth_loss_offdiag.item()
            if self.cfg.q_loss:
                metrics['q_loss'] = q_loss.item()
            eye_diff = torch.matmul(B.T, B) / B.shape[0] - torch.eye(B.shape[1], device=B.device)
            metrics['orth_linf'] = torch.max(torch.abs(eye_diff)).item()
            metrics['orth_l2'] = eye_diff.norm().item() / math.sqrt(B.shape[1])
            if isinstance(self.fb_opt, torch.optim.Adam):
                metrics["fb_opt_lr"] = self.fb_opt.param_groups[0]["lr"]
            if self.forward_fb_opt is not None:
                metrics["lr_f"] = self.forward_fb_opt.param_groups[0]["lr"]
            if self.backward_fb_opt is not None:
                metrics["lr_b"] = self.backward_fb_opt.param_groups[0]["lr"]
            if self.encoder_opt is not None:
                if self.cfg.obs_type == "vit":
                    metrics["encoder_lr_backbone"] = self.encoder_opt.param_groups[0]["lr"]
                    metrics["encoder_lr_projector"] = self.encoder_opt.param_groups[1]["lr"]
                else:
                    metrics["encoder_lr"] = self.encoder_opt.param_groups[0]["lr"]
            if idm_prediction is not None:
                with torch.no_grad():
                    prediction_error = idm_prediction - action
                    action_var = action.var(dim=0, unbiased=False).mean()
                    prediction_var = idm_prediction.var(dim=0, unbiased=False).mean()
                    error_var = prediction_error.var(dim=0, unbiased=False).mean()
                    eps = 1e-12
                    metrics["idm_action_mae"] = prediction_error.abs().mean().item()
                    metrics["idm_nmse"] = (idm_loss / (action_var + eps)).item()
                    metrics["idm_explained_variance"] = (
                        1.0 - error_var / (action_var + eps)
                    ).item()
                    metrics["action_std"] = action_var.sqrt().item()
                    metrics["idm_pred_std"] = prediction_var.sqrt().item()
                    metrics["h_norm"] = 0.5 * (
                        idm_source_obs.norm(dim=-1).mean()
                        + idm_source_next_obs.norm(dim=-1).mean()
                    ).item()
                    metrics["delta_h_norm"] = (
                        idm_source_next_obs - idm_source_obs
                    ).norm(dim=-1).mean().item()

        # optimize FB
        if self.encoder_opt is not None:
            self.encoder_opt.zero_grad(set_to_none=True)
        if self.backward_encoder_opt is not None:
            self.backward_encoder_opt.zero_grad(set_to_none=True)
        if self.idm_optimizer is not None:
            self.idm_optimizer.zero_grad(set_to_none=True)
        if self.fb_opt is not None:
            self.fb_opt.zero_grad(set_to_none=True)
        if self.forward_fb_opt is not None:
            self.forward_fb_opt.zero_grad(set_to_none=True)
        if self.backward_fb_opt is not None:
            self.backward_fb_opt.zero_grad(set_to_none=True)
        flare_b_optimizer = getattr(self, "flare_b_optimizer", None)
        if flare_b_optimizer is not None:
            flare_b_optimizer.zero_grad(set_to_none=True)

        diagnostics_adapter = self.encoder
        if self.cfg.idm_route == "backward_adapter":
            if self.backward_adapter is None:
                raise RuntimeError(
                    "backward_adapter IDM route has no backward adapter"
                )
            diagnostics_adapter = self.backward_adapter
        elif self.cfg.idm_route == "forward_adapter":
            if self.forward_adapter is None:
                raise RuntimeError("forward_adapter IDM route has no forward adapter")
            diagnostics_adapter = self.forward_adapter
        encoder_parameters = tuple(diagnostics_adapter.parameters())
        diagnostics_due = (
            logging_enabled
            and self.idm_head is not None
            and (
                self._idm_diagnostics_pending
                or self._idm_update_count % self.cfg.idm_diagnostics_interval == 0
            )
        )
        balance_due = (
            self.cfg.idm_encoder_mode == "balanced"
            and self.idm_head is not None
            and step >= self.cfg.idm_encoder_burnin_steps
            and self._idm_update_count % self.cfg.idm_diagnostics_interval == 0
        )
        component_grads_due = diagnostics_due or balance_due
        fb_encoder_grads: tp.Tuple[tp.Optional[torch.Tensor], ...] = ()
        idm_encoder_grads: tp.Tuple[tp.Optional[torch.Tensor], ...] = ()
        if component_grads_due and encoder_parameters:
            fb_encoder_grads = torch.autograd.grad(
                fb_loss,
                encoder_parameters,
                retain_graph=True,
                allow_unused=True,
            )
            idm_probe_loss = idm_loss
            if self.cfg.idm_encoder_mode != "legacy" and idm_encoder_coef == 0:
                # A zero gradient-only coefficient hides the raw encoder
                # gradient. Re-run only the tiny linear IDM head at sparse
                # diagnostic points so burn-in metrics remain truthful.
                raw_idm_prediction = self._predict_idm_action(
                    idm_source_obs,
                    idm_source_next_obs,
                )
                idm_probe_loss = self._compute_idm_loss(
                    idm_source_obs,
                    idm_source_next_obs,
                    action,
                    prediction=raw_idm_prediction,
                )
            idm_encoder_grads = torch.autograd.grad(
                idm_probe_loss,
                encoder_parameters,
                retain_graph=True,
                allow_unused=True,
            )

        next_balanced_state: tp.Optional[tp.Tuple[float, float, float]] = None
        fb_grad_norm = 0.0
        raw_idm_grad_norm = 0.0
        weighted_idm_grad_norm = 0.0
        weighted_dot = 0.0
        cosine_idm_grad_norm = 0.0
        if component_grads_due:
            fb_grad_norm = _tensor_grad_norm(fb_encoder_grads)
            probed_idm_grad_norm = _tensor_grad_norm(idm_encoder_grads)
            if self.cfg.idm_encoder_mode == "legacy":
                raw_idm_grad_norm = probed_idm_grad_norm
                weighted_idm_grad_norm = idm_encoder_coef * raw_idm_grad_norm
                weighted_dot = idm_encoder_coef * _tensor_grad_dot(
                    fb_encoder_grads, idm_encoder_grads
                )
                cosine_idm_grad_norm = weighted_idm_grad_norm
            else:
                if idm_encoder_coef > 0:
                    # The functional IDM gradient has already passed through
                    # the gradient-only scale, so it is the weighted gradient.
                    weighted_idm_grad_norm = probed_idm_grad_norm
                    weighted_dot = _tensor_grad_dot(
                        fb_encoder_grads,
                        idm_encoder_grads,
                    )
                    cosine_idm_grad_norm = weighted_idm_grad_norm
                    raw_idm_grad_norm = weighted_idm_grad_norm / idm_encoder_coef
                else:
                    raw_idm_grad_norm = probed_idm_grad_norm
                    weighted_dot = _tensor_grad_dot(
                        fb_encoder_grads,
                        idm_encoder_grads,
                    )
                    cosine_idm_grad_norm = raw_idm_grad_norm
            if balance_due:
                next_balanced_state = self._next_balanced_idm_state(
                    fb_grad_norm,
                    raw_idm_grad_norm,
                )

        # This remains the one and only training backward.  Functional
        # diagnostic gradients above never populate or modify parameter.grad.
        total_loss.backward()
        if logging_enabled:
            # Read the gradients that the optimizer will actually apply. The
            # aggregate covers the shared ForwardMap trunk exactly once; the
            # split metrics cover each twin F head's exclusive parameters.
            metrics['F1_grad_norm'] = _grad_norm(self.forward_net.F1.parameters())
            metrics['F2_grad_norm'] = _grad_norm(self.forward_net.F2.parameters())
            metrics['F_grad_norm'] = _grad_norm(self.forward_net.parameters())
            metrics['B_grad_norm'] = _grad_norm(self.backward_net.parameters())
            if self.forward_adapter is not None:
                metrics['forward_adapter_grad_norm'] = _grad_norm(
                    self.forward_adapter.parameters()
                )
            if self.backward_adapter is not None:
                metrics['backward_adapter_grad_norm'] = _grad_norm(
                    self.backward_adapter.parameters()
                )
            if self.cfg.pixel_separate_fb_encoders:
                assert self.forward_encoder is not None
                assert self.backward_encoder is not None
                metrics['forward_encoder_grad_norm'] = _grad_norm(
                    self.forward_encoder.parameters()
                )
                metrics['backward_encoder_grad_norm'] = _grad_norm(
                    self.backward_encoder.parameters()
                )
        if logging_enabled and self.encoder_opt is not None:
            # Current-batch gradient, deliberately measured after backward and
            # before clipping/optimizer.step().
            metrics["encoder_grad_norm"] = _grad_norm(encoder_parameters)
        if diagnostics_due:
            total_grad_norm = _grad_norm(encoder_parameters)
            eps = 1e-12
            metrics["encoder_grad_norm_fb"] = fb_grad_norm
            metrics["encoder_grad_norm_idm_unweighted"] = raw_idm_grad_norm
            metrics["encoder_grad_norm_idm_weighted"] = weighted_idm_grad_norm
            metrics["encoder_grad_norm_total"] = total_grad_norm
            metrics["encoder_grad_ratio_idm_fb"] = weighted_idm_grad_norm / (fb_grad_norm + eps)
            metrics["encoder_grad_cosine_fb_idm"] = weighted_dot / (
                fb_grad_norm * cosine_idm_grad_norm + eps
            )
            if self.cfg.idm_route != "none":
                metrics["idm_adapter_grad_norm"] = weighted_idm_grad_norm
                metrics["fb_adapter_grad_norm"] = fb_grad_norm
                metrics["idm_fb_adapter_grad_cosine"] = metrics[
                    "encoder_grad_cosine_fb_idm"
                ]
            assert self.idm_head is not None
            metrics["idm_head_grad_norm"] = _grad_norm(self.idm_head.parameters())
            if next_balanced_state is not None:
                metrics["idm_encoder_coef_next"] = next_balanced_state[0]
        if self.cfg.obs_type == "vit" and self.encoder_opt is not None and self.cfg.update_encoder and self.cfg.vit_encoder_grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.cfg.vit_encoder_grad_clip)
        if self.fb_opt is not None:
            self.fb_opt.step()
        if self.forward_fb_opt is not None:
            self.forward_fb_opt.step()
        if self.backward_fb_opt is not None:
            self.backward_fb_opt.step()
        if self.encoder_opt is not None:
            self.encoder_opt.step()
        if self.backward_encoder_opt is not None:
            self.backward_encoder_opt.step()
        if flare_b_optimizer is not None:
            flare_b_optimizer.step()
        if self.idm_optimizer is not None:
            self.idm_optimizer.step()
        if self.encoder_scheduler is not None and self.cfg.update_encoder:
            self.encoder_scheduler.step()
        if next_balanced_state is not None:
            (
                self._idm_effective_coef,
                self._idm_fb_grad_norm_ema,
                self._idm_raw_grad_norm_ema,
            ) = next_balanced_state
        if diagnostics_due:
            self._idm_diagnostics_pending = False
        self._idm_update_count += 1
        return metrics

    def _compute_idm_objective(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        action: torch.Tensor,
        step: int,
    ) -> tp.Tuple[torch.Tensor, torch.Tensor, float]:
        """Build IDM loss while scaling gradients only at its adapter input."""
        if self.idm_head is None:
            raise RuntimeError("Cannot compute IDM objective when idm_coef is zero")
        encoder_coef = self._idm_encoder_coefficient(step)
        idm_obs = obs
        idm_next_obs = next_obs
        if self.cfg.idm_encoder_mode != "legacy":
            idm_obs = _scale_idm_encoder_gradient(obs, encoder_coef)
            idm_next_obs = _scale_idm_encoder_gradient(next_obs, encoder_coef)
        prediction = self._predict_idm_action(idm_obs, idm_next_obs)
        loss = self._compute_idm_loss(
            idm_obs,
            idm_next_obs,
            action,
            prediction=prediction,
        )
        return loss, prediction, encoder_coef

    def _predict_idm_action(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
    ) -> torch.Tensor:
        if self.idm_head is None:
            raise RuntimeError("Cannot predict with IDM when idm_coef is zero")
        return self.idm_head(torch.cat([obs, next_obs], dim=-1))

    def _compute_idm_loss(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        action: torch.Tensor,
        prediction: tp.Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict the transition action from consecutive online features."""
        if prediction is None:
            prediction = self._predict_idm_action(obs, next_obs)
        return F.mse_loss(prediction, action)

    def update_actor(self, obs: torch.Tensor, z: torch.Tensor, step: int) -> tp.Dict[str, float]:
        metrics: tp.Dict[str, float] = {}
        if self.cfg.boltzmann:
            dist = self.actor(obs, z)
            action = dist.rsample()
        else:
            stddev = utils.schedule(self.cfg.stddev_schedule, step)
            dist = self.actor(obs, z, stddev)
            action = dist.sample(clip=self.cfg.stddev_clip)

        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        F1, F2 = self.forward_net(obs, z, action)
        Q1 = torch.einsum('sd, sd -> s', F1, z)
        Q2 = torch.einsum('sd, sd -> s', F2, z)
        if self.cfg.additional_metric:
            q1_success = Q1 > Q2
        Q = torch.min(Q1, Q2)
        actor_loss = (self.cfg.temp * log_prob - Q).mean() if self.cfg.boltzmann else -Q.mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.cfg.use_tb or self.cfg.use_wandb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['q'] = Q.mean().item()
            if self.cfg.additional_metric:
                metrics['q1_success'] = q1_success.float().mean().item()
            metrics['actor_logprob'] = log_prob.mean().item()
            # metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()

        return metrics

    def aug_and_encode(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.ndim == 2:
            return self.encoder(obs)
        if self.cfg.obs_type == "vit":
            return self.encoder(obs)
        obs = self.aug(obs.float())
        return self.encoder(obs)

    def backward_aug_and_encode(self, obs: torch.Tensor) -> torch.Tensor:
        flare_b_encoder = getattr(self, "flare_b_encoder", None)
        if flare_b_encoder is not None:
            return flare_b_encoder(obs)
        if self.backward_encoder is None:
            return self.aug_and_encode(obs)
        if self.cfg.obs_type == "pixels":
            obs = self.aug(obs.float())
            return self.backward_encoder(obs)
        # A separate backward adapter is only valid for cached 1-D DINO
        # embeddings, batched here as (batch, feature_dim).
        return self.backward_encoder(obs)

    def backward_target_aug_and_encode(self, obs: torch.Tensor) -> torch.Tensor:
        if self.backward_encoder_target is None:
            return self.backward_aug_and_encode(obs)
        return self.backward_encoder_target(obs)

    def update(self, replay_loader: ReplayBuffer, step: int) -> tp.Dict[str, float]:
        metrics: tp.Dict[str, float] = {}
        flare_b_encoder = getattr(self, "flare_b_encoder", None)

        if step % self.cfg.update_every_steps != 0:
            return metrics

        batch = replay_loader.sample(self.cfg.batch_size)
        batch = batch.to(self.cfg.device)

        # pdb.set_trace()
        obs = batch.obs
        action = batch.action
        discount = batch.discount
        next_obs = next_goal = target_next_goal = batch.next_obs
        idm_obs: tp.Optional[torch.Tensor] = None
        idm_next_obs: tp.Optional[torch.Tensor] = None
        backward_next_obs_for_idm: tp.Optional[torch.Tensor] = None
        if self.cfg.goal_space is not None:
            assert batch.next_goal is not None
            next_goal = target_next_goal = batch.next_goal

        if self.cfg.obs_type in VISUAL_ENCODER_OBS_TYPES:
            if self.cfg.pixel_separate_fb_encoders:
                # Augment each raw pixel tensor once per update. When the same
                # observation feeds both branches, the independent CNNs consume
                # the exact same random-shift realization.
                assert self.forward_encoder is not None
                augmented_obs = self.aug(batch.obs.float())
                augmented_next_obs = self.aug(batch.next_obs.float())
                obs = self.forward_encoder(augmented_obs)
                next_obs = self.forward_encoder(augmented_next_obs)
            else:
                obs = self.aug_and_encode(batch.obs)
                next_obs = self.aug_and_encode(batch.next_obs)
            if self.cfg.idm_route == "forward_adapter":
                idm_obs = obs
                idm_next_obs = next_obs

            if self.cfg.goal_space is None:
                if self.backward_encoder is not None or flare_b_encoder is not None:
                    if self.cfg.pixel_separate_fb_encoders:
                        next_goal = self.backward_encoder(augmented_next_obs)
                    else:
                        next_goal = self.backward_aug_and_encode(batch.next_obs)
                    backward_next_obs_for_idm = next_goal
                    if self.backward_encoder_target is None:
                        # The new separate-F/B topology has no target adapter;
                        # its target BackwardMap consumes a detached view of the
                        # same online B-side features.
                        target_next_goal = next_goal.detach()
                    else:
                        with torch.no_grad():
                            target_next_goal = self.backward_target_aug_and_encode(batch.next_obs)
                else:
                    next_goal = next_obs
                    target_next_goal = next_goal
                next_goal = _scale_gradient(
                    next_goal,
                    self.cfg.backward_encoder_grad_scale,
                )
            if self.cfg.goal_space is not None and (next_goal[-1].ndim != 1):
                next_goal = self.aug_and_encode(next_goal)
                target_next_goal = next_goal
            if not self.cfg.update_encoder:
                obs = obs.detach()
                next_obs = next_obs.detach()
                if flare_b_encoder is None:
                    next_goal = next_goal.detach()
                target_next_goal = target_next_goal.detach()

        # if len(batch.meta) == 1 and batch.meta[0].shape[-1] == self.cfg.z_dim:
        #     z = batch.meta[0]
        #     invalid = torch.linalg.norm(z, dim=1) < 1e-15
        #     if sum(invalid):
        #         z[invalid, :] = self.sample_z(sum(invalid)).to(self.cfg.device)
        # else:
        z = self.sample_z(self.cfg.batch_size, device=self.cfg.device)
        if not z.shape[-1] == self.cfg.z_dim:
            raise RuntimeError("There's something wrong with the logic here")
        # obs = self.aug_and_encode(batch.obs)
        # next_obs = self.aug_and_encode(batch.next_obs)
        # if not self.cfg.update_encoder:
        #     obs = obs.detach()
        #     next_obs = next_obs.detach()
        backward_input = batch.obs
        future_goal = batch.future_obs
        if self.cfg.goal_space is not None:
            assert batch.goal is not None
            backward_input = batch.goal
            future_goal = batch.future_goal
        if self.cfg.obs_type in VISUAL_ENCODER_OBS_TYPES:
            if self.cfg.goal_space is None:
                if self.cfg.pixel_separate_fb_encoders:
                    assert self.backward_encoder is not None
                    assert future_goal is not None
                    backward_input = self.backward_encoder(augmented_obs)
                    future_aug = self.aug(future_goal.float())
                    future_goal = self.backward_encoder(future_aug)
                else:
                    backward_input = self.backward_aug_and_encode(backward_input)
                    future_goal = self.backward_aug_and_encode(future_goal)
            elif backward_input[-1].ndim != 1:
                backward_input = self.aug_and_encode(backward_input)
                future_goal = self.aug_and_encode(future_goal)
            if not self.cfg.update_encoder and flare_b_encoder is None:
                backward_input = backward_input.detach()
                future_goal = future_goal.detach()

        if self.cfg.idm_route == "backward_adapter":
            if backward_next_obs_for_idm is None:
                raise RuntimeError(
                    "backward_adapter IDM routing requires visual next-observation "
                    "features from the backward adapter"
                )
            idm_obs = backward_input
            idm_next_obs = backward_next_obs_for_idm

        # if self.cfg.goal_space is None:
        #     backward_input = obs
        #     future_goal = self.aug_and_encode(batch.future_obs)
        #     next_goal = self.aug_and_encode(next_goal)
        # else:
        #     assert batch.goal is not None
        #     backward_input = batch.goal
        #     future_goal = batch.future_goal

        perm = torch.randperm(self.cfg.batch_size)
        backward_input = backward_input[perm]

        if self.cfg.mix_ratio > 0:
            mix_idxs: tp.Any = np.where(np.random.uniform(size=self.cfg.batch_size) < self.cfg.mix_ratio)[0]
            if not self.cfg.rand_weight:
                with torch.no_grad():
                    mix_z = self.backward_net(backward_input[mix_idxs]).detach()
            else:
                # generate random weight
                weight = torch.rand(size=(mix_idxs.shape[0], self.cfg.batch_size)).to(self.cfg.device)
                weight = F.normalize(weight, dim=1)
                uniform_rdv = torch.rand(mix_idxs.shape[0], 1).to(self.cfg.device)
                weight = uniform_rdv * weight
                with torch.no_grad():
                    mix_z = torch.matmul(weight, self.backward_net(backward_input).detach())
            if self.cfg.norm_z:
                mix_z = math.sqrt(self.cfg.z_dim) * F.normalize(mix_z, dim=1)
            z[mix_idxs] = mix_z

        # hindsight replay
        if self.cfg.future_ratio > 0:
            assert future_goal is not None
            future_idxs = np.where(np.random.uniform(size=self.cfg.batch_size) < self.cfg.future_ratio)
            z[future_idxs] = self.backward_net(future_goal[future_idxs]).detach()

        metrics.update(self.update_fb(obs=obs, action=action, discount=discount,
                                      next_obs=next_obs, next_goal=next_goal,
                                      target_next_goal=target_next_goal, z=z, step=step,
                                      idm_obs=idm_obs, idm_next_obs=idm_next_obs))

        # update actor
        if self.encoder_opt is not None or self.forward_fb_opt is not None:
            metrics.update(self.update_actor(obs.detach(), z, step))
        else:
            metrics.update(self.update_actor(obs, z, step))

        # update critic target
        utils.soft_update_params(self.forward_net, self.forward_target_net,
                                 self.cfg.fb_target_tau)
        utils.soft_update_params(self.backward_net, self.backward_target_net,
                                 self.cfg.fb_target_tau)
        if self.backward_encoder_target is not None:
            assert self.backward_encoder is not None
            utils.soft_update_params(self.backward_encoder, self.backward_encoder_target,
                                     self.cfg.fb_target_tau)

        # utils.soft_update_params(self.encoder, self.encoder_target,
        #                          self.cfg.enc_target_tau)

        # update inv cov
        # if step % self.cfg.update_cov_every_step == 0:
        #     logger.info("update online cov")
        #     obs_list = list()
        #     batch_size = 0
        #     while batch_size < 10000:
        #         batch = next(replay_loader)
        #         batch = batch.to(self.cfg.device)
        #         obs_list.append(batch.next_goal if self.cfg.goal_space is not None else batch.next_obs)
        #         batch_size += batch.next_obs.size(0)
        #     obs = torch.cat(obs_list, 0)
        #     with torch.no_grad():
        #         B = self.backward_net(obs)
        #     self.inv_cov = torch.inverse(self.online_cov(B))

        return metrics

    # def update(self, replay_loader: tp.Iterator[rb.EpisodeBatch], step: int) -> tp.Dict[str, float]:
    #     metrics: tp.Dict[str, float] = {}
    #
    #     if step % self.cfg.update_every_steps != 0:
    #         return metrics
    #
    #     for _ in range(self.cfg.num_fb_updates):
    #         batch = next(replay_loader)
    #         batch = batch.to(self.cfg.device)
    #         if self.cfg.mix_ratio > 0:
    #             assert self.cfg.batch_size % 3 == 0
    #             mini_batch_size = self.cfg.batch_size // 3
    #         else:
    #             assert self.cfg.batch_size % 2 == 0
    #             mini_batch_size = self.cfg.batch_size // 2
    #         idxs = list(range(mini_batch_size))
    #         idxs_prime = list(range(mini_batch_size, 2 * mini_batch_size))
    #
    #         # pdb.set_trace()
    #         obs = batch.obs[idxs]
    #         action = batch.action[idxs]
    #         discount = batch.discount[idxs]
    #         next_obs = next_goal = batch.next_obs[idxs]
    #         if self.cfg.goal_space is not None:
    #             assert batch.next_goal is not None
    #             next_goal = batch.next_goal[idxs]
    #         if len(batch.meta) == 1 and batch.meta[0].shape[-1] == self.cfg.z_dim:
    #             z = batch.meta[0][idxs]
    #             invalid = torch.linalg.norm(z, dim=1) < 1e-15
    #             if sum(invalid):
    #                 z[invalid, :] = self.sample_z(sum(invalid)).to(self.cfg.device)
    #         else:
    #             z = self.sample_z(mini_batch_size).to(self.cfg.device)
    #             if not z.shape[-1] == self.cfg.z_dim:
    #                 raise RuntimeError("There's something wrong with the logic here")
    #         # obs = self.aug_and_encode(batch.obs)
    #         # next_obs = self.aug_and_encode(batch.next_obs)
    #         # if not self.cfg.update_encoder:
    #         #     obs = obs.detach()
    #         #     next_obs = next_obs.detach()
    #
    #         backward_input = batch.obs
    #         future_goal = batch.future_obs
    #         if self.cfg.goal_space is not None:
    #             assert batch.goal is not None
    #             backward_input = batch.goal
    #             future_goal = batch.future_goal
    #
    #         # goal = backward_input[idxs]
    #         goal_prime = backward_input[idxs_prime]
    #
    #         if self.cfg.mix_ratio > 0:
    #             mix_idxs: tp.Any = np.where(np.random.uniform(size=mini_batch_size) < self.cfg.mix_ratio)[0]
    #             part = backward_input[2 * mini_batch_size:]
    #             if not self.cfg.rand_weight:
    #                 mix_z = self.backward_net(part[mix_idxs]).detach()
    #             else:
    #                 # generate random weight
    #                 weight = torch.rand(size=(mix_idxs.shape[0], mini_batch_size)).to(self.cfg.device)
    #                 weight = F.normalize(weight, dim=1)
    #                 uniform_rdv = torch.rand(mix_idxs.shape[0], 1).to(self.cfg.device)
    #                 weight = uniform_rdv * weight
    #                 mix_z = torch.matmul(weight, self.backward_net(part).detach())
    #             if self.cfg.norm_z:
    #                 mix_z = math.sqrt(self.cfg.z_dim) * F.normalize(mix_z, dim=1)
    #             z[mix_idxs] = mix_z
    #
    #         # hindsight replay
    #         if self.cfg.future_ratio > 0:
    #             assert future_goal is not None
    #             future_idxs = np.where(np.random.uniform(size=mini_batch_size) < self.cfg.future_ratio)
    #             future_goal = future_goal[idxs][future_idxs]
    #             z[future_idxs] = self.backward_net(future_goal).detach()
    #             goal_prime[future_idxs] = future_goal
    #         metrics.update(self.update_fb(obs=obs, action=action, discount=discount,
    #                                       next_obs=next_obs, next_goal=next_goal, goal_prime=goal_prime, z=z, step=step))
    #
    #         # update actor
    #         metrics.update(self.update_actor(obs, z, step))
    #
    #         # update critic target
    #         utils.soft_update_params(self.forward_net, self.forward_target_net,
    #                                  self.cfg.fb_target_tau)
    #         utils.soft_update_params(self.backward_net, self.backward_target_net,
    #                                  self.cfg.fb_target_tau)
    #
    #     return metrics

    # def update_fb(
    #     self,
    #     obs: torch.Tensor,
    #     action: torch.Tensor,
    #     discount: torch.Tensor,
    #     next_obs: torch.Tensor,
    #     next_goal: torch.Tensor,
    #     goal_prime: torch.Tensor,
    #     z: torch.Tensor,
    #     step: int
    # ) -> tp.Dict[str, float]:
    #     metrics: tp.Dict[str, float] = {}
    #     # compute target successor measure
    #     with torch.no_grad():
    #         if self.cfg.boltzmann:
    #             dist = self.actor(next_obs, z)
    #             next_action = dist.sample()
    #         else:
    #             stddev = utils.schedule(self.cfg.stddev_schedule, step)
    #             dist = self.actor(next_obs, z, stddev)
    #             next_action = dist.sample(clip=self.cfg.stddev_clip)
    #         target_F1, target_F2 = self.forward_target_net(next_obs, z, next_action)  # batch x z_dim
    #         target_B = self.backward_target_net(goal_prime)  # batch x z_dim
    #         target_M1 = torch.einsum('sd, td -> st', target_F1, target_B)  # batch x batch
    #         target_M2 = torch.einsum('sd, td -> st', target_F2, target_B)  # batch x batch
    #         target_M = torch.min(target_M1, target_M2)
    #
    #     # compute FB loss
    #     F1, F2 = self.forward_net(obs, z, action)
    #     B = self.backward_net(next_goal)
    #     B_prime = self.backward_net(goal_prime)
    #     M1_diag = torch.einsum('sd, sd -> s', F1, B)  # batch
    #     M2_diag = torch.einsum('sd, sd -> s', F2, B)  # batch
    #     M1 = torch.einsum('sd, td -> st', F1, B_prime)  # batch x batch
    #     M2 = torch.einsum('sd, td -> st', F2, B_prime)  # batch x batch
    #     fb_loss = 0.5 * (M1 - discount * target_M).pow(2).mean() - M1_diag.mean()
    #     fb_loss += 0.5 * (M2 - discount * target_M).pow(2).mean() - M2_diag.mean()
    #
    #     # ORTHONORMALITY LOSS FOR BACKWARD EMBEDDING
    #
    #     B_B_prime = torch.matmul(B, B_prime.T)
    #     B_diag = torch.einsum('sd, sd -> s', B, B)
    #     B_prime_diag = torch.einsum('sd, sd -> s', B_prime, B_prime)
    #     orth_loss = B_B_prime.pow(2).mean() - (B_diag.mean() + B_prime_diag.mean())
    #     fb_loss += self.cfg.ortho_coef * orth_loss
    #
    #     if self.cfg.use_tb or self.cfg.use_wandb or self.cfg.use_hiplog:
    #         metrics['target_M'] = target_M.mean().item()
    #         metrics['M1'] = M1.mean().item()
    #         metrics['F1'] = F1.mean().item()
    #         metrics['B'] = B.mean().item()
    #         metrics['B_norm'] = torch.norm(B, dim=-1).mean().item()
    #         metrics['z_norm'] = torch.norm(z, dim=-1).mean().item()
    #         metrics['fb_loss'] = fb_loss.item()
    #         metrics['orth_loss'] = orth_loss.item()
    #         eye_diff = torch.matmul(B.T, B) / B.shape[0] - torch.eye(B.shape[1], device=B.device)
    #         metrics['orth_linf'] = torch.max(torch.abs(eye_diff)).item()
    #         metrics['orth_l2'] = eye_diff.norm().item() / math.sqrt(B.shape[1])
    #         if isinstance(self.fb_opt, torch.optim.Adam):
    #             metrics["fb_opt_lr"] = self.fb_opt.param_groups[0]["lr"]
    #         if self.cfg.goal_space in ["simplified_walker", "simplified_quadruped"]:
    #             metrics['max_velocity'] = goal_prime[:, -1].max().item()
    #
    #     # optimize FB
    #     if self.encoder_opt is not None:
    #         self.encoder_opt.zero_grad(set_to_none=True)
    #     self.fb_opt.zero_grad(set_to_none=True)
    #     fb_loss.backward()
    #     self.fb_opt.step()
    #     if self.encoder_opt is not None:
    #         self.encoder_opt.step()
    #     return metrics
