# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pdb  # pylint: disable=unused-import
import sys
import unittest
import dataclasses
from collections import OrderedDict, deque
import typing as tp
from typing import Any

from dm_env import Environment
from dm_env import StepType, specs
import collections
import numpy as np
import torch


class UnsupportedPlatform(unittest.SkipTest, RuntimeError):
    """The platform is not supported for running"""


try:
    from dm_control import suite  # , manipulation
    from dm_control.suite.wrappers import action_scale, pixels
    from url_benchmark import custom_dmc_tasks as cdmc
except ImportError as e:
    raise UnsupportedPlatform(f"Import error (Note: DMC does not run on Mac):\n{e}") from e


S = tp.TypeVar("S", bound="TimeStep")
Env = tp.Union["EnvWrapper", Environment]


@dataclasses.dataclass
class TimeStep:
    step_type: StepType
    reward: float
    discount: float
    observation: np.ndarray
    physics: np.ndarray = dataclasses.field(default=np.ndarray([]), init=False)

    def first(self) -> bool:
        return self.step_type == StepType.FIRST  # type: ignore

    def mid(self) -> bool:
        return self.step_type == StepType.MID  # type: ignore

    def last(self) -> bool:
        return self.step_type == StepType.LAST  # type: ignore

    def __getitem__(self, attr: str) -> tp.Any:
        return getattr(self, attr)

    def _replace(self: S, **kwargs: tp.Any) -> S:
        for name, val in kwargs.items():
            setattr(self, name, val)
        return self


@dataclasses.dataclass
class GoalTimeStep(TimeStep):
    goal: np.ndarray


@dataclasses.dataclass
class ExtendedGoalTimeStep(GoalTimeStep):
    action: tp.Any


@dataclasses.dataclass
class ExtendedTimeStep(TimeStep):
    action: tp.Any


class EnvWrapper:
    def __init__(self, env: Env) -> None:
        self._env = env

    def _augment_time_step(self, time_step: TimeStep, action: tp.Optional[np.ndarray] = None) -> TimeStep:
        if not isinstance(time_step, TimeStep):
            # dm_env time step is a named tuple
            time_step = TimeStep(**time_step._asdict())
        if self.physics is not None:
            return time_step._replace(physics=self.physics.get_state())
        else:
            return time_step

    def reset(self) -> TimeStep:
        time_step = self._env.reset()
        return self._augment_time_step(time_step)

    def step(self, action: np.ndarray) -> TimeStep:
        time_step = self._env.step(action)
        return self._augment_time_step(time_step, action)

    def observation_spec(self) -> tp.Any:
        assert isinstance(self, EnvWrapper)
        return self._env.observation_spec()

    def action_spec(self) -> specs.Array:
        return self._env.action_spec()

    def render(self, *args: tp.Any, **kwargs: tp.Any) -> np.ndarray:
        return self._env.render(*args, **kwargs)  # type: ignore

    @property
    def base_env(self) -> tp.Any:
        env = self._env
        if isinstance(env, EnvWrapper):
            return self.base_env
        return env

    @property
    def physics(self) -> tp.Any:
        if hasattr(self._env, "physics"):
            return self._env.physics

    def __getattr__(self, name):
        return getattr(self._env, name)


class FlattenJacoObservationWrapper(EnvWrapper):
    def __init__(self, env: Env) -> None:
        super().__init__(env)
        self._obs_spec = OrderedDict()
        wrapped_obs_spec = env.observation_spec().copy()
        if 'front_close' in wrapped_obs_spec:
            spec = wrapped_obs_spec['front_close']
            # drop batch dim
            self._obs_spec['pixels'] = specs.BoundedArray(shape=spec.shape[1:],
                                                          dtype=spec.dtype,
                                                          minimum=spec.minimum,
                                                          maximum=spec.maximum,
                                                          name='pixels')
            wrapped_obs_spec.pop('front_close')

        for spec in wrapped_obs_spec.values():
            assert spec.dtype == np.float64
            assert type(spec) == specs.Array
        dim = np.sum(
            np.fromiter((int(np.prod(spec.shape))  # type: ignore
                         for spec in wrapped_obs_spec.values()), np.int32))

        self._obs_spec['observations'] = specs.Array(shape=(dim,),
                                                     dtype=np.float32,
                                                     name='observations')

    def observation_spec(self) -> tp.Any:
        return self._obs_spec

    def _augment_time_step(self, time_step: TimeStep, action: tp.Optional[np.ndarray] = None) -> TimeStep:
        super()._augment_time_step(time_step=time_step, action=action)
        obs = OrderedDict()

        # TODO: this is badly typed since observation is a dict in this case
        if 'front_close' in time_step.observation:
            pixels = time_step.observation['front_close']
            time_step.observation.pop('front_close')  # type: ignore
            pixels = np.squeeze(pixels)
            obs['pixels'] = pixels

        features = []
        for feature in time_step.observation.values():  # type: ignore
            features.append(feature.ravel())
        obs['observations'] = np.concatenate(features, axis=0)
        return time_step._replace(observation=obs)


class ActionRepeatWrapper(EnvWrapper):
    def __init__(self, env: tp.Any, num_repeats: int) -> None:
        super().__init__(env)
        self._num_repeats = num_repeats

    def step(self, action: np.ndarray) -> TimeStep:
        reward = 0.0
        discount = 1.0
        for _ in range(self._num_repeats):
            time_step = self._env.step(action)
            reward += (time_step.reward or 0.0) * discount
            discount *= time_step.discount
            if time_step.last():
                break

        return time_step._replace(reward=reward, discount=discount)


class FrameStackWrapper(EnvWrapper):
    def __init__(self, env: Env, num_frames: int, pixels_key: str = 'pixels') -> None:
        super().__init__(env)
        self._num_frames = num_frames
        self._frames: tp.Deque[np.ndarray] = deque([], maxlen=num_frames)
        self._pixels_key = pixels_key

        wrapped_obs_spec = env.observation_spec()
        assert pixels_key in wrapped_obs_spec

        pixels_shape = wrapped_obs_spec[pixels_key].shape
        # remove batch dim
        if len(pixels_shape) == 4:
            pixels_shape = pixels_shape[1:]
        self._obs_spec = specs.BoundedArray(shape=np.concatenate(
            [[pixels_shape[2] * num_frames], pixels_shape[:2]], axis=0),
            dtype=np.uint8,
            minimum=0,
            maximum=255,
            name='observation')

    def observation_spec(self) -> tp.Any:
        return self._obs_spec

    def _augment_time_step(self, time_step: TimeStep, action: tp.Optional[np.ndarray] = None) -> TimeStep:
        super()._augment_time_step(time_step=time_step, action=action)
        assert len(self._frames) == self._num_frames
        obs = np.concatenate(list(self._frames), axis=0)
        return time_step._replace(observation=obs)

    def _extract_pixels(self, time_step: TimeStep) -> np.ndarray:
        pixels_ = time_step.observation[self._pixels_key]
        # remove batch dim
        if len(pixels_.shape) == 4:
            pixels_ = pixels_[0]
        return pixels_.transpose(2, 0, 1).copy()

    def reset(self) -> TimeStep:
        time_step = self._env.reset()
        pixels_ = self._extract_pixels(time_step)
        for _ in range(self._num_frames):
            self._frames.append(pixels_)
        return self._augment_time_step(time_step)

    def step(self, action: np.ndarray) -> TimeStep:
        time_step = self._env.step(action)
        pixels_ = self._extract_pixels(time_step)
        self._frames.append(pixels_)
        return self._augment_time_step(time_step)


class PixelPerturbationWrapper(EnvWrapper):
    def __init__(
        self,
        env: Env,
        perturbation: str,
        seed: int = 0,
        pixels_key: str = "pixels",
    ) -> None:
        super().__init__(env)
        self._perturbation = perturbation
        self._rng = np.random.RandomState(seed)
        self._pixels_key = pixels_key
        self._episode_params: tp.Dict[str, tp.Any] = {}

    def observation_spec(self) -> tp.Any:
        return self._env.observation_spec()

    @staticmethod
    def _resize_nearest(image: np.ndarray, height: int, width: int) -> np.ndarray:
        y_idx = np.linspace(0, image.shape[0] - 1, height).astype(np.int64)
        x_idx = np.linspace(0, image.shape[1] - 1, width).astype(np.int64)
        return image[y_idx][:, x_idx]

    def _sample_color(self, hard: bool) -> tp.Dict[str, np.ndarray]:
        if hard:
            return {
                "contrast": self._rng.uniform(0.45, 1.75),
                "brightness": self._rng.uniform(-60.0, 60.0),
                "channel": self._rng.uniform(0.45, 1.65, size=(1, 1, 3)),
            }
        return {
            "contrast": self._rng.uniform(0.75, 1.25),
            "brightness": self._rng.uniform(-25.0, 25.0),
            "channel": self._rng.uniform(0.80, 1.20, size=(1, 1, 3)),
        }

    def _sample_background(self, hard: bool) -> tp.Dict[str, tp.Any]:
        grid = 16 if hard else 8
        return {
            "alpha": self._rng.uniform(0.28, 0.45) if hard else self._rng.uniform(0.12, 0.25),
            "texture": self._rng.randint(0, 256, size=(grid, grid, 3), dtype=np.uint8),
        }

    def _sample_camera(self, hard: bool) -> tp.Dict[str, float]:
        return {
            "zoom": self._rng.uniform(1.10, 1.35) if hard else self._rng.uniform(1.02, 1.12),
            "shift_y": self._rng.uniform(-0.18, 0.18) if hard else self._rng.uniform(-0.08, 0.08),
            "shift_x": self._rng.uniform(-0.18, 0.18) if hard else self._rng.uniform(-0.08, 0.08),
        }

    def _sample_episode_params(self) -> None:
        condition = self._perturbation
        hard = condition.endswith("_hard")
        self._episode_params = {
            "color": self._sample_color(hard),
            "background": self._sample_background(hard),
            "camera": self._sample_camera(hard),
        }

    def _apply_color(self, pixels: np.ndarray) -> np.ndarray:
        params = self._episode_params["color"]
        x = pixels.astype(np.float32)
        x = (x - 127.5) * float(params["contrast"]) + 127.5
        x = x * params["channel"] + float(params["brightness"])
        return np.clip(x, 0, 255).astype(np.uint8)

    def _apply_background(self, pixels: np.ndarray) -> np.ndarray:
        params = self._episode_params["background"]
        texture = self._resize_nearest(params["texture"], pixels.shape[0], pixels.shape[1]).astype(np.float32)
        alpha = float(params["alpha"])
        x = (1.0 - alpha) * pixels.astype(np.float32) + alpha * texture
        return np.clip(x, 0, 255).astype(np.uint8)

    def _apply_camera(self, pixels: np.ndarray) -> np.ndarray:
        params = self._episode_params["camera"]
        height, width = pixels.shape[:2]
        crop_h = max(2, int(round(height / float(params["zoom"]))))
        crop_w = max(2, int(round(width / float(params["zoom"]))))
        max_y = max(0, height - crop_h)
        max_x = max(0, width - crop_w)
        center_y = max_y / 2.0
        center_x = max_x / 2.0
        y0 = int(round(np.clip(center_y + params["shift_y"] * height, 0, max_y)))
        x0 = int(round(np.clip(center_x + params["shift_x"] * width, 0, max_x)))
        crop = pixels[y0:y0 + crop_h, x0:x0 + crop_w]
        return self._resize_nearest(crop, height, width).astype(np.uint8)

    def _perturb_pixels(self, pixels: np.ndarray) -> np.ndarray:
        batched = pixels.ndim == 4
        if batched:
            assert pixels.shape[0] == 1, f"expected a single pixel batch, got {pixels.shape}"
            pixels_ = pixels[0]
        else:
            pixels_ = pixels
        assert pixels_.ndim == 3 and pixels_.shape[2] == 3, f"expected HWC pixels, got {pixels.shape}"

        condition = self._perturbation
        if condition.startswith("combined"):
            pixels_ = self._apply_camera(pixels_)
            pixels_ = self._apply_background(pixels_)
            pixels_ = self._apply_color(pixels_)
        elif condition.startswith("color"):
            pixels_ = self._apply_color(pixels_)
        elif condition.startswith("background"):
            pixels_ = self._apply_background(pixels_)
        elif condition.startswith("camera"):
            pixels_ = self._apply_camera(pixels_)
        else:
            raise ValueError(f"Unknown visual perturbation {condition!r}")

        if batched:
            return pixels_[None]
        return pixels_

    def _augment_pixels(self, time_step: TimeStep) -> TimeStep:
        obs = time_step.observation.copy()
        obs[self._pixels_key] = self._perturb_pixels(obs[self._pixels_key])
        return time_step._replace(observation=obs)

    def reset(self) -> TimeStep:
        self._sample_episode_params()
        time_step = self._env.reset()
        return self._augment_pixels(time_step)

    def step(self, action: np.ndarray) -> TimeStep:
        time_step = self._env.step(action)
        return self._augment_pixels(time_step)


class GoalWrapper(EnvWrapper):
    def __init__(self, env: Env, goal_func: tp.Callable[[Env], np.ndarray], append_goal_to_observation: bool = False) -> None:
        """Adds a goal space with a predefined function.
        This can also append the observation with the goal to make sure the goal is achievable
        """
        super().__init__(env)
        self.append_goal_to_observation = append_goal_to_observation
        self.goal_func = goal_func

    def _augment_time_step(self, time_step: TimeStep, action: tp.Optional[np.ndarray] = None) -> TimeStep:
        goal = self.goal_func(self)
        obs = time_step.observation.copy()
        if self.append_goal_to_observation:
            k = "observations"
            obs[k] = np.concatenate([obs[k], goal], axis=0)
            # obs[k] = np.concatenate([obs[k], np.random.normal(size=goal.shape)], axis=0)
        ts = GoalTimeStep(
            step_type=time_step.step_type,
            reward=time_step.reward,
            discount=time_step.discount,
            observation=obs,
            goal=goal,
        )
        return super()._augment_time_step(time_step=ts, action=action)

    def observation_spec(self) -> specs.Array:
        spec = super().observation_spec().copy()
        k = "observations"
        if not self.append_goal_to_observation:
            return spec
        goal = self.goal_func(self)
        spec[k] = specs.Array((spec[k].shape[0] + goal.shape[0],), dtype=np.float32, name=k)
        return spec


class ActionDTypeWrapper(EnvWrapper):
    def __init__(self, env: Env, dtype) -> None:
        super().__init__(env)
        wrapped_action_spec = env.action_spec()
        self._action_spec = specs.BoundedArray(wrapped_action_spec.shape,
                                               dtype,
                                               wrapped_action_spec.minimum,
                                               wrapped_action_spec.maximum,
                                               'action')

    def action_spec(self) -> specs.BoundedArray:
        return self._action_spec

    def step(self, action) -> Any:
        action = action.astype(self._env.action_spec().dtype)
        return self._env.step(action)


class ObservationDTypeWrapper(EnvWrapper):
    def __init__(self, env: Env, dtype) -> None:
        super().__init__(env)
        self._dtype = dtype
        wrapped_obs_spec = env.observation_spec()['observations']
        self._obs_spec = specs.Array(wrapped_obs_spec.shape, dtype,
                                     'observation')

    def _augment_time_step(self, time_step: TimeStep, action: tp.Optional[np.ndarray] = None) -> TimeStep:
        obs = time_step.observation['observations'].astype(self._dtype)
        return time_step._replace(observation=obs)

    def observation_spec(self) -> Any:
        return self._obs_spec


class ExtendedGoalTimeStepWrapper(EnvWrapper):

    def _augment_time_step(self, time_step: TimeStep, action: tp.Optional[np.ndarray] = None) -> TimeStep:
        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        assert isinstance(time_step, GoalTimeStep)
        ts = ExtendedGoalTimeStep(observation=time_step.observation,
                                  step_type=time_step.step_type,
                                  action=action,
                                  reward=time_step.reward or 0.0,
                                  discount=time_step.discount or 1.0,
                                  goal=time_step.goal)
        return super()._augment_time_step(time_step=ts, action=action)


class ExtendedTimeStepWrapper(EnvWrapper):

    def _augment_time_step(self, time_step: TimeStep, action: tp.Optional[np.ndarray] = None) -> TimeStep:
        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        ts = ExtendedTimeStep(observation=time_step.observation,
                              step_type=time_step.step_type,
                              action=action,
                              reward=time_step.reward or 0.0,
                              discount=time_step.discount or 1.0)
        return super()._augment_time_step(time_step=ts, action=action)


class DinoV3EmbedWrapper(EnvWrapper):
    """
    Expects underlying env to have observation dict with pixels_key -> (H,W,3) uint8 (or (1,H,W,3)).
    Replaces time_step.observation with a 1D np.float32 embedding.
    """
    def __init__(
        self,
        env,
        processor,
        dino_model: torch.nn.Module,
        use_cls_token: bool = False,
        device: str = "cuda",
        pixels_key: str = "pixels",
        dino_key: str = "dino_emb",
        out_dim: tp.Optional[int] = None,
        use_amp: bool = True,
        render_shape: tp.Optional[tp.Tuple[int]] = None,
    ) -> None:
        super().__init__(env)
        self._processor = processor
        self.render_shape = render_shape
        self._model = dino_model.eval().to(device)
        for p in self._model.parameters():
            p.requires_grad_(False)
        self._device = device
        self._pixels_key = pixels_key
        self._dino_key = dino_key
        self._use_amp = use_amp
        self._use_cls_token = use_cls_token
        model_config = getattr(self._model, "config", None)
        self._is_vit_mae = getattr(model_config, "model_type", None) == "vit_mae"
        if self._is_vit_mae:
            # ViT-MAE applies random 75% patch masking even in eval mode.  For
            # use as a frozen visual encoder, expose the deterministic encoder
            # representation of the complete image instead.
            model_config.mask_ratio = 0.0


        # # Optional: define observation_spec if你后面有人会读spec
        # self._out_dim = out_dim  # if None, we'll infer on first reset()

        wrapped_obs_spec = env.observation_spec()
        assert isinstance(wrapped_obs_spec, collections.abc.MutableMapping), \
            f"Expected dict obs_spec, got {type(wrapped_obs_spec)}"
        assert pixels_key in wrapped_obs_spec, f"pixels_key={pixels_key} not in obs_spec keys={list(wrapped_obs_spec)}"

        if out_dim is None:
            out_dim = int(getattr(getattr(self._model, "config", None), "hidden_size", 0))
            if out_dim <= 0:
                raise ValueError("Cannot infer out_dim; please pass out_dim.")
        self._out_dim = int(out_dim)

        self._flat_spec = specs.Array(shape=(self._out_dim,), dtype=np.float32, name="observation")

        self._obs_spec = self._flat_spec

    @torch.no_grad()
    def _embed_pixels(self, pixels_uint8: np.ndarray) -> np.ndarray:
        # # pixels_uint8: (H,W,3) uint8
        x = torch.from_numpy(pixels_uint8).to(self._device)  # H W 3, uint8

        inputs = self._processor(images=x, return_tensors="pt").to(self._device)
        if self._is_vit_mae:
            pixel_values = inputs["pixel_values"]
            patch_size = self._model.config.patch_size
            if isinstance(patch_size, int):
                patch_height = patch_width = patch_size
            else:
                patch_height, patch_width = patch_size
            num_patches = (pixel_values.shape[-2] // patch_height) * (pixel_values.shape[-1] // patch_width)
            # Ordered noise preserves the original spatial token order.  With
            # mask_ratio=0 no patch is removed.
            noise = torch.arange(num_patches, device=self._device, dtype=pixel_values.dtype)
            noise = noise.unsqueeze(0).expand(pixel_values.shape[0], -1)
            outputs = self._model(**inputs, noise=noise)
        else:
            outputs = self._model(**inputs)
        if self._use_cls_token:
            emb = outputs.last_hidden_state[:, 0]  # CLS token
        else:
            emb = outputs.last_hidden_state[:, 1:, :].mean(dim=1)  # global avg pooling over patch tokens

        emb = emb.squeeze(0).float().detach().cpu().numpy()
        return emb.astype(np.float32)

    def _extract_pixels(self, time_step: TimeStep) -> np.ndarray:
        px = time_step.observation[self._pixels_key]
        if px.ndim == 4:  # (1,H,W,3)
            px = px[0]
        assert px.ndim == 3 and px.shape[2] == 3, f"expected (H,W,3), got {px.shape}"
        if not px.flags["C_CONTIGUOUS"] or any(s < 0 for s in px.strides):
            px = np.ascontiguousarray(px)
        return px

    def reset(self) -> TimeStep:
        ts = self._env.reset()
        px = self._extract_pixels(ts)
        emb = self._embed_pixels(px)

        ts2 = self._augment_time_step(ts)

        return ts2._replace(observation=emb)


    def step(self, action: np.ndarray) -> TimeStep:
        ts = self._env.step(action)
        px = self._extract_pixels(ts)
        emb = self._embed_pixels(px)
        ts2 = self._augment_time_step(ts, action)

        return ts2._replace(observation=emb)

    def observation_spec(self):
        return self._obs_spec


class EmbedStackWrapper(EnvWrapper):
    """
    Assumes underlying env returns time_step.observation as a 1D float32 vector (D,).
    Outputs stacked observation as (D * num_frames,) float32.
    """
    def __init__(self, env, num_frames: int, emb_key: str = "dino_emb", out_key: str = "dino_emb") -> None:
        super().__init__(env)
        self._num_frames = num_frames
        self._frames: tp.Deque[np.ndarray] = deque([], maxlen=num_frames)
        self._emb_key = emb_key
        self._out_key = out_key
        base_spec = env.observation_spec()
        # self._dict_obs = isinstance(base_spec, specs.Dict)

        if isinstance(base_spec, specs.Array) and len(base_spec.shape) == 1:
            self._dim = int(base_spec.shape[0])
            self._obs_spec = specs.Array(
                shape=(self._dim * self._num_frames,),
                dtype=np.float32,
                name="observation"
            )
        elif isinstance(base_spec, OrderedDict):
            emb_spec = base_spec[self._emb_key]
            if not (isinstance(emb_spec, specs.Array) and len(emb_spec.shape) == 1):
                # fallback: will infer on first reset
                self._dim = None
                self._obs_spec = None
            else:
                self._dim = int(emb_spec.shape[0])
                # preserve all other keys; only update out_key
                new_specs = OrderedDict(base_spec)  # specs.Dict is dict-like; this creates a shallow copy
                new_specs[self._out_key] = specs.Array(
                    shape=(self._dim * self._num_frames,),
                    dtype=np.float32,
                    name=self._out_key
                )
                self._obs_spec = OrderedDict(new_specs)
        else:
            # fallback: will infer on first reset
            self._dim = None
            self._obs_spec = None

    def observation_spec(self):
        if self._obs_spec is not None:
            return self._obs_spec
        # fallback: ask base env; might be specs.Array already
        return self._env.observation_spec()

    def _extract_emb_and_base_obs(self, obs: tp.Any) -> tp.Tuple[np.ndarray, tp.Optional[tp.Any]]:
        """
        Returns (emb, base_obs_dict_or_none).
        - If obs is ndarray: returns (obs, None)
        - If obs is dict-like: returns (obs[emb_key], obs)
        """
        if isinstance(obs, np.ndarray):
            emb = obs
            base_obs = None
        else:
            emb = obs[self._emb_key]
            base_obs = obs
        assert isinstance(emb, np.ndarray) and emb.ndim == 1
        return emb, base_obs

    def _write_stacked_back(self, base_obs: tp.Optional[tp.Any], stacked: np.ndarray) -> tp.Any:
        """
        - If base_obs is None -> return stacked ndarray
        - Else -> return dict-like obs with out_key replaced by stacked, preserving other keys
        """
        if base_obs is None:
            return stacked
        new_obs = OrderedDict(base_obs)  # keeps it simple; if you *must* keep OrderedDict, wrap accordingly
        new_obs[self._out_key] = stacked
        return new_obs

    def _build_spec_if_needed(self, emb: np.ndarray):
        if self._obs_spec is None:
            self._dim = int(emb.shape[0])
            base_spec = self._env.observation_spec()
            if isinstance(base_spec, OrderedDict):
                new_specs = OrderedDict(base_spec)
                new_specs[self._out_key] = specs.Array(
                    shape=(self._dim * self._num_frames,),
                    dtype=np.float32,
                    name=self._out_key
                )
                self._obs_spec = new_specs
            else:
                # original ndarray path
                self._obs_spec = specs.Array(
                    shape=(self._dim * self._num_frames,),
                    dtype=np.float32,
                    name="observation"
                )

    def _augment_time_step(self, time_step: TimeStep, action: tp.Optional[np.ndarray] = None) -> TimeStep:
        super()._augment_time_step(time_step=time_step, action=action)
        assert len(self._frames) == self._num_frames
        stacked = np.concatenate(list(self._frames), axis=0).astype(np.float32, copy=False)
        _, base_obs = self._extract_emb_and_base_obs(time_step.observation)
        new_obs = self._write_stacked_back(base_obs, stacked)

        return time_step._replace(observation=new_obs)

    def reset(self) -> TimeStep:
        ts = self._env.reset()
        emb, _ = self._extract_emb_and_base_obs(ts.observation)
        self._build_spec_if_needed(emb)
        # assert isinstance(emb, np.ndarray) and emb.ndim == 1
        self._frames.clear()
        # self._build_spec_if_needed(emb)
        for _ in range(self._num_frames):
            self._frames.append(emb)
        return self._augment_time_step(ts)

    def step(self, action: np.ndarray) -> TimeStep:
        ts = self._env.step(action)
        emb, _ = self._extract_emb_and_base_obs(ts.observation)
        # assert isinstance(emb, np.ndarray) and emb.ndim == 1
        self._frames.append(emb)
        return self._augment_time_step(ts, action)


def _make_jaco(obs_type, domain, task, frame_stack, action_repeat, seed,
               goal_space: tp.Optional[str] = None, append_goal_to_observation: bool = False
               ) -> FlattenJacoObservationWrapper:
    env = cdmc.make_jaco(task, obs_type, seed)
    if goal_space is not None:
        # inline because circular import
        from url_benchmark import goals as _goals  # pytlint: disable=import-outside-toplevel
        funcs = _goals.goal_spaces.funcs[domain]
        if goal_space not in funcs:
            raise ValueError(f"No goal space {goal_space} for {domain}, avail: {list(funcs)}")
        goal_func = funcs[goal_space]
        env = GoalWrapper(env, goal_func, append_goal_to_observation=append_goal_to_observation)
    env = ActionDTypeWrapper(env, np.float32)
    env = ActionRepeatWrapper(env, action_repeat)
    env = FlattenJacoObservationWrapper(env)
    return env


def _make_dmc(
    obs_type,
    domain,
    task,
    frame_stack,
    action_repeat,
    seed,
    goal_space: tp.Optional[str] = None,
    append_goal_to_observation: bool = False,
    render_shape: tp.Tuple[int, int] = (224, 224),
    visual_perturbation: tp.Optional[str] = None,
    visual_perturb_seed: int = 0,
):
    visualize_reward = False
    if (domain, task) in suite.ALL_TASKS:
        env = suite.load(domain,
                         task,
                         task_kwargs=dict(random=seed),
                         environment_kwargs=dict(flat_observation=True),
                         visualize_reward=visualize_reward)
    else:
        env = cdmc.make(domain,
                        task,
                        task_kwargs=dict(random=seed),
                        environment_kwargs=dict(flat_observation=True),
                        visualize_reward=visualize_reward)
    if goal_space is not None:
        # inline because circular import
        from url_benchmark import goals as _goals  # pytlint: disable=import-outside-toplevel
        funcs = _goals.goal_spaces.funcs[domain]
        if goal_space not in funcs:
            raise ValueError(f"No goal space {goal_space} for {domain}, avail: {list(funcs)}")
        goal_func = funcs[goal_space]
        env = GoalWrapper(env, goal_func, append_goal_to_observation=append_goal_to_observation)
    env = ActionDTypeWrapper(env, np.float32)
    env = ActionRepeatWrapper(env, action_repeat)
    if obs_type == 'pixels' or obs_type == 'dino' or obs_type == 'vit':
        # zoom in camera for quadruped
        camera_id = dict(quadruped=2).get(domain, 0)
        render_kwargs = dict(height=render_shape[0], width=render_shape[1], camera_id=camera_id)
        env = pixels.Wrapper(env,
                             pixels_only=True,
                             render_kwargs=render_kwargs)
        if visual_perturbation is not None:
            env = PixelPerturbationWrapper(env, visual_perturbation, seed=visual_perturb_seed)
    return env


def make(
    name: str, obs_type='states', frame_stack=1, action_repeat=1,
    seed=1, goal_space: tp.Optional[str] = None, append_goal_to_observation: bool = False, use_cls=True, dino_model = None, dino_processor = None,
    render_shape=(224, 224), visual_perturbation: tp.Optional[str] = None, visual_perturb_seed: int = 0,
    dino_frame_stack: int = 1,
 ) -> EnvWrapper:
    if append_goal_to_observation and goal_space is None:
        raise ValueError("Cannot append goal space since none is defined")
    assert obs_type in ['states', 'pixels', 'dino', 'vit']
    if name.startswith('point_mass_maze'):
        domain = 'point_mass_maze'
        _, _, _, task = name.split('_', 3)
    elif name.startswith('point_mass_'):
        domain = 'point_mass'
        task = name[len('point_mass_'):]
    else:
        domain, task = name.split('_', 1)
    domain = dict(cup='ball_in_cup').get(domain, domain)
    if sys.platform == "darwin":
        raise UnsupportedPlatform("Mac platform is not supported")

    if domain == 'jaco':
        if visual_perturbation is not None:
            raise ValueError("visual_perturbation is only supported for DMC pixel-rendered tasks")
        env = _make_jaco(obs_type, domain, task, frame_stack, action_repeat, seed,
                         goal_space=goal_space, append_goal_to_observation=append_goal_to_observation)
    else:
        env = _make_dmc(obs_type, domain, task, frame_stack, action_repeat, seed,
                        goal_space=goal_space, append_goal_to_observation=append_goal_to_observation,
                        render_shape=render_shape, visual_perturbation=visual_perturbation,
                        visual_perturb_seed=visual_perturb_seed)

    if obs_type == 'pixels':
        env = FrameStackWrapper(env, frame_stack)
    elif obs_type == 'vit':
        # ViT consumes raw pixels and handles representation learning in the agent.
        env = FrameStackWrapper(env, 1)
    elif obs_type == 'dino':
        env = DinoV3EmbedWrapper(
            env,
            processor=dino_processor,
            dino_model=dino_model,
            use_cls_token=use_cls,
            render_shape=render_shape,
        )
        # Stack frozen DINO embeddings in temporal order (oldest -> newest).
        # Each new environment observation still incurs exactly one DINO
        # forward; prior embeddings are retained by the wrapper.
        env = EmbedStackWrapper(env, dino_frame_stack)
    else:
        env = ObservationDTypeWrapper(env, np.float32)

    env = action_scale.Wrapper(env, minimum=-1.0, maximum=+1.0)
    if goal_space is not None:
        env = ExtendedGoalTimeStepWrapper(env)
    else:
        env = ExtendedTimeStepWrapper(env)
    return env


def extract_physics(env: Env) -> tp.Dict[str, float]:
    """Extract some physics available in the env"""
    output = {}
    names = ["torso_height", "torso_upright", "horizontal_velocity", "torso_velocity"]
    for name in names:
        if not hasattr(env.physics, name):
            continue
        val: tp.Union[float, np.ndarray] = getattr(env.physics, name)()
        if isinstance(val, (int, float)) or not val.ndim:
            output[name] = float(val)
        else:
            for k, v in enumerate(val):
                output[f"{name}#{k}"] = float(v)
    return output


class FloatStats:
    """Handle for keeping track of the statistics of a float variable"""

    def __init__(self) -> None:
        self.min = np.inf
        self.max = -np.inf
        self.mean = 0.0
        self._count = 0

    def add(self, value: float) -> "FloatStats":
        self.min = min(value, self.min)
        self.max = max(value, self.max)
        self._count += 1
        self.mean = (self._count - 1) / self._count * self.mean + 1 / self._count * value
        return self

    def items(self) -> tp.Iterator[tp.Tuple[str, float]]:
        for name, val in self.__dict__.items():
            if not name.startswith("_"):
                yield name, val


class PhysicsAggregator:
    """Aggregate stats on the physics of an environment"""

    def __init__(self) -> None:
        self.stats: tp.Dict[str, FloatStats] = {}

    def add(self, env: Env) -> "PhysicsAggregator":
        phy = extract_physics(env)
        for key, val in phy.items():
            self.stats.setdefault(key, FloatStats()).add(val)
        return self

    def dump(self) -> tp.Iterator[tp.Tuple[str, float]]:
        """Exports all statistics and reset the statistics"""
        for key, stats in self.stats.items():
            for stat, val in stats.items():
                yield (f'{key}/{stat}', val)
        self.stats.clear()
