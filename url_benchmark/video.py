# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
from pathlib import Path
import cv2
import imageio
import numpy as np
import wandb


def _frame_from_source(source: tp.Any, render_size: int, camera_id: int) -> tp.Optional[np.ndarray]:
    if isinstance(source, np.ndarray):
        frame = source
        if frame.ndim != 3:
            return None
        if frame.shape[0] >= 3:
            frame = frame[-3:].transpose(1, 2, 0)
        elif frame.shape[-1] != 3:
            return None
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        return cv2.resize(frame, dsize=(render_size, render_size), interpolation=cv2.INTER_CUBIC)

    if hasattr(source, 'physics') and source.physics is not None:
        return source.physics.render(height=render_size, width=render_size, camera_id=camera_id)
    if hasattr(source, 'base_env') and hasattr(source.base_env, 'render'):
        return source.base_env.render()
    if hasattr(source, 'render'):
        return source.render()
    return None


class VideoRecorder:
    def __init__(self,
                 root_dir: tp.Optional[tp.Union[str, Path]],
                 render_size: int = 256,
                 fps: int = 20,
                 camera_id: int = 0,
                 use_wandb: bool = False) -> None:
        self.save_dir: tp.Optional[Path] = None
        if root_dir is not None:
            self.save_dir = Path(root_dir) / 'eval_video'
            self.save_dir.mkdir(exist_ok=True)
        self.enabled = False
        self.render_size = render_size
        self.fps = fps
        self.frames: tp.List[np.ndarray] = []
        self.camera_id = camera_id
        self.use_wandb = use_wandb

    def init(self, env, enabled: bool = True) -> None:
        self.frames = []
        self.enabled = self.save_dir is not None and enabled
        self.record(env)

    def record(self, env) -> None:
        if self.enabled:
            frame = _frame_from_source(env, self.render_size, self.camera_id)
            if frame is not None:
                self.frames.append(frame)

    def log_to_wandb(self) -> None:
        frames = np.transpose(np.array(self.frames), (0, 3, 1, 2))
        fps, skip = 6, 8
        wandb.log({
            'eval/video':
            wandb.Video(frames[::skip, :, ::2, ::2], fps=fps, format="gif")
        })

    def save(self, file_name: str) -> None:
        if self.enabled:
            if self.use_wandb:
                self.log_to_wandb()
            assert self.save_dir is not None
            path = self.save_dir / file_name
            imageio.mimsave(str(path), self.frames, fps=self.fps)  # type: ignore


class TrainVideoRecorder:
    def __init__(self,
                 root_dir: tp.Optional[tp.Union[str, Path]],
                 render_size: int = 256,
                 fps: int = 20,
                 camera_id: int = 0,
                 use_wandb: bool = False) -> None:
        self.save_dir: tp.Optional[Path] = None
        if root_dir is not None:
            self.save_dir = Path(root_dir) / 'train_video'
            self.save_dir.mkdir(exist_ok=True)

        self.enabled = False
        self.render_size = render_size
        self.fps = fps
        self.frames: tp.List[np.ndarray] = []
        self.camera_id = camera_id
        self.use_wandb = use_wandb

    def init(self, source, enabled=True) -> None:
        self.frames = []
        self.enabled = self.save_dir is not None and enabled
        self.record(source)

    def record(self, source) -> None:
        if self.enabled:
            frame = _frame_from_source(source, self.render_size, self.camera_id)
            if frame is not None:
                self.frames.append(frame)

    def log_to_wandb(self) -> None:
        frames = np.transpose(np.array(self.frames), (0, 3, 1, 2))
        fps, skip = 6, 8
        wandb.log({
            'train/video':
            wandb.Video(frames[::skip, :, ::2, ::2], fps=fps, format="gif")
        })

    def save(self, file_name) -> None:
        if self.enabled:
            if self.use_wandb:
                self.log_to_wandb()
            assert self.save_dir is not None
            path = self.save_dir / file_name
            imageio.mimsave(str(path), self.frames, fps=self.fps)  # type: ignore
