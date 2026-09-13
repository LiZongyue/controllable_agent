import contextlib
import math
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator
from unittest import mock

import numpy as np
import torch

from url_benchmark import analyze_fb_pixel_b_mechanisms as diagnostic

# REDUNDANCY REVIEW: see test_analyze_fb_pixel_b_deferred_cuda.py -- this is the largest of
# five test files (368 lines here) covering private helpers of the offline, read-only
# analyze_fb_pixel_b_mechanisms.py checkpoint-analysis script. Not reachable from
# pretrain.py or any launch_*.sh; does not affect the training pipeline.


class _DeterministicPhysics:
    def __init__(self) -> None:
        self.state = np.zeros(1, dtype=np.float64)

    @contextlib.contextmanager
    def reset_context(self) -> Iterator[None]:
        yield

    def set_state(self, state: np.ndarray) -> None:
        self.state = np.asarray(state, dtype=np.float64).copy()

    def render(self, *, height: int, width: int, camera_id: int) -> np.ndarray:
        assert camera_id == 0
        pixels = np.arange(height * width * 3, dtype=np.uint16).reshape(
            height, width, 3
        )
        return ((pixels + int(self.state[0])) % 256).astype(np.uint8)


class _DeterministicEnv:
    def __init__(self) -> None:
        self.physics = _DeterministicPhysics()


REAL_EPISODE = Path(
    "/mnt/data_7tb/fanfeng/exoRL_datasets/cheetah/rnd/buffer_dino_cls/"
    "episode_000037_1000.npz"
)


class TestCnnPixelRenderContextChunking(unittest.TestCase):
    def test_stack_bytes_and_episode_row_order(self) -> None:
        states = np.arange(20, dtype=np.float64)[:, None]
        # Intentionally non-monotonic with repeats: this is the episode-local
        # order produced by selecting globally shuffled replay destinations.
        steps = np.asarray([5, 2, 8, 3, 5, 9, 2], dtype=np.int64)
        shape = (3, 4)
        reference = diagnostic._render_pixel_stacks(
            _DeterministicPhysics(), states, steps, shape
        )
        environments = []

        def factory() -> _DeterministicEnv:
            env = _DeterministicEnv()
            environments.append(env)
            return env

        chunked = diagnostic._render_pixel_stacks_context_chunked(
            states, steps, shape, context_chunk_size=3, env_factory=factory
        )

        self.assertEqual(len(environments), math.ceil(len(steps) / 3))
        self.assertTrue(np.array_equal(chunked[0], reference[0]))
        self.assertTrue(np.array_equal(chunked[1], reference[1]))
        self.assertEqual(
            diagnostic.hash_arrays(
                (("obs", chunked[0]), ("next_obs", chunked[1]))
            ),
            diagnostic.hash_arrays(
                (("obs", reference[0]), ("next_obs", reference[1]))
            ),
        )
        with mock.patch.dict(os.environ, {"MUJOCO_GL": "egl"}, clear=True):
            contract = diagnostic._cnn_pixel_render_context_contract(shape)
        self.assertEqual(contract["context_chunk_size_rows"], 8)
        self.assertEqual(
            contract["render_call"], {"height": 3, "width": 4, "camera_id": 0}
        )
        self.assertIn("unchanged destination indices", contract["row_chunking"])

    @unittest.skipUnless(
        os.environ.get("CNN_REAL_RENDER_EQUIVALENCE") == "1"
        and REAL_EPISODE.is_file(),
        "opt-in real DMC/EGL byte-equivalence test",
    )
    def test_real_episode_is_byte_identical_to_single_context(self) -> None:
        with np.load(REAL_EPISODE, allow_pickle=False) as payload:
            states = np.asarray(payload["physics"], dtype=np.float64)
        steps = np.asarray(
            [37, 2, 71, 18, 5, 39, 6, 70, 21, 3, 55], dtype=np.int64
        )
        render_shape = (84, 84)
        reference_env = diagnostic._make_cnn_pixel_render_env()
        reference = diagnostic._render_pixel_stacks(
            reference_env.physics, states, steps, render_shape
        )
        chunked = diagnostic._render_pixel_stacks_context_chunked(
            states, steps, render_shape
        )

        self.assertTrue(np.array_equal(chunked[0], reference[0]))
        self.assertTrue(np.array_equal(chunked[1], reference[1]))
        self.assertEqual(
            diagnostic.hash_arrays(
                (("obs", chunked[0]), ("next_obs", chunked[1]))
            ),
            diagnostic.hash_arrays(
                (("obs", reference[0]), ("next_obs", reference[1]))
            ),
        )


class TestLoadedAgentContract(unittest.TestCase):
    def _contract(self, *, use_cls: bool) -> dict:
        return {
            "obs_type": "pixels",
            "obs_shape": (9, 84, 84),
            "action_shape": (6,),
            "z_dim": diagnostic.Z_DIM,
            "batch_size": diagnostic.TRAIN_BATCH,
            "num_inference_steps": diagnostic.DEFAULT_INFERENCE_SIZE,
            "goal_space": None,
            "norm_z": True,
            "q_loss": False,
            "idm_coef": 0.0,
            "future_ratio": 0.0,
            "rand_weight": False,
            "boltzmann": False,
            "use_cls": use_cls,
            "dino_separate_backward_adapter": False,
            "dino_separate_fb_adapters": False,
            "dino_adapter_type": "linear",
            "pixel_separate_fb_encoders": False,
            "idm_route": "none",
            "idm_encoder_mode": "legacy",
            "idm_lr": None,
        }

    def test_pixel_lineages_keep_their_exact_serialized_use_cls(self) -> None:
        for experiment, use_cls in (
            (diagnostic.CNN_EXPERIMENTS[0], True),
            (diagnostic.CNN_EXPERIMENTS[1], False),
        ):
            observed = self._contract(use_cls=use_cls)
            expected = diagnostic.expected_loaded_agent_contract(
                experiment, (9, 84, 84)
            )
            self.assertEqual(expected, observed)

            observed["use_cls"] = not use_cls
            self.assertNotEqual(expected, observed)

    def test_pixel_semantic_fields_remain_strict(self) -> None:
        experiment = diagnostic.CNN_EXPERIMENTS[1]
        observed = self._contract(use_cls=False)
        observed["q_loss"] = True
        expected = diagnostic.expected_loaded_agent_contract(
            experiment, (9, 84, 84)
        )
        self.assertNotEqual(expected, observed)
        self.assertFalse(expected["q_loss"])

    def test_dino_use_cls_remains_required(self) -> None:
        experiment = diagnostic.EXPERIMENTS[0]
        observed = self._contract(use_cls=False)
        observed["obs_type"] = "dino"
        observed["obs_shape"] = (
            diagnostic.FRAME_STACK * diagnostic.EMBED_DIM,
        )
        expected = diagnostic.expected_loaded_agent_contract(
            experiment, observed["obs_shape"]
        )
        self.assertTrue(expected["use_cls"])
        self.assertNotEqual(expected, observed)

    def test_separate_cnn_and_dino_contracts_are_explicit(self) -> None:
        cnn = diagnostic.CNN_SEPARATE_EXPERIMENTS[0]
        cnn_contract = diagnostic.expected_loaded_agent_contract(
            cnn, (9, 84, 84)
        )
        self.assertTrue(cnn_contract["pixel_separate_fb_encoders"])
        self.assertFalse(cnn_contract["dino_separate_fb_adapters"])

        dino = diagnostic.DINO_SEPARATE_EXPERIMENTS[4]
        dino_contract = diagnostic.expected_loaded_agent_contract(
            dino, (diagnostic.FRAME_STACK * diagnostic.EMBED_DIM,)
        )
        self.assertTrue(dino_contract["dino_separate_fb_adapters"])
        self.assertEqual(dino_contract["dino_adapter_type"], "mlp_ln")
        self.assertEqual(dino_contract["idm_route"], "forward_adapter")
        self.assertEqual(dino_contract["idm_lr"], 1e-4)


class TestCheckpointLoaderCompatibility(unittest.TestCase):
    def test_branch_optimizers_are_selected_for_clean_separate_mode(self) -> None:
        forward_parameter = torch.nn.Parameter(torch.zeros(()))
        backward_parameter = torch.nn.Parameter(torch.zeros(()))
        agent = SimpleNamespace(
            cfg=SimpleNamespace(
                pixel_separate_fb_encoders=True,
                dino_separate_fb_adapters=False,
            ),
            fb_opt=None,
            forward_fb_opt=torch.optim.Adam([forward_parameter], lr=1e-4),
            backward_fb_opt=torch.optim.Adam([backward_parameter], lr=2e-4),
        )
        self.assertEqual(
            diagnostic.loaded_fb_optimizer_lrs(agent),
            {"forward_fb_opt": (1e-4,), "backward_fb_opt": (2e-4,)},
        )

    def test_resume_size_contract_accepts_unpinned_fresh_lineage(self) -> None:
        with tempfile.TemporaryDirectory(prefix="checkpoint-size-", dir="/tmp") as text:
            checkpoint = Path(text) / "snapshot_100000.pt"
            checkpoint.write_bytes(b"future-separate-checkpoint")
            self.assertEqual(
                diagnostic.validated_checkpoint_size(checkpoint, None),
                checkpoint.stat().st_size,
            )
            with self.assertRaisesRegex(ValueError, "checkpoint size mismatch"):
                diagnostic.validated_checkpoint_size(
                    checkpoint, checkpoint.stat().st_size + 1
                )


class TestChunkedQFloat64Terms(unittest.TestCase):
    def test_encode_forward_expands_constant_z_on_device(self) -> None:
        class DummyAgent:
            def __init__(self) -> None:
                self.z_shapes = []
                self.z_strides = []

            def aug_and_encode(self, raw_obs):
                return raw_obs

            def forward_net(self, obs, z, action):
                self.z_shapes.append(tuple(z.shape))
                self.z_strides.append(tuple(z.stride()))
                offset = action.mean(dim=1, keepdim=True)
                return z + offset, z - offset

        rows = 5
        task_z = np.linspace(-1.0, 1.0, diagnostic.Z_DIM).astype(
            np.float32
        )
        raw_obs = np.zeros((rows, 3), dtype=np.float32)
        action = np.arange(rows * 2, dtype=np.float32).reshape(rows, 2)
        agent = DummyAgent()
        f1, f2, observed_action = diagnostic.encode_forward(
            agent,
            raw_obs,
            task_z,
            action,
            "cpu",
            chunk_size=2,
            action_mode="replay",
        )

        offsets = action.mean(axis=1, keepdims=True)
        np.testing.assert_allclose(f1, task_z[None, :] + offsets)
        np.testing.assert_allclose(f2, task_z[None, :] - offsets)
        np.testing.assert_array_equal(observed_action, action)
        self.assertEqual(agent.z_shapes, [(2, diagnostic.Z_DIM)] * 2 + [(1, diagnostic.Z_DIM)])
        self.assertEqual(agent.z_strides[:2], [(0, 1), (0, 1)])

    def test_chunked_terms_match_full_float64_reference(self) -> None:
        rng = np.random.RandomState(20260901)
        rows = 521
        latent = 7
        z = rng.standard_normal((rows, latent)).astype(np.float32)
        forwards = tuple(
            rng.standard_normal((rows, latent)).astype(np.float32)
            for _ in range(2)
        )
        basis, _ = np.linalg.qr(rng.standard_normal((latent, latent)))
        projector = basis[:, :3] @ basis[:, :3].T

        z64 = np.asarray(z, dtype=np.float64)
        forward64 = tuple(
            np.asarray(value, dtype=np.float64) for value in forwards
        )
        expected_q = tuple(
            np.sum(value * z64, axis=1) for value in forward64
        )
        z_weak = z64 @ projector
        z_strong = z64 - z_weak
        expected_ratio = float(
            np.square(z_weak).sum() / np.square(z64).sum()
        )
        expected_weak = []
        expected_strong = []
        for value in forward64:
            value_weak = value @ projector
            value_strong = value - value_weak
            expected_weak.append(
                np.sum(value_weak * z_weak, axis=1)
            )
            expected_strong.append(
                np.sum(value_strong * z_strong, axis=1)
            )

        observed_q = diagnostic._q_row_dots_float64_chunked(
            forwards, z, chunk_rows=31
        )
        observed_ratio, observed_weak, observed_strong = (
            diagnostic._q_projected_parts_float64_chunked(
                forwards, z, projector, chunk_rows=31
            )
        )
        for observed, expected in zip(observed_q, expected_q):
            np.testing.assert_array_equal(observed, expected)
        self.assertAlmostEqual(observed_ratio, expected_ratio, places=14)
        for observed, expected in zip(observed_weak, expected_weak):
            np.testing.assert_allclose(
                observed, expected, rtol=2e-13, atol=2e-13
            )
        for observed, expected in zip(observed_strong, expected_strong):
            np.testing.assert_allclose(
                observed, expected, rtol=2e-13, atol=2e-13
            )

    def test_float64_matrix_conversions_are_bounded_by_chunk_rows(self) -> None:
        rng = np.random.RandomState(9)
        rows = 389
        latent = 11
        chunk_rows = 23
        z = rng.standard_normal((rows, latent)).astype(np.float32)
        forwards = tuple(
            rng.standard_normal((rows, latent)).astype(np.float32)
            for _ in range(2)
        )
        projector = np.eye(latent, dtype=np.float64)
        original_asarray = diagnostic.np.asarray
        converted_rows = []

        def tracking_asarray(value, *args, **kwargs):
            dtype = kwargs.get("dtype", args[0] if args else None)
            result = original_asarray(value, *args, **kwargs)
            if (
                dtype is not None
                and np.dtype(dtype) == np.dtype(np.float64)
                and result.ndim == 2
                and result.shape != projector.shape
            ):
                converted_rows.append(result.shape[0])
            return result

        with mock.patch.object(
            diagnostic.np, "asarray", side_effect=tracking_asarray
        ):
            diagnostic._q_row_dots_float64_chunked(
                forwards, z, chunk_rows=chunk_rows
            )
            diagnostic._q_projected_parts_float64_chunked(
                forwards, z, projector, chunk_rows=chunk_rows
            )
        self.assertTrue(converted_rows)
        self.assertLessEqual(max(converted_rows), chunk_rows)


if __name__ == "__main__":
    unittest.main()
