import argparse
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from url_benchmark import analyze_fb_pixel_b_mechanisms as diagnostic

# REDUNDANCY REVIEW: see test_analyze_fb_pixel_b_deferred_cuda.py -- the last of the five
# test files covering private helpers of the offline, read-only
# analyze_fb_pixel_b_mechanisms.py checkpoint-analysis script. Not reachable from
# pretrain.py or any launch_*.sh; does not affect the training pipeline.


class TestCnnRendererBackendContract(unittest.TestCase):
    def test_default_egl_contract_keeps_chunk_eight_and_records_device(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"MUJOCO_GL": "egl", "MUJOCO_EGL_DEVICE_ID": "3"},
            clear=True,
        ):
            contract = diagnostic._cnn_pixel_render_context_contract((84, 84))
        self.assertEqual(contract["mujoco_gl_backend"], "egl")
        self.assertEqual(contract["mujoco_egl_device_id"], "3")
        self.assertEqual(contract["context_chunk_size_rows"], 8)
        self.assertEqual(
            contract["render_call"], {"height": 84, "width": 84, "camera_id": 0}
        )

    def test_osmesa_contract_uses_safe_chunk_eight_and_diagnosis_limitation(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"MUJOCO_GL": "osmesa", "MUJOCO_EGL_DEVICE_ID": "ignored"},
            clear=True,
        ):
            contract = diagnostic._cnn_pixel_render_context_contract((84, 84))
        self.assertEqual(contract["mujoco_gl_backend"], "osmesa")
        self.assertIsNone(contract["mujoco_egl_device_id"])
        self.assertEqual(contract["context_chunk_size_rows"], 8)
        diagnosis = diagnostic.cnn_diagnosis_text([], [], contract)
        self.assertIn("re-rendered with OSMesa", diagnosis)
        self.assertIn("not the historical EGL online pixel replay", diagnosis)
        self.assertIn("renderer sensitivity remains", diagnosis)

    def test_cache_resume_fails_closed_when_backend_changes(self) -> None:
        with tempfile.TemporaryDirectory(prefix="cnn-render-cache-", dir="/tmp") as text:
            root = Path(text)
            exorl = root / "exorl"
            exorl.mkdir()
            bank = root / "bank.npz"
            bank.write_bytes(b"synthetic-bank")
            args = argparse.Namespace(
                output_dir=root,
                render_shape=(84, 84),
                exorl_dir=exorl,
                index_bank=bank,
                replay_seed=diagnostic.DEFAULT_SEED,
                z_seed=diagnostic.DEFAULT_Z_SEED,
                replay_size=1,
                probe_size=1,
                inference_size=1,
            )
            with mock.patch.dict(os.environ, {"MUJOCO_GL": "egl"}, clear=True):
                stored_context = diagnostic._cnn_pixel_render_context_contract((84, 84))
            metadata = {
                "schema_version": diagnostic.SCHEMA_VERSION,
                "obs_type": "pixels",
                "render_shape": [84, 84],
                "pixel_render_context": stored_context,
                "source_dir": str(exorl.resolve()),
                "source_index_bank": str(bank.resolve()),
                "source_index_bank_sha256": diagnostic.sha256_file(bank),
                "replay_seed": args.replay_seed,
                "z_seed": args.z_seed,
                "size": args.replay_size,
                "probe_size": args.probe_size,
                "inference_size": args.inference_size,
                "content_sha256": "not-reached-because-contract-mismatch",
            }
            cache = root / "fixed_rendered_replay.npz"
            diagnostic.atomic_npz(
                cache,
                obs=np.zeros((1, 9, 84, 84), dtype=np.uint8),
                next_obs=np.zeros((1, 9, 84, 84), dtype=np.uint8),
                action=np.zeros((1, 6), dtype=np.float32),
                discount=np.ones((1, 1), dtype=np.float32),
                physics=np.zeros((1, 18), dtype=np.float64),
                source_reward=np.zeros((1, 1), dtype=np.float32),
                random_z=np.zeros((1, diagnostic.Z_DIM), dtype=np.float32),
                episode_id=np.zeros(1, dtype=np.int32),
                step_in_episode=np.zeros(1, dtype=np.int32),
                index_bank_row=np.zeros(1, dtype=np.int32),
                metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
            )
            with mock.patch.dict(os.environ, {"MUJOCO_GL": "osmesa"}, clear=True):
                with self.assertRaisesRegex(ValueError, "pixel_render_context"):
                    diagnostic.load_replay_cache(cache, args, "pixels")


REAL_EPISODE = Path(
    "/mnt/data_7tb/fanfeng/exoRL_datasets/cheetah/rnd/buffer_dino_cls/"
    "episode_000037_1000.npz"
)


class TestRealOsmesaRendererEquivalence(unittest.TestCase):
    @unittest.skipUnless(
        os.environ.get("CNN_REAL_OSMESA_RENDER_EQUIVALENCE") == "1"
        and os.environ.get("MUJOCO_GL", "").lower() == "osmesa"
        and REAL_EPISODE.is_file()
        and diagnostic.DEFAULT_BANK.is_file(),
        "opt-in real OSMesa byte/digest-equivalence test",
    )
    def test_episode_37_small_reference_and_full_chunked_repeat(self) -> None:
        with np.load(diagnostic.DEFAULT_BANK, allow_pickle=False) as bank:
            episode_all = np.asarray(bank["episode_id"], dtype=np.int32)
            step_all = np.asarray(bank["step_in_episode"], dtype=np.int32)
        order = np.random.RandomState(diagnostic.DEFAULT_SEED).permutation(
            len(episode_all)
        )[: diagnostic.CNN_DEFAULT_REPLAY_SIZE]
        selected_steps = step_all[order][episode_all[order] == 37].astype(np.int64)
        self.assertEqual(len(selected_steps), 80)
        with np.load(REAL_EPISODE, allow_pickle=False) as payload:
            states = np.asarray(payload["physics"], dtype=np.float64)

        # Never run the pathological full-80 single-context reference.  A small
        # prefix proves assembly equivalence; two full chunked passes prove the
        # formal selected rows are stable across independent renderer contexts.
        reference_steps = selected_steps[:16]
        reference_env = diagnostic._make_cnn_pixel_render_env()
        reference = diagnostic._render_pixel_stacks(
            reference_env.physics, states, reference_steps, (84, 84)
        )
        del reference_env
        reference_chunked = diagnostic._render_pixel_stacks_context_chunked(
            states, reference_steps, (84, 84)
        )
        self.assertTrue(np.array_equal(reference_chunked[0], reference[0]))
        self.assertTrue(np.array_equal(reference_chunked[1], reference[1]))
        self.assertEqual(
            diagnostic.hash_arrays(
                (("obs", reference_chunked[0]), ("next_obs", reference_chunked[1]))
            ),
            diagnostic.hash_arrays(
                (("obs", reference[0]), ("next_obs", reference[1]))
            ),
        )

        first_full = diagnostic._render_pixel_stacks_context_chunked(
            states, selected_steps, (84, 84)
        )
        repeated_full = diagnostic._render_pixel_stacks_context_chunked(
            states, selected_steps, (84, 84)
        )
        self.assertTrue(np.array_equal(repeated_full[0], first_full[0]))
        self.assertTrue(np.array_equal(repeated_full[1], first_full[1]))
        self.assertEqual(
            diagnostic.hash_arrays(
                (("obs", repeated_full[0]), ("next_obs", repeated_full[1]))
            ),
            diagnostic.hash_arrays(
                (("obs", first_full[0]), ("next_obs", first_full[1]))
            ),
        )


if __name__ == "__main__":
    unittest.main()
