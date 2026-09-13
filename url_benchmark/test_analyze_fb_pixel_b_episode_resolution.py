import tempfile
import unittest
from pathlib import Path

import numpy as np

from url_benchmark import analyze_fb_pixel_b_mechanisms as diagnostic

# REDUNDANCY REVIEW: see test_analyze_fb_pixel_b_deferred_cuda.py -- another of the five
# test files covering private helpers of the offline, read-only
# analyze_fb_pixel_b_mechanisms.py checkpoint-analysis script. Not reachable from
# pretrain.py or any launch_*.sh; does not affect the training pipeline.


class TestSelectedExorlEpisodeResolution(unittest.TestCase):
    def test_exact_zero_padded_selected_paths_preserve_sorted_unique_ids(self) -> None:
        with tempfile.TemporaryDirectory(prefix="exorl-resolution-", dir="/tmp") as text:
            root = Path(text)
            expected = {
                0: root / "episode_000000_1000.npz",
                7: root / "episode_000007_1000.npz",
                123456: root / "episode_123456_1000.npz",
            }
            for path in expected.values():
                path.touch()
            # This unrelated file must never be scanned or selected.
            (root / "episode_999999_1000.npz").touch()

            observed = diagnostic._resolve_selected_exorl_episode_files(
                root, np.asarray([7, 0, 123456, 7], dtype=np.int32)
            )

            self.assertEqual(observed, expected)
            self.assertEqual(list(observed), [0, 7, 123456])
            self.assertEqual(
                diagnostic._exorl_episode_filename_contract()["template"],
                "episode_{episode_id:06d}_1000.npz",
            )

    def test_missing_exact_file_fails_without_suffix_or_glob_fallback(self) -> None:
        with tempfile.TemporaryDirectory(prefix="exorl-missing-", dir="/tmp") as text:
            root = Path(text)
            # Similar names deliberately exist, but the exact fixed-contract
            # episode_000008_1000.npz does not.
            (root / "episode_000008_0999.npz").touch()
            (root / "episode_8_1000.npz").touch()

            observed = diagnostic._resolve_selected_exorl_episode_files(
                root, np.asarray([8], dtype=np.int32)
            )
            exact_path = root / "episode_000008_1000.npz"
            self.assertEqual(observed, {8: exact_path})
            with self.assertRaisesRegex(
                FileNotFoundError, "episode_000008_1000[.]npz"
            ):
                diagnostic._require_exact_exorl_episode_file(8, observed[8])

    def test_episode_ids_outside_six_digit_contract_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="exorl-negative-", dir="/tmp") as text:
            for episode_id in (-1, 1_000_000):
                with self.subTest(episode_id=episode_id):
                    with self.assertRaisesRegex(ValueError, "zero-padded width 6"):
                        diagnostic._resolve_selected_exorl_episode_files(
                            Path(text), np.asarray([episode_id], dtype=np.int32)
                        )


if __name__ == "__main__":
    unittest.main()
