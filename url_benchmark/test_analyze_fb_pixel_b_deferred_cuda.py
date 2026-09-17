from __future__ import annotations

import argparse
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from url_benchmark import analyze_fb_pixel_b_mechanisms as diagnostic

# REDUNDANCY REVIEW: this file, together with test_analyze_fb_pixel_b_episode_parts.py,
# test_analyze_fb_pixel_b_episode_resolution.py, test_analyze_fb_pixel_b_mechanisms.py and
# test_analyze_fb_pixel_b_renderer_backend.py, tests private helpers of
# analyze_fb_pixel_b_mechanisms.py -- a read-only, run-once-per-checkpoint offline analysis
# script that pretrain.py never imports and no launch_*.sh ever calls. None of these ~1500
# lines of tests gate the training pipeline; they only protect a one-off diagnostic tool.


class TestDeferredCudaInitialization(unittest.TestCase):
    def test_explicit_cuda_reaches_replay_before_availability_probe(self) -> None:
        events = []
        with tempfile.TemporaryDirectory(prefix="cnn-cuda-order-", dir="/tmp") as text:
            args = argparse.Namespace(
                output_dir=Path(text), device="cuda:0", render_shape=(84, 84)
            )

            def initialized() -> bool:
                events.append("is_initialized")
                return False

            def build(*_args: object, **_kwargs: object) -> tuple[object, Path]:
                events.append("replay")
                return object(), Path(text) / "fixed_rendered_replay.npz"

            def available() -> bool:
                events.append("is_available")
                return False

            with mock.patch.object(
                    diagnostic, "validate_cnn_args", return_value=([], [], [])
                ), mock.patch.object(
                    diagnostic, "atomic_csv"
                ), mock.patch.object(
                    diagnostic.torch.cuda, "is_initialized", side_effect=initialized
                ), mock.patch.object(
                    diagnostic, "load_or_build_replay_cache", side_effect=build
                ), mock.patch.object(
                    diagnostic.torch.cuda, "is_available", side_effect=available
                ):
                with self.assertRaisesRegex(RuntimeError, "CUDA requested"):
                    diagnostic.main_cnn(args)
        self.assertEqual(events, ["is_initialized", "replay", "is_available"])

    def test_parser_does_not_probe_cuda_and_guard_precedes_replay(self) -> None:
        with tempfile.TemporaryDirectory(prefix="cnn-cuda-guard-", dir="/tmp") as text:
            root = Path(text)
            exorl = root / "exorl"
            bank = root / "bank.npz"
            exorl.mkdir()
            bank.touch()
            with mock.patch.object(
                    diagnostic.torch.cuda,
                    "is_available",
                    side_effect=AssertionError("premature CUDA availability probe"),
                ), mock.patch.object(diagnostic, "CNN_EXPERIMENTS", ()):
                args = diagnostic.build_parser().parse_args(
                    [
                        "--profile",
                        "cnn",
                        "--device",
                        "cuda:0",
                        "--output-dir",
                        str(root),
                        "--exorl-dir",
                        str(exorl),
                        "--index-bank",
                        str(bank),
                        "--replay-size",
                        "1024",
                        "--inference-size",
                        "1024",
                        "--allow-partial",
                    ]
                )
                with self.assertRaisesRegex(FileNotFoundError, "none of the requested"):
                    diagnostic.validate_cnn_args(args)
            self.assertEqual(args.device, "cuda:0")
            with mock.patch.object(
                    diagnostic, "validate_cnn_args", return_value=([], [], [])
                ), mock.patch.object(
                    diagnostic, "atomic_csv"
                ), mock.patch.object(
                    diagnostic.torch.cuda, "is_initialized", return_value=True
                ), mock.patch.object(
                    diagnostic, "load_or_build_replay_cache"
                ) as builder:
                with self.assertRaisesRegex(RuntimeError, "initialized before"):
                    diagnostic.main_cnn(args)
            builder.assert_not_called()

    def test_dino_default_is_resolved_before_validation(self) -> None:
        observed = []

        class Stop(Exception):
            pass

        def stop(args: argparse.Namespace) -> list[object]:
            observed.append(args.device)
            raise Stop

        with mock.patch.object(
            sys, "argv", ["diagnostic", "--profile", "dino"]
        ), mock.patch.object(
                diagnostic.torch.cuda, "is_available", return_value=True
            ) as available, mock.patch.object(
                diagnostic, "validate_args", side_effect=stop
            ):
            with self.assertRaises(Stop):
                diagnostic.main()
        self.assertEqual(observed, ["cuda:0"])
        available.assert_called_once_with()


class TestOptInFaultHandler(unittest.TestCase):
    def test_default_does_not_register_signal_handler(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(
            diagnostic.faulthandler, "register"
        ) as register:
            diagnostic._enable_opt_in_faulthandler()
        register.assert_not_called()

    def test_opt_in_routes_sigusr1_all_threads_to_current_stderr(self) -> None:
        current_stderr = mock.Mock(name="current_stderr")
        with mock.patch.dict(
                os.environ, {"CNN_ANALYSIS_FAULTHANDLER": "1"}, clear=True
            ), mock.patch.object(
                diagnostic.sys, "stderr", current_stderr
            ), mock.patch.object(
                diagnostic.faulthandler, "register"
            ) as register:
            diagnostic._enable_opt_in_faulthandler()
        register.assert_called_once_with(
            diagnostic.signal.SIGUSR1,
            file=current_stderr,
            all_threads=True,
        )


if __name__ == "__main__":
    unittest.main()
