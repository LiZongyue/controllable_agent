from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from url_benchmark import analyze_fb_pixel_b_mechanisms as diagnostic

# REDUNDANCY REVIEW: see test_analyze_fb_pixel_b_deferred_cuda.py -- this is one of five
# test files (801 lines here alone) covering private helpers of the offline, read-only
# analyze_fb_pixel_b_mechanisms.py checkpoint-analysis script. Not reachable from
# pretrain.py or any launch_*.sh; does not affect the training pipeline.


class TestCnnPixelEpisodePartCache(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(
            prefix="cnn-pixel-parts-", dir="/tmp"
        )
        self.root = Path(self.temporary.name)
        self.source = self.root / "episode_000037_1000.npz"
        self.source.write_bytes(b"strict synthetic source episode bytes")
        self.bank = self.root / "bank.npz"
        self.bank.write_bytes(b"strict synthetic index bank bytes")
        self.args = argparse.Namespace(
            output_dir=self.root,
            index_bank=self.bank,
            replay_seed=diagnostic.DEFAULT_SEED,
            replay_size=4,
            render_shape=(4, 5),
        )
        self.destinations = np.asarray([0, 3], dtype=np.int64)
        self.steps = np.asarray([2, 4], dtype=np.int64)
        self.index_bank_rows = np.asarray([7, 2], dtype=np.int32)
        self.render_context = diagnostic._cnn_pixel_render_context_contract(
            (4, 5), context_chunk_size=2
        )
        rng = np.random.RandomState(19)
        obs = rng.randint(0, 256, size=(2, 9, 4, 5), dtype=np.uint8)
        next_obs = np.empty_like(obs)
        next_obs[:, :6] = obs[:, 3:]
        next_obs[:, 6:] = rng.randint(
            0, 256, size=(2, 3, 4, 5), dtype=np.uint8
        )
        self.arrays = {
            "episode_id": np.asarray(37, dtype=np.int64),
            "destinations": self.destinations,
            "steps": self.steps,
            "index_bank_row": self.index_bank_rows,
            "obs": obs,
            "next_obs": next_obs,
            "action": rng.standard_normal((2, 6)).astype(np.float32),
            "discount": np.asarray([[0.99], [0.0]], dtype=np.float32),
            "physics": rng.standard_normal((2, 18)).astype(np.float64),
            "source_reward": rng.standard_normal((2, 1)).astype(np.float32),
        }
        self.contract = self._contract()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _contract(self, **overrides: object) -> dict:
        values = {
            "episode_id": 37,
            "episode_path": self.source,
            "source_episode_size": self.source.stat().st_size,
            "source_episode_sha256": diagnostic.sha256_file(self.source),
            "destinations": self.destinations,
            "steps": self.steps,
            "index_bank_rows": self.index_bank_rows,
            "pixel_render_context": self.render_context,
            "source_index_bank_sha256": diagnostic.sha256_file(self.bank),
        }
        values.update(overrides)
        return diagnostic._cnn_pixel_episode_part_contract(self.args, **values)

    def _load(self, path: Path, contract: dict | None = None) -> dict:
        return diagnostic._load_cnn_pixel_episode_part(
            path,
            self.contract if contract is None else contract,
            expected_destinations=self.destinations,
            expected_steps=self.steps,
            expected_index_bank_rows=self.index_bank_rows,
        )

    def test_atomic_roundtrip_has_strict_payload_and_audit_provenance(self) -> None:
        path = diagnostic._cnn_pixel_episode_part_path(self.root, 37)
        diagnostic._save_cnn_pixel_episode_part(path, self.arrays, self.contract)

        self.assertEqual(
            path,
            self.root / "fixed_rendered_replay_parts" / "episode_000037.npz",
        )
        self.assertTrue(path.is_file())
        self.assertEqual(list(path.parent.glob(f".{path.name}.tmp.*")), [])
        observed = self._load(path)
        for name, expected in self.arrays.items():
            self.assertTrue(np.array_equal(observed[name], expected), name)
        with np.load(path, allow_pickle=False) as payload:
            self.assertEqual(
                set(payload.files),
                set(diagnostic.CNN_PIXEL_EPISODE_PART_ARRAY_FIELDS) | {"metadata"},
            )
            metadata = json.loads(str(payload["metadata"].item()))
        self.assertEqual(
            metadata["part_schema_version"],
            diagnostic.CNN_PIXEL_EPISODE_PART_SCHEMA_VERSION,
        )
        self.assertEqual(metadata["pixel_render_context"], self.render_context)
        self.assertEqual(
            metadata["source_episode_sha256"], diagnostic.sha256_file(self.source)
        )
        self.assertEqual(
            metadata["content_sha256"],
            diagnostic._cnn_pixel_episode_part_arrays_digest(self.arrays),
        )
        cache_contract = diagnostic._cnn_pixel_episode_part_cache_contract(
            self.root, "pixels"
        )
        self.assertEqual(cache_contract["directory"], str(path.parent.resolve()))
        self.assertIn("retained", cache_contract["retention"])

    def test_selection_source_and_render_contract_mismatches_fail_closed(self) -> None:
        path = diagnostic._cnn_pixel_episode_part_path(self.root, 37)
        diagnostic._save_cnn_pixel_episode_part(path, self.arrays, self.contract)

        changed_steps = self.steps.copy()
        changed_steps[0] += 1
        changed_selection = self._contract(steps=changed_steps)
        changed_source = dict(self.contract)
        changed_source["source_episode_sha256"] = "0" * 64
        changed_render = json.loads(json.dumps(self.contract))
        changed_render["pixel_render_context"]["context_chunk_size_rows"] = 8
        for label, contract in (
            ("selection", changed_selection),
            ("source", changed_source),
            ("render", changed_render),
        ):
            with self.subTest(label=label):
                with self.assertRaisesRegex(ValueError, "contract mismatch"):
                    self._load(path, contract)

    def test_stream_validation_avoids_np_load_and_detects_contract_and_content(self) -> None:
        path = diagnostic._cnn_pixel_episode_part_path(self.root, 37)
        diagnostic._save_cnn_pixel_episode_part(path, self.arrays, self.contract)
        with mock.patch.object(
            diagnostic.np,
            "load",
            side_effect=AssertionError("Phase A must not call np.load on a part"),
        ) as numpy_load:
            diagnostic._stream_validate_cnn_pixel_episode_part(
                path,
                self.contract,
                expected_destinations=self.destinations,
                expected_steps=self.steps,
                expected_index_bank_rows=self.index_bank_rows,
            )
        numpy_load.assert_not_called()

        changed_contract = dict(self.contract)
        changed_contract["source_episode_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "contract mismatch"):
            diagnostic._stream_validate_cnn_pixel_episode_part(
                path,
                changed_contract,
                expected_destinations=self.destinations,
                expected_steps=self.steps,
                expected_index_bank_rows=self.index_bank_rows,
            )

        with np.load(path, allow_pickle=False) as payload:
            corrupt = {name: np.asarray(payload[name]) for name in payload.files}
        corrupt["action"] = corrupt["action"].copy()
        corrupt["action"][0, 0] += np.float32(1.0)
        diagnostic.atomic_npz(path, **corrupt)
        with self.assertRaisesRegex(ValueError, "content digest mismatch"):
            diagnostic._stream_validate_cnn_pixel_episode_part(
                path,
                self.contract,
                expected_destinations=self.destinations,
                expected_steps=self.steps,
                expected_index_bank_rows=self.index_bank_rows,
            )

    def test_scalar_digest_preamble_matches_hash_arrays_shape_normalization(self) -> None:
        scalar = np.asarray(37, dtype=np.int64)
        expected = hashlib.sha256()
        expected.update(b"episode_id")
        expected.update(b"int64")
        expected.update(np.asarray((1,), dtype=np.int64).tobytes())
        expected.update(scalar.tobytes())
        self.assertEqual(
            diagnostic.hash_arrays((("episode_id", scalar),)),
            expected.hexdigest(),
        )

        noncontiguous = np.arange(48, dtype=np.float32).reshape(6, 8)[:, ::2]
        self.assertFalse(noncontiguous.flags.c_contiguous)
        expected = hashlib.sha256()
        normalized = np.ascontiguousarray(noncontiguous)
        expected.update(b"noncontiguous")
        expected.update(str(normalized.dtype).encode())
        expected.update(np.asarray(normalized.shape, dtype=np.int64).tobytes())
        expected.update(normalized.tobytes())
        self.assertEqual(
            diagnostic.hash_arrays((("noncontiguous", noncontiguous),)),
            expected.hexdigest(),
        )

    def test_dtype_and_content_corruption_fail_before_resume(self) -> None:
        path = diagnostic._cnn_pixel_episode_part_path(self.root, 37)
        diagnostic._save_cnn_pixel_episode_part(path, self.arrays, self.contract)
        with np.load(path, allow_pickle=False) as payload:
            stored = {name: np.asarray(payload[name]) for name in payload.files}

        wrong_dtype = dict(stored)
        wrong_dtype["discount"] = wrong_dtype["discount"].astype(np.float64)
        diagnostic.atomic_npz(path, **wrong_dtype)
        with self.assertRaisesRegex(ValueError, "shape/dtype mismatch"):
            self._load(path)

        diagnostic._save_cnn_pixel_episode_part(
            path.with_name("episode_000038.npz"), self.arrays, self.contract
        )
        corrupt_path = path.with_name("episode_000038.npz")
        with np.load(corrupt_path, allow_pickle=False) as payload:
            corrupt = {name: np.asarray(payload[name]) for name in payload.files}
        corrupt["action"] = corrupt["action"].copy()
        corrupt["action"][0, 0] += np.float32(1.0)
        diagnostic.atomic_npz(corrupt_path, **corrupt)
        with self.assertRaisesRegex(ValueError, "content digest mismatch"):
            self._load(corrupt_path)

    def test_hidden_tmp_is_not_completion_and_dino_contract_is_unchanged(self) -> None:
        path = diagnostic._cnn_pixel_episode_part_path(self.root, 37)
        path.parent.mkdir(parents=True)
        hidden_tmp = path.with_name(f".{path.name}.tmp.12345")
        hidden_tmp.write_bytes(b"interrupted write")

        self.assertFalse(path.exists())
        with self.assertRaisesRegex(FileNotFoundError, "part is missing"):
            self._load(path)
        dino_root = self.root / "dino-output"
        self.assertIsNone(
            diagnostic._cnn_pixel_episode_part_cache_contract(dino_root, "dino")
        )
        self.assertFalse(
            (dino_root / diagnostic.CNN_PIXEL_EPISODE_PARTS_DIRNAME).exists()
        )

    def test_build_replay_resumes_part_without_renderer_and_preserves_assembly(self) -> None:
        row_count = diagnostic.TRAIN_BATCH
        episode_length = 8
        np.savez(
            self.bank,
            episode_id=np.full(row_count, 37, dtype=np.int32),
            step_in_episode=np.full(row_count, 2, dtype=np.int32),
            metadata=np.asarray(json.dumps({"synthetic": True}, sort_keys=True)),
        )
        rng = np.random.RandomState(23)
        np.savez(
            self.source,
            dino_emb=rng.standard_normal(
                (episode_length, diagnostic.EMBED_DIM)
            ).astype(np.float32),
            action=rng.standard_normal((episode_length, 6)).astype(np.float32),
            discount=np.ones((episode_length, 1), dtype=np.float32),
            physics=rng.standard_normal((episode_length, 18)).astype(np.float64),
            reward=rng.standard_normal((episode_length, 1)).astype(np.float32),
            dino_token=np.asarray("cls"),
            dino_model=np.asarray("facebook/dinov2-base"),
            dino_processor=np.asarray("facebook/dinov2-base"),
            image_size=np.asarray(224, dtype=np.int64),
            camera_id=np.asarray(0, dtype=np.int64),
        )
        args = argparse.Namespace(
            output_dir=self.root,
            exorl_dir=self.root,
            index_bank=self.bank,
            replay_size=row_count,
            replay_seed=diagnostic.DEFAULT_SEED,
            z_seed=diagnostic.DEFAULT_Z_SEED,
            probe_size=1,
            inference_size=1,
            render_shape=(4, 5),
            resume=False,
        )
        current = rng.randint(
            0, 256, size=(row_count, 9, 4, 5), dtype=np.uint8
        )
        following = np.empty_like(current)
        following[:, :6] = current[:, 3:]
        following[:, 6:] = rng.randint(
            0, 256, size=(row_count, 3, 4, 5), dtype=np.uint8
        )
        original_assemble = diagnostic._assemble_cnn_pixel_episode_parts

        def checked_assemble(namespace: argparse.Namespace, plans: list) -> tuple:
            self.assertTrue(plans)
            self.assertTrue(all(plan.part_path.is_file() for plan in plans))
            return original_assemble(namespace, plans)

        with mock.patch.object(
            diagnostic,
            "_render_pixel_stacks_context_chunked",
            return_value=(current, following),
        ) as renderer, mock.patch.object(
            diagnostic,
            "_assemble_cnn_pixel_episode_parts",
            side_effect=checked_assemble,
        ) as assembler:
            first = diagnostic.build_replay(args, obs_type="pixels")
        renderer.assert_called_once()
        assembler.assert_called_once()
        self.assertTrue(np.array_equal(first.obs, current))
        self.assertTrue(np.array_equal(first.next_obs, following))

        args.resume = True
        with mock.patch.object(
            diagnostic,
            "_render_pixel_stacks_context_chunked",
            side_effect=AssertionError("renderer must not run on validated resume"),
        ) as renderer:
            resumed = diagnostic.build_replay(args, obs_type="pixels")
        renderer.assert_not_called()
        for name in (
            "obs",
            "next_obs",
            "action",
            "discount",
            "physics",
            "source_reward",
            "random_z",
            "episode_id",
            "step_in_episode",
            "index_bank_row",
        ):
            self.assertTrue(np.array_equal(getattr(resumed, name), getattr(first, name)), name)
        self.assertEqual(
            resumed.metadata["content_sha256"], first.metadata["content_sha256"]
        )

        cache = self.root / "fixed_rendered_replay.npz"
        diagnostic.save_replay_cache(cache, first)
        diagnostic._stream_validate_replay_cache_against_replay(cache, first)
        original_numpy_load = diagnostic.np.load

        def reject_compressed_cache_load(path: object, *load_args, **load_kwargs):
            if Path(path) == cache:
                raise AssertionError(
                    "CNN compressed replay must be stream-validated, not np.load'ed"
                )
            return original_numpy_load(path, *load_args, **load_kwargs)

        with mock.patch.object(
            diagnostic.np, "load", side_effect=reject_compressed_cache_load
        ), mock.patch.object(
            diagnostic,
            "load_replay_cache",
            side_effect=AssertionError(
                "CNN load_or_build must retain global mmap arrays"
            ),
        ) as legacy_loader:
            cached, cached_path = diagnostic.load_or_build_replay_cache(
                args, "pixels"
            )
        legacy_loader.assert_not_called()
        self.assertEqual(cached_path, cache)
        self.assertIsInstance(cached.obs, np.memmap)
        self.assertTrue(np.array_equal(cached.obs, first.obs))

        del cached
        gc.collect()
        with original_numpy_load(cache, allow_pickle=False) as payload:
            corrupt = {
                name: np.asarray(payload[name]) for name in payload.files
            }
        corrupt["obs"] = corrupt["obs"].copy()
        corrupt["obs"][0, 0, 0, 0] ^= np.uint8(1)
        diagnostic.atomic_npz(cache, **corrupt)
        with self.assertRaisesRegex(ValueError, "field content mismatch"):
            diagnostic.load_or_build_replay_cache(args, "pixels")


class TestCnnPixelGlobalRowStreamingAssembly(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(
            prefix="cnn-pixel-global-stream-", dir="/tmp"
        )
        self.root = Path(self.temporary.name)
        self.bank = self.root / "bank.npz"
        self.bank.write_bytes(b"synthetic global-row streaming bank")
        self.args = argparse.Namespace(
            output_dir=self.root,
            index_bank=self.bank,
            replay_seed=diagnostic.DEFAULT_SEED,
            replay_size=12,
            render_shape=(8, 8),
        )
        self.render_context = diagnostic._cnn_pixel_render_context_contract(
            self.args.render_shape, context_chunk_size=2
        )
        self.plans = []
        self.part_arrays = []
        episode_ids = (11, 22, 33, 44)
        destinations_by_episode = (
            np.asarray([0, 4, 8], dtype=np.int64),
            np.asarray([1, 5, 9], dtype=np.int64),
            np.asarray([2, 6, 10], dtype=np.int64),
            np.asarray([3, 7, 11], dtype=np.int64),
        )
        for number, (episode_id, destinations) in enumerate(
            zip(episode_ids, destinations_by_episode), start=1
        ):
            source = self.root / f"episode_{episode_id:06d}_1000.npz"
            source.write_bytes(f"synthetic episode {episode_id}".encode())
            steps = np.asarray([2, 5, 7], dtype=np.int64) + number
            index_bank_rows = np.asarray(
                [100 + number, 50 + number, 10 + number], dtype=np.int32
            )
            contract = diagnostic._cnn_pixel_episode_part_contract(
                self.args,
                episode_id=episode_id,
                episode_path=source,
                source_episode_size=source.stat().st_size,
                source_episode_sha256=diagnostic.sha256_file(source),
                destinations=destinations,
                steps=steps,
                index_bank_rows=index_bank_rows,
                pixel_render_context=self.render_context,
                source_index_bank_sha256=diagnostic.sha256_file(self.bank),
            )
            rng = np.random.RandomState(500 + episode_id)
            obs = rng.randint(
                0,
                256,
                size=(3, 9, *self.args.render_shape),
                dtype=np.uint8,
            )
            next_obs = np.empty_like(obs)
            next_obs[:, :6] = obs[:, 3:]
            next_obs[:, 6:] = rng.randint(
                0,
                256,
                size=(3, 3, *self.args.render_shape),
                dtype=np.uint8,
            )
            arrays = {
                "episode_id": np.asarray(episode_id, dtype=np.int64),
                "destinations": destinations,
                "steps": steps,
                "index_bank_row": index_bank_rows,
                "obs": obs,
                "next_obs": next_obs,
                "action": rng.standard_normal((3, 6)).astype(np.float32),
                "discount": rng.uniform(0.0, 1.0, size=(3, 1)).astype(np.float32),
                "physics": rng.standard_normal((3, 18)).astype(np.float64),
                "source_reward": rng.standard_normal((3, 1)).astype(np.float32),
            }
            part_path = diagnostic._cnn_pixel_episode_part_path(
                self.root, episode_id
            )
            diagnostic._save_cnn_pixel_episode_part(part_path, arrays, contract)
            self.plans.append(
                diagnostic.CnnPixelEpisodePartPlan(
                    number=number,
                    total=len(episode_ids),
                    episode_id=episode_id,
                    source_path=source,
                    part_path=part_path,
                    destinations=destinations,
                    steps=steps,
                    index_bank_rows=index_bank_rows,
                    contract=contract,
                )
            )
            self.part_arrays.append(arrays)

        self.reference = {
            "obs": np.empty(
                (self.args.replay_size, 9, *self.args.render_shape), dtype=np.uint8
            ),
            "next_obs": np.empty(
                (self.args.replay_size, 9, *self.args.render_shape), dtype=np.uint8
            ),
            "action": np.empty((self.args.replay_size, 6), dtype=np.float32),
            "discount": np.empty((self.args.replay_size, 1), dtype=np.float32),
            "physics": np.empty((self.args.replay_size, 18), dtype=np.float64),
            "source_reward": np.empty(
                (self.args.replay_size, 1), dtype=np.float32
            ),
        }
        for plan, arrays in zip(self.plans, self.part_arrays):
            for field in self.reference:
                self.reference[field][plan.destinations] = arrays[field]

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _assemble(self) -> dict:
        values = diagnostic._assemble_cnn_pixel_episode_parts(
            self.args, self.plans
        )
        return dict(
            zip(
                (
                    "obs",
                    "next_obs",
                    "action",
                    "discount",
                    "physics",
                    "source_reward",
                ),
                values,
            )
        )

    def test_four_way_interleave_matches_episode_scatter_bytes_and_hash(self) -> None:
        observed = self._assemble()
        for field, expected in self.reference.items():
            self.assertEqual(observed[field].tobytes(), expected.tobytes(), field)
        fields = tuple(self.reference)
        self.assertEqual(
            diagnostic.hash_arrays((field, observed[field]) for field in fields),
            diagnostic.hash_arrays(
                (field, self.reference[field]) for field in fields
            ),
        )

    def test_short_zip_member_reads_are_accumulated(self) -> None:
        original_read = diagnostic.zipfile.ZipExtFile.read

        def short_read(stream: object, size: int = -1) -> bytes:
            if size > 256:
                size = 37
            return original_read(stream, size)

        with mock.patch.object(
            diagnostic.zipfile.ZipExtFile, "read", new=short_read
        ):
            observed = self._assemble()
        for field, expected in self.reference.items():
            self.assertTrue(np.array_equal(observed[field], expected), field)

    def test_duplicate_gap_and_nonincreasing_destinations_fail_closed(self) -> None:
        duplicate = list(self.plans)
        duplicate[1] = dataclasses.replace(
            duplicate[1],
            destinations=np.asarray([0, 5, 9], dtype=np.int64),
        )
        gap = list(self.plans)
        gap[3] = dataclasses.replace(
            gap[3],
            destinations=gap[3].destinations[:-1],
            steps=gap[3].steps[:-1],
            index_bank_rows=gap[3].index_bank_rows[:-1],
        )
        nonincreasing = list(self.plans)
        nonincreasing[0] = dataclasses.replace(
            nonincreasing[0], destinations=nonincreasing[0].destinations[::-1]
        )
        for label, plans, pattern in (
            ("duplicate", duplicate, "duplicate global rows"),
            ("gap", gap, "do not partition"),
            ("order", nonincreasing, "not strictly increasing"),
        ):
            with self.subTest(label=label):
                with self.assertRaisesRegex(ValueError, pattern):
                    diagnostic._assemble_cnn_pixel_episode_parts(self.args, plans)

    def test_truncated_archive_fails_and_closes_other_archives(self) -> None:
        path = self.plans[0].part_path
        payload = path.read_bytes()
        path.write_bytes(payload[:-32])
        descriptors_before = len(diagnostic.os.listdir("/proc/self/fd"))
        with self.assertRaisesRegex(ValueError, "ZIP|truncated|invalid"):
            self._assemble()
        self.assertEqual(
            len(diagnostic.os.listdir("/proc/self/fd")), descriptors_before
        )
        path.write_bytes(payload)
        # A second call proves exception unwinding closed every archive/member.
        observed = self._assemble()
        self.assertTrue(np.array_equal(observed["obs"], self.reference["obs"]))

    def test_bad_member_crc_fails_at_stream_eof_and_closes_archives(self) -> None:
        path = self.plans[0].part_path
        original_archive = path.read_bytes()
        with diagnostic.zipfile.ZipFile(path, "r") as source:
            members = [(info.filename, source.read(info)) for info in source.infolist()]
        with diagnostic.zipfile.ZipFile(
            path, "w", compression=diagnostic.zipfile.ZIP_STORED
        ) as destination:
            for name, data in members:
                destination.writestr(name, data)

        with diagnostic.zipfile.ZipFile(path, "r") as archive:
            info = archive.getinfo("action.npy")
            header_offset = info.header_offset
        corrupted = bytearray(path.read_bytes())
        name_size = int.from_bytes(
            corrupted[header_offset + 26 : header_offset + 28], "little"
        )
        extra_size = int.from_bytes(
            corrupted[header_offset + 28 : header_offset + 30], "little"
        )
        data_offset = header_offset + 30 + name_size + extra_size
        corrupted[data_offset + info.file_size - 1] ^= 1
        path.write_bytes(corrupted)

        descriptors_before = len(diagnostic.os.listdir("/proc/self/fd"))
        try:
            with self.assertRaisesRegex(ValueError, "CRC|failed reading|header"):
                self._assemble()
            self.assertEqual(
                len(diagnostic.os.listdir("/proc/self/fd")), descriptors_before
            )
        finally:
            path.write_bytes(original_archive)

    def test_only_one_part_archive_is_retained_at_a_time(self) -> None:
        original_zip = diagnostic.zipfile.ZipFile

        class TrackingZipFile(original_zip):
            active = 0
            peak = 0

            def __enter__(self):
                value = super().__enter__()
                type(self).active += 1
                type(self).peak = max(type(self).peak, type(self).active)
                return value

            def __exit__(self, *args):
                try:
                    return super().__exit__(*args)
                finally:
                    type(self).active -= 1

        with mock.patch.object(
            diagnostic.zipfile, "ZipFile", TrackingZipFile
        ):
            observed = self._assemble()
        self.assertEqual(TrackingZipFile.active, 0)
        self.assertEqual(TrackingZipFile.peak, 1)
        self.assertTrue(np.array_equal(observed["obs"], self.reference["obs"]))

    def test_file_backed_manifest_resume_and_content_digest(self) -> None:
        observed = self._assemble()
        directory = (
            self.root / diagnostic.CNN_PIXEL_GLOBAL_ARRAYS_DIRNAME
        )
        manifest_path = directory / diagnostic.CNN_PIXEL_GLOBAL_ARRAYS_MANIFEST
        self.assertTrue(manifest_path.is_file())
        self.assertEqual(
            list(self.root.glob(f".{directory.name}.tmp.*")), []
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for field, value in observed.items():
            self.assertIsInstance(value, np.memmap)
            self.assertEqual(value.mode, "c")
            self.assertTrue(value.flags.writeable)
            self.assertEqual(
                manifest["field_hashes"][field],
                diagnostic.hash_arrays(((field, value),)),
            )

        original_open = diagnostic.zipfile.ZipFile.open

        def reject_large_part_member(archive: object, name: str, *args, **kwargs):
            member_field = name[:-4] if name.endswith(".npy") else name
            if member_field in self.reference:
                raise AssertionError("resume must not reopen large part members")
            return original_open(archive, name, *args, **kwargs)

        with mock.patch.object(
            diagnostic.zipfile.ZipFile,
            "open",
            new=reject_large_part_member,
        ):
            resumed = self._assemble()
        for field, expected in self.reference.items():
            self.assertTrue(np.array_equal(resumed[field], expected), field)

        del observed, resumed
        gc.collect()
        obs_path = directory / "obs.npy"
        with obs_path.open("r+b") as stream:
            shape, _, dtype = diagnostic._read_cnn_pixel_part_npy_header(
                stream, obs_path, "obs"
            )
            self.assertEqual(shape, self.reference["obs"].shape)
            self.assertEqual(dtype, self.reference["obs"].dtype)
            offset = stream.tell()
            original = stream.read(1)
            stream.seek(offset)
            stream.write(bytes((original[0] ^ 1,)))
        with self.assertRaisesRegex(ValueError, "content digest mismatch"):
            self._assemble()

    def test_global_arrays_use_private_copy_on_write_mmaps(self) -> None:
        observed = self._assemble()
        obs = observed["obs"]
        self.assertIsInstance(obs, np.memmap)
        self.assertEqual(obs.mode, "c")
        self.assertTrue(obs.flags.writeable)

        directory = self.root / diagnostic.CNN_PIXEL_GLOBAL_ARRAYS_DIRNAME
        obs_path = directory / "obs.npy"
        original = int(obs[0, 0, 0, 0])
        obs[0, 0, 0, 0] ^= np.uint8(1)
        obs.flush()
        self.assertNotEqual(int(obs[0, 0, 0, 0]), original)

        durable = np.load(obs_path, mmap_mode="r", allow_pickle=False)
        try:
            self.assertEqual(int(durable[0, 0, 0, 0]), original)
        finally:
            del durable

    def test_pwrite_short_writes_and_failure_cleanup(self) -> None:
        original_pwrite = diagnostic.os.pwrite

        def short_pwrite(fd: int, data: object, offset: int) -> int:
            return original_pwrite(fd, memoryview(data)[:37], offset)

        with mock.patch.object(
            diagnostic.os, "pwrite", side_effect=short_pwrite
        ):
            observed = self._assemble()
        for field, expected in self.reference.items():
            self.assertTrue(np.array_equal(observed[field], expected), field)

        del observed
        gc.collect()
        directory = self.root / diagnostic.CNN_PIXEL_GLOBAL_ARRAYS_DIRNAME
        for child in directory.iterdir():
            child.unlink()
        directory.rmdir()

        calls = 0

        def failed_pwrite(fd: int, data: object, offset: int) -> int:
            nonlocal calls
            calls += 1
            if calls == 4:
                raise OSError("synthetic pwrite failure")
            return original_pwrite(fd, data, offset)

        with mock.patch.object(
            diagnostic.os, "pwrite", side_effect=failed_pwrite
        ):
            with self.assertRaisesRegex(ValueError, "failed writing"):
                self._assemble()
        self.assertFalse(directory.exists())
        self.assertEqual(
            list(self.root.glob(f".{directory.name}.tmp.*")), []
        )
        recovered = self._assemble()
        self.assertTrue(np.array_equal(recovered["obs"], self.reference["obs"]))

    def test_nonfinite_float_and_fortran_header_fail_closed(self) -> None:
        path = self.plans[0].part_path
        with np.load(path, allow_pickle=False) as payload:
            stored = {field: np.asarray(payload[field]) for field in payload.files}
        nonfinite = dict(stored)
        nonfinite["physics"] = nonfinite["physics"].copy()
        nonfinite["physics"][0, 0] = np.nan
        diagnostic.atomic_npz(path, **nonfinite)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            self._assemble()

        diagnostic.atomic_npz(path, **stored)
        changed_content = dict(stored)
        changed_content["action"] = changed_content["action"].copy()
        changed_content["action"][0, 0] += np.float32(1.0)
        diagnostic.atomic_npz(path, **changed_content)
        with self.assertRaisesRegex(ValueError, "content digest mismatch"):
            self._assemble()

        diagnostic.atomic_npz(path, **stored)
        fortran = dict(stored)
        fortran["action"] = np.asfortranarray(fortran["action"])
        diagnostic.atomic_npz(path, **fortran)
        with self.assertRaisesRegex(ValueError, "native C-order header mismatch"):
            self._assemble()

        diagnostic.atomic_npz(path, **stored)
        nonnative = dict(stored)
        opposite = ">f4" if diagnostic.sys.byteorder == "little" else "<f4"
        nonnative["action"] = nonnative["action"].astype(opposite)
        diagnostic.atomic_npz(path, **nonnative)
        with self.assertRaisesRegex(ValueError, "native C-order header mismatch"):
            self._assemble()


if __name__ == "__main__":
    unittest.main()
