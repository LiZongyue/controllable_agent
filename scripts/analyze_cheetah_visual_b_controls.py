#!/usr/bin/env python3
"""Read-only positive controls on the exact historical fixed CLS banks.

Run `main` before `supplement`. All outputs are new artifacts, never checkpoints.
Matrices/statistics/EVD are float64; historical neural inference remains float32.
"""
import argparse
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/cheetah_controls_mpl")
import numpy as np
import torch
from url_benchmark import analyze_fb_checkpoint_geometry as geometry
from url_benchmark import analyze_fb_pixel_b_mechanisms as mechanisms

OUTPUT = ROOT / "analysis_outputs/cheetah_visual_b_evd_controls_20260914"
GEOMETRY_BANK = ROOT / "analysis_outputs/cheetah_fb_dino_cls3_pixel_b_checkpoint_geometry_20260827/fixed_replay_batch.npz"
TASKS = ("cheetah_walk", "cheetah_run", "cheetah_walk_backward", "cheetah_run_backward")


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    geometry.atomic_json(path, value)


def records(output):
    data = json.loads((output / "checkpoints/checkpoint_manifest.json").read_text())
    if isinstance(data, dict):
        data = data.get("checkpoints", data.get("records"))
    return [x for x in data if x["compatible"]]


def load_agent(record):
    # Paths were discovered and checked against serialized state by the inventory.
    path = Path(record["path"])
    if geometry.sha256_file(path) != record["checkpoint_sha256"]:
        raise ValueError(f"checkpoint changed after inventory: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if int(payload["global_step"]) != int(record["global_step"]):
        raise ValueError("global_step mismatch")
    agent = payload["agent"]
    mechanisms.install_optimizer_guards(agent)
    def forbidden(_self, *args, **kwargs):
        raise RuntimeError("Agent update is forbidden during read-only diagnostics")
    for name in ("update", "update_fb", "update_actor", "update_critic"):
        if hasattr(agent, name):
            setattr(agent, name, types.MethodType(forbidden, agent))
    mechanisms.prepare_agent(agent, "cpu")
    for value in vars(agent).values():
        if isinstance(value, torch.nn.Module):
            value.requires_grad_(False)
    return agent


def encode_b(agent, obs, chunk=256):
    result = np.empty((len(obs), 50), np.float32)
    with torch.inference_mode():
        for start in range(0, len(obs), chunk):
            end = min(start + chunk, len(obs))
            inputs = torch.from_numpy(obs[start:end])
            result[start:end] = agent.backward_net(agent.backward_aug_and_encode(inputs)).numpy()
    if not np.isfinite(result).all():
        raise FloatingPointError("Non-finite B")
    return result


def weak_masks(eigenvalues):
    # Explicit names distinguish two genuinely different historical protocols.
    masks = {"bottom5": np.arange(50) < 5, "bottom10": np.arange(50) < 10}
    for alpha in (0.03, 0.1, 0.3):
        masks[f"geometry_mean_le_{alpha:g}"] = eigenvalues <= alpha * eigenvalues.mean()
        masks[f"mechanisms_max_lt_{alpha:g}"] = eigenvalues < alpha * eigenvalues[-1]
    return masks


def evd(x, centered=False):
    x = np.asarray(x, dtype=np.float64)
    mean = x.mean(axis=0)
    values = x - mean if centered else x
    matrix = values.T @ values / len(values)
    matrix = (matrix + matrix.T) * 0.5
    raw, vectors = np.linalg.eigh(matrix)
    floor = max(x.shape) * np.finfo(np.float64).eps * max(float(raw[-1]), 1.0)
    if raw[0] < -floor:
        raise FloatingPointError(f"Significant negative eigenvalue {raw[0]} < {-floor}")
    eigenvalues = np.maximum(raw, 0.0)
    trace = eigenvalues.sum()
    if trace <= 0:
        raise FloatingPointError("Zero covariance trace")
    normalized = eigenvalues / trace
    cumulative = np.cumsum(normalized[::-1])
    if not np.allclose(matrix, (vectors * raw) @ vectors.T, rtol=1e-10, atol=1e-10):
        raise AssertionError("EVD reconstruction")
    # Thin SVD has the same nonzero spectrum as sample Gram; no NxN allocation.
    svd_eigenvalues = np.linalg.svd(values, compute_uv=False)[::-1] ** 2 / len(values)
    svd_error = float(np.max(np.abs(eigenvalues - svd_eigenvalues)))
    if not np.allclose(eigenvalues, svd_eigenvalues, rtol=2e-7, atol=floor):
        raise AssertionError("EVD / thin SVD disagreement")
    summary = {
        "matrix": "centered" if centered else "uncentered", "n": len(x),
        "trace": float(trace), "lambda_min": float(eigenvalues[0]),
        "lambda_max": float(eigenvalues[-1]), "raw_lambda_min": float(raw[0]),
        "negative_clipped_count": int((raw < 0).sum()),
        "rank_tolerance": floor, "condition_floor": floor,
        "numerical_rank": int((eigenvalues > floor).sum()),
        "condition_raw": float(eigenvalues[-1] / eigenvalues[0]) if eigenvalues[0] > 0 else "inf",
        "condition_effective": float(eigenvalues[-1] / max(eigenvalues[0], floor)),
        "participation_ratio": float(trace ** 2 / (eigenvalues ** 2).sum()),
        "mean_energy_fraction": float(mean @ mean / np.mean(np.sum(x ** 2, axis=1))),
        "b_norm_min": float(np.linalg.norm(x, axis=1).min()),
        "b_norm_max": float(np.linalg.norm(x, axis=1).max()),
        "svd_max_abs_error": svd_error,
    }
    for k in (1, 2, 5, 10):
        summary[f"top{k}_fraction"] = float(cumulative[k - 1])
    for p in (90, 95, 99):
        summary[f"dims_{p}"] = int(np.searchsorted(cumulative, p / 100) + 1)
    for name, mask in weak_masks(eigenvalues).items():
        summary[f"{name}_dimension"] = int(mask.sum())
        summary[f"{name}_trace_fraction"] = float(normalized[mask].sum())
    arrays = dict(matrix=matrix, mean=mean, eigenvalues_raw_ascending=raw,
                  eigenvalues_ascending=eigenvalues, eigenvectors_columns_ascending=vectors,
                  descending_indices=np.arange(49, -1, -1),
                  trace_normalized_ascending=normalized, cumulative_descending=cumulative,
                  thin_svd_eigenvalues_ascending=svd_eigenvalues)
    return summary, arrays


def save_spectra(path, b, metadata):
    rows, arrays = [], {"B": b, "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True))}
    for centered in (False, True):
        summary, result = evd(b, centered)
        rows.append(summary)
        values = b.astype(np.float64) - result['mean'] if centered else b.astype(np.float64)
        result['B_eigen_coordinates'] = values @ result['eigenvectors_columns_ascending']
        arrays.update({summary["matrix"] + "_" + k: v for k, v in result.items()})
    geometry.atomic_npz(path, **arrays)
    return rows


def primary(output):
    batch = geometry.load_fixed_batch(GEOMETRY_BANK)
    all_records = records(output)
    seen, rows, mappings = {}, [], []
    for record in all_records:
        digest = record["b_online_sha256"]
        path = output / "primary" / (digest + ".npz")
        if digest not in seen:
            print(f"primary B {record['task']} frame={record['frame']} {path.name[:12]}", flush=True)
            agent = load_agent(record)
            b = encode_b(agent, batch.next_obs)
            seen[digest] = save_spectra(path, b, {
                "b_online_sha256": digest, "bank_sha256": batch.metadata["batch_sha256"],
                "bank": str(GEOMETRY_BANK), "inference_dtype": "float32", "statistics_dtype": "float64",
                "n": 1024, "ordering": "exact saved geometry bank",
            })
            del agent
            gc.collect()
        for row in seen[digest]:
            rows.append({"run_id": record["run_id"], "task": record["task"],
                         "frame": record["frame"], "global_step": record["global_step"],
                         "checkpoint_path": record["path"], "checkpoint_sha256": record["checkpoint_sha256"],
                         "model_sha256": record["model_sha256"], "b_online_sha256": digest,
                         "bundle": str(path.relative_to(output)), **row})
        mappings.append({**record, "primary_bundle": str(path.relative_to(output))})
        write_csv(output / "primary/spectrum_summary.csv", rows)
    write_json(output / "primary/checkpoint_mapping.json", {"records": mappings, "unique_b_count": len(seen)})
    spectral_rows = []
    for digest in seen:
        with np.load(output / "primary" / (digest + ".npz")) as bundle:
            for kind in ("uncentered", "centered"):
                for rank in range(50):
                    index = 49 - rank
                    spectral_rows.append({"b_online_sha256": digest, "matrix": kind,
                        "descending_rank": rank + 1,
                        "eigenvalue": bundle[kind + "_eigenvalues_ascending"][index],
                        "trace_fraction": bundle[kind + "_trace_normalized_ascending"][index],
                        "cumulative_fraction": bundle[kind + "_cumulative_descending"][rank]})
    write_csv(output / "primary/full_spectra_long.csv", spectral_rows)
    drift(output, mappings)
    print(f"PRIMARY COMPLETE: {len(all_records)} files, {len(seen)} unique B states", flush=True)


def drift(output, mappings):
    rows = []
    for run_id in sorted(set(x["run_id"] for x in mappings)):
        previous = None
        seen = set()
        for record in sorted((x for x in mappings if x["run_id"] == run_id), key=lambda x: x["frame"]):
            key = (record["frame"], record["b_online_sha256"])
            if key in seen:
                continue
            seen.add(key)
            with np.load(output / record["primary_bundle"]) as f:
                current = {k: f[k] for k in f.files}
            if previous is not None:
                previous_record, old = previous
                for kind in ("uncentered", "centered"):
                    b0, b1 = old["B"].astype(np.float64), current["B"].astype(np.float64)
                    if kind == "centered":
                        b0, b1 = b0 - b0.mean(0), b1 - b1.mean(0)
                    u0 = old[kind + "_eigenvectors_columns_ascending"]
                    u1 = current[kind + "_eigenvectors_columns_ascending"]
                    lam = current[kind + "_eigenvalues_ascending"]
                    values = geometry.drift_metrics(b0, u0, b1, lam, u1, (1, 2, 5, 10, 20))
                    for k in (1, 2, 5, 10, 20):
                        values[f"previous_top{k}_relative_eigengap"] = geometry.relative_top_eigengap(old[kind + "_eigenvalues_ascending"], k)
                    rows.append({"run_id": run_id, "task": record["task"], "matrix": kind,
                                 "previous_frame": previous_record["frame"], "frame": record["frame"], **values})
                    left, singular, right = np.linalg.svd(b1.T @ b0, full_matrices=False)
                    geometry.atomic_npz(output / "drift" / f"{record['task']}_{previous_record['frame']}_{record['frame']}_{kind}.npz",
                                        rotation_current_to_previous=left @ right, alignment_singular_values=singular)
            previous = (record, current)
    write_csv(output / "drift/consecutive.csv", rows)


def distribution(x):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    return {"n": len(x), "min": float(x.min()), "max": float(x.max()), "mean": float(x.mean()),
            "std": float(x.std()), **{f"p{p:g}": float(np.percentile(x, p)) for p in (1, 5, 25, 50, 75, 95, 99)},
            "positive_count": int((x > 0).sum()), "negative_count": int((x < 0).sum()),
            "zero_count": int((x == 0).sum())}


def reward_bank(output, replay):
    path = output / "supplement/rewards_and_velocity.npz"
    rewards = {}
    velocity = np.empty(len(replay.physics), np.float64)
    for task in TASKS:
        print(f"reward labels {task}", flush=True)
        rewarder = mechanisms.DmcReward(task)
        values = np.empty((len(replay.physics), 1), np.float32)
        for i, state in enumerate(replay.physics):
            values[i, 0] = rewarder.from_physics(state)
            if task == TASKS[0]:
                velocity[i] = rewarder._env.physics.speed()
        rewards[task] = values
    geometry.atomic_npz(path, velocity=velocity, **rewards)
    rows = []
    for selection, n in (("inference", 5120), ("probe", 1024), ("full_proxy", 20480)):
        rows.append({"subset": selection, "variable": "signed_horizontal_velocity", **distribution(velocity[:n]),
            **{f"velocity_{op}{threshold}_count": int(((velocity[:n] >= threshold) if op == 'ge' else (velocity[:n] <= threshold)).sum())
               for op, threshold in (("ge", 2), ("ge", 10), ("le", -2), ("le", -10))}})
        for task, values in rewards.items():
            x = values[:n, 0].astype(np.float64)
            rows.append({"subset": selection, "variable": task, **distribution(x),
                         "reward_gt_001_count": int((x > .01).sum()),
                         "reward_gt_01_count": int((x > .1).sum()),
                         "reward_ge_05_count": int((x >= .5).sum()),
                         "reward_ge_099_count": int((x >= .99).sum()),
                         "reward_weight_ess": float(x.sum() ** 2 / (x @ x)) if x @ x else 0})
    write_csv(output / "supplement/coverage.csv", rows)
    old = np.load(ROOT / "analysis_outputs/pixel_b_cheetah_mechanisms_20260827/fixed_replay_manifest.npz")
    checks = {task: bool(np.array_equal(rewards[task], old[f"proxy_reward_{task}"])) for task in TASKS[:2]}
    write_json(output / "supplement/reward_validation.json", {
        "legacy_labels_bitwise_equal": checks,
        "source_run_reward_correlation": float(np.corrcoef(replay.source_reward[:, 0], rewards['cheetah_run'][:, 0])[0, 1]),
        "reward_code_sha256": geometry.sha256_file(ROOT / 'url_benchmark/goals.py'),
        "custom_cheetah_sha256": geometry.sha256_file(ROOT / 'url_benchmark/custom_dmc_tasks/cheetah.py'),
        "time_semantics": "single-state ExORL proxy reward; not native two-substep accumulated reward"})
    if not all(checks.values()):
        raise ValueError("reward labels differ from exact old bank")
    return rewards


def projections(agent, obs, action, z, spectrum, prefix):
    arrays, rows = {}, []
    z = np.asarray(z, np.float32)
    f1, f2, _ = mechanisms.encode_forward(agent, obs, z, action, "cpu", 256, "replay")
    u = spectrum["uncentered_eigenvectors_columns_ascending"]
    lam = spectrum["uncentered_eigenvalues_ascending"]
    matrix = spectrum["uncentered_matrix"]
    z64 = z.astype(np.float64)
    z_coords = z64 @ u
    arrays["z"] = z
    qs, qmods = [], []
    for head, f in (("f1", f1), ("f2", f2)):
        f = f.astype(np.float64)
        coords = f @ u
        c_contrib = coords ** 2 * lam
        q_contrib = coords * z_coords
        c_direct = np.einsum("ni,ij,nj->n", f, matrix, f)
        q_direct = (f * z64).sum(1)
        if not np.allclose(c_direct, c_contrib.sum(1), rtol=1e-9, atol=1e-8):
            raise AssertionError("F^T C_B F mode identity failed")
        if not np.allclose(q_direct, q_contrib.sum(1), rtol=1e-9, atol=1e-8):
            raise AssertionError("F^T z mode identity failed")
        arrays.update({head + "_" + name: value for name, value in dict(
            F=f, eigen_coordinates=coords, covariance_modal_contribution=c_contrib,
            q_modal_contribution=q_contrib, covariance_energy=c_direct, Q=q_direct).items()})
        base = {"mode": prefix, "head": head, "action_mode": "fixed_replay", "n": len(f),
                "f_norm_rms": float(np.sqrt(np.mean(np.sum(f ** 2, 1)))),
                "covariance_energy_mean": float(c_direct.mean()), "q_mean": float(q_direct.mean()),
                "q_rms": float(np.sqrt(np.mean(q_direct ** 2))),
                "q_identity_abs_error": float(np.max(np.abs(q_direct - q_contrib.sum(1)))),
                "covariance_identity_abs_error": float(np.max(np.abs(c_direct - c_contrib.sum(1))))}
        for name, mask in weak_masks(lam).items():
            base[name + "_f_energy_fraction"] = float((coords[:, mask] ** 2).sum() / (coords ** 2).sum())
            base[name + "_covariance_fraction"] = float(c_contrib[:, mask].sum() / c_contrib.sum())
            base[name + "_q_mean"] = float(q_contrib[:, mask].sum(1).mean())
            base[name + "_q_rms"] = float(np.sqrt(np.mean(q_contrib[:, mask].sum(1) ** 2)))
        rows.append(base)
        qs.append(q_direct)
        qmods.append(q_contrib)
    arrays["Q_min_twin"] = np.minimum(qs[0], qs[1])
    arrays["Q_selected_head"] = np.where(qs[0] <= qs[1], 1, 2)
    arrays["Q_min_modal_contribution"] = np.where((qs[0] <= qs[1])[:, None], qmods[0], qmods[1])
    if not np.allclose(arrays["Q_min_modal_contribution"].sum(1), arrays["Q_min_twin"], rtol=1e-9, atol=1e-8):
        raise AssertionError("twin min mode identity")
    return arrays, rows


def supplement(output, bank_path):
    batch = geometry.load_fixed_batch(GEOMETRY_BANK)
    with np.load(bank_path) as f:
        replay = mechanisms.ReplayData(**{k: f[k] for k in mechanisms.REPLAY_CACHE_ARRAY_FIELDS},
                                       metadata=json.loads(str(f['metadata'].item())))
    replay.validate(1024, 5120)
    expected_digest = "7269e0dea9762932738b45b598b5951a13e43a8a11bbbe479993ced07d41636e"
    if mechanisms.hash_arrays((k, getattr(replay, k)) for k in mechanisms.REPLAY_CACHE_ARRAY_FIELDS) != expected_digest:
        raise ValueError("large bank differs from legacy content hash")
    rewards = reward_bank(output, replay)
    all_records = records(output)
    seen, b_seen = set(), {}
    task_rows, f_rows, large_rows, pair_rows = [], [], [], []
    for record in all_records:
        gdigest, bdigest = record['geometry_sha256'], record['b_online_sha256']
        if gdigest in seen:
            continue
        seen.add(gdigest)
        print(f"supplement {record['task']} {record['frame']} unique geometry={gdigest[:12]}", flush=True)
        agent = load_agent(record)
        if bdigest not in b_seen:
            b = encode_b(agent, replay.next_obs)
            with np.load(output / 'primary' / (bdigest + '.npz')) as f:
                spectrum = {k: f[k] for k in f.files}
            old_sort = np.argsort(batch.index_bank_row)
            new_sort = np.argsort(replay.index_bank_row[:1024])
            if not np.array_equal(batch.index_bank_row[old_sort], replay.index_bank_row[:1024][new_sort]):
                raise ValueError("geometry and mechanisms probe samples differ")
            if not np.allclose(spectrum['B'][old_sort], b[:1024][new_sort], rtol=1e-6, atol=1e-6):
                raise ValueError("B differs under probe permutation")
            large = save_spectra(output / 'supplement' / (bdigest + '_large_b.npz'), b,
                                {'bank_sha256': expected_digest, 'n': 20480, 'role': 'size_sensitivity_not_primary'})
            for row in large:
                large_rows.append({'b_online_sha256': bdigest, **row})
            b_seen[bdigest] = (b, spectrum)
        b, spectrum = b_seen[bdigest]
        z_values, raw_values, energy_values, coords_values = [], [], [], []
        for task in TASKS:
            raw = (rewards[task][:5120].astype(np.float64).T @ b[:5120].astype(np.float64) / 5120).reshape(50)
            norm = float(np.linalg.norm(raw))
            near_zero = norm <= max(1e-12, 1e-8 * math.sqrt(50) * float(np.mean(np.abs(rewards[task][:5120]))))
            # Historical float32 torch reward projection and F.normalize rule.
            native_raw = torch.from_numpy(rewards[task][:5120]).T @ torch.from_numpy(b[:5120]) / 5120
            z = (math.sqrt(50) * torch.nn.functional.normalize(native_raw, dim=1)).numpy().reshape(50)
            z_values.append(z if not near_zero else np.full(50, np.nan, np.float32))
            raw_values.append(raw)
            u = spectrum['uncentered_eigenvectors_columns_ascending']
            coords = raw @ u
            energy = coords ** 2 / norm ** 2 if not near_zero else np.full(50, np.nan)
            coords_values.append(coords)
            energy_values.append(energy)
            base = {'geometry_sha256': gdigest, 'b_online_sha256': bdigest, 'reward_task': task,
                    'raw_norm': norm, 'near_zero': near_zero,
                    'native_raw_max_abs_difference': float(np.max(np.abs(native_raw.numpy().ravel() - raw)))}
            for k in (1, 2, 5, 10):
                base[f'top{k}_energy_fraction'] = float(energy[-k:].sum())
            for name, mask in weak_masks(spectrum['uncentered_eigenvalues_ascending']).items():
                base[name + '_energy_fraction'] = float(energy[mask].sum())
            task_rows.append(base)
        raw_array = np.asarray(raw_values)
        valid = np.isfinite(np.asarray(z_values)).all(axis=1)
        units = np.full_like(raw_array, np.nan)
        units[valid] = raw_array[valid] / np.linalg.norm(raw_array[valid], axis=1, keepdims=True)
        cosines = units @ units.T
        for i, task1 in enumerate(TASKS):
            for j, task2 in enumerate(TASKS):
                pair_rows.append({'geometry_sha256': gdigest, 'task1': task1, 'task2': task2, 'cosine': cosines[i, j]})
        geometry.atomic_npz(output / 'supplement' / (gdigest + '_task_z.npz'),
            tasks=np.asarray(TASKS), z_raw=raw_array, proxy_z=np.asarray(z_values),
            raw_eigen_coordinates=np.asarray(coords_values), uncentered_energy_ascending=np.asarray(energy_values),
            centered_energy_ascending=(units @ spectrum['centered_eigenvectors_columns_ascending']) ** 2,
            cosine=cosines)
        # Validate one full repository inference path per unique geometry against cached-B algebra.
        if valid.any():
            check_index = int(np.flatnonzero(valid)[0])
            observed, exact_raw, note = mechanisms.infer_task_z_exact(agent, replay, b, rewards[TASKS[check_index]], 5120, 'cpu')
            if not np.allclose(observed, z_values[check_index], rtol=3e-6, atol=3e-6):
                raise AssertionError('repository proxy-z differs')
        modes = [('geometry_fixed_random', batch.obs, batch.action, batch.z),
                 ('mechanisms_fixed_random', replay.obs[:1024], replay.action[:1024], replay.random_z[:1024])]
        modes.extend((task + '_proxy_z', replay.obs[:1024], replay.action[:1024], z_values[i]) for i, task in enumerate(TASKS) if np.isfinite(z_values[i]).all())
        for mode, obs, action, z in modes:
            arrays, summaries = projections(agent, obs, action, z, spectrum, mode)
            arrays['metadata_json'] = np.asarray(json.dumps({'geometry_sha256': gdigest, 'b_online_sha256': bdigest,
                'mode': mode, 'action_mode': 'fixed_replay', 'q_aggregation': 'per-sample minimum of twin scalar Q',
                'probe_order': 'geometry' if mode.startswith('geometry_') else 'mechanisms',
                'basis_order': 'ascending uncentered primary eigenvalues'}))
            geometry.atomic_npz(output / 'supplement' / (gdigest + '_' + mode + '_F.npz'), **arrays)
            f_rows.extend({'geometry_sha256': gdigest, 'b_online_sha256': bdigest, **x} for x in summaries)
        write_csv(output / 'supplement/task_z_summary.csv', task_rows)
        write_csv(output / 'supplement/task_cosines.csv', pair_rows)
        write_csv(output / 'supplement/forward_summary.csv', f_rows)
        write_csv(output / 'supplement/large_bank_spectrum_summary.csv', large_rows)
        del agent
        gc.collect()
    write_json(output / 'supplement/completion.json', {'unique_geometry_count': len(seen), 'unique_b_count': len(b_seen),
        'task_count': 4, 'replay_size': 20480, 'inference_size': 5120, 'probe_size': 1024,
        'optimizer_steps_executed': 0, 'update_calls_executed': 0, 'bank_sha256': expected_digest})
    print('SUPPLEMENT COMPLETE', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('main', 'supplement'))
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--bank', type=Path)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.set_grad_enabled(False)
    if args.phase == 'main':
        primary(args.output)
    else:
        if args.bank is None:
            parser.error('--bank is required for supplement')
        supplement(args.output, args.bank)


if __name__ == '__main__':
    main()
