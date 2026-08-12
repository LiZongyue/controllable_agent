import numpy as np

from url_benchmark.reward_label_sensitivity import meta_from_features


def test_mean_projection_matches_expected_normalized_direction() -> None:
    backward = np.eye(3, dtype=np.float32)
    reward = np.asarray([1.0, 2.0, -1.0], dtype=np.float32)

    meta, diagnostics = meta_from_features(
        backward,
        reward,
        z_dim=3,
        norm_z=True,
        projection_method="mean",
    )

    expected = reward / np.linalg.norm(reward) * np.sqrt(3.0)
    np.testing.assert_allclose(meta["z"], expected, rtol=1e-6, atol=1e-6)
    assert diagnostics["projection_method"] == "mean"
    assert diagnostics["ridge_lambda"] is None


def test_mean_projection_is_bit_exact_with_legacy_expression() -> None:
    rng = np.random.RandomState(11)
    backward = rng.normal(size=(20480, 50)).astype(np.float32)
    reward = rng.normal(size=20480).astype(np.float32)
    legacy = np.mean(
        backward * reward[:, None], axis=0, dtype=np.float64
    ).astype(np.float32)

    meta, _ = meta_from_features(
        backward,
        reward,
        z_dim=50,
        norm_z=False,
        projection_method="mean",
    )

    assert np.array_equal(meta["z"], legacy)


def test_ridge_corrects_anisotropic_backward_covariance() -> None:
    rng = np.random.RandomState(7)
    backward = rng.normal(size=(4096, 3)).astype(np.float32)
    backward *= np.asarray([0.1, 1.0, 8.0], dtype=np.float32)
    true_z = np.asarray([1.0, -2.0, 0.5], dtype=np.float32)
    reward = backward @ true_z

    mean_meta, _ = meta_from_features(
        backward,
        reward,
        z_dim=3,
        norm_z=True,
        projection_method="mean",
    )
    ridge_meta, diagnostics = meta_from_features(
        backward,
        reward,
        z_dim=3,
        norm_z=True,
        projection_method="ridge",
        ridge_alpha=1e-6,
    )

    target = true_z / np.linalg.norm(true_z) * np.sqrt(3.0)
    mean_error = np.linalg.norm(mean_meta["z"] - target)
    ridge_error = np.linalg.norm(ridge_meta["z"] - target)
    assert ridge_error < 0.05 * mean_error
    assert diagnostics["ridge_lambda"] > 0
    assert diagnostics["regularized_condition_number"] >= 1


def test_ridge_is_finite_for_rank_deficient_features() -> None:
    backward = np.ones((4, 5), dtype=np.float32)
    reward = np.asarray([0.0, 1.0, 0.5, -0.5], dtype=np.float32)

    meta, diagnostics = meta_from_features(
        backward,
        reward,
        z_dim=5,
        norm_z=True,
        projection_method="ridge",
        ridge_alpha=1e-2,
    )

    assert np.isfinite(meta["z"]).all()
    assert np.isfinite(diagnostics["regularized_condition_number"])
