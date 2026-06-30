from __future__ import annotations

import numpy as np

from opttreat.simulations.DenseIndexDGP import DenseIndexDGP
from opttreat.simulations.DenseIndexDGP.value_known import (
    run_dense_index_value_known as dense_value_known,
)
from opttreat.simulations.DenseIndexDGP.welfare_known import (
    run_dense_index_welfare_known as dense_welfare_known,
)


def test_codewg_models_generate_common_setup_data() -> None:
    model = DenseIndexDGP(dim=4)
    data = model.generate_data(30)
    X = data.filter(like="X").to_numpy()
    h = model.h0(X)
    p = model.p0(X)

    assert X.shape == (30, 4)
    assert {"d", "y"}.issubset(data.columns)
    assert h.shape == (30,)
    assert p.shape == (30,)
    assert ((0.0 <= X) & (X <= 1.0)).all()
    assert ((0.0 < p) & (p < 1.0)).all()


def _patch_small_settings(monkeypatch, module) -> None:
    """Shrink the design so a smoke run finishes quickly."""
    monkeypatch.setattr(module, "DGP_DIM", 4)
    monkeypatch.setattr(module, "N_FEATURES", 12)
    monkeypatch.setattr(module, "THETA_SOBOL", 64)
    monkeypatch.setattr(module, "VARIANCE_SOBOL", 64)


def test_dense_index_welfare_known_runner_smoke(monkeypatch, tmp_path) -> None:
    _patch_small_settings(monkeypatch, dense_welfare_known)

    summary, draws = dense_welfare_known.run(ns=(120,), ite=2, jobs=1, output_dir=tmp_path)

    assert summary.shape[0] == 2
    assert draws.shape[0] == 4
    assert set(summary["spec"]) == {"dense_index"}
    assert set(summary["estimator"]) == {"plug_in", "loo"}
    assert np.isfinite(summary[["W_true", "W_mean", "bias", "se", "coverage"]].to_numpy()).all()
    assert np.isfinite(draws[["W_hat", "se"]].to_numpy()).all()
    assert (summary["se"] >= 0.0).all()
    assert (draws["se"] >= 0.0).all()
    assert summary["coverage"].between(0.0, 1.0).all()

    suffix = dense_welfare_known._suffix((120,), 2)
    stem = dense_welfare_known.STEM
    assert (tmp_path / f"{stem}_summary_{suffix}.csv").exists()
    assert (tmp_path / f"{stem}_draws_{suffix}.csv").exists()


def test_dense_index_value_known_runner_smoke(monkeypatch, tmp_path) -> None:
    _patch_small_settings(monkeypatch, dense_value_known)

    summary, draws = dense_value_known.run(ns=(120,), ite=2, jobs=1, output_dir=tmp_path)

    assert summary.shape[0] == 2
    assert draws.shape[0] == 4
    assert set(summary["spec"]) == {"dense_index"}
    assert set(summary["estimator"]) == {"plug_in", "loo"}
    assert np.isfinite(summary[["V_true", "V_mean", "bias", "se", "coverage"]].to_numpy()).all()
    assert np.isfinite(draws[["W_hat", "se"]].to_numpy()).all()
    assert (summary["se"] >= 0.0).all()
    assert (draws["se"] >= 0.0).all()
    assert summary["coverage"].between(0.0, 1.0).all()

    suffix = dense_value_known._suffix((120,), 2)
    stem = dense_value_known.STEM
    assert (tmp_path / f"{stem}_summary_{suffix}.csv").exists()
    assert (tmp_path / f"{stem}_draws_{suffix}.csv").exists()
