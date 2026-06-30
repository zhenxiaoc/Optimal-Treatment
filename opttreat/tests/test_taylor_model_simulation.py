from __future__ import annotations

import numpy as np

from opttreat.simulations.TaylorModel import run_taylor_rf


def test_taylor_default_specs_use_best_expansion() -> None:
    assert {spec["expansion"] for spec in run_taylor_rf.SPECS} == {"hyperbolic"}


def test_taylor_welfare_known_runner_smoke(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(run_taylor_rf, "N_FEATURES", 20)
    monkeypatch.setattr(run_taylor_rf, "THETA_SOBOL", 128)
    monkeypatch.setattr(run_taylor_rf, "VARIANCE_SOBOL", 128)

    specs = [{"expansion": "hyperbolic", "K": 4}]
    summary, draws = run_taylor_rf.run(specs=specs, ns=(120,), ite=2, jobs=1, output_dir=tmp_path)

    assert summary.shape[0] == 2
    assert draws.shape[0] == 4
    assert set(summary["estimator"]) == {"plug_in", "loo"}
    assert set(summary["expansion"]) == {"hyperbolic"}
    assert np.isfinite(summary[["W_true", "W_mean", "bias", "se", "coverage"]].to_numpy()).all()
    assert np.isfinite(draws[["W_hat", "se"]].to_numpy()).all()
    assert (summary["se"] >= 0.0).all()
    assert summary["coverage"].between(0.0, 1.0).all()

    suffix = run_taylor_rf._suffix(specs, (120,), 2)
    stem = run_taylor_rf.STEM
    assert (tmp_path / f"{stem}_summary_{suffix}.csv").exists()
    assert (tmp_path / f"{stem}_draws_{suffix}.csv").exists()
