from __future__ import annotations

import numpy as np
import pytest

from opttreat.simulations.ccg2025.ccg_welfare_known import run_ccg_welfare_known as welfare_known
from opttreat.simulations.ccg2025.ccg_welfare_unknown import run_ccg_welfare_unknown as welfare_unknown
from opttreat.simulations.ccg2025.ccg_value_known import run_ccg_value_known as value_known


CASES = [
    (welfare_known, "ccg_welfare_known"),
    (welfare_unknown, "ccg_welfare_unknown"),
    (value_known, "ccg_value_known"),
]


@pytest.mark.parametrize("module, stem", CASES)
def test_ccg_runner_reports_plugin_and_loo_with_analytical_band(module, stem, tmp_path):
    summary, draws = module.smoke_main(output_dir=tmp_path)

    # One smoke model and sample size yields a plug-in row and a LOO row.
    assert set(summary["estimator"]) == {"plug_in", "loo"}
    assert summary.shape[0] == 2
    assert draws.shape[0] == 2

    # Both estimators carry SieveVar inference.
    assert np.isfinite(summary[["W_true", "bias", "se", "coverage"]].to_numpy()).all()
    assert (summary["se"] >= 0.0).all()
    assert np.isfinite(draws[["W_hat", "se"]].to_numpy()).all()

    # The estimator and the analytical band default are recorded in the file names.
    suffix = "sieve_n90_rep1_loo0p05_band"
    assert (tmp_path / f"{stem}_summary_{suffix}.csv").exists()
    assert (tmp_path / f"{stem}_draws_{suffix}.csv").exists()


def test_ccg_value_known_plugin_within_value_scale(tmp_path):
    _, draws = value_known.smoke_main(output_dir=tmp_path)

    # The plug-in value functional E[1{h>0} * 9] lies in [0, 9]. (The LOO
    # correction is numerically meaningless at smoke sample sizes, so it is not
    # range-checked here.)
    plug_in = draws.loc[draws["estimator"] == "plug_in", "W_hat"]
    assert plug_in.between(0.0, 9.0).all()
