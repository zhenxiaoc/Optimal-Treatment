"""Reproduction tests for the KT empirical estimators (``TEST_Emp_*`` ports).

Reference values are the committed R results in
``R Codes/Code/results/TEST_Emp_*_results.csv``. The welfare estimators must
match R to floating-point precision; the value functional must match R exactly
on ``V_hat``, ``eps`` and the boundary-band count ``num`` (the standard error
depends on the density estimator and is intentionally not asserted -- the
Python port uses an exact Gaussian KDE rather than R's binned ``ks::Hscv``).
"""

from __future__ import annotations

import math

import pytest

from opttreat.config import VarianceConfig
from opttreat.empirical import value_sievevar, welfare_plugin, welfare_sievevar
from opttreat.variance import WelfarePlugInVariance, get_variance_estimator

# R reference rows keyed by cost flag: see the *_results.csv files.
R_WELFARE_PLUGIN = {
    False: dict(W_hat=1519.05908828218, SE=385.2326548977),
    True: dict(W_hat=857.716029068876, SE=360.180073726088),
}
R_WELFARE_NOTRIM = {
    False: dict(W_hat=1458.89116623546, SE=315.974366299183),
    True: dict(W_hat=768.389422134174, SE=295.244798420809),
}
R_WELFARE_TRIM = {
    False: dict(W_hat=1519.05908828218, SE=422.483795514627),
    True: dict(W_hat=857.716029068876, SE=394.703829342038),
}
R_VALUE_NOTRIM = {
    False: dict(V_hat=0.918247858614334, eps=6.99748032415983, num=496),
    True: dict(V_hat=0.850916187791391, eps=6.99748644713349, num=713),
}
R_VALUE_TRIM = {
    False: dict(V_hat=0.890821109934977, eps=7.02419679131273, num=931),
    True: dict(V_hat=0.797066384394375, eps=7.02420295339548, num=1411),
}


@pytest.mark.parametrize("cost", [False, True])
def test_welfare_plugin_matches_r(cost: bool) -> None:
    row = welfare_plugin(cost)
    ref = R_WELFARE_PLUGIN[cost]
    assert math.isclose(row["W_hat"], ref["W_hat"], rel_tol=1e-9)
    assert math.isclose(row["SE"], ref["SE"], rel_tol=1e-9)


@pytest.mark.parametrize(
    "trim, ref_table",
    [(False, R_WELFARE_NOTRIM), (True, R_WELFARE_TRIM)],
)
@pytest.mark.parametrize("cost", [False, True])
def test_welfare_sievevar_matches_r(trim: bool, ref_table: dict, cost: bool) -> None:
    row = welfare_sievevar(cost, trim=trim)
    ref = ref_table[cost]
    assert math.isclose(row["W_hat"], ref["W_hat"], rel_tol=1e-9)
    assert math.isclose(row["SE"], ref["SE"], rel_tol=1e-9)


@pytest.mark.parametrize(
    "trim, ref_table",
    [(False, R_VALUE_NOTRIM), (True, R_VALUE_TRIM)],
)
@pytest.mark.parametrize("cost", [False, True])
def test_value_point_and_band_match_r(trim: bool, ref_table: dict, cost: bool) -> None:
    # V_hat and eps do not depend on the Sobol grid; num does and needs the
    # full M=1e6 grid to match R's count exactly.
    row = value_sievevar(cost, trim=trim, M=1_000_000)
    ref = ref_table[cost]
    assert math.isclose(row["V_hat"], ref["V_hat"], rel_tol=1e-9)
    assert math.isclose(row["eps"], ref["eps"], rel_tol=1e-9)
    assert row["num"] == ref["num"]
    # The KDE-based SE is a real, finite number but is not expected to match R.
    assert math.isfinite(row["SE"]) and row["SE"] > 0


def test_welfare_plugin_variance_factory_and_welfare_only_guard() -> None:
    # The factory resolves the dedicated plug-in welfare variance estimator.
    estimator = get_variance_estimator(VarianceConfig("welfare_plugin", {}))
    assert isinstance(estimator, WelfarePlugInVariance)

    # It is defined for the welfare functional only and rejects value parameters.
    value_estimator = WelfarePlugInVariance(options={"param_type": "value_unknown"})
    with pytest.raises(ValueError, match="welfare functional only"):
        value_estimator.fit({})
