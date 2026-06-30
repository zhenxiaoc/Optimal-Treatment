"""Empirical-data analysis for OptTreat.

Python ports of the five ``TEST_Emp_*`` R scripts for the KT data set
(``KT_Data1.csv``): welfare and value functionals estimated with the shared
``sieve`` first stage, ``Welfare``/``ValueUnknownDist`` target parameters, and
``SieveVariance`` standard errors. See :mod:`opttreat.empirical.run_empirical`
for per-script details and the reproduction status against the R outputs.
"""

from .run_empirical import run_all, value_sievevar, welfare_plugin, welfare_sievevar

__all__ = [
    "welfare_plugin",
    "welfare_sievevar",
    "value_sievevar",
    "run_all",
]
