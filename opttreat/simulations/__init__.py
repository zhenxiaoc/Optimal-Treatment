"""Paper-specific runnable simulation workflows.

Each workflow folder (``ccg2025``, ``DenseIndexDGP``, ``TaylorModel``) holds
self-contained runners that call ``opttreat.estimation``, ``opttreat.parameters``
and ``opttreat.variance`` directly; there is no shared simulation engine.
"""
