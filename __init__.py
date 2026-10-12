"""Observational Fidelity for synthetic data linkage evaluation.

Pipeline: observed task -> learn name-error profile (supervised, pseudo-truth or
partly supervised) -> generate replicate synthetic tasks -> Observational
Fidelity gate + provenance audit -> linkage evaluation at a fixed threshold.
"""

__version__ = "1.0.0"
