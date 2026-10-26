"""Metrics module initialization"""
from metrics.evaluator import (
    evaluate_multilabel_metrics,
    evaluate_fairness_metrics,
    compute_statistical_significance,
    compute_bootstrap_auroc_ci
)

__all__ = [
    "evaluate_multilabel_metrics",
    "evaluate_fairness_metrics",
    "compute_statistical_significance",
    "compute_bootstrap_auroc_ci"
]
