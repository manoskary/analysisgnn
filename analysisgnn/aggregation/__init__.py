"""Aggregation strategies for AnalysisGNN predictions.

This package provides a registry of named aggregation strategies that operate
on Delta Lake data (pandas DataFrames) rather than PyTorch tensors.  Each
strategy takes the raw per-note probabilities, notes table, and hyperedges
table, and produces an argmax summary DataFrame matching the format used by
the Gradio display and reference CSVs.

Quickstart::

    from analysisgnn.aggregation import get_strategy

    strategy = get_strategy("mean")
    result = strategy.aggregate(probabilities, notes, hyperedges, metadata)
"""

from analysisgnn.aggregation.registry import get_strategy, list_strategies, register

__all__ = ["get_strategy", "list_strategies", "register"]
