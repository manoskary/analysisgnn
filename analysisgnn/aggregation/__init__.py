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

Scoring (for Roman-numeral candidate ranking)::

    from analysisgnn.aggregation.scoring import ScoringContext, ProductScorer

    ctx = ScoringContext(note_ids, notes_df, probs_df, edges_df)
    scorer = ProductScorer()
    result = scorer.score(ctx, {"quality": "major triad", "degree1": "1", ...})
"""

from analysisgnn.aggregation.registry import get_strategy, list_strategies, register
from analysisgnn.aggregation.scoring import (
    CORE_TASKS,
    VALIDATION_TASKS,
    GeometricMeanScorer,
    NoteContribution,
    ProductScorer,
    ScoringContext,
    ScoringResult,
    SeparateScorer,
    SeparateScoringResult,
    WeightedTaskScorer,
    nct_weight,
)

__all__ = [
    "get_strategy",
    "list_strategies",
    "register",
    "CORE_TASKS",
    "VALIDATION_TASKS",
    "ScoringContext",
    "ProductScorer",
    "GeometricMeanScorer",
    "WeightedTaskScorer",
    "SeparateScorer",
    "ScoringResult",
    "SeparateScoringResult",
    "NoteContribution",
    "nct_weight",
]
