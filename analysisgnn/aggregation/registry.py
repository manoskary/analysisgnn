"""Named strategy registry for aggregation strategies."""

from __future__ import annotations

from typing import Dict, List, Type

from analysisgnn.aggregation.base import AggregationStrategy

_STRATEGIES: Dict[str, Type[AggregationStrategy]] = {}


def register(name: str, cls: Type[AggregationStrategy]) -> None:
    """Register an aggregation strategy under *name*."""
    _STRATEGIES[name] = cls


def get_strategy(name: str) -> AggregationStrategy:
    """Instantiate and return the strategy registered under *name*.

    Raises
    ------
    KeyError
        If no strategy is registered under *name*.
    """
    if name not in _STRATEGIES:
        raise KeyError(
            f"Unknown aggregation strategy {name!r}. "
            f"Available: {list_strategies()}"
        )
    return _STRATEGIES[name]()


def list_strategies() -> List[str]:
    """Return the names of all registered strategies."""
    return sorted(_STRATEGIES)


# ---------------------------------------------------------------------------
# Auto-register built-in strategies on import
# ---------------------------------------------------------------------------


def _register_builtins() -> None:
    from analysisgnn.aggregation.mean import MeanAggregation

    # Import here to avoid circular imports; the NoneAggregation is defined
    # inline to keep things simple.

    class NoneAggregation(AggregationStrategy):
        """Passthrough strategy that returns the raw argmax summary."""

        def aggregate(self, probabilities, notes, hyperedges, metadata, tasks=None):
            return _argmax_summary_from_probs(probabilities, notes, tasks)

    register("none", NoneAggregation)
    register("mean", MeanAggregation)


def _argmax_summary_from_probs(
    probabilities, notes, tasks=None,
):
    """Build an argmax summary DataFrame from long-format probabilities.

    This mirrors :func:`analysisgnn.storage.delta_reader.argmax_summary` but
    operates on already-loaded DataFrames rather than reading from disk.
    """
    import pandas as pd

    argmax_df = probabilities[probabilities["is_argmax"]].copy()
    if tasks is not None:
        argmax_df = argmax_df[argmax_df["task"].isin(tasks)]

    labels_pivot = argmax_df.pivot(
        index="note_id", columns="task", values="class_label"
    )
    conf_pivot = argmax_df.pivot(
        index="note_id", columns="task", values="probability"
    )
    conf_pivot.columns = [f"{c}_confidence" for c in conf_pivot.columns]

    wide = pd.concat([labels_pivot, conf_pivot], axis=1).reset_index()

    # Drop columns from notes that collide with prediction task columns
    # (e.g., notes.staff vs the "staff" task prediction column).
    task_cols = set(labels_pivot.columns) | set(conf_pivot.columns)
    notes_clean = notes.drop(
        columns=[c for c in notes.columns if c in task_cols and c != "note_id"],
        errors="ignore",
    )

    result = notes_clean.merge(wide, on="note_id", how="left")
    return result


_register_builtins()
