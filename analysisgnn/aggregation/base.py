"""Abstract base class for aggregation strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import pandas as pd


class AggregationStrategy(ABC):
    """Abstract base class for aggregation strategies.

    Concrete subclasses implement :meth:`aggregate`, which takes Delta Lake
    data (pandas DataFrames) and returns an argmax summary DataFrame.
    """

    @abstractmethod
    def aggregate(
        self,
        probabilities: pd.DataFrame,
        notes: pd.DataFrame,
        hyperedges: pd.DataFrame,
        metadata: dict,
        tasks: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Aggregate probabilities and return an argmax summary DataFrame.

        Parameters
        ----------
        probabilities : pd.DataFrame
            Long-format probabilities from the Delta Lake ``probabilities/``
            table.  Columns: ``note_id``, ``task``, ``class_id``,
            ``class_label``, ``probability``, ``is_argmax``, ``rank``.
        notes : pd.DataFrame
            Notes table from the Delta Lake ``notes/`` table.
        hyperedges : pd.DataFrame
            Hyperedges table from the Delta Lake ``hyperedges/`` table.
        metadata : dict
            Contents of ``metadata.json``.
        tasks : list[str], optional
            If given, restrict aggregation to these tasks.  When ``None``
            (default), all tasks in the probabilities table are processed.

        Returns
        -------
        pd.DataFrame
            One row per note with columns:

            - All columns from the *notes* table
            - For each task: ``<task>`` (argmax label) and
              ``<task>_confidence`` (max probability after aggregation)

            This matches the format of the Gradio display DataFrame and the
            reference CSVs.
        """
        ...
