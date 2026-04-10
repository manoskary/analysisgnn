"""Delta Lake reader for AnalysisGNN inference results.

Thin convenience wrappers around ``deltalake.DeltaTable`` for loading
Delta Lake tables written by ``write_analysis_results()``.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import pandas as pd
from deltalake import DeltaTable


# ---------------------------------------------------------------------------
# Core loaders
# ---------------------------------------------------------------------------


def load_notes(output_dir: str) -> pd.DataFrame:
    """Load the ``notes/`` Delta table.

    Returns a DataFrame with columns: ``note_id``, ``onset_div``,
    ``onset_beat``, ``duration_div``, ``duration_beat``, ``pitch_midi``,
    ``pitch_spelling``, ``staff``, ``voice``, ``measure``, ``ts_beats``.
    """
    dt = DeltaTable(os.path.join(output_dir, "notes"))
    return dt.to_pandas()


def load_edges(
    output_dir: str,
    edge_types: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Load the ``edges/`` Delta table.

    Parameters
    ----------
    output_dir : str
        Path to the per-score Delta Lake directory.
    edge_types : list[str], optional
        If given (e.g., ``["onset", "consecutive"]``), filter to only those
        edge types.

    Returns
    -------
    pd.DataFrame
        Columns: ``src``, ``dst``, ``edge_type``.
    """
    dt = DeltaTable(os.path.join(output_dir, "edges"))
    df = dt.to_pandas()
    if edge_types is not None:
        df = df[df["edge_type"].isin(edge_types)].reset_index(drop=True)
    return df


def load_probabilities(
    output_dir: str,
    task: Optional[str] = None,
    top_k: Optional[int] = None,
) -> pd.DataFrame:
    """Load the ``probabilities/`` Delta table (long format).

    Parameters
    ----------
    output_dir : str
        Path to the per-score Delta Lake directory.
    task : str, optional
        If given, filter to that task only.
    top_k : int, optional
        If given, filter to rows with ``rank <= top_k`` (e.g., ``top_k=3``
        for top-3 predictions per note per task).

    Returns
    -------
    pd.DataFrame
        Columns: ``note_id``, ``task``, ``class_id``, ``class_label``,
        ``probability``, ``is_argmax``, ``rank``.
    """
    dt = DeltaTable(os.path.join(output_dir, "probabilities"))
    df = dt.to_pandas()
    if task is not None:
        df = df[df["task"] == task].reset_index(drop=True)
    if top_k is not None:
        df = df[df["rank"] <= top_k].reset_index(drop=True)
    return df


def load_hyperedges(
    output_dir: str,
    edge_type: Optional[str] = None,
) -> pd.DataFrame:
    """Load the ``hyperedges/`` Delta table.

    Parameters
    ----------
    output_dir : str
        Path to the per-score Delta Lake directory.
    edge_type : str, optional
        If given (e.g., ``"beat"``), filter to that grouping type.

    Returns
    -------
    pd.DataFrame
        Columns: ``group_id``, ``note_id``, ``edge_type``,
        ``parent_group_id``.
    """
    dt = DeltaTable(os.path.join(output_dir, "hyperedges"))
    df = dt.to_pandas()
    if edge_type is not None:
        df = df[df["edge_type"] == edge_type].reset_index(drop=True)
    return df


def load_metadata(output_dir: str) -> dict:
    """Load ``metadata.json`` and return as a Python dict."""
    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Listing / discovery
# ---------------------------------------------------------------------------


def list_group_types(output_dir: str) -> List[str]:
    """Return the distinct ``edge_type`` values in the ``hyperedges/`` table."""
    df = load_hyperedges(output_dir)
    return sorted(df["edge_type"].unique().tolist())


def list_tables(output_dir: str) -> List[str]:
    """Return names of all Delta tables in the directory.

    Scans for subdirectories containing a ``_delta_log/`` folder.
    """
    tables: List[str] = []
    if not os.path.isdir(output_dir):
        return tables
    for entry in sorted(os.listdir(output_dir)):
        entry_path = os.path.join(output_dir, entry)
        if os.path.isdir(entry_path) and os.path.isdir(
            os.path.join(entry_path, "_delta_log")
        ):
            tables.append(entry)
    return tables


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def export_table_to_csv(
    output_dir: str,
    table_name: str,
    csv_path: str,
) -> str:
    """Load a named Delta table and write it to CSV.

    Parameters
    ----------
    output_dir : str
        Path to the per-score Delta Lake directory.
    table_name : str
        Name of the table (e.g., ``"notes"``, ``"probabilities"``).
    csv_path : str
        Destination CSV path.

    Returns
    -------
    str
        The *csv_path* (for chaining / confirmation).
    """
    table_path = os.path.join(output_dir, table_name)
    dt = DeltaTable(table_path)
    df = dt.to_pandas()
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    df.to_csv(csv_path, index=False)
    return csv_path


# ---------------------------------------------------------------------------
# Convenience: argmax summary
# ---------------------------------------------------------------------------


def argmax_summary(
    output_dir: str,
    tasks: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Produce a wide-format summary with argmax labels per task.

    Each task becomes two columns: ``<task>`` (the argmax class label) and
    ``<task>_confidence`` (the argmax probability).  The result is joined
    with the ``notes`` table on ``note_id``.

    Parameters
    ----------
    output_dir : str
        Path to the per-score Delta Lake directory.
    tasks : list[str], optional
        If given, restrict to these tasks.  When ``None`` (default), all
        tasks found in the probabilities table are included.

    Returns
    -------
    pd.DataFrame
        One row per note, with note metadata columns plus per-task argmax
        columns.
    """
    notes_df = load_notes(output_dir)
    probs_df = load_probabilities(output_dir)

    # Filter to argmax rows only
    argmax_df = probs_df[probs_df["is_argmax"]].copy()

    if tasks is not None:
        argmax_df = argmax_df[argmax_df["task"].isin(tasks)]

    # Pivot: task -> class_label
    labels_pivot = argmax_df.pivot(
        index="note_id", columns="task", values="class_label"
    )

    # Pivot: task -> probability (confidence)
    conf_pivot = argmax_df.pivot(
        index="note_id", columns="task", values="probability"
    )
    conf_pivot.columns = [f"{c}_confidence" for c in conf_pivot.columns]

    # Combine label and confidence columns
    wide = pd.concat([labels_pivot, conf_pivot], axis=1)
    wide = wide.reset_index()

    # Join with notes
    result = notes_df.merge(wide, on="note_id", how="left")
    return result
