"""Delta Lake writer for AnalysisGNN inference results.

Writes raw per-note probability distributions, graph structure, and note metadata
to a Delta Lake directory layout as specified in AGENTS.md.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pyarrow as pa
from deltalake import write_deltalake


# ---------------------------------------------------------------------------
# Vocabulary resolution
# ---------------------------------------------------------------------------

from analysisgnn.utils.chord_representations import resolve_task_vocabulary


# ---------------------------------------------------------------------------
# Notes table
# ---------------------------------------------------------------------------


def _build_notes_table(
    note_array: np.ndarray,
    score,
) -> pa.Table:
    """Build a PyArrow table for the ``notes/`` Delta table."""
    n = len(note_array)

    # note_id: from partitura's 'id' field, or synthetic
    if "id" in note_array.dtype.names:
        note_ids = np.array([str(x) for x in note_array["id"]], dtype=object)
    else:
        note_ids = np.array([f"note_{i}" for i in range(n)], dtype=object)

    # pitch_spelling construction (matching score_note_table in hybrid_predictor.py)
    step = note_array["step"].astype(str)
    alter = note_array["alter"]
    octave = note_array["octave"] - 1
    accidental = np.where(
        alter == 0,
        "",
        np.where(
            alter > 0,
            np.char.multiply("#", alter.astype(int)),
            np.char.multiply("b", (-alter).astype(int)),
        ),
    )
    pitch_spelling = np.char.add(np.char.add(step, accidental), octave.astype(str))

    # measure numbers via partitura
    try:
        measures = score.parts[0].measure_number_map(note_array["onset_div"])
    except Exception:
        measures = np.full(n, -1, dtype=int)

    # ts_beats (may not be present)
    if "ts_beats" in note_array.dtype.names:
        ts_beats = note_array["ts_beats"].astype(np.float32)
    else:
        ts_beats = np.full(n, float("nan"), dtype=np.float32)

    data = {
        "note_id": pa.array(note_ids, type=pa.string()),
        "onset_div": pa.array(note_array["onset_div"].astype(np.int64), type=pa.int64()),
        "onset_beat": pa.array(note_array["onset_beat"].astype(np.float32), type=pa.float32()),
        "duration_div": pa.array(note_array["duration_div"].astype(np.int64), type=pa.int64()),
        "duration_beat": pa.array(note_array["duration_beat"].astype(np.float32), type=pa.float32()),
        "pitch_midi": pa.array(note_array["pitch"].astype(np.int64), type=pa.int64()),
        "pitch_spelling": pa.array(pitch_spelling, type=pa.string()),
        "staff": pa.array(note_array["staff"].astype(np.int64), type=pa.int64()),
        "voice": pa.array(note_array["voice"].astype(np.int64), type=pa.int64()),
        "measure": pa.array(measures.astype(np.int64), type=pa.int64()),
        "ts_beats": pa.array(ts_beats, type=pa.float32()),
    }
    return pa.table(data)


# ---------------------------------------------------------------------------
# Edges table
# ---------------------------------------------------------------------------

_EDGE_TYPE_TUPLES = [
    ("note", "onset", "note"),
    ("note", "consecutive", "note"),
    ("note", "during", "note"),
    ("note", "rest", "note"),
]


def _build_edges_table(data, note_ids: np.ndarray) -> pa.Table:
    """Build a PyArrow table for the ``edges/`` Delta table."""
    import torch

    src_list: List[str] = []
    dst_list: List[str] = []
    etype_list: List[str] = []

    edge_index_dict = data.edge_index_dict if hasattr(data, "edge_index_dict") else {}

    for etuple in _EDGE_TYPE_TUPLES:
        if etuple not in edge_index_dict:
            continue
        edge_type_name = etuple[1]  # e.g. "onset", "consecutive", ...
        ei = edge_index_dict[etuple]
        if isinstance(ei, torch.Tensor):
            ei = ei.cpu().numpy()
        num_edges = ei.shape[1]
        for j in range(num_edges):
            s, d = int(ei[0, j]), int(ei[1, j])
            if s < len(note_ids) and d < len(note_ids):
                src_list.append(note_ids[s])
                dst_list.append(note_ids[d])
                etype_list.append(edge_type_name)

    return pa.table({
        "src": pa.array(src_list, type=pa.string()),
        "dst": pa.array(dst_list, type=pa.string()),
        "edge_type": pa.array(etype_list, type=pa.string()),
    })


# ---------------------------------------------------------------------------
# Probabilities table (long format)
# ---------------------------------------------------------------------------


def _build_probabilities_table(
    predictions: Dict[str, "torch.Tensor"],
    task_dict: Dict[str, int],
    note_ids: np.ndarray,
) -> pa.Table:
    """Build a long-format probabilities table.

    One row per (note, task, class) combination.

    Columns:
        note_id, task, class_id, class_label, probability, is_argmax, rank
    """
    import torch

    all_note_ids: List[str] = []
    all_tasks: List[str] = []
    all_class_ids: List[int] = []
    all_class_labels: List[str] = []
    all_probs: List[float] = []
    all_is_argmax: List[bool] = []
    all_ranks: List[int] = []

    for task_name, tensor in predictions.items():
        if isinstance(tensor, torch.Tensor):
            probs = tensor.detach().cpu().float().numpy()
        else:
            probs = np.asarray(tensor, dtype=np.float32)

        if probs.ndim == 1:
            # Class IDs rather than probabilities — skip
            continue

        num_notes, num_classes = probs.shape
        actual_num_classes = task_dict.get(task_name, num_classes)
        vocab = resolve_task_vocabulary(task_name, num_classes)

        # Per-note argmax and rank
        argmax_ids = np.argmax(probs, axis=1)
        # Rank: 1 = highest probability (descending)
        ranks = np.empty_like(probs, dtype=np.int32)
        for i in range(num_notes):
            order = np.argsort(-probs[i])
            ranks[i, order] = np.arange(1, num_classes + 1)

        for c in range(num_classes):
            label = vocab[c] if c < len(vocab) else f"class_{c}"
            n = min(num_notes, len(note_ids))
            all_note_ids.extend(note_ids[:n].tolist())
            all_tasks.extend([task_name] * n)
            all_class_ids.extend([c] * n)
            all_class_labels.extend([label] * n)
            all_probs.extend(probs[:n, c].tolist())
            all_is_argmax.extend((argmax_ids[:n] == c).tolist())
            all_ranks.extend(ranks[:n, c].tolist())

    return pa.table({
        "note_id": pa.array(all_note_ids, type=pa.string()),
        "task": pa.array(all_tasks, type=pa.string()),
        "class_id": pa.array(all_class_ids, type=pa.int32()),
        "class_label": pa.array(all_class_labels, type=pa.string()),
        "probability": pa.array(all_probs, type=pa.float32()),
        "is_argmax": pa.array(all_is_argmax, type=pa.bool_()),
        "rank": pa.array(all_ranks, type=pa.int32()),
    })


# ---------------------------------------------------------------------------
# Hyperedges table
# ---------------------------------------------------------------------------


def _extract_group_ids_for_level(
    data, level: str, note_ids: np.ndarray,
) -> Optional[List[Tuple[str, str]]]:
    """Extract (group_id, note_id) pairs for a grouping level.

    Returns None if the grouping data is unavailable.
    """
    import torch

    num_notes = len(note_ids)

    if level == "onset":
        # Group by onset_div — available on data["note"].onset_div or note_array
        try:
            onset_div = data["note"].onset_div
            if isinstance(onset_div, torch.Tensor):
                onset_div = onset_div.cpu().numpy()
            else:
                onset_div = np.asarray(onset_div)
        except (AttributeError, KeyError):
            return None

        unique_onsets = np.unique(onset_div)
        onset_to_group = {int(o): i for i, o in enumerate(unique_onsets)}
        pairs = []
        for idx in range(min(len(onset_div), num_notes)):
            gid = onset_to_group[int(onset_div[idx])]
            pairs.append((f"onset_{gid}", note_ids[idx]))
        return pairs

    # beat and measure: try cluster attribute first, then edge-based
    cluster_attr = f"{level}_cluster"
    try:
        cluster = getattr(data["note"], cluster_attr)
        if isinstance(cluster, torch.Tensor):
            cluster = cluster.cpu().numpy()
        else:
            cluster = np.asarray(cluster)
        pairs = []
        for idx in range(min(len(cluster), num_notes)):
            pairs.append((f"{level}_{int(cluster[idx])}", note_ids[idx]))
        return pairs
    except (AttributeError, KeyError):
        pass

    # Fallback: edge-based grouping from (level, "connects", "note") edges
    edge_key = (level, "connects", "note")
    edge_index_dict = data.edge_index_dict if hasattr(data, "edge_index_dict") else {}
    if edge_key in edge_index_dict:
        import torch
        ei = edge_index_dict[edge_key]
        if isinstance(ei, torch.Tensor):
            ei = ei.cpu().numpy()
        # ei[0] = group node index, ei[1] = note node index
        pairs = []
        for j in range(ei.shape[1]):
            group_idx = int(ei[0, j])
            note_idx = int(ei[1, j])
            if note_idx < num_notes:
                pairs.append((f"{level}_{group_idx}", note_ids[note_idx]))
        return pairs

    return None


def _build_hyperedges_table(
    data, note_ids: np.ndarray,
) -> Tuple[pa.Table, List[str]]:
    """Build a PyArrow table for the ``hyperedges/`` Delta table.

    Returns (table, list_of_edge_types_present).
    """
    group_ids: List[str] = []
    nids: List[str] = []
    edge_types: List[str] = []
    parent_ids: List[Optional[str]] = []

    present_types: List[str] = []

    for level in ("onset", "beat", "measure"):
        pairs = _extract_group_ids_for_level(data, level, note_ids)
        if pairs is None:
            continue
        present_types.append(level)
        for gid, nid in pairs:
            group_ids.append(gid)
            nids.append(nid)
            edge_types.append(level)
            parent_ids.append(None)

    return pa.table({
        "group_id": pa.array(group_ids, type=pa.string()),
        "note_id": pa.array(nids, type=pa.string()),
        "edge_type": pa.array(edge_types, type=pa.string()),
        "parent_group_id": pa.array(parent_ids, type=pa.string()),
    }), present_types


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def _build_metadata(
    output_dir: str,
    note_ids: np.ndarray,
    predictions: Dict[str, Any],
    task_dict: Dict[str, int],
    edge_counts: Dict[str, int],
    hyperedge_types: List[str],
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the metadata.json contents."""
    class_vocabularies: Dict[str, List[str]] = {}
    for task_name in predictions:
        num_classes = task_dict.get(task_name)
        if num_classes is None:
            import torch
            t = predictions[task_name]
            if isinstance(t, torch.Tensor) and t.ndim == 2:
                num_classes = t.shape[1]
            else:
                continue
        class_vocabularies[task_name] = resolve_task_vocabulary(task_name, num_classes)

    meta = {
        "score_id": os.path.basename(output_dir),
        "inference_timestamp": datetime.now(timezone.utc).isoformat(),
        "task_dict": {k: int(v) for k, v in task_dict.items() if k in predictions},
        "class_vocabularies": class_vocabularies,
        "edge_types_included": list(edge_counts.keys()),
        "num_edges": {k: int(v) for k, v in edge_counts.items()},
        "hyperedge_types": hyperedge_types,
        "num_notes": int(len(note_ids)),
        "aggregation_history": [],
    }
    if extra:
        meta.update(extra)
    return meta


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def write_analysis_results(
    output_dir: str,
    score,
    note_array: np.ndarray,
    predictions: Dict[str, Any],
    data,
    task_dict: Dict[str, int],
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Write analysis results to a Delta Lake directory.

    Parameters
    ----------
    output_dir : str
        Target directory (will be created if it does not exist).
    score : partitura.Score
        The parsed score object.
    note_array : np.ndarray
        Structured note array, sorted by ``(onset_div, pitch)``.
    predictions : dict[str, torch.Tensor]
        Raw softmax predictions from ``ContinualAnalysisGNN.predict()``
        (with no aggregation).  Keys are task names, values are tensors
        of shape ``[num_notes, num_classes]``.
    data : torch_geometric.data.HeteroData
        The PyG graph produced by ``graphmuse.create_score_graph()``.
    task_dict : dict[str, int]
        Mapping of task name to number of classes.
    metadata : dict, optional
        Additional metadata to include (e.g., checkpoint paths, device).

    Returns
    -------
    str
        The ``output_dir`` path (for chaining).
    """
    os.makedirs(output_dir, exist_ok=True)

    # ---- note_ids (used as join key throughout) ----
    n = len(note_array)
    if "id" in note_array.dtype.names:
        note_ids = np.array([str(x) for x in note_array["id"]], dtype=object)
    else:
        note_ids = np.array([f"note_{i}" for i in range(n)], dtype=object)

    # ---- notes ----
    notes_table = _build_notes_table(note_array, score)
    write_deltalake(
        os.path.join(output_dir, "notes"),
        notes_table,
        mode="overwrite",
    )

    # ---- edges ----
    edges_table = _build_edges_table(data, note_ids)
    write_deltalake(
        os.path.join(output_dir, "edges"),
        edges_table,
        mode="overwrite",
    )

    # Edge counts for metadata
    edge_counts: Dict[str, int] = {}
    etype_col = edges_table.column("edge_type").to_pylist()
    for et in set(etype_col):
        edge_counts[et] = etype_col.count(et)

    # ---- probabilities (long format) ----
    probs_table = _build_probabilities_table(predictions, task_dict, note_ids)
    write_deltalake(
        os.path.join(output_dir, "probabilities"),
        probs_table,
        mode="overwrite",
    )

    # ---- hyperedges ----
    hyperedges_table, hyperedge_types = _build_hyperedges_table(data, note_ids)
    write_deltalake(
        os.path.join(output_dir, "hyperedges"),
        hyperedges_table,
        mode="overwrite",
    )

    # ---- metadata.json ----
    meta = _build_metadata(
        output_dir=output_dir,
        note_ids=note_ids,
        predictions=predictions,
        task_dict=task_dict,
        edge_counts=edge_counts,
        hyperedge_types=hyperedge_types,
        extra=metadata,
    )
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)

    return output_dir
