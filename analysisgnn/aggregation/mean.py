"""Mean aggregation strategy (onset -> beat -> measure).

Re-implements the existing aggregation pipeline from
``analysisgnn/models/analysis.py`` (``onsetwise_logit_aggregation``,
``beatwise_logit_aggregation``, ``measurewise_logit_aggregation``) but
operating on Delta Lake DataFrames (pandas) instead of PyTorch tensors.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from analysisgnn.aggregation.base import AggregationStrategy

# Task lists per level — must match DEFAULT_TASKS_BY_LEVEL in
# analysisgnn/models/posthoc_aggregator.py
_TASKS_BY_LEVEL: Dict[str, List[str]] = {
    "onset": [
        "cadence", "phrase", "root", "localkey", "quality",
        "inversion", "degree1", "degree2", "romanNumeral", "section",
    ],
    "beat": [
        "root", "localkey", "quality", "inversion", "degree1",
        "degree2", "romanNumeral", "cadence", "phrase", "section",
    ],
    "measure": ["localkey"],
}


class MeanAggregation(AggregationStrategy):
    """Onset -> beat -> measure mean aggregation.

    This mirrors the sequential pipeline in ``_aggregate_note_probs``
    (``analysisgnn/models/analysis.py``):

    1. **Onset level**: group notes by onset, compute mean probabilities
       over chord-tone notes within each group, broadcast back. Then apply
       change-point detection to create contiguous spans of identical argmax
       predictions.
    2. **Beat level**: group by beat, compute mean over chord tones, broadcast.
    3. **Measure level**: group by measure, compute mean over ALL notes
       (no chord-tone filter), for ``localkey`` only.
    """

    def aggregate(
        self,
        probabilities: pd.DataFrame,
        notes: pd.DataFrame,
        hyperedges: pd.DataFrame,
        metadata: dict,
        tasks: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        all_tasks = probabilities["task"].unique().tolist()
        if tasks is not None:
            all_tasks = [t for t in tasks if t in all_tasks]

        # Build a wide probability matrix per task: {task: DataFrame} where
        # each DataFrame has note_id as index and class_id columns.
        prob_wide = _build_prob_matrices(probabilities, all_tasks)

        # Get the chord-tone mask (tpc_in_label argmax == 1 for each note)
        chord_tone_mask = _get_chord_tone_mask(prob_wide)

        # Group memberships
        onset_groups = _get_group_membership(hyperedges, "onset")
        beat_groups = _get_group_membership(hyperedges, "beat")
        measure_groups = _get_group_membership(hyperedges, "measure")

        # Ordered note_ids (preserving the notes table row order)
        note_ids = notes["note_id"].tolist()

        # --- Level 1: Onset aggregation ---
        onset_tasks = [t for t in _TASKS_BY_LEVEL["onset"] if t in prob_wide]
        for task in onset_tasks:
            prob_wide[task] = _groupwise_mean_broadcast(
                prob_wide[task], onset_groups, chord_tone_mask, note_ids,
            )

        # Change-point detection (post onset aggregation)
        prob_wide = _change_point_broadcast(
            prob_wide, onset_tasks, onset_groups, chord_tone_mask, note_ids,
            notes,
        )

        # --- Level 2: Beat aggregation ---
        beat_tasks = [t for t in _TASKS_BY_LEVEL["beat"] if t in prob_wide]
        for task in beat_tasks:
            prob_wide[task] = _groupwise_mean_broadcast(
                prob_wide[task], beat_groups, chord_tone_mask, note_ids,
            )

        # --- Level 3: Measure aggregation (no chord-tone filter) ---
        measure_tasks = [t for t in _TASKS_BY_LEVEL["measure"] if t in prob_wide]
        for task in measure_tasks:
            prob_wide[task] = _groupwise_mean_broadcast(
                prob_wide[task], measure_groups, None, note_ids,
            )

        # Build the argmax summary from the aggregated probabilities
        return _to_argmax_summary(prob_wide, notes, all_tasks, probabilities)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_prob_matrices(
    probabilities: pd.DataFrame,
    tasks: List[str],
) -> Dict[str, pd.DataFrame]:
    """Build per-task wide probability matrices.

    Returns a dict mapping task name to a DataFrame with:
    - index: note_id (string)
    - columns: class_id (int) — one column per class
    - values: probability (float)
    """
    result: Dict[str, pd.DataFrame] = {}
    for task in tasks:
        task_df = probabilities[probabilities["task"] == task]
        if task_df.empty:
            continue
        wide = task_df.pivot(
            index="note_id", columns="class_id", values="probability"
        )
        # Ensure columns are sorted ints
        wide.columns = wide.columns.astype(int)
        wide = wide.sort_index(axis=1)
        result[task] = wide
    return result


def _get_chord_tone_mask(
    prob_wide: Dict[str, pd.DataFrame],
) -> Optional[Dict[str, bool]]:
    """Return a dict mapping note_id -> True if chord tone.

    Uses the argmax of ``tpc_in_label`` probabilities: class 1 = chord tone.
    Returns ``None`` if ``tpc_in_label`` is not present.
    """
    if "tpc_in_label" not in prob_wide:
        return None
    tpc_df = prob_wide["tpc_in_label"]
    argmax = tpc_df.values.argmax(axis=1)
    return {nid: bool(am) for nid, am in zip(tpc_df.index, argmax)}


def _get_group_membership(
    hyperedges: pd.DataFrame,
    edge_type: str,
) -> Dict[str, List[str]]:
    """Return {group_id: [note_id, ...]} for the given edge_type."""
    filtered = hyperedges[hyperedges["edge_type"] == edge_type]
    return filtered.groupby("group_id")["note_id"].apply(list).to_dict()


def _groupwise_mean_broadcast(
    prob_matrix: pd.DataFrame,
    groups: Dict[str, List[str]],
    chord_tone_mask: Optional[Dict[str, bool]],
    note_ids: List[str],
) -> pd.DataFrame:
    """Compute group-wise mean over eligible notes and broadcast back.

    For each group:
    1. Find eligible notes (chord tones if mask is provided, else all).
    2. If no eligible notes in the group, leave all notes unchanged.
    3. Compute the mean probability vector over eligible notes.
    4. Broadcast the mean to ALL notes in the group (including non-eligible).

    This matches ``_groupwise_mean_broadcast`` in ``analysis.py``.
    """
    result = prob_matrix.copy()
    note_set = set(prob_matrix.index)

    for group_id, member_note_ids in groups.items():
        # Filter to notes that exist in the probability matrix
        members = [nid for nid in member_note_ids if nid in note_set]
        if not members:
            continue

        # Determine eligible notes
        if chord_tone_mask is not None:
            eligible = [nid for nid in members if chord_tone_mask.get(nid, False)]
        else:
            eligible = members

        if not eligible:
            # No eligible notes — leave unchanged (matches PyTorch behavior:
            # valid mask intersection is empty, so no scatter_mean, clone kept)
            continue

        # Compute mean over eligible notes
        eligible_probs = prob_matrix.loc[eligible].values
        group_mean = eligible_probs.mean(axis=0)

        # Broadcast to ALL members (eligible AND ineligible)
        # In the PyTorch code, _groupwise_mean_broadcast only overwrites
        # eligible notes (idx = valid notes), NOT all members.
        # Let's match that exactly: only overwrite eligible notes.
        result.loc[eligible] = group_mean

    return result


def _change_point_broadcast(
    prob_wide: Dict[str, pd.DataFrame],
    tasks: List[str],
    onset_groups: Dict[str, List[str]],
    chord_tone_mask: Optional[Dict[str, bool]],
    note_ids: List[str],
    notes: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """Post-onset change-point detection and span broadcasting.

    For each task, find where the argmax label changes across consecutive
    onset times. Within each constant-label span, broadcast the probability
    distribution from the span's first onset to all notes in the span.

    This mirrors the change-point detection in ``onsetwise_logit_aggregation``
    (lines 421-452 of ``analysis.py``).
    """
    # Build a mapping from note_id to onset_div for ordering
    note_onset = notes.set_index("note_id")["onset_div"]

    # Get all note_ids that are valid (exist in the prob matrices)
    valid_note_ids = list(prob_wide[tasks[0]].index) if tasks else []
    if not valid_note_ids:
        return prob_wide

    # Filter to chord-tone notes only (for the change-point detection)
    if chord_tone_mask is not None:
        filtered_note_ids = [nid for nid in valid_note_ids
                             if chord_tone_mask.get(nid, False)]
    else:
        filtered_note_ids = valid_note_ids

    if len(filtered_note_ids) <= 1:
        return prob_wide

    # Get onset_div values for filtered notes and sort
    filtered_onsets = note_onset.loc[filtered_note_ids].values
    sort_idx = np.argsort(filtered_onsets)
    sorted_note_ids = [filtered_note_ids[i] for i in sort_idx]
    sorted_onsets = filtered_onsets[sort_idx]

    # Get unique onsets and the first note at each unique onset
    unique_onsets, first_indices = np.unique(sorted_onsets, return_index=True)
    unique_onset_note_ids = [sorted_note_ids[i] for i in first_indices]

    if len(unique_onsets) <= 1:
        return prob_wide

    for task in tasks:
        pm = prob_wide[task]

        # Get the argmax at each unique onset (from the onset-aggregated probs)
        onset_argmax = pm.loc[unique_onset_note_ids].values.argmax(axis=1)

        # Find change points
        changes = np.where(onset_argmax[1:] != onset_argmax[:-1])[0] + 1
        change_points = np.concatenate([[0], changes])

        # For each segment between change points, broadcast the segment's
        # probability distribution to all valid notes in that onset range
        for seg_idx in range(len(change_points) - 1):
            seg_start_onset = unique_onsets[change_points[seg_idx]]
            seg_end_onset = unique_onsets[change_points[seg_idx + 1]]

            # The probability distribution to broadcast is from the
            # change point's onset
            cp_note_id = unique_onset_note_ids[change_points[seg_idx]]
            cp_probs = pm.loc[cp_note_id].values

            # Find all valid notes in this onset range
            for nid in valid_note_ids:
                onset_val = note_onset[nid]
                if seg_start_onset <= onset_val < seg_end_onset:
                    pm.loc[nid] = cp_probs

        # Handle the last segment (from last change point to end)
        # The PyTorch code does NOT explicitly handle the last segment
        # (the loop is `for i in range(len(change_points) - 1)`), so
        # the last segment's notes keep their onset-aggregated values.
        # We match that behavior by not modifying them.

        prob_wide[task] = pm

    return prob_wide


def _to_argmax_summary(
    prob_wide: Dict[str, pd.DataFrame],
    notes: pd.DataFrame,
    tasks: List[str],
    original_probabilities: pd.DataFrame,
) -> pd.DataFrame:
    """Convert aggregated probability matrices back to an argmax summary.

    For tasks that were NOT aggregated (not in any level's task list), we
    use the original probabilities' argmax.
    """
    # Drop columns from notes that collide with prediction task columns
    # (e.g., notes.staff vs the "staff" task prediction column).
    task_cols = set(tasks) | {f"{t}_confidence" for t in tasks}
    result = notes.drop(
        columns=[c for c in notes.columns if c in task_cols and c != "note_id"],
        errors="ignore",
    ).copy()

    # Build a lookup for original class labels: {task: {class_id: label}}
    label_lookup: Dict[str, Dict[int, str]] = {}
    for task in tasks:
        task_probs = original_probabilities[original_probabilities["task"] == task]
        if task_probs.empty:
            continue
        lookup = task_probs.drop_duplicates(subset=["class_id"])[
            ["class_id", "class_label"]
        ]
        label_lookup[task] = dict(zip(lookup["class_id"], lookup["class_label"]))

    for task in tasks:
        if task in prob_wide:
            pm = prob_wide[task]
            argmax_ids = pm.values.argmax(axis=1)
            max_probs = pm.values.max(axis=1)

            # Map class_ids to labels
            task_labels = label_lookup.get(task, {})
            # Use the column names (class_ids) to resolve the actual class_id
            class_ids_arr = np.array(pm.columns)
            resolved_ids = class_ids_arr[argmax_ids]
            labels = [task_labels.get(int(cid), str(cid)) for cid in resolved_ids]

            # Build a Series indexed by note_id
            label_series = pd.Series(labels, index=pm.index, name=task)
            conf_series = pd.Series(max_probs, index=pm.index, name=f"{task}_confidence")

            result = result.merge(
                label_series.reset_index(), on="note_id", how="left",
            )
            result = result.merge(
                conf_series.reset_index(), on="note_id", how="left",
            )
        else:
            # Task not aggregated — use original argmax
            task_probs = original_probabilities[
                (original_probabilities["task"] == task)
                & (original_probabilities["is_argmax"])
            ]
            if task_probs.empty:
                continue
            label_df = task_probs[["note_id", "class_label", "probability"]].copy()
            label_df = label_df.rename(columns={
                "class_label": task,
                "probability": f"{task}_confidence",
            })
            result = result.merge(label_df, on="note_id", how="left")

    return result


# ---------------------------------------------------------------------------
# Single-level grouped mean aggregation
# ---------------------------------------------------------------------------


class GroupedMeanAggregation(AggregationStrategy):
    """Mean aggregation at a single hyperedge grouping level.

    Unlike :class:`MeanAggregation` which runs the full onset → beat →
    measure pipeline, this strategy applies mean aggregation at exactly one
    grouping level (e.g., only onset, only beat, only measure) for **all**
    requested tasks, with chord-tone filtering.

    Parameters
    ----------
    level : str
        Hyperedge type to group by (``"onset"``, ``"beat"``, ``"measure"``).
    """

    def __init__(self, level: str) -> None:
        self.level = level

    def aggregate(
        self,
        probabilities: pd.DataFrame,
        notes: pd.DataFrame,
        hyperedges: pd.DataFrame,
        metadata: dict,
        tasks: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        all_tasks = probabilities["task"].unique().tolist()
        if tasks is not None:
            all_tasks = [t for t in tasks if t in all_tasks]

        prob_wide = _build_prob_matrices(probabilities, all_tasks)
        chord_tone_mask = _get_chord_tone_mask(prob_wide)
        groups = _get_group_membership(hyperedges, self.level)
        note_ids = notes["note_id"].tolist()

        for task in all_tasks:
            if task in prob_wide:
                prob_wide[task] = _groupwise_mean_broadcast(
                    prob_wide[task], groups, chord_tone_mask, note_ids,
                )

        return _to_argmax_summary(prob_wide, notes, all_tasks, probabilities)
