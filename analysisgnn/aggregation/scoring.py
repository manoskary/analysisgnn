"""Scoring framework for Roman-numeral candidate ranking.

Operates on :class:`ScoringContext` — a thin wrapper over the Delta Lake
DataFrames (notes, probabilities, edges) for a group of notes.  Every
scorer has full access to each note's complete probability distributions,
properties, and graph connectivity, enabling:

- **Note filtering / weighting** based on note properties (NCT score,
  confidence threshold, voice, staff, etc.)
- **Distribution-level operations** (combine distributions before scoring,
  or score per-note then combine)
- **Graph-aware scoring** using edge connectivity
- **Full inspectability** — results carry a per-note breakdown and a
  structured computation trace

Task roles
----------
The 5 **core tasks** (quality, degree1, degree2, inversion, localkey)
define the OHR.  Every candidate uses exactly these 5 factors.

The 4 **validation tasks** (romanNumeral, root, bass, tonkey) provide
redundant cross-checks.  Incorporating them naively creates a task-count
bias; the :class:`SeparateScorer` keeps core and validation scores
orthogonal.

Usage
-----
::

    from analysisgnn.aggregation.scoring import ScoringContext, ProductScorer

    ctx = ScoringContext(note_ids, notes_df, probs_df, edges_df)
    scorer = ProductScorer()
    result = scorer.score(ctx, {"quality": "major triad", "degree1": "1", ...})
    # Inspect per-note breakdown:
    for c in result.contributions:
        print(c.note_id, c.weight, c.task_probabilities)
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Optional,
    Sequence,
)

import pandas as pd

# ── Task categories ──

CORE_TASKS: FrozenSet[str] = frozenset(
    {"quality", "degree1", "degree2", "inversion", "localkey"}
)

VALIDATION_TASKS: FrozenSet[str] = frozenset({"romanNumeral", "root", "bass", "tonkey"})

ALL_RN_TASKS: FrozenSet[str] = CORE_TASKS | VALIDATION_TASKS


# ── ScoringContext ──


class ScoringContext:
    """Thin wrapper over Delta Lake DataFrame slices for a note group.

    Holds references to the full DataFrames and a set of note IDs defining
    the group.  All accessors filter on the fly — no data is copied at
    construction time.  Sub-contexts (subsets of notes) are created cheaply
    via :meth:`subcontext`.

    Parameters
    ----------
    note_ids : sequence of str
        Note IDs defining this group (ordering is preserved).
    notes : pd.DataFrame
        Full notes table (all notes in the score).
    probabilities : pd.DataFrame
        Full long-format probabilities table.
    edges : pd.DataFrame
        Full edges table.
    hyperedges : pd.DataFrame or None
        Full hyperedges table (optional).
    metadata : dict or None
        ``metadata.json`` contents (optional).
    """

    __slots__ = (
        "_note_ids",
        "_note_id_set",
        "_notes",
        "_probabilities",
        "_edges",
        "_hyperedges",
        "_metadata",
    )

    def __init__(
        self,
        note_ids: Sequence[str],
        notes: pd.DataFrame,
        probabilities: pd.DataFrame,
        edges: pd.DataFrame,
        hyperedges: Optional[pd.DataFrame] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        self._note_ids = list(note_ids)
        self._note_id_set = frozenset(note_ids)
        self._notes = notes
        self._probabilities = probabilities
        self._edges = edges
        self._hyperedges = hyperedges
        self._metadata = metadata

    # ── identity ──

    @property
    def note_ids(self) -> List[str]:
        """Note IDs in this group (preserves insertion order)."""
        return list(self._note_ids)

    @property
    def note_id_set(self) -> FrozenSet[str]:
        """Note IDs as a frozen set (fast membership tests)."""
        return self._note_id_set

    def __len__(self) -> int:
        return len(self._note_ids)

    def __contains__(self, note_id: str) -> bool:
        return note_id in self._note_id_set

    # ── filtered views ──

    @property
    def notes(self) -> pd.DataFrame:
        """Notes table rows for this group."""
        return self._notes[self._notes["note_id"].isin(self._note_id_set)]

    @property
    def probabilities(self) -> pd.DataFrame:
        """Long-format probabilities for this group (all tasks, all classes)."""
        return self._probabilities[
            self._probabilities["note_id"].isin(self._note_id_set)
        ]

    @property
    def internal_edges(self) -> pd.DataFrame:
        """Edges where **both** endpoints are in this group."""
        return self._edges[
            self._edges["src"].isin(self._note_id_set)
            & self._edges["dst"].isin(self._note_id_set)
        ]

    @property
    def adjacent_edges(self) -> pd.DataFrame:
        """Edges where **at least one** endpoint is in this group."""
        return self._edges[
            self._edges["src"].isin(self._note_id_set)
            | self._edges["dst"].isin(self._note_id_set)
        ]

    @property
    def hyperedges(self) -> Optional[pd.DataFrame]:
        """Hyperedges involving notes in this group (or None)."""
        if self._hyperedges is None:
            return None
        return self._hyperedges[self._hyperedges["note_id"].isin(self._note_id_set)]

    @property
    def metadata(self) -> Optional[dict]:
        return self._metadata

    # ── per-note accessors ──

    def note(self, note_id: str) -> pd.Series:
        """Single note row as a Series."""
        return self._notes.loc[self._notes["note_id"] == note_id].iloc[0]

    def note_edges(self, note_id: str, edge_type: Optional[str] = None) -> pd.DataFrame:
        """Edges incident to a specific note, optionally filtered by type."""
        mask = (self._edges["src"] == note_id) | (self._edges["dst"] == note_id)
        if edge_type is not None:
            mask = mask & (self._edges["edge_type"] == edge_type)
        return self._edges[mask]

    # ── distribution accessors ──

    def distribution(self, note_id: str, task: str) -> pd.Series:
        """Full probability distribution for one note and one task.

        Returns a Series indexed by ``class_label``, values are
        probabilities summing to 1.
        """
        mask = (self._probabilities["note_id"] == note_id) & (
            self._probabilities["task"] == task
        )
        return self._probabilities.loc[mask].set_index("class_label")["probability"]

    def distribution_matrix(self, task: str) -> pd.DataFrame:
        """Probability matrix for one task across all notes in the group.

        Returns a DataFrame: rows = note_ids, columns = class_labels,
        values = probabilities.  Each row sums to 1.
        """
        task_probs = self._probabilities[
            (self._probabilities["task"] == task)
            & self._probabilities["note_id"].isin(self._note_id_set)
        ]
        return task_probs.pivot(
            index="note_id", columns="class_label", values="probability"
        ).reindex(self._note_ids)

    def top_k(self, task: str, k: int) -> pd.DataFrame:
        """Top-k predictions per note for a given task.

        Returns the probabilities rows where ``rank <= k``.
        """
        mask = (
            self._probabilities["note_id"].isin(self._note_id_set)
            & (self._probabilities["task"] == task)
            & (self._probabilities["rank"] <= k)
        )
        return self._probabilities[mask]

    def argmax(self, task: str) -> pd.DataFrame:
        """Argmax predictions for a task across all notes in the group.

        Returns DataFrame with columns ``note_id``, ``class_label``,
        ``probability``.
        """
        mask = (
            self._probabilities["note_id"].isin(self._note_id_set)
            & (self._probabilities["task"] == task)
            & self._probabilities["is_argmax"]
        )
        return self._probabilities.loc[mask, ["note_id", "class_label", "probability"]]

    @property
    def tasks(self) -> List[str]:
        """Task names available in the probabilities table."""
        return (
            self._probabilities.loc[
                self._probabilities["note_id"].isin(self._note_id_set), "task"
            ]
            .unique()
            .tolist()
        )

    # ── sub-context ──

    def subcontext(self, note_ids: Sequence[str]) -> ScoringContext:
        """Create a sub-context for a subset of notes.

        Shares the same underlying DataFrames — no data is copied.
        """
        return ScoringContext(
            note_ids=note_ids,
            notes=self._notes,
            probabilities=self._probabilities,
            edges=self._edges,
            hyperedges=self._hyperedges,
            metadata=self._metadata,
        )

    def __repr__(self) -> str:
        return f"ScoringContext({len(self._note_ids)} notes)"


# ── Result dataclasses ──


@dataclass
class NoteContribution:
    """One note's contribution to a :class:`ScoringResult`.

    Attributes
    ----------
    note_id : str
        The contributing note.
    weight : float
        Effective weight (1.0 = full weight; 0.0 = excluded).
    task_probabilities : dict[str, float]
        For each task in the candidate, the probability this note
        assigned to the candidate's class label.
    """

    note_id: str
    weight: float
    task_probabilities: Dict[str, float]


@dataclass
class ScoringResult:
    """Result of scoring a candidate against a note group.

    Carries the composite score *and* full provenance: which notes
    contributed, with what weights and per-task probabilities, plus a
    structured trace of computation steps.

    Attributes
    ----------
    score : float
        Composite score (interpretation depends on the scorer).
    log_score : float
        Log-domain equivalent.
    num_notes : int
        Number of notes that contributed (weight > 0).
    num_tasks : int
        Number of tasks scored.
    candidate : dict[str, str]
        The task -> class_label mapping that was scored.
    contributions : list[NoteContribution]
        Per-note breakdown (only notes with weight > 0).
    trace : list[dict]
        Structured computation log.  Each entry is a dict with at
        minimum a ``"step"`` key describing the operation.
    """

    score: float
    log_score: float
    num_notes: int
    num_tasks: int
    candidate: Dict[str, str]
    contributions: List[NoteContribution]
    trace: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class SeparateScoringResult:
    """Result with independent core and validation scores.

    Attributes
    ----------
    core : ScoringResult
        Score from the 5 core tasks.
    validation : ScoringResult
        Score from the validation tasks.
    combined : float
        Geometric mean of both sub-scores for ranking convenience.
    """

    core: ScoringResult
    validation: ScoringResult
    combined: float


# ── Internal arithmetic ──


def _safe_log(p: float) -> float:
    """``log(p)`` or ``-inf`` for non-positive values."""
    return math.log(p) if p > 0 else float("-inf")


def _product(values: Sequence[float]) -> tuple[float, float]:
    """Product and log-sum of *values*.  Returns ``(product, log_sum)``."""
    prod = 1.0
    log_sum = 0.0
    for v in values:
        prod *= v
        log_sum += _safe_log(v)
    return prod, log_sum


def _geometric_mean(values: Sequence[float]) -> tuple[float, float]:
    """Geometric mean and log-sum.  Returns ``(geomean, log_sum)``."""
    n = len(values)
    if n == 0:
        return 0.0, float("-inf")
    _, log_sum = _product(values)
    return math.exp(log_sum / n) if log_sum > float("-inf") else 0.0, log_sum


def _weighted_geometric_mean(
    values: Sequence[float],
    weights: Sequence[float],
) -> tuple[float, float]:
    """Weighted geometric mean.  Returns ``(wgeomean, weighted_log_sum)``."""
    w_log_sum = 0.0
    w_total = 0.0
    for v, w in zip(values, weights):
        w_log_sum += w * _safe_log(v)
        w_total += w
    if w_total == 0:
        return 0.0, float("-inf")
    return math.exp(w_log_sum / w_total), w_log_sum


# ── Scorer ABC ──


class Scorer(ABC):
    """Abstract base for candidate scorers operating on a :class:`ScoringContext`.

    Subclasses implement :meth:`score`, which receives the full context
    for a note group and a candidate's task -> class_label mapping.
    The scorer decides which notes to include, how to weight them,
    and how to combine their per-task probabilities.
    """

    @abstractmethod
    def score(
        self,
        context: ScoringContext,
        candidate: Dict[str, str],
    ) -> ScoringResult:
        """Score *candidate* against the notes in *context*.

        Parameters
        ----------
        context : ScoringContext
            Note group with access to distributions, properties, edges.
        candidate : dict[str, str]
            Mapping from task name to predicted class label
            (e.g. ``{"quality": "major triad", "degree1": "1", ...}``).

        Returns
        -------
        ScoringResult
            Composite score with per-note breakdown and trace.
        """
        ...

    def _collect_contributions(
        self,
        context: ScoringContext,
        candidate: Dict[str, str],
        note_weights: Dict[str, float],
        trace: List[Dict[str, Any]],
    ) -> List[NoteContribution]:
        """Look up per-note, per-task probabilities for the candidate.

        Shared implementation used by concrete scorers.  For each note
        with ``weight > 0`` in *note_weights*, looks up the probability
        the note assigns to each of the candidate's class labels.

        Parameters
        ----------
        context : ScoringContext
        candidate : dict[str, str]
            Task -> class_label.
        note_weights : dict[str, float]
            Note ID -> weight.  Notes with weight 0 are excluded.
        trace : list[dict]
            Trace list to append lookup details to.

        Returns
        -------
        list[NoteContribution]
        """
        contributions: List[NoteContribution] = []
        for note_id in context.note_ids:
            w = note_weights.get(note_id, 0.0)
            if w == 0.0:
                continue
            task_probs: Dict[str, float] = {}
            for task, label in candidate.items():
                dist = context.distribution(note_id, task)
                task_probs[task] = dist.get(label, 0.0) if len(dist) > 0 else 0.0
            contributions.append(
                NoteContribution(
                    note_id=note_id,
                    weight=w,
                    task_probabilities=task_probs,
                )
            )
        trace.append(
            {
                "step": "collect_contributions",
                "num_notes": len(contributions),
                "tasks": list(candidate.keys()),
            }
        )
        return contributions

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


# ── Concrete scorers ──


class ProductScorer(Scorer):
    """Per-note product, then product across notes.

    For each note, computes the product of probabilities across tasks.
    Then combines per-note scores by product, weighted by note weights.

    Parameters
    ----------
    note_weight_fn : callable or None
        ``(context, note_id) -> float``.  When ``None``, all notes get
        weight 1.0.
    """

    def __init__(
        self,
        note_weight_fn: Optional[Callable[[ScoringContext, str], float]] = None,
    ) -> None:
        self.note_weight_fn = note_weight_fn

    def score(
        self,
        context: ScoringContext,
        candidate: Dict[str, str],
    ) -> ScoringResult:
        trace: List[Dict[str, Any]] = []

        # Compute note weights
        weights = self._compute_weights(context, trace)

        # Collect per-note contributions
        contributions = self._collect_contributions(context, candidate, weights, trace)

        if not contributions:
            return ScoringResult(
                score=0.0,
                log_score=float("-inf"),
                num_notes=0,
                num_tasks=len(candidate),
                candidate=candidate,
                contributions=[],
                trace=trace,
            )

        # Per-note task-products, then combine across notes
        per_note_scores: List[float] = []
        for c in contributions:
            vals = list(c.task_probabilities.values())
            note_score, _ = _product(vals)
            per_note_scores.append(note_score)

        combined, log_combined = _product(per_note_scores)

        trace.append(
            {
                "step": "combine",
                "method": "product",
                "per_note_scores": {
                    c.note_id: s for c, s in zip(contributions, per_note_scores)
                },
                "combined": combined,
            }
        )

        return ScoringResult(
            score=combined,
            log_score=log_combined,
            num_notes=len(contributions),
            num_tasks=len(candidate),
            candidate=candidate,
            contributions=contributions,
            trace=trace,
        )

    def _compute_weights(
        self,
        context: ScoringContext,
        trace: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        weights: Dict[str, float] = {}
        for nid in context.note_ids:
            if self.note_weight_fn is not None:
                weights[nid] = self.note_weight_fn(context, nid)
            else:
                weights[nid] = 1.0
        trace.append({"step": "compute_weights", "weights": dict(weights)})
        return weights

    def __repr__(self) -> str:
        fn_name = (
            getattr(self.note_weight_fn, "__name__", repr(self.note_weight_fn))
            if self.note_weight_fn
            else "None"
        )
        return f"ProductScorer(note_weight_fn={fn_name})"


class GeometricMeanScorer(Scorer):
    """Geometric mean across notes and tasks.

    Computes the geometric mean of all (note, task) probabilities,
    normalising by both the number of notes and tasks.  This removes
    the bias that a simple product has toward smaller groups.

    Parameters
    ----------
    note_weight_fn : callable or None
        ``(context, note_id) -> float``.
    """

    def __init__(
        self,
        note_weight_fn: Optional[Callable[[ScoringContext, str], float]] = None,
    ) -> None:
        self.note_weight_fn = note_weight_fn

    def score(
        self,
        context: ScoringContext,
        candidate: Dict[str, str],
    ) -> ScoringResult:
        trace: List[Dict[str, Any]] = []

        weights = {
            nid: (self.note_weight_fn(context, nid) if self.note_weight_fn else 1.0)
            for nid in context.note_ids
        }
        trace.append({"step": "compute_weights", "weights": dict(weights)})

        contributions = self._collect_contributions(context, candidate, weights, trace)

        if not contributions:
            return ScoringResult(
                score=0.0,
                log_score=float("-inf"),
                num_notes=0,
                num_tasks=len(candidate),
                candidate=candidate,
                contributions=[],
                trace=trace,
            )

        # Gather all probability values (flat list across notes x tasks)
        all_values: List[float] = []
        for c in contributions:
            all_values.extend(c.task_probabilities.values())

        score, log_sum = _geometric_mean(all_values)

        trace.append(
            {
                "step": "combine",
                "method": "geometric_mean",
                "n_values": len(all_values),
                "score": score,
            }
        )

        return ScoringResult(
            score=score,
            log_score=log_sum,
            num_notes=len(contributions),
            num_tasks=len(candidate),
            candidate=candidate,
            contributions=contributions,
            trace=trace,
        )


class WeightedTaskScorer(Scorer):
    """Weighted geometric mean with per-task and per-note weights.

    Each (note, task) probability is weighted by the product of the
    note's weight and the task's weight.  The final score is the
    weighted geometric mean of all contributions.

    Parameters
    ----------
    task_weights : dict[str, float] or None
        Per-task weights (e.g. ``{"romanNumeral": 2.0}`` to upweight
        tasks with larger vocabularies).  Missing tasks use
        *default_task_weight*.
    default_task_weight : float
        Fallback weight for tasks not in *task_weights*.
    note_weight_fn : callable or None
        ``(context, note_id) -> float``.
    """

    def __init__(
        self,
        task_weights: Optional[Dict[str, float]] = None,
        default_task_weight: float = 1.0,
        note_weight_fn: Optional[Callable[[ScoringContext, str], float]] = None,
    ) -> None:
        self.task_weights: Dict[str, float] = dict(task_weights) if task_weights else {}
        self.default_task_weight = default_task_weight
        self.note_weight_fn = note_weight_fn

    def score(
        self,
        context: ScoringContext,
        candidate: Dict[str, str],
    ) -> ScoringResult:
        trace: List[Dict[str, Any]] = []

        note_weights = {
            nid: (self.note_weight_fn(context, nid) if self.note_weight_fn else 1.0)
            for nid in context.note_ids
        }
        trace.append({"step": "compute_weights", "note_weights": dict(note_weights)})

        contributions = self._collect_contributions(
            context, candidate, note_weights, trace
        )

        if not contributions:
            return ScoringResult(
                score=0.0,
                log_score=float("-inf"),
                num_notes=0,
                num_tasks=len(candidate),
                candidate=candidate,
                contributions=[],
                trace=trace,
            )

        # Collect (value, combined_weight) pairs
        values: List[float] = []
        combined_weights: List[float] = []
        for c in contributions:
            nw = c.weight
            for task, p in c.task_probabilities.items():
                tw = self.task_weights.get(task, self.default_task_weight)
                values.append(p)
                combined_weights.append(nw * tw)

        score, w_log_sum = _weighted_geometric_mean(values, combined_weights)

        trace.append(
            {
                "step": "combine",
                "method": "weighted_geometric_mean",
                "task_weights": {
                    t: self.task_weights.get(t, self.default_task_weight)
                    for t in candidate
                },
                "score": score,
            }
        )

        return ScoringResult(
            score=score,
            log_score=w_log_sum,
            num_notes=len(contributions),
            num_tasks=len(candidate),
            candidate=candidate,
            contributions=contributions,
            trace=trace,
        )

    def __repr__(self) -> str:
        return (
            f"WeightedTaskScorer(task_weights={self.task_weights!r}, "
            f"default_task_weight={self.default_task_weight!r})"
        )


class SeparateScorer:
    """Scores core and validation tasks independently.

    Avoids the task-count bias by keeping the two score dimensions
    orthogonal.  Each sub-group is scored by *inner_scorer*.

    The ``combined`` field is the geometric mean across all contributing
    (note, task) probabilities from both groups, provided for ranking.

    Parameters
    ----------
    inner_scorer : Scorer or None
        Scorer to apply to each task group (default: :class:`ProductScorer`).
    core_tasks : frozenset[str] or None
        Override core task names.
    validation_tasks : frozenset[str] or None
        Override validation task names.
    """

    def __init__(
        self,
        inner_scorer: Optional[Scorer] = None,
        core_tasks: Optional[FrozenSet[str]] = None,
        validation_tasks: Optional[FrozenSet[str]] = None,
    ) -> None:
        self.inner_scorer: Scorer = inner_scorer or ProductScorer()
        self.core_tasks: FrozenSet[str] = core_tasks or CORE_TASKS
        self.validation_tasks: FrozenSet[str] = validation_tasks or VALIDATION_TASKS

    def score(
        self,
        context: ScoringContext,
        candidate: Dict[str, str],
    ) -> SeparateScoringResult:
        """Score core and validation tasks independently.

        Parameters
        ----------
        context : ScoringContext
        candidate : dict[str, str]
            Full candidate (may contain both core and validation tasks).

        Returns
        -------
        SeparateScoringResult
        """
        core_candidate = {k: v for k, v in candidate.items() if k in self.core_tasks}
        val_candidate = {
            k: v for k, v in candidate.items() if k in self.validation_tasks
        }

        core_result = self.inner_scorer.score(context, core_candidate)
        val_result = self.inner_scorer.score(context, val_candidate)

        # Combined: geometric mean across all contributing probabilities
        all_values: List[float] = []
        for c in core_result.contributions:
            all_values.extend(c.task_probabilities.values())
        for c in val_result.contributions:
            all_values.extend(c.task_probabilities.values())

        if all_values and all(v > 0 for v in all_values):
            combined, _ = _geometric_mean(all_values)
        else:
            combined = core_result.score

        return SeparateScoringResult(
            core=core_result,
            validation=val_result,
            combined=combined,
        )

    def __repr__(self) -> str:
        return f"SeparateScorer(inner_scorer={self.inner_scorer!r})"


# ── Pre-built note weight functions ──


def nct_weight(context: ScoringContext, note_id: str) -> float:
    """Weight by chord-tone probability (downweight non-chord tones).

    Uses the ``tpc_in_label`` task: returns ``P("True")``, i.e. the
    model's confidence that this note is a chord tone.  Falls back to
    1.0 if the task is unavailable.
    """
    try:
        dist = context.distribution(note_id, "tpc_in_label")
    except (KeyError, IndexError):
        return 1.0
    return dist.get("True", 1.0) if len(dist) > 0 else 1.0


def binary_nct_filter(
    threshold: float = 0.5,
) -> Callable[[ScoringContext, str], float]:
    """Return a weight function that drops notes below *threshold*.

    Notes with ``P("True")`` for ``tpc_in_label`` below *threshold*
    get weight 0.0 (excluded); others get 1.0.
    """

    def _filter(context: ScoringContext, note_id: str) -> float:
        score = nct_weight(context, note_id)
        return 1.0 if score >= threshold else 0.0

    _filter.__name__ = f"binary_nct_filter(threshold={threshold})"
    return _filter
