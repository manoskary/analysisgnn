"""Tests for the aggregation scoring module.

Uses synthetic DataFrames mimicking the Delta Lake schema — no actual
Delta Lake or model inference required.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from analysisgnn.aggregation.scoring import (
    ALL_RN_TASKS,
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
    binary_nct_filter,
    nct_weight,
)


# ── Fixture helpers ──


def _make_notes(n: int = 4) -> pd.DataFrame:
    """Create a minimal notes DataFrame with n notes."""
    return pd.DataFrame(
        {
            "note_id": [f"n{i}" for i in range(n)],
            "onset_div": list(range(n)),
            "pitch_midi": [60 + i for i in range(n)],
            "measure": [1] * n,
        }
    )


def _make_probs(
    note_ids: list[str],
    task_dists: dict[str, dict[str, float]],
) -> pd.DataFrame:
    """Create a long-format probabilities DataFrame.

    Parameters
    ----------
    note_ids : list of str
    task_dists : dict[task_name, dict[class_label, probability]]
        Every note gets the **same** distribution for each task
        (use ``_make_probs_per_note`` for per-note variation).
    """
    rows = []
    for nid in note_ids:
        for task, dist in task_dists.items():
            sorted_items = sorted(dist.items(), key=lambda x: -x[1])
            for rank_idx, (label, prob) in enumerate(sorted_items):
                rows.append(
                    {
                        "note_id": nid,
                        "task": task,
                        "class_id": rank_idx,
                        "class_label": label,
                        "probability": prob,
                        "is_argmax": rank_idx == 0,
                        "rank": rank_idx + 1,
                    }
                )
    return pd.DataFrame(rows)


def _make_probs_per_note(
    per_note: dict[str, dict[str, dict[str, float]]],
) -> pd.DataFrame:
    """Per-note probability distributions.

    Parameters
    ----------
    per_note : dict[note_id, dict[task, dict[class_label, prob]]]
    """
    rows = []
    for nid, tasks in per_note.items():
        for task, dist in tasks.items():
            sorted_items = sorted(dist.items(), key=lambda x: -x[1])
            for rank_idx, (label, prob) in enumerate(sorted_items):
                rows.append(
                    {
                        "note_id": nid,
                        "task": task,
                        "class_id": rank_idx,
                        "class_label": label,
                        "probability": prob,
                        "is_argmax": rank_idx == 0,
                        "rank": rank_idx + 1,
                    }
                )
    return pd.DataFrame(rows)


def _make_edges(pairs: list[tuple[str, str, str]]) -> pd.DataFrame:
    """Create an edges DataFrame from (src, dst, edge_type) triples."""
    if not pairs:
        return pd.DataFrame(columns=["src", "dst", "edge_type"])
    return pd.DataFrame(pairs, columns=["src", "dst", "edge_type"])


# ── Standard fixtures ──

TASK_DISTS = {
    "quality": {"major triad": 0.9, "minor triad": 0.1},
    "degree1": {"1": 0.8, "5": 0.2},
    "inversion": {"0": 0.95, "1": 0.05},
    "localkey": {"G": 0.85, "D": 0.15},
    "degree2": {"None": 0.9, "5": 0.1},
}

CANDIDATE = {
    "quality": "major triad",
    "degree1": "1",
    "inversion": "0",
    "localkey": "G",
    "degree2": "None",
}


@pytest.fixture
def simple_context() -> ScoringContext:
    """4 notes, uniform distributions, onset edges between consecutive notes."""
    notes = _make_notes(4)
    probs = _make_probs(notes["note_id"].tolist(), TASK_DISTS)
    edges = _make_edges(
        [
            ("n0", "n1", "onset"),
            ("n1", "n2", "consecutive"),
            ("n2", "n3", "consecutive"),
        ]
    )
    return ScoringContext(
        note_ids=notes["note_id"].tolist(),
        notes=notes,
        probabilities=probs,
        edges=edges,
    )


# ── Task categories ──


class TestTaskCategories:
    def test_core_tasks_count(self) -> None:
        assert len(CORE_TASKS) == 5

    def test_validation_tasks_count(self) -> None:
        assert len(VALIDATION_TASKS) == 4

    def test_no_overlap(self) -> None:
        assert CORE_TASKS & VALIDATION_TASKS == set()

    def test_all_rn_tasks_is_union(self) -> None:
        assert ALL_RN_TASKS == CORE_TASKS | VALIDATION_TASKS


# ── ScoringContext ──


class TestScoringContext:
    def test_len(self, simple_context: ScoringContext) -> None:
        assert len(simple_context) == 4

    def test_contains(self, simple_context: ScoringContext) -> None:
        assert "n0" in simple_context
        assert "n99" not in simple_context

    def test_notes_filtered(self, simple_context: ScoringContext) -> None:
        assert list(simple_context.notes["note_id"]) == ["n0", "n1", "n2", "n3"]

    def test_distribution_sums_to_one(self, simple_context: ScoringContext) -> None:
        dist = simple_context.distribution("n0", "quality")
        assert dist.sum() == pytest.approx(1.0)

    def test_distribution_matrix_shape(self, simple_context: ScoringContext) -> None:
        mat = simple_context.distribution_matrix("quality")
        assert mat.shape == (4, 2)  # 4 notes, 2 classes
        assert list(mat.index) == ["n0", "n1", "n2", "n3"]

    def test_distribution_matrix_rows_sum_to_one(
        self, simple_context: ScoringContext
    ) -> None:
        mat = simple_context.distribution_matrix("quality")
        for nid in mat.index:
            assert mat.loc[nid].sum() == pytest.approx(1.0)

    def test_top_k(self, simple_context: ScoringContext) -> None:
        top1 = simple_context.top_k("quality", k=1)
        assert len(top1) == 4  # one per note
        assert all(top1["rank"] == 1)

    def test_argmax(self, simple_context: ScoringContext) -> None:
        am = simple_context.argmax("quality")
        assert len(am) == 4
        assert all(am["class_label"] == "major triad")

    def test_tasks(self, simple_context: ScoringContext) -> None:
        assert set(simple_context.tasks) == set(TASK_DISTS.keys())

    def test_internal_edges(self, simple_context: ScoringContext) -> None:
        ie = simple_context.internal_edges
        assert len(ie) == 3

    def test_note_edges(self, simple_context: ScoringContext) -> None:
        edges = simple_context.note_edges("n1")
        # n0->n1 (onset) and n1->n2 (consecutive)
        assert len(edges) == 2

    def test_note_edges_by_type(self, simple_context: ScoringContext) -> None:
        edges = simple_context.note_edges("n1", edge_type="onset")
        assert len(edges) == 1
        assert edges.iloc[0]["edge_type"] == "onset"

    def test_subcontext(self, simple_context: ScoringContext) -> None:
        sub = simple_context.subcontext(["n0", "n1"])
        assert len(sub) == 2
        assert list(sub.notes["note_id"]) == ["n0", "n1"]

    def test_subcontext_shares_data(self, simple_context: ScoringContext) -> None:
        """Sub-context references the same DataFrames (no copy)."""
        sub = simple_context.subcontext(["n0"])
        assert sub._notes is simple_context._notes
        assert sub._probabilities is simple_context._probabilities

    def test_adjacent_edges_includes_boundary(self) -> None:
        """Adjacent edges include edges crossing the group boundary."""
        notes = _make_notes(4)
        probs = _make_probs(["n0", "n1", "n2", "n3"], TASK_DISTS)
        edges = _make_edges([("n0", "n1", "onset"), ("n1", "n2", "consecutive")])
        ctx = ScoringContext(["n0", "n1"], notes, probs, edges)
        # n1->n2 crosses the boundary: n1 is in the group, n2 is not
        adj = ctx.adjacent_edges
        assert len(adj) == 2  # n0->n1 internal + n1->n2 crossing


# ── ProductScorer ──


class TestProductScorer:
    def test_single_note(self) -> None:
        notes = _make_notes(1)
        probs = _make_probs(["n0"], TASK_DISTS)
        ctx = ScoringContext(["n0"], notes, probs, _make_edges([]))
        r = ProductScorer().score(ctx, CANDIDATE)
        expected = 0.9 * 0.8 * 0.95 * 0.85 * 0.9
        assert r.score == pytest.approx(expected)
        assert r.num_notes == 1
        assert r.num_tasks == 5
        assert len(r.contributions) == 1

    def test_multiple_notes_uniform(self, simple_context: ScoringContext) -> None:
        r = ProductScorer().score(simple_context, CANDIDATE)
        # Each note produces the same per-note product; 4 notes combined
        per_note = 0.9 * 0.8 * 0.95 * 0.85 * 0.9
        expected = per_note**4
        assert r.score == pytest.approx(expected)
        assert r.num_notes == 4

    def test_contributions_have_task_probs(
        self, simple_context: ScoringContext
    ) -> None:
        r = ProductScorer().score(simple_context, CANDIDATE)
        for c in r.contributions:
            assert c.task_probabilities["quality"] == pytest.approx(0.9)
            assert c.task_probabilities["degree1"] == pytest.approx(0.8)

    def test_candidate_stored_in_result(self, simple_context: ScoringContext) -> None:
        r = ProductScorer().score(simple_context, CANDIDATE)
        assert r.candidate == CANDIDATE

    def test_trace_nonempty(self, simple_context: ScoringContext) -> None:
        r = ProductScorer().score(simple_context, CANDIDATE)
        assert len(r.trace) >= 2  # at least weights + combine

    def test_empty_context(self) -> None:
        notes = _make_notes(1)
        probs = _make_probs(["n0"], TASK_DISTS)
        ctx = ScoringContext([], notes, probs, _make_edges([]))
        r = ProductScorer().score(ctx, CANDIDATE)
        assert r.score == 0.0
        assert r.num_notes == 0

    def test_zero_probability_candidate(self, simple_context: ScoringContext) -> None:
        # "minor triad" has P=0.1, not 0; "augmented triad" has P=0
        bad_candidate = {**CANDIDATE, "quality": "augmented triad"}
        r = ProductScorer().score(simple_context, bad_candidate)
        assert r.score == 0.0


# ── GeometricMeanScorer ──


class TestGeometricMeanScorer:
    def test_single_note(self) -> None:
        notes = _make_notes(1)
        probs = _make_probs(["n0"], TASK_DISTS)
        ctx = ScoringContext(["n0"], notes, probs, _make_edges([]))
        r = GeometricMeanScorer().score(ctx, CANDIDATE)
        vals = [0.9, 0.8, 0.95, 0.85, 0.9]
        expected = math.exp(sum(math.log(v) for v in vals) / len(vals))
        assert r.score == pytest.approx(expected)

    def test_invariant_to_group_size_with_uniform_probs(self) -> None:
        """Geometric mean stays constant when adding notes with the same distributions."""
        for n in [1, 4, 8]:
            notes = _make_notes(n)
            nids = notes["note_id"].tolist()
            probs = _make_probs(nids, TASK_DISTS)
            ctx = ScoringContext(nids, notes, probs, _make_edges([]))
            r = GeometricMeanScorer().score(ctx, CANDIDATE)
            vals = [0.9, 0.8, 0.95, 0.85, 0.9]
            expected = math.exp(sum(math.log(v) for v in vals) / len(vals))
            assert r.score == pytest.approx(expected, abs=1e-9)


# ── WeightedTaskScorer ──


class TestWeightedTaskScorer:
    def test_uniform_weights_equals_geomean(
        self, simple_context: ScoringContext
    ) -> None:
        r_w = WeightedTaskScorer().score(simple_context, CANDIDATE)
        r_g = GeometricMeanScorer().score(simple_context, CANDIDATE)
        assert r_w.score == pytest.approx(r_g.score, rel=1e-9)

    def test_high_weight_dominates(self) -> None:
        notes = _make_notes(1)
        probs = _make_probs(["n0"], TASK_DISTS)
        ctx = ScoringContext(["n0"], notes, probs, _make_edges([]))
        scorer = WeightedTaskScorer(
            task_weights={"quality": 100.0}, default_task_weight=1.0
        )
        r = scorer.score(ctx, CANDIDATE)
        # Quality (P=0.9) dominates
        assert r.score > 0.85


# ── Note weighting ──


class TestNoteWeighting:
    def test_nct_weight_function(self) -> None:
        """nct_weight returns P('True') from tpc_in_label."""
        notes = _make_notes(2)
        probs = _make_probs_per_note(
            {
                "n0": {
                    "quality": {"major triad": 0.9, "minor triad": 0.1},
                    "tpc_in_label": {"True": 0.8, "False": 0.2},
                },
                "n1": {
                    "quality": {"major triad": 0.9, "minor triad": 0.1},
                    "tpc_in_label": {"True": 0.3, "False": 0.7},
                },
            }
        )
        ctx = ScoringContext(["n0", "n1"], notes, probs, _make_edges([]))
        assert nct_weight(ctx, "n0") == pytest.approx(0.8)
        assert nct_weight(ctx, "n1") == pytest.approx(0.3)

    def test_binary_nct_filter(self) -> None:
        """binary_nct_filter drops notes below threshold."""
        notes = _make_notes(2)
        probs = _make_probs_per_note(
            {
                "n0": {
                    "quality": {"major triad": 0.9, "minor triad": 0.1},
                    "tpc_in_label": {"True": 0.8, "False": 0.2},
                },
                "n1": {
                    "quality": {"major triad": 0.9, "minor triad": 0.1},
                    "tpc_in_label": {"True": 0.3, "False": 0.7},
                },
            }
        )
        ctx = ScoringContext(["n0", "n1"], notes, probs, _make_edges([]))
        filt = binary_nct_filter(threshold=0.5)
        assert filt(ctx, "n0") == 1.0
        assert filt(ctx, "n1") == 0.0

    def test_product_scorer_with_nct_filter(self) -> None:
        """ProductScorer with binary_nct_filter excludes NCT notes."""
        notes = _make_notes(2)
        probs = _make_probs_per_note(
            {
                "n0": {
                    "quality": {"major triad": 0.9, "minor triad": 0.1},
                    "degree1": {"1": 0.8, "5": 0.2},
                    "tpc_in_label": {"True": 0.9, "False": 0.1},
                },
                "n1": {
                    "quality": {"major triad": 0.7, "minor triad": 0.3},
                    "degree1": {"1": 0.6, "5": 0.4},
                    "tpc_in_label": {"True": 0.2, "False": 0.8},
                },
            }
        )
        ctx = ScoringContext(["n0", "n1"], notes, probs, _make_edges([]))
        candidate = {"quality": "major triad", "degree1": "1"}

        # Without filter: both notes contribute
        r_all = ProductScorer().score(ctx, candidate)
        assert r_all.num_notes == 2

        # With filter: only n0 contributes
        r_filt = ProductScorer(note_weight_fn=binary_nct_filter(0.5)).score(
            ctx, candidate
        )
        assert r_filt.num_notes == 1
        assert r_filt.contributions[0].note_id == "n0"


# ── SeparateScorer ──


class TestSeparateScorer:
    def test_core_and_validation_split(self) -> None:
        notes = _make_notes(1)
        full_dists = {
            **TASK_DISTS,
            "romanNumeral": {"I": 0.7, "V": 0.3},
            "root": {"G": 0.8, "D": 0.2},
        }
        probs = _make_probs(["n0"], full_dists)
        ctx = ScoringContext(["n0"], notes, probs, _make_edges([]))

        full_candidate = {
            **CANDIDATE,
            "romanNumeral": "I",
            "root": "G",
        }
        r = SeparateScorer().score(ctx, full_candidate)
        assert isinstance(r, SeparateScoringResult)
        assert r.core.num_tasks == 5
        assert r.validation.num_tasks == 2

    def test_core_only_candidate(self, simple_context: ScoringContext) -> None:
        r = SeparateScorer().score(simple_context, CANDIDATE)
        assert r.core.num_tasks == 5
        assert r.validation.num_tasks == 0
        # combined is the geometric mean of all (note, task) probabilities,
        # not the raw product score — so it won't match core.score for
        # multi-note contexts
        assert r.combined > 0


# ── Per-note variation ──


class TestPerNoteVariation:
    def test_disagreeing_notes_lower_score(self) -> None:
        """Notes that disagree with the candidate produce a lower score."""
        notes = _make_notes(2)

        # n0 agrees, n1 disagrees
        probs = _make_probs_per_note(
            {
                "n0": {"quality": {"major triad": 0.9, "minor triad": 0.1}},
                "n1": {"quality": {"major triad": 0.2, "minor triad": 0.8}},
            }
        )
        ctx = ScoringContext(["n0", "n1"], notes, probs, _make_edges([]))
        candidate = {"quality": "major triad"}

        r = ProductScorer().score(ctx, candidate)
        # n0 contributes 0.9, n1 contributes 0.2; product = 0.18
        assert r.score == pytest.approx(0.9 * 0.2)
        assert r.contributions[0].task_probabilities["quality"] == pytest.approx(0.9)
        assert r.contributions[1].task_probabilities["quality"] == pytest.approx(0.2)

    def test_geomean_handles_disagreement(self) -> None:
        notes = _make_notes(2)
        probs = _make_probs_per_note(
            {
                "n0": {"quality": {"major triad": 0.9, "minor triad": 0.1}},
                "n1": {"quality": {"major triad": 0.1, "minor triad": 0.9}},
            }
        )
        ctx = ScoringContext(["n0", "n1"], notes, probs, _make_edges([]))
        candidate = {"quality": "major triad"}
        r = GeometricMeanScorer().score(ctx, candidate)
        # geomean(0.9, 0.1) = sqrt(0.09) = 0.3
        assert r.score == pytest.approx(math.sqrt(0.09))


# ── Cross-scorer comparisons ──


class TestCrossScorerComparisons:
    def test_product_le_geomean(self, simple_context: ScoringContext) -> None:
        r_p = ProductScorer().score(simple_context, CANDIDATE)
        r_g = GeometricMeanScorer().score(simple_context, CANDIDATE)
        assert r_p.score <= r_g.score + 1e-9

    def test_all_scorers_agree_on_single_note_single_task(self) -> None:
        notes = _make_notes(1)
        probs = _make_probs(
            ["n0"], {"quality": {"major triad": 0.75, "minor triad": 0.25}}
        )
        ctx = ScoringContext(["n0"], notes, probs, _make_edges([]))
        candidate = {"quality": "major triad"}

        r_p = ProductScorer().score(ctx, candidate)
        r_g = GeometricMeanScorer().score(ctx, candidate)
        r_w = WeightedTaskScorer().score(ctx, candidate)
        assert r_p.score == pytest.approx(0.75)
        assert r_g.score == pytest.approx(0.75)
        assert r_w.score == pytest.approx(0.75)

    def test_repr_strings(self) -> None:
        assert "ProductScorer" in repr(ProductScorer())
        assert "GeometricMeanScorer" in repr(GeometricMeanScorer())
        assert "SeparateScorer" in repr(SeparateScorer())
        assert "WeightedTaskScorer" in repr(WeightedTaskScorer())
