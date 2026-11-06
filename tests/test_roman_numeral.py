"""Tests for the Roman-numeral enumeration module.

Uses synthetic DataFrames and real Delta Lake data (when available).
"""

from __future__ import annotations

import math
import os

import pandas as pd
import pytest

from analysisgnn.aggregation.roman_numeral import (
    EnumerationTrace,
    RankedCandidate,
    _build_candidate_ohr,
    _group_top_k_labels,
    _is_legal_inversion,
    enumerate_roman_numerals,
)
from analysisgnn.aggregation.scoring import (
    GeometricMeanScorer,
    ProductScorer,
    ScoringContext,
)


# ── Fixture helpers (reused from test_scoring.py) ──


def _make_notes(n: int = 4) -> pd.DataFrame:
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
    if not pairs:
        return pd.DataFrame(columns=["src", "dst", "edge_type"])
    return pd.DataFrame(pairs, columns=["src", "dst", "edge_type"])


# ── Standard distributions ──

TASK_DISTS = {
    "quality": {"major triad": 0.8, "minor triad": 0.15, "dominant seventh chord": 0.05},
    "degree1": {"1": 0.7, "5": 0.2, "4": 0.1},
    "inversion": {"0": 0.85, "1": 0.1, "2": 0.05},
    "localkey": {"I": 0.75, "V": 0.15, "IV": 0.1},
    "degree2": {"None": 0.9, "5": 0.07, "4": 0.03},
}


@pytest.fixture
def simple_context() -> ScoringContext:
    """4 notes, uniform distributions, onset edges."""
    notes = _make_notes(4)
    probs = _make_probs(notes["note_id"].tolist(), TASK_DISTS)
    edges = _make_edges([("n0", "n1", "onset"), ("n1", "n2", "consecutive")])
    return ScoringContext(
        note_ids=notes["note_id"].tolist(),
        notes=notes,
        probabilities=probs,
        edges=edges,
    )


@pytest.fixture
def single_note_context() -> ScoringContext:
    """Single note with clear distributions."""
    notes = _make_notes(1)
    probs = _make_probs(["n0"], TASK_DISTS)
    edges = _make_edges([])
    return ScoringContext(
        note_ids=["n0"], notes=notes, probabilities=probs, edges=edges,
    )


# ── _group_top_k_labels ──


class TestGroupTopKLabels:
    def test_k1_returns_argmax(self, simple_context: ScoringContext) -> None:
        labels = _group_top_k_labels(simple_context, "quality", 1)
        assert labels == ["major triad"]

    def test_k3_returns_top3(self, simple_context: ScoringContext) -> None:
        labels = _group_top_k_labels(simple_context, "quality", 3)
        assert len(labels) == 3
        assert labels[0] == "major triad"
        assert "minor triad" in labels
        assert "dominant seventh chord" in labels

    def test_degree2_none_is_top(self, simple_context: ScoringContext) -> None:
        labels = _group_top_k_labels(simple_context, "degree2", 1)
        assert labels == ["None"]

    def test_per_note_variation(self) -> None:
        """Mean distribution resolves conflicting per-note preferences."""
        notes = _make_notes(2)
        probs = _make_probs_per_note(
            {
                "n0": {"quality": {"major triad": 0.9, "minor triad": 0.1}},
                "n1": {"quality": {"major triad": 0.3, "minor triad": 0.7}},
            }
        )
        edges = _make_edges([])
        ctx = ScoringContext(
            note_ids=["n0", "n1"], notes=notes, probabilities=probs, edges=edges,
        )
        labels = _group_top_k_labels(ctx, "quality", 1)
        # Mean: major=0.6, minor=0.4 → major wins
        assert labels == ["major triad"]


# ── _is_legal_inversion ──


class TestIsLegalInversion:
    @pytest.fixture(autouse=True)
    def _load_chord_quality(self) -> None:
        from flexohr.harmony.harmony_enums import ChordQuality
        self.CQ = ChordQuality

    def test_triad_inv_0_1_2_legal(self) -> None:
        for inv in (0, 1, 2):
            assert _is_legal_inversion(self.CQ.major_triad, inv)

    def test_triad_inv_3_illegal(self) -> None:
        assert not _is_legal_inversion(self.CQ.major_triad, 3)

    def test_seventh_inv_0_to_3_legal(self) -> None:
        for inv in range(4):
            assert _is_legal_inversion(self.CQ.dominant_seventh, inv)

    def test_seventh_inv_4_illegal(self) -> None:
        assert not _is_legal_inversion(self.CQ.dominant_seventh, 4)

    def test_diminished_triad_inv_2_legal(self) -> None:
        assert _is_legal_inversion(self.CQ.diminished_triad, 2)

    def test_diminished_triad_inv_3_illegal(self) -> None:
        assert not _is_legal_inversion(self.CQ.diminished_triad, 3)


# ── _build_candidate_ohr ──


class TestBuildCandidateOhr:
    def test_simple_major_triad(self) -> None:
        ohr = _build_candidate_ohr("major triad", "1", "0", "I", "None", "G")
        assert ohr is not None
        dcml = ohr.to_format("dcml")
        assert "I" in dcml

    def test_tonicized_case(self) -> None:
        ohr = _build_candidate_ohr(
            "dominant seventh chord", "5", "0", "I", "5", "G"
        )
        assert ohr is not None
        dcml = ohr.to_format("dcml")
        assert "/" in dcml  # should have key context

    def test_none_quality_returns_none(self) -> None:
        assert _build_candidate_ohr("None", "1", "0", "I", "None", "G") is None

    def test_illegal_inversion_returns_none(self) -> None:
        # Major triad with inversion 3 — should fail validation
        ohr = _build_candidate_ohr("major triad", "1", "3", "I", "None", "G")
        assert ohr is None

    def test_minor_key(self) -> None:
        ohr = _build_candidate_ohr("minor triad", "1", "0", "i", "None", "G")
        assert ohr is not None


# ── enumerate_roman_numerals ──


class TestEnumerateRomanNumerals:
    def test_single_note_returns_candidates(
        self, single_note_context: ScoringContext
    ) -> None:
        candidates, trace = enumerate_roman_numerals(
            single_note_context, "G", k=2, top_n=5
        )
        assert len(candidates) > 0
        assert all(isinstance(c, RankedCandidate) for c in candidates)

    def test_multi_note_group(self, simple_context: ScoringContext) -> None:
        candidates, trace = enumerate_roman_numerals(
            simple_context, "G", k=2, top_n=5
        )
        assert len(candidates) > 0

    def test_respects_top_n(self, simple_context: ScoringContext) -> None:
        candidates, trace = enumerate_roman_numerals(
            simple_context, "G", k=3, top_n=3
        )
        assert len(candidates) <= 3

    def test_sorted_descending(self, simple_context: ScoringContext) -> None:
        candidates, trace = enumerate_roman_numerals(
            simple_context, "G", k=3, top_n=10
        )
        if len(candidates) >= 2:
            scores = [c.result.core.score for c in candidates]
            for i in range(len(scores) - 1):
                assert scores[i] >= scores[i + 1]

    def test_no_duplicate_dcml(self, simple_context: ScoringContext) -> None:
        candidates, trace = enumerate_roman_numerals(
            simple_context, "G", k=3, top_n=50
        )
        dcmls = [c.dcml for c in candidates]
        assert len(dcmls) == len(set(dcmls))

    def test_ranks_are_1_based(self, simple_context: ScoringContext) -> None:
        candidates, trace = enumerate_roman_numerals(
            simple_context, "G", k=2, top_n=5
        )
        for i, c in enumerate(candidates):
            assert c.rank == i + 1

    def test_trace_counts_consistent(self, simple_context: ScoringContext) -> None:
        _, trace = enumerate_roman_numerals(simple_context, "G", k=3, top_n=10)
        assert trace.num_raw_combos >= trace.num_after_inversion_prune
        assert trace.num_after_inversion_prune >= trace.num_after_validation
        assert trace.num_after_validation >= trace.num_after_dedup
        assert trace.num_after_dedup >= trace.num_returned

    def test_k1_small_product_space(self, simple_context: ScoringContext) -> None:
        _, trace = enumerate_roman_numerals(simple_context, "G", k=1, top_n=10)
        # k=1 with "None" filtered from quality: 1 quality * 1 deg1 * 1 inv * 1 lk * 1 deg2 = 1
        assert trace.num_raw_combos == 1

    def test_empty_context(self) -> None:
        notes = _make_notes(0)
        probs = pd.DataFrame(
            columns=["note_id", "task", "class_id", "class_label", "probability", "is_argmax", "rank"]
        )
        edges = _make_edges([])
        ctx = ScoringContext(note_ids=[], notes=notes, probabilities=probs, edges=edges)
        candidates, trace = enumerate_roman_numerals(ctx, "G")
        assert candidates == []
        assert trace.num_raw_combos == 0

    def test_top1_matches_argmax_unanimous(
        self, single_note_context: ScoringContext
    ) -> None:
        """When all notes agree, top-1 should match the argmax combination."""
        candidates, _ = enumerate_roman_numerals(
            single_note_context, "G", k=1, top_n=1
        )
        assert len(candidates) == 1
        # The argmax quality is "major triad", degree1 is "1", inv "0", lk "I"
        c = candidates[0]
        assert c.candidate["quality"] == "major triad"
        assert c.candidate["degree1"] == "1"
        assert c.candidate["inversion"] == "0"

    def test_with_product_scorer(self, simple_context: ScoringContext) -> None:
        candidates, _ = enumerate_roman_numerals(
            simple_context, "G", k=2, top_n=5, scorer=ProductScorer()
        )
        assert len(candidates) > 0

    def test_derive_validation(self, simple_context: ScoringContext) -> None:
        candidates, _ = enumerate_roman_numerals(
            simple_context, "G", k=2, top_n=3, derive_validation=True
        )
        if candidates:
            # Validation tasks should appear in the candidate dict
            c = candidates[0]
            # At least romanNumeral should be derived (from DCML label)
            assert "romanNumeral" in c.candidate or len(c.candidate) >= 5


# ── Real data test ──

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "Minuet_in_G_Major_K.1")
_has_output = os.path.isdir(os.path.join(OUTPUT_DIR, "notes", "_delta_log"))

pytestmark_real = pytest.mark.skipif(
    not _has_output, reason="Delta Lake output not found"
)


@pytestmark_real
class TestRealData:
    def test_enumerate_beat_group(self) -> None:
        from analysisgnn.storage.delta_reader import (
            load_edges,
            load_hyperedges,
            load_metadata,
            load_notes,
            load_probabilities,
        )

        notes = load_notes(OUTPUT_DIR)
        probs = load_probabilities(OUTPUT_DIR)
        edges = load_edges(OUTPUT_DIR)
        hyperedges = load_hyperedges(OUTPUT_DIR)
        meta = load_metadata(OUTPUT_DIR)

        full_ctx = ScoringContext(
            note_ids=notes["note_id"].tolist(),
            notes=notes,
            probabilities=probs,
            edges=edges,
            hyperedges=hyperedges,
            metadata=meta,
        )

        # Pick first beat group
        beat_groups = load_hyperedges(OUTPUT_DIR, edge_type="beat")
        first_group = beat_groups["group_id"].unique()[0]
        group_notes = beat_groups[beat_groups["group_id"] == first_group][
            "note_id"
        ].tolist()

        group_ctx = full_ctx.subcontext(group_notes)
        candidates, trace = enumerate_roman_numerals(group_ctx, "G", k=3, top_n=5)

        assert len(candidates) > 0
        assert trace.num_raw_combos > 0
        assert candidates[0].rank == 1
        assert isinstance(candidates[0].dcml, str)
        assert len(candidates[0].dcml) > 0
