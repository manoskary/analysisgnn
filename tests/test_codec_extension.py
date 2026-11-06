"""Tests for the FlexOHR AnalysisGNN codec extension.

Verifies that all 15 model quality labels (+ "None") can be decoded via the
``analysisgnn`` codec, including the 3 model-only labels added in Step 5:

- ``"augmented sixth"`` -> ``ChordQuality.generic_augmented_sixth`` (bidirectional)
- ``"incomplete dominant-seventh chord"`` -> ``ChordQuality.dominant_seventh`` (decode-only)
- ``"minor-augmented tetrachord"`` -> ``ChordQuality.minor_major_seventh`` (decode-only)

Also verifies that OHRs can be constructed from these qualities and that the
chord tables (interval structure, chord class, max inversion) are correct.
"""

from __future__ import annotations

import pytest

# Activate the codec
import flexohr.codecs.analysisgnn  # noqa: F401
from flexohr.harmony.harmony_enums import ChordClass, ChordQuality, CollectionType


# ── All 15 model quality labels (excluding "None") ──

MODEL_QUALITY_LABELS: list[str] = [
    "major triad",
    "minor triad",
    "diminished triad",
    "augmented triad",
    "minor seventh chord",
    "major seventh chord",
    "dominant seventh chord",
    "incomplete dominant-seventh chord",
    "diminished seventh chord",
    "half-diminished seventh chord",
    "augmented sixth",
    "German augmented sixth chord",
    "French augmented sixth chord",
    "Italian augmented sixth chord",
    "minor-augmented tetrachord",
]


class TestCodecDecoding:
    """Every model quality label decodes to a valid ChordQuality member."""

    @pytest.mark.parametrize("label", MODEL_QUALITY_LABELS)
    def test_all_model_labels_decode(self, label: str) -> None:
        q = ChordQuality.from_format(label, "analysisgnn")
        assert isinstance(q, ChordQuality)

    def test_generic_augmented_sixth_bidirectional(self) -> None:
        q = ChordQuality.from_format("augmented sixth", "analysisgnn")
        assert q is ChordQuality.generic_augmented_sixth
        assert q.to_format("analysisgnn") == "augmented sixth"

    def test_incomplete_dominant_seventh_decode_only(self) -> None:
        q = ChordQuality.from_format("incomplete dominant-seventh chord", "analysisgnn")
        assert q is ChordQuality.dominant_seventh
        # Encoding returns the primary label, not the alias
        assert q.to_format("analysisgnn") == "dominant seventh chord"

    def test_minor_augmented_tetrachord_decode_only(self) -> None:
        q = ChordQuality.from_format("minor-augmented tetrachord", "analysisgnn")
        assert q is ChordQuality.minor_major_seventh
        # Encoding returns the primary label, not the alias
        assert q.to_format("analysisgnn") == "minor-major seventh chord"

    def test_original_mappings_unaffected(self) -> None:
        """Adding aliases does not break existing bidirectional mappings."""
        q1 = ChordQuality.from_format("dominant seventh chord", "analysisgnn")
        assert q1 is ChordQuality.dominant_seventh

        q2 = ChordQuality.from_format("minor-major seventh chord", "analysisgnn")
        assert q2 is ChordQuality.minor_major_seventh


class TestChordTables:
    """Verify chord tables for the new generic_augmented_sixth member."""

    @pytest.fixture(autouse=True)
    def _load_tables(self) -> None:
        """Import chord tables after ensuring the import chain is resolved."""
        from flexohr.paradigms.pitchspace.scale import build_key_context

        # Trigger full import chain (works around circular import)
        build_key_context("C", "C")

        from flexohr.harmony.chord_tables import (
            CHORD_CLASS_MAX_INVERSION,
            CHORD_QUALITY_INTERVALS,
            CHORD_QUALITY_TO_CLASS,
        )

        self.intervals = CHORD_QUALITY_INTERVALS
        self.quality_to_class = CHORD_QUALITY_TO_CLASS
        self.max_inversion = CHORD_CLASS_MAX_INVERSION

    def test_generic_aug6_is_augmented_sixth_class(self) -> None:
        cls = self.quality_to_class[ChordQuality.generic_augmented_sixth]
        assert cls is ChordClass.augmented_sixth

    def test_generic_aug6_has_3_intervals(self) -> None:
        """Same 3-note structure as Italian sixth."""
        ivls = self.intervals[ChordQuality.generic_augmented_sixth]
        assert len(ivls) == 3

    def test_generic_aug6_intervals_match_italian(self) -> None:
        """Generic aug6 uses the Italian sixth's interval structure."""
        gen_ivls = self.intervals[ChordQuality.generic_augmented_sixth]
        it_ivls = self.intervals[ChordQuality.italian_sixth]
        # Same SIC values
        gen_sics = [sic for sic, _ in gen_ivls]
        it_sics = [sic for sic, _ in it_ivls]
        assert gen_sics == it_sics
        # Same tone functions
        gen_tfs = [tfs for _, tfs in gen_ivls]
        it_tfs = [tfs for _, tfs in it_ivls]
        assert gen_tfs == it_tfs

    def test_augmented_sixth_max_inversion(self) -> None:
        """Max inversion for augmented sixth class is 3."""
        assert self.max_inversion[ChordClass.augmented_sixth] == 3


class TestOHRConstruction:
    """OHRs can be constructed from the new quality labels."""

    def test_generic_aug6_ohr(self) -> None:
        from flexohr.core.ohr import OHR
        from flexohr.paradigms.pitchspace.scale import build_key_context
        from flexohr.paradigms.pitchspace.scale_degrees import SD

        ctx = build_key_context("C", "C")
        sd = SD.from_string("4", collection_type=CollectionType.major)
        q = ChordQuality.generic_augmented_sixth
        ohr = OHR.from_(q, sd, reference_ohr=ctx)
        dcml = ohr.to_format("dcml")
        assert isinstance(dcml, str)
        assert len(dcml) > 0

    def test_incomplete_dom7_constructs_as_dom7(self) -> None:
        """'incomplete dominant-seventh chord' maps to dominant_seventh."""
        from flexohr.core.ohr import OHR
        from flexohr.paradigms.pitchspace.scale import build_key_context
        from flexohr.paradigms.pitchspace.scale_degrees import SD

        ctx = build_key_context("C", "C")
        sd = SD.from_string("5", collection_type=CollectionType.major)
        q = ChordQuality.from_format("incomplete dominant-seventh chord", "analysisgnn")
        ohr = OHR.from_(q, sd, reference_ohr=ctx)
        dcml = ohr.to_format("dcml")
        assert isinstance(dcml, str)
        assert len(dcml) > 0

    def test_minor_aug_tetrachord_constructs_as_mm7(self) -> None:
        """'minor-augmented tetrachord' maps to minor_major_seventh."""
        from flexohr.core.ohr import OHR
        from flexohr.paradigms.pitchspace.scale import build_key_context
        from flexohr.paradigms.pitchspace.scale_degrees import SD

        ctx = build_key_context("C", "C")
        sd = SD.from_string("1", collection_type=CollectionType.major)
        q = ChordQuality.from_format("minor-augmented tetrachord", "analysisgnn")
        ohr = OHR.from_(q, sd, reference_ohr=ctx)
        dcml = ohr.to_format("dcml")
        assert isinstance(dcml, str)
        assert len(dcml) > 0


class TestEnumAlias:
    """Verify the short alias for generic_augmented_sixth."""

    def test_genAug6_alias(self) -> None:
        assert ChordQuality.genAug6 is ChordQuality.generic_augmented_sixth
