"""Top-k Roman-numeral candidate enumeration from GNN predictions.

Given a :class:`ScoringContext` for a note group, enumerates legal
Roman-numeral candidates (OHRs), scores them, deduplicates by DCML label,
and returns the top-k ranked results.

Public API::

    from analysisgnn.aggregation.roman_numeral import enumerate_roman_numerals

    candidates, trace = enumerate_roman_numerals(group_ctx, "G", k=3, top_n=5)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import product
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from analysisgnn.aggregation.scoring import (
    CORE_TASKS,
    VALIDATION_TASKS,
    GeometricMeanScorer,
    Scorer,
    ScoringContext,
    SeparateScorer,
    SeparateScoringResult,
)

if TYPE_CHECKING:
    from flexohr.core.ohr import OHR
    from flexohr.harmony.harmony_enums import ChordQuality

logger = logging.getLogger(__name__)


# ── Data classes ──


@dataclass
class RankedCandidate:
    """A scored Roman-numeral candidate."""

    ohr: OHR
    dcml: str
    candidate: Dict[str, str]
    result: SeparateScoringResult
    rank: int

    def __repr__(self) -> str:
        return f"RankedCandidate(#{self.rank} {self.dcml!r}, score={self.result.core.score:.6f})"

    def _repr_html_(self) -> str:
        rows = "".join(
            f"<tr><td>{t}</td><td>{l}</td></tr>"
            for t, l in self.candidate.items()
        )
        return (
            f"<table>"
            f"<tr><th colspan='2'>#{self.rank} <code>{self.dcml}</code> "
            f"(score={self.result.core.score:.6f})</th></tr>"
            f"{rows}</table>"
        )


@dataclass
class EnumerationTrace:
    """Diagnostic info about the enumeration process."""

    num_raw_combos: int
    num_after_inversion_prune: int
    num_after_validation: int
    num_after_dedup: int
    num_returned: int
    pruned_reasons: Dict[str, int] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"EnumerationTrace({self.num_raw_combos} raw "
            f"-> {self.num_after_dedup} dedup "
            f"-> {self.num_returned} returned)"
        )

    def _repr_html_(self) -> str:
        rows = [
            ("Raw combinations", self.num_raw_combos),
            ("After inversion prune", self.num_after_inversion_prune),
            ("After validation", self.num_after_validation),
            ("After dedup", self.num_after_dedup),
            ("Returned", self.num_returned),
        ]
        html = "<table><tr><th colspan='2'>EnumerationTrace</th></tr>"
        for k, v in rows:
            html += f"<tr><td><b>{k}</b></td><td>{v}</td></tr>"
        if self.pruned_reasons:
            reasons = ", ".join(f"{k}: {v}" for k, v in self.pruned_reasons.items())
            html += f"<tr><td><b>Pruned reasons</b></td><td>{reasons}</td></tr>"
        html += "</table>"
        return html


# ── Helpers ──


def _group_top_k_labels(context: ScoringContext, task: str, k: int) -> List[str]:
    """Top-k labels from the mean distribution across all notes in the group.

    Computes the element-wise mean of each note's probability distribution
    for *task*, then returns the *k* class labels with the highest mean
    probability.
    """
    mat = context.distribution_matrix(task)
    mean_dist = mat.mean(axis=0).sort_values(ascending=False)
    return mean_dist.head(k).index.tolist()


def _max_inversion_for_quality(quality_enum: ChordQuality) -> int:
    """Return the maximum legal 0-based inversion index for *quality_enum*.

    Triads: max 2 (root, first, second).  Seventh chords and augmented
    sixths: max 3.  Dyads/suspended/unknown: 0.
    """
    from flexohr.harmony.harmony_enums import ChordQuality as _CQ

    # Seventh chords and augmented sixths: max inversion 3
    _SEVENTH_OR_AUG6 = frozenset({
        _CQ.dominant_seventh, _CQ.minor_seventh, _CQ.major_seventh,
        _CQ.minor_major_seventh, _CQ.diminished_seventh,
        _CQ.half_diminished_seventh, _CQ.augmented_seventh,
        _CQ.augmented_major_seventh,
        _CQ.italian_sixth, _CQ.german_sixth, _CQ.french_sixth,
        _CQ.generic_augmented_sixth,
    })
    # Triads: max inversion 2
    _TRIADS = frozenset({
        _CQ.major_triad, _CQ.minor_triad, _CQ.diminished_triad,
        _CQ.augmented_triad,
    })

    if quality_enum in _SEVENTH_OR_AUG6:
        return 3
    if quality_enum in _TRIADS:
        return 2
    return 0


def _is_legal_inversion(quality_enum: ChordQuality, inversion: int) -> bool:
    """Check whether *inversion* is legal for the given chord quality."""
    return inversion <= _max_inversion_for_quality(quality_enum)


def _build_candidate_ohr(
    quality_label: str,
    degree1_label: str,
    inversion_label: str,
    localkey_label: str,
    degree2_label: str,
    global_key: str,
) -> Optional[OHR]:
    """Attempt to build an OHR from candidate labels.

    Returns ``None`` on any construction failure (catches all exceptions).
    """
    from flexohr import InversionBassConsistency, OHR as _OHR
    from flexohr.harmony.harmony_enums import ChordQuality as _CQ, CollectionType as _CT
    from flexohr.paradigms.pitchspace.scale import (
        build_key_context,
        infer_collection_type,
    )
    from flexohr.paradigms.pitchspace.scale_degrees import SD

    import flexohr.codecs.analysisgnn  # noqa: F401 — activate codec

    try:
        if quality_label == "None":
            return None

        quality = _CQ.from_format(quality_label, "analysisgnn")
        inv = int(float(inversion_label))

        # Legality pre-check (InversionBassConsistency may not catch this
        # on unresolved OHRs, so we check explicitly)
        if not _is_legal_inversion(quality, inv):
            return None

        # Key context.  AnalysisGNN uses "-" for flat (e.g. "A-"); FlexOHR
        # expects "b" (e.g. "Ab") — normalise before parsing.
        lk_flx = localkey_label.replace("-", "b")
        lk_coll = infer_collection_type(lk_flx)
        has_tonicization = degree2_label not in ("None", "")

        if has_tonicization:
            sd2 = SD.from_int(int(float(degree2_label)), collection_type=lk_coll)
            ref_ohr = build_key_context(
                global_key,
                lk_flx,
                tonicized_key=sd2,
                tonicized_coll=_CT.major,
            )
            tonic_coll = _CT.major
        else:
            ref_ohr = build_key_context(global_key, lk_flx)
            tonic_coll = lk_coll

        degree1_sd = SD.from_int(int(float(degree1_label)), collection_type=tonic_coll)
        ohr = _OHR.from_(quality, degree1_sd, inversion=inv, reference_ohr=ref_ohr)

        # Validate
        validator = InversionBassConsistency()
        try:
            inner = ohr.ohr()
        except (IndexError, AttributeError):
            inner = ohr
        if not validator(inner).is_valid:
            return None

        return ohr

    except Exception:
        return None


def _flx_to_agnn(name: str) -> str:
    """Convert FlexOHR pitch notation to AnalysisGNN (``b`` → ``-``)."""
    if len(name) <= 1:
        return name
    return name[0] + name[1:].replace("b", "-")


def _derive_validation_labels(ohr: OHR, global_key: str) -> Dict[str, str]:
    """Derive romanNumeral, root, bass, and tonkey labels from a resolved OHR.

    Returns a dict mapping validation task names to their expected class
    labels.  Only includes tasks where derivation succeeds.
    """
    from flexohr.harmony.harmony_enums import KeyFunction, ToneFunction
    from flexohr.paradigms.pitchspace.scale import find_scale_by_key_function

    result: Dict[str, str] = {}
    try:
        resolved = ohr.resolve()

        # root
        root_comp = next(
            resolved.components("b", depth=1, tone_function=ToneFunction.root), None
        )
        if root_comp is not None:
            result["root"] = _flx_to_agnn(root_comp.value.name)

        # bass
        bass_comp = next(
            resolved.components("b", depth=1, tone_function=ToneFunction.bass), None
        )
        if bass_comp is not None:
            result["bass"] = _flx_to_agnn(bass_comp.value.name)

        # tonkey
        tonicized = find_scale_by_key_function(resolved, KeyFunction.tonicized)
        if tonicized is not None:
            result["tonkey"] = _flx_to_agnn(tonicized.reference.value.name)
        else:
            local = find_scale_by_key_function(resolved, KeyFunction.local)
            if local is not None:
                result["tonkey"] = _flx_to_agnn(local.reference.value.name)

        # romanNumeral — root-position DCML chord portion (vocab excludes
        # inversion figures).  Inversion is a property of the inner chord
        # OHR (ohr.ohr()), so we must descend to the chord leaf before
        # overriding it — ohr.with_(inversion=...) on the outer OHR is a
        # no-op.
        try:
            from flexohr.harmony.harmony_enums import Inversion as _Inv
            chord_inner = ohr.ohr()
            root_pos = chord_inner.with_(inversion=_Inv.from_format("0", "analysisgnn"))
            dcml = root_pos.to_format("dcml")
        except Exception:
            dcml = ohr.to_format("dcml")
        parts = dcml.split("/")
        if parts:
            result["romanNumeral"] = parts[0]

    except Exception:
        pass

    return result


# ── Main enumeration ──


def enumerate_roman_numerals(
    context: ScoringContext,
    global_key: str,
    *,
    scorer: Optional[Scorer] = None,
    k: int = 3,
    top_n: int = 10,
    derive_validation: bool = False,
    localkey_mode_map: Optional[Dict[str, str]] = None,
) -> Tuple[List[RankedCandidate], EnumerationTrace]:
    """Enumerate and rank legal Roman-numeral candidates for a note group.

    Parameters
    ----------
    context : ScoringContext
        Note group to enumerate candidates for.
    global_key : str
        Global key of the piece (e.g. ``"G"``).
    scorer : Scorer or None
        Inner scorer for :class:`SeparateScorer`.
        Default: :class:`GeometricMeanScorer`.
    k : int
        Number of top predictions per core task to consider.
    top_n : int
        Maximum number of candidates to return.
    derive_validation : bool
        If ``True``, derive validation task labels (romanNumeral, root,
        bass, tonkey) from each OHR and include them in the
        :class:`SeparateScorer` evaluation.
    localkey_mode_map : dict or None
        Maps uppercase localkey labels (from the probabilities vocabulary)
        to mode-corrected labels (e.g. ``{"F": "f", "C": "c"}``).
        When ``None``, localkey labels are used as-is.

    Returns
    -------
    tuple[list[RankedCandidate], EnumerationTrace]
        Ranked candidates (1-based rank) and enumeration diagnostics.
    """
    if len(context.note_ids) == 0:
        return [], EnumerationTrace(0, 0, 0, 0, 0)

    inner_scorer = scorer or GeometricMeanScorer()
    sep_scorer = SeparateScorer(inner_scorer=inner_scorer)

    # 1. Extract top-k labels per core task
    available_tasks = set(context.tasks)
    task_labels: Dict[str, List[str]] = {}
    for task in ("quality", "degree1", "inversion", "localkey", "degree2"):
        if task in available_tasks:
            task_labels[task] = _group_top_k_labels(context, task, k)
        else:
            task_labels[task] = ["None"]

    # Filter "None" from quality candidates
    qualities = [q for q in task_labels["quality"] if q != "None"]
    if not qualities:
        return [], EnumerationTrace(0, 0, 0, 0, 0)

    degree1s = task_labels["degree1"]
    inversions = task_labels["inversion"]
    localkeys = task_labels["localkey"]
    # Apply mode correction to localkey labels (vocabulary is always uppercase)
    if localkey_mode_map:
        localkeys = [localkey_mode_map.get(lk, lk) for lk in localkeys]
    degree2s = task_labels["degree2"]

    # 2. Cartesian product
    combos = list(product(qualities, degree1s, inversions, localkeys, degree2s))
    num_raw = len(combos)
    pruned_reasons: Dict[str, int] = {}

    # 3. Prune and build OHRs
    valid_candidates: List[Tuple[OHR, str, Dict[str, str]]] = []
    seen_dcml: set = set()
    num_after_inv_prune = 0
    num_after_validation = 0

    from flexohr.harmony.harmony_enums import ChordQuality as _CQ

    import flexohr.codecs.analysisgnn  # noqa: F401 — activate codec

    for quality_l, degree1_l, inversion_l, localkey_l, degree2_l in combos:
        # Inversion legality pre-check
        try:
            quality_enum = _CQ.from_format(quality_l, "analysisgnn")
        except Exception:
            pruned_reasons["quality_parse"] = pruned_reasons.get("quality_parse", 0) + 1
            continue
        inv_int = int(float(inversion_l))
        if not _is_legal_inversion(quality_enum, inv_int):
            pruned_reasons["illegal_inversion"] = (
                pruned_reasons.get("illegal_inversion", 0) + 1
            )
            continue
        num_after_inv_prune += 1

        # Build OHR (includes InversionBassConsistency validation)
        ohr = _build_candidate_ohr(
            quality_l, degree1_l, inversion_l, localkey_l, degree2_l, global_key
        )
        if ohr is None:
            pruned_reasons["ohr_build_failed"] = (
                pruned_reasons.get("ohr_build_failed", 0) + 1
            )
            continue
        num_after_validation += 1

        # DCML dedup
        try:
            dcml = ohr.to_format("dcml")
        except Exception:
            pruned_reasons["dcml_format_failed"] = (
                pruned_reasons.get("dcml_format_failed", 0) + 1
            )
            continue
        if dcml in seen_dcml:
            pruned_reasons["duplicate_dcml"] = (
                pruned_reasons.get("duplicate_dcml", 0) + 1
            )
            continue
        seen_dcml.add(dcml)

        candidate_dict: Dict[str, str] = {
            "quality": quality_l,
            "degree1": degree1_l,
            "inversion": inversion_l,
            "localkey": localkey_l,
            "degree2": degree2_l,
        }
        if derive_validation:
            val_labels = _derive_validation_labels(ohr, global_key)
            candidate_dict.update(val_labels)

        valid_candidates.append((ohr, dcml, candidate_dict))

    num_after_dedup = len(valid_candidates)

    # 4. Score candidates
    ranked: List[RankedCandidate] = []
    for ohr, dcml, candidate_dict in valid_candidates:
        try:
            result = sep_scorer.score(context, candidate_dict)
        except Exception:
            continue
        ranked.append(RankedCandidate(ohr=ohr, dcml=dcml, candidate=candidate_dict,
                                      result=result, rank=0))

    # 5. Sort by core score descending
    ranked.sort(key=lambda c: c.result.core.score, reverse=True)

    # 6. Assign ranks and truncate
    for i, c in enumerate(ranked[:top_n]):
        c.rank = i + 1

    returned = ranked[:top_n]

    trace = EnumerationTrace(
        num_raw_combos=num_raw,
        num_after_inversion_prune=num_after_inv_prune,
        num_after_validation=num_after_validation,
        num_after_dedup=num_after_dedup,
        num_returned=len(returned),
        pruned_reasons=pruned_reasons,
    )

    return returned, trace
