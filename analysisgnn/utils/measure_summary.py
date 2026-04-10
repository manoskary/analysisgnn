from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from analysisgnn.utils.roman_decode import decode_roman_numeral


TONAL_SPACE_LABELS: Tuple[str, ...] = (
    "Tonic",
    "Predominant",
    "Dominant",
    "Cadential",
    "Modulatory/Transitional",
    "Ambiguous",
)

DOMINANT_DEGREES = {"5", "7", "#4"}
PREDOMINANT_DEGREES = {"2", "4", "-2", "b2"}
TONIC_DEGREES = {"1", "3", "6"}
DOMINANT_QUALITIES = {
    "dominant seventh chord",
    "incomplete dominant-seventh chord",
    "diminished seventh chord",
    "half-diminished seventh chord",
}
PREDOMINANT_QUALITIES = {
    "augmented sixth",
    "Italian augmented sixth chord",
    "French augmented sixth chord",
    "German augmented sixth chord",
}


def _value_or_none(value: Any) -> Optional[Any]:
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() == "none":
        return None
    return value


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _normalize_distribution(scores: Dict[str, float]) -> Dict[str, float]:
    total = sum(max(0.0, float(scores.get(label, 0.0))) for label in TONAL_SPACE_LABELS)
    if total <= 0.0:
        uniform = 1.0 / float(len(TONAL_SPACE_LABELS))
        return {label: uniform for label in TONAL_SPACE_LABELS}
    return {
        label: max(0.0, float(scores.get(label, 0.0))) / total
        for label in TONAL_SPACE_LABELS
    }


def _normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    total = sum(max(0.0, float(value)) for value in weights.values())
    if total <= 0.0:
        return {}
    return {key: max(0.0, float(value)) / total for key, value in weights.items()}


def _normalized_entropy(distribution: Dict[str, float]) -> float:
    probs = [max(0.0, float(value)) for value in distribution.values()]
    nonzero = [p for p in probs if p > 0.0]
    if not nonzero:
        return 1.0
    entropy = -sum(p * math.log(p) for p in nonzero)
    max_entropy = math.log(max(len(probs), 1))
    if max_entropy <= 0.0:
        return 0.0
    return float(entropy / max_entropy)


def build_complete_rn_from_onset_row(row: Dict[str, Any]) -> str:
    localkey = _value_or_none(row.get("localkey"))
    degree1 = _value_or_none(row.get("degree1"))
    quality = _value_or_none(row.get("quality"))
    inversion = row.get("inversion")
    if localkey is None or degree1 is None or quality is None or inversion is None:
        return ""
    degree2 = _value_or_none(row.get("degree2")) or "None"
    try:
        return decode_roman_numeral(
            degree1=str(degree1),
            degree2=str(degree2),
            inversion=int(inversion),
            quality=str(quality),
            localkey=str(localkey),
            include_key=False,
        )
    except Exception:
        return ""


def classify_onset_function(
    row: Dict[str, Any],
    *,
    final_window: bool = False,
) -> str:
    degree1 = str(_value_or_none(row.get("degree1")) or "").replace("b", "-")
    degree2 = _value_or_none(row.get("degree2"))
    localkey = _value_or_none(row.get("localkey"))
    tonkey = _value_or_none(row.get("tonkey"))
    quality = str(_value_or_none(row.get("quality")) or "")
    cadence = str(_value_or_none(row.get("cadence")) or "")
    cadence_conf = _safe_float(row.get("cadence_confidence"), 0.0)

    if final_window and cadence and cadence.lower() not in {"none", "0"} and cadence_conf >= 0.5:
        return "Cadential"
    if tonkey is not None and localkey is not None and str(tonkey) != str(localkey):
        return "Modulatory/Transitional"
    if degree2 is not None:
        return "Modulatory/Transitional"
    if degree1 in DOMINANT_DEGREES or quality in DOMINANT_QUALITIES:
        return "Dominant"
    if degree1 in PREDOMINANT_DEGREES or quality in PREDOMINANT_QUALITIES:
        return "Predominant"
    if degree1 in TONIC_DEGREES:
        return "Tonic"
    return "Ambiguous"


def summarize_measure_rows(
    onset_rows: Sequence[Dict[str, Any]],
    measure_rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    onset_by_measure: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for row in onset_rows:
        onset_by_measure[row.get("measure")].append(row)

    out_rows: List[Dict[str, Any]] = []
    previous_cadential = False

    for measure_meta in measure_rows:
        measure_id = measure_meta.get("measure")
        rows = sorted(
            onset_by_measure.get(measure_id, []),
            key=lambda item: (_safe_float(item.get("onset_div"), 0.0), _safe_float(item.get("onset_beat"), 0.0)),
        )
        if not rows:
            out_rows.append(
                {
                    "measure": measure_id,
                    "measure_index": measure_meta.get("measure_index"),
                    "note_count": int(measure_meta.get("note_count", 0) or 0),
                    "onset_count": 0,
                    "measure_start_beat": None,
                    "measure_end_beat": None,
                    "tonal_space_label": "Ambiguous",
                    "tonal_space_confidence": 0.0,
                    "mixedness": 1.0,
                    "transition_flag": True,
                    "bar_localkey": "",
                    "bar_localkey_confidence": 0.0,
                    "tonicization_target": "",
                    "tonicization_confidence": 0.0,
                    "modulation_confidence": 1.0,
                    "cadential_intent": "none",
                    "cadential_confidence": 0.0,
                    "harmonic_stability": "volatile",
                    "harmonic_change_density": 0.0,
                    "top2_label": "Ambiguous",
                    "top2_share": 0.0,
                    "romanNumeral_full_mode": "",
                    "distribution": {label: 0.0 for label in TONAL_SPACE_LABELS},
                    "no_evidence": True,
                }
            )
            previous_cadential = False
            continue

        measure_end_div = _safe_float(measure_meta.get("measure_end_div"), 0.0)
        functional_scores = {label: 0.0 for label in TONAL_SPACE_LABELS}
        localkey_weights: Dict[str, float] = defaultdict(float)
        tonkey_weights: Dict[str, float] = defaultdict(float)
        degree2_weights: Dict[str, float] = defaultdict(float)
        rn_weights: Dict[str, float] = defaultdict(float)

        cadential_bonus = 0.0
        approach_bonus = 0.0
        tonicization_support = 0.0
        total_weight = 0.0
        final_window_weight = 0.0
        dominant_free_start = True
        harmonic_changes = 0.0
        last_rn = None
        measure_start_beat = None
        measure_end_beat = None

        for idx, row in enumerate(rows):
            span_div = max(_safe_float(row.get("span_div"), 1.0), 1.0)
            total_weight += span_div
            onset_beat = row.get("onset_beat")
            span_beat = row.get("span_beat")
            if onset_beat is not None:
                onset_beat_f = _safe_float(onset_beat, 0.0)
                measure_start_beat = onset_beat_f if measure_start_beat is None else min(measure_start_beat, onset_beat_f)
                if span_beat is not None:
                    measure_end_beat = max(
                        measure_end_beat if measure_end_beat is not None else onset_beat_f,
                        onset_beat_f + max(_safe_float(span_beat, 0.0), 0.0),
                    )
                else:
                    measure_end_beat = max(
                        measure_end_beat if measure_end_beat is not None else onset_beat_f,
                        onset_beat_f,
                    )

            position = 0.0
            onset_div = _safe_float(row.get("onset_div"), 0.0)
            if measure_end_div > onset_div:
                position = (onset_div - _safe_float(measure_meta.get("measure_start_div"), onset_div)) / max(
                    measure_end_div - _safe_float(measure_meta.get("measure_start_div"), onset_div),
                    1.0,
                )
            final_window = position >= 0.75
            label = classify_onset_function(row, final_window=final_window)
            functional_scores[label] += span_div

            localkey = _value_or_none(row.get("localkey"))
            if localkey is not None:
                localkey_weights[str(localkey)] += span_div
            tonkey = _value_or_none(row.get("tonkey"))
            if tonkey is not None:
                tonkey_weights[str(tonkey)] += span_div
                tonicization_support += span_div
            degree2 = _value_or_none(row.get("degree2"))
            if degree2 is not None:
                degree2_weights[str(degree2)] += span_div
                tonicization_support += span_div

            rn = str(_value_or_none(row.get("romanNumeral_full")) or "")
            if rn:
                rn_weights[rn] += span_div
            if last_rn is not None and rn and rn != last_rn:
                harmonic_changes += 1.0
            if rn:
                last_rn = rn

            cadence = str(_value_or_none(row.get("cadence")) or "")
            cadence_conf = _safe_float(row.get("cadence_confidence"), 0.0)
            if final_window:
                final_window_weight += span_div
                if cadence and cadence.lower() not in {"none", "0"}:
                    cadential_bonus += span_div * max(cadence_conf, 0.5)
                elif label == "Dominant":
                    approach_bonus += span_div * max(_safe_float(row.get("localkey_confidence"), 0.0), 0.5)

            if idx == 0 and label in {"Dominant", "Cadential"}:
                dominant_free_start = False

        if total_weight <= 0.0:
            total_weight = 1.0

        bar_localkey = ""
        bar_localkey_conf = 0.0
        if localkey_weights:
            bar_localkey, best_weight = max(localkey_weights.items(), key=lambda item: item[1])
            bar_localkey_conf = float(best_weight / total_weight)

        tonicization_target = ""
        tonicization_conf = 0.0
        tonkey_candidates = {k: v for k, v in tonkey_weights.items() if k != bar_localkey}
        if tonkey_candidates:
            tonicization_target, target_weight = max(tonkey_candidates.items(), key=lambda item: item[1])
            tonicization_conf = float(target_weight / total_weight)
        elif degree2_weights:
            tonicization_target, target_weight = max(degree2_weights.items(), key=lambda item: item[1])
            tonicization_conf = float(target_weight / total_weight)

        localkey_entropy = _normalized_entropy(_normalize_weights(localkey_weights)) if localkey_weights else 1.0
        functional_scores["Cadential"] += cadential_bonus
        functional_scores["Modulatory/Transitional"] += max(
            tonicization_conf * total_weight,
            localkey_entropy * 0.5 * total_weight,
        )
        distribution = _normalize_distribution(functional_scores)
        sorted_labels = sorted(
            distribution.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        tonal_space_label, tonal_space_conf = sorted_labels[0]
        top2_label, top2_share = sorted_labels[1] if len(sorted_labels) > 1 else ("Ambiguous", 0.0)
        mixedness = _normalized_entropy(distribution)
        modulation_confidence = float(max(1.0 - bar_localkey_conf, tonicization_conf))
        transition_flag = bool(
            mixedness >= 0.35
            or harmonic_changes >= 1.0
            or modulation_confidence >= 0.40
        )

        if cadential_bonus > 0.0:
            cadential_intent = "cadential"
        elif approach_bonus > 0.0:
            cadential_intent = "approach"
        elif previous_cadential and dominant_free_start:
            cadential_intent = "resolution"
        else:
            cadential_intent = "none"
        cadential_confidence = float(min(1.0, cadential_bonus / max(final_window_weight, 1.0))) if final_window_weight > 0.0 else 0.0

        if mixedness < 0.25 and harmonic_changes < 0.5:
            harmonic_stability = "stable"
        elif tonal_space_label == "Modulatory/Transitional" or modulation_confidence >= 0.40:
            harmonic_stability = "transitioning"
        else:
            harmonic_stability = "volatile"

        roman_numeral_mode = ""
        if rn_weights:
            roman_numeral_mode = max(rn_weights.items(), key=lambda item: item[1])[0]

        out_rows.append(
            {
                "measure": measure_id,
                "measure_index": measure_meta.get("measure_index"),
                "note_count": int(measure_meta.get("note_count", 0) or 0),
                "onset_count": len(rows),
                "measure_start_beat": measure_start_beat,
                "measure_end_beat": measure_end_beat,
                "tonal_space_label": tonal_space_label,
                "tonal_space_confidence": float(tonal_space_conf),
                "mixedness": float(mixedness),
                "transition_flag": transition_flag,
                "bar_localkey": bar_localkey,
                "bar_localkey_confidence": float(bar_localkey_conf),
                "tonicization_target": tonicization_target,
                "tonicization_confidence": float(tonicization_conf),
                "modulation_confidence": modulation_confidence,
                "cadential_intent": cadential_intent,
                "cadential_confidence": cadential_confidence,
                "harmonic_stability": harmonic_stability,
                "harmonic_change_density": float(harmonic_changes),
                "top2_label": top2_label,
                "top2_share": float(top2_share),
                "romanNumeral_full_mode": roman_numeral_mode,
                "distribution": {label: float(distribution.get(label, 0.0)) for label in TONAL_SPACE_LABELS},
                "no_evidence": False,
            }
        )
        previous_cadential = cadential_intent == "cadential"

    return {
        "level": "measure",
        "mode": "summary_v1",
        "rows": out_rows,
    }
