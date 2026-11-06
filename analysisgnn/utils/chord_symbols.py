from __future__ import annotations

from typing import Any, Dict, Optional

from music21 import key as m21key
from music21 import pitch as m21pitch
from music21 import roman as m21roman

from analysisgnn.utils.roman_decode import decode_roman_numeral


REQUIRED_COMPONENT_TASKS = ("localkey", "degree1", "degree2", "quality", "inversion")
OPTIONAL_SUPPORT_TASKS = ("root", "bass")
SPECIAL_CONTEXT_QUALITIES = {
    "augmented sixth",
    "Italian augmented sixth chord",
    "French augmented sixth chord",
    "German augmented sixth chord",
}
SUPPORTED_QUALITY_SUFFIXES = {
    "major triad": "",
    "minor triad": "m",
    "diminished triad": "dim",
    "augmented triad": "aug",
    "dominant seventh chord": "7",
    "major seventh chord": "maj7",
    "minor seventh chord": "m7",
    "half-diminished seventh chord": "m7b5",
    "diminished seventh chord": "dim7",
    "minor-augmented tetrachord": "m(maj7)",
    "incomplete dominant-seventh chord": "7",
}


def _value_or_none(value: Any) -> Optional[Any]:
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() == "none":
        return None
    return value


def _repo_spelling_to_music21(text: str) -> str:
    value = str(text).strip()
    if not value:
        return value
    head = value[0]
    tail = value[1:].replace("b", "-")
    return f"{head}{tail}"


def _spelling_to_display(text: Any) -> str:
    value = _value_or_none(text)
    if value is None:
        return ""
    return str(value).strip().replace("-", "b")


def _pitch_class(spelling: Any) -> Optional[int]:
    value = _value_or_none(spelling)
    if value is None:
        return None
    try:
        return int(m21pitch.Pitch(_repo_spelling_to_music21(str(value))).pitchClass)
    except Exception:
        return None


def _parse_inversion(value: Any) -> Optional[int]:
    value = _value_or_none(value)
    if value is None:
        return None
    if isinstance(value, int):
        return int(value)
    try:
        return int(float(str(value)))
    except Exception:
        mapping = {
            "root": 0,
            "root position": 0,
            "6": 1,
            "63": 1,
            "first inversion": 1,
            "64": 2,
            "second inversion": 2,
            "65": 1,
            "43": 2,
            "2": 3,
            "42": 3,
            "third inversion": 3,
        }
        return mapping.get(str(value).strip().lower())


def build_complete_rn(
    localkey: Any,
    degree1: Any,
    degree2: Any,
    quality: Any,
    inversion: Any,
) -> str:
    localkey = _value_or_none(localkey)
    degree1 = _value_or_none(degree1)
    quality = _value_or_none(quality)
    inversion = _parse_inversion(inversion)
    if localkey is None or degree1 is None or quality is None or inversion is None:
        return ""
    degree2 = _value_or_none(degree2) or "None"
    try:
        return decode_roman_numeral(
            degree1=str(degree1),
            degree2=str(degree2),
            inversion=inversion,
            quality=str(quality),
            localkey=str(localkey),
            include_key=False,
        )
    except Exception:
        return ""


def derive_absolute_harmony(complete_rn: Any, localkey: Any) -> Dict[str, Any]:
    rn_value = _value_or_none(complete_rn)
    key_value = _value_or_none(localkey)
    out = {
        "root": "",
        "bass": "",
        "root_pitch_class": None,
        "bass_pitch_class": None,
        "supported": False,
    }
    if rn_value is None or key_value is None:
        return out
    try:
        rn_obj = m21roman.RomanNumeral(
            str(rn_value),
            m21key.Key(_repo_spelling_to_music21(str(key_value))),
        )
        root_name = rn_obj.root().name
        bass_name = rn_obj.bass().name
        out.update(
            {
                "root": _spelling_to_display(root_name),
                "bass": _spelling_to_display(bass_name),
                "root_pitch_class": _pitch_class(root_name),
                "bass_pitch_class": _pitch_class(bass_name),
                "supported": True,
            }
        )
    except Exception:
        return out
    return out


def build_leadsheet_symbol(root: Any, quality: Any, bass: Any = None) -> str:
    root_text = _spelling_to_display(root)
    quality_text = str(_value_or_none(quality) or "")
    if not root_text or quality_text in SPECIAL_CONTEXT_QUALITIES:
        return ""
    suffix = SUPPORTED_QUALITY_SUFFIXES.get(quality_text)
    if suffix is None:
        return ""
    symbol = f"{root_text}{suffix}"
    bass_text = _spelling_to_display(bass)
    if bass_text and bass_text != root_text:
        symbol = f"{symbol}/{bass_text}"
    return symbol


def build_context_symbol(complete_rn: Any, fallback_rn: Any = None) -> str:
    primary = _value_or_none(complete_rn)
    fallback = _value_or_none(fallback_rn)
    if primary is not None:
        return str(primary).strip()
    if fallback is not None:
        return str(fallback).strip()
    return ""


def _task_entry(row: Dict[str, Any], task: str) -> Dict[str, Any]:
    tasks = row.get("tasks", {}) if isinstance(row, dict) else {}
    if isinstance(tasks, dict):
        entry = tasks.get(task, {})
        if isinstance(entry, dict):
            return entry
    return {}


def _compatible_or_ambiguous(
    derived_spelling: str,
    predicted_spelling: Any,
) -> tuple[str, bool, bool]:
    predicted_text = _spelling_to_display(predicted_spelling)
    if not predicted_text:
        return derived_spelling, False, False
    derived_pc = _pitch_class(derived_spelling)
    predicted_pc = _pitch_class(predicted_text)
    if derived_pc is None or predicted_pc is None:
        return derived_spelling, False, True
    if derived_pc != predicted_pc:
        return derived_spelling, False, True
    return predicted_text, predicted_text != derived_spelling, False


def build_beat_chord_symbol_row(row: Dict[str, Any]) -> Dict[str, Any]:
    quality = _task_entry(row, "quality").get("label")
    localkey = _task_entry(row, "localkey").get("label")
    degree1 = _task_entry(row, "degree1").get("label")
    degree2 = _task_entry(row, "degree2").get("label")
    inversion = _task_entry(row, "inversion").get(
        "class_id", _task_entry(row, "inversion").get("label")
    )
    rn_label = _task_entry(row, "romanNumeral").get("label", "")
    existing_context = row.get("romanNumeral_full", "")
    derived_context = build_complete_rn(
        localkey=localkey,
        degree1=degree1,
        degree2=degree2,
        quality=quality,
        inversion=inversion,
    )
    context_symbol = build_context_symbol(existing_context, derived_context or rn_label)

    core_missing = any(
        _value_or_none(value) is None
        for value in (localkey, degree1, quality, _parse_inversion(inversion))
    )
    core_conflict = any(
        bool(_task_entry(row, task).get("conflict_flag"))
        for task in REQUIRED_COMPONENT_TASKS
    )
    derived_failed = not derived_context

    if core_missing or derived_failed:
        return {
            "chordSymbol_abs": "",
            "chordSymbol_context": build_context_symbol(context_symbol, rn_label),
            "chordSymbol_supported": False,
            "chordSymbol_ambiguous": bool(
                core_missing or core_conflict or context_symbol
            ),
            "chordSymbol_source": "roman_fallback",
        }

    if str(_value_or_none(quality) or "") in SPECIAL_CONTEXT_QUALITIES:
        return {
            "chordSymbol_abs": "",
            "chordSymbol_context": build_context_symbol(
                context_symbol, derived_context
            ),
            "chordSymbol_supported": False,
            "chordSymbol_ambiguous": bool(core_conflict),
            "chordSymbol_source": "unsupported_special",
        }

    harmony = derive_absolute_harmony(derived_context, localkey)
    if not harmony.get("supported"):
        return {
            "chordSymbol_abs": "",
            "chordSymbol_context": build_context_symbol(context_symbol, rn_label),
            "chordSymbol_supported": False,
            "chordSymbol_ambiguous": True,
            "chordSymbol_source": "roman_fallback",
        }

    source = "component_derived"
    ambiguous = bool(core_conflict)
    root_display = harmony.get("root", "")
    bass_display = harmony.get("bass", "")

    predicted_root = _task_entry(row, "root").get("label")
    chosen_root, used_pred_root, root_conflict = _compatible_or_ambiguous(
        root_display, predicted_root
    )
    root_display = chosen_root
    ambiguous = ambiguous or root_conflict

    predicted_bass = _task_entry(row, "bass").get("label")
    chosen_bass, used_pred_bass, bass_conflict = _compatible_or_ambiguous(
        bass_display, predicted_bass
    )
    bass_display = chosen_bass
    ambiguous = ambiguous or bass_conflict

    if used_pred_root or used_pred_bass:
        source = "component_plus_root_bass"

    if context_symbol and derived_context and context_symbol != derived_context:
        ambiguous = True

    abs_symbol = build_leadsheet_symbol(root_display, quality, bass_display)
    return {
        "chordSymbol_abs": abs_symbol,
        "chordSymbol_context": build_context_symbol(context_symbol, derived_context),
        "chordSymbol_supported": bool(abs_symbol),
        "chordSymbol_ambiguous": ambiguous,
        "chordSymbol_source": source,
    }
