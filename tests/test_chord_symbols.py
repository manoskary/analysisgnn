import pandas as pd

from analysisgnn.utils.chord_symbols import (
    build_beat_chord_symbol_row,
    build_complete_rn,
    build_leadsheet_symbol,
    derive_absolute_harmony,
)
from examples.gradio_hybrid_analysis_app import _beat_payload_to_dataframe


def _beat_row(
    *,
    roman_numeral_full="ii6",
    localkey="C",
    degree1="2",
    degree2="None",
    quality="minor triad",
    inversion=1,
    root="D",
    bass="F",
    include_root=True,
    include_bass=True,
    root_conflict=False,
    bass_conflict=False,
):
    tasks = {
        "localkey": {
            "label": localkey,
            "confidence": 0.9,
            "conflict_flag": False,
            "conflict_prob": 0.1,
        },
        "degree1": {
            "label": degree1,
            "confidence": 0.9,
            "conflict_flag": False,
            "conflict_prob": 0.1,
        },
        "degree2": {
            "label": degree2,
            "confidence": 0.9,
            "conflict_flag": False,
            "conflict_prob": 0.1,
        },
        "quality": {
            "label": quality,
            "confidence": 0.9,
            "conflict_flag": False,
            "conflict_prob": 0.1,
        },
        "inversion": {
            "label": str(inversion),
            "class_id": inversion,
            "confidence": 0.9,
            "conflict_flag": False,
            "conflict_prob": 0.1,
        },
    }
    if include_root:
        tasks["root"] = {
            "label": root,
            "confidence": 0.9,
            "conflict_flag": root_conflict,
            "conflict_prob": 0.1,
        }
    if include_bass:
        tasks["bass"] = {
            "label": bass,
            "confidence": 0.9,
            "conflict_flag": bass_conflict,
            "conflict_prob": 0.1,
        }
    return {
        "beat_id": 1,
        "beat_index": 0,
        "measure": 1,
        "onset_beat": 0.0,
        "note_count": 3,
        "romanNumeral_full": roman_numeral_full,
        "tasks": tasks,
    }


def test_build_complete_rn_from_components():
    assert build_complete_rn("C", "5", "None", "dominant seventh chord", 0) == "V7"


def test_derive_absolute_harmony_and_supported_suffix():
    harmony = derive_absolute_harmony("V7", "C")
    assert harmony["supported"] is True
    assert harmony["root"] == "G"
    assert harmony["bass"] == "G"
    assert (
        build_leadsheet_symbol(
            harmony["root"], "dominant seventh chord", harmony["bass"]
        )
        == "G7"
    )


def test_build_beat_chord_symbol_row_creates_slash_symbol():
    symbols = build_beat_chord_symbol_row(_beat_row())
    assert symbols["chordSymbol_abs"] == "Dm/F"
    assert symbols["chordSymbol_context"] == "ii6"
    assert symbols["chordSymbol_supported"] is True
    assert symbols["chordSymbol_ambiguous"] is False


def test_special_augmented_sixth_returns_context_only():
    row = _beat_row(
        roman_numeral_full="Ger65",
        degree1="#4",
        quality="German augmented sixth chord",
        inversion=1,
        root="F#",
        bass="Ab",
    )
    symbols = build_beat_chord_symbol_row(row)
    assert symbols["chordSymbol_abs"] == ""
    assert symbols["chordSymbol_context"] == "Ger65"
    assert symbols["chordSymbol_supported"] is False
    assert symbols["chordSymbol_source"] == "unsupported_special"


def test_root_disagreement_marks_ambiguous():
    row = _beat_row(root="E")
    symbols = build_beat_chord_symbol_row(row)
    assert symbols["chordSymbol_abs"] == "Dm/F"
    assert symbols["chordSymbol_ambiguous"] is True


def test_beat_payload_to_dataframe_adds_chord_symbol_columns():
    payload = {
        "rows": [_beat_row()],
        "tasks": [
            "localkey",
            "degree1",
            "degree2",
            "quality",
            "inversion",
            "root",
            "bass",
        ],
        "mode": "mean",
    }
    df = _beat_payload_to_dataframe(payload, payload["tasks"])
    assert isinstance(df, pd.DataFrame)
    assert list(df["chordSymbol_abs"]) == ["Dm/F"]
    assert list(df["chordSymbol_context"]) == ["ii6"]
    assert list(df["chordSymbol_supported"]) == [True]


def test_missing_component_falls_back_to_context_only():
    row = _beat_row()
    del row["tasks"]["quality"]
    symbols = build_beat_chord_symbol_row(row)
    assert symbols["chordSymbol_abs"] == ""
    assert symbols["chordSymbol_context"] == "ii6"
    assert symbols["chordSymbol_supported"] is False
    assert symbols["chordSymbol_source"] == "roman_fallback"
