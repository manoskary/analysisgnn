import pandas as pd

from analysisgnn.utils.measure_summary import classify_onset_function, summarize_measure_rows
from examples.gradio_hybrid_analysis_app import _measure_payload_to_dataframe, _parse_predict_output


def _onset_row(
    *,
    measure=1,
    onset_div=0,
    onset_beat=0.0,
    span_div=4.0,
    localkey="C",
    degree1="1",
    degree2="None",
    quality="major triad",
    inversion=0,
    roman_numeral_full="I",
    cadence="None",
    cadence_confidence=0.0,
    tonkey=None,
):
    return {
        "measure": measure,
        "onset_div": onset_div,
        "onset_beat": onset_beat,
        "span_div": span_div,
        "localkey": localkey,
        "degree1": degree1,
        "degree2": degree2,
        "quality": quality,
        "inversion": inversion,
        "romanNumeral_full": roman_numeral_full,
        "cadence": cadence,
        "cadence_confidence": cadence_confidence,
        "tonkey": tonkey,
        "localkey_confidence": 0.9,
    }


def _measure_meta(measure=1, start=0.0, end=16.0, note_count=4):
    return {
        "measure": measure,
        "measure_index": measure - 1,
        "measure_start_div": start,
        "measure_end_div": end,
        "note_count": note_count,
    }


def test_classify_onset_function_maps_core_functional_classes():
    assert classify_onset_function(_onset_row(degree1="1")) == "Tonic"
    assert classify_onset_function(_onset_row(degree1="2", roman_numeral_full="ii6")) == "Predominant"
    assert classify_onset_function(_onset_row(degree1="5", quality="dominant seventh chord", roman_numeral_full="V7")) == "Dominant"
    assert classify_onset_function(_onset_row(degree1="5", cadence="PAC", cadence_confidence=0.9), final_window=True) == "Cadential"
    assert classify_onset_function(_onset_row(degree1="5", degree2="5", roman_numeral_full="V/V")) == "Modulatory/Transitional"


def test_measure_summary_prefers_single_clear_function():
    payload = summarize_measure_rows(
        onset_rows=[
            _onset_row(measure=1, onset_div=0, onset_beat=0.0, span_div=8.0, degree1="1", roman_numeral_full="I"),
            _onset_row(measure=1, onset_div=8, onset_beat=2.0, span_div=8.0, degree1="1", roman_numeral_full="I"),
        ],
        measure_rows=[_measure_meta(measure=1)],
    )
    row = payload["rows"][0]
    assert row["tonal_space_label"] == "Tonic"
    assert row["tonal_space_confidence"] > 0.7
    assert row["mixedness"] < 0.25
    assert row["romanNumeral_full_mode"] == "I"


def test_measure_summary_marks_mixed_function_measure():
    payload = summarize_measure_rows(
        onset_rows=[
            _onset_row(measure=1, onset_div=0, onset_beat=0.0, span_div=4.0, degree1="1", roman_numeral_full="I"),
            _onset_row(measure=1, onset_div=4, onset_beat=1.0, span_div=4.0, degree1="2", quality="minor triad", roman_numeral_full="ii"),
            _onset_row(measure=1, onset_div=8, onset_beat=2.0, span_div=4.0, degree1="5", quality="dominant seventh chord", roman_numeral_full="V7"),
        ],
        measure_rows=[_measure_meta(measure=1)],
    )
    row = payload["rows"][0]
    assert row["mixedness"] > 0.25
    assert row["harmonic_change_density"] >= 2.0
    assert row["transition_flag"] is True


def test_measure_summary_emits_empty_measure_row():
    payload = summarize_measure_rows(onset_rows=[], measure_rows=[_measure_meta(measure=3, note_count=0)])
    row = payload["rows"][0]
    assert row["measure"] == 3
    assert row["tonal_space_label"] == "Ambiguous"
    assert row["no_evidence"] is True
    assert row["mixedness"] == 1.0


def test_measure_payload_to_dataframe_and_parse_output():
    measure_payload = {
        "level": "measure",
        "mode": "summary_v1",
        "rows": [
            {
                "measure": 1,
                "measure_index": 0,
                "note_count": 4,
                "onset_count": 2,
                "measure_start_beat": 0.0,
                "measure_end_beat": 4.0,
                "tonal_space_label": "Tonic",
                "tonal_space_confidence": 0.8,
                "mixedness": 0.1,
                "transition_flag": False,
                "bar_localkey": "C",
                "bar_localkey_confidence": 1.0,
                "tonicization_target": "",
                "tonicization_confidence": 0.0,
                "modulation_confidence": 0.0,
                "cadential_intent": "none",
                "cadential_confidence": 0.0,
                "harmonic_stability": "stable",
                "harmonic_change_density": 0.0,
                "top2_label": "Ambiguous",
                "top2_share": 0.1,
                "romanNumeral_full_mode": "I",
                "no_evidence": False,
            }
        ],
    }
    df = _measure_payload_to_dataframe(measure_payload)
    assert isinstance(df, pd.DataFrame)
    assert list(df["tonal_space_label"]) == ["Tonic"]

    preds, trace, beat_payload, parsed_measure = _parse_predict_output(
        (
            {"romanNumeral": "ok"},
            {"enabled": False, "steps": []},
            {"level": "beat", "rows": [], "tasks": [], "mode": "mean"},
            measure_payload,
        ),
        enable_iterative=False,
        enable_beat=True,
        enable_measure=True,
    )
    assert preds == {"romanNumeral": "ok"}
    assert trace["enabled"] is False
    assert beat_payload["level"] == "beat"
    assert parsed_measure["level"] == "measure"
