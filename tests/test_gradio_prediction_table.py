import pandas as pd
import torch

from analysisgnn.inference.hybrid_predictor import _decode_task_predictions
from examples.gradio_hybrid_analysis_app import _prepare_prediction_table_for_display


def test_prediction_table_rounds_confidence_and_places_complete_rn_before_cadence():
    df = pd.DataFrame(
        {
            "row": [0],
            "note_id": ["n1"],
            "onset_beat": [0.0],
            "measure": [1],
            "duration_beat": [1.0],
            "pitch_spelling": ["C4"],
            "pitch_midi": [60],
            "cadence": ["PAC"],
            "cadence_confidence": [0.98765],
            "localkey": ["C"],
            "localkey_confidence": [0.12345],
            "romanNumeral_full": ["I"],
        }
    )

    out = _prepare_prediction_table_for_display(df)

    assert out.columns.get_loc("romanNumeral_full") == out.columns.get_loc("pitch_midi") + 1
    assert out.columns.get_loc("romanNumeral_full") < out.columns.get_loc("cadence")
    assert out.loc[0, "cadence_confidence"] == 0.988
    assert out.loc[0, "localkey_confidence"] == 0.123


def test_cadence_decoder_maps_positive_classes_to_named_cadences():
    logits = torch.tensor(
        [
            [0.0, 10.0, 0.0, 0.0],
            [0.0, 0.0, 10.0, 0.0],
            [0.0, 0.0, 0.0, 10.0],
        ]
    )

    decoded, _, class_ids = _decode_task_predictions("cadence", logits)

    assert decoded.tolist() == ["PAC", "IAC", "HC"]
    assert class_ids.tolist() == [1, 2, 3]
