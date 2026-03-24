import torch

from analysisgnn.inference.beam_decoder import (
    _class_labels_for_task,
    build_smoothed_note_probs_from_class_ids,
    decode_onset_beam,
)


def _make_note_probs(num_notes: int, num_classes: int, preferred_idx: int) -> torch.Tensor:
    probs = torch.full((num_notes, num_classes), 1e-4, dtype=torch.float32)
    probs[:, preferred_idx] = 0.8
    if num_classes > 1:
        probs[:, (preferred_idx + 1) % num_classes] = 0.2
    probs = probs / probs.sum(dim=-1, keepdim=True)
    return probs


def test_class_labels_accept_sequence_and_pad_tail():
    degree_labels = _class_labels_for_task("degree1", 4)
    assert isinstance(degree_labels, list)
    assert degree_labels[0] == "-1"

    rn_labels = _class_labels_for_task("romanNumeral", 186)
    assert rn_labels[0] == "none"
    assert len(rn_labels) == 186
    # Tail padding should keep explicit class id placeholders.
    assert rn_labels[-1] == 185


def test_decode_onset_beam_returns_note_aligned_outputs():
    num_notes = 5
    onset_ids = torch.tensor([0, 0, 2, 2, 5], dtype=torch.long)
    task_num_classes = {
        "romanNumeral": 6,
        "localkey": 6,
        "quality": 6,
        "inversion": 4,
        "degree1": 6,
        "degree2": 6,
    }
    note_prob_dict = {
        "romanNumeral": _make_note_probs(num_notes, 6, 1),
        "localkey": _make_note_probs(num_notes, 6, 0),
        "quality": _make_note_probs(num_notes, 6, 0),
        "inversion": _make_note_probs(num_notes, 4, 0),
        "degree1": _make_note_probs(num_notes, 6, 0),
        "degree2": _make_note_probs(num_notes, 6, 0),
    }

    payload = decode_onset_beam(
        note_prob_dict=note_prob_dict,
        onset_ids=onset_ids,
        task_num_classes=task_num_classes,
        legal_rn_set=None,
        spec={"enabled": True, "beam_width": 3},
    )

    assert payload is not None
    assert payload["tasks"] == ["romanNumeral", "localkey", "quality", "inversion", "degree1", "degree2"]
    assert payload["onset_values"].shape[0] == 3
    assert payload["note_inverse"].shape[0] == num_notes
    assert payload["beam_trace"]["num_onsets"] == 3
    assert len(payload["beam_trace"]["steps"]) == 3

    for task in payload["tasks"]:
        note_ids = payload["note_class_ids"][task]
        onset_ids_task = payload["onset_class_ids"][task]
        assert note_ids.shape[0] == num_notes
        assert onset_ids_task.shape[0] == 3
        assert torch.equal(note_ids, onset_ids_task[payload["note_inverse"]])
        conf = payload["note_confidence"][task]
        assert torch.all(conf >= 0.0)
        assert torch.all(conf <= 1.0)


def test_decode_onset_beam_legality_fallback_keeps_path():
    num_notes = 4
    onset_ids = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    task_num_classes = {
        "romanNumeral": 6,
        "localkey": 6,
        "quality": 6,
        "inversion": 4,
        "degree1": 6,
        "degree2": 6,
    }
    note_prob_dict = {
        "romanNumeral": _make_note_probs(num_notes, 6, 1),
        "localkey": _make_note_probs(num_notes, 6, 0),
        "quality": _make_note_probs(num_notes, 6, 0),
        "inversion": _make_note_probs(num_notes, 4, 0),
        "degree1": _make_note_probs(num_notes, 6, 0),
        "degree2": _make_note_probs(num_notes, 6, 0),
    }
    # Impossible legal set enforces constrained-empty path and fallback to unconstrained best.
    payload = decode_onset_beam(
        note_prob_dict=note_prob_dict,
        onset_ids=onset_ids,
        task_num_classes=task_num_classes,
        legal_rn_set={"__impossible__"},
        spec={
            "enabled": True,
            "beam_width": 2,
            "topk_by_task": {
                "romanNumeral": 1,
                "localkey": 1,
                "quality": 1,
                "inversion": 1,
                "degree1": 1,
                "degree2": 1,
            },
        },
    )
    assert payload is not None
    assert payload["beam_trace"]["enabled"] is True
    assert payload["beam_trace"]["num_onsets"] == 2


def test_build_smoothed_note_probs_from_class_ids_normalizes():
    class_ids = torch.tensor([0, 2, 1, 2], dtype=torch.long)
    probs = build_smoothed_note_probs_from_class_ids(class_ids, num_classes=3, off_prob=0.03)
    assert probs.shape == (4, 3)
    row_sums = probs.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)
    winners = probs.argmax(dim=-1)
    assert torch.equal(winners, class_ids)
