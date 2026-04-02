import torch

from analysisgnn.inference.beam_decoder import (
    _class_labels_for_task,
    build_smoothed_note_probs_from_class_ids,
    decode_onset_beam,
)
from analysisgnn.inference.harmonic_state import get_default_harmonic_state_library
from analysisgnn.inference.harmonic_state import get_default_component_harmonic_state_library


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


def test_decode_onset_beam_structured_v2_returns_state_aligned_outputs():
    library = get_default_harmonic_state_library()
    state = next(s for s in library.states if s.localkey == "C" and s.roman_numeral == "I")
    num_notes = 6
    onset_ids = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
    task_num_classes = {
        "romanNumeral": 184,
        "localkey": 50,
        "quality": 16,
        "inversion": 4,
        "degree1": 22,
        "degree2": 22,
        "root": 38,
        "bass": 38,
        "tpc_in_label": 2,
    }
    note_prob_dict = {}
    for task, num_classes in task_num_classes.items():
        preferred = state.class_ids.get(task, 0)
        note_prob_dict[task] = _make_note_probs(num_notes, num_classes, preferred)
    payload = decode_onset_beam(
        note_prob_dict=note_prob_dict,
        onset_ids=onset_ids,
        task_num_classes=task_num_classes,
        legal_rn_set=None,
        spec={"enabled": True, "version": "structured_v2", "beam_width": 4, "nbest": 2},
    )
    assert payload is not None
    assert payload["beam_trace"]["version"] == "structured_v2"
    assert payload["onset_values"].shape[0] == 3
    assert payload["note_class_ids"]["romanNumeral"].shape[0] == num_notes
    assert payload["note_class_ids"]["romanNumeral"][0].item() == state.class_ids["romanNumeral"]
    assert payload["note_class_ids"]["localkey"][0].item() == state.class_ids["localkey"]
    assert len(payload["beam_trace"]["nbest"]) >= 1


def test_component_harmonic_state_library_contains_basic_tonic_state():
    library = get_default_component_harmonic_state_library()
    state = next(
        s
        for s in library.states
        if s.localkey == "C"
        and s.degree1 == "1"
        and s.degree2 == "None"
        and s.quality == "major triad"
        and int(s.inversion) == 0
    )
    fetched = library.get(
        state.class_ids["localkey"],
        state.class_ids["degree1"],
        state.class_ids["degree2"],
        state.class_ids["quality"],
        state.class_ids["inversion"],
    )
    assert fetched is not None
    assert fetched.complete_rn


def test_decode_onset_beam_component_v3_ignores_rn_head_and_returns_component_states():
    library = get_default_component_harmonic_state_library()
    state = next(
        s
        for s in library.states
        if s.localkey == "C"
        and s.degree1 == "1"
        and s.degree2 == "None"
        and s.quality == "major triad"
        and int(s.inversion) == 0
    )
    num_notes = 6
    onset_ids = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
    task_num_classes = {
        "romanNumeral": 184,
        "localkey": 50,
        "quality": 16,
        "inversion": 4,
        "degree1": 22,
        "degree2": 22,
        "root": 38,
        "bass": 38,
        "tpc_in_label": 2,
    }
    note_prob_dict = {
        "romanNumeral": _make_note_probs(num_notes, 184, 0),
        "localkey": _make_note_probs(num_notes, 50, state.class_ids["localkey"]),
        "quality": _make_note_probs(num_notes, 16, state.class_ids["quality"]),
        "inversion": _make_note_probs(num_notes, 4, state.class_ids["inversion"]),
        "degree1": _make_note_probs(num_notes, 22, state.class_ids["degree1"]),
        "degree2": _make_note_probs(num_notes, 22, state.class_ids["degree2"]),
        "root": _make_note_probs(num_notes, 38, state.class_ids.get("root", 0)),
        "bass": _make_note_probs(num_notes, 38, state.class_ids.get("bass", 0)),
        "tpc_in_label": _make_note_probs(num_notes, 2, 1),
    }
    payload = decode_onset_beam(
        note_prob_dict=note_prob_dict,
        onset_ids=onset_ids,
        task_num_classes=task_num_classes,
        legal_rn_set=None,
        spec={"enabled": True, "version": "component_v3", "beam_width": 4, "nbest": 2},
    )
    assert payload is not None
    assert payload["beam_trace"]["version"] == "component_v3"
    assert payload["onset_values"].shape[0] == 3
    assert payload["note_class_ids"]["localkey"][0].item() == state.class_ids["localkey"]
    assert payload["note_class_ids"]["degree1"][0].item() == state.class_ids["degree1"]
    assert payload["note_class_ids"]["quality"][0].item() == state.class_ids["quality"]
    assert payload["onset_margin"].shape[0] == 3
    assert len(payload["onset_complete_rn"]) == 3
