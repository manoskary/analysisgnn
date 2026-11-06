from analysisgnn.inference.harmonic_state import get_default_harmonic_state_library


def test_harmonic_state_library_contains_common_tonal_states():
    library = get_default_harmonic_state_library()
    target = None
    for state in library.states:
        if state.localkey == "C" and state.roman_numeral == "I":
            target = state
            break
    assert target is not None
    assert target.class_ids["romanNumeral"] >= 0
    assert target.class_ids["localkey"] >= 0
    assert target.quality in {"major triad", "minor triad"}
    assert target.complete_rn in {"I", "C: I", "I64", "I6", "I7"} or target.complete_rn.startswith("I")


def test_harmonic_state_library_pair_lookup_is_deterministic():
    library = get_default_harmonic_state_library()
    sample = next(state for state in library.states if state.localkey == "C" and state.roman_numeral == "V7")
    again = library.get(sample.class_ids["romanNumeral"], sample.class_ids["localkey"])
    assert again is not None
    assert again.state_id == sample.state_id
    assert again.quality == sample.quality
