"""
Bar-level texture descriptors computed from note arrays or partitura parts.

This module adapts the descriptor ideas from the Symbolic Texture Dataset
without relying on music21. It operates on partitura note arrays or simple
note lists and is optimized for batched, per-bar processing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

CONSONANT_INTERVALS = {0, 3, 4, 5, 7, 8, 9}
HARMONIC_THIRD_SIXTH = {3, 4, 8, 9}
HARMONIC_FOURTH_FIFTH = {5, 7}
HARMONIC_OCTAVE = {12}

INTERVAL_GROUPS = {
    "seconds": {1, 2, -1, -2},
    "thirds": {3, 4, -3, -4},
    "fourths_fifths": {5, 7, -5, -7},
    "octaves": {12, -12},
    "unison": {0},
}


@dataclass
class BarSlice:
    start: float
    duration: float
    pitches: List[int]


def note_list_to_array(
    notes: Sequence[Dict[str, float | int]],
    *,
    default_ts_beats: int = 4,
    default_ts_beat_type: int = 4,
) -> np.ndarray:
    """Convert a list of note dicts into a structured array."""
    onset = np.asarray([n.get("onset_beat", n.get("onset", 0.0)) for n in notes], dtype=float)
    duration = np.asarray([n.get("duration_beat", n.get("duration", 0.0)) for n in notes], dtype=float)
    pitch = np.asarray([n.get("pitch", n.get("midi", 0)) for n in notes], dtype=int)
    ts_beats = np.asarray([n.get("ts_beats", default_ts_beats) for n in notes], dtype=int)
    ts_beat_type = np.asarray([n.get("ts_beat_type", default_ts_beat_type) for n in notes], dtype=int)

    dtype = [
        ("onset_beat", "f4"),
        ("duration_beat", "f4"),
        ("pitch", "i4"),
        ("ts_beats", "i4"),
        ("ts_beat_type", "i4"),
    ]
    out = np.zeros(len(notes), dtype=dtype)
    out["onset_beat"] = onset
    out["duration_beat"] = duration
    out["pitch"] = pitch
    out["ts_beats"] = ts_beats
    out["ts_beat_type"] = ts_beat_type
    return out


def note_array_from_part(part, include_grace_notes: bool = False) -> np.ndarray:
    """Return a partitura note array with time signature metadata."""
    import partitura
    import partitura.score

    if isinstance(part, list) or isinstance(part, partitura.score.PartGroup):
        part = partitura.score.merge_parts(part)
    return part.note_array(
        include_time_signature=True,
        include_grace_notes=include_grace_notes,
        include_metrical_position=True,
        include_pitch_spelling=False,
    )


def _get_field(note_array: np.ndarray, name: str, default: float | int) -> np.ndarray:
    if name in note_array.dtype.names:
        return note_array[name]
    return np.full(len(note_array), default)


def _stats(values: np.ndarray) -> Tuple[float, float, float, float, float]:
    if values.size == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    sorted_values = np.sort(values)
    avg = float(sorted_values.mean())
    std = float(np.mean(np.abs(sorted_values - avg)))
    min_v = float(sorted_values[0])
    max_v = float(sorted_values[-1])
    if sorted_values.size % 2 == 0:
        med = float((sorted_values[sorted_values.size // 2] + sorted_values[sorted_values.size // 2 - 1]) / 2)
    else:
        med = float(sorted_values[(sorted_values.size - 1) // 2])
    return avg, std, min_v, max_v, med


def _weighted_stats(values: np.ndarray, weights: np.ndarray) -> Tuple[float, float, float, float, float]:
    if values.size == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    total_weight = float(np.sum(weights))
    if total_weight <= 0:
        return 0.0, 0.0, float(values.min()), float(values.max()), float(values.mean())
    avg = float(np.sum(values * weights) / total_weight)
    std = float(np.sum(np.abs(values - avg) * weights) / total_weight)
    min_v = float(values[0])
    max_v = float(values[-1])

    cumulative = np.cumsum(weights)
    median_target = total_weight / 2.0
    idx = np.searchsorted(cumulative, median_target)
    idx = min(idx, values.size - 1)
    if idx > 0 and np.isclose(cumulative[idx], median_target, atol=1e-6):
        med = float((values[idx] + values[idx - 1]) / 2.0)
    else:
        med = float(values[idx])
    return avg, std, min_v, max_v, med


def _compute_measure_boundaries(note_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    onset = _get_field(note_array, "onset_beat", 0.0).astype(float)
    duration = _get_field(note_array, "duration_beat", 0.0).astype(float)
    ts_beats = _get_field(note_array, "ts_beats", 4).astype(float)
    ts_beat_type = _get_field(note_array, "ts_beat_type", 4).astype(float)

    if onset.size == 0:
        return np.asarray([0.0]), np.asarray([0.0])

    order = np.argsort(onset)
    onset = onset[order]
    duration = duration[order]
    ts_beats = ts_beats[order]
    ts_beat_type = ts_beat_type[order]

    change_idx = [0]
    for i in range(1, onset.size):
        if ts_beats[i] != ts_beats[i - 1] or ts_beat_type[i] != ts_beat_type[i - 1]:
            change_idx.append(i)

    starts: List[float] = []
    ends: List[float] = []
    piece_end = float(np.max(onset + duration))

    for idx, start_idx in enumerate(change_idx):
        segment_start = float(onset[start_idx])
        beats_per_bar = float(ts_beats[start_idx])
        next_start = piece_end
        if idx + 1 < len(change_idx):
            next_start = float(onset[change_idx[idx + 1]])

        current = segment_start
        if not starts:
            starts.append(current)
        while current + beats_per_bar <= next_start + 1e-6:
            current += beats_per_bar
            starts.append(current)

        if current < next_start:
            starts.append(next_start)

    starts = sorted(set(starts))
    starts = [s for s in starts if s < piece_end - 1e-6]
    if not starts:
        return np.asarray([0.0]), np.asarray([piece_end])

    ends = starts[1:] + [piece_end]
    return np.asarray(starts), np.asarray(ends)


def _compute_onsets(onset: np.ndarray, pitch: np.ndarray) -> Dict[float, List[int]]:
    if onset.size == 0:
        return {}
    order = np.argsort(onset)
    onset_sorted = onset[order]
    pitch_sorted = pitch[order]
    unique_times, idx_start, counts = np.unique(onset_sorted, return_index=True, return_counts=True)
    onset_dict: Dict[float, List[int]] = {}
    for time, start_idx, count in zip(unique_times, idx_start, counts):
        onset_dict[float(time)] = pitch_sorted[start_idx : start_idx + count].tolist()
    return onset_dict


def _compute_slices(
    onset: np.ndarray,
    duration: np.ndarray,
    pitch: np.ndarray,
    bar_length: float,
) -> List[BarSlice]:
    if onset.size == 0:
        return [BarSlice(start=0.0, duration=bar_length, pitches=[])]

    events: List[Tuple[float, int, int]] = []
    for start, dur, p in zip(onset, duration, pitch):
        end = start + dur
        events.append((start, 1, int(p)))
        events.append((end, 0, int(p)))

    events.sort(key=lambda x: (x[0], x[1]))

    active: Dict[int, int] = {}
    slices: List[BarSlice] = []
    current_time = 0.0
    idx = 0

    while idx < len(events):
        time = events[idx][0]
        if time > current_time:
            pitches = sorted(active.keys())
            slices.append(BarSlice(start=current_time, duration=time - current_time, pitches=pitches))
            current_time = time
        while idx < len(events) and np.isclose(events[idx][0], time):
            _, event_order, p = events[idx]
            if event_order == 0:
                count = active.get(p, 0) - 1
                if count <= 0:
                    active.pop(p, None)
                else:
                    active[p] = count
            else:
                active[p] = active.get(p, 0) + 1
            idx += 1

    if current_time < bar_length:
        slices.append(BarSlice(start=current_time, duration=bar_length - current_time, pitches=sorted(active.keys())))

    return slices


def _n_gaps(chord: List[int]) -> int:
    if len(chord) < 2:
        return 0
    chord_sorted = sorted(chord)
    return sum(1 for a, b in zip(chord_sorted[:-1], chord_sorted[1:]) if b - a > 5)


def _harmonicity(chord: List[int]) -> float:
    if len(chord) <= 1:
        return 1.0
    count = 0
    total = 0
    for i in range(len(chord)):
        for j in range(i + 1, len(chord)):
            total += 1
            interval = abs(chord[j] - chord[i]) % 12
            if interval in CONSONANT_INTERVALS:
                count += 1
    return float(count / total) if total else 1.0


def _harmonic_intervals(onset_dict: Dict[float, List[int]]) -> Tuple[float, float, float]:
    if not onset_dict:
        return 0.0, 0.0, 0.0
    total = len(onset_dict)
    third_sixth = 0
    fourth_fifth = 0
    octave = 0
    for chord in onset_dict.values():
        has_third_sixth = False
        has_fourth_fifth = False
        has_octave = False
        for i in range(len(chord)):
            for j in range(i + 1, len(chord)):
                interval = abs(chord[j] - chord[i])
                if interval in HARMONIC_THIRD_SIXTH:
                    has_third_sixth = True
                elif interval in HARMONIC_FOURTH_FIFTH:
                    has_fourth_fifth = True
                elif interval in HARMONIC_OCTAVE:
                    has_octave = True
        third_sixth += int(has_third_sixth)
        fourth_fifth += int(has_fourth_fifth)
        octave += int(has_octave)
    return third_sixth / total, fourth_fifth / total, octave / total


def _melodic_intervals(
    slice_dict: Dict[float, List[int]],
    onset_dict: Dict[float, List[int]],
    interval_key: str,
) -> float:
    targets = INTERVAL_GROUPS[interval_key]
    slice_times = sorted(slice_dict.keys())
    if not slice_times or not onset_dict:
        return 0.0

    total_notes = 0
    matched = 0
    time_index = {t: i for i, t in enumerate(slice_times)}
    for onset_time, pitches in onset_dict.items():
        idx = time_index.get(onset_time)
        if idx is None or idx <= 0:
            continue
        prev_pitches = slice_dict[slice_times[idx - 1]]
        if not prev_pitches:
            total_notes += len(pitches)
            continue
        prev_set = set(prev_pitches)
        for pitch in pitches:
            total_notes += 1
            for delta in targets:
                if pitch + delta in prev_set:
                    matched += 1
                    break
    return matched / total_notes if total_notes else 0.0


def _onset_synchrony(
    onset_dict: Dict[float, List[int]],
    slice_dict: Dict[float, List[int]],
    bar_length: float,
    *,
    function: str = "lin",
) -> float:
    if bar_length <= 0:
        return 0.0
    if function == "lin":
        v_function = lambda x: abs(2 * x - 1)
    elif function == "cos":
        v_function = lambda x: (np.cos(2 * np.pi * x) / 2) + 0.5
    else:
        raise ValueError("function must be 'lin' or 'cos'")

    slice_times = sorted(slice_dict.keys())
    slice_times.append(bar_length)
    onset_counts = {t: len(p) for t, p in onset_dict.items()}

    total = 0.0
    for i in range(len(slice_times) - 1):
        start = slice_times[i]
        end = slice_times[i + 1]
        notes = len(slice_dict.get(start, []))
        onsets = onset_counts.get(start, 0)
        if notes == 0:
            local = 0.5
        else:
            local = v_function(onsets / notes)
        total += (end - start) * local
    return total / bar_length


def _semblant_motions(slice_dict: Dict[float, List[int]]) -> float:
    slice_times = sorted(slice_dict.keys())
    nb_motions = 0
    nb_semblant = 0
    for i in range(len(slice_times) - 1):
        current = slice_dict[slice_times[i]]
        next_slice = slice_dict[slice_times[i + 1]]
        current_pitches = [p for p in current if p not in next_slice]
        next_pitches = [p for p in next_slice if p not in current]
        if not current_pitches or not next_pitches:
            continue
        if min(current_pitches) < min(next_pitches):
            if max(current_pitches) < max(next_pitches):
                nb_semblant += 1
                nb_motions += 1
            elif max(current_pitches) > max(next_pitches):
                nb_motions += 1
        elif min(current_pitches) > min(next_pitches):
            if max(current_pitches) > max(next_pitches):
                nb_semblant += 1
                nb_motions += 1
            elif max(current_pitches) < max(next_pitches):
                nb_motions += 1
    if nb_motions == 0:
        return 1.0
    return nb_semblant / nb_motions


def _bar_feature_names() -> List[str]:
    return [
        "bar_start_beat",
        "bar_end_beat",
        "bar_length_beats",
        "has_even_meter",
        "n_notes",
        "n_notes_per_beat",
        "n_active_notes",
        "n_pitches",
        "n_pitches_per_beat",
        "n_pitchclasses",
        "novelty",
        "pitch_avg",
        "pitch_std",
        "pitch_min",
        "pitch_max",
        "pitch_med",
        "duration_avg",
        "duration_std",
        "duration_min",
        "duration_max",
        "duration_med",
        "n_onsets",
        "n_onsets_per_beat",
        "onset_reg_avg",
        "onset_reg_std",
        "onset_reg_min",
        "onset_reg_max",
        "onset_reg_med",
        "onset_n_voices_avg",
        "onset_n_voices_std",
        "onset_n_voices_min",
        "onset_n_voices_max",
        "onset_n_voices_med",
        "onset_n_gaps_avg",
        "onset_n_gaps_std",
        "onset_n_gaps_min",
        "onset_n_gaps_max",
        "onset_n_gaps_med",
        "harmonic_third_sixth",
        "harmonic_fourth_fifth",
        "harmonic_octave",
        "n_slices",
        "n_slices_per_beat",
        "prop_silence",
        "longest_silence",
        "slice_width_avg",
        "slice_width_std",
        "slice_width_min",
        "slice_width_max",
        "slice_width_med",
        "slice_n_voices_avg",
        "slice_n_voices_std",
        "slice_n_voices_min",
        "slice_n_voices_max",
        "slice_n_voices_med",
        "slice_n_gaps_avg",
        "slice_n_gaps_std",
        "slice_n_gaps_min",
        "slice_n_gaps_max",
        "slice_n_gaps_med",
        "harmonicity_avg",
        "harmonicity_std",
        "harmonicity_min",
        "harmonicity_max",
        "harmonicity_med",
        "melodic_seconds",
        "melodic_thirds",
        "melodic_fourths_fifths",
        "melodic_octaves",
        "melodic_unison",
        "onset_synchrony",
        "semblant_motion",
    ]


def compute_bar_texture(
    note_array: np.ndarray,
    *,
    bar_starts: Optional[np.ndarray] = None,
    bar_ends: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Compute bar-level texture descriptors from a note array."""
    onset = _get_field(note_array, "onset_beat", 0.0).astype(float)
    duration = _get_field(note_array, "duration_beat", 0.0).astype(float)
    pitch = _get_field(note_array, "pitch", 0).astype(int)
    ts_beats = _get_field(note_array, "ts_beats", 4).astype(float)

    if bar_starts is None or bar_ends is None:
        bar_starts, bar_ends = _compute_measure_boundaries(note_array)

    feature_names = _bar_feature_names()
    features = np.zeros((len(bar_starts), len(feature_names)), dtype=float)

    for idx, (start, end) in enumerate(zip(bar_starts, bar_ends)):
        bar_length = float(end - start)
        if bar_length <= 0:
            continue

        onset_mask = (onset >= start) & (onset < end)
        active_mask = (onset < end) & ((onset + duration) > start)

        onset_times = onset[onset_mask] - start
        onset_pitches = pitch[onset_mask]
        active_onset = np.maximum(onset[active_mask], start) - start
        active_end = np.minimum(onset[active_mask] + duration[active_mask], end) - start
        active_pitches = pitch[active_mask]
        active_durations = active_end - active_onset

        onset_dict = _compute_onsets(onset_times, onset_pitches)
        slices = _compute_slices(active_onset, active_durations, active_pitches, bar_length)
        slice_dict = {s.start: s.pitches for s in slices}

        n_notes = float(onset_pitches.size)
        n_active = float(active_pitches.size)
        n_pitches = float(len(np.unique(active_pitches))) if active_pitches.size else 0.0
        n_pitchclasses = float(len(np.unique(active_pitches % 12))) if active_pitches.size else 0.0
        novelty = n_pitches / n_notes if n_notes else 0.0

        pitch_stats = _weighted_stats(active_pitches.astype(float), active_durations)
        duration_stats = _stats(active_durations)

        n_onsets = float(len(onset_dict))
        onset_times_sorted = np.sort(list(onset_dict.keys())) if onset_dict else np.asarray([])
        if onset_times_sorted.size:
            intervals = np.diff(np.r_[onset_times_sorted, bar_length])
        else:
            intervals = np.asarray([bar_length])
        onset_reg_stats = _stats(intervals)

        onset_n_voices = np.asarray([len(p) for p in onset_dict.values()], dtype=float)
        onset_n_voices_stats = _stats(onset_n_voices)

        onset_n_gaps = np.asarray([_n_gaps(p) for p in onset_dict.values()], dtype=float)
        onset_n_gaps_stats = _stats(onset_n_gaps)

        third_sixth, fourth_fifth, octave = _harmonic_intervals(onset_dict)

        n_slices = float(len(slices))
        slice_durations = np.asarray([s.duration for s in slices], dtype=float)
        slice_chords = [s.pitches for s in slices]

        silence_mask = np.asarray([len(chord) == 0 for chord in slice_chords])
        prop_silence = float(np.sum(slice_durations[silence_mask]) / bar_length) if bar_length else 0.0
        longest_silence = float(slice_durations[silence_mask].max()) / bar_length if silence_mask.any() else 0.0

        slice_widths = np.asarray(
            [
                0 if len(chord) == 0 else (1 if len(chord) == 1 else max(chord) - min(chord) + 1)
                for chord in slice_chords
            ],
            dtype=float,
        )
        slice_width_stats = _weighted_stats(slice_widths, slice_durations)

        slice_n_voices = np.asarray([len(chord) for chord in slice_chords], dtype=float)
        slice_n_voices_stats = _weighted_stats(slice_n_voices, slice_durations)

        slice_n_gaps = np.asarray([_n_gaps(chord) for chord in slice_chords], dtype=float)
        slice_n_gaps_stats = _weighted_stats(slice_n_gaps, slice_durations)

        harmonicities = np.asarray([_harmonicity(chord) for chord in slice_chords], dtype=float)
        harmonicity_stats = _weighted_stats(harmonicities, slice_durations)

        melodic_seconds = _melodic_intervals(slice_dict, onset_dict, "seconds")
        melodic_thirds = _melodic_intervals(slice_dict, onset_dict, "thirds")
        melodic_fourths_fifths = _melodic_intervals(slice_dict, onset_dict, "fourths_fifths")
        melodic_octaves = _melodic_intervals(slice_dict, onset_dict, "octaves")
        melodic_unison = _melodic_intervals(slice_dict, onset_dict, "unison")

        onset_synchrony = _onset_synchrony(onset_dict, slice_dict, bar_length)
        semblant_motion = _semblant_motions(slice_dict)

        has_even_meter = float(int(round(bar_length)) % 2 == 0)

        features[idx] = np.array(
            [
                start,
                end,
                bar_length,
                has_even_meter,
                n_notes,
                n_notes / bar_length,
                n_active,
                n_pitches,
                n_pitches / bar_length,
                n_pitchclasses,
                novelty,
                *pitch_stats,
                *duration_stats,
                n_onsets,
                n_onsets / bar_length,
                *onset_reg_stats,
                *onset_n_voices_stats,
                *onset_n_gaps_stats,
                third_sixth,
                fourth_fifth,
                octave,
                n_slices,
                n_slices / bar_length,
                prop_silence,
                longest_silence,
                *slice_width_stats,
                *slice_n_voices_stats,
                *slice_n_gaps_stats,
                *harmonicity_stats,
                melodic_seconds,
                melodic_thirds,
                melodic_fourths_fifths,
                melodic_octaves,
                melodic_unison,
                onset_synchrony,
                semblant_motion,
            ],
            dtype=float,
        )

    return features, feature_names


def bar_texture_dataframe(
    note_array: np.ndarray,
    *,
    bar_starts: Optional[np.ndarray] = None,
    bar_ends: Optional[np.ndarray] = None,
):
    """Return a pandas DataFrame of bar-level texture descriptors."""
    import pandas as pd

    feats, names = compute_bar_texture(note_array, bar_starts=bar_starts, bar_ends=bar_ends)
    return pd.DataFrame(feats, columns=names)
