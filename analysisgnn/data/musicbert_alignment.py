from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from analysisgnn.data.remi_bpe_aligner import BpeNoteAlignment, NoteInfo
from analysisgnn.utils.dcl_tsv_utils import load_labeled_pitch_array
from analysisgnn.utils.globals import INTERVAL_TO_SEMITONES

LOGGER = logging.getLogger(__name__)


def load_pitch_array(
    tsv_path: str,
    spec_file: dict | str,
    converters: Optional[dict] = None,
    drop_na_subset: Optional[List[str]] = None,
) -> pd.DataFrame:
    df = load_labeled_pitch_array(
        spec_file=spec_file,
        pitch_array_tsv=tsv_path,
        converters=converters,
        dropna_subset=drop_na_subset,
    )
    return df


def _to_float_array(values: pd.Series) -> np.ndarray:
    def _convert(value):
        if hasattr(value, "numerator") and hasattr(value, "denominator"):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                try:
                    from fractions import Fraction

                    return float(Fraction(value))
                except Exception:
                    return float("nan")
        return float(value)

    return values.map(_convert).to_numpy(dtype=float)


def _compute_divs_per_beat(onset_div: np.ndarray, onset_beat: np.ndarray) -> float:
    unique_divs = np.unique(onset_div)
    unique_beats = np.unique(onset_beat)
    if unique_divs.size < 2 or unique_beats.size < 2:
        return 1.0
    diff_divs = np.diff(unique_divs)
    diff_beats = np.diff(unique_beats)
    for beat, divs in zip(diff_beats, diff_divs):
        if not np.isclose(beat, 0.0):
            return float(divs / beat)
    return 1.0


def _extract_duration_beat(df: pd.DataFrame, divs_per_beat: float) -> np.ndarray:
    if "duration_beat" in df.columns:
        return df["duration_beat"].to_numpy(dtype=float)
    if "duration_div" in df.columns:
        return df["duration_div"].to_numpy(dtype=float) / divs_per_beat
    raise ValueError("Pitch array missing duration_beat/duration_div columns.")


def _extract_pitch(df: pd.DataFrame) -> np.ndarray:
    for col in ("pitch", "s_midi", "s_pitch"):
        if col in df.columns:
            return df[col].to_numpy(dtype=int)
    raise ValueError("Pitch array missing pitch/s_midi/s_pitch column.")


def _extract_velocity(df: pd.DataFrame, length: int) -> np.ndarray:
    for col in ("velocity", "s_velocity"):
        if col in df.columns:
            return df[col].to_numpy(dtype=int)
    return np.full(length, 64, dtype=int)


def _extract_time_signatures(df: pd.DataFrame, length: int) -> Tuple[np.ndarray, np.ndarray]:
    if "ts_beats" in df.columns:
        ts_beats = pd.to_numeric(df["ts_beats"], errors="coerce").fillna(4).astype(int).to_numpy()
    else:
        ts_beats = np.full(length, 4, dtype=int)
    if "ts_beat_type" in df.columns:
        ts_beat_type = (
            pd.to_numeric(df["ts_beat_type"], errors="coerce").fillna(4).astype(int).to_numpy()
        )
    else:
        ts_beat_type = np.full(length, 4, dtype=int)
    return ts_beats, ts_beat_type


def _ticks_per_quarter(tokenizer, max_denom: int) -> int:
    max_pos_per_beat = getattr(tokenizer.config, "max_num_pos_per_beat", None)
    if max_pos_per_beat is None:
        max_pos_per_beat = max(tokenizer.config.beat_res.values())
    return int(max_pos_per_beat * max_denom / 4)


def _collect_time_signatures(
    ts_beats: np.ndarray,
    ts_beat_type: np.ndarray,
    onset_ticks: np.ndarray,
) -> List[Tuple[int, int, int]]:
    changes: List[Tuple[int, int, int]] = []
    last_sig: Optional[Tuple[int, int]] = None
    for idx, (num, denom) in enumerate(zip(ts_beats, ts_beat_type)):
        sig = (int(num), int(denom))
        if last_sig is None or sig != last_sig:
            tick = int(onset_ticks[idx])
            changes.append((sig[0], sig[1], max(tick, 0)))
            last_sig = sig
    if not changes:
        return [(4, 4, 0)]
    if changes[0][2] != 0:
        first = changes[0]
        changes.insert(0, (first[0], first[1], 0))
    # Deduplicate identical signatures at the same time
    cleaned: List[Tuple[int, int, int]] = []
    for sig in changes:
        if cleaned and cleaned[-1] == sig:
            continue
        cleaned.append(sig)
    return cleaned


def _build_note_metadata(
    onset_ticks: np.ndarray,
    duration_ticks: np.ndarray,
    pitch: np.ndarray,
    velocity: np.ndarray,
    ts_beats: np.ndarray,
    ts_beat_type: np.ndarray,
    ticks_per_quarter: int,
    onset_beat: np.ndarray,
) -> Dict[int, NoteInfo]:
    note_meta: Dict[int, NoteInfo] = {}
    for idx in range(len(onset_ticks)):
        beat_in_bar = onset_beat[idx]
        bar_len = float(ts_beats[idx]) if ts_beats[idx] > 0 else 4.0
        bar_index = int(np.floor(beat_in_bar / bar_len)) if bar_len else 0
        position_beats = beat_in_bar - (bar_index * bar_len)
        position_tick = int(
            round(position_beats * (4.0 / max(ts_beat_type[idx], 1)) * ticks_per_quarter)
        )
        note_meta[idx] = NoteInfo(
            note_id=int(idx),
            onset_tick=int(onset_ticks[idx]),
            pitch=int(pitch[idx]),
            duration_tick=int(duration_ticks[idx]),
            velocity=int(velocity[idx]),
            track=0,
            bar=max(bar_index, 0),
            position=max(position_tick, 0),
        )
    return note_meta


def _build_midi(
    note_meta: Dict[int, NoteInfo],
    ticks_per_quarter: int,
    time_signatures: List[Tuple[int, int, int]],
    tempo: int = 120,
):
    import symusic

    score = symusic.Score(ticks_per_quarter)
    track = symusic.Track(program=0, is_drum=False)
    for note in note_meta.values():
        start = int(note.onset_tick)
        duration = int(note.duration_tick)
        if duration <= 0:
            duration = 1
        track.notes.append(symusic.Note(start, duration, int(note.pitch), int(note.velocity)))
    score.tracks.append(track)
    for numer, denom, time in time_signatures:
        score.time_signatures.append(symusic.TimeSignature(int(time), int(numer), int(denom)))
    score.tempos.append(symusic.Tempo(0, float(tempo)))
    return score


def _update_note_meta_from_preprocessed(
    note_meta: Dict[int, NoteInfo],
    pre_score,
) -> Dict[int, NoteInfo]:
    if not pre_score.tracks:
        return note_meta
    pre_notes = pre_score.tracks[0].notes
    if len(pre_notes) != len(note_meta):
        LOGGER.warning(
            "Preprocessed note count mismatch (%s vs %s); alignment coverage may drop.",
            len(pre_notes),
            len(note_meta),
        )
    count = min(len(pre_notes), len(note_meta))
    updated: Dict[int, NoteInfo] = {}
    for idx in range(count):
        pre_note = pre_notes[idx]
        original = note_meta[idx]
        updated[idx] = NoteInfo(
            note_id=original.note_id,
            onset_tick=int(pre_note.time),
            pitch=int(pre_note.pitch),
            duration_tick=int(pre_note.duration),
            velocity=int(pre_note.velocity),
            track=original.track,
            bar=original.bar,
            position=original.position,
        )
    # Preserve any trailing notes if lengths mismatch.
    for idx in range(count, len(note_meta)):
        updated[idx] = note_meta[idx]
    return updated


def _parse_duration_from_desc(desc, start_tick: int) -> Optional[int]:
    if desc is None:
        return None
    if isinstance(desc, str):
        for token in desc.replace(",", " ").split():
            if token.isdigit():
                return int(token)
        return None
    if isinstance(desc, (int, np.integer)):
        if desc >= start_tick:
            return int(desc - start_tick)
        return int(desc)
    if isinstance(desc, float):
        value = int(round(desc))
        if value >= start_tick:
            return int(value - start_tick)
        return value
    return None


def _pop_note_index(
    by_key: Dict[Tuple[int, int, int], List[int]],
    by_start_pitch: Dict[Tuple[int, int], List[int]],
    start: int,
    pitch: int,
    duration: Optional[int],
) -> Optional[int]:
    if duration is not None:
        key = (start, pitch, duration)
        if key in by_key and by_key[key]:
            return by_key[key].pop(0)
        for delta in (1, -1, 2, -2):
            alt = (start, pitch, duration + delta)
            if alt in by_key and by_key[alt]:
                return by_key[alt].pop(0)
    fallback = by_start_pitch.get((start, pitch))
    if fallback:
        return fallback.pop(0)
    return None


def _build_base_token_mapping(seq, note_meta: Dict[int, NoteInfo]) -> List[int]:
    if seq.events is None or seq.tokens is None:
        raise ValueError("Tokenizer did not return events/tokens for alignment.")
    if len(seq.events) != len(seq.tokens):
        raise ValueError("Token/events length mismatch in tokenizer output.")

    by_key: Dict[Tuple[int, int, int], List[int]] = defaultdict(list)
    by_start_pitch: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for idx, note in note_meta.items():
        key = (int(note.onset_tick), int(note.pitch), int(note.duration_tick))
        by_key[key].append(idx)
        by_start_pitch[(int(note.onset_tick), int(note.pitch))].append(idx)

    base_token_to_note = [-1] * len(seq.tokens)
    pending_note_idx: Optional[int] = None
    pending_note_time: Optional[int] = None
    pitch_types = {"Pitch", "NoteOn", "PitchDrum", "DrumOn"}

    for token_idx, event in enumerate(seq.events):
        event_type = getattr(event, "type_", getattr(event, "type", None))
        if event_type in pitch_types:
            start_tick = int(event.time)
            pitch = int(event.value)
            duration = _parse_duration_from_desc(event.desc, start_tick)
            note_idx = _pop_note_index(by_key, by_start_pitch, start_tick, pitch, duration)
            if note_idx is not None:
                base_token_to_note[token_idx] = note_idx
            else:
                LOGGER.debug(
                    "Unmatched note token at %s (pitch=%s, dur=%s).",
                    start_tick,
                    pitch,
                    duration,
                )
            pending_note_idx = note_idx
            pending_note_time = start_tick
            continue

        if event_type in {"Velocity", "Duration"} and pending_note_idx is not None:
            if pending_note_time == int(event.time):
                base_token_to_note[token_idx] = pending_note_idx
                if event_type == "Duration":
                    pending_note_idx = None
                    pending_note_time = None
            else:
                pending_note_idx = None
                pending_note_time = None

    return base_token_to_note


def _bpe_offsets(seq, tokenizer) -> Tuple[List[int], List[Tuple[int, int]]]:
    split_mode = getattr(tokenizer.config, "encode_ids_split", None)
    if split_mode == "bar":
        subseqs = seq.split_per_bars()
    elif split_mode == "beat":
        subseqs = seq.split_per_beats()
    else:
        subseqs = [seq]

    bpe_ids: List[int] = []
    bpe_offsets: List[Tuple[int, int]] = []
    base_offset = 0
    for subseq in subseqs:
        tokenizer.complete_sequence(subseq, complete_bytes=True)
        encoding = tokenizer._model.encode([subseq.bytes], is_pretokenized=True)
        bpe_ids.extend(encoding.ids)
        for start, end in encoding.offsets:
            bpe_offsets.append((base_offset + start, base_offset + end))
        base_offset += len(subseq.tokens)
    return bpe_ids, bpe_offsets


def build_alignment_from_tsv(
    tsv_path: str,
    tokenizer,
    spec_file: dict | str,
    converters: Optional[dict] = None,
    drop_na_subset: Optional[List[str]] = None,
    interval: str = "P1",
) -> BpeNoteAlignment:
    df = load_pitch_array(
        tsv_path=tsv_path,
        spec_file=spec_file,
        converters=converters,
        drop_na_subset=drop_na_subset,
    )

    pitch = _extract_pitch(df)
    velocity = _extract_velocity(df, len(df))
    ts_beats, ts_beat_type = _extract_time_signatures(df, len(df))

    beat_unit = 4.0 / np.maximum(ts_beat_type, 1)
    if "continuous_beats" in df.columns:
        onset_beat = _to_float_array(df["continuous_beats"])
    elif "quarterbeats_playthrough" in df.columns:
        quarterbeats = _to_float_array(df["quarterbeats_playthrough"])
        onset_beat = quarterbeats / beat_unit
    elif "onset_beat" in df.columns:
        onset_beat = _to_float_array(df["onset_beat"])
    else:
        raise ValueError("Pitch array missing timing columns for onset beats.")

    if "onset_div" in df.columns:
        divs_per_beat = _compute_divs_per_beat(df["onset_div"].to_numpy(dtype=float), onset_beat)
    else:
        divs_per_beat = 1.0
    duration_beat = _extract_duration_beat(df, divs_per_beat)

    semitone = INTERVAL_TO_SEMITONES.get(interval, 0)
    if semitone:
        pitch = (pitch + semitone) % 128

    max_denom = int(np.max(ts_beat_type)) if len(ts_beat_type) else 4
    ticks_per_quarter = _ticks_per_quarter(tokenizer, max_denom)
    onset_ticks = np.round(onset_beat * beat_unit * ticks_per_quarter).astype(int)
    duration_ticks = np.round(duration_beat * beat_unit * ticks_per_quarter).astype(int)
    duration_ticks = np.maximum(duration_ticks, 1)

    shift = 0
    if onset_ticks.size and onset_ticks.min() < 0:
        shift = int(-onset_ticks.min())
        onset_ticks = onset_ticks + shift
    onset_ticks = onset_ticks.astype(int)

    time_signatures = _collect_time_signatures(ts_beats, ts_beat_type, onset_ticks)

    note_meta = _build_note_metadata(
        onset_ticks=onset_ticks,
        duration_ticks=duration_ticks,
        pitch=pitch,
        velocity=velocity,
        ts_beats=ts_beats,
        ts_beat_type=ts_beat_type,
        ticks_per_quarter=ticks_per_quarter,
        onset_beat=onset_beat,
    )
    score = _build_midi(note_meta, ticks_per_quarter, time_signatures)
    pre_score = tokenizer.preprocess_score(score)
    note_meta = _update_note_meta_from_preprocessed(note_meta, pre_score)

    seq = tokenizer.encode(pre_score, no_preprocess_score=True)
    if seq.events is None or seq.tokens is None:
        raise ValueError("Tokenizer did not provide events/tokens for alignment.")

    base_token_to_note = _build_base_token_mapping(seq, note_meta)
    bpe_ids, bpe_offsets = _bpe_offsets(seq, tokenizer)
    if len(seq.ids) != len(bpe_ids):
        raise ValueError("BPE id length mismatch between tokenizer output and offsets.")
    if seq.ids != bpe_ids:
        LOGGER.warning("BPE ids mismatch between tokenizer output and offsets; using tokenizer ids.")
        bpe_ids = list(seq.ids)

    edges: List[List[float]] = []
    for token_idx, (start, end) in enumerate(bpe_offsets):
        note_counts: Dict[int, int] = defaultdict(int)
        for base_idx in range(start, end):
            note_idx = base_token_to_note[base_idx]
            if note_idx >= 0:
                note_counts[note_idx] += 1
        total = sum(note_counts.values())
        if total == 0:
            continue
        for note_idx, count in note_counts.items():
            weight = float(count / total)
            edges.append([float(token_idx), float(note_idx), weight])

    input_ids = np.asarray(bpe_ids, dtype=np.int64)
    attention_mask = np.ones_like(input_ids, dtype=np.int64)
    token2note = np.asarray(edges, dtype=np.float32)
    if token2note.ndim == 1:
        token2note = token2note.reshape(0, 3)
    num_notes = len(note_meta)

    return BpeNoteAlignment(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token2note=token2note,
        num_notes=num_notes,
        note_meta=note_meta,
    )


def save_alignment_npz(alignment: BpeNoteAlignment, output_path: str) -> None:
    np.savez_compressed(
        output_path,
        input_ids=alignment.input_ids,
        attention_mask=alignment.attention_mask,
        token2note=alignment.token2note,
        num_notes=np.asarray(alignment.num_notes, dtype=np.int64),
        note_meta=alignment.note_meta,
    )
