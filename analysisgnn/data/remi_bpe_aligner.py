from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass(frozen=True)
class NoteInfo:
    note_id: int
    onset_tick: int
    pitch: int
    duration_tick: int
    velocity: int
    track: int
    bar: int
    position: int


@dataclass(frozen=True)
class BpeNoteAlignment:
    input_ids: np.ndarray
    attention_mask: np.ndarray
    token2note: np.ndarray
    num_notes: int
    note_meta: Dict[int, NoteInfo]


def build_alignment(
    midi_path: str,
    tokenizer,
    *,
    token2note: Optional[np.ndarray] = None,
    note_meta: Optional[Dict[int, NoteInfo]] = None,
    num_notes: Optional[int] = None,
) -> BpeNoteAlignment:
    tok_seq = tokenizer(midi_path)
    input_ids = np.asarray(tok_seq.ids, dtype=np.int64)
    attention_mask = np.ones_like(input_ids, dtype=np.int64)

    if token2note is None or note_meta is None or num_notes is None:
        raise NotImplementedError(
            "Token-to-note alignment is required to build MusicBERT note embeddings. "
            "Provide token2note, note_meta, and num_notes or extend the tokenizer "
            "pipeline to emit base-event to note mappings."
        )

    return BpeNoteAlignment(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token2note=token2note.astype(np.float32),
        num_notes=num_notes,
        note_meta=note_meta,
    )
