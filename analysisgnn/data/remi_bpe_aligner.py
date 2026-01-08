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
    """Build a BpeNoteAlignment object for a tokenized MIDI file.

    Parameters
    ----------
    midi_path:
        Path to the MIDI file to tokenize.
    tokenizer:
        Callable that takes ``midi_path`` and returns an object with an ``ids`` attribute
        containing the token ids for the file.
    token2note:
        Optional NumPy array describing the alignment between BPE tokens and notes.
        The expected shape is ``(N, 3)``, where each row has the form
        ``[token_idx, note_idx, weight]``:

        * ``token_idx``: integer index into ``input_ids`` (token position).
        * ``note_idx``: integer note index in ``[0, num_notes)``.
        * ``weight``: non-negative float weight indicating how strongly the token is
          associated with the note (for example, alignment probability or a normalized
          contribution; the exact semantics are defined by the caller).

        Multiple rows may share the same ``token_idx`` and/or ``note_idx`` when a token
        aligns to multiple notes or vice versa.
    note_meta:
        Optional mapping from ``note_idx`` to :class:`NoteInfo` instances containing
        metadata for each note.
    num_notes:
        Optional total number of notes in the piece. This should be consistent with the
        maximum ``note_idx`` referenced in ``token2note``.

    Returns
    -------
    BpeNoteAlignment
        The alignment information, including token ids, attention mask, token-to-note
        alignment matrix, and per-note metadata.
    """
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


def load_alignment_npz(path: str) -> BpeNoteAlignment:
    payload = np.load(path, allow_pickle=True)
    note_meta = {}
    if "note_meta" in payload:
        note_meta = payload["note_meta"].item()

    return BpeNoteAlignment(
        input_ids=payload["input_ids"],
        attention_mask=payload["attention_mask"],
        token2note=payload["token2note"],
        num_notes=int(payload["num_notes"]),
        note_meta=note_meta,
    )


def attach_alignment_to_graph(graph, alignment: BpeNoteAlignment) -> None:
    graph.input_ids = alignment.input_ids.tolist()
    graph.attention_mask = alignment.attention_mask.tolist()
    graph.token2note = alignment.token2note.tolist()
    graph.num_notes = alignment.num_notes
