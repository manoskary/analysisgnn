import numpy as np
import pytest

from analysisgnn.data.remi_bpe_aligner import build_alignment, NoteInfo


class DummyTokSeq:
    def __init__(self, ids):
        self.ids = ids


class DummyTokenizer:
    def __call__(self, midi_path):
        return DummyTokSeq([10, 20, 30])


def test_build_alignment_requires_token2note():
    tokenizer = DummyTokenizer()
    with pytest.raises(NotImplementedError):
        build_alignment("dummy.mid", tokenizer)


def test_build_alignment_with_mapping():
    tokenizer = DummyTokenizer()
    token2note = np.array([[0, 0, 1.0], [1, 0, 0.5], [2, 0, 0.5]], dtype=np.float32)
    note_meta = {
        0: NoteInfo(
            note_id=0,
            onset_tick=0,
            pitch=60,
            duration_tick=480,
            velocity=100,
            track=0,
            bar=0,
            position=0,
        )
    }
    alignment = build_alignment(
        "dummy.mid",
        tokenizer,
        token2note=token2note,
        note_meta=note_meta,
        num_notes=1,
    )
    assert alignment.num_notes == 1
    assert alignment.input_ids.tolist() == [10, 20, 30]
    assert alignment.attention_mask.tolist() == [1, 1, 1]
    assert alignment.token2note.shape == (3, 3)
