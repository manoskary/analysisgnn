import torch

from analysisgnn.modules import TokenToNotePooler


def test_token_to_note_pooler_weight_invariance():
    pooler = TokenToNotePooler()
    token_states = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])

    edges = torch.tensor(
        [
            [0.0, 0.0, 1.0],
        ]
    )
    pooled, mask = pooler(token_states, [edges], [1])

    duplicated_edges = torch.tensor(
        [
            [0.0, 0.0, 0.5],
            [0.0, 0.0, 0.5],
        ]
    )
    pooled_dup, mask_dup = pooler(token_states, [duplicated_edges], [1])

    torch.testing.assert_close(pooled, pooled_dup)
    assert torch.equal(mask, mask_dup)


def test_token_to_note_pooler_sums_weights():
    pooler = TokenToNotePooler()
    token_states = torch.tensor([[[1.0, 1.0], [2.0, 2.0]]])
    edges = torch.tensor(
        [
            [0.0, 0.0, 0.25],
            [1.0, 0.0, 0.75],
        ]
    )
    pooled, mask = pooler(token_states, [edges], [1])
    expected = torch.tensor([[[1.75, 1.75]]])
    torch.testing.assert_close(pooled, expected)
    assert mask.tolist() == [[True]]
