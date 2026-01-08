from typing import List, Tuple

import torch
import torch.nn as nn


class TokenToNotePooler(nn.Module):
    """Pool token-level embeddings into note-level embeddings using weighted aggregation.

    This module aggregates token embeddings from a sequence model (e.g., MusicBERT) into
    note-level embeddings by performing a weighted sum according to token-to-note alignment
    edges. Multiple tokens can contribute to a single note with different weights, and the
    pooling operation sums these weighted contributions.

    The pooling mechanism is useful when working with tokenized music representations where
    multiple tokens may correspond to a single musical note (e.g., BPE-tokenized REMI events
    that represent note onset, pitch, duration, etc.).
    """

    def forward(
        self,
        token_states: torch.Tensor,
        token2note: List[torch.Tensor],
        num_notes: List[int],
    ) -> Tuple[torch.Tensor, torch.BoolTensor]:
        """Pool token embeddings into note embeddings using weighted aggregation.

        Parameters
        ----------
        token_states : torch.Tensor
            Token-level embeddings of shape ``(batch_size, seq_len, hidden_dim)`` from a
            sequence model such as MusicBERT.
        token2note : List[torch.Tensor]
            List of alignment edge tensors, one per batch item. Each tensor has shape
            ``(num_edges, 3)`` where each row is ``[token_idx, note_idx, weight]``:

            * ``token_idx``: integer index into the token sequence (0 to seq_len-1)
            * ``note_idx``: integer index for the target note (0 to num_notes-1)
            * ``weight``: non-negative float weight for the contribution of this token
              to this note

            Multiple edges may share the same ``token_idx`` (one token contributing to
            multiple notes) or the same ``note_idx`` (multiple tokens contributing to one
            note). The pooler sums all weighted token embeddings for each note.
        num_notes : List[int]
            Number of notes in each batch item. Used to construct the output tensor and mask.

        Returns
        -------
        pooled : torch.Tensor
            Pooled note embeddings of shape ``(batch_size, max_notes, hidden_dim)`` where
            ``max_notes = max(num_notes)``. For each note, the embedding is the sum of
            weighted token embeddings as specified by the alignment edges. Positions beyond
            ``num_notes[i]`` for batch item ``i`` are zero-filled.
        note_mask : torch.BoolTensor
            Boolean mask of shape ``(batch_size, max_notes)`` indicating valid note positions.
            ``note_mask[i, j]`` is ``True`` if ``j < num_notes[i]``, else ``False``.

        Examples
        --------
        >>> pooler = TokenToNotePooler()
        >>> token_states = torch.randn(1, 10, 768)  # 1 batch, 10 tokens, 768-dim embeddings
        >>> # Two tokens (indices 2 and 5) contribute to note 0 with weights 0.6 and 0.4
        >>> edges = torch.tensor([[2, 0, 0.6], [5, 0, 0.4]])
        >>> pooled, mask = pooler(token_states, [edges], [1])
        >>> pooled.shape
        torch.Size([1, 1, 768])
        >>> mask.tolist()
        [[True]]
        """
        batch_size, _, hidden_dim = token_states.shape
        max_notes = max(num_notes) if num_notes else 0
        device = token_states.device

        pooled = torch.zeros((batch_size, max_notes, hidden_dim), device=device)
        note_mask = torch.zeros((batch_size, max_notes), dtype=torch.bool, device=device)

        for batch_idx in range(batch_size):
            if num_notes[batch_idx] == 0:
                continue

            edges = token2note[batch_idx]
            token_idx = edges[:, 0].long()
            note_idx = edges[:, 1].long()
            weights = edges[:, 2].to(token_states.dtype).unsqueeze(-1)

            weighted_states = token_states[batch_idx, token_idx] * weights
            pooled[batch_idx, :num_notes[batch_idx]].index_add_(
                0,
                note_idx,
                weighted_states,
            )
            note_mask[batch_idx, :num_notes[batch_idx]] = True

        return pooled, note_mask
