from typing import List, Tuple

import torch
import torch.nn as nn


class TokenToNotePooler(nn.Module):
    def forward(
        self,
        token_states: torch.Tensor,
        token2note: List[torch.Tensor],
        num_notes: List[int],
    ) -> Tuple[torch.Tensor, torch.BoolTensor]:
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
