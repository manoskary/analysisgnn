from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from analysisgnn.modules import TokenToNotePooler
from analysisgnn.models.musicbert_backbone import MusicBertAdapterConfig, MusicBertBackbone


class MusicBertNoteEncoder(nn.Module):
    """Note-level encoder using MusicBERT with token-to-note pooling.
    
    This encoder implements a two-stage process for generating note-level representations
    from tokenized musical sequences:
    
    1. Token Embedding: A pretrained MusicBERT backbone (transformers AutoModel) encodes
       the input token sequence into contextualized token embeddings.
    2. Token-to-Note Pooling: A TokenToNotePooler aggregates token embeddings into 
       note-level representations using weighted pooling based on token-to-note alignments.
    
    The backbone can be optionally fine-tuned with LoRA adapters or frozen for feature
    extraction. The pooler uses alignment information to map multiple tokens that represent
    parts of the same note into a single note-level embedding.
    
    Args:
        pretrained_name: HuggingFace model identifier for the pretrained MusicBERT model
            (e.g., "manoskary/musicbert-large").
        adapter_cfg: Optional configuration for LoRA adapters to enable parameter-efficient
            fine-tuning of the backbone. If None, no adapters are used.
        freeze_backbone: Whether to freeze the backbone model parameters. If True, the
            backbone is used as a frozen feature extractor. If False or if LoRA adapters
            are used, the backbone parameters are trainable.
    """
    def __init__(
        self,
        pretrained_name: str,
        adapter_cfg: Optional[MusicBertAdapterConfig] = None,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = MusicBertBackbone(
            pretrained_name=pretrained_name,
            adapter_cfg=adapter_cfg,
            freeze_base=freeze_backbone,
        )
        self.pooler = TokenToNotePooler()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token2note: List[torch.Tensor],
        num_notes: List[int],
    ) -> Tuple[torch.Tensor, torch.BoolTensor]:
        """Encode tokens to note-level representations.
        
        Args:
            input_ids: Token IDs for the input sequence, shape (batch_size, seq_len).
            attention_mask: Attention mask for the input sequence, shape (batch_size, seq_len).
            token2note: List of alignment tensors, one per batch item. Each tensor has shape
                (num_edges, 3) where each row is [token_idx, note_idx, weight] indicating
                that token_idx contributes to note_idx with the given weight.
            num_notes: List of integers indicating the number of notes in each batch item.
        
        Returns:
            A tuple of (pooled_states, note_mask):
                - pooled_states: Note-level representations, shape (batch_size, max_notes, hidden_dim).
                - note_mask: Boolean mask indicating valid notes, shape (batch_size, max_notes).
        """
        max_len = getattr(self.backbone.model.config, "max_position_embeddings", None)
        if max_len is None or input_ids.size(1) <= max_len:
            token_states = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            return self.pooler(token_states=token_states, token2note=token2note, num_notes=num_notes)

        return self._encode_long_sequences(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token2note=token2note,
            num_notes=num_notes,
            max_len=max_len,
        )

    def _encode_long_sequences(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token2note: List[torch.Tensor],
        num_notes: List[int],
        max_len: int,
    ) -> Tuple[torch.Tensor, torch.BoolTensor]:
        device = input_ids.device
        batch_size = input_ids.size(0)
        hidden_dim = int(self.backbone.model.config.hidden_size)
        max_notes = max(num_notes) if num_notes else 0

        pooled = torch.zeros((batch_size, max_notes, hidden_dim), device=device)
        note_mask = torch.zeros((batch_size, max_notes), dtype=torch.bool, device=device)

        for batch_idx in range(batch_size):
            if num_notes[batch_idx] == 0:
                continue
            seq_len = int(attention_mask[batch_idx].sum().item())
            if seq_len == 0:
                continue
            note_mask[batch_idx, : num_notes[batch_idx]] = True
            edges = token2note[batch_idx]
            if edges.numel() == 0:
                continue

            for start in range(0, seq_len, max_len):
                end = min(start + max_len, seq_len)
                chunk_mask = (edges[:, 0] >= start) & (edges[:, 0] < end)
                if not torch.any(chunk_mask):
                    continue

                chunk_edges = edges[chunk_mask].clone()
                chunk_edges[:, 0] -= start
                chunk_ids = input_ids[batch_idx, start:end].unsqueeze(0)
                chunk_attn = attention_mask[batch_idx, start:end].unsqueeze(0)
                token_states = self.backbone(input_ids=chunk_ids, attention_mask=chunk_attn)
                pooled_chunk, _ = self.pooler(
                    token_states=token_states,
                    token2note=[chunk_edges],
                    num_notes=[num_notes[batch_idx]],
                )
                pooled[batch_idx, : num_notes[batch_idx]] += pooled_chunk[0, : num_notes[batch_idx]]

        return pooled, note_mask
