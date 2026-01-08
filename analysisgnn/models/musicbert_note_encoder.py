from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from analysisgnn.modules import TokenToNotePooler
from analysisgnn.models.musicbert_backbone import MusicBertAdapterConfig, MusicBertBackbone


class MusicBertNoteEncoder(nn.Module):
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
        token_states = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return self.pooler(token_states=token_states, token2note=token2note, num_notes=num_notes)
