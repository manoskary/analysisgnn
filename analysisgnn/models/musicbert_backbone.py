from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn


@dataclass
class MusicBertAdapterConfig:
    use_lora: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    target_modules: Optional[List[str]] = None


class MusicBertBackbone(nn.Module):
    def __init__(
        self,
        pretrained_name: str,
        adapter_cfg: Optional[MusicBertAdapterConfig] = None,
        freeze_base: bool = True,
    ) -> None:
        super().__init__()
        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(pretrained_name)
        self.adapter_cfg = adapter_cfg

        if adapter_cfg and adapter_cfg.use_lora:
            from peft import LoraConfig, get_peft_model

            target_modules = adapter_cfg.target_modules or [
                "q_proj",
                "k_proj",
                "v_proj",
                "out_proj",
            ]
            lora_cfg = LoraConfig(
                r=adapter_cfg.lora_r,
                lora_alpha=adapter_cfg.lora_alpha,
                lora_dropout=adapter_cfg.lora_dropout,
                target_modules=target_modules,
            )
            self.model = get_peft_model(self.model, lora_cfg)
            freeze_base = False

        if freeze_base:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state
