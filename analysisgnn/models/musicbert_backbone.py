from dataclasses import dataclass
from typing import List, Optional
from contextlib import nullcontext

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
        use_autocast: bool = True,
        autocast_dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(pretrained_name)
        self.adapter_cfg = adapter_cfg
        self.use_autocast = use_autocast
        self.autocast_dtype = autocast_dtype

        if adapter_cfg and adapter_cfg.use_lora:
            from peft import LoraConfig, get_peft_model

            target_modules = adapter_cfg.target_modules
            if not target_modules:
                target_modules = self._infer_lora_target_modules(self.model)
            if not target_modules:
                raise ValueError(
                    "Could not infer LoRA target modules for this backbone. "
                    "Set MusicBertAdapterConfig.target_modules explicitly."
                )
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

    @staticmethod
    def _infer_lora_target_modules(model: nn.Module) -> List[str]:
        module_names = {name.split(".")[-1] for name, _ in model.named_modules()}
        candidate_groups = (
            ["q_proj", "k_proj", "v_proj", "out_proj"],
            ["query", "key", "value"],
        )
        for group in candidate_groups:
            found = [name for name in group if name in module_names]
            if found:
                return found
        return []

    def _autocast_context(self, device_type: str):
        if not self.use_autocast or torch.is_autocast_enabled():
            return nullcontext()
        if device_type != "cuda":
            return nullcontext()
        dtype = self.autocast_dtype or torch.get_autocast_gpu_dtype()
        return torch.autocast(device_type="cuda", dtype=dtype)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        with self._autocast_context(input_ids.device.type):
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state
