from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch_scatter


DEFAULT_TASKS_BY_LEVEL: Dict[str, List[str]] = {
    "onset": [
        "cadence",
        "phrase",
        "root",
        "localkey",
        "quality",
        "inversion",
        "degree1",
        "degree2",
        "romanNumeral",
        "section",
    ],
    "beat": [
        "root",
        "localkey",
        "quality",
        "inversion",
        "degree1",
        "degree2",
        "romanNumeral",
        "cadence",
        "phrase",
        "section",
    ],
    "measure": ["localkey"],
}

DEFAULT_BEAT_OUTPUT_TASKS: List[str] = [
    "cadence",
    "phrase",
    "romanNumeral",
    "root",
    "bass",
    "degree1",
    "degree2",
    "inversion",
    "localkey",
]

HARMONIC_BEAT_TASKS = {
    "romanNumeral",
    "root",
    "bass",
    "degree1",
    "degree2",
    "inversion",
    "localkey",
}


@dataclass
class AggregationBundleMetadata:
    feature_schema_version: int = 1
    input_dim: int = 7
    hidden_dim: int = 16
    dropout: float = 0.1
    consistency_policy: str = "none"
    harmonic_filter_policy: str = "all_notes"
    use_conflict_heads: bool = False


class WeightedGroupVoter(nn.Module):
    def __init__(self, input_dim: int = 7, hidden_dim: int = 16, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class BeatConflictHead(nn.Module):
    def __init__(self, input_dim: int = 7, hidden_dim: int = 16, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class PosthocAggregationBundle(nn.Module):
    """Task/level-specific weighted aggregation scorer bundle."""

    def __init__(
        self,
        tasks_by_level: Optional[Dict[str, Iterable[str]]] = None,
        input_dim: int = 7,
        hidden_dim: int = 16,
        dropout: float = 0.1,
        feature_schema_version: int = 1,
        consistency_policy: str = "none",
        harmonic_filter_policy: str = "all_notes",
        use_conflict_heads: bool = False,
    ) -> None:
        super().__init__()
        level_map = tasks_by_level or DEFAULT_TASKS_BY_LEVEL
        self.tasks_by_level: Dict[str, List[str]] = {
            level: sorted({str(task).strip() for task in tasks if str(task).strip()})
            for level, tasks in level_map.items()
        }
        self.metadata = AggregationBundleMetadata(
            feature_schema_version=int(feature_schema_version),
            input_dim=int(input_dim),
            hidden_dim=int(hidden_dim),
            dropout=float(dropout),
            consistency_policy=str(consistency_policy),
            harmonic_filter_policy=str(harmonic_filter_policy),
            use_conflict_heads=bool(use_conflict_heads),
        )
        self.scorers = nn.ModuleDict()
        self.conflict_heads = nn.ModuleDict()
        for level, tasks in self.tasks_by_level.items():
            for task in tasks:
                self.scorers[self._key(level, task)] = WeightedGroupVoter(
                    input_dim=self.metadata.input_dim,
                    hidden_dim=self.metadata.hidden_dim,
                    dropout=self.metadata.dropout,
                )
                if self.metadata.use_conflict_heads and level == "beat":
                    self.conflict_heads[self._key(level, task)] = BeatConflictHead(
                        input_dim=self.metadata.input_dim,
                        hidden_dim=self.metadata.hidden_dim,
                        dropout=self.metadata.dropout,
                    )

        self._collect_entropy_stats = False
        self._entropy_terms: List[torch.Tensor] = []

    @staticmethod
    def _key(level: str, task: str) -> str:
        return f"{level}::{task}"

    def supports(self, level: str, task: str) -> bool:
        return self._key(level, task) in self.scorers

    def scorer(self, level: str, task: str) -> Optional[WeightedGroupVoter]:
        key = self._key(level, task)
        if key not in self.scorers:
            return None
        return self.scorers[key]

    def supports_conflict(self, level: str, task: str) -> bool:
        key = self._key(level, task)
        return key in self.conflict_heads

    def conflict_head(self, level: str, task: str) -> Optional[BeatConflictHead]:
        key = self._key(level, task)
        if key not in self.conflict_heads:
            return None
        return self.conflict_heads[key]

    def enable_entropy_stats(self, enabled: bool = True) -> None:
        self._collect_entropy_stats = bool(enabled)

    def reset_entropy_stats(self) -> None:
        self._entropy_terms = []

    def mean_entropy(self) -> Optional[torch.Tensor]:
        if not self._entropy_terms:
            return None
        return torch.stack(self._entropy_terms).mean()

    def aggregate_task(
        self,
        *,
        level: str,
        task: str,
        task_probs: torch.Tensor,
        all_task_probs: Dict[str, torch.Tensor],
        graph,
        group_ids: torch.Tensor,
        eligible_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        out, _, _, _ = self.aggregate_task_with_conflict(
            level=level,
            task=task,
            task_probs=task_probs,
            all_task_probs=all_task_probs,
            graph=graph,
            group_ids=group_ids,
            eligible_mask=eligible_mask,
        )
        return out

    def aggregate_task_with_conflict(
        self,
        *,
        level: str,
        task: str,
        task_probs: torch.Tensor,
        all_task_probs: Dict[str, torch.Tensor],
        graph,
        group_ids: torch.Tensor,
        eligible_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        scorer = self.scorer(level, task)
        if scorer is None:
            return task_probs, None, None, None
        if task_probs.ndim != 2 or group_ids.numel() != task_probs.size(0):
            return task_probs, None, None, None

        device = task_probs.device
        n = int(task_probs.size(0))
        if eligible_mask is None:
            eligible_mask = torch.ones(n, dtype=torch.bool, device=device)
        else:
            eligible_mask = eligible_mask.to(device=device, dtype=torch.bool)
        valid = eligible_mask & (group_ids.to(device=device) >= 0)
        if not torch.any(valid):
            return task_probs, None, None, None

        features = self._build_note_features(
            task_probs=task_probs,
            all_task_probs=all_task_probs,
            graph=graph,
            num_nodes=n,
        )
        scores = scorer(features)

        valid_idx = torch.where(valid)[0]
        valid_groups = group_ids[valid_idx].to(device=device, dtype=torch.long)
        _, inverse = torch.unique(valid_groups, sorted=True, return_inverse=True)
        num_groups = int(inverse.max().item()) + 1 if inverse.numel() > 0 else 0
        if num_groups == 0:
            return task_probs, None, None, None

        valid_scores = scores[valid_idx]
        group_max = torch_scatter.scatter_max(valid_scores, inverse, dim=0, dim_size=num_groups)[0]
        stable = valid_scores - group_max[inverse]
        exp_scores = torch.exp(stable)
        denom = torch_scatter.scatter_add(exp_scores, inverse, dim=0, dim_size=num_groups) + 1e-12
        weights = exp_scores / denom[inverse]

        if self._collect_entropy_stats:
            entropy = -weights * torch.log(torch.clamp(weights, min=1e-12))
            group_entropy = torch_scatter.scatter_add(entropy, inverse, dim=0, dim_size=num_groups)
            self._entropy_terms.append(group_entropy.mean())

        weighted = task_probs[valid_idx] * weights.unsqueeze(-1)
        group_probs = torch_scatter.scatter_add(weighted, inverse, dim=0, dim_size=num_groups)
        group_probs = torch.clamp(group_probs, min=1e-9)
        group_probs = group_probs / torch.clamp(group_probs.sum(dim=-1, keepdim=True), min=1e-9)

        out = task_probs.clone()
        out[valid_idx] = group_probs[inverse]

        node_conflict_prob: Optional[torch.Tensor] = None
        group_conflict_prob: Optional[torch.Tensor] = None
        group_values: Optional[torch.Tensor] = None
        conflict_head = self.conflict_head(level, task)
        if conflict_head is not None and level == "beat":
            group_values = torch.unique(valid_groups, sorted=True)
            group_features = torch_scatter.scatter_mean(
                features[valid_idx],
                inverse,
                dim=0,
                dim_size=num_groups,
            )
            conflict_logits = conflict_head(group_features)
            group_conflict_prob = torch.sigmoid(conflict_logits)
            node_conflict_prob = torch.zeros(n, dtype=task_probs.dtype, device=device)
            node_conflict_prob[valid_idx] = group_conflict_prob[inverse]
        return out, node_conflict_prob, group_conflict_prob, group_values

    def _build_note_features(
        self,
        *,
        task_probs: torch.Tensor,
        all_task_probs: Dict[str, torch.Tensor],
        graph,
        num_nodes: int,
    ) -> torch.Tensor:
        eps = 1e-9
        p_max, _ = task_probs.max(dim=-1)
        p_max = p_max.clamp(min=eps, max=1.0)
        entropy = -(task_probs.clamp(min=eps) * torch.log(task_probs.clamp(min=eps))).sum(dim=-1)
        top2 = torch.topk(task_probs, k=min(2, task_probs.size(-1)), dim=-1).values
        if top2.size(-1) == 1:
            margin = top2[:, 0]
        else:
            margin = top2[:, 0] - top2[:, 1]

        note_store = graph["note"]
        pitch = getattr(note_store, "pitch", None)
        if isinstance(pitch, torch.Tensor) and pitch.numel() >= num_nodes:
            pitch_norm = pitch[:num_nodes].to(dtype=task_probs.dtype, device=task_probs.device) / 127.0
        else:
            pitch_norm = torch.zeros(num_nodes, dtype=task_probs.dtype, device=task_probs.device)

        duration_div = getattr(note_store, "duration_div", None)
        if isinstance(duration_div, torch.Tensor) and duration_div.numel() >= num_nodes:
            dur_norm = torch.tanh(
                duration_div[:num_nodes].to(dtype=task_probs.dtype, device=task_probs.device) / 480.0
            )
        else:
            dur_norm = torch.zeros(num_nodes, dtype=task_probs.dtype, device=task_probs.device)

        is_onset = getattr(note_store, "is_note_onset", None)
        if isinstance(is_onset, torch.Tensor) and is_onset.numel() >= num_nodes:
            onset_flag = is_onset[:num_nodes].to(dtype=task_probs.dtype, device=task_probs.device)
        else:
            onset_flag = torch.ones(num_nodes, dtype=task_probs.dtype, device=task_probs.device)

        chord_tone_prob = torch.ones(num_nodes, dtype=task_probs.dtype, device=task_probs.device)
        tpc = all_task_probs.get("tpc_in_label")
        if isinstance(tpc, torch.Tensor) and tpc.ndim == 2 and tpc.size(0) >= num_nodes:
            idx = 1 if tpc.size(-1) > 1 else 0
            chord_tone_prob = tpc[:num_nodes, idx].to(dtype=task_probs.dtype, device=task_probs.device)

        return torch.stack(
            [p_max, entropy, margin, pitch_norm, dur_norm, onset_flag, chord_tone_prob],
            dim=-1,
        )

    def to_serializable_metadata(self) -> Dict[str, object]:
        return {
            "tasks_by_level": {k: list(v) for k, v in self.tasks_by_level.items()},
            "feature_schema_version": int(self.metadata.feature_schema_version),
            "input_dim": int(self.metadata.input_dim),
            "hidden_dim": int(self.metadata.hidden_dim),
            "dropout": float(self.metadata.dropout),
            "consistency_policy": str(self.metadata.consistency_policy),
            "harmonic_filter_policy": str(self.metadata.harmonic_filter_policy),
            "use_conflict_heads": bool(self.metadata.use_conflict_heads),
        }

    @classmethod
    def from_serializable_metadata(cls, metadata: Dict[str, object]) -> "PosthocAggregationBundle":
        return cls(
            tasks_by_level=metadata.get("tasks_by_level", DEFAULT_TASKS_BY_LEVEL),
            input_dim=int(metadata.get("input_dim", 7)),
            hidden_dim=int(metadata.get("hidden_dim", 16)),
            dropout=float(metadata.get("dropout", 0.1)),
            feature_schema_version=int(metadata.get("feature_schema_version", 1)),
            consistency_policy=str(metadata.get("consistency_policy", "none")),
            harmonic_filter_policy=str(metadata.get("harmonic_filter_policy", "all_notes")),
            use_conflict_heads=bool(metadata.get("use_conflict_heads", False)),
        )
