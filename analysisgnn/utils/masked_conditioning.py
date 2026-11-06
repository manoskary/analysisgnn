"""
Utilities for label-conditioned masked prediction.

This module normalizes user-facing masked specs into tensors that can be fed
into the model as conditioning evidence.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from analysisgnn.utils.chord_representations import available_representations
from analysisgnn.utils.node_masking import create_node_mask


@dataclass
class MaskedConditioningSpec:
    """Normalized conditioning data for masked prediction."""

    node_mask: Optional[torch.Tensor]
    known_labels_by_task: Dict[str, torch.Tensor]
    known_indices_by_task: Dict[str, torch.Tensor]
    masked_tasks: List[str]
    constraint_mode: str = "hard"
    feedback_mode: str = "single_pass"

    def to(self, device: torch.device) -> "MaskedConditioningSpec":
        node_mask = self.node_mask.to(device) if self.node_mask is not None else None
        known_labels_by_task = {
            task: labels.to(device) for task, labels in self.known_labels_by_task.items()
        }
        known_indices_by_task = {
            task: idx.to(device) for task, idx in self.known_indices_by_task.items()
        }
        return MaskedConditioningSpec(
            node_mask=node_mask,
            known_labels_by_task=known_labels_by_task,
            known_indices_by_task=known_indices_by_task,
            masked_tasks=list(self.masked_tasks),
            constraint_mode=self.constraint_mode,
            feedback_mode=self.feedback_mode,
        )


def _normalize_indices(value: Any, name: str) -> List[int]:
    if value is None:
        return []
    if isinstance(value, (int, np.integer)):
        return [int(value)]
    if torch.is_tensor(value):
        value = value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    raise TypeError(f"{name} must be an int or a list of ints")


def _validate_indices(indices: List[int], num_nodes: int, name: str) -> None:
    for idx in indices:
        if idx < 0 or idx >= num_nodes:
            raise ValueError(f"{name} index {idx} out of range for num_nodes={num_nodes}")


def _encode_label(task: str, value: Any, tasks_num_classes: Optional[Dict[str, int]]) -> int:
    if torch.is_tensor(value):
        value = value.detach().cpu().item()
    if isinstance(value, (np.integer, int)):
        label_id = int(value)
    elif isinstance(value, str):
        if task not in available_representations:
            raise ValueError(f"Task '{task}' does not support string labels")
        class_list = available_representations[task].classList
        if value not in class_list:
            raise ValueError(f"Label '{value}' not in class list for task '{task}'")
        label_id = class_list.index(value)
    else:
        raise TypeError(f"Unsupported label type for task '{task}': {type(value).__name__}")

    if tasks_num_classes and task in tasks_num_classes:
        if label_id < 0 or label_id >= tasks_num_classes[task]:
            raise ValueError(
                f"Label id {label_id} out of range for task '{task}' "
                f"(0..{tasks_num_classes[task] - 1})"
            )
    return label_id


def _normalize_task_list(
    tasks: Optional[List[str]],
    known_labels: Optional[Dict[str, Any]],
    tasks_num_classes: Dict[str, int],
) -> List[str]:
    resolved = []
    if tasks:
        resolved.extend(tasks)
    if known_labels:
        resolved.extend(list(known_labels.keys()))
    dedup = []
    for task in resolved:
        if task not in tasks_num_classes:
            continue
        if task in dedup:
            continue
        dedup.append(task)
    return dedup


def _normalize_node_mask_spec(
    node_mask_spec: Optional[Any],
    num_nodes: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    if node_mask_spec is None:
        return None
    if isinstance(node_mask_spec, dict):
        context_weight = float(node_mask_spec.get("context_weight", 0.1))
        if context_weight <= 0 or context_weight > 1:
            raise ValueError("context_weight must be in (0, 1]")
        targets = _normalize_indices(node_mask_spec.get("targets", None), "node_mask.targets")
        context = _normalize_indices(node_mask_spec.get("context", None), "node_mask.context")
        unlabeled = _normalize_indices(node_mask_spec.get("unlabeled", None), "node_mask.unlabeled")
        _validate_indices(targets, num_nodes, "node_mask.targets")
        _validate_indices(context, num_nodes, "node_mask.context")
        _validate_indices(unlabeled, num_nodes, "node_mask.unlabeled")

        target_set = set(targets)
        context_set = set(context)
        unlabeled_set = set(unlabeled)
        if target_set & context_set or target_set & unlabeled_set or context_set & unlabeled_set:
            raise ValueError("node_mask targets/context/unlabeled must be disjoint")
        if not targets:
            targets = sorted(set(range(num_nodes)) - context_set - unlabeled_set)

        return create_node_mask(
            num_nodes=num_nodes,
            target_indices=torch.tensor(targets, dtype=torch.long, device=device),
            context_indices=torch.tensor(context, dtype=torch.long, device=device) if context else None,
            unlabeled_indices=torch.tensor(unlabeled, dtype=torch.long, device=device) if unlabeled else None,
            context_weight=context_weight,
            device=device,
        )

    if torch.is_tensor(node_mask_spec):
        node_mask = node_mask_spec.detach().to(device=device, dtype=torch.float32)
    else:
        node_mask = torch.tensor(node_mask_spec, dtype=torch.float32, device=device)
    if node_mask.numel() != num_nodes:
        raise ValueError(f"node_mask length {node_mask.numel()} != num_nodes {num_nodes}")
    if node_mask.min() < 0 or node_mask.max() > 1:
        raise ValueError("node_mask values must be in [0, 1]")
    return node_mask


def _normalize_known_labels(
    known_labels_spec: Optional[Dict[str, Any]],
    masked_tasks: List[str],
    num_nodes: int,
    tasks_num_classes: Dict[str, int],
    device: torch.device,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    known_labels_by_task: Dict[str, torch.Tensor] = {}
    known_indices_by_task: Dict[str, torch.Tensor] = {}

    for task in masked_tasks:
        known = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
        known_labels_by_task[task] = known
        known_indices_by_task[task] = torch.zeros(0, dtype=torch.long, device=device)

    if not known_labels_spec:
        return known_labels_by_task, known_indices_by_task

    for task, spec in known_labels_spec.items():
        if task not in known_labels_by_task:
            continue
        if isinstance(spec, dict):
            indices = _normalize_indices(spec.get("indices", None), f"known_labels.{task}.indices")
            labels = spec.get("labels", None)
            if labels is None:
                raise ValueError(f"known_labels.{task}.labels is required")
            if torch.is_tensor(labels):
                labels = labels.detach().cpu().tolist()
            if isinstance(labels, np.ndarray):
                labels = labels.tolist()
            label_list = list(labels) if isinstance(labels, (list, tuple)) else [labels]
            if len(indices) == 0:
                continue
            if len(label_list) == 1 and len(indices) > 1:
                label_list = label_list * len(indices)
            if len(label_list) != len(indices):
                raise ValueError(f"known_labels.{task} labels/indices length mismatch")
            _validate_indices(indices, num_nodes, f"known_labels.{task}.indices")
            encoded = [_encode_label(task, value, tasks_num_classes) for value in label_list]
            idx_tensor = torch.tensor(indices, dtype=torch.long, device=device)
            known_labels_by_task[task][idx_tensor] = torch.tensor(encoded, dtype=torch.long, device=device)
            known_indices_by_task[task] = idx_tensor
            continue

        if torch.is_tensor(spec):
            dense = spec.detach().to(device=device, dtype=torch.long).flatten()
        else:
            dense = torch.tensor(spec, dtype=torch.long, device=device).flatten()
        if dense.numel() != num_nodes:
            raise ValueError(f"known_labels.{task} length {dense.numel()} != num_nodes {num_nodes}")
        num_classes = tasks_num_classes[task]
        dense = torch.where(
            (dense >= 0) & (dense < num_classes),
            dense,
            torch.full_like(dense, -1),
        )
        known_labels_by_task[task] = dense
        known_indices_by_task[task] = torch.where(dense >= 0)[0]

    return known_labels_by_task, known_indices_by_task


def build_known_labels_from_overrides(
    overrides: Dict[str, Dict[str, torch.Tensor]],
    num_nodes: int,
    tasks_num_classes: Dict[str, int],
    device: torch.device,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    known_labels_by_task: Dict[str, torch.Tensor] = {}
    known_indices_by_task: Dict[str, torch.Tensor] = {}
    if not overrides:
        return known_labels_by_task, known_indices_by_task

    for task, payload in overrides.items():
        if task not in tasks_num_classes:
            continue
        if "indices" not in payload or "labels" not in payload:
            continue
        indices = payload["indices"].to(device=device, dtype=torch.long)
        if indices.numel() == 0:
            continue
        labels = payload["labels"].to(device=device, dtype=torch.long)
        if labels.numel() == num_nodes:
            labels = labels[indices]
        num_classes = tasks_num_classes[task]
        valid = (labels >= 0) & (labels < num_classes)
        indices = indices[valid]
        labels = labels[valid]
        if indices.numel() == 0:
            continue
        known = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
        known[indices] = labels
        known_labels_by_task[task] = known
        known_indices_by_task[task] = indices

    return known_labels_by_task, known_indices_by_task


def normalize_masked_conditioning_spec(
    masked_spec: Optional[Dict[str, Any]],
    num_nodes: int,
    tasks_num_classes: Dict[str, int],
    device: torch.device,
    base_node_mask: Optional[torch.Tensor] = None,
    base_known_labels_by_task: Optional[Dict[str, torch.Tensor]] = None,
    base_known_indices_by_task: Optional[Dict[str, torch.Tensor]] = None,
    base_masked_tasks: Optional[List[str]] = None,
    base_constraint_mode: str = "hard",
    base_feedback_mode: str = "single_pass",
) -> Optional[MaskedConditioningSpec]:
    """
    Normalize a masked conditioning spec into dense tensors.

    Returns None when there is no conditioning information.
    """
    base_known_labels_by_task = base_known_labels_by_task or {}
    base_known_indices_by_task = base_known_indices_by_task or {}
    base_masked_tasks = base_masked_tasks or []

    if masked_spec is None:
        if not base_masked_tasks and not base_known_labels_by_task and base_node_mask is None:
            return None
        return MaskedConditioningSpec(
            node_mask=base_node_mask,
            known_labels_by_task=base_known_labels_by_task,
            known_indices_by_task=base_known_indices_by_task,
            masked_tasks=list(base_masked_tasks or base_known_labels_by_task.keys()),
            constraint_mode=base_constraint_mode,
            feedback_mode=base_feedback_mode,
        )

    if not isinstance(masked_spec, dict):
        raise TypeError("masked_spec must be a dict")

    masked_tasks = _normalize_task_list(
        masked_spec.get("masked_tasks", base_masked_tasks),
        masked_spec.get("known_labels", None),
        tasks_num_classes,
    )
    if not masked_tasks and base_known_labels_by_task:
        masked_tasks = list(base_known_labels_by_task.keys())

    if not masked_tasks:
        raise ValueError("masked_spec requires non-empty masked_tasks")

    node_mask = _normalize_node_mask_spec(masked_spec.get("node_mask", None), num_nodes, device)
    if node_mask is None:
        node_mask = base_node_mask

    known_labels_by_task, known_indices_by_task = _normalize_known_labels(
        known_labels_spec=masked_spec.get("known_labels", None),
        masked_tasks=masked_tasks,
        num_nodes=num_nodes,
        tasks_num_classes=tasks_num_classes,
        device=device,
    )

    for task in masked_tasks:
        if task not in known_labels_by_task:
            known_labels_by_task[task] = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
            known_indices_by_task[task] = torch.zeros(0, dtype=torch.long, device=device)

    for task, known in base_known_labels_by_task.items():
        if task not in known_labels_by_task:
            continue
        if known.numel() != num_nodes:
            continue
        base_known = known.to(device=device, dtype=torch.long)
        override_mask = (known_labels_by_task[task] < 0) & (base_known >= 0)
        known_labels_by_task[task][override_mask] = base_known[override_mask]
        known_indices_by_task[task] = torch.where(known_labels_by_task[task] >= 0)[0]

    constraint_mode = masked_spec.get("constraint_mode", base_constraint_mode)
    feedback_mode = masked_spec.get("feedback_mode", base_feedback_mode)
    if constraint_mode not in {"hard", "soft"}:
        raise ValueError("constraint_mode must be 'hard' or 'soft'")
    if feedback_mode != "single_pass":
        raise ValueError("Only feedback_mode='single_pass' is currently supported")

    return MaskedConditioningSpec(
        node_mask=node_mask,
        known_labels_by_task=known_labels_by_task,
        known_indices_by_task=known_indices_by_task,
        masked_tasks=masked_tasks,
        constraint_mode=constraint_mode,
        feedback_mode=feedback_mode,
    )
