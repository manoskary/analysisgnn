"""
Utilities for normalizing user edit specs into node masks and label overrides.

User edit spec format (dict):
    {
        "node_mask": {
            "targets": [0, 1, 2],
            "context": [10, 11],
            "unlabeled": [20, 21],
            "context_weight": 0.1
        },
        "label_overrides": {
            "romanNumeral": {
                "indices": [10, 11],
                "labels": ["V", "I6"]
            },
            "localkey": {
                "indices": [10],
                "labels": [5]
            }
        },
        "context_weight": 0.1
    }

Notes:
- Indices refer to the note order used in prediction: sorted by onset_div, then pitch.
- labels can be int class indices, or strings for tasks in available_representations.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from analysisgnn.utils.chord_representations import available_representations
from analysisgnn.utils.node_masking import create_node_mask


def _normalize_indices(value: Any, name: str) -> list:
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


def _validate_indices(indices: list, num_nodes: int, name: str) -> None:
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
                f"Label id {label_id} out of range for task '{task}' (0..{tasks_num_classes[task] - 1})"
            )
    return label_id


def normalize_user_edits(
    user_edits: Optional[Dict[str, Any]],
    num_nodes: int,
    tasks_num_classes: Optional[Dict[str, int]] = None,
    device: Optional[torch.device] = None,
) -> Tuple[Optional[torch.Tensor], Dict[str, Dict[str, torch.Tensor]]]:
    """
    Normalize a user edit spec into a node mask and per-task label overrides.

    Returns:
        node_mask: Tensor of shape [num_nodes] or None
        overrides: Dict[task] -> {"indices": Tensor, "labels": Tensor}
    """
    if user_edits is None:
        return None, {}
    if not isinstance(user_edits, dict):
        raise TypeError("user_edits must be a dict")

    if device is None:
        device = torch.device("cpu")

    context_weight = user_edits.get("context_weight", 0.1)
    node_mask_spec = user_edits.get("node_mask", None)
    label_overrides = user_edits.get("label_overrides", {})

    overrides: Dict[str, Dict[str, torch.Tensor]] = {}
    for task, spec in label_overrides.items():
        if spec is None:
            continue
        if not isinstance(spec, dict):
            raise TypeError(f"label_overrides for task '{task}' must be a dict")
        indices = _normalize_indices(spec.get("indices", None), f"label_overrides.{task}.indices")
        labels = spec.get("labels", None)
        if labels is None:
            raise ValueError(f"label_overrides.{task}.labels is required")
        if torch.is_tensor(labels):
            labels = labels.detach().cpu().tolist()
        if isinstance(labels, np.ndarray):
            labels = labels.tolist()
        if isinstance(labels, (list, tuple)):
            label_list = list(labels)
        else:
            label_list = [labels]

        if len(indices) == 0:
            continue
        if len(label_list) == 1 and len(indices) > 1:
            label_list = label_list * len(indices)
        if len(label_list) != len(indices):
            raise ValueError(f"label_overrides.{task} labels/indices length mismatch")

        _validate_indices(indices, num_nodes, f"label_overrides.{task}.indices")
        encoded = [_encode_label(task, v, tasks_num_classes) for v in label_list]
        labels_tensor = torch.zeros(num_nodes, dtype=torch.long, device=device)
        indices_tensor = torch.tensor(indices, dtype=torch.long, device=device)
        labels_tensor[indices_tensor] = torch.tensor(encoded, dtype=torch.long, device=device)
        overrides[task] = {"indices": indices_tensor, "labels": labels_tensor}

    node_mask = None
    if isinstance(node_mask_spec, dict):
        context_weight = node_mask_spec.get("context_weight", context_weight)
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

        node_mask = create_node_mask(
            num_nodes=num_nodes,
            target_indices=torch.tensor(targets, dtype=torch.long, device=device),
            context_indices=torch.tensor(context, dtype=torch.long, device=device) if context else None,
            unlabeled_indices=torch.tensor(unlabeled, dtype=torch.long, device=device) if unlabeled else None,
            context_weight=context_weight,
            device=device,
        )
    elif node_mask_spec is not None:
        if torch.is_tensor(node_mask_spec):
            mask_tensor = node_mask_spec.detach().to(device=device, dtype=torch.float32)
        else:
            mask_tensor = torch.tensor(node_mask_spec, dtype=torch.float32, device=device)
        if mask_tensor.numel() != num_nodes:
            raise ValueError(f"node_mask length {mask_tensor.numel()} != num_nodes {num_nodes}")
        if mask_tensor.min() < 0 or mask_tensor.max() > 1:
            raise ValueError("node_mask values must be in [0, 1]")
        node_mask = mask_tensor
    elif overrides:
        context_indices = sorted(
            set().union(*[set(v["indices"].detach().cpu().tolist()) for v in overrides.values()])
        )
        targets = sorted(set(range(num_nodes)) - set(context_indices))
        node_mask = create_node_mask(
            num_nodes=num_nodes,
            target_indices=torch.tensor(targets, dtype=torch.long, device=device),
            context_indices=torch.tensor(context_indices, dtype=torch.long, device=device) if context_indices else None,
            context_weight=context_weight,
            device=device,
        )

    return node_mask, overrides
