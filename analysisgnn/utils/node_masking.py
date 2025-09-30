"""
Utilities for semi-supervised node masking in AnalysisGNN.

This module provides utilities for managing three types of nodes:
- T (Target): nodes to predict and evaluate (mask value = 1.0)
- C (Context): nodes with known labels for conditioning (mask value = 0.0 < x < 1.0)
- U (Unlabeled): nodes to ignore (mask value = 0.0)
"""

import torch
from typing import Optional, Dict, Tuple


def create_node_mask(
    num_nodes: int,
    target_indices: Optional[torch.Tensor] = None,
    context_indices: Optional[torch.Tensor] = None,
    unlabeled_indices: Optional[torch.Tensor] = None,
    context_weight: float = 0.1,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Create a node mask tensor for semi-supervised learning.
    
    Args:
        num_nodes: Total number of nodes
        target_indices: Indices of target nodes (T) to predict and evaluate
        context_indices: Indices of context nodes (C) with known labels
        unlabeled_indices: Indices of unlabeled nodes (U) to ignore
        context_weight: Weight for context nodes (default: 0.1)
        device: Device to create the mask on
    
    Returns:
        Tensor of shape [num_nodes] with mask values:
        - 1.0 for target nodes
        - context_weight for context nodes
        - 0.0 for unlabeled nodes
    """
    if device is None:
        device = torch.device('cpu')
    
    # Initialize mask with zeros (unlabeled by default)
    mask = torch.zeros(num_nodes, dtype=torch.float32, device=device)
    
    # Set target nodes to 1.0
    if target_indices is not None:
        mask[target_indices] = 1.0
    
    # Set context nodes to context_weight
    if context_indices is not None:
        mask[context_indices] = context_weight
    
    # Explicitly set unlabeled nodes to 0.0 (redundant but explicit)
    if unlabeled_indices is not None:
        mask[unlabeled_indices] = 0.0
    
    return mask


def split_nodes_by_mask(
    mask: torch.Tensor,
    target_threshold: float = 0.9,
    context_threshold: float = 0.01
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Split nodes into target, context, and unlabeled based on mask values.
    
    Args:
        mask: Node mask tensor
        target_threshold: Minimum value to be considered a target node
        context_threshold: Minimum value to be considered a context node
    
    Returns:
        Tuple of (target_indices, context_indices, unlabeled_indices)
    """
    target_indices = torch.where(mask >= target_threshold)[0]
    context_indices = torch.where((mask >= context_threshold) & (mask < target_threshold))[0]
    unlabeled_indices = torch.where(mask < context_threshold)[0]
    
    return target_indices, context_indices, unlabeled_indices


def clamp_logits_to_labels(
    logits: torch.Tensor,
    labels: torch.Tensor,
    context_indices: torch.Tensor,
    num_classes: Optional[int] = None,
    temperature: float = 1.0
) -> torch.Tensor:
    """
    Replace logits at context node positions with one-hot encoded ground truth.
    
    This ensures context nodes always predict their known labels correctly,
    effectively acting as fixed evidence in the model.
    
    Args:
        logits: Predicted logits of shape [num_nodes, num_classes]
        labels: Ground truth labels of shape [num_nodes]
        context_indices: Indices of context nodes
        num_classes: Number of classes (inferred from logits if not provided)
        temperature: Temperature for one-hot encoding (default: 1.0 for hard labels)
    
    Returns:
        Logits with context nodes replaced by one-hot ground truth
    """
    if num_classes is None:
        num_classes = logits.shape[-1]
    
    # Clone logits to avoid in-place modification
    clamped_logits = logits.clone()
    
    # Create one-hot encoding for context node labels
    if len(context_indices) > 0:
        context_labels = labels[context_indices]
        # Use very large values (scaled by temperature) for the correct class
        one_hot = torch.zeros(len(context_indices), num_classes, 
                            dtype=logits.dtype, device=logits.device)
        one_hot.scatter_(1, context_labels.unsqueeze(1), 1.0 / temperature)
        
        # Replace context node logits with one-hot values (scaled to logit space)
        # Use large values to ensure correct predictions
        clamped_logits[context_indices] = one_hot * 10.0
    
    return clamped_logits


def clamp_logits_dict(
    logits_dict: Dict[str, torch.Tensor],
    labels_dict: Dict[str, torch.Tensor],
    context_indices: torch.Tensor,
    tasks_num_classes: Optional[Dict[str, int]] = None,
    temperature: float = 1.0
) -> Dict[str, torch.Tensor]:
    """
    Apply logit clamping to all tasks in a multi-task setting.
    
    Args:
        logits_dict: Dictionary of task predictions
        labels_dict: Dictionary of task labels
        context_indices: Indices of context nodes
        tasks_num_classes: Optional dictionary of number of classes per task
        temperature: Temperature for one-hot encoding
    
    Returns:
        Dictionary of clamped logits per task
    """
    clamped_dict = {}
    
    for task_name in logits_dict.keys():
        if task_name in labels_dict:
            num_classes = tasks_num_classes.get(task_name) if tasks_num_classes else None
            clamped_dict[task_name] = clamp_logits_to_labels(
                logits_dict[task_name],
                labels_dict[task_name],
                context_indices,
                num_classes=num_classes,
                temperature=temperature
            )
        else:
            clamped_dict[task_name] = logits_dict[task_name]
    
    return clamped_dict


def create_label_embeddings(
    labels: torch.Tensor,
    num_classes: int,
    embedding_dim: int,
    mask: torch.Tensor,
    mask_token_id: Optional[int] = None
) -> torch.Tensor:
    """
    Create label embeddings for conditioning, with [MASK] tokens for non-context nodes.
    
    Args:
        labels: Ground truth labels of shape [num_nodes]
        num_classes: Number of classes
        embedding_dim: Dimension of embeddings
        mask: Node mask tensor (context nodes have values > 0 and < 1)
        mask_token_id: ID to use for masked positions (default: num_classes)
    
    Returns:
        Label embeddings ready for injection into the model
    """
    if mask_token_id is None:
        mask_token_id = num_classes  # Use an extra ID for [MASK]
    
    # Clone labels and replace non-context positions with mask token
    masked_labels = labels.clone()
    non_context = mask < 0.5  # Assuming context has weight >= 0.1
    masked_labels[non_context] = mask_token_id
    
    # Create embedding layer (this should ideally be created once and reused)
    embedding = torch.nn.Embedding(num_classes + 1, embedding_dim)  # +1 for [MASK]
    label_embeds = embedding(masked_labels.long())
    
    # Apply stop-gradient for context nodes
    label_embeds = label_embeds.detach()
    
    return label_embeds


def validate_node_mask(
    mask: torch.Tensor,
    min_target_ratio: float = 0.1,
    max_context_ratio: float = 0.9
) -> Tuple[bool, str]:
    """
    Validate a node mask to ensure it meets basic sanity checks.
    
    Args:
        mask: Node mask tensor
        min_target_ratio: Minimum ratio of target nodes
        max_context_ratio: Maximum ratio of context nodes
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if mask.min() < 0 or mask.max() > 1:
        return False, "Mask values must be in range [0, 1]"
    
    target_count = (mask > 0.9).sum().item()
    context_count = ((mask > 0.01) & (mask <= 0.9)).sum().item()
    total_count = mask.numel()
    
    target_ratio = target_count / total_count
    context_ratio = context_count / total_count
    
    if target_ratio < min_target_ratio:
        return False, f"Target ratio {target_ratio:.3f} is below minimum {min_target_ratio}"
    
    if context_ratio > max_context_ratio:
        return False, f"Context ratio {context_ratio:.3f} exceeds maximum {max_context_ratio}"
    
    if target_count == 0:
        return False, "No target nodes found in mask"
    
    return True, "Mask is valid"
