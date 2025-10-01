"""
Example: Semi-Supervised Node Masking for Selective Prediction

This example demonstrates how to use semi-supervised node masking in AnalysisGNN
to leverage ground-truth labels on some nodes (context) while predicting others (targets).

Use cases:
- Predicting harmony at phrase boundaries while using known harmonies within phrases
- Selective cadence detection with known harmonic context
- Incorporating expert annotations as fixed evidence during training
"""

import sys
import os

# Add parent directory to path if running standalone
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

try:
    # Try importing from installed package
    from analysisgnn.utils.node_masking import (
        create_node_mask,
        split_nodes_by_mask,
        clamp_logits_to_labels,
        clamp_logits_dict,
        validate_node_mask,
    )
    from analysisgnn.models.chord import MultiTaskLoss
except ImportError:
    # Fallback: Import modules directly without package initialization
    import importlib.util
    
    # Load node_masking module
    node_masking_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 
        'analysisgnn', 'utils', 'node_masking.py'
    )
    spec = importlib.util.spec_from_file_location('node_masking', node_masking_path)
    node_masking = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(node_masking)
    
    create_node_mask = node_masking.create_node_mask
    split_nodes_by_mask = node_masking.split_nodes_by_mask
    clamp_logits_to_labels = node_masking.clamp_logits_to_labels
    clamp_logits_dict = node_masking.clamp_logits_dict
    validate_node_mask = node_masking.validate_node_mask
    
    # For MultiTaskLoss, we'll use a simplified version
    class MultiTaskLoss(nn.Module):
        def __init__(self, tasks, loss_ft, requires_grad=True, context_weight=0.1):
            super().__init__()
            self.tasks = tasks
            self.loss_ft = loss_ft
            self.requires_grad = requires_grad
            self.context_weight = context_weight
            if requires_grad:
                self.params = nn.Parameter(torch.ones(len(tasks), requires_grad=True))
            else:
                self.params = torch.ones(len(tasks), requires_grad=False)

        def forward(self, pred, gt, node_mask=None):
            out = {}
            for task in gt.keys():
                task_loss = self.loss_ft[task](pred[task], gt[task])
                
                if node_mask is not None:
                    if task_loss.dim() == 0:
                        loss_fn = self.loss_ft[task]
                        if isinstance(loss_fn, nn.CrossEntropyLoss):
                            weight = loss_fn.weight
                            ignore_index = loss_fn.ignore_index if hasattr(loss_fn, 'ignore_index') else -100
                            label_smoothing = loss_fn.label_smoothing if hasattr(loss_fn, 'label_smoothing') else 0.0
                            unreduced_loss_fn = nn.CrossEntropyLoss(
                                weight=weight, 
                                ignore_index=ignore_index,
                                label_smoothing=label_smoothing,
                                reduction='none'
                            )
                            task_loss = unreduced_loss_fn(pred[task], gt[task])
                    
                    mask_weights = torch.where(node_mask > 0.5, 
                                              torch.ones_like(node_mask), 
                                              node_mask * self.context_weight)
                    task_loss = (task_loss * mask_weights).sum() / (mask_weights.sum() + 1e-8)
                else:
                    if task_loss.dim() > 0:
                        task_loss = task_loss.mean()
                
                out[task] = task_loss
            
            loss_sum = 0
            for i, loss in enumerate(out.values()):
                if self.requires_grad:
                    loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
                else:
                    loss_sum += loss
            out["total"] = loss_sum
            return out


def example_basic_masking():
    """Basic example: Creating and using node masks."""
    print("=" * 60)
    print("Example 1: Basic Node Masking")
    print("=" * 60)
    
    # Scenario: 100 notes in a musical passage
    num_nodes = 100
    
    # We want to predict harmony on the first 40 notes (targets)
    # Use the next 30 notes with known labels as context
    # Ignore the last 30 notes (unlabeled)
    target_indices = torch.arange(0, 40)
    context_indices = torch.arange(40, 70)
    
    # Create node mask
    mask = create_node_mask(
        num_nodes,
        target_indices=target_indices,
        context_indices=context_indices,
        context_weight=0.1  # Context nodes contribute 10% of target loss
    )
    
    print(f"Total nodes: {num_nodes}")
    print(f"Target nodes (full loss): {len(target_indices)} nodes")
    print(f"Context nodes (10% loss): {len(context_indices)} nodes")
    print(f"Unlabeled nodes (no loss): {num_nodes - 70} nodes")
    
    # Validate the mask
    is_valid, message = validate_node_mask(mask)
    print(f"\nMask validation: {is_valid}")
    print(f"Message: {message}")
    
    # Split mask back to indices
    tgt_idx, ctx_idx, unl_idx = split_nodes_by_mask(mask)
    print(f"\nRecovered indices:")
    print(f"  Targets: {len(tgt_idx)} nodes")
    print(f"  Context: {len(ctx_idx)} nodes")
    print(f"  Unlabeled: {len(unl_idx)} nodes")


def example_loss_computation():
    """Example: Computing masked loss for multi-task learning."""
    print("\n" + "=" * 60)
    print("Example 2: Masked Loss Computation")
    print("=" * 60)
    
    # Setup tasks
    tasks = {
        'localkey': 50,  # 50 possible local keys
        'degree': 22,    # 22 degree classes
        'quality': 15,   # 15 chord qualities
    }
    
    # Create loss module with node masking support
    loss_ft = nn.ModuleDict({
        task: nn.CrossEntropyLoss(reduction='none')
        for task in tasks.keys()
    })
    loss_module = MultiTaskLoss(
        list(tasks.keys()),
        loss_ft,
        requires_grad=False,
        context_weight=0.1  # Context nodes contribute 10% of loss
    )
    
    # Simulate predictions and labels
    batch_size = 50
    predictions = {
        task: torch.randn(batch_size, n_classes)
        for task, n_classes in tasks.items()
    }
    labels = {
        task: torch.randint(0, n_classes, (batch_size,))
        for task, n_classes in tasks.items()
    }
    
    # Create mask: 30 targets, 10 context, 10 unlabeled
    mask = create_node_mask(
        batch_size,
        target_indices=torch.arange(0, 30),
        context_indices=torch.arange(30, 40),
        context_weight=0.1
    )
    
    # Compute losses
    loss_without_mask = loss_module(predictions, labels)
    loss_with_mask = loss_module(predictions, labels, node_mask=mask)
    
    print(f"Loss without mask: {loss_without_mask['total'].item():.4f}")
    print(f"Loss with mask: {loss_with_mask['total'].item():.4f}")
    print(f"\nReduction: {(1 - loss_with_mask['total'] / loss_without_mask['total']) * 100:.1f}%")
    print("(Loss is reduced because we ignore unlabeled nodes and down-weight context)")


def example_logit_clamping():
    """Example: Clamping logits for context nodes to ground truth."""
    print("\n" + "=" * 60)
    print("Example 3: Logit Clamping for Context Nodes")
    print("=" * 60)
    
    # Setup
    num_nodes = 20
    num_classes = 10
    
    # Simulate model predictions and ground truth
    logits = torch.randn(num_nodes, num_classes)
    labels = torch.randint(0, num_classes, (num_nodes,))
    
    # Define context nodes (indices 5-9)
    context_indices = torch.arange(5, 10)
    
    print(f"Total nodes: {num_nodes}")
    print(f"Context nodes: {context_indices.tolist()}")
    
    # Check accuracy before clamping
    predictions_before = logits.argmax(dim=1)
    accuracy_before = (predictions_before == labels).float().mean()
    context_accuracy_before = (predictions_before[context_indices] == labels[context_indices]).float().mean()
    
    print(f"\nBefore clamping:")
    print(f"  Overall accuracy: {accuracy_before.item():.2%}")
    print(f"  Context accuracy: {context_accuracy_before.item():.2%}")
    
    # Clamp logits for context nodes (use the imported function)
    clamped_logits = clamp_logits_to_labels(
        logits, labels, context_indices, num_classes
    )
    
    # Check accuracy after clamping
    predictions_after = clamped_logits.argmax(dim=1)
    accuracy_after = (predictions_after == labels).float().mean()
    context_accuracy_after = (predictions_after[context_indices] == labels[context_indices]).float().mean()
    
    print(f"\nAfter clamping:")
    print(f"  Overall accuracy: {accuracy_after.item():.2%}")
    print(f"  Context accuracy: {context_accuracy_after.item():.2%}")
    print(f"\n✓ Context nodes now predict their ground truth labels!")


def example_training_integration():
    """Example: Integration with PyTorch Lightning training."""
    print("\n" + "=" * 60)
    print("Example 4: Training Integration")
    print("=" * 60)
    
    print("""
    To use node masking in your training pipeline:
    
    1. Add node masks to your data:
       
       ```python
       # In your dataset or data module
       graph["note"].node_mask = create_node_mask(
           num_nodes=len(graph["note"].x),
           target_indices=target_indices,
           context_indices=context_indices,
           context_weight=0.1
       )
       ```
    
    2. The training step will automatically detect and use the mask:
       
       ```python
       # In ChordPredictionPLModel.training_step
       # The mask is automatically extracted if present
       node_mask = graph["note"].node_mask[:batch_size] if hasattr(...) else None
       
       # Logits are clamped for context nodes
       if node_mask is not None:
           batch_pred = clamp_logits_dict(batch_pred, batch_labels, context_indices, tasks)
       
       # Loss computation uses the mask
       loss = self.train_loss(batch_pred, batch_labels, node_mask=node_mask)
       ```
    
    3. Benefits:
       - Context nodes guide predictions with ground truth
       - Reduced computational cost (no loss on unlabeled nodes)
       - Better sample efficiency in low-data regimes
       - Natural way to incorporate expert annotations
    """)


def example_use_cases():
    """Example: Real-world use cases."""
    print("\n" + "=" * 60)
    print("Example 5: Real-World Use Cases")
    print("=" * 60)
    
    print("""
    Use Case 1: Phrase Boundary Analysis
    ────────────────────────────────────
    Problem: Predict cadences at phrase boundaries.
    Solution: Mark phrase-internal notes as context, boundaries as targets.
    
    - Target nodes (T): Notes at phrase boundaries
    - Context nodes (C): Notes within phrases with known harmony
    - Unlabeled nodes (U): Non-harmonic passing tones
    
    Benefits:
    - Focus predictions on important structural points
    - Use stable harmonic context to improve boundary detection
    
    
    Use Case 2: Semi-Supervised Learning
    ────────────────────────────────────
    Problem: Limited expert annotations available.
    Solution: Use high-confidence predictions as context for uncertain ones.
    
    - Target nodes (T): Uncertain or ambiguous harmonies
    - Context nodes (C): Clear, unambiguous harmonies
    - Unlabeled nodes (U): Non-chord tones
    
    Benefits:
    - Better use of limited annotations
    - Iterative refinement (update context based on confident predictions)
    
    
    Use Case 3: Transfer Learning
    ────────────────────────────────
    Problem: Adapt model from one style/corpus to another.
    Solution: Use source domain predictions as context in target domain.
    
    - Target nodes (T): Target domain notes needing prediction
    - Context nodes (C): Source domain predictions (pseudo-labels)
    - Unlabeled nodes (U): Low-confidence predictions
    
    Benefits:
    - Leverage existing model knowledge
    - Gradual adaptation to new domain
    
    
    Use Case 4: Interactive Annotation
    ──────────────────────────────────
    Problem: Help musicologists with partially completed analyses.
    Solution: Fix expert annotations, predict remaining nodes.
    
    - Target nodes (T): Unannotated notes
    - Context nodes (C): Expert-annotated notes (fixed)
    - Unlabeled nodes (U): Skipped or deleted annotations
    
    Benefits:
    - Incorporates expert knowledge as hard constraints
    - Speeds up annotation process
    - Ensures consistency with expert decisions
    """)


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("Semi-Supervised Node Masking Examples for AnalysisGNN")
    print("=" * 60 + "\n")
    
    example_basic_masking()
    example_loss_computation()
    example_logit_clamping()
    example_training_integration()
    example_use_cases()
    
    print("\n" + "=" * 60)
    print("Examples completed successfully!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
