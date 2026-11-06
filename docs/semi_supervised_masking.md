# Semi-Supervised Node Masking in AnalysisGNN

## Overview

Semi-supervised node masking allows AnalysisGNN to make selective predictions on a subset of nodes while leveraging known labels on other nodes as contextual information. This feature enables more efficient training and improved predictions in scenarios where not all nodes require predictions or where partial ground-truth information is available.

## Node Types

The framework supports three types of nodes:

1. **Target (T) Nodes**: Nodes that require predictions and are fully evaluated
   - Mask value: `1.0`
   - Full loss contribution
   - Primary focus of prediction

2. **Context (C) Nodes**: Nodes with known labels used as conditioning
   - Mask value: `0.0 < value < 1.0` (typically `0.1`)
   - Down-weighted loss contribution
   - Labels fixed during prediction (logit clamping)
   - Provide supervisory signal without being primary targets

3. **Unlabeled (U) Nodes**: Nodes to ignore
   - Mask value: `0.0`
   - No loss contribution
   - Completely ignored during training

## Key Features

- **Loss Masking**: Computes primary loss on target nodes, auxiliary loss on context nodes, ignores unlabeled
- **Label Conditioning**: Injects known labels for context nodes with stop-gradient
- **Logit Clamping**: Replaces predicted logits with one-hot ground-truth for context nodes
- **Backward Compatible**: Works seamlessly with existing code when masks are not provided

## Quick Start

```python
from analysisgnn.utils.node_masking import create_node_mask

# Create mask
mask = create_node_mask(
    num_nodes=100,
    target_indices=torch.tensor([0, 1, 2, ...]),  # Predict these
    context_indices=torch.tensor([50, 51, ...]),   # Use as context
    context_weight=0.1
)

# Add to your graph
graph["note"].node_mask = mask

# Training automatically uses the mask!
```

## Use Cases

### 1. Phrase Boundary Analysis
- **Target**: Notes at phrase boundaries
- **Context**: Notes within phrases with known harmony
- **Benefit**: Focus predictions on structural points

### 2. Semi-Supervised Learning
- **Target**: Uncertain or ambiguous harmonies
- **Context**: Clear, high-confidence harmonies
- **Benefit**: Better use of limited annotations

### 3. Transfer Learning
- **Target**: Target domain notes
- **Context**: Source domain predictions (pseudo-labels)
- **Benefit**: Leverage existing model knowledge

### 4. Interactive Annotation
- **Target**: Unannotated notes
- **Context**: Expert-annotated notes (fixed)
- **Benefit**: Incorporates expert knowledge as constraints

## Examples

See `examples/semi_supervised_masking_example.py` for comprehensive examples.

Run:
```bash
python examples/semi_supervised_masking_example.py
```

## API Reference

### Core Functions

**`create_node_mask(num_nodes, target_indices, context_indices, context_weight)`**
- Creates a node mask tensor
- Returns: `torch.Tensor` of shape `[num_nodes]`

**`split_nodes_by_mask(mask)`**
- Splits mask into node type indices
- Returns: `(target_indices, context_indices, unlabeled_indices)`

**`clamp_logits_dict(logits_dict, labels_dict, context_indices, tasks)`**
- Clamps predictions for context nodes to ground truth
- Returns: Dictionary of clamped logits

**`validate_node_mask(mask)`**
- Validates a node mask for sanity
- Returns: `(is_valid, error_message)`

## Benefits

1. **Improved Efficiency**: Reduced computational cost by ignoring unlabeled nodes
2. **Better Predictions**: Context labels guide target predictions
3. **Sample Efficiency**: Better use of limited annotations
4. **Flexibility**: Supports various semi-supervised learning scenarios
5. **Easy Integration**: Minimal changes to existing code

## Implementation Details

The loss for a masked batch is computed as:

```
L = Σ_i [w_i * CrossEntropy(pred_i, label_i)] / Σ_i w_i

where w_i = {
    1.0          if mask_i > 0.9  (target)
    mask_i * α   if 0.01 < mask_i < 0.9  (context, α = context_weight)
    0.0          if mask_i < 0.01  (unlabeled)
}
```

Context nodes have their logits replaced with one-hot ground truth before loss computation, ensuring they always predict correctly.
