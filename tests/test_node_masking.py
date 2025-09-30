#!/usr/bin/env python3
"""
Unit tests for semi-supervised node masking functionality.
"""

import unittest
import sys
import os
import torch
import torch.nn as nn

# Add the parent directory to the path so we can import analysisgnn
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysisgnn.utils.node_masking import (
    create_node_mask,
    split_nodes_by_mask,
    clamp_logits_to_labels,
    clamp_logits_dict,
    validate_node_mask,
)
from analysisgnn.models.chord import MultiTaskLoss


class TestNodeMaskCreation(unittest.TestCase):
    """Test node mask creation utilities."""
    
    def test_create_basic_mask(self):
        """Test basic node mask creation."""
        num_nodes = 100
        target_indices = torch.tensor([0, 1, 2, 3, 4])
        context_indices = torch.tensor([5, 6, 7, 8, 9])
        
        mask = create_node_mask(
            num_nodes, 
            target_indices=target_indices,
            context_indices=context_indices,
            context_weight=0.1
        )
        
        self.assertEqual(mask.shape, (num_nodes,))
        self.assertTrue((mask[target_indices] == 1.0).all())
        self.assertTrue((mask[context_indices] == 0.1).all())
        self.assertTrue((mask[10:] == 0.0).all())
    
    def test_mask_all_targets(self):
        """Test mask with only target nodes."""
        num_nodes = 10
        target_indices = torch.arange(num_nodes)
        
        mask = create_node_mask(num_nodes, target_indices=target_indices)
        
        self.assertTrue((mask == 1.0).all())
    
    def test_mask_with_device(self):
        """Test mask creation on specific device."""
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        
        num_nodes = 50
        mask = create_node_mask(num_nodes, device=device)
        
        self.assertEqual(mask.device.type, device.type)


class TestNodeMaskSplitting(unittest.TestCase):
    """Test splitting nodes by mask values."""
    
    def test_split_basic_mask(self):
        """Test splitting a basic mask."""
        mask = torch.tensor([1.0, 1.0, 1.0, 0.1, 0.1, 0.0, 0.0])
        
        target_idx, context_idx, unlabeled_idx = split_nodes_by_mask(mask)
        
        self.assertEqual(len(target_idx), 3)
        self.assertEqual(len(context_idx), 2)
        self.assertEqual(len(unlabeled_idx), 2)
        self.assertTrue((target_idx == torch.tensor([0, 1, 2])).all())
        self.assertTrue((context_idx == torch.tensor([3, 4])).all())
        self.assertTrue((unlabeled_idx == torch.tensor([5, 6])).all())
    
    def test_split_all_targets(self):
        """Test splitting when all nodes are targets."""
        mask = torch.ones(10)
        
        target_idx, context_idx, unlabeled_idx = split_nodes_by_mask(mask)
        
        self.assertEqual(len(target_idx), 10)
        self.assertEqual(len(context_idx), 0)
        self.assertEqual(len(unlabeled_idx), 0)


class TestLogitClamping(unittest.TestCase):
    """Test logit clamping for context nodes."""
    
    def test_clamp_basic_logits(self):
        """Test basic logit clamping."""
        num_nodes = 10
        num_classes = 5
        
        # Create random logits and labels
        logits = torch.randn(num_nodes, num_classes)
        labels = torch.randint(0, num_classes, (num_nodes,))
        context_indices = torch.tensor([2, 3, 4])
        
        clamped_logits = clamp_logits_to_labels(
            logits, labels, context_indices, num_classes
        )
        
        # Check that context nodes predict their labels correctly
        context_preds = clamped_logits[context_indices].argmax(dim=1)
        context_labels = labels[context_indices]
        self.assertTrue((context_preds == context_labels).all())
        
        # Check that non-context nodes are unchanged
        non_context_mask = torch.ones(num_nodes, dtype=torch.bool)
        non_context_mask[context_indices] = False
        self.assertTrue(torch.allclose(
            logits[non_context_mask], 
            clamped_logits[non_context_mask]
        ))
    
    def test_clamp_empty_context(self):
        """Test clamping with no context nodes."""
        logits = torch.randn(10, 5)
        labels = torch.randint(0, 5, (10,))
        context_indices = torch.tensor([])
        
        clamped_logits = clamp_logits_to_labels(
            logits, labels, context_indices
        )
        
        # Should be unchanged
        self.assertTrue(torch.allclose(logits, clamped_logits))
    
    def test_clamp_dict(self):
        """Test clamping for multiple tasks."""
        num_nodes = 10
        tasks = {
            'task1': 5,
            'task2': 8,
            'task3': 3
        }
        
        logits_dict = {
            task: torch.randn(num_nodes, n_classes) 
            for task, n_classes in tasks.items()
        }
        labels_dict = {
            task: torch.randint(0, n_classes, (num_nodes,))
            for task, n_classes in tasks.items()
        }
        context_indices = torch.tensor([1, 2, 3])
        
        clamped_dict = clamp_logits_dict(
            logits_dict, labels_dict, context_indices, tasks
        )
        
        # Check all tasks
        for task in tasks.keys():
            context_preds = clamped_dict[task][context_indices].argmax(dim=1)
            context_labels = labels_dict[task][context_indices]
            self.assertTrue((context_preds == context_labels).all())


class TestMaskValidation(unittest.TestCase):
    """Test mask validation utilities."""
    
    def test_valid_mask(self):
        """Test validation of a valid mask."""
        mask = torch.tensor([1.0] * 50 + [0.1] * 20 + [0.0] * 30)
        
        is_valid, message = validate_node_mask(mask)
        
        self.assertTrue(is_valid)
        self.assertEqual(message, "Mask is valid")
    
    def test_invalid_values(self):
        """Test validation fails for out-of-range values."""
        mask = torch.tensor([1.5, 1.0, 0.5, -0.1])
        
        is_valid, message = validate_node_mask(mask)
        
        self.assertFalse(is_valid)
        self.assertIn("range", message.lower())
    
    def test_no_targets(self):
        """Test validation fails when no targets exist."""
        mask = torch.tensor([0.1, 0.1, 0.0, 0.0])
        
        is_valid, message = validate_node_mask(mask)
        
        self.assertFalse(is_valid)
        self.assertIn("target", message.lower())
    
    def test_insufficient_targets(self):
        """Test validation fails with too few targets."""
        mask = torch.tensor([1.0] + [0.0] * 99)
        
        is_valid, message = validate_node_mask(mask, min_target_ratio=0.1)
        
        self.assertFalse(is_valid)


class TestMultiTaskLossWithMasking(unittest.TestCase):
    """Test MultiTaskLoss with node masking."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tasks = {'task1': 5, 'task2': 8}
        self.loss_ft = nn.ModuleDict({
            task: nn.CrossEntropyLoss(reduction='none') 
            for task in self.tasks.keys()
        })
        self.loss_module = MultiTaskLoss(
            list(self.tasks.keys()), 
            self.loss_ft,
            requires_grad=False,
            context_weight=0.1
        )
    
    def test_loss_without_mask(self):
        """Test loss computation without masking."""
        batch_size = 20
        pred = {
            task: torch.randn(batch_size, n_classes)
            for task, n_classes in self.tasks.items()
        }
        gt = {
            task: torch.randint(0, n_classes, (batch_size,))
            for task, n_classes in self.tasks.items()
        }
        
        loss_dict = self.loss_module(pred, gt)
        
        self.assertIn('total', loss_dict)
        self.assertTrue(loss_dict['total'].item() >= 0)
    
    def test_loss_with_target_mask(self):
        """Test loss computation with all target nodes."""
        batch_size = 20
        pred = {
            task: torch.randn(batch_size, n_classes)
            for task, n_classes in self.tasks.items()
        }
        gt = {
            task: torch.randint(0, n_classes, (batch_size,))
            for task, n_classes in self.tasks.items()
        }
        mask = torch.ones(batch_size)
        
        loss_dict = self.loss_module(pred, gt, node_mask=mask)
        
        self.assertIn('total', loss_dict)
        self.assertTrue(loss_dict['total'].item() >= 0)
    
    def test_loss_with_mixed_mask(self):
        """Test loss computation with mixed target/context/unlabeled nodes."""
        batch_size = 30
        pred = {
            task: torch.randn(batch_size, n_classes)
            for task, n_classes in self.tasks.items()
        }
        gt = {
            task: torch.randint(0, n_classes, (batch_size,))
            for task, n_classes in self.tasks.items()
        }
        # 10 targets, 10 context, 10 unlabeled
        mask = torch.cat([
            torch.ones(10),
            torch.full((10,), 0.1),
            torch.zeros(10)
        ])
        
        loss_dict = self.loss_module(pred, gt, node_mask=mask)
        
        self.assertIn('total', loss_dict)
        self.assertTrue(loss_dict['total'].item() >= 0)
    
    def test_loss_ignores_unlabeled(self):
        """Test that unlabeled nodes contribute zero loss."""
        batch_size = 10
        pred = {
            task: torch.randn(batch_size, n_classes)
            for task, n_classes in self.tasks.items()
        }
        gt = {
            task: torch.randint(0, n_classes, (batch_size,))
            for task, n_classes in self.tasks.items()
        }
        
        # Compute loss with all targets
        mask_all = torch.ones(batch_size)
        loss_all = self.loss_module(pred, gt, node_mask=mask_all)
        
        # Compute loss with half unlabeled
        mask_half = torch.cat([torch.ones(5), torch.zeros(5)])
        loss_half = self.loss_module(pred, gt, node_mask=mask_half)
        
        # Loss with half should be roughly half (accounting for normalization)
        # We just check it's positive and less than full loss
        self.assertTrue(loss_half['total'].item() > 0)
        self.assertTrue(loss_half['total'].item() <= loss_all['total'].item())


class TestIntegration(unittest.TestCase):
    """Integration tests for the full workflow."""
    
    def test_end_to_end_workflow(self):
        """Test complete workflow: mask creation -> loss computation -> clamping."""
        # Setup
        num_nodes = 50
        num_classes = 5
        tasks = {'task1': num_classes}
        
        # Create mask with targets, context, and unlabeled
        target_indices = torch.arange(0, 20)
        context_indices = torch.arange(20, 30)
        mask = create_node_mask(
            num_nodes,
            target_indices=target_indices,
            context_indices=context_indices,
            context_weight=0.1
        )
        
        # Validate mask
        is_valid, _ = validate_node_mask(mask)
        self.assertTrue(is_valid)
        
        # Split mask
        tgt_idx, ctx_idx, unl_idx = split_nodes_by_mask(mask)
        self.assertEqual(len(tgt_idx), 20)
        self.assertEqual(len(ctx_idx), 10)
        self.assertEqual(len(unl_idx), 20)
        
        # Create predictions and labels
        logits = torch.randn(num_nodes, num_classes)
        labels = torch.randint(0, num_classes, (num_nodes,))
        
        # Clamp logits for context nodes
        clamped_logits = clamp_logits_to_labels(
            logits, labels, ctx_idx, num_classes
        )
        
        # Verify context predictions are correct
        context_preds = clamped_logits[ctx_idx].argmax(dim=1)
        context_labels = labels[ctx_idx]
        self.assertTrue((context_preds == context_labels).all())
        
        # Compute loss with mask
        loss_ft = nn.ModuleDict({
            'task1': nn.CrossEntropyLoss(reduction='none')
        })
        loss_module = MultiTaskLoss(
            ['task1'], loss_ft, requires_grad=False, context_weight=0.1
        )
        
        loss_dict = loss_module(
            {'task1': clamped_logits},
            {'task1': labels},
            node_mask=mask
        )
        
        self.assertIn('total', loss_dict)
        self.assertTrue(loss_dict['total'].item() >= 0)


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)
