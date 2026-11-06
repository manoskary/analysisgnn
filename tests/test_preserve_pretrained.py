#!/usr/bin/env python3
"""
Unit tests for pretrained-preservation utilities in ContinualAnalysisGNN.
"""

import os
import sys
import unittest

import torch

# Add repo root for local imports.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysisgnn.models.analysis import ContinualAnalysisGNN


def _tiny_hparams():
    return {
        "model": "HybridGNN",
        "disable_graph_encoder": False,
        "metadata": (
            ["note"],
            [
                ("note", "onset", "note"),
                ("note", "consecutive", "note"),
                ("note", "during", "note"),
                ("note", "rest", "note"),
                ("note", "consecutive_rev", "note"),
                ("note", "during_rev", "note"),
                ("note", "rest_rev", "note"),
            ],
        ),
        "in_channels": 16,
        "hidden_channels": 32,
        "out_channels": 16,
        "task_dict": {
            "romanNumeral": 7,
            "localkey": 5,
        },
        "num_layers": 2,
        "dropout": 0.1,
        "num_epochs": 3,
        "epochs_per_task": [3],
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "main_tasks": ["all"],
        "lambda_dctn": 0.0,
        "lambda_featl": 0.0,
        "lambda_ewc": 0.0,
        "use_ewc": False,
        "cl_training": False,
        "has_memories": False,
        "use_smote": False,
        "mt_strategy": "fixed",
        "masked_prediction_train": True,
        "masked_tasks": ["romanNumeral"],
        "mask_ratio": 0.15,
        "known_ratio": 0.15,
        "mask_sampling_policy": "hybrid",
        "constraint_mode": "hard",
        "feedback_mode": "single_pass",
        "preserve_pretrained": True,
        "preserve_kd_lambda": 1.0,
        "preserve_feat_lambda": 0.1,
        "preserve_l2sp_lambda": 1e-4,
        "preserve_temperature": 2.0,
        "preserve_tasks": ["all_nonmasked"],
        "unmasked_batch_prob": 0.30,
        "freeze_graph_encoder_stage_epochs": 1,
        "preserve_stage_b_epochs": 1,
        "preserve_stage_a_lr": 1e-4,
        "preserve_stage_b_lr": 5e-5,
        "preserve_stage_c_lr": 2e-5,
        "preserve_max_regression_abs": 0.015,
    }


class TestPreservePretrained(unittest.TestCase):
    def setUp(self):
        self.model = ContinualAnalysisGNN(_tiny_hparams(), note_encoder=None)
        self.model.train()

    def test_teacher_is_frozen(self):
        self.model._ensure_preservation_teacher_ready()
        teacher_model = self.model.__dict__.get("_preserve_teacher_model")
        self.assertIsNotNone(teacher_model)
        self.assertFalse(any(p.requires_grad for p in teacher_model.parameters()))

    def test_kd_mask_excludes_masked_targets(self):
        labels = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        # target, context, unlabeled, target
        raw_mask = torch.tensor([1.0, 0.1, 0.0, 1.0], dtype=torch.float32)
        mask = self.model._build_kd_task_mask("romanNumeral", labels, raw_mask)
        self.assertEqual(mask.tolist(), [False, True, True, False])

    def test_l2sp_zero_when_unchanged(self):
        self.model._init_l2sp_anchor_if_needed()
        penalty = self.model._compute_l2sp_penalty(device=torch.device("cpu"))
        self.assertTrue(torch.isclose(penalty, torch.tensor(0.0), atol=1e-8))

    def test_unmasked_batch_toggle(self):
        self.model.unmasked_batch_prob = 1.0
        self.assertTrue(self.model._should_use_unmasked_batch())
        self.model.unmasked_batch_prob = 0.0
        self.assertFalse(self.model._should_use_unmasked_batch())


if __name__ == "__main__":
    unittest.main(verbosity=2)
