#!/usr/bin/env python3
"""
Unit tests for masked conditioning normalization utilities.
"""

import os
import sys
import unittest

import torch

# Add the parent directory to the path so we can import analysisgnn
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysisgnn.utils.masked_conditioning import (
    build_known_labels_from_overrides,
    normalize_masked_conditioning_spec,
)
from analysisgnn.utils.user_edits import normalize_user_edits_to_masked_conditioning


TASKS = {
    "romanNumeral": 185,
    "localkey": 50,
    "quality": 15,
}


class TestMaskedConditioningSpec(unittest.TestCase):
    def test_normalize_masked_spec_indices_labels(self):
        spec = {
            "masked_tasks": ["romanNumeral", "localkey"],
            "known_labels": {
                "romanNumeral": {"indices": [1, 3], "labels": [10, 12]},
                "localkey": {"indices": [0], "labels": [4]},
            },
            "constraint_mode": "hard",
            "feedback_mode": "single_pass",
        }
        normalized = normalize_masked_conditioning_spec(
            masked_spec=spec,
            num_nodes=6,
            tasks_num_classes=TASKS,
            device=torch.device("cpu"),
        )
        self.assertIsNotNone(normalized)
        self.assertEqual(normalized.masked_tasks, ["romanNumeral", "localkey"])
        self.assertEqual(int(normalized.known_labels_by_task["romanNumeral"][1].item()), 10)
        self.assertEqual(int(normalized.known_labels_by_task["romanNumeral"][3].item()), 12)
        self.assertEqual(int(normalized.known_labels_by_task["localkey"][0].item()), 4)
        self.assertEqual(normalized.known_indices_by_task["romanNumeral"].numel(), 2)

    def test_build_known_labels_from_overrides(self):
        overrides = {
            "romanNumeral": {
                "indices": torch.tensor([0, 2], dtype=torch.long),
                "labels": torch.tensor([5, -1, 7, -1], dtype=torch.long),
            }
        }
        known_labels_by_task, known_indices_by_task = build_known_labels_from_overrides(
            overrides=overrides,
            num_nodes=4,
            tasks_num_classes=TASKS,
            device=torch.device("cpu"),
        )
        self.assertIn("romanNumeral", known_labels_by_task)
        self.assertEqual(int(known_labels_by_task["romanNumeral"][0].item()), 5)
        self.assertEqual(int(known_labels_by_task["romanNumeral"][2].item()), 7)
        self.assertEqual(known_indices_by_task["romanNumeral"].tolist(), [0, 2])


class TestUserEditsIntegration(unittest.TestCase):
    def test_user_edits_to_masked_conditioning(self):
        user_edits = {
            "node_mask": {
                "targets": [2, 3, 4],
                "context": [0, 1],
                "context_weight": 0.1,
            },
            "label_overrides": {
                "romanNumeral": {"indices": [0, 1], "labels": [8, 9]},
                "localkey": {"indices": [0], "labels": [2]},
            },
        }
        node_mask, overrides, conditioning = normalize_user_edits_to_masked_conditioning(
            user_edits=user_edits,
            num_nodes=5,
            tasks_num_classes=TASKS,
            device=torch.device("cpu"),
            masked_spec=None,
        )
        self.assertIsNotNone(node_mask)
        self.assertIn("romanNumeral", overrides)
        self.assertIsNotNone(conditioning)
        self.assertIn("romanNumeral", conditioning.known_labels_by_task)
        self.assertEqual(int(conditioning.known_labels_by_task["romanNumeral"][0].item()), 8)
        self.assertEqual(int(conditioning.known_labels_by_task["romanNumeral"][1].item()), 9)
        self.assertEqual(conditioning.constraint_mode, "hard")
        self.assertEqual(conditioning.feedback_mode, "single_pass")


if __name__ == "__main__":
    unittest.main(verbosity=2)
