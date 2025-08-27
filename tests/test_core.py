#!/usr/bin/env python3
"""
Unit tests for AnalysisGNN core functionality
"""

import unittest
import sys
import os

# Add the parent directory to the path so we can import analysisgnn
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestAnalysisGNNImports(unittest.TestCase):
    """Test that core modules can be imported."""
    
    def test_package_import(self):
        """Test that the main package can be imported."""
        try:
            import analysisgnn
            self.assertTrue(hasattr(analysisgnn, '__version__'))
        except ImportError as e:
            self.fail(f"Failed to import analysisgnn: {e}")
    
    def test_model_import(self):
        """Test that the main model can be imported."""
        try:
            from analysisgnn.models.analysis import ContinualAnalysisGNN
            self.assertTrue(callable(ContinualAnalysisGNN))
        except ImportError as e:
            self.fail(f"Failed to import ContinualAnalysisGNN: {e}")
    
    def test_datamodule_import(self):
        """Test that the data module can be imported."""
        try:
            from analysisgnn.data.datamodules.analysis import AnalysisDataModule
            self.assertTrue(callable(AnalysisDataModule))
        except ImportError as e:
            self.fail(f"Failed to import AnalysisDataModule: {e}")
    
    def test_utils_import(self):
        """Test that utility modules can be imported."""
        try:
            from analysisgnn.utils import chord_representations
            self.assertTrue(hasattr(chord_representations, 'available_representations'))
        except ImportError as e:
            self.fail(f"Failed to import chord_representations: {e}")


class TestDependencies(unittest.TestCase):
    """Test that required dependencies are available."""
    
    def test_torch_import(self):
        """Test that PyTorch is available."""
        try:
            import torch
            self.assertTrue(hasattr(torch, '__version__'))
        except ImportError as e:
            self.fail(f"PyTorch not available: {e}")
    
    def test_pytorch_lightning_import(self):
        """Test that PyTorch Lightning is available."""
        try:
            import pytorch_lightning as pl
            self.assertTrue(hasattr(pl, '__version__'))
        except ImportError as e:
            self.fail(f"PyTorch Lightning not available: {e}")
    
    def test_torch_geometric_import(self):
        """Test that PyTorch Geometric is available."""
        try:
            import torch_geometric
            self.assertTrue(hasattr(torch_geometric, '__version__'))
        except ImportError as e:
            self.fail(f"PyTorch Geometric not available: {e}")
    
    def test_partitura_import(self):
        """Test that Partitura is available."""
        try:
            import partitura as pt
            self.assertTrue(hasattr(pt, '__version__'))
        except ImportError as e:
            self.fail(f"Partitura not available: {e}")
    
    def test_music21_import(self):
        """Test that Music21 is available."""
        try:
            import music21
            self.assertTrue(hasattr(music21, '__version__'))
        except ImportError as e:
            self.fail(f"Music21 not available: {e}")


class TestConfiguration(unittest.TestCase):
    """Test configuration and setup."""
    
    def test_task_dict_exists(self):
        """Test that the task dictionary is properly defined."""
        from analysisgnn.train.train_analysisgnn import TASK_DICT
        self.assertIsInstance(TASK_DICT, dict)
        self.assertGreater(len(TASK_DICT), 0)
        
        # Check some expected tasks
        expected_tasks = ['cadence', 'localkey', 'romanNumeral']
        for task in expected_tasks:
            self.assertIn(task, TASK_DICT)
            self.assertIsInstance(TASK_DICT[task], int)


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)
