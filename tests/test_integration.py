#!/usr/bin/env python3
"""
Integration tests for AnalysisGNN
"""

import unittest
import tempfile
import os
import sys

# Add the parent directory to the path so we can import analysisgnn
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestAnalysisGNNIntegration(unittest.TestCase):
    """Integration tests for AnalysisGNN functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @unittest.skipIf('torch' not in sys.modules and 'torch' not in globals(), 
                     "PyTorch not available")
    def test_model_creation(self):
        """Test that we can create a model instance."""
        try:
            from analysisgnn.models.analysis import ContinualAnalysisGNN
            
            # Minimal configuration for testing
            config = {
                "model": "HybridGNN",
                "num_layers": 2,
                "hidden_channels": 64,
                "out_channels": 32,
                "dropout": 0.3,
                "lr": 0.01,
                "weight_decay": 0.001,
                "in_channels": 10,
                "task_dict": {
                    "cadence": 4,
                    "localkey": 12,
                },
                "metadata": {
                    "feature_size": 10
                }
            }
            
            # This should not raise an exception
            model = ContinualAnalysisGNN(config)
            self.assertIsNotNone(model)
            
        except ImportError:
            self.skipTest("Required dependencies not available")
        except Exception as e:
            # Some errors are expected due to missing graphmuse
            if "graphmuse" in str(e).lower():
                self.skipTest("GraphMuse dependency not available")
            else:
                raise
    
    def test_command_line_tools_exist(self):
        """Test that command line tools are available after installation."""
        # Test that the entry points are defined in setup.py
        import analysisgnn
        
        # Check if we can import the main functions
        try:
            from analysisgnn.train.train_analysisgnn import main as train_main
            self.assertTrue(callable(train_main))
        except ImportError as e:
            if "graphmuse" in str(e).lower():
                self.skipTest("GraphMuse dependency not available")
            else:
                raise
        
        try:
            from analysisgnn.inference.predict_analysis import main as predict_main
            self.assertTrue(callable(predict_main))
        except ImportError as e:
            if "graphmuse" in str(e).lower():
                self.skipTest("GraphMuse dependency not available")
            else:
                raise


if __name__ == '__main__':
    unittest.main(verbosity=2)
