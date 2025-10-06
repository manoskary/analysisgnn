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
    
    def test_model_creation(self):
        """Test that we can create a model instance."""
        try:
            from analysisgnn.models.analysis import ContinualAnalysisGNN
            
            # Minimal configuration for testing with correct metadata structure
            # metadata should be a tuple of (node_types, edge_types)
            node_types = ['note', 'beat', 'measure']
            edge_types = [
                ('note', 'onset', 'note'),
                ('note', 'consecutive', 'note'),
                ('note', 'during', 'beat'),
                ('beat', 'next', 'beat')
            ]
            
            config = {
                "model": "hybridgnn",
                "num_layers": 2,
                "hidden_channels": 64,
                "out_channels": 32,
                "dropout": 0.3,
                "lr": 0.001,
                "weight_decay": 0.0001,
                "in_channels": 10,
                "num_epochs": 10,
                "task_dict": {
                    "cadence": 4,
                    "localkey": 12,
                },
                "metadata": (node_types, edge_types),
                "main_tasks": ["cadence", "localkey"]
            }
            
            # This should not raise an exception
            model = ContinualAnalysisGNN(config)
            self.assertIsNotNone(model)
            self.assertTrue(hasattr(model, 'model'))
            
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
    
    def test_inference_dry_run(self):
        """Test that inference can be initialized (dry run without actual prediction)."""
        try:
            from analysisgnn.inference.predict_analysis import load_model, download_default_score
            import os
            
            # Check if cached model exists
            model_path = "./artifacts/models/model.ckpt"
            if not os.path.exists(model_path):
                self.skipTest("Model checkpoint not available for testing")
            
            # Try to load the model
            device = "cpu"
            model = load_model(model_path, device)
            self.assertIsNotNone(model)
            
            # Check if default score can be obtained
            score_path = download_default_score()
            self.assertTrue(os.path.exists(score_path))
            
        except ImportError as e:
            if "graphmuse" in str(e).lower():
                self.skipTest("GraphMuse dependency not available")
            else:
                raise
        except Exception as e:
            # Allow certain expected errors
            if "checkpoint" in str(e).lower() or "artifact" in str(e).lower():
                self.skipTest(f"Checkpoint not available: {str(e)}")
            else:
                raise
    
    def test_training_parser(self):
        """Test that training argument parser works correctly."""
        try:
            from analysisgnn.train.train_analysisgnn import get_parser
            
            parser = get_parser()
            self.assertIsNotNone(parser)
            
            # Test parsing default arguments
            args = parser.parse_args([])
            self.assertEqual(args.gpus, "-1")  # Default CPU
            self.assertEqual(args.num_layers, 3)
            self.assertEqual(args.model, "HybridGNN")
            
            # Test parsing custom arguments
            args = parser.parse_args([
                "--num_layers", "2",
                "--hidden_channels", "128",
                "--model", "HGT"
            ])
            self.assertEqual(args.num_layers, 2)
            self.assertEqual(args.hidden_channels, 128)
            self.assertEqual(args.model, "HGT")
            
        except ImportError as e:
            if "graphmuse" in str(e).lower():
                self.skipTest("GraphMuse dependency not available")
            else:
                raise


if __name__ == '__main__':
    unittest.main(verbosity=2)
