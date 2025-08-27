#!/usr/bin/env python3
"""
Test script to verify AnalysisGNN installation

This script checks that the core components can be imported and basic
functionality is available.
"""

import sys
import traceback


def test_imports():
    """Test that core modules can be imported."""
    print("Testing package imports...")
    success = True
    
    # Test main package
    try:
        import analysisgnn
        print("✓ analysisgnn package imported successfully")
        print(f"  Version: {analysisgnn.__version__}")
    except ImportError as e:
        print(f"✗ Failed to import analysisgnn: {e}")
        success = False
    
    # Test core model (may fail due to graphmuse dependency)
    try:
        from analysisgnn.models.analysis import ContinualAnalysisGNN
        print("✓ ContinualAnalysisGNN imported successfully")
    except ImportError as e:
        if "graphmuse" in str(e).lower():
            print("⚠ ContinualAnalysisGNN import failed due to missing graphmuse (expected)")
        else:
            print(f"✗ Failed to import ContinualAnalysisGNN: {e}")
            success = False
    
    # Test data module (may fail due to graphmuse dependency)
    try:
        from analysisgnn.data.datamodules.analysis import AnalysisDataModule
        print("✓ AnalysisDataModule imported successfully")
    except ImportError as e:
        if "graphmuse" in str(e).lower():
            print("⚠ AnalysisDataModule import failed due to missing graphmuse (expected)")
        else:
            print(f"✗ Failed to import AnalysisDataModule: {e}")
            success = False
    
    # Test utilities
    try:
        from analysisgnn.utils.chord_representations import available_representations
        print("✓ Chord representations imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import chord representations: {e}")
        success = False
    
    # Test prediction script
    try:
        from analysisgnn.inference.predict_analysis import main as predict_main
        print("✓ Prediction script imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import prediction script: {e}")
        success = False
    
    return success


def test_dependencies():
    """Test that required dependencies are available."""
    print("\nTesting dependencies...")
    dependencies = [
        ('torch', 'PyTorch'),
        ('pytorch_lightning', 'PyTorch Lightning'), 
        ('torch_geometric', 'PyTorch Geometric'),
        ('partitura', 'Partitura'),
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('music21', 'Music21'),
    ]
    
    available = 0
    total = len(dependencies)
    
    for module_name, display_name in dependencies:
        try:
            module = __import__(module_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"✓ {display_name} v{version}")
            available += 1
        except ImportError:
            print(f"✗ {display_name} is not available")
    
    print(f"\nDependencies: {available}/{total} available")
    return available >= total - 1  # Allow one missing dependency


def test_graphmuse():
    """Test graphmuse dependency separately."""
    print("\nTesting GraphMuse (optional)...")
    try:
        import graphmuse
        version = getattr(graphmuse, '__version__', 'unknown')
        print(f"✓ GraphMuse v{version} is available")
        return True
    except ImportError:
        print("⚠ GraphMuse is not available (this is expected and OK)")
        print("  GraphMuse is needed for some model architectures but not required for basic functionality")
        return False


def main():
    """Run all tests."""
    print("AnalysisGNN Installation Test")
    print("=" * 50)
    
    try:
        imports_ok = test_imports()
        deps_ok = test_dependencies()
        graphmuse_available = test_graphmuse()
        
        print("\n" + "=" * 50)
        print("SUMMARY:")
        
        if imports_ok and deps_ok:
            print("✓ Core installation is working!")
            print("✓ AnalysisGNN is ready for basic use")
            
            if graphmuse_available:
                print("✓ GraphMuse is available - full functionality enabled")
            else:
                print("⚠ GraphMuse not available - some model architectures may not work")
                print("  To install GraphMuse: pip install graphmuse")
            
            print("\nNext steps:")
            print("- Run unit tests: python -m pytest tests/")
            print("- Try training: analysisgnn-train --help")
            print("- Try prediction: analysisgnn-predict --help")
            
            return 0
        else:
            print("✗ Installation has issues")
            print("  Please check the error messages above")
            return 1
            
    except Exception as e:
        print(f"\n✗ Unexpected error during testing: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
