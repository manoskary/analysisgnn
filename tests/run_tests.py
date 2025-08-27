#!/usr/bin/env python3
"""
Test runner for AnalysisGNN tests

This script runs all tests in the tests directory and provides a summary.
"""

import unittest
import sys
import os

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_all_tests():
    """Discover and run all tests."""
    # Discover tests in the current directory
    loader = unittest.TestLoader()
    start_dir = os.path.dirname(os.path.abspath(__file__))
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return success/failure
    return result.wasSuccessful()


def main():
    """Main test runner."""
    print("Running AnalysisGNN Test Suite")
    print("=" * 50)
    
    success = run_all_tests()
    
    if success:
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed!")
        return 1


if __name__ == '__main__':
    exit(main())
