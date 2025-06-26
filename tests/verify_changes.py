#!/usr/bin/env python3
"""
Verification script to test the models path extraction functionality.
"""

import json
import tempfile
from pathlib import Path

# Import the ParallelTester class
import importlib.util
spec = importlib.util.spec_from_file_location("parallel_testing", "5b_parallel_testing.py")
parallel_testing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parallel_testing)
ParallelTester = parallel_testing.ParallelTester


def test_models_path_extraction():
    """Test that models path is correctly extracted from config files."""
    print("Testing models path extraction...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a test config file
        config_path = Path(temp_dir) / "test_config.json"
        expected_models_path = "/test/models/path"
        
        config_data = {
            "name": "TestConfig",
            "test": {
                "all_models_path": expected_models_path,
                "model_path": "/test/model/path",
                "visualize_batch": False
            }
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_data, f)
        
        # Mock the _load_available_configs method
        original_load = ParallelTester._load_available_configs
        
        def mock_load_available_configs(self):
            return {"test_config": str(config_path)}
        
        # Replace the method temporarily
        ParallelTester._load_available_configs = mock_load_available_configs
        
        try:
            # Test the ParallelTester
            tester = ParallelTester(configs_path=temp_dir)
            
            # Verify that the models path was extracted correctly
            actual_path = str(tester.models_path)
            print(f"Expected models path: {expected_models_path}")
            print(f"Actual models path: {actual_path}")
            
            if actual_path == expected_models_path:
                print("✅ SUCCESS: Models path extracted correctly!")
                return True
            else:
                print("❌ FAILED: Models path extraction failed!")
                return False
                
        finally:
            # Restore the original method
            ParallelTester._load_available_configs = original_load


def test_no_configs_behavior():
    """Test behavior when no config files are available."""
    print("\nTesting behavior with no config files...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Mock the _load_available_configs method to return empty dict
        original_load = ParallelTester._load_available_configs
        
        def mock_load_available_configs(self):
            return {}
        
        # Replace the method temporarily
        ParallelTester._load_available_configs = mock_load_available_configs
        
        try:
            # Should raise ValueError when no configs are available
            try:
                tester = ParallelTester(configs_path=temp_dir)
                print("❌ FAILED: Should have raised ValueError for no configs!")
                return False
            except ValueError as e:
                if "No configuration files found" in str(e):
                    print("✅ SUCCESS: Correctly raised ValueError for no configs!")
                    return True
                else:
                    print(f"❌ FAILED: Unexpected error: {e}")
                    return False
                    
        finally:
            # Restore the original method
            ParallelTester._load_available_configs = original_load


def test_missing_field_behavior():
    """Test behavior when config files don't have the all_models_path field."""
    print("\nTesting behavior with missing all_models_path field...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a test config file without all_models_path
        config_path = Path(temp_dir) / "test_config.json"
        config_data = {
            "name": "TestConfig",
            "test": {
                "model_path": "/test/model/path"
                # Missing all_models_path
            }
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_data, f)
        
        # Mock the _load_available_configs method
        original_load = ParallelTester._load_available_configs
        
        def mock_load_available_configs(self):
            return {"test_config": str(config_path)}
        
        # Replace the method temporarily
        ParallelTester._load_available_configs = mock_load_available_configs
        
        try:
            # Should use default path when all_models_path is missing
            tester = ParallelTester(configs_path=temp_dir)
            expected_default = "/home/olmozavala/DATA/AirPollution/OUTPUT/models"
            actual_path = str(tester.models_path)
            
            print(f"Expected default path: {expected_default}")
            print(f"Actual path: {actual_path}")
            
            if actual_path == expected_default:
                print("✅ SUCCESS: Correctly used default path!")
                return True
            else:
                print("❌ FAILED: Did not use default path!")
                return False
                
        finally:
            # Restore the original method
            ParallelTester._load_available_configs = original_load


if __name__ == "__main__":
    print("=" * 60)
    print("VERIFICATION OF MODELS PATH EXTRACTION FUNCTIONALITY")
    print("=" * 60)
    
    # Run all tests
    test1_passed = test_models_path_extraction()
    test2_passed = test_no_configs_behavior()
    test3_passed = test_missing_field_behavior()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Test 1 (Extract models path): {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"Test 2 (No configs): {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    print(f"Test 3 (Missing field): {'✅ PASSED' if test3_passed else '❌ FAILED'}")
    
    if all([test1_passed, test2_passed, test3_passed]):
        print("\n🎉 ALL TESTS PASSED! The models path extraction is working correctly.")
    else:
        print("\n⚠️  SOME TESTS FAILED! Please check the implementation.") 