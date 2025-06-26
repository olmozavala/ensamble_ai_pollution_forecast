#!/usr/bin/env python3
"""
Test script for the ParallelTester class to verify models path extraction.
"""

import json
import tempfile
import os
from pathlib import Path
from unittest.mock import patch
import pytest
import sys
sys.path.append('.')

# Import the class to test
from parallel_testing import ParallelTester


def create_test_config(models_path: str) -> dict:
    """Create a test configuration with the specified models path."""
    return {
        "name": "TestConfig",
        "test": {
            "all_models_path": models_path,
            "model_path": "/test/model/path",
            "visualize_batch": False
        }
    }


def test_extract_models_path_from_configs():
    """Test that models path is correctly extracted from config files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a test config file
        config_path = Path(temp_dir) / "test_config.json"
        expected_models_path = "/test/models/path"
        
        config_data = create_test_config(expected_models_path)
        with open(config_path, 'w') as f:
            json.dump(config_data, f)
        
        # Test the ParallelTester
        with patch('parallel_testing.Path') as mock_path:
            mock_path.return_value = Path(temp_dir)
            
            # Mock the _load_available_configs method to return our test config
            with patch.object(ParallelTester, '_load_available_configs') as mock_load:
                mock_load.return_value = {"test_config": str(config_path)}
                
                tester = ParallelTester(configs_path=temp_dir)
                
                # Verify that the models path was extracted correctly
                assert str(tester.models_path) == expected_models_path


def test_extract_models_path_no_configs():
    """Test behavior when no config files are available."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create an empty directory
        with patch('parallel_testing.Path') as mock_path:
            mock_path.return_value = Path(temp_dir)
            
            # Mock the _load_available_configs method to return empty dict
            with patch.object(ParallelTester, '_load_available_configs') as mock_load:
                mock_load.return_value = {}
                
                # Should raise ValueError when no configs are available
                with pytest.raises(ValueError, match="No configuration files found"):
                    ParallelTester(configs_path=temp_dir)


def test_extract_models_path_missing_field():
    """Test behavior when config files don't have the all_models_path field."""
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
        
        # Test the ParallelTester
        with patch('parallel_testing.Path') as mock_path:
            mock_path.return_value = Path(temp_dir)
            
            # Mock the _load_available_configs method
            with patch.object(ParallelTester, '_load_available_configs') as mock_load:
                mock_load.return_value = {"test_config": str(config_path)}
                
                # Should use default path when all_models_path is missing
                tester = ParallelTester(configs_path=temp_dir)
                assert str(tester.models_path) == "/home/olmozavala/DATA/AirPollution/OUTPUT/models"


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"]) 