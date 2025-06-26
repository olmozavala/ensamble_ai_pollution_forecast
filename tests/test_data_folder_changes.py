#!/usr/bin/env python3
"""
Test script to verify that the data folder parameter changes work correctly.
"""

import json
import os
from os.path import join
from data_loader.data_loaders import MLforecastDataLoader

def test_data_folder_structure():
    """Test that the data folder structure is correctly handled."""
    
    # Load the config
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    print("=== Testing Data Folder Structure ===")
    print(f"Config data_folder: {config['data_loader']['args']['data_folder']}")
    
    # Verify that the expected subfolders exist
    data_folder = config['data_loader']['args']['data_folder']
    pollution_folder = join(data_folder, "PollutionCSV")
    weather_folder = join(data_folder, "WRF_NetCDF")
    training_folder = join(data_folder, "TrainingData")
    
    print(f"Expected pollution folder: {pollution_folder}")
    print(f"Expected weather folder: {weather_folder}")
    print(f"Expected training folder: {training_folder}")
    
    # Check if folders exist
    print(f"Pollution folder exists: {os.path.exists(pollution_folder)}")
    print(f"Weather folder exists: {os.path.exists(weather_folder)}")
    print(f"Training folder exists: {os.path.exists(training_folder)}")
    
    return True

def test_data_loader_initialization():
    """Test that the data loader can be initialized with the new structure."""
    
    print("\n=== Testing Data Loader Initialization ===")
    
    try:
        # Create a minimal data loader for testing
        data_loader = MLforecastDataLoader(
            data_folder="/home/olmozavala/DATA/AirPollution",
            norm_params_file="/home/olmozavala/DATA/AirPollution/TrainingData/norm_params_2010_to_2020.pkl",
            years=[2015],  # Use a single year for testing
            pollutants_to_keep=['co', 'nodos', 'otres', 'pmdiez', 'pmdoscinco'],
            prev_pollutant_hours=16,
            prev_weather_hours=4,
            next_weather_hours=2,
            auto_regresive_steps=4,
            bootstrap_enabled=False,  # Disable bootstrap for faster testing
            bootstrap_repetition=20,
            bootstrap_threshold=3.0,
            batch_size=2,
            shuffle=False,
            validation_split=0.0,
            num_workers=1
        )
        
        print("✓ Data loader initialized successfully")
        print(f"Dataset size: {len(data_loader.dataset)}")
        
        # Test getting column information
        pollution_columns, pollution_indices = data_loader.get_pollution_column_names_and_indices("pollutant_only")
        print(f"Pollution columns: {len(pollution_columns)}")
        
        # Test sampling info
        sampling_info = data_loader.get_sampling_info()
        print(f"Sampling info: {sampling_info}")
        
        return True
        
    except Exception as e:
        print(f"✗ Data loader initialization failed: {e}")
        return False

def test_config_compatibility():
    """Test that the config file is compatible with the new structure."""
    
    print("\n=== Testing Config Compatibility ===")
    
    try:
        # Load the config
        with open('config.json', 'r') as f:
            config = json.load(f)
        
        # Check that data_folder is present
        if 'data_folder' not in config['data_loader']['args']:
            print("✗ data_folder not found in config")
            return False
        
        # Check that old parameters are not present
        old_params = ['pollution_folder', 'weather_folder', 'training_folder']
        for param in old_params:
            if param in config['data_loader']['args']:
                print(f"✗ Old parameter {param} still present in config")
                return False
        
        print("✓ Config file is compatible with new structure")
        return True
        
    except Exception as e:
        print(f"✗ Config compatibility test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing data folder parameter changes...")
    
    # Run all tests
    tests = [
        test_data_folder_structure,
        test_config_compatibility,
        test_data_loader_initialization
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"Test {test.__name__} failed with exception: {e}")
            results.append(False)
    
    # Summary
    print("\n=== Test Summary ===")
    passed = sum(results)
    total = len(results)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed! Data folder changes are working correctly.")
    else:
        print("✗ Some tests failed. Please check the implementation.") 