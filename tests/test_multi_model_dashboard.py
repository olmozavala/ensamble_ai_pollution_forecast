#!/usr/bin/env python3
"""
Test script for the multi-model comparison dashboard.
This script tests the key functions without requiring actual data files.
"""

import pandas as pd
import numpy as np
from typing import Dict
import sys
import os

# Add the current directory to the path so we can import from the main script
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import functions from the main dashboard script
from dashb6brd_singlemoallmodels import calculate_rmse_by_hour, create_rmse_comparison_plot, create_model_summary_table, create_model_grouping_analysis

def create_test_data() -> Dict[str, pd.DataFrame]:
    """
    Create synthetic test data for multiple models with two-level folder structure.
    
    Returns:
        Dictionary with test model data using combined folder names
    """
    # Create synthetic data for testing
    np.random.seed(42)
    
    # Generate timestamps
    timestamps = pd.date_range('2023-01-01', periods=100, freq='H')
    
    models_data = {}
    
    # Model 1: Baseline model with two-level structure
    model1_data = []
    for hour in range(1, 25):  # 24 hours
        for timestamp in timestamps:
            # Create synthetic target and prediction data
            target_values = np.random.normal(50, 15, 10)  # 10 stations
            # Add some systematic error that increases with hour
            prediction_error = np.random.normal(0, 5 + hour * 0.5, 10)
            pred_values = target_values + prediction_error
            
            # Create DataFrame for this hour
            hour_data = {'timestamp': timestamp, 'predicted_hour': hour}
            
            # Add target and prediction columns for each station
            for i in range(10):
                hour_data[f'target_cont_otres_station_{i}'] = target_values[i]
                hour_data[f'pred_cont_otres_station_{i}'] = pred_values[i]
            
            model1_data.append(hour_data)
    
    models_data['BaseModel_10340'] = pd.DataFrame(model1_data)
    
    # Model 2: Improved model (lower RMSE)
    model2_data = []
    for hour in range(1, 25):
        for timestamp in timestamps:
            target_values = np.random.normal(50, 15, 10)
            # Lower error for improved model
            prediction_error = np.random.normal(0, 3 + hour * 0.3, 10)
            pred_values = target_values + prediction_error
            
            hour_data = {'timestamp': timestamp, 'predicted_hour': hour}
            for i in range(10):
                hour_data[f'target_cont_otres_station_{i}'] = target_values[i]
                hour_data[f'pred_cont_otres_station_{i}'] = pred_values[i]
            
            model2_data.append(hour_data)
    
    models_data['ImprovedModel_20450'] = pd.DataFrame(model2_data)
    
    # Model 3: Advanced model (even lower RMSE)
    model3_data = []
    for hour in range(1, 25):
        for timestamp in timestamps:
            target_values = np.random.normal(50, 15, 10)
            # Even lower error for advanced model
            prediction_error = np.random.normal(0, 2 + hour * 0.2, 10)
            pred_values = target_values + prediction_error
            
            hour_data = {'timestamp': timestamp, 'predicted_hour': hour}
            for i in range(10):
                hour_data[f'target_cont_otres_station_{i}'] = target_values[i]
                hour_data[f'pred_cont_otres_station_{i}'] = pred_values[i]
            
            model3_data.append(hour_data)
    
    models_data['AdvancedModel_30560'] = pd.DataFrame(model3_data)
    
    # Model 4: Another BaseModel variant
    model4_data = []
    for hour in range(1, 25):
        for timestamp in timestamps:
            target_values = np.random.normal(50, 15, 10)
            # Similar to base model but slightly different
            prediction_error = np.random.normal(0, 4.5 + hour * 0.4, 10)
            pred_values = target_values + prediction_error
            
            hour_data = {'timestamp': timestamp, 'predicted_hour': hour}
            for i in range(10):
                hour_data[f'target_cont_otres_station_{i}'] = target_values[i]
                hour_data[f'pred_cont_otres_station_{i}'] = pred_values[i]
            
            model4_data.append(hour_data)
    
    models_data['BaseModel_10341'] = pd.DataFrame(model4_data)
    
    return models_data

def test_rmse_calculation():
    """Test RMSE calculation function."""
    print("Testing RMSE calculation...")
    
    # Create test data
    test_data = create_test_data()
    
    for model_name, data in test_data.items():
        print(f"\nTesting {model_name}:")
        rmse_by_hour = calculate_rmse_by_hour(data, 'otres')
        
        print(f"  Hours with RMSE data: {len(rmse_by_hour)}")
        print(f"  RMSE range: {min(rmse_by_hour.values()):.3f} - {max(rmse_by_hour.values()):.3f}")
        print(f"  Mean RMSE: {np.mean(list(rmse_by_hour.values())):.3f}")
        
        # Verify we have RMSE for all 24 hours
        assert len(rmse_by_hour) == 24, f"Expected 24 hours, got {len(rmse_by_hour)}"
        assert all(not np.isnan(val) for val in rmse_by_hour.values()), "Found NaN values in RMSE"
    
    print("✓ RMSE calculation test passed!")

def test_plot_creation():
    """Test plot creation functions."""
    print("\nTesting plot creation...")
    
    test_data = create_test_data()
    
    # Test RMSE comparison plot
    try:
        rmse_fig = create_rmse_comparison_plot(test_data, 'otres')
        print("✓ RMSE comparison plot created successfully")
    except Exception as e:
        print(f"✗ Error creating RMSE comparison plot: {e}")
        return False
    
    # Test summary table
    try:
        summary_fig = create_model_summary_table(test_data, 'otres')
        print("✓ Summary table created successfully")
    except Exception as e:
        print(f"✗ Error creating summary table: {e}")
        return False
    
    # Test model grouping analysis
    try:
        grouping_fig = create_model_grouping_analysis(test_data, 'otres')
        print("✓ Model grouping analysis created successfully")
    except Exception as e:
        print(f"✗ Error creating model grouping analysis: {e}")
        return False
    
    print("✓ Plot creation tests passed!")
    return True

def test_data_loading_simulation():
    """Test the data loading logic with simulated file structure."""
    print("\nTesting data loading simulation...")
    
    # Simulate the expected file structure
    test_data = create_test_data()
    
    # Verify the data structure matches expectations
    for model_name, data in test_data.items():
        # Check required columns
        required_cols = ['timestamp', 'predicted_hour']
        for col in required_cols:
            assert col in data.columns, f"Missing required column: {col}"
        
        # Check pollutant columns
        target_cols = [col for col in data.columns if col.startswith('target_cont_otres_')]
        pred_cols = [col for col in data.columns if col.startswith('pred_cont_otres_')]
        
        assert len(target_cols) > 0, f"No target columns found for {model_name}"
        assert len(pred_cols) > 0, f"No prediction columns found for {model_name}"
        assert len(target_cols) == len(pred_cols), f"Mismatch in target/prediction columns for {model_name}"
        
        # Check that model name follows the expected pattern (folder1_folder2)
        assert '_' in model_name, f"Model name {model_name} should contain underscore"
        parts = model_name.split('_')
        assert len(parts) >= 2, f"Model name {model_name} should have at least two parts"
        
        print(f"✓ {model_name}: {len(target_cols)} station columns, {len(data)} records")
    
    print("✓ Data loading simulation test passed!")

def test_model_grouping():
    """Test the model grouping functionality."""
    print("\nTesting model grouping...")
    
    test_data = create_test_data()
    
    # Test that models are properly grouped by first level folder
    model_groups = {}
    for model_name in test_data.keys():
        group_name = model_name.split('_')[0]  # First part before underscore
        if group_name not in model_groups:
            model_groups[group_name] = []
        model_groups[group_name].append(model_name)
    
    # Verify we have the expected groups
    expected_groups = {'BaseModel', 'ImprovedModel', 'AdvancedModel'}
    actual_groups = set(model_groups.keys())
    
    assert actual_groups == expected_groups, f"Expected groups {expected_groups}, got {actual_groups}"
    
    # Verify BaseModel has 2 variants
    assert len(model_groups['BaseModel']) == 2, f"Expected 2 BaseModel variants, got {len(model_groups['BaseModel'])}"
    
    print(f"✓ Model grouping test passed! Found groups: {actual_groups}")
    for group, models in model_groups.items():
        print(f"  {group}: {models}")

def main():
    """Run all tests."""
    print("Running multi-model dashboard tests...\n")
    
    try:
        test_rmse_calculation()
        test_plot_creation()
        test_data_loading_simulation()
        test_model_grouping()
        
        print("\n🎉 All tests passed! The dashboard should work correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 