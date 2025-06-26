import pytest
import os
from os.path import join
import torch
import pickle
import numpy as np
from data_loader.data_loaders import MLforecastDataset, MLforecastDataLoader

def test_dataset_initialization(test_data_dir, sample_pollution_data, sample_weather_data):
    """Test that the dataset can be initialized with sample data."""
    # Create a temporary training folder
    training_folder = join(test_data_dir, 'training')
    os.makedirs(training_folder, exist_ok=True)
    
    # Create a temporary normalization params file
    norm_params_file = join(training_folder, 'norm_params_2010_to_2010.pkl')
    
    # Initialize dataset
    dataset = MLforecastDataset(
        pollution_folder=os.path.dirname(sample_pollution_data),
        weather_folder=os.path.dirname(sample_weather_data),
        norm_params_file=norm_params_file,
        training_folder=training_folder,
        years=[2010],
        pollutants_to_keep=['co', 'nodos', 'otres'],
        prev_pollutant_hours=24,
        prev_weather_hours=2,
        next_weather_hours=1,
        auto_regresive_steps=1
    )
    
    # Test that dataset was initialized correctly
    assert dataset is not None
    assert len(dataset) > 0
    assert isinstance(dataset.pollution_data, pd.DataFrame)
    assert isinstance(dataset.weather_data, np.ndarray)

def test_dataset_getitem(test_data_dir, sample_pollution_data, sample_weather_data):
    """Test that we can get items from the dataset."""
    # Create a temporary training folder
    training_folder = join(test_data_dir, 'training')
    os.makedirs(training_folder, exist_ok=True)
    
    # Create a temporary normalization params file
    norm_params_file = join(training_folder, 'norm_params_2010_to_2010.pkl')
    
    # Initialize dataset
    dataset = MLforecastDataset(
        pollution_folder=os.path.dirname(sample_pollution_data),
        weather_folder=os.path.dirname(sample_weather_data),
        norm_params_file=norm_params_file,
        training_folder=training_folder,
        years=[2010],
        pollutants_to_keep=['co', 'nodos', 'otres'],
        prev_pollutant_hours=24,
        prev_weather_hours=2,
        next_weather_hours=1,
        auto_regresive_steps=1
    )
    
    # Get an item
    x, y = dataset[24]  # Get item after enough previous hours
    
    # Test x components
    assert isinstance(x, list)
    assert len(x) == 2
    assert isinstance(x[0], torch.Tensor)  # pollution data
    assert isinstance(x[1], torch.Tensor)  # weather data
    
    # Test y components
    assert isinstance(y, list)
    assert len(y) == 2
    assert isinstance(y[0], torch.Tensor)  # pollution predictions
    assert isinstance(y[1], torch.Tensor)  # imputed pollution predictions

def test_dataloader(test_data_dir, sample_pollution_data, sample_weather_data):
    """Test that the dataloader works correctly."""
    # Create a temporary training folder
    training_folder = join(test_data_dir, 'training')
    os.makedirs(training_folder, exist_ok=True)
    
    # Initialize dataloader
    dataloader = MLforecastDataLoader(
        pollution_folder=os.path.dirname(sample_pollution_data),
        weather_folder=os.path.dirname(sample_weather_data),
        training_folder=training_folder,
        years=[2010],
        pollutants_to_keep=['co', 'nodos', 'otres'],
        prev_pollutant_hours=24,
        prev_weather_hours=2,
        next_weather_hours=1,
        auto_regresive_steps=1,
        batch_size=4,
        shuffle=True,
        validation_split=0.2,
        num_workers=0  # Use 0 workers for testing
    )
    
    # Test that dataloader was initialized correctly
    assert dataloader is not None
    assert len(dataloader) > 0
    
    # Test getting a batch
    for batch_idx, (x, y) in enumerate(dataloader):
        # Check batch components
        assert isinstance(x, list)
        assert len(x) == 2
        assert isinstance(x[0], torch.Tensor)  # pollution data
        assert isinstance(x[1], torch.Tensor)  # weather data
        
        assert isinstance(y, list)
        assert len(y) == 2
        assert isinstance(y[0], torch.Tensor)  # pollution predictions
        assert isinstance(y[1], torch.Tensor)  # imputed pollution predictions
        
        # Check batch size
        assert x[0].size(0) == 4  # batch size
        assert x[1].size(0) == 4  # batch size
        assert y[0].size(0) == 4  # batch size
        assert y[1].size(0) == 4  # batch size
        
        break  # Only test first batch 