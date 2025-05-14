from typing import List, Optional, Callable
import torch
from torchvision import datasets, transforms
from base import BaseDataLoader
from preproc_data import preproc_pollution, preproc_weather, intersect_dates, visualize_pollutant_vs_weather_var
import xarray as xr
import os
from os.path import join
import pandas as pd
import re
import numpy as np
from torch.utils.data import Dataset
from pandas import DataFrame
from xarray import Dataset as XDataset
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

class MLforecastDataset(Dataset):
    """
    Custom dataset for ML forecasting that loads and preprocesses pollution and weather data.
    
    Args:
        pollution_folder (str): Path to folder containing pollution CSV files
        weather_folder (str): Path to folder containing weather netCDF files
        years (list): List of years to process
        prev_pol_hours (int, optional): Number of previous hours of pollution data to use. Defaults to 24.
        prev_weather_hours (int, optional): Number of previous hours of weather data to use. Defaults to 2.
        next_weather_hours (int, optional): Number of future hours of weather data to predict. Defaults to 1.
        transform (callable, optional): Optional transform to be applied to samples
        
    Attributes:
        pollution_data (pandas.DataFrame): Preprocessed pollution data
        weather_data (xarray.Dataset): Preprocessed weather data
        total_dates (int): Number of timesteps in the dataset
        features (numpy.ndarray): Weather data converted to numpy array
    """
    def __init__(self, 
                 pollution_folder: str, 
                 weather_folder: str, 
                 years: List[int], 
                 prev_pol_hours: int = 24, 
                 prev_weather_hours: int = 2,
                 next_weather_hours: int = 1, 
                 transform: Optional[Callable] = None) -> None:
        """Initialize the dataset."""

        self.prev_pol_hours = prev_pol_hours
        self.prev_weather_hours = prev_weather_hours
        self.next_weather_hours = next_weather_hours

        self.pollution_data: DataFrame = preproc_pollution(pollution_folder, years)
        self.weather_data: XDataset = preproc_weather(weather_folder, years)
        
        visualize_pollutant_vs_weather_var(self.pollution_data, self.weather_data, output_file=join(pollution_folder, "imgs", "pollution_vs_weather.png"),
                                          pollutant_col='cont_otres_MER', weather_var='T2')

        self.weather_data, self.pollution_data = intersect_dates(
            self.pollution_data, self.weather_data)

        # Save both arrays as pickle files together
        
        self.total_dates: int = len(self.pollution_data)
        self.features: np.ndarray = self.weather_data.to_array().values
        self.transform = transform


        print("Done initializing dataset!")
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self.total_dates
        
    def __getitem__(self, idx: int) -> int:
        """Get a single sample from the dataset."""

        return 0


class MLforecastDataLoader(BaseDataLoader):
    """
    DataLoader for the MLforecastDataset that handles batching and shuffling.
    
    Args:
        pollution_folder (str): Path to folder containing pollution CSV files
        weather_folder (str): Path to folder containing weather netCDF files
        years (list): List of years to process
        batch_size (int): Number of samples per batch
        shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
        validation_split (float, optional): Fraction of data to use for validation. Defaults to 0.0.
        num_workers (int, optional): Number of worker processes for data loading. Defaults to 1.
    """
    def __init__(self, 
                 pollution_folder: str, 
                 weather_folder: str, 
                 years: List[int], 
                 batch_size: int, 
                 shuffle: bool = True,
                 validation_split: float = 0.0, 
                 num_workers: int = 1) -> None:
        """Initialize the data loader."""
        self.dataset = MLforecastDataset(pollution_folder, weather_folder, years)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

    
if __name__ == '__main__':
    # Test parameters
    root_folder = "/home/olmozavala/DATA/AirPollution"
    pollution_folder = join(root_folder, "PollutionCSV")
    weather_folder = join(root_folder, "WRF_NetCDF")
    years = [2010]
    batch_size = 32
    
    # Create dataset instance
    dataset = MLforecastDataset(pollution_folder, weather_folder, years)
    print(f"Dataset size: {len(dataset)}")
    
    # Create dataloader
    # dataloader = MLforecastDataLoader(
    #     pollution_folder=pollution_folder,
    #     weather_folder=weather_folder,
    #     years=years,
    #     batch_size=batch_size,
    #     shuffle=True
    # )
    
    # # Test loading a batch
    # for batch_idx, batch in enumerate(dataloader):
    #     print(f"Batch {batch_idx} shape: {batch.shape}")
    #     break
