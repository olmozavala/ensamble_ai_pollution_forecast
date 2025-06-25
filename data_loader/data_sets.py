from typing import List, Optional, Callable, Tuple
from torch.utils.data import Dataset
from pandas import DataFrame
from xarray import Dataset as XDataset
import numpy as np
import pickle
import pandas as pd
import os
from os.path import join
import torch
import time
from data_loader.preproc_data import preproc_pollution, preproc_weather, intersect_dates, bootstrap_high_ozone_events
from proj_preproc.normalization import normalize_data, denormalize_data, create_normalization_data

class MLforecastDataset(Dataset):
    """
    Custom dataset for ML forecasting that loads and preprocesses pollution and weather data.
    
    Args:
        data_folder (str): Path to the main data folder containing subfolders for pollution, weather, and training data
        years (list): List of years to process
        prev_pollutant_hours (int, optional): Number of previous hours of pollution data to use. Defaults to 24.
        prev_weather_hours (int, optional): Number of previous hours of weather data to use. Defaults to 2.
        next_weather_hours (int, optional): Number of future hours of weather data to predict. Defaults to 1.
        auto_regresive_steps (int, optional): Number of auto-regressive steps. Defaults to 1.
        bootstrap_enabled (bool, optional): Whether to bootstrap high ozone events. Defaults to False.
        bootstrap_repetition (int, optional): Number of times to replicate high ozone events. Defaults to 20.
        bootstrap_threshold (float, optional): Threshold for high ozone events. Defaults to 2.5.
        transform (callable, optional): Optional transform to be applied to samples
        
    Attributes:
        pollution_data (pandas.DataFrame): Preprocessed pollution data
        weather_data (xarray.Dataset): Preprocessed weather data
        total_dates (int): Number of timesteps in the dataset
        features (numpy.ndarray): Weather data converted to numpy array
    """
    def __init__(self, 
                 data_folder: str, 
                 norm_params_file: str,
                 years: List[int], 
                 pollutants_to_keep: List[str],
                 prev_pollutant_hours: int = 24, 
                 prev_weather_hours: int = 2,
                 next_weather_hours: int = 1, 
                 auto_regresive_steps: int = 1,
                 bootstrap_enabled: bool = False,
                 bootstrap_repetition: int = 20,
                 bootstrap_threshold: float = 2.5,
                 transform: Optional[Callable] = None) -> None:
        """Initialize the dataset."""
        
        # Create subfolder paths
        self.pollution_folder = join(data_folder, "PollutionCSV")
        self.weather_folder = join(data_folder, "WRF_NetCDF")
        self.training_folder = join(data_folder, "TrainingData")

        # Save the data to pickle files with year range in filename
        start_year = min(years)
        end_year = max(years)

        pollution_data_file = join(self.training_folder, f'pollution_data_{start_year}_to_{end_year}.pkl')
        weather_data_file = join(self.training_folder, f'weather_data_{start_year}_to_{end_year}.pkl')

        self.prev_pollutant_hours = prev_pollutant_hours
        self.prev_weather_hours = prev_weather_hours
        self.next_weather_hours = next_weather_hours
        self.auto_regresive_steps = auto_regresive_steps
        self.pollutants_to_keep = pollutants_to_keep
        self.bootstrap_enabled = bootstrap_enabled
        self.bootstrap_repetition = bootstrap_repetition
        self.bootstrap_threshold = bootstrap_threshold

        self.data = {}
        if not os.path.exists(pollution_data_file) or not os.path.exists(weather_data_file):
            print("Preprocessing data and saving to pickle files")
            pollution_data: DataFrame = preproc_pollution(self.pollution_folder, years, pollutants_to_keep)
            weather_data: XDataset = preproc_weather(self.weather_folder, years)
            
            pollution_data, weather_data = intersect_dates( pollution_data, weather_data)


            # Normalize data
            if not os.path.exists(norm_params_file):
                print("Creating normalization parameters file")
                create_normalization_data(norm_params_file, pollution_data, weather_data)

            pollution_data, weather_data_xarray = normalize_data(norm_params_file, pollution_data, weather_data)

            self.pollution_data = pollution_data
            # Replace all the nan values with 0
            print("Replacing all the pollution nan values with 0")
            # Check if there are still nan values
            self.pollution_data.fillna(0, inplace=True)
            self.pollution_data.isna().any().any()

            # Concatenate weather data and transform to numpy array
            print("Concatenating weather data and transforming to numpy array")
            weather_vars = list(weather_data_xarray.data_vars)
            weather_arrays = [weather_data_xarray[var].values.astype(np.float32) for var in weather_vars]
            weather_array = np.array(weather_arrays)

            # Switch first and second axis
            self.weather_data = weather_array.swapaxes(1, 0)

            # Replace all the nan values with 0
            print("Replacing all the weather nan values with 0")
            self.weather_data = np.nan_to_num(self.weather_data)
            print(f"Weather array final shape: {self.weather_data.shape}")
            with open(join(self.training_folder, f'pollution_data_{start_year}_to_{end_year}.pkl'), 'wb') as f:
                pickle.dump(self.pollution_data, f)
            with open(join(self.training_folder, f'weather_data_{start_year}_to_{end_year}.pkl'), 'wb') as f:
                pickle.dump(self.weather_data, f)
        else:
            print("Loading data from pickle files")
            self.pollution_data = pd.read_pickle(pollution_data_file)
            self.weather_data = pd.read_pickle(weather_data_file)

        self.dates = self.pollution_data.index

        # ======================== Apply bootstrap if enabled ========================
        self.random_sampler_weights = np.ones(len(self.pollution_data))
        if self.bootstrap_enabled:
            print("Identifying high ozone events for bootstrapping...")
            self.bootstrap_indexes = bootstrap_high_ozone_events(self.pollution_data, self.bootstrap_threshold)
            print(f"Found {len(self.bootstrap_indexes)} high ozone events to bootstrap")
            self.random_sampler_weights[self.bootstrap_indexes] = self.bootstrap_repetition
        else:
            print("Bootstrap disabled - skipping high ozone event identification")

        # ======================== Replacing all pollutants columns with the mean, min, and max of the stations ========================(except otres) 
        self.pollutant_columns = [col for col in self.pollution_data.columns if col.startswith('cont_')]
        # Calculate mean, min, and max values for each pollutant across all stations, except otres
        pollutant_stats = {}
        otres_columns = []
        for pollutant in self.pollutants_to_keep:
            # Get all columns for this pollutant across stations
            pollutant_cols = [col for col in self.pollutant_columns if f'cont_{pollutant}_' in col]
            
            if pollutant == 'otres':
                # For otres, keep original columns
                otres_columns = pollutant_cols
            else:
                # For other pollutants, calculate mean, min, and max across stations
                pollutant_mean = self.pollution_data[pollutant_cols].mean(axis=1)
                pollutant_min = self.pollution_data[pollutant_cols].min(axis=1)
                pollutant_max = self.pollution_data[pollutant_cols].max(axis=1)
                
                mean_col_name = f'cont_{pollutant}_mean'
                min_col_name = f'cont_{pollutant}_min'
                max_col_name = f'cont_{pollutant}_max'
                
                pollutant_stats[mean_col_name] = pollutant_mean
                pollutant_stats[min_col_name] = pollutant_min
                pollutant_stats[max_col_name] = pollutant_max
            
        # Create new dataframe with mean, min, and max values
        stats_pollution_df = pd.DataFrame(pollutant_stats)

        # Create aggregated imputed columns for mean, min, max of stations
        imputed_stats = {}
        for pollutant in self.pollutants_to_keep:
            if pollutant != 'otres':
                # Get all imputed columns for this pollutant across stations
                imputed_cols = [col for col in self.pollution_data.columns if f'i_cont_{pollutant}_' in col]
                
                # If any station has 1 for imputed value, set 1 for aggregated column
                imputed_mean = (self.pollution_data[imputed_cols].mean(axis=1) > 0).astype(int)
                imputed_min = (self.pollution_data[imputed_cols].min(axis=1) > 0).astype(int) 
                imputed_max = (self.pollution_data[imputed_cols].max(axis=1) > 0).astype(int)
                
                mean_col_name = f'i_cont_{pollutant}_mean'
                min_col_name = f'i_cont_{pollutant}_min'
                max_col_name = f'i_cont_{pollutant}_max'
                
                imputed_stats[mean_col_name] = imputed_mean
                imputed_stats[min_col_name] = imputed_min
                imputed_stats[max_col_name] = imputed_max
                
        # Add aggregated imputed columns to pollution data
        imputed_stats_df = pd.DataFrame(imputed_stats)
        self.pollution_data = pd.concat([self.pollution_data, imputed_stats_df], axis=1)
        
        # Drop all pollutant columns except otres
        cols_to_drop = [col for col in self.pollutant_columns if col not in otres_columns]
        self.pollution_data = self.pollution_data.drop(columns=cols_to_drop)
        
        # Add the stats columns for non-otres pollutants
        self.pollution_data = pd.concat([self.pollution_data, stats_pollution_df], axis=1)

        # ======================== Dropping the imputed columns from the pollution data (except otres) ========================
        # Select the imputed columns names and indices that start with 'i_cont_' (imputed continuous pollutants)
        self.imputed_mask_columns = [col for col in self.pollution_data.columns if col.startswith('i_cont_') and not any(x in col for x in ['otres', 'min', 'max', 'mean'])]
        # Remove the imputed columns from the pollution data (it keeps the pollutant column and the time related columns)
        self.pollution_data = self.pollution_data.drop(columns=self.imputed_mask_columns) # Pollutants and time related columns
        # Get the updated imputed columns names
        self.imputed_mask_columns = [col for col in self.pollution_data.columns if col.startswith('i_cont_')]
        self.imputed_mask_columns_idx = [i for i, col in enumerate(self.pollution_data.columns) if col.startswith('i_cont_')]

        # ======================== Dropping the imputed columns from the pollution data ========================
        # Save the imputed columns to a separate dataframe
        self.pollutant_columns = [col for col in self.pollution_data.columns if col.startswith('cont_')]
        self.pollutant_imputed_data = self.pollution_data[self.imputed_mask_columns]
        # Remove the imputed columns from the pollution data (it keeps the pollutant column and the time related columns)
        self.x_input_data = self.pollution_data.drop(columns=self.imputed_mask_columns) # Pollutants and time related columns
        self.y_output_data = self.pollution_data[self.pollutant_columns] # Pollutants only data    

        # ======================== Getting names and indices of the columns for the input data ========================
        # Get both the indices and column names for pollutant columns
        self.pollutant_columns = [col for col in self.x_input_data.columns if col.startswith('cont_')]
        self.pollutant_columns_idx = [self.x_input_data.columns.get_loc(col) for col in self.pollutant_columns]
        # Get the columns and names of the time related inputs
        self.time_related_columns = [col for col in self.x_input_data.columns if col.endswith(('day', 'week', 'year'))]
        self.time_related_columns_idx = [self.x_input_data.columns.get_loc(col) for col in self.time_related_columns]

        # ======================== Getting names and indices of the target columns ========================
        # This has to be done after y_output_data is defined
        self.target_columns = self.y_output_data.columns
        self.target_columns_idx = [self.y_output_data.columns.get_loc(col) for col in self.target_columns]

        self.total_dates: int = len(self.pollution_data)
        self.transform = transform
        print("Done initializing dataset!")
    
    def get_column_and_index_names(self, column_type: str):
        """Get the column names and indices of the pollution dataframe"""
        if column_type == "pollutant_only":
            return self.pollutant_columns, self.pollutant_columns_idx
        elif column_type == "imputed_mask":
            return self.imputed_mask_columns, self.imputed_mask_columns_idx
        elif column_type == "time":
            return self.time_related_columns, self.time_related_columns_idx
        elif column_type == "target":
            return self.target_columns, self.target_columns_idx
        else:
            raise ValueError(f"Invalid column type: {column_type}")

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self.total_dates
        
    def __getitem__(self, idx: int) -> int:
        """Get a single sample from the dataset."""

        start_time = time.time()
        
        # Validate that the index is within the bounds of the dataset considering the previous and next weather hours
        while idx < self.prev_pollutant_hours or idx >= self.total_dates - self.next_weather_hours - self.auto_regresive_steps:
            idx = np.random.randint(0, self.total_dates)
        validation_time = time.time() - start_time

        # Define the input data. x_pollution contains the pollution data and the weather data in a two dimentional tuple
        x_pollution = self.x_input_data.iloc[idx-self.prev_pollutant_hours:idx]
        x_weather = self.weather_data[idx-self.prev_weather_hours:idx + self.next_weather_hours + self.auto_regresive_steps]

        # Define the output data. y_imputed_pollutant_data contains the imputed pollutant data and y_pollutant_data contains the pollutant data
        y_imputed_pollutant_data = self.pollutant_imputed_data.iloc[idx:idx + self.auto_regresive_steps]
        y_pollutant_data = self.y_output_data.iloc[idx:idx + self.auto_regresive_steps]
        data_loading_time = time.time() - start_time - validation_time
        
        # Convert numpy arrays to tensors
        x = [torch.FloatTensor(x_pollution.to_numpy().astype(np.float32)), 
             torch.FloatTensor(x_weather.astype(np.float32))]
        y = [torch.FloatTensor(y_pollutant_data.to_numpy().astype(np.float32)), 
             torch.FloatTensor(y_imputed_pollutant_data.to_numpy().astype(np.float32))]
        tensor_conversion_time = time.time() - start_time - validation_time - data_loading_time 

        # print(f"Timing breakdown:")
        # print(f"  Validation: {validation_time:.4f}s")
        # print(f"  Data loading: {data_loading_time:.4f}s") 
        # print(f"  Tensor conversion: {tensor_conversion_time:.4f}s")
        # print(f"  Total time: {time.time() - start_time:.4f}s")

        # x contains the pollution data and the weather data in a two dimentional tuple
        # y contains the target data and the imputed data in a two dimentional tuple
        # z contains the current datetime of the data from the pollution dataframe index
        timestamp = self.dates[idx].timestamp()
        rounded_timestamp = round(timestamp / 3600) * 3600  # Round to nearest hour
        current_datetime = torch.tensor(rounded_timestamp, dtype=torch.float)

        return x, y, current_datetime


if __name__ == "__main__":
    # Test the dataset directly
    import matplotlib.pyplot as plt
    
    # Test parameters
    data_folder = "/home/olmozavala/DATA/AirPollution"
    years = [2015]
    
    start_year = min(years)
    end_year = max(years)
    norm_params_file = join(data_folder, "TrainingData", f"norm_params_{start_year}_to_{end_year}.pkl")
    pollutants_to_keep = ['co', 'nodos', 'otres', 'pmdiez', 'pmdoscinco', 'nox', 'no', 'sodos', 'pmco']
    
    # Test dataset creation
    print("=== Testing MLforecastDataset ===")
    dataset = MLforecastDataset(
        data_folder=data_folder,
        norm_params_file=norm_params_file,
        years=years,
        pollutants_to_keep=pollutants_to_keep,
        prev_pollutant_hours=16,
        prev_weather_hours=4,
        next_weather_hours=2,
        auto_regresive_steps=4,
        bootstrap_enabled=True,
        bootstrap_repetition=20,
        bootstrap_threshold=2.5
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Bootstrap enabled: {dataset.bootstrap_enabled}")
    print(f"Bootstrap indexes: {len(dataset.bootstrap_indexes)}")
    print(f"Weight range: {dataset.random_sampler_weights.min()} - {dataset.random_sampler_weights.max()}")
    
    # Test getting a sample
    sample = dataset[0]
    print(f"Sample shapes:")
    print(f"  x_pollution: {sample[0][0].shape}")
    print(f"  x_weather: {sample[0][1].shape}")
    print(f"  y_pollution: {sample[1][0].shape}")
    print(f"  y_imputed: {sample[1][1].shape}")
    print(f"  datetime: {sample[2]}")
    
    print("Dataset test completed successfully!")

