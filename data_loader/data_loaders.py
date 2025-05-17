from typing import List, Optional, Callable
from proj_preproc.normalization import normalize_data, denormalize_data, create_normalization_data
from proj_preproc.viz import visualize_pollutant_vs_weather_var
import os
from os.path import join
from torch.utils.data import Dataset
from pandas import DataFrame
from xarray import Dataset as XDataset
import numpy as np
import pickle
import pandas as pd
from base import BaseDataLoader
from data_loader.preproc_data import preproc_pollution, preproc_weather, intersect_dates
import torch
import time

class MLforecastDataset(Dataset):
    """
    Custom dataset for ML forecasting that loads and preprocesses pollution and weather data.
    
    Args:
        pollution_folder (str): Path to folder containing pollution CSV files
        weather_folder (str): Path to folder containing weather netCDF files
        years (list): List of years to process
        prev_pollutant_hours (int, optional): Number of previous hours of pollution data to use. Defaults to 24.
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
                 norm_params_file: str,
                 training_folder: str,
                 years: List[int], 
                 pollutants_to_keep: List[str],
                 prev_pollutant_hours: int = 24, 
                 prev_weather_hours: int = 2,
                 next_weather_hours: int = 1, 
                 auto_regresive_steps: int = 1,
                 transform: Optional[Callable] = None) -> None:
        """Initialize the dataset."""

        # Save the data to pickle files with year range in filename
        start_year = min(years)
        end_year = max(years)

        pollution_data_file = join(training_folder, f'pollution_data_{start_year}_to_{end_year}.pkl')
        weather_data_file = join(training_folder, f'weather_data_{start_year}_to_{end_year}.pkl')

        self.prev_pollutant_hours = prev_pollutant_hours
        self.prev_weather_hours = prev_weather_hours
        self.next_weather_hours = next_weather_hours
        self.auto_regresive_steps = auto_regresive_steps

        self.data = {}
        if not os.path.exists(pollution_data_file) or not os.path.exists(weather_data_file):
            print("Preprocessing data and saving to pickle files")
            pollution_data: DataFrame = preproc_pollution(pollution_folder, years, pollutants_to_keep)
            weather_data: XDataset = preproc_weather(weather_folder, years)
            
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
            
            print(f"Weather array final shape: {weather_array.shape}")

            with open(join(training_folder, f'pollution_data_{start_year}_to_{end_year}.pkl'), 'wb') as f:
                pickle.dump(self.pollution_data, f)
            with open(join(training_folder, f'weather_data_{start_year}_to_{end_year}.pkl'), 'wb') as f:
                pickle.dump(self.weather_data, f)

            # ================== Only for visualization ==================
            self.pollutant_imputed_columns = [col for col in self.pollution_data.columns if col.startswith('i_cont_')]

            # Save the imputed columns to a separate dataframe
            self.pollutant_imputed_data = self.pollution_data[self.pollutant_imputed_columns]

            viz_stations = ['PED', 'MER', 'UIZ']
            viz_pollutants = ['otres', 'co']
            viz_weather_vars = list(weather_data_xarray.data_vars)
            # Visualize the data
            for pollutant in viz_pollutants:
                for station in viz_stations:
                    for weather_var in viz_weather_vars:
                        visualize_pollutant_vs_weather_var(self.pollution_data, 
                                                            weather_data_xarray, 
                                                            self.pollutant_imputed_data,
                                                            join(training_folder, 'imgs', f'{station} and {pollutant} vs {weather_var}.png'),
                                                            pollutant_col=f'cont_{pollutant}_{station}', weather_var=weather_var, hours_to_plot=range(48))

        else:
            print("Loading data from pickle files")
            self.pollution_data = pd.read_pickle(pollution_data_file)
            self.weather_data = pd.read_pickle(weather_data_file)

        # Select all the columns that start with 'i_cont_' (imputed continuous pollutants)
        self.pollutant_imputed_columns = [col for col in self.pollution_data.columns if col.startswith('i_cont_')]
        self.pollutant_only_columns = [col for col in self.pollution_data.columns if col.startswith('cont_')]

        # Save the imputed columns to a separate dataframe
        self.pollutant_imputed_data = self.pollution_data[self.pollutant_imputed_columns]
        # Remove the imputed columns from the pollution data
        self.pollution_data = self.pollution_data.drop(columns=self.pollutant_imputed_columns)
            
        self.total_dates: int = len(self.pollution_data)
        self.dates = self.pollution_data.index
        self.transform = transform
        print("Done initializing dataset!")
    
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

        # Get the pollution data for the previous hours up to current index
        x_pollution = self.pollution_data.iloc[idx-self.prev_pollutant_hours:idx]
        x_weather = self.weather_data[idx-self.prev_weather_hours:idx + self.next_weather_hours + self.auto_regresive_steps]
        y_imputed_pollutant_data = self.pollutant_imputed_data.iloc[idx:idx + self.auto_regresive_steps]
        y_pollutant_data = self.pollution_data.iloc[idx:idx + self.auto_regresive_steps]
        y_pollutant_data = y_pollutant_data[self.pollutant_only_columns]
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

        return x, y


class MLforecastDataLoader(BaseDataLoader):
    """
    DataLoader for the MLforecastDataset that handles batching and shuffling.
    
    Args:
        pollution_folder (str): Path to folder containing pollution CSV files
        weather_folder (str): Path to folder containing weather netCDF files
        training_folder (str): Path to folder for saving/loading processed data
        years (list): List of years to process
        pollutants_to_keep (list): List of pollutants to keep
        prev_pollutant_hours (int): Number of previous pollution hours to use
        prev_weather_hours (int): Number of previous weather hours to use
        next_weather_hours (int): Number of future weather hours to predict
        auto_regresive_steps (int): Number of steps for auto-regression
        batch_size (int): Number of samples per batch
        shuffle (bool): Whether to shuffle the data
        validation_split (float): Fraction of data to use for validation
        num_workers (int): Number of worker processes for data loading
    """
    def __init__(self, 
                pollution_folder: str,
                weather_folder: str,
                training_folder: str,
                years: List[int],
                pollutants_to_keep: List[str],
                prev_pollutant_hours: int = 24,
                prev_weather_hours: int = 2,
                next_weather_hours: int = 1,
                auto_regresive_steps: int = 1,
                batch_size: int = 32,
                shuffle: bool = True,
                validation_split: float = 0.0,
                num_workers: int = 1) -> None:
        """Initialize the data loader."""
        
        # Create norm_params_file path
        start_year = min(years)
        end_year = max(years)
        norm_params_file = join(training_folder, f"norm_params_{start_year}_to_{end_year}.pkl")

        # Create the dataset
        self.dataset = MLforecastDataset(
            pollution_folder=pollution_folder,
            weather_folder=weather_folder,
            norm_params_file=norm_params_file,
            training_folder=training_folder,
            years=years,
            pollutants_to_keep=pollutants_to_keep,
            prev_pollutant_hours=prev_pollutant_hours,
            prev_weather_hours=prev_weather_hours,
            next_weather_hours=next_weather_hours,
            auto_regresive_steps=auto_regresive_steps
        )

        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

    
if __name__ == '__main__':
    # Test parameters
    root_folder = "/home/olmozavala/DATA/AirPollution"
    pollution_folder = join(root_folder, "PollutionCSV")
    weather_folder = join(root_folder, "WRF_NetCDF")
    training_folder = join(root_folder, "TrainingData")
    years = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]

    start_year = min(years)
    end_year = max(years)

    norm_params_file = join(training_folder, f"norm_params_{start_year}_to_{end_year}.pkl")
    pollutants_to_keep = ['co', 'nodos', 'otres', 'pmdiez', 'pmdoscinco']
    # Data loader parameters
    batch_size = 8
    prev_pollutant_hours = 24
    prev_weather_hours = 1
    next_weather_hours = 1
    auto_regresive_steps = 2

    # Print the paramters
    print(f"Batch size: {batch_size}")
    print(f"Prev pollutant hours: {prev_pollutant_hours}")
    print(f"Prev weather hours: {prev_weather_hours}")
    print(f"Next weather hours: {next_weather_hours}")
    print(f"Auto regresive steps: {auto_regresive_steps}")
    
    dataset = MLforecastDataset(
        pollution_folder, weather_folder, norm_params_file, training_folder, years, pollutants_to_keep,
        prev_pollutant_hours, prev_weather_hours, next_weather_hours, auto_regresive_steps)
       
    # Create dataloader
    dataloader = MLforecastDataLoader(
        pollution_folder=pollution_folder,
        weather_folder=weather_folder,
        training_folder=training_folder,
        years=years,
        pollutants_to_keep=pollutants_to_keep,
        prev_pollutant_hours=prev_pollutant_hours,
        prev_weather_hours=prev_weather_hours,
        next_weather_hours=next_weather_hours,
        auto_regresive_steps=auto_regresive_steps,
        batch_size=batch_size,
        shuffle=True
    )
    
    # Test loading a batch
    for batch_idx, batch in enumerate(dataloader):
        # Print the shape of each element in the batch (x, y)
        print(f"Batch {batch_idx}")
        print(f"  x pollution shape: {batch[0][0].shape} (batch, prev_pollutant_hours, stations*contaminants)")
        print(f"  x weather shape: {batch[0][1].shape} (batch, prev_weather_hours + next_weather_hours + auto_regresive_steps + 1, fields, lat, lon)")
        print(f"  y pollution shape: {batch[1][0].shape} (batch, auto_regresive_steps, stations*contaminants)")
        print(f"  y imputed pollution shape: {batch[1][1].shape} (batch, auto_regresive_steps, stations*contaminants)")
        break
