from typing import List, Optional, Callable, Tuple
from proj_preproc.normalization import normalize_data, denormalize_data, create_normalization_data
from proj_preproc.viz import visualize_pollutant_vs_weather_var, visualize_batch_data
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
            print(f"Weather array final shape: {self.weather_data.shape}")
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

        # ======================== Getting names and indices of the columns ========================
        # Get both the indices and column names for pollutant columns
        self.pollutant_columns = [col for col in self.pollution_data.columns if col.startswith('cont_')]
        self.pollutant_columns_idx = [self.pollution_data.columns.get_loc(col) for col in self.pollutant_columns]
        # Get the columns and names of the time related inputs
        self.time_related_columns = [col for col in self.pollution_data.columns if col.endswith(('day', 'week', 'year'))]
        self.time_related_columns_idx = [self.pollution_data.columns.get_loc(col) for col in self.time_related_columns]
        # Select the imputed columns names and indeces that start with 'i_cont_' (imputed continuous pollutants)
        self.imputed_mask_columns = [col for col in self.pollution_data.columns if col.startswith('i_cont_')]
        self.imputed_mask_columns_idx = [self.pollution_data.columns.get_loc(col) for col in self.imputed_mask_columns]

        # ======================== Dropping the imputed columns from the pollution data ========================
        # Save the imputed columns to a separate dataframe
        self.pollutant_imputed_data = self.pollution_data[self.imputed_mask_columns]
        # Remove the imputed columns from the pollution data (it keeps the pollutant column and the time related columns)
        self.x_input_data = self.pollution_data.drop(columns=self.imputed_mask_columns) # Pollutants and time related columns
        self.y_output_data = self.pollution_data[self.pollutant_columns] # Pollutants only data    
            
        self.total_dates: int = len(self.pollution_data)
        self.dates = self.pollution_data.index
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
        timestamp = self.x_input_data.iloc[idx].name.timestamp()
        rounded_timestamp = round(timestamp / 3600) * 3600  # Round to nearest hour
        current_datetime = torch.tensor(rounded_timestamp, dtype=torch.float)

        return x, y, current_datetime


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

    def get_pollution_dataframe(self) -> pd.DataFrame:
        """Get the pollution dataframe"""
        return self.dataset.pollution_data

    def get_pollution_column_names_and_indices(self, column_type: str) -> Tuple[List[str], List[int]]:
        """Get the column names and indices of the pollution dataframe"""
        return self.dataset.get_column_and_index_names(column_type)

    
if __name__ == '__main__':

    import matplotlib.pyplot as plt
    # Test parameters
    root_folder = "/home/olmozavala/DATA/AirPollution"
    pollution_folder = join(root_folder, "PollutionCSV")
    weather_folder = join(root_folder, "WRF_NetCDF")
    training_folder = join(root_folder, "TrainingData")
    years = [2015]

    start_year = min(years)
    end_year = max(years)

    norm_params_file = join(training_folder, f"norm_params_{start_year}_to_{end_year}.pkl")
    pollutants_to_keep = ['co', 'nodos', 'otres', 'pmdiez', 'pmdoscinco']
    # Data loader parameters
    batch_size = 8
    prev_pollutant_hours = 24
    prev_weather_hours = 24
    next_weather_hours = 1
    auto_regresive_steps = 10

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
    data_loader = MLforecastDataLoader(
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

    pollution_column_names, pollution_column_indices = data_loader.get_pollution_column_names_and_indices("pollutant_only")
    imputed_mask_columns, imputed_mask_columns_indices = data_loader.get_pollution_column_names_and_indices("imputed_mask")
    time_related_columns, time_related_columns_indices = data_loader.get_pollution_column_names_and_indices("time")

    # Print the time related columns
    print(f"Time related columns: {time_related_columns}")
    print(f"Time related columns indices: {time_related_columns_indices}")

    for batch_idx, batch in enumerate(data_loader):
        # Print the shape of each element in the batch (x, y)
        print(f"Batch {batch_idx}")
        print(f"  x pollution shape: {batch[0][0].shape} (batch, prev_pollutant_hours, stations*contaminants + time related columns)")
        print(f"  x weather shape: {batch[0][1].shape} (batch, prev_weather_hours + next_weather_hours + auto_regresive_steps + 1, fields, lat, lon)")
        print(f"  y pollution shape: {batch[1][0].shape} (batch, auto_regresive_steps, stations*contaminants)")
        print(f"  y imputed pollution shape: {batch[1][1].shape} (batch, auto_regresive_steps, stations*contaminants)")

        # Here we can plot the data to be sure that the data is loaded correctly
        pollution_data = batch[0][0].numpy()[0,:,:]  # Final shape is (prev_pollutant_hours, stations*contaminants + time related columns)
        weather_data = batch[0][1].numpy()[0,:,:,:,:]  # Final shape is (prev_weather_hours + next_weather_hours + auto_regresive_steps + 1, fields, lat, lon)
        target_data = batch[1][0].numpy()[0,:,:]  # Final shape is (auto_regresive_steps, stations*contaminants)
        imputed_data = batch[1][1].numpy()[0,:,:]  # Final shape is (auto_regresive_steps, stations*contaminants)
        current_datetime = pd.to_datetime(batch[2][0].item(), unit='s')

        # Plot the pollution data
        # Find all indices that contain "otres" in the name
        contaminant_name = "pmdiez"  # To plot multiple stations
        # contaminant_name = "cont_otres_MER"  # To plot a single station
        weather_var_idx = 0
        weather_var_name = "T2"
        plot_pollutant_indices = [i for i, name in enumerate(pollution_column_names) if contaminant_name in name]
        
        # Create figure to plot pollution data features
        output_folder = "/home/olmozavala/DATA/AirPollution/TrainingData/batch_imgs"
        visualize_batch_data(pollution_data, target_data, imputed_data, weather_data, 
                             plot_pollutant_indices, pollution_column_names, time_related_columns, time_related_columns_indices, weather_var_name, 
                             current_datetime, output_folder, batch_idx, prev_weather_hours, next_weather_hours, 
                             auto_regresive_steps, weather_var_idx, contaminant_name)
        if batch_idx > 5:
            break