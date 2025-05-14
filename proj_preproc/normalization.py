import os
import pickle
from typing import List, Tuple
from pandas import DataFrame
from xarray import Dataset
from data_loader.preproc_data import preproc_pollution, preproc_weather, visualize_pollutant_vs_weather_var
from os.path import join


def create_normalization_data(output_file: str, pollution_data: DataFrame, weather_data: Dataset, overwrite: bool = False) -> int:
    """
    Create normalization data for pollution and weather data.
    """

    pollutants = ['co', 'no', 'nodos', 'nox', 'otres', 'pmco', 'pmdoscinco', 'pmdiez', 'sodos']
    weather_vars = ['T2', 'U10', 'V10', 'SWDOWN', 'GLW','RH', 'RAIN', 'WS10']

    norm_params = {}
    # Check if the file exists
    if os.path.exists(output_file) and not overwrite:
        print(f"File {output_file} already exists. Exiting...")
        return 0

    for pollutant in pollutants:
        print(f"Processing {pollutant}...")
        
        # Get the data for the pollutant (filter all that contain the pollutant)
        pol_columns = [col for col in pollution_data.columns if col.startswith(f'cont_{pollutant}')]
        print(pol_columns)

        # For each pollutant get the mean and std value
        mean = pollution_data[pol_columns].mean().mean()
        std = pollution_data[pol_columns].std().mean()

        # Save the mean and std as a pickle file
        norm_params[pollutant] = {'mean': mean, 'std': std}

    print("\nPollutant normalization parameters:")
    for pollutant, params in norm_params.items():
        print(f"\n{pollutant}:")
        print(f"  Mean: {params['mean']:.4f}")
        print(f"  Std:  {params['std']:.4f}")

    for weather_var in weather_vars:
        print(f"Processing {weather_var}...")

        # Get the data for the weather variable (filter all that contain the weather variable)
        mean = weather_data[weather_var].mean(skipna=True).values.item()
        std = weather_data[weather_var].std(skipna=True).values.item()

        # Save the mean and std as a pickle file
        norm_params[weather_var] = {'mean': mean, 'std': std}

    print("\nWeather normalization parameters:")
    for weather_var, params in norm_params.items():
        print(f"\n{weather_var}:")
        print(f"  Mean: {params['mean']:.4f}")
        print(f"  Std:  {params['std']:.4f}")

    # Save the norm_params as a pickle file
    with open(output_file, 'wb') as f:
        pickle.dump(norm_params, f)

    return 1

def normalize_data(norm_params_file: str, pollution_data: DataFrame, weather_data: Dataset) -> Tuple[DataFrame, Dataset]:
    """
    Normalize the data using the normalization parameters.
    """

    print(f"Normalizing data using {norm_params_file}...")
    # Load the normalization parameters
    with open(norm_params_file, 'rb') as f:
        norm_params = pickle.load(f)

    # Normalize the data
    for pollutant in norm_params.keys():
        print(f"Normalizing {pollutant}...")
        # Get the mean and std for the pollutant
        mean = norm_params[pollutant]['mean']
        std = norm_params[pollutant]['std']

        # Get all the columns that start with the cont_pollutant
        pol_columns = [col for col in pollution_data.columns if col.startswith(f'cont_{pollutant}')]
        print(pol_columns)

        # Normalize the data
        pollution_data[pol_columns] = (pollution_data[pol_columns] - mean) / std

    # Normalize the weather data
    for weather_var in norm_params.keys():
        print(f"Normalizing {weather_var}...")
        # Get the mean and std for the weather variable
        mean = norm_params[weather_var]['mean']
        std = norm_params[weather_var]['std']

        # Normalize the data
        weather_data[weather_var] = (weather_data[weather_var] - mean) / std
    
    return pollution_data, weather_data

if __name__ == "__main__":

    root_folder = "/home/olmozavala/DATA/AirPollution"
    pollution_folder = join(root_folder, "PollutionCSV")
    weather_folder = join(root_folder, "WRF_NetCDF")
    # years = list(range(2010, 2021))  # 2010 to 2020 inclusive
    years = [2010]

    pollution_data: DataFrame = preproc_pollution(pollution_folder, years)
    weather_data: Dataset = preproc_weather(weather_folder, years)

    # Create the normalization data
    norm_params_file = join(root_folder, 'TrainingData', "norm_params.pkl")
    create_normalization_data(norm_params_file, pollution_data, weather_data, overwrite=False)

    # Normalize the data
    pollution_data, weather_data = normalize_data(norm_params_file, pollution_data, weather_data)
 
    visualize_pollutant_vs_weather_var(pollution_data, weather_data, output_file=join(pollution_folder, "imgs", "pollution_vs_weather_normalized.png"),
                                        pollutant_col='cont_otres_MER', weather_var='T2')