from typing import List, Tuple
from proj_preproc.viz import visualize_pollutant_vs_weather_var
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from os.path import join
import xarray as xr
from xarray import Dataset
import os
import re
from pandas import DataFrame

def read_pollution_data(pollution_folder: str, years: List[int]) -> List[DataFrame]:
    """
    Read pollution data from CSV files for specified years.
    
    Args:
        pollution_folder (str): Path to folder containing pollution CSV files
        years (list): List of years to read data for
        
    Returns:
        list: List of pandas DataFrames containing pollution data for each year
    """
    pollution_data: List[DataFrame] = []
    # Read pollution data for each year
    for year in years:
        print(f"Reading pollution data for year: {year}")
        pollution_df = pd.read_csv(join(pollution_folder, f"data_imputed_{year}.csv"))
        # Set the first column as the index and convert to datetime
        pollution_df.set_index(pollution_df.columns[0], inplace=True)
        pollution_df.index = pd.to_datetime(pollution_df.index)
        pollution_data.append(pollution_df)

    return pollution_data

def preproc_pollution(pollution_folder: str, years: List[int]) -> DataFrame:
    """
    Preprocess pollution data by removing unused columns and combining years.
    
    Args:
        pollution_folder (str): Path to folder containing pollution CSV files
        years (list): List of years to process
        
    Returns:
        pandas.DataFrame: Preprocessed pollution data with:
            - Removed atmospheric columns (atmos_*)
            - Removed meteorological columns (met_*)
            - Combined data from all years
    """
    pollution_data: DataFrame = pd.concat(read_pollution_data(pollution_folder, years))
                
    # %% Preprocess pollution data
    names = pollution_data.columns
    print(f"Total rows: {len(pollution_data)}. Total columns: {len(names)}.")

    # %% Count columns by prefix
    cont_columns: List[str] = [col for col in names if col.startswith('cont_')]
    i_cont_columns: List[str] = [col for col in names if col.startswith('i_cont_')]
    atmos_columns: List[str] = [col for col in names if any(col.endswith(f'h{i}') for i in range(24))]
    met_columns: List[str] = [col for col in names if col.startswith('met_')]
    time_columns: List[str] = [col for col in names if col.endswith(('day', 'week', 'year'))]

    print(f"\nColumn counts by prefix:")
    print(f"cont_: {len(cont_columns)}")
    print(f"i_cont_: {len(i_cont_columns)}") 
    print(f"atmos_: {len(atmos_columns)}")
    print(f"met_: {len(met_columns)}")
    print(f"time_: {len(time_columns)}")

    # Calculate remaining columns (missing ones)
    total_columns = len(names)
    counted_columns = len(cont_columns) + len(i_cont_columns) + len(atmos_columns) + len(met_columns) + len(time_columns)
    missing_columns = total_columns - counted_columns

    print(f"Other columns: {missing_columns}")

    print(f"Removing unused columns....")
    # Remove all atmos_ columns
    pollution_data = pollution_data.drop(columns=atmos_columns)
    # Remove all met_columns
    pollution_data = pollution_data.drop(columns=met_columns)

    # Print total remaining columns
    print(f"Total remaining columns: {len(pollution_data.columns)}")

    print(f"First 5 dates: {pollution_data.index[:5]}")

    return pollution_data

def read_weather_data(weather_folder: str, years: List[int]) -> Dataset:
    """
    Read weather data from netCDF files for specified years.
    
    Args:
        weather_folder (str): Path to folder containing weather netCDF files
        years (list): List of years to read data for
        
    Returns:
        xarray.Dataset: Combined weather data from all matching files
    """
    weather_data: Dataset = []
    # Create regex pattern to match any of the years
    year_pattern: str = '|'.join(str(year) for year in years)
    weather_pattern: str = r'.*(' + year_pattern + r').*\.nc$'
    
    # Get all matching weather files
    weather_files: List[str] = [join(weather_folder, f) for f in os.listdir(weather_folder) 
                    if re.match(weather_pattern, f)]
    
    # Read all weather files at once using open_mfdataset
    print("Reading weather files...")
    weather_data = xr.open_mfdataset(weather_files, combine='by_coords')

    return weather_data

def preproc_weather(weather_folder: str, years: List[int]) -> Dataset:
    """
    Preprocess weather data by reading and combining files.
    
    Args:
        weather_folder (str): Path to folder containing weather netCDF files
        years (list): List of years to process
        
    Returns:
        xarray.Dataset: Preprocessed weather data
    """
    weather_data: Dataset = read_weather_data(weather_folder, years)

    return weather_data

def intersect_dates(pollution_data: DataFrame, weather_data: Dataset) -> Tuple[Dataset, DataFrame]:
    """
    Find common dates between pollution and weather datasets and filter both to those dates.
    
    Args:
        pollution_data (pandas.DataFrame): Preprocessed pollution data
        weather_data (xarray.Dataset): Preprocessed weather data
        
    Returns:
        tuple: (weather_data, pollution_data) filtered to only include common dates
    """

    #  Convert pollution index to datetime if it's not already
    if pollution_data.index.dtype == 'int64':
        # Assuming the integers are UNIX timestamps in seconds
        pollution_times = pd.to_datetime(pollution_data.index, unit='s')
    else:
        pollution_times = pollution_data.index
        
    # Convert weather timestamps to pandas datetime for comparison
    weather_times = pd.to_datetime(weather_data.time.values)

    print("Weather times example:", weather_times[:5])
    print("Pollution times example:", pollution_times[:5])
    print("Weather times dtype:", weather_times.dtype)
    print("Pollution times dtype:", pollution_times.dtype)
    
    # Find common dates
    common_dates = sorted(set(weather_times).intersection(set(pollution_times)))
    print(f"Found {len(common_dates)} common dates")
    
    # Filter both datasets to only include common dates
    weather_data = weather_data.sel(time=common_dates)
    pollution_data = pollution_data.loc[common_dates]
    
    print(f"Weather data dimensions: {weather_data.dims}")
    print(f"Pollution data shape: {pollution_data.shape}")

    return weather_data, pollution_data


if __name__ == "__main__":
    root_folder = "/home/olmozavala/DATA/AirPollution"
    pollution_folder = join(root_folder, "PollutionCSV")
    weather_folder = join(root_folder, "WRF_NetCDF")
    years = [2010]

    pollution_data: DataFrame = preproc_pollution(pollution_folder, years)
    weather_data: Dataset = preproc_weather(weather_folder, years)

    weather_data, pollution_data = intersect_dates(pollution_data, weather_data)

    visualize_pollutant_vs_weather_var(pollution_data, weather_data, output_file=join(pollution_folder, "pollution_vs_weather.png"),
                                      pollutant_col='cont_otres_MER', weather_var='T2')