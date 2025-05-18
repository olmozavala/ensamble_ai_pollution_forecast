import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pandas import DataFrame
from xarray import Dataset
import numpy as np

def visualize_pollutant_vs_weather_var(pollution_data: DataFrame, 
                                       weather_data: Dataset, 
                                       imputed_hours: DataFrame = None,
                                       output_file: str = 'pollutant_vs_weather_var.png', 
                                       pollutant_col: str = 'cont_otres_MER', 
                                       weather_var: str = 'T2',
                                       hours_to_plot: range = range(48)) -> None:
    """
    Create visualization comparing pollution (ozone) and weather (temperature) data.
    
    Args:
        pollution_data (pandas.DataFrame): Preprocessed pollution data
        weather_data (xarray.Dataset): Preprocessed weather data
        
    Saves:
        pollution_vs_weather.png: Plot showing first 48 hours of:
            - Ozone levels from AJU station
            - Average temperature across all lat/lon points
    """
    print(f"Visualizing pollution ({pollutant_col}) vs weather ({weather_var})...")
    # Validate that they are 'aligned' by plotting the first 48 hours of cont_otres_AJU from pollution with
    # average of T2M from weather
    ozone = pollution_data.iloc[hours_to_plot][pollutant_col]
    temp = weather_data[weather_var][hours_to_plot].mean(dim=['lat', 'lon'])
    current_imputed_hours = imputed_hours.iloc[hours_to_plot][f"i_{pollutant_col}"]

    # Create figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()
    
    # Plot ozone on left y-axis using the index for dates
    ax1.plot(ozone.index, ozone.values, 'grey', label=pollutant_col)
    ax1.set_ylabel(pollutant_col, color='grey')
    ax1.tick_params(axis='y', labelcolor='grey')

    # If imputed_hours is not None, plot the imputed hours as scatter plot
    if imputed_hours is not None:
        ax1.scatter(current_imputed_hours.index, current_imputed_hours.to_numpy(), color='blue', label='Imputed', s=20)
        ax1.tick_params(axis='y', labelcolor='blue')
    
    # Plot temperature on right y-axis
    ax2.plot(temp.time, temp.values, 'darkred', label=weather_var)
    ax2.set_ylabel(weather_var, color='darkred')
    ax2.tick_params(axis='y', labelcolor='darkred')
    
    plt.title(f'{pollutant_col} vs {weather_var}')
    ax1.set_xlabel('Date')
    
    # Rotate x-axis labels for better readability
    ax1.tick_params(axis='x', rotation=45)
    
    # Format x-axis date ticks
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    
    # Add legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_file)
    plt.close()

def visualize_pollutant_vs_weather_var_np(pollution_data: np.ndarray, 
                                       weather_data: np.ndarray, 
                                       target_data: np.ndarray,
                                       imputed_hours: np.ndarray = None,
                                       output_file: str = 'pollutant_vs_weather_var_np.png', 
                                       pollutant_col_idx: int = 0, 
                                       weather_var_idx: int = 0,
                                       hours_to_plot: range = range(1)) -> None:
    """
    Create visualization comparing pollution (ozone) and weather (temperature) data.
    
    Args:
        pollution_data (pandas.DataFrame): Preprocessed pollution data
        weather_data (xarray.Dataset): Preprocessed weather data
        
    Saves:
        pollution_vs_weather.png: Plot showing first 48 hours of:
            - Ozone levels from AJU station
            - Average temperature across all lat/lon points
    """
    print(f"Visualizing pollution ({pollutant_col_idx}) vs weather ({weather_var_idx})...")
    # Validate that they are 'aligned' by plotting the first 48 hours of cont_otres_AJU from pollution with
    # average of T2M from weather
    ozone = pollution_data[hours_to_plot, pollutant_col_idx]
    temp = weather_data[weather_var_idx][hours_to_plot].mean(dim=['lat', 'lon'])
    current_imputed_hours = target_data[hours_to_plot, pollutant_col_idx]

    # Create figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()
    
    # Plot ozone on left y-axis using the index for dates
    ax1.plot(ozone, 'grey', label=pollutant_col_idx)
    ax1.set_ylabel(f'{pollutant_col_idx}', color='grey')
    ax1.tick_params(axis='y', labelcolor='grey')

    # If imputed_hours is not None, plot the imputed hours as scatter plot
    if imputed_hours is not None:
        ax1.scatter(current_imputed_hours, color='blue', label='Imputed', s=20)
        ax1.tick_params(axis='y', labelcolor='blue')
    
    # Plot temperature on right y-axis
    ax2.plot(temp.time, temp.values, 'darkred', label=weather_var_idx)
    ax2.tick_params(axis='y', labelcolor='darkred')
    
    plt.title(f'{pollutant_col_idx} vs {weather_var_idx} - {hours_to_plot}')
    ax1.set_xlabel('Date')
    
    # Rotate x-axis labels for better readability
    ax1.tick_params(axis='x', rotation=45)
    
    # Format x-axis date ticks
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    
    # Add legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_file)
    plt.close()

