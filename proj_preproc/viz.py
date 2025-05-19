import matplotlib.pyplot as plt
from os.path import join
import matplotlib.dates as mdates
from pandas import DataFrame
from xarray import Dataset
import numpy as np
import pandas as pd

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

def visualize_batch_data( pollution_data: np.ndarray,
                                            target_data: np.ndarray,
                                            imputed_data: np.ndarray,
                                            weather_data: np.ndarray,
                                            plot_pollutant_indices: list,
                                            pollution_column_names: list,
                                            time_related_columns: list,
                                            time_related_indices: list,
                                            weather_var_name: str,
                                            current_datetime: pd.Timestamp, 
                                            output_folder: str,
                                            batch_idx: int,
                                            prev_weather_hours: int,
                                            next_weather_hours: int,
                                            auto_regresive_steps: int,
                                            weather_var_idx: int,
                                            contaminant_name: str
                                        ) -> None:
    """
    Visualize pollution, target, imputed and weather data from a batch.

    Args:
        pollution_data (np.ndarray): Input pollution data of shape (prev_pollutant_hours, num_features)
        target_data (np.ndarray): Target pollution data of shape (auto_regresive_steps, num_features) 
        imputed_data (np.ndarray): Imputed pollution data of shape (auto_regresive_steps, num_features)
        weather_data (np.ndarray): Weather data of shape (timesteps, fields, lat, lon)
        plot_pollutant_indices (list): Indices of pollutants to plot
        pollution_column_names (list): Names of pollution data columns
        time_related_columns (list): Names of time related columns to plot
        time_related_indices (list): Indices of time related columns to plot
        weather_var_name (str): Name of weather variable to plot
        current_datetime (pd.Timestamp): Current datetime of the data
        output_folder (str): Folder to save plots
        batch_idx (int): Index of current batch
        prev_weather_hours (int): Number of previous weather hours
        next_weather_hours (int): Number of future weather hours
        auto_regresive_steps (int): Number of autoregressive prediction steps
        weather_var_idx (int): Index of weather variable to plot
        contaminant_name (str): Name of contaminant being plotted

    Creates two plots:
    1. Time series of pollution data with targets, imputed values and weather variable
    2. Spatial maps of weather data for each timestep
    """

    fig, axs = plt.subplots(figsize=(15, 8))
    axs.plot(pollution_data[:, plot_pollutant_indices])
    axs.set_title('Pollution Data - Otres')
    axs.set_xlabel('Hour') 
    axs.legend([pollution_column_names[i] for i in plot_pollutant_indices])
    # Include the target data
    for i in range(len(plot_pollutant_indices)):
        x_range = range(pollution_data.shape[0], pollution_data.shape[0] + target_data.shape[0])
        axs.plot(x_range, target_data[:, plot_pollutant_indices[i]], color='red', label='Target' if i == 0 else None)
    # Finally include the imputed data
    axs.scatter([range(pollution_data.shape[0], pollution_data.shape[0] + target_data.shape[0]) for _ in range(len(plot_pollutant_indices))],
                imputed_data[:, plot_pollutant_indices], color='blue', label='Imputed')
    # Include the average of the weather data
    axs.scatter(range(pollution_data.shape[0] - prev_weather_hours - 1, pollution_data.shape[0] + next_weather_hours + auto_regresive_steps - 1), 
                weather_data[:, weather_var_idx, :, :].mean(axis=(1,2)), color='black', label=weather_var_name)

    # Add legend
    axs.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    # Title is the current datetime
    axs.set_title(f'Pollution Data - {contaminant_name} - {current_datetime.strftime("%Y-%m-%d %H:%M")}')
    plt.tight_layout()
    plt.savefig(join(output_folder, f'{batch_idx}_{contaminant_name}_pollution_data_plot.png'))
    plt.close(fig)

    # Second plot the weather data
    start_hour = 0 - prev_weather_hours
    time_steps = weather_data.shape[0]
    fig, axs = plt.subplots(1, time_steps, figsize=(15, 8))
    for i in range(time_steps):
        axs[i].imshow(weather_data[i, weather_var_idx, :, :])
        axs[i].set_title(f'Weather Data - {weather_var_name} - Hour {start_hour + i}', fontsize=10)
        axs[i].set_xlabel('Lon') 
        axs[i].set_ylabel('Lat')
    plt.tight_layout()
    plt.savefig(join(output_folder, f'{batch_idx}_{weather_var_name}_weather_data_plot.png'))
    plt.close(fig)

    # Third plot,  the time related columns
    fig, axs = plt.subplots(figsize=(15, 8))
    # Time related columns are all the ones that are not pollutants
    axs.plot(pollution_data[:, time_related_indices])
    axs.set_title(f'Pollution Data - {contaminant_name} - {current_datetime.strftime("%Y-%m-%d %H:%M")}')
    axs.legend(time_related_columns)
    plt.tight_layout()
    plt.savefig(join(output_folder, f'{batch_idx}_{contaminant_name}_time_related_columns_plot.png'))
    plt.close(fig)

def visualize_pollution_input( pollution_data: np.ndarray,
                               output_folder: str,
                               plot_pollutant_indices: list,
                               pollution_column_names: list,
                               contaminant_name: str,
                               predicted_hour: int
                               ) -> None:
    """
    Visualize pollution input data.
    """
    fig, axs = plt.subplots(figsize=(15, 8))
    axs.plot(pollution_data[0, :, plot_pollutant_indices])
    axs.set_title('Pollution Data - Otres')
    axs.set_xlabel('Hour') 
    axs.legend([pollution_column_names[i] for i in plot_pollutant_indices])
    plt.tight_layout()
    plt.savefig(join(output_folder, f'{predicted_hour}_{contaminant_name}_pollution_data_plot.png'))
    plt.close(fig)