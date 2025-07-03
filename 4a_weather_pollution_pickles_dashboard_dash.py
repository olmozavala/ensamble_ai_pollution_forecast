#!/usr/bin/env python3
"""
Interactive Weather Data Dashboard using Dash

This module provides an interactive dashboard to explore weather data pickle files
with a slider to iterate over time and display 8 weather variables as 25x25 images.

RUN INSTRUCTIONS:
    python 4a_weather_pollution_pickles_dashboard_dash.py
    
    The dashboard will be available at: http://localhost:8050
    
REQUIRED PACKAGES:
    pip install dash dash-bootstrap-components plotly numpy pandas cmocean
"""

import os
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, callback_context
import dash_bootstrap_components as dbc
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Weather variable names and their corresponding color scales
WEATHER_VARIABLES = ['T2', 'U10', 'V10', 'RAINC', 'RAINNC', 'SWDOWN', 'GLW', 'RH']
WEATHER_COLORSCALES = {
    'T2': 'thermal',      # Temperature - thermal colorscale
    'U10': 'RdBu_r',      # Wind U component - diverging colorscale
    'V10': 'RdBu_r',      # Wind V component - diverging colorscale
    'RAINC': 'Blues',     # Convective precipitation - blue colorscale
    'RAINNC': 'Blues',    # Non-convective precipitation - blue colorscale
    'SWDOWN': 'YlOrRd',   # Solar radiation - yellow-orange-red colorscale
    'GLW': 'thermal',     # Longwave radiation - thermal colorscale
    'RH': 'viridis'       # Relative humidity - viridis colorscale
}

# Pollution variable names and their corresponding color scales
POLLUTION_VARIABLES = ['co', 'nodos', 'otres', 'pmdiez', 'pmdoscinco', 'nox', 'no', 'sodos', 'pmco']
POLLUTION_COLORSCALES = {
    'co': 'Reds',         # Carbon monoxide - red colorscale
    'nodos': 'Purples',   # NO2 - purple colorscale
    'otres': 'Blues',     # O3 - blue colorscale
    'pmdiez': 'Oranges',  # PM10 - orange colorscale
    'pmdoscinco': 'Greens', # PM2.5 - green colorscale
    'nox': 'Reds',        # NOx - red colorscale
    'no': 'Reds',         # NO - red colorscale
    'sodos': 'Purples',   # SO2 - purple colorscale
    'pmco': 'Oranges'     # PMCO - orange colorscale
}


class InteractiveWeatherDashboard:
    """
    Interactive dashboard for exploring weather data pickle files using Dash.
    
    This class provides methods to load weather data and create an interactive
    interface with sliders and visualizations.
    """
    
    def __init__(self, data_folder: str = "/unity/f1/ozavala/DATA/AirPollution/TrainingData"):
        """
        Initialize the InteractiveWeatherDashboard.
        
        Args:
            data_folder: Path to the TrainingData directory containing pickle files
        """
        self.data_folder = Path(data_folder)
        self.weather_files: List[Path] = []
        self.pollution_files: List[Path] = []
        self.current_weather_data: Optional[np.ndarray] = None
        self.current_pollution_data: Optional[np.ndarray] = None
        self.current_weather_file: Optional[str] = None
        self.current_pollution_file: Optional[str] = None
        self.stations: List[str] = []
        
    def find_weather_files(self) -> List[Path]:
        """
        Find all weather data pickle files in the data folder.
        
        Returns:
            List of paths to weather data pickle files
        """
        weather_files = list(self.data_folder.glob("weather_data_*.pkl"))
        self.weather_files = sorted(weather_files)
        logger.info(f"Found {len(self.weather_files)} weather data files")
        return self.weather_files
    
    def find_pollution_files(self) -> List[Path]:
        """
        Find all pollution data pickle files in the data folder.
        
        Returns:
            List of paths to pollution data pickle files
        """
        pollution_files = list(self.data_folder.glob("pollution_data_*.pkl"))
        self.pollution_files = sorted(pollution_files)
        logger.info(f"Found {len(self.pollution_files)} pollution data files")
        return self.pollution_files
    
    def get_matching_file_pairs(self) -> List[Tuple[Path, Path]]:
        """
        Find matching weather-pollution file pairs based on year ranges.
        
        Returns:
            List of tuples containing (weather_file, pollution_file) pairs
        """
        pairs = []
        
        for weather_file in self.weather_files:
            # Extract year range from weather filename
            weather_name = weather_file.stem  # e.g., "weather_data_2010_to_2020"
            year_range = weather_name.replace("weather_data_", "")
            
            # Find matching pollution file
            pollution_name = f"pollution_data_{year_range}.pkl"
            pollution_file = self.data_folder / pollution_name
            
            if pollution_file.exists():
                pairs.append((weather_file, pollution_file))
                logger.info(f"Found matching pair: {weather_file.name} <-> {pollution_file.name}")
            else:
                logger.warning(f"No matching pollution file found for {weather_file.name}")
        
        return pairs
    
    def load_pickle_file(self, file_path: Union[str, Path]) -> Union[np.ndarray, pd.DataFrame]:
        """
        Load a pickle file and return the data (numpy array or pandas DataFrame).
        
        Args:
            file_path: Path to the pickle file to load
            
        Returns:
            The loaded data as numpy array or pandas DataFrame
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            pickle.UnpicklingError: If the file can't be unpickled
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            if not isinstance(data, (np.ndarray, pd.DataFrame)):
                raise ValueError(f"Expected numpy array or pandas DataFrame, got {type(data).__name__}")
            
            logger.info(f"Successfully loaded {file_path.name} ({file_path.stat().st_size / 1024 / 1024:.1f} MB) - Type: {type(data).__name__}")
            return data
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            raise
    
    def process_pollution_dataframe(self, df: pd.DataFrame) -> np.ndarray:
        """
        Process pollution DataFrame into a numpy array for analysis.
        
        Args:
            df: Pollution data DataFrame
            
        Returns:
            Processed numpy array with shape (time_steps, stations, pollutants)
        """
        logger.info(f"Processing pollution DataFrame with shape: {df.shape}")
        logger.info(f"DataFrame columns: {list(df.columns)}")
        logger.info(f"DataFrame index: {type(df.index)}")
        
        # Check if DataFrame has a time index
        if isinstance(df.index, pd.DatetimeIndex):
            logger.info(f"Time range: {df.index.min()} to {df.index.max()}")
        
        # Extract unique stations and pollutants from column names
        # Only process columns starting with 'cont_'
        stations = set()
        pollutants = set()
        
        for col in df.columns:
            if col.startswith('cont_'):
                # Parse column name: cont_pollutant_STATION
                parts = col.split('_')
                if len(parts) >= 3:
                    pollutant = parts[1]  # Skip 'cont'
                    station = '_'.join(parts[2:])  # Join remaining parts as station name
                    
                    stations.add(station)
                    pollutants.add(pollutant)
        
        # Convert to sorted lists
        stations_list = sorted(list(stations))
        pollutants_list = sorted(list(pollutants))
        
        logger.info(f"Found {len(stations_list)} stations: {stations_list}")
        logger.info(f"Found {len(pollutants_list)} pollutants: {pollutants_list}")
        
        # Map pollutants to our standard pollution variables
        # Only include pollutants that are in our POLLUTION_VARIABLES list
        valid_pollutants = [p for p in pollutants_list if p in POLLUTION_VARIABLES]
        logger.info(f"Valid pollutants for analysis: {valid_pollutants}")
        
        # Create the output array: (time_steps, stations, pollutants)
        time_steps = len(df)
        num_stations = len(stations_list)
        num_pollutants = len(valid_pollutants)
        
        # Initialize output array with NaN values
        output_array = np.full((time_steps, num_stations, num_pollutants), np.nan)
        
        # Fill the array with data from DataFrame
        for i, station in enumerate(stations_list):
            for j, pollutant in enumerate(valid_pollutants):
                # Look for columns that match this station-pollutant combination
                col_name = f"cont_{pollutant}_{station}"
                if col_name in df.columns:
                    output_array[:, i, j] = df[col_name].values
                    logger.debug(f"Found data for {col_name}")
                else:
                    logger.warning(f"Column not found: {col_name}")
        
        # Store station and pollutant names for later use
        self.station_names = stations_list
        self.pollutant_names = valid_pollutants
        
        logger.info(f"Processed array shape: {output_array.shape}")
        logger.info(f"Non-NaN values: {np.count_nonzero(~np.isnan(output_array))}")
        
        return output_array
    
    def load_data_server_side(self, weather_filename: str, pollution_filename: str = None) -> bool:
        """
        Load weather and pollution data server-side and store it in memory.
        
        Args:
            weather_filename: Name of the weather data file to load
            pollution_filename: Name of the pollution data file to load (optional)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load weather data
            weather_path = self.data_folder / weather_filename
            weather_data = self.load_pickle_file(weather_path)
            
            if isinstance(weather_data, pd.DataFrame):
                raise ValueError("Weather data should be a numpy array, not a DataFrame")
            
            self.current_weather_data = weather_data
            self.current_weather_file = weather_filename
            
            # Load pollution data if provided
            if pollution_filename:
                pollution_path = self.data_folder / pollution_filename
                pollution_data = self.load_pickle_file(pollution_path)
                
                if isinstance(pollution_data, pd.DataFrame):
                    # Process DataFrame into numpy array
                    self.current_pollution_data = self.process_pollution_dataframe(pollution_data)
                else:
                    # Already a numpy array
                    self.current_pollution_data = pollution_data
                
                self.current_pollution_file = pollution_filename
                
                # Extract station names from pollution data
                if len(self.current_pollution_data.shape) >= 3:
                    num_stations = self.current_pollution_data.shape[1]
                    # Use actual station names if available, otherwise generate generic names
                    if hasattr(self, 'station_names') and self.station_names:
                        self.stations = self.station_names
                    else:
                        self.stations = [f"Station_{i+1}" for i in range(num_stations)]
                else:
                    self.stations = ["Station_1"]  # Default if shape is different
            else:
                self.current_pollution_data = None
                self.current_pollution_file = None
                self.stations = []
            
            logger.info(f"Data loaded server-side: Weather={weather_filename}, Pollution={pollution_filename}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            return False
    
    def get_current_frame(self, time_idx: int) -> np.ndarray:
        """
        Get the current frame data for a specific time index.
        
        Args:
            time_idx: Time index to retrieve
            
        Returns:
            The frame data as numpy array
        """
        if self.current_weather_data is None:
            raise ValueError("No weather data loaded")
        
        if time_idx >= self.current_weather_data.shape[0]:
            raise ValueError(f"Time index {time_idx} out of range")
        
        return self.current_weather_data[time_idx, :, :, :]
    
    def get_pollution_data_for_station(self, station_idx: int) -> np.ndarray:
        """
        Get pollution data for a specific station.
        
        Args:
            station_idx: Station index to retrieve
            
        Returns:
            Pollution data array for the station
        """
        if self.current_pollution_data is None:
            raise ValueError("No pollution data loaded")
        
        if station_idx >= self.current_pollution_data.shape[1]:
            raise ValueError(f"Station index {station_idx} out of range")
        
        return self.current_pollution_data[:, station_idx, :]
    
    def get_time_series_data(self, start_time: int, window_size: int) -> np.ndarray:
        """
        Get time series data for spatially averaged variables over a time window.
        
        Args:
            start_time: Starting time index
            window_size: Number of time steps in the window
            
        Returns:
            Time series data array with shape (window_size, num_variables)
        """
        if self.current_weather_data is None:
            raise ValueError("No weather data loaded")
        
        end_time = min(start_time + window_size, self.current_weather_data.shape[0])
        actual_window_size = end_time - start_time
        
        # Extract the time window and calculate spatial averages
        time_window_data = self.current_weather_data[start_time:end_time, :, :, :]
        
        # Calculate spatial average for each variable and time step
        # Shape: (window_size, num_variables)
        time_series = np.mean(time_window_data, axis=(2, 3))
        
        return time_series
    
    def get_pollution_time_series_data(self, start_time: int, window_size: int, station_idx: int) -> np.ndarray:
        """
        Get time series data for pollution variables at a specific station over a time window.
        
        Args:
            start_time: Starting time index
            window_size: Number of time steps in the window
            station_idx: Station index
            
        Returns:
            Pollution time series data array with shape (window_size, num_pollutants)
        """
        if self.current_pollution_data is None:
            raise ValueError("No pollution data loaded")
        
        end_time = min(start_time + window_size, self.current_pollution_data.shape[0])
        
        # Extract the time window for the specific station
        pollution_data = self.get_pollution_data_for_station(station_idx)
        time_series = pollution_data[start_time:end_time, :]
        
        return time_series
    
    def get_data_info(self, filename: str) -> Dict[str, Any]:
        """
        Get information about the weather data array.
        
        Args:
            filename: Name of the file for reference
            
        Returns:
            Dictionary containing data information
        """
        if self.current_weather_data is None:
            raise ValueError("No weather data loaded")
        
        info = {
            "filename": filename,
            "shape": self.current_weather_data.shape,
            "dtype": str(self.current_weather_data.dtype),
            "size_mb": self.current_weather_data.nbytes / 1024 / 1024,
            "time_steps": self.current_weather_data.shape[0],
            "variables": len(WEATHER_VARIABLES),
            "spatial_dims": self.current_weather_data.shape[2:],
            "nan_count": int(np.isnan(self.current_weather_data).sum())
        }
        
        # Add pollution data info if available
        if self.current_pollution_data is not None:
            info.update({
                "pollution_filename": self.current_pollution_file,
                "pollution_shape": self.current_pollution_data.shape,
                "pollution_size_mb": self.current_pollution_data.nbytes / 1024 / 1024,
                "pollution_variables": len(POLLUTION_VARIABLES),
                "num_stations": len(self.stations),
                "pollution_nan_count": int(np.isnan(self.current_pollution_data).sum())
            })
        
        return info
    
    def create_weather_variable_plot(self, frame_data: np.ndarray, var_idx: int, 
                                   var_name: str) -> go.Figure:
        """
        Create a plot for a specific weather variable.
        
        Args:
            frame_data: The frame data array (8, 25, 25)
            var_idx: Variable index to plot
            var_name: Name of the weather variable
            
        Returns:
            Plotly figure with the plot
        """
        # Extract the 2D slice for this variable
        slice_data = frame_data[var_idx, :, :]
        
        # Get the appropriate colorscale for this variable
        colorscale = WEATHER_COLORSCALES.get(var_name, 'viridis')
        
        # Create the heatmap
        fig = go.Figure(data=go.Heatmap(
            z=slice_data,
            colorscale=colorscale,
            showscale=True,
            colorbar=dict(title=f'{var_name} Value')
        ))
        
        fig.update_layout(
            title=f'{var_name}',
            xaxis_title='Longitude Index',
            yaxis_title='Latitude Index',
            width=600,
            height=500
        )
        
        # Add statistics as annotation
        stats_text = f'Min: {np.min(slice_data):.3f}<br>Max: {np.max(slice_data):.3f}<br>Mean: {np.mean(slice_data):.3f}'
        fig.add_annotation(
            x=0.02, y=0.98,
            xref='paper', yref='paper',
            text=stats_text,
            showarrow=False,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='black',
            borderwidth=1
        )
        
        return fig
    
    def create_all_variables_plot(self, frame_data: np.ndarray) -> go.Figure:
        """
        Create a plot showing all 8 weather variables.
        
        Args:
            frame_data: The frame data array (8, 25, 25)
            
        Returns:
            Plotly figure with all variables
        """
        # Create subplot grid: 2 rows, 4 columns
        fig = make_subplots(
            rows=2, cols=4,
            subplot_titles=WEATHER_VARIABLES,
            specs=[[{"secondary_y": False}] * 4] * 2,
            vertical_spacing=0.15,  # Increased spacing
            horizontal_spacing=0.05
        )
        
        for i, var_name in enumerate(WEATHER_VARIABLES):
            row = (i // 4) + 1
            col = (i % 4) + 1
            
            # Extract the 2D slice for this variable
            slice_data = frame_data[i, :, :]
            
            # Get the appropriate colorscale for this variable
            colorscale = WEATHER_COLORSCALES.get(var_name, 'viridis')
            
            # Add heatmap to subplot
            fig.add_trace(
                go.Heatmap(
                    z=slice_data,
                    colorscale=colorscale,
                    showscale=False,
                    name=var_name
                ),
                row=row, col=col
            )
            
            # Add statistics as annotation
            stats_text = f'Min: {np.min(slice_data):.2f}<br>Max: {np.max(slice_data):.2f}'
            fig.add_annotation(
                x=0.02, y=0.98,
                xref=f'x{i+1}', yref=f'y{i+1}',
                text=stats_text,
                showarrow=False,
                font=dict(size=8),
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='black',
                borderwidth=1
            )
        
        fig.update_layout(
            title=f'All Weather Variables',
            width=1200,
            height=600,
            showlegend=False
        )
        
        return fig
    
    def create_time_series_plot(self, time_series_data: np.ndarray, start_time: int, 
                               window_size: int) -> go.Figure:
        """
        Create time series plots for all weather variables.
        
        Args:
            time_series_data: Time series data array (window_size, num_variables)
            start_time: Starting time index
            window_size: Window size in hours
            
        Returns:
            Plotly figure with time series plots
        """
        # Create subplot grid: 2 rows, 4 columns
        fig = make_subplots(
            rows=2, cols=4,
            subplot_titles=WEATHER_VARIABLES,
            specs=[[{"secondary_y": False}] * 4] * 2,
            vertical_spacing=0.15,  # Increased spacing
            horizontal_spacing=0.05
        )
        
        # Create time axis (hours from start)
        time_hours = np.arange(len(time_series_data))
        
        for i, var_name in enumerate(WEATHER_VARIABLES):
            row = (i // 4) + 1
            col = (i % 4) + 1
            
            # Get time series for this variable
            var_data = time_series_data[:, i]
            
            # Get the appropriate colorscale for this variable
            colorscale = WEATHER_COLORSCALES.get(var_name, 'viridis')
            
            # Add line plot to subplot
            fig.add_trace(
                go.Scatter(
                    x=time_hours,
                    y=var_data,
                    mode='lines+markers',
                    name=var_name,
                    line=dict(color=px.colors.qualitative.Set1[i % len(px.colors.qualitative.Set1)]),
                    marker=dict(size=4)
                ),
                row=row, col=col
            )
            
            # Add statistics as annotation
            stats_text = f'Min: {np.min(var_data):.2f}<br>Max: {np.max(var_data):.2f}<br>Mean: {np.mean(var_data):.2f}'
            fig.add_annotation(
                x=0.02, y=0.98,
                xref=f'x{i+1}', yref=f'y{i+1}',
                text=stats_text,
                showarrow=False,
                font=dict(size=8),
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='black',
                borderwidth=1
            )
            
            # Update axis labels
            fig.update_xaxes(title_text="Hours", row=row, col=col)
            fig.update_yaxes(title_text=var_name, row=row, col=col)
        
        fig.update_layout(
            title=f'Time Series: Spatially Averaged Weather Variables<br>'
                  f'Starting from Time Step {start_time}, Window: {len(time_series_data)} hours',
            width=None,
            height=1000,  # Increased height
            showlegend=False
        )
        
        return fig
    
    def create_pollution_time_series_plot(self, pollution_time_series_data: np.ndarray, 
                                        start_time: int, window_size: int, 
                                        station_name: str) -> go.Figure:
        """
        Create time series plots for all pollution variables at a specific station.
        
        Args:
            pollution_time_series_data: Pollution time series data array (window_size, num_pollutants)
            start_time: Starting time index
            window_size: Window size in hours
            station_name: Name of the station
            
        Returns:
            Plotly figure with pollution time series plots
        """
        # Use actual pollutant names if available
        pollutant_names = getattr(self, 'pollutant_names', POLLUTION_VARIABLES)
        num_pollutants = len(pollutant_names)
        
        # Calculate grid dimensions
        if num_pollutants <= 9:
            rows, cols = 3, 3
        elif num_pollutants <= 12:
            rows, cols = 3, 4
        else:
            rows, cols = 4, 4
        
        # Create subplot grid
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=pollutant_names,
            specs=[[{"secondary_y": False}] * cols] * rows,
            vertical_spacing=0.15,  # Increased spacing
            horizontal_spacing=0.05
        )
        
        # Create time axis (hours from start)
        time_hours = np.arange(len(pollution_time_series_data))
        
        for i, pollutant_name in enumerate(pollutant_names):
            row = (i // cols) + 1
            col = (i % cols) + 1
            
            # Get time series for this pollutant
            pollutant_data = pollution_time_series_data[:, i]
            
            # Get the appropriate colorscale for this pollutant
            colorscale = POLLUTION_COLORSCALES.get(pollutant_name, 'Reds')
            
            # Add line plot to subplot
            fig.add_trace(
                go.Scatter(
                    x=time_hours,
                    y=pollutant_data,
                    mode='lines+markers',
                    name=pollutant_name,
                    line=dict(color=px.colors.qualitative.Set1[i % len(px.colors.qualitative.Set1)]),
                    marker=dict(size=4)
                ),
                row=row, col=col
            )
            
            # Add statistics as annotation
            stats_text = f'Min: {np.min(pollutant_data):.2f}<br>Max: {np.max(pollutant_data):.2f}<br>Mean: {np.mean(pollutant_data):.2f}'
            fig.add_annotation(
                x=0.02, y=0.98,
                xref=f'x{i+1}', yref=f'y{i+1}',
                text=stats_text,
                showarrow=False,
                font=dict(size=8),
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='black',
                borderwidth=1
            )
            
            # Update axis labels
            fig.update_xaxes(title_text="Hours", row=row, col=col)
            fig.update_yaxes(title_text=pollutant_name, row=row, col=col)
        
        fig.update_layout(
            title=f'Pollution Time Series: {station_name}<br>'
                  f'Starting from Time Step {start_time}, Window: {len(pollution_time_series_data)} hours',
            width=None,
            height=1100,  # Increased height
            showlegend=False
        )
        
        return fig
    
    def create_combined_time_series_plot(self, weather_time_series_data: np.ndarray,
                                       pollution_time_series_data: np.ndarray,
                                       start_time: int, window_size: int,
                                       station_name: str) -> go.Figure:
        """
        Create combined time series plots showing both weather and pollution data.
        
        Args:
            weather_time_series_data: Weather time series data array (window_size, num_weather_variables)
            pollution_time_series_data: Pollution time series data array (window_size, num_pollutants)
            start_time: Starting time index
            window_size: Window size in hours
            station_name: Name of the station
            
        Returns:
            Plotly figure with combined time series plots
        """
        # Create subplot grid: 4 rows, 4 columns (8 weather + 9 pollution = 17 variables)
        fig = make_subplots(
            rows=4, cols=4,
            subplot_titles=WEATHER_VARIABLES + getattr(self, 'pollutant_names', POLLUTION_VARIABLES),
            specs=[[{"secondary_y": False}] * 4] * 4,
            vertical_spacing=0.15,  # Increased spacing
            horizontal_spacing=0.05
        )
        
        # Create time axis (hours from start)
        time_hours = np.arange(len(weather_time_series_data))
        
        # Plot weather variables (first 8 subplots)
        for i, var_name in enumerate(WEATHER_VARIABLES):
            row = (i // 4) + 1
            col = (i % 4) + 1
            
            var_data = weather_time_series_data[:, i]
            
            fig.add_trace(
                go.Scatter(
                    x=time_hours,
                    y=var_data,
                    mode='lines+markers',
                    name=var_name,
                    line=dict(color='blue'),
                    marker=dict(size=3)
                ),
                row=row, col=col
            )
            
            # Add statistics
            stats_text = f'Min: {np.min(var_data):.2f}<br>Max: {np.max(var_data):.2f}<br>Mean: {np.mean(var_data):.2f}'
            fig.add_annotation(
                x=0.02, y=0.98,
                xref=f'x{i+1}', yref=f'y{i+1}',
                text=stats_text,
                showarrow=False,
                font=dict(size=7),
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='black',
                borderwidth=1
            )
            
            fig.update_xaxes(title_text="Hours", row=row, col=col)
            fig.update_yaxes(title_text=var_name, row=row, col=col)
        
        # Plot pollution variables (remaining subplots)
        pollutant_names = getattr(self, 'pollutant_names', POLLUTION_VARIABLES)
        for i, pollutant_name in enumerate(pollutant_names):
            # Calculate position in the 4x4 grid (weather takes first 8 positions)
            grid_pos = 8 + i
            row = (grid_pos // 4) + 1
            col = (grid_pos % 4) + 1
            
            pollutant_data = pollution_time_series_data[:, i]
            
            fig.add_trace(
                go.Scatter(
                    x=time_hours,
                    y=pollutant_data,
                    mode='lines+markers',
                    name=pollutant_name,
                    line=dict(color='red'),
                    marker=dict(size=3)
                ),
                row=row, col=col
            )
            
            # Add statistics
            stats_text = f'Min: {np.min(pollutant_data):.2f}<br>Max: {np.max(pollutant_data):.2f}<br>Mean: {np.mean(pollutant_data):.2f}'
            fig.add_annotation(
                x=0.02, y=0.98,
                xref=f'x{grid_pos+1}', yref=f'y{grid_pos+1}',
                text=stats_text,
                showarrow=False,
                font=dict(size=7),
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='black',
                borderwidth=1
            )
            
            fig.update_xaxes(title_text="Hours", row=row, col=col)
            fig.update_yaxes(title_text=pollutant_name, row=row, col=col)
        
        pollutant_names = getattr(self, 'pollutant_names', POLLUTION_VARIABLES)
        fig.update_layout(
            title=f'Combined Time Series: Weather + Pollution at {station_name}<br>'
                  f'Starting from Time Step {start_time}, Window: {len(weather_time_series_data)} hours<br>'
                  f'Weather Variables: {len(WEATHER_VARIABLES)}, Pollution Variables: {len(pollutant_names)}',
            width=None,
            height=1200,  # Increased height
            showlegend=False
        )
        
        return fig


# Initialize the dashboard
dashboard = InteractiveWeatherDashboard()

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Find weather and pollution files
weather_files = dashboard.find_weather_files()
pollution_files = dashboard.find_pollution_files()
file_pairs = dashboard.get_matching_file_pairs()

# Create file options for dropdown (show weather files that have matching pollution files)
file_options = []
for weather_file, pollution_file in file_pairs:
    label = f"{weather_file.name} + {pollution_file.name}"
    value = f"{weather_file.name}|{pollution_file.name}"
    file_options.append({'label': label, 'value': value})

# If no pairs found, show weather files only
if not file_options:
    file_options = [{'label': f.name, 'value': f.name} for f in weather_files]

# App layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Weather Data Interactive Dashboard", className="text-center mb-4"),
            html.Hr()
        ])
    ]),
    
    # File selection
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Data Selection"),
                dbc.CardBody([
                    dcc.Dropdown(
                        id='file-dropdown',
                        options=file_options,
                        value=file_options[0]['value'] if file_options else None,
                        placeholder="Select a weather data file..."
                    ),
                    html.Br(),
                    dbc.Button("Load Data", id="load-button", color="primary", className="mt-2")
                ])
            ])
        ], width=6),
        
        # Data info
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Data Information"),
                dbc.CardBody(id="data-info")
            ])
        ], width=6)
    ], className="mb-4"),
    
    # Time navigation
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Time Navigation"),
                dbc.CardBody([
                    dcc.Slider(
                        id='time-slider',
                        min=0,
                        max=100,  # Will be updated dynamically
                        value=0,
                        marks={},
                        step=1,
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                    html.Div(id="time-info", className="mt-2")
                ])
            ])
        ])
    ], className="mb-4"),
    
    # Tabs for different views
    dbc.Row([
        dbc.Col([
            dbc.Tabs([
                dbc.Tab([
                    html.Div(id="all-variables-plot")
                ], label="All Variables View", tab_id="tab-all"),
                
                dbc.Tab([
                    dbc.Row([
                        dbc.Col([
                            dcc.Dropdown(
                                id='variable-dropdown',
                                options=[{'label': var, 'value': i} for i, var in enumerate(WEATHER_VARIABLES)],
                                value=0,
                                placeholder="Select a weather variable..."
                            )
                        ], width=4)
                    ], className="mb-3"),
                    html.Div(id="single-variable-plot")
                ], label="Single Variable View", tab_id="tab-single"),
                
                dbc.Tab([
                    dbc.Row([
                        dbc.Col([
                            html.H6("Window Size (hours):"),
                            dcc.Slider(
                                id='window-size-slider',
                                min=48,
                                max=168,  # 24*7 = 168 hours
                                value=48,
                                step=1,
                                marks={48: "48h", 72: "72h", 96: "96h", 120: "120h", 144: "144h", 168: "168h"},
                                tooltip={"placement": "bottom", "always_visible": True}
                            )
                        ], width=4),
                        dbc.Col([
                            html.H6("Start Time:"),
                            dcc.Slider(
                                id='start-time-slider',
                                min=0,
                                max=100,
                                value=0,
                                step=1,
                                marks={},
                                tooltip={"placement": "bottom", "always_visible": True}
                            )
                        ], width=4),
                        dbc.Col([
                            html.H6("Station:"),
                            dcc.Dropdown(
                                id='station-dropdown',
                                options=[],
                                value=None,
                                placeholder="Select a station..."
                            )
                        ], width=4)
                    ], className="mb-3"),
                    html.Div(id="time-series-plot"),
                    html.Div(id="time-series-info", className="mb-2")
                ], label="Weather Time Series", tab_id="tab-time-series"),
                
                dbc.Tab([
                    dbc.Row([
                        dbc.Col([
                            html.H6("Window Size (hours):"),
                            dcc.Slider(
                                id='pollution-window-slider',
                                min=48,
                                max=168,
                                value=48,
                                step=1,
                                marks={48: "48h", 72: "72h", 96: "96h", 120: "120h", 144: "144h", 168: "168h"},
                                tooltip={"placement": "bottom", "always_visible": True}
                            )
                        ], width=4),
                        dbc.Col([
                            html.H6("Start Time:"),
                            dcc.Slider(
                                id='pollution-start-slider',
                                min=0,
                                max=100,
                                value=0,
                                step=1,
                                marks={},
                                tooltip={"placement": "bottom", "always_visible": True}
                            )
                        ], width=4),
                        dbc.Col([
                            html.H6("Station:"),
                            dcc.Dropdown(
                                id='pollution-station-dropdown',
                                options=[],
                                value=None,
                                placeholder="Select a station..."
                            )
                        ], width=4)
                    ], className="mb-3"),
                    html.Div(id="pollution-time-series-plot"),
                    html.Div(id="pollution-time-series-info", className="mb-2")
                ], label="Pollution Time Series", tab_id="tab-pollution-time-series"),
                
                dbc.Tab([
                    dbc.Row([
                        dbc.Col([
                            html.H6("Window Size (hours):"),
                            dcc.Slider(
                                id='combined-window-slider',
                                min=48,
                                max=168,
                                value=48,
                                step=1,
                                marks={48: "48h", 72: "72h", 96: "96h", 120: "120h", 144: "144h", 168: "168h"},
                                tooltip={"placement": "bottom", "always_visible": True}
                            )
                        ], width=4),
                        dbc.Col([
                            html.H6("Start Time:"),
                            dcc.Slider(
                                id='combined-start-slider',
                                min=0,
                                max=100,
                                value=0,
                                step=1,
                                marks={},
                                tooltip={"placement": "bottom", "always_visible": True}
                            )
                        ], width=4),
                        dbc.Col([
                            html.H6("Station:"),
                            dcc.Dropdown(
                                id='combined-station-dropdown',
                                options=[],
                                value=None,
                                placeholder="Select a station..."
                            )
                        ], width=4)
                    ], className="mb-3"),
                    html.Div(id="combined-time-series-info", className="mb-2"),
                    html.Div(id="combined-time-series-plot")
                ], label="Combined Time Series", tab_id="tab-combined-time-series")
            ], id="tabs")
        ])
    ]),
    
    # Footer
    dbc.Row([
        dbc.Col([
            html.Hr(),
            html.H5("Weather Variables Description:"),
            html.Ul([
                html.Li("T2: 2-meter temperature (°C) - Thermal colorscale"),
                html.Li("U10: 10-meter U wind component (m/s) - Balance colorscale"),
                html.Li("V10: 10-meter V wind component (m/s) - Balance colorscale"),
                html.Li("RAINC: Cumulative convective precipitation (mm) - Rain colorscale"),
                html.Li("RAINNC: Cumulative non-convective precipitation (mm) - Rain colorscale"),
                html.Li("SWDOWN: Downward shortwave flux at ground surface (W/m²) - Solar colorscale"),
                html.Li("GLW: Downward longwave flux at ground surface (W/m²) - Thermal colorscale"),
                html.Li("RH: Relative humidity (%) - Haline colorscale")
            ])
        ])
    ]),
    
    # Store for current frame data
    dcc.Store(id='current-frame-store'),
    dcc.Store(id='data-info-store')
    
], fluid=True)


# Callbacks
@app.callback(
    [Output('data-info-store', 'data'),
     Output('time-slider', 'max'),
     Output('time-slider', 'marks'),
     Output('start-time-slider', 'max'),
     Output('start-time-slider', 'marks'),
     Output('pollution-start-slider', 'max'),
     Output('pollution-start-slider', 'marks'),
     Output('combined-start-slider', 'max'),
     Output('combined-start-slider', 'marks'),
     Output('station-dropdown', 'options'),
     Output('pollution-station-dropdown', 'options'),
     Output('combined-station-dropdown', 'options'),
     Output('data-info', 'children')],
    [Input('load-button', 'n_clicks')],
    [dash.dependencies.State('file-dropdown', 'value')]
)
def load_data(n_clicks, selected_file):
    """Load weather and pollution data server-side when button is clicked."""
    if n_clicks is None or selected_file is None:
        return [dash.no_update] * 12 + [dash.no_update]
    
    try:
        # Parse selected file (could be a pair or single file)
        if '|' in selected_file:
            # File pair selected
            weather_filename, pollution_filename = selected_file.split('|')
            success = dashboard.load_data_server_side(weather_filename, pollution_filename)
        else:
            # Single weather file selected
            success = dashboard.load_data_server_side(selected_file)
        
        if not success:
            return [dash.no_update] * 12 + [
                dbc.Alert("Error loading data", color="danger")
            ]
        
        info = dashboard.get_data_info(selected_file)
        
        # Create marks for time sliders
        max_time = info['time_steps'] - 1
        marks = {i: str(i) for i in range(0, max_time + 1, max(1, max_time // 10))}
        start_time_marks = {i: str(i) for i in range(0, max_time + 1, max(1, max_time // 20))}
        
        # Create station options
        station_options = [{'label': station, 'value': i} for i, station in enumerate(dashboard.stations)]
        
        # Create data info cards
        info_cards = [
            dbc.Row([
                dbc.Col(dbc.Card([
                    dbc.CardBody([
                        html.H6("Time Steps", className="card-title"),
                        html.P(str(info['time_steps']), className="card-text")
                    ])
                ]), width=3),
                dbc.Col(dbc.Card([
                    dbc.CardBody([
                        html.H6("Weather Variables", className="card-title"),
                        html.P(str(info['variables']), className="card-text")
                    ])
                ]), width=3),
                dbc.Col(dbc.Card([
                    dbc.CardBody([
                        html.H6("Spatial Grid", className="card-title"),
                        html.P(f"{info['spatial_dims'][0]}×{info['spatial_dims'][1]}", className="card-text")
                    ])
                ]), width=3),
                dbc.Col(dbc.Card([
                    dbc.CardBody([
                        html.H6("File Size", className="card-title"),
                        html.P(f"{info['size_mb']:.1f} MB", className="card-text")
                    ])
                ]), width=3)
            ])
        ]
        
        # Add pollution info if available
        if 'pollution_filename' in info:
            info_cards.append(
                dbc.Row([
                    dbc.Col(dbc.Card([
                        dbc.CardBody([
                            html.H6("Pollution File", className="card-title"),
                            html.P(info['pollution_filename'], className="card-text")
                        ])
                    ]), width=4),
                    dbc.Col(dbc.Card([
                        dbc.CardBody([
                            html.H6("Pollution Variables", className="card-title"),
                            html.P(str(info['pollution_variables']), className="card-text")
                        ])
                    ]), width=4),
                    dbc.Col(dbc.Card([
                        dbc.CardBody([
                            html.H6("Stations", className="card-title"),
                            html.P(str(info['num_stations']), className="card-text")
                        ])
                    ]), width=4)
                ], className="mt-3")
            )
        
        return (info, max_time, marks, max_time, start_time_marks, 
                max_time, start_time_marks, max_time, start_time_marks,
                station_options, station_options, station_options, info_cards)
        
    except Exception as e:
        return [dash.no_update] * 12 + [
            dbc.Alert(f"Error loading data: {str(e)}", color="danger")
        ]


@app.callback(
    [Output('current-frame-store', 'data'),
     Output('time-info', 'children')],
    [Input('time-slider', 'value'),
     Input('data-info-store', 'data')]
)
def update_current_frame(time_idx, data_info):
    """Update current frame data when time slider changes."""
    if data_info is None or dashboard.current_weather_data is None:
        return dash.no_update, ""
    
    try:
        # Get current frame data
        frame_data = dashboard.get_current_frame(time_idx)
        
        # Convert to list for JSON serialization (only the current frame)
        frame_list = frame_data.tolist()
        
        max_time = data_info.get('time_steps', 0) - 1
        time_info = f"Currently viewing time step {time_idx} of {max_time + 1} total time steps"
        
        return frame_list, time_info
        
    except Exception as e:
        return dash.no_update, f"Error updating frame: {str(e)}"


@app.callback(
    Output('all-variables-plot', 'children'),
    [Input('current-frame-store', 'data')]
)
def update_all_variables_plot(frame_data):
    """Update the all variables plot."""
    if frame_data is None:
        return html.P("Please load data first.")
    
    try:
        frame_array = np.array(frame_data)
        fig = dashboard.create_all_variables_plot(frame_array)
        return dcc.Graph(figure=fig)
    except Exception as e:
        return html.P(f"Error creating plot: {str(e)}")


@app.callback(
    Output('single-variable-plot', 'children'),
    [Input('current-frame-store', 'data'),
     Input('variable-dropdown', 'value')]
)
def update_single_variable_plot(frame_data, var_idx):
    """Update the single variable plot."""
    if frame_data is None or var_idx is None:
        return html.P("Please load data and select a variable first.")
    
    try:
        frame_array = np.array(frame_data)
        var_name = WEATHER_VARIABLES[var_idx]
        fig = dashboard.create_weather_variable_plot(frame_array, var_idx, var_name)
        
        # Add statistics
        var_data = frame_array[var_idx, :, :]
        stats = [
            dbc.Row([
                dbc.Col(dbc.Card([
                    dbc.CardBody([
                        html.H6(f"{var_name} Min", className="card-title"),
                        html.P(f"{np.min(var_data):.3f}", className="card-text")
                    ])
                ]), width=3),
                dbc.Col(dbc.Card([
                    dbc.CardBody([
                        html.H6(f"{var_name} Max", className="card-title"),
                        html.P(f"{np.max(var_data):.3f}", className="card-text")
                    ])
                ]), width=3),
                dbc.Col(dbc.Card([
                    dbc.CardBody([
                        html.H6(f"{var_name} Mean", className="card-title"),
                        html.P(f"{np.mean(var_data):.3f}", className="card-text")
                    ])
                ]), width=3),
                dbc.Col(dbc.Card([
                    dbc.CardBody([
                        html.H6(f"{var_name} Std", className="card-title"),
                        html.P(f"{np.std(var_data):.3f}", className="card-text")
                    ])
                ]), width=3)
            ], className="mb-3"),
            dcc.Graph(figure=fig)
        ]
        
        return stats
        
    except Exception as e:
        return html.P(f"Error creating plot: {str(e)}")


@app.callback(
    Output('time-series-info', 'children'),
    [Input('start-time-slider', 'value'),
     Input('window-size-slider', 'value'),
     Input('data-info-store', 'data')]
)
def update_time_series_info(start_time, window_size, data_info):
    """Update time series information display."""
    if data_info is None:
        return ""
    
    max_time = data_info.get('time_steps', 0) - 1
    end_time = min(start_time + window_size, max_time + 1)
    actual_window = end_time - start_time
    
    return html.Div([
        dbc.Alert([
            html.H6("Time Series Configuration:", className="alert-heading"),
            html.P(f"• Start Time: {start_time} hours"),
            html.P(f"• Window Size: {window_size} hours"),
            html.P(f"• End Time: {end_time} hours"),
            html.P(f"• Actual Window: {actual_window} hours"),
            html.P(f"• Available Data: {max_time + 1} total hours")
        ], color="info")
    ])


@app.callback(
    Output('time-series-plot', 'children'),
    [Input('start-time-slider', 'value'),
     Input('window-size-slider', 'value'),
     Input('data-info-store', 'data')]
)
def update_time_series_plot(start_time, window_size, data_info):
    """Update the weather time series plot."""
    if data_info is None or dashboard.current_weather_data is None:
        return html.P("Please load data first.")
    
    try:
        time_series_data = dashboard.get_time_series_data(start_time, window_size)
        fig = dashboard.create_time_series_plot(time_series_data, start_time, window_size)
        return dcc.Graph(figure=fig, style={'width': '100%'})
    except Exception as e:
        return html.P(f"Error creating time series plot: {str(e)}")


@app.callback(
    Output('pollution-time-series-info', 'children'),
    [Input('pollution-start-slider', 'value'),
     Input('pollution-window-slider', 'value'),
     Input('pollution-station-dropdown', 'value'),
     Input('data-info-store', 'data')]
)
def update_pollution_time_series_info(start_time, window_size, station_idx, data_info):
    """Update pollution time series information display."""
    if data_info is None or station_idx is None:
        return ""
    
    max_time = data_info.get('time_steps', 0) - 1
    end_time = min(start_time + window_size, max_time + 1)
    actual_window = end_time - start_time
    station_name = dashboard.stations[station_idx] if station_idx < len(dashboard.stations) else f"Station_{station_idx+1}"
    
    return html.Div([
        dbc.Alert([
            html.H6("Pollution Time Series Configuration:", className="alert-heading"),
            html.P(f"• Station: {station_name}"),
            html.P(f"• Start Time: {start_time} hours"),
            html.P(f"• Window Size: {window_size} hours"),
            html.P(f"• End Time: {end_time} hours"),
            html.P(f"• Actual Window: {actual_window} hours"),
            html.P(f"• Available Data: {max_time + 1} total hours")
        ], color="warning")
    ])


@app.callback(
    Output('pollution-time-series-plot', 'children'),
    [Input('pollution-start-slider', 'value'),
     Input('pollution-window-slider', 'value'),
     Input('pollution-station-dropdown', 'value'),
     Input('data-info-store', 'data')]
)
def update_pollution_time_series_plot(start_time, window_size, station_idx, data_info):
    """Update the pollution time series plot."""
    if data_info is None or dashboard.current_pollution_data is None or station_idx is None:
        return html.P("Please load data and select a station first.")
    
    try:
        pollution_time_series_data = dashboard.get_pollution_time_series_data(start_time, window_size, station_idx)
        station_name = dashboard.stations[station_idx] if station_idx < len(dashboard.stations) else f"Station_{station_idx+1}"
        fig = dashboard.create_pollution_time_series_plot(pollution_time_series_data, start_time, window_size, station_name)
        return dcc.Graph(figure=fig, style={'width': '100%'})
    except Exception as e:
        return html.P(f"Error creating pollution time series plot: {str(e)}")


@app.callback(
    Output('combined-time-series-info', 'children'),
    [Input('combined-start-slider', 'value'),
     Input('combined-window-slider', 'value'),
     Input('combined-station-dropdown', 'value'),
     Input('data-info-store', 'data')]
)
def update_combined_time_series_info(start_time, window_size, station_idx, data_info):
    """Update combined time series information display."""
    if data_info is None or station_idx is None:
        return ""
    
    max_time = data_info.get('time_steps', 0) - 1
    end_time = min(start_time + window_size, max_time + 1)
    actual_window = end_time - start_time
    station_name = dashboard.stations[station_idx] if station_idx < len(dashboard.stations) else f"Station_{station_idx+1}"
    
    return html.Div([
        dbc.Alert([
            html.H6("Combined Time Series Configuration:", className="alert-heading"),
            html.P(f"• Station: {station_name}"),
            html.P(f"• Start Time: {start_time} hours"),
            html.P(f"• Window Size: {window_size} hours"),
            html.P(f"• End Time: {end_time} hours"),
            html.P(f"• Actual Window: {actual_window} hours"),
            html.P(f"• Available Data: {max_time + 1} total hours"),
            html.P(f"• Weather Variables: {len(WEATHER_VARIABLES)}"),
            html.P(f"• Pollution Variables: {len(POLLUTION_VARIABLES)}")
        ], color="success")
    ])


@app.callback(
    Output('combined-time-series-plot', 'children'),
    [Input('combined-start-slider', 'value'),
     Input('combined-window-slider', 'value'),
     Input('combined-station-dropdown', 'value'),
     Input('data-info-store', 'data')]
)
def update_combined_time_series_plot(start_time, window_size, station_idx, data_info):
    """Update the combined time series plot."""
    if data_info is None or dashboard.current_weather_data is None or dashboard.current_pollution_data is None or station_idx is None:
        return html.P("Please load both weather and pollution data and select a station first.")
    
    try:
        weather_time_series_data = dashboard.get_time_series_data(start_time, window_size)
        pollution_time_series_data = dashboard.get_pollution_time_series_data(start_time, window_size, station_idx)
        station_name = dashboard.stations[station_idx] if station_idx < len(dashboard.stations) else f"Station_{station_idx+1}"
        
        fig = dashboard.create_combined_time_series_plot(
            weather_time_series_data, pollution_time_series_data, 
            start_time, window_size, station_name
        )
        return dcc.Graph(figure=fig, style={'width': '100%'})
    except Exception as e:
        return html.P(f"Error creating combined time series plot: {str(e)}")


if __name__ == '__main__':
    print("Starting Weather Data Interactive Dashboard...")
    print("Dashboard will be available at: http://localhost:8050")
    print("To access from other machines, use: http://[your-ip]:8050")
    print("Press Ctrl+C to stop the server")
    print("-" * 60)
    
    try:
        app.run(
            debug=True, 
            host='0.0.0.0', 
            port=8050,
            dev_tools_hot_reload=True
        )
    except KeyboardInterrupt:
        print("\nDashboard stopped by user")
    except Exception as e:
        print(f"Error starting dashboard: {e}")
        print("Make sure all required packages are installed:")
        print("   pip install dash dash-bootstrap-components plotly numpy pandas cmocean")


# Pytest test functions
def test_interactive_dashboard_initialization() -> None:
    """Test InteractiveWeatherDashboard initialization."""
    dashboard = InteractiveWeatherDashboard()
    assert dashboard.data_folder == Path("/unity/f1/ozavala/DATA/AirPollution/TrainingData")
    assert dashboard.weather_files == []
    assert dashboard.current_weather_data is None


def test_find_weather_files() -> None:
    """Test finding weather files."""
    dashboard = InteractiveWeatherDashboard()
    files = dashboard.find_weather_files()
    assert isinstance(files, list)
    assert all(f.name.startswith("weather_data_") for f in files)
    assert all(f.suffix == ".pkl" for f in files)


def test_get_data_info() -> None:
    """Test getting data information."""
    dashboard = InteractiveWeatherDashboard()
    test_data = np.random.randn(100, 8, 25, 25).astype(np.float32)
    dashboard.current_weather_data = test_data
    info = dashboard.get_data_info("test_file.pkl")
    
    assert info["filename"] == "test_file.pkl"
    assert info["shape"] == (100, 8, 25, 25)
    assert info["time_steps"] == 100
    assert info["variables"] == 8
    assert info["spatial_dims"] == (25, 25)


def test_get_current_frame() -> None:
    """Test getting current frame data."""
    dashboard = InteractiveWeatherDashboard()
    test_data = np.random.randn(10, 8, 25, 25).astype(np.float32)
    dashboard.current_weather_data = test_data
    
    frame = dashboard.get_current_frame(0)
    assert frame.shape == (8, 25, 25)
    assert np.array_equal(frame, test_data[0, :, :, :])


def test_create_weather_variable_plot() -> None:
    """Test creating weather variable plot."""
    dashboard = InteractiveWeatherDashboard()
    test_frame = np.random.randn(8, 25, 25).astype(np.float32)
    
    fig = dashboard.create_weather_variable_plot(test_frame, 0, "T2")
    assert isinstance(fig, go.Figure)


def test_create_all_variables_plot() -> None:
    """Test creating all variables plot."""
    dashboard = InteractiveWeatherDashboard()
    test_frame = np.random.randn(8, 25, 25).astype(np.float32)
    
    fig = dashboard.create_all_variables_plot(test_frame)
    assert isinstance(fig, go.Figure)


def test_get_time_series_data() -> None:
    """Test getting time series data."""
    dashboard = InteractiveWeatherDashboard()
    test_data = np.random.randn(100, 8, 25, 25).astype(np.float32)
    dashboard.current_weather_data = test_data
    
    time_series = dashboard.get_time_series_data(0, 48)
    assert time_series.shape == (48, 8)
    assert np.allclose(time_series, np.mean(test_data[0:48, :, :, :], axis=(2, 3)))


def test_create_time_series_plot() -> None:
    """Test creating time series plot."""
    dashboard = InteractiveWeatherDashboard()
    test_time_series = np.random.randn(48, 8).astype(np.float32)
    
    fig = dashboard.create_time_series_plot(test_time_series, 0, 48)
    assert isinstance(fig, go.Figure)


def test_get_pollution_data_for_station() -> None:
    """Test getting pollution data for a station."""
    dashboard = InteractiveWeatherDashboard()
    test_pollution_data = np.random.randn(100, 5, 9).astype(np.float32)  # time, stations, pollutants
    dashboard.current_pollution_data = test_pollution_data
    
    station_data = dashboard.get_pollution_data_for_station(0)
    assert station_data.shape == (100, 9)
    assert np.array_equal(station_data, test_pollution_data[:, 0, :])


def test_get_pollution_time_series_data() -> None:
    """Test getting pollution time series data."""
    dashboard = InteractiveWeatherDashboard()
    test_pollution_data = np.random.randn(100, 5, 9).astype(np.float32)
    dashboard.current_pollution_data = test_pollution_data
    
    time_series = dashboard.get_pollution_time_series_data(0, 48, 0)
    assert time_series.shape == (48, 9)
    assert np.array_equal(time_series, test_pollution_data[0:48, 0, :])


def test_create_pollution_time_series_plot() -> None:
    """Test creating pollution time series plot."""
    dashboard = InteractiveWeatherDashboard()
    test_pollution_time_series = np.random.randn(48, 9).astype(np.float32)
    
    fig = dashboard.create_pollution_time_series_plot(test_pollution_time_series, 0, 48, "Test_Station")
    assert isinstance(fig, go.Figure)


def test_create_combined_time_series_plot() -> None:
    """Test creating combined time series plot."""
    dashboard = InteractiveWeatherDashboard()
    test_weather_time_series = np.random.randn(48, 8).astype(np.float32)
    test_pollution_time_series = np.random.randn(48, 9).astype(np.float32)
    
    fig = dashboard.create_combined_time_series_plot(
        test_weather_time_series, test_pollution_time_series, 0, 48, "Test_Station"
    )
    assert isinstance(fig, go.Figure)


def test_get_matching_file_pairs() -> None:
    """Test finding matching file pairs."""
    dashboard = InteractiveWeatherDashboard()
    # This test would require actual files, so we just test the method exists
    pairs = dashboard.get_matching_file_pairs()
    assert isinstance(pairs, list)


def test_process_pollution_dataframe() -> None:
    """Test processing pollution DataFrame."""
    dashboard = InteractiveWeatherDashboard()
    
    # Create a test DataFrame with pollution data
    import pandas as pd
    import numpy as np
    
    # Simulate pollution data with time index and pollutant columns
    dates = pd.date_range('2015-01-01', periods=100, freq='H')
    stations = ['UIZ', 'AJU', 'SFE']
    pollutants = ['co', 'nodos', 'otres', 'pmdiez', 'pmdoscinco', 'nox', 'no', 'sodos', 'pmco']
    
    # Create DataFrame with cont_pollutant_station column format
    columns = []
    for pollutant in pollutants:
        for station in stations:
            columns.append(f'cont_{pollutant}_{station}')
    
    # Create DataFrame with random pollution values
    data = np.random.randn(100, len(columns))
    df = pd.DataFrame(data, index=dates, columns=columns)
    
    # Process the DataFrame
    result = dashboard.process_pollution_dataframe(df)
    
    # Check the result
    assert isinstance(result, np.ndarray)
    assert result.shape == (100, 3, 9)  # time_steps, stations, pollutants
    assert hasattr(dashboard, 'station_names')
    assert hasattr(dashboard, 'pollutant_names')
    assert len(dashboard.station_names) == 3
    assert len(dashboard.pollutant_names) == 9 