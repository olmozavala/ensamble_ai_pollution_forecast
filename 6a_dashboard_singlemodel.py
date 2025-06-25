# %%
import os
import pandas as pd
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import argparse
from parse_config import ConfigParser
import numpy as np
from typing import List, Dict, Optional, Tuple

# --- CONFIG FROM ARGPARSE/CONFIG FILE ---
args = argparse.ArgumentParser(description='Air Pollution Dashboard')
args.add_argument('-c', '--config', default='config.json', type=str,
                  help='config file path (default: config.json)')
args.add_argument('-r', '--resume', default=None, type=str,
                  help='path to latest checkpoint (default: None)')
args.add_argument('-d', '--device', default=None, type=str,
                  help='indices of GPUs to enable (default: all)')
config = ConfigParser.from_args(args)

# Use single_model_path instead of prediction_path
SINGLE_MODEL_PATH = config['analyze']['single_model_path']

def load_data_from_single_model(model_path: str) -> pd.DataFrame:
    """
    Load all CSV files from a single model folder and concatenate into a DataFrame.
    
    Args:
        model_path: Path to the single model folder containing CSV files
        
    Returns:
        DataFrame with all data concatenated and 'predicted_hour' column added
    """
    if not os.path.exists(model_path):
        raise ValueError(f"Model path does not exist: {model_path}")
    
    all_dfs = []
    csv_files = [f for f in os.listdir(model_path) if f.endswith('.csv')]
    
    if not csv_files:
        raise ValueError(f"No CSV files found in {model_path}")
    
    for fname in sorted(csv_files):
        print(f"Loading {fname}")
        # Handle filenames like "FourPrevHoursFourAutoRegresiveStepsFourWeatherFields_forecast_8.csv"
        if 'forecast_' in fname:
            hour = int(fname.split('forecast_')[-1].split('.')[0])
        else:
            # Fallback to original parsing
            hour = int(fname.split('_')[-1].split('.')[0])
        df = pd.read_csv(os.path.join(model_path, fname))
        df['predicted_hour'] = hour
        all_dfs.append(df)
    
    return pd.concat(all_dfs, ignore_index=True)

def analyze_data_availability(data: pd.DataFrame, pollutant: str) -> Dict[str, bool]:
    """
    Analyze what data is available for a given pollutant.
    
    Args:
        data: DataFrame containing the prediction data
        pollutant: Name of the pollutant to analyze
        
    Returns:
        Dictionary indicating what data is available
    """
    # Check for individual station data
    target_station_cols = [col for col in data.columns if col.startswith(f'target_cont_{pollutant}_')]
    pred_station_cols = [col for col in data.columns if col.startswith(f'pred_cont_{pollutant}_')]

    # Remove aggregated columns (min, max, mean) from station columns
    target_station_cols = [col for col in target_station_cols 
                          if not any(x in col for x in ['min', 'max', 'mean'])]
    pred_station_cols = [col for col in pred_station_cols 
                        if not any(x in col for x in ['min', 'max', 'mean'])]
    
    # Check for aggregated data columns
    has_cont_min = f'pred_cont_{pollutant}_min' in data.columns
    has_cont_max = f'pred_cont_{pollutant}_max' in data.columns
    has_cont_mean = f'pred_cont_{pollutant}_mean' in data.columns
    
    return {
        'has_multiple_stations': len(target_station_cols) > 1,
        'has_individual_stations': len(target_station_cols) > 0,
        'stations': [col.split('_')[-1] for col in target_station_cols],
        'has_cont_min': has_cont_min,
        'has_cont_max': has_cont_max,
        'has_cont_mean': has_cont_mean,
        'target_station_cols': target_station_cols,
        'pred_station_cols': pred_station_cols
    }

def create_station_based_plots(data: pd.DataFrame, pollutant: str, selected_station: str, 
                              start_time: pd.Timestamp, end_time: pd.Timestamp, 
                              forecast_hours: List[int]) -> List[go.Figure]:
    """
    Create the original 4 plots when we have multiple stations.
    
    Args:
        data: DataFrame containing the prediction data
        pollutant: Name of the pollutant
        selected_station: Selected station name
        start_time: Start time for the window
        end_time: End time for the window
        forecast_hours: List of forecast hours
        
    Returns:
        List of 4 plotly figures
    """
    # Get all station columns for the selected pollutant
    true_cols_all_stations = [col for col in data.columns if col.startswith(f'target_cont_{pollutant}_')]
    pred_cols_all_stations = [col for col in data.columns if col.startswith(f'pred_cont_{pollutant}_')]
    
    figures = []
    
    # Figure 1: Original plot (selected station + mean/max all stations)
    fig1 = go.Figure()

    # Plot True Values for the selected window
    true_col = f'target_cont_{pollutant}_{selected_station}'
    if true_col in data.columns:
        # Use the first forecast hour's data as the source for true values
        df_true_source = data[data['predicted_hour'] == forecast_hours[0]].copy()
        df_true_source['timestamp'] = pd.to_datetime(df_true_source['timestamp'])
        
        # The true values are for the timestamp itself, no shift needed
        df_true_window = df_true_source[
            (df_true_source['timestamp'] >= start_time) & (df_true_source['timestamp'] <= end_time)
        ]
        
        fig1.add_trace(go.Scatter(
            x=df_true_window['timestamp'],
            y=df_true_window[true_col],
            mode='lines',
            name='True Value',
            line=dict(color='black', width=2, dash='dash')
        ))

    # Calculate mean and max across all stations for the selected pollutant
    if true_cols_all_stations:
        df_true_source_all = data[data['predicted_hour'] == forecast_hours[0]].copy()
        df_true_source_all['timestamp'] = pd.to_datetime(df_true_source_all['timestamp'])
        
        # Filter for the time window
        df_true_window_all = df_true_source_all[
            (df_true_source_all['timestamp'] >= start_time) & (df_true_source_all['timestamp'] <= end_time)
        ]
        
        # Calculate mean and max across all stations
        df_true_window_all['mean_all_stations'] = df_true_window_all[true_cols_all_stations].mean(axis=1)
        df_true_window_all['max_all_stations'] = df_true_window_all[true_cols_all_stations].max(axis=1)
        
        # Plot mean across all stations
        fig1.add_trace(go.Scatter(
            x=df_true_window_all['timestamp'],
            y=df_true_window_all['mean_all_stations'],
            mode='lines',
            name='Mean (All Stations)',
            line=dict(color='blue', width=2, dash='dash')
        ))
        
        # Plot max across all stations
        fig1.add_trace(go.Scatter(
            x=df_true_window_all['timestamp'],
            y=df_true_window_all['max_all_stations'],
            mode='lines',
            name='Max (All Stations)',
            line=dict(color='red', width=2, dash='dash')
        ))

    # Plot Predicted Values for all forecast hours, shifted to align
    pred_col = f'pred_cont_{pollutant}_{selected_station}'
    if pred_col in data.columns:
        colors = [f'hsl({int(h)}, 80%, 50%)' for h in np.linspace(0, 330, len(forecast_hours))]

        for i, hour in enumerate(forecast_hours):
            df_forecast = data[data['predicted_hour'] == hour].copy()
            df_forecast['timestamp'] = pd.to_datetime(df_forecast['timestamp'])

            # Shift timestamp to get the actual time the forecast is for
            time_shift = pd.Timedelta(hours=-hour + 1)
            df_forecast['target_time'] = df_forecast['timestamp'] + time_shift

            # Filter based on the target_time being in the window
            df_window = df_forecast[
                (df_forecast['target_time'] >= start_time) & (df_forecast['target_time'] <= end_time)
            ]

            if not df_window.empty:
                fig1.add_trace(go.Scatter(
                    x=df_window['target_time'],
                    y=df_window[pred_col],
                    mode='lines',
                    name=f'Forecast Hour {hour}',
                    line=dict(color=colors[i], width=1)
                ))

    fig1.update_layout(
        title=f'{pollutant.upper()} at {selected_station} - All Forecasts vs. True',
        xaxis_title='Timestamp',
        yaxis_title='Concentration',
        legend_title='Forecast Hour'
    )
    
    figures.append(fig1)
    
    # Figure 2: Max values across all stations
    fig2 = go.Figure()
    
    # Plot true max across all stations
    if true_cols_all_stations:
        df_true_source_all = data[data['predicted_hour'] == forecast_hours[0]].copy()
        df_true_source_all['timestamp'] = pd.to_datetime(df_true_source_all['timestamp'])
        
        df_true_window_all = df_true_source_all[
            (df_true_source_all['timestamp'] >= start_time) & (df_true_source_all['timestamp'] <= end_time)
        ]
        
        df_true_window_all['max_all_stations'] = df_true_window_all[true_cols_all_stations].max(axis=1)
        
        fig2.add_trace(go.Scatter(
            x=df_true_window_all['timestamp'],
            y=df_true_window_all['max_all_stations'],
            mode='lines',
            name='True Max (All Stations)',
            line=dict(color='black', width=3, dash='dash')
        ))
    
    # Plot predicted max across all stations for each forecast hour
    if pred_cols_all_stations:
        colors = [f'hsl({int(h)}, 80%, 50%)' for h in np.linspace(0, 330, len(forecast_hours))]
        
        for i, hour in enumerate(forecast_hours):
            df_forecast = data[data['predicted_hour'] == hour].copy()
            df_forecast['timestamp'] = pd.to_datetime(df_forecast['timestamp'])
            
            # Shift timestamp to get the actual time the forecast is for
            time_shift = pd.Timedelta(hours=-hour)
            df_forecast['target_time'] = df_forecast['timestamp'] + time_shift
            
            # Filter based on the target_time being in the window
            df_window = df_forecast[
                (df_forecast['target_time'] >= start_time) & (df_forecast['target_time'] <= end_time)
            ]
            
            if not df_window.empty:
                # Calculate max across all stations for this forecast hour
                df_window['pred_max_all_stations'] = df_window[pred_cols_all_stations].max(axis=1)
                
                fig2.add_trace(go.Scatter(
                    x=df_window['target_time'],
                    y=df_window['pred_max_all_stations'],
                    mode='lines',
                    name=f'Pred Max Hour {hour}',
                    line=dict(color=colors[i], width=1)
                ))
    
    fig2.update_layout(
        title=f'{pollutant.upper()} - Max Values Across All Stations',
        xaxis_title='Timestamp',
        yaxis_title='Max Concentration',
        legend_title='Forecast Hour'
    )
    
    figures.append(fig2)
    
    # Figure 3: Mean values across all stations
    fig3 = go.Figure()
    
    # Plot true mean across all stations
    if true_cols_all_stations:
        df_true_source_all = data[data['predicted_hour'] == forecast_hours[0]].copy()
        df_true_source_all['timestamp'] = pd.to_datetime(df_true_source_all['timestamp'])
        
        df_true_window_all = df_true_source_all[
            (df_true_source_all['timestamp'] >= start_time) & (df_true_source_all['timestamp'] <= end_time)
        ]
        
        df_true_window_all['mean_all_stations'] = df_true_window_all[true_cols_all_stations].mean(axis=1)
        
        fig3.add_trace(go.Scatter(
            x=df_true_window_all['timestamp'],
            y=df_true_window_all['mean_all_stations'],
            mode='lines',
            name='True Mean (All Stations)',
            line=dict(color='black', width=3, dash='dash')
        ))
    
    # Plot predicted mean across all stations for each forecast hour
    if pred_cols_all_stations:
        colors = [f'hsl({int(h)}, 80%, 50%)' for h in np.linspace(0, 330, len(forecast_hours))]
        
        for i, hour in enumerate(forecast_hours):
            df_forecast = data[data['predicted_hour'] == hour].copy()
            df_forecast['timestamp'] = pd.to_datetime(df_forecast['timestamp'])
            
            # Shift timestamp to get the actual time the forecast is for
            time_shift = pd.Timedelta(hours=-hour + 1)
            df_forecast['target_time'] = df_forecast['timestamp'] + time_shift
            
            # Filter based on the target_time being in the window
            df_window = df_forecast[
                (df_forecast['target_time'] >= start_time) & (df_forecast['target_time'] <= end_time)
            ]
            
            if not df_window.empty:
                # Calculate mean across all stations for this forecast hour
                df_window['pred_mean_all_stations'] = df_window[pred_cols_all_stations].mean(axis=1)
                
                fig3.add_trace(go.Scatter(
                    x=df_window['target_time'],
                    y=df_window['pred_mean_all_stations'],
                    mode='lines',
                    name=f'Pred Mean Hour {hour}',
                    line=dict(color=colors[i], width=1)
                ))
    
    fig3.update_layout(
        title=f'{pollutant.upper()} - Mean Values Across All Stations',
        xaxis_title='Timestamp',
        yaxis_title='Mean Concentration',
        legend_title='Forecast Hour'
    )
    
    figures.append(fig3)
    
    # Figure 4: Min values across all stations
    fig4 = go.Figure()
    
    # Plot true min across all stations
    if true_cols_all_stations:
        df_true_source_all = data[data['predicted_hour'] == forecast_hours[0]].copy()
        df_true_source_all['timestamp'] = pd.to_datetime(df_true_source_all['timestamp'])
        
        df_true_window_all = df_true_source_all[
            (df_true_source_all['timestamp'] >= start_time) & (df_true_source_all['timestamp'] <= end_time)
        ]
        
        df_true_window_all['min_all_stations'] = df_true_window_all[true_cols_all_stations].min(axis=1)
        
        fig4.add_trace(go.Scatter(
            x=df_true_window_all['timestamp'],
            y=df_true_window_all['min_all_stations'],
            mode='lines',
            name='True Min (All Stations)',
            line=dict(color='black', width=3, dash='dash')
        ))
    
    # Plot predicted min across all stations for each forecast hour
    if pred_cols_all_stations:
        colors = [f'hsl({int(h)}, 80%, 50%)' for h in np.linspace(0, 330, len(forecast_hours))]
        
        for i, hour in enumerate(forecast_hours):
            df_forecast = data[data['predicted_hour'] == hour].copy()
            df_forecast['timestamp'] = pd.to_datetime(df_forecast['timestamp'])
            
            # Shift timestamp to get the actual time the forecast is for
            time_shift = pd.Timedelta(hours=-hour + 1)
            df_forecast['target_time'] = df_forecast['timestamp'] + time_shift
            
            # Filter based on the target_time being in the window
            df_window = df_forecast[
                (df_forecast['target_time'] >= start_time) & (df_forecast['target_time'] <= end_time)
            ]
            
            if not df_window.empty:
                # Calculate min across all stations for this forecast hour
                df_window['pred_min_all_stations'] = df_window[pred_cols_all_stations].min(axis=1)
                
                fig4.add_trace(go.Scatter(
                    x=df_window['target_time'],
                    y=df_window['pred_min_all_stations'],
                    mode='lines',
                    name=f'Pred Min Hour {hour}',
                    line=dict(color=colors[i], width=1)
                ))
    
    fig4.update_layout(
        title=f'{pollutant.upper()} - Min Values Across All Stations',
        xaxis_title='Timestamp',
        yaxis_title='Min Concentration',
        legend_title='Forecast Hour'
    )
    
    figures.append(fig4)
    
    return figures

def create_aggregated_plots(data: pd.DataFrame, pollutant: str, start_time: pd.Timestamp, 
                           end_time: pd.Timestamp, forecast_hours: List[int], 
                           availability: Dict[str, bool]) -> List[go.Figure]:
    """
    Create plots based on available aggregated data (cont_pollutant_min, max, mean).
    
    Args:
        data: DataFrame containing the prediction data
        pollutant: Name of the pollutant
        start_time: Start time for the window
        end_time: End time for the window
        forecast_hours: List of forecast hours
        availability: Dictionary indicating what data is available
        
    Returns:
        List of plotly figures based on available data
    """
    figures = []
    colors = [f'hsl({int(h)}, 80%, 50%)' for h in np.linspace(0, 330, len(forecast_hours))]
    
    # Create plots for each available aggregated metric
    if availability['has_cont_mean']:
        fig_mean = go.Figure()
        
        # Plot true mean
        df_true_source = data[data['predicted_hour'] == forecast_hours[0]].copy()
        df_true_source['timestamp'] = pd.to_datetime(df_true_source['timestamp'])
        
        df_true_window = df_true_source[
            (df_true_source['timestamp'] >= start_time) & (df_true_source['timestamp'] <= end_time)
        ]
        
        if f'pred_cont_{pollutant}_mean' in df_true_window.columns:
            fig_mean.add_trace(go.Scatter(
                x=df_true_window['timestamp'],
                y=df_true_window[f'pred_cont_{pollutant}_mean'],
                mode='lines',
                name='True Mean',
                line=dict(color='black', width=3, dash='dash')
            ))
        
        # Plot predicted mean for each forecast hour
        for i, hour in enumerate(forecast_hours):
            df_forecast = data[data['predicted_hour'] == hour].copy()
            df_forecast['timestamp'] = pd.to_datetime(df_forecast['timestamp'])
            
            time_shift = pd.Timedelta(hours=-hour + 1)
            df_forecast['target_time'] = df_forecast['timestamp'] + time_shift
            
            df_window = df_forecast[
                (df_forecast['target_time'] >= start_time) & (df_forecast['target_time'] <= end_time)
            ]
            
            if not df_window.empty and f'pred_cont_{pollutant}_mean' in df_window.columns:
                fig_mean.add_trace(go.Scatter(
                    x=df_window['target_time'],
                    y=df_window[f'pred_cont_{pollutant}_mean'],
                    mode='lines',
                    name=f'Pred Mean Hour {hour}',
                    line=dict(color=colors[i], width=1)
                ))
        
        fig_mean.update_layout(
            title=f'{pollutant.upper()} - Mean Values',
            xaxis_title='Timestamp',
            yaxis_title='Mean Concentration',
            legend_title='Forecast Hour'
        )
        
        figures.append(fig_mean)
    
    if availability['has_cont_max']:
        fig_max = go.Figure()
        
        # Plot true max
        df_true_source = data[data['predicted_hour'] == forecast_hours[0]].copy()
        df_true_source['timestamp'] = pd.to_datetime(df_true_source['timestamp'])
        
        df_true_window = df_true_source[
            (df_true_source['timestamp'] >= start_time) & (df_true_source['timestamp'] <= end_time)
        ]
        
        if f'pred_cont_{pollutant}_max' in df_true_window.columns:
            fig_max.add_trace(go.Scatter(
                x=df_true_window['timestamp'],
                y=df_true_window[f'pred_cont_{pollutant}_max'],
                mode='lines',
                name='True Max',
                line=dict(color='black', width=3, dash='dash')
            ))
        
        # Plot predicted max for each forecast hour
        for i, hour in enumerate(forecast_hours):
            df_forecast = data[data['predicted_hour'] == hour].copy()
            df_forecast['timestamp'] = pd.to_datetime(df_forecast['timestamp'])
            
            time_shift = pd.Timedelta(hours=-hour + 1)
            df_forecast['target_time'] = df_forecast['timestamp'] + time_shift
            
            df_window = df_forecast[
                (df_forecast['target_time'] >= start_time) & (df_forecast['target_time'] <= end_time)
            ]
            
            if not df_window.empty and f'pred_cont_{pollutant}_max' in df_window.columns:
                fig_max.add_trace(go.Scatter(
                    x=df_window['target_time'],
                    y=df_window[f'pred_cont_{pollutant}_max'],
                    mode='lines',
                    name=f'Pred Max Hour {hour}',
                    line=dict(color=colors[i], width=1)
                ))
        
        fig_max.update_layout(
            title=f'{pollutant.upper()} - Max Values',
            xaxis_title='Timestamp',
            yaxis_title='Max Concentration',
            legend_title='Forecast Hour'
        )
        
        figures.append(fig_max)
    
    if availability['has_cont_min']:
        fig_min = go.Figure()
        
        # Plot true min
        df_true_source = data[data['predicted_hour'] == forecast_hours[0]].copy()
        df_true_source['timestamp'] = pd.to_datetime(df_true_source['timestamp'])
        
        df_true_window = df_true_source[
            (df_true_source['timestamp'] >= start_time) & (df_true_source['timestamp'] <= end_time)
        ]
        
        if f'pred_cont_{pollutant}_min' in df_true_window.columns:
            fig_min.add_trace(go.Scatter(
                x=df_true_window['timestamp'],
                y=df_true_window[f'pred_cont_{pollutant}_min'],
                mode='lines',
                name='True Min',
                line=dict(color='black', width=3, dash='dash')
            ))
        
        # Plot predicted min for each forecast hour
        for i, hour in enumerate(forecast_hours):
            df_forecast = data[data['predicted_hour'] == hour].copy()
            df_forecast['timestamp'] = pd.to_datetime(df_forecast['timestamp'])
            
            time_shift = pd.Timedelta(hours=-hour + 1)
            df_forecast['target_time'] = df_forecast['timestamp'] + time_shift
            
            df_window = df_forecast[
                (df_forecast['target_time'] >= start_time) & (df_forecast['target_time'] <= end_time)
            ]
            
            if not df_window.empty and f'cont_{pollutant}_min' in df_window.columns:
                fig_min.add_trace(go.Scatter(
                    x=df_window['target_time'],
                    y=df_window[f'cont_{pollutant}_min'],
                    mode='lines',
                    name=f'Pred Min Hour {hour}',
                    line=dict(color=colors[i], width=1)
                ))
        
        fig_min.update_layout(
            title=f'{pollutant.upper()} - Min Values',
            xaxis_title='Timestamp',
            yaxis_title='Min Concentration',
            legend_title='Forecast Hour'
        )
        
        figures.append(fig_min)
    
    return figures

# %% Load data from single model path
print(f"Loading data from: {SINGLE_MODEL_PATH}")
data = load_data_from_single_model(SINGLE_MODEL_PATH)

# %%
# Get pollutants and analyze data availability
pollutants = list(set(col.split('_')[-2] for col in data.columns if col.startswith('target_cont_')))
print(f"Available pollutants: {pollutants}")

# Analyze data availability for each pollutant
pollutant_availability = {}
for pollutant in pollutants:
    pollutant_availability[pollutant] = analyze_data_availability(data, pollutant)
    print(f"\nPollutant {pollutant} availability:")
    for key, value in pollutant_availability[pollutant].items():
        if key not in ['target_station_cols', 'pred_station_cols']:
            print(f"  {key}: {value}")

# Get time range
min_hour = data['predicted_hour'].min()
max_hour = data['predicted_hour'].max()

# Get timestamps (assume all CSVs have the same timestamps)
timestamps = pd.to_datetime(data['timestamp'].unique())

# %% --- DASH APP ---
app = dash.Dash(__name__)

# Get unique forecast hours for the dropdown
forecast_hours = sorted(data['predicted_hour'].unique())

# Create layout with conditional station dropdown
app.layout = html.Div([
    html.H1('Air Pollution Prediction Dashboard'),
    html.Div([
        html.Div([
            html.Label('Pollutant:'),
            dcc.Dropdown(
                id='pollutant-dropdown',
                options=[{'label': p, 'value': p} for p in pollutants],
                value='otres' if 'otres' in pollutants else (pollutants[0] if pollutants else None),
                clearable=False
            ),
        ], style={'width': '49%', 'display': 'inline-block'}),
        html.Div([
            html.Label('Station:'),
            dcc.Dropdown(
                id='station-dropdown',
                options=[],  # Will be populated based on selected pollutant
                value=None,
                clearable=False
            ),
        ], style={'width': '49%', 'display': 'inline-block', 'float': 'right'}),
    ], style={'width': '80%', 'margin': 'auto'}),
    
    html.Div([
        html.Label(id='time-slider-label', children='Select Start Time (24-hour window)'),
        dcc.Slider(
            id='time-slider',
            min=0,
            max=len(timestamps) - 24,  # Initial max, will be updated by callback
            step=1,
            value=0,
            marks=None,
            tooltip={"placement": "bottom", "always_visible": True}
        )
    ], style={'width': '80%', 'margin': 'auto', 'paddingTop': '20px'}),

    html.Div([
        html.Label('Select Window Size (hours)'),
        dcc.Slider(
            id='window-size-slider',
            min=24,
            max=24 * 7,
            step=12,
            value=24,
            marks={i: f'{i}h' for i in range(24, 24 * 7 + 1, 24)},
            tooltip={"placement": "bottom", "always_visible": True},
        )
    ], style={'width': '80%', 'margin': 'auto', 'paddingTop': '20px'}),

    html.Div(id='plots-container', style={
        'marginTop': '20px',
        'display': 'grid',
        'gridTemplateColumns': '1fr 1fr',
        'gap': '20px',
        'width': '100%',
        'marginLeft': 'auto',
        'marginRight': 'auto'
    })
])

@app.callback(
    Output('station-dropdown', 'options'),
    Output('station-dropdown', 'value'),
    [Input('pollutant-dropdown', 'value')]
)
def update_station_dropdown(selected_pollutant):
    """Update station dropdown based on selected pollutant."""
    if not selected_pollutant:
        return [], None
    
    availability = pollutant_availability.get(selected_pollutant, {})
    
    if availability.get('has_multiple_stations', False):
        # We have multiple stations, show station dropdown
        stations = availability.get('stations', [])
        options = [{'label': s, 'value': s} for s in stations]
        value = stations[0] if stations else None
        return options, value
    else:
        # No multiple stations, hide station dropdown
        return [], None

@app.callback(
    Output('plots-container', 'children'),
    [Input('pollutant-dropdown', 'value'),
     Input('station-dropdown', 'value'),
     Input('time-slider', 'value'),
     Input('window-size-slider', 'value')]
)
def update_plots(selected_pollutant, selected_station, start_index, window_size):
    """Update plots based on selected parameters and data availability."""
    if not selected_pollutant:
        return []

    # Define the window from the sliders
    start_time = timestamps[start_index]
    end_time = start_time + pd.Timedelta(hours=window_size - 1)
    
    availability = pollutant_availability.get(selected_pollutant, {})
    
    # Determine which type of plots to create
    if availability.get('has_multiple_stations', False) and selected_station:
        # Create station-based plots (original 4 plots)
        figures = create_station_based_plots(data, selected_pollutant, selected_station, 
                                           start_time, end_time, forecast_hours)
    else:
        # Create aggregated plots based on available data
        figures = create_aggregated_plots(data, selected_pollutant, start_time, 
                                        end_time, forecast_hours, availability)
    
    # Convert figures to Dash Graph components
    return [dcc.Graph(figure=fig) for fig in figures]

@app.callback(
    [Output('time-slider', 'max'),
     Output('time-slider-label', 'children')],
    [Input('window-size-slider', 'value')]
)
def update_time_slider_properties(window_size):
    """Update time slider properties based on window size."""
    max_val = len(timestamps) - window_size
    label = f'Select Start Time ({window_size}-hour window)'
    return max_val, label

if __name__ == '__main__':
    app.run(debug=True, port=8072)
