# %%
import os
import pandas as pd
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import argparse
from parse_config import ConfigParser
import numpy as np

# --- CONFIG FROM ARGPARSE/CONFIG FILE ---
args = argparse.ArgumentParser(description='Air Pollution Dashboard')
args.add_argument('-c', '--config', default='config.json', type=str,
                  help='config file path (default: config.json)')
args.add_argument('-r', '--resume', default=None, type=str,
                  help='path to latest checkpoint (default: None)')
args.add_argument('-d', '--device', default=None, type=str,
                  help='indices of GPUs to enable (default: all)')
config = ConfigParser.from_args(args)

PREDICTION_FOLDER = config['analyze']['prediction_path']

# %% Load all CSVs and concatenate into a single DataFrame with a 'predicted_hour' column
all_dfs = []
for fname in sorted(os.listdir(PREDICTION_FOLDER)):
    if fname.endswith('.csv'):
        print("Loading", fname)
        # Handle filenames like "FourPrevHoursFourAutoRegresiveStepsFourWeatherFields_forecast_8.csv"
        if 'forecast_' in fname:
            hour = int(fname.split('forecast_')[-1].split('.')[0])
        else:
            # Fallback to original parsing
            hour = int(fname.split('_')[-1].split('.')[0])
        df = pd.read_csv(os.path.join(PREDICTION_FOLDER, fname))
        df['predicted_hour'] = hour
        all_dfs.append(df)

if not all_dfs:
    raise ValueError(f"No CSV files found in {PREDICTION_FOLDER}")

data = pd.concat(all_dfs, ignore_index=True)

# %%
# Get pollutants
stations = list(set(col.split('_')[-1] for col in data.columns if col.startswith('target_cont_')))
pollutants = list(set(col.split('_')[-2] for col in data.columns if col.startswith('target_cont_')))
print(f"Stations: {stations}")
print(f"Pollutants: {pollutants}")

# Get time range
min_hour = data['predicted_hour'].min()
max_hour = data['predicted_hour'].max()

# Get timestamps (assume all CSVs have the same timestamps)
timestamps = pd.to_datetime(data['timestamp'].unique())

# %% --- DASH APP ---
app = dash.Dash(__name__)

# Get unique forecast hours for the dropdown
forecast_hours = sorted(data['predicted_hour'].unique())

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
                options=[{'label': s, 'value': s} for s in stations],
                value='MER' if 'MER' in stations else (stations[0] if stations else None),
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

    html.Div(id='plots-container', style={'marginTop': '20px'})
])

@app.callback(
    Output('plots-container', 'children'),
    [Input('pollutant-dropdown', 'value'),
     Input('station-dropdown', 'value'),
     Input('time-slider', 'value'),
     Input('window-size-slider', 'value')]
)
def update_plots(selected_pollutant, selected_station, start_index, window_size):
    if not all([selected_pollutant, selected_station]):
        return []

    # Define the window from the sliders
    start_time = timestamps[start_index]
    end_time = start_time + pd.Timedelta(hours=window_size - 1)
    
    # Get all station columns for the selected pollutant
    true_cols_all_stations = [col for col in data.columns if col.startswith(f'target_cont_{selected_pollutant}_')]
    pred_cols_all_stations = [col for col in data.columns if col.startswith(f'pred_cont_{selected_pollutant}_')]
    
    figures = []
    
    # Figure 1: Original plot (selected station + mean/max all stations)
    fig1 = go.Figure()

    # Plot True Values for the selected window
    true_col = f'target_cont_{selected_pollutant}_{selected_station}'
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
    pred_col = f'pred_cont_{selected_pollutant}_{selected_station}'
    if pred_col in data.columns:
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
                fig1.add_trace(go.Scatter(
                    x=df_window['target_time'],
                    y=df_window[pred_col],
                    mode='lines',
                    name=f'Forecast Hour {hour}',
                    line=dict(color=colors[i], width=1)
                ))

    fig1.update_layout(
        title=f'{selected_pollutant.upper()} at {selected_station} - All Forecasts vs. True',
        xaxis_title='Timestamp',
        yaxis_title='Concentration',
        legend_title='Forecast Hour',
        height=600
    )
    
    figures.append(dcc.Graph(figure=fig1))
    
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
        title=f'{selected_pollutant.upper()} - Max Values Across All Stations',
        xaxis_title='Timestamp',
        yaxis_title='Max Concentration',
        legend_title='Forecast Hour',
        height=600
    )
    
    figures.append(dcc.Graph(figure=fig2))
    
    # Figure 3: Min values across all stations
    fig3 = go.Figure()
    
    # Plot true min across all stations
    if true_cols_all_stations:
        df_true_source_all = data[data['predicted_hour'] == forecast_hours[0]].copy()
        df_true_source_all['timestamp'] = pd.to_datetime(df_true_source_all['timestamp'])
        
        df_true_window_all = df_true_source_all[
            (df_true_source_all['timestamp'] >= start_time) & (df_true_source_all['timestamp'] <= end_time)
        ]
        
        df_true_window_all['min_all_stations'] = df_true_window_all[true_cols_all_stations].min(axis=1)
        
        fig3.add_trace(go.Scatter(
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
            time_shift = pd.Timedelta(hours=-hour)
            df_forecast['target_time'] = df_forecast['timestamp'] + time_shift
            
            # Filter based on the target_time being in the window
            df_window = df_forecast[
                (df_forecast['target_time'] >= start_time) & (df_forecast['target_time'] <= end_time)
            ]
            
            if not df_window.empty:
                # Calculate min across all stations for this forecast hour
                df_window['pred_min_all_stations'] = df_window[pred_cols_all_stations].min(axis=1)
                
                fig3.add_trace(go.Scatter(
                    x=df_window['target_time'],
                    y=df_window['pred_min_all_stations'],
                    mode='lines',
                    name=f'Pred Min Hour {hour}',
                    line=dict(color=colors[i], width=1)
                ))
    
    fig3.update_layout(
        title=f'{selected_pollutant.upper()} - Min Values Across All Stations',
        xaxis_title='Timestamp',
        yaxis_title='Min Concentration',
        legend_title='Forecast Hour',
        height=600
    )
    
    figures.append(dcc.Graph(figure=fig3))
    
    return figures

@app.callback(
    [Output('time-slider', 'max'),
     Output('time-slider-label', 'children')],
    [Input('window-size-slider', 'value')]
)
def update_time_slider_properties(window_size):
    max_val = len(timestamps) - window_size
    label = f'Select Start Time ({window_size}-hour window)'
    return max_val, label

if __name__ == '__main__':
    app.run(debug=True)
