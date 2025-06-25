
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
from sklearn.metrics import mean_squared_error
import glob

# --- CONFIG FROM ARGPARSE/CONFIG FILE ---
args = argparse.ArgumentParser(description='Air Pollution Multi-Model Comparison Dashboard')
args.add_argument('-c', '--config', default='config.json', type=str,
                  help='config file path (default: config.json)')
args.add_argument('-r', '--resume', default=None, type=str,
                  help='path to latest checkpoint (default: None)')
args.add_argument('-d', '--device', default=None, type=str,
                  help='indices of GPUs to enable (default: all)')
config = ConfigParser.from_args(args)

# Use prediction_path from config
PREDICTION_PATH = config['test']['prediction_path']

def load_all_models_data(prediction_path: str) -> Dict[str, pd.DataFrame]:
    """
    Load data from all model subfolders in the prediction path.
    Looks two levels deep and combines folder names for model identification.
    
    Args:
        prediction_path: Path to the predictions directory containing model subfolders
        
    Returns:
        Dictionary mapping model names to their concatenated DataFrames
    """
    if not os.path.exists(prediction_path):
        raise ValueError(f"Prediction path does not exist: {prediction_path}")
    
    models_data = {}
    
    # Get all first-level subdirectories
    first_level_folders = [f for f in os.listdir(prediction_path) 
                          if os.path.isdir(os.path.join(prediction_path, f))]
    
    print(f"Found {len(first_level_folders)} first-level folders: {first_level_folders}")
    
    for first_level_folder in first_level_folders:
        first_level_path = os.path.join(prediction_path, first_level_folder)
        
        # Get all second-level subdirectories
        second_level_folders = [f for f in os.listdir(first_level_path) 
                               if os.path.isdir(os.path.join(first_level_path, f))]
        
        print(f"  Found {len(second_level_folders)} second-level folders in {first_level_folder}: {second_level_folders}")
        
        for second_level_folder in second_level_folders:
            # Create combined model name
            model_name = f"{first_level_folder}_{second_level_folder}"
            model_path = os.path.join(first_level_path, second_level_folder)
            
            print(f"Loading data from model: {model_name}")
            
            try:
                # Load all CSV files from this model folder
                all_dfs = []
                csv_files = glob.glob(os.path.join(model_path, "*.csv"))
                
                if not csv_files:
                    print(f"Warning: No CSV files found in {model_path}")
                    continue
                
                for csv_file in sorted(csv_files):
                    # Extract hour from filename (e.g., "modelname_forecast_8.csv" -> 8)
                    filename = os.path.basename(csv_file)
                    if 'forecast_' in filename:
                        hour = int(filename.split('forecast_')[-1].split('.')[0])
                    else:
                        # Fallback parsing
                        hour = int(filename.split('_')[-1].split('.')[0])
                    
                    df = pd.read_csv(csv_file)
                    df['predicted_hour'] = hour
                    all_dfs.append(df)
                
                if all_dfs:
                    models_data[model_name] = pd.concat(all_dfs, ignore_index=True)
                    print(f"  Loaded {len(all_dfs)} CSV files for model {model_name}")
                else:
                    print(f"  No valid CSV files found for model {model_name}")
                    
            except Exception as e:
                print(f"Error loading model {model_name}: {e}")
                continue
    
    return models_data

def calculate_rmse_by_hour(data: pd.DataFrame, pollutant: str = 'otres') -> Dict[int, float]:
    """
    Calculate RMSE for each predicted hour for a specific pollutant.
    
    Args:
        data: DataFrame containing predictions and targets
        pollutant: Name of the pollutant to analyze
        
    Returns:
        Dictionary mapping predicted hour to RMSE value
    """
    rmse_by_hour = {}
    
    # Get all target and prediction columns for the pollutant
    target_cols = [col for col in data.columns if col.startswith(f'target_cont_{pollutant}_')]
    pred_cols = [col for col in data.columns if col.startswith(f'pred_cont_{pollutant}_')]
    
    # Remove aggregated columns (min, max, mean) from station columns
    target_cols = [col for col in target_cols 
                  if not any(x in col for x in ['min', 'max', 'mean'])]
    pred_cols = [col for col in pred_cols 
                if not any(x in col for x in ['min', 'max', 'mean'])]
    
    if not target_cols or not pred_cols:
        print(f"Warning: No target or prediction columns found for pollutant {pollutant}")
        return rmse_by_hour
    
    # Calculate RMSE for each predicted hour
    for hour in sorted(data['predicted_hour'].unique()):
        hour_data = data[data['predicted_hour'] == hour].copy()
        
        if hour_data.empty:
            continue
        
        # Calculate RMSE across all stations for this hour
        total_squared_error = 0
        total_valid_points = 0
        
        for target_col, pred_col in zip(target_cols, pred_cols):
            if target_col in hour_data.columns and pred_col in hour_data.columns:
                # Remove NaN values
                valid_mask = ~(np.isnan(hour_data[target_col]) | np.isnan(hour_data[pred_col]))
                if valid_mask.any():
                    squared_errors = (hour_data.loc[valid_mask, target_col] - 
                                    hour_data.loc[valid_mask, pred_col]) ** 2
                    total_squared_error += squared_errors.sum()
                    total_valid_points += valid_mask.sum()
        
        if total_valid_points > 0:
            rmse = np.sqrt(total_squared_error / total_valid_points)
            rmse_by_hour[hour] = rmse
        else:
            rmse_by_hour[hour] = np.nan
    
    return rmse_by_hour

def create_rmse_comparison_plot(models_data: Dict[str, pd.DataFrame], 
                               pollutant: str = 'otres') -> go.Figure:
    """
    Create a plot comparing RMSE across all models by predicted hour.
    
    Args:
        models_data: Dictionary mapping model names to their DataFrames
        pollutant: Name of the pollutant to analyze
        
    Returns:
        Plotly figure showing RMSE comparison
    """
    fig = go.Figure()
    
    # Generate colors for different models
    colors = [f'hsl({int(i * 360 / len(models_data))}, 70%, 50%)' 
              for i in range(len(models_data))]
    
    for i, (model_name, data) in enumerate(models_data.items()):
        rmse_by_hour = calculate_rmse_by_hour(data, pollutant)
        
        if rmse_by_hour:
            hours = sorted(rmse_by_hour.keys())
            rmse_values = [rmse_by_hour[hour] for hour in hours]
            
            # Remove NaN values
            valid_indices = [j for j, val in enumerate(rmse_values) if not np.isnan(val)]
            valid_hours = [hours[j] for j in valid_indices]
            valid_rmse = [rmse_values[j] for j in valid_indices]
            
            if valid_rmse:
                fig.add_trace(go.Scatter(
                    x=valid_hours,
                    y=valid_rmse,
                    mode='lines+markers',
                    name=model_name,
                    line=dict(color=colors[i], width=2),
                    marker=dict(size=6)
                ))
    
    fig.update_layout(
        title=f'{pollutant.upper()} RMSE Comparison Across Models by Predicted Hour',
        xaxis_title='Predicted Hour',
        yaxis_title='RMSE',
        legend_title='Models',
        hovermode='x unified'
    )
    
    return fig

def create_model_summary_table(models_data: Dict[str, pd.DataFrame], 
                              pollutant: str = 'otres') -> go.Figure:
    """
    Create a summary table showing overall metrics for each model.
    
    Args:
        models_data: Dictionary mapping model names to their DataFrames
        pollutant: Name of the pollutant to analyze
        
    Returns:
        Plotly figure with summary table
    """
    summary_data = []
    
    for model_name, data in models_data.items():
        rmse_by_hour = calculate_rmse_by_hour(data, pollutant)
        
        if rmse_by_hour:
            valid_rmse = [val for val in rmse_by_hour.values() if not np.isnan(val)]
            
            if valid_rmse:
                summary_data.append({
                    'Model': model_name,
                    'Mean RMSE': f"{np.mean(valid_rmse):.4f}",
                    'Std RMSE': f"{np.std(valid_rmse):.4f}",
                    'Min RMSE': f"{np.min(valid_rmse):.4f}",
                    'Max RMSE': f"{np.max(valid_rmse):.4f}",
                    'Hours Available': len(valid_rmse)
                })
    
    if not summary_data:
        # Create empty table
        fig = go.Figure(data=[go.Table(
            header=dict(values=['Model', 'Mean RMSE', 'Std RMSE', 'Min RMSE', 'Max RMSE', 'Hours Available']),
            cells=dict(values=[[], [], [], [], [], []])
        )])
    else:
        # Create table with data
        fig = go.Figure(data=[go.Table(
            header=dict(values=list(summary_data[0].keys())),
            cells=dict(values=[[row[key] for row in summary_data] for key in summary_data[0].keys()])
        )])
    
    fig.update_layout(
        title=f'{pollutant.upper()} Model Performance Summary',
        height=200 + len(summary_data) * 50
    )
    
    return fig

def create_hourly_performance_heatmap(models_data: Dict[str, pd.DataFrame], 
                                     pollutant: str = 'otres') -> go.Figure:
    """
    Create a heatmap showing RMSE performance for each model and hour.
    
    Args:
        models_data: Dictionary mapping model names to their DataFrames
        pollutant: Name of the pollutant to analyze
        
    Returns:
        Plotly figure with heatmap
    """
    # Collect all hours and models
    all_hours = set()
    model_rmse_data = {}
    
    for model_name, data in models_data.items():
        rmse_by_hour = calculate_rmse_by_hour(data, pollutant)
        model_rmse_data[model_name] = rmse_by_hour
        all_hours.update(rmse_by_hour.keys())
    
    all_hours = sorted(all_hours)
    model_names = list(models_data.keys())
    
    # Create matrix for heatmap
    heatmap_data = []
    for model_name in model_names:
        row = []
        for hour in all_hours:
            rmse = model_rmse_data[model_name].get(hour, np.nan)
            row.append(rmse if not np.isnan(rmse) else None)
        heatmap_data.append(row)
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=all_hours,
        y=model_names,
        colorscale='Viridis',
        colorbar=dict(title='RMSE'),
        text=[[f"{val:.3f}" if val is not None else "N/A" for val in row] for row in heatmap_data],
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title=f'{pollutant.upper()} RMSE Heatmap by Model and Hour',
        xaxis_title='Predicted Hour',
        yaxis_title='Models',
        height=400
    )
    
    return fig

def create_model_grouping_analysis(models_data: Dict[str, pd.DataFrame], 
                                  pollutant: str = 'otres') -> go.Figure:
    """
    Create a plot showing performance by model group (first level folder).
    
    Args:
        models_data: Dictionary mapping model names to their DataFrames
        pollutant: Name of the pollutant to analyze
        
    Returns:
        Plotly figure showing performance by model group
    """
    # Group models by their first level folder
    model_groups = {}
    for model_name in models_data.keys():
        group_name = model_name.split('_')[0]  # First part before underscore
        if group_name not in model_groups:
            model_groups[group_name] = []
        model_groups[group_name].append(model_name)
    
    fig = go.Figure()
    
    # Generate colors for different groups
    colors = [f'hsl({int(i * 360 / len(model_groups))}, 70%, 50%)' 
              for i in range(len(model_groups))]
    
    for i, (group_name, group_models) in enumerate(model_groups.items()):
        group_rmse_values = []
        
        for model_name in group_models:
            data = models_data[model_name]
            rmse_by_hour = calculate_rmse_by_hour(data, pollutant)
            if rmse_by_hour:
                valid_rmse = [val for val in rmse_by_hour.values() if not np.isnan(val)]
                if valid_rmse:
                    group_rmse_values.extend(valid_rmse)
        
        if group_rmse_values:
            fig.add_trace(go.Box(
                y=group_rmse_values,
                name=group_name,
                boxpoints='outliers',
                marker_color=colors[i]
            ))
    
    fig.update_layout(
        title=f'{pollutant.upper()} RMSE Distribution by Model Group',
        yaxis_title='RMSE',
        xaxis_title='Model Groups',
        showlegend=False
    )
    
    return fig

# %% Load data from all models
print(f"Loading data from prediction path: {PREDICTION_PATH}")
models_data = load_all_models_data(PREDICTION_PATH)

if not models_data:
    print("No model data found. Please check the prediction path.")
    exit(1)

print(f"Successfully loaded data for {len(models_data)} models")

# %% --- DASH APP ---
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1('Air Pollution Multi-Model Comparison Dashboard'),
    html.H3(f'Comparing {len(models_data)} models for OTRES pollutant'),
    
    html.Div([
        html.H4('Model Performance Summary'),
        dcc.Graph(id='summary-table')
    ], style={'marginBottom': '30px'}),
    
    html.Div([
        html.H4('RMSE Comparison by Predicted Hour'),
        dcc.Graph(id='rmse-comparison-plot')
    ], style={'marginBottom': '30px'}),
    
    html.Div([
        html.H4('RMSE Heatmap by Model and Hour'),
        dcc.Graph(id='rmse-heatmap')
    ], style={'marginBottom': '30px'}),
    
    html.Div([
        html.H4('RMSE Distribution by Model Group'),
        dcc.Graph(id='model-grouping-plot')
    ], style={'marginBottom': '30px'}),
    
    html.Div([
        html.H4('Model Information'),
        html.Div(id='model-info')
    ])
])

@app.callback(
    [Output('summary-table', 'figure'),
     Output('rmse-comparison-plot', 'figure'),
     Output('rmse-heatmap', 'figure'),
     Output('model-grouping-plot', 'figure'),
     Output('model-info', 'children')],
    [Input('summary-table', 'id')]  # Dummy input to trigger callback
)
def update_plots(dummy_input):
    """Update all plots with the loaded model data."""
    
    # Create summary table
    summary_fig = create_model_summary_table(models_data, 'otres')
    
    # Create RMSE comparison plot
    rmse_fig = create_rmse_comparison_plot(models_data, 'otres')
    
    # Create heatmap
    heatmap_fig = create_hourly_performance_heatmap(models_data, 'otres')
    
    # Create model grouping analysis
    grouping_fig = create_model_grouping_analysis(models_data, 'otres')
    
    # Create model information
    model_info = []
    for model_name, data in models_data.items():
        info = html.Div([
            html.H5(f"Model: {model_name}"),
            html.P(f"Total records: {len(data):,}"),
            html.P(f"Date range: {data['timestamp'].min()} to {data['timestamp'].max()}"),
            html.Hr()
        ])
        model_info.append(info)
    
    return summary_fig, rmse_fig, heatmap_fig, grouping_fig, model_info

if __name__ == '__main__':
    app.run(debug=True, port=8073) 