# %%
import os
import pandas as pd
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
from plotly.subplots import make_subplots
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

def create_model_id_mapping(models_data: Dict[str, pd.DataFrame]) -> Dict[str, str]:
    """
    Create a mapping from model names to shorter IDs.
    
    Args:
        models_data: Dictionary mapping model names to their DataFrames
        
    Returns:
        Dictionary mapping model names to their IDs
    """
    model_names = list(models_data.keys())
    model_id_mapping = {}
    
    for i, model_name in enumerate(model_names, 1):
        model_id_mapping[model_name] = f"M{i:02d}"
    
    return model_id_mapping

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
                               model_id_mapping: Dict[str, str],
                               pollutant: str = 'otres') -> go.Figure:
    """
    Create a plot comparing RMSE across all models by predicted hour.
    
    Args:
        models_data: Dictionary mapping model names to their DataFrames
        model_id_mapping: Dictionary mapping model names to their IDs
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
                model_id = model_id_mapping[model_name]
                fig.add_trace(go.Scatter(
                    x=valid_hours,
                    y=valid_rmse,
                    mode='lines+markers',
                    name=model_id,
                    line=dict(color=colors[i], width=2),
                    marker=dict(size=6),
                    hovertemplate=f'<b>{model_id}</b><br>' +
                                 f'Model: {model_name}<br>' +
                                 'Hour: %{x}<br>' +
                                 'RMSE: %{y:.4f}<extra></extra>'
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
                              model_id_mapping: Dict[str, str],
                              pollutant: str = 'otres') -> go.Figure:
    """
    Create a summary table showing overall metrics for each model.
    
    Args:
        models_data: Dictionary mapping model names to their DataFrames
        model_id_mapping: Dictionary mapping model names to their IDs
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
                model_id = model_id_mapping[model_name]
                summary_data.append({
                    'Model ID': model_id,
                    'Model Name': model_name,
                    'Mean RMSE': f"{np.mean(valid_rmse):.4f}",
                    'Std RMSE': f"{np.std(valid_rmse):.4f}",
                    'Min RMSE': f"{np.min(valid_rmse):.4f}",
                    'Max RMSE': f"{np.max(valid_rmse):.4f}",
                    'Hours Available': len(valid_rmse)
                })
    
    if not summary_data:
        # Create empty table
        fig = go.Figure(data=[go.Table(
            header=dict(values=['Model ID', 'Model Name', 'Mean RMSE', 'Std RMSE', 'Min RMSE', 'Max RMSE', 'Hours Available']),
            cells=dict(values=[[], [], [], [], [], [], []]),
            columnwidth=[0.1, 0.3, 0.1, 0.1, 0.1, 0.1, 0.1]
        )])
    else:
        # Create table with data
        fig = go.Figure(data=[go.Table(
            header=dict(values=list(summary_data[0].keys())),
            cells=dict(values=[[row[key] for row in summary_data] for key in summary_data[0].keys()]),
            columnwidth=[0.1, 0.3, 0.1, 0.1, 0.1, 0.1, 0.1]
        )])
    
    fig.update_layout(
        title=f'{pollutant.upper()} Model Performance Summary',
        height=200 + len(summary_data) * 50
    )
    
    return fig

def create_model_grouping_analysis(models_data: Dict[str, pd.DataFrame], 
                                  model_id_mapping: Dict[str, str],
                                  pollutant: str = 'otres') -> go.Figure:
    """
    Create a plot showing performance by individual model.
    
    Args:
        models_data: Dictionary mapping model names to their DataFrames
        model_id_mapping: Dictionary mapping model names to their IDs
        pollutant: Name of the pollutant to analyze
        
    Returns:
        Plotly figure showing performance by individual model
    """
    fig = go.Figure()
    
    # Generate colors for different models
    colors = [f'hsl({int(i * 360 / len(models_data))}, 70%, 50%)' 
              for i in range(len(models_data))]
    
    for i, (model_name, data) in enumerate(models_data.items()):
        rmse_by_hour = calculate_rmse_by_hour(data, pollutant)
        if rmse_by_hour:
            valid_rmse = [val for val in rmse_by_hour.values() if not np.isnan(val)]
            if valid_rmse:
                model_id = model_id_mapping[model_name]
                fig.add_trace(go.Box(
                    y=valid_rmse,
                    name=model_id,
                    boxpoints='outliers',
                    marker_color=colors[i],
                    hovertemplate=f'<b>{model_id}</b><br>' +
                                 f'Model: {model_name}<br>' +
                                 'RMSE: %{y:.4f}<extra></extra>'
                ))
    
    fig.update_layout(
        title=f'{pollutant.upper()} RMSE Distribution by Individual Model',
        yaxis_title='RMSE',
        xaxis_title='Models',
        showlegend=False
    )
    
    return fig

def calculate_confusion_matrix_for_threshold(data: pd.DataFrame, threshold: int, 
                                           pollutant: str = 'otres') -> Tuple[int, int, int, int]:
    """
    Calculate confusion matrix for a specific ozone threshold.
    
    Args:
        data: DataFrame containing predictions and targets
        threshold: Ozone threshold in ppbs
        pollutant: Name of the pollutant to analyze
        
    Returns:
        Tuple of (TP, FP, TN, FN) values
    """
    # Get target and prediction columns for the pollutant
    target_cols = [col for col in data.columns if col.startswith(f'target_cont_{pollutant}_')]
    pred_cols = [col for col in data.columns if col.startswith(f'pred_cont_{pollutant}_')]
    
    # Remove aggregated columns (min, max, mean) from station columns
    target_cols = [col for col in target_cols 
                  if not any(x in col for x in ['min', 'max', 'mean'])]
    pred_cols = [col for col in pred_cols 
                if not any(x in col for x in ['min', 'max', 'mean'])]
    
    if not target_cols or not pred_cols:
        return 0, 0, 0, 0
    
    # Calculate confusion matrix across all stations
    tp, fp, tn, fn = 0, 0, 0, 0
    
    for target_col, pred_col in zip(target_cols, pred_cols):
        if target_col in data.columns and pred_col in data.columns:
            # Remove NaN values
            valid_mask = ~(np.isnan(data[target_col]) | np.isnan(data[pred_col]))
            if valid_mask.any():
                valid_data = data.loc[valid_mask]
                
                # True values
                actual_high = valid_data[target_col] >= threshold
                predicted_high = valid_data[pred_col] >= threshold
                
                # Confusion matrix
                tp += ((actual_high) & (predicted_high)).sum()
                fp += ((~actual_high) & (predicted_high)).sum()
                tn += ((~actual_high) & (~predicted_high)).sum()
                fn += ((actual_high) & (~predicted_high)).sum()
    
    return tp, fp, tn, fn

def calculate_classification_metrics(tp: int, fp: int, tn: int, fn: int) -> Dict[str, float]:
    """
    Calculate accuracy, sensitivity, and specificity from confusion matrix values.
    
    Args:
        tp: True Positives
        fp: False Positives
        tn: True Negatives
        fn: False Negatives
        
    Returns:
        Dictionary with accuracy, sensitivity, and specificity
    """
    total = tp + fp + tn + fn
    if total == 0:
        return {'accuracy': 0.0, 'sensitivity': 0.0, 'specificity': 0.0}
    
    accuracy = (tp + tn) / total
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    return {
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity
    }

def create_confusion_matrix_plot(models_data: Dict[str, pd.DataFrame], 
                                model_id_mapping: Dict[str, str],
                                threshold: int, pollutant: str = 'otres') -> go.Figure:
    """
    Create confusion matrix plots for all models at a specific threshold.
    
    Args:
        models_data: Dictionary mapping model names to their DataFrames
        model_id_mapping: Dictionary mapping model names to their IDs
        threshold: Ozone threshold in ppbs
        pollutant: Name of the pollutant to analyze
        
    Returns:
        Plotly figure with subplots showing confusion matrices
    """
    model_names = list(models_data.keys())
    n_models = len(model_names)
    
    # Calculate number of rows and columns for subplot grid
    cols = min(3, n_models)  # Max 3 columns
    rows = (n_models + cols - 1) // cols  # Ceiling division
    
    # Create subplot titles using model IDs
    subplot_titles = [model_id_mapping[model_name] for model_name in model_names]
    
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=subplot_titles,
        specs=[[{"type": "heatmap"} for _ in range(cols)] for _ in range(rows)],
        vertical_spacing=0.25,
        horizontal_spacing=0.10
    )
    
    for i, model_name in enumerate(model_names):
        data = models_data[model_name]
        tp, fp, tn, fn = calculate_confusion_matrix_for_threshold(data, threshold, pollutant)
        
        # Create confusion matrix values (rearranged for better visualization)
        cm_values = [[tp, fn], [fp, tn]]
        cm_text = [[f"{val:.3f}" if isinstance(val, float) else str(val) for val in row] for row in cm_values]
        
        # Calculate row and column position
        row = (i // cols) + 1
        col = (i % cols) + 1
        
        # Create heatmap
        heatmap = go.Heatmap(
            z=cm_values,
            x=['Predicted High', 'Predicted Low'],
            y=['Actual High', 'Actual Low'],
            text=cm_text,
            texttemplate="%{text}",
            textfont={"size": 14},
            colorscale='Blues',
            showscale=False
        )
        
        # Add to subplot
        fig.add_trace(heatmap, row=row, col=col)
    
    # Update layout
    fig.update_layout(
        title=f'Confusion Matrices for {pollutant.upper()} ≥ {threshold} ppbs',
        height=200 * rows,
        width=1400,
        showlegend=False
    )
    
    # Update axes labels
    for i in range(n_models):
        row = (i // cols) + 1
        col = (i % cols) + 1
        fig.update_xaxes(title_text="Prediction", row=row, col=col)
        fig.update_yaxes(title_text="Actual", row=row, col=col)
    
    return fig

def create_model_id_mapping_table(model_id_mapping: Dict[str, str]) -> go.Figure:
    """
    Create a table showing the mapping between model IDs and full model names.
    
    Args:
        model_id_mapping: Dictionary mapping model names to their IDs
        
    Returns:
        Plotly figure with mapping table
    """
    # Prepare data for table
    model_ids = []
    model_names = []
    
    for model_name, model_id in model_id_mapping.items():
        model_ids.append(model_id)
        model_names.append(model_name)
    
    fig = go.Figure(data=[go.Table(
        header=dict(values=['Model ID', 'Model Name']),
        cells=dict(values=[model_ids, model_names]),
        columnwidth=[0.2, 0.8]
    )])
    
    fig.update_layout(
        title='Model ID Mapping',
        height=200 + len(model_ids) * 40
    )
    
    return fig

def create_classification_summary_table(models_data: Dict[str, pd.DataFrame], 
                                       model_id_mapping: Dict[str, str],
                                       thresholds: List[int], pollutant: str = 'otres') -> go.Figure:
    """
    Create a heatmap showing classification metrics for all models and thresholds.
    
    Args:
        models_data: Dictionary mapping model names to their DataFrames
        model_id_mapping: Dictionary mapping model names to their IDs
        thresholds: List of ozone thresholds to evaluate
        pollutant: Name of the pollutant to analyze
        
    Returns:
        Plotly figure with heatmap
    """
    model_ids = []
    metric_names = []
    metric_values = []
    
    for model_name, data in models_data.items():
        model_id = model_id_mapping[model_name]
        model_ids.append(model_id)
        
        for threshold in thresholds:
            tp, fp, tn, fn = calculate_confusion_matrix_for_threshold(data, threshold, pollutant)
            metrics = calculate_classification_metrics(tp, fp, tn, fn)
            
            # Add metrics for this threshold
            metric_names.extend([
                f'{threshold}ppb_Accuracy',
                f'{threshold}ppb_Sensitivity', 
                f'{threshold}ppb_Specificity'
            ])
            metric_values.extend([
                round(metrics['accuracy'], 3),
                round(metrics['sensitivity'], 3),
                round(metrics['specificity'], 3)
            ])
    
    if not model_ids:
        return go.Figure()
    
    # Reshape data for heatmap (models as rows, metrics as columns)
    n_models = len(model_ids)
    n_metrics_per_model = len(thresholds) * 3  # 3 metrics per threshold
    
    # Create the heatmap data matrix
    heatmap_data = []
    for i in range(n_models):
        row = metric_values[i * n_metrics_per_model:(i + 1) * n_metrics_per_model]
        heatmap_data.append(row)
    
    # Get unique metric names (they repeat for each model)
    unique_metric_names = metric_names[:n_metrics_per_model]
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=unique_metric_names,
        y=model_ids,
        colorscale='RdYlGn',  # Red to Yellow to Green
        zmin=0,
        zmax=1,
        text=[[f"{val:.3f}" for val in row] for row in heatmap_data],
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverongaps=False,
        colorbar=dict(title='Performance Score')
    ))
    
    fig.update_layout(
        title=f'{pollutant.upper()} Classification Performance Heatmap',
        xaxis_title='Metrics',
        yaxis_title='Models',
        height=300 + len(model_ids) * 30,
        width=1400
    )
    
    return fig

def save_model_performance_summary_csv(models_data: Dict[str, pd.DataFrame], 
                                      model_id_mapping: Dict[str, str],
                                      pollutant: str = 'otres',
                                      output_file: str = 'SUMMARY.csv') -> None:
    """
    Save model performance summary data to a CSV file, sorted by Mean RMSE.
    
    Args:
        models_data: Dictionary mapping model names to their DataFrames
        model_id_mapping: Dictionary mapping model names to their IDs
        pollutant: Name of the pollutant to analyze
        output_file: Output CSV file path
    """
    summary_data = []
    
    for model_name, data in models_data.items():
        rmse_by_hour = calculate_rmse_by_hour(data, pollutant)
        
        if rmse_by_hour:
            valid_rmse = [val for val in rmse_by_hour.values() if not np.isnan(val)]
            
            if valid_rmse:
                model_id = model_id_mapping[model_name]
                summary_data.append({
                    'Model_ID': model_id,
                    'Model_Name': model_name,
                    'Mean_RMSE': np.mean(valid_rmse),
                    'Std_RMSE': np.std(valid_rmse),
                    'Min_RMSE': np.min(valid_rmse),
                    'Max_RMSE': np.max(valid_rmse),
                    'Hours_Available': len(valid_rmse)
                })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        # Sort by Mean_RMSE (ascending - best performing models first)
        summary_df = summary_df.sort_values('Mean_RMSE')
        summary_df.to_csv(output_file, index=False)
        print(f"Model performance summary saved to {output_file} (sorted by Mean RMSE)")
    else:
        print("No summary data to save")

def save_rmse_by_hour_csv(models_data: Dict[str, pd.DataFrame], 
                         model_id_mapping: Dict[str, str],
                         pollutant: str = 'otres',
                         output_file: str = 'SUMMARY_BY_HOUR.csv') -> None:
    """
    Save RMSE by hour data for all models to a CSV file.
    
    Args:
        models_data: Dictionary mapping model names to their DataFrames
        model_id_mapping: Dictionary mapping model names to their IDs
        pollutant: Name of the pollutant to analyze
        output_file: Output CSV file path
    """
    all_data = []
    
    for model_name, data in models_data.items():
        rmse_by_hour = calculate_rmse_by_hour(data, pollutant)
        model_id = model_id_mapping[model_name]
        
        for hour, rmse in rmse_by_hour.items():
            all_data.append({
                'Model_ID': model_id,
                'Model_Name': model_name,
                'Predicted_Hour': hour,
                'RMSE': rmse if not np.isnan(rmse) else None
            })
    
    if all_data:
        rmse_df = pd.DataFrame(all_data)
        rmse_df.to_csv(output_file, index=False)
        print(f"RMSE by hour data saved to {output_file}")
    else:
        print("No RMSE by hour data to save")

# %% Load data from all models
print(f"Loading data from prediction path: {PREDICTION_PATH}")
models_data = load_all_models_data(PREDICTION_PATH)

if not models_data:
    print("No model data found. Please check the prediction path.")
    exit(1)

print(f"Successfully loaded data for {len(models_data)} models")

# Create model ID mapping
model_id_mapping = create_model_id_mapping(models_data)
print("Model ID mapping created:")
for model_name, model_id in model_id_mapping.items():
    print(f"  {model_id}: {model_name}")

# Save summary CSV files
print("\nSaving summary CSV files...")
save_model_performance_summary_csv(models_data, model_id_mapping, 'otres', 'SUMMARY.csv')
save_rmse_by_hour_csv(models_data, model_id_mapping, 'otres', 'SUMMARY_BY_HOUR.csv')
print("CSV files saved successfully!\n")

# %% --- DASH APP ---
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1('Air Pollution Multi-Model Comparison Dashboard'),
    html.H3(f'Comparing {len(models_data)} models for OTRES pollutant'),
    
    html.Div([
        html.H4('Model ID Mapping'),
        dcc.Graph(id='model-id-mapping-table')
    ], style={'marginBottom': '30px'}),
    
    html.Div([
        html.H4('Model Performance Summary'),
        dcc.Graph(id='summary-table')
    ], style={'marginBottom': '30px'}),
    
    html.Div([
        html.H4('RMSE Comparison by Predicted Hour'),
        dcc.Graph(id='rmse-comparison-plot')
    ], style={'marginBottom': '30px'}),
    
    html.Div([
        html.H4('RMSE Distribution by Individual Model'),
        dcc.Graph(id='model-grouping-plot')
    ], style={'marginBottom': '30px'}),
    
    html.Div([
        html.H4('Ozone Classification Performance Summary'),
        dcc.Graph(id='classification-summary-table')
    ], style={'marginBottom': '30px'}),
    
    html.Div([
        html.H4('Confusion Matrices - Ozone ≥ 90 ppbs'),
        dcc.Graph(id='confusion-matrix-90')
    ], style={'marginBottom': '30px'}),
    
    html.Div([
        html.H4('Confusion Matrices - Ozone ≥ 120 ppbs'),
        dcc.Graph(id='confusion-matrix-120')
    ], style={'marginBottom': '30px'}),
    
    html.Div([
        html.H4('Confusion Matrices - Ozone ≥ 150 ppbs'),
        dcc.Graph(id='confusion-matrix-150')
    ], style={'marginBottom': '30px'}),
    
    html.Div([
        html.H4('Model Information'),
        html.Div(id='model-info')
    ])
])

@app.callback(
    [Output('model-id-mapping-table', 'figure'),
     Output('summary-table', 'figure'),
     Output('rmse-comparison-plot', 'figure'),
     Output('model-grouping-plot', 'figure'),
     Output('classification-summary-table', 'figure'),
     Output('confusion-matrix-90', 'figure'),
     Output('confusion-matrix-120', 'figure'),
     Output('confusion-matrix-150', 'figure'),
     Output('model-info', 'children')],
    [Input('summary-table', 'id')]  # Dummy input to trigger callback
)
def update_plots(dummy_input):
    """Update all plots with the loaded model data."""
    
    # Create model ID mapping table
    mapping_fig = create_model_id_mapping_table(model_id_mapping)
    
    # Create summary table
    summary_fig = create_model_summary_table(models_data, model_id_mapping, 'otres')
    
    # Create RMSE comparison plot
    rmse_fig = create_rmse_comparison_plot(models_data, model_id_mapping, 'otres')
    
    # Create model grouping analysis
    grouping_fig = create_model_grouping_analysis(models_data, model_id_mapping, 'otres')
    
    # Create classification summary table
    thresholds = [90, 120, 150]
    classification_summary_fig = create_classification_summary_table(models_data, model_id_mapping, thresholds, 'otres')
    
    # Create confusion matrix plots for each threshold
    confusion_90_fig = create_confusion_matrix_plot(models_data, model_id_mapping, 90, 'otres')
    confusion_120_fig = create_confusion_matrix_plot(models_data, model_id_mapping, 120, 'otres')
    confusion_150_fig = create_confusion_matrix_plot(models_data, model_id_mapping, 150, 'otres')
    
    # Create model information
    model_info = []
    for model_name, data in models_data.items():
        model_id = model_id_mapping[model_name]
        info = html.Div([
            html.H5(f"Model {model_id}: {model_name}"),
            html.P(f"Total records: {len(data):,}"),
            html.P(f"Date range: {data['timestamp'].min()} to {data['timestamp'].max()}"),
            html.Hr()
        ])
        model_info.append(info)
    
    return mapping_fig, summary_fig, rmse_fig, grouping_fig, classification_summary_fig, confusion_90_fig, confusion_120_fig, confusion_150_fig, model_info

if __name__ == '__main__':
    app.run(debug=True, port=8085) 