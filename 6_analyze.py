import argparse
import os
import pandas as pd
import numpy as np
from parse_config import ConfigParser
from os.path import join
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from sklearn.metrics import r2_score

def compute_rmse(predictions: pd.DataFrame, targets: pd.DataFrame) -> float:
    """Compute RMSE between predictions and targets."""
    # Strip the 'pred_' and 'target_' prefixes to align columns
    predictions.columns = [col.replace("pred_", "") for col in predictions.columns]
    targets.columns = [col.replace("target_", "") for col in targets.columns]

    # Ensure the same column order
    predictions = predictions[targets.columns]
    return np.sqrt(np.mean((predictions - targets) ** 2))

def compute_mae(predictions: pd.DataFrame, targets: pd.DataFrame) -> float:
    """Compute MAE between predictions and targets."""
    # Strip the 'pred_' and 'target_' prefixes to align columns
    predictions.columns = [col.replace("pred_", "") for col in predictions.columns]
    targets.columns = [col.replace("target_", "") for col in targets.columns]

    # Ensure the same column order
    predictions = predictions[targets.columns]
    return np.mean(np.abs(predictions - targets))

def compute_r2(predictions: pd.DataFrame, targets: pd.DataFrame) -> float:
    """Compute R² score between predictions and targets."""
    # Strip the 'pred_' and 'target_' prefixes to align columns
    predictions.columns = [col.replace("pred_", "") for col in predictions.columns]
    targets.columns = [col.replace("target_", "") for col in targets.columns]

    # Ensure the same column order
    predictions = predictions[targets.columns]
    return r2_score(targets, predictions)

def analyze_model_predictions(model_folder: str, config: ConfigParser) -> Dict:
    """
    Analyze predictions for a single model.
    Returns a dictionary with various metrics.
    """
    # Get all CSV files for this model
    csv_files = sorted([f for f in os.listdir(model_folder) if f.endswith('.csv')])
    
    # Initialize results dictionary
    results = {
        'model_name': os.path.basename(model_folder),
        'overall_rmse': {},  # RMSE for each pollutant averaged across all stations and hours
        'overall_mae': {},   # MAE for each pollutant averaged across all stations and hours
        'overall_r2': {},    # R² for each pollutant averaged across all stations and hours
        'hourly_rmse': {},   # RMSE for each pollutant by predicted hour
        'hourly_mae': {},    # MAE for each pollutant by predicted hour
        'hourly_r2': {},     # R² for each pollutant by predicted hour
        'time_of_day_rmse': {},  # RMSE for each pollutant by hour of the day
        'time_of_day_mae': {},   # MAE for each pollutant by hour of the day
        'time_of_day_r2': {}     # R² for each pollutant by hour of the day
    }
    
    # Get unique pollutants from the first file
    first_file = pd.read_csv(join(model_folder, csv_files[0]))
    pollutants = list(set(col.split('_')[2] for col in first_file.columns if col.startswith('pred_')))
    
    # Initialize DataFrames to store all predictions and targets
    all_predictions = []
    all_targets = []
    all_timestamps = []
    
    # Read all CSV files
    for csv_file in csv_files:
        df = pd.read_csv(join(model_folder, csv_file))
        predicted_hour = int(csv_file.split('_')[-1].split('.')[0])
        
        # Get predictions and targets for each pollutant
        for pollutant in pollutants:
            pred_cols = [col for col in df.columns if col.startswith(f'pred_cont_{pollutant}')]
            target_cols = [col for col in df.columns if col.startswith(f'target_cont_{pollutant}')]
            
            # Store predictions and targets
            all_predictions.append(df[pred_cols])
            all_targets.append(df[target_cols])
            all_timestamps.append(df['timestamp'])
    
    # Compute overall metrics for each pollutant
    for pollutant in pollutants:
        pred_cols = [col for col in df.columns if col.startswith(f'pred_cont_{pollutant}')]
        target_cols = [col for col in df.columns if col.startswith(f'target_cont_{pollutant}')]
        
        results['overall_rmse'][pollutant] = compute_rmse(df[pred_cols], df[target_cols])
        results['overall_mae'][pollutant] = compute_mae(df[pred_cols], df[target_cols])
        results['overall_r2'][pollutant] = compute_r2(df[pred_cols], df[target_cols])

        # Print the overall metrics for the pollutant
        print(f"Overall metrics for {pollutant}:")
        print(f"  RMSE: {results['overall_rmse'][pollutant]:.6f}")
        print(f"  MAE: {results['overall_mae'][pollutant]:.6f}")
        print(f"  R²: {results['overall_r2'][pollutant]:.6f}")
    
    # Compute metrics by predicted hour
    max_hour = 0
    for csv_file in csv_files:
        hour = int(csv_file.split('_')[-1].split('.')[0])
        df = pd.read_csv(join(model_folder, csv_file))
        max_hour = max(max_hour, hour)
        results['hourly_rmse'][hour] = {}
        results['hourly_mae'][hour] = {}
        results['hourly_r2'][hour] = {}
        
        for pollutant in pollutants:
            pred_cols = [col for col in df.columns if col.startswith(f'pred_cont_{pollutant}')]
            target_cols = [col for col in df.columns if col.startswith(f'target_cont_{pollutant}')]
            
            results['hourly_rmse'][hour][pollutant] = compute_rmse(df[pred_cols], df[target_cols])
            results['hourly_mae'][hour][pollutant] = compute_mae(df[pred_cols], df[target_cols])
            results['hourly_r2'][hour][pollutant] = compute_r2(df[pred_cols], df[target_cols])
    
    # Compute metrics by hour of the day
    timestamps = pd.to_datetime(all_timestamps[0])
    hour_of_day = timestamps.dt.hour
    
    for hour in range(max_hour):
        mask = hour_of_day == hour
        results['time_of_day_rmse'][hour] = {}
        results['time_of_day_mae'][hour] = {}
        results['time_of_day_r2'][hour] = {}
        
        for pollutant in pollutants:
            pred_cols = [col for col in df.columns if col.startswith(f'pred_cont_{pollutant}')]
            target_cols = [col for col in df.columns if col.startswith(f'target_cont_{pollutant}')]
            
            if pred_cols and target_cols:
                results['time_of_day_rmse'][hour][pollutant] = compute_rmse(df[pred_cols][mask], df[target_cols][mask])
                results['time_of_day_mae'][hour][pollutant] = compute_mae(df[pred_cols][mask], df[target_cols][mask])
                results['time_of_day_r2'][hour][pollutant] = compute_r2(df[pred_cols][mask], df[target_cols][mask])
    
    return results

def plot_analysis_results(results: Dict, output_path: str):
    """Plot the analysis results."""
    os.makedirs(output_path, exist_ok=True)
    
    # Plot overall metrics
    metrics = ['overall_rmse', 'overall_mae', 'overall_r2']
    titles = ['Overall RMSE by Pollutant', 'Overall MAE by Pollutant', 'Overall R² by Pollutant']
    ylabels = ['RMSE', 'MAE', 'R²']
    
    for metric, title, ylabel in zip(metrics, titles, ylabels):
        plt.figure(figsize=(12, 6))
        metric_df = pd.DataFrame(results[metric], index=[0])
        metric_df.plot(kind='bar')
        plt.title(title)
        plt.ylabel(ylabel)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(join(output_path, f'{metric}.png'))
        plt.close()
    
    # Plot hourly metrics
    metrics = ['hourly_rmse', 'hourly_mae', 'hourly_r2']
    titles = ['RMSE by Predicted Hour', 'MAE by Predicted Hour', 'R² by Predicted Hour']
    ylabels = ['RMSE', 'MAE', 'R²']
    
    for metric, title, ylabel in zip(metrics, titles, ylabels):
        # Plot for each pollutant
        hourly_metric = pd.DataFrame(results[metric]).T
        hourly_metric = hourly_metric.sort_index()  # Ensure rows are sorted by hour
        for pollutant in hourly_metric.columns:
            plt.figure(figsize=(15, 8))
            hourly_metric[pollutant].sort_index().plot(kind='line', marker='o')
            plt.title(f'{title} - {pollutant}')
            plt.xlabel('Predicted Hour')
            plt.ylabel(ylabel)
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(join(output_path, f'{metric}_{pollutant}.png'))
            plt.close()
        
        # Plot for all pollutants
        plt.figure(figsize=(15, 8))
        hourly_metric.sort_index().plot(kind='line', marker='o')
        plt.title(title)
        plt.xlabel('Predicted Hour')
        plt.ylabel(ylabel)
        plt.legend(title='Pollutant')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(join(output_path, f'{metric}_all.png'))
        plt.close()
    
    # Plot time of day metrics
    metrics = ['time_of_day_rmse', 'time_of_day_mae', 'time_of_day_r2']
    titles = ['RMSE by Hour of Day', 'MAE by Hour of Day', 'R² by Hour of Day']
    ylabels = ['RMSE', 'MAE', 'R²']
    
    for metric, title, ylabel in zip(metrics, titles, ylabels):
        plt.figure(figsize=(15, 8))
        time_of_day_metric = pd.DataFrame(results[metric]).T
        time_of_day_metric.sort_index().plot(kind='line', marker='o')
        plt.title(title)
        plt.xlabel('Hour of Day')
        plt.ylabel(ylabel)
        plt.legend(title='Pollutant')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(join(output_path, f'{metric}.png'))
        plt.close()

def main(config):
    logger = config.get_logger('analyze')
    
    prediction_path = config['analyze']['prediction_path']
    output_path = config['analyze']['output_path']
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Get all model folders
    model_folders = [f for f in os.listdir(prediction_path) 
                    if os.path.isdir(join(prediction_path, f))]
    
    all_results = {}
    
    # Analyze each model
    for model_folder in model_folders:
        logger.info(f'Analyzing model: {model_folder}')
        model_path = join(prediction_path, model_folder)
        results = analyze_model_predictions(model_path, config)
        all_results[model_folder] = results
        
        # Save results to CSV
        model_output_path = join(output_path, model_folder)
        os.makedirs(model_output_path, exist_ok=True)
        
        # Save overall metrics
        for metric in ['overall_rmse', 'overall_mae', 'overall_r2']:
            pd.DataFrame(results[metric], index=[0]).to_csv(
                join(model_output_path, f'{metric}.csv'))
        
        # Save hourly metrics
        for metric in ['hourly_rmse', 'hourly_mae', 'hourly_r2']:
            pd.DataFrame(results[metric]).T.to_csv(
                join(model_output_path, f'{metric}.csv'))
        
        # Save time of day metrics
        for metric in ['time_of_day_rmse', 'time_of_day_mae', 'time_of_day_r2']:
            pd.DataFrame(results[metric]).T.to_csv(
                join(model_output_path, f'{metric}.csv'))
        
        # Plot results
        plot_analysis_results(results, model_output_path)
    
    # Create summary comparison for each metric
    metrics = ['overall_rmse', 'overall_mae', 'overall_r2']
    titles = ['Model Comparison - Overall RMSE', 'Model Comparison - Overall MAE', 'Model Comparison - Overall R²']
    ylabels = ['RMSE', 'MAE', 'R²']
    
    for metric, title, ylabel in zip(metrics, titles, ylabels):
        summary = pd.DataFrame({
            model: results[metric] 
            for model, results in all_results.items()
        }).T
        
        summary.to_csv(join(output_path, f'model_comparison_{metric}.csv'))
        
        plt.figure(figsize=(15, 8))
        summary.plot(kind='bar')
        plt.title(title)
        plt.ylabel(ylabel)
        plt.xticks(rotation=45)
        plt.legend(title='Pollutant')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(join(output_path, f'model_comparison_{metric}.png'))
        plt.close()

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Analyze Model Predictions')
    args.add_argument('-c', '--config', default='config.json', type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    
    config = ConfigParser.from_args(args)
    main(config) 