import argparse
import os
import pandas as pd
import numpy as np
from parse_config import ConfigParser
from os.path import join
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

def compute_rmse(predictions: pd.DataFrame, targets: pd.DataFrame) -> float:
    """Compute RMSE between predictions and targets."""
    # Strip the 'pred_' and 'target_' prefixes to align columns
    predictions.columns = [col.replace("pred_", "") for col in predictions.columns]
    targets.columns = [col.replace("target_", "") for col in targets.columns]

    # Ensure the same column order
    predictions = predictions[targets.columns]
    return np.sqrt(np.mean((predictions - targets) ** 2))

def analyze_model_predictions(model_folder: str, config: ConfigParser) -> Dict:
    """
    Analyze predictions for a single model.
    Returns a dictionary with various RMSE metrics.
    """
    # Get all CSV files for this model
    csv_files = sorted([f for f in os.listdir(model_folder) if f.endswith('.csv')])
    
    # Initialize results dictionary
    results = {
        'model_name': os.path.basename(model_folder),
        'overall_rmse': {},  # RMSE for each pollutant averaged across all stations and hours
        'hourly_rmse': {},   # RMSE for each pollutant by predicted hour
        'time_of_day_rmse': {}  # RMSE for each pollutant by hour of the day
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
    
    # Compute overall RMSE for each pollutant
    for pollutant in pollutants:
        pred_cols = [col for col in df.columns if col.startswith(f'pred_cont_{pollutant}')]
        target_cols = [col for col in df.columns if col.startswith(f'target_cont_{pollutant}')]
        
        results['overall_rmse'][pollutant] = compute_rmse(
            df[pred_cols], 
            df[target_cols]
        )

        # Print the overall RMSE for the pollutant
        print(f"Overall RMSE for {pollutant}: {results['overall_rmse'][pollutant]:.6f}")
    
    # Compute RMSE by predicted hour
    max_hour = 0
    for csv_file in csv_files:
        hour = int(csv_file.split('_')[-1].split('.')[0])
        df = pd.read_csv(join(model_folder, csv_file))
        max_hour = max(max_hour, hour)
        results['hourly_rmse'][hour] = {}
        
        for pollutant in pollutants:
            pred_cols = [col for col in df.columns if col.startswith(f'pred_cont_{pollutant}')]
            target_cols = [col for col in df.columns if col.startswith(f'target_cont_{pollutant}')]
            
            results['hourly_rmse'][hour][pollutant] = compute_rmse(
                df[pred_cols], 
                df[target_cols]
            )
    
    # Compute RMSE by hour of the day
    timestamps = pd.to_datetime(all_timestamps[0])
    hour_of_day = timestamps.dt.hour
    
    for hour in range(max_hour):
        mask = hour_of_day == hour
        results['time_of_day_rmse'][hour] = {}
        
        for pollutant in pollutants:
            pred_cols = [col for col in df.columns if col.startswith(f'pred_cont_{pollutant}')]
            target_cols = [col for col in df.columns if col.startswith(f'target_cont_{pollutant}')]
            
            if pred_cols and target_cols:
                results['time_of_day_rmse'][hour][pollutant] = compute_rmse(
                    df[pred_cols][mask], 
                    df[target_cols][mask]
                )
    
    return results

def plot_analysis_results(results: Dict, output_path: str):
    """Plot the analysis results."""
    os.makedirs(output_path, exist_ok=True)
    
    # Plot overall RMSE
    plt.figure(figsize=(12, 6))
    overall_rmse = pd.DataFrame(results['overall_rmse'], index=[0])
    overall_rmse.plot(kind='bar')
    plt.title('Overall RMSE by Pollutant')
    plt.ylabel('RMSE')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(join(output_path, 'overall_rmse.png'))
    plt.close()
    
    # Plot RMSE by predicted hour for each pollutant
    hourly_rmse = pd.DataFrame(results['hourly_rmse']).T
    for pollutant in hourly_rmse.columns:
        plt.figure(figsize=(15, 8))
        hourly_rmse[pollutant].sort_index().plot(kind='line', marker='o')
        plt.title(f'RMSE by Predicted Hour - {pollutant}')
        plt.xlabel('Predicted Hour') 
        plt.ylabel('RMSE')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(join(output_path, f'hourly_rmse_{pollutant}.png'))
        plt.close()

    # Plot RMSE by predicted hour
    plt.figure(figsize=(15, 8))
    hourly_rmse = pd.DataFrame(results['hourly_rmse']).T
    hourly_rmse.plot(kind='line', marker='o')
    plt.title('RMSE by Predicted Hour')
    plt.xlabel('Predicted Hour')
    plt.ylabel('RMSE')
    plt.legend(title='Pollutant')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(join(output_path, 'hourly_rmse_all.png'))
    plt.close()
 
    
    # Plot RMSE by predicted hour for all pollutants
    plt.figure(figsize=(15, 8))
    hourly_rmse.plot(kind='line', marker='o')
    plt.title('RMSE by Predicted Hour')
    plt.xlabel('Predicted Hour')
    plt.ylabel('RMSE')
    plt.grid(True)
    
    # Plot RMSE by hour of day
    plt.figure(figsize=(15, 8))
    time_of_day_rmse = pd.DataFrame(results['time_of_day_rmse']).T
    time_of_day_rmse.plot(kind='line', marker='o')
    plt.title('RMSE by Hour of Day')
    plt.xlabel('Hour of Day')
    plt.ylabel('RMSE')
    plt.legend(title='Pollutant')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(join(output_path, 'time_of_day_rmse.png'))
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
        
        # Save overall RMSE
        pd.DataFrame(results['overall_rmse'], index=[0]).to_csv(
            join(model_output_path, 'overall_rmse.csv'))
        
        # Save hourly RMSE
        pd.DataFrame(results['hourly_rmse']).T.to_csv(
            join(model_output_path, 'hourly_rmse.csv'))
        
        # Save time of day RMSE
        pd.DataFrame(results['time_of_day_rmse']).T.to_csv(
            join(model_output_path, 'time_of_day_rmse.csv'))
        
        # Plot results
        plot_analysis_results(results, model_output_path)
    
    # Create summary comparison
    summary = pd.DataFrame({
        model: results['overall_rmse'] 
        for model, results in all_results.items()
    }).T
    
    summary.to_csv(join(output_path, 'model_comparison.csv'))
    
    # Plot model comparison
    plt.figure(figsize=(15, 8))
    summary.plot(kind='bar')
    plt.title('Model Comparison - Overall RMSE')
    plt.ylabel('RMSE')
    plt.xticks(rotation=45)
    plt.legend(title='Pollutant')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(join(output_path, 'model_comparison.png'))
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