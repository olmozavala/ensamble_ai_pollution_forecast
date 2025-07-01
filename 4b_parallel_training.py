#!/usr/bin/env python3
"""
Parallel Training Script for Air Pollution Forecasting

This script generates multiple configuration files with different parameter combinations
and runs training in parallel using subprocess. It takes the base config.json and creates
variations with different hyperparameters to test.

The script includes options to manipulate:
- prev_weather_hours and next_weather_hours (affects weather_time_dims calculation)
- prev_pollutant_hours, attention_heads, transformer blocks
- pollutants_to_keep, bootstrap settings, auto_regressive_steps

Author: AI Assistant
Date: 2024
"""

import json
import os
import subprocess
import time
import itertools
from pathlib import Path
from typing import Dict, List, Any, Tuple
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('parallel_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ParallelTrainer:
    """Handles parallel training with different configuration parameters."""
    
    def __init__(self, base_config_path: str = "config.json", logs_dir: str = None):
        """
        Initialize the parallel trainer.
        
        Args:
            base_config_path: Path to the base configuration file
            logs_dir: Directory to save training logs (default: logs/parallel_training)
        """
        self.base_config_path = base_config_path
        self.output_dir = Path("saved_confs/parallel_configs")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create logs directory for individual training logs
        if logs_dir is None:
            logs_dir = "logs/parallel_training"
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Load base configuration
        with open(base_config_path, 'r') as f:
            self.base_config = json.load(f)
        
        # Define parameter combinations to test
        self.parameter_combinations = self._define_parameter_combinations()
        
    def _define_parameter_combinations(self) -> List[Dict[str, Any]]:
        """
        Define all parameter combinations to test.
        
        Returns:
            List of dictionaries containing parameter combinations
        """
        # Define the parameter options
        param_options = {
            'prev_pollutant_hours': [8, 24],
            'attention_heads': [4],
            'weather_transformer_blocks': [4],
            'pollution_transformer_blocks': [4],
            'pollutants_to_keep': [
                ["co", "nodos", "otres", "pmdiez", "pmdoscinco", "nox", "no", "sodos", "pmco"],  # all
                ["otres"]
            ],
            'bootstrap_enabled': [True, False],
            'bootstrap_threshold': [2],
            'auto_regresive_steps': [8, 24],
            'prev_weather_hours': [4, 8],
            'next_weather_hours': [2]
        }
        
        # Generate all combinations
        combinations = []
        param_names = list(param_options.keys())
        param_values = list(param_options.values())
        
        for combination in itertools.product(*param_values):
            param_dict = dict(zip(param_names, combination))
            
            # Skip invalid combinations (e.g., bootstrap_threshold when bootstrap is disabled)
            if not param_dict['bootstrap_enabled'] and param_dict['bootstrap_threshold'] != 3:
                continue
                
            combinations.append(param_dict)
        
        logger.info(f"Generated {len(combinations)} parameter combinations")
        return combinations
    
    def _update_config(self, base_config: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update the base configuration with new parameters.
        
        Args:
            base_config: Base configuration dictionary
            params: Parameters to update
            
        Returns:
            Updated configuration dictionary
        """
        config = json.loads(json.dumps(base_config))  # Deep copy
        
        # Update architecture parameters
        config['arch']['args']['prev_pollutant_hours'] = params['prev_pollutant_hours']
        config['arch']['args']['attention_heads'] = params['attention_heads']
        config['arch']['args']['weather_transformer_blocks'] = params['weather_transformer_blocks']
        config['arch']['args']['pollution_transformer_blocks'] = params['pollution_transformer_blocks']
        
        # Calculate weather_time_dims as sum of prev_weather_hours + next_weather_hours + 1
        weather_time_dims = params['prev_weather_hours'] + params['next_weather_hours'] + 1
        config['arch']['args']['weather_time_dims'] = weather_time_dims
        
        # Compute input_features dynamically based on pollutants_to_keep
        # 30 stations of ozone, 3 (min,avg, max) for other pollutants, 12 time features
        input_features = 30 + (len(params['pollutants_to_keep']) - 1) * 3 + 12
        config['arch']['args']['input_features'] = input_features
        
        # Update data loader parameters
        config['data_loader']['args']['prev_pollutant_hours'] = params['prev_pollutant_hours']
        config['data_loader']['args']['prev_weather_hours'] = params['prev_weather_hours']
        config['data_loader']['args']['next_weather_hours'] = params['next_weather_hours']
        config['data_loader']['args']['pollutants_to_keep'] = params['pollutants_to_keep']
        config['data_loader']['args']['bootstrap_enabled'] = params['bootstrap_enabled']
        config['data_loader']['args']['bootstrap_threshold'] = params['bootstrap_threshold']
        config['data_loader']['args']['auto_regresive_steps'] = params['auto_regresive_steps']
        
        # Update trainer parameters
        config['trainer']['auto_regresive_steps'] = params['auto_regresive_steps']
        
        # Update model name to reflect parameters
        pollutants_str = "_".join(params['pollutants_to_keep'][:3])  # First 3 pollutants
        if len(params['pollutants_to_keep']) == 9:
            pollutants_str = "all"
        
        config['name'] = f"Parallel_{pollutants_str}_prev{params['prev_pollutant_hours']}_heads{params['attention_heads']}_w{params['weather_transformer_blocks']}_p{params['pollution_transformer_blocks']}_ar{params['auto_regresive_steps']}_bootstrap{params['bootstrap_enabled']}_thresh{params['bootstrap_threshold']}_weather{params['prev_weather_hours']}_{params['next_weather_hours']}"
        
        return config
    
    def _generate_config_filename(self, params: Dict[str, Any]) -> str:
        """
        Generate a filename for the configuration based on parameters.
        
        Args:
            params: Parameter dictionary
            
        Returns:
            Configuration filename
        """
        pollutants_str = "_".join(params['pollutants_to_keep'][:3])
        if len(params['pollutants_to_keep']) > 3:
            pollutants_str = "all"
        
        filename = f"parallel_{pollutants_str}_prev{params['prev_pollutant_hours']}_heads{params['attention_heads']}_w{params['weather_transformer_blocks']}_p{params['pollution_transformer_blocks']}_ar{params['auto_regresive_steps']}_bootstrap{params['bootstrap_enabled']}_thresh{params['bootstrap_threshold']}_weather{params['prev_weather_hours']}_{params['next_weather_hours']}.json"
        
        return filename
    
    def _run_training(self, config_path: str, gpu_id: int) -> Tuple[str, int]:
        """
        Run training for a specific configuration.
        
        Args:
            config_path: Path to the configuration file
            gpu_id: GPU ID to use for training
            
        Returns:
            Tuple of (config_name, return_code)
        """
        config_name = Path(config_path).stem
        
        # Create log files for this training run
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_filename = f"{config_name}_{timestamp}_gpu{gpu_id}.log"
        log_path = self.logs_dir / log_filename
        
        # Set CUDA device
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
        try:
            logger.info(f"Starting training for {config_name} on GPU {gpu_id}")
            logger.info(f"Log file: {log_path}")
            
            # Write header to log file
            with open(log_path, 'w') as log_file:
                log_file.write(f"Training Log for {config_name}\n")
                log_file.write(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                log_file.write(f"GPU ID: {gpu_id}\n")
                log_file.write(f"Config file: {config_path}\n")
                log_file.write("=" * 80 + "\n\n")
            
            # Run the training script
            cmd = ['.venv/bin/python', '4_train.py', '-c', config_path]
            result = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            # Write output to log file
            with open(log_path, 'a') as log_file:
                log_file.write("STDOUT:\n")
                log_file.write("-" * 40 + "\n")
                log_file.write(result.stdout)
                log_file.write("\n\nSTDERR:\n")
                log_file.write("-" * 40 + "\n")
                log_file.write(result.stderr)
                log_file.write(f"\n\nReturn code: {result.returncode}\n")
                log_file.write(f"Completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            if result.returncode == 0:
                logger.info(f"Training completed successfully for {config_name}")
                logger.info(f"Log saved to: {log_path}")
            else:
                logger.error(f"Training failed for {config_name}: {result.stderr}")
                logger.error(f"Log saved to: {log_path}")
                
            return config_name, result.returncode
            
        except subprocess.TimeoutExpired:
            # Write timeout error to log file
            with open(log_path, 'a') as log_file:
                log_file.write(f"\n\nERROR: Training timed out after 1 hour\n")
                log_file.write(f"Timeout occurred at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            logger.error(f"Training timed out for {config_name}")
            logger.error(f"Log saved to: {log_path}")
            return config_name, -1
        except Exception as e:
            # Write exception error to log file
            with open(log_path, 'a') as log_file:
                log_file.write(f"\n\nERROR: Exception occurred during training\n")
                log_file.write(f"Exception: {str(e)}\n")
                log_file.write(f"Exception occurred at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            logger.error(f"Error running training for {config_name}: {str(e)}")
            logger.error(f"Log saved to: {log_path}")
            return config_name, -1
    
    def generate_configs(self) -> List[str]:
        """
        Generate all configuration files.
        
        Returns:
            List of generated configuration file paths
        """
        config_paths = []
        
        for i, params in enumerate(self.parameter_combinations):
            # Update configuration with new parameters
            updated_config = self._update_config(self.base_config, params)
            
            # Generate filename
            filename = self._generate_config_filename(params)
            config_path = self.output_dir / filename
            
            # Save configuration
            with open(config_path, 'w') as f:
                json.dump(updated_config, f, indent=4)
            
            config_paths.append(str(config_path))
            logger.info(f"Generated config {i+1}/{len(self.parameter_combinations)}: {filename}")
        
        return config_paths
    
    def run_parallel_training(self, max_parallel: int = None) -> Dict[str, int]:
        """
        Run training in parallel for all generated configurations.
        
        Args:
            max_parallel: Maximum number of parallel processes (default: n_gpu from config)
            
        Returns:
            Dictionary mapping config names to return codes
        """
        if max_parallel is None:
            max_parallel = self.base_config.get('n_gpu', 4)
        
        # Generate all configurations
        config_paths = self.generate_configs()
        
        logger.info(f"Starting parallel training with {len(config_paths)} configurations using {max_parallel} processes")
        
        results = {}
        
        # Use ProcessPoolExecutor for parallel execution
        with ProcessPoolExecutor(max_workers=max_parallel) as executor:
            # Submit all training jobs
            future_to_config = {}
            for i, config_path in enumerate(config_paths):
                gpu_id = i % max_parallel
                future = executor.submit(self._run_training, config_path, gpu_id)
                future_to_config[future] = config_path
            
            # Collect results as they complete
            for future in as_completed(future_to_config):
                config_path = future_to_config[future]
                try:
                    config_name, return_code = future.result()
                    results[config_name] = return_code
                except Exception as e:
                    config_name = Path(config_path).stem
                    logger.error(f"Exception occurred for {config_name}: {str(e)}")
                    results[config_name] = -1
        
        return results
    
    def print_summary(self, results: Dict[str, int]) -> None:
        """
        Print a summary of training results.
        
        Args:
            results: Dictionary mapping config names to return codes
        """
        successful = sum(1 for code in results.values() if code == 0)
        failed = len(results) - successful
        
        logger.info("=" * 50)
        logger.info("TRAINING SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Total configurations: {len(results)}")
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Success rate: {successful/len(results)*100:.1f}%")
        
        if failed > 0:
            logger.info("\nFailed configurations:")
            for config_name, return_code in results.items():
                if return_code != 0:
                    logger.info(f"  - {config_name} (return code: {return_code})")
        
        # Create summary log file
        self._create_summary_log(results)
    
    def _create_summary_log(self, results: Dict[str, int]) -> None:
        """
        Create a summary log file with details about all training runs.
        
        Args:
            results: Dictionary mapping config names to return codes
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        summary_filename = f"training_summary_{timestamp}.log"
        summary_path = self.logs_dir / summary_filename
        
        successful = sum(1 for code in results.values() if code == 0)
        failed = len(results) - successful
        
        with open(summary_path, 'w') as summary_file:
            summary_file.write("PARALLEL TRAINING SUMMARY\n")
            summary_file.write("=" * 50 + "\n")
            summary_file.write(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            summary_file.write(f"Total configurations: {len(results)}\n")
            summary_file.write(f"Successful: {successful}\n")
            summary_file.write(f"Failed: {failed}\n")
            summary_file.write(f"Success rate: {successful/len(results)*100:.1f}%\n\n")
            
            summary_file.write("DETAILED RESULTS:\n")
            summary_file.write("-" * 30 + "\n")
            
            # List all log files in the logs directory
            log_files = list(self.logs_dir.glob(f"*_{timestamp[:8]}*.log"))
            log_files.sort()
            
            for log_file in log_files:
                # Extract config name from log filename
                filename = log_file.name
                if filename.startswith("training_summary_"):
                    continue
                    
                # Parse the filename to get config name and status
                parts = filename.replace(".log", "").split("_")
                if len(parts) >= 2:
                    config_name = "_".join(parts[:-2])  # Remove timestamp and gpu
                    gpu_id = parts[-1].replace("gpu", "")
                    
                    # Find the result for this config
                    return_code = results.get(config_name, -1)
                    status = "SUCCESS" if return_code == 0 else "FAILED"
                    
                    summary_file.write(f"{config_name} (GPU {gpu_id}): {status} (code: {return_code})\n")
                    summary_file.write(f"  Log file: {log_file.name}\n")
            
            summary_file.write(f"\nAll individual logs saved in: {self.logs_dir}\n")
        
        logger.info(f"Summary log created: {summary_path}")
    
    def list_logs(self) -> List[Path]:
        """
        List all available log files.
        
        Returns:
            List of log file paths
        """
        log_files = list(self.logs_dir.glob("*.log"))
        log_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)  # Sort by modification time
        return log_files
    
    def analyze_logs(self, config_name: str = None) -> Dict[str, Any]:
        """
        Analyze log files to extract training statistics.
        
        Args:
            config_name: Specific configuration to analyze (if None, analyze all)
            
        Returns:
            Dictionary with analysis results
        """
        log_files = self.list_logs()
        
        if config_name:
            log_files = [f for f in log_files if config_name in f.name]
        
        analysis = {
            'total_logs': len(log_files),
            'successful': 0,
            'failed': 0,
            'timeout': 0,
            'exceptions': 0,
            'configs': {}
        }
        
        for log_file in log_files:
            try:
                with open(log_file, 'r') as f:
                    content = f.read()
                
                # Extract config name from filename
                filename = log_file.name
                if filename.startswith("training_summary_"):
                    continue
                
                parts = filename.replace(".log", "").split("_")
                if len(parts) >= 2:
                    config = "_".join(parts[:-2])
                    
                    if config not in analysis['configs']:
                        analysis['configs'][config] = {
                            'log_files': [],
                            'status': 'unknown',
                            'gpu_id': None,
                            'start_time': None,
                            'end_time': None
                        }
                    
                    analysis['configs'][config]['log_files'].append(log_file.name)
                    
                    # Determine status from log content
                    if "Training completed successfully" in content:
                        analysis['configs'][config]['status'] = 'success'
                        analysis['successful'] += 1
                    elif "Training timed out" in content:
                        analysis['configs'][config]['status'] = 'timeout'
                        analysis['timeout'] += 1
                    elif "Exception occurred" in content:
                        analysis['configs'][config]['status'] = 'exception'
                        analysis['exceptions'] += 1
                    else:
                        analysis['configs'][config]['status'] = 'failed'
                        analysis['failed'] += 1
                    
                    # Extract GPU ID
                    if "gpu" in parts[-1]:
                        analysis['configs'][config]['gpu_id'] = parts[-1].replace("gpu", "")
                    
                    # Extract timestamps
                    import re
                    start_match = re.search(r"Started at: (.+)", content)
                    if start_match:
                        analysis['configs'][config]['start_time'] = start_match.group(1)
                    
                    end_match = re.search(r"Completed at: (.+)", content)
                    if end_match:
                        analysis['configs'][config]['end_time'] = end_match.group(1)
                        
            except Exception as e:
                logger.warning(f"Could not analyze log file {log_file}: {e}")
        
        return analysis


def main():
    """Main function to run parallel training."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run parallel training with different configurations')
    parser.add_argument('--config', default='config.json', type=str,
                       help='Base configuration file path')
    parser.add_argument('--max-parallel', type=int, default=None,
                       help='Maximum number of parallel processes')
    parser.add_argument('--generate-only', action='store_true',
                       help='Only generate configs, do not run training')
    parser.add_argument('--logs-dir', type=str, default=None,
                       help='Directory to save training logs (default: logs/parallel_training)')
    parser.add_argument('--list-logs', action='store_true',
                       help='List all available log files')
    parser.add_argument('--analyze-logs', action='store_true',
                       help='Analyze existing log files and show statistics')
    parser.add_argument('--config-name', type=str, default=None,
                       help='Specific configuration name for log analysis')
    
    args = parser.parse_args()
    
    # Initialize parallel trainer
    trainer = ParallelTrainer(args.config, args.logs_dir)
    
    if args.list_logs:
        # List all log files
        log_files = trainer.list_logs()
        logger.info(f"Found {len(log_files)} log files in {trainer.logs_dir}:")
        for log_file in log_files:
            logger.info(f"  - {log_file.name}")
    elif args.analyze_logs:
        # Analyze log files
        analysis = trainer.analyze_logs(args.config_name)
        logger.info("LOG ANALYSIS RESULTS:")
        logger.info(f"Total log files: {analysis['total_logs']}")
        logger.info(f"Successful runs: {analysis['successful']}")
        logger.info(f"Failed runs: {analysis['failed']}")
        logger.info(f"Timeout runs: {analysis['timeout']}")
        logger.info(f"Exception runs: {analysis['exceptions']}")
        
        if analysis['configs']:
            logger.info("\nConfiguration details:")
            for config, details in analysis['configs'].items():
                logger.info(f"  {config}: {details['status']} (GPU {details['gpu_id']})")
                if details['start_time']:
                    logger.info(f"    Started: {details['start_time']}")
                if details['end_time']:
                    logger.info(f"    Completed: {details['end_time']}")
    elif args.generate_only:
        # Only generate configurations
        config_paths = trainer.generate_configs()
        logger.info(f"Generated {len(config_paths)} configuration files in {trainer.output_dir}")
    else:
        # Run parallel training
        results = trainer.run_parallel_training(args.max_parallel)
        trainer.print_summary(results)


if __name__ == "__main__":
    main()
