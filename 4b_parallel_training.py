#!/usr/bin/env python3
"""
Parallel Training Script for Air Pollution Forecasting

This script generates multiple configuration files with different parameter combinations
and runs training in parallel using subprocess. It takes the base config.json and creates
variations with different hyperparameters to test.

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
    
    def __init__(self, base_config_path: str = "config.json"):
        """
        Initialize the parallel trainer.
        
        Args:
            base_config_path: Path to the base configuration file
        """
        self.base_config_path = base_config_path
        self.output_dir = Path("saved_confs/parallel_configs")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
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
            'weather_transformer_blocks': [5],
            'pollution_transformer_blocks': [5],
            'pollutants_to_keep': [
                ["co", "nodos", "otres", "pmdiez", "pmdoscinco", "nox", "no", "sodos", "pmco"],  # all
                ["otres"],  # only otres
            ],
            'bootstrap_enabled': [True, False],
            'bootstrap_threshold': [2],
            'auto_regresive_steps': [8, 24]
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
        
        # Compute input_features dynamically based on pollutants_to_keep
        input_features = 30 + (len(params['pollutants_to_keep']) - 1) * 3 + 12
        config['arch']['args']['input_features'] = input_features
        
        # Update data loader parameters
        config['data_loader']['args']['prev_pollutant_hours'] = params['prev_pollutant_hours']
        config['data_loader']['args']['pollutants_to_keep'] = params['pollutants_to_keep']
        config['data_loader']['args']['bootstrap_enabled'] = params['bootstrap_enabled']
        config['data_loader']['args']['bootstrap_threshold'] = params['bootstrap_threshold']
        config['data_loader']['args']['auto_regresive_steps'] = params['auto_regresive_steps']
        
        # Update trainer parameters
        config['trainer']['auto_regresive_steps'] = params['auto_regresive_steps']
        
        # Update model name to reflect parameters
        pollutants_str = "_".join(params['pollutants_to_keep'][:3])  # First 3 pollutants
        if len(params['pollutants_to_keep']) > 3:
            pollutants_str += "_all"
        
        config['name'] = f"Parallel_{pollutants_str}_prev{params['prev_pollutant_hours']}_heads{params['attention_heads']}_w{params['weather_transformer_blocks']}_p{params['pollution_transformer_blocks']}_ar{params['auto_regresive_steps']}_bootstrap{params['bootstrap_enabled']}"
        
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
            pollutants_str = "all_pollutants"
        
        filename = f"parallel_{pollutants_str}_prev{params['prev_pollutant_hours']}_heads{params['attention_heads']}_w{params['weather_transformer_blocks']}_p{params['pollution_transformer_blocks']}_ar{params['auto_regresive_steps']}_bootstrap{params['bootstrap_enabled']}_thresh{params['bootstrap_threshold']}.json"
        
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
        
        # Set CUDA device
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
        try:
            logger.info(f"Starting training for {config_name} on GPU {gpu_id}")
            
            # Run the training script
            cmd = ['.venv/bin/python', '4_train.py', '-c', config_path]
            result = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            if result.returncode == 0:
                logger.info(f"Training completed successfully for {config_name}")
            else:
                logger.error(f"Training failed for {config_name}: {result.stderr}")
                
            return config_name, result.returncode
            
        except subprocess.TimeoutExpired:
            logger.error(f"Training timed out for {config_name}")
            return config_name, -1
        except Exception as e:
            logger.error(f"Error running training for {config_name}: {str(e)}")
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
    
    args = parser.parse_args()
    
    # Initialize parallel trainer
    trainer = ParallelTrainer(args.config)
    
    if args.generate_only:
        # Only generate configurations
        config_paths = trainer.generate_configs()
        logger.info(f"Generated {len(config_paths)} configuration files in {trainer.output_dir}")
    else:
        # Run parallel training
        results = trainer.run_parallel_training(args.max_parallel)
        trainer.print_summary(results)


if __name__ == "__main__":
    main()
