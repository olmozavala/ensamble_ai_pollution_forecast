#!/usr/bin/env python3
"""
Parallel Testing Script for Air Pollution Forecasting

This script scans the models directory to find all trained models, matches them with
corresponding configuration files, and runs testing in parallel for all models.

Author: AI Assistant
Date: 2024
"""

import argparse
import json
import os
import subprocess
import time
import glob
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('parallel_testing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ParallelTester:
    """Handles parallel testing for all trained models."""
    
    def __init__(self, configs_path: str = "saved_confs/parallel_configs"):
        """
        Initialize the parallel tester.
        
        Args:
            configs_path: Path to the parallel configs directory
        """
        self.configs_path = Path(configs_path)
        self.output_dir = Path("saved_confs/parallel_configs")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load all available config files
        self.available_configs = self._load_available_configs()
        
        # Extract models path from the first available config
        self.models_path = self._extract_models_path_from_configs()
        
    def _load_available_configs(self) -> Dict[str, str]:
        """
        Load all available configuration files and create a mapping.
        
        Returns:
            Dictionary mapping config names to file paths
        """
        config_files = list(self.configs_path.glob("*.json"))
        config_mapping = {}
        
        for config_file in config_files:
            # Extract the base name without extension
            base_name = config_file.stem
            
            # Remove the 'parallel_' prefix and '_threshX' suffix for matching
            clean_name = base_name.replace('parallel_', '')
            clean_name = re.sub(r'_thresh\d+$', '', clean_name)
            
            config_mapping[clean_name] = str(config_file)
            
        logger.info(f"Loaded {len(config_mapping)} configuration files")
        return config_mapping
    
    def _extract_models_path_from_configs(self) -> Path:
        """
        Extract the models path from the available configuration files.
        
        Returns:
            Path to the models directory
        """
        if not self.available_configs:
            raise ValueError("No configuration files found. Cannot determine models path.")
        
        # Try to get models path from the first available config
        for config_name, config_path in self.available_configs.items():
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                # Check if the config has the test section with all_models_path
                if 'test' in config and 'all_models_path' in config['test']:
                    models_path = Path(config['test']['all_models_path'])
                    logger.info(f"Using models path from config {config_name}: {models_path}")
                    return models_path
                    
            except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
                logger.warning(f"Could not read models path from config {config_name}: {e}")
                continue
        
        # If no config has the models path, use a default
        default_path = Path("/home/olmozavala/DATA/AirPollution/OUTPUT/models")
        logger.warning(f"Could not find models path in any config. Using default: {default_path}")
        return default_path
    
    def _find_model_config_match(self, model_name: str) -> Optional[str]:
        """
        Find the matching configuration file for a given model name.
        
        Args:
            model_name: Name of the model folder
            
        Returns:
            Path to the matching config file, or None if not found
        """
        # Clean the model name for matching
        # Remove any timestamp suffixes and normalize
        clean_model_name = model_name
        
        # Try exact match first
        if clean_model_name in self.available_configs:
            return self.available_configs[clean_model_name]
        
        # Try partial matches
        for config_name, config_path in self.available_configs.items():
            if config_name in clean_model_name or clean_model_name in config_name:
                logger.info(f"Found partial match: {model_name} -> {config_name}")
                return config_path
        
        logger.warning(f"No config match found for model: {model_name}")
        return None
    
    def _discover_models(self) -> List[Tuple[str, str, str]]:
        """
        Discover all trained models in the models directory.
        
        Returns:
            List of tuples (model_folder, run_folder, full_path)
        """
        models = []
        
        # Scan the models directory
        for model_folder in self.models_path.iterdir():
            if not model_folder.is_dir():
                continue
                
            model_name = model_folder.name
            
            # Look for subdirectories (run folders) within each model folder
            for run_folder in model_folder.iterdir():
                if not run_folder.is_dir():
                    continue
                    
                # Check if this is a timestamp folder (contains model files)
                if re.match(r'\d{4}_\d{6}', run_folder.name):
                    # Check if model_best.pth exists
                    model_file = run_folder / "model_best.pth"
                    if model_file.exists():
                        models.append((model_name, run_folder.name, str(run_folder)))
                        logger.info(f"Found model: {model_name}/{run_folder.name}")
        
        logger.info(f"Discovered {len(models)} trained models")
        return models
    
    def _update_config_for_testing(self, config_path: str, model_path: str) -> Dict[str, Any]:
        """
        Update configuration for testing with the specific model path.
        
        Args:
            config_path: Path to the original config file
            model_path: Path to the specific model folder
            
        Returns:
            Updated configuration dictionary
        """
        # Load the original config
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Update the model path
        config['test']['model_path'] = model_path
        
        # Update the analyze path
        model_name = Path(model_path).parent.name
        run_name = Path(model_path).name
        
        return config
    
    def _generate_test_config_filename(self, model_name: str, run_name: str) -> str:
        """
        Generate a filename for the test configuration.
        
        Args:
            model_name: Name of the model folder
            run_name: Name of the run folder
            
        Returns:
            Test configuration filename
        """
        # Clean the names for filename
        clean_model_name = model_name.replace('/', '_').replace('\\', '_')
        clean_run_name = run_name.replace('/', '_').replace('\\', '_')
        
        filename = f"test_{clean_model_name}_{clean_run_name}.json"
        return filename
    
    def _run_testing(self, config_path: str, gpu_id: int) -> Tuple[str, int]:
        """
        Run testing for a specific configuration.
        
        Args:
            config_path: Path to the configuration file
            gpu_id: GPU ID to use for testing
            
        Returns:
            Tuple of (config_name, return_code)
        """
        config_name = Path(config_path).stem
        
        # Set CUDA device
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
        try:
            logger.info(f"Starting testing for {config_name} on GPU {gpu_id}")
            
            # Run the testing script
            cmd = ['/home/olmozavala/CODE/ensamble_ai_pollution_forecast/.venv/bin/python', '5_test.py', '-c', config_path]
            result = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                timeout=7200  # 2 hour timeout for testing
            )
            
            if result.returncode == 0:
                logger.info(f"Testing completed successfully for {config_name}")
            else:
                logger.error(f"Testing failed for {config_name}: {result.stderr}")
                
            return config_name, result.returncode
            
        except subprocess.TimeoutExpired:
            logger.error(f"Testing timed out for {config_name}")
            return config_name, -1
        except Exception as e:
            logger.error(f"Error running testing for {config_name}: {str(e)}")
            return config_name, -1
    
    def generate_test_configs(self) -> List[str]:
        """
        Generate test configuration files for all discovered models.
        
        Returns:
            List of generated test configuration file paths
        """
        # Discover all models
        models = self._discover_models()
        test_config_paths = []
        
        for model_name, run_name, model_path in models:
            # Find matching config
            config_path = self._find_model_config_match(model_name)
            if config_path is None:
                logger.warning(f"Skipping {model_name}/{run_name} - no config match found")
                continue
            
            # Update config for testing
            updated_config = self._update_config_for_testing(config_path, model_path)
            
            # Generate test config filename
            filename = self._generate_test_config_filename(model_name, run_name)
            test_config_path = self.output_dir / filename
            
            # Save test configuration
            with open(test_config_path, 'w') as f:
                json.dump(updated_config, f, indent=4)
            
            test_config_paths.append(str(test_config_path))
            logger.info(f"Generated test config for {model_name}/{run_name}")
        
        return test_config_paths
    
    def run_parallel_testing(self, max_parallel: int = None) -> Dict[str, int]:
        """
        Run testing in parallel for all generated configurations.
        
        Args:
            max_parallel: Maximum number of parallel processes (default: 4)
            
        Returns:
            Dictionary mapping config names to return codes
        """
        if max_parallel is None:
            max_parallel = 4  # Default to 4 parallel processes
        
        # Generate all test configurations
        test_config_paths = self.generate_test_configs()
        
        if not test_config_paths:
            logger.warning("No test configurations generated. Exiting.")
            return {}
        
        logger.info(f"Starting parallel testing with {len(test_config_paths)} configurations using {max_parallel} processes")
        
        results = {}
        
        # Use ProcessPoolExecutor for parallel execution
        with ProcessPoolExecutor(max_workers=max_parallel) as executor:
            # Submit all testing jobs
            future_to_config = {}
            for i, config_path in enumerate(test_config_paths):
                gpu_id = i % max_parallel
                future = executor.submit(self._run_testing, config_path, gpu_id)
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
        Print a summary of testing results.
        
        Args:
            results: Dictionary mapping config names to return codes
        """
        if not results:
            logger.info("No results to summarize.")
            return
            
        successful = sum(1 for code in results.values() if code == 0)
        failed = len(results) - successful
        
        logger.info("=" * 50)
        logger.info("TESTING SUMMARY")
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
    """Main function to run parallel testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run parallel testing for all trained models')
    parser.add_argument('--configs-path', default='saved_confs/parallel_configs', type=str,
                       help='Path to the parallel configs directory')
    parser.add_argument('--max-parallel', type=int, default=None,
                       help='Maximum number of parallel processes')
    parser.add_argument('--generate-only', action='store_true',
                       help='Only generate test configs, do not run testing')
    
    args = parser.parse_args()
    
    # Initialize parallel tester
    tester = ParallelTester(args.configs_path)
    
    if args.generate_only:
        # Only generate test configurations
        test_config_paths = tester.generate_test_configs()
        logger.info(f"Generated {len(test_config_paths)} test configuration files in {tester.output_dir}")
    else:
        # Run parallel testing
        results = tester.run_parallel_testing(args.max_parallel)
        tester.print_summary(results)


if __name__ == "__main__":
    main() 