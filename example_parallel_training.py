#!/usr/bin/env python3
"""
Example script demonstrating how to use the parallel training functionality.

This script shows different ways to use the ParallelTrainer class for
hyperparameter optimization and parallel model training.
"""

import logging
from pathlib import Path
from 4b_parallel_training import ParallelTrainer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_generate_configs_only():
    """Example: Generate configuration files without running training."""
    logger.info("Example 1: Generating configuration files only")
    
    # Initialize trainer with base config
    trainer = ParallelTrainer("config.json")
    
    # Generate all configuration files
    config_paths = trainer.generate_configs()
    
    logger.info(f"Generated {len(config_paths)} configuration files")
    logger.info(f"Files saved in: {trainer.output_dir}")
    
    # Show first few generated files
    for i, config_path in enumerate(config_paths[:3]):
        logger.info(f"  {i+1}. {Path(config_path).name}")
    
    return config_paths


def example_run_parallel_training():
    """Example: Run parallel training with all configurations."""
    logger.info("Example 2: Running parallel training")
    
    # Initialize trainer
    trainer = ParallelTrainer("config.json")
    
    # Run parallel training (uses n_gpu from config.json)
    results = trainer.run_parallel_training()
    
    # Print summary
    trainer.print_summary(results)
    
    return results


def example_custom_parallel_training():
    """Example: Run parallel training with custom number of parallel processes."""
    logger.info("Example 3: Custom parallel training with 2 processes")
    
    # Initialize trainer
    trainer = ParallelTrainer("config.json")
    
    # Run with custom number of parallel processes
    results = trainer.run_parallel_training(max_parallel=2)
    
    # Print summary
    trainer.print_summary(results)
    
    return results


def example_filtered_combinations():
    """Example: Show how to modify parameter combinations for specific testing."""
    logger.info("Example 4: Custom parameter combinations")
    
    # Initialize trainer
    trainer = ParallelTrainer("config.json")
    
    # Modify parameter combinations for faster testing
    # (This would require modifying the _define_parameter_combinations method)
    logger.info("Original parameter combinations:")
    original_combinations = trainer._define_parameter_combinations()
    logger.info(f"  Total combinations: {len(original_combinations)}")
    
    # Show first combination as example
    if original_combinations:
        first_combo = original_combinations[0]
        logger.info("  Example combination:")
        for key, value in first_combo.items():
            logger.info(f"    {key}: {value}")


def main():
    """Main function demonstrating different usage examples."""
    logger.info("Parallel Training Examples")
    logger.info("=" * 40)
    
    try:
        # Example 1: Generate configs only
        example_generate_configs_only()
        logger.info("")
        
        # Example 2: Show parameter combinations
        example_filtered_combinations()
        logger.info("")
        
        # Example 3: Custom parallel training (commented out to avoid actual training)
        # example_custom_parallel_training()
        logger.info("Note: Actual training examples are commented out to avoid execution")
        logger.info("Uncomment the relevant lines to run actual training")
        
    except Exception as e:
        logger.error(f"Error in examples: {str(e)}")
        raise


if __name__ == "__main__":
    main() 