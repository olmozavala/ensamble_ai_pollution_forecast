# Parallel Training for Air Pollution Forecasting

This module provides functionality to run parallel training with different hyperparameter combinations for the air pollution forecasting model.

## Overview

The parallel training system consists of:

- **`4b_parallel_training.py`**: Main parallel training script
- **`test_parallel_training.py`**: Test suite with pytest functions
- **`example_parallel_training.py`**: Example usage demonstrations
- **`saved_confs/parallel_configs/`**: Directory where generated configs are saved

## Features

- **Automatic Configuration Generation**: Creates multiple config files with different parameter combinations
- **Parallel Execution**: Runs multiple training jobs simultaneously using available GPUs
- **Comprehensive Logging**: Detailed logging of training progress and results
- **Error Handling**: Robust error handling with timeout protection
- **Summary Reporting**: Provides summary of successful and failed training runs

## Parameter Combinations Tested

The system tests the following parameter combinations:

| Parameter | Values |
|-----------|--------|
| `prev_pollutant_hours` | 4, 8, 16, 24 |
| `attention_heads` | 2, 4 |
| `weather_transformer_blocks` | 3, 5 |
| `pollution_transformer_blocks` | 3, 5 |
| `pollutants_to_keep` | `["otres"]`, `["otres", "nox", "no"]`, all pollutants |
| `bootstrap_enabled` | true, false |
| `bootstrap_threshold` | 2, 3 |
| `auto_regresive_steps` | 4, 8, 16, 24 |

## Usage

### Basic Usage

```bash
# Run parallel training with all parameter combinations
python 4b_parallel_training.py

# Generate config files only (without running training)
python 4b_parallel_training.py --generate-only

# Use custom number of parallel processes
python 4b_parallel_training.py --max-parallel 2

# Use different base configuration
python 4b_parallel_training.py --config my_config.json
```

### Command Line Arguments

- `--config`: Base configuration file path (default: `config.json`)
- `--max-parallel`: Maximum number of parallel processes (default: `n_gpu` from config)
- `--generate-only`: Only generate configuration files, do not run training

### Programmatic Usage

```python
from 4b_parallel_training import ParallelTrainer

# Initialize trainer
trainer = ParallelTrainer("config.json")

# Generate configuration files
config_paths = trainer.generate_configs()

# Run parallel training
results = trainer.run_parallel_training(max_parallel=4)

# Print summary
trainer.print_summary(results)
```

## Usage Examples

### Example 1: Quick Configuration Review

Before running full training, review the generated configurations:

```bash
# Generate all configuration files without running training
python 4b_parallel_training.py --generate-only

# Check how many configurations were generated
ls saved_confs/parallel_configs/ | wc -l

# Review a few sample configurations
ls saved_confs/parallel_configs/ | head -10

# Examine a specific configuration file
cat saved_confs/parallel_configs/parallel_otres_prev8_heads2_w3_p3_ar4_bootstrapTrue_thresh2.json
```

**Expected Output:**
```
2025-06-25 16:48:38,016 - INFO - Generated 1152 configuration files in saved_confs/parallel_configs
1152
parallel_all_pollutants_prev16_heads2_w3_p3_ar16_bootstrapFalse_thresh3.json
parallel_all_pollutants_prev16_heads2_w3_p3_ar16_bootstrapTrue_thresh2.json
...
```

### Example 2: Limited Parallel Training

Run training with a reduced number of parallel processes for testing:

```bash
# Run with only 2 parallel processes (useful for testing or limited GPU resources)
python 4b_parallel_training.py --max-parallel 2

# Monitor progress
tail -f parallel_training.log

# Check GPU usage
nvidia-smi
```

### Example 3: Custom Base Configuration

Use a different base configuration file:

```bash
# Create a custom base configuration
cp config.json my_custom_config.json

# Edit the custom configuration as needed
# (modify parameters, paths, etc.)

# Run parallel training with custom base config
python 4b_parallel_training.py --config my_custom_config.json
```

### Example 4: Programmatic Usage for Custom Workflows

```python
from 4b_parallel_training import ParallelTrainer
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Initialize with custom configuration
trainer = ParallelTrainer("my_custom_config.json")

# Generate configurations
print("Generating configurations...")
config_paths = trainer.generate_configs()
print(f"Generated {len(config_paths)} configurations")

# Run training with specific number of parallel processes
print("Starting parallel training...")
results = trainer.run_parallel_training(max_parallel=3)

# Analyze results
print("Training completed. Summary:")
trainer.print_summary(results)

# Filter successful configurations
successful_configs = [name for name, code in results.items() if code == 0]
print(f"Successful configurations: {len(successful_configs)}")
```

### Example 5: Monitoring and Debugging

```bash
# Monitor training progress in real-time
tail -f parallel_training.log

# Check specific training job logs
ls saved/runs/

# Monitor GPU usage
watch -n 1 nvidia-smi

# Check disk space usage
du -sh saved_confs/parallel_configs/
du -sh /unity/f1/ozavala/DATA/AirPollution/OUTPUT/

# Kill all training processes if needed
pkill -f "4_train.py"
```

### Example 6: Batch Processing with Different Configurations

```bash
# Run different parameter subsets
# First, test with only 2 attention heads
python 4b_parallel_training.py --max-parallel 4

# After completion, modify the script to test only 4 attention heads
# Edit 4b_parallel_training.py and change attention_heads to [4]
# Then run again
python 4b_parallel_training.py --max-parallel 4
```

### Example 7: Error Recovery and Restart

```bash
# If training was interrupted, check what completed
grep "Training completed successfully" parallel_training.log

# Restart training (it will skip already completed configurations)
python 4b_parallel_training.py

# Or run specific configurations manually
python 4_train.py -c saved_confs/parallel_configs/parallel_otres_prev8_heads2_w3_p3_ar4_bootstrapTrue_thresh2.json
```

### Example 8: Performance Optimization

```bash
# For memory-constrained systems, reduce batch size in config.json
# Then run with fewer parallel processes
python 4b_parallel_training.py --max-parallel 2

# For faster testing, modify parameter ranges in the script
# Edit _define_parameter_combinations() to use fewer values
# Then run
python 4b_parallel_training.py --generate-only
```

### Example 9: Integration with Existing Workflow

```bash
# Generate configurations
python 4b_parallel_training.py --generate-only

# Run training in background
nohup python 4b_parallel_training.py > training_output.log 2>&1 &

# Monitor background process
tail -f training_output.log

# Check process status
ps aux | grep 4b_parallel_training
```

### Example 10: Results Analysis

```bash
# After training completion, analyze results
python 6_analyze.py

# Compare different configurations
# (You may need to create a custom analysis script)

# Generate summary report
echo "Training Summary:" > training_summary.txt
grep "TRAINING SUMMARY" parallel_training.log >> training_summary.txt
cat training_summary.txt
```

## Configuration File Structure

Generated configuration files are saved in `saved_confs/parallel_configs/` with descriptive names that include the parameter values:

```
parallel_otres_nox_no_prev8_heads4_w3_p3_ar16_bootstrapTrue_thresh2.json
```

The naming convention is:
- `parallel_` prefix
- Pollutant names (first 3 or "all_pollutants")
- `prev{hours}` for previous pollutant hours
- `heads{num}` for attention heads
- `w{num}` for weather transformer blocks
- `p{num}` for pollution transformer blocks
- `ar{num}` for auto-regressive steps
- `bootstrap{bool}` for bootstrap enabled
- `thresh{num}` for bootstrap threshold

## GPU Management

The system automatically manages GPU usage:

1. Uses `CUDA_VISIBLE_DEVICES` to assign specific GPUs to each training process
2. Distributes training jobs across available GPUs in round-robin fashion
3. Respects the `n_gpu` setting from the base configuration
4. Supports custom parallel process limits

## Logging

The system provides comprehensive logging:

- **Console Output**: Real-time progress updates
- **Log File**: `parallel_training.log` with detailed information
- **Training Logs**: Each training process generates its own logs via `4_train.py`

## Error Handling

The system includes robust error handling:

- **Timeout Protection**: 1-hour timeout per training job
- **Process Isolation**: Each training job runs in a separate process
- **Error Recovery**: Failed jobs don't affect other running jobs
- **Summary Reporting**: Detailed report of successful and failed jobs

## Testing

Run the test suite to verify functionality:

```bash
# Run all tests
pytest test_parallel_training.py -v

# Run specific test
pytest test_parallel_training.py::TestParallelTrainer::test_generate_configs -v
```

## Example Workflow

1. **Prepare Base Configuration**: Ensure your `config.json` is properly configured
2. **Generate Configs**: Run with `--generate-only` to review generated configurations
3. **Run Training**: Execute full parallel training
4. **Monitor Progress**: Check logs and console output
5. **Review Results**: Use the summary report to identify successful configurations

## Monitoring and Debugging

### Check Training Progress

```bash
# Monitor log file
tail -f parallel_training.log

# Check generated configs
ls -la saved_confs/parallel_configs/

# Monitor GPU usage
nvidia-smi
```

### Common Issues

1. **Out of Memory**: Reduce `max_parallel` or batch size
2. **Timeout Errors**: Increase timeout in `_run_training` method
3. **Configuration Errors**: Check generated config files for syntax errors
4. **GPU Conflicts**: Ensure `n_gpu` matches available GPUs

## Performance Considerations

- **Memory Usage**: Each parallel process loads the full model and data
- **GPU Memory**: Monitor GPU memory usage with `nvidia-smi`
- **Disk Space**: Generated configs and training outputs require significant space
- **Network**: Ensure stable network connection for data loading

## Customization

### Adding New Parameters

To test additional parameters, modify the `_define_parameter_combinations` method:

```python
param_options = {
    'new_parameter': [value1, value2, value3],
    # ... existing parameters
}
```

### Modifying Parameter Ranges

Adjust the parameter values in `_define_parameter_combinations`:

```python
'prev_pollutant_hours': [8, 16],  # Reduced range for faster testing
'attention_heads': [2],           # Single value for focused testing
```

### Custom Configuration Updates

Modify the `_update_config` method to handle new parameters:

```python
# Update new parameter in configuration
config['new_section']['args']['new_parameter'] = params['new_parameter']
```

## Dependencies

- Python 3.7+
- PyTorch
- NumPy
- Pathlib
- Concurrent.futures
- Subprocess
- Logging
- Pytest (for testing)

## License

This module follows the same license as the main project. 