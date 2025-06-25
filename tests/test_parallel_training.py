#!/usr/bin/env python3
"""
Test suite for parallel training functionality.

This module contains pytest functions to test the ParallelTrainer class
and its various methods for generating configurations and running parallel training.
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import subprocess

from 4b_parallel_training import ParallelTrainer


class TestParallelTrainer:
    """Test class for ParallelTrainer functionality."""
    
    @pytest.fixture
    def temp_config(self) -> str:
        """Create a temporary configuration file for testing."""
        config = {
            "name": "TestConfig",
            "n_gpu": 2,
            "arch": {
                "type": "MultiStreamTransformerModel",
                "args": {
                    "weather_time_dims": 7,
                    "prev_pollutant_hours": 16,
                    "weather_fields": 8,
                    "input_features": 66,
                    "weather_embedding_size": 128,
                    "pollution_embedding_size": 64,
                    "attention_heads": 2,
                    "lat_size": 25,
                    "lon_size": 25,
                    "dropout": 0.1,
                    "weather_transformer_blocks": 5,
                    "pollution_transformer_blocks": 5
                }
            },
            "data_loader": {
                "type": "MLforecastDataLoader",
                "args": {
                    "data_folder": "/test/data",
                    "norm_params_file": "/test/norm.pkl",
                    "years": [2020, 2021],
                    "pollutants_to_keep": ["co", "nodos", "otres"],
                    "prev_pollutant_hours": 16,
                    "prev_weather_hours": 4,
                    "next_weather_hours": 2,
                    "auto_regresive_steps": 24,
                    "bootstrap_enabled": True,
                    "bootstrap_repetition": 20,
                    "bootstrap_threshold": 3.0,
                    "batch_size": 2048,
                    "shuffle": True,
                    "validation_split": 0.1,
                    "num_workers": 4
                }
            },
            "trainer": {
                "epochs": 100,
                "save_dir": "/test/output/",
                "save_period": 1,
                "verbosity": 2,
                "monitor": "min val_loss",
                "early_stop": 20,
                "tensorboard": True,
                "log_dir": "saved/runs",
                "auto_regresive_steps": 24,
                "epochs_before_increase_auto_regresive_steps": 4
            },
            "test": {
                "data_loader": {
                    "auto_regresive_steps": 24
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config, f)
            return f.name
    
    @pytest.fixture
    def trainer(self, temp_config: str) -> ParallelTrainer:
        """Create a ParallelTrainer instance for testing."""
        return ParallelTrainer(temp_config)
    
    def test_define_parameter_combinations(self, trainer: ParallelTrainer) -> None:
        """Test that parameter combinations are generated correctly."""
        combinations = trainer._define_parameter_combinations()
        
        # Should generate combinations
        assert len(combinations) > 0
        
        # Check that each combination has all required keys
        required_keys = {
            'prev_pollutant_hours', 'attention_heads', 'weather_transformer_blocks',
            'pollution_transformer_blocks', 'pollutants_to_keep', 'bootstrap_enabled',
            'bootstrap_threshold', 'auto_regresive_steps'
        }
        
        for combo in combinations:
            assert all(key in combo for key in required_keys)
    
    def test_update_config(self, trainer: ParallelTrainer) -> None:
        """Test that configuration is updated correctly with new parameters."""
        params = {
            'prev_pollutant_hours': 8,
            'attention_heads': 4,
            'weather_transformer_blocks': 3,
            'pollution_transformer_blocks': 3,
            'pollutants_to_keep': ['otres', 'nox'],
            'bootstrap_enabled': False,
            'bootstrap_threshold': 3,
            'auto_regresive_steps': 16
        }
        
        updated_config = trainer._update_config(trainer.base_config, params)
        
        # Check that parameters were updated correctly
        assert updated_config['arch']['args']['prev_pollutant_hours'] == 8
        assert updated_config['arch']['args']['attention_heads'] == 4
        assert updated_config['arch']['args']['weather_transformer_blocks'] == 3
        assert updated_config['arch']['args']['pollution_transformer_blocks'] == 3
        assert updated_config['data_loader']['args']['pollutants_to_keep'] == ['otres', 'nox']
        assert updated_config['data_loader']['args']['bootstrap_enabled'] == False
        assert updated_config['data_loader']['args']['bootstrap_threshold'] == 3
        assert updated_config['data_loader']['args']['auto_regresive_steps'] == 16
        assert updated_config['trainer']['auto_regresive_steps'] == 16
        assert updated_config['test']['data_loader']['auto_regresive_steps'] == 16
    
    def test_generate_config_filename(self, trainer: ParallelTrainer) -> None:
        """Test that configuration filenames are generated correctly."""
        params = {
            'prev_pollutant_hours': 8,
            'attention_heads': 4,
            'weather_transformer_blocks': 3,
            'pollution_transformer_blocks': 3,
            'pollutants_to_keep': ['otres', 'nox', 'no'],
            'bootstrap_enabled': True,
            'bootstrap_threshold': 2,
            'auto_regresive_steps': 16
        }
        
        filename = trainer._generate_config_filename(params)
        
        # Check that filename contains expected components
        assert 'parallel_' in filename
        assert 'otres_nox_no' in filename
        assert 'prev8' in filename
        assert 'heads4' in filename
        assert 'w3' in filename
        assert 'p3' in filename
        assert 'ar16' in filename
        assert 'bootstrapTrue' in filename
        assert 'thresh2' in filename
        assert filename.endswith('.json')
    
    def test_generate_configs(self, trainer: ParallelTrainer) -> None:
        """Test that configuration files are generated correctly."""
        # Create a temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer.output_dir = Path(temp_dir)
            
            config_paths = trainer.generate_configs()
            
            # Should generate configuration files
            assert len(config_paths) > 0
            
            # Check that files exist and are valid JSON
            for config_path in config_paths:
                assert Path(config_path).exists()
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    assert 'name' in config
                    assert 'arch' in config
                    assert 'data_loader' in config
    
    @patch('subprocess.run')
    def test_run_training_success(self, mock_run: MagicMock, trainer: ParallelTrainer) -> None:
        """Test successful training execution."""
        # Mock successful subprocess run
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_run.return_value = mock_result
        
        config_path = "/test/config.json"
        gpu_id = 0
        
        config_name, return_code = trainer._run_training(config_path, gpu_id)
        
        assert return_code == 0
        mock_run.assert_called_once()
    
    @patch('subprocess.run')
    def test_run_training_failure(self, mock_run: MagicMock, trainer: ParallelTrainer) -> None:
        """Test failed training execution."""
        # Mock failed subprocess run
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "Training failed"
        mock_run.return_value = mock_result
        
        config_path = "/test/config.json"
        gpu_id = 0
        
        config_name, return_code = trainer._run_training(config_path, gpu_id)
        
        assert return_code == 1
    
    @patch('subprocess.run')
    def test_run_training_timeout(self, mock_run: MagicMock, trainer: ParallelTrainer) -> None:
        """Test training timeout."""
        # Mock timeout exception
        mock_run.side_effect = subprocess.TimeoutExpired(cmd=['python'], timeout=3600)
        
        config_path = "/test/config.json"
        gpu_id = 0
        
        config_name, return_code = trainer._run_training(config_path, gpu_id)
        
        assert return_code == -1
    
    def test_print_summary(self, trainer: ParallelTrainer, capsys: pytest.CaptureFixture) -> None:
        """Test that summary is printed correctly."""
        results = {
            'config1': 0,  # Success
            'config2': 0,  # Success
            'config3': 1,  # Failure
            'config4': -1  # Error
        }
        
        trainer.print_summary(results)
        
        captured = capsys.readouterr()
        output = captured.out
        
        # Check that summary information is present
        assert "TRAINING SUMMARY" in output
        assert "Total configurations: 4" in output
        assert "Successful: 2" in output
        assert "Failed: 2" in output
        assert "Success rate: 50.0%" in output


def test_parameter_combinations_filtering() -> None:
    """Test that invalid parameter combinations are filtered out."""
    # Create a minimal config for testing
    config = {
        "n_gpu": 2,
        "arch": {"args": {}},
        "data_loader": {"args": {}},
        "trainer": {},
        "test": {"data_loader": {}}
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config, f)
        config_path = f.name
    
    trainer = ParallelTrainer(config_path)
    combinations = trainer._define_parameter_combinations()
    
    # Check that combinations with bootstrap disabled have threshold = 3
    for combo in combinations:
        if not combo['bootstrap_enabled']:
            assert combo['bootstrap_threshold'] == 3


if __name__ == "__main__":
    pytest.main([__file__]) 