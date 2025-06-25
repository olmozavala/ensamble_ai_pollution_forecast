from typing import List, Optional, Callable, Tuple
from proj_preproc.viz import visualize_pollutant_vs_weather_var, visualize_batch_data
from torch.utils.data import WeightedRandomSampler, SubsetRandomSampler
import os
from os.path import join
import numpy as np
import pandas as pd
from base import BaseDataLoader
from data_loader.data_sets import MLforecastDataset
import torch
import time

class MLforecastDataLoader(BaseDataLoader):
    """
    DataLoader for the MLforecastDataset that handles batching and shuffling.
    
    Args:
        dataset (MLforecastDataset): The dataset to load
        batch_size (int): Number of samples per batch
        shuffle (bool): Whether to shuffle the data
        validation_split (float): Fraction of data to use for validation
        num_workers (int): Number of worker processes for data loading
    """
    def __init__(self, 
                pollution_folder: str,
                weather_folder: str,
                norm_params_file: str,
                training_folder: str,
                years: List[int],
                pollutants_to_keep: List[str],
                prev_pollutant_hours: int,
                prev_weather_hours: int,
                next_weather_hours: int,
                auto_regresive_steps: int,
                bootstrap_enabled: bool,
                bootstrap_repetition: int,
                bootstrap_threshold: float,
                batch_size: int = 32,
                shuffle: bool = True,
                validation_split: float = 0.0,
                num_workers: int = 1) -> None:
        """Initialize the data loader."""
        
        # Store the dataset
        self.dataset = MLforecastDataset(
        pollution_folder=pollution_folder,
        weather_folder=weather_folder,
        norm_params_file=norm_params_file,
        training_folder=training_folder,
        years=years,
        pollutants_to_keep=pollutants_to_keep,
        prev_pollutant_hours=prev_pollutant_hours,
        prev_weather_hours=prev_weather_hours,
        next_weather_hours=next_weather_hours,
        auto_regresive_steps=auto_regresive_steps,
        bootstrap_enabled=bootstrap_enabled,
        bootstrap_repetition=bootstrap_repetition,
        bootstrap_threshold=bootstrap_threshold
    )
        
        # Create weighted sampler if bootstrap is enabled
        if hasattr(self.dataset, 'random_sampler_weights') and self.dataset.bootstrap_enabled:
            self.weights = self.dataset.random_sampler_weights
            self.sampler = WeightedRandomSampler(self.weights, len(self.weights), replacement=True)
            # Turn off shuffle when using sampler
            shuffle = False
        else:
            self.sampler = None
            self.weights = None

        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

    def _split_sampler(self, split):
        """Override to handle weighted sampling with validation split."""
        if split == 0.0:
            # No validation split, use weighted sampler if available
            if self.sampler is not None:
                return self.sampler, None
            return None, None

        # Handle validation split with weighted sampling
        if self.sampler is not None:
            # For weighted sampling with validation split, we need to create separate samplers
            # This is a simplified approach - you might want to implement more sophisticated validation splitting
            n_samples = len(self.dataset)
            idx_full = np.arange(n_samples)
            
            if isinstance(split, int):
                assert split > 0
                assert split < n_samples, "validation set size is configured to be larger than entire dataset."
                len_valid = split
            else:
                len_valid = int(n_samples * split)

            valid_idx = idx_full[-len_valid:]
            train_idx = np.delete(idx_full, np.arange(n_samples-len_valid, n_samples))
            
            # Create weighted sampler for training data only
            train_weights = self.weights[train_idx]
            train_sampler = WeightedRandomSampler(train_weights, len(train_weights), replacement=True)
            valid_sampler = SubsetRandomSampler(valid_idx)
            
            return train_sampler, valid_sampler
        else:
            # Fall back to original implementation for non-weighted sampling
            return super()._split_sampler(split)

    def get_pollution_dataframe(self) -> pd.DataFrame:
        """Get the pollution dataframe"""
        return self.dataset.pollution_data

    def get_pollution_column_names_and_indices(self, column_type: str) -> Tuple[List[str], List[int]]:
        """Get the column names and indices of the pollution dataframe"""
        return self.dataset.get_column_and_index_names(column_type)

    def get_sampling_info(self) -> dict:
        """Get information about the sampling configuration."""
        info = {
            'bootstrap_enabled': self.dataset.bootstrap_enabled,
            'total_samples': len(self.dataset),
            'using_weighted_sampling': self.sampler is not None
        }
        
        if self.dataset.bootstrap_enabled:
            info.update({
                'bootstrap_threshold': self.dataset.bootstrap_threshold,
                'bootstrap_repetition': self.dataset.bootstrap_repetition,
                'high_ozone_events': len(self.dataset.bootstrap_indexes),
                'weight_range': (self.weights.min(), self.weights.max()) if self.weights is not None else None
            })
        
        return info

if __name__ == '__main__':

    import matplotlib.pyplot as plt
    # Test parameters
    root_folder = "/home/olmozavala/DATA/AirPollution"
    pollution_folder = join(root_folder, "PollutionCSV")
    weather_folder = join(root_folder, "WRF_NetCDF")
    training_folder = join(root_folder, "TrainingData")
    years = [2015]

    start_year = min(years)
    end_year = max(years)

    norm_params_file = join(training_folder, f"norm_params_{start_year}_to_{end_year}.pkl")
    pollutants_to_keep = ['co', 'nodos', 'otres', 'pmdiez', 'pmdoscinco']
    # Data loader parameters
    batch_size = 2
    prev_pollutant_hours = 16 
    prev_weather_hours = 4
    next_weather_hours = 2
    auto_regresive_steps = 4
    bootstrap_enabled = True
    bootstrap_repetition = 20
    bootstrap_threshold = 3.0

    # Print the paramters
    print(f"Batch size: {batch_size}")
    print(f"Prev pollutant hours: {prev_pollutant_hours}")
    print(f"Prev weather hours: {prev_weather_hours}")
    print(f"Next weather hours: {next_weather_hours}")
    print(f"Auto regresive steps: {auto_regresive_steps}")

    # Create dataset


    # Create dataloader (weighted sampling is handled internally)
    data_loader = MLforecastDataLoader(
        pollution_folder=pollution_folder,
        weather_folder=weather_folder,
        norm_params_file=norm_params_file,
        training_folder=training_folder,
        years=years,
        pollutants_to_keep=pollutants_to_keep,
        prev_pollutant_hours=prev_pollutant_hours,
        prev_weather_hours=prev_weather_hours,
        next_weather_hours=next_weather_hours,
        auto_regresive_steps=auto_regresive_steps,
        bootstrap_enabled=bootstrap_enabled,
        bootstrap_repetition=bootstrap_repetition,
        bootstrap_threshold=bootstrap_threshold,
        batch_size=batch_size,
        shuffle=True
    )

    # Make a histogram of the weights
    plt.plot(data_loader.weights)
    plt.savefig("bootstrap_weights_plot.png")
    plt.show()

    
    # Test loading a batch
    pollution_column_names, pollution_column_indices = data_loader.get_pollution_column_names_and_indices("pollutant_only")
    imputed_mask_columns, imputed_mask_columns_indices = data_loader.get_pollution_column_names_and_indices("imputed_mask")
    time_related_columns, time_related_columns_indices = data_loader.get_pollution_column_names_and_indices("time")
    target_columns, target_columns_indices = data_loader.get_pollution_column_names_and_indices("target")

    # Print sampling information
    sampling_info = data_loader.get_sampling_info()
    print("\n=== Sampling Information ===")
    for key, value in sampling_info.items():
        print(f"  {key}: {value}")

    # Print the time related columns
    print(f"\nTime related columns: {time_related_columns}")
    print(f"Time related columns indices: {time_related_columns_indices}")

    for batch_idx, batch in enumerate(data_loader):
        # Print the shape of each element in the batch (x, y)
        print(f"Batch {batch_idx}")
        # Print total number of contaminants    
        print(f"  Total number of contaminants: {len(pollutants_to_keep)}")
        print(f"  x pollution shape: {batch[0][0].shape} (batch({batch_size}), prev_pollutant_hours({prev_pollutant_hours}), stations(30)(ozone) + (contaminants({len(pollutants_to_keep)}) - 1)*3(means, min, max) + time related columns(12))")
        print(f"  x weather shape: {batch[0][1].shape} (batch({batch_size}), prev_weather_hours({prev_weather_hours}) + next_weather_hours({next_weather_hours}) + auto_regresive_steps({auto_regresive_steps}) + 1, fields(8), lat(25), lon(25))")
        print(f"  y pollution shape: {batch[1][0].shape} (batch({batch_size}), auto_regresive_steps({auto_regresive_steps}), stations(30)(ozone) + (contaminants({len(pollutants_to_keep)}) - 1)*3(means, min, max))")
        print(f"  y imputed pollution shape: {batch[1][1].shape} (batch({batch_size}), auto_regresive_steps({auto_regresive_steps}), stations(30)(ozone))")

        # Here we can plot the data to be sure that the data is loaded correctly
        pollution_data = batch[0][0].numpy()[0,:,:]  # Final shape is (prev_pollutant_hours, stations(30)(ozone) + (contaminants - 1)*3(means, min, max) + time related columns(12))
        weather_data = batch[0][1].numpy()[0,:,:,:,:]  # Final shape is (prev_weather_hours + next_weather_hours + auto_regresive_steps + 1, fields, lat, lon)
        target_data = batch[1][0].numpy()[0,:,:]  # Final shape is (auto_regresive_steps, stations(30)(ozone) + (contaminants - 1)*3(means, min, max))
        imputed_data = batch[1][1].numpy()[0,:,:]  # Final shape is (auto_regresive_steps, stations(30)(ozone))
        current_datetime = pd.to_datetime(batch[2][0].item(), unit='s')

        # Plot the pollution data
        # contaminant_names = ["otres", "pmdiez", "nodos", "co", "pmdoscinco"]  # To plot multiple stations
        contaminant_names = ["co"]  # To plot multiple stations
        for contaminant_name in contaminant_names:
            # contaminant_name = "cont_otres_MER"  # To plot a single station
            weather_var_idx = 0
            weather_var_name = "T2"
            plot_pollutant_names = [name for name in pollution_column_names if contaminant_name == name.split('_')[1]]
            plot_pollutant_indices = [pollution_column_indices[i] for i, name in enumerate(pollution_column_names) if contaminant_name == name.split('_')[1]]
            plot_target_indices = [target_columns_indices[i] for i, name in enumerate(target_columns) if contaminant_name == name.split('_')[1]]
            
            # Create figure to plot pollution data features
            output_folder = "/home/olmozavala/DATA/AirPollution/TrainingData/batch_imgs"
            visualize_batch_data(pollution_data, 
                                target_data, 
                                imputed_data, 
                                weather_data, 
                                plot_pollutant_indices, 
                                plot_pollutant_names, 
                                target_columns, 
                                plot_target_indices, 
                                time_related_columns, 
                                time_related_columns_indices, 
                                weather_var_name, 
                                current_datetime, 
                                output_folder, 
                                batch_idx, 
                                prev_weather_hours, 
                                next_weather_hours, 
                                auto_regresive_steps, 
                                weather_var_idx, 
                                contaminant_name)
        if batch_idx > 1:
            break