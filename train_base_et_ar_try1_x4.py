#!/usr/bin/env python
# coding: utf-8
# %% Version 0.1 de modelo base predictor
#########################################################
# imports, y declacraciones de variables de configuración:
##########################################################
import sys
from os.path import join

# Append custom utility path
sys.path.append("./eoas_pyutils")

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset  # , TensorDataset
from torchsummary import summary

# TensorBoard import
from torch.utils.tensorboard import SummaryWriter

# Scientific and Data Analysis Libraries
from datetime import datetime

# Import from project-specific modules
from proj_io.inout import (
    create_folder,
    add_previous_hours,
    save_columns,
)

# Import constants
from conf.localConstants import constants

# Import from refactoring utility
from refactoring import (
    apply_bootstrap,
    split_train_validation_and_test,
    normalizeData,
    normalizeDataWithLoadedScaler,
    filter_data,
    load_imputed_data,
    add_forecasted_hours,
    plot_time_series_predictions,
    save_base_predictor,
    evaluate_and_plot_ar_predictions_new,
)

# %%
import torch
import os
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import time

# %% Configuración Inicial
model_name_user = "TestPSpyt"
experi_id = "24hwin_md_lr005v2_LossMasked_train_et_ar_try1_x3"
cur_pollutant = "otres"
cur_station = "MER"
now = datetime.utcnow().strftime("%Y_%m_%d_%H_%M")
model_name = f"{model_name_user}_{experi_id}_{cur_pollutant}_{now}"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# grid_size = 4
# merged_specific_folder = f"{grid_size*grid_size}"

data_folder = "/ZION/AirPollutionData/Data/"
input_folder = join(data_folder, "MergedDataCSV/16/BK2/")
output_folder = join(data_folder, "TrainingTestsPS2024")
norm_folder = join(output_folder, "norm")
split_info_folder = join(output_folder, "Splits")
imputed_files_folder = "/ZION/AirPollutionData/Data/MergedDataCSV/16/Imputed/bk1"


# Stations data
stations_2020 = [
    "UIZ",
    "AJU",
    "ATI",
    "CUA",
    "SFE",
    "SAG",
    "CUT",
    "PED",
    "TAH",
    "GAM",
    "IZT",
    "CCA",
    "HGM",
    "LPR",
    "MGH",
    "CAM",
    "FAC",
    "TLA",
    "MER",
    "XAL",
    "LLA",
    "TLI",
    "UAX",
    "BJU",
    "MPA",
    "MON",
    "NEZ",
    "INN",
    "AJM",
    "VIF",
]
stations = stations_2020
pollutants = "cont_otres"

# Preprocessing and train vars
hours_before = 0  # 24
replace_nan_value = 0
forecasted_hours = 1
norm_type = "meanzero var 1"
val_perc = 0.1
test_perc = 0
bootstrap = True
boostrap_factor = 15
boostrap_threshold = 2.9
start_year = 2010
end_year = 2013
test_year = 2019

batch_size = 2048  # 32  # 4096

# Dataset features and additional configurations
sequence_length = 24
num_ar_features = 58
num_flag_features = 58
num_var_features = 12
output_dim = num_ar_features
num_hours_to_forecast = 6
num_points = 200  # The number of data points to plot
target_column_index = 0  # The column from Y_df_train_tensor to plot


# Training parameters
learning_rate_base = 0.0025
num_epochs_base = 250 # 5000
patience_base = 600

learning_rate_ar = 0.0001
num_epochs_ar = 100 # 000
patience_ar = 600

# %% Creación de carpetas necesarias
folders = ["Splits", "Parameters", "models", "logs", "imgs", "norm"]
for folder in folders:
    create_folder(join(output_folder, folder))

# %%
##########################################################
# Load the imputed data
data_imputed_df = load_imputed_data(start_year, end_year, imputed_files_folder)
print(data_imputed_df)

print(data_imputed_df.tail())

data_imputed = data_imputed_df

# %% Preprocessing setup
# Filtrar las columnas imputadas y sus flags solo para los grupos específicos
selected_prefixes = [
    "cont_otres_",
    "cont_nodos_",
    "i_cont_otres_",
    "i_cont_nodos_"
]
additional_columns = [
    "half_sin_day",
    "half_cos_day",
    "half_sin_week",
    "half_cos_week",
    "half_sin_year",
    "half_cos_year",
    "sin_day",
    "cos_day",
    "sin_week",
    "cos_week",
    "sin_year",
    "cos_year",
]
imputed_columns_filtered = [
    col
    for col in data_imputed.columns
    if any(col.startswith(prefix) for prefix in selected_prefixes)
    and (col in data_imputed.columns or col.endswith("_i"))
] + additional_columns

# Crear el subset con las columnas seleccionadas
data_imputed_subset = data_imputed[imputed_columns_filtered].copy()

# Reordenar las columnas: primero las columnas seleccionadas, luego additional_columns, y finalmente las que empiezan con "i_"
ordered_columns = (
    [
        col
        for col in imputed_columns_filtered
        if not col.startswith("i_") and col not in additional_columns
    ]
    + additional_columns
    + [col for col in imputed_columns_filtered if col.startswith("i_")]
)
data_imputed_subset = data_imputed_subset[ordered_columns]

# Mostrar información del subset
print("Subset de datos con columnas imputadas y sus flags:")
print(data_imputed_subset.info())
print(data_imputed_subset.head())

# %%
for each in data_imputed_subset.columns:
    print(each)


# %% convertir valores 'none' en flags a 1, los valores 'row_avg'a 2, 'last_day_same_hour' a 3
# Convertir valores 'none' en flags a 1, los valores 'row_avg' a 2, 'last_day_same_hour' a 3
for col in data_imputed_subset.columns:
    if col.startswith("i_"):
        data_imputed_subset[col] = data_imputed_subset[col].replace(
            #{"none": 1, "row_avg": 2, "last_day_same_hour": 3}
            {"none": 1, "row_avg": 0, "last_day_same_hour": 0}
        )

print(data_imputed_subset.head())


# %% Preprocessing steps functions
def preprocessing_data_step0(data, gen_col_csv=True, file_name_norm=None):
    """Preprocessing data"""
    # if file_name_norm:

    #     data_norm_df = normalizeDataWithLoadedScaler(data, file_name_norm)
    # else:
    #     file_name_norm = join(norm_folder, f"{model_name}_scaler.pkl")
    #     print("Normalizing data....")
    #     data_norm_df = normalizeData(data, "mean_zero", file_name_norm)
    # Guardar el orden original de las columnas
    original_column_order = data.columns.tolist()
    
    # Separar las columnas de flags (i_) y las columnas de datos
    flag_columns = [col for col in data.columns if col.startswith("i_")]
    data_columns = [col for col in data.columns if not col.startswith("i_")]
    
    # Guardar las flags antes de normalizar
    flags_df = data[flag_columns].copy()
    
    # Normalizar solo las columnas de datos
    data_to_normalize = data[data_columns].copy()
    
    if file_name_norm:
        data_norm_df = normalizeDataWithLoadedScaler(data_to_normalize, file_name_norm)
    else:
        file_name_norm = join(norm_folder, f"{model_name}_scaler.pkl")
        print("Normalizing data....")
        data_norm_df = normalizeData(data_to_normalize, "mean_zero", file_name_norm)
    
    # Volver a combinar los datos normalizados con las flags sin normalizar
    combined_df = pd.concat([data_norm_df, flags_df], axis=1)
    
    # IMPORTANTE: Restaurar exactamente el mismo orden de columnas original
    data_norm_df = combined_df[original_column_order]
    
    # Verificar que el orden se mantuvo correctamente
    assert list(data_norm_df.columns) == original_column_order, "El orden de las columnas ha cambiado"
    
    # Here we remove all the data of other pollutants
    X_df = filter_data(
        data_norm_df,
        filter_type="single_pollutant",
        filtered_pollutant=["otres", "pmdiez", "pmdoscinco", "nodos"],
    )

    print(X_df.columns.values)
    print(
        f"X {X_df.shape}, Memory usage: {X_df.memory_usage().sum() / 1024 ** 2:02f} MB"
    )

    print("Building X...")
    X_df = add_previous_hours(X_df, hours_before=hours_before)

    print(
        "Building Y...:Adding the forecasted hours of the pollutant as the predicted column Y..."
    )
    Y_df = add_forecasted_hours(X_df, range(1, forecasted_hours + 1))

    X_df = X_df.iloc[hours_before:, :]
    Y_df = Y_df.iloc[hours_before:, :]
    column_y_csv = join(output_folder, "Y_columns.csv")
    column_x_csv = join(output_folder, "X_columns.csv")
    if gen_col_csv:
        save_columns(Y_df, column_y_csv)
        save_columns(X_df, column_x_csv)

    print("Done!")

    print(f"Original {data_norm_df.shape}")
    print(
        f"X {X_df.shape}, Memory usage: {X_df.memory_usage().sum() / 1024 ** 2:02f} MB"
    )
    print(
        f"Y {Y_df.shape}, Memory usage: {Y_df.memory_usage().sum() / 1024 ** 2:02f} MB"
    )

    return X_df, Y_df, column_x_csv, column_y_csv, file_name_norm

import pandas as pd
import numpy as np
def preprocessing_data_step1(X_df, Y_df):
    """Preprocessing data"""
    print("Splitting training and validation data by year....")
    splits_file = None
    # Here we remove the datetime indexes so we need to consider that
    train_idxs, val_idxs, _ = split_train_validation_and_test(
        len(X_df), val_perc, test_perc, shuffle_ids=False, file_name=splits_file
    )

    # Y_df.reset_index(drop=True, inplace=True)

    X_df_train = X_df.iloc[train_idxs]
    Y_df_train = Y_df.iloc[train_idxs]

    X_df_val = X_df.iloc[val_idxs]
    Y_df_val = Y_df.iloc[val_idxs]

    print(
        f"X train {X_df_train.shape}, Memory usage: {X_df_train.memory_usage().sum() / 1024 ** 2:02f} MB"
    )
    print(
        f"Y train {Y_df_train.shape}, Memory usage: {Y_df_train.memory_usage().sum() / 1024 ** 2:02f} MB"
    )
    print(
        f"X val {X_df_val.shape}, Memory usage: {X_df_val.memory_usage().sum() / 1024 ** 2:02f} MB"
    )
    print(
        f"Y val {Y_df_val.shape}, Memory usage: {Y_df_val.memory_usage().sum() / 1024 ** 2:02f} MB"
    )

    print("Done!")

    if bootstrap:
        # Bootstrapping the data
        station = "MER"
        print("Bootstrapping the data...")
        print(
            f"X train {X_df_train.shape}, Memory usage: {X_df_train.memory_usage().sum() / 1024 ** 2:02f} MB"
        )
        print(
            f"Y train {Y_df_train.shape}, Memory usage: {Y_df_train.memory_usage().sum() / 1024 ** 2:02f} MB"
        )
        X_df_train, Y_df_train = apply_bootstrap(
            X_df_train,
            Y_df_train,
            cur_pollutant,
            station,
            boostrap_threshold,
            forecasted_hours,
            boostrap_factor,
        )
        print(
            f"X train bootstrapped {X_df_train.shape}, Memory usage: {X_df_train.memory_usage().sum() / 1024 ** 2:02f} MB"
        )
        print(
            f"Y train bootstrapped {Y_df_train.shape}, Memory usage: {Y_df_train.memory_usage().sum() / 1024 ** 2:02f} MB"
        )
        print(
            f"X val {X_df_val.shape}, Memory usage: {X_df_val.memory_usage().sum() / 1024 ** 2:02f} MB"
        )
        print(
            f"Y val {Y_df_val.shape}, Memory usage: {Y_df_val.memory_usage().sum() / 1024 ** 2:02f} MB"
        )

    # Managing nan values..
    print(f"Replacing nan values with {replace_nan_value}...")
    X_df_train.fillna(replace_nan_value, inplace=True)
    X_df_val.fillna(replace_nan_value, inplace=True)
    Y_df_train.fillna(replace_nan_value, inplace=True)
    Y_df_val.fillna(replace_nan_value, inplace=True)

    print(f"Train examples: {X_df_train.shape[0]}")
    print(f"Validation examples {X_df_val.shape[0]}")

    print(type(X_df_val))
    print(len(X_df_val))
    return X_df_train, Y_df_train, X_df_val, Y_df_val


# %% Preprocesssing, normalize, bootstrap and split data
X_df, Y_df, column_x_csv, column_y_csv, file_name_norm = preprocessing_data_step0(
    data_imputed_subset
)
# Drop the last row to ensure data conformity between X_df and Y_df
X_df = X_df[:-1]
Y_df = Y_df[:-1]
print(file_name_norm)

# %%
X_df_train, Y_df_train, X_df_val, Y_df_val = preprocessing_data_step1(X_df, Y_df)


X_df_train = X_df_train.astype("float32")
Y_df_train = Y_df_train.astype("float32")
X_df_val = X_df_val.astype("float32")
Y_df_val = Y_df_val.astype("float32")
# %%
print(X_df_train.head())
print(X_df_train.tail())

# %% Conversion to PyTorch tensors
X_df_train_tensor = torch.tensor(X_df_train.values, dtype=torch.float32)
Y_df_train_tensor = torch.tensor(Y_df_train.values, dtype=torch.float32)
X_df_val_tensor = torch.tensor(X_df_val.values, dtype=torch.float32)
Y_df_val_tensor = torch.tensor(Y_df_val.values, dtype=torch.float32)

# Verification and dimensions
print(type(X_df_train_tensor), X_df_train_tensor.shape)
print(type(Y_df_train_tensor), Y_df_train_tensor.shape)


##########################################################
# %% # Define the base predictor model CLASS, AND AUTOREGRESSIVE PREDICTOR
##########################################################
class BasePredictor(nn.Module):
    #  modelo más deep propuesto por
    # T4
    def __init__(
        self,
        num_ar_features,
        num_flag_features,
        num_var_features,
        num_classes,
        sequence_length,
    ):
        super(BasePredictor, self).__init__()
        self.sequence_length = sequence_length

        # AR part
        self.ar_layers = nn.Sequential(
            nn.Linear(num_ar_features * sequence_length, 150),
            nn.ReLU(),
            nn.BatchNorm1d(150),
            nn.Dropout(0.5),
        )

        # Flag part
        self.flag_layers = nn.Sequential(
            nn.Linear(num_flag_features * sequence_length, 100),
            nn.ReLU(),
            nn.BatchNorm1d(100),
            nn.Dropout(0.5),
        )

        # Var part
        self.var_layers = nn.Sequential(
            nn.Linear(num_var_features * sequence_length, 50),
            nn.ReLU(),
            nn.BatchNorm1d(50),
            nn.Dropout(0.5),
        )

        # Combined deeper layers
        self.combined_layers = nn.Sequential(
            nn.Linear(150 + 100 + 50, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, ar_input, flag_input, var_input):
        ar_out = self.ar_layers(ar_input)
        flag_out = self.flag_layers(flag_input)
        var_out = self.var_layers(var_input)

        # Concatenate the outputs
        combined_out = torch.cat((ar_out, flag_out, var_out), dim=1)

        # Pass through deeper combined layers
        final_out = self.combined_layers(combined_out)
        return final_out


class AutoregressivePredictor(nn.Module):
    def __init__(
        self,
        base_predictor,
        num_ar_features,
        num_flag_features,
        num_var_features,
        output_dim,
        sequence_length,
        num_hours_to_forecast,
    ):
        super(AutoregressivePredictor, self).__init__()
        self.base_predictor = base_predictor
        self.num_ar_features = num_ar_features
        self.num_flag_features = num_flag_features
        self.num_var_features = num_var_features
        self.output_dim = output_dim
        self.sequence_length = sequence_length
        self.num_hours_to_forecast = num_hours_to_forecast

        # Calculate expected input dimensions for the base predictor
        self.expected_ar_dim = sequence_length * num_ar_features
        self.expected_flag_dim = sequence_length * num_flag_features
        self.expected_var_dim = sequence_length * num_var_features
        
        # Inicialización de pesos
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    def forward(
        self,
        initial_input_ar,
        initial_input_flag,
        initial_input_var,
        X_data,
        F_data,
        V_data,
        sequence_start_idx,
    ):
        batch_size = initial_input_ar.size(0)
        predictions = []
        flags = []

        # Validación de dimensiones
        if initial_input_ar.size(1) != self.expected_ar_dim:
            raise ValueError(
                f"Expected AR input dimension {self.expected_ar_dim}, but got {initial_input_ar.size(1)}"
            )
        if initial_input_flag.size(1) != self.expected_flag_dim:
            raise ValueError(
                f"Expected flag input dimension {self.expected_flag_dim}, but got {initial_input_flag.size(1)}"
            )
        if initial_input_var.size(1) != self.expected_var_dim:
            raise ValueError(
                f"Expected var input dimension {self.expected_var_dim}, but got {initial_input_var.size(1)}"
            )

        current_input_ar = initial_input_ar.clone()
        current_input_flag = initial_input_flag.clone()
        current_input_var = initial_input_var.clone()

        current_idx = sequence_start_idx + self.sequence_length

        for _ in range(self.num_hours_to_forecast):
            # Make prediction with the current input
            pred = self.base_predictor(
                current_input_ar, current_input_flag, current_input_var
            )
            predictions.append(pred)
            flags.append(current_input_flag[:, -self.num_flag_features:])

            # Update inputs for next step
            current_input_ar, current_input_flag, current_input_var = (
                self.update_inputs(
                    pred, X_data, F_data, V_data, current_idx, batch_size
                )
            )
            current_idx += 1

        return torch.stack(predictions, dim=1), torch.stack(flags, dim=1)

    def update_inputs(self, preds, X_data, F_data, V_data, current_idx, batch_size):
        """
        Updates the input sequences for the next autoregressive step (simplified).

        Args:
            preds (torch.Tensor): Predictions from the base predictor.
            X_data (torch.Tensor): Full AR data tensor.
            F_data (torch.Tensor): Full flag data tensor.
            V_data (torch.Tensor): Full var data tensor.
            current_idx (int): Current index in the data tensors.
            batch_size (int): Batch size.

        Returns:
            tuple: Updated AR, flag, and var input sequences.
        """
        seq_len = self.sequence_length
        device = preds.device

        # --- AR Features ---
        # Determine if we're out of bounds for AR features
        ar_out_of_bounds = current_idx - seq_len + 1 < 0 or current_idx >= len(X_data)

        if ar_out_of_bounds:
            # If out of bounds, use zeros
            ar_seq = torch.zeros(
                (batch_size, seq_len - 1, self.num_ar_features), device=device
            )
        else:
            # Get the AR sequence (excluding the oldest time step)
            ar_seq = (
                X_data[current_idx - seq_len + 1 : current_idx, : self.num_ar_features]
                .unsqueeze(0)
                .repeat_interleave(batch_size, dim=0)
                .to(device)
            )

        # Pad with zeros if needed
        if ar_seq.size(1) < seq_len - 1:
            pad_size = seq_len - 1 - ar_seq.size(1)
            ar_padding = torch.zeros(
                (batch_size, pad_size, self.num_ar_features), device=device
            )
            ar_seq = torch.cat([ar_padding, ar_seq], dim=1)

        # Add the new prediction as the latest time step
        ar_new_seq = torch.cat([ar_seq, preds.unsqueeze(1)], dim=1)
        ar_batch = ar_new_seq.reshape(batch_size, -1)

        # --- Flag and Var Features (Refactored) ---
        def get_features(data, num_features):
            # Determine if we're out of bounds
            out_of_bounds = current_idx >= len(data) or current_idx + seq_len > len(
                data
            )
            if out_of_bounds:
                # If out of bounds, use zeros
                features = torch.zeros(
                    (batch_size, seq_len, num_features), device=device
                )
            else:
                # Get the sequence
                features = (
                    data[current_idx : current_idx + seq_len, :num_features]
                    .unsqueeze(0)
                    .repeat_interleave(batch_size, dim=0)
                    .to(device)
                )

            # Pad with zeros if needed
            if features.size(1) < seq_len:
                available = max(0, len(data) - current_idx)
                pad_size = seq_len - available
                padding = torch.zeros(
                    (batch_size, pad_size, num_features), device=device
                )
                features = torch.cat([features, padding], dim=1)

            return features.reshape(batch_size, -1)

        flag_batch = get_features(F_data, self.num_flag_features)
        var_batch = get_features(V_data, self.num_var_features)

        return ar_batch, flag_batch, var_batch


##########################################################
# %% Define the dataset and dataloader classes
##########################################################
class TimeSeriesDataset(Dataset):
    def __init__(
        self,
        X_data,
        F_data,
        V_data,
        Y_data,
        sequence_length,
        num_ar_features=num_ar_features,
        num_flag_features=num_flag_features,
        num_var_features=num_var_features,
    ):
        self.X_data = X_data
        self.F_data = F_data
        self.V_data = V_data
        self.Y_data = Y_data
        self.sequence_length = sequence_length
        self.num_ar_features = num_ar_features
        self.num_flag_features = num_flag_features
        self.num_var_features = num_var_features

    def __len__(self):
        return len(self.X_data) - self.sequence_length

    def __getitem__(self, idx):
        X_sequence = self.X_data[idx : idx + self.sequence_length, :]
        F_sequence = self.F_data[idx : idx + self.sequence_length, :]
        V_sequence = self.V_data[idx : idx + self.sequence_length, :]
        Y_target = self.Y_data[idx + self.sequence_length, :]

        X_sequence_flat = X_sequence.reshape(-1)
        F_sequence_flat = F_sequence.reshape(-1)
        V_sequence_flat = V_sequence.reshape(-1)

        return X_sequence_flat, F_sequence_flat, V_sequence_flat, Y_target


##########################################################
# Define the dataset and dataloader classes FOR THE AR MODEL TRAINING
##########################################################
class AutoregressiveTimeSeriesDataset(Dataset):
    def __init__(
        self,
        X_ar_data,
        X_flag_data,
        X_var_data,
        Y_data,
        sequence_length,
        num_hours_to_forecast,
        num_ar_features,
        num_flag_features,
        num_var_features,
    ):
        """
        Dataset for autoregressive time series forecasting with separate AR, flag, and var inputs.

        Args:
            X_ar_data (torch.Tensor): Tensor containing the AR features.
            X_flag_data (torch.Tensor): Tensor containing the flag features.
            X_var_data (torch.Tensor): Tensor containing the var features.
            Y_data (torch.Tensor): Tensor containing the target values.
            sequence_length (int): Length of the input sequence.
            num_hours_to_forecast (int): Number of hours to forecast.
            num_ar_features (int): Number of AR features.
            num_flag_features (int): Number of flag features.
            num_var_features (int): Number of var features.
        """
        self.X_ar_data = X_ar_data
        self.X_flag_data = X_flag_data
        self.X_var_data = X_var_data
        self.Y_data = Y_data
        self.sequence_length = sequence_length
        self.num_hours_to_forecast = num_hours_to_forecast
        self.num_ar_features = num_ar_features
        self.num_flag_features = num_flag_features
        self.num_var_features = num_var_features

    def __len__(self):
        """
        Returns the length of the dataset.
        """
        return (
            len(self.X_ar_data) - self.sequence_length - self.num_hours_to_forecast + 1
        )

    def __getitem__(self, idx):
        """
        Retrieves a data sample at the specified index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the flattened AR sequence, flattened flag sequence,
                   flattened var sequence, and the target values for the forecasted hours.
        """
        # Get the AR sequence
        X_ar_sequence = self.X_ar_data[idx : idx + self.sequence_length, :]
        X_ar_sequence_flat = X_ar_sequence.reshape(-1)

        # Get the flag sequence
        X_flag_sequence = self.X_flag_data[idx : idx + self.sequence_length, :]
        X_flag_sequence_flat = X_flag_sequence.reshape(-1)

        # Get the var sequence
        X_var_sequence = self.X_var_data[idx : idx + self.sequence_length, :]
        X_var_sequence_flat = X_var_sequence.reshape(-1)

        # Get the target values for the forecasted hours
        Y_targets = self.Y_data[
            idx
            + self.sequence_length : idx
            + self.sequence_length
            + self.num_hours_to_forecast,
            :,
        ]

        return X_ar_sequence_flat, X_flag_sequence_flat, X_var_sequence_flat, Y_targets


##########################################################
# %% BASE PREDICTOR INSTANTIATION
##########################################################
base_predictor = BasePredictor(
    num_ar_features, num_flag_features, num_var_features, output_dim, sequence_length
).to(device)

# Print the model summary
print("Base predictor model summary:")
summary(
    base_predictor,
    [
        (num_ar_features * sequence_length,),
        (num_flag_features * sequence_length,),
        (num_var_features * sequence_length,),
    ],
)
print(f"sequence_length: {sequence_length}")
print(f"num_ar_features: {num_ar_features}")
print(f"num_flag_features: {num_flag_features}")
print(f"num_var_features: {num_var_features}")
print(f"output_dim: {output_dim}")


# %% SOME TRAINING PARAMETERS AND PROCESSINGS DECLARATIONS
# Verify and APPLYING DataParallel if multiple GPUs are available <- IMPORTANT
# Removing DataParallel due to issues when reloading weights.
# if torch.cuda.device_count() > 1:
#     if not isinstance(base_predictor, torch.nn.DataParallel):
#         print(f"Using {torch.cuda.device_count()} GPUs!")
#         base_predictor = nn.DataParallel(base_predictor)

# Create the dataset and dataloader
# Assuming you have X_df_train_tensor, Y_df_train_tensor, X_df_val_tensor, Y_df_val_tensor
# Let's split X_df_train_tensor into X_df_train_ar_tensor, X_df_train_flag_tensor and X_df_train_var_tensor
X_df_train_ar_tensor = X_df_train_tensor[:, :num_ar_features].to(device)
X_df_train_flag_tensor = X_df_train_tensor[
    :, num_ar_features : num_ar_features + num_flag_features
].to(device)
X_df_train_var_tensor = X_df_train_tensor[:, -num_var_features:].to(device)

X_df_val_ar_tensor = X_df_val_tensor[:, :num_ar_features].to(device)
X_df_val_flag_tensor = X_df_val_tensor[
    :, num_ar_features : num_ar_features + num_flag_features
].to(device)
X_df_val_var_tensor = X_df_val_tensor[:, -num_var_features:].to(device)

Y_df_train_tensor = Y_df_train_tensor[:, :num_ar_features].to(device)
Y_df_val_tensor = Y_df_val_tensor[:, :num_ar_features].to(device)

train_dataset = TimeSeriesDataset(
    X_df_train_ar_tensor,
    X_df_train_flag_tensor,
    X_df_train_var_tensor,
    Y_df_train_tensor,
    sequence_length,
    num_ar_features,
    num_flag_features,
    num_var_features,
)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Supongamos que hemos cargado X_df_val_tensor y Y_df_val_tensor
val_dataset = TimeSeriesDataset(
    X_df_val_ar_tensor,
    X_df_val_flag_tensor,
    X_df_val_var_tensor,
    Y_df_val_tensor,
    sequence_length,
    num_ar_features,
    num_flag_features,
    num_var_features,
)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# %%

class ObservedOnlyMSELoss(nn.Module):
    def __init__(self, observed_flag=1, weight_factor=1/2.7):
        super(ObservedOnlyMSELoss, self).__init__()
        self.observed_flag = observed_flag
        self.weight_factor = weight_factor
        
    def forward(self, predictions, targets, flags):
        """
        Calcula MSE considerando valores observados (donde flags == observed_flag)
        y valores imputados con un factor de peso reducido.
        
        Args:
            predictions: tensor de predicciones del modelo
            targets: tensor de valores reales
            flags: tensor de banderas (1 para valores observados/none, otros valores para imputados)
            
        Returns:
            loss: pérdida MSE considerando valores observados y ponderando valores imputados
        """
        # Crear máscara para valores observados (flags == observed_flag)
        mask = (flags == self.observed_flag).float()
        
        # Crear máscara para valores imputados (flags != observed_flag)
        non_mask = (flags != self.observed_flag).float()
        
        # Calcular error cuadrático para valores observados y no observados
        squared_error_observed = ((predictions - targets) ** 2) * mask
        squared_error_non_observed = ((predictions - targets) ** 2) * non_mask * self.weight_factor
        
        # Sumar ambos errores
        total_squared_error = squared_error_observed + squared_error_non_observed
        
        # Calcular suma y normalizar por el número total de valores
        num_total = torch.sum(mask + non_mask)
        
        # En caso de que no haya valores, devolver 0
        if num_total == 0:
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)
        
        loss = torch.sum(total_squared_error) / num_total
        return loss



# %%
##########################################################
# FIRST TRAINING LOOP FOR THE BASE MODEL
##########################################################
# FUNCTION DECLARATIONS FOR MORE TRAINING STEPS ON BASE MODEL   1hr
##########################################################
# Define the optimizer and loss function
optimizer = optim.Adam(base_predictor.parameters(), lr=learning_rate_base)
#criterion = nn.MSELoss()
criterion = ObservedOnlyMSELoss(observed_flag=1)  # 1 corresponde a "none" (valores observados)


# %%
def train_base_predictor_step(
    base_predictor,
    train_loader,
    val_loader,
    device,
    model_name,
    criterion,
    optimizer,
    num_epochs_base=250,
    learning_rate=0.01,
    pesos_name_seq=False,
    epoch_offset=0
):
    """
    Entrena el modelo base por num_epochs_base epochs adicionales.
    Guarda pesos cada ciclo y grafica el loss y val_loss.

    Args:
        pesos_name_seq (bool): Si True, guarda los pesos con sufijo de epoch. Si False, sobreescribe.
        epoch_offset (int): Offset para los epochs si se entrena por bloques.
    """
    start_time = time.time()  # Start timing

    # TensorBoard writer
    log_dir = f"./tensorboard_logs/{model_name}_base_model"
    writer = SummaryWriter(log_dir=log_dir)

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs_base):
        base_predictor.train()
        epoch_loss = 0.0
        for x_batch, f_batch, v_batch, y_batch in train_loader:
            x_batch, f_batch, v_batch, y_batch = (
                x_batch.to(device),
                f_batch.to(device),
                v_batch.to(device),
                y_batch.to(device),
            )

            x_batch = x_batch.reshape(x_batch.size(0), -1)
            f_batch = f_batch.reshape(f_batch.size(0), -1)
            v_batch = v_batch.reshape(v_batch.size(0), -1)

            optimizer.zero_grad()
            outputs = base_predictor(x_batch, f_batch, v_batch)
            f_batch_last_hour = f_batch[:, -num_ar_features:]

            # Calcular la pérdida solo con los flags correspondientes a la última hora
            loss = criterion(outputs, y_batch, f_batch_last_hour)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        writer.add_scalar("Loss/train", avg_train_loss, epoch_offset + epoch)
        train_losses.append(avg_train_loss)

        # Validation
        base_predictor.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, f_batch, v_batch, y_batch in val_loader:
                x_batch, f_batch, v_batch, y_batch = (
                    x_batch.to(device),
                    f_batch.to(device),
                    v_batch.to(device),
                    y_batch.to(device),
                )

                x_batch = x_batch.reshape(x_batch.size(0), -1)
                f_batch = f_batch.reshape(f_batch.size(0), -1)
                v_batch = v_batch.reshape(v_batch.size(0), -1)

                outputs = base_predictor(x_batch, f_batch, v_batch)
                f_batch_last_hour = f_batch[:, -num_ar_features:]

                # Calcular la pérdida solo con los flags correspondientes a la última hora
                loss = criterion(outputs, y_batch, f_batch_last_hour)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        writer.add_scalar("Loss/val", avg_val_loss, epoch_offset + epoch)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch_offset + epoch + 1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    writer.close()

    # Guardar pesos
    model_path = f"{model_name}_base_model.pth"
    if pesos_name_seq:
        model_path = f"{model_name}_base_model_ep{epoch_offset + num_epochs_base}.pth"
    torch.save(base_predictor.state_dict(), model_path)
    print(f"Pesos guardados en: {model_path}")

    # Evaluar predicciones vs observaciones
    plot_time_series_predictions(
        base_predictor,
        X_df_train_ar_tensor,
        X_df_train_flag_tensor,
        X_df_train_var_tensor,
        Y_df_train_tensor,
        sequence_length,
        device,
        num_points=200,
        target_column_index=0,  # ajusta esto si tu columna objetivo es distinta
    )

    # Leer eventos de TensorBoard para graficar
    ea = EventAccumulator(log_dir)
    ea.Reload()

    train_events = ea.Scalars("Loss/train")
    val_events = ea.Scalars("Loss/val")

    train_steps = [e.step for e in train_events]
    train_vals = [e.value for e in train_events]

    val_steps = [e.step for e in val_events]
    val_vals = [e.value for e in val_events]

    plt.figure(figsize=(10, 5))
    plt.plot(train_steps, train_vals, label="Train Loss")
    plt.plot(val_steps, val_vals, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # End timing and print duration
    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_time_hours = elapsed_time / 3600
    print(f"Tiempo total de ejecución paso: {elapsed_time_hours:.2f} horas")

# # %%
# model_path = f"{model_name}_base_model.pth"
# torch.save(base_predictor.state_dict(), model_path)
# print(f"Pesos guardados en: {model_path}")


print(base_predictor)
print(f"sequence_length: {sequence_length}")
print(f"num_ar_features: {num_ar_features}")
print(f"num_flag_features: {num_flag_features}")
print(f"num_var_features: {num_var_features}")
print(f"output_dim: {output_dim}")

# %%
# Paso 1
train_base_predictor_step(
    base_predictor, train_loader, val_loader, device, model_name,
    criterion, optimizer,
    num_epochs_base=num_epochs_base, pesos_name_seq=False, epoch_offset=0
)

# %%

# %% Paso 2
train_base_predictor_step(
    base_predictor, train_loader, val_loader, device, model_name,
    criterion, optimizer,
    num_epochs_base=num_epochs_base, pesos_name_seq=False, epoch_offset=num_epochs_base*1
)

# %%
train_base_predictor_step(
    base_predictor, train_loader, val_loader, device, model_name,
    criterion, optimizer,
    num_epochs_base=num_epochs_base, pesos_name_seq=False, epoch_offset=num_epochs_base*2
)



# %%

# # %% Paso 3
# train_base_predictor_step(
#     base_predictor, train_loader, val_loader, device, model_name,
#     criterion, optimizer,
#     num_epochs_base=250, pesos_name_seq=False, epoch_offset=500
# )




# Create the dataset and dataloader
# Assuming you have X_df_train_tensor, Y_df_train_tensor, X_df_val_tensor, Y_df_val_tensor
# Let's split X_df_train_tensor into X_df_train_ar_tensor, X_df_train_flag_tensor and X_df_train_var_tensor
X_df_train_ar_tensor = X_df_train_tensor[:, :num_ar_features].to(device)
X_df_train_flag_tensor = X_df_train_tensor[
    :, num_ar_features : num_ar_features + num_flag_features
].to(device)
X_df_train_var_tensor = X_df_train_tensor[:, -num_var_features:].to(device)

X_df_val_ar_tensor = X_df_val_tensor[:, :num_ar_features].to(device)
X_df_val_flag_tensor = X_df_val_tensor[
    :, num_ar_features : num_ar_features + num_flag_features
].to(device)
X_df_val_var_tensor = X_df_val_tensor[:, -num_var_features:].to(device)

Y_df_train_tensor = Y_df_train_tensor[:, :num_ar_features].to(device)
Y_df_val_tensor = Y_df_val_tensor[:, :num_ar_features].to(device)

# Crear los datasets AR
train_ar_dataset = AutoregressiveTimeSeriesDataset(
    X_df_train_ar_tensor,
    X_df_train_flag_tensor,
    X_df_train_var_tensor,
    Y_df_train_tensor,
    sequence_length,
    num_hours_to_forecast,
    num_ar_features,
    num_flag_features,
    num_var_features,
)

val_ar_dataset = AutoregressiveTimeSeriesDataset(
    X_df_val_ar_tensor,
    X_df_val_flag_tensor,
    X_df_val_var_tensor,
    Y_df_val_tensor,
    sequence_length,
    num_hours_to_forecast,
    num_ar_features,
    num_flag_features,
    num_var_features,
)

# Crear los dataloaders
train_ar_loader = DataLoader(train_ar_dataset, batch_size=batch_size, shuffle=True)
val_ar_loader = DataLoader(val_ar_dataset, batch_size=batch_size, shuffle=False)

# %% PLOT TEST OF BASE MODEL PREDICTIONS
model = base_predictor
plot_time_series_predictions(
    model,
    X_df_train_ar_tensor,
    X_df_train_flag_tensor,
    X_df_train_var_tensor,
    Y_df_train_tensor,
    sequence_length,
    device,
    200,
    target_column_index,
)
# %%







# %% ################################################################
# %% ################################################################





# %%
##########################################################
# FIRST TRAINING LOOP FOR THE AR MODEL
##########################################################
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import numpy as np
import time
def train_ar_predictor(
    ar_predictor,
    train_loader,
    device,
    model_name,
    criterion,
    optimizer,
    X_df_train_ar_tensor,
    X_df_train_flag_tensor,
    X_df_train_var_tensor,
    num_epochs_ar=100,
    learning_rate=0.0001,
    patience=20,
    val_loader=None,
    X_df_val_ar_tensor=None,
    X_df_val_flag_tensor=None,
    X_df_val_var_tensor=None,
):
    start_time = time.time()
    log_dir = f"./tensorboard_logs/{model_name}_ar_model"
    writer = SummaryWriter(log_dir=log_dir)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    def log_gradients(model, writer, epoch):
        for name, param in model.named_parameters():
            if param.grad is not None:
                writer.add_histogram(f'gradients/{name}', param.grad, epoch)

    for epoch in range(num_epochs_ar):
        epoch_loss = 0.0
        start_epoch_time = time.time()
        ar_predictor.train()

        for batch_idx, (x_ar_batch_flat, x_flag_batch_flat, x_var_batch_flat, y_batch) in enumerate(train_loader):
            x_ar_batch_flat = x_ar_batch_flat.to(device)
            x_flag_batch_flat = x_flag_batch_flat.to(device)
            x_var_batch_flat = x_var_batch_flat.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            
            # Obtener predicciones y flags
            predictions, prediction_flags = ar_predictor(
                x_ar_batch_flat,
                x_flag_batch_flat,
                x_var_batch_flat,
                X_df_train_ar_tensor,
                X_df_train_flag_tensor,
                X_df_train_var_tensor,
                0,
            )

            # Calcular pérdida para la predicción actual
            step_pred = predictions.squeeze(1)  # Eliminar dimensión extra si existe
            step_flags = prediction_flags.squeeze(1)  # Eliminar dimensión extra si existe
            loss = criterion(step_pred, y_batch, step_flags)
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(ar_predictor.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Log gradients
            if batch_idx % 100 == 0:
                log_gradients(ar_predictor, writer, epoch)

            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        writer.add_scalar("Loss/train", avg_train_loss, epoch)

        # Validación
        if val_loader is not None:
            val_loss = 0.0
            ar_predictor.eval()
            with torch.no_grad():
                for x_ar_val_batch_flat, x_flag_val_batch_flat, x_var_val_batch_flat, y_val_batch in val_loader:
                    x_ar_val_batch_flat = x_ar_val_batch_flat.to(device)
                    x_flag_val_batch_flat = x_flag_val_batch_flat.to(device)
                    x_var_val_batch_flat = x_var_val_batch_flat.to(device)
                    y_val_batch = y_val_batch.to(device)

                    val_predictions, val_flags = ar_predictor(
                        x_ar_val_batch_flat,
                        x_flag_val_batch_flat,
                        x_var_val_batch_flat,
                        X_df_val_ar_tensor,
                        X_df_val_flag_tensor,
                        X_df_val_var_tensor,
                        0,
                    )

                    # Calcular pérdida de validación
                    step_pred = val_predictions.squeeze(1)
                    step_flags = val_flags.squeeze(1)
                    val_step_loss = criterion(step_pred, y_val_batch, step_flags)
                    val_loss += val_step_loss.item()

            avg_val_loss = val_loss / len(val_loader)
            writer.add_scalar('Loss/val', avg_val_loss, epoch)

            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save(ar_predictor.state_dict(), f'{model_name}_ar_best.pth')
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping en epoch {epoch}")
                break

        # Print progress
        epoch_time = time.time() - start_epoch_time
        if (epoch + 1) % 1 == 0:
            if val_loader is not None:
                print(f"Epoch [{epoch + 1}/{num_epochs_ar}], "
                      f"Train Loss: {avg_train_loss:.4f}, "
                      f"Val Loss: {avg_val_loss:.4f}, "
                      f"Epoch Time: {epoch_time:.2f}s")
            else:
                print(f"Epoch [{epoch + 1}/{num_epochs_ar}], "
                      f"Train Loss: {avg_train_loss:.4f}, "
                      f"Epoch Time: {epoch_time:.2f}s")

    writer.close()
    print("AR model training finished.")

    # Cargar el mejor modelo
    ar_predictor.load_state_dict(torch.load(f'{model_name}_ar_best.pth'))
    
    # Guardar el modelo final
    save_base_predictor(ar_predictor, f'{model_name}_ar_final')

    total_time = time.time() - start_time
    print(f"\nTiempo total de entrenamiento: {total_time/3600:.2f} horas")

    return ar_predictor


# %% step 1
# Primero instanciamos el modelo AutoregressivePredictor
ar_predictor = AutoregressivePredictor(
    base_predictor=base_predictor,  # Usamos el base_predictor ya entrenado
    num_ar_features=num_ar_features,
    num_flag_features=num_flag_features,
    num_var_features=num_var_features,
    output_dim=output_dim,
    sequence_length=sequence_length,
    num_hours_to_forecast=num_hours_to_forecast
).to(device)

# Ahora sí podemos configurar el entrenamiento
#criterion = nn.MSELoss()  # Usando MSE regular para el modelo AR
criterion = ObservedOnlyMSELoss(observed_flag=1)  # 1 corresponde a "none" (valores observados)
optimizer = optim.Adam(ar_predictor.parameters(), lr=0.0001)  # Learning rate reducido

# Y procedemos con el entrenamiento
ar_predictor = train_ar_predictor(
    ar_predictor,
    train_ar_loader,  # Usar train_ar_loader en lugar de train_loader
    device,
    model_name,
    criterion,
    optimizer,
    X_df_train_ar_tensor,
    X_df_train_flag_tensor,
    X_df_train_var_tensor,
    num_epochs_ar=num_epochs_ar,
    learning_rate=0.0001,
    patience=20,
    val_loader=val_ar_loader,  # Usar val_ar_loader en lugar de val_loader
    X_df_val_ar_tensor=X_df_val_ar_tensor,
    X_df_val_flag_tensor=X_df_val_flag_tensor,
    X_df_val_var_tensor=X_df_val_var_tensor,
)

# %% step 2
# Hyperparameters
# Train the AR model
ar_predictor = train_ar_predictor(
    ar_predictor,
    train_ar_loader,
    device,
    model_name,
    criterion,
    optimizer,
    X_df_train_ar_tensor,
    X_df_train_flag_tensor,
    X_df_train_var_tensor,
    num_epochs_ar=num_epochs_ar,
    learning_rate=0.0001,
    patience=20,
    val_loader=val_ar_loader,
    X_df_val_ar_tensor=X_df_val_ar_tensor,
    X_df_val_flag_tensor=X_df_val_flag_tensor,
    X_df_val_var_tensor=X_df_val_var_tensor,
)

# %% step 3
# Hyperparameters
# Train the AR model
ar_predictor = train_ar_predictor(
    ar_predictor,
    train_ar_loader,
    device,
    model_name,
    criterion,
    optimizer,
    X_df_train_ar_tensor,
    X_df_train_flag_tensor,
    X_df_train_var_tensor,
    num_epochs_ar = num_epochs_ar,
    learning_rate=0.0001,
    patience=20,
    val_loader=val_ar_loader,
    X_df_val_ar_tensor=X_df_val_ar_tensor,
    X_df_val_flag_tensor=X_df_val_flag_tensor,
    X_df_val_var_tensor=X_df_val_var_tensor,
)

# # %% step 4
# # Hyperparameters
# # Train the AR model
# ar_predictor = train_ar_predictor(
#     ar_predictor,
#     train_ar_loader,
#     device,
#     model_name,
#     criterion,
#     optimizer,
#     X_df_train_ar_tensor,
#     X_df_train_flag_tensor,
#     X_df_train_var_tensor,
#     num_epochs_ar=250,
#     learning_rate=0.0001,
#     patience=20,
#     val_loader=val_ar_loader,
#     X_df_val_ar_tensor=X_df_val_ar_tensor,
#     X_df_val_flag_tensor=X_df_val_flag_tensor,
#     X_df_val_var_tensor=X_df_val_var_tensor,
# )

# %% step 3

def plot_ar_predictions_7days(
    ar_predictor,
    X_df_train_ar_tensor,
    X_df_train_flag_tensor,
    X_df_train_var_tensor,
    Y_df_train_tensor,
    sequence_length,
    device,
    start_idx=0,
    num_hours=168,  # 7 días * 24 horas
    target_column_index=0
):
    """
    Plots a specific 7-day sequence of predictions from the AR predictor against actual values.

    Args:
        ar_predictor: The trained AutoregressivePredictor model.
        X_df_train_ar_tensor: The input tensor for AR features.
        X_df_train_flag_tensor: The input tensor for Flag features.
        X_df_train_var_tensor: The input tensor for Var features.
        Y_df_train_tensor: The target data tensor.
        sequence_length: The length of the input sequences.
        device: The device to use ('cuda' or 'cpu').
        start_idx: Starting index for the sequence.
        num_hours: Number of hours to plot (default: 168 for 7 days).
        target_column_index: The index of the column in Y_df_train_tensor to plot.
    """
    ar_predictor.eval()  # Set the model to evaluation mode
    
    with torch.no_grad():
        # Prepare input sequences
        input_ar = X_df_train_ar_tensor[start_idx:start_idx + sequence_length].reshape(-1).unsqueeze(0).to(device)
        input_flag = X_df_train_flag_tensor[start_idx:start_idx + sequence_length].reshape(-1).unsqueeze(0).to(device)
        input_var = X_df_train_var_tensor[start_idx:start_idx + sequence_length].reshape(-1).unsqueeze(0).to(device)
        
        # Get predictions
        predictions, _ = ar_predictor(
            input_ar,
            input_flag,
            input_var,
            X_df_train_ar_tensor,
            X_df_train_flag_tensor,
            X_df_train_var_tensor,
            start_idx
        )
        
        # Get actual values
        actual_values = Y_df_train_tensor[start_idx + sequence_length:start_idx + sequence_length + num_hours, target_column_index]
        
        # Plot
        plt.figure(figsize=(20, 6))
        plt.plot(actual_values.cpu().numpy(), label='Valores Reales', marker='o', linestyle='-')
        plt.plot(predictions[0, :num_hours, target_column_index].cpu().numpy(), label='Predicciones', marker='x', linestyle='-')
        plt.title(f'Predicciones AR vs Valores Reales - 7 días (168 horas)')
        plt.xlabel('Hora')
        plt.ylabel('Valor')
        plt.legend()
        plt.grid(True)
        plt.show()


# %%

plot_ar_predictions_7days(
    ar_predictor,
    X_df_train_ar_tensor,
    X_df_train_flag_tensor,
    X_df_train_var_tensor,
    Y_df_train_tensor,
    sequence_length,
    device,
    start_idx=35,
    num_hours=168,  # 7 días * 24 horas
    target_column_index=0
)

# %%

plot_ar_predictions_7days(
    ar_predictor,
    X_df_train_ar_tensor,
    X_df_train_flag_tensor,
    X_df_train_var_tensor,
    Y_df_train_tensor,
    sequence_length,
    device,
    start_idx=0,
    num_hours=168,  # 7 días * 24 horas
    target_column_index=0
)

# %% Configuración para el nuevo predictor AR de 24 horas
num_hours_to_forecast_24h = 24  # Nuevo número de horas a predecir

# Instanciar el nuevo predictor AR de 24 horas
ar_predictor_24h = AutoregressivePredictor(
    base_predictor=base_predictor,  # Reutilizamos el base predictor entrenado
    num_ar_features=num_ar_features,
    num_flag_features=num_flag_features,
    num_var_features=num_var_features,
    output_dim=output_dim,
    sequence_length=sequence_length,
    num_hours_to_forecast=num_hours_to_forecast_24h
).to(device)

# Configurar el optimizador y criterio para el nuevo modelo
optimizer_24h = optim.Adam(ar_predictor_24h.parameters(), lr=0.0001)
criterion_24h = criterion

# %%

# # %% Paso 3
# train_base_predictor_step(
#     base_predictor, train_loader, val_loader, device, model_name,
#     criterion, optimizer,
#     num_epochs_base=250, pesos_name_seq=False, epoch_offset=500
# )




# Create the dataset and dataloader
# Assuming you have X_df_train_tensor, Y_df_train_tensor, X_df_val_tensor, Y_df_val_tensor
# Let's split X_df_train_tensor into X_df_train_ar_tensor, X_df_train_flag_tensor and X_df_train_var_tensor
X_df_train_ar_tensor = X_df_train_tensor[:, :num_ar_features].to(device)
X_df_train_flag_tensor = X_df_train_tensor[
    :, num_ar_features : num_ar_features + num_flag_features
].to(device)
X_df_train_var_tensor = X_df_train_tensor[:, -num_var_features:].to(device)

X_df_val_ar_tensor = X_df_val_tensor[:, :num_ar_features].to(device)
X_df_val_flag_tensor = X_df_val_tensor[
    :, num_ar_features : num_ar_features + num_flag_features
].to(device)
X_df_val_var_tensor = X_df_val_tensor[:, -num_var_features:].to(device)

Y_df_train_tensor = Y_df_train_tensor[:, :num_ar_features].to(device)
Y_df_val_tensor = Y_df_val_tensor[:, :num_ar_features].to(device)

# Crear los datasets AR
train_ar_dataset = AutoregressiveTimeSeriesDataset(
    X_df_train_ar_tensor,
    X_df_train_flag_tensor,
    X_df_train_var_tensor,
    Y_df_train_tensor,
    sequence_length,
    num_hours_to_forecast_24h,
    num_ar_features,
    num_flag_features,
    num_var_features,
)

val_ar_dataset = AutoregressiveTimeSeriesDataset(
    X_df_val_ar_tensor,
    X_df_val_flag_tensor,
    X_df_val_var_tensor,
    Y_df_val_tensor,
    sequence_length,
    num_hours_to_forecast_24h,
    num_ar_features,
    num_flag_features,
    num_var_features,
)

# Crear los dataloaders
train_ar_loader = DataLoader(train_ar_dataset, batch_size=batch_size, shuffle=True)
val_ar_loader = DataLoader(val_ar_dataset, batch_size=batch_size, shuffle=False)

# %%

# Entrenar el nuevo modelo AR de 24 horas
ar_predictor_24h = train_ar_predictor(
    ar_predictor_24h,
    train_ar_loader,
    device,
    model_name + "_24h",  # Nombre diferente para el modelo de 24h
    criterion_24h,
    optimizer_24h,
    X_df_train_ar_tensor,
    X_df_train_flag_tensor,
    X_df_train_var_tensor,
    num_epochs_ar=100,
    learning_rate=0.0001,
    patience=20,
    val_loader=val_ar_loader,
    X_df_val_ar_tensor=X_df_val_ar_tensor,
    X_df_val_flag_tensor=X_df_val_flag_tensor,
    X_df_val_var_tensor=X_df_val_var_tensor,
)


# %%

# Visualizar predicciones del nuevo modelo de 24 horas
plot_ar_predictions_7days(
    ar_predictor_24h,
    X_df_train_ar_tensor,
    X_df_train_flag_tensor,
    X_df_train_var_tensor,
    Y_df_train_tensor,
    sequence_length,
    device,
    start_idx=0,
    num_hours=168,  # 7 días * 24 horas
    target_column_index=0
)


# %%

# Guardar el modelo de 24 horas
save_base_predictor(ar_predictor_24h, f'{model_name}_ar_24h_final')

# %%
