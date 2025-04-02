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
)

# %% Configuración Inicial
model_name_user = "TestPSpyt"
cur_pollutant = "otres"
cur_station = "MER"
now = datetime.utcnow().strftime("%Y_%m_%d_%H_%M")
model_name = f"{model_name_user}_{cur_pollutant}_{now}"

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
end_year = 2012
test_year = 2013

batch_size = 2048  # 32  # 4096

# Dataset features and additional configurations
sequence_length = 12
num_ar_features = 106
num_flag_features = 106
num_var_features = 12
output_dim = num_ar_features
num_hours_to_forecast = 48
num_points = 200  # The number of data points to plot
target_column_index = 0  # The column from Y_df_train_tensor to plot


# Training parameters
learning_rate_base = 0.001
num_epochs_base = 5000
patience_base = 600

learning_rate_ar = 0.0005
num_epochs_ar = 5000
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
    "cont_pmdiez_",
    "cont_pmdoscinco_",
    "cont_nodos_",
    "cont_co_",
    "i_",
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
            {"none": 1, "row_avg": 2, "last_day_same_hour": 3}
        )

print(data_imputed_subset.head())


# %% Preprocessing steps functions
def preprocessing_data_step0(data, gen_col_csv=True, file_name_norm=None):
    """Preprocessing data"""
    if file_name_norm:

        data_norm_df = normalizeDataWithLoadedScaler(data, file_name_norm)
    else:
        file_name_norm = join(norm_folder, f"{model_name}_scaler.pkl")
        print("Normalizing data....")
        data_norm_df = normalizeData(data, "mean_zero", file_name_norm)

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
        """
        Forward pass for the AutoregressivePredictor.

        Args:
            initial_input_ar (torch.Tensor): Initial AR input sequence. Shape: [batch_size, expected_ar_dim]
            initial_input_flag (torch.Tensor): Initial flag input sequence. Shape: [batch_size, expected_flag_dim]
            initial_input_var (torch.Tensor): Initial var input sequence. Shape: [batch_size, expected_var_dim]
            X_data (torch.Tensor): Full AR data tensor.
            F_data (torch.Tensor): Full flag data tensor.
            V_data (torch.Tensor): Full var data tensor.
            sequence_start_idx (int): Starting index for the sequence.

        Returns:
            torch.Tensor: Stacked predictions for the forecasted hours.
        """
        batch_size = initial_input_ar.size(0)
        predictions = []

        # Ensure inputs have the correct dimensions
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

            # Update the input for the next step
            current_input_ar, current_input_flag, current_input_var = (
                self.update_inputs(
                    pred, X_data, F_data, V_data, current_idx, batch_size
                )
            )
            current_idx += 1

        return torch.stack(predictions, dim=1)

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
        num_ar_features=106,
        num_flag_features=106,
        num_var_features=12,
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


# %% SOME TRAINING PARAMETERS AND PROCESSINGS DECLARATIONS
# Verify and APPLYING DataParallel if multiple GPUs are available <- IMPORTANT
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    base_predictor = nn.DataParallel(base_predictor)

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
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

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
##########################################################
# FIRST TRAINING LOOP FOR THE BASE MODEL
##########################################################
# FUNCTION DECLARATIONS FOR MORE TRAINING STEPS ON BASE MODEL   1hr
##########################################################
# Define the optimizer and loss function
optimizer = optim.Adam(base_predictor.parameters(), lr=learning_rate_base)
criterion = nn.MSELoss()


def train_base_predictor(
    base_predictor,
    train_loader,
    device,
    model_name,
    criterion,
    optimizer,
    num_epochs_base=100,
    learning_rate=0.01,
):
    """
    Trains the base predictor model.

    Args:
        base_predictor (nn.Module): The base predictor model.
        train_loader (DataLoader): The DataLoader for the training data.
        device (str): The device to use ('cuda' or 'cpu').
        model_name (str): The name of the model for TensorBoard logging.
        num_epochs (int, optional): The number of training epochs. Defaults to 100.
        learning_rate (float, optional): The learning rate for the optimizer. Defaults to 0.01.

    Returns:
        nn.Module: The trained base predictor model.
    """
    # Initialize TensorBoard writer
    log_dir = f"./tensorboard_logs/{model_name}_base_model"
    writer = SummaryWriter(log_dir=log_dir)

    # Training loop
    for epoch in range(num_epochs_base):
        epoch_loss = 0.0
        for x_batch, f_batch, v_batch, y_batch in train_loader:
            x_batch, f_batch, v_batch, y_batch = (
                x_batch.to(device),
                f_batch.to(device),
                v_batch.to(device),
                y_batch.to(device),
            )

            # Reshape the input tensors to match the expected input shape of the model
            x_batch = x_batch.reshape(x_batch.size(0), -1)
            f_batch = f_batch.reshape(f_batch.size(0), -1)
            v_batch = v_batch.reshape(v_batch.size(0), -1)

            optimizer.zero_grad()
            outputs = base_predictor(x_batch, f_batch, v_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Log the average loss for the epoch
        avg_loss = epoch_loss / len(train_loader)
        writer.add_scalar("Loss/train", avg_loss, epoch)

        print(f"Epoch: {epoch+1}/{num_epochs_base}, Loss: {avg_loss:.4f}")

    # Close the TensorBoard writer
    writer.close()

    return base_predictor


# %%
base_predictor = train_base_predictor(
    base_predictor,
    train_loader,
    device,
    model_name,
    criterion,
    optimizer,
    num_epochs_base=num_epochs_base,
    learning_rate=learning_rate_base,
)

# %%
plot_time_series_predictions(
    base_predictor,
    X_df_train_ar_tensor,
    X_df_train_flag_tensor,
    X_df_train_var_tensor,
    Y_df_train_tensor,
    sequence_length,
    device,
    num_points,
    target_column_index,
)

# %%
##########################################################
# SAVING OF WEIGHTS FOR THE BASE(AR) MODEL
##########################################################
save_base_predictor(base_predictor, model_name)

# %%
