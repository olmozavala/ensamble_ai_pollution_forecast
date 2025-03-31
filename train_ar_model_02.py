#!/usr/bin/env python
# coding: utf-8
# %% vamos a tratar de usar el dataset con datos imputados, y calcularlo solo con los datos autoreregresivos planeados e imputados:
##########################################################
# imports, y declacraciones de variables de configuración:,
##########################################################

# %%
# Set working directory
import os
import sys
#os.chdir("/home/pedro/git2/gitflow/air_pollution_forecast")

sys.path.append("./eoas_pyutils")

# %%
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
from torch.utils.data import Dataset, DataLoader
#from pytorch_proj import (
#    split_train_validation_and_test
#)
# apply_bootstrap
#     normalizeData,
#     plot_forecast_hours,
# %%
from sklearn import preprocessing


def save_splits(file_name, train_ids, val_ids, test_ids):
    """
    This function saves the training, validation and test indexes. It assumes that there are
    more training examples than validation and test examples. It also uses
    :param file_name:
    :param train_ids:
    :param val_ids:
    :param test_ids:
    :return:
    """
    print("Saving split information...")
    info_splits = DataFrame({F'Train({len(train_ids)})': train_ids})
    info_splits[F'Validation({len(val_ids)})'] = -1
    info_splits[F'Validation({len(val_ids)})'][0:len(val_ids)] = val_ids
    info_splits[F'Test({len(test_ids)})'] = -1
    info_splits[F'Test({len(test_ids)})'][0:len(test_ids)] = test_ids
    info_splits.to_csv(file_name, index=None)

def split_train_validation_and_test(num_examples, val_percentage, test_percentage, 
                                    shuffle_ids=True, file_name = ''):
    """
    Splits a number into training, validation, and test randomly
    :param num_examples: int of the number of examples
    :param val_percentage: int of the percentage desired for validation
    :param test_percentage: int of the percentage desired for testing
    :return:
    """
    all_samples_idxs = np.arange(num_examples)

    if shuffle_ids:
        np.random.shuffle(all_samples_idxs)

    test_examples = int(np.ceil(num_examples * test_percentage))
    val_examples = int(np.ceil(num_examples * val_percentage))
    # Train and validation indexes
    train_idxs = all_samples_idxs[0:len(all_samples_idxs) - test_examples - val_examples]
    val_idxs = all_samples_idxs[len(all_samples_idxs) - test_examples - val_examples:len(all_samples_idxs) - test_examples]
    test_idxs = all_samples_idxs[len(all_samples_idxs) - test_examples:]
    train_idxs.sort()
    val_idxs.sort()
    test_idxs.sort()

    if file_name != '':
        save_splits(file_name, train_idxs, val_idxs, test_idxs)


    return [train_idxs, val_idxs, test_idxs]


def apply_bootstrap(X_df, Y_df, contaminant, station, boostrap_threshold, forecasted_hours, boostrap_factor=1):
    '''
    This function will boostrap the data based on the threshold and the forecasted hours
    '''

    bootstrap_column = f"cont_{contaminant}_{station}"
    print("Bootstrapping the data...")
    # Searching all the index where X or Y is above the threshold

    # Adding index when the current time is above the threshold
    bootstrap_idx = X_df.loc[:,bootstrap_column] > boostrap_threshold

    # Searching index when any of the forecasted hours is above the threshold
    y_cols = Y_df.columns.values
    for i in range(1, forecasted_hours+1):
        # print(bootstrap_idx.sum())  
        c_column = f"plus_{i:02d}_{bootstrap_column}"
        if c_column in y_cols:
            bootstrap_idx = bootstrap_idx | (Y_df.loc[:, c_column] > boostrap_threshold)

    X_df = pd.concat([X_df, *[X_df[bootstrap_idx] for i in range(boostrap_factor)]])
    Y_df = pd.concat([Y_df, *[Y_df[bootstrap_idx] for i in range(boostrap_factor)]])

    return X_df, Y_df

def normalizeData(data, norm_type, file_name):
    if norm_type == "min_max":
        scaler = preprocessing.MinMaxScaler()
    if norm_type == "mean_zero":
        scaler = preprocessing.StandardScaler()

    scaler = scaler.fit(data)
    data_norm_np = scaler.transform(data)
    data_norm_df = DataFrame(data_norm_np, columns=data.columns, index=data.index)

    # ******************* Saving Normalization params, scaler object **********************
    scaler.path_file = file_name
    with open(scaler.path_file, 'wb') as f: #scaler.path_file must be defined during training.
        pickle.dump(scaler, f)
    print(f'Scaler/normalizer object saved to: {scaler.path_file}')
    print(F'Done! Current shape: {data_norm_df.shape} ')
    return data_norm_df



# %%
#from proj_prediction.prediction import analyze_column
#from proj_preproc.preproc import loadScaler

from proj_io.inout import (
    create_folder,
    add_previous_hours,
    get_column_names,
    read_merged_files,
    save_columns,
)


# %%
from conf.localConstants import constants
from torchsummary import summary
import matplotlib.pyplot as plt
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np
import pandas as pd
import pickle
import sys
from os.path import join
import os
from pandas import DataFrame
# Un intento de introducir forzamientos en modelos autorregresivos
# Se intenta forzar un modelo AR con forzamientos aplanados en modelos base



# %%

def filter_data(df, filter_type="none", filtered_pollutant="", filtered_station=""):
    """
    Filtra el DataFrame según el tipo de filtro especificado.
    Los posibles valores de filter_type son: 'none', 'single_pollutant', 'single_pollutant_and_station'
    """
    all_contaminant_columns, all_meteo_columns, all_time_colums = get_column_names(df)

    if filter_type == "single_pollutant":
        filtered_pollutants = (
            filtered_pollutant
            if isinstance(filtered_pollutant, list)
            else [filtered_pollutant]
        )
        keep_cols = [
            x
            for x in df.columns
            if any(pollutant in x for pollutant in filtered_pollutants)
        ]
        keep_cols += all_time_colums.tolist() + all_meteo_columns.tolist()

    elif filter_type == "single_pollutant_and_station":
        keep_cols = (
            [f"cont_{filtered_pollutant}_{filtered_station}"]
            + all_time_colums.tolist()
            + all_meteo_columns
        )

    elif filter_type == "none":
        return df.copy()

    print(f"Keeping columns: {len(keep_cols)} original columns: {len(df.columns)}")
    return df[keep_cols].copy()


# %% Configuración Inicial
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hours_before = 0  # 24
replace_nan_value = 0
data_folder = "/ZION/AirPollutionData/Data/"
grid_size = 4
merged_specific_folder = f"{grid_size*grid_size}"
input_folder = join(data_folder, "MergedDataCSV/16/BK2/")
output_folder = join(data_folder, "TrainingTestsPS2024")
norm_folder = join(output_folder, "norm")
split_info_folder = join(output_folder, "Splits")

val_perc = 0.1
test_perc = 0
epochs = 5000
batch_size = 2048 # 32  # 4096
bootstrap = True
boostrap_factor = 15
boostrap_threshold = 2.9
model_name_user = "TestPSpyt"
start_year = 2010
end_year = 2012
test_year = 2013
cur_pollutant = "otres"
cur_station = "MER"
forecasted_hours = 1
norm_type = "meanzero var 1"
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


imputed_files_folder = "/ZION/AirPollutionData/Data/MergedDataCSV/16/Imputed/bk1"

now = datetime.utcnow().strftime("%Y_%m_%d_%H_%M")
model_name = f"{model_name_user}_{cur_pollutant}_{now}"


# Configuración del Data set features
sequence_length = 12
num_ar_features = 106
num_flag_features = 106
num_var_features = 12
output_dim = num_ar_features
num_hours_to_forecast = 12
num_points = 200  # The number of data points to plot
target_column_index = 0  # The column from Y_df_train_tensor to plot

learning_rate_base = 0.0005
num_epochs_base = 5000
patience_base = 600

# Some training parameters
learning_rate_ar = 0.0005
num_epochs_ar = 5000
patience_ar = 600




# %% Creación de carpetas necesarias
folders = ["Splits", "Parameters", "models", "logs", "imgs", "norm"]
for folder in folders:
    create_folder(join(output_folder, folder))

##########################################################
# %% # Load the imputed data
# Function to load imputed data for all years and recreate the DataFrame
# LOAD THE DATA AND PREPROCESSING
##########################################################
def load_imputed_data(start_year, end_year, folder_path):
    all_data = []
    for year in range(start_year, end_year + 1):
        file_path = os.path.join(folder_path, f"data_imputed_{year}.csv")
        print(f"Loading data from {file_path}")
        yearly_data = pd.read_csv(file_path, index_col=0, parse_dates=True)
        all_data.append(yearly_data)
    data_imputed_df = pd.concat(all_data)
    data_imputed_df.index = pd.to_datetime(data_imputed_df.index)
    return data_imputed_df


data_imputed_df = load_imputed_data(start_year, end_year, imputed_files_folder)
print(data_imputed_df)

print(data_imputed_df.tail())

data_imputed = data_imputed_df

# %%
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


# %%
def preprocessing_data_step0(data, gen_col_csv=True, file_name_norm=None):
    """Preprocessing data"""
    if file_name_norm:

        def normalizeDataWithLoadedScaler(data, file_name):
            """
            Normalize data using a pre-saved scaler object.

            :param data: DataFrame to be normalized
            :param file_name: Path to the scaler .pkl file
            :return: Normalized DataFrame
            """
            # Load the scaler object from the file
            with open(file_name, "rb") as f:
                scaler = pickle.load(f)

            # Transform the data using the loaded scaler
            data_norm_np = scaler.transform(data)
            data_norm_df = pd.DataFrame(
                data_norm_np, columns=data.columns, index=data.index
            )

            print(f"Scaler/normalizer object loaded from: {file_name}")
            print(f"Done! Current shape: {data_norm_df.shape}")
            return data_norm_df

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

    def add_forecasted_hours(df, forecasted_hours=range(1, 25)):
        """
        This function adds the forecasted hours of all columns in the dataframe
        forecasted_hours: Array with the hours to forecast
        """
        new_Y_columns = {}

        # Loop to create the shifted columns
        for c_hour in forecasted_hours:
            for c_column in df.columns:
                new_column_name = f"plus_{c_hour:02d}_{c_column}"
                new_Y_columns[new_column_name] = df[c_column].shift(-c_hour)

        # Concatenate all new columns at once
        Y_df = pd.concat([pd.DataFrame(new_Y_columns)], axis=1)

        print(f"Shape of Y: {Y_df.shape}")
        print("Done!")
        return Y_df

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
    # splits_file = join(split_info_folder, f'splits_{model_name}.csv')
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
# %%
print(X_df_train.tail())

# %%
# Conversion to PyTorch tensors
X_df_train_tensor = torch.tensor(X_df_train.values, dtype=torch.float32)
Y_df_train_tensor = torch.tensor(Y_df_train.values, dtype=torch.float32)
X_df_val_tensor = torch.tensor(X_df_val.values, dtype=torch.float32)
Y_df_val_tensor = torch.tensor(Y_df_val.values, dtype=torch.float32)

# Verification and dimensions
print(type(X_df_train_tensor), X_df_train_tensor.shape)
print(type(Y_df_train_tensor), Y_df_train_tensor.shape)

# %%

# Split the data into AR, flag, and var parts BEFORE creating the datasets

# Assuming you have X_df_train_tensor, Y_df_train_tensor, X_df_val_tensor, Y_df_val_tensor
# Let's split X_df_train_tensor into X_df_train_ar_tensor, X_df_train_flag_tensor and X_df_train_var_tensor
X_df_train_ar_tensor = X_df_train_tensor[:, :num_ar_features]
X_df_train_flag_tensor = X_df_train_tensor[
    :, num_ar_features : num_ar_features + num_flag_features
]
X_df_train_var_tensor = X_df_train_tensor[:, -num_var_features:]

X_df_val_ar_tensor = X_df_val_tensor[:, :num_ar_features]
X_df_val_flag_tensor = X_df_val_tensor[
    :, num_ar_features : num_ar_features + num_flag_features
]
X_df_val_var_tensor = X_df_val_tensor[:, -num_var_features:]


# %%
Y_df_train_tensor = Y_df_train_tensor[:, :num_ar_features]
len(Y_df_train_tensor[0])
Y_df_val_tensor = Y_df_val_tensor[:, :num_ar_features]
len(Y_df_val_tensor[0])

# %%
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
# Define the dataset and dataloader classes FOR THE BASE MODEL TRAINING
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

# %% 
# BASE PREDICTOR INSTANTIATION
base_predictor = BasePredictor(
    num_ar_features, num_flag_features, num_var_features, output_dim, sequence_length
).to(device)

# SOME TRAINING PARAMETERS AND PROCESSINGS DECLARATIONS

# Verify and APPLYING DataParallel if multiple GPUs are available <- IMPORTANT
if torch.cuda.device_count() > 1:
    print(f'Using {torch.cuda.device_count()} GPUs!')
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


# DEFINING *PLOTS* FOR THE BASE MODEL EVALUATION AND DEBUGGING

def plot_time_series_predictions(
    base_predictor,
    X_df_train_ar_tensor,
    X_df_train_flag_tensor,
    X_df_train_var_tensor,
    Y_df_train_tensor,
    sequence_length,
    device,
    num_points=200,
    target_column_index=0,
):
    """
    Plots a time series of predictions vs. actual values for a specific column over a sequence of data points.

    Args:
        base_predictor: The trained BasePredictor model.
        X_df_train_ar_tensor: The input tensor for AR features.
        X_df_train_flag_tensor: The input tensor for Flag features.
        X_df_train_var_tensor: The input tensor for Var features.
        Y_df_train_tensor: The target data tensor.
        sequence_length: The length of the input sequences.
        device: The device to use ('cuda' or 'cpu').
        num_points: The number of data points to include in the time series plot.
        target_column_index: The index of the column in Y_df_train_tensor to plot.
    """
    base_predictor.eval()  # Set the model to evaluation mode

    predictions = []
    actual_values = []

    # Total possible points considering the sequence length
    total_possible_points = len(X_df_train_ar_tensor) - sequence_length
    if num_points > total_possible_points:
        print(
            f"Requested num_points ({num_points}) larger than available data points ({total_possible_points}). Adjusting to {total_possible_points}."
        )
        num_points = total_possible_points

    with torch.no_grad():
        for i in range(num_points):
            start_idx = i  # Index to slice the sequence from the tensors

            input_ar = (
                X_df_train_ar_tensor[start_idx : start_idx + sequence_length]
                .reshape(-1)
                .unsqueeze(0)
                .to(device)
            )
            input_flag = (
                X_df_train_flag_tensor[start_idx : start_idx + sequence_length]
                .reshape(-1)
                .unsqueeze(0)
                .to(device)
            )
            input_var = (
                X_df_train_var_tensor[start_idx : start_idx + sequence_length]
                .reshape(-1)
                .unsqueeze(0)
                .to(device)
            )

            # Make the prediction
            prediction = base_predictor(input_ar, input_flag, input_var)

            # Actual value for the target column
            actual_value = Y_df_train_tensor[
                start_idx + sequence_length, target_column_index
            ]

            # Extract the prediction for the target column
            prediction_for_column = prediction[0, target_column_index]

            predictions.append(prediction_for_column.cpu().numpy())
            actual_values.append(actual_value.cpu().numpy())

    # Plotting
    plt.figure(figsize=(16, 6))
    plt.plot(actual_values, label="Actual Values", marker="o", linestyle="-")
    plt.plot(predictions, label="Predictions", marker="x", linestyle="-")

    plt.title(
        f"Time Series Predictions vs Actual Values (Column: {target_column_index}, {num_points} points)"
    )
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.show()

# %% Base model
##########################################################
# Base model weights Loading , and AR model instantiation
##########################################################
def load_base_predictor(
    base_predictor_class, model_name, device, save_dir="./saved_models"
):
    """
    Loads the trained base predictor model.

    Args:
        base_predictor_class (nn.Module): The class of the model to initialize.
        model_name (str): Name of the model (used for filename).
        device (str): The device to load the model onto ('cuda' or 'cpu').
        save_dir (str, optional): Directory where the model is saved. Defaults to "./saved_models".

    Returns:
        nn.Module: The loaded model.
    """
    save_path = f"{save_dir}/{model_name}_base_model.pth"
    # model = base_predictor_class().to(device)  # Inicializar la arquitectura
    base_predictor = base_predictor_class(
    num_ar_features, num_flag_features, num_var_features, output_dim, sequence_length
).to(device)

    model.load_state_dict(torch.load(save_path, map_location=device))
    model.eval()  # Modo evaluación
    print(f"✅ Modelo cargado desde: {save_path}")
    return model

model_path_base = "/OZONO/HOME/pedro/git2/gitflow/air_pollution_forecast/saved_models/TestPSpyt_otres_2025_03_26_17_49_ar_5_base_model.pth"

base_predictor = BasePredictor(
    num_ar_features, num_flag_features, num_var_features, output_dim, sequence_length
).to(device)

# Wrap the model with nn.DataParallel if multiple GPUs are available
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    base_predictor = nn.DataParallel(base_predictor)

base_predictor.load_state_dict(torch.load(model_path_base, map_location=device))
base_predictor.eval()  # Modo evaluación
print(f"✅ Modelo cargado desde: {model_path_base}")

# PLOT TEST OF BASE MODEL PREDICTIONS
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


# ##########################################################
# FUNCTION DECLARATIONS FOR MORE TRAINING STEPS ON BASE MODEL   1hr
##########################################################

def train_base_predictor(
    base_predictor, train_loader, device, model_name, num_epochs_base=100, learning_rate=0.01
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

    # Define the optimizer and loss function
    optimizer = optim.Adam(base_predictor.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

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
# %% 
##########################################################    
#Instantiate the AutoregressivePredictor:
##########################################################    

ar_predictor = AutoregressivePredictor(
    base_predictor=base_predictor,
    num_ar_features=num_ar_features,
    num_flag_features=num_flag_features,
    num_var_features=num_var_features,
    output_dim=output_dim,
    sequence_length=sequence_length,
    num_hours_to_forecast=num_hours_to_forecast,
).to(device)


# 3. Prepare an example input for prediction:
#    - We need an initial sequence of AR, flag, and var features.
#    - We'll take the first `sequence_length` data points from the training data.

# Choose a sequence_start_idx (e.g., 0)
sequence_start_idx = 0

# Create the initial input sequences
initial_input_ar = X_df_train_ar_tensor[
    sequence_start_idx : sequence_start_idx + sequence_length, :
]
initial_input_flag = X_df_train_flag_tensor[
    sequence_start_idx : sequence_start_idx + sequence_length, :
]
initial_input_var = X_df_train_var_tensor[
    sequence_start_idx : sequence_start_idx + sequence_length, :
]

# Flatten the initial input sequences and add a batch dimension
initial_input_ar_flat = initial_input_ar.reshape(1, -1).to(
    device
)  # (1, sequence_length * num_ar_features)
initial_input_flag_flat = initial_input_flag.reshape(1, -1).to(
    device
)  # (1, sequence_length * num_flag_features)
initial_input_var_flat = initial_input_var.reshape(1, -1).to(
    device
)  # (1, sequence_length * num_var_features)

# 4. Make a prediction 
ar_predictor.eval()
with torch.no_grad():
    predictions = ar_predictor(
        initial_input_ar_flat,
        initial_input_flag_flat,
        initial_input_var_flat,
        X_df_train_ar_tensor,
        X_df_train_flag_tensor,
        X_df_train_var_tensor,
        sequence_start_idx,
    )

# 5. Print the shape and the first prediction:
# Should be: (1, num_hours_to_forecast, output_dim)
print("Shape of predictions:", predictions.shape)
print("First prediction:", predictions[0, 0, :])




# %%%
# # DEBUGGING INFO
# Después de inicializar ar_predictor
# print(f"Dimensiones esperadas para AR: {ar_predictor.expected_ar_dim}")
# print(f"Dimensiones esperadas para flag: {ar_predictor.expected_flag_dim}")
# print(f"Dimensiones esperadas para var: {ar_predictor.expected_var_dim}")

# print_dimensions(train_ar_loader)
# %%
# TRAINING THE AUTOREGRESSIVE MODEL 
# Assuming you have X_df_train_ar_tensor, X_df_train_flag_tensor, X_df_train_var_tensor, Y_df_train_tensor
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

batch_size = 64  # Or your desired batch size
train_ar_loader = DataLoader(train_ar_dataset, batch_size=batch_size, shuffle=False)


# %%
# FUN DEFINITION FOR PLOT TEST OF AR MODEL PREDICTIONS
def evaluate_and_plot_ar_predictions_new(
    ar_predictor,
    X_ar_data,
    X_flag_data,
    X_var_data,
    Y_data,
    sequence_start_idx,
    num_hours_to_forecast,
    sequence_length,
    device,
    columns_to_plot,
    num_ar_features,
    num_flag_features,
    num_var_features,
):
    """
    Evaluates the autoregressive model and plots the predictions versus the actual values for specific columns.
    This version is adapted to use the new AutoregressivePredictor and separate input datasets (X_ar_data, X_flag_data, X_var_data).

    Args:
        ar_predictor: The trained AutoregressivePredictor model.
        X_ar_data: The input AR data tensor.
        X_flag_data: The input flag data tensor.
        X_var_data: The input var data tensor.
        Y_data: The target data tensor.
        sequence_start_idx: The starting index of the sequence to evaluate.
        num_hours_to_forecast: The number of hours to forecast.
        sequence_length: The length of the input sequence.
        device: The device to use ('cuda' or 'cpu').
        columns_to_plot: A list of column indices to plot.
        num_ar_features: Number of AR features.
        num_flag_features: Number of flag features.
        num_var_features: Number of var features.
    """

    # Check if the sequence fits in the dataset:
    if sequence_start_idx + sequence_length + num_hours_to_forecast > len(X_ar_data):
        print(
            f"Error: sequence_start_idx {sequence_start_idx} + sequence_length {sequence_length} + num_hours_to_forecast {num_hours_to_forecast} exceeds dataset length {len(X_ar_data)}"
        )
        return

    # Create the initial input sequences
    initial_input_ar = X_ar_data[
        sequence_start_idx : sequence_start_idx + sequence_length, :
    ]
    initial_input_flag = X_flag_data[
        sequence_start_idx : sequence_start_idx + sequence_length, :
    ]
    initial_input_var = X_var_data[
        sequence_start_idx : sequence_start_idx + sequence_length, :
    ]

    # Flatten the initial input sequences and add a batch dimension
    initial_input_ar_flat = initial_input_ar.reshape(1, -1).to(
        device
    )  # (1, sequence_length * num_ar_features)
    initial_input_flag_flat = initial_input_flag.reshape(1, -1).to(
        device
    )  # (1, sequence_length * num_flag_features)
    initial_input_var_flat = initial_input_var.reshape(1, -1).to(
        device
    )  # (1, sequence_length * num_var_features)

    # Get the actual targets for the future hours
    actual_targets = Y_data[
        sequence_start_idx
        + sequence_length : sequence_start_idx
        + sequence_length
        + num_hours_to_forecast,
        :,
    ]

    # Evaluate the model
    ar_predictor.eval()
    with torch.no_grad():
        predictions = ar_predictor(
            initial_input_ar_flat,
            initial_input_flag_flat,
            initial_input_var_flat,
            X_ar_data,
            X_flag_data,
            X_var_data,
            sequence_start_idx,
        )

    # Reshape for easier plotting
    pred_np = predictions.cpu().numpy().squeeze()
    actual_np = actual_targets.cpu().numpy()

    # Plot only the specified columns
    plt.figure(figsize=(12, 6))
    for col_idx in columns_to_plot:
        if col_idx >= actual_np.shape[1]:
            print(
                f"Warning: column index {col_idx} is out of bounds for the actual data (shape: {actual_np.shape}). Skipping."
            )
            continue
        plt.plot(pred_np[:, col_idx], label=f"Pred Columna {col_idx}", marker="o")
        plt.plot(
            actual_np[:, col_idx],
            label=f"True Columna {col_idx}",
            linestyle="dashed",
            marker="x",
        )

    plt.title(f"AR Predictions vs True Values (Starting from idx {sequence_start_idx})")
    plt.xlabel("Hours Ahead")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.show()


# %% Example Usage:

columns_to_plot = [0, 2, 5, 104, 105]  # Example columns to plot
sequence_start_idx = 0  # Example starting index

# Evaluate and plot on the training data
evaluate_and_plot_ar_predictions_new(
    ar_predictor,
    X_df_train_ar_tensor,
    X_df_train_flag_tensor,
    X_df_train_var_tensor,
    Y_df_train_tensor,
    sequence_start_idx,
    num_hours_to_forecast,
    sequence_length,
    device,
    columns_to_plot,
    num_ar_features,
    num_flag_features,
    num_var_features,
)

# Evaluate and plot on the validation data
evaluate_and_plot_ar_predictions_new(
    ar_predictor,
    X_df_val_ar_tensor,
    X_df_val_flag_tensor,
    X_df_val_var_tensor,
    Y_df_val_tensor,
    sequence_start_idx,
    num_hours_to_forecast,
    sequence_length,
    device,
    columns_to_plot,
    num_ar_features,
    num_flag_features,
    num_var_features,
)
# %%
# Example Usage (assuming you have defined base_predictor, train_dataset, device, and model_name):
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


# VOY A ESTAR CORRIENDO ESTA SECCIÓN, HASTA QUE CONSIDERE QUE EL MODELO MÁS MENOS PREDICE, OK:
# base_predictor = train_base_predictor(base_predictor, train_loader, device, model_name, num_epochs_base=num_epochs_base, learning_rate=learning_rate_base)

# plot_time_series_predictions(
#     base_predictor,
#     X_df_train_ar_tensor,
#     X_df_train_flag_tensor,
#     X_df_train_var_tensor,
#     Y_df_train_tensor,
#     sequence_length,
#     device,
#     num_points,
#     target_column_index,
# )


# %%
##########################################################
# FIRST TRAINING LOOP FOR THE AR MODEL
##########################################################

##########################################################    
# AR MODEL TRAINING LOOP STEP
##########################################################    
# Assuming you have already defined:
# - AutoregressivePredictor class (ar_predictor)
# - AutoregressiveTimeSeriesDataset class (train_ar_dataset, val_ar_dataset)
# - X_df_train_ar_tensor, X_df_train_flag_tensor, X_df_train_var_tensor, Y_df_train_tensor
# - X_df_val_ar_tensor, X_df_val_flag_tensor, X_df_val_var_tensor, Y_df_val_tensor
# - num_ar_features, num_flag_features, num_var_features, output_dim, sequence_length, device
# - batch_size
# - model_name (for TensorBoard logging)
# - num_hours_to_forecast

# Hyperparameters
criterion = nn.MSELoss()

# Create the DataLoaders (assuming they are already defined)
# train_ar_loader = DataLoader(train_ar_dataset, batch_size=batch_size, shuffle=True)
# val_ar_loader = DataLoader(val_ar_dataset, batch_size=batch_size, shuffle=False)

# Initialize TensorBoard writer
log_dir = f"./tensorboard_logs/{model_name}_ar_model"
writer = SummaryWriter(log_dir=log_dir)

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
# Training loop
optimizer = optim.Adam(ar_predictor.parameters(), lr=learning_rate_ar)
ar_predictor.train()

for epoch in range(num_epochs_ar):
    epoch_loss = 0
    for (
        x_ar_batch_flat,
        x_flag_batch_flat,
        x_var_batch_flat,
        y_batch,
    ) in train_ar_loader:
        x_ar_batch_flat, x_flag_batch_flat, x_var_batch_flat, y_batch = (
            x_ar_batch_flat.to(device),
            x_flag_batch_flat.to(device),
            x_var_batch_flat.to(device),
            y_batch.to(device),
        )

        # Prepare the initial input for the AutoregressivePredictor
        # No need to reshape here, as the data is already flattened in the dataset
        initial_input_ar = x_ar_batch_flat
        initial_input_flag = x_flag_batch_flat
        initial_input_var = x_var_batch_flat

        # Make predictions
        predictions = ar_predictor(
            initial_input_ar,
            initial_input_flag,
            initial_input_var,
            X_df_train_ar_tensor,
            X_df_train_flag_tensor,
            X_df_train_var_tensor,
            0,  # sequence_start_idx = 0 for training
        )

        # Calculate the loss
        loss = criterion(predictions, y_batch)

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    # Log training loss to TensorBoard
    writer.add_scalar("Loss/train", epoch_loss / len(train_ar_loader), epoch)

    # Validation (optional, but recommended)
    ar_predictor.eval()
    val_loss = 0.0
    # with torch.no_grad():
    #     for x_ar_val_batch_flat, x_flag_val_batch_flat, x_var_val_batch_flat, y_val_batch in val_ar_loader:
    #         x_ar_val_batch_flat, x_flag_val_batch_flat, x_var_val_batch_flat, y_val_batch = x_ar_val_batch_flat.to(device), x_flag_val_batch_flat.to(device), x_var_val_batch_flat.to(device), y_val_batch.to(device)

    #         # Prepare the initial input for the AutoregressivePredictor
    #         initial_input_ar_val = x_ar_val_batch_flat
    #         initial_input_flag_val = x_flag_val_batch_flat
    #         initial_input_var_val = x_var_val_batch_flat

    #         val_predictions = ar_predictor(
    #             initial_input_ar_val,
    #             initial_input_flag_val,
    #             initial_input_var_val,
    #             X_df_val_ar_tensor,
    #             X_df_val_flag_tensor,
    #             X_df_val_var_tensor,
    #             0  # sequence_start_idx = 0 for validation
    #         )
    #         val_loss += criterion(val_predictions, y_val_batch).item()

    # val_loss /= len(val_ar_loader)

    # # Log validation loss to TensorBoard
    # writer.add_scalar('Loss/val', val_loss, epoch)

    if (epoch + 1) % 10 == 0:
        # , Val Loss: {val_loss:.4f}')
        print(
            f"Epoch [{epoch + 1}/{num_epochs_ar}], Loss: {epoch_loss/len(train_ar_loader):.4f}"
        )

# Close the TensorBoard writer
writer.close()

print("Training finished.")

# %%
##########################################################
# SAVING OF WEIGHTS FOR THE BASE(AR) MODEL
##########################################################

def save_base_predictor(base_predictor, model_name, save_dir="./saved_models"):
    """
    Saves the trained base predictor model.

    Args:
        base_predictor (nn.Module): The trained model to save.
        model_name (str): Name of the model (used for filename).
        save_dir (str, optional): Directory to save the model. Defaults to "./saved_models".
    """
    import os

    os.makedirs(save_dir, exist_ok=True)

    save_path = f"{save_dir}/{model_name}_base_model.pth"
    torch.save(base_predictor.state_dict(), save_path)
    print(f"✅ Modelo guardado en: {save_path}")



save_base_predictor(base_predictor, f'{model_name}_ar_8')

# %%
