import os
import pickle
import pandas as pd
from pandas import DataFrame
import numpy as np
from sklearn import preprocessing
from datetime import datetime
from os.path import join
from proj_io.inout import (
    create_folder,
    add_previous_hours,
    get_column_names,
    read_merged_files,
    save_columns,
)


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
    info_splits = DataFrame({f"Train({len(train_ids)})": train_ids})
    info_splits[f"Validation({len(val_ids)})"] = -1
    info_splits[f"Validation({len(val_ids)})"][0 : len(val_ids)] = val_ids
    info_splits[f"Test({len(test_ids)})"] = -1
    info_splits[f"Test({len(test_ids)})"][0 : len(test_ids)] = test_ids
    info_splits.to_csv(file_name, index=None)


def split_train_validation_and_test(
    num_examples, val_percentage, test_percentage, shuffle_ids=True, file_name=""
):
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
    train_idxs = all_samples_idxs[
        0 : len(all_samples_idxs) - test_examples - val_examples
    ]
    val_idxs = all_samples_idxs[
        len(all_samples_idxs)
        - test_examples
        - val_examples : len(all_samples_idxs)
        - test_examples
    ]
    test_idxs = all_samples_idxs[len(all_samples_idxs) - test_examples :]
    train_idxs.sort()
    val_idxs.sort()
    test_idxs.sort()

    if file_name != "":
        save_splits(file_name, train_idxs, val_idxs, test_idxs)

    return [train_idxs, val_idxs, test_idxs]


def apply_bootstrap(
    X_df,
    Y_df,
    contaminant,
    station,
    boostrap_threshold,
    forecasted_hours,
    boostrap_factor=1,
):
    """
    This function will boostrap the data based on the threshold and the forecasted hours
    """

    bootstrap_column = f"cont_{contaminant}_{station}"
    print("Bootstrapping the data...")
    # Searching all the index where X or Y is above the threshold

    # Adding index when the current time is above the threshold
    bootstrap_idx = X_df.loc[:, bootstrap_column] > boostrap_threshold

    # Searching index when any of the forecasted hours is above the threshold
    y_cols = Y_df.columns.values
    for i in range(1, forecasted_hours + 1):
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
    with open(
        scaler.path_file, "wb"
    ) as f:  # scaler.path_file must be defined during training.
        pickle.dump(scaler, f)
    print(f"Scaler/normalizer object saved to: {scaler.path_file}")
    print(f"Done! Current shape: {data_norm_df.shape} ")
    return data_norm_df


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
    data_norm_df = pd.DataFrame(data_norm_np, columns=data.columns, index=data.index)

    print(f"Scaler/normalizer object loaded from: {file_name}")
    print(f"Done! Current shape: {data_norm_df.shape}")
    return data_norm_df


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


# normalizeDataWithLoadedScaler,add_forecasted_hours
# %%
# %% DEFINING *PLOTS* FOR THE BASE MODEL EVALUATION AND DEBUGGING
import matplotlib.pyplot as plt


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


##########################################################
# SAVING OF WEIGHTS FOR THE BASE(AR) MODEL
##########################################################
import torch


def save_base_predictor(base_predictor, model_name, ar=None, save_dir="./saved_models"):
    """
    Saves the trained base predictor model.

    Args:
        base_predictor (nn.Module): The trained model to save.
        model_name (str): Name of the model (used for filename).
        save_dir (str, optional): Directory to save the model. Defaults to "./saved_models".
    """
    import os

    os.makedirs(save_dir, exist_ok=True)
    suffix = 'base_model.pth'
    if ar:
        suffix = 'ar_model.pth'
    
    save_path = f"{save_dir}/{model_name}_{suffix}"
    torch.save(base_predictor.state_dict(), save_path)
    print(f"✅ Modelo guardado en: {save_path}")


# def load_base_predictor(
#     base_predictor_class, model_name, device, save_dir="./saved_models"
# ):
#     """
#     Loads the trained base predictor model.

#     Args:
#         base_predictor_class (nn.Module): The class of the model to initialize.
#         model_name (str): Name of the model (used for filename).
#         device (str): The device to load the model onto ('cuda' or 'cpu').
#         save_dir (str, optional): Directory where the model is saved. Defaults to "./saved_models".

#     Returns:
#         nn.Module: The loaded model.
#     """
#     save_path = f"{save_dir}/{model_name}_base_model.pth"
#     # model = base_predictor_class().to(device)  # Inicializar la arquitectura
#     base_predictor = base_predictor_class(
#     num_ar_features, num_flag_features, num_var_features, output_dim, sequence_length
# ).to(device)

#     model.load_state_dict(torch.load(save_path, map_location=device))
#     model.eval()  # Modo evaluación
#     print(f"✅ Modelo cargado desde: {save_path}")
#     return model



###############################
# AR functios
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

