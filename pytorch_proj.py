import os
import pickle
import pandas as pd
from pandas import DataFrame
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn import preprocessing
from datetime import datetime
import matplotlib.pyplot as plt
from conf.localConstants import constants 
from proj_io.inout import create_folder, read_merged_files, save_columns 
from proj_preproc.preproc import loadScaler 
from proj_prediction.prediction import plot_forecast_hours, analyze_column 


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
# %%
    
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

# %%

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

# %%
def plot_forecast_hours(column_to_plot, y_true_df, y_pred_descaled_df, 
                        output_results_folder_img=None, show_grid=True, 
                        x_label='Forecasted Time [hours]', y_label='Pollutant Level $O_3$ [ppb]', 
                        title_str=None, save_fig=True):
    """
    Generate a plot for forecast hours.

    Parameters:
    ...
    """
    plt.close('all')
    plot_this_many = 24 * 20  # Number of points to plot
    
    # Retrieve columns for plotting
    y_true_column = y_true_df[column_to_plot]
    y_pred_column = y_pred_descaled_df[column_to_plot]
    
    x_plot = range(len(y_true_column))
    
    # Plotting
    plt.figure(figsize=[12, 6])
    
    plt.xlabel(x_label, fontsize=18)
    plt.ylabel(y_label, fontsize=18)
    
    plt.plot(x_plot[0:plot_this_many], y_true_column[0:plot_this_many], marker='.', label='Observed')
    plt.plot(x_plot[0:plot_this_many], y_pred_column[0:plot_this_many], marker='.', label='Forecasted')
    
    if title_str is not None:
        plt.title(title_str, fontsize=20)
    else:
        plt.title(f'Forecast Levels for {column_to_plot}', fontsize=20)
        
    if show_grid:
        plt.grid(True)
    
    plt.legend(fontsize=16)
    plt.tick_params(axis='x', labelsize=14)
    plt.tick_params(axis='y', labelsize=16)
    
    if save_fig and output_results_folder_img:
        plt.savefig(join(output_results_folder_img, f'hours_plot_{column_to_plot.lower()}.png'), dpi=300)
    
    plt.show()
