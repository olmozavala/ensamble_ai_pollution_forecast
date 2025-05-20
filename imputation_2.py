# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import os
import pandas as pd
import matplotlib.pyplot as plt

# Set working directory
os.chdir('/home/pedro/git2/gitflow/air_pollution_forecast')

from proj_io.inout import create_folder, add_forecasted_hours, add_previous_hours, get_column_names, read_merged_files, save_columns

# Constants
START_YEAR = 2010
END_YEAR = 2024
INPUT_FOLDER = '/ZION/AirPollutionData/Data/MergedDataCSV/16/'

# Read data
data_df = read_merged_files(INPUT_FOLDER, START_YEAR, END_YEAR)

# %%
# Ensure datetime index
data_df.index = pd.to_datetime(data_df.index.to_series().apply(lambda x: f"{x} 00:00:00" if len(str(x)) == 10 else str(x)))

# %%
# Create imputation columns with prefix i_
def create_imputation_columns(df, columns):
    df_imputed = df.copy()
    for col in columns:
        new_col_name = f"i_{col}"
        df_imputed[new_col_name] = df_imputed[col].apply(lambda x: 'none' if pd.notna(x) else -1)
    return df_imputed

columns_otres = [col for col in data_df.columns if col.startswith('cont_otres_')]
data_imputed = create_imputation_columns(data_df, columns_otres)

# %%
# Impute with row average or persistence
def impute_with_row_avg_or_persistence(df, value_columns):
    df_imputed = df.copy()
    for col in value_columns:
        flag_col_name = f"i_{col}"
        mask_missing = (df[flag_col_name] == -1)
        mask_avg = mask_missing & (df[value_columns].notna().sum(axis=1) > 5)
        df_imputed.loc[mask_avg, col] = df_imputed.loc[mask_avg, value_columns].mean(axis=1, skipna=True)
        df_imputed.loc[mask_avg, flag_col_name] = 'row_avg'
        mask_remaining = mask_missing & (df_imputed[flag_col_name] == -1)
        prev_day_indices = df.index[mask_remaining] - pd.Timedelta(days=1)
        valid_last_day_indices = df.index.intersection(prev_day_indices)
        mask_last_day = df.index.isin(valid_last_day_indices)
        df_imputed.loc[mask_last_day, col] = df.loc[prev_day_indices.intersection(valid_last_day_indices), col].values
        df_imputed.loc[mask_last_day, flag_col_name] = 'last_day_same_hour'
        df_imputed.loc[mask_remaining & (df_imputed[flag_col_name] == -1), col] = -1
    return df_imputed

value_columns_otres = [col for col in data_df.columns if col.startswith('cont_otres_')]
data_imputed = impute_with_row_avg_or_persistence(data_imputed, value_columns_otres)

# %%
# Reindex data
full_index = pd.date_range(start='2010-01-01 00:00:00', end='2024-12-31 23:00:00', freq='H')
data_imputed = data_imputed.reindex(full_index)

# %%
# Count imputation flags
def count_imputation_flags(df, flag_columns):
    counts = {'-1': 0, 'none': 0, 'row_avg': 0}
    for col in flag_columns:
        value_counts = df[col].value_counts()
        counts['-1'] += value_counts.get(-1, 0)
        counts['none'] += value_counts.get('none', 0)
        counts['row_avg'] += value_counts.get('row_avg', 0)
    return counts

flag_columns_otres = [col for col in data_imputed.columns if col.startswith('i_')]
flag_counts = count_imputation_flags(data_imputed, flag_columns_otres)
print("Conteo de flags de imputación:")
print(flag_counts)

# %%
# Plot time series
def plot_time_series(df, columns, title):
    plt.figure(figsize=(14, 8))
    for col in columns:
        plt.plot(df.index, df[col], label=col, alpha=0.5)
    plt.title(title)
    plt.xlabel('Fecha')
    plt.ylabel('Valor')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

plot_time_series(data_imputed, columns_otres, 'Series Temporales de cont_otres_')

# %%
# Procesar otras columnas de contaminación
column_groups = ['cont_pmdiez_', 'cont_pmdoscinco_', 'cont_nodos_', 'cont_co_']

for group in column_groups:
    columns = [col for col in data_df.columns if col.startswith(group)]
    data_imputed = create_imputation_columns(data_imputed, columns)
    value_columns = [col for col in data_df.columns if col.startswith(group)]
    data_imputed = impute_with_row_avg_or_persistence(data_imputed, value_columns)

# %%
# Contar flags de imputación para todas las categorías
for group in column_groups:
    flag_columns = [col for col in data_imputed.columns if col.startswith(f"i_{group}")]
    flag_counts = count_imputation_flags(data_imputed, flag_columns)
    print(f"\nConteo de flags de imputación para {group}:")
    print(flag_counts)

# %%
# Export DataFrame to CSV
output_folder = '/ZION/AirPollutionData/Data/MergedDataCSV/16/Imputed/'
create_folder(output_folder)
data_imputed.to_csv(os.path.join(output_folder, 'data_imputed_full.csv'))

# %%
# Export DataFrame by year
for year in range(START_YEAR, END_YEAR + 1):
    yearly_data = data_imputed[data_imputed.index.year == year]
    yearly_data.to_csv(os.path.join(output_folder, f'data_imputed_{year}.csv'))

# %%

# # Load imputed data for all years
# def load_imputed_data(start_year, end_year, folder_path):
#     all_data = []
#     for year in range(start_year, end_year + 1):
#         file_path = os.path.join(folder_path, f'data_imputed_{year}.csv')
#         print(f"Loading data from {file_path}")
#         yearly_data = pd.read_csv(file_path, index_col=0, parse_dates=True)
#         all_data.append(yearly_data)
#     data_imputed_df = pd.concat(all_data)
#     data_imputed_df.index = pd.to_datetime(data_imputed_df.index)
#     return data_imputed_df

# # Load the imputed data
# data_imputed_df = load_imputed_data(START_YEAR, END_YEAR, output_folder)
# data_imputed = data_imputed_df

# # %%
# # Filtrar las columnas imputadas
# selected_prefixes = ['cont_otres_', 'cont_pmdiez_', 'cont_pmdoscinco_', 'cont_nodos_', 'cont_co_', 'i_']
# additional_columns = [
#     'half_sin_day', 'half_cos_day', 'half_sin_week', 'half_cos_week', 
#     'half_sin_year', 'half_cos_year', 'sin_day', 'cos_day', 
#     'sin_week', 'cos_week', 'sin_year', 'cos_year'
# ]
# imputed_columns_filtered = [col for col in data_imputed.columns 
#                             if any(col.startswith(prefix) for prefix in selected_prefixes)] + additional_columns

# # Crear el subset con las columnas seleccionadas
# data_imputed_subset = data_imputed[imputed_columns_filtered].copy()

# # %%
# for each in data_imputed_subset.columns:
#     print(each)



# # %%
# # %%
# # ---
# # jupyter:
# #   jupytext:
# #     text_representation:
# #       extension: .py
# #       format_name: percent
# #       format_version: '1.3'
# #       jupytext_version: 1.16.4
# #   kernelspec:
# #     display_name: Python 3
# #     language: python
# #     name: python3
# # ---

# # %%
# import os
# import pandas as pd
# import matplotlib.pyplot as plt

# # Set working directory
# os.chdir('/home/pedro/git2/gitflow/air_pollution_forecast')

# from proj_io.inout import create_folder, add_forecasted_hours, add_previous_hours, get_column_names, read_merged_files, save_columns

# # Constants
# START_YEAR = 2010
# END_YEAR = 2024
# INPUT_FOLDER = '/ZION/AirPollutionData/Data/MergedDataCSV/16/'

# # Read data
# data_df = read_merged_files(INPUT_FOLDER, START_YEAR, END_YEAR)

# # %%
# # Ensure datetime index
# data_df.index = pd.to_datetime(data_df.index.to_series().apply(lambda x: f"{x} 00:00:00" if len(str(x)) == 10 else str(x)))

# # %%
# # Create imputation columns
# def create_imputation_columns(df, columns):
#     df_imputed = df.copy()
#     for col in columns:
#         new_col_name = f"{col}_i"
#         df_imputed[new_col_name] = df_imputed[col].apply(lambda x: 'none' if pd.notna(x) else -1)
#     return df_imputed

# columns_otres = [col for col in data_df.columns if col.startswith('cont_otres_')]
# data_imputed = create_imputation_columns(data_df, columns_otres)

# # %%
# # Impute with row average or persistence
# def impute_with_row_avg_or_persistence(df, value_columns):
#     df_imputed = df.copy()
#     for col in value_columns:
#         flag_col_name = f"{col}_i"
#         mask_missing = (df[flag_col_name] == -1)
#         mask_avg = mask_missing & (df[value_columns].notna().sum(axis=1) > 5)
#         df_imputed.loc[mask_avg, col] = df_imputed.loc[mask_avg, value_columns].mean(axis=1, skipna=True)
#         df_imputed.loc[mask_avg, flag_col_name] = 'row_avg'
#         mask_remaining = mask_missing & (df_imputed[flag_col_name] == -1)
#         prev_day_indices = df.index[mask_remaining] - pd.Timedelta(days=1)
#         valid_last_day_indices = df.index.intersection(prev_day_indices)
#         mask_last_day = df.index.isin(valid_last_day_indices)
#         df_imputed.loc[mask_last_day, col] = df.loc[prev_day_indices.intersection(valid_last_day_indices), col].values
#         df_imputed.loc[mask_last_day, flag_col_name] = 'last_day_same_hour'
#         # Ensure remaining NaNs are set to -1
#         df_imputed.loc[mask_remaining & (df_imputed[flag_col_name] == -1), col] = -1
#     return df_imputed

# value_columns_otres = [col for col in data_df.columns if col.startswith('cont_otres_') and not col.endswith('_i')]
# data_imputed = impute_with_row_avg_or_persistence(data_imputed, value_columns_otres)

# # %%
# # Reindex data
# full_index = pd.date_range(start='2010-01-01 00:00:00', end='2024-12-31 23:00:00', freq='H')
# data_imputed = data_imputed.reindex(full_index)

# # %%
# # Count imputation flags
# def count_imputation_flags(df, flag_columns):
#     counts = {'-1': 0, 'none': 0, 'row_avg': 0}
#     for col in flag_columns:
#         value_counts = df[col].value_counts()
#         counts['-1'] += value_counts.get(-1, 0)
#         counts['none'] += value_counts.get('none', 0)
#         counts['row_avg'] += value_counts.get('row_avg', 0)
#     return counts

# flag_columns_otres = [col for col in data_imputed.columns if col.endswith('_i')]
# flag_counts = count_imputation_flags(data_imputed, flag_columns_otres)
# print("Conteo de flags de imputación:")
# print(f"- '-1': {flag_counts['-1']}")
# print(f"- 'none': {flag_counts['none']}")
# print(f"- 'row_avg': {flag_counts['row_avg']}")

# # %%
# # Plot time series
# def plot_time_series(df, columns, title):
#     plt.figure(figsize=(14, 8))
#     for col in columns:
#         plt.plot(df.index, df[col], label=col, alpha=0.5)
#     plt.title(title)
#     plt.xlabel('Fecha')
#     plt.ylabel('Valor')
#     plt.legend(loc='upper right')
#     plt.tight_layout()
#     plt.show()

# plot_time_series(data_imputed, columns_otres, 'Series Temporales de cont_otres_')


# # %%
# # Plot reindexed time series
# plot_time_series(data_df, columns_otres, 'Series Temporales original df)')

# # %%
# print(data_imputed)

# # %%
# # Procesar columnas cont_pmdiez_
# columns_pmdiez = [col for col in data_df.columns if col.startswith('cont_pmdiez_')]
# data_imputed = create_imputation_columns(data_imputed, columns_pmdiez)

# value_columns_pmdiez = [col for col in data_df.columns if col.startswith('cont_pmdiez_') and not col.endswith('_i')]
# data_imputed = impute_with_row_avg_or_persistence(data_imputed, value_columns_pmdiez)

# # %%
# # Procesar columnas cont_pmdoscinco_
# columns_pmdoscinco = [col for col in data_df.columns if col.startswith('cont_pmdoscinco_')]
# data_imputed = create_imputation_columns(data_imputed, columns_pmdoscinco)

# value_columns_pmdoscinco = [col for col in data_df.columns if col.startswith('cont_pmdoscinco_') and not col.endswith('_i')]
# data_imputed = impute_with_row_avg_or_persistence(data_imputed, value_columns_pmdoscinco)

# # %%
# # Contar flags de imputación para todas las categorías
# flag_columns_pmdiez = [col for col in data_imputed.columns if col.startswith('cont_pmdiez_') and col.endswith('_i')]
# flag_columns_pmdoscinco = [col for col in data_imputed.columns if col.startswith('cont_pmdoscinco_') and col.endswith('_i')]

# flag_counts_pmdiez = count_imputation_flags(data_imputed, flag_columns_pmdiez)
# flag_counts_pmdoscinco = count_imputation_flags(data_imputed, flag_columns_pmdoscinco)

# print("\nConteo de flags de imputación para cont_pmdiez_:")
# print(f"- '-1': {flag_counts_pmdiez['-1']}")
# print(f"- 'none': {flag_counts_pmdiez['none']}")
# print(f"- 'row_avg': {flag_counts_pmdiez['row_avg']}")

# print("\nConteo de flags de imputación para cont_pmdoscinco_:")
# print(f"- '-1': {flag_counts_pmdoscinco['-1']}")
# print(f"- 'none': {flag_counts_pmdoscinco['none']}")
# print(f"- 'row_avg': {flag_counts_pmdoscinco['row_avg']}")

# # %%
# # Graficar series temporales imputadas vs originales
# plot_time_series(data_imputed, columns_pmdiez, 'Series Temporales de cont_pmdiez_ (Imputadas)')
# plot_time_series(data_df, columns_pmdiez, 'Series Temporales de cont_pmdiez_ (Originales)')

# plot_time_series(data_imputed, columns_pmdoscinco, 'Series Temporales de cont_pmdoscinco_ (Imputadas)')
# plot_time_series(data_df, columns_pmdoscinco, 'Series Temporales de cont_pmdoscinco_ (Originales)')

# # %%
# print(data_imputed)
# # %%
# # %%
# # Procesar columnas cont_nodos_
# columns_nodos = [col for col in data_df.columns if col.startswith('cont_nodos_')]
# data_imputed = create_imputation_columns(data_imputed, columns_nodos)

# value_columns_nodos = [col for col in data_df.columns if col.startswith('cont_nodos_') and not col.endswith('_i')]
# data_imputed = impute_with_row_avg_or_persistence(data_imputed, value_columns_nodos)

# # %%
# # Procesar columnas cont_co_
# columns_co = [col for col in data_df.columns if col.startswith('cont_co_')]
# data_imputed = create_imputation_columns(data_imputed, columns_co)

# value_columns_co = [col for col in data_df.columns if col.startswith('cont_co_') and not col.endswith('_i')]
# data_imputed = impute_with_row_avg_or_persistence(data_imputed, value_columns_co)

# # %%
# # Contar flags de imputación para cont_nodos_ y cont_co_
# flag_columns_nodos = [col for col in data_imputed.columns if col.startswith('cont_nodos_') and col.endswith('_i')]
# flag_columns_co = [col for col in data_imputed.columns if col.startswith('cont_co_') and col.endswith('_i')]

# flag_counts_nodos = count_imputation_flags(data_imputed, flag_columns_nodos)
# flag_counts_co = count_imputation_flags(data_imputed, flag_columns_co)

# print("\nConteo de flags de imputación para cont_nodos_:")
# print(f"- '-1': {flag_counts_nodos['-1']}")
# print(f"- 'none': {flag_counts_nodos['none']}")
# print(f"- 'row_avg': {flag_counts_nodos['row_avg']}")

# print("\nConteo de flags de imputación para cont_co_:")
# print(f"- '-1': {flag_counts_co['-1']}")
# print(f"- 'none': {flag_counts_co['none']}")
# print(f"- 'row_avg': {flag_counts_co['row_avg']}")

# # %%
# # Graficar series temporales imputadas vs originales para cont_nodos_ y cont_co_
# plot_time_series(data_imputed, columns_nodos, 'Series Temporales de cont_nodos_ (Imputadas)')
# plot_time_series(data_df, columns_nodos, 'Series Temporales de cont_nodos_ (Originales)')

# plot_time_series(data_imputed, columns_co, 'Series Temporales de cont_co_ (Imputadas)')
# plot_time_series(data_df, columns_co, 'Series Temporales de cont_co_ (Originales)')

# # %%
# print(data_imputed)
# # %%
# for each in data_imputed.columns:
#     print(each)

# # %%
# # Export entire DataFrame to CSV
# output_folder = '/ZION/AirPollutionData/Data/MergedDataCSV/16/Imputed/'
# #'/home/pedro/git2/gitflow/air_pollution_forecast/output/'
# create_folder(output_folder)
# data_imputed.to_csv(os.path.join(output_folder, 'data_imputed_full.csv'))
# # %%
# # Export DataFrame to CSV by year
# for year in range(START_YEAR, END_YEAR + 1):
#     yearly_data = data_imputed[data_imputed.index.year == year]
#     yearly_data.to_csv(os.path.join(output_folder, f'data_imputed_{year}.csv'))
# # %%
# # Function to load imputed data for all years and recreate the DataFrame
# def load_imputed_data(start_year, end_year, folder_path):
#     all_data = []
#     for year in range(start_year, end_year + 1):
#         file_path = os.path.join(folder_path, f'data_imputed_{year}.csv')
#         print(f"Loading data from {file_path}")
#         yearly_data = pd.read_csv(file_path, index_col=0, parse_dates=True)
#         all_data.append(yearly_data)
#     data_imputed_df = pd.concat(all_data)
#     data_imputed_df.index = pd.to_datetime(data_imputed_df.index)
#     return data_imputed_df

# # Load the imputed data
# data_imputed_df = load_imputed_data(START_YEAR, END_YEAR, output_folder)
# print(data_imputed_df)

# # %% 
# data_imputed  = data_imputed_df

# # %% 
# # Filtrar las columnas imputadas y sus flags solo para los grupos específicos
# selected_prefixes = ['cont_otres_', 'cont_pmdiez_', 'cont_pmdoscinco_', 'cont_nodos_', 'cont_co_']
# additional_columns = [
#     'half_sin_day', 'half_cos_day', 'half_sin_week', 'half_cos_week', 
#     'half_sin_year', 'half_cos_year', 'sin_day', 'cos_day', 
#     'sin_week', 'cos_week', 'sin_year', 'cos_year'
# ]
# imputed_columns_filtered = [col for col in data_imputed.columns 
#                             if any(col.startswith(prefix) for prefix in selected_prefixes) and 
#                             (col in data_imputed.columns or col.endswith('_i'))] + additional_columns

# # Crear el subset con las columnas seleccionadas
# data_imputed_subset = data_imputed[imputed_columns_filtered].copy()

# # Mostrar información del subset
# print("Subset de datos con columnas imputadas y sus flags:")
# print(data_imputed_subset.info())
# print(data_imputed_subset.head())

# # %%
# for each in data_imputed_subset.columns:
#     print(each)

# # %%
# for each in data_imputed_subset.columns:
#     print(each)

