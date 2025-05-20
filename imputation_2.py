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

def generate_climatology(df, value_columns):
    """
    Genera un DataFrame con la climatología para cada hora del año usando un promedio móvil de 3 días.
    La climatología se calcula usando todos los años disponibles en el DataFrame.
    Los valores NaN son ignorados en los cálculos.
    
    Args:
        df: DataFrame con índice datetime
        value_columns: Lista de columnas para las que calcular la climatología
        
    Returns:
        DataFrame con la climatología para cada hora del año
    """
    # Crear un DataFrame para almacenar la climatología usando el año 2010 como referencia
    climatology = pd.DataFrame(index=pd.date_range(start='2010-01-01 00:00:00', 
                                                 end='2010-12-31 23:00:00', 
                                                 freq='h'))
    
    # Para cada columna, calcular el promedio por hora del año usando todos los años disponibles
    for col in value_columns:
        # Agrupar por mes, día y hora, calculando el promedio a través de todos los años
        # skipna=True para ignorar NaN en el cálculo del promedio
        hourly_means = df.groupby([df.index.month, df.index.day, df.index.hour])[col].mean(skipna=True)
        
        # Crear un índice datetime para el año 2010 (solo como referencia)
        hourly_means_dict = {}
        for (month, day, hour), value in hourly_means.items():
            try:
                date = pd.Timestamp(f'2010-{month:02d}-{day:02d} {hour:02d}:00:00')
                # Solo guardamos el valor si no es NaN
                if not pd.isna(value):
                    hourly_means_dict[date] = value
            except ValueError:
                continue
        
        # Convertir el diccionario a Series y asignar a climatology
        climatology[col] = pd.Series(hourly_means_dict)
        
        # Aplicar promedio móvil de 3 días para suavizar la climatología
        # min_periods=1 para permitir promedios con menos de 3 puntos
        climatology[col] = climatology[col].rolling(window=3, center=True, min_periods=1).mean()
        
        # Manejar los bordes del año usando valores del otro extremo del año
        # Solo si tenemos valores válidos
        if not climatology[col].isna().all():
            first_valid = climatology[col].first_valid_index()
            last_valid = climatology[col].last_valid_index()
            
            if first_valid is not None and last_valid is not None:
                climatology[col].iloc[0] = (climatology[col].iloc[-1] + climatology[col].iloc[0] + climatology[col].iloc[1]) / 3
                climatology[col].iloc[-1] = (climatology[col].iloc[-2] + climatology[col].iloc[-1] + climatology[col].iloc[0]) / 3
    
    return climatology

columns_otres = [col for col in data_df.columns if col.startswith('cont_otres_')]
data_imputed = create_imputation_columns(data_df, columns_otres)

# Generar climatología para cont_otres_
climatology_otres = generate_climatology(data_df, columns_otres)

# Visualizar la climatología
def plot_climatology(climatology_df, title="Climatología"):
    """
    Genera un gráfico alargado de la climatología para todas las columnas.
    
    Args:
        climatology_df: DataFrame con la climatología
        title: Título del gráfico
    """
    plt.figure(figsize=(20, 6))
    for col in climatology_df.columns:
        plt.plot(climatology_df.index, climatology_df[col], label=col, alpha=0.7)
    
    plt.title(title)
    plt.xlabel('Fecha')
    plt.ylabel('Valor')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    
    # Guardar el gráfico
    output_folder = '/ZION/AirPollutionData/Data/MergedDataCSV/16/Climatology/'
    create_folder(output_folder)
    plt.savefig(os.path.join(output_folder, f'{title.lower().replace(" ", "_")}.png'))
    plt.close()

plot_climatology(climatology_otres, "Climatología de Otres")

# Guardar la climatología
output_folder = '/ZION/AirPollutionData/Data/MergedDataCSV/16/Climatology/'
create_folder(output_folder)
climatology_otres.to_csv(os.path.join(output_folder, 'climatology_otres.csv'))

# %%
# Calculate climatology
def calculate_climatology(df, value_columns):
    """
    Calcula la climatología para cada hora del año usando un promedio móvil de 3 días.
    
    Args:
        df: DataFrame con índice datetime
        value_columns: Lista de columnas para las que calcular la climatología
        
    Returns:
        DataFrame con la climatología para cada hora del año
    """
    # Crear un DataFrame para almacenar la climatología
    climatology = pd.DataFrame(index=pd.date_range(start='2010-01-01 00:00:00', 
                                                 end='2010-12-31 23:00:00', 
                                                 freq='H'))
    
    # Para cada columna, calcular el promedio por hora del año
    for col in value_columns:
        # Agrupar por mes, día y hora
        hourly_means = df.groupby([df.index.month, df.index.day, df.index.hour])[col].mean()
        
        # Crear un índice datetime para el año 2010
        hourly_means.index = pd.to_datetime([f'2010-{m:02d}-{d:02d} {h:02d}:00:00' 
                                           for m, d, h in hourly_means.index])
        
        # Aplicar promedio móvil de 3 días
        climatology[col] = hourly_means.rolling(window=3, center=True, min_periods=1).mean()
        
        # Manejar los bordes del año
        climatology[col].iloc[0] = (climatology[col].iloc[-1] + climatology[col].iloc[0] + climatology[col].iloc[1]) / 3
        climatology[col].iloc[-1] = (climatology[col].iloc[-2] + climatology[col].iloc[-1] + climatology[col].iloc[0]) / 3
    
    return climatology

# Impute with row average or persistence
def impute_with_row_avg_or_persistence(df, value_columns):
    df_imputed = df.copy()
    
    # Calcular la climatología
    climatology = calculate_climatology(df, value_columns)
    
    for col in value_columns:
        flag_col_name = f"i_{col}"
        mask_missing = (df[flag_col_name] == -1)
        
        # Primero intentar con promedio de fila
        mask_avg = mask_missing & (df[value_columns].notna().sum(axis=1) > 5)
        df_imputed.loc[mask_avg, col] = df_imputed.loc[mask_avg, value_columns].mean(axis=1, skipna=True)
        df_imputed.loc[mask_avg, flag_col_name] = 'row_avg'
        
        # Luego intentar con persistencia
        mask_remaining = mask_missing & (df_imputed[flag_col_name] == -1)
        prev_day_indices = df.index[mask_remaining] - pd.Timedelta(days=1)
        valid_last_day_indices = df.index.intersection(prev_day_indices)
        mask_last_day = df.index.isin(valid_last_day_indices)
        df_imputed.loc[mask_last_day, col] = df.loc[prev_day_indices.intersection(valid_last_day_indices), col].values
        df_imputed.loc[mask_last_day, flag_col_name] = 'last_day_same_hour'
        
        # Finalmente, usar climatología para los valores restantes
        mask_remaining = mask_missing & (df_imputed[flag_col_name] == -1)
        for idx in df.index[mask_remaining]:
            # Obtener el mes, día y hora correspondiente
            month = idx.month
            day = idx.day
            hour = idx.hour
            # Usar el valor climatológico correspondiente
            climatology_idx = pd.Timestamp(f'2010-{month:02d}-{day:02d} {hour:02d}:00:00')
            df_imputed.loc[idx, col] = climatology.loc[climatology_idx, col]
            df_imputed.loc[idx, flag_col_name] = 'climatology'
    
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
    counts = {'-1': 0, 'none': 0, 'row_avg': 0, 'last_day_same_hour': 0, 'climatology': 0}
    for col in flag_columns:
        value_counts = df[col].value_counts()
        counts['-1'] += value_counts.get(-1, 0)
        counts['none'] += value_counts.get('none', 0)
        counts['row_avg'] += value_counts.get('row_avg', 0)
        counts['last_day_same_hour'] += value_counts.get('last_day_same_hour', 0)
        counts['climatology'] += value_counts.get('climatology', 0)
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
    
    # Generar climatología para cada grupo
    climatology = generate_climatology(data_df, columns)
    # Visualizar la climatología
    plot_climatology(climatology, f"Climatología de {group.replace('cont_', '').replace('_', '')}")
    # Guardar la climatología
    climatology.to_csv(os.path.join(output_folder, f'climatology_{group.replace("cont_", "").replace("_", "")}.csv'))

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

