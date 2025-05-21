# %% CORREGIR que la climatología se calcule con los dataos observados, en vez de con datos medio-imputados...
# Imports y configuración
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from os.path import join
from proj_io.inout import create_folder

# Set working directory
os.chdir('/home/pedro/git2/gitflow/air_pollution_forecast')

# Constantes
START_YEAR = 2010
END_YEAR = 2024
INPUT_FOLDER = '/ZION/AirPollutionData/Data/MergedDataCSV/16/'
OUTPUT_FOLDER = '/ZION/AirPollutionData/Data/MergedDataCSV/16/Imputed/'
CLIMATOLOGY_FOLDER = '/ZION/AirPollutionData/Data/MergedDataCSV/16/Climatology/'
PLOTS_FOLDER = '/ZION/AirPollutionData/Data/MergedDataCSV/16/Imputed/Plots/'

# Crear directorios si no existen
for folder in [OUTPUT_FOLDER, CLIMATOLOGY_FOLDER, PLOTS_FOLDER]:
    create_folder(folder)

print(f"Directorios configurados:")
print(f"INPUT_FOLDER: {INPUT_FOLDER}")
print(f"OUTPUT_FOLDER: {OUTPUT_FOLDER}")
print(f"CLIMATOLOGY_FOLDER: {CLIMATOLOGY_FOLDER}")
print(f"PLOTS_FOLDER: {PLOTS_FOLDER}")

# %%
# Funciones de utilidad
def create_folder(folder_path: str) -> None:
    """Crea una carpeta si no existe."""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def read_merged_files(input_folder: str, start_year: int, end_year: int) -> pd.DataFrame:
    """
    Lee y combina los archivos de datos de contaminación.
    
    Args:
        input_folder: Ruta a la carpeta con los archivos
        start_year: Año inicial
        end_year: Año final
        
    Returns:
        DataFrame con los datos combinados
    """
    print(f"Reading years {start_year} to {end_year}...")
    for c_year in range(start_year, end_year+1):
        db_file_name = join(input_folder, f"{c_year}_AllStations.csv")
        print(f"============ Reading data for: {c_year}: {db_file_name}")
        if c_year == start_year:
            data = pd.read_csv(db_file_name, index_col=0)
        else:
            data = pd.concat([data, pd.read_csv(db_file_name, index_col=0)])

    print(f'Data shape: {data.shape} Data axes {data.axes}')
    print("Done!")
    return data

# %%
# Funciones de climatología
def generate_climatology(df: pd.DataFrame, value_columns: list) -> pd.DataFrame:
    """Genera climatología para las columnas especificadas."""
    # Crear un DataFrame para almacenar la climatología usando el año 2012 como referencia (año bisiesto)
    climatology = pd.DataFrame(index=pd.date_range(start='2012-01-01 00:00:00', 
                                                 end='2012-12-31 23:00:00', 
                                                 freq='h'))
    
    # Para cada columna, calcular el promedio por hora del año usando todos los años disponibles
    for col in value_columns:
        # Agrupar por mes, día y hora, calculando el promedio a través de todos los años
        hourly_means = df.groupby([df.index.month, df.index.day, df.index.hour])[col].mean()
        
        # Crear un índice datetime para el año 2012 (solo como referencia)
        hourly_means_dict = {}
        for (month, day, hour), value in hourly_means.items():
            try:
                date = pd.Timestamp(f'2012-{month:02d}-{day:02d} {hour:02d}:00:00')
                # Solo guardamos el valor si no es NaN
                if not pd.isna(value):
                    hourly_means_dict[date] = value
            except ValueError:
                continue
        
        # Convertir el diccionario a Series y asignar a climatology
        climatology[col] = pd.Series(hourly_means_dict)
        
        # Aplicar promedio móvil de 3 días para suavizar la climatología
        climatology[col] = climatology[col].rolling(window=3, center=True, min_periods=1).mean()
        
        # Manejar los bordes del año usando valores del otro extremo del año
        if not climatology[col].isna().all():
            first_valid = climatology[col].first_valid_index()
            last_valid = climatology[col].last_valid_index()
            
            if first_valid is not None and last_valid is not None:
                climatology[col].iloc[0] = (climatology[col].iloc[-1] + climatology[col].iloc[0] + climatology[col].iloc[1]) / 3
                climatology[col].iloc[-1] = (climatology[col].iloc[-2] + climatology[col].iloc[-1] + climatology[col].iloc[0]) / 3
    
    return climatology

def calculate_climatology(df: pd.DataFrame, value_columns: list) -> pd.DataFrame:
    """Calcula climatología con promedio móvil."""
    # Crear un DataFrame para almacenar la climatología
    climatology = pd.DataFrame(index=pd.date_range(start='2012-01-01 00:00:00', 
                                                 end='2012-12-31 23:00:00', 
                                                 freq='H'))
    
    # Para cada columna, calcular el promedio por hora del año
    for col in value_columns:
        # Agrupar por mes, día y hora
        hourly_means = df.groupby([df.index.month, df.index.day, df.index.hour])[col].mean()
        
        # Crear un índice datetime para el año 2012 (bisiesto)
        hourly_means.index = pd.to_datetime([f'2012-{m:02d}-{d:02d} {h:02d}:00:00' 
                                           for m, d, h in hourly_means.index])
        
        # Aplicar promedio móvil de 3 días
        climatology[col] = hourly_means.rolling(window=3, center=True, min_periods=1).mean()
        
        # Manejar los bordes del año
        climatology[col].iloc[0] = (climatology[col].iloc[-1] + climatology[col].iloc[0] + climatology[col].iloc[1]) / 3
        climatology[col].iloc[-1] = (climatology[col].iloc[-2] + climatology[col].iloc[-1] + climatology[col].iloc[0]) / 3
    
    return climatology

# %%
# Funciones de imputación básica
def create_imputation_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Crea columnas de imputación."""
    df_imputed = df.copy()
    for col in columns:
        new_col_name = f"i_{col}"
        df_imputed[new_col_name] = df_imputed[col].apply(lambda x: 'none' if pd.notna(x) else -1)
    return df_imputed

def impute_with_row_avg_or_persistence(df: pd.DataFrame, value_columns: list) -> pd.DataFrame:
    """Imputa valores usando promedio de fila o persistencia."""
    df_imputed = df.copy()
    for col in value_columns:
        flag_col_name = f"i_{col}"
        mask_missing = (df[flag_col_name] == -1)
        
        # Imputar con promedio de fila
        mask_avg = mask_missing & (df[value_columns].notna().sum(axis=1) > 5)
        df_imputed.loc[mask_avg, col] = df_imputed.loc[mask_avg, value_columns].mean(axis=1, skipna=True)
        df_imputed.loc[mask_avg, flag_col_name] = 'row_avg'
        
        # Imputar con persistencia
        mask_remaining = mask_missing & (df_imputed[flag_col_name] == -1)
        prev_day_indices = df.index[mask_remaining] - pd.Timedelta(days=1)
        valid_last_day_indices = df.index.intersection(prev_day_indices)
        mask_last_day = df.index.isin(valid_last_day_indices)
        df_imputed.loc[mask_last_day, col] = df.loc[prev_day_indices.intersection(valid_last_day_indices), col].values
        df_imputed.loc[mask_last_day, flag_col_name] = 'last_day_same_hour'
    
    return df_imputed

# %%
# Funciones de imputación con climatología
def impute_with_climatology(df: pd.DataFrame, value_columns: list) -> pd.DataFrame:
    """Imputa valores usando climatología."""
    df_imputed = df.copy()
    climatology = calculate_climatology(df, value_columns)
    
    for col in value_columns:
        flag_col_name = f"i_{col}"
        mask_missing = (df[flag_col_name] == -1)
        
        for idx in df.index[mask_missing]:
            month = idx.month
            day = idx.day
            hour = idx.hour
            climatology_idx = pd.Timestamp(f'2012-{month:02d}-{day:02d} {hour:02d}:00:00')
            df_imputed.loc[idx, col] = climatology.loc[climatology_idx, col]
            df_imputed.loc[idx, flag_col_name] = 'climatology'
    
    return df_imputed

# %%
# Cargar y preparar datos
data_df = read_merged_files(INPUT_FOLDER, START_YEAR, END_YEAR)
data_df.index = pd.to_datetime(data_df.index.to_series().apply(
    lambda x: f"{x} 00:00:00" if len(str(x)) == 10 else str(x)))

# %%
# Procesar cada grupo de contaminantes
column_groups = ['cont_otres_', 'cont_pmdiez_', 'cont_pmdoscinco_', 'cont_nodos_', 'cont_co_']
data_imputed = data_df.copy()

for group in column_groups:
    print(f"\nProcesando {group}...")
    columns = [col for col in data_df.columns if col.startswith(group)]
    
    # Crear columnas de imputación
    data_imputed = create_imputation_columns(data_imputed, columns)
    
    # Imputar con métodos básicos
    data_imputed = impute_with_row_avg_or_persistence(data_imputed, columns)
    
    # Imputar con climatología
    data_imputed = impute_with_climatology(data_imputed, columns)

# %%
# Guardar resultados
create_folder(OUTPUT_FOLDER)
data_imputed.to_csv(os.path.join(OUTPUT_FOLDER, 'data_imputed_full.csv'))

# Guardar por año
for year in range(START_YEAR, END_YEAR + 1):
    yearly_data = data_imputed[data_imputed.index.year == year]
    yearly_data.to_csv(os.path.join(OUTPUT_FOLDER, f'data_imputed_{year}.csv'))

# %%
# =============================================================================
# ANÁLISIS Y VISUALIZACIÓN
# =============================================================================

# %%
# Funciones de análisis de clustering
def prepare_clustering_data(climatology_df: pd.DataFrame) -> tuple:
    """Prepara los datos para el clustering."""
    X = climatology_df.T
    X = pd.DataFrame(X)
    X = X.fillna(X.mean())
    
    if np.isinf(X.values).any():
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.mean())
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    if not np.all(np.isfinite(X_scaled)):
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    
    return X_scaled, scaler

def analizar_contaminante(df: pd.DataFrame, prefix: str, output_folder: str) -> tuple:
    """Analiza un contaminante específico usando clustering."""
    # Obtener columnas del contaminante
    columns = [col for col in df.columns if col.startswith(prefix)]
    
    # Generar climatología
    climatology = generate_climatology(df, columns)
    
    # Preparar datos para clustering
    X_scaled, scaler = prepare_clustering_data(climatology)
    
    # Clustering jerárquico
    linkage_matrix = linkage(X_scaled, method='ward')
    
    # Visualizar dendrograma
    plt.figure(figsize=(12, 8))
    dendrogram(linkage_matrix, labels=climatology.columns, leaf_rotation=90)
    plt.title(f'Dendrograma de Estaciones {prefix.replace("cont_", "").replace("_", "")}')
    plt.xlabel('Estaciones')
    plt.ylabel('Distancia')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'dendrograma_{prefix.replace("cont_", "").replace("_", "")}.png'))
    plt.close()
    
    # K-means clustering
    n_clusters = 3
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Guardar resultados del clustering
    cluster_results = pd.DataFrame({
        'Estacion': climatology.columns,
        'Cluster': clusters
    })
    cluster_results.to_csv(os.path.join(output_folder, f'clusters_{prefix.replace("cont_", "").replace("_", "")}.csv'))
    
    # Visualizar perfiles medios por cluster
    plt.figure(figsize=(15, 6))
    for i in range(n_clusters):
        estaciones_cluster = cluster_results[cluster_results['Cluster'] == i]['Estacion']
        perfil_medio = climatology[estaciones_cluster].mean(axis=1)
        plt.plot(climatology.index, perfil_medio, label=f'Cluster {i}')
    
    plt.title(f'Perfiles Medios por Cluster - {prefix.replace("cont_", "").replace("_", "")}')
    plt.xlabel('Fecha')
    plt.ylabel('Valor')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'perfiles_cluster_{prefix.replace("cont_", "").replace("_", "")}.png'))
    plt.close()
    
    return climatology, cluster_results, None

# %%
# Funciones de visualización
def plot_time_series_imputed(df: pd.DataFrame, prefix: str, title_suffix: str = "(Imputados)") -> None:
    """Visualiza las series temporales de los datos imputados."""
    # Obtener columnas del contaminante (excluyendo las columnas de flags)
    columns = [col for col in df.columns if col.startswith(prefix) and not col.startswith(f"i_{prefix}")]
    
    plt.figure(figsize=(20, 6))
    for col in columns:
        plt.plot(df.index, df[col], label=col, alpha=0.5)
    
    plt.title(f'Series Temporales de {prefix.replace("cont_", "").replace("_", "")} {title_suffix}')
    plt.xlabel('Fecha')
    plt.ylabel('Valor')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    
    # Guardar el gráfico
    create_folder(PLOTS_FOLDER)
    plt.savefig(os.path.join(PLOTS_FOLDER, f'series_temporales_{prefix.replace("cont_", "").replace("_", "")}.png'))
    plt.close()

def plot_comparison_original_imputed(original_df: pd.DataFrame, 
                                  imputed_df: pd.DataFrame, 
                                  prefix: str, 
                                  year: int = 2023) -> None:
    """Visualiza la comparación entre datos originales e imputados."""
    # Obtener columnas del contaminante
    columns = [col for col in original_df.columns if col.startswith(prefix) and not col.startswith(f"i_{prefix}")]
    
    # Filtrar datos por año
    original_year = original_df[original_df.index.year == year]
    imputed_year = imputed_df[imputed_df.index.year == year]
    
    # Crear subplots para cada estación
    n_stations = len(columns)
    n_cols = 2
    n_rows = (n_stations + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
    axes = axes.flatten()
    
    for idx, col in enumerate(columns):
        ax = axes[idx]
        ax.plot(original_year.index, original_year[col], label='Original', alpha=0.5)
        ax.plot(imputed_year.index, imputed_year[col], label='Imputado', alpha=0.5)
        ax.set_title(f'{col}')
        ax.set_xlabel('Fecha')
        ax.set_ylabel('Valor')
        ax.legend()
        ax.grid(True)
    
    # Ocultar subplots vacíos
    for idx in range(len(columns), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle(f'Comparación Original vs Imputado - {prefix.replace("cont_", "").replace("_", "")} - {year}')
    plt.tight_layout()
    
    # Guardar el gráfico
    create_folder(PLOTS_FOLDER)
    plt.savefig(os.path.join(PLOTS_FOLDER, f'comparacion_{prefix.replace("cont_", "").replace("_", "")}_{year}.png'))
    plt.close()

# %%
# Ejecutar análisis y visualización
# Crear carpetas necesarias
create_folder(CLIMATOLOGY_FOLDER)
create_folder(PLOTS_FOLDER)

# Análisis de clustering
resultados = {}
for contaminante in column_groups:
    print(f"\nAnalizando {contaminante}...")
    try:
        climatology, clusters, distancias = analizar_contaminante(
            data_df,
            contaminante,
            CLIMATOLOGY_FOLDER
        )
        resultados[contaminante] = {
            'climatology': climatology,
            'clusters': clusters,
            'distancias': distancias
        }
    except Exception as e:
        print(f"Error al analizar {contaminante}: {str(e)}")
        continue

# Visualizar series temporales
for contaminante in column_groups:
    print(f"\nGenerando gráfico de series temporales para {contaminante}...")
    plot_time_series_imputed(data_imputed, contaminante)

# Visualizar comparaciones
for contaminante in column_groups:
    print(f"\nGenerando comparación para {contaminante}...")
    plot_comparison_original_imputed(data_df, data_imputed, contaminante)

# %%
# Guardar resumen de resultados
def guardar_resumen(resultados: dict, output_folder: str) -> None:
    """Guarda un resumen de los resultados del clustering."""
    resumen = []
    
    for contaminante, datos in resultados.items():
        try:
            # Obtener distribución de clusters
            cluster_dist = datos['clusters']['Cluster'].value_counts().sort_index()
            
            resumen.append({
                'Contaminante': contaminante,
                'Número de estaciones': len(datos['clusters']),
                'Distribución de clusters': cluster_dist.to_dict()
            })
        except Exception as e:
            print(f"Error al procesar resumen para {contaminante}: {str(e)}")
            continue
    
    # Crear DataFrame con el resumen
    resumen_df = pd.DataFrame(resumen)
    resumen_df.to_csv(os.path.join(output_folder, 'resumen_clustering.csv'), index=False)
    
    # Imprimir resumen
    print("\nResumen del análisis de clustering:")
    print(resumen_df)

guardar_resumen(resultados, CLIMATOLOGY_FOLDER) 
# %%
