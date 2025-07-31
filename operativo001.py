#!/usr/bin/env python3
"""
Script para procesar múltiples archivos WRF y crear una ventana temporal extendida
para inferencia en tiempo real del modelo de contaminación.

Este script:
1. Calcula la ventana meteorológica necesaria basada en la configuración del modelo
2. Identifica y procesa múltiples archivos WRF para cubrir la ventana completa
3. Combina los datos en un archivo único con la ventana temporal extendida
4. Genera archivos procesados listos para inferencia en tiempo real

MODO OPERATIVO: Este script puede ejecutarse desde terminal con argumentos de fecha
"""

# %%
# =============================================================================
# IMPORTS Y CONFIGURACIÓN INICIAL
# =============================================================================

from proj_preproc.wrf import crop_variables_xr, calculate_relative_humidity_metpy
import os
import xarray as xr
import numpy as np
import pandas as pd
import argparse
import glob
from datetime import datetime, timedelta
import json
from parse_config import ConfigParser
# %%
# =============================================================================
# CONFIGURACIÓN POR DEFECTO
# =============================================================================

# Configuración por defecto (puede ser sobrescrita por argumentos de terminal)
DEFAULT_CONFIG = {
    'config_file': 'config22_zion.json',
    'target_datetime': '2023-05-05 09:00:00',  # Esta fecha está en CDMX
    'input_folder': '/ServerData/WRF_2017_Kraken/',
    'output_folder': '/dev/shm/tem_ram_forecast/', # './tem_var/',
    'bbox': [18.75, 20, -99.75, -98.5],
    'resolution': 1/20
}

# %%
# =============================================================================
# PARSEO DE ARGUMENTOS DE LÍNEA DE COMANDOS
# =============================================================================

def parse_arguments():
    """
    Parsea argumentos de línea de comandos.
    
    Returns:
        argparse.Namespace: Argumentos parseados
    """
    parser = argparse.ArgumentParser(
        description='Procesamiento de archivos WRF para ventana temporal extendida',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--target-datetime',
        type=str,
        help='Fecha objetivo en formato YYYY-MM-DD HH:MM:SS (hora CDMX)'
    )
    
    parser.add_argument(
        '--config-file',
        type=str,
        default=DEFAULT_CONFIG['config_file'],
        help='Archivo de configuración del modelo'
    )
    
    parser.add_argument(
        '--input-folder',
        type=str,
        default=DEFAULT_CONFIG['input_folder'],
        help='Carpeta con archivos WRF originales'
    )
    
    parser.add_argument(
        '--output-folder',
        type=str,
        default=DEFAULT_CONFIG['output_folder'],
        help='Carpeta de salida para archivos procesados'
    )
    
    parser.add_argument(
        '--bbox',
        type=float,
        nargs=4,
        default=DEFAULT_CONFIG['bbox'],
        metavar=('LAT_MIN', 'LAT_MAX', 'LON_MIN', 'LON_MAX'),
        help='Bounding box [lat_min, lat_max, lon_min, lon_max]'
    )
    
    parser.add_argument(
        '--resolution',
        type=float,
        default=DEFAULT_CONFIG['resolution'],
        help='Resolución en grados'
    )
    
    return parser.parse_args()

# %%
# =============================================================================
# FUNCIÓN PARA CONFIGURAR PARÁMETROS
# =============================================================================

def setup_configuration():
    """
    Configura los parámetros del script basado en argumentos de terminal y configuración por defecto.
    
    Returns:
        dict: Configuración final a usar
    """
    # Parsear argumentos
    args = parse_arguments()
    
    # Crear configuración final
    config = DEFAULT_CONFIG.copy()
    
    # Sobrescribir con argumentos de terminal si están disponibles
    if args.target_datetime:
        config['target_datetime'] = args.target_datetime
        print(f"📅 Fecha objetivo desde terminal: {args.target_datetime}")
    else:
        print(f"📅 Usando fecha por defecto: {config['target_datetime']}")
    
    config['config_file'] = args.config_file
    config['input_folder'] = args.input_folder
    config['output_folder'] = args.output_folder
    config['bbox'] = args.bbox
    config['resolution'] = args.resolution
    
    return config

# %%
# =============================================================================
# MOSTRAR CONFIGURACIÓN INICIAL (SOLO EN MODO NOTEBOOK)
# =============================================================================

# Solo mostrar configuración si NO se está ejecutando como script principal
if __name__ != "__main__":
    # Obtener configuración final para modo notebook/debug
    try:
        import sys
        # Guardar argumentos originales
        original_argv = sys.argv[:]
        # Simular ejecución sin argumentos para notebook
        sys.argv = [sys.argv[0]]
        
        SCRIPT_CONFIG = setup_configuration()
        
        print("🔧 CONFIGURACIÓN DEL SCRIPT (MODO NOTEBOOK)")
        print("-" * 50)
        for key, value in SCRIPT_CONFIG.items():
            print(f"   {key}: {value}")
        
        # Restaurar argumentos originales
        sys.argv = original_argv
        
    except Exception as e:
        print(f"⚠️  Error configurando modo notebook: {str(e)}")
        # Usar configuración por defecto
        SCRIPT_CONFIG = DEFAULT_CONFIG.copy()
        print("🔧 USANDO CONFIGURACIÓN POR DEFECTO")
        print("-" * 50)
        for key, value in SCRIPT_CONFIG.items():
            print(f"   {key}: {value}")

# %%
# =============================================================================
# CONFIGURACIÓN DE ZONAS HORARIAS
# =============================================================================

# Configuración de zonas horarias
TIMEZONE_CONFIG = {
    'UTC_OFFSET': -6,  # CDMX está en GMT-6 (UTC-6)
    'TIMEZONE_NAME': 'CDMX',
    'DESCRIPTION': 'Hora de Ciudad de México (GMT-6)'
}

print("🕐 CONFIGURACIÓN DE ZONAS HORARIAS")
print("-" * 50)
print(f"   Zona horaria: {TIMEZONE_CONFIG['TIMEZONE_NAME']}")
print(f"   Offset UTC: {TIMEZONE_CONFIG['UTC_OFFSET']} horas")
print(f"   Descripción: {TIMEZONE_CONFIG['DESCRIPTION']}")

# %%
# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

def get_month_folder_name(month):
    """Convierte número de mes a formato de carpeta (ej: 06_junio)"""
    month_names = {
        1: '01_enero', 2: '02_febrero', 3: '03_marzo', 4: '04_abril',
        5: '05_mayo', 6: '06_junio', 7: '07_julio', 8: '08_agosto',
        9: '09_septiembre', 10: '10_octubre', 11: '11_noviembre', 12: '12_diciembre'
    }
    return month_names[month]

def convert_cdmx_to_utc(cdmx_datetime):
    """
    Convierte una fecha/hora de CDMX a UTC.
    
    Args:
        cdmx_datetime: datetime en zona horaria CDMX
    
    Returns:
        datetime: datetime en UTC
    """
    # CDMX está en GMT-6, por lo que para convertir a UTC sumamos 6 horas
    utc_datetime = cdmx_datetime + timedelta(hours=abs(TIMEZONE_CONFIG['UTC_OFFSET']))
    return utc_datetime

def convert_utc_to_cdmx(utc_datetime):
    """
    Convierte una fecha/hora de UTC a CDMX.
    
    Args:
        utc_datetime: datetime en UTC
    
    Returns:
        datetime: datetime en zona horaria CDMX
    """
    # Para convertir de UTC a CDMX restamos 6 horas
    cdmx_datetime = utc_datetime + timedelta(hours=TIMEZONE_CONFIG['UTC_OFFSET'])
    return cdmx_datetime

def get_wrf_file_path(target_date, input_folder):
    """
    Obtiene la ruta del archivo WRF para una fecha específica.
    
    Args:
        target_date: fecha en CDMX (datetime.date)
        input_folder: carpeta de entrada
    
    Returns:
        str: ruta del archivo WRF
    """
    year_folder = str(target_date.year)
    month_folder = get_month_folder_name(target_date.month)
    
    # NOTA: Usar d02 (alta resolución) en lugar de d01 para consistencia con 1_MakeNetcdf_From_WRF.py
    # d02 es el dominio de mayor resolución espacial usado en el entrenamiento
    file_pattern = f"wrfout_d02_{target_date.strftime('%Y-%m-%d')}_00.nc"
    file_path = os.path.join(input_folder, year_folder, month_folder, file_pattern)
    
    return file_path

# %%
# =============================================================================
# CÁLCULO DE VENTANA METEOROLÓGICA
# =============================================================================

def calculate_weather_window(config):
    """
    Calcula la ventana meteorológica necesaria basada en la configuración del modelo.
    
    Args:
        config: ConfigParser con la configuración del modelo
    
    Returns:
        int: Número total de horas necesarias
    """
    prev_weather_hours = config['data_loader']['args']['prev_weather_hours']
    next_weather_hours = config['data_loader']['args']['next_weather_hours']
    auto_regresive_steps = config['test']['data_loader']['auto_regresive_steps']
    
    weather_window = prev_weather_hours + next_weather_hours + auto_regresive_steps + 1
    
    print(f"📊 CÁLCULO DE VENTANA METEOROLÓGICA:")
    print(f"   - prev_weather_hours: {prev_weather_hours}")
    print(f"   - next_weather_hours: {next_weather_hours}")
    print(f"   - auto_regresive_steps: {auto_regresive_steps}")
    print(f"   - weather_window total: {weather_window} horas")
    
    return weather_window

def get_required_dates(target_datetime_cdmx, weather_window):
    """
    Calcula las fechas de archivos WRF necesarias para cubrir la ventana meteorológica.
    Considera la diferencia horaria entre CDMX y UTC.
    
    Args:
        target_datetime_cdmx: datetime objetivo en CDMX
        weather_window: número de horas necesarias
    
    Returns:
        list: Lista de fechas (datetime.date) necesarias
    """
    print(f"🕐 CÁLCULO DE FECHAS CONSIDERANDO ZONA HORARIA")
    print("-" * 50)
    print(f"   Fecha objetivo (CDMX): {target_datetime_cdmx}")
    
    # Convertir fecha objetivo a UTC para cálculos
    target_datetime_utc = convert_cdmx_to_utc(target_datetime_cdmx)
    print(f"   Fecha objetivo (UTC): {target_datetime_utc}")
    
    # Calcular días requeridos (cada archivo tiene 24 horas)
    required_days = int(np.ceil(weather_window / 24))
    
    # Calcular fechas necesarias en UTC
    start_date_utc = target_datetime_utc.date() - timedelta(days=required_days-1)
    end_date_utc = target_datetime_utc.date() + timedelta(days=required_days-1)
    
    print(f"   Rango en UTC: {start_date_utc} a {end_date_utc}")
    
    # Generar lista de fechas en UTC
    required_dates_utc = []
    current_date_utc = start_date_utc
    while current_date_utc <= end_date_utc:
        required_dates_utc.append(current_date_utc)
        current_date_utc += timedelta(days=1)
    
    # Convertir fechas UTC a CDMX para mostrar
    required_dates_cdmx = [convert_utc_to_cdmx(datetime.combine(d, datetime.min.time())).date() 
                          for d in required_dates_utc]
    
    print(f"📅 FECHAS REQUERIDAS:")
    print(f"   - Días necesarios: {required_days}")
    print(f"   - Rango UTC: {start_date_utc} a {end_date_utc}")
    print(f"   - Rango CDMX: {required_dates_cdmx[0]} a {required_dates_cdmx[-1]}")
    print(f"   - Fechas UTC: {[d.strftime('%Y-%m-%d') for d in required_dates_utc]}")
    print(f"   - Fechas CDMX: {[d.strftime('%Y-%m-%d') for d in required_dates_cdmx]}")
    
    # Retornar fechas en UTC (que es como están nombrados los archivos WRF)
    return required_dates_utc

# %%
# =============================================================================
# PROCESAMIENTO DE ARCHIVOS WRF INDIVIDUALES
# =============================================================================

def process_wrf_file_for_window(file_path, bbox, resolution):
    """
    Procesa un archivo WRF individual para la ventana temporal.
    
    Args:
        file_path: Ruta al archivo WRF
        bbox: Bounding box [minlat, maxlat, minlon, maxlon]
        resolution: Resolución en grados
    
    Returns:
        xarray.Dataset: Dataset procesado o None si hay error
    """
    print(f"🔧 Procesando archivo: {os.path.basename(file_path)}")
    
    # Variables a procesar
    variable_names = ['T2', 'U10', 'V10', 'RAINC', 'RAINNC', 'SWDOWN', 'GLW', 'Q2', 'PSFC']
    times = range(24)  # Procesar 24 horas
    
    try:
        # Cargar el archivo
        ds = xr.open_dataset(file_path, decode_times=False)
        
        # Obtener la fecha del nombre del archivo (en UTC)
        file_name = os.path.basename(file_path)
        date_str = file_name.split('_')[2]  # Obtiene la parte de la fecha del nombre
        file_date_utc = datetime.strptime(date_str, '%Y-%m-%d')
        
        # Convertir a CDMX para mostrar
        file_date_cdmx = convert_utc_to_cdmx(file_date_utc)
        
        print(f"   📅 Fecha del archivo (UTC): {file_date_utc.strftime('%Y-%m-%d')}")
        print(f"   📅 Fecha del archivo (CDMX): {file_date_cdmx.strftime('%Y-%m-%d')}")
        
        # Procesar variables de lluvia
        if 'RAINC' in variable_names and 'RAINNC' in variable_names:
            print("   🌧️  Calculando lluvia total...")
            ds['RAIN'] = ds['RAINC'] + ds['RAINNC']
            rain_values = ds['RAIN'].values
            rain_diff = np.zeros_like(rain_values)
            rain_diff[1:,:,:] = rain_values[1:,:,:] - rain_values[:-1,:,:]
            ds['RAIN'] = xr.DataArray(rain_diff, dims=ds['RAIN'].dims, coords=ds['RAIN'].coords)
            ds['RAIN'] = ds['RAIN'].where(ds['RAIN'] > 0, 0)
            ds = ds.drop_vars(['RAINC', 'RAINNC'])
            variable_names.remove('RAINC')
            variable_names.remove('RAINNC')
            variable_names.append('RAIN')

        # Calcular velocidad del viento
        if 'U10' in variable_names and 'V10' in variable_names:
            print("   💨 Calculando velocidad del viento...")
            ds['WS10'] = np.sqrt(ds['U10']**2 + ds['V10']**2)
            variable_names.append('WS10')

        # Calcular humedad relativa
        print("   💧 Calculando humedad relativa...")
        T2 = ds['T2'].values
        PSFC = ds['PSFC'].values
        Q2 = ds['Q2'].values
        
        RH = calculate_relative_humidity_metpy(T2, PSFC, Q2)
        ds['RH'] = xr.DataArray(RH, dims=ds['T2'].dims, coords=ds['T2'].coords)
        variable_names.append('RH')
        variable_names.remove('Q2')
        variable_names.remove('PSFC')

        # Recortar al área de interés
        print(f"   ✂️  Recortando al área de interés: {bbox}")
        cropped_ds, newLAT, newLon = crop_variables_xr(ds, variable_names, bbox, times)

        # Crear nueva grid con resolución específica
        new_lat = np.arange(bbox[0], bbox[1], resolution)
        new_lon = np.arange(bbox[2], bbox[3], resolution)
        
        # Interpolar a nueva grid
        print(f"   🔄 Interpolando a resolución: {resolution} grados")
        cropped_ds = cropped_ds.interp(
            lat=new_lat,
            lon=new_lon,
            method='linear'
        )

        # Ajustar tiempo a hora local (GMT-6)
        # El archivo WRF está en UTC, pero queremos mostrar en CDMX
        first_datetime_utc = file_date_utc.replace(hour=0, minute=0, second=0, microsecond=0)
        first_datetime_cdmx = convert_utc_to_cdmx(first_datetime_utc)
        
        cropped_ds['time'].attrs.update({
            'units': f'hours since {first_datetime_cdmx.strftime("%Y-%m-%d %H:%M:%S")}',
            'calendar': 'standard',
            'axis': 'T',
            'long_name': 'time',
            'standard_name': 'time',
            'timezone': 'CDMX (GMT-6)'
        })

        print(f"   ✅ Archivo procesado exitosamente")
        print(f"   📊 Variables finales: {list(cropped_ds.data_vars.keys())}")
        print(f"   📐 Dimensiones: {dict(cropped_ds.dims)}")
        print(f"   🕐 Tiempo base (CDMX): {first_datetime_cdmx}")
        
        return cropped_ds

    except Exception as e:
        print(f"   ❌ Error procesando archivo: {str(e)}")
        return None

def create_zero_dataset(date_utc, bbox, resolution):
    """
    Crea un dataset de ceros para fechas donde no hay archivos WRF disponibles.

    Args:
    date_utc: Fecha en UTC para el dataset
    bbox: Bounding box [minlat, maxlat, minlon, maxlon]
    resolution: Resolución en grados

    Returns:
    xarray.Dataset: Dataset de ceros
    """
    try:
        # Crear coordenadas
        lat = np.arange(bbox[0], bbox[1], resolution)
        lon = np.arange(bbox[2], bbox[3], resolution)
        time = range(24)
        
        # Variables meteorológicas
        variables = ['T2', 'U10', 'V10', 'RAIN', 'SWDOWN', 'GLW', 'WS10', 'RH']
        
        # Crear dataset
        data_vars = {}
        for var in variables:
            data_vars[var] = xr.DataArray(
                np.zeros((24, len(lat), len(lon))),
                coords=[('time', time), ('lat', lat), ('lon', lon)],
                attrs={'units': 'unknown', 'long_name': var}
            )
        
        ds = xr.Dataset(data_vars)
        
        # Ajustar tiempo a hora local (GMT-6)
        first_datetime_utc = date_utc.replace(hour=0, minute=0, second=0, microsecond=0)
        first_datetime_cdmx = convert_utc_to_cdmx(first_datetime_utc)
        
        ds['time'].attrs.update({
            'units': f'hours since {first_datetime_cdmx.strftime("%Y-%m-%d %H:%M:%S")}',
            'calendar': 'standard',
            'axis': 'T',
            'long_name': 'time',
            'standard_name': 'time',
            'timezone': 'CDMX (GMT-6)'
        })
        
        return ds
        
    except Exception as e:
        print(f"   ❌ Error creando dataset de ceros: {str(e)}")
        return None

# %%
# =============================================================================
# COMBINACIÓN DE DATASETS
# =============================================================================

def combine_wrf_datasets(datasets_list, weather_window):
    """
    Combina múltiples datasets WRF en uno solo con la ventana temporal extendida.

    Args:
        datasets_list: Lista de datasets WRF procesados
        weather_window: Número de horas necesarias

    Returns:
        xarray.Dataset: Dataset combinado
    """
    print(f"\n🔗 COMBINANDO DATASETS WRF")
    print("-" * 50)
    
    if not datasets_list:
        print("❌ No hay datasets para combinar")
        return None
    
    # Filtrar datasets válidos
    valid_datasets = [ds for ds in datasets_list if ds is not None]
    
    if not valid_datasets:
        print("❌ No hay datasets válidos para combinar")
        return None
    
    print(f"📊 Datasets válidos: {len(valid_datasets)}")
    
    try:
        # Concatenar datasets a lo largo del eje temporal
        combined_ds = xr.concat(valid_datasets, dim='time')
        
        # Verificar que tenemos suficientes horas
        total_hours = len(combined_ds['time'])
        print(f"📊 Horas totales disponibles: {total_hours}")
        print(f"📊 Horas requeridas: {weather_window}")
        
        if total_hours < weather_window:
            print(f"⚠️  Horas insuficientes: {total_hours} < {weather_window}")
            print("   Usando todas las horas disponibles")
        else:
            # Tomar solo las horas necesarias
            combined_ds = combined_ds.isel(time=slice(0, weather_window))
            print(f"✅ Tomadas las primeras {weather_window} horas")
        
        print(f"📊 Dataset combinado:")
        print(f"   - Variables: {list(combined_ds.data_vars.keys())}")
        print(f"   - Dimensiones: {dict(combined_ds.dims)}")
        print(f"   - Rango temporal: {combined_ds['time'].values[0]} a {combined_ds['time'].values[-1]}")
        
        return combined_ds
        
    except Exception as e:
        print(f"❌ Error combinando datasets: {str(e)}")
        return None

# %%
# =============================================================================
# FUNCIÓN PRINCIPAL DE PROCESAMIENTO
# =============================================================================

def process_wrf_window(target_datetime_cdmx, config, input_folder, output_folder, bbox, resolution):
    """
    Procesa múltiples archivos WRF para crear archivos individuales por día.
    
    Args:
        target_datetime_cdmx: datetime objetivo en CDMX
        config: configuración del modelo
        input_folder: carpeta con archivos WRF originales
        output_folder: carpeta de salida
        bbox: bounding box [minlat, maxlat, minlon, maxlon]
        resolution: resolución en grados
    
    Returns:
        bool: True si el procesamiento fue exitoso
    """
    print(f"\n🔧 PROCESANDO VENTANA WRF PARA ARCHIVOS INDIVIDUALES")
    print("-" * 60)
    print(f"   Fecha objetivo (CDMX): {target_datetime_cdmx}")
    print(f"   Carpeta de entrada: {input_folder}")
    print(f"   Carpeta de salida: {output_folder}")
    print(f"   Bounding box: {bbox}")
    print(f"   Resolución: {resolution}")
    
    try:
        # Calcular ventana meteorológica
        weather_window = calculate_weather_window(config)
        required_dates_utc = get_required_dates(target_datetime_cdmx, weather_window)
        
        print(f"\n📊 VENTANA METEOROLÓGICA:")
        print(f"   - Horas requeridas: {weather_window}")
        print(f"   - Días requeridos: {len(required_dates_utc)}")
        
        # Crear directorio de salida si no existe
        os.makedirs(output_folder, exist_ok=True)
        
        # Lista para almacenar resultados
        processed_files = []
        missing_files = []
        
        # Procesar cada archivo individualmente
        for i, date_utc in enumerate(required_dates_utc):
            date_cdmx = convert_utc_to_cdmx(datetime.combine(date_utc, datetime.min.time())).date()
            
            print(f"\n📅 PROCESANDO DÍA {i+1}/{len(required_dates_utc)}:")
            print(f"   - Fecha UTC: {date_utc.strftime('%Y-%m-%d')}")
            print(f"   - Fecha CDMX: {date_cdmx.strftime('%Y-%m-%d')}")
            
            # Obtener ruta del archivo WRF
            wrf_file_path = get_wrf_file_path(date_utc, input_folder)
            
            # Verificar si existe el archivo
            if os.path.exists(wrf_file_path):
                print(f"   ✅ Archivo encontrado: {os.path.basename(wrf_file_path)}")
                
                # Procesar archivo individual
                try:
                    processed_ds = process_wrf_file_for_window(wrf_file_path, bbox, resolution)
                    
                    if processed_ds is not None:
                        # Generar nombre de archivo de salida
                        output_filename = f"{date_cdmx.strftime('%Y-%m-%d')}.nc"
                        output_file_path = os.path.join(output_folder, output_filename)
                        
                        # Guardar archivo procesado
                        print(f"   💾 Guardando: {output_filename}")
                        processed_ds.to_netcdf(output_file_path)
                        
                        # Verificar que se guardó correctamente
                        if os.path.exists(output_file_path):
                            print(f"   ✅ Archivo guardado exitosamente")
                            
                            # Mostrar información del archivo
                            print(f"   📊 Variables: {list(processed_ds.data_vars.keys())}")
                            print(f"   📐 Dimensiones: {dict(processed_ds.dims)}")
                            
                            processed_files.append({
                                'date_utc': date_utc,
                                'date_cdmx': date_cdmx,
                                'input_file': wrf_file_path,
                                'output_file': output_file_path,
                                'variables': list(processed_ds.data_vars.keys()),
                                'dimensions': dict(processed_ds.dims)
                            })
                        else:
                            print(f"   ❌ Error: Archivo no se guardó correctamente")
                            missing_files.append((date_utc, date_cdmx, "Error guardando archivo"))
                    else:
                        print(f"   ❌ Error: No se pudo procesar el archivo")
                        missing_files.append((date_utc, date_cdmx, "Error procesando archivo"))
                        
                except Exception as e:
                    print(f"   ❌ Error procesando archivo: {str(e)}")
                    missing_files.append((date_utc, date_cdmx, f"Error: {str(e)}"))
                    
            else:
                print(f"   ❌ Archivo no encontrado: {os.path.basename(wrf_file_path)}")
                
                # Crear dataset de ceros para archivo faltante
                try:
                    print(f"   🔧 Creando dataset de ceros...")
                    zero_ds = create_zero_dataset(date_utc, bbox, resolution)
                    
                    if zero_ds is not None:
                        # Generar nombre de archivo de salida
                        output_filename = f"{date_cdmx.strftime('%Y-%m-%d')}.nc"
                        output_file_path = os.path.join(output_folder, output_filename)
                        
                        # Guardar archivo de ceros
                        print(f"   💾 Guardando dataset de ceros: {output_filename}")
                        zero_ds.to_netcdf(output_file_path)
                        
                        if os.path.exists(output_file_path):
                            print(f"   ✅ Dataset de ceros guardado exitosamente")
                            
                            processed_files.append({
                                'date_utc': date_utc,
                                'date_cdmx': date_cdmx,
                                'input_file': 'N/A (ceros)',
                                'output_file': output_file_path,
                                'variables': list(zero_ds.data_vars.keys()),
                                'dimensions': dict(zero_ds.dims),
                                'note': 'Dataset de ceros (archivo original faltante)'
                            })
                        else:
                            print(f"   ❌ Error: Dataset de ceros no se guardó")
                            missing_files.append((date_utc, date_cdmx, "Error guardando dataset de ceros"))
                    else:
                        print(f"   ❌ Error: No se pudo crear dataset de ceros")
                        missing_files.append((date_utc, date_cdmx, "Error creando dataset de ceros"))
                        
                except Exception as e:
                    print(f"   ❌ Error creando dataset de ceros: {str(e)}")
                    missing_files.append((date_utc, date_cdmx, f"Error dataset de ceros: {str(e)}"))
        
        # Resumen final
        print(f"\n📋 RESUMEN DEL PROCESAMIENTO:")
        print("-" * 50)
        print(f"   - Total de días requeridos: {len(required_dates_utc)}")
        print(f"   - Archivos procesados exitosamente: {len(processed_files)}")
        print(f"   - Archivos con errores: {len(missing_files)}")
        
        if processed_files:
            print(f"\n✅ ARCHIVOS PROCESADOS EXITOSAMENTE:")
            for file_info in processed_files:
                print(f"   - {file_info['date_cdmx'].strftime('%Y-%m-%d')} CDMX")
                print(f"     → {os.path.basename(file_info['output_file'])}")
                if 'note' in file_info:
                    print(f"     → Nota: {file_info['note']}")
        
        if missing_files:
            print(f"\n❌ ARCHIVOS CON ERRORES:")
            for date_utc, date_cdmx, error in missing_files:
                print(f"   - {date_cdmx.strftime('%Y-%m-%d')} CDMX: {error}")
        
        # Verificar archivos en disco
        print(f"\n🔍 VERIFICACIÓN DE ARCHIVOS EN DISCO:")
        print("-" * 50)
        
        files_on_disk = []
        for date_utc in required_dates_utc:
            date_cdmx = convert_utc_to_cdmx(datetime.combine(date_utc, datetime.min.time())).date()
            expected_file = os.path.join(output_folder, f"{date_cdmx.strftime('%Y-%m-%d')}.nc")
            
            if os.path.exists(expected_file):
                files_on_disk.append(expected_file)
                print(f"   ✅ {os.path.basename(expected_file)}")
            else:
                print(f"   ❌ {os.path.basename(expected_file)} (faltante)")
        
        print(f"\n📊 ESTADÍSTICAS FINALES:")
        print(f"   - Archivos esperados: {len(required_dates_utc)}")
        print(f"   - Archivos en disco: {len(files_on_disk)}")
        print(f"   - Tasa de éxito: {(len(files_on_disk) / len(required_dates_utc)) * 100:.1f}%")
        
        # Retornar éxito si al menos se procesó un archivo
        success = len(processed_files) > 0
        
        if success:
            print(f"\n🎉 PROCESAMIENTO COMPLETADO EXITOSAMENTE!")
            
            # Verificar archivos generados
            print(f"\n📁 ARCHIVOS GENERADOS EN: {output_folder}")
            
            # Listar archivos .nc en el directorio de salida
            nc_files = [f for f in os.listdir(output_folder) if f.endswith('.nc')]
            
            if nc_files:
                print(f"   ✅ Se encontraron {len(nc_files)} archivos:")
                for nc_file in sorted(nc_files):
                    file_path = os.path.join(output_folder, nc_file)
                    file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                    print(f"   📄 {nc_file} ({file_size:.1f} MB)")
            else:
                print(f"   ❌ No se encontraron archivos .nc")
        else:
            print(f"\n❌ ERROR EN EL PROCESAMIENTO")
        
        return success
        
    except Exception as e:
        print(f"\n❌ ERROR GENERAL EN EL PROCESAMIENTO: {str(e)}")
        return False


# =============================================================================
# CÓDIGO DE NOTEBOOK/DEBUG (SOLO SE EJECUTA EN MODO NOTEBOOK)
# =============================================================================

if __name__ != "__main__":
    # %%
    # =============================================================================
    # CARGA DE CONFIGURACIÓN
    # =============================================================================

    print("📋 CARGANDO CONFIGURACIÓN")
    print("-" * 50)

    try:
        from utils import read_json
        config_dict = read_json(SCRIPT_CONFIG['config_file'])
        config = ConfigParser(config_dict)
        
        print(f"✅ Configuración cargada desde: {SCRIPT_CONFIG['config_file']}")
        
        # Mostrar parámetros clave
        prev_weather_hours = config['data_loader']['args']['prev_weather_hours']
        next_weather_hours = config['data_loader']['args']['next_weather_hours']
        auto_regresive_steps = config['test']['data_loader']['auto_regresive_steps']
        
        print(f"📊 Parámetros del modelo:")
        print(f"   - prev_weather_hours: {prev_weather_hours}")
        print(f"   - next_weather_hours: {next_weather_hours}")
        print(f"   - auto_regresive_steps: {auto_regresive_steps}")
        
    except Exception as e:
        print(f"❌ Error cargando configuración: {str(e)}")
        config = None

    # %%
    # =============================================================================
    # PREPARACIÓN DE FECHA OBJETIVO
    # =============================================================================

    print("📅 PREPARANDO FECHA OBJETIVO")
    print("-" * 50)

    try:
        target_datetime_cdmx = datetime.strptime(SCRIPT_CONFIG['target_datetime'], '%Y-%m-%d %H:%M:%S')
        target_datetime_utc = convert_cdmx_to_utc(target_datetime_cdmx)
        
        print(f"✅ Fecha objetivo (CDMX): {target_datetime_cdmx}")
        print(f"✅ Fecha objetivo (UTC): {target_datetime_utc}")
        print(f"🕐 Diferencia horaria: {TIMEZONE_CONFIG['UTC_OFFSET']} horas")
        
    except Exception as e:
        print(f"❌ Error parseando fecha: {str(e)}")
        target_datetime_cdmx = None
        target_datetime_utc = None

    # %%
    # =============================================================================
    # CÁLCULO DE VENTANA METEOROLÓGICA
    # =============================================================================

    if config is not None and target_datetime_cdmx is not None:
        print("📊 CALCULANDO VENTANA METEOROLÓGICA")
        print("-" * 50)
        
        weather_window = calculate_weather_window(config)
        required_dates_utc = get_required_dates(target_datetime_cdmx, weather_window)
        
        print(f"✅ Ventana meteorológica calculada: {weather_window} horas")
        print(f"✅ Fechas requeridas: {len(required_dates_utc)} días")

    # %%
    # =============================================================================
    # VERIFICACIÓN DE ARCHIVOS WRF
    # =============================================================================

    print("🔍 VERIFICANDO ARCHIVOS WRF")
    print("-" * 50)

    if 'required_dates_utc' in locals():
        available_files = []
        missing_files = []
        
        for date_utc in required_dates_utc:
            file_path = get_wrf_file_path(date_utc, SCRIPT_CONFIG['input_folder'])
            date_cdmx = convert_utc_to_cdmx(datetime.combine(date_utc, datetime.min.time())).date()
            
            if os.path.exists(file_path):
                available_files.append((date_utc, date_cdmx, file_path))
                print(f"✅ {date_utc.strftime('%Y-%m-%d')} UTC ({date_cdmx.strftime('%Y-%m-%d')} CDMX): {os.path.basename(file_path)}")
            else:
                missing_files.append((date_utc, date_cdmx, file_path))
                print(f"❌ {date_utc.strftime('%Y-%m-%d')} UTC ({date_cdmx.strftime('%Y-%m-%d')} CDMX): No encontrado")
        
        print(f"\n📊 RESUMEN:")
        print(f"   - Archivos disponibles: {len(available_files)}")
        print(f"   - Archivos faltantes: {len(missing_files)}")
        
        if missing_files:
            print(f"\n⚠️  ARCHIVOS FALTANTES:")
            for date_utc, date_cdmx, file_path in missing_files:
                print(f"   - {date_utc.strftime('%Y-%m-%d')} UTC ({date_cdmx.strftime('%Y-%m-%d')} CDMX)")
                print(f"     Ruta esperada: {file_path}")

    # %%
    # =============================================================================
    # DIAGNÓSTICO DE VARIABLES
    # =============================================================================

    print("🔍 DIAGNÓSTICO DE VARIABLES")
    print("-" * 50)

    # Verificar variables requeridas
    required_vars = ['config', 'target_datetime_cdmx']
    available_vars = []

    for var in required_vars:
        if var in locals():
            available_vars.append(var)
            print(f"✅ {var}: Disponible")
        else:
            print(f"❌ {var}: FALTANTE")

    print(f"\n📊 RESUMEN:")
    print(f"   - Variables requeridas: {len(required_vars)}")
    print(f"   - Variables disponibles: {len(available_vars)}")
    print(f"   - Variables faltantes: {len(required_vars) - len(available_vars)}")

    # Verificar archivo de configuración
    config_file = SCRIPT_CONFIG['config_file']
    if os.path.exists(config_file):
        print(f"✅ Archivo de configuración existe: {config_file}")
    else:
        print(f"❌ Archivo de configuración NO existe: {config_file}")

    # Verificar si se cargó la configuración
    if 'config' in locals() and config is not None:
        print(f"✅ Configuración cargada correctamente")
        try:
            # Verificar parámetros específicos
            prev_weather_hours = config['data_loader']['args']['prev_weather_hours']
            next_weather_hours = config['data_loader']['args']['next_weather_hours']
            auto_regresive_steps = config['test']['data_loader']['auto_regresive_steps']
            print(f"✅ Parámetros del modelo verificados")
        except Exception as e:
            print(f"❌ Error verificando parámetros del modelo: {str(e)}")
    else:
        print(f"❌ Configuración NO cargada")

    # Verificar fecha objetivo
    if 'target_datetime_cdmx' in locals() and target_datetime_cdmx is not None:
        print(f"✅ Fecha objetivo cargada: {target_datetime_cdmx}")
    else:
        print(f"❌ Fecha objetivo NO cargada")

    # %%
    # =============================================================================
    # PROCESAMIENTO DE VENTANA WRF
    # =============================================================================

    print("🚀 EJECUTANDO PROCESAMIENTO DE VENTANA WRF")
    print("-" * 50)

    # Verificación más detallada
    config_ok = 'config' in locals() and config is not None
    date_ok = 'target_datetime_cdmx' in locals() and target_datetime_cdmx is not None

    print(f"📊 VERIFICACIÓN FINAL:")
    print(f"   - Configuración: {'✅ OK' if config_ok else '❌ FALTA'}")
    print(f"   - Fecha objetivo: {'✅ OK' if date_ok else '❌ FALTA'}")

    if config_ok and date_ok:
        print(f"✅ Todas las variables están disponibles")
        
        # Crear directorio de salida si no existe
        os.makedirs(SCRIPT_CONFIG['output_folder'], exist_ok=True)
        
        # Procesar ventana WRF
        success = process_wrf_window(
            target_datetime_cdmx=target_datetime_cdmx,
            config=config,
            input_folder=SCRIPT_CONFIG['input_folder'],
            output_folder=SCRIPT_CONFIG['output_folder'],
            bbox=SCRIPT_CONFIG['bbox'],
            resolution=SCRIPT_CONFIG['resolution']
        )
        
        if success:
            print(f"\n🎉 PROCESAMIENTO COMPLETADO EXITOSAMENTE!")
            
            # Verificar archivos generados
            print(f"\n📁 ARCHIVOS GENERADOS EN: {SCRIPT_CONFIG['output_folder']}")
            
            # Listar archivos .nc en el directorio de salida
            nc_files = [f for f in os.listdir(SCRIPT_CONFIG['output_folder']) if f.endswith('.nc')]
            
            if nc_files:
                print(f"   ✅ Se encontraron {len(nc_files)} archivos:")
                for nc_file in sorted(nc_files):
                    file_path = os.path.join(SCRIPT_CONFIG['output_folder'], nc_file)
                    file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                    print(f"   📄 {nc_file} ({file_size:.1f} MB)")
                    
                    # Mostrar información del archivo
                    try:
                        ds = xr.open_dataset(file_path)
                        print(f"     📊 Variables: {list(ds.data_vars.keys())}")
                        print(f"     📐 Dimensiones: {dict(ds.dims)}")
                        ds.close()
                    except Exception as e:
                        print(f"     ⚠️  Error leyendo archivo: {str(e)}")
            else:
                print(f"   ❌ No se encontraron archivos .nc")
        else:
            print(f"\n❌ ERROR EN EL PROCESAMIENTO")
    else:
        print("❌ No se puede ejecutar el procesamiento - faltan variables de configuración")
        
        if not config_ok:
            print("   💡 Solución: Verificar que el archivo config22_zion.json existe y es válido")
        if not date_ok:
            print("   💡 Solución: Verificar que la fecha objetivo está en formato correcto")


# %%
# =============================================================================
# HASTA AQUÍ SE CORRIÓ PARA GENERAR LOS NETCDFS DE LAS FECHAS QUE SE NECESITAN
# =============================================================================

# %%
# =============================================================================
# FUNCIÓN PRINCIPAL PARA EJECUCIÓN DESDE LÍNEA DE COMANDOS
# =============================================================================

def clean_output_folder(output_folder):
    """
    Limpia archivos *.nc existentes en la carpeta de salida.
    
    Args:
        output_folder: Carpeta de salida a limpiar
    
    Returns:
        int: Número de archivos eliminados
    """
    print(f"🗑️  LIMPIANDO CARPETA DE SALIDA")
    print("-" * 50)
    print(f"   Carpeta: {output_folder}")
    
    # Verificar si la carpeta existe
    if not os.path.exists(output_folder):
        print(f"   ⚠️  La carpeta no existe, se creará automáticamente")
        return 0
    
    # Buscar archivos *.nc
    nc_pattern = os.path.join(output_folder, "*.nc")
    nc_files = glob.glob(nc_pattern)
    
    if not nc_files:
        print(f"   ✅ No hay archivos *.nc para limpiar")
        return 0
    
    print(f"   📊 Se encontraron {len(nc_files)} archivos *.nc para eliminar:")
    
    deleted_count = 0
    errors = []
    
    for nc_file in nc_files:
        try:
            file_size = os.path.getsize(nc_file) / (1024 * 1024)  # MB
            print(f"   🗑️  Eliminando: {os.path.basename(nc_file)} ({file_size:.1f} MB)")
            os.remove(nc_file)
            deleted_count += 1
        except Exception as e:
            error_msg = f"Error eliminando {os.path.basename(nc_file)}: {str(e)}"
            errors.append(error_msg)
            print(f"   ❌ {error_msg}")
    
    print(f"\n📊 RESUMEN DE LIMPIEZA:")
    print(f"   - Archivos encontrados: {len(nc_files)}")
    print(f"   - Archivos eliminados: {deleted_count}")
    print(f"   - Errores: {len(errors)}")
    
    if errors:
        print(f"\n❌ ERRORES EN LA LIMPIEZA:")
        for error in errors:
            print(f"   - {error}")
    
    if deleted_count > 0:
        print(f"✅ Limpieza completada exitosamente")
    
    return deleted_count

def main():
    """
    Función principal para ejecutar el script desde línea de comandos.
    """
    print("🚀 INICIANDO PROCESAMIENTO WRF")
    print("=" * 60)
    
    try:
        # Obtener configuración
        script_config = setup_configuration()
        
        # Crear directorio de salida si no existe
        os.makedirs(script_config['output_folder'], exist_ok=True)
        
        # =========================================================================
        # LIMPIEZA EXPLÍCITA DE ARCHIVOS *.nc EN OUTPUT_FOLDER
        # =========================================================================
        print(f"\n🗑️  LIMPIANDO ARCHIVOS *.nc EN OUTPUT_FOLDER")
        print("-" * 50)
        print(f"   Carpeta objetivo: {script_config['output_folder']}")
        
        # Buscar archivos *.nc específicamente en output_folder
        nc_pattern = os.path.join(script_config['output_folder'], "*.nc")
        nc_files_to_delete = glob.glob(nc_pattern)
        
        if nc_files_to_delete:
            print(f"   📊 Encontrados {len(nc_files_to_delete)} archivos *.nc para eliminar:")
            
            for nc_file in nc_files_to_delete:
                try:
                    file_size = os.path.getsize(nc_file) / (1024 * 1024)  # MB
                    filename = os.path.basename(nc_file)
                    print(f"   🗑️  Eliminando: {filename} ({file_size:.1f} MB)")
                    os.remove(nc_file)
                except Exception as e:
                    print(f"   ❌ Error eliminando {os.path.basename(nc_file)}: {str(e)}")
            
            print(f"   ✅ Limpieza de archivos *.nc completada")
        else:
            print(f"   ✅ No hay archivos *.nc para eliminar")
        
        # =========================================================================
        # CONTINUAR CON PROCESAMIENTO NORMAL
        # =========================================================================
        
        print("\n📋 CARGANDO CONFIGURACIÓN DEL MODELO")
        print("-" * 50)
        
        # Cargar configuración del modelo
        from utils import read_json
        config_dict = read_json(script_config['config_file'])
        config = ConfigParser(config_dict)
        
        print(f"✅ Configuración cargada desde: {script_config['config_file']}")
        
        # Mostrar parámetros clave
        prev_weather_hours = config['data_loader']['args']['prev_weather_hours']
        next_weather_hours = config['data_loader']['args']['next_weather_hours']
        auto_regresive_steps = config['test']['data_loader']['auto_regresive_steps']
        
        print(f"📊 Parámetros del modelo:")
        print(f"   - prev_weather_hours: {prev_weather_hours}")
        print(f"   - next_weather_hours: {next_weather_hours}")
        print(f"   - auto_regresive_steps: {auto_regresive_steps}")
        
        print("\n📅 PREPARANDO FECHA OBJETIVO")
        print("-" * 50)
        
        # Preparar fecha objetivo
        target_datetime_cdmx = datetime.strptime(script_config['target_datetime'], '%Y-%m-%d %H:%M:%S')
        target_datetime_utc = convert_cdmx_to_utc(target_datetime_cdmx)
        
        print(f"✅ Fecha objetivo (CDMX): {target_datetime_cdmx}")
        print(f"✅ Fecha objetivo (UTC): {target_datetime_utc}")
        print(f"🕐 Diferencia horaria: {TIMEZONE_CONFIG['UTC_OFFSET']} horas")
        
        print("\n📊 CALCULANDO VENTANA METEOROLÓGICA")
        print("-" * 50)
        
        # Calcular ventana meteorológica
        weather_window = calculate_weather_window(config)
        required_dates_utc = get_required_dates(target_datetime_cdmx, weather_window)
        
        print(f"✅ Ventana meteorológica calculada: {weather_window} horas")
        print(f"✅ Fechas requeridas: {len(required_dates_utc)} días")
        
        print("\n🔍 VERIFICANDO ARCHIVOS WRF")
        print("-" * 50)
        
        # Verificar archivos WRF disponibles
        available_files = []
        missing_files = []
        
        for date_utc in required_dates_utc:
            file_path = get_wrf_file_path(date_utc, script_config['input_folder'])
            date_cdmx = convert_utc_to_cdmx(datetime.combine(date_utc, datetime.min.time())).date()
            
            if os.path.exists(file_path):
                available_files.append((date_utc, date_cdmx, file_path))
                print(f"✅ {date_utc.strftime('%Y-%m-%d')} UTC ({date_cdmx.strftime('%Y-%m-%d')} CDMX): {os.path.basename(file_path)}")
            else:
                missing_files.append((date_utc, date_cdmx, file_path))
                print(f"❌ {date_utc.strftime('%Y-%m-%d')} UTC ({date_cdmx.strftime('%Y-%m-%d')} CDMX): No encontrado")
        
        print(f"\n📊 RESUMEN:")
        print(f"   - Archivos disponibles: {len(available_files)}")
        print(f"   - Archivos faltantes: {len(missing_files)}")
        
        if missing_files:
            print(f"\n⚠️  ARCHIVOS FALTANTES:")
            for date_utc, date_cdmx, file_path in missing_files:
                print(f"   - {date_utc.strftime('%Y-%m-%d')} UTC ({date_cdmx.strftime('%Y-%m-%d')} CDMX)")
                print(f"     Ruta esperada: {file_path}")
        
        print("\n🚀 EJECUTANDO PROCESAMIENTO DE VENTANA WRF")
        print("-" * 50)
        
        # Procesar ventana WRF
        success = process_wrf_window(
            target_datetime_cdmx=target_datetime_cdmx,
            config=config,
            input_folder=script_config['input_folder'],
            output_folder=script_config['output_folder'],
            bbox=script_config['bbox'],
            resolution=script_config['resolution']
        )
        
        if success:
            print(f"\n🎉 PROCESAMIENTO COMPLETADO EXITOSAMENTE!")
            
            # Verificar archivos generados
            print(f"\n📁 ARCHIVOS GENERADOS EN: {script_config['output_folder']}")
            
            # Listar archivos .nc en el directorio de salida
            nc_files = [f for f in os.listdir(script_config['output_folder']) if f.endswith('.nc')]
            
            if nc_files:
                print(f"   ✅ Se encontraron {len(nc_files)} archivos:")
                for nc_file in sorted(nc_files):
                    file_path = os.path.join(script_config['output_folder'], nc_file)
                    file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                    print(f"   📄 {nc_file} ({file_size:.1f} MB)")
                    
                    # Mostrar información del archivo
                    try:
                        ds = xr.open_dataset(file_path)
                        print(f"     📊 Variables: {list(ds.data_vars.keys())}")
                        print(f"     📐 Dimensiones: {dict(ds.dims)}")
                        ds.close()
                    except Exception as e:
                        print(f"     ⚠️  Error leyendo archivo: {str(e)}")
            else:
                print(f"   ❌ No se encontraron archivos .nc")
        else:
            print(f"\n❌ ERROR EN EL PROCESAMIENTO")
            return 1
        
        return 0
        
    except Exception as e:
        print(f"\n❌ ERROR GENERAL: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

# %%
# =============================================================================
# EJECUCIÓN DEL SCRIPT
# =============================================================================

if __name__ == "__main__":
    # Ejecutar función principal cuando el script se llama desde línea de comandos
    exit_code = main()
    exit(exit_code)
else:
    # Ejecutar configuración cuando se importa como módulo o en notebook
    print("📓 Modo notebook/import detectado - configuración cargada")
    print("💡 Para ejecutar procesamiento completo, llama a main()")


# =============================================================================
# CÓDIGO ORIGINAL DE NOTEBOOK/DEBUG (MANTENER PARA COMPATIBILIDAD)
# =============================================================================



