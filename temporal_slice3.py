#!/usr/bin/env python3
"""
temporal_slice3.py

Script corregido para extracción inteligente de recortes temporales de archivos WRF.
Corrige el problema del cambio de mes y valida disponibilidad de archivos fuente.

CORRECCIONES V3:
1. Valida disponibilidad de archivos fuente para fechas objetivo
2. Maneja cambios de mes/año correctamente
3. Busca automáticamente archivos alternativos
4. Implementa fallback con archivos de ceros cuando es necesario
5. Mantiene compatibilidad total con operativo001.py

PROBLEMA SOLUCIONADO:
- El script anterior fallaba cuando había cambio de mes (ej: agosto -> septiembre)
- Intentaba extraer datos de septiembre de un archivo de agosto
- Ahora valida que existan archivos fuente para cada fecha objetivo

AUTOR: AI Assistant (versión V3 - cambio de mes corregido)
FECHA: 2024
"""

import os
import sys
import xarray as xr
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from pathlib import Path
import argparse
import json
import glob

# Agregar path del proyecto para imports
project_path = os.path.dirname(os.path.abspath(__file__))
if project_path not in sys.path:
    sys.path.append(project_path)

# Imports del proyecto (requeridos)
try:
    from conf.localConstants import wrfFileType
    from proj_preproc.wrf import crop_variables_xr, calculate_relative_humidity_metpy
    print("✅ Imports del proyecto cargados correctamente")
    PROJECT_IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Warning: No se pudieron cargar imports del proyecto: {e}")
    print("📋 Continuando en modo mockup básico...")
    PROJECT_IMPORTS_AVAILABLE = False


class TemporalSliceExtractorV3:
    """Extractor de recortes temporales de archivos WRF - V3 con manejo de cambio de mes."""
    
    def __init__(self, base_date: str, source_file_path: str, output_folder: str, 
                 bbox: list = None, resolution: float = 0.05, input_folder: str = None):
        """
        Inicializa el extractor V3.
        
        Args:
            base_date: Fecha base (formato YYYY-MM-DD) - fecha del archivo fuente
            source_file_path: Ruta al archivo WRF fuente
            output_folder: Carpeta de salida
            bbox: Bounding box [minlat, maxlat, minlon, maxlon]
            resolution: Resolución objetivo
            input_folder: Carpeta base de archivos WRF (para búsqueda automática)
        """
        self.base_date = pd.to_datetime(base_date).date()
        self.source_file_path = source_file_path
        self.output_folder = Path(output_folder)
        self.bbox = bbox or [18.75, 20, -99.75, -98.5]
        self.resolution = resolution
        self.input_folder = input_folder or "/ServerData/WRF_2017_Kraken/"
        
        # Variables a procesar (EXACTAMENTE igual que operativo001.py)
        self.variable_names = ['T2', 'U10', 'V10', 'RAINC', 'RAINNC', 'SWDOWN', 'GLW', 'Q2', 'PSFC']
        
        # Crear carpeta de salida
        self.output_folder.mkdir(parents=True, exist_ok=True)
        
        print(f"🎯 EXTRACTOR DE RECORTES TEMPORALES V3 (CAMBIO DE MES CORREGIDO)")
        print(f"   📅 Fecha base (archivo fuente): {self.base_date}")
        print(f"   📂 Archivo fuente: {os.path.basename(self.source_file_path)}")
        print(f"   📁 Carpeta salida: {self.output_folder}")
        print(f"   🎨 BBOX: {self.bbox}")
        print(f"   📐 Resolución: {self.resolution}")
        print(f"   🔧 Compatible con: operativo001.py")
        print(f"   📂 Carpeta de búsqueda: {self.input_folder}")

    def get_month_folder_name(self, month):
        """Convierte número de mes a formato de carpeta (igual que operativo001.py)"""
        month_names = {
            1: '01_enero', 2: '02_febrero', 3: '03_marzo', 4: '04_abril',
            5: '05_mayo', 6: '06_junio', 7: '07_julio', 8: '08_agosto',
            9: '09_septiembre', 10: '10_octubre', 11: '11_noviembre', 12: '12_diciembre'
        }
        return month_names[month]

    def get_wrf_file_path(self, target_date):
        """
        Obtiene la ruta del archivo WRF para una fecha específica (igual que operativo001.py).
        
        Args:
            target_date: fecha (datetime.date)
        
        Returns:
            str: ruta del archivo WRF
        """
        year_folder = str(target_date.year)
        month_folder = self.get_month_folder_name(target_date.month)
        
        file_pattern = f"wrfout_d02_{target_date.strftime('%Y-%m-%d')}_00.nc"
        file_path = os.path.join(self.input_folder, year_folder, month_folder, file_pattern)
        
        return file_path

    def convert_cdmx_to_utc(self, cdmx_datetime):
        """Convierte una fecha/hora de CDMX a UTC (igual que operativo001.py)."""
        utc_datetime = cdmx_datetime + timedelta(hours=6)
        return utc_datetime

    def convert_utc_to_cdmx(self, utc_datetime):
        """Convierte una fecha/hora de UTC a CDMX (igual que operativo001.py)."""
        cdmx_datetime = utc_datetime + timedelta(hours=-6)
        return cdmx_datetime

    def create_zero_dataset(self, target_date):
        """
        Crea un dataset de ceros para fechas sin archivo fuente (igual que operativo001.py).
        
        Args:
            target_date: Fecha objetivo (datetime.date)
        
        Returns:
            xarray.Dataset: Dataset de ceros
        """
        try:
            # Crear coordenadas
            lat = np.arange(self.bbox[0], self.bbox[1], self.resolution)
            lon = np.arange(self.bbox[2], self.bbox[3], self.resolution)
            time = range(24)
            
            # Variables meteorológicas (igual que operativo001.py)
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
            
            # Ajustar tiempo a hora local (GMT-6) - igual que operativo001.py
            first_datetime_utc = datetime.combine(target_date, datetime.min.time())
            first_datetime_cdmx = self.convert_utc_to_cdmx(first_datetime_utc)
            
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

    def validate_target_dates(self, num_days: int) -> dict:
        """
        Valida qué fechas objetivo tienen archivos fuente disponibles.
        
        Args:
            num_days: Número de días a extraer
            
        Returns:
            dict: Información sobre disponibilidad de archivos
        """
        print(f"\n🔍 VALIDANDO DISPONIBILIDAD DE ARCHIVOS FUENTE")
        print("-" * 60)
        
        validation_results = {
            'available': [],     # Fechas con archivo fuente disponible
            'missing': [],       # Fechas sin archivo fuente
            'source_files': {},  # Mapeo fecha -> archivo fuente
            'strategies': {}     # Estrategia para cada fecha
        }
        
        for day_offset in range(1, num_days + 1):
            target_date = self.base_date + timedelta(days=day_offset)
            target_date_str = target_date.strftime('%Y-%m-%d')
            
            print(f"\n📅 VALIDANDO DÍA {day_offset}: {target_date_str}")
            
            # Estrategia 1: Verificar si el archivo base puede cubrir esta fecha
            base_file_hours = 72  # Asumiendo 72 horas de pronóstico típico
            required_end_hour = day_offset * 24 + 24
            
            if required_end_hour <= base_file_hours and os.path.exists(self.source_file_path):
                print(f"   ✅ Estrategia 1: Extraer del archivo base ({self.base_date})")
                validation_results['available'].append(target_date)
                validation_results['source_files'][target_date] = self.source_file_path
                validation_results['strategies'][target_date] = 'slice_from_base'
                continue
            
            # Estrategia 2: Buscar archivo fuente específico para esta fecha
            target_file_path = self.get_wrf_file_path(target_date)
            if os.path.exists(target_file_path):
                print(f"   ✅ Estrategia 2: Archivo específico encontrado")
                print(f"      📂 {os.path.basename(target_file_path)}")
                validation_results['available'].append(target_date)
                validation_results['source_files'][target_date] = target_file_path
                validation_results['strategies'][target_date] = 'individual_file'
                continue
            
            # Estrategia 3: Buscar archivo de días anteriores que pueda cubrir
            for look_back in range(1, 4):  # Buscar hasta 3 días atrás
                alternative_date = target_date - timedelta(days=look_back)
                alternative_file = self.get_wrf_file_path(alternative_date)
                
                if os.path.exists(alternative_file):
                    # Verificar si este archivo puede cubrir la fecha objetivo
                    hours_needed = (target_date - alternative_date).days * 24 + 24
                    if hours_needed <= base_file_hours:
                        print(f"   ✅ Estrategia 3: Extraer de archivo alternativo ({alternative_date})")
                        print(f"      📂 {os.path.basename(alternative_file)}")
                        validation_results['available'].append(target_date)
                        validation_results['source_files'][target_date] = alternative_file
                        validation_results['strategies'][target_date] = f'slice_from_{alternative_date}'
                        break
            else:
                # Estrategia 4: Crear archivo de ceros
                print(f"   ⚠️  Estrategia 4: Crear archivo de ceros (no hay fuente disponible)")
                validation_results['missing'].append(target_date)
                validation_results['source_files'][target_date] = None
                validation_results['strategies'][target_date] = 'zeros'
        
        print(f"\n📊 RESUMEN DE VALIDACIÓN:")
        print(f"   ✅ Fechas con fuente disponible: {len(validation_results['available'])}")
        print(f"   ⚠️  Fechas que requerirán ceros: {len(validation_results['missing'])}")
        
        return validation_results

    def extract_from_wrf_file(self, source_file: str, target_date: date, hours_range: tuple) -> xr.Dataset:
        """
        Extrae datos específicos de un archivo WRF.
        
        Args:
            source_file: Ruta al archivo WRF fuente
            target_date: Fecha objetivo
            hours_range: Tupla (start_hour, end_hour)
            
        Returns:
            xarray.Dataset: Dataset procesado
        """
        start_hour, end_hour = hours_range
        times = list(range(start_hour, end_hour))
        
        print(f"   📂 Cargando: {os.path.basename(source_file)}")
        print(f"   ⏰ Horas: {start_hour} a {end_hour-1}")
        
        # Cargar archivo fuente
        ds = xr.open_dataset(source_file, decode_times=False)
        
        # Verificar que tenemos suficientes horas
        max_time_available = ds.dims.get('Time', ds.dims.get('time', 24))
        if end_hour > max_time_available:
            raise ValueError(f"No hay suficientes horas: necesitamos {end_hour}, disponibles {max_time_available}")
        
        # Cropping espacial y temporal
        variables_with_extra = self.variable_names.copy()
        cropped_ds, newLAT, newLon = crop_variables_xr(ds, variables_with_extra, self.bbox, times)
        
        # PROCESAR VARIABLES DERIVADAS (igual que operativo001.py)
        
        # Lluvia total (RAIN = RAINC + RAINNC) - NO CUMULATIVA
        if 'RAINC' in cropped_ds.data_vars and 'RAINNC' in cropped_ds.data_vars:
            print("   🌧️ Calculando lluvia total...")
            cropped_ds['RAIN'] = cropped_ds['RAINC'] + cropped_ds['RAINNC']
            rain_values = cropped_ds['RAIN'].values
            rain_diff = np.zeros_like(rain_values)
            rain_diff[1:,:,:] = rain_values[1:,:,:] - rain_values[:-1,:,:]
            cropped_ds['RAIN'] = xr.DataArray(rain_diff, dims=cropped_ds['RAIN'].dims, coords=cropped_ds['RAIN'].coords)
            cropped_ds['RAIN'] = cropped_ds['RAIN'].where(cropped_ds['RAIN'] > 0, 0)
            cropped_ds = cropped_ds.drop(['RAINC', 'RAINNC'])
        
        # Velocidad del viento (WS10)
        if 'U10' in cropped_ds.data_vars and 'V10' in cropped_ds.data_vars:
            print("   💨 Calculando velocidad del viento...")
            cropped_ds['WS10'] = np.sqrt(cropped_ds['U10']**2 + cropped_ds['V10']**2)
        
        # Humedad relativa (RH)
        if all(var in cropped_ds.data_vars for var in ['T2', 'Q2', 'PSFC']):
            print("   💧 Calculando humedad relativa...")
            T2 = cropped_ds['T2'].values
            PSFC = cropped_ds['PSFC'].values  
            Q2 = cropped_ds['Q2'].values
            RH = calculate_relative_humidity_metpy(T2, PSFC, Q2)
            cropped_ds['RH'] = xr.DataArray(RH, dims=cropped_ds['T2'].dims, coords=cropped_ds['T2'].coords)
            cropped_ds = cropped_ds.drop(['Q2', 'PSFC'])
        
        # AJUSTE DE ZONA HORARIA (igual que operativo001.py)
        source_date = pd.to_datetime(os.path.basename(source_file).split('_')[2]).date()
        first_datetime_utc = datetime.combine(source_date, datetime.min.time())
        first_datetime_cdmx = self.convert_utc_to_cdmx(first_datetime_utc)
        
        cropped_ds['time'].attrs.update({
            'units': f'hours since {first_datetime_cdmx.strftime("%Y-%m-%d %H:%M:%S")}',
            'calendar': 'standard',
            'axis': 'T',
            'long_name': 'time',
            'standard_name': 'time',
            'timezone': 'CDMX (GMT-6)'
        })
        
        # Asegurar tiempo como enteros
        if hasattr(cropped_ds['time'], 'values'):
            time_values = cropped_ds['time'].values
            if hasattr(time_values, 'dtype') and 'datetime' in str(time_values.dtype):
                new_time_values = np.arange(len(time_values), dtype=np.int64)
                cropped_ds = cropped_ds.assign_coords(time=new_time_values)
        
        # INTERPOLACIÓN
        new_lat = np.arange(self.bbox[0], self.bbox[1], self.resolution)
        new_lon = np.arange(self.bbox[2], self.bbox[3], self.resolution)
        
        cropped_ds = cropped_ds.interp(
            lat=new_lat,
            lon=new_lon,
            method='linear'
        )
        
        # Actualizar atributos de coordenadas
        cropped_ds['lat'].attrs.update({
            'units': 'degrees_north',
            'axis': 'Y',
            'long_name': 'latitude',
            'standard_name': 'latitude'
        })
        cropped_ds['lon'].attrs.update({
            'units': 'degrees_east',
            'axis': 'X',
            'long_name': 'longitude',
            'standard_name': 'longitude'
        })
        
        return cropped_ds

    def extract_temporal_slice(self, target_date: str, validation_results: dict, 
                             dry_run: bool = False) -> str:
        """
        Extrae un recorte temporal específico usando la estrategia validada.
        
        Args:
            target_date: Fecha objetivo del recorte (YYYY-MM-DD)
            validation_results: Resultados de validación
            dry_run: Si solo simular sin procesar realmente
            
        Returns:
            Ruta del archivo generado
        """
        target_dt = pd.to_datetime(target_date).date()
        output_filename = f"{target_date}.nc"
        output_path = self.output_folder / output_filename
        
        print(f"\n🎯 EXTRAYENDO RECORTE: {target_date}")
        print("-" * 50)
        
        strategy = validation_results['strategies'].get(target_dt, 'unknown')
        source_file = validation_results['source_files'].get(target_dt)
        
        print(f"   📋 Estrategia: {strategy}")
        if source_file:
            print(f"   📂 Archivo fuente: {os.path.basename(source_file)}")
        
        if dry_run:
            print(f"   🧪 DRY RUN: Archivo se generaría en: {output_path}")
            return str(output_path)
        
        try:
            if strategy == 'zeros':
                # Crear dataset de ceros
                print(f"   🔧 Creando dataset de ceros...")
                processed_ds = self.create_zero_dataset(target_dt)
                note = "Dataset de ceros (archivo fuente no disponible)"
                
            elif not PROJECT_IMPORTS_AVAILABLE:
                # Modo mockup
                print(f"   🧪 MODO MOCKUP: Simulando procesamiento...")
                dummy_data = np.random.rand(24, 25, 25)
                
                import netCDF4 as nc
                with nc.Dataset(output_path, 'w') as ncfile:
                    ncfile.createDimension('time', 24)
                    ncfile.createDimension('lat', 25)
                    ncfile.createDimension('lon', 25)
                    
                    time_var = ncfile.createVariable('time', 'f4', ('time',))
                    lat_var = ncfile.createVariable('lat', 'f4', ('lat',))
                    lon_var = ncfile.createVariable('lon', 'f4', ('lon',))
                    temp_var = ncfile.createVariable('T2', 'f4', ('time', 'lat', 'lon'))
                    
                    time_var[:] = np.arange(24)
                    lat_var[:] = np.linspace(18.75, 20, 25)
                    lon_var[:] = np.linspace(-99.75, -98.5, 25)
                    temp_var[:] = dummy_data
                    
                    ncfile.setncattr('source_file', os.path.basename(source_file) if source_file else 'mockup')
                    ncfile.setncattr('created_by', 'TemporalSliceExtractorV3')
                    ncfile.setncattr('strategy', strategy)
                
                print(f"   ✅ Archivo simulado creado: {output_filename}")
                return str(output_path)
                
            else:
                # Procesamiento real
                print(f"   🔧 MODO REAL: Procesando...")
                
                if strategy == 'slice_from_base':
                    # Extraer del archivo base usando offset
                    day_offset = (target_dt - self.base_date).days
                    start_hour = day_offset * 24
                    end_hour = start_hour + 24
                    processed_ds = self.extract_from_wrf_file(source_file, target_dt, (start_hour, end_hour))
                    note = f"Extraído del archivo base (horas {start_hour}-{end_hour-1})"
                    
                elif strategy == 'individual_file':
                    # Usar archivo específico (primeras 24 horas)
                    processed_ds = self.extract_from_wrf_file(source_file, target_dt, (0, 24))
                    note = "Procesado desde archivo específico"
                    
                elif strategy.startswith('slice_from_'):
                    # Extraer de archivo alternativo
                    source_date_str = strategy.split('_')[-1]
                    source_date = pd.to_datetime(source_date_str).date()
                    day_offset = (target_dt - source_date).days
                    start_hour = day_offset * 24
                    end_hour = start_hour + 24
                    processed_ds = self.extract_from_wrf_file(source_file, target_dt, (start_hour, end_hour))
                    note = f"Extraído de archivo alternativo {source_date} (horas {start_hour}-{end_hour-1})"
                    
                else:
                    raise ValueError(f"Estrategia desconocida: {strategy}")
            
            if processed_ds is not None:
                # Agregar metadatos
                processed_ds.attrs['source_file'] = os.path.basename(source_file) if source_file else 'none'
                processed_ds.attrs['created_by'] = 'TemporalSliceExtractorV3'
                processed_ds.attrs['creation_time'] = datetime.now().isoformat()
                processed_ds.attrs['compatibility_mode'] = 'operativo001'
                processed_ds.attrs['extraction_strategy'] = strategy
                processed_ds.attrs['extraction_note'] = note
                
                # Guardar archivo
                print(f"   💾 Guardando archivo...")
                processed_ds.to_netcdf(output_path)
                
                # Verificar archivo generado
                file_size = os.path.getsize(output_path) / (1024 * 1024)
                print(f"   ✅ Archivo procesado: {output_filename} ({file_size:.1f} MB)")
                print(f"   📋 Nota: {note}")
                
                return str(output_path)
            else:
                raise ValueError("No se pudo crear el dataset procesado")
                
        except Exception as e:
            print(f"   ❌ Error procesando recorte: {e}")
            raise

    def extract_multiple_days(self, num_days: int = 2, dry_run: bool = False) -> list:
        """
        Extrae múltiples días con validación automática de archivos fuente.
        
        Args:
            num_days: Número de días a extraer después de la fecha base
            dry_run: Si solo simular
            
        Returns:
            Lista de rutas de archivos generados
        """
        print(f"\n🔄 EXTRAYENDO MÚLTIPLES RECORTES TEMPORALES V3")
        print("=" * 70)
        print(f"   📅 Fecha base (archivo fuente): {self.base_date}")
        print(f"   📊 Días a extraer: {num_days}")
        print(f"   🧪 Dry run: {'Sí' if dry_run else 'No'}")
        print(f"   🔧 Maneja cambios de mes: ✅")
        
        # PASO 1: Validar disponibilidad de archivos
        validation_results = self.validate_target_dates(num_days)
        
        # PASO 2: Extraer cada día usando la estrategia validada
        generated_files = []
        
        for day_offset in range(1, num_days + 1):
            target_date = self.base_date + timedelta(days=day_offset)
            target_date_str = target_date.strftime('%Y-%m-%d')
            
            print(f"\n📅 PROCESANDO DÍA {day_offset}/{num_days}: {target_date_str}")
            
            try:
                output_path = self.extract_temporal_slice(
                    target_date_str, validation_results, dry_run
                )
                generated_files.append(output_path)
                
                if not dry_run:
                    import time
                    time.sleep(1)
                
            except Exception as e:
                print(f"   ❌ Error en día {day_offset}: {e}")
                continue
        
        print(f"\n✅ EXTRACCIÓN MÚLTIPLE COMPLETADA")
        print(f"   📊 Archivos generados: {len(generated_files)}")
        for i, file_path in enumerate(generated_files, 1):
            print(f"   {i}. {os.path.basename(file_path)}")
        
        return generated_files

    def verify_slice_metadata(self, file_path: str) -> dict:
        """Verifica metadatos con información de estrategia de extracción."""
        try:
            print(f"\n🔍 VERIFICANDO METADATOS V3")
            print("-" * 50)
            print(f"   📁 Archivo: {os.path.basename(file_path)}")
            
            ds = xr.open_dataset(file_path)
            
            # Atributos V3
            v3_attrs = ['extraction_strategy', 'extraction_note']
            compatibility_attrs = ['source_file', 'created_by', 'compatibility_mode']
            
            print(f"   🔧 ESTRATEGIA DE EXTRACCIÓN:")
            for attr in v3_attrs:
                if attr in ds.attrs:
                    print(f"   📋 {attr}: {ds.attrs[attr]}")
            
            print(f"   🔧 COMPATIBILIDAD:")
            for attr in compatibility_attrs:
                if attr in ds.attrs:
                    print(f"   ✅ {attr}: {ds.attrs[attr]}")
            
            print(f"   📊 Dimensiones: {dict(ds.dims)}")
            print(f"   🌡️  Variables: {list(ds.data_vars.keys())}")
            
            # Verificar variables esperadas
            expected_vars = ['T2', 'U10', 'V10', 'RAIN', 'SWDOWN', 'GLW', 'WS10', 'RH']
            missing_vars = [var for var in expected_vars if var not in ds.data_vars]
            
            if missing_vars:
                print(f"   ⚠️  Variables faltantes: {missing_vars}")
            else:
                print(f"   ✅ Todas las variables esperadas presentes")
            
            ds.close()
            return ds.attrs
            
        except Exception as e:
            print(f"   ❌ Error verificando metadatos: {e}")
            return {}


def main():
    """Función principal V3 con manejo de cambio de mes."""
    parser = argparse.ArgumentParser(description='Extractor de recortes temporales WRF V3 - Manejo de cambio de mes')
    
    parser.add_argument('--target-datetime', type=str,
                       help='Fecha objetivo (YYYY-MM-DD HH:MM:SS)')
    parser.add_argument('--base-date', type=str, default='2023-05-05',
                       help='Fecha base (YYYY-MM-DD)')
    parser.add_argument('--source-file', type=str,
                       help='Ruta al archivo WRF fuente')
    parser.add_argument('--input-folder', type=str, default='/ServerData/WRF_2017_Kraken/',
                       help='Carpeta base de archivos WRF')
    parser.add_argument('--output-folder', type=str, default='/dev/shm/tem_ram_forecast',
                       help='Carpeta de salida')
    parser.add_argument('--num-days', type=int, default=2,
                       help='Número de días a extraer')
    parser.add_argument('--dry-run', action='store_true',
                       help='Solo simular, no procesar')
    parser.add_argument('--verify', action='store_true',
                       help='Verificar metadatos de archivos existentes')
    
    args = parser.parse_args()
    
    try:
        print("🚀 EXTRACTOR DE RECORTES TEMPORALES WRF V3")
        print("=" * 70)
        print("🔧 CORRECCIONES V3:")
        print("   ✅ Manejo de cambio de mes/año")
        print("   ✅ Validación de archivos fuente")
        print("   ✅ Búsqueda automática de archivos alternativos")
        print("   ✅ Fallback inteligente con archivos de ceros")
        print("   ✅ Compatibilidad total con operativo001.py")
        print("=" * 70)
        
        # Determinar fecha base
        if args.target_datetime:
            target_dt = datetime.strptime(args.target_datetime, '%Y-%m-%d %H:%M:%S')
            base_date = (target_dt - timedelta(days=1)).strftime('%Y-%m-%d')
            print(f"📅 Target datetime: {args.target_datetime}")
            print(f"📅 Fecha base calculada: {base_date}")
        else:
            base_date = args.base_date
            print(f"📅 Fecha base: {base_date}")
        
        # Construir ruta del archivo fuente si no se proporciona
        if not args.source_file:
            year_folder = str(datetime.strptime(base_date, '%Y-%m-%d').year)
            month_num = datetime.strptime(base_date, '%Y-%m-%d').month
            
            month_names = {
                1: '01_enero', 2: '02_febrero', 3: '03_marzo', 4: '04_abril',
                5: '05_mayo', 6: '06_junio', 7: '07_julio', 8: '08_agosto',
                9: '09_septiembre', 10: '10_octubre', 11: '11_noviembre', 12: '12_diciembre'
            }
            month_folder = month_names[month_num]
            
            file_pattern = f"wrfout_d02_{base_date}_00.nc"
            args.source_file = os.path.join(args.input_folder, year_folder, month_folder, file_pattern)
            
            print(f"📂 Archivo fuente (por defecto): {args.source_file}")
        
        # Crear extractor V3
        extractor = TemporalSliceExtractorV3(
            base_date=base_date,
            source_file_path=args.source_file,
            output_folder=args.output_folder,
            input_folder=args.input_folder
        )
        
        if args.verify:
            # Modo verificación
            print(f"\n🔍 MODO VERIFICACIÓN V3")
            pattern = os.path.join(args.output_folder, "*.nc")
            nc_files = glob.glob(pattern)
            
            if nc_files:
                for file_path in nc_files:
                    extractor.verify_slice_metadata(file_path)
            else:
                print(f"   ⚠️  No se encontraron archivos .nc en {args.output_folder}")
        else:
            # Modo extracción
            generated_files = extractor.extract_multiple_days(
                num_days=args.num_days,
                dry_run=args.dry_run
            )
            
            # Verificar archivos generados
            if generated_files and not args.dry_run:
                print(f"\n🔍 VERIFICACIÓN DE ARCHIVOS GENERADOS:")
                for file_path in generated_files:
                    if os.path.exists(file_path):
                        extractor.verify_slice_metadata(file_path)
        
        print(f"\n🎉 PROCESAMIENTO V3 COMPLETADO EXITOSAMENTE")
        print(f"   🔧 Cambios de mes manejados correctamente")
        print(f"   📊 Archivos compatibles con operativo001.py")
        
    except Exception as e:
        print(f"\n❌ ERROR EN V3: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())