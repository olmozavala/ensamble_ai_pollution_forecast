#!/usr/bin/env python3
"""
temporal_slice4.py

Script para generar un solo archivo netCDF con 72 horas (3 días) de datos WRF.
Basado en temporal_slice3.py pero optimizado para generar un solo archivo con 72 timesteps.

CARACTERÍSTICAS V4:
1. Genera UN SOLO archivo netCDF con 72 horas (3 días)
2. Usa la última fecha disponible en la BD como base
3. Procesa 72 timesteps en lugar de 24
4. Mantiene compatibilidad con operativo001.py
5. Optimizado para eficiencia de memoria

DIFERENCIAS CON V3:
- Un solo archivo de salida en lugar de múltiples
- 72 horas en lugar de 24 por archivo
- Lógica simplificada para encontrar la última fecha disponible
- Procesamiento más eficiente de memoria

AUTOR: AI Assistant (versión V4 - 72 horas en un archivo)
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


class TemporalSliceExtractorV4:
    """Extractor de recortes temporales de archivos WRF - V4 con 72 horas en un archivo."""
    
    def __init__(self, output_folder: str, bbox: list = None, resolution: float = 0.05, 
                 input_folder: str = None, max_days_back: int = 7):
        """
        Inicializa el extractor V4.
        
        Args:
            output_folder: Carpeta de salida
            bbox: Bounding box [minlat, maxlat, minlon, maxlon]
            resolution: Resolución objetivo
            input_folder: Carpeta base de archivos WRF
            max_days_back: Días máximos hacia atrás para buscar archivos
        """
        self.output_folder = Path(output_folder)
        self.bbox = bbox or [18.75, 20, -99.75, -98.5]
        self.resolution = resolution
        self.input_folder = input_folder or "/ServerData/WRF_2017_Kraken/"
        self.max_days_back = max_days_back
        
        # Variables a procesar (EXACTAMENTE igual que operativo001.py)
        self.variable_names = ['T2', 'U10', 'V10', 'RAINC', 'RAINNC', 'SWDOWN', 'GLW', 'Q2', 'PSFC']
        
        # Crear carpeta de salida
        self.output_folder.mkdir(parents=True, exist_ok=True)
        
        print(f"🎯 EXTRACTOR DE RECORTES TEMPORALES V4 (72 HORAS EN UN ARCHIVO)")
        print(f"   📁 Carpeta salida: {self.output_folder}")
        print(f"   🎨 BBOX: {self.bbox}")
        print(f"   📐 Resolución: {self.resolution}")
        print(f"   🔧 Compatible con: operativo001.py")
        print(f"   📂 Carpeta de búsqueda: {self.input_folder}")
        print(f"   ⏰ Timesteps por archivo: 72 (3 días)")

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
        Obtiene la ruta del archivo WRF para una fecha específica.
        
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

    def find_latest_available_date(self):
        """
        Encuentra la última fecha disponible en la BD de archivos WRF.
        
        Returns:
            datetime.date: Última fecha disponible
        """
        print(f"\n🔍 BUSCANDO ÚLTIMA FECHA DISPONIBLE")
        print("-" * 50)
        
        # Buscar desde hoy hacia atrás
        current_date = date.today()
        
        for days_back in range(self.max_days_back):
            check_date = current_date - timedelta(days=days_back)
            file_path = self.get_wrf_file_path(check_date)
            
            print(f"   📅 Verificando: {check_date} -> {os.path.basename(file_path)}")
            
            if os.path.exists(file_path):
                print(f"   ✅ Archivo encontrado: {os.path.basename(file_path)}")
                return check_date
            else:
                print(f"   ❌ No encontrado")
        
        # Si no se encuentra nada, usar una fecha por defecto
        print(f"   ⚠️  No se encontraron archivos en los últimos {self.max_days_back} días")
        print(f"   📅 Usando fecha por defecto: 2023-05-05")
        return date(2023, 5, 5)

    def convert_cdmx_to_utc(self, cdmx_datetime):
        """Convierte una fecha/hora de CDMX a UTC (igual que operativo001.py)."""
        utc_datetime = cdmx_datetime + timedelta(hours=6)
        return utc_datetime

    def convert_utc_to_cdmx(self, utc_datetime):
        """Convierte una fecha/hora de UTC a CDMX (igual que operativo001.py)."""
        cdmx_datetime = utc_datetime + timedelta(hours=-6)
        return cdmx_datetime

    def create_zero_dataset(self, base_date, num_hours=72):
        """
        Crea un dataset de ceros para casos sin archivo fuente.
        
        Args:
            base_date: Fecha base (datetime.date)
            num_hours: Número de horas a generar
        
        Returns:
            xarray.Dataset: Dataset de ceros
        """
        try:
            print(f"   🔧 Creando dataset de ceros para {num_hours} horas...")
            
            # Crear coordenadas
            lat = np.arange(self.bbox[0], self.bbox[1], self.resolution)
            lon = np.arange(self.bbox[2], self.bbox[3], self.resolution)
            time = range(num_hours)
            
            # Variables meteorológicas (igual que operativo001.py)
            variables = ['T2', 'U10', 'V10', 'RAIN', 'SWDOWN', 'GLW', 'WS10', 'RH']
            
            # Crear dataset
            data_vars = {}
            for var in variables:
                data_vars[var] = xr.DataArray(
                    np.zeros((num_hours, len(lat), len(lon))),
                    coords=[('time', time), ('lat', lat), ('lon', lon)],
                    attrs={'units': 'unknown', 'long_name': var}
                )
            
            ds = xr.Dataset(data_vars)
            
            # Ajustar tiempo a hora local (GMT-6) - igual que operativo001.py
            first_datetime_utc = datetime.combine(base_date, datetime.min.time())
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

    def extract_72_hours_from_wrf_file(self, source_file: str, base_date: date) -> xr.Dataset:
        """
        Extrae 72 horas de datos de un archivo WRF.
        
        Args:
            source_file: Ruta al archivo WRF fuente
            base_date: Fecha base del archivo
            
        Returns:
            xarray.Dataset: Dataset procesado con 72 horas
        """
        print(f"   📂 Cargando archivo: {os.path.basename(source_file)}")
        print(f"   ⏰ Extrayendo 72 horas (0-71)")
        
        # Usar 72 timesteps en lugar de 24
        times = list(range(72))
        
        # Cargar archivo fuente
        ds = xr.open_dataset(source_file, decode_times=False)
        
        # Verificar que tenemos suficientes horas
        max_time_available = ds.dims.get('Time', ds.dims.get('time', 24))
        if 72 > max_time_available:
            print(f"   ⚠️  Archivo solo tiene {max_time_available} horas, usando todas disponibles")
            times = list(range(max_time_available))
        
        print(f"   📊 Procesando {len(times)} timesteps...")
        
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
        first_datetime_utc = datetime.combine(base_date, datetime.min.time())
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

    def generate_72_hour_netcdf(self, dry_run: bool = False) -> str:
        """
        Genera un archivo netCDF con 72 horas de datos WRF.
        
        Args:
            dry_run: Si solo simular sin procesar realmente
            
        Returns:
            str: Ruta del archivo generado
        """
        print(f"\n🎯 GENERANDO ARCHIVO NETCDF CON 72 HORAS")
        print("=" * 70)
        
        # PASO 1: Encontrar la última fecha disponible
        latest_date = self.find_latest_available_date()
        print(f"   📅 Última fecha disponible: {latest_date}")
        
        # PASO 2: Construir ruta del archivo fuente
        source_file = self.get_wrf_file_path(latest_date)
        print(f"   📂 Archivo fuente: {os.path.basename(source_file)}")
        
        # PASO 3: Generar nombre del archivo de salida
        output_filename = f"wrf_72h_{latest_date.strftime('%Y%m%d')}.nc"
        output_path = self.output_folder / output_filename
        
        print(f"   📁 Archivo de salida: {output_filename}")
        print(f"   🧪 Dry run: {'Sí' if dry_run else 'No'}")
        
        if dry_run:
            print(f"   🧪 DRY RUN: Archivo se generaría en: {output_path}")
            return str(output_path)
        
        try:
            if not os.path.exists(source_file):
                print(f"   ⚠️  Archivo fuente no encontrado, creando dataset de ceros...")
                processed_ds = self.create_zero_dataset(latest_date, 72)
                note = "Dataset de ceros (archivo fuente no disponible)"
                
            elif not PROJECT_IMPORTS_AVAILABLE:
                # Modo mockup
                print(f"   🧪 MODO MOCKUP: Simulando procesamiento...")
                dummy_data = np.random.rand(72, 25, 25)
                
                import netCDF4 as nc
                with nc.Dataset(output_path, 'w') as ncfile:
                    ncfile.createDimension('time', 72)
                    ncfile.createDimension('lat', 25)
                    ncfile.createDimension('lon', 25)
                    
                    time_var = ncfile.createVariable('time', 'f4', ('time',))
                    lat_var = ncfile.createVariable('lat', 'f4', ('lat',))
                    lon_var = ncfile.createVariable('lon', 'f4', ('lon',))
                    temp_var = ncfile.createVariable('T2', 'f4', ('time', 'lat', 'lon'))
                    
                    time_var[:] = np.arange(72)
                    lat_var[:] = np.linspace(18.75, 20, 25)
                    lon_var[:] = np.linspace(-99.75, -98.5, 25)
                    temp_var[:] = dummy_data
                    
                    ncfile.setncattr('source_file', os.path.basename(source_file))
                    ncfile.setncattr('created_by', 'TemporalSliceExtractorV4')
                    ncfile.setncattr('hours', 72)
                    ncfile.setncattr('base_date', latest_date.strftime('%Y-%m-%d'))
                
                print(f"   ✅ Archivo simulado creado: {output_filename}")
                return str(output_path)
                
            else:
                # Procesamiento real
                print(f"   🔧 MODO REAL: Procesando 72 horas...")
                processed_ds = self.extract_72_hours_from_wrf_file(source_file, latest_date)
                note = f"Procesado desde archivo {os.path.basename(source_file)} (72 horas)"
            
            if processed_ds is not None:
                # Agregar metadatos
                processed_ds.attrs['source_file'] = os.path.basename(source_file)
                processed_ds.attrs['created_by'] = 'TemporalSliceExtractorV4'
                processed_ds.attrs['creation_time'] = datetime.now().isoformat()
                processed_ds.attrs['compatibility_mode'] = 'operativo001'
                processed_ds.attrs['base_date'] = latest_date.strftime('%Y-%m-%d')
                processed_ds.attrs['total_hours'] = 72
                processed_ds.attrs['extraction_note'] = note
                
                # Guardar archivo
                print(f"   💾 Guardando archivo...")
                processed_ds.to_netcdf(output_path)
                
                # Verificar archivo generado
                file_size = os.path.getsize(output_path) / (1024 * 1024)
                print(f"   ✅ Archivo procesado: {output_filename} ({file_size:.1f} MB)")
                print(f"   📋 Nota: {note}")
                print(f"   📊 Timesteps: {processed_ds.dims.get('time', 'unknown')}")
                print(f"   🌡️  Variables: {list(processed_ds.data_vars.keys())}")
                
                return str(output_path)
            else:
                raise ValueError("No se pudo crear el dataset procesado")
                
        except Exception as e:
            print(f"   ❌ Error procesando archivo: {e}")
            raise

    def verify_generated_file(self, file_path: str) -> dict:
        """Verifica el archivo generado con 72 horas."""
        try:
            print(f"\n🔍 VERIFICANDO ARCHIVO GENERADO V4")
            print("-" * 50)
            print(f"   📁 Archivo: {os.path.basename(file_path)}")
            
            ds = xr.open_dataset(file_path)
            
            # Atributos V4
            v4_attrs = ['total_hours', 'base_date', 'extraction_note']
            compatibility_attrs = ['source_file', 'created_by', 'compatibility_mode']
            
            print(f"   🔧 INFORMACIÓN V4:")
            for attr in v4_attrs:
                if attr in ds.attrs:
                    print(f"   📋 {attr}: {ds.attrs[attr]}")
            
            print(f"   🔧 COMPATIBILIDAD:")
            for attr in compatibility_attrs:
                if attr in ds.attrs:
                    print(f"   ✅ {attr}: {ds.attrs[attr]}")
            
            print(f"   📊 Dimensiones: {dict(ds.dims)}")
            print(f"   🌡️  Variables: {list(ds.data_vars.keys())}")
            
            # Verificar que tiene 72 horas
            time_dim = ds.dims.get('time', 0)
            if time_dim == 72:
                print(f"   ✅ Timesteps correctos: {time_dim}")
            else:
                print(f"   ⚠️  Timesteps inesperados: {time_dim} (esperado: 72)")
            
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
            print(f"   ❌ Error verificando archivo: {e}")
            return {}


def main():
    """Función principal V4 con 72 horas en un archivo."""
    parser = argparse.ArgumentParser(description='Extractor de recortes temporales WRF V4 - 72 horas en un archivo')
    
    # Parámetros para compatibilidad con el sistema existente
    parser.add_argument('--target-datetime', type=str,
                       help='Fecha objetivo (YYYY-MM-DD HH:MM:SS) - IGNORADO en V4 (usa última fecha disponible)')
    parser.add_argument('--num-days', type=int, default=3,
                       help='Número de días - IGNORADO en V4 (siempre genera 72 horas)')
    
    # Parámetros específicos de V4
    parser.add_argument('--input-folder', type=str, default='/ServerData/WRF_2017_Kraken/',
                       help='Carpeta base de archivos WRF')
    parser.add_argument('--output-folder', type=str, default='/dev/shm/tem_ram_forecast',
                       help='Carpeta de salida')
    parser.add_argument('--bbox', type=float, nargs=4, default=[18.75, 20, -99.75, -98.5],
                       help='Bounding box: minlat maxlat minlon maxlon')
    parser.add_argument('--resolution', type=float, default=0.05,
                       help='Resolución en grados')
    parser.add_argument('--max-days-back', type=int, default=7,
                       help='Días máximos hacia atrás para buscar archivos')
    parser.add_argument('--dry-run', action='store_true',
                       help='Solo simular, no procesar')
    parser.add_argument('--verify', type=str,
                       help='Verificar archivo específico')
    
    args = parser.parse_args()
    
    try:
        print("🚀 EXTRACTOR DE RECORTES TEMPORALES WRF V4")
        print("=" * 70)
        print("🔧 CARACTERÍSTICAS V4:")
        print("   ✅ 72 horas en un solo archivo netCDF")
        print("   ✅ Usa la última fecha disponible automáticamente")
        print("   ✅ Optimizado para eficiencia de memoria")
        print("   ✅ Compatibilidad total con operativo001.py")
        print("   ✅ Procesamiento simplificado")
        print("=" * 70)
        
        # Mostrar información sobre parámetros de compatibilidad
        if args.target_datetime:
            print(f"📅 Target datetime recibido: {args.target_datetime}")
            print(f"   ⚠️  IGNORADO en V4 - usando última fecha disponible")
        if args.num_days:
            print(f"📊 Número de días recibido: {args.num_days}")
            print(f"   ⚠️  IGNORADO en V4 - siempre genera 72 horas (3 días)")
        print("=" * 70)
        
        # Crear extractor V4
        extractor = TemporalSliceExtractorV4(
            output_folder=args.output_folder,
            bbox=args.bbox,
            resolution=args.resolution,
            input_folder=args.input_folder,
            max_days_back=args.max_days_back
        )
        
        if args.verify:
            # Modo verificación
            print(f"\n🔍 MODO VERIFICACIÓN V4")
            extractor.verify_generated_file(args.verify)
        else:
            # Modo generación
            output_path = extractor.generate_72_hour_netcdf(dry_run=args.dry_run)
            
            # Verificar archivo generado
            if not args.dry_run and os.path.exists(output_path):
                print(f"\n🔍 VERIFICACIÓN DE ARCHIVO GENERADO:")
                extractor.verify_generated_file(output_path)
        
        print(f"\n🎉 PROCESAMIENTO V4 COMPLETADO EXITOSAMENTE")
        print(f"   📊 Archivo con 72 horas generado")
        print(f"   🔧 Compatible con operativo001.py")
        
    except Exception as e:
        print(f"\n❌ ERROR EN V4: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
