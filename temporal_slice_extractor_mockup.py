#!/usr/bin/env python3
"""
temporal_slice_extractor_mockup.py

Script standalone para demostrar la extracción inteligente de recortes temporales
de archivos WRF. Permite generar archivos de días futuros usando recortes de un
archivo base que contiene múltiples días de pronóstico.

EJEMPLO DE USO:
- Tenemos archivo del 5 de mayo de 2023 (72 horas de pronóstico)
- Queremos generar archivos del 6 y 7 de mayo
- Extraemos horas 24-47 para el día 6, horas 48-71 para el día 7

AUTOR: AI Assistant
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

# Agregar path del proyecto para imports
project_path = os.path.dirname(os.path.abspath(__file__))
if project_path not in sys.path:
    sys.path.append(project_path)

# Imports del proyecto (requeridos)
try:
    from conf.localConstants import wrfFileType
    from proj_preproc.wrf import crop_variables_xr, calculate_relative_humidity_metpy
    print("✅ Imports del proyecto cargados correctamente")
except ImportError as e:
    print(f"⚠️  Warning: No se pudieron cargar imports del proyecto: {e}")
    print("📋 Continuando en modo mockup básico...")


class TemporalSliceExtractor:
    """Extractor de recortes temporales de archivos WRF."""
    
    def __init__(self, base_date: str, source_file_path: str, output_folder: str, 
                 bbox: list = None, resolution: float = 0.05):
        """
        Inicializa el extractor.
        
        Args:
            base_date: Fecha base (formato YYYY-MM-DD)
            source_file_path: Ruta al archivo WRF fuente
            output_folder: Carpeta de salida
            bbox: Bounding box [minlat, maxlat, minlon, maxlon]
            resolution: Resolución objetivo
        """
        self.base_date = pd.to_datetime(base_date).date()
        self.source_file_path = source_file_path
        self.output_folder = Path(output_folder)
        self.bbox = bbox or [18.75, 20, -99.75, -98.5]
        self.resolution = resolution
        
        # Variables a procesar (mismas que el procesamiento normal)
        self.variable_names = ['T2', 'U10', 'V10', 'RAINC', 'RAINNC', 'SWDOWN', 'GLW']
        
        # Crear carpeta de salida
        self.output_folder.mkdir(parents=True, exist_ok=True)
        
        print(f"🎯 EXTRACTOR DE RECORTES TEMPORALES INICIALIZADO")
        print(f"   📅 Fecha base: {self.base_date}")
        print(f"   📂 Archivo fuente: {os.path.basename(self.source_file_path)}")
        print(f"   📁 Carpeta salida: {self.output_folder}")
        print(f"   🎨 BBOX: {self.bbox}")
        print(f"   📐 Resolución: {self.resolution}")

    def extract_temporal_slice(self, target_date: str, day_offset: int, 
                             add_label: bool = True, dry_run: bool = False) -> str:
        """
        Extrae un recorte temporal específico del archivo fuente.
        
        Args:
            target_date: Fecha objetivo del recorte (YYYY-MM-DD)
            day_offset: Offset en días desde la fecha base (1=día siguiente, 2=pasado mañana)
            add_label: Si agregar label identificativo al archivo
            dry_run: Si solo simular sin procesar realmente
            
        Returns:
            Ruta del archivo generado
        """
        target_dt = pd.to_datetime(target_date).date()
        
        # Calcular rango de horas a extraer
        start_hour = day_offset * 24
        end_hour = start_hour + 24
        
        print(f"\n🎯 EXTRAYENDO RECORTE TEMPORAL")
        print("-" * 50)
        print(f"   📅 Fecha objetivo: {target_date}")
        print(f"   📊 Offset de días: +{day_offset}")
        print(f"   ⏰ Rango de horas: {start_hour} a {end_hour-1}")
        print(f"   🏷️  Con label: {'Sí' if add_label else 'No'}")
        print(f"   🧪 Dry run: {'Sí' if dry_run else 'No'}")
        
        # Nombre del archivo de salida
        if add_label:
            output_filename = f"{target_date}_SLICE_DAY{day_offset:+d}.nc"
        else:
            output_filename = f"{target_date}.nc"
            
        output_path = self.output_folder / output_filename
        
        if dry_run:
            print(f"   🧪 DRY RUN: Archivo se generaría en: {output_path}")
            return str(output_path)
        
        try:
            # Verificar archivo fuente
            if not os.path.exists(self.source_file_path):
                raise FileNotFoundError(f"Archivo fuente no encontrado: {self.source_file_path}")
            
            print(f"   📂 Cargando archivo fuente...")
            
            # MOCKUP: Si no tenemos las librerías del proyecto, simular
            if 'crop_variables_xr' not in globals():
                print(f"   🧪 MODO MOCKUP: Simulando procesamiento...")
                import sys
                print(f"not well processd nc file for {target_date} and {day_offset}")
                sys.exit(0)
                # Simular creación de archivo
                dummy_data = np.random.rand(24, 25, 25)  # 24 horas, 25x25 grid
                
                # Crear archivo netCDF simulado
                import netCDF4 as nc
                
                with nc.Dataset(output_path, 'w') as ncfile:
                    # Dimensiones
                    ncfile.createDimension('time', 24)
                    ncfile.createDimension('lat', 25)
                    ncfile.createDimension('lon', 25)
                    
                    # Variables
                    time_var = ncfile.createVariable('time', 'f4', ('time',))
                    lat_var = ncfile.createVariable('lat', 'f4', ('lat',))
                    lon_var = ncfile.createVariable('lon', 'f4', ('lon',))
                    temp_var = ncfile.createVariable('T2', 'f4', ('time', 'lat', 'lon'))
                    
                    # Datos
                    time_var[:] = np.arange(24)
                    lat_var[:] = np.linspace(18.75, 20, 25)
                    lon_var[:] = np.linspace(-99.75, -98.5, 25)
                    temp_var[:] = dummy_data
                    
                    # Atributos especiales para identificar recortes
                    ncfile.setncattr('source_file', os.path.basename(self.source_file_path))
                    ncfile.setncattr('extraction_method', 'temporal_slice')
                    ncfile.setncattr('day_offset', day_offset)
                    ncfile.setncattr('original_hours_range', f"{start_hour}-{end_hour-1}")
                    ncfile.setncattr('base_date', str(self.base_date))
                    ncfile.setncattr('target_date', target_date)
                    ncfile.setncattr('created_by', 'TemporalSliceExtractor')
                    ncfile.setncattr('creation_time', datetime.now().isoformat())
                    
                print(f"   ✅ Archivo simulado creado: {output_filename}")
                
            else:
                print(f"   🔧 MODO REAL: Procesando con librerías del proyecto...")
                
                # Cargar archivo fuente
                ds = xr.open_dataset(self.source_file_path, decode_times=False)
                print(f"   📊 Dimensiones originales: {dict(ds.dims)}")
                
                # Verificar que tenemos suficientes horas
                max_time_available = ds.dims.get('Time', ds.dims.get('time', 24))
                if end_hour > max_time_available:
                    raise ValueError(f"No hay suficientes horas en el archivo: necesitamos hasta {end_hour}, disponibles {max_time_available}")
                
                # Definir rango de horas a extraer
                times = list(range(start_hour, end_hour))
                print(f"   ⏰ Extrayendo horas: {times}")
                
                # Cropping y procesamiento - incluir Q2 y PSFC temporalmente para calcular RH
                print(f"   ✂️  Aplicando cropping espacial y temporal...")
                variables_with_extra = self.variable_names + ['Q2', 'PSFC']
                cropped_ds, newLAT, newLon = crop_variables_xr(ds, variables_with_extra, self.bbox, times)
                
                # Procesar variables derivadas (igual que en 1_MakeNetcdf_From_WRF.py)
                print(f"   🧮 Calculando variables derivadas...")
                
                # Lluvia total (RAIN = RAINC + RAINNC) - NO CUMULATIVA
                if 'RAINC' in cropped_ds.data_vars and 'RAINNC' in cropped_ds.data_vars:
                    print("   🌧️ Calculando lluvia total...")
                    cropped_ds['RAIN'] = cropped_ds['RAINC'] + cropped_ds['RAINNC']
                    # Hacer RAIN no cumulativa (diferencia entre pasos)
                    rain_values = cropped_ds['RAIN'].values
                    rain_diff = np.zeros_like(rain_values)
                    rain_diff[1:,:,:] = rain_values[1:,:,:] - rain_values[:-1,:,:]
                    cropped_ds['RAIN'] = xr.DataArray(rain_diff, dims=cropped_ds['RAIN'].dims, coords=cropped_ds['RAIN'].coords)
                    # Set negative values to 0
                    cropped_ds['RAIN'] = cropped_ds['RAIN'].where(cropped_ds['RAIN'] > 0, 0)
                    cropped_ds = cropped_ds.drop(['RAINC', 'RAINNC'])
                
                # Velocidad del viento (WS10)
                if 'U10' in cropped_ds.data_vars and 'V10' in cropped_ds.data_vars:
                    print("   💨 Calculando velocidad del viento...")
                    cropped_ds['WS10'] = np.sqrt(cropped_ds['U10']**2 + cropped_ds['V10']**2)
                
                # Humedad relativa (RH) - usar Q2 y PSFC del archivo original
                if all(var in cropped_ds.data_vars for var in ['T2', 'Q2', 'PSFC']):
                    print("   💧 Calculando humedad relativa...")
                    T2 = cropped_ds['T2'].values
                    PSFC = cropped_ds['PSFC'].values  
                    Q2 = cropped_ds['Q2'].values
                    RH = calculate_relative_humidity_metpy(T2, PSFC, Q2)
                    cropped_ds['RH'] = xr.DataArray(RH, dims=cropped_ds['T2'].dims, coords=cropped_ds['T2'].coords)
                    # Remover Q2 y PSFC para mantener solo las variables finales
                    cropped_ds = cropped_ds.drop(['Q2', 'PSFC'])
                
                # APLICAR AJUSTE DE DELTA TIME (igual que en 1_MakeNetcdf_From_WRF.py)
                print(f"   🕐 Aplicando ajuste de zona horaria...")
                target_dt = pd.to_datetime(target_date)
                first_datetime = target_dt.replace(hour=0, minute=0, second=0, microsecond=0) - pd.Timedelta(hours=6)
                
                # Actualizar atributos de tiempo para seguir convenciones CF (MANTENIENDO ENTEROS)
                cropped_ds['time'].attrs.update({
                    'units': f'hours since {first_datetime.strftime("%Y-%m-%d %H:%M:%S")}',
                    'calendar': 'standard',
                    'axis': 'T',
                    'long_name': 'time',
                    'standard_name': 'time'
                })
                
                # Asegurar que el tiempo se mantenga como enteros (no fechas)
                if hasattr(cropped_ds['time'], 'values'):
                    # Convertir de vuelta a enteros si se convirtió a fechas
                    time_values = cropped_ds['time'].values
                    if hasattr(time_values, 'dtype') and 'datetime' in str(time_values.dtype):
                        print(f"   🔧 Convirtiendo tiempo de fechas a enteros...")
                        # Crear nuevos valores de tiempo como enteros
                        new_time_values = np.arange(len(time_values), dtype=np.int64)
                        cropped_ds = cropped_ds.assign_coords(time=new_time_values)
                
                # INTERPOLACIÓN A RESOLUCIÓN OBJETIVO (igual que en 1_MakeNetcdf_From_WRF.py)
                print(f"   📐 Aplicando interpolación a resolución {self.resolution}...")
                
                # Crear nuevas coordenadas en la resolución objetivo
                new_lat = np.arange(self.bbox[0], self.bbox[1], self.resolution)
                new_lon = np.arange(self.bbox[2], self.bbox[3], self.resolution)
                
                # Interpolar el dataset a las nuevas coordenadas
                cropped_ds = cropped_ds.interp(
                    lat=new_lat,
                    lon=new_lon,
                    method='linear'
                )
                
                # Actualizar atributos de lat/lon
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
                
                # Agregar metadatos de recorte
                cropped_ds.attrs['source_file'] = os.path.basename(self.source_file_path)
                cropped_ds.attrs['extraction_method'] = 'temporal_slice'
                cropped_ds.attrs['day_offset'] = day_offset
                cropped_ds.attrs['original_hours_range'] = f"{start_hour}-{end_hour-1}"
                cropped_ds.attrs['base_date'] = str(self.base_date)
                cropped_ds.attrs['target_date'] = target_date
                cropped_ds.attrs['created_by'] = 'TemporalSliceExtractor'
                cropped_ds.attrs['creation_time'] = datetime.now().isoformat()
                
                # Guardar archivo
                print(f"   💾 Guardando archivo...")
                cropped_ds.to_netcdf(output_path)
                
                print(f"   ✅ Archivo real procesado: {output_filename}")
            
            # Verificar archivo generado
            file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
            print(f"   📁 Tamaño: {file_size:.1f} MB")
            
            return str(output_path)
            
        except Exception as e:
            print(f"   ❌ Error procesando recorte: {e}")
            raise

    def extract_multiple_days(self, num_days: int = 2, add_labels: bool = True, dry_run: bool = False) -> list:
        """
        Extrae múltiples días de recortes temporales.
        
        Args:
            num_days: Número de días a extraer después de la fecha base
            add_labels: Si agregar labels a los archivos
            dry_run: Si solo simular
            
        Returns:
            Lista de rutas de archivos generados
        """
        print(f"\n🔄 EXTRAYENDO MÚLTIPLES RECORTES TEMPORALES")
        print("=" * 60)
        print(f"   📅 Fecha base: {self.base_date}")
        print(f"   📊 Días a extraer: {num_days}")
        print(f"   🏷️  Con labels: {'Sí' if add_labels else 'No'}")
        print(f"   🧪 Dry run: {'Sí' if dry_run else 'No'}")
        
        generated_files = []
        
        for day_offset in range(1, num_days + 1):
            target_date = self.base_date + timedelta(days=day_offset)
            target_date_str = target_date.strftime('%Y-%m-%d')
            
            print(f"\n📅 PROCESANDO DÍA {day_offset}/{num_days}:")
            
            try:
                output_path = self.extract_temporal_slice(
                    target_date_str, day_offset, add_labels, dry_run
                )
                generated_files.append(output_path)
                import time
                time.sleep(3)
                
            except Exception as e:
                print(f"   ❌ Error en día {day_offset}: {e}")
                continue
        
        print(f"\n✅ EXTRACCIÓN MÚLTIPLE COMPLETADA")
        print(f"   📊 Archivos generados: {len(generated_files)}")
        for i, file_path in enumerate(generated_files, 1):
            print(f"   {i}. {os.path.basename(file_path)}")
        
        return generated_files

    def verify_slice_metadata(self, file_path: str) -> dict:
        """Verifica y muestra metadatos de un archivo de recorte."""
        try:
            print(f"\n🔍 VERIFICANDO METADATOS DE RECORTE")
            print("-" * 50)
            print(f"   📁 Archivo: {os.path.basename(file_path)}")
            
            ds = xr.open_dataset(file_path)
            
            metadata = {}
            slice_attrs = ['source_file', 'extraction_method', 'day_offset', 
                          'original_hours_range', 'base_date', 'target_date', 
                          'created_by', 'creation_time']
            
            for attr in slice_attrs:
                if attr in ds.attrs:
                    metadata[attr] = ds.attrs[attr]
                    print(f"   {attr}: {ds.attrs[attr]}")
            
            print(f"   📊 Dimensiones: {dict(ds.dims)}")
            print(f"   🌡️  Variables: {list(ds.data_vars.keys())}")
            
            ds.close()
            return metadata
            
        except Exception as e:
            print(f"   ❌ Error verificando metadatos: {e}")
            return {}


def main():
    """Función principal del script de demostración."""
    parser = argparse.ArgumentParser(description='Extractor de recortes temporales WRF')
    
    # Nuevo argumento para target-datetime
    parser.add_argument('--target-datetime', type=str,
                       help='Fecha objetivo (YYYY-MM-DD HH:MM:SS) - Si se proporciona, se usa en lugar de base-date')
    
    parser.add_argument('--base-date', type=str, default='2023-05-05',
                       help='Fecha base (YYYY-MM-DD) - Solo usado si no se proporciona target-datetime')
    parser.add_argument('--source-file', type=str,
                       help='Ruta al archivo WRF fuente')
    parser.add_argument('--output-folder', type=str, default='/dev/shm/tem_ram_forecast',
                       help='Carpeta de salida')
    parser.add_argument('--num-days', type=int, default=2,
                       help='Número de días a extraer')
    parser.add_argument('--dry-run', action='store_true',
                       help='Solo simular, no procesar')
    parser.add_argument('--no-labels', action='store_true',
                       help='No agregar labels a los archivos')
    parser.add_argument('--verify', action='store_true',
                       help='Verificar metadatos de archivos existentes')
    
    args = parser.parse_args()
    
    try:
        print("🚀 DEMO: EXTRACTOR DE RECORTES TEMPORALES WRF")
        print("=" * 60)
        
        # Determinar fecha base basada en target-datetime o base-date
        if args.target_datetime:
            from datetime import datetime, timedelta
            target_dt = datetime.strptime(args.target_datetime, '%Y-%m-%d %H:%M:%S')
            base_date = (target_dt - timedelta(days=1)).strftime('%Y-%m-%d')# anteriormente tenia delta 1 day..
            print(f"📅 Target datetime: {args.target_datetime}")
            print(f"📅 Fecha base calculada: {base_date}")
        else:
            base_date = args.base_date
            print(f"📅 Fecha base: {base_date}")
        
        print(f"🎯 Escenario: Solo disponible archivo del {base_date}")
        print(f"📊 Objetivo: Generar archivos de {args.num_days} días siguientes")
        print("=" * 60)
        
        # Ruta por defecto para el ejemplo - usar la misma lógica que process_wrf_files_like_in_train.py
        if not args.source_file:
            # Construir ruta usando la misma lógica que el script principal
            year_folder = str(datetime.strptime(base_date, '%Y-%m-%d').year)
            month_num = datetime.strptime(base_date, '%Y-%m-%d').month
            
            # Función para obtener nombre de carpeta de mes
            month_names = {
                1: '01_enero', 2: '02_febrero', 3: '03_marzo', 4: '04_abril',
                5: '05_mayo', 6: '06_junio', 7: '07_julio', 8: '08_agosto',
                9: '09_septiembre', 10: '10_octubre', 11: '11_noviembre', 12: '12_diciembre'
            }
            month_folder = month_names[month_num]
            
            # Construir ruta completa
            input_folder = "/ServerData/WRF_2017_Kraken/"
            file_pattern = f"wrfout_d02_{base_date}_00.nc"
            args.source_file = os.path.join(input_folder, year_folder, month_folder, file_pattern)
            
            print(f"📂 Archivo fuente (por defecto): {args.source_file}")
        
        # Crear extractor
        extractor = TemporalSliceExtractor(
            base_date=base_date,
            source_file_path=args.source_file,
            output_folder=args.output_folder
        )
        
        if args.verify:
            # Modo verificación
            print(f"\n🔍 MODO VERIFICACIÓN")
            pattern = os.path.join(args.output_folder, "*SLICE*.nc")
            import glob
            slice_files = glob.glob(pattern)
            
            if slice_files:
                for file_path in slice_files:
                    extractor.verify_slice_metadata(file_path)
            else:
                print(f"   ⚠️  No se encontraron archivos de recorte en {args.output_folder}")
        else:
            # Modo extracción - SIEMPRE sin labels para nombres simples
            generated_files = extractor.extract_multiple_days(
                num_days=args.num_days,
                add_labels=False,  # Siempre False para nombres simples
                dry_run=args.dry_run
            )
            
            # Verificar archivos generados
            if generated_files and not args.dry_run:
                print(f"\n🔍 VERIFICACIÓN DE ARCHIVOS GENERADOS:")
                for file_path in generated_files:
                    if os.path.exists(file_path):
                        extractor.verify_slice_metadata(file_path)
        
        print(f"\n🎉 DEMO COMPLETADA EXITOSAMENTE")
        
    except Exception as e:
        print(f"\n❌ ERROR EN DEMO: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())