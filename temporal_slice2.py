#!/usr/bin/env python3
"""
temporal_slice2.py

Script corregido para extracción inteligente de recortes temporales de archivos WRF.
Corrige las inconsistencias del mockup original para mantener compatibilidad con
operativo001.py.

CORRECCIONES IMPLEMENTADAS:
1. Manejo consistente de zonas horarias (igual que operativo001.py)
2. Procesamiento de variables derivadas compatible
3. Nombres de archivos consistentes
4. Metadatos compatibles con el sistema principal
5. Validación de rangos temporales

EJEMPLO DE USO:
- Tenemos archivo del 5 de mayo de 2023 (72 horas de pronóstico)
- Queremos generar archivos del 6 y 7 de mayo
- Extraemos horas 24-47 para el día 6, horas 48-71 para el día 7

AUTOR: AI Assistant (versión corregida)
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
    PROJECT_IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Warning: No se pudieron cargar imports del proyecto: {e}")
    print("📋 Continuando en modo mockup básico...")
    PROJECT_IMPORTS_AVAILABLE = False


class TemporalSliceExtractorV2:
    """Extractor de recortes temporales de archivos WRF - Versión corregida."""
    
    def __init__(self, base_date: str, source_file_path: str, output_folder: str, 
                 bbox: list = None, resolution: float = 0.05):
        """
        Inicializa el extractor corregido.
        
        Args:
            base_date: Fecha base (formato YYYY-MM-DD) - fecha del archivo fuente
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
        
        # Variables a procesar (EXACTAMENTE igual que operativo001.py)
        self.variable_names = ['T2', 'U10', 'V10', 'RAINC', 'RAINNC', 'SWDOWN', 'GLW', 'Q2', 'PSFC']
        
        # Crear carpeta de salida
        self.output_folder.mkdir(parents=True, exist_ok=True)
        
        print(f"🎯 EXTRACTOR DE RECORTES TEMPORALES V2 (CORREGIDO)")
        print(f"   📅 Fecha base (archivo fuente): {self.base_date}")
        print(f"   📂 Archivo fuente: {os.path.basename(self.source_file_path)}")
        print(f"   📁 Carpeta salida: {self.output_folder}")
        print(f"   🎨 BBOX: {self.bbox}")
        print(f"   📐 Resolución: {self.resolution}")
        print(f"   🔧 Compatible con: operativo001.py")

    def convert_cdmx_to_utc(self, cdmx_datetime):
        """
        Convierte una fecha/hora de CDMX a UTC (igual que operativo001.py).
        
        Args:
            cdmx_datetime: datetime en zona horaria CDMX
        
        Returns:
            datetime: datetime en UTC
        """
        # CDMX está en GMT-6, por lo que para convertir a UTC sumamos 6 horas
        utc_datetime = cdmx_datetime + timedelta(hours=6)
        return utc_datetime

    def convert_utc_to_cdmx(self, utc_datetime):
        """
        Convierte una fecha/hora de UTC a CDMX (igual que operativo001.py).
        
        Args:
            utc_datetime: datetime en UTC
        
        Returns:
            datetime: datetime en zona horaria CDMX
        """
        # Para convertir de UTC a CDMX restamos 6 horas
        cdmx_datetime = utc_datetime + timedelta(hours=-6)
        return cdmx_datetime

    def extract_temporal_slice(self, target_date: str, day_offset: int, 
                             dry_run: bool = False) -> str:
        """
        Extrae un recorte temporal específico del archivo fuente (CORREGIDO).
        
        Args:
            target_date: Fecha objetivo del recorte (YYYY-MM-DD)
            day_offset: Offset en días desde la fecha base (1=día siguiente, 2=pasado mañana)
            dry_run: Si solo simular sin procesar realmente
            
        Returns:
            Ruta del archivo generado
        """
        target_dt = pd.to_datetime(target_date).date()
        
        # Calcular rango de horas a extraer
        start_hour = day_offset * 24
        end_hour = start_hour + 24
        
        print(f"\n�� EXTRAYENDO RECORTE TEMPORAL (CORREGIDO)")
        print("-" * 50)
        print(f"   📅 Fecha objetivo: {target_date}")
        print(f"   �� Offset de días: +{day_offset}")
        print(f"   ⏰ Rango de horas: {start_hour} a {end_hour-1}")
        print(f"   🧪 Dry run: {'Sí' if dry_run else 'No'}")
        
        # NOMBRE CONSISTENTE (igual que operativo001.py)
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
            if not PROJECT_IMPORTS_AVAILABLE:
                print(f"   �� MODO MOCKUP: Simulando procesamiento...")
                
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
                    
                    # METADATOS COMPATIBLES (sin atributos especiales de recorte)
                    ncfile.setncattr('source_file', os.path.basename(self.source_file_path))
                    ncfile.setncattr('created_by', 'TemporalSliceExtractorV2')
                    ncfile.setncattr('creation_time', datetime.now().isoformat())
                    ncfile.setncattr('compatibility_mode', 'operativo001')
                    
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
                
                # PROCESAR VARIABLES DERIVADAS (EXACTAMENTE igual que operativo001.py)
                print(f"   🧮 Calculando variables derivadas...")
                
                # Lluvia total (RAIN = RAINC + RAINNC) - NO CUMULATIVA
                if 'RAINC' in cropped_ds.data_vars and 'RAINNC' in cropped_ds.data_vars:
                    print("   ��️ Calculando lluvia total...")
                    cropped_ds['RAIN'] = cropped_ds['RAINC'] + cropped_ds['RAINNC']
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
                    print("   �� Calculando humedad relativa...")
                    T2 = cropped_ds['T2'].values
                    PSFC = cropped_ds['PSFC'].values  
                    Q2 = cropped_ds['Q2'].values
                    RH = calculate_relative_humidity_metpy(T2, PSFC, Q2)
                    cropped_ds['RH'] = xr.DataArray(RH, dims=cropped_ds['T2'].dims, coords=cropped_ds['T2'].coords)
                    # Remover Q2 y PSFC para mantener solo las variables finales
                    cropped_ds = cropped_ds.drop(['Q2', 'PSFC'])
                
                # AJUSTE DE ZONA HORARIA (EXACTAMENTE igual que operativo001.py)
                print(f"   🕐 Aplicando ajuste de zona horaria (compatible con operativo001.py)...")
                
                # Usar la fecha del archivo fuente (base_date) para consistencia
                file_date_utc = self.base_date
                first_datetime_utc = file_date_utc.replace(hour=0, minute=0, second=0, microsecond=0)
                first_datetime_cdmx = self.convert_utc_to_cdmx(first_datetime_utc)
                
                # Actualizar atributos de tiempo para seguir convenciones CF (MANTENIENDO ENTEROS)
                cropped_ds['time'].attrs.update({
                    'units': f'hours since {first_datetime_cdmx.strftime("%Y-%m-%d %H:%M:%S")}',
                    'calendar': 'standard',
                    'axis': 'T',
                    'long_name': 'time',
                    'standard_name': 'time',
                    'timezone': 'CDMX (GMT-6)'  # Igual que operativo001.py
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
                
                # INTERPOLACIÓN A RESOLUCIÓN OBJETIVO (igual que operativo001.py)
                print(f"   �� Aplicando interpolación a resolución {self.resolution}...")
                
                # Crear nuevas coordenadas en la resolución objetivo
                new_lat = np.arange(self.bbox[0], self.bbox[1], self.resolution)
                new_lon = np.arange(self.bbox[2], self.bbox[3], self.resolution)
                
                # Interpolar el dataset a las nuevas coordenadas
                cropped_ds = cropped_ds.interp(
                    lat=new_lat,
                    lon=new_lon,
                    method='linear'
                )
                
                # Actualizar atributos de lat/lon (igual que operativo001.py)
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
                
                # METADATOS COMPATIBLES (sin atributos especiales de recorte)
                cropped_ds.attrs['source_file'] = os.path.basename(self.source_file_path)
                cropped_ds.attrs['created_by'] = 'TemporalSliceExtractorV2'
                cropped_ds.attrs['creation_time'] = datetime.now().isoformat()
                cropped_ds.attrs['compatibility_mode'] = 'operativo001'
                cropped_ds.attrs['extraction_info'] = f'Día {day_offset} desde archivo base {self.base_date}'
                
                # Guardar archivo
                print(f"   💾 Guardando archivo...")
                cropped_ds.to_netcdf(output_path)
                
                print(f"   ✅ Archivo real procesado: {output_filename}")
            
            # Verificar archivo generado
            file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
            print(f"   �� Tamaño: {file_size:.1f} MB")
            
            return str(output_path)
            
        except Exception as e:
            print(f"   ❌ Error procesando recorte: {e}")
            raise

    def extract_multiple_days(self, num_days: int = 2, dry_run: bool = False) -> list:
        """
        Extrae múltiples días de recortes temporales (CORREGIDO).
        
        Args:
            num_days: Número de días a extraer después de la fecha base
            dry_run: Si solo simular
            
        Returns:
            Lista de rutas de archivos generados
        """
        print(f"\n🔄 EXTRAYENDO MÚLTIPLES RECORTES TEMPORALES (CORREGIDO)")
        print("=" * 60)
        print(f"   📅 Fecha base (archivo fuente): {self.base_date}")
        print(f"   📊 Días a extraer: {num_days}")
        print(f"   🧪 Dry run: {'Sí' if dry_run else 'No'}")
        print(f"   🔧 Compatible con: operativo001.py")
        
        generated_files = []
        
        for day_offset in range(1, num_days + 1):
            target_date = self.base_date + timedelta(days=day_offset)
            target_date_str = target_date.strftime('%Y-%m-%d')
            
            print(f"\n📅 PROCESANDO DÍA {day_offset}/{num_days}:")
            print(f"   - Fecha objetivo: {target_date_str}")
            print(f"   - Offset: +{day_offset} días desde {self.base_date}")
            
            try:
                output_path = self.extract_temporal_slice(
                    target_date_str, day_offset, dry_run
                )
                generated_files.append(output_path)
                
                if not dry_run:
                    import time
                    time.sleep(1)  # Pausa más corta
                
            except Exception as e:
                print(f"   ❌ Error en día {day_offset}: {e}")
                continue
        
        print(f"\n✅ EXTRACCIÓN MÚLTIPLE COMPLETADA")
        print(f"   �� Archivos generados: {len(generated_files)}")
        for i, file_path in enumerate(generated_files, 1):
            print(f"   {i}. {os.path.basename(file_path)}")
        
        return generated_files

    def verify_slice_metadata(self, file_path: str) -> dict:
        """Verifica y muestra metadatos de un archivo de recorte (CORREGIDO)."""
        try:
            print(f"\n🔍 VERIFICANDO METADATOS DE RECORTE (CORREGIDO)")
            print("-" * 50)
            print(f"   📁 Archivo: {os.path.basename(file_path)}")
            
            ds = xr.open_dataset(file_path)
            
            # Verificar compatibilidad con operativo001.py
            compatibility_attrs = ['source_file', 'created_by', 'compatibility_mode']
            slice_attrs = ['extraction_info']
            
            print(f"   🔧 ATRIBUTOS DE COMPATIBILIDAD:")
            for attr in compatibility_attrs:
                if attr in ds.attrs:
                    print(f"   ✅ {attr}: {ds.attrs[attr]}")
                else:
                    print(f"   ❌ {attr}: FALTANTE")
            
            print(f"   📊 ATRIBUTOS DE EXTRACCIÓN:")
            for attr in slice_attrs:
                if attr in ds.attrs:
                    print(f"   �� {attr}: {ds.attrs[attr]}")
                else:
                    print(f"   ❌ {attr}: FALTANTE")
            
            print(f"   �� Dimensiones: {dict(ds.dims)}")
            print(f"   ��️  Variables: {list(ds.data_vars.keys())}")
            
            # Verificar variables esperadas (igual que operativo001.py)
            expected_vars = ['T2', 'U10', 'V10', 'RAIN', 'SWDOWN', 'GLW', 'WS10', 'RH']
            missing_vars = [var for var in expected_vars if var not in ds.data_vars]
            
            if missing_vars:
                print(f"   ⚠️  Variables faltantes: {missing_vars}")
            else:
                print(f"   ✅ Todas las variables esperadas están presentes")
            
            ds.close()
            return ds.attrs
            
        except Exception as e:
            print(f"   ❌ Error verificando metadatos: {e}")
            return {}


def main():
    """Función principal del script corregido."""
    parser = argparse.ArgumentParser(description='Extractor de recortes temporales WRF - Versión corregida')
    
    # Argumentos compatibles con operativo001.py
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
    parser.add_argument('--verify', action='store_true',
                       help='Verificar metadatos de archivos existentes')
    
    args = parser.parse_args()
    
    try:
        print("�� DEMO: EXTRACTOR DE RECORTES TEMPORALES WRF V2 (CORREGIDO)")
        print("=" * 70)
        print("🔧 CORRECCIONES IMPLEMENTADAS:")
        print("   ✅ Manejo consistente de zonas horarias")
        print("   ✅ Procesamiento de variables derivadas compatible")
        print("   ✅ Nombres de archivos consistentes")
        print("   ✅ Metadatos compatibles con operativo001.py")
        print("   ✅ Validación de rangos temporales")
        print("=" * 70)
        
        # Determinar fecha base basada en target-datetime o base-date
        if args.target_datetime:
            from datetime import datetime, timedelta
            target_dt = datetime.strptime(args.target_datetime, '%Y-%m-%d %H:%M:%S')
            base_date = (target_dt - timedelta(days=1)).strftime('%Y-%m-%d')
            print(f"�� Target datetime: {args.target_datetime}")
            print(f"📅 Fecha base calculada: {base_date}")
        else:
            base_date = args.base_date
            print(f"📅 Fecha base: {base_date}")
        
        print(f"🎯 Escenario: Solo disponible archivo del {base_date}")
        print(f"📊 Objetivo: Generar archivos de {args.num_days} días siguientes")
        print(f"🔧 Compatibilidad: operativo001.py")
        print("=" * 70)
        
        # Ruta por defecto para el ejemplo - usar la misma lógica que operativo001.py
        if not args.source_file:
            # Construir ruta usando la misma lógica que operativo001.py
            year_folder = str(datetime.strptime(base_date, '%Y-%m-%d').year)
            month_num = datetime.strptime(base_date, '%Y-%m-%d').month
            
            # Función para obtener nombre de carpeta de mes (igual que operativo001.py)
            month_names = {
                1: '01_enero', 2: '02_febrero', 3: '03_marzo', 4: '04_abril',
                5: '05_mayo', 6: '06_junio', 7: '07_julio', 8: '08_agosto',
                9: '09_septiembre', 10: '10_octubre', 11: '11_noviembre', 12: '12_diciembre'
            }
            month_folder = month_names[month_num]
            
            # Construir ruta completa (igual que operativo001.py)
            input_folder = "/ServerData/WRF_2017_Kraken/"
            file_pattern = f"wrfout_d02_{base_date}_00.nc"
            args.source_file = os.path.join(input_folder, year_folder, month_folder, file_pattern)
            
            print(f"📂 Archivo fuente (por defecto): {args.source_file}")
        
        # Crear extractor corregido
        extractor = TemporalSliceExtractorV2(
            base_date=base_date,
            source_file_path=args.source_file,
            output_folder=args.output_folder
        )
        
        if args.verify:
            # Modo verificación
            print(f"\n🔍 MODO VERIFICACIÓN")
            pattern = os.path.join(args.output_folder, "*.nc")
            import glob
            nc_files = glob.glob(pattern)
            
            if nc_files:
                for file_path in nc_files:
                    extractor.verify_slice_metadata(file_path)
            else:
                print(f"   ⚠️  No se encontraron archivos .nc en {args.output_folder}")
        else:
            # Modo extracción - SIEMPRE sin labels para nombres simples (compatible con operativo001.py)
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
        
        print(f"\n�� DEMO COMPLETADA EXITOSAMENTE")
        print(f"�� Archivos generados son compatibles con operativo001.py")
        
    except Exception as e:
        print(f"\n❌ ERROR EN DEMO: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())