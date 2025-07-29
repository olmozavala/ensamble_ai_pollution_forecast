#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
forecast_utils2.py - Utilidades para sistema de pronóstico de contaminación

Módulo que contiene todas las funciones y clases necesarias para realizar inferencia
en tiempo real de contaminación del aire.
"""

import os
import glob
import yaml
import pickle
import netrc
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch
import xarray as xr
import psycopg2
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tqdm import tqdm

from proj_io.inout import generateDateColumns
from proj_preproc.normalization import normalize_data, denormalize_data


class ForecastBatchProcessor:
    """Procesador de lotes para barrido de fechas y generación de CSV de pronósticos."""
    
    def __init__(self, forecast_system: 'ForecastSystem', config_file_path: str):
        self.forecast_system = forecast_system
        self.config_file_path = config_file_path
        self.config = forecast_system.config
        self.results_df = None
        
    def generate_date_range(self, start_date: str, end_date: str, 
                          frequency: str = 'D') -> List[str]:
        """
        Genera rango de fechas para el barrido.
        
        Args:
            start_date: Fecha inicial (YYYY-MM-DD HH:MM:SS)
            end_date: Fecha final (YYYY-MM-DD HH:MM:SS)
            frequency: Frecuencia ('D' diario, 'H' horario, etc.)
            
        Returns:
            Lista de fechas en formato string
        """
        print(f"📅 GENERANDO RANGO DE FECHAS")
        print(f"   Inicio: {start_date}")
        print(f"   Fin: {end_date}")
        print(f"   Frecuencia: {frequency}")
        
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        date_range = pd.date_range(start=start_dt, end=end_dt, freq=frequency)
        date_strings = [dt.strftime('%Y-%m-%d %H:%M:%S') for dt in date_range]
        
        print(f"   ✅ Generadas {len(date_strings)} fechas")
        return date_strings
    
    def validate_dates_for_processing(self, date_list: List[str]) -> List[str]:
        """
        Valida que las fechas tienen suficientes datos históricos disponibles.
        
        Args:
            date_list: Lista de fechas a validar
            
        Returns:
            Lista de fechas válidas para procesamiento
        """
        print(f"🔍 VALIDANDO FECHAS PARA PROCESAMIENTO")
        
        min_required_hours = self.config['data_loader']['args']['prev_pollutant_hours']
        print(f"   🕐 Horas históricas requeridas: {min_required_hours}")
        
        valid_dates = []
        invalid_dates = []
        
        for date_str in date_list:
            target_dt = pd.to_datetime(date_str)
            
            # Calcular la fecha mínima requerida
            min_required_dt = target_dt - pd.Timedelta(hours=min_required_hours)
            
            # Verificar que la fecha objetivo no sea muy temprana
            # (esto es una validación básica, se puede hacer más sofisticada)
            if target_dt.year < 2020:  # Asumiendo que datos antes de 2020 no están disponibles
                invalid_dates.append(date_str)
                continue
            
            valid_dates.append(date_str)
        
        print(f"   ✅ Fechas válidas: {len(valid_dates)}")
        print(f"   ❌ Fechas inválidas: {len(invalid_dates)}")
        
        if invalid_dates:
            print(f"   ⚠️ Fechas saltadas (datos insuficientes): {invalid_dates[:5]}...")
        
        return valid_dates
    
    def create_forecast_columns(self, num_pollutants: int = 54, 
                              num_horizons: int = 24) -> List[str]:
        """
        Crea nombres de columnas para el CSV de pronósticos.
        
        Args:
            num_pollutants: Número de contaminantes
            num_horizons: Número de horizontes de pronóstico (horas)
            
        Returns:
            Lista de nombres de columnas
        """
        columns = []
        
        # Generar columnas por horizonte
        for horizon in range(1, num_horizons + 1):
            for pollutant_idx in range(num_pollutants):
                col_name = f"col{pollutant_idx:02d}_h_plus_{horizon:02d}"
                columns.append(col_name)
        
        print(f"📊 ESTRUCTURA DEL CSV:")
        print(f"   Contaminantes: {num_pollutants}")
        print(f"   Horizontes: {num_horizons}")
        print(f"   Total columnas: {len(columns)}")
        print(f"   Ejemplo columnas: {columns[:5]} ... {columns[-5:]}")
        
        return columns
    
    def run_batch_forecast(self, start_date: str, end_date: str,
                         frequency: str = 'D',
                         output_csv: str = 'forecast_batch_results.csv',
                         save_progress: bool = True,
                         resume_from_checkpoint: bool = True) -> pd.DataFrame:
        """
        Ejecuta barrido de fechas y genera CSV de pronósticos.
        
        Args:
            start_date: Fecha inicial (YYYY-MM-DD HH:MM:SS)
            end_date: Fecha final (YYYY-MM-DD HH:MM:SS)
            frequency: Frecuencia del barrido
            output_csv: Archivo CSV de salida
            save_progress: Guardar progreso incremental
            resume_from_checkpoint: Reanudar desde último checkpoint
            
        Returns:
            DataFrame con todos los pronósticos
        """
        print("🚀 INICIANDO BARRIDO DE FECHAS")
        print("=" * 60)
        
        # Generar fechas
        date_list = self.generate_date_range(start_date, end_date, frequency)
        
        # Validar fechas para procesamiento
        valid_date_list = self.validate_dates_for_processing(date_list)
        
        if not valid_date_list:
            print("❌ No hay fechas válidas para procesar")
            return pd.DataFrame()
        
        # Crear columnas
        forecast_columns = self.create_forecast_columns()
        
        # Inicializar DataFrame de resultados
        results_df = pd.DataFrame(
            index=pd.to_datetime([dt for dt in valid_date_list]),
            columns=forecast_columns,
            dtype=np.float32
        )
        
        # Checkpoint para reanudar
        checkpoint_file = output_csv.replace('.csv', '_checkpoint.csv')
        processed_dates = set()
        
        if resume_from_checkpoint and os.path.exists(checkpoint_file):
            print(f"📂 CARGANDO CHECKPOINT: {checkpoint_file}")
            try:
                checkpoint_df = pd.read_csv(checkpoint_file, index_col=0, parse_dates=True)
                results_df.loc[checkpoint_df.index, :] = checkpoint_df
                processed_dates = set(checkpoint_df.index.strftime('%Y-%m-%d %H:%M:%S'))
                print(f"   ✅ Cargadas {len(processed_dates)} fechas previas")
            except Exception as e:
                print(f"   ⚠️ Error cargando checkpoint: {e}")
        
        # Filtrar fechas ya procesadas
        remaining_dates = [dt for dt in valid_date_list if dt not in processed_dates]
        
        if not remaining_dates:
            print("✅ Todas las fechas ya están procesadas")
            return results_df
        
        print(f"\n📋 PLAN DE EJECUCIÓN:")
        print(f"   Total fechas generadas: {len(date_list)}")
        print(f"   Fechas válidas: {len(valid_date_list)}")
        print(f"   Ya procesadas: {len(processed_dates)}")
        print(f"   Por procesar: {len(remaining_dates)}")
        
        # Procesamiento secuencial
        print(f"\n🔄 INICIANDO PROCESAMIENTO SECUENCIAL")
        print("=" * 60)
        
        successful_forecasts = 0
        failed_forecasts = 0
        error_categories = {
            'datos_insuficientes': 0,
            'datos_faltantes': 0,
            'errores_wrf': 0,
            'otros_errores': 0
        }
        
        for i, target_date in enumerate(tqdm(remaining_dates, desc="Procesando fechas")):
            try:
                print(f"\n📅 PROCESANDO {i+1}/{len(remaining_dates)}: {target_date}")
                print("-" * 50)
                
                # Ejecutar pronóstico para esta fecha
                predictions = self.forecast_system.run_forecast(
                    target_date, 
                    self.config_file_path,
                    output_folder=f"./temp_batch_output_{target_date.replace(' ', '_').replace(':', '')}"
                )
                
                # Estructurar datos para CSV
                forecast_data = self._structure_forecast_data(predictions)
                
                # Guardar en DataFrame principal
                target_dt = pd.to_datetime(target_date)
                results_df.loc[target_dt, :] = forecast_data
                
                successful_forecasts += 1
                print(f"   ✅ Pronóstico exitoso para {target_date}")
                
                # Guardar progreso incremental
                if save_progress and (i + 1) % 1 == 0:  # Guardar cada pronóstico
                    try:
                        # Guardar solo las filas no nulas
                        non_null_rows = results_df.dropna(how='all')
                        non_null_rows.to_csv(checkpoint_file)
                        print(f"   💾 Checkpoint guardado: {len(non_null_rows)} fechas")
                    except Exception as e:
                        print(f"   ⚠️ Error guardando checkpoint: {e}")
                
            except Exception as e:
                failed_forecasts += 1
                error_type = type(e).__name__
                
                # Categorizar el error para mensaje más informativo
                if "no hay suficientes datos" in str(e).lower():
                    error_msg = f"datos insuficientes ({str(e)[:100]}...)"
                    error_categories['datos_insuficientes'] += 1
                elif "not in index" in str(e).lower():
                    error_msg = f"datos faltantes en base de datos ({str(e)[:100]}...)"
                    error_categories['datos_faltantes'] += 1
                elif "wrf" in str(e).lower():
                    error_msg = f"error procesando archivos WRF ({str(e)[:100]}...)"
                    error_categories['errores_wrf'] += 1
                else:
                    error_msg = f"{error_type}: {str(e)[:100]}..."
                    error_categories['otros_errores'] += 1
                
                print(f"   ❌ Error procesando {target_date}: {error_msg}")
                
                # Llenar con NaN para mantener estructura
                target_dt = pd.to_datetime(target_date)
                results_df.loc[target_dt, :] = np.nan
                
                # Continuar con siguiente fecha
                continue
        
        # Guardar resultados finales
        print(f"\n💾 GUARDANDO RESULTADOS FINALES")
        print("-" * 50)
        
        # Eliminar filas completamente nulas
        final_results = results_df.dropna(how='all')
        final_results.to_csv(output_csv)
        
        print(f"✅ Archivo guardado: {output_csv}")
        print(f"📊 Registros finales: {len(final_results)}")
        
        # Resumen final
        print(f"\n📈 RESUMEN FINAL:")
        print("=" * 60)
        print(f"   ✅ Pronósticos exitosos: {successful_forecasts}")
        print(f"   ❌ Pronósticos fallidos: {failed_forecasts}")
        print(f"   📊 Tasa de éxito: {(successful_forecasts/(successful_forecasts+failed_forecasts)*100):.1f}%")
        print(f"   📁 Archivo final: {output_csv}")
        print(f"   💾 Checkpoint: {checkpoint_file}")
        
        # Desglose de errores
        if failed_forecasts > 0:
            print(f"\n🔍 DESGLOSE DE ERRORES:")
            for error_type, count in error_categories.items():
                if count > 0:
                    print(f"   {error_type.replace('_', ' ').title()}: {count}")
            
            print(f"\n💡 RECOMENDACIONES:")
            if error_categories['datos_insuficientes'] > 0:
                print("   • Considere usar fechas más recientes (después de 2020)")
                print("   • Verifique que hay al menos 24 horas de datos históricos")
            if error_categories['datos_faltantes'] > 0:
                print("   • Algunos datos pueden estar faltando en la base de datos")
                print("   • Verifique la conectividad a la base de datos")
            if error_categories['errores_wrf'] > 0:
                print("   • Problemas con archivos meteorológicos WRF")
                print("   • Verifique que operativo001.py funciona correctamente")
            if error_categories['otros_errores'] > 0:
                print("   • Revise los logs para errores específicos")
        
        # Limpiar archivos temporales
        self._cleanup_temp_files()
        
        return final_results
    
    def _structure_forecast_data(self, predictions: pd.DataFrame) -> np.ndarray:
        """
        Estructura los datos de pronóstico en el formato requerido para CSV.
        
        Args:
            predictions: DataFrame con predicciones [time_steps, pollutants]
            
        Returns:
            Array con datos estructurados [pollutants * horizons]
        """
        num_horizons = len(predictions)  # 24 horas
        num_pollutants = len(predictions.columns)  # 54 contaminantes
        
        # Crear array de salida
        forecast_data = np.zeros(num_pollutants * num_horizons)
        
        # Llenar datos por horizonte
        for h in range(num_horizons):
            for p in range(num_pollutants):
                index = h * num_pollutants + p
                if h < len(predictions) and p < len(predictions.columns):
                    forecast_data[index] = predictions.iloc[h, p]
                else:
                    forecast_data[index] = np.nan
        
        return forecast_data
    
    def _cleanup_temp_files(self):
        """Limpia archivos temporales generados durante el procesamiento."""
        print("\n🧹 LIMPIANDO ARCHIVOS TEMPORALES...")
        
        # Buscar directorios temporales
        temp_dirs = glob.glob('./temp_batch_output_*')
        
        if temp_dirs:
            for temp_dir in temp_dirs:
                try:
                    import shutil
                    shutil.rmtree(temp_dir)
                    print(f"   🗑️ Eliminado: {temp_dir}")
                except Exception as e:
                    print(f"   ⚠️ Error eliminando {temp_dir}: {e}")
        else:
            print("   ✅ No hay archivos temporales para limpiar")
    
    def analyze_results(self, results_csv: str) -> Dict[str, Any]:
        """
        Analiza los resultados del barrido de fechas.
        
        Args:
            results_csv: Archivo CSV con resultados
            
        Returns:
            Diccionario con estadísticas de análisis
        """
        print(f"📊 ANALIZANDO RESULTADOS: {results_csv}")
        
        if not os.path.exists(results_csv):
            print(f"❌ Archivo no encontrado: {results_csv}")
            return {}
        
        # Cargar datos
        df = pd.read_csv(results_csv, index_col=0, parse_dates=True)
        
        # Estadísticas básicas
        stats = {
            'total_dates': len(df),
            'total_columns': len(df.columns),
            'missing_values': df.isnull().sum().sum(),
            'completion_rate': (1 - df.isnull().sum().sum() / df.size) * 100,
            'date_range': {
                'start': df.index.min(),
                'end': df.index.max()
            },
            'value_stats': {
                'mean': df.mean().mean(),
                'std': df.std().mean(),
                'min': df.min().min(),
                'max': df.max().max()
            }
        }
        
        print(f"   📅 Fechas procesadas: {stats['total_dates']}")
        print(f"   📊 Columnas: {stats['total_columns']}")
        print(f"   ✅ Tasa de completitud: {stats['completion_rate']:.1f}%")
        print(f"   📈 Valores promedio: {stats['value_stats']['mean']:.3f}")
        
        return stats


class WRFProcessor:
    """Procesador de archivos WRF que ejecuta operativo001.py para generar archivos necesarios."""
    
    def __init__(self, operativo_script_path: str = 'operativo001.py'):
        self.operativo_script_path = operativo_script_path
        
    def process_wrf_files(self, target_datetime: str, config_file_path: str, 
                         verbose: bool = True) -> bool:
        """
        Ejecuta operativo001.py para generar archivos WRF necesarios.
        
        Args:
            target_datetime: Fecha objetivo en formato 'YYYY-MM-DD HH:MM:SS'
            config_file_path: Ruta al archivo de configuración JSON
            verbose: Mostrar mensajes detallados
            
        Returns:
            True si el procesamiento fue exitoso, False en caso contrario
        """
        if verbose:
            print("🚀 EJECUTANDO OPERATIVO001.PY PARA GENERAR ARCHIVOS WRF")
            print("=" * 60)
        
        # Construir comando
        comando = [
            sys.executable,
            self.operativo_script_path,
            '--target-datetime', target_datetime,
            '--config-file', config_file_path
        ]
        
        if verbose:
            print(f"📅 Fecha objetivo: {target_datetime}")
            print(f"🔧 Comando: {' '.join(comando)}")
            print(f"📁 Directorio: {os.getcwd()}")
        
        # Verificar que el script existe
        if not os.path.exists(self.operativo_script_path):
            if verbose:
                print(f"❌ ERROR: No se encuentra {self.operativo_script_path}")
                available_scripts = [f for f in os.listdir('.') if f.endswith('.py')]
                print(f"   Scripts disponibles: {available_scripts}")
            return False
        
        if verbose:
            print(f"✅ Script encontrado: {self.operativo_script_path}")
        
        try:
            if verbose:
                print("\n🔄 EJECUTANDO PROCESAMIENTO WRF...")
                print("-" * 50)
            
            # Ejecutar el script
            proceso = subprocess.run(
                comando,
                capture_output=False,  # Mostrar salida en tiempo real
                text=True,
                cwd=os.getcwd()
            )
            
            if verbose:
                print("-" * 50)
            
            if proceso.returncode == 0:
                if verbose:
                    print("✅ PROCESAMIENTO WRF EXITOSO")
                    print(f"   Código de salida: {proceso.returncode}")
                return True
            else:
                if verbose:
                    print("❌ ERROR EN PROCESAMIENTO WRF")
                    print(f"   Código de salida: {proceso.returncode}")
                return False
                
        except Exception as e:
            if verbose:
                print(f"❌ ERROR EJECUTANDO {self.operativo_script_path}: {str(e)}")
                import traceback
                traceback.print_exc()
            return False
        
        finally:
            if verbose:
                print(f"\n🔚 EJECUCIÓN DE {self.operativo_script_path} COMPLETADA")
                print("=" * 60)


class DatabaseManager:
    """Gestión de conexiones y consultas a la base de datos de contaminación."""
    
    def __init__(self):
        self.connection = None
        
    def connect(self) -> psycopg2.extensions.connection:
        """Establece conexión con la base de datos PostgreSQL."""
        try:
            print("🔌 Conectando a base de datos...")
            secrets = netrc.netrc()
            login, account, passw = secrets.hosts['OWGIS']
            
            host = '132.248.8.238'
            self.connection = psycopg2.connect(
                database="contingencia", 
                user=login, 
                host=host, 
                password=passw
            )
            print(f"✅ Conectado a {host}")
            return self.connection
            
        except Exception as e:
            print(f"❌ Error conectando a base de datos: {e}")
            raise
    
    def disconnect(self):
        """Cierra la conexión a la base de datos."""
        if self.connection:
            self.connection.close()
            self.connection = None
            print("🔌 Conexión cerrada")
    
    def execute_query(self, query: str) -> List[Tuple]:
        """Ejecuta una consulta SQL y retorna los resultados."""
        if not self.connection:
            self.connect()
            
        cur = self.connection.cursor()
        try:
            cur.execute(query)
            rows = cur.fetchall()
            return rows
        finally:
            cur.close()
    
    def get_dataframe_from_query(self, query: str, columns: List[str]) -> pd.DataFrame:
        """Ejecuta consulta y retorna DataFrame."""
        data = self.execute_query(query)
        return pd.DataFrame(data, columns=columns)


class WRFDataLoader:
    """Cargador de datos meteorológicos WRF."""
    
    def __init__(self, wrf_folder: str):
        self.wrf_folder = Path(wrf_folder)
        
    def load_data(self, use_all_nc: bool = True, 
                  specific_files: Optional[List[str]] = None) -> xr.Dataset:
        """
        Carga datos WRF desde archivos netCDF.
        
        Args:
            use_all_nc: Si True, usa todos los archivos .nc en la carpeta
            specific_files: Lista de archivos específicos (solo si use_all_nc=False)
            
        Returns:
            Dataset combinado de WRF
        """
        if use_all_nc:
            print(f"🔍 Buscando archivos .nc en: {self.wrf_folder}")
            wrf_files = sorted(glob.glob(str(self.wrf_folder / "*.nc")))
            print(f"📁 Encontrados {len(wrf_files)} archivos")
        else:
            if not specific_files:
                raise ValueError("specific_files requerido cuando use_all_nc=False")
            wrf_files = [str(self.wrf_folder / f) for f in specific_files 
                        if (self.wrf_folder / f).exists()]
            print(f"📁 Encontrados {len(wrf_files)} de {len(specific_files)} archivos")
        
        if not wrf_files:
            raise FileNotFoundError(f"No se encontraron archivos WRF en {self.wrf_folder}")
        
        print(f"📖 Leyendo {len(wrf_files)} archivos WRF...")
        try:
            dataset = xr.open_mfdataset(wrf_files, combine='by_coords')
            print(f"✅ Dataset combinado: {dict(dataset.dims)}")
            return dataset
        except Exception as e:
            print(f"❌ Error leyendo archivos WRF: {e}")
            raise


class PollutionDataManager:
    """Gestor de datos de contaminación."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        
        # Configuración de estaciones y contaminantes
        self.all_stations = [
            "ACO", "AJM", "AJU", "ARA", "ATI", "AZC", "BJU", "CAM", "CCA", "CES", 
            "CFE", "CHO", "COR", "COY", "CUA", "CUI", "CUT", "DIC", "EAJ", "EDL", 
            "FAC", "FAN", "GAM", "HAN", "HGM", "IBM", "IMP", "INN", "IZT", "LAA", 
            "LAG", "LLA", "LOM", "LPR", "LVI", "MCM", "MER", "MGH", "MIN", "MON", 
            "MPA", "NET", "NEZ", "PED", "PER", "PLA", "POT", "SAG", "SFE", "SHA", 
            "SJA", "SNT", "SUR", "TAC", "TAH", "TAX", "TEC", "TLA", "TLI", "TPN", 
            "UAX", "UIZ", "UNM", "VAL", "VIF", "XAL", "XCH"
        ]
        
        self.pollutant_tables = [
            "cont_otres", "cont_co", "cont_nox", "cont_no", "cont_sodos", 
            "cont_pmdiez", "cont_pmdoscinco", "cont_pmco", "cont_nodos"
        ]
    
    def get_contaminant_data(self, target_datetime: str, hours_back: int = 30,
                           expected_columns: List[str] = None,
                           min_required_hours: int = 24) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Obtiene datos de contaminantes desde la base de datos.
        
        Args:
            target_datetime: Fecha objetivo en formato 'YYYY-MM-DD HH:MM:SS'
            hours_back: Horas hacia atrás a consultar
            expected_columns: Lista de columnas esperadas
            min_required_hours: Mínimo de horas históricas requeridas
            
        Returns:
            Tupla (datos_contaminantes, datos_imputados)
        """
        target_dt = pd.to_datetime(target_datetime)
        
        # Asegurar que cargamos suficientes horas históricas
        actual_hours_back = max(hours_back, min_required_hours + 5)  # +5 de buffer
        start_dt = target_dt - pd.Timedelta(hours=actual_hours_back)
        
        start_date = start_dt.strftime('%Y-%m-%d %H:%M:%S')
        end_date = target_dt.strftime('%Y-%m-%d %H:%M:%S')
        
        print(f"📅 Consultando datos desde {start_date} hasta {end_date}")
        print(f"   🕐 Horas solicitadas: {hours_back}, horas garantizadas: {actual_hours_back}")
        
        # Crear índice temporal
        date_range = pd.date_range(start=start_dt, end=target_dt, freq='H')
        
        if expected_columns is None:
            expected_columns = self._generate_expected_columns()
        
        # Inicializar DataFrames
        final_df = pd.DataFrame(index=date_range, columns=expected_columns).astype(float)
        final_df.fillna(np.nan, inplace=True)
        
        imputed_columns = [f'i_{col}' for col in expected_columns]
        df_imputed = pd.DataFrame(index=date_range, columns=imputed_columns).fillna(0)
        
        # Consultar cada tipo de contaminante
        pollutant_types = self._get_pollutant_types_from_columns(expected_columns)
        
        for pollutant in pollutant_types:
            print(f"🔍 Consultando {pollutant}...")
            data = self._query_pollutant_data(pollutant, start_date, end_date)
            
            if data:
                df = pd.DataFrame(data, columns=['fecha', 'val', 'id_est'])
                df['fecha'] = pd.to_datetime(df['fecha'])
                df_pivot = df.pivot(index='fecha', columns='id_est', values='val')
                df_pivot.columns = [f'{pollutant}_{col}' for col in df_pivot.columns]
                
                # Actualizar datos principales e indicadores de imputación
                for col in df_pivot.columns:
                    if col in final_df.columns:
                        final_df.loc[df_pivot.index, col] = df_pivot[col]
                        mask_col = f'i_{col}'
                        df_imputed.loc[df_pivot.index, mask_col] = 0
                        df_imputed.loc[final_df[col].isna(), mask_col] = 1
        
        # Agregar variables de tiempo
        print("🕐 Generando variables cíclicas de tiempo...")
        time_cols, time_values = generateDateColumns(date_range, flip_order=True)
        time_df = pd.DataFrame(dict(zip(time_cols, time_values)), index=date_range)
        combined_df = pd.concat([final_df, time_df], axis=1)
        
        return combined_df, df_imputed
    
    def _generate_expected_columns(self) -> List[str]:
        """Genera lista de columnas esperadas basada en CSV original."""
        # Lista hardcodeada basada en el CSV original de entrenamiento
        columns = [
            # Contaminantes OTRES
            'cont_otres_UIZ', 'cont_otres_AJU', 'cont_otres_ATI', 'cont_otres_CUA', 'cont_otres_SFE',
            'cont_otres_SAG', 'cont_otres_CUT', 'cont_otres_PED', 'cont_otres_TAH', 'cont_otres_GAM',
            'cont_otres_IZT', 'cont_otres_CCA', 'cont_otres_HGM', 'cont_otres_LPR', 'cont_otres_MGH',
            'cont_otres_CAM', 'cont_otres_FAC', 'cont_otres_TLA', 'cont_otres_MER', 'cont_otres_XAL',
            'cont_otres_LLA', 'cont_otres_TLI', 'cont_otres_UAX', 'cont_otres_BJU', 'cont_otres_MPA',
            'cont_otres_MON', 'cont_otres_NEZ', 'cont_otres_INN', 'cont_otres_AJM', 'cont_otres_VIF',
            
            # PM2.5
            'cont_pmdoscinco_UIZ', 'cont_pmdoscinco_AJU', 'cont_pmdoscinco_SFE', 'cont_pmdoscinco_SAG',
            'cont_pmdoscinco_PED', 'cont_pmdoscinco_GAM', 'cont_pmdoscinco_CCA', 'cont_pmdoscinco_HGM',
            'cont_pmdoscinco_MGH', 'cont_pmdoscinco_CAM', 'cont_pmdoscinco_TLA', 'cont_pmdoscinco_MER',
            'cont_pmdoscinco_XAL', 'cont_pmdoscinco_UAX', 'cont_pmdoscinco_BJU', 'cont_pmdoscinco_MPA',
            'cont_pmdoscinco_MON', 'cont_pmdoscinco_NEZ', 'cont_pmdoscinco_INN', 'cont_pmdoscinco_AJM',
            
            # NOX
            'cont_nox_UIZ', 'cont_nox_ATI', 'cont_nox_CUA', 'cont_nox_SFE', 'cont_nox_SAG',
            'cont_nox_CUT', 'cont_nox_PED', 'cont_nox_TAH', 'cont_nox_IZT', 'cont_nox_CCA',
            'cont_nox_HGM', 'cont_nox_LPR', 'cont_nox_MGH', 'cont_nox_CAM', 'cont_nox_FAC',
            'cont_nox_TLA', 'cont_nox_MER', 'cont_nox_XAL', 'cont_nox_LLA', 'cont_nox_TLI',
            'cont_nox_UAX', 'cont_nox_MON', 'cont_nox_NEZ', 'cont_nox_AJM', 'cont_nox_VIF',
            
            # CO
            'cont_co_UIZ', 'cont_co_ATI', 'cont_co_CUA', 'cont_co_SFE', 'cont_co_SAG',
            'cont_co_CUT', 'cont_co_PED', 'cont_co_TAH', 'cont_co_IZT', 'cont_co_CCA',
            'cont_co_HGM', 'cont_co_LPR', 'cont_co_MGH', 'cont_co_CAM', 'cont_co_FAC',
            'cont_co_TLA', 'cont_co_MER', 'cont_co_XAL', 'cont_co_LLA', 'cont_co_TLI',
            'cont_co_UAX', 'cont_co_BJU', 'cont_co_MPA', 'cont_co_MON', 'cont_co_NEZ',
            'cont_co_INN', 'cont_co_AJM', 'cont_co_VIF',
            
            # NO DOS
            'cont_nodos_UIZ', 'cont_nodos_ATI', 'cont_nodos_CUA', 'cont_nodos_SFE', 'cont_nodos_SAG',
            'cont_nodos_CUT', 'cont_nodos_PED', 'cont_nodos_TAH', 'cont_nodos_GAM', 'cont_nodos_IZT',
            'cont_nodos_CCA', 'cont_nodos_HGM', 'cont_nodos_LPR', 'cont_nodos_MGH', 'cont_nodos_CAM',
            'cont_nodos_FAC', 'cont_nodos_TLA', 'cont_nodos_MER', 'cont_nodos_XAL', 'cont_nodos_LLA',
            'cont_nodos_TLI', 'cont_nodos_UAX', 'cont_nodos_BJU', 'cont_nodos_MPA', 'cont_nodos_MON',
            'cont_nodos_NEZ', 'cont_nodos_AJM', 'cont_nodos_VIF',
            
            # NO
            'cont_no_UIZ', 'cont_no_ATI', 'cont_no_CUA', 'cont_no_SFE', 'cont_no_SAG',
            'cont_no_CUT', 'cont_no_PED', 'cont_no_TAH', 'cont_no_IZT', 'cont_no_CCA',
            'cont_no_HGM', 'cont_no_LPR', 'cont_no_MGH', 'cont_no_CAM', 'cont_no_FAC',
            'cont_no_TLA', 'cont_no_MER', 'cont_no_XAL', 'cont_no_LLA', 'cont_no_TLI',
            'cont_no_UAX', 'cont_no_MON', 'cont_no_NEZ', 'cont_no_AJM', 'cont_no_VIF',
            
            # SO DOS
            'cont_sodos_UIZ', 'cont_sodos_ATI', 'cont_sodos_CUA', 'cont_sodos_SFE', 'cont_sodos_SAG',
            'cont_sodos_CUT', 'cont_sodos_PED', 'cont_sodos_TAH', 'cont_sodos_IZT', 'cont_sodos_CCA',
            'cont_sodos_HGM', 'cont_sodos_LPR', 'cont_sodos_MGH', 'cont_sodos_CAM', 'cont_sodos_FAC',
            'cont_sodos_TLA', 'cont_sodos_MER', 'cont_sodos_XAL', 'cont_sodos_LLA', 'cont_sodos_TLI',
            'cont_sodos_UAX', 'cont_sodos_BJU', 'cont_sodos_MPA', 'cont_sodos_MON', 'cont_sodos_NEZ',
            'cont_sodos_INN', 'cont_sodos_AJM', 'cont_sodos_VIF',
            
            # PM10
            'cont_pmdiez_UIZ', 'cont_pmdiez_ATI', 'cont_pmdiez_CUA', 'cont_pmdiez_SFE', 'cont_pmdiez_SAG',
            'cont_pmdiez_CUT', 'cont_pmdiez_PED', 'cont_pmdiez_TAH', 'cont_pmdiez_GAM', 'cont_pmdiez_IZT',
            'cont_pmdiez_CCA', 'cont_pmdiez_HGM', 'cont_pmdiez_LPR', 'cont_pmdiez_MGH', 'cont_pmdiez_CAM',
            'cont_pmdiez_FAC', 'cont_pmdiez_TLA', 'cont_pmdiez_MER', 'cont_pmdiez_XAL', 'cont_pmdiez_TLI',
            'cont_pmdiez_UAX', 'cont_pmdiez_BJU', 'cont_pmdiez_MPA', 'cont_pmdiez_MON', 'cont_pmdiez_NEZ',
            'cont_pmdiez_INN', 'cont_pmdiez_AJM', 'cont_pmdiez_VIF',
            
            # PM CO
            'cont_pmco_UIZ', 'cont_pmco_SFE', 'cont_pmco_SAG', 'cont_pmco_PED', 'cont_pmco_GAM',
            'cont_pmco_HGM', 'cont_pmco_MGH', 'cont_pmco_CAM', 'cont_pmco_TLA', 'cont_pmco_MER',
            'cont_pmco_XAL', 'cont_pmco_BJU', 'cont_pmco_MPA', 'cont_pmco_INN', 'cont_pmco_AJM'
        ]
        return columns
    
    def _get_pollutant_types_from_columns(self, columns: List[str]) -> List[str]:
        """Extrae tipos de contaminantes únicos de las columnas."""
        types = set()
        for col in columns:
            if col.startswith('cont_'):
                parts = col.split('_')
                if len(parts) >= 2:
                    types.add(f"cont_{parts[1]}")
        return sorted(list(types))
    
    def _query_pollutant_data(self, pollutant: str, start_date: str, end_date: str) -> List[Tuple]:
        """Consulta datos de un contaminante específico."""
        query = f"""
        SELECT fecha, val, id_est 
        FROM {pollutant}
        WHERE fecha BETWEEN '{start_date}' AND '{end_date}'
        AND id_est IN ('{"','".join(self.all_stations)}')
        ORDER BY fecha, id_est;
        """
        return self.db.execute_query(query)


class DataAligner:
    """Alineador de datos de contaminación y meteorología."""
    
    @staticmethod
    def align_weather_pollution_utc6(pollution_data: pd.DataFrame, 
                                   weather_data: xr.Dataset,
                                   target_datetime: str,
                                   utc_offset: int = 0) -> Tuple[pd.DataFrame, xr.Dataset, int]:
        """
        Alinea datos de contaminación y meteorología en UTC-6.
        
        Args:
            pollution_data: Datos de contaminación
            weather_data: Datos meteorológicos
            target_datetime: Fecha objetivo
            utc_offset: Offset UTC (default 0, ya aplicado)
            
        Returns:
            Tupla (pollution_aligned, weather_aligned, target_index)
        """
        target_dt = pd.to_datetime(target_datetime)
        print(f"🎯 Target datetime (UTC-6): {target_dt}")
        
        # 1. Convertir weather a UTC-6
        print("🌤️ Convirtiendo weather de UTC a UTC-6...")
        weather_aligned = weather_data.copy()
        weather_aligned['time'] = (pd.to_datetime(weather_data.time.values) + 
                                 pd.Timedelta(hours=utc_offset))
        
        # 2. Filtrar y alinear contaminación
        print("🏭 Alineando datos de contaminación...")
        pollution_filtered = pollution_data[pollution_data.index <= target_dt]
        complete_time_index = pd.date_range(
            start=weather_aligned.time.min().item(),
            end=weather_aligned.time.max().item(),
            freq='H'
        )
        
        pollution_aligned = pd.DataFrame(
            index=complete_time_index, 
            columns=pollution_filtered.columns
        )
        pollution_aligned.loc[pollution_filtered.index, 
                            pollution_filtered.columns] = pollution_filtered
        pollution_aligned.fillna(-1, inplace=True)
        
        # 3. Encontrar índice objetivo
        target_index = pollution_aligned.index.get_loc(target_dt)
        
        print(f"✅ Alineación completada")
        print(f"   📊 Pollution shape: {pollution_aligned.shape}")
        print(f"   🌤️ Weather dims: {weather_aligned.dims}")
        print(f"   📍 Target index: {target_index}")
        
        return pollution_aligned, weather_aligned, target_index


class ModelInference:
    """Motor de inferencia para modelos de pronóstico."""
    
    def __init__(self, model: torch.nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.model.eval()
    
    def prepare_input_tensors(self, pollution_aligned: pd.DataFrame,
                            weather_aligned: xr.Dataset,
                            target_index: int,
                            prev_pollutant_hours: int,
                            prev_weather_hours: int,
                            next_weather_hours: int,
                            auto_regressive_steps: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepara tensores de entrada para el modelo.
        
        Args:
            pollution_aligned: Datos de contaminación alineados
            weather_aligned: Datos meteorológicos alineados
            target_index: Índice del momento objetivo
            prev_pollutant_hours: Horas previas de contaminantes
            prev_weather_hours: Horas previas de meteorología
            next_weather_hours: Horas futuras de meteorología
            auto_regressive_steps: Pasos autorregresivos
            
        Returns:
            Tupla (x_pollution, x_weather)
        """
        print("🔧 Preparando tensores de entrada...")
        
        # Validar que hay suficientes datos históricos
        required_start_index = target_index - prev_pollutant_hours + 1
        if required_start_index < 0:
            raise ValueError(
                f"No hay suficientes datos históricos. "
                f"Se requieren {prev_pollutant_hours} horas previas, "
                f"pero solo hay {target_index + 1} horas disponibles. "
                f"Fecha objetivo: {pollution_aligned.index[target_index]}"
            )
        
        # Validar que los índices requeridos existen
        if target_index >= len(pollution_aligned):
            raise ValueError(
                f"target_index ({target_index}) fuera del rango del DataFrame "
                f"(tamaño: {len(pollution_aligned)})"
            )
        
        print(f"   📊 Validación exitosa: target_index={target_index}, "
              f"rango=[{required_start_index}:{target_index + 1}]")
        
        # Ventana de contaminación
        pollution_window = pollution_aligned.iloc[required_start_index:target_index + 1]
        
        # Verificar que la ventana tiene el tamaño correcto
        if len(pollution_window) != prev_pollutant_hours:
            raise ValueError(
                f"Ventana de contaminación tiene tamaño incorrecto. "
                f"Esperado: {prev_pollutant_hours}, obtenido: {len(pollution_window)}"
            )
        
        # Ventana de meteorología
        weather_start_index = target_index - prev_weather_hours
        weather_end_index = target_index + next_weather_hours + auto_regressive_steps
        
        # Validar índices meteorológicos
        if weather_start_index < 0:
            raise ValueError(
                f"No hay suficientes datos meteorológicos históricos. "
                f"Se requieren {prev_weather_hours} horas previas, "
                f"pero solo hay {target_index + 1} horas disponibles."
            )
        
        if weather_end_index > len(weather_aligned.time):
            raise ValueError(
                f"No hay suficientes datos meteorológicos futuros. "
                f"Se requieren hasta el índice {weather_end_index}, "
                f"pero solo hay {len(weather_aligned.time)} timesteps disponibles."
            )
        
        weather_window = weather_aligned.isel(
            time=slice(weather_start_index, weather_end_index)
        )
        
        print(f"   🌤️ Ventana meteorológica: [{weather_start_index}:{weather_end_index}] "
              f"= {len(weather_window.time)} timesteps")
        
        # Convertir a tensores
        x_pollution = torch.tensor(
            pollution_window.values, 
            dtype=torch.float32
        ).unsqueeze(0).to(self.device)
        
        # Procesar datos meteorológicos
        weather_vars = list(weather_window.data_vars)
        weather_arrays = [weather_window[var].values.astype(np.float32) 
                         for var in weather_vars]
        weather_array = np.array(weather_arrays)  # [vars, time, lat, lon]
        weather_array = weather_array.swapaxes(1, 0)  # [time, vars, lat, lon]
        weather_array = np.nan_to_num(weather_array)  # Limpiar NaN
        
        x_weather = torch.tensor(
            weather_array, 
            dtype=torch.float32
        ).unsqueeze(0).to(self.device)
        
        print(f"✅ Tensores preparados:")
        print(f"   📊 Pollution: {x_pollution.shape}")
        print(f"   🌤️ Weather: {x_weather.shape}")
        
        return x_pollution, x_weather
    
    def run_autoregressive_inference(self, x_pollution: torch.Tensor,
                                   x_weather: torch.Tensor,
                                   target_datetime: str,
                                   auto_regressive_steps: int,
                                   weather_window_size: int) -> List[Dict]:
        """
        Ejecuta inferencia autorregresiva.
        
        Args:
            x_pollution: Tensor de contaminación
            x_weather: Tensor meteorológico
            target_datetime: Fecha objetivo
            auto_regressive_steps: Número de pasos autorregresivos
            weather_window_size: Tamaño de ventana meteorológica
            
        Returns:
            Lista de predicciones con metadatos
        """
        print(f"🚀 Iniciando inferencia autorregresiva para {auto_regressive_steps} pasos")
        
        predictions = []
        current_pollution = x_pollution.clone()
        target_dt = pd.to_datetime(target_datetime)
        
        for step in range(auto_regressive_steps):
            # Datetime de predicción
            pred_datetime = target_dt + pd.Timedelta(hours=step + 1)
            
            # Ventana meteorológica para este paso
            weather_start = step
            weather_end = step + weather_window_size
            current_weather = x_weather[:, weather_start:weather_end, :, :, :]
            
            print(f"   🔄 Paso {step + 1}/{auto_regressive_steps} - {pred_datetime}")
            
            # Predicción
            with torch.no_grad():
                output = self.model(current_weather, current_pollution)
            
            # Guardar predicción
            predictions.append({
                'step': step + 1,
                'datetime': pred_datetime,
                'prediction': output.cpu().numpy()
            })
            
            # Actualizar contaminación para siguiente paso
            if step < auto_regressive_steps - 1:
                new_pollution = torch.zeros_like(current_pollution)
                new_pollution[:, :-1, :] = current_pollution[:, 1:, :].clone()
                new_pollution[:, -1, :output.shape[1]] = output
                current_pollution = new_pollution
        
        print(f"✅ Inferencia completada: {len(predictions)} predicciones")
        return predictions


class ResultsProcessor:
    """Procesador de resultados y visualizaciones."""
    
    def __init__(self, norm_params_file: str):
        with open(norm_params_file, 'r') as f:
            self.norm_params = yaml.safe_load(f)
    
    def denormalize_predictions(self, predictions: List[Dict],
                              output_columns: List[str]) -> pd.DataFrame:
        """
        Desnormaliza predicciones usando parámetros guardados.
        
        Args:
            predictions: Lista de predicciones del modelo
            output_columns: Nombres de columnas de salida
            
        Returns:
            DataFrame con predicciones desnormalizadas
        """
        print("🔄 Desnormalizando predicciones...")
        
        # Extraer datos
        datetimes = [pred['datetime'] for pred in predictions]
        prediction_values = np.array([pred['prediction'].squeeze() for pred in predictions])
        
        # Ajustar dimensiones si es necesario
        num_cols_to_use = min(prediction_values.shape[1], len(output_columns))
        output_columns_adj = output_columns[:num_cols_to_use]
        prediction_values_adj = prediction_values[:, :num_cols_to_use]
        
        # Crear DataFrame normalizado
        predictions_normalized = pd.DataFrame(
            prediction_values_adj,
            columns=output_columns_adj,
            index=datetimes
        )
        
        # Desnormalizar
        try:
            predictions_denormalized = denormalize_data(
                self.norm_params,
                predictions_normalized,
                data_type='pollutants'
            )
            print("✅ Desnormalización automática exitosa")
        except Exception as e:
            print(f"⚠️ Desnormalización automática falló: {e}")
            print("🔧 Usando desnormalización manual...")
            predictions_denormalized = self._manual_denormalization(
                predictions_normalized, output_columns_adj
            )
        
        return predictions_denormalized
    
    def _manual_denormalization(self, df_norm: pd.DataFrame, 
                              columns: List[str]) -> pd.DataFrame:
        """Desnormalización manual por tipo de contaminante."""
        df_denorm = df_norm.copy()
        
        for col in columns:
            if col.startswith('cont_'):
                parts = col.split('_')
                if len(parts) >= 2:
                    pollutant_type = parts[1]
                    
                    if pollutant_type in self.norm_params.get('pollutants', {}):
                        mean_val = self.norm_params['pollutants'][pollutant_type]['mean']
                        std_val = self.norm_params['pollutants'][pollutant_type]['std']
                        
                        df_denorm[col] = df_norm[col] * std_val + mean_val
                        print(f"   ✅ {col}: μ={mean_val:.3f}, σ={std_val:.3f}")
        
        return df_denorm
    
    def create_summary_plots(self, predictions_denormalized: pd.DataFrame,
                           target_datetime: str, output_folder: str = "./output/"):
        """
        Crea plots de resumen de las predicciones.
        
        Args:
            predictions_denormalized: DataFrame con predicciones desnormalizadas
            target_datetime: Fecha objetivo inicial
            output_folder: Carpeta para guardar plots
        """
        print("📊 Generando plots de resumen...")
        
        os.makedirs(output_folder, exist_ok=True)
        target_dt = pd.to_datetime(target_datetime)
        datetimes = predictions_denormalized.index
        
        # Plot 1: Primeras 9 columnas
        num_cols = min(9, len(predictions_denormalized.columns))
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle(f'Predicciones de Contaminación - Top {num_cols} Variables\n'
                    f'{len(predictions_denormalized)} horas desde {target_datetime}',
                    fontsize=16, fontweight='bold')
        axes = axes.flatten()
        
        for i in range(num_cols):
            ax = axes[i]
            col_name = predictions_denormalized.columns[i]
            values = predictions_denormalized.iloc[:, i]
            
            ax.plot(datetimes, values, 'o-', linewidth=2, markersize=3,
                   color='green', alpha=0.8)
            ax.axvline(x=target_dt, color='blue', linestyle='--', alpha=0.7)
            
            ax.grid(True, alpha=0.3)
            ax.set_ylabel('Concentración', fontsize=8)
            ax.set_title(col_name, fontsize=9, fontweight='bold')
            
            # Formato de fechas
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=8))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, fontsize=6)
            
            # Estadísticas
            stats_text = f'μ={values.mean():.1f}\nσ={values.std():.1f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   fontsize=7, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Ocultar subplots vacíos
        for i in range(num_cols, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plot_path = os.path.join(output_folder, f'predictions_summary_{target_dt.strftime("%Y%m%d_%H%M")}.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Plot guardado: {plot_path}")
        
        return plot_path


class ForecastSystem:
    """Sistema integrado de pronóstico de contaminación."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.db_manager = DatabaseManager()
        self.results_processor = None
        self.wrf_processor = WRFProcessor()
        
    def setup(self, wrf_folder: str, model: torch.nn.Module, device: torch.device):
        """Configura el sistema de pronóstico."""
        self.wrf_loader = WRFDataLoader(wrf_folder)
        self.pollution_manager = PollutionDataManager(self.db_manager)
        self.model_inference = ModelInference(model, device)
        
        norm_params_file = self.config['data_loader']['args']['norm_params_file']
        self.results_processor = ResultsProcessor(norm_params_file)
        
    def run_forecast(self, target_datetime: str, config_file_path: str, 
                     output_folder: str = "./output/") -> pd.DataFrame:
        """
        Ejecuta pronóstico completo.
        
        Args:
            target_datetime: Fecha objetivo en formato 'YYYY-MM-DD HH:MM:SS'
            config_file_path: Ruta al archivo de configuración JSON
            output_folder: Carpeta de salida
            
        Returns:
            DataFrame con predicciones desnormalizadas
        """
        try:
            print("🚀 INICIANDO PRONÓSTICO COMPLETO")
            print("=" * 60)
            
            # 1. Procesar archivos WRF (ejecutar operativo001.py)
            print("\n1️⃣ PROCESANDO ARCHIVOS WRF")
            wrf_success = self.wrf_processor.process_wrf_files(
                target_datetime, config_file_path, verbose=True
            )
            
            if not wrf_success:
                raise Exception("Error procesando archivos WRF con operativo001.py")
            
            # 2. Cargar datos meteorológicos
            print("\n2️⃣ CARGANDO DATOS METEOROLÓGICOS")
            weather_data = self.wrf_loader.load_data(use_all_nc=True)
            
            # 3. Obtener datos de contaminación
            print("\n3️⃣ OBTENIENDO DATOS DE CONTAMINACIÓN")
            min_required_hours = self.config['data_loader']['args']['prev_pollutant_hours']
            pollution_data, imputed_data = self.pollution_manager.get_contaminant_data(
                target_datetime, 
                hours_back=30,
                min_required_hours=min_required_hours
            )
            
            # 4. Normalizar datos
            print("\n4️⃣ NORMALIZANDO DATOS")
            norm_params_file = self.config['data_loader']['args']['norm_params_file']
            pollution_normalized, weather_normalized = normalize_data(
                norm_params_file, pollution_data, weather_data
            )
            
            # 5. Procesar datos
            print("\n5️⃣ PROCESANDO DATOS")
            pollution_processed = self._process_pollution_data(pollution_normalized)
            
            # 6. Alinear datos
            print("\n6️⃣ ALINEANDO DATOS TEMPORALMENTE")
            pollution_aligned, weather_aligned, target_index = DataAligner.align_weather_pollution_utc6(
                pollution_processed, weather_normalized, target_datetime
            )
            
            # 7. Preparar tensores
            print("\n7️⃣ PREPARANDO TENSORES DE ENTRADA")
            x_pollution, x_weather = self.model_inference.prepare_input_tensors(
                pollution_aligned, weather_aligned, target_index,
                self.config['data_loader']['args']['prev_pollutant_hours'],
                self.config['data_loader']['args']['prev_weather_hours'],
                self.config['data_loader']['args']['next_weather_hours'],
                self.config['test']['data_loader']['auto_regresive_steps']
            )
            
            # 8. Ejecutar inferencia
            print("\n8️⃣ EJECUTANDO INFERENCIA")
            predictions = self.model_inference.run_autoregressive_inference(
                x_pollution, x_weather, target_datetime,
                self.config['test']['data_loader']['auto_regresive_steps'],
                (self.config['data_loader']['args']['prev_weather_hours'] +
                 self.config['data_loader']['args']['next_weather_hours'] + 1)
            )
            
            # 9. Desnormalizar resultados
            print("\n9️⃣ DESNORMALIZANDO RESULTADOS")
            # Aquí necesitarías cargar las columnas objetivo desde el archivo YAML
            column_file_path = os.path.join(os.path.dirname(__file__), f"column_names_{self.config['name']}.yml")
            with open(column_file_path, 'r') as f:
                column_names_dict = yaml.unsafe_load(f)
            output_columns = column_names_dict['pollution_columns']
            
            predictions_denormalized = self.results_processor.denormalize_predictions(
                predictions, output_columns
            )
            
            # 🔟 Generar visualizaciones
            print("\n🔟 GENERANDO VISUALIZACIONES")
            self.results_processor.create_summary_plots(
                predictions_denormalized, target_datetime, output_folder
            )
            
            print("\n✅ PRONÓSTICO COMPLETADO EXITOSAMENTE")
            print("=" * 60)
            
            return predictions_denormalized
            
        except Exception as e:
            print(f"\n❌ ERROR EN PRONÓSTICO: {e}")
            import traceback
            traceback.print_exc()
            raise
        finally:
            self.db_manager.disconnect()
    
    def _process_pollution_data(self, pollution_data: pd.DataFrame) -> pd.DataFrame:
        """Procesa datos de contaminación (agregaciones, limpieza, etc.)."""
        print("🔧 Procesando datos de contaminación...")
        
        # Reemplazar NaN con 0
        pollution_data.fillna(0, inplace=True)
        
        # Obtener configuración de contaminantes a mantener
        pollutants_to_keep = self.config['data_loader']['args']['pollutants_to_keep']
        
        # Filtrar columnas de contaminantes
        pollutant_columns = [col for col in pollution_data.columns if col.startswith('cont_')]
        
        # Calcular estadísticas por contaminante (excepto otres)
        pollutant_stats = {}
        otres_columns = []
        
        for pollutant in pollutants_to_keep:
            pollutant_cols = [col for col in pollutant_columns if f'cont_{pollutant}_' in col]
            
            if pollutant == 'otres':
                otres_columns = pollutant_cols
            else:
                # Calcular mean, min, max para otros contaminantes
                if pollutant_cols:
                    pollutant_mean = pollution_data[pollutant_cols].mean(axis=1)
                    pollutant_min = pollution_data[pollutant_cols].min(axis=1)
                    pollutant_max = pollution_data[pollutant_cols].max(axis=1)
                    
                    pollutant_stats[f'cont_{pollutant}_mean'] = pollutant_mean
                    pollutant_stats[f'cont_{pollutant}_min'] = pollutant_min
                    pollutant_stats[f'cont_{pollutant}_max'] = pollutant_max
        
        # Crear DataFrame con estadísticas
        if pollutant_stats:
            stats_df = pd.DataFrame(pollutant_stats, index=pollution_data.index)
            
            # Eliminar columnas originales (excepto otres)
            cols_to_drop = [col for col in pollutant_columns if col not in otres_columns]
            pollution_processed = pollution_data.drop(columns=cols_to_drop)
            
            # Agregar estadísticas
            pollution_processed = pd.concat([pollution_processed, stats_df], axis=1)
        else:
            pollution_processed = pollution_data
        
        # Eliminar columnas de imputación si existen
        imputed_cols = [col for col in pollution_processed.columns if col.startswith('i_cont_')]
        if imputed_cols:
            pollution_processed = pollution_processed.drop(columns=imputed_cols)
        
        print(f"   ✅ Datos procesados: {pollution_processed.shape}")
        return pollution_processed 