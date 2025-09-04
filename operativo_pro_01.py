#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
operativo_pro_01.py - Sistema de Inferencia en Tiempo Real para Pronóstico de Contaminación

Script modularizado para realizar pronósticos de contaminación del aire
usando modelos de deep learning y datos meteorológicos WRF.

VERSIÓN: 2.0
FECHA: 2024
"""

import argparse
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path
import yaml
import torch

# Imports del proyecto
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from utils import read_json

# Import del módulo de utilidades
from operativo_files.forecast_utils2 import ForecastSystem, DatabaseManager, WRFDataLoader, ModelInference, ResultsProcessor, WRFProcessor, ForecastBatchProcessor

# Imports adicionales para contingencia SQLite
import sqlite3
import pandas as pd
import numpy as np
from typing import List, Tuple

# Import para variables de tiempo
from proj_io.inout import generateDateColumns


class SQLiteDatabaseManager:
    """Gestor de base de datos SQLite para contingencia."""
    
    def __init__(self, db_path: str = '/home/pedro/git2/gitflow2/hack_sqlite/ensamble_ai_pollution_forecast/contingencia_sqlite_bd.db'):
        self.db_path = db_path
        self.connection = None
        
        # Mapeo de contaminantes compatible con el sistema existente
        self.pollutant_mapping = {
            'cont_otres': 'O3',
            'cont_co': 'CO',
            'cont_nodos': 'NO2',
            'cont_pmdiez': 'PM10',
            'cont_pmdoscinco': 'PM2.5',
            'cont_nox': 'NOx',
            'cont_no': 'NO',
            'cont_sodos': 'SO2',
            'cont_pmco': 'PMco'
        }
    
    def connect(self):
        """Establece conexión con la base de datos SQLite."""
        try:
            print(f"🔌 Conectando a SQLite: {self.db_path}")
            if not os.path.exists(self.db_path):
                raise FileNotFoundError(f"Base de datos SQLite no encontrada: {self.db_path}")
            
            self.connection = sqlite3.connect(self.db_path)
            print("✅ Conectado a SQLite de contingencia")
            return self.connection
            
        except Exception as e:
            print(f"❌ Error conectando a SQLite: {e}")
            print("💡 Asegúrate de que la base SQLite esté disponible en la ruta especificada")
            raise
    
    def disconnect(self):
        """Cierra la conexión a la base de datos."""
        if self.connection:
            self.connection.close()
            self.connection = None
            print("🔌 Conexión SQLite cerrada")
    
    def execute_query(self, query: str) -> List[Tuple]:
        """Ejecuta una consulta SQL y retorna los resultados."""
        if not self.connection:
            self.connect()
            
        cursor = self.connection.cursor()
        try:
            cursor.execute(query)
            rows = cursor.fetchall()
            return rows
        finally:
            cursor.close()
    
    def get_dataframe_from_query(self, query: str, columns: List[str]) -> pd.DataFrame:
        """Ejecuta consulta y retorna DataFrame."""
        data = self.execute_query(query)
        return pd.DataFrame(data, columns=columns)


class SQLitePollutionDataManager:
    """Gestor de datos de contaminación usando SQLite."""
        
    def __init__(self, db_manager: SQLiteDatabaseManager):
        self.db = db_manager
        
        # Configuración de estaciones - igual que el original
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
        Obtiene datos de contaminantes desde SQLite y genera exactamente 66 columnas.
        30 ozono individuales + 24 estadísticas + 12 variables tiempo = 66 total
        """
        print(f"📊 OBTENIENDO DATOS DESDE SQLITE (MODO 66 COLUMNAS)")
        print(f"🎯 Fecha objetivo: {target_datetime}")
        print(f"⏪ Horas hacia atrás: {hours_back}")
        
        target_dt = pd.to_datetime(target_datetime)
        
        # Asegurar suficientes horas históricas
        actual_hours_back = max(hours_back, min_required_hours + 5)
        start_dt = target_dt - pd.Timedelta(hours=actual_hours_back)
        
        start_date = start_dt.strftime('%Y-%m-%d %H:%M:%S')
        end_date = target_dt.strftime('%Y-%m-%d %H:%M:%S')  
        
        print(f"📅 Consultando SQLite desde {start_date} hasta {end_date}")
        print(f"   🕐 Horas solicitadas: {hours_back}, horas garantizadas: {actual_hours_back}")
        
        # Crear índice temporal
        date_range = pd.date_range(start=start_dt, end=target_dt, freq='H')
        
        if expected_columns is None:
            expected_columns = self._generate_expected_columns()
        
        # Inicializar DataFrame con 54 columnas de contaminantes
        final_df = pd.DataFrame(index=date_range, columns=expected_columns).astype(float)
        final_df.fillna(np.nan, inplace=True)
        
        imputed_columns = [f'i_{col}' for col in expected_columns]
        df_imputed = pd.DataFrame(index=date_range, columns=imputed_columns).fillna(0)
        
        # 1. OBTENER DATOS INDIVIDUALES DE OZONO (30 columnas)
        print(f"   🔍 Consultando cont_otres (30 estaciones individuales)...")
        ozono_data = self._query_pollutant_data_sqlite('cont_otres', start_date, end_date)
        
        if ozono_data:
            for fecha, val, id_est in ozono_data:
                col_name = f"cont_otres_{id_est}"
                if col_name in expected_columns:
                    try:
                        timestamp = pd.to_datetime(fecha)
                        if timestamp in final_df.index:
                            final_df.loc[timestamp, col_name] = float(val) if val is not None else np.nan
                    except (ValueError, TypeError):
                        continue
            print(f"   ✅ cont_otres: {len(ozono_data)} registros procesados")
        
        # 2. OBTENER Y CALCULAR ESTADÍSTICAS PARA OTROS CONTAMINANTES (24 columnas)
        stats_pollutants = ['cont_co', 'cont_nodos', 'cont_pmdiez', 'cont_pmdoscinco', 
                           'cont_nox', 'cont_no', 'cont_sodos', 'cont_pmco']
        
        for pollutant in stats_pollutants:
            try:
                print(f"   🔍 Consultando {pollutant} para estadísticas...")
                data_rows = self._query_pollutant_data_sqlite(pollutant, start_date, end_date)
                
                if data_rows:
                    # Crear DataFrame temporal para este contaminante
                    temp_df = pd.DataFrame(data_rows, columns=['fecha', 'val', 'id_est'])
                    temp_df['fecha'] = pd.to_datetime(temp_df['fecha'])
                    temp_df['val'] = pd.to_numeric(temp_df['val'], errors='coerce')
                    
                    # Calcular estadísticas por timestamp
                    stats_by_time = temp_df.groupby('fecha')['val'].agg(['mean', 'min', 'max']).reset_index()
                    
                    # Llenar estadísticas en el DataFrame final
                    for _, row in stats_by_time.iterrows():
                        timestamp = row['fecha']
                        if timestamp in final_df.index:
                            final_df.loc[timestamp, f'{pollutant}_mean'] = row['mean']
                            final_df.loc[timestamp, f'{pollutant}_min'] = row['min']
                            final_df.loc[timestamp, f'{pollutant}_max'] = row['max']
                    
                    print(f"   ✅ {pollutant}: {len(data_rows)} registros → estadísticas calculadas")
                else:
                    print(f"   ⚠️ {pollutant}: Sin datos")
                    
            except Exception as e:
                print(f"   ❌ Error en {pollutant}: {e}")
                continue
        
        # 3. AGREGAR VARIABLES DE TIEMPO (12 columnas)
        print("🕐 Generando variables cíclicas de tiempo...")
        time_cols, time_values = generateDateColumns(date_range, flip_order=True)
        time_df = pd.DataFrame(dict(zip(time_cols, time_values)), index=date_range)
        
        # 4. COMBINAR TODO: 54 contaminantes + 12 tiempo = 66 total
        combined_df = pd.concat([final_df, time_df], axis=1)
        
        # Calcular estadísticas de completitud solo para contaminantes
        total_expected = len(final_df) * len(expected_columns)
        available_data = final_df.count().sum()
        completeness = (available_data / total_expected) * 100 if total_expected > 0 else 0
        
        print(f"📈 Completitud contaminantes: {completeness:.1f}% ({available_data:,}/{total_expected:,})")
        print(f"📊 DataFrame final: {combined_df.shape[1]} columnas (54 contaminantes + 12 tiempo = 66 total)")
        
        return combined_df, df_imputed
    
    def _query_pollutant_data_sqlite(self, pollutant: str, start_date: str, end_date: str) -> List[Tuple]:
        """Consulta datos de SQLite para un contaminante específico."""
        query = f"""
        SELECT fecha, val, id_est 
        FROM {pollutant}
        WHERE fecha BETWEEN '{start_date}' AND '{end_date}'
        AND val IS NOT NULL
        AND id_est IN ({','.join([f"'{station}'" for station in self.all_stations])})
        ORDER BY fecha, id_est
        """
        return self.db.execute_query(query)
    
    def _generate_expected_columns(self) -> List[str]:
        """Genera las 54 columnas exactas que espera el modelo (30 ozono + 24 estadísticas)."""
        columns = [
            # 30 columnas individuales de OZONO
            'cont_otres_UIZ', 'cont_otres_AJU', 'cont_otres_ATI', 'cont_otres_CUA', 'cont_otres_SFE',
            'cont_otres_SAG', 'cont_otres_CUT', 'cont_otres_PED', 'cont_otres_TAH', 'cont_otres_GAM',
            'cont_otres_IZT', 'cont_otres_CCA', 'cont_otres_HGM', 'cont_otres_LPR', 'cont_otres_MGH',
            'cont_otres_CAM', 'cont_otres_FAC', 'cont_otres_TLA', 'cont_otres_MER', 'cont_otres_XAL',
            'cont_otres_LLA', 'cont_otres_TLI', 'cont_otres_UAX', 'cont_otres_BJU', 'cont_otres_MPA',
            'cont_otres_MON', 'cont_otres_NEZ', 'cont_otres_INN', 'cont_otres_AJM', 'cont_otres_VIF',
            
            # 24 columnas de ESTADÍSTICAS (mean, min, max) para 8 contaminantes
            'cont_co_mean', 'cont_co_min', 'cont_co_max',
            'cont_nodos_mean', 'cont_nodos_min', 'cont_nodos_max',
            'cont_pmdiez_mean', 'cont_pmdiez_min', 'cont_pmdiez_max',
            'cont_pmdoscinco_mean', 'cont_pmdoscinco_min', 'cont_pmdoscinco_max',
            'cont_nox_mean', 'cont_nox_min', 'cont_nox_max',
            'cont_no_mean', 'cont_no_min', 'cont_no_max',
            'cont_sodos_mean', 'cont_sodos_min', 'cont_sodos_max',
            'cont_pmco_mean', 'cont_pmco_min', 'cont_pmco_max'
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


class SQLiteForecastSystem(ForecastSystem):
    """Sistema de pronóstico usando SQLite como contingencia."""
    
    def __init__(self, config, sqlite_db_path: str = '/home/pedro/git2/gitflow2/hack_sqlite/ensamble_ai_pollution_forecast/contingencia_sqlite_bd.db'):
        # No llamar super().__init__ porque vamos a sobrescribir el db_manager
        self.config = config
        self.db_manager = SQLiteDatabaseManager(sqlite_db_path)
        self.results_processor = None
        self.wrf_processor = WRFProcessor()
        print("🔄 Sistema configurado para usar SQLite de contingencia")
    
    def setup(self, wrf_folder: str, model: torch.nn.Module, device: torch.device):
        """Configura el sistema usando SQLite en lugar de PostgreSQL."""
        self.wrf_loader = WRFDataLoader(wrf_folder)
        self.pollution_manager = SQLitePollutionDataManager(self.db_manager)  # Usar versión SQLite
        self.model_inference = ModelInference(model, device)
        
        norm_params_file = self.config['data_loader']['args']['norm_params_file']
        self.results_processor = ResultsProcessor(norm_params_file)
        print("✅ Sistema SQLite configurado completamente")


def log_to_file(message, filename="last_log.txt"):
    """
    Función simple para guardar un mensaje en un archivo de log
    
    Args:
        message: El mensaje a guardar
        filename: Nombre del archivo (por defecto 'last_log.txt')
    """
    import datetime
    
    # Obtener timestamp actual
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Crear mensaje con timestamp
    log_entry = f"[{timestamp}] {message}\n"
    
    # Escribir en el archivo
    with open(filename, 'a', encoding='utf-8') as f:
        f.write(log_entry)

# Ejemplo de uso:
# log_to_file(" �� Última hora: 2024-01-15 14:30:25")
class ForecastConfig:
    """Configuración centralizada del sistema de pronóstico."""
    
    # =============================================================================
    # CONTINGENCY FLAG: USAR SQLITE EN LUGAR DE POSTGRESQL
    # =============================================================================
    USE_SQLITE_CONTINGENCY = False  # Cambiar a True para usar SQLite de contingencia
    SQLITE_DB_PATH = '/home/pedro/git2/gitflow2/hack_sqlite/ensamble_ai_pollution_forecast/contingencia_sqlite_bd.db'
    
    # =============================================================================
    # EXPORTATION FLAGS: CONTROLAR GUARDADO DE PRONÓSTICOS
    # =============================================================================
    SAVE_TO_POSTGRES = True     # Guardar pronósticos en PostgreSQL (AMATE-OPERATIVO)
    SAVE_TO_SQLITE = False      # Guardar pronósticos en SQLite local (forecast_predictions.db)
    
    # =============================================================================
    # DEBUGGING FLAG: DESACTIVAR ARGPARSE
    # =============================================================================
    FORCE_DEFAULTS_FOR_DEBUG = False  # Cambiar a False para usar argparse normalmente
    
    # =============================================================================
    # VALORES POR DEFECTO (SIEMPRE COMO FALLBACK)
    # =============================================================================
    DEFAULT_TARGET_DATETIME = '2025-08-03 15:00:00'  # Fecha de agosto 2025 para pruebas
    DEFAULT_CONFIG_FILE = 'operativo_files/test_Parallel_all_prev24_heads4_w4_p4_ar8_bootstrapTrue_thresh2_weather4_2_0701_101128.json'
    DEFAULT_DEBUG_MODE = True
    DEFAULT_PLOTS_MODE = True
    DEFAULT_WRF_FOLDER = '/dev/shm/tem_ram_forecast/'
    DEFAULT_OUTPUT_FOLDER = './tem_var/'
    DEFAULT_VERBOSE = True
    DEFAULT_SAVE_INPUT_VECTORS = False
    
    def __init__(self):
        """Inicializa configuración con valores por defecto."""
        # Cambiar esta línea para usar siempre la hora anterior
        self.target_datetime = self.get_last_hour_datetime()  # ← CAMBIAR AQUÍ
        self.config_file_path = self.DEFAULT_CONFIG_FILE
        self.debug_mode = self.DEFAULT_DEBUG_MODE
        self.plots_mode = self.DEFAULT_PLOTS_MODE
        self.wrf_folder = self.DEFAULT_WRF_FOLDER
        self.output_folder = self.DEFAULT_OUTPUT_FOLDER
        self.verbose = self.DEFAULT_VERBOSE
        self.save_input_vectors = self.DEFAULT_SAVE_INPUT_VECTORS
    
    # def get_last_hour_datetime(self) -> str:
    #     """
    #     Calcula la hora anterior a la última hora en punto basada en la hora actual.
        
    #     Returns:
    #         str: Fecha y hora en formato 'YYYY-MM-DD HH:00:00'
    #     """
    #     now = datetime.now()
    #     # Redondear hacia abajo a la hora en punto y restar 1 hora
    #     last_hour = now.replace(minute=0, second=0, microsecond=0)
    #     previous_hour = last_hour + timedelta(hours=1)

    #     print(f" 🕐 Última hora: {last_hour.strftime('%Y-%m-%d %H:%M:%S')}")
    #     log_to_file(f" 🕐 Última hora: {last_hour.strftime('%Y-%m-%d %H:%M:%S')}")
    #     import time
    #     time.sleep(5)
    #     log_to_file(f" 🕐 Última hora: {last_hour.strftime('%Y-%m-%d %H:%M:%S')}")
    #     return previous_hour.strftime('%Y-%m-%d %H:%M:%S')

    def get_last_hour_datetime(self) -> str:
        """
        Obtiene la última hora disponible de ozono en la base de datos.
        Si no hay datos en la BD, fallback a la hora del sistema.
        
        Returns:
            str: Fecha y hora en formato 'YYYY-MM-DD HH:MM:%S'
        """
        try:
            # Importar y usar el módulo específico de PostgreSQL
            from postgres_query_helper import get_ozone_target_datetime
            return get_ozone_target_datetime()
            
        except ImportError as e:
            print(f" ❌ Error importando módulo PostgreSQL: {e}")
            log_to_file(f" ❌ Error importando módulo PostgreSQL: {e}")
        except Exception as e:
            print(f" ❌ Error consultando BD: {e}")
            log_to_file(f" ❌ Error consultando BD: {e}")
        
        # Fallback: usar hora del sistema (método anterior)
        print(" 🔄 Fallback: usando hora del sistema")
        log_to_file(" 🔄 Fallback: usando hora del sistema")
        
        now = datetime.now()
        last_hour = now.replace(minute=0, second=0, microsecond=0)
        previous_hour = last_hour - timedelta(hours=1)
        
        print(f" 🕐 Última hora (sistema): {previous_hour.strftime('%Y-%m-%d %H:%M:%S')}")
        log_to_file(f" 🕐 Última hora (sistema): {previous_hour.strftime('%Y-%m-%d %H:%M:%S')}")
        
        return previous_hour.strftime('%Y-%m-%d %H:%M:%S')


    def parse_arguments(self) -> bool:
        """
        Parsea argumentos de línea de comandos si FORCE_DEFAULTS_FOR_DEBUG está desactivado.
        
        Returns:
            True si se usaron argumentos de línea de comandos, False si se usaron defaults
        """
        if __name__ == "__main__" and not self.FORCE_DEFAULTS_FOR_DEBUG:
            print("🔧 USANDO ARGUMENTOS DE LÍNEA DE COMANDOS (FORCE_DEFAULTS_FOR_DEBUG = False)")
            try:
                args = self._create_argument_parser().parse_args()
                self._apply_parsed_arguments(args)
                print("✅ ARGUMENTOS PARSEADOS EXITOSAMENTE")
                return True
            except Exception as e:
                print(f"❌ ERROR EN ARGPARSE: {e}")
                print("🔄 FALLBACK: Usando valores por defecto")
                return False
        else:
            mode_msg = "🐛 DEBUGGING MODE" if __name__ == "__main__" else "📓 NOTEBOOK MODE"
            print(f"{mode_msg}: USANDO DEFAULTS (FORCE_DEFAULTS_FOR_DEBUG = True)")
            return False
    
    def _create_argument_parser(self) -> argparse.ArgumentParser:
        """Crea el parser de argumentos de línea de comandos."""
        parser = argparse.ArgumentParser(description='Sistema de Inferencia de Contaminación')
        
        # Argumentos requeridos - ahora opcional con valor por defecto automático
        parser.add_argument('--target-datetime', type=str, required=False,
                          help='Fecha objetivo (YYYY-MM-DD HH:MM:SS). Si no se proporciona, se usa la última hora en punto.')
        
        # Argumentos opcionales con detección automática
        parser.add_argument('--config-file', type=str, default=argparse.SUPPRESS,
                          help='Archivo de configuración JSON')
        parser.add_argument('--debug', action='store_true', default=argparse.SUPPRESS,
                          help='Modo debug')
        parser.add_argument('--no-debug', action='store_false', dest='debug', default=argparse.SUPPRESS,
                          help='Desactivar debug')
        parser.add_argument('--plots', action='store_true', default=argparse.SUPPRESS,
                          help='Generar gráficos')
        parser.add_argument('--no-plots', action='store_false', dest='plots', default=argparse.SUPPRESS,
                          help='No generar gráficos')
        parser.add_argument('--wrf-folder', type=str, default=argparse.SUPPRESS,
                          help='Carpeta WRF')
        parser.add_argument('--output-folder', type=str, default=argparse.SUPPRESS,
                          help='Carpeta de salida')
        parser.add_argument('--verbose', '-v', action='store_true', default=argparse.SUPPRESS,
                          help='Modo verbose')
        parser.add_argument('--save-input-vectors', action='store_true', default=argparse.SUPPRESS,
                          help='Guardar vectores de entrada en CSV para análisis')
        parser.add_argument('--no-save-input-vectors', action='store_false', dest='save_input_vectors', default=argparse.SUPPRESS,
                          help='No guardar vectores de entrada')
        
        return parser
    
    def _apply_parsed_arguments(self, args: argparse.Namespace):
        """Aplica argumentos parseados a la configuración."""
        # target_datetime - usar última hora si no se proporciona
        if hasattr(args, 'target_datetime') and args.target_datetime is not None:
            self.target_datetime = args.target_datetime
            print(f"   ✅ target_datetime: {self.target_datetime} (desde argparse)")
        else:
            self.target_datetime = self.get_last_hour_datetime()
            print(f"   🕐 target_datetime: {self.target_datetime} (última hora automática)")
        
        # Mapeo de argumentos opcionales
        arg_mappings = {
            'config_file': 'config_file_path',
            'wrf_folder': 'wrf_folder',
            'output_folder': 'output_folder',
            'debug': 'debug_mode',
            'plots': 'plots_mode',
            'verbose': 'verbose',
            'save_input_vectors': 'save_input_vectors'
        }
        
        parsed_args = vars(args)
        for arg_name, attr_name in arg_mappings.items():
            if arg_name in parsed_args:
                old_value = getattr(self, attr_name)
                setattr(self, attr_name, parsed_args[arg_name])
                print(f"   ✅ {arg_name}: {parsed_args[arg_name]} (desde argparse)")
            else:
                default_value = getattr(self, attr_name)
                print(f"   📋 {arg_name}: {default_value} (default)")
    
    def setup_output_directory(self):
        """Crea el directorio de salida si no existe."""
        os.makedirs(self.output_folder, exist_ok=True)
        if self.verbose:
            print(f"📁 Directorio de salida: {self.output_folder}")
    
    def print_final_configuration(self):
        """Imprime la configuración final a usar."""
        print("🔧 CONFIGURACIÓN FINAL:")
        print(f"   📅 Target: {self.target_datetime}")
        print(f"   📁 Config: {self.config_file_path}")
        print(f"   🐛 Debug: {self.debug_mode}")
        print(f"   📊 Plots: {self.plots_mode}")
        print(f"   💾 Output: {self.output_folder}")
        print(f"   🗣️ Verbose: {self.verbose}")
        print(f"   📋 Save Input Vectors: {self.save_input_vectors}")
        
        # Mostrar modo de base de datos
        if self.USE_SQLITE_CONTINGENCY:
            print(f"   🔄 Base de Datos: SQLite CONTINGENCIA ({self.SQLITE_DB_PATH})")
            print(f"   ⚠️  MODO: Usando datos de contingencia desde SQLite")
        else:
            print(f"   🔄 Base de Datos: PostgreSQL (producción)")
        
        # Mostrar configuración de exportación
        print(f"   💾 Exportación:")
        print(f"     🐘 PostgreSQL: {'✅ Activado' if self.SAVE_TO_POSTGRES else '❌ Desactivado'}")
        print(f"     🗄️ SQLite: {'✅ Activado' if self.SAVE_TO_SQLITE else '❌ Desactivado'}")
        
        print("-" * 50)


class ModelManager:
    """Gestor del modelo de deep learning."""
    
    def __init__(self, config_path: str, verbose: bool = True):
        self.config_path = config_path
        self.verbose = verbose
        self.config = None
        self.model = None
        self.device = None
        self.logger = None
        
    def load_configuration(self) -> ConfigParser:
        """Carga la configuración del modelo."""
        if self.verbose:
            print("🚀 CARGANDO CONFIGURACIÓN DEL MODELO...")
        
        raw_config = read_json(self.config_path)
        self.config = ConfigParser(raw_config)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.verbose:
            print(f"🔧 Dispositivo: {self.device}")
        
        self.logger = self.config.get_logger('inference')
        
        model_name = self.config['name']
        if self.verbose:
            print(f"🤖 Modelo: {model_name}")
        
        return self.config
    
    def build_and_load_model(self) -> torch.nn.Module:
        """Construye y carga el modelo preentrenado."""
        if self.verbose:
            print("🏗️ CONSTRUYENDO Y CARGANDO MODELO...")
        
        # Construir arquitectura
        self.model = self.config.init_obj('arch', module_arch)
        if self.verbose:
            print("📊 Arquitectura del modelo construida")
        
        # Cargar checkpoint
        model_path = os.path.join(
            self.config['test']['all_models_path'],
            self.config['test']['model_path'],
            'model_best.pth'
        )
        
        if self.verbose:
            print(f"📂 Cargando checkpoint desde: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        state_dict = checkpoint['state_dict']
        
        # Manejar prefijos de DataParallel
        has_module_prefix = any(key.startswith('module.') for key in state_dict.keys())
        if has_module_prefix:
            if self.verbose:
                print("🔧 Removiendo prefijo 'module.' del state dict")
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            state_dict = new_state_dict
        
        self.model.load_state_dict(state_dict)
        
        # Preparar para inferencia
        self.model = self.model.to(self.device)
        self.model.eval()
        
        if self.verbose:
            print("✅ Modelo cargado y listo para inferencia")
        
        return self.model
    
    def get_model_parameters(self) -> dict:
        """Obtiene parámetros del modelo necesarios para la inferencia."""
        params = {
            'prev_pollutant_hours': self.config['data_loader']['args']['prev_pollutant_hours'],
            'prev_weather_hours': self.config['data_loader']['args']['prev_weather_hours'],
            'next_weather_hours': self.config['data_loader']['args']['next_weather_hours'],
            'auto_regressive_steps': self.config['test']['data_loader']['auto_regresive_steps'],
            'norm_params_file': self.config['data_loader']['args']['norm_params_file']
        }
        
        params['weather_window_size'] = (params['prev_weather_hours'] + 
                                       params['next_weather_hours'] + 1)
        
        if self.verbose:
            print("📋 PARÁMETROS DEL MODELO:")
            for key, value in params.items():
                print(f"   {key}: {value}")
        
        return params


def main():
    """Función principal del sistema de pronóstico."""
    try:
        print("=" * 60)
        print("🌟 SISTEMA DE PRONÓSTICO DE CONTAMINACIÓN v2.0")
        print("=" * 60)
        
        # 1. Configuración
        print("\n1️⃣ CONFIGURACIÓN DEL SISTEMA")
        config = ForecastConfig()
        used_argparse = config.parse_arguments()
        config.setup_output_directory()
        config.print_final_configuration()
        
        # 2. Gestión del modelo
        print("\n2️⃣ GESTIÓN DEL MODELO")
        model_manager = ModelManager(config.config_file_path, config.verbose)
        model_config = model_manager.load_configuration()
        model = model_manager.build_and_load_model()
        model_params = model_manager.get_model_parameters()
        
        # 3. Configuración del sistema de pronóstico
        print("\n3️⃣ INICIALIZANDO SISTEMA DE PRONÓSTICO")
        
        # Verificar si usar SQLite o PostgreSQL
        if config.USE_SQLITE_CONTINGENCY:
            print("🔄 MODO CONTINGENCIA: Usando SQLite")
            forecast_system = SQLiteForecastSystem(model_config, config.SQLITE_DB_PATH)
        else:
            print("🔄 MODO NORMAL: Usando PostgreSQL")
        forecast_system = ForecastSystem(model_config)
        
        forecast_system.setup(config.wrf_folder, model, model_manager.device)
        
        # 4. Ejecución del pronóstico
        print("\n4️⃣ EJECUTANDO PRONÓSTICO")
        predictions_denormalized = forecast_system.run_forecast(
            config.target_datetime,
            config.config_file_path,
            config.output_folder,
            append_input_vectors=False,
            save_input_vectors=config.save_input_vectors
        )
        
        # 5. Resultados finales
        print("\n5️⃣ RESULTADOS FINALES")
        print(f"✅ Pronóstico completado exitosamente")
        print(f"📊 Predicciones generadas: {len(predictions_denormalized)} pasos temporales")
        print(f"📅 Período: {predictions_denormalized.index[0]} → {predictions_denormalized.index[-1]}")
        print(f"🗂️ Variables: {len(predictions_denormalized.columns)} contaminantes")
        print(f"💾 Resultados guardados en: {config.output_folder}")
        
        # Estadísticas básicas
        print(f"\n📈 ESTADÍSTICAS BÁSICAS:")
        print(f"   Promedio general: {predictions_denormalized.mean().mean():.2f}")
        print(f"   Desviación estándar: {predictions_denormalized.std().mean():.2f}")
        print(f"   Valor mínimo: {predictions_denormalized.min().min():.2f}")
        print(f"   Valor máximo: {predictions_denormalized.max().max():.2f}")
        
        # Guardar resultados en CSV
        output_csv = os.path.join(config.output_folder, 
                                f'predictions_{config.target_datetime.replace(" ", "_").replace(":", "")}.csv')
        predictions_denormalized.to_csv(output_csv)
        print(f"💾 CSV guardado: {output_csv}")
        
        # ===== GUARDADO EN BASES DE DATOS =====
        print("\n6️⃣ GUARDANDO EN BASES DE DATOS")
        
        # Variables para resumen final
        postgres_success = False
        sqlite_success = False
        postgres_result = {}
        sqlite_result = {}
        
        # 1. POSTGRESQL (Producción - ID Tipo Pronóstico = 7)
        if config.SAVE_TO_POSTGRES:
            try:
                from save_predictions_postgres import save_predictions_to_postgres
                
                print("🐘 EXPORTANDO A POSTGRESQL...")
                postgres_result = save_predictions_to_postgres(
                    predictions_denormalized,
                    config.target_datetime,
                    verbose=config.verbose
                )
                
                postgres_success = postgres_result.get('success', False)
                if postgres_success:
                    print(f"✅ PostgreSQL: {postgres_result['total_saved']} registros guardados")
                    print(f"   🌟 Ozono: {postgres_result['ozono_saved']}/30 estaciones")
                    print(f"   📊 Estadísticas: {postgres_result['stats_saved']}/24 combinaciones")
                else:
                    print("⚠️ PostgreSQL: No se pudieron guardar datos")
                    
            except ImportError:
                print("⚠️ Módulo PostgreSQL no disponible")
            except Exception as e:
                print(f"⚠️ Error en PostgreSQL: {e}")
        else:
            print("🐘 PostgreSQL: ❌ Desactivado en configuración")
        
        # 2. SQLITE LOCAL (Respaldo/Debug)
        if config.SAVE_TO_SQLITE:
            try:
                from save_predictions_sqlite import save_predictions_to_sqlite
                
                print("\n🗄️ EXPORTANDO A SQLITE...")
                sqlite_result = save_predictions_to_sqlite(
                    predictions_denormalized,
                    config.target_datetime,
                    db_path="forecast_predictions.db",
                    verbose=config.verbose
                )
                
                sqlite_success = sqlite_result.get('total_saved', 0) > 0
                if sqlite_success:
                    print(f"✅ SQLite: {sqlite_result['total_saved']} registros guardados")
                    print(f"   🌟 Ozono: {sqlite_result['ozono_saved']}/30 estaciones")
                    print(f"   📊 Estadísticas: {sqlite_result['stats_saved']}/24 combinaciones")
                else:
                    print("⚠️ SQLite: No se pudieron guardar datos")
                    
            except ImportError:
                print("⚠️ Módulo SQLite no disponible")
            except Exception as e:
                print(f"⚠️ Error en SQLite: {e}")
        else:
            print("🗄️ SQLite: ❌ Desactivado en configuración")
        
        # RESUMEN DE GUARDADO
        print(f"\n📈 RESUMEN DE EXPORTACIÓN:")
        if postgres_success or sqlite_success:
            if postgres_success:
                print(f"   🐘 PostgreSQL: ✅ Exitoso")
            if sqlite_success:
                print(f"   🗄️ SQLite: ✅ Exitoso")
            print("   💾 Al menos una base de datos guardó correctamente")
        else:
            print("   ⚠️ Ninguna base de datos pudo guardar los pronósticos")
            print("   📄 Solo se guardó archivo CSV")
        
        print("\n" + "=" * 60)
        print("🎉 PRONÓSTICO COMPLETADO EXITOSAMENTE")
        print("=" * 60)
        
        return predictions_denormalized
        
    except Exception as e:
        print(f"\n❌ ERROR CRÍTICO EN EL SISTEMA: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    

if __name__ == "__main__":
    # Ejecutar función principal
    results = main() 
