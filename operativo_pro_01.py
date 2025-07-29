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
from datetime import datetime
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


class ForecastConfig:
    """Configuración centralizada del sistema de pronóstico."""
    
    # =============================================================================
    # DEBUGGING FLAG: DESACTIVAR ARGPARSE
    # =============================================================================
    FORCE_DEFAULTS_FOR_DEBUG = False  # Cambiar a False para usar argparse normalmente
    
    # =============================================================================
    # VALORES POR DEFECTO (SIEMPRE COMO FALLBACK)
    # =============================================================================
    DEFAULT_TARGET_DATETIME = '2023-05-10 07:00:00'
    DEFAULT_CONFIG_FILE = 'operativo_files/test_Parallel_all_prev24_heads4_w4_p4_ar8_bootstrapTrue_thresh2_weather4_2_0701_101128.json'
    DEFAULT_DEBUG_MODE = True
    DEFAULT_PLOTS_MODE = True
    DEFAULT_WRF_FOLDER = '/dev/shm/tem_ram_forecast/'
    DEFAULT_OUTPUT_FOLDER = './tem_var/'
    DEFAULT_VERBOSE = True
    
    def __init__(self):
        """Inicializa configuración con valores por defecto."""
        self.target_datetime = self.DEFAULT_TARGET_DATETIME
        self.config_file_path = self.DEFAULT_CONFIG_FILE
        self.debug_mode = self.DEFAULT_DEBUG_MODE
        self.plots_mode = self.DEFAULT_PLOTS_MODE
        self.wrf_folder = self.DEFAULT_WRF_FOLDER
        self.output_folder = self.DEFAULT_OUTPUT_FOLDER
        self.verbose = self.DEFAULT_VERBOSE
        
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
        
        # Argumentos requeridos
        parser.add_argument('--target-datetime', type=str, required=True,
                          help='Fecha objetivo (YYYY-MM-DD HH:MM:SS)')
        
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
        
        return parser
    
    def _apply_parsed_arguments(self, args: argparse.Namespace):
        """Aplica argumentos parseados a la configuración."""
        # target_datetime es requerido
        self.target_datetime = args.target_datetime
        print(f"   ✅ target_datetime: {self.target_datetime} (desde argparse)")
        
        # Mapeo de argumentos opcionales
        arg_mappings = {
            'config_file': 'config_file_path',
            'wrf_folder': 'wrf_folder',
            'output_folder': 'output_folder',
            'debug': 'debug_mode',
            'plots': 'plots_mode',
            'verbose': 'verbose'
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
        forecast_system = ForecastSystem(model_config)
        forecast_system.setup(config.wrf_folder, model, model_manager.device)
        
        # 4. Ejecución del pronóstico
        print("\n4️⃣ EJECUTANDO PRONÓSTICO")
        predictions_denormalized = forecast_system.run_forecast(
            config.target_datetime,
            config.config_file_path,
            config.output_folder
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