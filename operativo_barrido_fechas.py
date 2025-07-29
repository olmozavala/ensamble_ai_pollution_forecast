#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
operativo_barrido_fechas.py - Script para Barrido de Fechas y Generación de CSV de Pronósticos

Script especializado para generar pronósticos para múltiples fechas y crear un CSV 
estructurado para evaluaciones posteriores.

FUNCIONALIDAD:
- Barrido de fechas con frecuencia configurable
- Generación de CSV con estructura: fecha + col00_h_plus_01 ... col53_h_plus_24
- Checkpoints para reanudar procesamiento
- Procesamiento secuencial (no paralelizable por restricciones WRF)
- Análisis de resultados

VERSIÓN: 1.0
"""

import argparse
import sys
import os
from datetime import datetime, timedelta
import pandas as pd
import torch

# Imports del proyecto
from parse_config import ConfigParser
from utils import read_json

# Import del módulo de utilidades profesional
from operativo_pro_01 import ForecastConfig, ModelManager
from forecast_utils2 import ForecastSystem, ForecastBatchProcessor


class BatchForecastConfig(ForecastConfig):
    """Configuración extendida para barrido de fechas."""
    
    # =============================================================================
    # VALORES ADICIONALES PARA BARRIDO DE FECHAS
    # =============================================================================
    DEFAULT_START_DATE = '2023-05-01 07:00:00'
    DEFAULT_END_DATE = '2023-05-07 07:00:00'
    DEFAULT_FREQUENCY = 'D'  # Diario
    DEFAULT_OUTPUT_CSV = 'forecast_batch_results.csv'
    DEFAULT_SAVE_PROGRESS = True
    DEFAULT_RESUME_CHECKPOINT = True
    
    def __init__(self):
        super().__init__()
        self.start_date = self.DEFAULT_START_DATE
        self.end_date = self.DEFAULT_END_DATE
        self.frequency = self.DEFAULT_FREQUENCY
        self.output_csv = self.DEFAULT_OUTPUT_CSV
        self.save_progress = self.DEFAULT_SAVE_PROGRESS
        self.resume_checkpoint = self.DEFAULT_RESUME_CHECKPOINT
    
    def _create_argument_parser(self) -> argparse.ArgumentParser:
        """Crea el parser de argumentos para barrido de fechas."""
        parser = argparse.ArgumentParser(
            description='Sistema de Barrido de Fechas para Pronósticos de Contaminación'
        )
        
        # Argumentos específicos del barrido
        parser.add_argument('--start-date', type=str, required=True,
                          help='Fecha inicial (YYYY-MM-DD HH:MM:SS)')
        parser.add_argument('--end-date', type=str, required=True,
                          help='Fecha final (YYYY-MM-DD HH:MM:SS)')
        parser.add_argument('--frequency', type=str, default=argparse.SUPPRESS,
                          help='Frecuencia del barrido (D=diario, H=horario)')
        parser.add_argument('--output-csv', type=str, default=argparse.SUPPRESS,
                          help='Archivo CSV de salida')
        parser.add_argument('--no-save-progress', action='store_false', dest='save_progress',
                          default=argparse.SUPPRESS, help='No guardar progreso incremental')
        parser.add_argument('--no-resume', action='store_false', dest='resume_checkpoint',
                          default=argparse.SUPPRESS, help='No reanudar desde checkpoint')
        
        # Argumentos heredados opcionales
        parser.add_argument('--config-file', type=str, default=argparse.SUPPRESS,
                          help='Archivo de configuración JSON')
        parser.add_argument('--wrf-folder', type=str, default=argparse.SUPPRESS,
                          help='Carpeta WRF')
        parser.add_argument('--output-folder', type=str, default=argparse.SUPPRESS,
                          help='Carpeta de salida temporal')
        parser.add_argument('--verbose', '-v', action='store_true', default=argparse.SUPPRESS,
                          help='Modo verbose')
        
        return parser
    
    def _apply_parsed_arguments(self, args: argparse.Namespace):
        """Aplica argumentos parseados a la configuración."""
        # Argumentos requeridos del barrido
        self.start_date = args.start_date
        self.end_date = args.end_date
        print(f"   ✅ start_date: {self.start_date} (desde argparse)")
        print(f"   ✅ end_date: {self.end_date} (desde argparse)")
        
        # Mapeo de argumentos opcionales del barrido
        batch_arg_mappings = {
            'frequency': 'frequency',
            'output_csv': 'output_csv',
            'save_progress': 'save_progress',
            'resume_checkpoint': 'resume_checkpoint'
        }
        
        # Mapeo de argumentos heredados
        inherited_arg_mappings = {
            'config_file': 'config_file_path',
            'wrf_folder': 'wrf_folder',
            'output_folder': 'output_folder',
            'verbose': 'verbose'
        }
        
        # Procesar argumentos del barrido
        parsed_args = vars(args)
        for arg_name, attr_name in batch_arg_mappings.items():
            if arg_name in parsed_args:
                setattr(self, attr_name, parsed_args[arg_name])
                print(f"   ✅ {arg_name}: {parsed_args[arg_name]} (desde argparse)")
            else:
                default_value = getattr(self, attr_name)
                print(f"   📋 {arg_name}: {default_value} (default)")
        
        # Procesar argumentos heredados
        for arg_name, attr_name in inherited_arg_mappings.items():
            if arg_name in parsed_args:
                setattr(self, attr_name, parsed_args[arg_name])
                print(f"   ✅ {arg_name}: {parsed_args[arg_name]} (desde argparse)")
            else:
                default_value = getattr(self, attr_name)
                print(f"   📋 {arg_name}: {default_value} (default)")
    
    def print_final_configuration(self):
        """Imprime la configuración final para barrido."""
        print("🔧 CONFIGURACIÓN FINAL DEL BARRIDO:")
        print(f"   📅 Fecha inicial: {self.start_date}")
        print(f"   📅 Fecha final: {self.end_date}")
        print(f"   ⏰ Frecuencia: {self.frequency}")
        print(f"   📄 CSV salida: {self.output_csv}")
        print(f"   💾 Guardar progreso: {self.save_progress}")
        print(f"   🔄 Reanudar: {self.resume_checkpoint}")
        print(f"   📁 Config: {self.config_file_path}")
        print(f"   🌤️ WRF folder: {self.wrf_folder}")
        print(f"   📂 Output folder: {self.output_folder}")
        print(f"   🗣️ Verbose: {self.verbose}")
        print("-" * 50)


def main():
    """Función principal para barrido de fechas."""
    try:
        print("=" * 60)
        print("📅 SISTEMA DE BARRIDO DE FECHAS PARA PRONÓSTICOS v1.0")
        print("=" * 60)
        
        # 1. Configuración específica del barrido
        print("\n1️⃣ CONFIGURACIÓN DEL BARRIDO")
        config = BatchForecastConfig()
        
        # Usar argparse siempre para barridos (no debugging mode)
        if len(sys.argv) > 1:
            try:
                args = config._create_argument_parser().parse_args()
                config._apply_parsed_arguments(args)
                print("✅ ARGUMENTOS PARSEADOS EXITOSAMENTE")
            except Exception as e:
                print(f"❌ ERROR EN ARGPARSE: {e}")
                sys.exit(1)
        else:
            print("❌ ERROR: Barrido de fechas requiere argumentos")
            print("💡 Uso: python operativo_barrido_fechas.py --start-date '2023-05-01 07:00:00' --end-date '2023-05-07 07:00:00'")
            sys.exit(1)
        
        config.setup_output_directory()
        config.print_final_configuration()
        
        # 2. Validar rango de fechas
        print("\n2️⃣ VALIDANDO RANGO DE FECHAS")
        start_dt = pd.to_datetime(config.start_date)
        end_dt = pd.to_datetime(config.end_date)
        
        if start_dt >= end_dt:
            print("❌ ERROR: Fecha inicial debe ser anterior a fecha final")
            sys.exit(1)
        
        date_range = pd.date_range(start=start_dt, end=end_dt, freq=config.frequency)
        total_dates = len(date_range)
        
        print(f"✅ Rango válido: {total_dates} fechas")
        print(f"   📅 Desde: {start_dt}")
        print(f"   📅 Hasta: {end_dt}")
        print(f"   ⏰ Frecuencia: {config.frequency}")
        
        # Estimar tiempo
        if config.frequency == 'D':
            est_hours = total_dates * 0.5  # ~30 min por fecha
        else:
            est_hours = total_dates * 0.1  # ~6 min por fecha horaria
        
        print(f"   ⏱️ Tiempo estimado: {est_hours:.1f} horas")
        
        # 3. Gestión del modelo
        print("\n3️⃣ GESTIÓN DEL MODELO")
        model_manager = ModelManager(config.config_file_path, config.verbose)
        model_config = model_manager.load_configuration()
        model = model_manager.build_and_load_model()
        
        # 4. Configuración del sistema de pronóstico
        print("\n4️⃣ INICIALIZANDO SISTEMA DE BARRIDO")
        forecast_system = ForecastSystem(model_config)
        forecast_system.setup(config.wrf_folder, model, model_manager.device)
        
        # 5. Crear procesador de lotes
        batch_processor = ForecastBatchProcessor(forecast_system, config.config_file_path)
        
        # 6. Ejecutar barrido
        print("\n5️⃣ EJECUTANDO BARRIDO DE FECHAS")
        results_df = batch_processor.run_batch_forecast(
            start_date=config.start_date,
            end_date=config.end_date,
            frequency=config.frequency,
            output_csv=config.output_csv,
            save_progress=config.save_progress,
            resume_from_checkpoint=config.resume_checkpoint
        )
        
        # 7. Análisis de resultados
        print("\n6️⃣ ANÁLISIS DE RESULTADOS")
        analysis = batch_processor.analyze_results(config.output_csv)
        
        # 8. Resultados finales
        print("\n7️⃣ RESULTADOS FINALES")
        print(f"✅ Barrido completado exitosamente")
        print(f"📊 Fechas procesadas: {len(results_df)}")
        print(f"📄 Archivo CSV: {config.output_csv}")
        print(f"💾 Tamaño del archivo: {os.path.getsize(config.output_csv) / (1024*1024):.1f} MB")
        
        if analysis:
            print(f"📈 Tasa de completitud: {analysis['completion_rate']:.1f}%")
            print(f"📊 Valores promedio: {analysis['value_stats']['mean']:.3f}")
        
        print("\n" + "=" * 60)
        print("🎉 BARRIDO DE FECHAS COMPLETADO EXITOSAMENTE")
        print("=" * 60)
        
        print("\n💡 PRÓXIMOS PASOS:")
        print(f"   📊 Usar {config.output_csv} para evaluaciones")
        print(f"   📈 Analizar resultados con herramientas de ML")
        print(f"   🔍 Validar pronósticos contra datos reales")
        
        return results_df
        
    except Exception as e:
        print(f"\n❌ ERROR CRÍTICO EN BARRIDO: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # Ejecutar función principal
    results = main() 