#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ejemplo_barrido_fechas.py - Ejemplos de uso del sistema de barrido de fechas

Este script muestra diferentes formas de usar el sistema de barrido de fechas
para generar CSV de pronósticos para evaluaciones.
"""

import os
import pandas as pd
from datetime import datetime, timedelta

# Imports del sistema profesional
from operativo_pro_01 import ForecastConfig, ModelManager
from forecast_utils2 import ForecastSystem, ForecastBatchProcessor


def ejemplo_barrido_semanal():
    """Ejemplo de barrido semanal (7 días)."""
    print("=" * 60)
    print("📅 EJEMPLO 1: BARRIDO SEMANAL")
    print("=" * 60)
    
    try:
        # 1. Configuración
        config = ForecastConfig()
        
        # 2. Gestión del modelo
        model_manager = ModelManager(config.config_file_path, config.verbose)
        model_config = model_manager.load_configuration()
        model = model_manager.build_and_load_model()
        
        # 3. Sistema de pronóstico
        forecast_system = ForecastSystem(model_config)
        forecast_system.setup(config.wrf_folder, model, model_manager.device)
        
        # 4. Procesador de lotes
        batch_processor = ForecastBatchProcessor(forecast_system, config.config_file_path)
        
        # 5. Ejecutar barrido semanal
        results_df = batch_processor.run_batch_forecast(
            start_date='2024-02-01 07:00:00',
            end_date='2024-06-06 07:00:00',
            frequency='D',  # Diario
            output_csv='forecast_semanal.csv',
            save_progress=True,
            resume_from_checkpoint=True
        )
        
        print(f"✅ Barrido semanal completado: {len(results_df)} fechas")
        return results_df
        
    except Exception as e:
        print(f"❌ Error en barrido semanal: {e}")
        return None


def ejemplo_barrido_horario():
    """Ejemplo de barrido horario (24 horas)."""
    print("\n" + "=" * 60)
    print("⏰ EJEMPLO 2: BARRIDO HORARIO")
    print("=" * 60)
    
    try:
        # 1. Configuración
        config = ForecastConfig()
        
        # 2. Gestión del modelo
        model_manager = ModelManager(config.config_file_path, config.verbose)
        model_config = model_manager.load_configuration()
        model = model_manager.build_and_load_model()
        
        # 3. Sistema de pronóstico
        forecast_system = ForecastSystem(model_config)
        forecast_system.setup(config.wrf_folder, model, model_manager.device)
        
        # 4. Procesador de lotes
        batch_processor = ForecastBatchProcessor(forecast_system, config.config_file_path)
        
        # 5. Ejecutar barrido horario
        results_df = batch_processor.run_batch_forecast(
            start_date='2023-05-15 00:00:00',
            end_date='2023-05-15 23:00:00',
            frequency='H',  # Horario
            output_csv='forecast_horario.csv',
            save_progress=True,
            resume_from_checkpoint=True
        )
        
        print(f"✅ Barrido horario completado: {len(results_df)} fechas")
        return results_df
        
    except Exception as e:
        print(f"❌ Error en barrido horario: {e}")
        return None


def ejemplo_barrido_personalizado():
    """Ejemplo de barrido personalizado con configuración específica."""
    print("\n" + "=" * 60)
    print("🔧 EJEMPLO 3: BARRIDO PERSONALIZADO")
    print("=" * 60)
    
    try:
        # 1. Configuración personalizada
        config = ForecastConfig()
        config.wrf_folder = '/dev/shm/tem_ram_forecast/'
        config.output_folder = './temp_barrido_personalizado/'
        config.verbose = True
        
        # 2. Gestión del modelo
        model_manager = ModelManager(config.config_file_path, config.verbose)
        model_config = model_manager.load_configuration()
        model = model_manager.build_and_load_model()
        
        # 3. Sistema de pronóstico
        forecast_system = ForecastSystem(model_config)
        forecast_system.setup(config.wrf_folder, model, model_manager.device)
        
        # 4. Procesador de lotes
        batch_processor = ForecastBatchProcessor(forecast_system, config.config_file_path)
        
        # 5. Ejecutar barrido personalizado (cada 6 horas)
        results_df = batch_processor.run_batch_forecast(
            start_date='2024-02-01 07:00:00',
            end_date='2024-06-06 07:00:00',
            frequency='1H',  # Cada 6 horas
            output_csv='forecast_personalizado.csv',
            save_progress=True,
            resume_from_checkpoint=True
        )
        
        print(f"✅ Barrido persona  lizado completado: {len(results_df)} fechas")
        return results_df
        
    except Exception as e:
        print(f"❌ Error en barrido personalizado: {e}")
        return None


def ejemplo_analisis_resultados():
    """Ejemplo de análisis de resultados de barrido."""
    print("\n" + "=" * 60)
    print("📊 EJEMPLO 4: ANÁLISIS DE RESULTADOS")
    print("=" * 60)
    
    # Archivos CSV de ejemplos anteriores
    csv_files = [
        'forecast_semanal.csv',
        'forecast_horario.csv',
        'forecast_personalizado.csv'
    ]
    
    for csv_file in csv_files:
        if os.path.exists(csv_file):
            print(f"\n📄 ANALIZANDO: {csv_file}")
            
            # Crear procesador para análisis
            config = ForecastConfig()
            model_manager = ModelManager(config.config_file_path, False)
            model_config = model_manager.load_configuration()
            model = model_manager.build_and_load_model()
            forecast_system = ForecastSystem(model_config)
            forecast_system.setup(config.wrf_folder, model, model_manager.device)
            batch_processor = ForecastBatchProcessor(forecast_system, config.config_file_path)
            
            # Análisis
            analysis = batch_processor.analyze_results(csv_file)
            
            if analysis:
                print(f"   📊 Fechas: {analysis['total_dates']}")
                print(f"   📈 Completitud: {analysis['completion_rate']:.1f}%")
                print(f"   📋 Columnas: {analysis['total_columns']}")
                print(f"   📉 Promedio: {analysis['value_stats']['mean']:.3f}")
                print(f"   📊 Rango: [{analysis['value_stats']['min']:.3f}, {analysis['value_stats']['max']:.3f}]")
        else:
            print(f"\n⚠️ Archivo no encontrado: {csv_file}")


def ejemplo_estructura_csv():
    """Ejemplo mostrando la estructura del CSV generado."""
    print("\n" + "=" * 60)
    print("📋 EJEMPLO 5: ESTRUCTURA DEL CSV")
    print("=" * 60)
    
    # Crear procesador para mostrar estructura
    config = ForecastConfig()
    model_manager = ModelManager(config.config_file_path, False)
    model_config = model_manager.load_configuration()
    model = model_manager.build_and_load_model()
    forecast_system = ForecastSystem(model_config)
    forecast_system.setup(config.wrf_folder, model, model_manager.device)
    batch_processor = ForecastBatchProcessor(forecast_system, config.config_file_path)
    
    # Mostrar estructura de columnas
    columns = batch_processor.create_forecast_columns(num_pollutants=54, num_horizons=24)
    
    print(f"\n📊 ESTRUCTURA DEL CSV:")
    print(f"   Total columnas: {len(columns)}")
    print(f"   Formato: colXX_h_plus_YY")
    print(f"   XX: Índice de contaminante (00-53)")
    print(f"   YY: Horizonte de pronóstico (01-24)")
    
    print(f"\n📝 EJEMPLOS DE COLUMNAS:")
    print(f"   Primeras 10: {columns[:10]}")
    print(f"   Últimas 10: {columns[-10:]}")
    
    print(f"\n💡 INTERPRETACIÓN:")
    print(f"   col00_h_plus_01: Contaminante 0, 1 hora adelante")
    print(f"   col00_h_plus_24: Contaminante 0, 24 horas adelante")
    print(f"   col53_h_plus_01: Contaminante 53, 1 hora adelante")
    print(f"   col53_h_plus_24: Contaminante 53, 24 horas adelante")


def ejemplo_uso_linea_comandos():
    """Ejemplo de uso desde línea de comandos."""
    print("\n" + "=" * 60)
    print("💻 EJEMPLO 6: USO DESDE LÍNEA DE COMANDOS")
    print("=" * 60)
    
    print("🔧 COMANDOS DE EJEMPLO:")
    print()
    
    # Barrido semanal
    print("1️⃣ Barrido semanal:")
    print("   python operativo_barrido_fechas.py \\")
    print("     --start-date '2023-05-01 07:00:00' \\")
    print("     --end-date '2023-05-07 07:00:00' \\")
    print("     --frequency D \\")
    print("     --output-csv forecast_semanal.csv")
    print()
    
    # Barrido horario
    print("2️⃣ Barrido horario:")
    print("   python operativo_barrido_fechas.py \\")
    print("     --start-date '2023-05-15 00:00:00' \\")
    print("     --end-date '2023-05-15 23:00:00' \\")
    print("     --frequency H \\")
    print("     --output-csv forecast_horario.csv")
    print()
    
    # Barrido personalizado
    print("3️⃣ Barrido personalizado:")
    print("   python operativo_barrido_fechas.py \\")
    print("     --start-date '2023-05-10 00:00:00' \\")
    print("     --end-date '2023-05-12 18:00:00' \\")
    print("     --frequency 6H \\")
    print("     --output-csv forecast_personalizado.csv \\")
    print("     --config-file mi_config.json \\")
    print("     --verbose")
    print()
    
    # Sin checkpoint
    print("4️⃣ Sin checkpoint (empezar desde cero):")
    print("   python operativo_barrido_fechas.py \\")
    print("     --start-date '2023-05-01 07:00:00' \\")
    print("     --end-date '2023-05-07 07:00:00' \\")
    print("     --no-resume \\")
    print("     --no-save-progress")


def main():
    """Función principal que ejecuta todos los ejemplos."""
    print("🎯 EJEMPLOS DE USO DEL SISTEMA DE BARRIDO DE FECHAS")
    
    # Advertencia sobre tiempo de ejecución
    print("\n⚠️  ADVERTENCIA:")
    print("   Los ejemplos con procesamiento real pueden tomar mucho tiempo")
    print("   Se recomienda ejecutar uno a la vez y monitorear el progreso")
    
    # Ejemplo 1: Barrido semanal (comentado para no ejecutar automáticamente)
    # print("\n🔄 Ejecutando ejemplo semanal...")
    # ejemplo_barrido_semanal()
    
    # Ejemplo 2: Barrido horario (comentado para no ejecutar automáticamente)
    # print("\n🔄 Ejecutando ejemplo horario...")
    # ejemplo_barrido_horario()
    
    # Ejemplo 3: Barrido personalizado (comentado para no ejecutar automáticamente)
    print("\n🔄 Ejecutando ejemplo personalizado...")
    ejemplo_barrido_personalizado()
    
    # Ejemplo 4: Análisis de resultados (si existen archivos)
    print("\n🔄 Ejecutando análisis de resultados...")
    ejemplo_analisis_resultados()
    
    # Ejemplo 5: Estructura del CSV
    print("\n🔄 Mostrando estructura del CSV...")
    ejemplo_estructura_csv()
    
    # Ejemplo 6: Uso desde línea de comandos
    print("\n🔄 Mostrando uso desde línea de comandos...")
    ejemplo_uso_linea_comandos()
    
    print("\n" + "=" * 60)
    print("🎉 EJEMPLOS COMPLETADOS")
    print("=" * 60)
    
    print("\n💡 PARA EJECUTAR UN BARRIDO REAL:")
    print("   1. Descomenta uno de los ejemplos arriba")
    print("   2. O usa operativo_barrido_fechas.py desde línea de comandos")
    print("   3. Monitorea el progreso y los checkpoints")
    print("   4. Usa los CSV generados para evaluaciones")


if __name__ == "__main__":
    main() 