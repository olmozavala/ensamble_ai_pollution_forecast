#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ejemplo_uso_mejorado.py - Ejemplo de uso del sistema mejorado con manejo de errores

Este script demuestra cómo usar el sistema de pronóstico con las mejoras
de validación y manejo de errores.
"""

import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from operativo_pro_01 import ForecastConfig, ModelManager, main
from .forecast_utils2 import ForecastSystem, ForecastBatchProcessor
import torch
import os

def ejemplo_pronostico_individual():
    """Ejemplo de pronóstico individual con validación mejorada."""
    print("="*60)
    print("🔬 EJEMPLO: PRONÓSTICO INDIVIDUAL CON VALIDACIÓN")
    print("="*60)
    
    try:
        # Configurar fecha (usar fecha reciente con datos disponibles)
        target_datetime = '2023-05-15 07:00:00'
        
        # Ejecutar pronóstico
        print(f"📅 Ejecutando pronóstico para: {target_datetime}")
        predictions = main()
        
        print(f"✅ Pronóstico exitoso")
        print(f"📊 Predicciones: {len(predictions)} pasos temporales")
        print(f"🗂️ Variables: {len(predictions.columns)} contaminantes")
        
        return predictions
        
    except Exception as e:
        print(f"❌ Error en pronóstico individual: {e}")
        return None


def ejemplo_barrido_con_validacion():
    """Ejemplo de barrido de fechas con validación previa."""
    print("\n" + "="*60)
    print("🔬 EJEMPLO: BARRIDO CON VALIDACIÓN PREVIA")
    print("="*60)
    
    try:
        # Configurar sistema
        config_path = 'test_Parallel_all_prev24_heads4_w4_p4_ar8_bootstrapTrue_thresh2_weather4_2_0701_101128.json'
        
        # Simular configuración del sistema
        from parse_config import ConfigParser
        from utils import read_json
        
        raw_config = read_json(config_path)
        config = ConfigParser(raw_config)
        
        # Crear sistema de pronóstico
        forecast_system = ForecastSystem(config)
        
        # Crear procesador de lotes
        batch_processor = ForecastBatchProcessor(forecast_system, config_path)
        
        # Ejecutar barrido con fechas que tienen datos disponibles
        print("📅 Ejecutando barrido de fechas con validación...")
        results = batch_processor.run_batch_forecast(
            start_date='2023-05-10 00:00:00',
            end_date='2023-05-12 18:00:00',
            frequency='6H',  # Cada 6 horas
            output_csv='forecast_validado.csv',
            save_progress=True,
            resume_from_checkpoint=True
        )
        
        print(f"✅ Barrido completado")
        print(f"📊 Resultados: {len(results)} fechas procesadas")
        
        return results
        
    except Exception as e:
        print(f"❌ Error en barrido: {e}")
        return None


def ejemplo_analisis_errores():
    """Ejemplo de análisis de errores en resultados."""
    print("\n" + "="*60)
    print("🔬 EJEMPLO: ANÁLISIS DE ERRORES")
    print("="*60)
    
    # Verificar si existe archivo de resultados
    csv_files = ['forecast_validado.csv', 'forecast_personalizado.csv']
    
    for csv_file in csv_files:
        if os.path.exists(csv_file):
            print(f"📊 Analizando archivo: {csv_file}")
            
            # Cargar datos
            df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
            
            # Estadísticas básicas
            print(f"   📅 Fechas: {len(df)}")
            print(f"   📊 Columnas: {len(df.columns)}")
            print(f"   ✅ Datos válidos: {(~df.isnull()).sum().sum()}")
            print(f"   ❌ Datos faltantes: {df.isnull().sum().sum()}")
            print(f"   📈 Tasa de completitud: {((~df.isnull()).sum().sum() / df.size * 100):.1f}%")
            
            # Identificar fechas con problemas
            fechas_problema = df[df.isnull().all(axis=1)]
            if not fechas_problema.empty:
                print(f"   ⚠️ Fechas con errores: {len(fechas_problema)}")
                print(f"   📋 Ejemplos: {fechas_problema.index[:3].tolist()}")
            
            return df
    
    print("❌ No se encontraron archivos de resultados para analizar")
    return None


def ejemplo_recomendaciones():
    """Muestra recomendaciones para mejorar el rendimiento."""
    print("\n" + "="*60)
    print("💡 RECOMENDACIONES PARA MEJORAR EL RENDIMIENTO")
    print("="*60)
    
    print("1. 📅 SELECCIÓN DE FECHAS:")
    print("   • Use fechas después de 2020 para mejor disponibilidad de datos")
    print("   • Evite fechas muy cercanas al presente (pueden faltar datos WRF)")
    print("   • Verifique que hay al menos 24 horas de datos históricos")
    
    print("\n2. 🔧 CONFIGURACIÓN DEL SISTEMA:")
    print("   • Aumente 'hours_back' si hay problemas de datos insuficientes")
    print("   • Use checkpoints para reanudar procesamiento interrumpido")
    print("   • Procese en lotes pequeños para mejor manejo de memoria")
    
    print("\n3. 🐛 MANEJO DE ERRORES:")
    print("   • Revise los logs para errores específicos")
    print("   • Verifique conectividad a la base de datos")
    print("   • Asegúrese de que operativo001.py funciona correctamente")
    
    print("\n4. 📈 OPTIMIZACIÓN:")
    print("   • Use frecuencias más espaciadas (6H, 12H) para pruebas")
    print("   • Implemente validación previa de datos")
    print("   • Monitoree el uso de memoria y espacio en disco")


def main():
    """Función principal que ejecuta todos los ejemplos."""
    print("🌟 SISTEMA MEJORADO DE PRONÓSTICO - EJEMPLOS DE USO")
    print("="*60)
    
    # Ejemplo 1: Pronóstico individual
    resultado_individual = ejemplo_pronostico_individual()
    
    # Ejemplo 2: Barrido con validación
    # resultado_barrido = ejemplo_barrido_con_validacion()
    
    # Ejemplo 3: Análisis de errores
    resultado_analisis = ejemplo_analisis_errores()
    
    # Ejemplo 4: Recomendaciones
    #ejemplo_recomendaciones()
    
    print("\n" + "="*60)
    print("🎉 EJEMPLOS COMPLETADOS")
    print("="*60)


if __name__ == "__main__":
    main() 