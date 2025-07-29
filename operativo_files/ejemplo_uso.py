#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ejemplo_uso.py - Ejemplo de uso del sistema de pronóstico

Este script muestra cómo usar el nuevo sistema modularizado para realizar
pronósticos de contaminación.
"""
# %%
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from operativo_pro_01 import main

# Ejecutar con configuración por defecto
predicciones = main()

# %%
import os
# sys.path ya configurado arriba
from operativo_pro_01 import main, ForecastConfig, ModelManager
from .forecast_utils2 import ForecastSystem

def ejemplo_basico():
    """Ejemplo básico usando la función main."""
    print("=" * 60)
    print("🌟 EJEMPLO BÁSICO - USANDO FUNCIÓN MAIN")
    print("=" * 60)
    
    # Simplemente ejecutar el main que maneja todo automáticamente
    try:
        predicciones = main()
        print(f"\n✅ Ejemplo completado exitosamente")
        print(f"📊 Predicciones generadas: {len(predicciones)} horas")
        print(f"📈 Columnas: {len(predicciones.columns)} variables")
        return predicciones
    except Exception as e:
        print(f"❌ Error en ejemplo básico: {e}")
        return None

def ejemplo_personalizado():
    """Ejemplo con configuración personalizada."""
    print("\n" + "=" * 60)
    print("🔧 EJEMPLO PERSONALIZADO - CONFIGURACIÓN MANUAL")
    print("=" * 60)
    
    try:
        # 1. Configuración personalizada
        config = ForecastConfig()
        config.target_datetime = '2023-12-15 10:00:00'  # Fecha diferente
        config.output_folder = './resultados_personalizados/'
        config.verbose = True
        config.setup_output_directory()
        config.print_final_configuration()
        
        # 2. El resto sigue igual que en main()
        print("\n🤖 Cargando modelo...")
        model_manager = ModelManager(config.config_file_path, config.verbose)
        model_config = model_manager.load_configuration()
        model = model_manager.build_and_load_model()
        
        # 3. Sistema de pronóstico
        print("\n🚀 Ejecutando pronóstico personalizado...")
        forecast_system = ForecastSystem(model_config)
        forecast_system.setup(config.wrf_folder, model, model_manager.device)
        
        predicciones = forecast_system.run_forecast(
            config.target_datetime,
            config.config_file_path,
            config.output_folder
        )
        
        print(f"\n✅ Ejemplo personalizado completado")
        print(f"📊 Predicciones: {len(predicciones)} x {len(predicciones.columns)}")
        return predicciones
        
    except Exception as e:
        print(f"❌ Error en ejemplo personalizado: {e}")
        import traceback
        traceback.print_exc()
        return None

def ejemplo_solo_configuracion():
    """Ejemplo mostrando solo la configuración sin ejecutar."""
    print("\n" + "=" * 60)
    print("⚙️ EJEMPLO SOLO CONFIGURACIÓN")
    print("=" * 60)
    
    # Mostrar configuración con argumentos de línea de comandos
    print("\n1️⃣ Configuración con defaults:")
    config1 = ForecastConfig()
    config1.print_final_configuration()
    
    print("\n2️⃣ Configuración personalizada:")
    config2 = ForecastConfig()
    config2.target_datetime = '2023-12-20 15:30:00'
    config2.wrf_folder = '/otro/path/wrf/'
    config2.debug_mode = False
    config2.plots_mode = False
    config2.print_final_configuration()

if __name__ == "__main__":
    print("🎯 EJEMPLOS DE USO DEL SISTEMA DE PRONÓSTICO")
    
    # Ejemplo 1: Uso básico
    predicciones_basicas = ejemplo_basico()
    
    # Ejemplo 2: Configuración personalizada
    predicciones_personalizadas = ejemplo_personalizado()
    
    # Ejemplo 3: Solo configuración (sin ejecutar)
    ejemplo_solo_configuracion()
    
    print("\n" + "=" * 60)
    print("🎉 TODOS LOS EJEMPLOS COMPLETADOS")
    print("=" * 60)
    
    # Resumen
    if predicciones_basicas is not None:
        print(f"✅ Ejemplo básico: {len(predicciones_basicas)} horas")
    if predicciones_personalizadas is not None:
        print(f"✅ Ejemplo personalizado: {len(predicciones_personalizadas)} horas")
    
    print("\n💡 Para usar en tu código:")
    print("   from operativo_pro_01 import main")
    print("   predicciones = main()")
    print("\n💡 Para configuración personalizada:")
    print("   from operativo_pro_01 import ForecastConfig, ModelManager")
    print("   from .forecast_utils2 import ForecastSystem") 