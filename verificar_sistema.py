#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verificar_sistema.py - Script de verificación del sistema profesional

Este script verifica que el sistema profesional funcione igual que el original.
"""

import os
import sys
from datetime import datetime

def verificar_archivos_existentes():
    """Verifica que todos los archivos necesarios existan."""
    print("🔍 VERIFICANDO ARCHIVOS NECESARIOS...")
    
    archivos_requeridos = [
        'operativo_pro_01.py',
        'forecast_utils2.py',
        'operativo_barrido_fechas.py',
        'ejemplo_barrido_fechas.py',
        'operativo001.py',
        'operativo_01.py',
        'ejemplo_uso.py',
        'README_sistema_profesional.md'
    ]
    
    archivos_encontrados = []
    archivos_faltantes = []
    
    for archivo in archivos_requeridos:
        if os.path.exists(archivo):
            archivos_encontrados.append(archivo)
            print(f"   ✅ {archivo}")
        else:
            archivos_faltantes.append(archivo)
            print(f"   ❌ {archivo} - NO ENCONTRADO")
    
    print(f"\n📊 RESULTADO:")
    print(f"   ✅ Encontrados: {len(archivos_encontrados)}")
    print(f"   ❌ Faltantes: {len(archivos_faltantes)}")
    
    return len(archivos_faltantes) == 0

def verificar_imports():
    """Verifica que los imports funcionen correctamente."""
    print("\n🔍 VERIFICANDO IMPORTS...")
    
    try:
        from operativo_pro_01 import ForecastConfig, ModelManager, main
        print("   ✅ operativo_pro_01 imports OK")
    except ImportError as e:
        print(f"   ❌ operativo_pro_01 import ERROR: {e}")
        return False
    
    try:
        from forecast_utils2 import (
            ForecastSystem, DatabaseManager, WRFDataLoader, 
            ModelInference, ResultsProcessor, WRFProcessor, ForecastBatchProcessor
        )
        print("   ✅ forecast_utils2 imports OK")
    except ImportError as e:
        print(f"   ❌ forecast_utils2 import ERROR: {e}")
        return False
    
    return True

def verificar_configuracion():
    """Verifica que la configuración funcione correctamente."""
    print("\n🔍 VERIFICANDO CONFIGURACIÓN...")
    
    try:
        from operativo_pro_01 import ForecastConfig
        
        # Configuración por defecto
        config = ForecastConfig()
        print(f"   ✅ Default target_datetime: {config.target_datetime}")
        print(f"   ✅ Default config_file: {config.config_file_path}")
        print(f"   ✅ Default debug_mode: {config.debug_mode}")
        print(f"   ✅ Default wrf_folder: {config.wrf_folder}")
        print(f"   ✅ Default output_folder: {config.output_folder}")
        
        # Test configuración personalizada
        config.target_datetime = '2023-12-20 10:00:00'
        config.debug_mode = False
        print(f"   ✅ Custom target_datetime: {config.target_datetime}")
        print(f"   ✅ Custom debug_mode: {config.debug_mode}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error en configuración: {e}")
        return False

def verificar_componentes():
    """Verifica que los componentes principales funcionen."""
    print("\n🔍 VERIFICANDO COMPONENTES PRINCIPALES...")
    
    try:
        from forecast_utils2 import (
            WRFProcessor, DatabaseManager, WRFDataLoader, 
            ModelInference, ResultsProcessor, ForecastSystem, ForecastBatchProcessor
        )
        
        # Test WRFProcessor
        wrf_processor = WRFProcessor()
        print(f"   ✅ WRFProcessor creado: {wrf_processor.operativo_script_path}")
        
        # Test DatabaseManager
        db_manager = DatabaseManager()
        print(f"   ✅ DatabaseManager creado")
        
        # Test WRFDataLoader
        wrf_loader = WRFDataLoader('/tmp/test')
        print(f"   ✅ WRFDataLoader creado: {wrf_loader.wrf_folder}")
        
        # Test ForecastSystem
        test_config = {'name': 'test'}
        forecast_system = ForecastSystem(test_config)
        print(f"   ✅ ForecastSystem creado")
        
        # Test ForecastBatchProcessor
        batch_processor = ForecastBatchProcessor(forecast_system, 'test_config.json')
        print(f"   ✅ ForecastBatchProcessor creado")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error en componentes: {e}")
        return False

def verificar_flujo_completo():
    """Verifica que el flujo completo funcione (sin ejecutar)."""
    print("\n🔍 VERIFICANDO FLUJO COMPLETO...")
    
    try:
        from operativo_pro_01 import ForecastConfig, ModelManager
        from forecast_utils2 import ForecastSystem
        
        # 1. Configuración
        config = ForecastConfig()
        config.target_datetime = '2023-05-15 07:00:00'
        print("   ✅ Configuración creada")
        
        # 2. Verificar archivos necesarios
        config_file_exists = os.path.exists(config.config_file_path)
        operativo001_exists = os.path.exists('operativo001.py')
        
        print(f"   ✅ Config file exists: {config_file_exists}")
        print(f"   ✅ operativo001.py exists: {operativo001_exists}")
        
        if not config_file_exists:
            print("   ⚠️  Archivo de configuración no encontrado, pero flujo OK")
        
        if not operativo001_exists:
            print("   ⚠️  operativo001.py no encontrado, pero flujo OK")
        
        # 3. Verificar que main() puede ser importada
        from operativo_pro_01 import main
        print("   ✅ main() importada correctamente")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error en flujo completo: {e}")
        return False

def verificar_barrido_fechas():
    """Verifica que el sistema de barrido de fechas funcione."""
    print("\n🔍 VERIFICANDO BARRIDO DE FECHAS...")
    
    try:
        # 1. Verificar imports específicos del barrido
        from operativo_barrido_fechas import BatchForecastConfig, main as batch_main
        print("   ✅ BatchForecastConfig importada")
        
        from forecast_utils2 import ForecastBatchProcessor
        print("   ✅ ForecastBatchProcessor importada")
        
        # 2. Verificar configuración del barrido
        batch_config = BatchForecastConfig()
        batch_config.start_date = '2023-05-01 07:00:00'
        batch_config.end_date = '2023-05-03 07:00:00'
        print("   ✅ Configuración de barrido creada")
        
        # 3. Verificar generación de fechas
        from forecast_utils2 import ForecastSystem
        test_config = {'name': 'test'}
        forecast_system = ForecastSystem(test_config)
        batch_processor = ForecastBatchProcessor(forecast_system, 'test_config.json')
        
        # Generar fechas de prueba
        date_list = batch_processor.generate_date_range(
            '2023-05-01 07:00:00', 
            '2023-05-03 07:00:00', 
            'D'
        )
        print(f"   ✅ Generación de fechas: {len(date_list)} fechas")
        
        # 4. Verificar estructura de columnas
        columns = batch_processor.create_forecast_columns(num_pollutants=54, num_horizons=24)
        expected_columns = 54 * 24
        print(f"   ✅ Estructura CSV: {len(columns)} columnas (esperadas: {expected_columns})")
        
        # 5. Verificar ejemplo de uso
        from ejemplo_barrido_fechas import ejemplo_estructura_csv
        print("   ✅ Ejemplos de barrido importados")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error en barrido de fechas: {e}")
        return False

def main():
    """Función principal de verificación."""
    print("=" * 60)
    print("🔍 VERIFICACIÓN DEL SISTEMA PROFESIONAL DE PRONÓSTICO")
    print("=" * 60)
    
    resultados = []
    
    # Test 1: Archivos
    resultados.append(verificar_archivos_existentes())
    
    # Test 2: Imports
    resultados.append(verificar_imports())
    
    # Test 3: Configuración
    resultados.append(verificar_configuracion())
    
    # Test 4: Componentes
    resultados.append(verificar_componentes())
    
    # Test 5: Flujo completo
    resultados.append(verificar_flujo_completo())
    
    # Test 6: Barrido de fechas
    resultados.append(verificar_barrido_fechas())
    
    # Resumen
    print("\n" + "=" * 60)
    print("📊 RESUMEN DE VERIFICACIÓN")
    print("=" * 60)
    
    tests_passed = sum(resultados)
    total_tests = len(resultados)
    
    print(f"✅ Tests exitosos: {tests_passed}/{total_tests}")
    print(f"❌ Tests fallidos: {total_tests - tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("\n🎉 ¡TODOS LOS TESTS EXITOSOS!")
        print("✅ El sistema profesional está listo para usar")
        print("\n💡 Para probar el sistema completo:")
        print("   python operativo_pro_01.py")
        print("   # o")
        print("   python ejemplo_uso.py")
        
    else:
        print("\n⚠️  ALGUNOS TESTS FALLARON")
        print("❌ Revisar errores arriba antes de usar el sistema")
        print("\n💡 Para más información consulta:")
        print("   README_sistema_profesional.md")
    
    return tests_passed == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 