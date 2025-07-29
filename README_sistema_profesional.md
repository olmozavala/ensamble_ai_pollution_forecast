# Sistema Profesional de Pronóstico de Contaminación v2.0

## 📋 Descripción

Sistema modularizado y profesional para realizar pronósticos de contaminación del aire en tiempo real usando modelos de deep learning y datos meteorológicos WRF.

## 🆕 Mejoras de la Versión 2.0

### ✅ Características Nuevas
- **Modularización completa**: Código organizado en clases y módulos reutilizables
- **Gestión profesional de errores**: Manejo robusto de excepciones
- **Configuración centralizada**: Sistema unificado de configuración
- **Logging mejorado**: Mensajes informativos y de progreso
- **Documentación completa**: Docstrings y comentarios profesionales
- **Validaciones automáticas**: Verificación de datos y configuraciones
- **Procesamiento WRF automático**: Ejecuta operativo001.py para generar archivos WRF
- **Compatibilidad preservada**: Mantiene la funcionalidad del sistema original

### 🔧 Conservado del Original
- **FORCE_DEFAULTS_FOR_DEBUG**: Sistema de debugging sin argparse
- **Configuración por argumentos**: Soporte completo para línea de comandos
- **Ejecución de operativo001.py**: Procesamiento WRF automático como el original
- **Funcionalidad idéntica**: Mismos resultados que el sistema original

## 📁 Estructura de Archivos

```
├── operativo_pro_01.py        # Script principal profesional
├── forecast_utils2.py         # Módulo de utilidades profesional
├── operativo_barrido_fechas.py # Script para barrido de fechas
├── ejemplo_uso.py             # Ejemplos de uso básico
├── ejemplo_barrido_fechas.py  # Ejemplos de barrido de fechas
├── verificar_sistema.py       # Script de verificación
├── README_sistema_profesional.md  # Esta documentación
└── operativo_01.py            # Script original (conservado)
```

## 🔍 Verificación del Sistema

Antes de usar el sistema, puedes verificar que todo funcione correctamente:

```bash
python verificar_sistema.py
```

Este script verifica:
- ✅ Archivos necesarios existentes
- ✅ Imports funcionando correctamente
- ✅ Configuración operativa
- ✅ Componentes principales
- ✅ Flujo completo del sistema

## 🚀 Uso Rápido

### Método 1: Ejecución Simple
```python
from operativo_pro_01 import main

# Ejecutar con configuración por defecto
predicciones = main()
```

### Método 4: Barrido de Fechas
```bash
# Barrido semanal
python operativo_barrido_fechas.py \
  --start-date '2023-05-01 07:00:00' \
  --end-date '2023-05-07 07:00:00' \
  --frequency D \
  --output-csv forecast_semanal.csv

# Barrido horario
python operativo_barrido_fechas.py \
  --start-date '2023-05-15 00:00:00' \
  --end-date '2023-05-15 23:00:00' \
  --frequency H \
  --output-csv forecast_horario.csv
```

### Método 2: Configuración Personalizada
```python
from operativo_pro_01 import ForecastConfig, ModelManager
from forecast_utils2 import ForecastSystem

# Configurar parámetros
config = ForecastConfig()
config.target_datetime = '2023-12-15 10:00:00'
config.output_folder = './mi_output/'

# Ejecutar pronóstico
model_manager = ModelManager(config.config_file_path)
model_config = model_manager.load_configuration()
model = model_manager.build_and_load_model()

forecast_system = ForecastSystem(model_config)
forecast_system.setup(config.wrf_folder, model, model_manager.device)
predicciones = forecast_system.run_forecast(config.target_datetime, config.config_file_path, config.output_folder)
```

### Método 3: Línea de Comandos
```bash
# Cambiar FORCE_DEFAULTS_FOR_DEBUG = False en operativo_pro_01.py
python operativo_pro_01.py --target-datetime "2023-12-13 07:00:00" --verbose
```

## ⚙️ Configuración

### Variables por Defecto
```python
DEFAULT_TARGET_DATETIME = '2023-12-13 07:00:00'
DEFAULT_CONFIG_FILE = 'test_Parallel_all_prev24_heads4_w4_p4_ar8_bootstrapTrue_thresh2_weather4_2_0701_101128.json'
DEFAULT_WRF_FOLDER = '/dev/shm/tem_ram_forecast/'
DEFAULT_OUTPUT_FOLDER = './tem_var/'
```

### Modo Debug (FORCE_DEFAULTS_FOR_DEBUG)
- **True**: Usa valores por defecto, ignora argumentos de línea de comandos
- **False**: Permite argumentos de línea de comandos

## 📊 Clases Principales

### 🔧 ForecastConfig
Configuración centralizada del sistema.
```python
config = ForecastConfig()
config.target_datetime = '2023-12-15 10:00:00'
config.verbose = True
```

### 🤖 ModelManager
Gestión del modelo de deep learning.
```python
model_manager = ModelManager(config_path, verbose=True)
model = model_manager.build_and_load_model()
```

### 🌤️ WRFProcessor
Procesador que ejecuta operativo001.py para generar archivos WRF.
```python
wrf_processor = WRFProcessor('operativo001.py')
success = wrf_processor.process_wrf_files(target_datetime, config_file)
```

### 📁 WRFDataLoader
Cargador de datos meteorológicos WRF.
```python
wrf_loader = WRFDataLoader('/path/to/wrf/')
weather_data = wrf_loader.load_data(use_all_nc=True)
```

### 🏭 PollutionDataManager
Gestor de datos de contaminación desde PostgreSQL.
```python
pollution_manager = PollutionDataManager(db_manager)
data, imputed = pollution_manager.get_contaminant_data(target_datetime)
```

### 🚀 ModelInference
Motor de inferencia autorregresiva.
```python
inference = ModelInference(model, device)
predictions = inference.run_autoregressive_inference(...)
```

### 📈 ResultsProcessor
Procesador de resultados y visualizaciones.
```python
processor = ResultsProcessor(norm_params_file)
denormalized = processor.denormalize_predictions(predictions, columns)
```

### 🎯 ForecastSystem
Sistema integrado que coordina todos los componentes.
```python
forecast_system = ForecastSystem(config)
forecast_system.setup(wrf_folder, model, device)
results = forecast_system.run_forecast(target_datetime, config_file_path, output_folder)
```

### 📅 ForecastBatchProcessor
Procesador de lotes para barrido de fechas.
```python
batch_processor = ForecastBatchProcessor(forecast_system, config_file_path)
results = batch_processor.run_batch_forecast(
    start_date='2023-05-01 07:00:00',
    end_date='2023-05-07 07:00:00',
    frequency='D',
    output_csv='forecast_batch.csv'
)
```

## 📈 Flujo de Trabajo

1. **Configuración**: Establecer parámetros del sistema
2. **Carga del Modelo**: Cargar modelo preentrenado y configuración
3. **Procesamiento WRF**: Ejecutar operativo001.py para generar archivos WRF
4. **Datos Meteorológicos**: Leer archivos WRF generados desde disco
5. **Datos de Contaminación**: Consultar base de datos PostgreSQL
6. **Normalización**: Aplicar parámetros de normalización
7. **Procesamiento**: Agregar estadísticas por contaminante
8. **Alineación Temporal**: Sincronizar datos meteorológicos y de contaminación
9. **Inferencia**: Ejecutar predicción autorregresiva
10. **Desnormalización**: Convertir a unidades reales
11. **Visualización**: Generar gráficos de resultados
12. **Guardado**: Exportar a CSV y gráficos

## 🔍 Parámetros del Modelo

- **prev_pollutant_hours**: Horas previas de contaminantes (24)
- **prev_weather_hours**: Horas previas de meteorología (2)
- **next_weather_hours**: Horas futuras de meteorología (4)
- **auto_regressive_steps**: Pasos de predicción autorregresiva (24)

## 📊 Salidas

### Archivos Generados
- `predictions_YYYYMMDD_HHMM.csv`: Predicciones en formato CSV
- `predictions_summary_YYYYMMDD_HHMM.png`: Gráfico de resumen
- Logs de ejecución en consola

### Formato de Datos
```python
# DataFrame con índice temporal y columnas de contaminantes
predicciones.index     # DatetimeIndex con horas de predicción
predicciones.columns   # Columnas de contaminantes (cont_otres_UIZ, etc.)
predicciones.values    # Concentraciones desnormalizadas
```

## 🐛 Debugging

### Mensajes de Log
- 🚀 Inicio de operaciones
- ✅ Operaciones exitosas
- ⚠️ Advertencias
- ❌ Errores
- 🔧 Operaciones técnicas
- 📊 Estadísticas

### Variables de Debug
```python
config.verbose = True    # Mensajes detallados
config.debug_mode = True # Información adicional
config.plots_mode = True # Generar gráficos
```

## 🔄 Comparación con Versión Original

| Aspecto | Original | Profesional v2.0 |
|---------|----------|------------------|
| Estructura | Script monolítico | Modular (clases) |
| Configuración | Variables globales | Clase centralizada |
| Manejo de errores | Básico | Robusto |
| Documentación | Comentarios | Docstrings completos |
| Reutilización | Limitada | Alta |
| Mantenimiento | Difícil | Fácil |
| Funcionalidad | ✅ Completa | ✅ Idéntica |

## 📞 Contacto y Soporte

Para dudas o problemas:
1. Revisar logs de ejecución
2. Verificar configuración de base de datos
3. Comprobar archivos WRF disponibles
4. Validar parámetros de normalización

## 📅 Barrido de Fechas para Evaluaciones

### 🎯 Propósito
El sistema de barrido de fechas permite generar pronósticos para múltiples fechas de forma secuencial, creando un CSV estructurado ideal para evaluaciones posteriores.

### 📊 Estructura del CSV Generado
```
fecha,col00_h_plus_01,col00_h_plus_02,...,col00_h_plus_24,col01_h_plus_01,...,col53_h_plus_24
2023-05-01 07:00:00,0.123,0.145,...,0.234,0.456,...,0.789
2023-05-02 07:00:00,0.234,0.256,...,0.345,0.567,...,0.890
...
```

**Explicación de columnas:**
- `colXX_h_plus_YY`: Contaminante XX (0-53), horizonte YY (1-24 horas)
- 54 contaminantes × 24 horizontes = 1,296 columnas por fecha
- Cada fila representa un pronóstico completo para una fecha específica

### 🚀 Uso del Barrido

#### Línea de Comandos
```bash
# Barrido semanal diario
python operativo_barrido_fechas.py \
  --start-date '2023-05-01 07:00:00' \
  --end-date '2023-05-07 07:00:00' \
  --frequency D \
  --output-csv forecast_semanal.csv

# Barrido horario
python operativo_barrido_fechas.py \
  --start-date '2023-05-15 00:00:00' \
  --end-date '2023-05-15 23:00:00' \
  --frequency H \
  --output-csv forecast_horario.csv

# Barrido cada 6 horas
python operativo_barrido_fechas.py \
  --start-date '2023-05-01 00:00:00' \
  --end-date '2023-05-03 18:00:00' \
  --frequency 6H \
  --output-csv forecast_6h.csv
```

#### Programáticamente
```python
from operativo_pro_01 import ForecastConfig, ModelManager
from forecast_utils2 import ForecastSystem, ForecastBatchProcessor

# Configurar sistema
config = ForecastConfig()
model_manager = ModelManager(config.config_file_path)
model_config = model_manager.load_configuration()
model = model_manager.build_and_load_model()

forecast_system = ForecastSystem(model_config)
forecast_system.setup(config.wrf_folder, model, model_manager.device)

# Ejecutar barrido
batch_processor = ForecastBatchProcessor(forecast_system, config.config_file_path)
results = batch_processor.run_batch_forecast(
    start_date='2023-05-01 07:00:00',
    end_date='2023-05-07 07:00:00',
    frequency='D',
    output_csv='forecast_batch.csv'
)
```

### 🔧 Características Avanzadas

#### Checkpoints y Reanudación
```bash
# El sistema guarda progreso automáticamente
python operativo_barrido_fechas.py \
  --start-date '2023-05-01 07:00:00' \
  --end-date '2023-05-31 07:00:00' \
  --frequency D \
  --output-csv forecast_mayo.csv
  
# Si se interrumpe, reanudar automáticamente
python operativo_barrido_fechas.py \
  --start-date '2023-05-01 07:00:00' \
  --end-date '2023-05-31 07:00:00' \
  --frequency D \
  --output-csv forecast_mayo.csv  # Continúa desde checkpoint
```

#### Opciones de Configuración
```bash
# Sin guardar progreso
--no-save-progress

# Sin reanudar desde checkpoint
--no-resume

# Configuración personalizada
--config-file mi_config.json
--wrf-folder /mi/path/wrf/
--verbose
```

### 📊 Análisis de Resultados

#### Verificar Completitud
```python
from forecast_utils2 import ForecastBatchProcessor
from operativo_pro_01 import ForecastConfig, ModelManager

# Configurar procesador
config = ForecastConfig()
model_manager = ModelManager(config.config_file_path)
model_config = model_manager.load_configuration()
model = model_manager.build_and_load_model()
forecast_system = ForecastSystem(model_config)
batch_processor = ForecastBatchProcessor(forecast_system, config.config_file_path)

# Analizar resultados
analysis = batch_processor.analyze_results('forecast_batch.csv')
print(f"Completitud: {analysis['completion_rate']:.1f}%")
print(f"Fechas procesadas: {analysis['total_dates']}")
```

### ⚠️ Consideraciones Importantes

1. **Procesamiento Secuencial**: Los archivos WRF no se pueden generar en paralelo
2. **Tiempo de Ejecución**: ~30 minutos por fecha diaria, ~6 minutos por fecha horaria
3. **Espacio en Disco**: Los CSV pueden ser grandes (varios GB para barridos largos)
4. **Memoria**: Cada fecha requiere ~2GB de RAM durante el procesamiento
5. **Checkpoints**: Se guardan automáticamente en `archivo_checkpoint.csv`

### 💡 Casos de Uso

- **Evaluación de Modelos**: Generar pronósticos para períodos históricos
- **Análisis de Rendimiento**: Comparar horizontes de predicción
- **Validación Cruzada**: Crear datasets para evaluación temporal
- **Estudios de Sensibilidad**: Analizar variabilidad por fechas/estaciones

## 🛡️ Sistema de Validación y Manejo de Errores (v2.1)

### ✅ Mejoras Implementadas

#### Validación Previa
- **Filtrado automático**: Elimina fechas sin datos suficientes antes del procesamiento
- **Verificación de 24 horas**: Garantiza datos históricos necesarios
- **Validación de índices**: Previene errores de "not in index"

#### Manejo de Errores Inteligente
- **Categorización automática**: Clasifica errores por tipo (datos insuficientes, datos faltantes, errores WRF)
- **Reportes detallados**: Muestra estadísticas de errores y recomendaciones
- **Continuidad**: El sistema continúa procesando aunque algunas fechas fallen

#### Carga Automática Extendida
- **Buffer de datos**: Carga automáticamente +5 horas adicionales
- **Garantía de datos**: Asegura mínimo 24 horas históricas
- **Detección temprana**: Identifica problemas antes de la inferencia

### 🔧 Uso con Validación

```bash
# Barrido con validación automática
python operativo_barrido_fechas.py \
  --start-date '2023-05-01 00:00:00' \
  --end-date '2023-05-07 00:00:00' \
  --frequency D \
  --output-csv forecast_validado.csv
```

### 📊 Ejemplo de Reporte de Errores

```
📈 RESUMEN FINAL:
============================================================
   ✅ Pronósticos exitosos: 9
   ❌ Pronósticos fallidos: 3
   📊 Tasa de éxito: 75.0%

🔍 DESGLOSE DE ERRORES:
   Datos Faltantes: 2
   Datos Insuficientes: 1

💡 RECOMENDACIONES:
   • Algunos datos pueden estar faltando en la base de datos
   • Verifique la conectividad a la base de datos
   • Considere usar fechas más recientes (después de 2020)
```

### 🔍 Análisis de Errores

```python
# Usar el ejemplo mejorado
python ejemplo_uso_mejorado.py
```

Este script incluye:
- Pronóstico individual con validación
- Barrido con validación previa
- Análisis de errores detallado
- Recomendaciones automáticas

## 🔮 Próximas Mejoras

- [ ] Soporte para múltiples modelos
- [ ] Cache de datos para mejores tiempos
- [ ] API REST para integración
- [ ] Dashboard web de resultados
- [ ] Notificaciones automáticas
- [ ] Validación automática de resultados
- [ ] Paralelización de barridos (cuando sea posible) 