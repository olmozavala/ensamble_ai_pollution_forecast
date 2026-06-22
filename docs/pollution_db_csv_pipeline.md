# Pipeline DB → CSV para entrenamiento ML

Dos scripts autocontenidos que reemplazan el flujo disperso (`x_make_csv_from_db_yml.py`, `x2_MData.py`, `imputation_7_fixed.py` + transformaciones manuales) por un pipeline reproducible con rutas centralizadas.

```
PostgreSQL (AMATE-OPERATIVO)
        │
        ▼  x_db_full_pipeline.py
   export/  merged/  imputed/
        │
        ▼  x_db_ml_features_csv.py
   mlforecast/
        │
        ▼  copiar a PollutionCSV/  →  4_train.py (MLforecastFeatureMapDataLoader)
```

---

## Estructura de datos

Todo vive bajo `{DATA_ROOT}/DataPollutionDB_CSV_YML/` (por defecto `/ZION/AirPollutionData/Data/DataPollutionDB_CSV_YML/`):

| Carpeta | Script | Contenido |
|---------|--------|-----------|
| `export/` | paso 1 | `cont_{slug}_{EST}.csv` por estación y contaminante |
| `merged/` | paso 2 | `{year}_AllStations.csv` (30 estaciones operativas) |
| `imputed/` | paso 3 | `data_imputed_7fix_{year}.csv` (imputación completa) |
| `climatology/` | paso 3 | climatología horaria de referencia |
| `plots/` | paso 3 (opcional) | gráficas de análisis de imputación |
| `mlforecast/` | paso 4 | `data_imputed_7_{year}.csv` (listo para entrenar) |

---

## Script 1: `x_db_full_pipeline.py`

Pipeline en tres pasos activables con flags al inicio del archivo.

### Pasos

1. **Export** — Lee tablas `cont_*` de PostgreSQL y escribe un CSV por estación/contaminante.
2. **Merge** — Une 30 estaciones (`MERGE_STATIONS`) en `{year}_AllStations.csv` con features temporales (`half_sin_*`, `sin_*`, etc.).
3. **Imputación** — Row average → persistencia → climatología; genera `data_imputed_7fix_{year}.csv`.

### Parámetros principales (editar en el script)

| Variable | Default | Descripción |
|----------|---------|-------------|
| `DATA_ROOT` | `/ZION/AirPollutionData/Data` | Raíz de datos |
| `RUN_EXPORT` / `RUN_MERGE` / `RUN_IMPUTATION` | `True` | Activar/desactivar pasos |
| `RUN_IMPUTATION_ANALYSIS` | `False` | Dendrogramas/plots (lento; requiere sklearn, matplotlib, seaborn) |
| `NETRC_HOST` | `AMATE-OPERATIVO` | Host en `~/.netrc` |
| `DATABASE` | `contingencia` | Base PostgreSQL |
| `EXPORT_YEARS` | `2000..2026` | Años del merge |
| `MERGE_STATIONS` | 30 estaciones | Subconjunto operativo |
| `IMPUTATION_OUTPUT_PREFIX` | `data_imputed_7fix` | Prefijo de salida imputada |

### Requisitos

```bash
pip install pandas numpy psycopg2-binary
```

- Credenciales PostgreSQL en `~/.netrc` (máquina `AMATE-OPERATIVO`).
- Acceso de lectura a la base `contingencia`.

### Ejecución

```bash
python x_db_full_pipeline.py
```

Solo imputación (si export/merge ya existen):

```python
RUN_EXPORT = False
RUN_MERGE = False
RUN_IMPUTATION = True
```

---

## Script 2: `x_db_ml_features_csv.py`

Transforma `data_imputed_7fix_{year}.csv` → `data_imputed_7_{year}.csv` con agregaciones **causales** (sin leakage: `shift(1)` antes de rolling).

### Transformaciones

| Tipo | Implementación |
|------|----------------|
| `3h` / `6h` | Media móvil causal de 3 o 6 horas |
| `ALL` | Media espacial sobre `MERGE_STATIONS` |
| `mda8` | MA8 de 8h (`min_periods=6`) → `max` en ventana −24h |

Lookback: 32 h del año anterior (para arranque de enero).

### Config actual (`POLLUTANT_SPECS`)

| Slug | Spatial | Transforms | Extra | Notas |
|------|---------|------------|-------|-------|
| co, no, pmdiez, pmdoscinco | per_station | `3h` | `ALL/mda8` | drop fuentes raw |
| nodos | ALL | `3h` | `ALL/mda8` | drop fuentes |
| sodos | ALL | — | `ALL/mda8` | horario + mda8 |
| otres | per_station | — | `ALL/mda8` | conserva horario por estación |
| nox | — | — | — | `enabled: false` |

**Esquema de salida (config actual):** 12 time + 159 `cont_*` + 159 `i_cont_*` = **330 columnas**.

### Parámetros principales

| Variable | Default | Descripción |
|----------|---------|-------------|
| `DATA_ROOT` | `/ZION/AirPollutionData/Data` | Misma raíz que el pipeline full |
| `PROCESS_YEARS` | `2000..2026` | Años a procesar |
| `MERGE_STATIONS` | 30 estaciones | Debe coincidir con el merge |
| `POLLUTANT_SPECS` | ver script | Esquema de columnas de salida |

### Requisitos

```bash
pip install pandas numpy
# opcional: pyyaml (export_config_yaml)
```

### Ejecución

```bash
python x_db_ml_features_csv.py
```

Al terminar imprime inventario de columnas y valida que los headers sean idénticos entre años.

---

## Integración con entrenamiento

### 1. Copiar CSV al sandbox de entrenamiento

```bash
scp -P 9022 \
  /ZION/AirPollutionData/Data/DataPollutionDB_CSV_YML/mlforecast/data_imputed_7_*.csv \
  psegura@chacmool.atmosfera.unam.mx:/home/psegura/netcdfs/PollutionCSV/
```

### 2. JSON de entrenamiento (`MLforecastFeatureMapDataLoader`)

El `pollution_feature_map` debe referir columnas **exactas** del CSV (con sufijos `_3h`, `_mda8`, etc.). No usar entradas planas `mode: ALL` sin `transform` si la columna no existe.

Config alineada con `POLLUTANT_SPECS` actual:

- Archivo: `operativo_files/test_Parallel_featuremap_v2_3h_nox_off_psegura.json`
- `input_features`: **171** (159 cont + 12 time)
- `norm_params_file`: usar YAML nuevo (p. ej. `norm_params_2010_to_2020_featuremap_v2.yml`)

Formato de cada entrada del mapa:

```json
"co_3h": {
  "pollutant": "co",
  "mode": "stations",
  "stations": ["UIZ", "AJU", "..."],
  "transform": "3h"
},
"co_mda8": {
  "pollutant": "co",
  "mode": "ALL",
  "transform": "mda8"
}
```

### 3. Primer entrenamiento con CSV nuevo

```bash
# Borrar caché de esquema anterior
rm -f /home/psegura/netcdfs/TrainingData/pollution_data_*.pkl

cd /home/psegura/ensamble_repo_proj/ensamble_ai_pollution_forecast
python 4_train.py -c test_Parallel_featuremap_v2_3h_nox_off_psegura.json
```

---

## Verificación rápida tras clone

```bash
# 1. Dependencias mínimas
pip install pandas numpy psycopg2-binary

# 2. Ajustar DATA_ROOT en ambos scripts si no tienes /ZION montado

# 3. Solo features (si ya tienes imputed/)
python x_db_ml_features_csv.py

# 4. Comprobar un año
head -n1 /path/to/mlforecast/data_imputed_7_2020.csv | tr ',' '\n' | wc -l
# Esperado: 330 columnas (+ índice fecha)

# 5. Headers consistentes entre años
for y in 2018 2019 2020; do
  md5sum /path/to/mlforecast/data_imputed_7_${y}.csv | cut -c1-8
done
# El hash del header debe coincidir (o comparar con diff <(head -1 f1) <(head -1 f2))
```

---

## Estaciones operativas (30)

`UIZ`, `AJU`, `ATI`, `CUA`, `SFE`, `SAG`, `CUT`, `PED`, `TAH`, `GAM`, `IZT`, `CCA`, `HGM`, `LPR`, `MGH`, `CAM`, `FAC`, `TLA`, `MER`, `XAL`, `LLA`, `TLI`, `UAX`, `BJU`, `MPA`, `MON`, `NEZ`, `INN`, `AJM`, `VIF`

---

## Notas

- Ambos scripts son **autocontenidos**: no importan otros módulos del repo.
- Los CSV generados no se versionan en git (solo los scripts y esta documentación).
- Si cambias `POLLUTANT_SPECS`, regenera CSV **y** el JSON de entrenamiento (`input_features` + `pollution_feature_map`).
- `x_db_ml_features_dashboard.py` (visualización Dash) es opcional y no forma parte de este commit mínimo.
