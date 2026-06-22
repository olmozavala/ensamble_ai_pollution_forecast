# %% [markdown]
# # x_db_full_pipeline.py
#
# Pipeline completo autocontenido:
#
# 1. **Export** DB → CSV por estación (`x_make_csv_from_db_yml.py`)
# 2. **Merge** → `{year}_AllStations.csv` (`x2_MData.py`)
# 3. **Imputación** row_avg → persistencia → climatología (`imputation_7_fixed.py`)
#
# Salidas centralizadas bajo `{DATA_ROOT}/DataPollutionDB_CSV_YML/`:
# `export/`, `merged/`, `imputed/`, `climatology/`, `plots/`.
#
# ```bash
# python x_db_full_pipeline.py
# ```
#
# Requisitos pasos 1–2: `psycopg2`, `pandas`, `numpy`, `~/.netrc`.
# Paso 3 (solo si `RUN_IMPUTATION_ANALYSIS=True`): `matplotlib`, `seaborn`, `sklearn`, `scipy`.

from __future__ import annotations

import logging
import netrc
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2

log = logging.getLogger("x_db_full_pipeline")

# %%
# ── Parámetros (editar) ───────────────────────────────────────────────────────

_REPO_ROOT = Path(__file__).resolve().parent

# ── Rutas (todo bajo {DATA_ROOT}/DataPollutionDB_CSV_YML/) ───────────────────
DATA_ROOT: Path = Path("/ZION/AirPollutionData/Data")
PIPELINE_HUB: str = "DataPollutionDB_CSV_YML"
EXPORT_SUBFOLDER: str = "export"          # paso 1: cont_*_{EST}.csv
MERGED_SUBFOLDER: str = "merged"          # paso 2: {year}_AllStations.csv
IMPUTED_SUBFOLDER: str = "imputed"        # paso 3: data_imputed_7fix_*.csv
CLIMATOLOGY_SUBFOLDER: str = "climatology"
PLOTS_SUBFOLDER: str = "plots"

# ── Pasos ───────────────────────────────────────────────────────────────────
RUN_EXPORT: bool = True
RUN_MERGE: bool = True
RUN_IMPUTATION: bool = True
RUN_IMPUTATION_ANALYSIS: bool = False  # dendrogramas, plots, clustering (lento)

# ── PostgreSQL ────────────────────────────────────────────────────────────────
NETRC_HOST: str = "AMATE-OPERATIVO"
DATABASE: str = "contingencia"
START_DATE: str = "1980-01-01"
END_DATE: str = "2026-06-02"

POLLUTANT_TABLES: list[str] = [
    "cont_otres", "cont_pmco", "cont_pmdoscinco", "cont_nox", "cont_codos",
    "cont_co", "cont_nodos", "cont_no", "cont_sodos", "cont_pmdiez",
]

EXPORT_STATIONS: list[str] = [
    "ACO", "AJM", "AJU", "ARA", "ATI", "AZC", "BJU", "CAM", "CCA", "CES", "CFE",
    "CHO", "COR", "COY", "CUA", "CUI", "CUT", "DIC", "EAJ", "EDL", "FAC", "FAN",
    "GAM", "HAN", "HGM", "IBM", "IMP", "INN", "IZT", "LAA", "LAG", "LLA", "LOM",
    "LPR", "LVI", "MCM", "MER", "MGH", "MIN", "MON", "MPA", "NET", "NEZ", "PED",
    "PER", "PLA", "POT", "SAG", "SFE", "SHA", "SJA", "SNT", "SUR", "TAC", "TAH",
    "TAX", "TEC", "TLA", "TLI", "TPN", "UAX", "UIZ", "UNM", "VAL", "VIF", "XAL",
    "XCH",
]

MERGE_STATIONS: list[str] = [
    "UIZ", "AJU", "ATI", "CUA", "SFE", "SAG", "CUT", "PED", "TAH", "GAM",
    "IZT", "CCA", "HGM", "LPR", "MGH", "CAM", "FAC", "TLA", "MER", "XAL",
    "LLA", "TLI", "UAX", "BJU", "MPA", "MON", "NEZ", "INN", "AJM", "VIF",
]

# ── Merge ─────────────────────────────────────────────────────────────────────
MERGE_MODE: str = "by_year"
EXPORT_YEARS: list[int] = list(range(2000, 2027))
INCLUDE_TIME_FEATURES: bool = True
TIME_FEATURE_STYLE: str = "operativo"

# ── Imputación (imputation_7_fixed) ───────────────────────────────────────────
# Por defecto mismos años que merge; ajusta si quieres otro rango
IMPUTATION_START_YEAR: int = min(EXPORT_YEARS) if EXPORT_YEARS else 2000
IMPUTATION_END_YEAR: int = max(EXPORT_YEARS) if EXPORT_YEARS else 2025
ROW_AVG_MIN_VALID: int = 5
CLIMATOLOGY_REFERENCE_YEAR: int = 2012
IMPUTATION_OUTPUT_PREFIX: str = "data_imputed_7fix"  # → data_imputed_7fix_{year}.csv

INDEX_LABEL: str = "fecha"
CSV_FLOAT_FORMAT: str = "%.2f"
INTEGER_POLLUTANTS: set[str] = {"cont_otres"}


def resolve_config() -> dict:
    hub = DATA_ROOT / PIPELINE_HUB
    export_folder = hub / EXPORT_SUBFOLDER
    merged_folder = hub / MERGED_SUBFOLDER
    imputed_folder = hub / IMPUTED_SUBFOLDER
    climatology_folder = hub / CLIMATOLOGY_SUBFOLDER
    plots_folder = hub / PLOTS_SUBFOLDER
    return {
        "run_export": RUN_EXPORT,
        "run_merge": RUN_MERGE,
        "run_imputation": RUN_IMPUTATION,
        "run_imputation_analysis": RUN_IMPUTATION_ANALYSIS,
        "pipeline_hub": hub,
        "db_output_folder": export_folder,
        "db_input_folder": export_folder,
        "merged_output_folder": merged_folder,
        "merged_input_folder": merged_folder,
        "imputed_output_folder": imputed_folder,
        "climatology_folder": climatology_folder,
        "plots_folder": plots_folder,
        "pollutants": list(POLLUTANT_TABLES),
        "export_stations": list(EXPORT_STATIONS),
        "merge_stations": list(MERGE_STATIONS),
        "start_date": START_DATE,
        "end_date": END_DATE,
        "netrc_host": NETRC_HOST,
        "database": DATABASE,
        "years": list(EXPORT_YEARS),
        "imputation_start_year": IMPUTATION_START_YEAR,
        "imputation_end_year": IMPUTATION_END_YEAR,
        "row_avg_min_valid": ROW_AVG_MIN_VALID,
        "climatology_reference_year": CLIMATOLOGY_REFERENCE_YEAR,
        "imputation_output_prefix": IMPUTATION_OUTPUT_PREFIX,
        "merge_mode": MERGE_MODE.strip().lower(),
        "include_time_features": INCLUDE_TIME_FEATURES,
        "time_feature_style": TIME_FEATURE_STYLE.strip().lower(),
        "index_label": INDEX_LABEL,
        "float_format": CSV_FLOAT_FORMAT,
        "integer_pollutants": set(INTEGER_POLLUTANTS),
    }


def ensure_folder(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# %%
# ── Paso 1: export DB ─────────────────────────────────────────────────────────

def get_postgres_conn(netrc_host: str, database: str):
    log.info("Conectando a base de datos...")
    login, account, password = netrc.netrc().hosts[netrc_host]
    host = account
    try:
        conn = psycopg2.connect(
            database=database, user=login, host=host, password=password,
        )
    except Exception as exc:
        raise ConnectionError(f"No se pudo conectar a {host}/{database}") from exc
    log.info("Conectado a %s", host)
    return conn


def get_pollutant_from_date_range(
    conn, table: str, start_date: str, end_date: str, stations: list[str],
) -> list[tuple]:
    stations_str = "','".join(stations)
    sql = f"""
        SELECT fecha, val, id_est FROM {table}
        WHERE fecha BETWEEN '{start_date}' AND '{end_date}'
          AND id_est IN ('{stations_str}')
        ORDER BY fecha;
    """
    with conn.cursor() as cur:
        cur.execute(sql)
        return cur.fetchall()


def export_pollution_csvs(cfg: dict) -> None:
    output_folder: Path = cfg["db_output_folder"]
    ensure_folder(output_folder)
    conn = get_postgres_conn(cfg["netrc_host"], cfg["database"])
    try:
        for cur_station in cfg["export_stations"]:
            print(f" ====================== {cur_station} ====================== ")
            for cur_pollutant in cfg["pollutants"]:
                print(f"\t ---------------------- {cur_pollutant} ---------------------- ")
                cur_data = np.array(
                    get_pollutant_from_date_range(
                        conn, cur_pollutant, cfg["start_date"],
                        cfg["end_date"], [cur_station],
                    )
                )
                if len(cur_data) == 0:
                    print("\t\t Warning!!!  NO DATA")
                    continue
                dates = np.array([row[0] for row in cur_data])
                print(f"\tTotal rows {cur_station}-{cur_pollutant}: {len(dates)}")
                pd.DataFrame({cur_pollutant: cur_data[:, 1]}, index=dates).to_csv(
                    output_folder / f"{cur_pollutant}_{cur_station}.csv",
                    index_label=cfg["index_label"],
                )
    finally:
        conn.close()


# %%
# ── Paso 2: merge ─────────────────────────────────────────────────────────────

_DAY_S = 24 * 3600
_WEEK_S = _DAY_S * 7
_YEAR_S = _DAY_S * 365.2425

_TIME_SPECS_OPERATIVO: list[tuple[str, float, str]] = [
    ("half_sin_day", _DAY_S / 2, "sin"), ("half_cos_day", _DAY_S / 2, "cos"),
    ("half_sin_week", _WEEK_S / 2, "sin"), ("half_cos_week", _WEEK_S / 2, "cos"),
    ("half_sin_year", _YEAR_S / 2, "sin"), ("half_cos_year", _YEAR_S / 2, "cos"),
    ("sin_day", _DAY_S, "sin"), ("cos_day", _DAY_S, "cos"),
    ("sin_week", _WEEK_S, "sin"), ("cos_week", _WEEK_S, "cos"),
    ("sin_year", _YEAR_S, "sin"), ("cos_year", _YEAR_S, "cos"),
]
_TIME_SPECS_MERGE_DATA: list[tuple[str, float, str]] = [
    ("sin_day", _DAY_S, "sin"), ("cos_day", _DAY_S, "cos"),
    ("sin_year", _YEAR_S, "sin"), ("cos_year", _YEAR_S, "cos"),
    ("sin_week", _WEEK_S, "sin"), ("cos_week", _WEEK_S, "cos"),
]


def pollutant_station_column(pollutant: str, station: str) -> str:
    return f"{pollutant}_{station}"


def compute_time_features(datetimes: pd.DatetimeIndex, style: str) -> pd.DataFrame:
    specs = _TIME_SPECS_OPERATIVO if style == "operativo" else _TIME_SPECS_MERGE_DATA
    if style not in ("operativo", "merge_data"):
        raise ValueError(f"TIME_FEATURE_STYLE inválido: {style!r}")
    ix = pd.DatetimeIndex(datetimes)
    ts = ix.asi8.astype(np.float64) / 1e9
    two_pi = 2 * np.pi
    return pd.DataFrame(
        {n: (np.sin if f == "sin" else np.cos)(two_pi * ts / p) for n, p, f in specs},
        index=ix,
    )


def read_station_csv(path: Path, pollutant: str, station: str, integer_pollutants: set[str]):
    if not path.is_file():
        return None
    dtype = {pollutant: np.int32} if pollutant in integer_pollutants else None
    data = pd.read_csv(path, index_col=0, parse_dates=True, dtype=dtype)
    return data.rename(columns={pollutant: pollutant_station_column(pollutant, station)})


def build_pollutant_panel(cfg: dict, pollutant: str, *, wide_column_names: bool):
    not_found: list[str] = []
    panel = None
    for station in cfg["merge_stations"]:
        chunk = read_station_csv(
            cfg["db_input_folder"] / f"{pollutant}_{station}.csv",
            pollutant, station, cfg["integer_pollutants"],
        )
        if chunk is None:
            not_found.append(station)
            continue
        if not wide_column_names:
            chunk = chunk.rename(columns={chunk.columns[0]: station})
        panel = chunk if panel is None else pd.concat([panel, chunk], axis=1)
    if panel is None:
        raise FileNotFoundError(f"Sin CSV para {pollutant}")
    return panel, not_found


def build_all_pollutants_panel(cfg: dict):
    not_found_by_table: dict[str, list[str]] = {}
    panel = None
    for pollutant in cfg["pollutants"]:
        try:
            chunk, not_found = build_pollutant_panel(cfg, pollutant, wide_column_names=True)
        except FileNotFoundError:
            not_found_by_table[pollutant] = list(cfg["merge_stations"])
            continue
        not_found_by_table[pollutant] = not_found
        panel = chunk if panel is None else pd.concat([panel, chunk], axis=1)
    if panel is None:
        raise FileNotFoundError("Sin datos para ningún contaminante")
    return panel, not_found_by_table


def finalize_merged_frame(cfg: dict, values: pd.DataFrame) -> pd.DataFrame:
    if not cfg["include_time_features"]:
        return values
    return pd.concat([
        compute_time_features(pd.to_datetime(values.index), cfg["time_feature_style"]),
        values,
    ], axis=1)


def filter_year(panel: pd.DataFrame, year: int) -> pd.DataFrame:
    datetimes = pd.to_datetime(panel.index)
    mask = (datetimes >= np.datetime64(f"{year}-01-01")) & (
        datetimes < np.datetime64(f"{year + 1}-01-01")
    )
    return panel.loc[mask]


def merge_by_year(cfg: dict) -> None:
    ensure_folder(cfg["merged_output_folder"])
    panel, not_found_by_table = build_all_pollutants_panel(cfg)
    for pollutant, missing in not_found_by_table.items():
        if missing:
            print(f"\t{pollutant}: estaciones sin CSV: {missing}")
    for year in cfg["years"]:
        filtered = filter_year(panel, year)
        if filtered.empty:
            continue
        out_path = cfg["merged_output_folder"] / f"{year}_AllStations.csv"
        finalize_merged_frame(cfg, filtered).to_csv(
            out_path, float_format=cfg["float_format"], index_label=cfg["index_label"],
        )
        print(f"\tSaved merge: {out_path}")


def run_merge(cfg: dict) -> None:
    if cfg["merge_mode"] not in ("by_year", "by_year_all_pollutants"):
        raise ValueError("Este pipeline full solo soporta MERGE_MODE='by_year'")
    merge_by_year(cfg)


# %%
# ── Paso 3: imputación (imputation_7_fixed) ─────────────────────────────────────

def read_merged_files(input_folder: Path, start_year: int, end_year: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for year in range(start_year, end_year + 1):
        path = input_folder / f"{year}_AllStations.csv"
        if not path.is_file():
            log.warning("Archivo no encontrado (se omite): %s", path)
            continue
        frames.append(pd.read_csv(path, index_col=0))
    if not frames:
        raise FileNotFoundError(
            f"No hay {{year}}_AllStations.csv en {input_folder} "
            f"para {start_year}–{end_year}"
        )
    return pd.concat(frames, axis=0)


def normalize_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.index = pd.to_datetime(
        out.index.to_series().apply(
            lambda x: f"{x} 00:00:00" if len(str(x)) == 10 else str(x)
        )
    )
    return out


def drop_wrf_columns(df: pd.DataFrame) -> pd.DataFrame:
    wrf_cols = [c for c in df.columns if any(c.endswith(f"_h{h}") for h in range(24))]
    return df.drop(columns=wrf_cols) if wrf_cols else df


def create_complete_time_index(start_year: int, end_year: int) -> pd.DatetimeIndex:
    return pd.date_range(
        start=f"{start_year}-01-01 00:00:00",
        end=f"{end_year}-12-31 23:00:00",
        freq="h",
    )


def detect_contaminant_groups(df: pd.DataFrame) -> list[str]:
    groups: set[str] = set()
    for col in df.columns:
        if not col.startswith("cont_"):
            continue
        parts = col.split("_")
        if len(parts) >= 2:
            groups.add(f"cont_{parts[1]}_")
    return sorted(groups)


def generate_climatology(
    df: pd.DataFrame,
    value_columns: list[str],
    reference_year: int,
) -> pd.DataFrame:
    climatology = pd.DataFrame(
        index=pd.date_range(
            start=f"{reference_year}-01-01 00:00:00",
            end=f"{reference_year}-12-31 23:00:00",
            freq="h",
        )
    )
    for col in value_columns:
        hourly_means = df.groupby([df.index.month, df.index.day, df.index.hour])[col].mean()
        hourly_means_dict: dict[pd.Timestamp, float] = {}
        for (month, day, hour), value in hourly_means.items():
            try:
                date = pd.Timestamp(f"{reference_year}-{month:02d}-{day:02d} {hour:02d}:00:00")
                if not pd.isna(value):
                    hourly_means_dict[date] = value
            except ValueError:
                continue
        climatology[col] = pd.Series(hourly_means_dict)
        climatology[col] = climatology[col].rolling(window=3, center=True, min_periods=1).mean()
        if not climatology[col].isna().all():
            climatology.loc[climatology.index[0], col] = (
                climatology.loc[climatology.index[-1], col]
                + climatology.loc[climatology.index[0], col]
                + climatology.loc[climatology.index[1], col]
            ) / 3
            climatology.loc[climatology.index[-1], col] = (
                climatology.loc[climatology.index[-2], col]
                + climatology.loc[climatology.index[-1], col]
                + climatology.loc[climatology.index[0], col]
            ) / 3
    return climatology


def impute_with_row_avg(
    df: pd.DataFrame, value_columns: list[str], min_valid: int,
) -> pd.DataFrame:
    df_imputed = df.copy()
    for col in value_columns:
        flag_col = f"i_{col}"
        mask_missing = df_imputed[flag_col] == -1
        mask_enough = df[value_columns].notna().sum(axis=1) > min_valid
        mask = mask_missing & mask_enough
        if mask.any():
            row_avg = df_imputed.loc[mask, value_columns].mean(axis=1, skipna=True)
            df_imputed.loc[mask, col] = row_avg
            df_imputed.loc[mask, flag_col] = "row_avg"
    return df_imputed


def impute_with_persistence(df: pd.DataFrame, value_columns: list[str]) -> pd.DataFrame:
    df_imputed = df.copy()
    for col in value_columns:
        flag_col = f"i_{col}"
        mask_missing = df_imputed[flag_col] == -1
        if not mask_missing.any():
            continue
        for idx in df_imputed.index[mask_missing]:
            prev_idx = idx - pd.Timedelta(days=1)
            if prev_idx in df_imputed.index:
                prev_val = df_imputed.loc[prev_idx, col]
                if pd.notna(prev_val):
                    df_imputed.loc[idx, col] = prev_val
                    df_imputed.loc[idx, flag_col] = "last_day_same_hour"
    return df_imputed


def impute_with_climatology(
    df: pd.DataFrame,
    climatology: pd.DataFrame,
    value_columns: list[str],
    reference_year: int,
) -> pd.DataFrame:
    df_imputed = df.copy()
    for col in value_columns:
        flag_col = f"i_{col}"
        mask_missing = df_imputed[flag_col] == -1
        for idx in df_imputed.index[mask_missing]:
            clim_idx = pd.Timestamp(
                f"{reference_year}-{idx.month:02d}-{idx.day:02d} {idx.hour:02d}:00:00"
            )
            df_imputed.loc[idx, col] = climatology.loc[clim_idx, col]
            df_imputed.loc[idx, flag_col] = "climatology"
    return df_imputed


def _group_slug(group_prefix: str) -> str:
    return group_prefix.replace("cont_", "").replace("_", "")


def run_imputation_analysis(
    cfg: dict,
    data_df: pd.DataFrame,
    data_imputed: pd.DataFrame,
    column_groups: list[str],
) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns  # noqa: F401
    from scipy.cluster.hierarchy import dendrogram, linkage
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    ensure_folder(cfg["plots_folder"])
    ensure_folder(cfg["climatology_folder"])
    resultados: dict = {}

    for group in column_groups:
        columns = [c for c in data_df.columns if c.startswith(group)]
        climatology = generate_climatology(
            data_df, columns, cfg["climatology_reference_year"],
        )
        x = climatology.T.fillna(climatology.T.mean())
        x = x.replace([np.inf, -np.inf], np.nan).fillna(x.mean())
        x_scaled = StandardScaler().fit_transform(x)
        x_scaled = np.nan_to_num(x_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        slug = _group_slug(group)
        linkage_matrix = linkage(x_scaled, method="ward")
        plt.figure(figsize=(12, 8))
        dendrogram(linkage_matrix, labels=climatology.columns, leaf_rotation=90)
        plt.title(f"Dendrograma {slug}")
        plt.tight_layout()
        plt.savefig(cfg["climatology_folder"] / f"dendrograma_{slug}.png")
        plt.close()

        clusters = KMeans(n_clusters=3, random_state=42).fit_predict(x_scaled)
        cluster_df = pd.DataFrame({"Estacion": climatology.columns, "Cluster": clusters})
        cluster_df.to_csv(cfg["climatology_folder"] / f"clusters_{slug}_7fix.csv", index=False)
        resultados[group] = {"clusters": cluster_df}

        plt.figure(figsize=(15, 6))
        for i in range(3):
            est = cluster_df.loc[cluster_df["Cluster"] == i, "Estacion"]
            plt.plot(climatology.index, climatology[est].mean(axis=1), label=f"Cluster {i}")
        plt.title(f"Perfiles cluster {slug}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(cfg["climatology_folder"] / f"perfiles_cluster_{slug}.png")
        plt.close()

        columns_plot = [c for c in data_imputed.columns if c.startswith(group) and not c.startswith(f"i_{group}")]
        plt.figure(figsize=(20, 6))
        for col in columns_plot:
            plt.plot(data_imputed.index, data_imputed[col], alpha=0.5, label=col)
        plt.title(f"Series imputadas {slug}")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(cfg["plots_folder"] / f"series_temporales_{slug}.png")
        plt.close()

    resumen = []
    for group, datos in resultados.items():
        dist = datos["clusters"]["Cluster"].value_counts().sort_index()
        resumen.append({
            "Contaminante": group,
            "Número de estaciones": len(datos["clusters"]),
            "Distribución de clusters": dist.to_dict(),
        })
    pd.DataFrame(resumen).to_csv(
        cfg["climatology_folder"] / "resumen_clustering_7fix.csv", index=False,
    )


def run_imputation(cfg: dict) -> pd.DataFrame:
    """Imputación por grupos cont_*; exporta CSV anual y full."""
    ensure_folder(cfg["imputed_output_folder"])
    ensure_folder(cfg["climatology_folder"])

    start_y = cfg["imputation_start_year"]
    end_y = cfg["imputation_end_year"]
    ref_y = cfg["climatology_reference_year"]
    prefix = cfg["imputation_output_prefix"]

    data_df = read_merged_files(cfg["merged_input_folder"], start_y, end_y)
    data_df = normalize_datetime_index(data_df)
    data_df = drop_wrf_columns(data_df)

    cont_cols = [c for c in data_df.columns if c.startswith("cont_")]
    total_nan = int(data_df[cont_cols].isna().sum().sum())
    log.info("NaN en cont_* antes de imputar: %d", total_nan)

    complete_index = create_complete_time_index(start_y, end_y)
    column_groups = detect_contaminant_groups(data_df)

    data_imputed = pd.DataFrame(index=complete_index)
    for col in cont_cols:
        data_imputed[col] = np.nan
        data_imputed[f"i_{col}"] = -1

    mask_observed = data_imputed.index.isin(data_df.index)
    for col in cont_cols:
        data_imputed.loc[mask_observed, col] = data_df[col]
        flags = data_df[col].apply(lambda x: "none" if pd.notna(x) else -1)
        data_imputed.loc[mask_observed, f"i_{col}"] = flags

    for group in column_groups:
        columns = [c for c in data_df.columns if c.startswith(group)]
        log.info("Imputando grupo %s (%d columnas)", group, len(columns))
        climatology = generate_climatology(data_df, columns, ref_y)
        data_imputed = impute_with_row_avg(
            data_imputed, columns, cfg["row_avg_min_valid"],
        )
        data_imputed = impute_with_persistence(data_imputed, columns)
        data_imputed = impute_with_climatology(
            data_imputed, climatology, columns, ref_y,
        )
        slug = _group_slug(group)
        climatology.to_csv(
            cfg["climatology_folder"] / f"climatology_{slug}_7fix.csv",
        )

    for col in cont_cols:
        data_df[col] = data_imputed[col]
        data_df[f"i_{col}"] = data_imputed[f"i_{col}"]

    # Versión limpia: sin WRF ni banderas no-cont
    data_clean = drop_wrf_columns(data_df.copy())
    non_cont_flags = [
        c for c in data_clean.columns
        if c.startswith("i_") and not c.startswith("i_cont_")
    ]
    data_clean = data_clean.drop(columns=non_cont_flags)

    data_clean.to_csv(cfg["imputed_output_folder"] / f"{prefix}_full.csv")
    for year in range(start_y, end_y + 1):
        yearly = data_clean[data_clean.index.year == year]
        if yearly.empty:
            continue
        out = cfg["imputed_output_folder"] / f"{prefix}_{year}.csv"
        yearly.to_csv(out)
        log.info("Imputado guardado: %s  shape=%s", out, yearly.shape)

    if cfg["run_imputation_analysis"]:
        run_imputation_analysis(cfg, data_df, data_imputed, column_groups)

    remaining = int(data_imputed[cont_cols].isna().sum().sum())
    log.info("NaN restantes en cont_*: %d", remaining)
    return data_clean


# %%
def run_pipeline(cfg: dict) -> None:
    steps: list[tuple[str, Callable[[], None]]] = []
    if cfg["run_export"]:
        steps.append(("export DB", lambda: export_pollution_csvs(cfg)))
    if cfg["run_merge"]:
        steps.append(("merge", lambda: run_merge(cfg)))
    if cfg["run_imputation"]:
        steps.append(("imputación", lambda: run_imputation(cfg)))

    if not steps:
        log.warning("Ningún paso activo (RUN_EXPORT/MERGE/IMPUTATION=False)")
        return

    for i, (name, fn) in enumerate(steps, start=1):
        log.info("=== Paso %d/%d: %s ===", i, len(steps), name)
        fn()


# %%
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    user_cfg = resolve_config()

    time_info = (
        "off" if not user_cfg["include_time_features"] else user_cfg["time_feature_style"]
    )
    log.info("Hub: %s", user_cfg["pipeline_hub"])
    log.info("  export: %s", user_cfg["db_output_folder"])
    log.info("  merged: %s", user_cfg["merged_output_folder"])
    log.info("  imputed: %s", user_cfg["imputed_output_folder"])
    log.info(
        "export=%s merge=%s impute=%s analysis=%s | merge=%s time=%s | "
        "impute_years=%s..%s",
        user_cfg["run_export"],
        user_cfg["run_merge"],
        user_cfg["run_imputation"],
        user_cfg["run_imputation_analysis"],
        user_cfg["merge_mode"],
        time_info,
        user_cfg["imputation_start_year"],
        user_cfg["imputation_end_year"],
    )
    run_pipeline(user_cfg)

# %%
