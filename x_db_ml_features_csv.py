# %% [markdown]
# # x_db_ml_features_csv.py
#
# Pipeline CSV ML forecast: transforma `data_imputed_7fix_{year}.csv` → `data_imputed_7_{year}.csv`
# compatible con `MLforecastFeatureMapDataLoader` (repo entrenamiento).
#
# - Agregación espacial/temporal **causal** (shift=1, sin ventanas centrales).
# - Esquema de columnas fijo entre años.
# - Config YAML-like inline (`POLLUTANT_SPECS`).
#
# ```bash
# python x_db_ml_features_csv.py
# ```
#
# Requisitos: `pandas`, `numpy`, `pyyaml` (opcional, solo export config).

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("x_db_ml_features_csv")

# %%
# ── Parámetros (editar) ───────────────────────────────────────────────────────

_REPO_ROOT = Path(__file__).resolve().parent

DATA_ROOT: Path = Path("/ZION/AirPollutionData/Data")
PIPELINE_HUB: str = "DataPollutionDB_CSV_YML"
IMPUTED_SUBFOLDER: str = "imputed"
MLFORECAST_SUBFOLDER: str = "mlforecast"

INPUT_PREFIX: str = "data_imputed_7fix"
OUTPUT_PREFIX: str = "data_imputed_7"
INDEX_LABEL: str = "fecha"
CSV_FLOAT_FORMAT: str = "%.2f"

PROCESS_YEARS: list[int] = list(range(2000, 2027))
LOOKBACK_HOURS: int = 32  # mda8: 24 + 8

MERGE_STATIONS: list[str] = [
    "UIZ", "AJU", "ATI", "CUA", "SFE", "SAG", "CUT", "PED", "TAH", "GAM",
    "IZT", "CCA", "HGM", "LPR", "MGH", "CAM", "FAC", "TLA", "MER", "XAL",
    "LLA", "TLI", "UAX", "BJU", "MPA", "MON", "NEZ", "INN", "AJM", "VIF",
]

SLUG_ORDER: tuple[str, ...] = (
    "otres", "pmdoscinco", "co", "nodos", "no", "sodos", "pmdiez", "nox",
)

TIME_COLS_ORDER: tuple[str, ...] = (
    "half_sin_day", "half_cos_day", "half_sin_week", "half_cos_week",
    "half_sin_year", "half_cos_year",
    "sin_day", "cos_day", "sin_week", "cos_week", "sin_year", "cos_year",
)

TRANSFORM_WINDOWS: dict[str, int] = {"3h": 3, "6h": 6}
MDA8_MA8_WINDOW: int = 8
MDA8_MA8_MIN_PERIODS: int = 6
MDA8_MAX_WINDOW: int = 24
MDA8_FLAG_LOOKBACK: int = 32

# ── POLLUTANT_SPECS (YAML-like) ───────────────────────────────────────────────
POLLUTANT_SPECS: dict[str, dict] = {
    "co": {
        "enabled": True,
        "spatial": "per_station",
        "transforms": ["3h"],
        "extra": [{"spatial": "ALL", "transform": "mda8"}],
        "drop_sources": True,
    },
    "no": {
        "enabled": True,
        "spatial": "per_station",
        "transforms": ["3h"],
        "extra": [{"spatial": "ALL", "transform": "mda8"}],
        "drop_sources": True,
    },
    "nodos": {
        "enabled": True,
        "spatial": "ALL",
        "transforms": ["3h"],
        "extra": [
            {"spatial": "ALL", "transform": "mda8"},
        ],
        "drop_sources": True,
    },
    "nox": {
        "enabled": False,
        "spatial": "per_station",
        "transforms": ["3h"],
        "drop_sources": True,
    },
    "sodos": {
        "enabled": True,
        "spatial": "ALL",
        "transforms": [],
        "extra": [{"spatial": "ALL", "transform": "mda8"}],
        "drop_sources": True,
    },
    "otres": {
        "enabled": True,
        "spatial": "per_station",
        "transforms": [],
        "extra": [{"spatial": "ALL", "transform": "mda8"}],
        "drop_sources": False,
    },
    "pmdiez": {
        "enabled": True,
        "spatial": "per_station",
        "transforms": ["3h"],
        "extra": [{"spatial": "ALL", "transform": "mda8"}],
        "drop_sources": True,
    },
    "pmdoscinco": {
        "enabled": True,
        "spatial": "per_station",
        "transforms": ["3h"],
        "extra": [{"spatial": "ALL", "transform": "mda8"}],
        "drop_sources": True,
    },
}

PASSTHROUGH_SLUGS: list[str] = []


# %%
# ── Config / schema ─────────────────────────────────────────────────────────────

@dataclass
class ColumnPlan:
    sources_to_read: set[str] = field(default_factory=set)
    output_columns: list[str] = field(default_factory=list)
    column_order: list[str] = field(default_factory=list)


def resolve_config() -> dict:
    hub = DATA_ROOT / PIPELINE_HUB
    return {
        "pipeline_hub": hub,
        "input_folder": hub / IMPUTED_SUBFOLDER,
        "output_folder": hub / MLFORECAST_SUBFOLDER,
        "input_pattern": f"{INPUT_PREFIX}_{{year}}.csv",
        "output_pattern": f"{OUTPUT_PREFIX}_{{year}}.csv",
        "years": list(PROCESS_YEARS),
        "stations": list(MERGE_STATIONS),
        "slug_order": list(SLUG_ORDER),
        "pollutant_specs": {k: dict(v) for k, v in POLLUTANT_SPECS.items()},
        "passthrough_slugs": list(PASSTHROUGH_SLUGS),
        "time_cols": list(TIME_COLS_ORDER),
        "lookback_hours": LOOKBACK_HOURS,
        "index_label": INDEX_LABEL,
        "float_format": CSV_FLOAT_FORMAT,
    }


def validate_config(cfg: dict) -> None:
    if not cfg["years"]:
        raise ValueError("PROCESS_YEARS vacío")
    for slug, spec in cfg["pollutant_specs"].items():
        if not spec.get("enabled", True):
            continue
        spatial = spec.get("spatial")
        if spatial not in ("per_station", "ALL"):
            raise ValueError(f"{slug}: spatial inválido {spatial!r}")
        for t in spec.get("transforms", []) + [e.get("transform") for e in spec.get("extra", [])]:
            if t and t not in (*TRANSFORM_WINDOWS, "mda8"):
                raise ValueError(f"{slug}: transform desconocido {t!r}")


def _transform_suffix(transform: str) -> str:
    return "_mda8" if transform == "mda8" else f"_{transform}"


def _cont_name(slug: str, station: str, transform: str | None = None) -> str:
    if transform:
        return f"cont_{slug}_{station}{_transform_suffix(transform)}"
    return f"cont_{slug}_{station}"


def _i_cont_name(cont_name: str) -> str:
    return f"i_{cont_name}"


def _pair_columns(cont_name: str) -> tuple[str, str]:
    return cont_name, _i_cont_name(cont_name)


def _source_cols(slug: str, stations: list[str]) -> list[str]:
    cols: list[str] = []
    for st in stations:
        cols.append(_cont_name(slug, st))
        cols.append(_i_cont_name(_cont_name(slug, st)))
    return cols


def _output_cols_for_layer(
    slug: str,
    spatial: str,
    transforms: list[str],
    stations: list[str],
) -> list[str]:
    """Genera pares cont/i_cont para una capa spatial+transforms."""
    out: list[str] = []
    if spatial == "per_station":
        for transform in transforms or [None]:
            for st in sorted(stations):
                name = _cont_name(slug, st, transform)
                out.extend(_pair_columns(name))
    elif spatial == "ALL":
        for transform in transforms or [None]:
            name = _cont_name(slug, "ALL", transform)
            out.extend(_pair_columns(name))
    return out


def build_output_schema(cfg: dict) -> ColumnPlan:
    stations = cfg["stations"]
    specs = cfg["pollutant_specs"]
    plan = ColumnPlan()

    for slug in cfg["slug_order"]:
        spec = specs.get(slug)
        if not spec or not spec.get("enabled", True):
            continue

        plan.sources_to_read.update(_source_cols(slug, stations))

        spatial = spec["spatial"]
        transforms = list(spec.get("transforms", []))
        drop_sources = spec.get("drop_sources", False)

        if spatial == "per_station":
            if not drop_sources and not transforms:
                plan.output_columns.extend(_output_cols_for_layer(slug, "per_station", [], stations))
            elif not drop_sources and transforms:
                plan.output_columns.extend(_output_cols_for_layer(slug, "per_station", [], stations))
            for t in transforms:
                plan.output_columns.extend(_output_cols_for_layer(slug, "per_station", [t], stations))

        if spatial == "ALL":
            plan.output_columns.extend(_output_cols_for_layer(slug, "ALL", transforms or [None], stations))

        for extra in spec.get("extra", []):
            ex_t = extra.get("transform")
            ex_sp = extra.get("spatial", "ALL")
            if ex_t:
                plan.output_columns.extend(_output_cols_for_layer(slug, ex_sp, [ex_t], stations))

    for slug in cfg["passthrough_slugs"]:
        plan.sources_to_read.update(_source_cols(slug, stations))
        plan.output_columns.extend(_output_cols_for_layer(slug, "per_station", [None], stations))

    time_cols = [c for c in cfg["time_cols"]]
    plan.column_order = time_cols + plan.output_columns
    return plan


# %%
# ── Flags y transforms ────────────────────────────────────────────────────────

def impute_flag_to_binary(series: pd.Series) -> pd.Series:
    def _f(v) -> int:
        if pd.isna(v):
            return 0
        if isinstance(v, str):
            s = v.strip().lower()
            return 0 if s in ("none", "-1", "0") else 1
        try:
            return 0 if float(v) <= 0 else 1
        except (TypeError, ValueError):
            return 0

    return series.map(_f).astype(np.int8)


def rolling_causal_mean(values: pd.Series, window: int, min_periods: int = 1) -> pd.Series:
    return values.shift(1).rolling(window, min_periods=min_periods).mean()


def rolling_causal_flag_max(flags: pd.Series, window: int, min_periods: int = 1) -> pd.Series:
    rolled = flags.shift(1).rolling(window, min_periods=min_periods).max()
    return rolled.fillna(0).astype(np.int8)


def mda8_max_in_24h(values: pd.Series, flags: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    MA8 causal (8h, min_periods=6) → max sobre ventana -24h (sin incluir t).
    """
    ma8 = rolling_causal_mean(values, MDA8_MA8_WINDOW, min_periods=MDA8_MA8_MIN_PERIODS)
    mda8 = ma8.shift(1).rolling(MDA8_MAX_WINDOW, min_periods=1).max()
    mda8_flag = rolling_causal_flag_max(flags, MDA8_FLAG_LOOKBACK, min_periods=1)
    return mda8, mda8_flag


def spatial_mean_all(
    df: pd.DataFrame,
    value_cols: list[str],
    flag_cols: list[str],
) -> tuple[pd.Series, pd.Series]:
    present_val = [c for c in value_cols if c in df.columns]
    present_flag = [c for c in flag_cols if c in df.columns]
    if not present_val:
        idx = df.index
        return pd.Series(np.nan, index=idx), pd.Series(0, index=idx, dtype=np.int8)

    values = df[present_val].mean(axis=1, skipna=True)
    if present_flag:
        flags_bin = df[present_flag].apply(impute_flag_to_binary)
        imputed = (flags_bin.max(axis=1) > 0).astype(np.int8)
    else:
        imputed = pd.Series(0, index=df.index, dtype=np.int8)
    return values, imputed


def apply_transform(
    values: pd.Series,
    flags: pd.Series,
    transform: str,
) -> tuple[pd.Series, pd.Series]:
    if transform == "mda8":
        return mda8_max_in_24h(values, flags)

    window = TRANSFORM_WINDOWS.get(transform)
    if window is None:
        raise ValueError(f"Transform desconocido: {transform!r}")

    out_val = rolling_causal_mean(values, window, min_periods=1)
    out_flag = rolling_causal_flag_max(flags, window, min_periods=1)
    return out_val, out_flag


def ensure_folder(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# %%
# ── Carga ─────────────────────────────────────────────────────────────────────

def normalize_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    idx = out.index.astype(str)
    idx = idx.where(idx.str.contains(":"), idx + " 00:00:00")
    out.index = pd.to_datetime(idx)
    out.index.name = INDEX_LABEL
    return out.sort_index()


def _drop_unwanted_columns(df: pd.DataFrame) -> pd.DataFrame:
    drop: list[str] = []
    for c in df.columns:
        if c.startswith("met_"):
            drop.append(c)
        elif any(c.endswith(f"_h{h}") for h in range(24)):
            drop.append(c)
    return df.drop(columns=drop, errors="ignore")


def load_year_csv(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, index_col=0)
    df = normalize_datetime_index(_drop_unwanted_columns(df))
    if columns is not None:
        keep = [c for c in columns if c in df.columns]
        df = df[keep]
    return df


def load_with_lookback(
    year: int,
    cfg: dict,
    plan: ColumnPlan,
) -> pd.DataFrame:
    input_folder = cfg["input_folder"]
    pattern = cfg["input_pattern"]
    lookback = cfg["lookback_hours"]
    time_cols = cfg["time_cols"]

    read_cols = sorted(plan.sources_to_read | set(time_cols))
    cur_path = input_folder / pattern.format(year=year)
    df_cur = load_year_csv(cur_path)
    avail = [c for c in read_cols if c in df_cur.columns]
    df_cur = df_cur[avail]

    if year <= min(cfg["years"]):
        return df_cur

    prev_path = input_folder / pattern.format(year=year - 1)
    if not prev_path.is_file():
        log.warning("Sin año anterior %s; bordes con historial parcial", year - 1)
        return df_cur

    df_prev = load_year_csv(prev_path)
    avail_prev = [c for c in read_cols if c in df_prev.columns]
    df_prev = df_prev[avail_prev]

    year_start = pd.Timestamp(f"{year}-01-01 00:00:00")
    tail_start = year_start - pd.Timedelta(hours=lookback)
    df_tail = df_prev.loc[df_prev.index >= tail_start]

    common_cols = sorted(set(df_cur.columns) & set(df_tail.columns))
    if not common_cols:
        return df_cur

    combined = pd.concat([df_tail[common_cols], df_cur[common_cols]], axis=0)
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    return combined


# %%
# ── Procesamiento por contaminante ────────────────────────────────────────────

def _station_value_flag(
    df: pd.DataFrame,
    slug: str,
    station: str,
) -> tuple[pd.Series, pd.Series]:
    vcol = _cont_name(slug, station)
    fcol = _i_cont_name(vcol)
    if vcol not in df.columns:
        nan = pd.Series(np.nan, index=df.index)
        zero = pd.Series(0, index=df.index, dtype=np.int8)
        return nan, zero
    values = df[vcol].astype(float)
    flags = impute_flag_to_binary(df[fcol]) if fcol in df.columns else pd.Series(0, index=df.index, dtype=np.int8)
    return values, flags


def process_pollutant(
    df: pd.DataFrame,
    slug: str,
    spec: dict,
    stations: list[str],
) -> dict[str, pd.Series]:
    """Devuelve dict cont_col → Series (valores + flags bajo i_cont_*)."""
    out: dict[str, pd.Series] = {}
    spatial = spec["spatial"]
    transforms = list(spec.get("transforms", []))
    drop_sources = spec.get("drop_sources", False)

    value_cols = [_cont_name(slug, st) for st in stations]
    flag_cols = [_i_cont_name(c) for c in value_cols]

    def emit_pair(cont: str, values: pd.Series, flags: pd.Series) -> None:
        out[cont] = values
        out[_i_cont_name(cont)] = flags.astype(np.int8)

    # ── Capa per_station ────────────────────────────────────────────────────
    if spatial == "per_station":
        if not drop_sources and not transforms:
            for st in stations:
                v, f = _station_value_flag(df, slug, st)
                emit_pair(_cont_name(slug, st), v, f)

        for transform in transforms:
            for st in stations:
                v, f = _station_value_flag(df, slug, st)
                tv, tf = apply_transform(v, f, transform)
                emit_pair(_cont_name(slug, st, transform), tv, tf)

    # ── Capa ALL (spatial) ──────────────────────────────────────────────────
    if spatial == "ALL":
        v_all, f_all = spatial_mean_all(df, value_cols, flag_cols)
        if not transforms:
            emit_pair(_cont_name(slug, "ALL"), v_all, f_all)
        for transform in transforms:
            tv, tf = apply_transform(v_all, f_all, transform)
            emit_pair(_cont_name(slug, "ALL", transform), tv, tf)

    # ── Extras (ALL o per_station) ───────────────────────────────────────────
    for extra in spec.get("extra", []):
        ex_spatial = extra.get("spatial", "ALL")
        ex_transform = extra.get("transform")
        if not ex_transform:
            continue
        if ex_spatial == "ALL":
            v_all, f_all = spatial_mean_all(df, value_cols, flag_cols)
            tv, tf = apply_transform(v_all, f_all, ex_transform)
            emit_pair(_cont_name(slug, "ALL", ex_transform), tv, tf)
        elif ex_spatial == "per_station":
            for st in stations:
                v, f = _station_value_flag(df, slug, st)
                tv, tf = apply_transform(v, f, ex_transform)
                emit_pair(_cont_name(slug, st, ex_transform), tv, tf)

    return out


def process_buffer(
    df: pd.DataFrame,
    cfg: dict,
) -> dict[str, pd.Series]:
    """Aplica todos los contaminantes habilitados sobre el buffer completo."""
    merged: dict[str, pd.Series] = {}
    for slug in cfg["slug_order"]:
        spec = cfg["pollutant_specs"].get(slug)
        if not spec or not spec.get("enabled", True):
            continue
        part = process_pollutant(df, slug, spec, cfg["stations"])
        merged.update(part)
    return merged


def slice_year(series_dict: dict[str, pd.Series], year: int) -> dict[str, pd.Series]:
    year_start = pd.Timestamp(f"{year}-01-01 00:00:00")
    year_end = pd.Timestamp(f"{year + 1}-01-01 00:00:00")
    return {
        k: v.loc[(v.index >= year_start) & (v.index < year_end)]
        for k, v in series_dict.items()
    }


def finalize_output_df(
    df_time: pd.DataFrame,
    series_dict: dict[str, pd.Series],
    plan: ColumnPlan,
) -> pd.DataFrame:
    idx = df_time.index
    frames: dict[str, pd.Series] = {}
    for col in plan.column_order:
        if col in df_time.columns:
            frames[col] = df_time[col]
        elif col in series_dict:
            frames[col] = series_dict[col]
        elif col.startswith("cont_"):
            frames[col] = pd.Series(np.nan, index=idx)
        else:
            frames[col] = pd.Series(0, index=idx, dtype=np.int8)

    out = pd.DataFrame(frames, index=idx)

    cont_cols = [c for c in out.columns if c.startswith("cont_")]
    n_nan = int(out[cont_cols].isna().sum().sum()) if cont_cols else 0
    if n_nan:
        log.warning("Rellenando %d NaN en cont_* con 0.0", n_nan)
        out[cont_cols] = out[cont_cols].fillna(0.0)

    i_cols = [c for c in out.columns if c.startswith("i_cont_")]
    out[i_cols] = out[i_cols].fillna(0).astype(np.int8)

    return out


def process_year(year: int, cfg: dict, plan: ColumnPlan) -> pd.DataFrame | None:
    input_path = cfg["input_folder"] / cfg["input_pattern"].format(year=year)
    if not input_path.is_file():
        log.warning("Input no encontrado (omitido): %s", input_path)
        return None

    df_buf = load_with_lookback(year, cfg, plan)
    series_all = process_buffer(df_buf, cfg)
    series_year = slice_year(series_all, year)

    time_avail = [c for c in cfg["time_cols"] if c in df_buf.columns]
    year_start = pd.Timestamp(f"{year}-01-01 00:00:00")
    year_end = pd.Timestamp(f"{year + 1}-01-01 00:00:00")
    mask = (df_buf.index >= year_start) & (df_buf.index < year_end)
    df_time = df_buf.loc[mask, time_avail].copy()

    return finalize_output_df(df_time, series_year, plan)


# %%
# ── Validación ────────────────────────────────────────────────────────────────

def validate_written_headers(output_folder: Path, pattern: str, years: list[int]) -> bool:
    headers: dict[int, list[str]] = {}
    for year in years:
        path = output_folder / pattern.format(year=year)
        if not path.is_file():
            continue
        headers[year] = list(pd.read_csv(path, nrows=0).columns)

    if not headers:
        log.warning("No hay archivos escritos para validar headers")
        return False

    ref_year = next(iter(headers))
    ref_cols = headers[ref_year]
    ok = True
    for year, cols in headers.items():
        if cols != ref_cols:
            log.error("Header distinto en %s (ref %s): %d vs %d cols", year, ref_year, len(cols), len(ref_cols))
            ok = False
    if ok:
        log.info("Headers idénticos en %d archivos (%d columnas)", len(headers), len(ref_cols))
    return ok


def parse_cont_column(col: str) -> dict | None:
    """Parsea cont_{slug}_{station|ALL}[_{3h|6h|mda8}]."""
    if not col.startswith("cont_"):
        return None
    parts = col.split("_")
    if len(parts) < 3:
        return None
    slug = parts[1]
    tail = "_".join(parts[2:])
    transform = "raw"
    for suffix in ("mda8", "3h", "6h"):
        token = f"_{suffix}"
        if tail.endswith(token):
            transform = suffix
            tail = tail[: -len(token)]
            break
    spatial = "ALL" if tail == "ALL" else "per_station"
    return {
        "slug": slug,
        "station": tail,
        "spatial": spatial,
        "transform": transform,
    }


def _format_station_keys(stations: list[str], indent: int = 6, per_line: int = 10) -> None:
    """Imprime lista completa de claves de estación, varias por línea."""
    if not stations:
        print(" " * indent + "(ninguna)")
        return
    pad = " " * indent
    for i in range(0, len(stations), per_line):
        chunk = stations[i : i + per_line]
        print(pad + ", ".join(chunk))


def source_stations_for_slug(plan: ColumnPlan, slug: str) -> list[str]:
    """Estaciones leídas del CSV imputado para un slug (claves fuente)."""
    prefix = f"cont_{slug}_"
    found: set[str] = set()
    for col in plan.sources_to_read:
        if not col.startswith(prefix):
            continue
        meta = parse_cont_column(col)
        if meta and meta["spatial"] == "per_station":
            found.add(meta["station"])
    return sorted(found)


def output_stations_by_slug(
    cont_cols: list[str],
) -> dict[str, dict[str, dict[str, list[str]]]]:
    """slug → transform → spatial → [estaciones]."""
    by_slug: dict[str, dict[str, dict[str, list[str]]]] = {}
    for col in cont_cols:
        meta = parse_cont_column(col)
        if not meta:
            continue
        slug = meta["slug"]
        by_slug.setdefault(slug, {}).setdefault(meta["transform"], {}).setdefault(
            meta["spatial"], []
        ).append(meta["station"])
    for slug in by_slug:
        for transform in by_slug[slug]:
            for spatial in by_slug[slug][transform]:
                by_slug[slug][transform][spatial] = sorted(set(by_slug[slug][transform][spatial]))
    return by_slug


def preserved_stations_per_slug(
    layers: dict[str, dict[str, list[str]]],
) -> list[str]:
    """Unión de claves per_station presentes en cualquier capa de salida."""
    out: set[str] = set()
    for spatial_map in layers.values():
        out.update(spatial_map.get("per_station", []))
    return sorted(out)


def print_station_keys_inventory(
    plan: ColumnPlan,
    cfg: dict,
    by_slug: dict[str, dict[str, dict[str, list[str]]]],
) -> None:
    """Sección dedicada: claves de estación por contaminante (fuente vs salida vs drop)."""
    width = 72
    thin = "─" * width
    specs = cfg["pollutant_specs"]

    print("  CLAVES DE ESTACIÓN POR CONTAMINANTE")
    print("  (claves = sufijo en cont_{slug}_{CLAVE}[_{transform}])")
    print(thin)

    for slug in cfg["slug_order"]:
        spec = specs.get(slug, {})
        if not spec.get("enabled", True):
            continue

        source = source_stations_for_slug(plan, slug)
        layers = by_slug.get(slug, {})
        preserved = preserved_stations_per_slug(layers)
        drop_sources = spec.get("drop_sources", False)

        print(f"  {slug.upper()}")
        print(f"    fuente (input imputed): {len(source)} estaciones")
        _format_station_keys(source, indent=6)

        if not layers:
            print("    salida: (sin columnas cont_*)")
            print()
            continue

        transform_order = ["raw", "3h", "6h", "mda8"]
        for transform in transform_order:
            if transform not in layers:
                continue
            spatial_map = layers[transform]
            label = transform if transform != "raw" else "horario"
            for spatial in ("per_station", "ALL"):
                stations = spatial_map.get(spatial, [])
                if not stations:
                    continue
                scope = f"{len(stations)} estaciones" if spatial == "per_station" else "ALL"
                print(f"    salida [{label}, {scope}]:")
                _format_station_keys(stations, indent=6)

        if drop_sources and source:
            has_all = any("ALL" in sm for sm in layers.values())
            if not preserved and has_all:
                print(f"    drop fuentes: las {len(source)} claves individuales")
                _format_station_keys(source, indent=6)
                print("      → reemplazadas por columna ALL agregada")
            elif preserved:
                raw_stations = sorted(layers.get("raw", {}).get("per_station", []))
                if not raw_stations and preserved:
                    print(
                        f"    drop fuentes: columnas horarias raw; "
                        f"se conservan {len(preserved)} claves en capas transformadas"
                    )

        if preserved:
            print(f"    resumen claves en salida ({len(preserved)}):")
            _format_station_keys(preserved, indent=6)
        else:
            all_out = sorted(
                {s for sm in layers.values() for s in sm.get("ALL", [])}
            )
            if all_out:
                print(f"    resumen salida: solo ALL (sin claves por estación)")
        print()


def _abbrev_list(items: list[str], max_show: int = 6) -> str:
    if len(items) <= max_show:
        return ", ".join(items)
    head = ", ".join(items[:max_show])
    return f"{head}, … (+{len(items) - max_show} más)"


def _spec_description(slug: str, spec: dict) -> str:
    if not spec.get("enabled", True):
        return "deshabilitado"
    spatial = spec.get("spatial", "?")
    transforms = spec.get("transforms") or []
    extras = spec.get("extra") or []
    drop = spec.get("drop_sources", False)
    parts: list[str] = []
    if spatial == "per_station":
        parts.append("por estación")
    elif spatial == "ALL":
        parts.append("ALL")
    if transforms:
        parts.append("→ " + ", ".join(transforms))
    for ex in extras:
        parts.append(f"+ extra {ex.get('spatial', 'ALL')}/{ex.get('transform')}")
    if drop:
        parts.append("drop fuentes")
    return " · ".join(parts)


def print_column_inventory(
    plan: ColumnPlan,
    cfg: dict,
    written_years: list[int] | None = None,
) -> None:
    """Imprime inventario visual de columnas de salida al final del pipeline."""
    width = 72
    sep = "=" * width
    thin = "─" * width

    cont_cols = [c for c in plan.output_columns if c.startswith("cont_")]
    i_cols = [c for c in plan.output_columns if c.startswith("i_cont_")]
    n_time = len(cfg["time_cols"])
    n_total = n_time + len(plan.output_columns)

    years_str = "—"
    if written_years:
        if len(written_years) == 1:
            years_str = str(written_years[0])
        else:
            years_str = f"{min(written_years)}..{max(written_years)} ({len(written_years)} archivos)"

    print(f"\n{sep}")
    print("  INVENTARIO CSV ML FORECAST")
    print(f"  Patrón salida: {OUTPUT_PREFIX}_{{year}}.csv")
    print(sep)
    print(f"  Carpeta     : {cfg['output_folder']}")
    print(f"  Años escritos: {years_str}")
    print(f"  Columnas    : {n_total} total")
    print(f"                  {n_time} time features")
    print(f"                  {len(cont_cols)} cont_*  +  {len(i_cols)} i_cont_* (pareadas)")
    print(thin)

    print("  TIME FEATURES")
    print(f"    {_abbrev_list(cfg['time_cols'], max_show=12)}")
    print(thin)

    by_slug = output_stations_by_slug(cont_cols)

    specs = cfg["pollutant_specs"]
    for slug in cfg["slug_order"]:
        spec = specs.get(slug, {})
        enabled = spec.get("enabled", True)
        print(f"  {slug.upper()}" + ("" if enabled else "  [OFF]"))
        if not enabled:
            print(f"    config: {_spec_description(slug, spec)}")
            continue

        print(f"    config: {_spec_description(slug, spec)}")
        layers = by_slug.get(slug, {})
        if not layers:
            print("    (sin columnas en salida)")
            continue

        transform_order = ["raw", "3h", "6h", "mda8"]
        for transform in transform_order:
            if transform not in layers:
                continue
            spatial_map = layers[transform]
            for spatial in ("per_station", "ALL"):
                stations = sorted(spatial_map.get(spatial, []))
                if not stations:
                    continue
                label = transform if transform != "raw" else "horario"
                scope = f"ALL" if spatial == "ALL" else f"{len(stations)} estaciones"
                example = _cont_name(slug, stations[0], None if transform == "raw" else transform)
                if spatial == "ALL":
                    example = _cont_name(slug, "ALL", None if transform == "raw" else transform)
                print(f"    • {label:<8} [{scope:<14}]  ej. {example}")
                if spatial == "per_station":
                    print(f"      claves ({len(stations)}):")
                    _format_station_keys(stations, indent=8)

        n_slug = sum(len(v) for t in layers.values() for v in t.values())
        print(f"    subtotal: {n_slug} cont_* (+ {n_slug} i_cont_*)")
        print()

    disabled = [s for s in cfg["slug_order"] if not specs.get(s, {}).get("enabled", True)]
    if disabled:
        print(thin)
        print(f"  EXCLUIDOS (enabled=false): {', '.join(disabled)}")

    print(thin)
    print_station_keys_inventory(plan, cfg, by_slug)
    print(thin)
    print("  ORDEN EN CSV (column_order)")
    print(f"    1. time features ({n_time})")
    pos = n_time + 1
    for slug in cfg["slug_order"]:
        if not specs.get(slug, {}).get("enabled", True):
            continue
        slug_cont = [
            c for c in cont_cols
            if (meta := parse_cont_column(c)) and meta["slug"] == slug
        ]
        if not slug_cont:
            continue
        n_pairs = len(slug_cont)
        print(
            f"    {pos:>4}. {slug}: {slug_cont[0]} … {slug_cont[-1]}"
            f"  ({n_pairs} cont + {n_pairs} i_cont)"
        )
        pos += n_pairs * 2
    print(sep)
    print()


def log_column_summary(plan: ColumnPlan, cfg: dict) -> None:
    """Resumen breve al inicio (logging)."""
    cont_cols = [c for c in plan.output_columns if c.startswith("cont_")]
    log.info(
        "Esquema salida: %d time + %d pares cont/i_cont = %d columnas",
        len(cfg["time_cols"]),
        len(cont_cols),
        len(plan.column_order),
    )


# %%
# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_pipeline(cfg: dict | None = None) -> list[int]:
    if cfg is None:
        cfg = resolve_config()
    validate_config(cfg)
    plan = build_output_schema(cfg)

    ensure_folder(cfg["output_folder"])
    log_column_summary(plan, cfg)

    written: list[int] = []
    for year in cfg["years"]:
        log.info("=== Procesando %s ===", year)
        df_out = process_year(year, cfg, plan)
        if df_out is None:
            continue
        out_path = cfg["output_folder"] / cfg["output_pattern"].format(year=year)
        df_out.to_csv(
            out_path,
            float_format=cfg["float_format"],
            index_label=cfg["index_label"],
        )
        log.info("Guardado: %s  shape=%s", out_path, df_out.shape)
        written.append(year)

    if written:
        validate_written_headers(cfg["output_folder"], cfg["output_pattern"], written)
        print_column_inventory(plan, cfg, written)
    return written


def export_config_yaml(path: str | Path = "x_db_ml_features_config.yml") -> None:
    try:
        import yaml
    except ImportError:
        log.warning("pyyaml no instalado; omitiendo export config")
        return
    payload = {
        "data_root": str(DATA_ROOT),
        "pipeline_hub": PIPELINE_HUB,
        "years": PROCESS_YEARS,
        "stations": MERGE_STATIONS,
        "pollutants": POLLUTANT_SPECS,
        "passthrough_slugs": PASSTHROUGH_SLUGS,
    }
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(payload, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    log.info("Config exportado → %s", path)


# %%
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    user_cfg = resolve_config()
    log.info("Input : %s", user_cfg["input_folder"])
    log.info("Output: %s", user_cfg["output_folder"])
    log.info("Años  : %s..%s", min(user_cfg["years"]), max(user_cfg["years"]))
    run_pipeline(user_cfg)
