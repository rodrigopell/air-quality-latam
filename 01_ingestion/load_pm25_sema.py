"""
load_pm25_sema.py — Carga y estandarización de datos PurpleAir / SEMA
                    Península de Yucatán (Quintana Roo)

Fuente: Secretaría de Medio Ambiente de Quintana Roo (SEMA)
Sensores: PurpleAir PA-II (ópticos, corrección ATM)
Período:  ~oct-2020 a oct-2025 (varía por estación)
Variable: PM2.5 ATM [µg/m³], resolución horaria (o diaria según sensor)

Región primaria: Península de Yucatán — datos REALES con cobertura completa.
Guatemala es región de expansión con datos sintéticos; no usa este módulo.

Formatos de timestamp presentes en los CSVs:
  - DD/MM/YYYY              (e.g. Bacalar — solo fecha, múltiples lecturas/día)
  - YYYY-MM-DDTHH:MM:SSZ   (ISO UTC horario, e.g. Bomberos Cancún, Chetumal)

Estaciones disponibles (10 sensores PurpleAir):
  BAC  Bacalar           COZ  Cozumel          PL   Playa del Carmen
  BCUN Bomberos Cancún   PM   Puerto Morelos   TUL  Tulum
  C_A  Chetumal Aerop.   C_N  Chetumal Norte   C_Z  Chetumal Zoo
  UAC  UAC Cancún        ← sensor institucional real; cobertura limitada pero válido

Uso rápido:
    from 01_ingestion.load_pm25_sema import load_all_stations, load_station, station_summary
    gdf     = load_all_stations()          # GeoDataFrame con las 10 estaciones
    df_bac  = load_station("BAC")          # una sola estación
    summary = station_summary()            # tabla de cobertura por estación

Autor: Rodrigo, Ingeniero Ambiental, Universidad Anáhuac Cancún
"""

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import geopandas as gpd
    from shapely.geometry import Point
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PM25_SEMA_DIR, PM25_FILE_STATION_MAP, STATIONS_QR, PHYSICAL_LIMITS

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ─────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────

# Separadores a intentar en orden
_CSV_SEPS = [";", ",", "\t"]

# Formatos de fecha para intentar parsear (dd/mm/yyyy sin hora = dato diario)
_DATE_FMTS = [
    "%Y-%m-%dT%H:%M:%SZ",   # ISO UTC  2021-03-29T19:00:00Z
    "%Y-%m-%dT%H:%M:%S%z",  # ISO con tz
    "%d/%m/%Y %H:%M:%S",    # DD/MM/YYYY HH:MM:SS
    "%d/%m/%Y %H:%M",
    "%d/%m/%Y",              # Bacalar: solo fecha → granularidad diaria
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d",
]

PM25_LIMIT = PHYSICAL_LIMITS.get("PM2.5", (0, 500))


# ─────────────────────────────────────────────
# LECTURA Y LIMPIEZA
# ─────────────────────────────────────────────

def _read_csv_robust(filepath: Path) -> pd.DataFrame:
    """Lee un CSV intentando diferentes separadores y encodings."""
    encodings = ["utf-8", "latin-1", "cp1252"]
    last_err = None
    for enc in encodings:
        for sep in _CSV_SEPS:
            try:
                df = pd.read_csv(filepath, sep=sep, encoding=enc)
                if len(df.columns) >= 2:
                    return df
            except Exception as e:
                last_err = e
    raise IOError(f"No se pudo leer {filepath.name}: {last_err}")


def _parse_timestamp(series: pd.Series) -> pd.Series:
    """
    Parsea una columna de strings de fecha/hora usando múltiples formatos.
    Retorna pd.Series con dtype datetime64[ns, UTC].
    """
    # Primer intento: pandas automático (maneja ISO 8601 con 'Z')
    try:
        parsed = pd.to_datetime(series, utc=True)
        if parsed.notna().sum() > 0:
            return parsed
    except Exception:
        pass

    # Intentos con formatos explícitos
    for fmt in _DATE_FMTS:
        try:
            parsed = pd.to_datetime(series, format=fmt, utc=True, errors="coerce")
            if parsed.notna().mean() > 0.5:   # al menos 50 % de filas válidas
                return parsed
        except Exception:
            pass

    # Último recurso: inferencia libre
    return pd.to_datetime(series, infer_datetime_format=True, utc=True, errors="coerce")


def _detect_pm25_col(df: pd.DataFrame) -> str | None:
    """Detecta la columna de PM2.5 en el DataFrame."""
    candidates = [
        "pm2_5_atm", "pm2.5_atm", "PM2.5_CF1_ug/m3", "PM2.5_ATM_ug/m3",
        "pm2_5", "pm25", "PM2.5",
    ]
    for col in candidates:
        if col in df.columns:
            return col
    # Búsqueda flexible
    for col in df.columns:
        if re.search(r"pm\s*2[\._]?5", col, re.IGNORECASE):
            return col
    return None


def _detect_time_col(df: pd.DataFrame) -> str | None:
    """Detecta la columna de tiempo."""
    candidates = ["time_stamp", "created_at", "datetime", "fecha", "date", "time"]
    for col in candidates:
        if col in df.columns:
            return col
    for col in df.columns:
        if re.search(r"time|fecha|date", col, re.IGNORECASE):
            return col
    return None


def _clean_pm25(series: pd.Series) -> pd.Series:
    """
    Limpieza básica de PM2.5:
    - Convierte a numérico (coerce errores)
    - Reemplaza 0 con NaN (sensor apagado)
    - Elimina valores fuera del rango físico plausible
    """
    s = pd.to_numeric(series, errors="coerce")
    s = s.where(s > 0)                            # 0 → NaN (sensor inactivo)
    s = s.where((s >= PM25_LIMIT[0]) & (s <= PM25_LIMIT[1]))
    return s


# ─────────────────────────────────────────────
# FUNCIONES PÚBLICAS
# ─────────────────────────────────────────────

def load_station(
    station_id: str,
    data_dir: Path | None = None,
    drop_missing: bool = False,
) -> pd.DataFrame:
    """
    Carga los datos PM2.5 de una estación individual.

    Parameters
    ----------
    station_id : str
        ID de la estación (BAC, BCUN, C_A, C_N, C_Z, COZ, PL, PM, TUL, UAC).
    data_dir : Path | None
        Directorio con los CSV. Por defecto usa PM25_SEMA_DIR del config.
    drop_missing : bool
        Si True, elimina filas con PM2.5 NaN.

    Returns
    -------
    pd.DataFrame con columnas:
        datetime   — timestamp UTC (DatetimeTZDtype)
        pm2_5      — concentración PM2.5 ATM [µg/m³]
        station_id — identificador de estación
        lat, lon   — coordenadas (float)
        nombre     — nombre descriptivo de la estación
    """
    sid = station_id.upper()
    if sid not in STATIONS_QR:
        raise ValueError(
            f"Estación desconocida '{station_id}'. "
            f"Disponibles: {sorted(STATIONS_QR)}"
        )

    # Localizar archivo CSV
    base_dir = Path(data_dir) if data_dir else PM25_SEMA_DIR
    fname = next(
        (k for k, v in PM25_FILE_STATION_MAP.items() if v == sid), None
    )
    if fname is None:
        raise FileNotFoundError(f"Sin archivo CSV registrado para estación {sid}")

    filepath = base_dir / fname
    if not filepath.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {filepath}")

    # Leer
    raw = _read_csv_robust(filepath)

    # Detectar columnas clave
    time_col = _detect_time_col(raw)
    pm_col = _detect_pm25_col(raw)

    if time_col is None:
        raise ValueError(f"No se encontró columna de tiempo en {fname}")
    if pm_col is None:
        raise ValueError(f"No se encontró columna PM2.5 en {fname}")

    # Filtrar filas de "PROMEDIO" u otras cadenas no-dato
    mask_valid_time = raw[time_col].astype(str).str.match(
        r"^\d", na=False
    )
    raw = raw[mask_valid_time].copy()

    df = pd.DataFrame()
    df["datetime"] = _parse_timestamp(raw[time_col])
    df["pm2_5"] = _clean_pm25(raw[pm_col])

    # Metadata de estación
    meta = STATIONS_QR[sid]
    df["station_id"] = sid
    df["lat"] = meta["lat"]
    df["lon"] = meta["lon"]
    df["nombre"] = meta["nombre"]

    # Ordenar por timestamp
    df = df.sort_values("datetime").reset_index(drop=True)

    # Si hay timestamps duplicados (e.g. Bacalar con formato solo-fecha),
    # añadir índice incremental para distinguirlos en lugar de eliminarlos.
    n_dup = df.duplicated(subset=["datetime"]).sum()
    if n_dup > 0:
        logger.warning(
            f"Estación {sid}: {n_dup} timestamps duplicados. "
            "El CSV no incluye hora — se conservan todas las lecturas sin deduplicar. "
            "Usar df.groupby('datetime')['pm2_5'].mean() para agregar a diario."
        )

    if drop_missing:
        df = df.dropna(subset=["pm2_5"]).reset_index(drop=True)

    valid_pct = df["pm2_5"].notna().mean() * 100
    logger.info(
        f"Estación {sid} ({meta['nombre']}): {len(df)} registros, "
        f"{valid_pct:.1f}% con PM2.5 válido"
    )
    return df


def load_all_stations(
    data_dir: Path | None = None,
    drop_missing: bool = False,
    exclude_stations: list[str] | None = None,
) -> pd.DataFrame | "gpd.GeoDataFrame":
    """
    Carga todas las estaciones y las concatena en un único DataFrame.

    Parameters
    ----------
    data_dir : Path | None
        Directorio con los CSV. Por defecto usa PM25_SEMA_DIR del config.
    drop_missing : bool
        Eliminar filas con PM2.5 NaN.
    exclude_stations : list[str] | None
        Lista de station_id a excluir. Por defecto None (incluye todas).
        UAC se conserva aunque tenga cobertura limitada: es un sensor
        institucional real de la Universidad Anáhuac Cancún.

    Returns
    -------
    GeoDataFrame (si geopandas disponible) o DataFrame con columnas:
        datetime, pm2_5, station_id, lat, lon, nombre
    """
    if exclude_stations is None:
        exclude_stations = []   # todas las estaciones incluidas por defecto

    exclude_set = {s.upper() for s in exclude_stations}

    dfs = []
    for sid in sorted(STATIONS_QR):
        if sid in exclude_set:
            logger.info(f"Estación {sid} excluida según parámetro.")
            continue
        try:
            df = load_station(sid, data_dir=data_dir, drop_missing=drop_missing)
            dfs.append(df)
        except Exception as e:
            logger.warning(f"Estación {sid}: no se pudo cargar — {e}")

    if not dfs:
        raise RuntimeError("No se cargó ninguna estación.")

    combined = pd.concat(dfs, ignore_index=True)

    if HAS_GEOPANDAS:
        geometry = [Point(xy) for xy in zip(combined["lon"], combined["lat"])]
        gdf = gpd.GeoDataFrame(combined, geometry=geometry, crs="EPSG:4326")
        logger.info(
            f"GeoDataFrame total: {len(gdf):,} registros, "
            f"{gdf['station_id'].nunique()} estaciones"
        )
        return gdf

    logger.info(
        f"DataFrame total: {len(combined):,} registros, "
        f"{combined['station_id'].nunique()} estaciones"
    )
    return combined


def station_summary(data_dir: Path | None = None) -> pd.DataFrame:
    """
    Genera una tabla resumen de cobertura por estación.

    Returns
    -------
    pd.DataFrame con: station_id, nombre, n_total, n_validos, pct_valido,
                      fecha_inicio, fecha_fin, lat, lon
    """
    rows = []
    for sid in sorted(STATIONS_QR):
        try:
            df = load_station(sid, data_dir=data_dir)
            meta = STATIONS_QR[sid]
            rows.append({
                "station_id":   sid,
                "nombre":       meta["nombre"],
                "municipio":    meta["municipio"],
                "lat":          meta["lat"],
                "lon":          meta["lon"],
                "n_total":      len(df),
                "n_validos":    int(df["pm2_5"].notna().sum()),
                "pct_valido":   round(df["pm2_5"].notna().mean() * 100, 1),
                "fecha_inicio": df["datetime"].min(),
                "fecha_fin":    df["datetime"].max(),
                "pm25_media":   round(df["pm2_5"].mean(), 2),
                "pm25_max":     round(df["pm2_5"].max(), 2),
            })
        except Exception as e:
            logger.warning(f"Estación {sid}: error en resumen — {e}")

    return pd.DataFrame(rows).sort_values("station_id").reset_index(drop=True)


# ─────────────────────────────────────────────
# DEMO / USO DIRECTO
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=== PM2.5 SEMA Quintana Roo — demostración ===\n")

    print("Tabla resumen de estaciones:")
    summary = station_summary()
    print(summary.to_string(index=False))
    print()

    print("Cargando todas las estaciones (excepto UAC)...")
    gdf = load_all_stations()
    print(f"Total registros: {len(gdf):,}")
    print(f"Columnas: {list(gdf.columns)}")
    print(gdf.head(3).to_string())
