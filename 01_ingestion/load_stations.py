"""
load_stations.py — Carga y estandarización de datos de estaciones de monitoreo
Soporta CSV con separadores y encodings variados, formatos de fecha LATAM,
y detección fuzzy de columnas de contaminantes y coordenadas.

Autor: Rodrigo, Ingeniero Ambiental, Universidad Anáhuac Cancún
"""

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# Configurar logging para este módulo
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# MAPAS FUZZY DE COLUMNAS
# ─────────────────────────────────────────────

# Patrones regex para detectar contaminantes (insensible a mayúsculas)
POLLUTANT_PATTERNS = {
    "PM2.5":  r"pm\s*2[\._]?5|pm25|particulas_finas|fine_pm",
    "PM10":   r"pm\s*10|pm10|particulas_gruesas|coarse_pm",
    "O3":     r"o3|ozono|ozone",
    "NO2":    r"no2|dioxido.*nitrogeno|nitrogen.*dioxide",
    "NO":     r"\bno\b|oxido.*nitrico",
    "CO":     r"\bco\b|monoxido.*carbono|carbon.*monoxide",
    "SO2":    r"so2|dioxido.*azufre|sulfur.*dioxide",
    "CO2":    r"co2|dioxido.*carbono",
    "BC":     r"\bbc\b|black.?carbon|carbono.?negro",
}

# Patrones para columnas de coordenadas
LAT_PATTERNS  = r"lat(itud)?|y_coord|northing"
LON_PATTERNS  = r"lon(gitud)?|lng|x_coord|easting"

# Patrones para columnas de fecha/hora
DATE_PATTERNS = r"fecha|date|datetime|timestamp|time|hora"

# Formatos de fecha frecuentes en LATAM
DATE_FORMATS = [
    "%d/%m/%Y %H:%M:%S",
    "%d/%m/%Y %H:%M",
    "%d/%m/%Y",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d",
    "%d-%m-%Y %H:%M:%S",
    "%d-%m-%Y %H:%M",
    "%d-%m-%Y",
    "%Y/%m/%d %H:%M:%S",
    "%Y/%m/%d",
    "%m/%d/%Y %H:%M:%S",
    "%m/%d/%Y",
]

# Encodings a probar en orden
ENCODINGS = ["utf-8", "utf-8-sig", "latin-1", "cp1252", "iso-8859-1"]


# ─────────────────────────────────────────────
# FUNCIONES AUXILIARES
# ─────────────────────────────────────────────

def _detect_separator(filepath: str) -> str:
    """
    Detecta el separador del CSV probando coma, punto y coma y tabulador.
    Devuelve el separador con mayor número de ocurrencias en la primera línea.
    """
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            primera_linea = f.readline()
    except Exception:
        return ","

    conteos = {
        ",":  primera_linea.count(","),
        ";":  primera_linea.count(";"),
        "\t": primera_linea.count("\t"),
        "|":  primera_linea.count("|"),
    }
    sep = max(conteos, key=conteos.get)
    logger.debug(f"Separador detectado: {repr(sep)} (conteos: {conteos})")
    return sep


def _read_with_encoding(filepath: str, sep: str) -> pd.DataFrame:
    """
    Intenta leer el CSV con múltiples encodings.
    Lanza ValueError si ninguno funciona.
    """
    for enc in ENCODINGS:
        try:
            df = pd.read_csv(filepath, sep=sep, encoding=enc, low_memory=False)
            logger.info(f"Archivo leído con encoding '{enc}' — {len(df)} filas, {len(df.columns)} columnas")
            return df
        except UnicodeDecodeError:
            logger.debug(f"Encoding '{enc}' falló, probando siguiente...")
        except Exception as e:
            logger.warning(f"Error inesperado con encoding '{enc}': {e}")
    raise ValueError(
        f"No se pudo leer '{filepath}' con ninguno de los encodings: {ENCODINGS}. "
        "Verifica el archivo o especifica el encoding manualmente."
    )


def _detect_date_column(df: pd.DataFrame) -> str | None:
    """
    Detecta la columna de fecha/hora por nombre fuzzy.
    Devuelve el nombre de la columna o None.
    """
    for col in df.columns:
        if re.search(DATE_PATTERNS, col, re.IGNORECASE):
            logger.debug(f"Columna de fecha detectada: '{col}'")
            return col
    logger.warning("No se detectó columna de fecha/hora.")
    return None


def _parse_dates(series: pd.Series) -> pd.Series:
    """
    Intenta parsear una serie de texto a datetime con múltiples formatos.
    """
    for fmt in DATE_FORMATS:
        try:
            parsed = pd.to_datetime(series, format=fmt, errors="raise")
            logger.info(f"Fechas parseadas con formato '{fmt}'")
            return parsed
        except (ValueError, TypeError):
            continue
    # Último intento: inferencia automática de pandas
    try:
        parsed = pd.to_datetime(series, infer_datetime_format=True, errors="coerce")
        n_null = parsed.isna().sum()
        if n_null > 0:
            logger.warning(f"Inferencia automática de fechas: {n_null} valores no parseados → NaT")
        return parsed
    except Exception as e:
        logger.error(f"No se pudo parsear la columna de fechas: {e}")
        return series


def _detect_pollutant_columns(df: pd.DataFrame) -> dict:
    """
    Detecta columnas de contaminantes por nombre fuzzy.
    Devuelve dict {nombre_estandar: nombre_columna_original}.
    """
    mapping = {}
    for col in df.columns:
        col_lower = col.lower().replace(" ", "_")
        for pollutant, pattern in POLLUTANT_PATTERNS.items():
            if re.search(pattern, col_lower, re.IGNORECASE):
                if pollutant not in mapping:  # primera coincidencia gana
                    mapping[pollutant] = col
                    logger.debug(f"Contaminante '{pollutant}' → columna '{col}'")
    logger.info(f"Contaminantes detectados: {list(mapping.keys())}")
    return mapping


def _detect_coordinate_columns(df: pd.DataFrame) -> tuple[str | None, str | None]:
    """
    Detecta columnas de latitud y longitud por nombre fuzzy.
    Devuelve (col_lat, col_lon) o (None, None).
    """
    col_lat, col_lon = None, None
    for col in df.columns:
        col_norm = col.lower().replace(" ", "_")
        if re.search(LAT_PATTERNS, col_norm, re.IGNORECASE) and col_lat is None:
            col_lat = col
        if re.search(LON_PATTERNS, col_norm, re.IGNORECASE) and col_lon is None:
            col_lon = col
    if col_lat:
        logger.debug(f"Latitud → '{col_lat}', Longitud → '{col_lon}'")
    else:
        logger.warning("No se detectaron columnas de coordenadas.")
    return col_lat, col_lon


# ─────────────────────────────────────────────
# FUNCIÓN PRINCIPAL
# ─────────────────────────────────────────────

def load_csv_stations(
    filepath: str,
    encoding: str | None = None,
    station_id_col: str | None = None,
    crs: str = "EPSG:4326",
) -> gpd.GeoDataFrame:
    """
    Carga un CSV de estaciones de monitoreo de calidad del aire y devuelve
    un GeoDataFrame estandarizado con CRS EPSG:4326.

    Parámetros
    ----------
    filepath : str
        Ruta al archivo CSV.
    encoding : str | None
        Encoding explícito. Si es None, se prueba automáticamente.
    station_id_col : str | None
        Nombre de la columna de ID de estación. Si es None, se busca por patrón.
    crs : str
        Sistema de referencia de coordenadas de salida (default: EPSG:4326).

    Retorna
    -------
    gpd.GeoDataFrame
        GeoDataFrame con columnas estandarizadas: datetime, lat, lon,
        geometry, y una columna por cada contaminante detectado.
    """
    filepath = Path(filepath)
    logger.info(f"─── Cargando archivo: {filepath} ───")

    if not filepath.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {filepath}")

    # 1. Detectar separador
    sep = _detect_separator(str(filepath))

    # 2. Leer CSV con encoding
    if encoding:
        try:
            df = pd.read_csv(filepath, sep=sep, encoding=encoding, low_memory=False)
            logger.info(f"Archivo leído con encoding '{encoding}': {len(df)} filas")
        except UnicodeDecodeError:
            logger.warning(f"Encoding '{encoding}' falló, probando automáticamente...")
            df = _read_with_encoding(str(filepath), sep)
    else:
        df = _read_with_encoding(str(filepath), sep)

    # Eliminar columnas completamente vacías
    n_antes = len(df.columns)
    df = df.dropna(axis=1, how="all")
    if len(df.columns) < n_antes:
        logger.info(f"Eliminadas {n_antes - len(df.columns)} columnas vacías")

    # 3. Detectar y parsear fechas
    date_col = _detect_date_column(df)
    if date_col:
        df["datetime"] = _parse_dates(df[date_col])
        if date_col != "datetime":
            df = df.drop(columns=[date_col])
        df = df.sort_values("datetime").reset_index(drop=True)
        logger.info(f"Rango de fechas: {df['datetime'].min()} → {df['datetime'].max()}")
    else:
        df["datetime"] = pd.NaT

    # 4. Detectar contaminantes y renombrar
    pollutant_map = _detect_pollutant_columns(df)
    rename_dict = {v: k for k, v in pollutant_map.items() if v != k}
    if rename_dict:
        df = df.rename(columns=rename_dict)
        logger.info(f"Columnas renombradas: {rename_dict}")

    # Convertir columnas de contaminantes a numérico
    for poll in pollutant_map.keys():
        if poll in df.columns:
            df[poll] = pd.to_numeric(df[poll], errors="coerce")

    # 5. Detectar coordenadas
    col_lat, col_lon = _detect_coordinate_columns(df)

    if col_lat and col_lon:
        df[col_lat] = pd.to_numeric(df[col_lat], errors="coerce")
        df[col_lon] = pd.to_numeric(df[col_lon], errors="coerce")
        df = df.rename(columns={col_lat: "lat", col_lon: "lon"})

        # Construir geometría
        geometry = [
            Point(lon, lat) if pd.notna(lon) and pd.notna(lat) else None
            for lat, lon in zip(df["lat"], df["lon"])
        ]
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=crs)
        n_sin_geo = sum(g is None for g in geometry)
        if n_sin_geo:
            logger.warning(f"{n_sin_geo} filas sin geometría válida (coordenadas nulas)")
    else:
        logger.warning("Sin coordenadas — creando GeoDataFrame sin geometría")
        gdf = gpd.GeoDataFrame(df, crs=crs)

    # 6. Detectar/estandarizar columna de ID de estación
    if station_id_col and station_id_col in gdf.columns:
        gdf = gdf.rename(columns={station_id_col: "station_id"})
    else:
        id_candidates = [c for c in gdf.columns
                         if re.search(r"id|estacion|station|codigo|code|clave", c, re.IGNORECASE)]
        if id_candidates:
            gdf = gdf.rename(columns={id_candidates[0]: "station_id"})
            logger.debug(f"ID de estación → '{id_candidates[0]}'")
        else:
            logger.warning("No se detectó columna de ID de estación.")

    logger.info(
        f"GeoDataFrame final: {len(gdf)} filas, {len(gdf.columns)} columnas. "
        f"Contaminantes: {list(pollutant_map.keys())}"
    )
    return gdf


# ─────────────────────────────────────────────
# UTILIDAD: GENERAR DATOS SINTÉTICOS PARA PRUEBAS
# ─────────────────────────────────────────────

def generate_synthetic_data(
    region: str = "Guatemala",
    n_stations: int = 8,
    days: int = 365,
    output_path: str | None = None,
) -> pd.DataFrame:
    """
    Genera un DataFrame sintético de calidad del aire para pruebas.

    Parámetros
    ----------
    region : str
        'Guatemala' o 'Quintana_Roo'
    n_stations : int
        Número de estaciones simuladas.
    days : int
        Días de datos a generar.
    output_path : str | None
        Si se especifica, guarda el CSV en esa ruta.

    Retorna
    -------
    pd.DataFrame con columnas: station_id, fecha, lat, lon, PM2.5, PM10, O3, NO2, CO
    """
    rng = np.random.default_rng(42)

    bboxes = {
        "Guatemala":    {"lat": (13.7, 17.8), "lon": (-92.2, -88.2)},
        "Quintana_Roo": {"lat": (18.0, 21.6), "lon": (-88.0, -86.7)},
    }
    bbox = bboxes.get(region, bboxes["Guatemala"])

    # Posiciones fijas por estación
    station_lats = rng.uniform(*bbox["lat"], n_stations)
    station_lons = rng.uniform(*bbox["lon"], n_stations)
    station_ids  = [f"EST_{i+1:03d}" for i in range(n_stations)]

    fechas = pd.date_range("2023-01-01", periods=days * 24, freq="h")

    rows = []
    for i, sid in enumerate(station_ids):
        # Patrón diurno + estacional + ruido
        horas = np.arange(len(fechas))
        diurno   = 10 * np.sin(2 * np.pi * (horas % 24) / 24 - np.pi / 2) + 10
        estacional = 5 * np.sin(2 * np.pi * horas / (365 * 24)) + 5

        pm25 = np.clip(15 + diurno + estacional + rng.normal(0, 5, len(fechas)), 0, 200)
        pm10 = np.clip(pm25 * 1.8 + rng.normal(0, 8, len(fechas)), 0, 400)
        o3   = np.clip(40 + 20 * np.sin(2 * np.pi * (horas % 24) / 24) + rng.normal(0, 10, len(fechas)), 0, 200)
        no2  = np.clip(20 + diurno * 0.8 + rng.normal(0, 5, len(fechas)), 0, 200)
        co   = np.clip(0.5 + diurno * 0.03 + rng.normal(0, 0.1, len(fechas)), 0, 10)

        # Introducir ~5% de datos faltantes
        for arr in [pm25, pm10, o3, no2, co]:
            mask = rng.random(len(arr)) < 0.05
            arr[mask] = np.nan

        for j, fecha in enumerate(fechas):
            rows.append({
                "station_id": sid,
                "fecha": fecha.strftime("%d/%m/%Y %H:%M"),
                "lat": round(station_lats[i], 4),
                "lon": round(station_lons[i], 4),
                "PM2.5": round(pm25[j], 2) if not np.isnan(pm25[j]) else None,
                "PM10":  round(pm10[j], 2) if not np.isnan(pm10[j]) else None,
                "O3":    round(o3[j], 2)   if not np.isnan(o3[j])   else None,
                "NO2":   round(no2[j], 2)  if not np.isnan(no2[j])  else None,
                "CO":    round(co[j], 3)   if not np.isnan(co[j])   else None,
            })

    df = pd.DataFrame(rows)
    logger.info(f"Datos sintéticos generados: {len(df)} registros, {n_stations} estaciones, {days} días")

    if output_path:
        df.to_csv(output_path, index=False, encoding="utf-8")
        logger.info(f"Guardado en: {output_path}")

    return df


# ─────────────────────────────────────────────
# EJEMPLO DE USO
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config import RAW_DIR

    # Generar datos sintéticos y guardarlos
    csv_path = RAW_DIR / "estaciones_guatemala_ejemplo.csv"
    df_sint = generate_synthetic_data(region="Guatemala", n_stations=5, days=30, output_path=str(csv_path))
    print(df_sint.head())

    # Cargar con la función principal
    gdf = load_csv_stations(str(csv_path))
    print("\nGeoDataFrame resultante:")
    print(gdf.dtypes)
    print(gdf.head())
