"""
fetch_cams.py — Descarga de datos del servicio Copernicus CAMS (reanálisis atmosférico)
Cubre contaminantes: PM2.5, O3, NO2, CO, SO2 y más.

Configuración requerida:
  1. Crear cuenta en https://ads.atmosphere.copernicus.eu/
  2. Instalar cliente: pip install cdsapi
  3. Crear archivo ~/.cdsapirc con:
       url: https://ads.atmosphere.copernicus.eu/api/v2
       key: TU_UID:TU_API_KEY
  4. (Opcional) Definir variables de entorno CDSAPI_URL y CDSAPI_KEY

Dataset principal: cams-global-reanalysis-eac4
Docs: https://ads.atmosphere.copernicus.eu/cdsapp#!/dataset/cams-global-reanalysis-eac4

Autor: Rodrigo, Ingeniero Ambiental, Universidad Anáhuac Cancún
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import cdsapi
    HAS_CDSAPI = True
except ImportError:
    HAS_CDSAPI = False
    logging.warning(
        "cdsapi no disponible. Instalar con: pip install cdsapi\n"
        "Luego configurar ~/.cdsapirc con tus credenciales de Copernicus ADS."
    )

try:
    import xarray as xr
    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False
    logging.warning("xarray no disponible. Instalar con: pip install xarray netCDF4")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RAW_DIR

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
)

CAMS_RAW_DIR = RAW_DIR / "cams"
CAMS_RAW_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# VARIABLES CAMS DISPONIBLES
# ─────────────────────────────────────────────

# Mapeo de nombre amigable → nombre oficial en CAMS EAC4
CAMS_VARIABLES = {
    "PM2.5":  "particulate_matter_2.5um",
    "PM10":   "particulate_matter_10um",
    "O3":     "ozone",
    "NO2":    "nitrogen_dioxide",
    "NO":     "nitrogen_monoxide",
    "CO":     "carbon_monoxide",
    "SO2":    "sulphur_dioxide",
    "dust":   "dust_aerosol_0.55-0.9um_mixing_ratio",
    "BC":     "black_carbon_aerosol_mixing_ratio",
}

# Dataset principal de CAMS
CAMS_DATASET = "cams-global-reanalysis-eac4"

# Niveles de presión disponibles en CAMS (hPa)
SURFACE_LEVEL = "surface"
PRESSURE_LEVELS = [1000, 925, 850, 700, 500, 300, 200, 100]

# Horas disponibles (UTC)
CAMS_HOURS = ["00:00", "03:00", "06:00", "09:00", "12:00", "15:00", "18:00", "21:00"]


# ─────────────────────────────────────────────
# FUNCIONES DE DESCARGA
# ─────────────────────────────────────────────

def fetch_cams_reanalysis(
    variable: str,
    bbox: list[float],
    date_from: str,
    date_to: str,
    hours: list[str] | None = None,
    pressure_level: str | int = SURFACE_LEVEL,
    output_filename: str | None = None,
) -> Path | None:
    """
    Descarga el reanálisis CAMS EAC4 para una variable, bbox y período.

    Parámetros
    ----------
    variable : str
        Contaminante: 'PM2.5', 'PM10', 'O3', 'NO2', 'CO', 'SO2', 'dust', 'BC'.
        O el nombre CAMS directo (ej: 'particulate_matter_2.5um').
    bbox : list[float]
        [lon_min, lat_min, lon_max, lat_max] — área de descarga.
        CAMS acepta formato [lat_max, lon_min, lat_min, lon_max] (N/W/S/E).
    date_from : str
        Fecha inicial 'YYYY-MM-DD'.
    date_to : str
        Fecha final 'YYYY-MM-DD'.
    hours : list[str] | None
        Horas UTC a descargar. Default: todas las disponibles.
    pressure_level : str | int
        'surface' o nivel de presión en hPa.
    output_filename : str | None
        Nombre del archivo de salida. Si es None, se genera automáticamente.

    Retorna
    -------
    Path al archivo NetCDF descargado, o None si falló.

    Notas
    -----
    - Las descargas CAMS pueden tardar varios minutos según el período.
    - El archivo se guarda en data/raw/cams/.
    - CAMS EAC4 tiene resolución de ~0.75° (~80 km).
    """
    if not HAS_CDSAPI:
        raise ImportError(
            "cdsapi no disponible. Instalar con: pip install cdsapi\n"
            "Configurar ~/.cdsapirc con las credenciales de Copernicus ADS."
        )

    # Normalizar nombre de variable
    cams_var = CAMS_VARIABLES.get(variable, variable)
    var_short = variable.replace(".", "").replace(" ", "_").lower()

    if hours is None:
        hours = CAMS_HOURS

    # CAMS usa formato de bbox: Norte/Oeste/Sur/Este
    lon_min, lat_min, lon_max, lat_max = bbox
    cams_area = [lat_max, lon_min, lat_min, lon_max]

    # Generar lista de fechas
    dates = _date_range_list(date_from, date_to)
    date_str = "/".join(dates)

    # Nombre de archivo de salida
    if output_filename is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"cams_eac4_{var_short}_{date_from}_{date_to}_{ts}.nc"
    output_path = CAMS_RAW_DIR / output_filename

    logger.info(
        f"Solicitando CAMS EAC4 | Variable: {cams_var} | "
        f"Período: {date_from} → {date_to} | Área: {cams_area}"
    )

    # Construir request según si es superficie o nivel de presión
    request = {
        "variable": cams_var,
        "date": date_str,
        "time": hours,
        "area": cams_area,
        "format": "netcdf",
    }

    if pressure_level == SURFACE_LEVEL or pressure_level == "surface":
        # Sin nivel de presión → superficie
        pass
    else:
        request["pressure_level"] = str(pressure_level)

    try:
        client = cdsapi.Client(quiet=False)
        client.retrieve(
            CAMS_DATASET,
            request,
            str(output_path),
        )
        size_mb = output_path.stat().st_size / 1024 / 1024
        logger.info(f"Descarga completada: {output_path.name} ({size_mb:.1f} MB)")
        return output_path

    except Exception as e:
        logger.error(
            f"Error descargando CAMS: {e}\n"
            "Verifica:\n"
            "  1. ~/.cdsapirc configurado correctamente\n"
            "  2. Acceso aprobado en https://ads.atmosphere.copernicus.eu/\n"
            "  3. Variable y período válidos"
        )
        return None


def fetch_cams_multiple_variables(
    variables: list[str],
    bbox: list[float],
    date_from: str,
    date_to: str,
) -> dict[str, Path]:
    """
    Descarga múltiples variables CAMS para la misma región y período.

    Retorna
    -------
    dict {variable: path_netcdf}
    """
    results = {}
    for var in variables:
        logger.info(f"Descargando variable: {var}")
        path = fetch_cams_reanalysis(var, bbox, date_from, date_to)
        if path:
            results[var] = path
    return results


# ─────────────────────────────────────────────
# FUNCIONES DE CARGA Y PROCESAMIENTO
# ─────────────────────────────────────────────

def load_cams_nc(
    filepath: str | Path,
    bbox: list[float] | None = None,
) -> "xr.Dataset":
    """
    Carga un archivo NetCDF de CAMS y lo prepara para análisis.

    Parámetros
    ----------
    filepath : str | Path
        Ruta al archivo .nc descargado de CAMS.
    bbox : list[float] | None
        [lon_min, lat_min, lon_max, lat_max] para recorte espacial adicional.

    Retorna
    -------
    xr.Dataset con dimensiones estandarizadas (time, latitude, longitude).
    """
    if not HAS_XARRAY:
        raise ImportError("xarray es necesario. Instalar: pip install xarray netCDF4")

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {filepath}")

    logger.info(f"Cargando CAMS NetCDF: {filepath.name}")
    ds = xr.open_dataset(filepath, engine="netcdf4")

    # Estandarizar nombres de coordenadas
    rename_map = {}
    for coord in ds.coords:
        if coord.lower() in ("lon", "longitude", "x"):
            rename_map[coord] = "lon"
        elif coord.lower() in ("lat", "latitude", "y"):
            rename_map[coord] = "lat"
        elif coord.lower() in ("valid_time", "time", "t"):
            rename_map[coord] = "time"
    if rename_map:
        ds = ds.rename(rename_map)

    # Recorte espacial
    if bbox and "lat" in ds.coords and "lon" in ds.coords:
        lon_min, lat_min, lon_max, lat_max = bbox
        ds = ds.sel(
            lat=slice(lat_min, lat_max),
            lon=slice(lon_min, lon_max)
        )
        logger.info(f"Recortado a bbox {bbox}")

    logger.info(f"Dataset: {dict(ds.dims)} | Variables: {list(ds.data_vars)}")
    return ds


def cams_to_dataframe(
    ds: "xr.Dataset",
    variable: str | None = None,
) -> pd.DataFrame:
    """
    Convierte un xr.Dataset de CAMS a DataFrame largo con columnas:
    [lat, lon, datetime, variable, valor]

    Parámetros
    ----------
    ds : xr.Dataset
        Dataset CAMS cargado con load_cams_nc().
    variable : str | None
        Variable a convertir. Si es None, usa la primera disponible.

    Retorna
    -------
    pd.DataFrame con columnas: lat, lon, datetime, variable, valor
    """
    if not HAS_XARRAY:
        raise ImportError("xarray es necesario.")

    if variable is None:
        variable = list(ds.data_vars)[0]
        logger.info(f"Variable no especificada — usando: {variable}")

    if variable not in ds.data_vars:
        raise KeyError(f"Variable '{variable}' no encontrada. Disponibles: {list(ds.data_vars)}")

    da = ds[variable]

    # Convertir a DataFrame
    df = da.to_dataframe().reset_index()

    # Renombrar columnas de tiempo
    time_cols = [c for c in df.columns if "time" in c.lower()]
    if time_cols:
        df = df.rename(columns={time_cols[0]: "datetime"})

    df = df.rename(columns={variable: "valor"})
    df["variable"] = variable

    # Seleccionar y ordenar columnas
    cols = [c for c in ["lat", "lon", "datetime", "variable", "valor"] if c in df.columns]
    df = df[cols]

    # Limpiar
    df = df.dropna(subset=["valor"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values(["datetime", "lat", "lon"]).reset_index(drop=True)

    logger.info(f"DataFrame CAMS '{variable}': {len(df)} filas")
    return df


def cams_daily_mean(filepath: str | Path, variable: str, bbox: list[float]) -> pd.DataFrame:
    """
    Carga un archivo CAMS y calcula la media diaria por celda del grid.

    Retorna DataFrame con columnas: [lat, lon, date, media_diaria]
    """
    ds = load_cams_nc(filepath, bbox=bbox)
    df = cams_to_dataframe(ds, variable)

    df["date"] = pd.to_datetime(df["datetime"]).dt.date
    daily = (
        df.groupby(["lat", "lon", "date"])["valor"]
        .mean()
        .reset_index()
        .rename(columns={"valor": f"{variable}_daily_mean"})
    )
    logger.info(f"Media diaria calculada: {len(daily)} filas ({daily['date'].nunique()} días)")
    return daily


# ─────────────────────────────────────────────
# UTILIDADES
# ─────────────────────────────────────────────

def _date_range_list(date_from: str, date_to: str) -> list[str]:
    """
    Genera lista de fechas 'YYYY-MM-DD' entre date_from y date_to.
    """
    d0 = datetime.strptime(date_from, "%Y-%m-%d")
    d1 = datetime.strptime(date_to, "%Y-%m-%d")
    dates = []
    while d0 <= d1:
        dates.append(d0.strftime("%Y-%m-%d"))
        d0 += timedelta(days=1)
    return dates


def check_cdsapi_config() -> bool:
    """
    Verifica si cdsapi está configurado correctamente.
    Retorna True si la configuración es válida.
    """
    if not HAS_CDSAPI:
        logger.error("cdsapi no instalado. Ejecutar: pip install cdsapi")
        return False

    cdsapirc = Path.home() / ".cdsapirc"
    if not cdsapirc.exists():
        logger.error(
            "Archivo ~/.cdsapirc no encontrado.\n"
            "Crear con el contenido:\n"
            "  url: https://ads.atmosphere.copernicus.eu/api/v2\n"
            "  key: TU_UID:TU_API_KEY\n"
            "Obtener credenciales en: https://ads.atmosphere.copernicus.eu/"
        )
        return False

    logger.info(f"Configuración cdsapi encontrada en {cdsapirc}")
    return True


# ─────────────────────────────────────────────
# EJEMPLO DE USO
# ─────────────────────────────────────────────
if __name__ == "__main__":
    GUATEMALA_BBOX = [-92.2, 13.7, -88.2, 17.8]

    # Verificar configuración
    if not check_cdsapi_config():
        print("\nConfigura ~/.cdsapirc antes de continuar.")
    else:
        print("=== Descargando PM2.5 CAMS para Guatemala ===")
        nc_path = fetch_cams_reanalysis(
            variable="PM2.5",
            bbox=GUATEMALA_BBOX,
            date_from="2024-01-01",
            date_to="2024-01-07",
        )

        if nc_path:
            print("\n=== Cargando y convirtiendo a DataFrame ===")
            ds = load_cams_nc(nc_path, bbox=GUATEMALA_BBOX)
            print(ds)

            # Primera variable disponible
            var = list(ds.data_vars)[0]
            df = cams_to_dataframe(ds, var)
            print(df.head(10))
            print(f"\nEstadísticas de {var}:")
            print(df["valor"].describe())
