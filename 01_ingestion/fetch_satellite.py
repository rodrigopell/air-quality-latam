"""
fetch_satellite.py — Descarga y carga de datos satelitales de calidad del aire
Soporta:
  - NASA EarthData: MODIS Aerosol Optical Depth (AOD)
  - MERRA-2: análisis de reanálisis atmosférico (NetCDF)

Credenciales NASA EarthData:
  1. Registrarse en https://urs.earthdata.nasa.gov/
  2. Aprobar el acuerdo de uso de GESDISC (para MERRA-2)
  3. Crear archivo ~/.netrc con:
       machine urs.earthdata.nasa.gov login TU_USUARIO password TU_CONTRASEÑA
  4. O definir variables de entorno:
       NASA_EARTHDATA_USER y NASA_EARTHDATA_PASSWORD

Autor: Rodrigo, Ingeniero Ambiental, Universidad Anáhuac Cancún
"""

import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

try:
    import xarray as xr
    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False
    logging.warning("xarray no disponible — funciones NetCDF deshabilitadas. Instalar con: pip install xarray netCDF4")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RAW_DIR, NASA_EARTHDATA_URL, NASA_EARTHDATA_TOKEN

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
)

SATELLITE_RAW_DIR = RAW_DIR / "satellite"
SATELLITE_RAW_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# AUTENTICACIÓN NASA EARTHDATA
# ─────────────────────────────────────────────

def _get_nasa_session() -> requests.Session:
    """
    Crea una sesión requests autenticada con NASA EarthData.
    Usa token Bearer si está disponible, o credenciales .netrc como respaldo.
    """
    session = requests.Session()

    if NASA_EARTHDATA_TOKEN:
        session.headers.update({"Authorization": f"Bearer {NASA_EARTHDATA_TOKEN}"})
        logger.debug("Usando token Bearer de NASA EarthData")
    else:
        # Intentar credenciales de .netrc
        user = os.getenv("NASA_EARTHDATA_USER", "")
        pwd  = os.getenv("NASA_EARTHDATA_PASSWORD", "")
        if user and pwd:
            session.auth = (user, pwd)
            logger.debug("Usando credenciales de variables de entorno NASA_EARTHDATA_USER/PASSWORD")
        else:
            logger.warning(
                "Sin credenciales NASA EarthData. Algunas descargas fallarán.\n"
                "Solución: define NASA_EARTHDATA_TOKEN o NASA_EARTHDATA_USER + NASA_EARTHDATA_PASSWORD."
            )
    return session


# ─────────────────────────────────────────────
# MODIS AOD
# ─────────────────────────────────────────────

# Colecciones MODIS disponibles en LAADS DAAC
MODIS_COLLECTIONS = {
    "MOD04_L2":  "Terra MODIS AOD (resolución 10km, Level-2)",
    "MYD04_L2":  "Aqua MODIS AOD (resolución 10km, Level-2)",
    "MOD04_3K":  "Terra MODIS AOD (resolución 3km, Level-2)",
    "MOD08_M3":  "Terra MODIS AOD mensual (Level-3, 1°×1°)",
}


def fetch_modis_aod(
    bbox: list[float],
    date_from: str,
    date_to: str,
    collection: str = "MOD08_M3",
    save_dir: Path | None = None,
) -> list[Path]:
    """
    Descarga archivos de Aerosol Optical Depth (AOD) de MODIS vía NASA LAADS DAAC.

    Parámetros
    ----------
    bbox : list[float]
        [lon_min, lat_min, lon_max, lat_max]
    date_from : str
        Fecha inicial 'YYYY-MM-DD'
    date_to : str
        Fecha final 'YYYY-MM-DD'
    collection : str
        Colección MODIS (ver MODIS_COLLECTIONS).
    save_dir : Path | None
        Directorio de guardado. Default: data/raw/satellite/modis/

    Retorna
    -------
    list[Path]: Rutas de archivos HDF descargados.

    Notas
    -----
    Los archivos MODIS están en formato HDF4 (.hdf).
    Para leerlos necesitas: conda install -c conda-forge pyhdf
    o convertirlos con gdal_translate.
    """
    if save_dir is None:
        save_dir = SATELLITE_RAW_DIR / "modis"
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    session = _get_nasa_session()

    # Construir URL de búsqueda en LAADS DAAC
    search_url = (
        f"{NASA_EARTHDATA_URL}/api/v1/files/product"
        f"?products={collection}"
        f"&temporalRanges={date_from}..{date_to}"
        f"&spatialPattern=BBOX"
        f"&coordinates={bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
        f"&resultLimit=100"
    )

    logger.info(f"Buscando archivos MODIS {collection} para bbox {bbox}")
    downloaded = []

    try:
        resp = session.get(search_url, timeout=30)
        resp.raise_for_status()
        files_info = resp.json()

        if not files_info:
            logger.warning("No se encontraron archivos MODIS para los parámetros dados.")
            return []

        logger.info(f"Encontrados {len(files_info)} archivos — descargando...")

        for file_info in files_info:
            file_url = file_info.get("fileURL") or file_info.get("downloadsLink")
            if not file_url:
                continue

            filename = Path(file_url).name
            local_path = save_dir / filename

            if local_path.exists():
                logger.info(f"Ya existe (omitiendo): {filename}")
                downloaded.append(local_path)
                continue

            try:
                with session.get(file_url, stream=True, timeout=120) as r:
                    r.raise_for_status()
                    with open(local_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                logger.info(f"Descargado: {filename} ({local_path.stat().st_size / 1024:.1f} KB)")
                downloaded.append(local_path)
                time.sleep(0.5)
            except Exception as e:
                logger.error(f"Error descargando {filename}: {e}")

    except requests.exceptions.HTTPError as e:
        logger.error(f"Error HTTP buscando archivos MODIS: {e}")
    except Exception as e:
        logger.error(f"Error inesperado: {e}")

    return downloaded


# ─────────────────────────────────────────────
# MERRA-2
# ─────────────────────────────────────────────

# Variables MERRA-2 relevantes para calidad del aire
MERRA2_VARIABLES = {
    "DUSMASS25":   "Concentración de polvo sahariano PM2.5 (kg/m³)",
    "PM25_RH35_GCC": "PM2.5 a HR 35% — GOCART (µg/m³)",
    "BCSMASS":     "Masa de carbono negro (kg/m³)",
    "OCSMASS":     "Masa de carbono orgánico (kg/m³)",
    "SO4SMASS":    "Masa de sulfatos (kg/m³)",
    "SSSMASS25":   "Masa de sal marina PM2.5 (kg/m³)",
}

# Colecciones MERRA-2 en GES DISC
MERRA2_COLLECTIONS = {
    "M2T1NXAER": "Aerosoles horarios (tavg1_2d_aer_Nx)",
    "M2TMNXAER": "Aerosoles mensuales (tavgM_2d_aer_Nx)",
}


def load_merra2_nc(
    filepath: str | Path,
    variables: list[str] | None = None,
    bbox: list[float] | None = None,
) -> "xr.Dataset":
    """
    Carga un archivo NetCDF de MERRA-2 usando xarray.
    Extrae las variables relevantes de calidad del aire y aplica
    recorte espacial opcional.

    Parámetros
    ----------
    filepath : str | Path
        Ruta al archivo .nc4 o .nc de MERRA-2.
    variables : list[str] | None
        Lista de variables a extraer. Default: todas las de MERRA2_VARIABLES.
    bbox : list[float] | None
        [lon_min, lat_min, lon_max, lat_max] para recorte espacial.

    Retorna
    -------
    xr.Dataset con las variables seleccionadas y coordenadas (lat, lon, time).

    Notas sobre descarga de MERRA-2
    --------------------------------
    1. Registrarse en https://urs.earthdata.nasa.gov/
    2. Aprobar "NASA GESDISC DATA ARCHIVE" en Applications > Authorized Apps
    3. Descargar vía OPeNDAP o GES DISC:
       https://disc.gsfc.nasa.gov/datasets/M2T1NXAER_5.12.4/summary
    4. Ejemplo wget:
       wget --load-cookies ~/.urs_cookies --save-cookies ~/.urs_cookies \\
            --auth-no-challenge=on --keep-session-cookies \\
            "URL_DEL_ARCHIVO"
    """
    if not HAS_XARRAY:
        raise ImportError("xarray es necesario. Instalar: pip install xarray netCDF4")

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {filepath}")

    logger.info(f"Cargando MERRA-2: {filepath.name}")

    # Abrir dataset
    ds = xr.open_dataset(filepath, engine="netcdf4")
    logger.info(f"Variables disponibles: {list(ds.data_vars)}")
    logger.info(f"Coordenadas: {list(ds.coords)}")

    # Seleccionar variables solicitadas
    if variables is None:
        variables = [v for v in MERRA2_VARIABLES if v in ds.data_vars]
        if not variables:
            logger.warning(
                "No se encontraron variables MERRA-2 conocidas. "
                f"Variables disponibles: {list(ds.data_vars)}"
            )
    else:
        variables = [v for v in variables if v in ds.data_vars]
        not_found = [v for v in variables if v not in ds.data_vars]
        if not_found:
            logger.warning(f"Variables no encontradas en el archivo: {not_found}")

    if variables:
        ds = ds[variables]

    # Recorte espacial
    if bbox is not None:
        lon_min, lat_min, lon_max, lat_max = bbox
        # MERRA-2 usa lat/lon (o lat/lon en diferentes nombres según versión)
        lat_dim = "lat" if "lat" in ds.coords else "latitude"
        lon_dim = "lon" if "lon" in ds.coords else "longitude"
        ds = ds.sel(
            {lat_dim: slice(lat_min, lat_max),
             lon_dim: slice(lon_min, lon_max)}
        )
        logger.info(f"Recortado a bbox {bbox}: {ds.dims}")

    # Añadir atributos descriptivos
    for var in ds.data_vars:
        if var in MERRA2_VARIABLES:
            ds[var].attrs["descripcion"] = MERRA2_VARIABLES[var]

    logger.info(f"Dataset cargado: {dict(ds.dims)} | Variables: {list(ds.data_vars)}")
    return ds


def merra2_to_dataframe(
    ds: "xr.Dataset",
    variable: str,
) -> pd.DataFrame:
    """
    Convierte una variable de un xr.Dataset MERRA-2 a DataFrame largo.

    Retorna DataFrame con columnas: [lat, lon, datetime, variable, valor]
    """
    if not HAS_XARRAY:
        raise ImportError("xarray es necesario. Instalar: pip install xarray netCDF4")

    if variable not in ds.data_vars:
        raise KeyError(f"Variable '{variable}' no encontrada. Disponibles: {list(ds.data_vars)}")

    da = ds[variable]
    df = da.to_dataframe().reset_index()

    # Renombrar columnas de tiempo
    time_cols = [c for c in df.columns if "time" in c.lower()]
    if time_cols:
        df = df.rename(columns={time_cols[0]: "datetime"})

    # Renombrar coordenadas a lat/lon
    lat_cols = [c for c in df.columns if c in ("lat", "latitude")]
    lon_cols = [c for c in df.columns if c in ("lon", "longitude")]
    if lat_cols and lat_cols[0] != "lat":
        df = df.rename(columns={lat_cols[0]: "lat"})
    if lon_cols and lon_cols[0] != "lon":
        df = df.rename(columns={lon_cols[0]: "lon"})

    df = df.rename(columns={variable: "valor"})
    df["variable"] = variable
    df = df[["lat", "lon", "datetime", "variable", "valor"]].dropna(subset=["valor"])
    logger.info(f"DataFrame MERRA-2 '{variable}': {len(df)} filas")
    return df


def fetch_merra2_opendap(
    variable: str,
    date: str,
    bbox: list[float],
    save_dir: Path | None = None,
) -> Path | None:
    """
    Descarga un archivo MERRA-2 para una fecha y variable específicas
    usando el endpoint OPeNDAP de GES DISC.

    Parámetros
    ----------
    variable : str
        Variable MERRA-2 (ej: 'DUSMASS25').
    date : str
        Fecha 'YYYY-MM-DD'.
    bbox : list[float]
        [lon_min, lat_min, lon_max, lat_max].
    save_dir : Path | None
        Directorio de guardado.

    Retorna
    -------
    Path al archivo descargado, o None si falló.

    Nota: Requiere credenciales NASA EarthData configuradas.
    """
    if save_dir is None:
        save_dir = SATELLITE_RAW_DIR / "merra2"
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Construir URL OPeNDAP (colección M2T1NXAER)
    dt = datetime.strptime(date, "%Y-%m-%d")
    year, month, day = dt.year, dt.month, dt.day

    base_url = "https://goldsmr4.gesdisc.eosdis.nasa.gov/data"
    collection = "MERRA2/M2T1NXAER.5.12.4"
    filename = f"MERRA2_400.tavg1_2d_aer_Nx.{year}{month:02d}{day:02d}.nc4"
    url = f"{base_url}/{collection}/{year}/{month:02d}/{filename}"

    local_path = save_dir / filename
    if local_path.exists():
        logger.info(f"Ya existe: {filename}")
        return local_path

    session = _get_nasa_session()
    logger.info(f"Descargando MERRA-2: {filename}")

    try:
        with session.get(url, stream=True, timeout=300) as r:
            if r.status_code == 401:
                logger.error(
                    "Error 401: Credenciales inválidas o no tienes acceso a GES DISC.\n"
                    "Solución: Visita https://urs.earthdata.nasa.gov/ → Applications → "
                    "Authorized Apps → Approve 'NASA GESDISC DATA ARCHIVE'"
                )
                return None
            r.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=32768):
                    f.write(chunk)
        size_mb = local_path.stat().st_size / 1024 / 1024
        logger.info(f"Descargado: {filename} ({size_mb:.1f} MB)")
        return local_path
    except Exception as e:
        logger.error(f"Error descargando MERRA-2: {e}")
        return None


# ─────────────────────────────────────────────
# EJEMPLO DE USO
# ─────────────────────────────────────────────
if __name__ == "__main__":
    GUATEMALA_BBOX = [-92.2, 13.7, -88.2, 17.8]

    print("=== Descarga MODIS AOD para Guatemala ===")
    archivos = fetch_modis_aod(
        bbox=GUATEMALA_BBOX,
        date_from="2024-01-01",
        date_to="2024-01-31",
        collection="MOD08_M3",
    )
    print(f"Archivos descargados: {len(archivos)}")

    print("\n=== Cargar NetCDF MERRA-2 (si existe) ===")
    ejemplo_nc = SATELLITE_RAW_DIR / "merra2" / "ejemplo.nc4"
    if ejemplo_nc.exists():
        ds = load_merra2_nc(str(ejemplo_nc), bbox=GUATEMALA_BBOX)
        print(ds)
        df = merra2_to_dataframe(ds, "DUSMASS25")
        print(df.head())
    else:
        print(f"Archivo de ejemplo no encontrado en {ejemplo_nc}")
        print("Descarga un archivo MERRA-2 de: https://disc.gsfc.nasa.gov/")
