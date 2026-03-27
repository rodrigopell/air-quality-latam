"""
load_era5_local.py — Carga de datos ERA5 descargados localmente
                     Península de Yucatán / Quintana Roo

Región primaria: Península de Yucatán — datos REALES con cobertura completa.
Guatemala es región de expansión; para Guatemala usar fetch_cams.py vía API.

Archivos fuente (obtenidos de Copernicus Climate Data Store):
  Viento:        era_5_v_YYYY.nc  → variables u10, v10  [m s⁻¹]
  Precipitación: era_5_p_YYYY.nc  → variable  tp        [m] (acumulado horario)
  Temperatura:   era_5_t_YYYY.nc  → variable  t2m       [K] → convertida a °C

Cobertura espacial: ~17–23°N, 89.6–86.6°W (grilla 0.25°, 25 lat × 13 lon)
Cobertura temporal:
  - Serie corta (Datos_entrada): 2020–2025, resolución horaria
  - Serie larga (ERA_5_V/P/T):  1996–2026, resolución horaria

Uso rápido:
    from 01_ingestion.load_era5_local import load_era5, era5_to_dataframe
    ds = load_era5("viento", years=range(2020, 2026))
    df = era5_to_dataframe(ds, lat=21.17, lon=-86.83)  # serie puntual Cancún
    df = era5_to_dataframe(ds, lat=20.97, lon=-89.62)  # serie puntual Mérida

Autor: Rodrigo, Ingeniero Ambiental, Universidad Anáhuac Cancún
"""

import logging
import re
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

try:
    import xarray as xr
    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False

try:
    import netCDF4 as nc4
    HAS_NC4 = True
except ImportError:
    HAS_NC4 = False

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import ERA5_SUBDIRS, ERA5_LONG_DIRS

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ─────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────

# Variable principal en cada tipo de archivo
ERA5_MAIN_VAR = {
    "viento":        ["u10", "v10"],   # componentes U y V del viento a 10 m
    "precipitacion": ["tp"],           # precipitación total acumulada [m]
    "temperatura":   ["t2m"],          # temperatura a 2 m [K]
}

# Patrón de nombre de archivo: año extraído del nombre
_FNAME_YEAR_RE = re.compile(r"(\d{4})")


def _find_nc_files(variable: str, years=None, use_long_series: bool = False) -> list[Path]:
    """
    Devuelve lista ordenada de archivos .nc para el tipo de variable y años solicitados.

    Parameters
    ----------
    variable : str
        Uno de: 'viento', 'precipitacion', 'temperatura'
    years : iterable[int] | None
        Años a cargar. None carga todos los disponibles.
    use_long_series : bool
        True = usar carpeta ERA_5_V/P/T (1996–2026).
        False = usar Datos_entrada (2020–2025).
    """
    dirs = ERA5_LONG_DIRS if use_long_series else ERA5_SUBDIRS
    if variable not in dirs:
        raise ValueError(f"Variable desconocida '{variable}'. Opciones: {list(dirs)}")

    folder = Path(dirs[variable])
    if not folder.exists():
        raise FileNotFoundError(f"Carpeta ERA5 no encontrada: {folder}")

    files = sorted(folder.glob("*.nc"))
    if not files:
        raise FileNotFoundError(f"No hay archivos .nc en {folder}")

    if years is not None:
        year_set = set(years)
        files = [
            f for f in files
            if (m := _FNAME_YEAR_RE.search(f.stem)) and int(m.group(1)) in year_set
        ]
        if not files:
            raise FileNotFoundError(
                f"No se encontraron archivos para años {sorted(year_set)} en {folder}"
            )

    logger.info(f"ERA5 '{variable}': {len(files)} archivos encontrados en {folder}")
    return files


# ─────────────────────────────────────────────
# CARGA PRINCIPAL
# ─────────────────────────────────────────────

def load_era5(
    variable: str,
    years=None,
    use_long_series: bool = False,
) -> "xr.Dataset":
    """
    Carga y concatena archivos ERA5 locales en un Dataset xarray.

    Parameters
    ----------
    variable : str
        'viento' | 'precipitacion' | 'temperatura'
    years : iterable[int] | None
        Años a incluir. None = todos los disponibles.
    use_long_series : bool
        True = serie larga 1996–2026. False = serie corta 2020–2025.

    Returns
    -------
    xr.Dataset con coordenadas (valid_time, latitude, longitude) y las
    variables meteorológicas con sus atributos originales (CF-1.7).

    Raises
    ------
    ImportError si xarray no está instalado.
    FileNotFoundError si no se encuentran archivos.
    """
    if not HAS_XARRAY:
        raise ImportError("Instalar xarray: pip install xarray netCDF4")

    files = _find_nc_files(variable, years=years, use_long_series=use_long_series)

    # Abrir todos los archivos y concatenar en tiempo
    datasets = []
    for f in files:
        try:
            ds = xr.open_dataset(f, engine="netcdf4")
            # Renombrar dimensión de tiempo si es necesario
            if "valid_time" in ds.dims:
                ds = ds.rename({"valid_time": "time"})
            datasets.append(ds)
        except Exception as e:
            logger.warning(f"No se pudo abrir {f.name}: {e}")

    if not datasets:
        raise FileNotFoundError("Ningún archivo ERA5 se pudo abrir.")

    ds_all = xr.concat(datasets, dim="time")

    # Convertir tiempo a datetime64 UTC
    if "time" in ds_all.coords and ds_all["time"].dtype != "datetime64[ns]":
        times_s = ds_all["time"].values.astype("int64")
        times_dt = pd.to_datetime(times_s, unit="s", utc=True)
        ds_all = ds_all.assign_coords(time=times_dt)

    # Ajustes específicos por variable
    if variable == "temperatura" and "t2m" in ds_all:
        ds_all["t2m_C"] = ds_all["t2m"] - 273.15
        ds_all["t2m_C"].attrs = {
            "long_name": "2 metre temperature",
            "units": "°C",
        }

    if variable == "viento" and "u10" in ds_all and "v10" in ds_all:
        ds_all["wind_speed"] = np.sqrt(ds_all["u10"] ** 2 + ds_all["v10"] ** 2)
        ds_all["wind_speed"].attrs = {
            "long_name": "10 metre wind speed",
            "units": "m s**-1",
        }
        ds_all["wind_dir"] = (
            np.degrees(np.arctan2(ds_all["u10"], ds_all["v10"])) % 360
        )
        ds_all["wind_dir"].attrs = {
            "long_name": "10 metre wind direction (meteorological)",
            "units": "degrees",
        }

    if variable == "precipitacion" and "tp" in ds_all:
        ds_all["tp_mm"] = ds_all["tp"] * 1000.0
        ds_all["tp_mm"].attrs = {
            "long_name": "Total precipitation",
            "units": "mm",
        }

    logger.info(
        f"ERA5 '{variable}' cargado: {ds_all.sizes.get('time', '?')} pasos de tiempo, "
        f"grilla {ds_all.sizes.get('latitude', '?')}×{ds_all.sizes.get('longitude', '?')}"
    )
    return ds_all


# ─────────────────────────────────────────────
# EXTRACCIÓN DE SERIES TEMPORALES
# ─────────────────────────────────────────────

def era5_to_dataframe(
    ds: "xr.Dataset",
    lat: float,
    lon: float,
    method: str = "nearest",
) -> pd.DataFrame:
    """
    Extrae serie temporal de un punto geográfico del Dataset ERA5.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset devuelto por load_era5().
    lat, lon : float
        Coordenadas del punto de interés (WGS84).
    method : str
        Método de selección espacial: 'nearest' (por defecto).

    Returns
    -------
    pd.DataFrame con columna 'datetime' (UTC) y una columna por variable.
    """
    if not HAS_XARRAY:
        raise ImportError("Instalar xarray: pip install xarray netCDF4")

    pt = ds.sel(latitude=lat, longitude=lon, method=method)
    df = pt.to_dataframe().reset_index()

    # Conservar sólo columnas útiles
    drop_cols = {"number", "expver", "latitude", "longitude"}
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    df = df.rename(columns={"time": "datetime"})

    logger.info(
        f"Serie extraída en lat={lat:.4f}, lon={lon:.4f}: {len(df)} registros"
    )
    return df


def era5_regional_mean(
    ds: "xr.Dataset",
    lat_min: float = 17.0,
    lat_max: float = 23.0,
    lon_min: float = -89.6,
    lon_max: float = -86.6,
) -> pd.DataFrame:
    """
    Calcula la media espacial del Dataset ERA5 dentro del bounding box dado.

    Por defecto usa toda la región de Quintana Roo cubierta por los archivos.
    """
    if not HAS_XARRAY:
        raise ImportError("Instalar xarray: pip install xarray netCDF4")

    region = ds.sel(
        latitude=slice(lat_max, lat_min),   # ERA5: latitud decrece
        longitude=slice(lon_min, lon_max),
    )
    mean_ds = region.mean(dim=["latitude", "longitude"])
    df = mean_ds.to_dataframe().reset_index()
    df = df.drop(columns=[c for c in ["number", "expver"] if c in df.columns])
    df = df.rename(columns={"time": "datetime"})
    return df


# ─────────────────────────────────────────────
# CARGA BÁSICA SIN XARRAY (fallback con netCDF4)
# ─────────────────────────────────────────────

def load_era5_nc4(variable: str, year: int, use_long_series: bool = False) -> dict:
    """
    Carga un único año de ERA5 usando netCDF4 (sin xarray).
    Retorna diccionario con arrays numpy y metadatos.

    Útil si xarray no está disponible.
    """
    if not HAS_NC4:
        raise ImportError("Instalar netCDF4: pip install netCDF4")

    files = _find_nc_files(variable, years=[year], use_long_series=use_long_series)
    if not files:
        raise FileNotFoundError(f"Archivo ERA5 '{variable}' año {year} no encontrado")

    f = nc4.Dataset(files[0])
    times_s = f.variables["valid_time"][:].data.astype("int64")
    datetimes = pd.to_datetime(times_s, unit="s", utc=True)

    result = {
        "datetime": datetimes,
        "latitude": f.variables["latitude"][:].data,
        "longitude": f.variables["longitude"][:].data,
    }

    for var_name in ERA5_MAIN_VAR.get(variable, []):
        if var_name in f.variables:
            result[var_name] = f.variables[var_name][:]
            result[f"{var_name}_units"] = getattr(f.variables[var_name], "units", "")
            result[f"{var_name}_long_name"] = getattr(
                f.variables[var_name], "long_name", var_name
            )

    f.close()

    # Conversiones
    if "t2m" in result:
        result["t2m_C"] = result["t2m"] - 273.15

    if "u10" in result and "v10" in result:
        result["wind_speed"] = np.sqrt(result["u10"] ** 2 + result["v10"] ** 2)

    if "tp" in result:
        result["tp_mm"] = result["tp"] * 1000.0

    logger.info(f"ERA5 nc4 '{variable}' {year}: {len(datetimes)} registros cargados")
    return result


# ─────────────────────────────────────────────
# DEMO / USO DIRECTO
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=== ERA5 local — demostración ===\n")

    for var in ["viento", "precipitacion", "temperatura"]:
        try:
            ds = load_era5(var, years=[2022], use_long_series=False)
            df = era5_to_dataframe(ds, lat=21.17, lon=-86.83)  # Cancún aprox.
            print(f"{var}: {len(df)} filas | columnas: {list(df.columns)}")
            print(df.head(3).to_string(), "\n")
        except Exception as e:
            print(f"{var}: ERROR — {e}\n")
