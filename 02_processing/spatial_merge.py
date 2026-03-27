"""
spatial_merge.py — Análisis y operaciones espaciales para datos de calidad del aire
Incluye: reproyección, joins espaciales, interpolación IDW/Kriging, conversión
raster→puntos y merge estación-satélite.

Autor: Rodrigo, Ingeniero Ambiental, Universidad Anáhuac Cancún
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

try:
    import xarray as xr
    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False

try:
    from pykrige.ok import OrdinaryKriging
    HAS_PYKRIGE = True
except ImportError:
    HAS_PYKRIGE = False
    logging.warning("pykrige no disponible — Kriging usará fallback IDW. Instalar: pip install pykrige")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DEFAULT_CRS, INTERPOLATION_POWER, DEFAULT_RESOLUTION_DEG, SPATIAL_DIR

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
)


# ─────────────────────────────────────────────
# 1. REPROYECCIÓN
# ─────────────────────────────────────────────

def reproject(gdf: gpd.GeoDataFrame, target_crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
    """
    Reproyecta un GeoDataFrame al CRS destino.

    Parámetros
    ----------
    gdf : gpd.GeoDataFrame
    target_crs : str
        CRS destino (ej: 'EPSG:4326', 'EPSG:32615').

    Retorna
    -------
    gpd.GeoDataFrame reproyectado.
    """
    if gdf.crs is None:
        logger.warning("GeoDataFrame sin CRS definido — asumiendo EPSG:4326")
        gdf = gdf.set_crs("EPSG:4326")

    if str(gdf.crs) == target_crs:
        logger.debug(f"Ya en CRS {target_crs} — sin reproyección")
        return gdf

    logger.info(f"Reproyectando: {gdf.crs} → {target_crs}")
    return gdf.to_crs(target_crs)


# ─────────────────────────────────────────────
# 2. JOIN ESPACIAL CON LÍMITES ADMINISTRATIVOS
# ─────────────────────────────────────────────

def _download_natural_earth(scale: str = "10m", category: str = "cultural",
                             name: str = "admin_1_states_provinces") -> gpd.GeoDataFrame:
    """
    Descarga límites administrativos de Natural Earth vía geopandas.
    Filtra por Guatemala y México.
    """
    try:
        url = f"https://naturalearth.s3.amazonaws.com/{scale}_{category}/{name}.zip"
        logger.info(f"Descargando Natural Earth: {url}")
        gdf = gpd.read_file(url)
        return gdf
    except Exception as e:
        logger.error(f"Error descargando Natural Earth: {e}")
        raise


def spatial_join_stations_admin(
    gdf_stations: gpd.GeoDataFrame,
    shapefile_admin: str | Path | None = None,
    country_filter: list[str] | None = None,
    admin_level: str = "departamento",
) -> gpd.GeoDataFrame:
    """
    Realiza un join espacial entre estaciones y límites administrativos.
    Si no se proporciona shapefile, descarga Guatemala desde Natural Earth.

    Parámetros
    ----------
    gdf_stations : gpd.GeoDataFrame
        GeoDataFrame de estaciones con geometría de puntos.
    shapefile_admin : str | Path | None
        Ruta al shapefile de límites administrativos.
        Si es None, descarga Guatemala desde Natural Earth.
    country_filter : list[str] | None
        Filtrar por código de país ISO ('GT', 'MX', etc.).
    admin_level : str
        Nombre descriptivo del nivel administrativo para logging.

    Retorna
    -------
    gpd.GeoDataFrame con columnas administrativas añadidas.
    """
    # Cargar o descargar límites administrativos
    if shapefile_admin is None:
        local_admin = SPATIAL_DIR / "admin_latam.gpkg"
        if local_admin.exists():
            logger.info(f"Cargando límites administrativos desde: {local_admin}")
            gdf_admin = gpd.read_file(local_admin)
        else:
            logger.info("Descargando límites administrativos de Natural Earth...")
            try:
                gdf_admin = _download_natural_earth()
                gdf_admin.to_file(local_admin, driver="GPKG")
                logger.info(f"Guardado en: {local_admin}")
            except Exception:
                logger.error("No se pudieron descargar los límites — join espacial omitido")
                return gdf_stations
    else:
        logger.info(f"Cargando shapefile: {shapefile_admin}")
        gdf_admin = gpd.read_file(shapefile_admin)

    # Filtrar por país si se especifica
    if country_filter:
        iso_col = next((c for c in gdf_admin.columns
                        if "iso" in c.lower() or "country" in c.lower()), None)
        if iso_col:
            gdf_admin = gdf_admin[gdf_admin[iso_col].isin(country_filter)]
            logger.info(f"Filtrado por países {country_filter}: {len(gdf_admin)} polígonos")

    # Asegurar mismo CRS
    gdf_admin = gdf_admin.to_crs(DEFAULT_CRS)
    gdf_stations = gdf_stations.to_crs(DEFAULT_CRS)

    # Join espacial
    logger.info(f"Ejecutando join espacial ({len(gdf_stations)} estaciones × {len(gdf_admin)} polígonos)")
    gdf_joined = gpd.sjoin(gdf_stations, gdf_admin, how="left", predicate="within")

    n_unmatched = gdf_joined["index_right"].isna().sum()
    if n_unmatched:
        logger.warning(f"{n_unmatched} estaciones fuera de los polígonos administrativos")

    logger.info(f"Join completado: {len(gdf_joined)} registros")
    return gdf_joined.drop(columns=["index_right"], errors="ignore")


# ─────────────────────────────────────────────
# 3. INTERPOLACIÓN IDW
# ─────────────────────────────────────────────

def interpolate_idw(
    gdf: gpd.GeoDataFrame,
    variable: str,
    bbox: list[float] | None = None,
    resolution: float = DEFAULT_RESOLUTION_DEG,
    power: float = INTERPOLATION_POWER,
    min_stations: int = 3,
) -> gpd.GeoDataFrame:
    """
    Interpolación Inverse Distance Weighting sobre un grid regular.

    Parámetros
    ----------
    gdf : gpd.GeoDataFrame
        Estaciones con geometría de puntos y columna `variable`.
    variable : str
        Columna a interpolar.
    bbox : list[float] | None
        [lon_min, lat_min, lon_max, lat_max]. Default: extent de los puntos.
    resolution : float
        Resolución del grid en grados.
    power : float
        Potencia del IDW (mayor = más influencia local).
    min_stations : int
        Mínimo de estaciones con datos válidos para interpolar.

    Retorna
    -------
    gpd.GeoDataFrame con grid interpolado y columna `variable`.
    """
    gdf_clean = gdf[gdf[variable].notna()].copy()
    if len(gdf_clean) < min_stations:
        logger.warning(
            f"Solo {len(gdf_clean)} estaciones con datos de '{variable}' "
            f"(mínimo: {min_stations}) — saltando IDW"
        )
        return gpd.GeoDataFrame()

    # Extraer coordenadas
    gdf_clean = gdf_clean.to_crs(DEFAULT_CRS)
    if "lat" not in gdf_clean.columns:
        gdf_clean["lon"] = gdf_clean.geometry.x
        gdf_clean["lat"] = gdf_clean.geometry.y

    lons_s = gdf_clean["lon"].values
    lats_s = gdf_clean["lat"].values
    vals_s = gdf_clean[variable].values

    # Definir bbox
    if bbox is None:
        bbox = [lons_s.min(), lats_s.min(), lons_s.max(), lats_s.max()]
    lon_min, lat_min, lon_max, lat_max = bbox

    # Crear grid
    grid_lons = np.arange(lon_min, lon_max + resolution, resolution)
    grid_lats = np.arange(lat_min, lat_max + resolution, resolution)
    grid_lon_2d, grid_lat_2d = np.meshgrid(grid_lons, grid_lats)

    # IDW vectorizado
    logger.info(f"IDW: {len(gdf_clean)} estaciones → grid {grid_lons.size}×{grid_lats.size} ({resolution}°)")
    grid_vals = _idw_vectorized(lons_s, lats_s, vals_s, grid_lon_2d, grid_lat_2d, power)

    # Construir GeoDataFrame de salida
    rows = []
    for i in range(grid_lats.size):
        for j in range(grid_lons.size):
            rows.append({
                "lat": grid_lat_2d[i, j],
                "lon": grid_lon_2d[i, j],
                variable: grid_vals[i, j],
                "geometry": Point(grid_lon_2d[i, j], grid_lat_2d[i, j]),
            })

    gdf_grid = gpd.GeoDataFrame(rows, crs=DEFAULT_CRS)
    logger.info(f"Grid IDW generado: {len(gdf_grid)} celdas")
    return gdf_grid


def _idw_vectorized(
    x_s: np.ndarray, y_s: np.ndarray, z_s: np.ndarray,
    x_grid: np.ndarray, y_grid: np.ndarray,
    power: float = 2.0,
    eps: float = 1e-10,
) -> np.ndarray:
    """
    Implementación vectorizada de IDW.
    Usa broadcasting de numpy para eficiencia.
    """
    z_grid = np.zeros_like(x_grid, dtype=float)

    for i in range(x_grid.shape[0]):
        for j in range(x_grid.shape[1]):
            dist = np.sqrt((x_s - x_grid[i, j])**2 + (y_s - y_grid[i, j])**2) + eps
            weights = 1.0 / dist**power
            z_grid[i, j] = np.sum(weights * z_s) / np.sum(weights)

    return z_grid


# ─────────────────────────────────────────────
# 4. INTERPOLACIÓN KRIGING
# ─────────────────────────────────────────────

def interpolate_kriging(
    gdf: gpd.GeoDataFrame,
    variable: str,
    bbox: list[float] | None = None,
    resolution: float = 0.05,
    variogram_model: str = "spherical",
) -> gpd.GeoDataFrame:
    """
    Kriging ordinario usando pykrige. Si no está disponible, usa fallback IDW.

    Parámetros
    ----------
    gdf : gpd.GeoDataFrame
        Estaciones con datos.
    variable : str
        Columna a interpolar.
    bbox : list[float] | None
        [lon_min, lat_min, lon_max, lat_max].
    resolution : float
        Resolución del grid en grados.
    variogram_model : str
        Modelo de variograma: 'spherical', 'exponential', 'gaussian', 'linear'.

    Retorna
    -------
    gpd.GeoDataFrame con grid interpolado.
    """
    if not HAS_PYKRIGE:
        logger.warning("pykrige no disponible — usando IDW como fallback")
        return interpolate_idw(gdf, variable, bbox, resolution)

    gdf_clean = gdf[gdf[variable].notna()].copy().to_crs(DEFAULT_CRS)
    if len(gdf_clean) < 4:
        logger.warning(f"Muy pocas estaciones para Kriging ({len(gdf_clean)}) — usando IDW")
        return interpolate_idw(gdf, variable, bbox, resolution)

    if "lat" not in gdf_clean.columns:
        gdf_clean["lon"] = gdf_clean.geometry.x
        gdf_clean["lat"] = gdf_clean.geometry.y

    lons_s = gdf_clean["lon"].values
    lats_s = gdf_clean["lat"].values
    vals_s = gdf_clean[variable].values

    if bbox is None:
        bbox = [lons_s.min(), lats_s.min(), lons_s.max(), lats_s.max()]
    lon_min, lat_min, lon_max, lat_max = bbox

    grid_lons = np.arange(lon_min, lon_max + resolution, resolution)
    grid_lats = np.arange(lat_min, lat_max + resolution, resolution)

    logger.info(
        f"Kriging ordinario: {len(gdf_clean)} estaciones | variograma: {variogram_model} | "
        f"grid {len(grid_lons)}×{len(grid_lats)}"
    )

    try:
        ok = OrdinaryKriging(
            lons_s, lats_s, vals_s,
            variogram_model=variogram_model,
            verbose=False,
            enable_plotting=False,
        )
        z_grid, sigma_grid = ok.execute("grid", grid_lons, grid_lats)
    except Exception as e:
        logger.error(f"Kriging falló: {e} — usando IDW como fallback")
        return interpolate_idw(gdf, variable, bbox, resolution)

    rows = []
    for i, lat in enumerate(grid_lats):
        for j, lon in enumerate(grid_lons):
            rows.append({
                "lat": lat,
                "lon": lon,
                variable: float(z_grid[i, j]),
                f"{variable}_variance": float(sigma_grid[i, j]),
                "geometry": Point(lon, lat),
            })

    gdf_grid = gpd.GeoDataFrame(rows, crs=DEFAULT_CRS)
    logger.info(f"Grid Kriging generado: {len(gdf_grid)} celdas")
    return gdf_grid


# ─────────────────────────────────────────────
# 5. RASTER → PUNTOS
# ─────────────────────────────────────────────

def raster_to_points(
    nc_dataset: "xr.Dataset",
    variable: str,
    bbox: list[float],
    time_index: int = 0,
) -> gpd.GeoDataFrame:
    """
    Extrae puntos de un raster xarray dentro de un bbox.

    Parámetros
    ----------
    nc_dataset : xr.Dataset
        Dataset con dimensiones (time, lat, lon) o similar.
    variable : str
        Variable a extraer.
    bbox : list[float]
        [lon_min, lat_min, lon_max, lat_max].
    time_index : int
        Índice de tiempo a extraer (si tiene dimensión temporal).

    Retorna
    -------
    gpd.GeoDataFrame con puntos y valor de la variable.
    """
    if not HAS_XARRAY:
        raise ImportError("xarray es necesario. Instalar: pip install xarray netCDF4")

    if variable not in nc_dataset.data_vars:
        raise KeyError(f"Variable '{variable}' no encontrada. Disponibles: {list(nc_dataset.data_vars)}")

    lon_min, lat_min, lon_max, lat_max = bbox
    lat_dim = "lat" if "lat" in nc_dataset.coords else "latitude"
    lon_dim = "lon" if "lon" in nc_dataset.coords else "longitude"

    # Recortar al bbox
    ds_crop = nc_dataset.sel(
        {lat_dim: slice(lat_min, lat_max),
         lon_dim: slice(lon_min, lon_max)}
    )

    da = ds_crop[variable]

    # Seleccionar tiempo si tiene esa dimensión
    if "time" in da.dims and len(da.time) > 0:
        da = da.isel(time=time_index)
        logger.info(f"Extrayendo tiempo[{time_index}]: {da.time.values if 'time' in da.coords else 'N/A'}")

    # Convertir a DataFrame
    df = da.to_dataframe().reset_index().dropna(subset=[variable])
    lat_col = lat_dim if lat_dim in df.columns else "lat"
    lon_col = lon_dim if lon_dim in df.columns else "lon"

    geometry = [Point(lon, lat) for lat, lon in zip(df[lat_col], df[lon_col])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=DEFAULT_CRS)
    gdf = gdf.rename(columns={lat_col: "lat", lon_col: "lon"})
    logger.info(f"Raster → puntos: {len(gdf)} puntos extraídos de bbox {bbox}")
    return gdf


# ─────────────────────────────────────────────
# 6. MERGE ESTACIÓN - SATÉLITE
# ─────────────────────────────────────────────

def merge_station_satellite(
    gdf_stations: gpd.GeoDataFrame,
    ds_satellite: "xr.Dataset",
    variable: str,
    method: str = "nearest",
    max_distance_km: float = 50.0,
    time_tolerance_hours: int = 3,
) -> pd.DataFrame:
    """
    Merge espaciotemporal entre mediciones de estaciones y datos satelitales.
    Encuentra el punto de satélite más cercano a cada estación.

    Parámetros
    ----------
    gdf_stations : gpd.GeoDataFrame
        Estaciones con geometría y columna 'datetime'.
    ds_satellite : xr.Dataset
        Dataset satelital con dimensiones (time, lat, lon).
    variable : str
        Variable satelital a extraer.
    method : str
        'nearest': punto más cercano.
        'bilinear': interpolación bilineal (si está disponible).
    max_distance_km : float
        Distancia máxima aceptable para el match (km).
    time_tolerance_hours : int
        Tolerancia temporal para el match (horas).

    Retorna
    -------
    pd.DataFrame con columnas de estación y columna satelital añadida.
    """
    if not HAS_XARRAY:
        raise ImportError("xarray es necesario para merge con satélite.")

    if variable not in ds_satellite.data_vars:
        raise KeyError(f"Variable '{variable}' no en dataset. Disponibles: {list(ds_satellite.data_vars)}")

    gdf_stations = gdf_stations.to_crs(DEFAULT_CRS).copy()
    if "lat" not in gdf_stations.columns:
        gdf_stations["lon"] = gdf_stations.geometry.x
        gdf_stations["lat"] = gdf_stations.geometry.y

    lat_dim = "lat" if "lat" in ds_satellite.coords else "latitude"
    lon_dim = "lon" if "lon" in ds_satellite.coords else "longitude"

    sat_lats = ds_satellite[lat_dim].values
    sat_lons = ds_satellite[lon_dim].values

    logger.info(
        f"Merge estación-satélite: {len(gdf_stations)} estaciones | "
        f"variable: {variable} | método: {method}"
    )

    resultados = []

    for _, row in gdf_stations.iterrows():
        # Encontrar índice lat/lon más cercano
        lat_idx = np.argmin(np.abs(sat_lats - row["lat"]))
        lon_idx = np.argmin(np.abs(sat_lons - row["lon"]))

        # Calcular distancia aproximada (en km, usando fórmula plana)
        dist_lat_km = abs(sat_lats[lat_idx] - row["lat"]) * 111.0
        dist_lon_km = abs(sat_lons[lon_idx] - row["lon"]) * 111.0 * np.cos(np.radians(row["lat"]))
        dist_km = np.sqrt(dist_lat_km**2 + dist_lon_km**2)

        if dist_km > max_distance_km:
            sat_value = np.nan
            logger.debug(f"Estación {row.get('station_id', '?')}: satélite más cercano a {dist_km:.1f} km > {max_distance_km} km")
        else:
            # Extraer valor (si hay dimensión temporal, buscar match temporal)
            if "time" in ds_satellite.dims and "datetime" in gdf_stations.columns:
                try:
                    sat_time = pd.to_datetime(ds_satellite.time.values)
                    station_dt = pd.to_datetime(row["datetime"])
                    time_diff = np.abs((sat_time - station_dt).total_seconds() / 3600)
                    time_idx = np.argmin(time_diff)
                    if time_diff[time_idx] <= time_tolerance_hours:
                        sat_value = float(ds_satellite[variable].isel(
                            {lat_dim: lat_idx, lon_dim: lon_idx, "time": time_idx}
                        ).values)
                    else:
                        sat_value = np.nan
                except Exception:
                    sat_value = float(ds_satellite[variable].isel(
                        {lat_dim: lat_idx, lon_dim: lon_idx}
                    ).values.flat[0])
            else:
                sat_value = float(ds_satellite[variable].isel(
                    {lat_dim: lat_idx, lon_dim: lon_idx}
                ).values.flat[0])

        resultado = row.to_dict()
        resultado[f"sat_{variable}"] = sat_value
        resultado["sat_dist_km"] = round(dist_km, 2)
        resultados.append(resultado)

    df_merged = pd.DataFrame(resultados)
    n_matched = df_merged[f"sat_{variable}"].notna().sum()
    logger.info(f"Merge completado: {n_matched}/{len(gdf_stations)} estaciones con dato satelital")
    return df_merged


# ─────────────────────────────────────────────
# EJEMPLO DE USO
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from ingestion.load_stations import generate_synthetic_data

    df_sint = generate_synthetic_data(n_stations=8, days=1)

    # Construir GeoDataFrame
    geometry = [Point(row["lon"], row["lat"]) for _, row in df_sint.iterrows()]
    gdf = gpd.GeoDataFrame(df_sint, geometry=geometry, crs="EPSG:4326")

    # IDW
    BBOX_GT = [-92.2, 13.7, -88.2, 17.8]
    gdf_grid = interpolate_idw(gdf, "PM2.5", bbox=BBOX_GT, resolution=0.5)
    print(f"Grid IDW: {len(gdf_grid)} celdas")
    print(gdf_grid.head())
