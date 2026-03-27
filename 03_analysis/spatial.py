"""
spatial.py — Análisis espacial avanzado para datos de calidad del aire
Incluye: hotspots, mapas de concentración, patrones estacionales,
análisis de transporte y mapas de excedencia.

Autor: Rodrigo, Ingeniero Ambiental, Universidad Anáhuac Cancún
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    import matplotlib.cm as cm
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import contextily as ctx
    HAS_CONTEXTILY = True
except ImportError:
    HAS_CONTEXTILY = False
    logging.warning("contextily no disponible (mapas base) — instalar: pip install contextily")

try:
    import esda
    import libpysal
    HAS_ESDA = True
except ImportError:
    HAS_ESDA = False
    logging.warning("esda/libpysal no disponibles (Moran) — instalar: pip install esda libpysal")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import FIGURES_DIR, MAPS_DIR, WHO_GUIDELINES_2021, AQI_CATEGORIES

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
)

ESTACIONES_DEL_ANIO = {
    "Seca": [11, 12, 1, 2, 3, 4],
    "Lluviosa": [5, 6, 7, 8, 9, 10],
}


# ─────────────────────────────────────────────
# 1. ANÁLISIS DE HOTSPOTS (Moran Local)
# ─────────────────────────────────────────────

def hotspot_analysis(
    gdf: gpd.GeoDataFrame,
    variable: str,
    k_neighbors: int = 5,
    significance: float = 0.05,
) -> gpd.GeoDataFrame:
    """
    Análisis de hotspots usando el estadístico I de Moran local (LISA).
    Identifica clusters de altas concentraciones (High-High) y bajas (Low-Low).

    Parámetros
    ----------
    gdf : gpd.GeoDataFrame
        Estaciones con columna `variable`.
    variable : str
        Variable a analizar.
    k_neighbors : int
        K vecinos más cercanos para la matriz de pesos espaciales.
    significance : float
        Nivel de significancia para identificar hotspots.

    Retorna
    -------
    gpd.GeoDataFrame con columnas adicionales:
        - moran_ii: estadístico I local
        - moran_p: p-value
        - hotspot_tipo: 'HH' (hotspot), 'LL' (coldspot), 'HL', 'LH', 'NS'
    """
    if not HAS_ESDA:
        logger.warning(
            "esda no disponible para análisis de Moran. "
            "Retornando GeoDataFrame sin análisis.\n"
            "Instalar: pip install esda libpysal"
        )
        gdf = gdf.copy()
        gdf["hotspot_tipo"] = "NS"
        gdf["moran_ii"]     = np.nan
        gdf["moran_p"]      = np.nan
        return gdf

    gdf_clean = gdf[gdf[variable].notna()].copy()
    if len(gdf_clean) < k_neighbors + 1:
        logger.warning(f"Muy pocos puntos ({len(gdf_clean)}) para análisis de Moran")
        gdf["hotspot_tipo"] = "NS"
        return gdf

    gdf_clean = gdf_clean.to_crs("EPSG:4326")

    # Matriz de pesos espaciales: K vecinos más cercanos
    w = libpysal.weights.KNN.from_dataframe(gdf_clean, k=k_neighbors)
    w.transform = "R"  # estandarización por filas

    # Moran Local
    y = gdf_clean[variable].values
    local_moran = esda.Moran_Local(y, w, permutations=999)

    gdf_clean["moran_ii"] = local_moran.Is
    gdf_clean["moran_p"]  = local_moran.p_sim

    # Clasificar tipos de hotspot
    mean_y = y.mean()
    tipos = []
    for i, (ii, p, val) in enumerate(zip(local_moran.Is, local_moran.p_sim, y)):
        if p >= significance:
            tipos.append("NS")  # no significativo
        elif ii > 0 and val >= mean_y:
            tipos.append("HH")  # hotspot: concentraciones altas rodeadas de altas
        elif ii > 0 and val < mean_y:
            tipos.append("LL")  # coldspot: concentraciones bajas rodeadas de bajas
        elif ii < 0 and val >= mean_y:
            tipos.append("HL")  # outlier: alta rodeada de bajas
        else:
            tipos.append("LH")  # outlier: baja rodeada de altas

    gdf_clean["hotspot_tipo"] = tipos

    n_hotspots = sum(t == "HH" for t in tipos)
    logger.info(
        f"Análisis Moran para '{variable}': {n_hotspots} hotspots HH, "
        f"{sum(t == 'LL' for t in tipos)} coldspots LL"
    )

    # Unir resultados con GeoDataFrame original
    result = gdf.merge(
        gdf_clean[["hotspot_tipo", "moran_ii", "moran_p"]],
        left_index=True, right_index=True, how="left"
    )
    result["hotspot_tipo"] = result.get("hotspot_tipo_y", result.get("hotspot_tipo", "NS")).fillna("NS")
    return result


# ─────────────────────────────────────────────
# 2. MAPA DE CONCENTRACIONES
# ─────────────────────────────────────────────

def concentration_map(
    gdf: gpd.GeoDataFrame,
    variable: str,
    output_path: str | Path | None = None,
    title: str | None = None,
    cmap: str = "YlOrRd",
    basemap: bool = True,
    dpi: int = 150,
) -> Path | None:
    """
    Genera un mapa estático de concentraciones usando matplotlib + contextily.

    Parámetros
    ----------
    gdf : gpd.GeoDataFrame
        Puntos o polígonos con columna `variable`.
    variable : str
    output_path : str | Path | None
    title : str | None
    cmap : str
        Colormap de matplotlib.
    basemap : bool
        Si True, añade mapa base OpenStreetMap (requiere contextily).
    dpi : int

    Retorna
    -------
    Path al PNG generado.
    """
    if not HAS_MATPLOTLIB:
        logger.error("matplotlib no disponible")
        return None

    if output_path is None:
        var_safe = variable.replace(".", "").replace(" ", "_")
        output_path = MAPS_DIR / f"conc_{var_safe}.png"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    gdf_plot = gdf[gdf[variable].notna()].copy()
    if len(gdf_plot) == 0:
        logger.warning(f"Sin datos válidos de '{variable}' para mapear")
        return None

    # Reproyectar a Web Mercator para basemap
    if basemap and HAS_CONTEXTILY:
        gdf_plot = gdf_plot.to_crs("EPSG:3857")

    fig, ax = plt.subplots(figsize=(10, 8))

    # Determinar si son puntos o polígonos
    geom_type = gdf_plot.geometry.geom_type.iloc[0]
    vmin, vmax = gdf_plot[variable].quantile(0.02), gdf_plot[variable].quantile(0.98)

    if "Point" in geom_type:
        sc = gdf_plot.plot(
            column=variable, ax=ax, cmap=cmap,
            vmin=vmin, vmax=vmax,
            markersize=80, alpha=0.8,
            legend=True,
            legend_kwds={"label": f"{variable} (µg/m³)", "shrink": 0.6},
        )
    else:
        gdf_plot.plot(
            column=variable, ax=ax, cmap=cmap,
            vmin=vmin, vmax=vmax, alpha=0.7,
            legend=True,
            legend_kwds={"label": f"{variable} (µg/m³)", "shrink": 0.6},
        )

    # Añadir mapa base
    if basemap and HAS_CONTEXTILY:
        try:
            ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, zoom="auto")
        except Exception as e:
            logger.debug(f"No se pudo añadir mapa base: {e}")

    ax.set_title(title or f"Concentración de {variable}", fontsize=13)
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    logger.info(f"Mapa guardado: {output_path}")
    return output_path


# ─────────────────────────────────────────────
# 3. PATRÓN ESPACIAL ESTACIONAL
# ─────────────────────────────────────────────

def seasonal_spatial_pattern(
    gdf: gpd.GeoDataFrame,
    variable: str,
    datetime_col: str = "datetime",
) -> dict:
    """
    Calcula promedios por estación del año (seca/lluviosa para LATAM)
    y devuelve un dict con GeoDataFrames por estación.

    Parámetros
    ----------
    gdf : gpd.GeoDataFrame
        Con columna `datetime` y `variable`.
    variable : str
    datetime_col : str

    Retorna
    -------
    dict {nombre_estacion: GeoDataFrame_con_promedio}
    """
    if datetime_col not in gdf.columns:
        raise ValueError(f"Columna '{datetime_col}' no encontrada")

    gdf = gdf.copy()
    gdf[datetime_col] = pd.to_datetime(gdf[datetime_col])
    gdf["mes"] = gdf[datetime_col].dt.month

    station_col = next((c for c in ["station_id", "location_id"] if c in gdf.columns), None)
    if not station_col:
        logger.warning("Sin columna de ID de estación — usando geometría como identificador")

    resultados = {}
    for nombre_estacion, meses in ESTACIONES_DEL_ANIO.items():
        mask = gdf["mes"].isin(meses)
        sub = gdf[mask]

        if len(sub) == 0:
            logger.warning(f"Sin datos para estación '{nombre_estacion}'")
            continue

        if station_col:
            promedios = (sub.groupby(station_col)[variable]
                         .mean()
                         .reset_index()
                         .rename(columns={variable: f"{variable}_promedio"}))
            gdf_est = gdf[[station_col, "geometry"]].drop_duplicates(station_col)
            gdf_est = gdf_est.merge(promedios, on=station_col, how="left")
        else:
            promedios = sub.groupby("geometry")[variable].mean().reset_index()
            gdf_est = promedios

        resultados[nombre_estacion] = gdf_est
        logger.info(
            f"Estación '{nombre_estacion}': {len(gdf_est)} estaciones, "
            f"promedio {variable}={gdf_est[f'{variable}_promedio'].mean():.2f}"
        )

    return resultados


# ─────────────────────────────────────────────
# 4. ANÁLISIS DE TRANSPORTE DE POLVO
# ─────────────────────────────────────────────

def transport_corridor_analysis(
    gdf: gpd.GeoDataFrame,
    variable: str,
    wind_dir_col: str = "direccion_viento",
    wind_speed_col: str = "velocidad_viento",
    n_sectors: int = 8,
) -> pd.DataFrame:
    """
    Analiza la relación entre dirección de viento y concentraciones de contaminante.
    Útil para identificar corredores de transporte de polvo sahariano.

    Parámetros
    ----------
    gdf : gpd.GeoDataFrame
        Con columnas de contaminante y viento.
    variable : str
        Contaminante a analizar.
    wind_dir_col : str
        Columna de dirección de viento (0-360°).
    wind_speed_col : str
        Columna de velocidad de viento.
    n_sectors : int
        Número de sectores direccionales (8 = N, NE, E, SE, S, SW, W, NW).

    Retorna
    -------
    pd.DataFrame con concentración promedio por sector de viento.
    """
    required = [variable, wind_dir_col]
    missing = [c for c in required if c not in gdf.columns]
    if missing:
        raise KeyError(f"Columnas no encontradas: {missing}")

    df = gdf[[variable, wind_dir_col] +
             ([wind_speed_col] if wind_speed_col in gdf.columns else [])].copy()
    df = df.dropna(subset=required)

    # Discretizar dirección en sectores
    sector_size = 360 / n_sectors
    sector_labels = {
        8: ["N", "NE", "E", "SE", "S", "SO", "O", "NO"],
        4: ["N", "E", "S", "O"],
        16: [f"{i*22.5:.0f}°" for i in range(16)],
    }
    labels = sector_labels.get(n_sectors, [f"Sector_{i}" for i in range(n_sectors)])

    df["sector"] = pd.cut(
        (df[wind_dir_col] + sector_size / 2) % 360,
        bins=np.linspace(0, 360, n_sectors + 1),
        labels=labels,
        include_lowest=True,
    )

    resultados = (df.groupby("sector", observed=True)[variable]
                  .agg(["mean", "std", "count"])
                  .rename(columns={"mean": f"conc_media", "std": "conc_std", "count": "n"})
                  .round(3))

    if wind_speed_col in df.columns:
        velocidad_media = df.groupby("sector", observed=True)[wind_speed_col].mean().round(2)
        resultados["velocidad_media"] = velocidad_media

    logger.info(f"Análisis de transporte para '{variable}': {n_sectors} sectores")
    return resultados


# ─────────────────────────────────────────────
# 5. MAPA DE EXCEDENCIAS
# ─────────────────────────────────────────────

def exceedance_map(
    gdf: gpd.GeoDataFrame,
    variable: str,
    threshold: float | None = None,
    output_path: str | Path | None = None,
    temporal_col: str = "datetime",
    dpi: int = 150,
) -> Path | None:
    """
    Genera un mapa mostrando el % de tiempo que cada estación
    excede el umbral OMS (o personalizado).

    Parámetros
    ----------
    gdf : gpd.GeoDataFrame
    variable : str
    threshold : float | None
        Umbral en µg/m³. Si None, usa guía OMS 2021 24h.
    output_path : str | Path | None
    temporal_col : str
    dpi : int

    Retorna
    -------
    Path al PNG generado.
    """
    if not HAS_MATPLOTLIB:
        return None

    # Obtener umbral OMS si no se especifica
    if threshold is None:
        poll_key = variable.replace(".", "")
        if poll_key == "PM25":
            poll_key = "PM2.5"
        guidelines = WHO_GUIDELINES_2021.get(poll_key, {})
        threshold = guidelines.get("24h") or guidelines.get("8h") or 25.0
        logger.info(f"Usando umbral OMS para {variable}: {threshold} µg/m³")

    # Calcular % excedencia por estación
    station_col = next((c for c in ["station_id", "location_id"] if c in gdf.columns), None)
    if not station_col:
        logger.warning("Sin columna de ID de estación")
        return None

    if temporal_col in gdf.columns:
        gdf = gdf.copy()
        gdf[temporal_col] = pd.to_datetime(gdf[temporal_col])
        gdf["date"] = gdf[temporal_col].dt.date
        medias_diarias = gdf.groupby([station_col, "date"])[variable].mean()
        excedencias = (medias_diarias > threshold).groupby(level=0).mean() * 100
    else:
        excedencias = (gdf.groupby(station_col)[variable].apply(
            lambda s: (s.dropna() > threshold).mean() * 100
        ))

    excedencias = excedencias.reset_index(name="pct_excedencia")

    gdf_exc = (gdf[[station_col, "geometry"]]
               .drop_duplicates(station_col)
               .merge(excedencias, on=station_col))

    if output_path is None:
        var_safe = variable.replace(".", "").replace(" ", "_")
        output_path = MAPS_DIR / f"exceedance_{var_safe}.png"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))

    gdf_exc_proj = gdf_exc.to_crs("EPSG:3857") if HAS_CONTEXTILY else gdf_exc

    sc = gdf_exc_proj.plot(
        column="pct_excedencia", ax=ax,
        cmap="Reds", vmin=0, vmax=100,
        markersize=100, alpha=0.85,
        legend=True,
        legend_kwds={"label": "% días sobre umbral OMS", "shrink": 0.6},
    )

    if HAS_CONTEXTILY:
        try:
            ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
        except Exception:
            pass

    ax.set_title(f"% Excedencia {variable} (umbral OMS: {threshold} µg/m³)", fontsize=12)
    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    logger.info(f"Mapa de excedencia guardado: {output_path}")
    return output_path


# ─────────────────────────────────────────────
# EJEMPLO DE USO
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from ingestion.load_stations import generate_synthetic_data

    df = generate_synthetic_data(n_stations=10, days=90)
    df["datetime"] = pd.to_datetime(df["fecha"], format="%d/%m/%Y %H:%M")

    geometry = [Point(row["lon"], row["lat"]) for _, row in df.iterrows()]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    print("=== Mapa de concentraciones ===")
    path = concentration_map(gdf, "PM2.5", basemap=False)
    print(f"Guardado en: {path}")

    print("\n=== Patrón estacional ===")
    patrones = seasonal_spatial_pattern(gdf, "PM2.5")
    for est, gdf_est in patrones.items():
        col = "PM2.5_promedio"
        if col in gdf_est.columns:
            print(f"{est}: {gdf_est[col].mean():.2f} µg/m³")

    print("\n=== Mapa de excedencias ===")
    path_exc = exceedance_map(gdf, "PM2.5")
    print(f"Guardado en: {path_exc}")
