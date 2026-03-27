"""
maps.py — Mapas interactivos y animados de calidad del aire
Incluye: mapa folium con AQI, coroplético de municipios/departamentos,
animación temporal con Plotly y overlay de datos satelitales.

Autor: Rodrigo, Ingeniero Ambiental, Universidad Anáhuac Cancún
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd

try:
    import folium
    from folium import plugins as folium_plugins
    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False
    logging.warning("folium no disponible — instalar: pip install folium")

try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    logging.warning("plotly no disponible — instalar: pip install plotly")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import MAPS_DIR, AQI_CATEGORIES, WHO_GUIDELINES_2021, DEFAULT_CRS

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
)

# Opciones de tiles para folium
FOLIUM_TILES = {
    "osm":       "OpenStreetMap",
    "cartolight": "CartoDB positron",
    "cartodark":  "CartoDB dark_matter",
    "esri":       "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
}


def _aqi_to_color(value: float, vmin: float = 0, vmax: float = 100) -> str:
    """
    Mapea un valor numérico a un color hexadecimal basado en la escala AQI.
    Retorna color por defecto (#CCCCCC) si el valor es NaN.
    """
    if pd.isna(value):
        return "#CCCCCC"
    normalized = (value - vmin) / max(vmax - vmin, 1)
    for cat_name, cat_info in AQI_CATEGORIES.items():
        lo, hi = cat_info["rango"]
        if lo <= value <= hi:
            return cat_info["color_hex"]
    if value > 300:
        return "#7E0023"
    return "#00E400"


def _get_station_stats(gdf: gpd.GeoDataFrame, variable: str, station_col: str) -> pd.DataFrame:
    """
    Calcula estadísticas por estación para usar en popups del mapa.
    """
    cols = [station_col, variable]
    extra_cols = ["lat", "lon", "geometry"]
    available = [c for c in extra_cols if c in gdf.columns]
    cols += available

    stats = (gdf[cols]
             .groupby(station_col)[variable]
             .agg(["mean", "max", "min", "count"])
             .round(2)
             .reset_index())
    stats.columns = [station_col, "promedio", "maximo", "minimo", "n_obs"]

    if "lat" in gdf.columns and "lon" in gdf.columns:
        coords = gdf.groupby(station_col)[["lat", "lon"]].first().reset_index()
        stats = stats.merge(coords, on=station_col)

    return stats


# ─────────────────────────────────────────────
# 1. MAPA FOLIUM INTERACTIVO
# ─────────────────────────────────────────────

def interactive_map_folium(
    gdf: gpd.GeoDataFrame,
    variable: str,
    output_path: str | Path | None = None,
    station_col: str | None = None,
    center: list[float] | None = None,
    zoom_start: int = 7,
    tiles: str = "osm",
) -> Path | None:
    """
    Genera un mapa folium interactivo con CircleMarkers coloreados por AQI,
    popup con info de estación, leyenda AQI y control de capas.

    Parámetros
    ----------
    gdf : gpd.GeoDataFrame
        Datos de estaciones con columna `variable`.
    variable : str
        Contaminante a visualizar.
    output_path : str | Path | None
        Ruta del HTML de salida. Default: outputs/maps/mapa_{variable}.html
    station_col : str | None
        Columna de ID de estación.
    center : list[float] | None
        [lat, lon] centro del mapa. Default: centroide de los datos.
    zoom_start : int
        Nivel de zoom inicial.
    tiles : str
        Estilo de mapa base ('osm', 'cartolight', 'cartodark').

    Retorna
    -------
    Path al archivo HTML generado.
    """
    if not HAS_FOLIUM:
        logger.error("folium no disponible. Instalar: pip install folium")
        return None

    if output_path is None:
        var_safe = variable.replace(".", "").replace(" ", "_")
        output_path = MAPS_DIR / f"mapa_{var_safe}.html"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    gdf_clean = gdf[gdf[variable].notna()].copy().to_crs(DEFAULT_CRS)
    if len(gdf_clean) == 0:
        logger.warning("Sin datos válidos para el mapa")
        return None

    # Extraer coordenadas
    if "lat" not in gdf_clean.columns:
        gdf_clean["lon"] = gdf_clean.geometry.x
        gdf_clean["lat"] = gdf_clean.geometry.y

    if station_col is None:
        station_col = next((c for c in ["station_id", "location_id", "station_name"]
                            if c in gdf_clean.columns), None)

    # Calcular estadísticas por estación
    if station_col:
        stats_df = _get_station_stats(gdf_clean, variable, station_col)
        gdf_plot = stats_df.merge(
            gdf_clean[[station_col, "geometry"]].drop_duplicates(station_col),
            on=station_col, how="left"
        )
        if "lat" not in gdf_plot.columns:
            gdf_plot["lon"] = gdf_clean.groupby(station_col)["lon"].first().values[:len(gdf_plot)]
            gdf_plot["lat"] = gdf_clean.groupby(station_col)["lat"].first().values[:len(gdf_plot)]
        val_col = "promedio"
    else:
        gdf_plot = gdf_clean.copy()
        gdf_plot["promedio"] = gdf_plot[variable]
        val_col = variable

    # Centro del mapa
    if center is None:
        center = [gdf_plot["lat"].mean(), gdf_plot["lon"].mean()]

    # Crear mapa
    tile_url = FOLIUM_TILES.get(tiles, FOLIUM_TILES["osm"])
    m = folium.Map(location=center, zoom_start=zoom_start, tiles=tile_url)

    # Capa de estaciones
    layer_estaciones = folium.FeatureGroup(name="Estaciones de monitoreo")

    vmin = gdf_plot[val_col].quantile(0.05)
    vmax = gdf_plot[val_col].quantile(0.95)

    for _, row in gdf_plot.iterrows():
        val = row[val_col]
        color = _aqi_to_color(val, vmin, vmax)

        # Popup con información detallada
        popup_html = f"""
        <div style='font-family: Arial; font-size: 12px; min-width: 180px;'>
            <b>{row.get(station_col, 'Estación')}</b><br>
            <hr style='margin: 3px 0;'>
            <b>{variable}:</b> {val:.1f} µg/m³<br>
        """
        if "maximo" in row:
            popup_html += f"Máximo: {row['maximo']:.1f} | Mínimo: {row['minimo']:.1f}<br>"
        if "n_obs" in row:
            popup_html += f"Observaciones: {row['n_obs']}<br>"
        popup_html += f"Coordenadas: {row['lat']:.4f}, {row['lon']:.4f}"
        popup_html += "</div>"

        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=10,
            color="white",
            weight=1.5,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=f"{row.get(station_col, '')}: {val:.1f} µg/m³",
        ).add_to(layer_estaciones)

    layer_estaciones.add_to(m)

    # Leyenda AQI
    legend_html = """
    <div style='position: fixed; bottom: 30px; right: 30px; z-index: 1000;
                background-color: white; padding: 10px; border: 2px solid grey;
                border-radius: 5px; font-family: Arial; font-size: 11px;'>
        <b>Escala AQI</b><br>
    """
    for cat_name, cat_info in AQI_CATEGORIES.items():
        lo, hi = cat_info["rango"]
        nombre = cat_name.replace("_", " ")
        legend_html += (
            f'<i style="background:{cat_info["color_hex"]}; width:12px; height:12px; '
            f'display:inline-block; border:1px solid #ccc;"></i> '
            f'{nombre} ({lo}-{hi})<br>'
        )
    legend_html += "</div>"

    m.get_root().html.add_child(folium.Element(legend_html))

    # Minimap
    try:
        folium_plugins.MiniMap(toggle_display=True).add_to(m)
    except Exception:
        pass

    # Control de capas y escala
    folium.LayerControl().add_to(m)

    try:
        folium.plugins.MeasureControl().add_to(m)
    except Exception:
        pass

    m.save(str(output_path))
    logger.info(f"Mapa folium guardado: {output_path}")
    return output_path


# ─────────────────────────────────────────────
# 2. MAPA COROPLÉTICO
# ─────────────────────────────────────────────

def choropleth_map(
    gdf_admin: gpd.GeoDataFrame,
    variable: str,
    output_path: str | Path | None = None,
    admin_name_col: str = "name",
    title: str | None = None,
) -> Path | None:
    """
    Genera un mapa coroplético de municipios/departamentos.

    Parámetros
    ----------
    gdf_admin : gpd.GeoDataFrame
        GeoDataFrame con polígonos administrativos y columna `variable`.
    variable : str
    output_path : str | Path | None
    admin_name_col : str
        Columna con nombres de unidades administrativas.
    title : str | None

    Retorna
    -------
    Path al HTML generado.
    """
    if not HAS_FOLIUM:
        logger.error("folium no disponible")
        return None

    if output_path is None:
        var_safe = variable.replace(".", "").replace(" ", "_")
        output_path = MAPS_DIR / f"coropletico_{var_safe}.html"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    gdf_admin = gdf_admin.to_crs(DEFAULT_CRS)
    centroid = gdf_admin.geometry.centroid
    center = [centroid.y.mean(), centroid.x.mean()]

    m = folium.Map(location=center, zoom_start=7, tiles="CartoDB positron")

    folium.Choropleth(
        geo_data=gdf_admin.to_json(),
        data=gdf_admin,
        columns=[admin_name_col, variable] if admin_name_col in gdf_admin.columns else [gdf_admin.index.name or "index", variable],
        key_on=f"feature.properties.{admin_name_col}" if admin_name_col in gdf_admin.columns else "feature.id",
        fill_color="YlOrRd",
        fill_opacity=0.7,
        line_opacity=0.5,
        legend_name=f"{variable} (µg/m³)",
        nan_fill_color="lightgray",
    ).add_to(m)

    # Tooltips
    folium.GeoJson(
        gdf_admin.to_json(),
        style_function=lambda x: {"fillOpacity": 0, "weight": 0.5},
        tooltip=folium.GeoJsonTooltip(
            fields=[admin_name_col, variable] if all(c in gdf_admin.columns for c in [admin_name_col, variable]) else [variable],
            aliases=["Nombre:", f"{variable} (µg/m³):"],
        ),
    ).add_to(m)

    if title:
        title_html = f"""
        <div style='position: fixed; top: 10px; left: 50%; transform: translateX(-50%);
                    z-index: 1000; background: white; padding: 8px 15px;
                    border-radius: 5px; font-family: Arial; font-weight: bold; font-size: 14px;'>
            {title}
        </div>"""
        m.get_root().html.add_child(folium.Element(title_html))

    m.save(str(output_path))
    logger.info(f"Mapa coroplético guardado: {output_path}")
    return output_path


# ─────────────────────────────────────────────
# 3. MAPA ANIMADO TEMPORAL
# ─────────────────────────────────────────────

def animation_map(
    gdf_timeseries: gpd.GeoDataFrame,
    variable: str,
    output_path: str | Path | None = None,
    time_col: str = "datetime",
    station_col: str | None = None,
    freq_resample: str = "D",
) -> Path | None:
    """
    Genera un mapa animado por fecha usando Plotly Express con slider temporal.

    Parámetros
    ----------
    gdf_timeseries : gpd.GeoDataFrame
        Serie temporal de estaciones.
    variable : str
    output_path : str | Path | None
        Ruta del HTML. Default: outputs/maps/animacion_{variable}.html
    time_col : str
    station_col : str | None
    freq_resample : str
        Frecuencia de agrupación temporal.

    Retorna
    -------
    Path al HTML de Plotly.
    """
    if not HAS_PLOTLY:
        logger.error("plotly no disponible. Instalar: pip install plotly")
        return None

    if output_path is None:
        var_safe = variable.replace(".", "").replace(" ", "_")
        output_path = MAPS_DIR / f"animacion_{var_safe}.html"
    output_path = Path(output_path)

    gdf = gdf_timeseries.copy().to_crs(DEFAULT_CRS)
    if "lat" not in gdf.columns:
        gdf["lon"] = gdf.geometry.x
        gdf["lat"] = gdf.geometry.y

    if station_col is None:
        station_col = next((c for c in ["station_id", "location_id"] if c in gdf.columns), None)

    gdf[time_col] = pd.to_datetime(gdf[time_col])

    # Resamplear a frecuencia deseada
    if station_col:
        df_resampled = (gdf.groupby([station_col, pd.Grouper(key=time_col, freq=freq_resample)])
                        [variable].mean()
                        .reset_index())
        coords = gdf[[station_col, "lat", "lon"]].drop_duplicates(station_col)
        df_resampled = df_resampled.merge(coords, on=station_col)
    else:
        df_resampled = gdf.copy()

    df_resampled[time_col] = df_resampled[time_col].dt.strftime("%Y-%m-%d")
    df_resampled = df_resampled.dropna(subset=["lat", "lon", variable])

    # Escala de color AQI
    color_scale = [
        [0.0,   "#00E400"],
        [0.1,   "#FFFF00"],
        [0.3,   "#FF7E00"],
        [0.5,   "#FF0000"],
        [0.7,   "#8F3F97"],
        [1.0,   "#7E0023"],
    ]

    vmax = df_resampled[variable].quantile(0.95)

    fig = px.scatter_mapbox(
        df_resampled,
        lat="lat",
        lon="lon",
        color=variable,
        size=variable,
        animation_frame=time_col,
        hover_name=station_col if station_col else None,
        hover_data={variable: ":.1f"},
        color_continuous_scale=color_scale,
        size_max=25,
        range_color=[0, vmax],
        mapbox_style="open-street-map",
        title=f"Evolución temporal de {variable}",
        labels={variable: f"{variable} (µg/m³)"},
    )

    fig.update_layout(
        height=600,
        coloraxis_colorbar=dict(
            title=f"{variable}<br>(µg/m³)",
        ),
    )

    fig.write_html(str(output_path))
    logger.info(f"Mapa animado guardado: {output_path}")
    return output_path


# ─────────────────────────────────────────────
# 4. OVERLAY DE DATOS SATELITALES
# ─────────────────────────────────────────────

def satellite_overlay(
    gdf_stations: gpd.GeoDataFrame,
    ds_satellite,
    variable: str,
    date: str,
    output_path: str | Path | None = None,
    station_col: str | None = None,
    station_var: str | None = None,
    dpi: int = 150,
) -> Path | None:
    """
    Overlay de datos satelitales (raster) sobre mapa de estaciones.
    Visualiza datos satelitales como fondo coloreado + estaciones como puntos.

    Parámetros
    ----------
    gdf_stations : gpd.GeoDataFrame
        Estaciones de monitoreo.
    ds_satellite : xr.Dataset
        Dataset con datos satelitales (requiere xarray).
    variable : str
        Variable del dataset satelital.
    date : str
        Fecha a visualizar 'YYYY-MM-DD'.
    output_path : str | Path | None
    station_col : str | None
    station_var : str | None
        Variable de estaciones a superponer (si es diferente de `variable`).
    dpi : int

    Retorna
    -------
    Path al PNG generado.
    """
    if not HAS_MATPLOTLIB:
        return None

    try:
        import xarray as xr
    except ImportError:
        logger.error("xarray requerido para overlay satelital")
        return None

    if output_path is None:
        var_safe = variable.replace(".", "").replace(" ", "_")
        output_path = MAPS_DIR / f"satellite_overlay_{var_safe}_{date}.png"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plotear raster satelital
    try:
        lat_dim = "lat" if "lat" in ds_satellite.coords else "latitude"
        lon_dim = "lon" if "lon" in ds_satellite.coords else "longitude"
        da = ds_satellite[variable]

        # Seleccionar fecha si hay dimensión temporal
        if "time" in da.dims:
            date_dt = pd.to_datetime(date)
            da = da.sel(time=date_dt, method="nearest")

        # Plot raster
        da.plot(ax=ax, cmap="YlOrRd", alpha=0.7,
                cbar_kwargs={"label": f"{variable} (satelital)"})
    except Exception as e:
        logger.warning(f"No se pudo plotear raster satelital: {e}")

    # Superponer estaciones
    gdf_plot = gdf_stations.to_crs(DEFAULT_CRS).copy()
    if "lat" not in gdf_plot.columns:
        gdf_plot["lon"] = gdf_plot.geometry.x
        gdf_plot["lat"] = gdf_plot.geometry.y

    sv = station_var or variable
    if sv in gdf_plot.columns:
        sc = ax.scatter(
            gdf_plot["lon"], gdf_plot["lat"],
            c=gdf_plot[sv], cmap="RdPu",
            s=80, edgecolors="black", linewidths=0.8,
            zorder=5, label=f"Estaciones ({sv})",
        )
        plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.02, label=f"{sv} estación (µg/m³)")
    else:
        ax.scatter(gdf_plot["lon"], gdf_plot["lat"],
                   c="blue", s=60, zorder=5, label="Estaciones")

    ax.set_title(f"Overlay Satelital + Estaciones — {variable} ({date})", fontsize=12)
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    logger.info(f"Overlay satelital guardado: {output_path}")
    return output_path


# ─────────────────────────────────────────────
# EJEMPLO DE USO
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from ingestion.load_stations import generate_synthetic_data

    df = generate_synthetic_data(n_stations=8, days=30)
    df["datetime"] = pd.to_datetime(df["fecha"], format="%d/%m/%Y %H:%M")

    from shapely.geometry import Point
    geometry = [Point(row["lon"], row["lat"]) for _, row in df.iterrows()]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    # Promediar por estación para el mapa estático
    promedios = (gdf.groupby("station_id")
                 .agg({"PM2.5": "mean", "lat": "first", "lon": "first"})
                 .reset_index())
    geometry_prom = [Point(row["lon"], row["lat"]) for _, row in promedios.iterrows()]
    gdf_prom = gpd.GeoDataFrame(promedios, geometry=geometry_prom, crs="EPSG:4326")

    print("=== Generando mapa folium ===")
    path = interactive_map_folium(gdf_prom, "PM2.5", station_col="station_id")
    print(f"Guardado: {path}")

    print("\n=== Generando mapa animado ===")
    path_anim = animation_map(gdf, "PM2.5", station_col="station_id")
    print(f"Guardado: {path_anim}")
