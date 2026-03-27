"""
dashboard.py — Dashboard interactivo de calidad del aire con Streamlit
Incluye: mapa folium, series de tiempo, estadísticas y correlaciones.

Para ejecutar:
    streamlit run 04_visualization/dashboard.py

Autor: Rodrigo, Ingeniero Ambiental, Universidad Anáhuac Cancún
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# ──────────────────────────────────────────────────────────
# Importaciones con manejo de dependencias opcionales
# ──────────────────────────────────────────────────────────
try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False
    print("Streamlit no disponible. Instalar: pip install streamlit")

try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    import folium
    from streamlit_folium import st_folium
    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False

import sys
import importlib.util

# Agregar raíz del proyecto al path (soporta carpetas con prefijo numérico)
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

# Registrar alias de paquetes sin prefijo numérico en sys.modules
for _num_name, _alias in [
    ("01_ingestion",  "ingestion"),
    ("02_processing", "processing"),
    ("03_analysis",   "analysis"),
]:
    _pkg_path = _ROOT / _num_name
    _spec = importlib.util.spec_from_file_location(
        _alias,
        _pkg_path / "__init__.py",
        submodule_search_locations=[str(_pkg_path)],
    )
    _pkg = importlib.util.module_from_spec(_spec)
    sys.modules[_alias] = _pkg
    _spec.loader.exec_module(_pkg)

from config import POLLUTANTS, REGIONS, AQI_CATEGORIES, WHO_GUIDELINES_2021, PROCESSED_DIR

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# CARGA Y CACHÉ DE DATOS
# ─────────────────────────────────────────────

def _load_demo_data() -> gpd.GeoDataFrame:
    """
    Carga datos de ejemplo: primero busca CSV en data/processed/,
    si no existe genera datos sintéticos.
    """
    # Buscar archivo procesado
    csv_files = list(PROCESSED_DIR.glob("*.csv"))
    if csv_files:
        df = pd.read_csv(csv_files[0])
        df["datetime"] = pd.to_datetime(df.get("datetime", df.get("fecha", None)))
    else:
        # Generar datos sintéticos
        from ingestion.load_stations import generate_synthetic_data
        df = generate_synthetic_data(n_stations=8, days=90)
        df["datetime"] = pd.to_datetime(df["fecha"], format="%d/%m/%Y %H:%M")

    # Construir GeoDataFrame
    if "lat" in df.columns and "lon" in df.columns:
        geometry = [
            Point(row["lon"], row["lat"]) if pd.notna(row.get("lon")) and pd.notna(row.get("lat")) else None
            for _, row in df.iterrows()
        ]
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    else:
        gdf = gpd.GeoDataFrame(df)

    return gdf


@st.cache_data(ttl=3600, show_spinner="Cargando datos...")
def load_data() -> gpd.GeoDataFrame:
    """Carga datos con caché de Streamlit (1 hora de TTL)."""
    return _load_demo_data()


# ─────────────────────────────────────────────
# COMPONENTES DE LA INTERFAZ
# ─────────────────────────────────────────────

def render_sidebar(gdf: gpd.GeoDataFrame) -> dict:
    """
    Renderiza el sidebar con filtros y retorna un dict con las selecciones.
    """
    st.sidebar.title("Filtros")
    st.sidebar.markdown("---")

    # Selector de región
    regiones = list(REGIONS.keys()) + ["Todas las regiones"]
    region = st.sidebar.selectbox(
        "Región",
        regiones,
        index=0,
        help="Selecciona la región geográfica de análisis",
    )

    # Selector de contaminante
    contaminantes_disponibles = [p for p in POLLUTANTS if p in gdf.columns]
    if not contaminantes_disponibles:
        contaminantes_disponibles = POLLUTANTS[:3]
    contaminante = st.sidebar.selectbox(
        "Contaminante",
        contaminantes_disponibles,
        index=0,
    )

    # Rango de fechas
    if "datetime" in gdf.columns:
        fecha_min = gdf["datetime"].min().date() if hasattr(gdf["datetime"].min(), "date") else None
        fecha_max = gdf["datetime"].max().date() if hasattr(gdf["datetime"].max(), "date") else None
    else:
        fecha_min = fecha_max = None

    if fecha_min and fecha_max:
        fechas = st.sidebar.date_input(
            "Rango de fechas",
            value=(fecha_min, fecha_max),
            min_value=fecha_min,
            max_value=fecha_max,
        )
        fecha_inicio = pd.Timestamp(fechas[0]) if len(fechas) > 0 else fecha_min
        fecha_fin    = pd.Timestamp(fechas[1]) if len(fechas) > 1 else fecha_max
    else:
        fecha_inicio = fecha_fin = None

    # Selector de estaciones
    station_col = next((c for c in ["station_id", "location_id", "station_name"] if c in gdf.columns), None)
    if station_col:
        estaciones_disp = sorted(gdf[station_col].dropna().unique().tolist())
        estaciones_sel = st.sidebar.multiselect(
            "Estaciones",
            estaciones_disp,
            default=estaciones_disp[:min(5, len(estaciones_disp))],
        )
    else:
        estaciones_sel = []
        station_col = None

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        **Air Quality LATAM**
        Universidad Anáhuac Cancún
        Ingeniería Ambiental
        """
    )

    return {
        "region":         region,
        "contaminante":   contaminante,
        "fecha_inicio":   fecha_inicio,
        "fecha_fin":      fecha_fin,
        "estaciones":     estaciones_sel,
        "station_col":    station_col,
    }


def apply_filters(gdf: gpd.GeoDataFrame, filtros: dict) -> gpd.GeoDataFrame:
    """Aplica los filtros del sidebar al GeoDataFrame."""
    df = gdf.copy()

    # Filtro temporal
    if filtros["fecha_inicio"] and filtros["fecha_fin"] and "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df[
            (df["datetime"] >= filtros["fecha_inicio"]) &
            (df["datetime"] <= filtros["fecha_fin"])
        ]

    # Filtro de estaciones
    if filtros["estaciones"] and filtros["station_col"]:
        df = df[df[filtros["station_col"]].isin(filtros["estaciones"])]

    # Filtro de región (bbox)
    if filtros["region"] != "Todas las regiones" and filtros["region"] in REGIONS:
        bbox = REGIONS[filtros["region"]]["bbox"]
        if "lon" in df.columns and "lat" in df.columns:
            df = df[
                (df["lon"] >= bbox[0]) & (df["lon"] <= bbox[2]) &
                (df["lat"] >= bbox[1]) & (df["lat"] <= bbox[3])
            ]

    return df


# ─────────────────────────────────────────────
# TABS DE CONTENIDO
# ─────────────────────────────────────────────

def tab_mapa(df: pd.DataFrame, filtros: dict):
    """Tab 1: Mapa interactivo y tabla de estaciones."""
    st.subheader("Mapa de calidad del aire")
    variable = filtros["contaminante"]
    station_col = filtros["station_col"]

    # Calcular promedio por estación para el mapa
    if station_col and station_col in df.columns and "lat" in df.columns:
        df_mapa = (df.groupby(station_col)
                   .agg({variable: "mean", "lat": "first", "lon": "first"})
                   .reset_index()
                   .dropna(subset=["lat", "lon", variable]))
    else:
        df_mapa = df.dropna(subset=["lat", "lon", variable] if "lat" in df.columns else [variable])

    if HAS_FOLIUM and HAS_PLOTLY and len(df_mapa) > 0:
        # Mapa folium
        center_lat = df_mapa["lat"].mean()
        center_lon = df_mapa["lon"].mean()
        m = folium.Map(location=[center_lat, center_lon], zoom_start=7,
                       tiles="CartoDB positron")

        from processing.aqi_index import calculate_aqi
        for _, row in df_mapa.iterrows():
            val = row[variable]
            aqi_result = calculate_aqi(val, variable)
            color = aqi_result["color_hex"]

            popup_html = f"""
            <b>{row.get(station_col, 'Estación')}</b><br>
            {variable}: <b>{val:.1f} µg/m³</b><br>
            AQI: {aqi_result['aqi_value']} — {aqi_result['category']}<br>
            {aqi_result['health_message_es'][:80]}...
            """
            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=12,
                color="white",
                weight=2,
                fill=True,
                fill_color=color,
                fill_opacity=0.9,
                popup=folium.Popup(popup_html, max_width=250),
                tooltip=f"{row.get(station_col, '')}: {val:.1f} µg/m³",
            ).add_to(m)

        st_folium(m, width=700, height=450)

    elif HAS_PLOTLY and len(df_mapa) > 0:
        # Fallback a mapa Plotly si folium no está disponible
        fig = px.scatter_mapbox(
            df_mapa, lat="lat", lon="lon",
            color=variable, size=variable,
            size_max=20, color_continuous_scale="YlOrRd",
            mapbox_style="open-street-map",
            hover_name=station_col if station_col else None,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Instalar folium y/o plotly para visualizar el mapa")

    # Tabla de estaciones
    st.subheader("Resumen por estación")
    if station_col and station_col in df.columns:
        tabla = (df.groupby(station_col)[variable]
                 .agg(["mean", "max", "min", "count"])
                 .round(2)
                 .rename(columns={"mean": "Promedio", "max": "Máximo",
                                  "min": "Mínimo", "count": "N obs"})
                 .sort_values("Promedio", ascending=False))
        st.dataframe(tabla, use_container_width=True)


def tab_series_tiempo(df: pd.DataFrame, filtros: dict):
    """Tab 2: Series de tiempo interactivas."""
    st.subheader("Series de tiempo")
    variable = filtros["contaminante"]
    station_col = filtros["station_col"]

    if not HAS_PLOTLY:
        st.warning("Instalar plotly: pip install plotly")
        return

    if "datetime" not in df.columns:
        st.warning("Sin columna de fecha/hora en los datos")
        return

    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])

    # Frecuencia de agregación
    freq = st.selectbox("Agregación temporal", ["Horario", "Diario", "Semanal"],
                        index=1)
    freq_map = {"Horario": "h", "Diario": "D", "Semanal": "W"}

    if station_col and station_col in df.columns:
        df_resampled = (df.groupby([station_col, pd.Grouper(key="datetime", freq=freq_map[freq])])
                        [variable].mean()
                        .reset_index())
        fig = px.line(
            df_resampled, x="datetime", y=variable, color=station_col,
            title=f"Serie temporal de {variable} ({freq})",
            labels={"datetime": "Fecha", variable: f"{variable} (µg/m³)"},
        )
    else:
        df_resampled = df.resample(freq_map[freq], on="datetime")[variable].mean().reset_index()
        fig = px.line(df_resampled, x="datetime", y=variable,
                      title=f"Serie temporal de {variable} ({freq})")

    # Línea umbral OMS
    guidelines = WHO_GUIDELINES_2021.get(variable.replace(".", ""), {})
    umbral = guidelines.get("24h") or guidelines.get("8h")
    if umbral:
        fig.add_hline(y=umbral, line_dash="dash", line_color="orange",
                      annotation_text=f"Umbral OMS: {umbral} µg/m³")

    fig.update_layout(height=400, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)


def tab_estadisticas(df: pd.DataFrame, filtros: dict):
    """Tab 3: Estadísticas descriptivas y boxplots."""
    st.subheader("Estadísticas descriptivas")
    variable = filtros["contaminante"]
    station_col = filtros["station_col"]

    if not HAS_PLOTLY:
        st.warning("Instalar plotly: pip install plotly")
        return

    col1, col2 = st.columns(2)

    with col1:
        # Métricas clave
        serie = df[variable].dropna()
        if len(serie) > 0:
            st.metric("Promedio", f"{serie.mean():.1f} µg/m³")
            st.metric("Máximo",   f"{serie.max():.1f} µg/m³")
            guidelines = WHO_GUIDELINES_2021.get(variable.replace(".", ""), {})
            umbral = guidelines.get("24h") or guidelines.get("8h")
            if umbral:
                pct_exc = (serie > umbral).mean() * 100
                st.metric(f"% sobre umbral OMS ({umbral} µg/m³)", f"{pct_exc:.1f}%")

    with col2:
        # Histograma
        fig_hist = px.histogram(
            df.dropna(subset=[variable]), x=variable,
            title=f"Distribución de {variable}",
            labels={variable: f"{variable} (µg/m³)"},
            nbins=40, color_discrete_sequence=["#1f77b4"],
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # Tabla estadística
    if station_col and station_col in df.columns:
        from analysis.descriptive import full_descriptive
        try:
            stats_df = full_descriptive(df, variable)
            st.dataframe(stats_df, use_container_width=True)
        except Exception:
            st.dataframe(df.groupby(station_col)[variable].describe().round(2),
                         use_container_width=True)

    # Boxplot comparativo
    if station_col and station_col in df.columns and HAS_PLOTLY:
        fig_box = px.box(
            df.dropna(subset=[variable]),
            x=station_col, y=variable,
            title=f"Distribución de {variable} por estación",
            labels={variable: f"{variable} (µg/m³)"},
        )
        if umbral:
            fig_box.add_hline(y=umbral, line_dash="dash", line_color="orange",
                              annotation_text=f"Umbral OMS: {umbral}")
        st.plotly_chart(fig_box, use_container_width=True)


def tab_correlaciones(df: pd.DataFrame, filtros: dict):
    """Tab 4: Heatmap de correlaciones interactivo."""
    st.subheader("Correlaciones entre contaminantes")

    if not HAS_PLOTLY:
        st.warning("Instalar plotly: pip install plotly")
        return

    # Seleccionar variables
    cols_numericas = [c for c in POLLUTANTS if c in df.columns]
    if len(cols_numericas) < 2:
        st.warning("Se necesitan al menos 2 variables de contaminantes para correlaciones")
        return

    variables_sel = st.multiselect(
        "Variables a correlacionar",
        cols_numericas,
        default=cols_numericas[:min(5, len(cols_numericas))],
    )

    if len(variables_sel) < 2:
        st.info("Selecciona al menos 2 variables")
        return

    metodo = st.radio("Método", ["Pearson", "Spearman"], horizontal=True)

    df_sub = df[variables_sel].dropna()
    corr = df_sub.corr(method=metodo.lower())

    fig = px.imshow(
        corr,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        text_auto=".2f",
        title=f"Correlación {metodo} entre contaminantes (n={len(df_sub)})",
        aspect="auto",
    )
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)

    # Scatter matrix interactivo
    if st.checkbox("Mostrar scatter matrix"):
        fig_scatter = px.scatter_matrix(
            df_sub.sample(min(500, len(df_sub)), random_state=42),
            dimensions=variables_sel,
            title="Scatter matrix de contaminantes",
        )
        fig_scatter.update_traces(marker_size=2, opacity=0.5)
        st.plotly_chart(fig_scatter, use_container_width=True)


# ─────────────────────────────────────────────
# MAIN — DASHBOARD PRINCIPAL
# ─────────────────────────────────────────────

def main():
    """Función principal del dashboard."""
    if not HAS_STREAMLIT:
        print("ERROR: Streamlit no está instalado.")
        print("Instalar: pip install streamlit")
        print("Ejecutar: streamlit run 04_visualization/dashboard.py")
        return

    # Configuración de página
    st.set_page_config(
        page_title="Air Quality LATAM",
        page_icon="🌿",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Encabezado
    st.title("🌿 Air Quality LATAM")
    st.markdown(
        "**Dashboard de calidad del aire — Guatemala y América Latina**  "
        "Universidad Anáhuac Cancún | Ingeniería Ambiental"
    )
    st.markdown("---")

    # Cargar datos
    with st.spinner("Cargando datos..."):
        gdf = load_data()

    if gdf is None or len(gdf) == 0:
        st.error("No se pudieron cargar los datos. Verificar las rutas de datos.")
        return

    # Sidebar con filtros
    filtros = render_sidebar(gdf)

    # Aplicar filtros
    df_filtrado = apply_filters(gdf, filtros)

    # Métricas de resumen en el encabezado
    variable = filtros["contaminante"]
    n_estaciones = df_filtrado[filtros["station_col"]].nunique() if filtros["station_col"] else "N/A"
    n_registros  = len(df_filtrado)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Registros filtrados", f"{n_registros:,}")
    with col2:
        st.metric("Estaciones", n_estaciones)
    with col3:
        if variable in df_filtrado.columns:
            val_promedio = df_filtrado[variable].mean()
            st.metric(f"Promedio {variable}", f"{val_promedio:.1f} µg/m³" if pd.notna(val_promedio) else "N/D")
    with col4:
        if "datetime" in df_filtrado.columns:
            rango = f"{pd.to_datetime(df_filtrado['datetime']).min().date()} → {pd.to_datetime(df_filtrado['datetime']).max().date()}"
            st.metric("Período", rango)

    st.markdown("---")

    # Tabs principales
    tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Mapa", "📈 Series de tiempo", "📊 Estadísticas", "🔗 Correlaciones"])

    with tab1:
        tab_mapa(df_filtrado, filtros)

    with tab2:
        tab_series_tiempo(df_filtrado, filtros)

    with tab3:
        tab_estadisticas(df_filtrado, filtros)

    with tab4:
        tab_correlaciones(df_filtrado, filtros)


# ─────────────────────────────────────────────
# PUNTO DE ENTRADA
# ─────────────────────────────────────────────
if __name__ == "__main__":
    main()
