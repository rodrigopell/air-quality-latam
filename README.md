# Air Quality LATAM

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

Repositorio completo de análisis de calidad del aire para **Guatemala** y **América Latina**.
Integra datos de estaciones terrestres, satélites y reanálisis atmosférico para análisis
espaciotemporal, cálculo de AQI, detección de anomalías y generación de reportes PDF.

---

## Descripción y motivación / Description and Motivation

**ES:** Guatemala y muchas regiones de LATAM tienen cobertura limitada de monitoreo de calidad
del aire. Este repositorio combina datos de múltiples fuentes (OpenAQ, NASA MERRA-2, Copernicus
CAMS) para cerrar esa brecha y apoyar la toma de decisiones en salud pública y política ambiental.

**EN:** Guatemala and many LATAM regions have limited air quality monitoring coverage. This
repository bridges that gap by combining data from multiple sources (OpenAQ, NASA MERRA-2,
Copernicus CAMS) to support public health and environmental policy decisions.

---

## Estructura del proyecto / Project Structure

```
air-quality-latam/
├── 01_ingestion/           # Carga y descarga de datos
│   ├── load_stations.py    # CSV de estaciones con detección automática
│   ├── fetch_openaq.py     # API v3 de OpenAQ (GT, MX, LATAM)
│   ├── fetch_satellite.py  # MODIS AOD + MERRA-2 NetCDF
│   └── fetch_cams.py       # Copernicus CAMS reanálisis
├── 02_processing/          # Limpieza y transformación
│   ├── clean.py            # AirQualityCleaner con pipeline completo
│   ├── spatial_merge.py    # IDW, Kriging, joins espaciales
│   └── aqi_index.py        # EPA AQI, IMECA, NowCast PM2.5
├── 03_analysis/            # Análisis estadístico y espacial
│   ├── spatial.py          # Hotspots, patrones estacionales
│   ├── descriptive.py      # Estadísticas completas + gráficos
│   ├── timeseries.py       # STL, ARIMA, anomalías
│   └── correlations.py     # Pearson, Spearman, lag, parcial
├── 04_visualization/       # Visualización y reportes
│   ├── maps.py             # Folium, Plotly, animaciones
│   ├── dashboard.py        # Dashboard Streamlit interactivo
│   └── reports.py          # Reporte PDF con ReportLab
├── data/
│   ├── raw/                # Datos originales sin modificar
│   │   ├── openaq/         # JSON de OpenAQ
│   │   ├── satellite/      # HDF y NC4 de MODIS y MERRA-2
│   │   └── cams/           # NetCDF de Copernicus CAMS
│   ├── processed/          # Datos limpios y procesados
│   └── spatial/            # Shapefiles y límites administrativos
├── outputs/
│   ├── figures/            # PNG de gráficos estadísticos
│   ├── maps/               # HTML y PNG de mapas
│   └── reports/            # PDF de reportes
├── notebooks/
│   └── 00_exploracion.ipynb  # Notebook de exploración inicial
├── config.py               # Configuración global
└── requirements.txt        # Dependencias
```

---

## Instalación / Installation

```bash
# 1. Clonar el repositorio
git clone https://github.com/tu-usuario/air-quality-latam.git
cd air-quality-latam

# 2. Crear entorno virtual (recomendado)
python -m venv .venv
source .venv/bin/activate    # Linux/Mac
.venv\Scripts\activate       # Windows

# 3. Instalar dependencias
pip install -r requirements.txt
```

---

## Quickstart (3 comandos)

```bash
# 1. Explorar datos de ejemplo (genera datos sintéticos de Guatemala)
python 01_ingestion/load_stations.py

# 2. Limpiar y calcular AQI
python 02_processing/clean.py && python 02_processing/aqi_index.py

# 3. Lanzar el dashboard interactivo
streamlit run 04_visualization/dashboard.py
```

El dashboard estará disponible en `http://localhost:8501`

---

## Fuentes de datos / Data Sources

| Fuente | Tipo | Cobertura | Acceso |
|--------|------|-----------|--------|
| [OpenAQ](https://openaq.org/) | Estaciones terrestres | Global | API gratuita |
| [NASA EarthData](https://earthdata.nasa.gov/) | MODIS AOD, MERRA-2 | Global | Registro gratuito |
| [Copernicus CAMS](https://atmosphere.copernicus.eu/) | Reanálisis atmosférico | Global | Registro gratuito |
| [MARN Guatemala](https://www.marn.gob.gt/) | Estaciones nacionales GT | Guatemala | Solicitud directa |
| [SEMARNAT México](https://www.gob.mx/semarnat) | Red SINAICA | México | Descarga directa |

---

## Configuración de API keys

```bash
# Crear archivo .env en la raíz del proyecto
export OPENAQ_API_KEY="tu_clave_aqui"
export NASA_EARTHDATA_TOKEN="tu_token_aqui"

# Para Copernicus CAMS, crear ~/.cdsapirc:
# url: https://ads.atmosphere.copernicus.eu/api/v2
# key: UID:API_KEY
```

---

## Ejemplos de uso

```python
# Descargar datos de Guatemala
from ingestion.fetch_openaq import fetch_openaq
df = fetch_openaq("GT", "pm25", "2024-01-01", "2024-12-31")

# Calcular AQI
from processing.aqi_index import aqi_dataframe
df_aqi = aqi_dataframe(df, pollutant="PM2.5")

# Generar mapa interactivo
from visualization.maps import interactive_map_folium
import geopandas as gpd
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat))
interactive_map_folium(gdf, "PM2.5", "outputs/maps/mapa_gt.html")

# Generar reporte PDF
from visualization.reports import generate_report
generate_report(df, "Guatemala", "2024-01-01", "2024-12-31")
```

---

## Cómo contribuir / Contributing

1. Haz fork del repositorio
2. Crea una rama: `git checkout -b feature/nueva-funcionalidad`
3. Haz commit: `git commit -m 'Añadir nueva funcionalidad'`
4. Push: `git push origin feature/nueva-funcionalidad`
5. Abre un Pull Request

Por favor, sigue el estilo de código existente (docstrings en español, logging detallado,
manejo de errores robusto).

---

## Autor / Author

**Rodrigo** — Ingeniero Ambiental
Universidad Anáhuac Cancún
Cancún, Quintana Roo, México

---

## Licencia / License

MIT License — libre para uso académico y comercial con atribución.

```
Copyright (c) 2024 Rodrigo, Universidad Anáhuac Cancún

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```
