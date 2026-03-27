# Air Quality LATAM

**Framework de análisis de calidad del aire para regiones con escasez de datos en América Latina**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![OpenAQ](https://img.shields.io/badge/Data-OpenAQ-teal.svg)](https://openaq.org)
[![NASA](https://img.shields.io/badge/Data-NASA%20EarthData-orange.svg)](https://earthdata.nasa.gov)

---

## Por qué existe este proyecto

El monitoreo de calidad del aire en América Latina es severamente limitado. Las herramientas existentes están diseñadas para redes densas de sensores y datos estandarizados. Este proyecto fue construido para el escenario opuesto: **regiones donde los datos son escasos, fragmentados o simplemente inexistentes.**

Se combina información de estaciones en superficie con sensores de bajo costo (PurpleAir), reanálisis meteorológico (ERA5) y teledetección satelital (NASA MERRA-2, MODIS AOD, Copernicus CAMS) para producir análisis de calidad del aire donde el monitoreo tradicional es insuficiente.

---

## Regiones de cobertura

| Región | Estado | Datos disponibles |
|--------|--------|-------------------|
| **Península de Yucatán** | **Primaria — datos reales** | 10 sensores PurpleAir/SEMA (PM2.5, 2020–2025) + ERA5 horario (1996–2026) |
| Guatemala | Expansión — datos sintéticos | Pendiente integración con MARN / OpenAQ |
| LATAM | Contexto regional | Copernicus CAMS, NASA MERRA-2, OpenAQ API |

### Estaciones PurpleAir activas — Quintana Roo

| ID | Nombre | Municipio | Período |
|----|--------|-----------|---------|
| BAC | Bacalar | Bacalar | oct-2020 → sep-2025 |
| BCUN | Bomberos Cancún | Benito Juárez | mar-2021 → may-2023 |
| C_A | Chetumal Aeropuerto | Othón P. Blanco | oct-2020 → oct-2025 |
| C_N | Chetumal Norte | Othón P. Blanco | oct-2020 → feb-2024 |
| C_Z | Chetumal Zoo | Othón P. Blanco | oct-2020 → oct-2025 |
| COZ | Cozumel | Cozumel | ene-2022 → abr-2025 |
| PL | Playa del Carmen | Solidaridad | mar-2021 → feb-2023 |
| PM | Puerto Morelos | Benito Juárez | mar-2021 → jun-2024 |
| TUL | Tulum | Tulum | nov-2022 → dic-2022 |
| UAC | UAC Cancún | Benito Juárez | mar-2022 → dic-2023 |

---

## Qué hace

- Ingesta datos de estaciones locales (CSV/Excel), API OpenAQ, NASA EarthData y Copernicus CAMS
- Carga ERA5 local (viento, temperatura, precipitación) para análisis de correlación meteorológica
- Limpia, estandariza y valida datos multi-fuente de calidad del aire
- Calcula EPA AQI, IMECA y estándares OMS 2021
- Realiza interpolación espacial (IDW, kriging) para llenar vacíos de monitoreo
- Genera mapas interactivos, análisis de series de tiempo y estudios de correlación
- Produce reportes PDF automatizados por región y período

---

## Contaminantes cubiertos

PM2.5 · PM10 · O3 · NO2 · CO · SO2 · Polvo sahariano (transporte transoceánico)

---

## Fuentes de datos

| Fuente | Tipo | Cobertura | Estado |
|--------|------|-----------|--------|
| PurpleAir / SEMA Quintana Roo | Sensores ópticos PM2.5 | Quintana Roo | **Datos reales integrados** |
| ERA5 (ECMWF / Copernicus) | Reanálisis meteorológico | Península Yucatán | **Datos reales integrados** |
| OpenAQ API v3 | Agregador multi-fuente | Global | Vía API (requiere key) |
| NASA MERRA-2 | Reanálisis atmosférico | Global | Vía NASA EarthData |
| MODIS AOD | Aerosol óptico satelital | Global | Vía NASA EarthData |
| Copernicus CAMS | Reanálisis calidad del aire | Global | Vía CDS API |
| MARN Guatemala | Estaciones en superficie | Guatemala | Pendiente integración |

---

## Inicio rápido

```bash
git clone https://github.com/rodrigopell/air-quality-latam.git
cd air-quality-latam
pip install -r requirements.txt
```

### Cargar datos reales de PM2.5 (Península de Yucatán)

```python
from 01_ingestion.load_pm25_sema import load_all_stations, station_summary

# Resumen de cobertura por estación
print(station_summary())

# Cargar todas las estaciones como GeoDataFrame
gdf = load_all_stations()

# Estación individual
from 01_ingestion.load_pm25_sema import load_station
df_cancun = load_station("BCUN")
```

### Cargar ERA5 meteorológico

```python
from 01_ingestion.load_era5_local import load_era5, era5_to_dataframe

# Viento horario 2020-2024
ds = load_era5("viento", years=range(2020, 2025))
df = era5_to_dataframe(ds, lat=21.17, lon=-86.83)  # Cancún

# También: "precipitacion", "temperatura"
# Serie larga 1996-2026: use_long_series=True
ds_largo = load_era5("temperatura", years=range(1996, 2026), use_long_series=True)
```

### Dashboard interactivo

```bash
streamlit run 04_visualization/dashboard.py
```

Abrir http://localhost:8501 en el navegador.

---

## Estructura del proyecto

```
air-quality-latam/
├── 01_ingestion/
│   ├── load_pm25_sema.py     # PurpleAir/SEMA Quintana Roo (datos reales)
│   ├── load_era5_local.py    # ERA5 local (datos reales, 1996-2026)
│   ├── load_stations.py      # Loader genérico CSV/Excel
│   ├── fetch_openaq.py       # API OpenAQ v3
│   ├── fetch_cams.py         # Copernicus CAMS (requiere cuenta ADS)
│   └── fetch_satellite.py    # NASA MODIS / MERRA-2
├── 02_processing/            # Limpieza, merge espacial, cálculo AQI
├── 03_analysis/              # Estadísticas, series de tiempo, correlaciones
├── 04_visualization/         # Mapas, dashboard Streamlit, reportes PDF
├── data/
│   └── raw/
│       └── pm25_peninsula_yucatan/   # CSVs por estación + metadata
├── outputs/                  # Figuras, mapas, reportes generados
└── notebooks/                # Análisis exploratorio
```

---

## Contribuciones

Bienvenidas contribuciones, especialmente de investigadores e ingenieros ambientales trabajando en regiones con escasez de datos. Abrir un issue o hacer un PR.

---

## Autor

Rodrigo Pellecer · Ingeniero Ambiental
Universidad Anáhuac Cancún · rodrigo.pellecergodoy@gmail.com

---

## Licencia

MIT
