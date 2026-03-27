"""
config.py — Configuración global del proyecto Air Quality LATAM
Autor: Rodrigo, Ingeniero Ambiental, Universidad Anáhuac Cancún
"""

import os
from pathlib import Path

# ─────────────────────────────────────────────
# RUTAS BASE
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SPATIAL_DIR = DATA_DIR / "spatial"
OUTPUTS_DIR = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
MAPS_DIR = OUTPUTS_DIR / "maps"
REPORTS_DIR = OUTPUTS_DIR / "reports"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"

# Crear directorios si no existen
for _dir in [RAW_DIR / "openaq", RAW_DIR / "satellite", RAW_DIR / "cams",
             PROCESSED_DIR, SPATIAL_DIR, FIGURES_DIR, MAPS_DIR, REPORTS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# CONTAMINANTES
# ─────────────────────────────────────────────
POLLUTANTS = ["PM2.5", "PM10", "O3", "NO2", "CO", "SO2"]

# Unidades estándar por contaminante
POLLUTANT_UNITS = {
    "PM2.5": "µg/m³",
    "PM10":  "µg/m³",
    "O3":    "µg/m³",
    "NO2":   "µg/m³",
    "CO":    "mg/m³",
    "SO2":   "µg/m³",
}

# Límites físicos plausibles (para limpieza de datos)
PHYSICAL_LIMITS = {
    "PM2.5": (0, 500),
    "PM10":  (0, 1000),
    "O3":    (0, 400),
    "NO2":   (0, 500),
    "CO":    (0, 50),
    "SO2":   (0, 500),
}

# Factores de conversión ppb → µg/m³ a 25°C, 1 atm
# µg/m³ = ppb × factor
PPB_TO_UGM3 = {
    "O3":  1.9957,
    "NO2": 1.8816,
    "CO":  1.1456,
    "SO2": 2.6196,
}

# ─────────────────────────────────────────────
# REGIONES DE INTERÉS (bounding boxes: [lon_min, lat_min, lon_max, lat_max])
# ─────────────────────────────────────────────
REGIONS = {
    "Guatemala": {
        "bbox": [-92.2, 13.7, -88.2, 17.8],
        "country_code": "GT",
        "crs": "EPSG:4326",
        "descripcion": "República de Guatemala",
    },
    "Quintana_Roo": {
        "bbox": [-88.0, 18.0, -86.7, 21.6],
        "country_code": "MX",
        "crs": "EPSG:4326",
        "descripcion": "Estado de Quintana Roo, México",
    },
    "LATAM": {
        "bbox": [-120.0, -56.0, -34.0, 32.0],
        "country_code": None,
        "crs": "EPSG:4326",
        "descripcion": "América Latina y el Caribe",
    },
}

# ─────────────────────────────────────────────
# UMBRALES OMS 2021 (µg/m³ o mg/m³ según contaminante)
# Guías de Calidad del Aire OMS 2021
# ─────────────────────────────────────────────
WHO_GUIDELINES_2021 = {
    "PM2.5": {
        "anual":    5.0,   # µg/m³
        "24h":     15.0,   # µg/m³
        "descripcion": "Material Particulado fino (≤2.5 µm)",
    },
    "PM10": {
        "anual":   15.0,
        "24h":     45.0,
        "descripcion": "Material Particulado grueso (≤10 µm)",
    },
    "O3": {
        "8h":      100.0,  # µg/m³
        "descripcion": "Ozono troposférico",
    },
    "NO2": {
        "anual":   10.0,
        "24h":     25.0,
        "descripcion": "Dióxido de nitrógeno",
    },
    "CO": {
        "24h":      4.0,   # mg/m³
        "8h":      10.0,
        "descripcion": "Monóxido de carbono",
    },
    "SO2": {
        "24h":     40.0,   # µg/m³
        "descripcion": "Dióxido de azufre",
    },
}

# ─────────────────────────────────────────────
# PALETAS DE COLOR POR NIVEL DE RIESGO (AQI EPA)
# ─────────────────────────────────────────────
AQI_CATEGORIES = {
    "Bueno":              {"rango": (0, 50),   "color_hex": "#00E400", "color_name": "verde"},
    "Moderado":           {"rango": (51, 100),  "color_hex": "#FFFF00", "color_name": "amarillo"},
    "Insalubre_sensibles":{"rango": (101, 150), "color_hex": "#FF7E00", "color_name": "naranja"},
    "Insalubre":          {"rango": (151, 200), "color_hex": "#FF0000", "color_name": "rojo"},
    "Muy_insalubre":      {"rango": (201, 300), "color_hex": "#8F3F97", "color_name": "morado"},
    "Peligroso":          {"rango": (301, 500), "color_hex": "#7E0023", "color_name": "granate"},
}

# Paleta simplificada para mapas
RISK_COLORMAP = {
    "bueno":      "#00E400",
    "moderado":   "#FFFF00",
    "malo":       "#FF7E00",
    "muy_malo":   "#FF0000",
    "peligroso":  "#8F3F97",
    "critico":    "#7E0023",
}

# ─────────────────────────────────────────────
# API KEYS (se leen de variables de entorno, nunca hardcodeadas)
# ─────────────────────────────────────────────
OPENAQ_API_KEY   = os.getenv("OPENAQ_API_KEY", "")
NASA_EARTHDATA_TOKEN = os.getenv("NASA_EARTHDATA_TOKEN", "")
COPERNICUS_KEY   = os.getenv("COPERNICUS_API_KEY", "")

# ─────────────────────────────────────────────
# SISTEMA DE REFERENCIA COORDENADAS
# ─────────────────────────────────────────────
DEFAULT_CRS = "EPSG:4326"   # WGS84 geográfico
PROJECTED_CRS_GT = "EPSG:32615"  # UTM Zona 15N (Guatemala)
PROJECTED_CRS_MX = "EPSG:32614"  # UTM Zona 14N (Quintana Roo)

# ─────────────────────────────────────────────
# PARÁMETROS DE ANÁLISIS
# ─────────────────────────────────────────────
DEFAULT_RESOLUTION_DEG = 0.1   # resolución de grid IDW en grados
INTERPOLATION_POWER    = 2      # potencia para IDW
MAX_GAP_HOURS          = 6      # máximo gap para interpolación temporal

# ─────────────────────────────────────────────
# FUENTES DE DATOS
# ─────────────────────────────────────────────
OPENAQ_BASE_URL = "https://api.openaq.org/v3"
NASA_EARTHDATA_URL = "https://ladsweb.modaps.eosdis.nasa.gov"
NATURAL_EARTH_URL = "https://naturalearth.s3.amazonaws.com"
