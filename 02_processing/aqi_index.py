"""
aqi_index.py — Cálculo del Índice de Calidad del Aire (AQI)
Implementa estándares EPA (EE.UU.), IMECA (México) y guías OMS 2021.
Incluye NowCast para PM2.5.

Autor: Rodrigo, Ingeniero Ambiental, Universidad Anáhuac Cancún
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import AQI_CATEGORIES

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
)

# ─────────────────────────────────────────────
# BREAKPOINTS EPA AQI (Tabla 2 de 40 CFR Part 58)
# Fuente: https://www.airnow.gov/sites/default/files/2020-05/aqi-technical-assistance-document-sept2018.pdf
# Formato: {pollutant: [(C_low, C_high, I_low, I_high), ...]}
# Concentraciones en µg/m³ (PM) o ppm (gases) según estándar EPA
# Aquí todo estandarizado a µg/m³
# ─────────────────────────────────────────────

EPA_BREAKPOINTS = {
    "PM2.5": [
        # (C_low, C_high, I_low, I_high) — media 24h en µg/m³
        (0.0,   9.0,    0,   50),
        (9.1,  35.4,   51,  100),
        (35.5, 55.4,  101,  150),
        (55.5, 125.4, 151,  200),
        (125.5, 225.4, 201, 300),
        (225.5, 325.4, 301, 400),
        (325.5, 500.4, 401, 500),
    ],
    "PM10": [
        # Media 24h en µg/m³
        (0,    54,    0,   50),
        (55,  154,   51,  100),
        (155, 254,  101,  150),
        (255, 354,  151,  200),
        (355, 424,  201,  300),
        (425, 504,  301,  400),
        (505, 604,  401,  500),
    ],
    "O3": [
        # Media 8h en µg/m³ (convertido de ppm: 1 ppm O3 = 1995.7 µg/m³ @ 25°C)
        (0,    107,   0,   50),
        (108,  140,  51,  100),
        (141,  170, 101,  150),
        (171,  210, 151,  200),
        (211,  400, 201,  300),
        # Nota: >300 AQI para O3 usa promedio 1h (no implementado aquí por simplicidad)
    ],
    "NO2": [
        # Media 1h en µg/m³
        (0,    53,    0,   50),
        (54,  100,   51,  100),
        (101, 360,  101,  150),
        (361, 649,  151,  200),
        (650, 1249, 201,  300),
        (1250, 1649, 301, 400),
        (1650, 2049, 401, 500),
    ],
    "CO": [
        # Media 8h en mg/m³ (nota: EPA usa ppm, aquí en mg/m³: 1 ppm CO = 1.1456 mg/m³)
        (0.0,   4.4,   0,   50),
        (4.5,   9.4,  51,  100),
        (9.5,  12.4, 101,  150),
        (12.5, 15.4, 151,  200),
        (15.5, 30.4, 201,  300),
        (30.5, 40.4, 301,  400),
        (40.5, 50.4, 401,  500),
    ],
    "SO2": [
        # Media 1h en µg/m³
        (0,    91,    0,   50),
        (92,  197,   51,  100),
        (198, 304,  101,  150),
        (305, 604,  151,  200),
        (605, 804,  201,  300),
        (805, 1004, 301,  400),
        (1005, 1200, 401, 500),
    ],
}

# Mensajes de salud en español por categoría AQI
HEALTH_MESSAGES_ES = {
    "Bueno": (
        "La calidad del aire es satisfactoria y la contaminación "
        "representa poco o ningún riesgo."
    ),
    "Moderado": (
        "La calidad del aire es aceptable. Sin embargo, puede haber riesgo moderado "
        "para personas muy sensibles a la contaminación."
    ),
    "Insalubre_sensibles": (
        "Los grupos sensibles pueden experimentar efectos en la salud. "
        "El público en general probablemente no se verá afectado."
    ),
    "Insalubre": (
        "Todo el mundo puede comenzar a experimentar efectos en la salud. "
        "Los grupos sensibles pueden experimentar efectos más graves."
    ),
    "Muy_insalubre": (
        "Alertas sanitarias: todos pueden experimentar efectos en la salud más graves. "
        "Evitar actividades físicas al aire libre."
    ),
    "Peligroso": (
        "Advertencia de emergencia: toda la población está en riesgo. "
        "Permanecer en interiores con ventanas cerradas."
    ),
}

# ─────────────────────────────────────────────
# BREAKPOINTS IMECA (México) para PM2.5
# Fuente: NOM-172-SEMARNAT-2019
# ─────────────────────────────────────────────

IMECA_BREAKPOINTS_PM25 = [
    # (C_low, C_high, I_low, I_high)
    (0,    25,    0,   50),   # Buena
    (25.1, 45,   51,  100),   # Aceptable
    (45.1, 79,  101,  150),   # Mala
    (79.1, 147, 151,  200),   # Muy mala
    (147.1, 250, 201, 300),   # Extremadamente mala
]

# ─────────────────────────────────────────────
# CATEGORÍAS IMECA
# ─────────────────────────────────────────────

IMECA_CATEGORIES = {
    (0, 50):    {"nombre": "Buena",              "color_hex": "#00E400"},
    (51, 100):  {"nombre": "Aceptable",          "color_hex": "#FFFF00"},
    (101, 150): {"nombre": "Mala",               "color_hex": "#FF7E00"},
    (151, 200): {"nombre": "Muy mala",           "color_hex": "#FF0000"},
    (201, 300): {"nombre": "Extremadamente mala","color_hex": "#8F3F97"},
}


# ─────────────────────────────────────────────
# FUNCIONES PRINCIPALES
# ─────────────────────────────────────────────

def _interpolate_aqi(C: float, breakpoints: list[tuple]) -> float | None:
    """
    Calcula el AQI usando interpolación lineal entre breakpoints.
    Formula: I = ((I_high - I_low) / (C_high - C_low)) * (C - C_low) + I_low

    Retorna None si la concentración está fuera del rango.
    """
    for C_low, C_high, I_low, I_high in breakpoints:
        if C_low <= C <= C_high:
            aqi = ((I_high - I_low) / (C_high - C_low)) * (C - C_low) + I_low
            return round(aqi)
    if C > breakpoints[-1][1]:
        return 500  # Hazardous extremo
    return None


def _get_aqi_category(aqi_value: int) -> tuple[str, str, str]:
    """
    Determina la categoría, color y mensaje de salud para un valor AQI.
    Retorna (categoria, color_hex, mensaje_salud).
    """
    for cat_name, cat_info in AQI_CATEGORIES.items():
        lo, hi = cat_info["rango"]
        if lo <= aqi_value <= hi:
            return cat_name, cat_info["color_hex"], HEALTH_MESSAGES_ES.get(cat_name, "")
    if aqi_value > 500:
        return "Peligroso", "#7E0023", HEALTH_MESSAGES_ES["Peligroso"]
    return "Desconocido", "#808080", "Sin información."


def calculate_aqi(
    concentration: float,
    pollutant: str,
    standard: str = "EPA",
) -> dict:
    """
    Calcula el AQI para un contaminante y concentración dados.

    Parámetros
    ----------
    concentration : float
        Concentración del contaminante en µg/m³ (PM) o mg/m³ (CO).
    pollutant : str
        Nombre del contaminante: 'PM2.5', 'PM10', 'O3', 'NO2', 'CO', 'SO2'.
    standard : str
        'EPA': estándar de EE.UU. (default).
        'IMECA': estándar mexicano (solo PM2.5 implementado).
        'OMS': guías OMS 2021 (retorna excedencia en lugar de índice numérico).

    Retorna
    -------
    dict con claves:
        - aqi_value: valor numérico del AQI (int)
        - category: categoría de calidad (str)
        - color_hex: color hexadecimal (#RRGGBB)
        - health_message_es: mensaje de salud en español
        - standard: estándar utilizado
        - concentration: concentración ingresada
        - pollutant: contaminante
    """
    if concentration is None or np.isnan(concentration) or concentration < 0:
        return {
            "aqi_value": None,
            "category": "Sin datos",
            "color_hex": "#CCCCCC",
            "health_message_es": "Datos no disponibles.",
            "standard": standard,
            "concentration": concentration,
            "pollutant": pollutant,
        }

    poll_upper = pollutant.upper().replace(".", "")
    if poll_upper == "PM25":
        poll_upper = "PM2.5"

    aqi_value = None

    if standard == "EPA":
        bp = EPA_BREAKPOINTS.get(poll_upper)
        if bp is None:
            logger.warning(f"Contaminante '{pollutant}' no tiene breakpoints EPA definidos")
            aqi_value = None
        else:
            aqi_value = _interpolate_aqi(concentration, bp)

    elif standard == "IMECA":
        if poll_upper != "PM2.5":
            logger.warning("IMECA solo implementado para PM2.5 en esta versión")
        bp = IMECA_BREAKPOINTS_PM25
        aqi_value = _interpolate_aqi(concentration, bp)

    elif standard == "OMS":
        # Para OMS retornamos un índice basado en múltiplos del umbral
        from config import WHO_GUIDELINES_2021
        guidelines = WHO_GUIDELINES_2021.get(poll_upper, {})
        umbral_24h = guidelines.get("24h") or guidelines.get("8h")
        if umbral_24h and umbral_24h > 0:
            ratio = concentration / umbral_24h
            aqi_value = int(min(ratio * 100, 500))
        else:
            aqi_value = None

    if aqi_value is None:
        return {
            "aqi_value": None,
            "category": "No calculable",
            "color_hex": "#CCCCCC",
            "health_message_es": f"AQI no calculable para {pollutant} con estándar {standard}.",
            "standard": standard,
            "concentration": concentration,
            "pollutant": pollutant,
        }

    category, color_hex, message = _get_aqi_category(aqi_value)
    return {
        "aqi_value": aqi_value,
        "category": category,
        "color_hex": color_hex,
        "health_message_es": message,
        "standard": standard,
        "concentration": concentration,
        "pollutant": pollutant,
    }


def calculate_nowcast_pm25(concentrations_12h: list[float]) -> float | None:
    """
    Calcula el NowCast para PM2.5 según el algoritmo EPA.

    El NowCast usa las últimas 12 horas de datos para calcular
    una concentración ponderada que refleja las condiciones actuales.

    Parámetros
    ----------
    concentrations_12h : list[float]
        Lista de concentraciones horarias, ordenadas de más reciente a más antigua.
        Debe tener entre 3 y 12 valores (las últimas 12 horas).
        None o NaN para horas sin datos.

    Retorna
    -------
    float: NowCast PM2.5 en µg/m³, o None si hay menos de 2 horas válidas.

    Referencia
    ----------
    https://www.airnow.gov/sites/default/files/2020-05/
    aqi-technical-assistance-document-sept2018.pdf (Appendix D)
    """
    # Reemplazar None por NaN
    concs = [c if (c is not None and not np.isnan(c)) else np.nan
             for c in concentrations_12h[:12]]

    # Necesitamos al menos la hora más reciente y una de las primeras 3
    if len(concs) < 3:
        return None

    valid_concs = [c for c in concs if not np.isnan(c)]
    if len(valid_concs) < 2:
        return None

    # Al menos una de las 3 horas más recientes debe ser válida
    recent_3 = [c for c in concs[:3] if not np.isnan(c)]
    if not recent_3:
        return None

    # Calcular factor de ponderación w
    c_min = np.nanmin(concs)
    c_max = np.nanmax(concs)
    if c_max == 0:
        return 0.0

    w_raw = 1 - (c_max - c_min) / c_max
    w = max(w_raw, 0.5)  # mínimo 0.5 según EPA

    # Calcular pesos por hora (hora 1 es la más reciente)
    numerador = 0.0
    denominador = 0.0
    for i, c in enumerate(concs):
        if not np.isnan(c):
            peso = w ** i
            numerador += peso * c
            denominador += peso

    if denominador == 0:
        return None

    nowcast = numerador / denominador
    return round(nowcast, 1)


def aqi_dataframe(
    df: pd.DataFrame,
    pollutant: str = "PM2.5",
    concentration_col: str | None = None,
    standard: str = "EPA",
) -> pd.DataFrame:
    """
    Aplica calculate_aqi a un DataFrame completo.
    Añade columnas: aqi, categoria, color, mensaje_salud.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame con columna de concentración.
    pollutant : str
        Contaminante a calcular.
    concentration_col : str | None
        Nombre de la columna de concentración.
        Si es None, busca columna con nombre del contaminante.
    standard : str
        Estándar a usar ('EPA', 'IMECA', 'OMS').

    Retorna
    -------
    pd.DataFrame con columnas AQI añadidas.
    """
    df = df.copy()

    if concentration_col is None:
        # Buscar columna por nombre del contaminante
        candidates = [c for c in df.columns
                      if pollutant.lower() in c.lower() or c.lower() in pollutant.lower()]
        if not candidates:
            if "value" in df.columns:
                concentration_col = "value"
            else:
                raise ValueError(
                    f"No se encontró columna para '{pollutant}'. "
                    f"Especificar concentration_col. Columnas disponibles: {list(df.columns)}"
                )
        else:
            concentration_col = candidates[0]

    logger.info(
        f"Calculando AQI ({standard}) para '{pollutant}' "
        f"desde columna '{concentration_col}' ({len(df)} filas)"
    )

    resultados = df[concentration_col].apply(
        lambda c: calculate_aqi(c, pollutant, standard)
    )

    df["aqi"]          = resultados.apply(lambda r: r["aqi_value"])
    df["categoria"]    = resultados.apply(lambda r: r["category"])
    df["color"]        = resultados.apply(lambda r: r["color_hex"])
    df["salud_mensaje"] = resultados.apply(lambda r: r["health_message_es"])

    # Estadística de distribución de categorías
    dist = df["categoria"].value_counts()
    logger.info(f"Distribución de categorías AQI:\n{dist.to_string()}")

    return df


# ─────────────────────────────────────────────
# EJEMPLO DE USO
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Ejemplos de cálculo AQI ===\n")

    ejemplos = [
        ("PM2.5", 8.0,   "EPA"),
        ("PM2.5", 25.0,  "EPA"),
        ("PM2.5", 55.0,  "EPA"),
        ("PM2.5", 150.0, "EPA"),
        ("PM10",  80.0,  "EPA"),
        ("O3",    120.0, "EPA"),
        ("NO2",   150.0, "EPA"),
        ("CO",    5.5,   "EPA"),
        ("PM2.5", 40.0,  "IMECA"),
        ("PM2.5", 12.0,  "OMS"),
    ]

    for pollutant, conc, std in ejemplos:
        result = calculate_aqi(conc, pollutant, std)
        print(
            f"{std} | {pollutant:6s} {conc:6.1f} µg/m³ → "
            f"AQI: {str(result['aqi_value']):5s} | "
            f"{result['category']:20s} | {result['color_hex']}"
        )

    print("\n=== NowCast PM2.5 ===")
    ultimas_12h = [35, 40, 38, 45, 50, 55, 48, 42, 38, None, 30, 28]
    nowcast = calculate_nowcast_pm25(ultimas_12h)
    print(f"Concentraciones (hora más reciente primero): {ultimas_12h}")
    print(f"NowCast PM2.5: {nowcast} µg/m³")

    print("\n=== DataFrame con AQI ===")
    df_test = pd.DataFrame({
        "datetime": pd.date_range("2024-01-01", periods=10, freq="h"),
        "PM2.5": [5, 12, 38, 60, 100, 140, 200, 300, 400, None],
    })
    df_aqi = aqi_dataframe(df_test, pollutant="PM2.5")
    print(df_aqi[["datetime", "PM2.5", "aqi", "categoria", "color"]].to_string())
