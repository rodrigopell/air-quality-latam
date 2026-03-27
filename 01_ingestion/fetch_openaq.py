"""
fetch_openaq.py — Descarga de datos de calidad del aire desde la API v3 de OpenAQ
Soporta paginación automática, rate limiting, guardado de JSON raw y
estandarización a DataFrame.

API docs: https://docs.openaq.org/
Registro gratuito en: https://explore.openaq.org/register

Autor: Rodrigo, Ingeniero Ambiental, Universidad Anáhuac Cancún
"""

import json
import logging
import time
from datetime import datetime, date
from pathlib import Path

import pandas as pd
import requests

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import OPENAQ_API_KEY, RAW_DIR, OPENAQ_BASE_URL

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
)

# Directorio de almacenamiento raw
OPENAQ_RAW_DIR = RAW_DIR / "openaq"
OPENAQ_RAW_DIR.mkdir(parents=True, exist_ok=True)

# Pausa entre requests (segundos) para respetar rate limit de OpenAQ
RATE_LIMIT_SLEEP = 0.5

# Parámetros válidos en OpenAQ v3
VALID_PARAMETERS = {
    "pm25": "PM2.5",
    "pm10": "PM10",
    "o3":   "O3",
    "no2":  "NO2",
    "co":   "CO",
    "so2":  "SO2",
    "no":   "NO",
    "bc":   "BC",
}


def _build_headers() -> dict:
    """Construye los headers HTTP. Incluye API key si está disponible."""
    headers = {"Accept": "application/json"}
    if OPENAQ_API_KEY:
        headers["X-API-Key"] = OPENAQ_API_KEY
        logger.debug("Usando API key de OpenAQ")
    else:
        logger.warning(
            "No se encontró OPENAQ_API_KEY. Usando acceso anónimo "
            "(límite reducido de requests). Define la variable de entorno."
        )
    return headers


def fetch_locations(
    country_code: str,
    parameter: str,
    limit: int = 200,
) -> list[dict]:
    """
    Obtiene la lista de estaciones (locations) para un país y parámetro.

    Parámetros
    ----------
    country_code : str
        Código ISO 3166-1 alpha-2 (ej: 'GT', 'MX', 'CO').
    parameter : str
        Parámetro contaminante (ej: 'pm25', 'no2').
    limit : int
        Máximo de resultados por página.

    Retorna
    -------
    list[dict]: Lista de objetos location de OpenAQ.
    """
    url = f"{OPENAQ_BASE_URL}/locations"
    headers = _build_headers()
    locations = []
    page = 1

    while True:
        params = {
            "countries_id": country_code,
            "parameters_name": parameter.lower(),
            "limit": limit,
            "page": page,
        }
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            locations.extend(results)
            logger.info(f"Página {page}: {len(results)} estaciones obtenidas (total: {len(locations)})")

            # Verificar si hay más páginas
            meta = data.get("meta", {})
            found = meta.get("found", 0)
            if len(locations) >= found or len(results) == 0:
                break

            page += 1
            time.sleep(RATE_LIMIT_SLEEP)

        except requests.exceptions.HTTPError as e:
            logger.error(f"Error HTTP {resp.status_code} en /locations: {e}")
            break
        except requests.exceptions.ConnectionError:
            logger.error("Error de conexión. Verifica tu acceso a internet.")
            break
        except Exception as e:
            logger.error(f"Error inesperado en fetch_locations: {e}")
            break

    logger.info(f"Total de estaciones encontradas: {len(locations)}")
    return locations


def fetch_openaq(
    country_code: str,
    parameter: str,
    date_from: str | date | datetime,
    date_to: str | date | datetime,
    limit: int = 1000,
    save_raw: bool = True,
) -> pd.DataFrame:
    """
    Descarga mediciones de OpenAQ API v3 para un país, parámetro y rango de fechas.
    Realiza paginación automática hasta obtener todos los registros disponibles.

    Parámetros
    ----------
    country_code : str
        Código ISO del país (ej: 'GT' para Guatemala, 'MX' para México).
    parameter : str
        Parámetro a descargar: 'pm25', 'pm10', 'o3', 'no2', 'co', 'so2'.
    date_from : str | date | datetime
        Fecha de inicio (YYYY-MM-DD o datetime).
    date_to : str | date | datetime
        Fecha de fin (YYYY-MM-DD o datetime).
    limit : int
        Registros por página (max 1000 en OpenAQ v3).
    save_raw : bool
        Si True, guarda el JSON raw en data/raw/openaq/.

    Retorna
    -------
    pd.DataFrame con columnas estandarizadas:
        location_id, station_name, lat, lon, datetime, parameter, value, unit, country
    """
    # Normalizar fechas a string ISO
    if isinstance(date_from, (date, datetime)):
        date_from = date_from.strftime("%Y-%m-%dT%H:%M:%SZ")
    elif "T" not in str(date_from):
        date_from = f"{date_from}T00:00:00Z"

    if isinstance(date_to, (date, datetime)):
        date_to = date_to.strftime("%Y-%m-%dT%H:%M:%SZ")
    elif "T" not in str(date_to):
        date_to = f"{date_to}T23:59:59Z"

    param_lower = parameter.lower()
    logger.info(
        f"Descargando OpenAQ | País: {country_code} | Parámetro: {param_lower} | "
        f"{date_from} → {date_to}"
    )

    url = f"{OPENAQ_BASE_URL}/measurements"
    headers = _build_headers()
    all_results = []
    page = 1

    while True:
        params = {
            "countries_id": country_code,
            "parameters_name": param_lower,
            "date_from": date_from,
            "date_to": date_to,
            "limit": min(limit, 1000),
            "page": page,
        }

        try:
            resp = requests.get(url, headers=headers, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()

        except requests.exceptions.HTTPError as e:
            status = resp.status_code if resp else "?"
            if status == 429:
                logger.warning("Rate limit alcanzado — esperando 10 segundos...")
                time.sleep(10)
                continue
            logger.error(f"Error HTTP {status}: {e}")
            break
        except requests.exceptions.Timeout:
            logger.error("Timeout en la solicitud. Reintentando en 5 segundos...")
            time.sleep(5)
            continue
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Error de conexión: {e}")
            break
        except Exception as e:
            logger.error(f"Error inesperado: {e}")
            break

        results = data.get("results", [])
        if not results:
            logger.info(f"No hay más resultados en página {page}")
            break

        all_results.extend(results)
        meta = data.get("meta", {})
        found = meta.get("found", 0)
        logger.info(f"Página {page}: +{len(results)} registros (total: {len(all_results)}/{found})")

        if len(all_results) >= found or len(results) < limit:
            break

        page += 1
        time.sleep(RATE_LIMIT_SLEEP)

    if not all_results:
        logger.warning("No se obtuvieron datos. Verifica el país, parámetro y rango de fechas.")
        return pd.DataFrame()

    # Guardar JSON raw
    if save_raw:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        raw_filename = OPENAQ_RAW_DIR / f"openaq_{country_code}_{param_lower}_{ts}.json"
        with open(raw_filename, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        logger.info(f"JSON raw guardado en: {raw_filename}")

    # Estandarizar a DataFrame
    df = _parse_measurements(all_results, country_code)
    logger.info(f"DataFrame final: {len(df)} registros")
    return df


def _parse_measurements(results: list[dict], country_code: str) -> pd.DataFrame:
    """
    Convierte la lista de mediciones de OpenAQ v3 a DataFrame estandarizado.
    """
    rows = []
    for r in results:
        try:
            # Extraer campos según estructura de OpenAQ v3
            loc = r.get("location", {}) or {}
            coords = loc.get("coordinates", {}) or {}

            row = {
                "location_id":   r.get("locationId") or r.get("location_id"),
                "station_name":  loc.get("name") or r.get("location"),
                "lat":           coords.get("latitude"),
                "lon":           coords.get("longitude"),
                "datetime":      r.get("date", {}).get("utc") or r.get("dateTime"),
                "parameter":     r.get("parameter"),
                "value":         r.get("value"),
                "unit":          r.get("unit"),
                "country":       country_code,
            }
            rows.append(row)
        except Exception as e:
            logger.debug(f"Error parseando registro: {e}")
            continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Convertir datetime
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df["datetime"] = df["datetime"].dt.tz_convert(None)  # remover timezone para facilitar joins

    # Convertir columnas numéricas
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["lat"]   = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"]   = pd.to_numeric(df["lon"], errors="coerce")

    # Estandarizar nombre de parámetro
    df["parameter"] = df["parameter"].str.lower().map(VALID_PARAMETERS).fillna(df["parameter"])

    # Eliminar valores negativos (físicamente imposibles)
    n_neg = (df["value"] < 0).sum()
    if n_neg:
        logger.warning(f"Eliminando {n_neg} valores negativos")
        df = df[df["value"] >= 0]

    df = df.drop_duplicates(subset=["location_id", "datetime", "parameter"])
    df = df.sort_values(["location_id", "datetime"]).reset_index(drop=True)
    return df


def fetch_multiple_countries(
    country_codes: list[str],
    parameters: list[str],
    date_from: str,
    date_to: str,
) -> pd.DataFrame:
    """
    Descarga datos para múltiples países y parámetros.
    Útil para análisis regionales LATAM.

    Parámetros
    ----------
    country_codes : list[str]
        Lista de códigos de países (ej: ['GT', 'MX', 'CO', 'PE']).
    parameters : list[str]
        Lista de parámetros (ej: ['pm25', 'no2']).
    date_from, date_to : str
        Rango de fechas.

    Retorna
    -------
    pd.DataFrame combinado de todos los países y parámetros.
    """
    dfs = []
    for country in country_codes:
        for param in parameters:
            logger.info(f"Procesando: {country} — {param}")
            df = fetch_openaq(country, param, date_from, date_to)
            if not df.empty:
                dfs.append(df)
            time.sleep(1.0)  # pausa extra entre países

    if not dfs:
        logger.warning("No se obtuvieron datos en ninguna combinación país/parámetro.")
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"DataFrame combinado: {len(combined)} registros de {len(dfs)} descargas")
    return combined


# ─────────────────────────────────────────────
# EJEMPLOS DE USO
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Ejemplo 1: Guatemala — PM2.5, último mes ===")
    df_gt = fetch_openaq(
        country_code="GT",
        parameter="pm25",
        date_from="2024-01-01",
        date_to="2024-01-31",
    )
    if not df_gt.empty:
        print(df_gt.head())
        print(f"Estaciones únicas: {df_gt['station_name'].nunique()}")

    print("\n=== Ejemplo 2: México — NO2 ===")
    df_mx = fetch_openaq(
        country_code="MX",
        parameter="no2",
        date_from="2024-01-01",
        date_to="2024-01-07",
    )
    if not df_mx.empty:
        print(df_mx.describe())
