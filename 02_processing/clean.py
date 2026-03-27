"""
clean.py — Clase AirQualityCleaner para limpieza y control de calidad de datos
Incluye: eliminación de duplicados, manejo de valores faltantes, detección de
outliers (IQR, z-score, límites físicos), estandarización de unidades y validación.

Autor: Rodrigo, Ingeniero Ambiental, Universidad Anáhuac Cancún
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PHYSICAL_LIMITS, PPB_TO_UGM3, POLLUTANTS

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
)


class AirQualityCleaner:
    """
    Clase para limpieza y control de calidad de datos de calidad del aire.

    Ejemplo de uso:
    --------------
    cleaner = AirQualityCleaner(df)
    df_limpio = cleaner.clean_pipeline()
    reporte = cleaner.generate_qc_report(df_limpio)
    """

    def __init__(self, df: pd.DataFrame, pollutant_cols: list[str] | None = None):
        """
        Parámetros
        ----------
        df : pd.DataFrame
            DataFrame de entrada con datos de calidad del aire.
        pollutant_cols : list[str] | None
            Lista de columnas de contaminantes. Si es None, se detectan automáticamente
            buscando columnas que coincidan con los nombres en POLLUTANTS.
        """
        self.df_original = df.copy()
        self.pollutant_cols = pollutant_cols or self._detect_pollutant_cols(df)
        logger.info(
            f"AirQualityCleaner inicializado: {len(df)} filas, "
            f"contaminantes detectados: {self.pollutant_cols}"
        )

    def _detect_pollutant_cols(self, df: pd.DataFrame) -> list[str]:
        """Detecta columnas de contaminantes presentes en el DataFrame."""
        return [c for c in df.columns if c in POLLUTANTS or
                any(p.lower() in c.lower() for p in POLLUTANTS)]

    # ─────────────────────────────────────────────
    # 1. ELIMINAR DUPLICADOS
    # ─────────────────────────────────────────────

    def remove_duplicates(
        self,
        df: pd.DataFrame,
        subset: list[str] | None = None,
        keep: str = "first",
    ) -> pd.DataFrame:
        """
        Elimina filas duplicadas por estación y datetime.

        Parámetros
        ----------
        df : pd.DataFrame
        subset : list[str] | None
            Columnas para identificar duplicados. Default: ['station_id', 'datetime'].
        keep : str
            'first', 'last' o False para eliminar todos los duplicados.

        Retorna
        -------
        pd.DataFrame sin duplicados.
        """
        if subset is None:
            # Construir subset según columnas disponibles
            candidates = ["station_id", "datetime", "parameter", "location_id"]
            subset = [c for c in candidates if c in df.columns]
            if not subset:
                logger.warning("No se encontraron columnas clave para deduplicación — usando todas")
                subset = None

        n_antes = len(df)
        df_clean = df.drop_duplicates(subset=subset, keep=keep).copy()
        n_eliminados = n_antes - len(df_clean)

        if n_eliminados:
            logger.info(f"Duplicados eliminados: {n_eliminados} ({n_eliminados/n_antes*100:.1f}%)")
        else:
            logger.info("Sin duplicados encontrados")

        return df_clean.reset_index(drop=True)

    # ─────────────────────────────────────────────
    # 2. MANEJO DE VALORES FALTANTES
    # ─────────────────────────────────────────────

    def handle_missing(
        self,
        df: pd.DataFrame,
        method: str = "interpolate",
        max_gap_hours: int = 6,
        pollutant_cols: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Maneja valores faltantes en series temporales.

        Parámetros
        ----------
        df : pd.DataFrame
            Debe tener columna 'datetime' y columnas de contaminantes.
        method : str
            'interpolate': interpolación lineal para gaps ≤ max_gap_hours.
            'ffill': relleno hacia adelante.
            'bfill': relleno hacia atrás.
            'none': solo marca los gaps, no rellena.
        max_gap_hours : int
            Máximo gap en horas a rellenar. Gaps mayores se dejan como NaN.
        pollutant_cols : list[str] | None
            Columnas a procesar. Default: las detectadas en __init__.

        Retorna
        -------
        pd.DataFrame con valores faltantes manejados.
        """
        cols = pollutant_cols or self.pollutant_cols
        df = df.copy()

        if "datetime" not in df.columns:
            logger.warning("Sin columna 'datetime' — saltando interpolación temporal")
            return df

        df = df.sort_values("datetime")

        # Procesar por estación (si existe columna station_id)
        station_col = next((c for c in ["station_id", "location_id"] if c in df.columns), None)

        if station_col:
            grupos = df.groupby(station_col)
        else:
            grupos = [(None, df)]

        dfs_procesados = []
        total_rellenados = 0

        for station, grupo in grupos:
            grupo = grupo.copy().sort_values("datetime").reset_index(drop=True)

            for col in cols:
                if col not in grupo.columns:
                    continue

                n_nulos_antes = grupo[col].isna().sum()
                if n_nulos_antes == 0:
                    continue

                if method == "interpolate":
                    # Calcular diferencia de tiempo entre registros
                    if len(grupo) > 1:
                        dt_diff = grupo["datetime"].diff().dt.total_seconds() / 3600
                        # Marcar como NaN los gaps mayores a max_gap_hours antes de interpolar
                        gap_mask = (dt_diff > max_gap_hours).cumsum()
                        for _, seg in grupo.groupby(gap_mask):
                            idx = seg.index
                            grupo.loc[idx, col] = grupo.loc[idx, col].interpolate(
                                method="linear", limit=max_gap_hours
                            )
                    else:
                        grupo[col] = grupo[col].interpolate(method="linear")

                elif method == "ffill":
                    grupo[col] = grupo[col].fillna(method="ffill", limit=max_gap_hours)
                elif method == "bfill":
                    grupo[col] = grupo[col].fillna(method="bfill", limit=max_gap_hours)
                elif method == "none":
                    pass

                n_rellenados = n_nulos_antes - grupo[col].isna().sum()
                total_rellenados += n_rellenados

            dfs_procesados.append(grupo)

        df_clean = pd.concat(dfs_procesados, ignore_index=True)
        logger.info(
            f"Valores faltantes: método='{method}', max_gap={max_gap_hours}h, "
            f"rellenados={total_rellenados}"
        )
        return df_clean

    # ─────────────────────────────────────────────
    # 3. DETECCIÓN Y ELIMINACIÓN DE OUTLIERS
    # ─────────────────────────────────────────────

    def remove_outliers(
        self,
        df: pd.DataFrame,
        method: str = "iqr",
        factor: float = 3.0,
        zscore_threshold: float = 3.5,
        pollutant_cols: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Detecta y elimina outliers en columnas de contaminantes.

        Parámetros
        ----------
        df : pd.DataFrame
        method : str
            'iqr': método IQR con factor ajustable.
            'zscore': z-score con umbral configurable.
            'physical': límites físicos por contaminante (más conservador).
            'all': aplica los tres métodos en cascada.
        factor : float
            Factor para IQR: outlier si valor > Q3 + factor*IQR o < Q1 - factor*IQR.
        zscore_threshold : float
            Umbral para z-score (típicamente 3.0 a 3.5).
        pollutant_cols : list[str] | None
            Columnas a procesar.

        Retorna
        -------
        pd.DataFrame con outliers reemplazados por NaN.
        """
        cols = pollutant_cols or self.pollutant_cols
        df = df.copy()
        total_outliers = 0

        for col in cols:
            if col not in df.columns:
                continue

            serie = df[col].copy()
            n_validos = serie.notna().sum()
            if n_validos < 4:
                logger.debug(f"Columna '{col}': muy pocos datos ({n_validos}) para detectar outliers")
                continue

            outlier_mask = pd.Series(False, index=df.index)

            if method in ("iqr", "all"):
                q1 = serie.quantile(0.25)
                q3 = serie.quantile(0.75)
                iqr = q3 - q1
                if iqr > 0:
                    lower = q1 - factor * iqr
                    upper = q3 + factor * iqr
                    outlier_mask |= (serie < lower) | (serie > upper)

            if method in ("zscore", "all"):
                mean = serie.mean()
                std  = serie.std()
                if std > 0:
                    z = (serie - mean).abs() / std
                    outlier_mask |= (z > zscore_threshold)

            if method in ("physical", "all"):
                # Usar nombre de columna para identificar contaminante
                limit_key = next(
                    (k for k in PHYSICAL_LIMITS if k.lower() in col.lower() or col.lower() in k.lower()),
                    None
                )
                if limit_key:
                    lo, hi = PHYSICAL_LIMITS[limit_key]
                    outlier_mask |= (serie < lo) | (serie > hi)

            n_out = outlier_mask.sum()
            if n_out > 0:
                df.loc[outlier_mask, col] = np.nan
                total_outliers += n_out
                logger.debug(
                    f"Outliers en '{col}': {n_out} ({n_out/n_validos*100:.1f}%) → NaN"
                )

        logger.info(f"Total outliers eliminados: {total_outliers} (método: {method})")
        return df

    # ─────────────────────────────────────────────
    # 4. ESTANDARIZACIÓN DE UNIDADES
    # ─────────────────────────────────────────────

    def standardize_units(
        self,
        df: pd.DataFrame,
        unit_col: str = "unit",
        temp_celsius: float = 25.0,
        pressure_hpa: float = 1013.25,
    ) -> pd.DataFrame:
        """
        Convierte unidades entre ppb, ppm y µg/m³ o mg/m³.

        Parámetros
        ----------
        df : pd.DataFrame
            Si tiene columna 'unit', usa esa para detectar la unidad por fila.
            Si no, asume µg/m³ como unidad estándar.
        unit_col : str
            Nombre de la columna con la unidad de medida.
        temp_celsius : float
            Temperatura para conversión ppb → µg/m³.
        pressure_hpa : float
            Presión atmosférica para conversión.

        Retorna
        -------
        pd.DataFrame con columna 'value' (o columnas de contaminantes) en µg/m³.
        """
        df = df.copy()

        # Factor de corrección por temperatura y presión
        # (relativo a condiciones estándar 25°C, 1013.25 hPa)
        temp_factor = (273.15 + 25.0) / (273.15 + temp_celsius)
        press_factor = pressure_hpa / 1013.25
        correction = temp_factor * press_factor

        # Caso 1: DataFrame con columnas 'parameter', 'value', 'unit' (formato largo OpenAQ)
        if "parameter" in df.columns and "value" in df.columns and unit_col in df.columns:
            mask_ppb = df[unit_col].str.lower().isin(["ppb", "ppbv"])
            mask_ppm = df[unit_col].str.lower().isin(["ppm", "ppmv"])

            for param, factor in PPB_TO_UGM3.items():
                param_mask = df["parameter"].str.upper() == param.upper()

                # ppb → µg/m³
                conv_mask = mask_ppb & param_mask
                if conv_mask.sum() > 0:
                    df.loc[conv_mask, "value"] *= factor * correction
                    df.loc[conv_mask, unit_col] = "µg/m³"
                    logger.info(f"Convertido {conv_mask.sum()} registros {param} ppb → µg/m³")

                # ppm → µg/m³ (= ppb * 1000 * factor)
                conv_mask_ppm = mask_ppm & param_mask
                if conv_mask_ppm.sum() > 0:
                    df.loc[conv_mask_ppm, "value"] *= factor * correction * 1000
                    df.loc[conv_mask_ppm, unit_col] = "µg/m³"
                    logger.info(f"Convertido {conv_mask_ppm.sum()} registros {param} ppm → µg/m³")

        # Caso 2: DataFrame con columnas por contaminante (formato ancho)
        else:
            if unit_col in df.columns:
                for col in self.pollutant_cols:
                    if col not in df.columns:
                        continue
                    factor_key = next(
                        (k for k in PPB_TO_UGM3 if k.upper() == col.upper()), None
                    )
                    if not factor_key:
                        continue

                    mask_ppb = df[unit_col].str.lower() == "ppb"
                    if mask_ppb.sum() > 0:
                        df.loc[mask_ppb, col] *= PPB_TO_UGM3[factor_key] * correction
                        logger.info(f"Convertido {col} ppb → µg/m³ ({mask_ppb.sum()} filas)")

        return df

    # ─────────────────────────────────────────────
    # 5. VALIDACIÓN DE COORDENADAS
    # ─────────────────────────────────────────────

    def validate_coordinates(self, gdf) -> Any:
        """
        Verifica que lat ∈ [-90, 90] y lon ∈ [-180, 180].
        Marca como NaN las coordenadas fuera de rango.

        Parámetros
        ----------
        gdf : GeoDataFrame o DataFrame con columnas 'lat' y 'lon'.

        Retorna
        -------
        GeoDataFrame/DataFrame con coordenadas inválidas corregidas.
        """
        gdf = gdf.copy()
        n_invalidos = 0

        if "lat" in gdf.columns:
            mask_lat = (gdf["lat"] < -90) | (gdf["lat"] > 90)
            if mask_lat.sum() > 0:
                logger.warning(f"Latitudes fuera de rango: {mask_lat.sum()} → NaN")
                gdf.loc[mask_lat, "lat"] = np.nan
                n_invalidos += mask_lat.sum()

        if "lon" in gdf.columns:
            mask_lon = (gdf["lon"] < -180) | (gdf["lon"] > 180)
            if mask_lon.sum() > 0:
                logger.warning(f"Longitudes fuera de rango: {mask_lon.sum()} → NaN")
                gdf.loc[mask_lon, "lon"] = np.nan
                n_invalidos += mask_lon.sum()

        if n_invalidos == 0:
            logger.info("Coordenadas válidas: todas dentro de rango")

        return gdf

    # ─────────────────────────────────────────────
    # 6. REPORTE DE CALIDAD
    # ─────────────────────────────────────────────

    def generate_qc_report(self, df: pd.DataFrame) -> dict:
        """
        Genera un reporte de control de calidad con estadísticas detalladas.

        Retorna
        -------
        dict con:
            - n_total: total de registros
            - n_validos: registros con al menos un valor de contaminante
            - pct_datos_validos: % datos completos por contaminante
            - rango_fechas: (fecha_min, fecha_max)
            - duracion_dias: días cubiertos
            - gaps_detectados: lista de gaps > 1h
            - outliers_por_columna: estadísticas de outliers
            - estaciones: número de estaciones únicas
        """
        reporte = {
            "n_total": len(df),
            "n_columnas": len(df.columns),
            "contaminantes": self.pollutant_cols,
        }

        # Rango de fechas
        if "datetime" in df.columns:
            dt = pd.to_datetime(df["datetime"])
            fecha_min = dt.min()
            fecha_max = dt.max()
            reporte["rango_fechas"] = (str(fecha_min), str(fecha_max))
            reporte["duracion_dias"] = (fecha_max - fecha_min).days if pd.notna(fecha_min) else None
        else:
            reporte["rango_fechas"] = None
            reporte["duracion_dias"] = None

        # % datos válidos por contaminante
        pct_validos = {}
        for col in self.pollutant_cols:
            if col in df.columns:
                n_val = df[col].notna().sum()
                pct_validos[col] = round(n_val / len(df) * 100, 1) if len(df) > 0 else 0
        reporte["pct_datos_validos"] = pct_validos

        # Detectar gaps en serie temporal (diferencias > 1h entre registros consecutivos)
        gaps = []
        if "datetime" in df.columns and len(df) > 1:
            station_col = next((c for c in ["station_id", "location_id"] if c in df.columns), None)
            grupos = df.groupby(station_col) if station_col else [(None, df)]
            for station, grupo in grupos:
                grupo_sorted = grupo.sort_values("datetime")
                dt_diff = pd.to_datetime(grupo_sorted["datetime"]).diff()
                gap_idx = dt_diff[dt_diff > pd.Timedelta(hours=1)]
                for idx, gap in gap_idx.items():
                    gaps.append({
                        "estacion": station,
                        "fecha_inicio": str(pd.to_datetime(grupo_sorted.loc[idx, "datetime"]) - gap),
                        "duracion_horas": round(gap.total_seconds() / 3600, 1),
                    })
        reporte["gaps_detectados"] = gaps[:20]  # limitar a 20 primeros
        reporte["n_gaps_total"] = len(gaps)

        # Estadísticas básicas de outliers (usando IQR)
        outlier_stats = {}
        for col in self.pollutant_cols:
            if col in df.columns:
                serie = df[col].dropna()
                if len(serie) > 3:
                    q1, q3 = serie.quantile(0.25), serie.quantile(0.75)
                    iqr = q3 - q1
                    n_out = ((serie < q1 - 3*iqr) | (serie > q3 + 3*iqr)).sum()
                    outlier_stats[col] = {
                        "mean": round(serie.mean(), 2),
                        "std":  round(serie.std(), 2),
                        "min":  round(serie.min(), 2),
                        "max":  round(serie.max(), 2),
                        "n_outliers_iqr3": int(n_out),
                        "pct_outliers": round(n_out / len(serie) * 100, 2),
                    }
        reporte["estadisticas_por_contaminante"] = outlier_stats

        # Número de estaciones
        station_col = next((c for c in ["station_id", "location_id", "station_name"] if c in df.columns), None)
        if station_col:
            reporte["n_estaciones"] = df[station_col].nunique()
        else:
            reporte["n_estaciones"] = None

        return reporte

    # ─────────────────────────────────────────────
    # 7. PIPELINE COMPLETO
    # ─────────────────────────────────────────────

    def clean_pipeline(
        self,
        df: pd.DataFrame | None = None,
        outlier_method: str = "iqr",
        missing_method: str = "interpolate",
        max_gap_hours: int = 6,
        standardize_units: bool = True,
        validate_coords: bool = True,
    ) -> pd.DataFrame:
        """
        Ejecuta el pipeline completo de limpieza en orden:
        1. remove_duplicates
        2. validate_coordinates
        3. standardize_units
        4. remove_outliers
        5. handle_missing

        Parámetros
        ----------
        df : pd.DataFrame | None
            Si es None, usa self.df_original.
        outlier_method : str
            Método de detección de outliers ('iqr', 'zscore', 'physical', 'all').
        missing_method : str
            Método para valores faltantes ('interpolate', 'ffill', 'bfill', 'none').
        max_gap_hours : int
            Máximo gap a rellenar.
        standardize_units : bool
            Si True, aplica conversión de unidades.
        validate_coords : bool
            Si True, valida coordenadas lat/lon.

        Retorna
        -------
        pd.DataFrame limpio.
        """
        if df is None:
            df = self.df_original.copy()

        n_inicial = len(df)
        logger.info(f"=== Iniciando pipeline de limpieza ({n_inicial} registros) ===")

        # Paso 1: Duplicados
        df = self.remove_duplicates(df)

        # Paso 2: Coordenadas
        if validate_coords:
            df = self.validate_coordinates(df)

        # Paso 3: Unidades
        if standardize_units and "unit" in df.columns:
            df = self.standardize_units(df)

        # Paso 4: Outliers
        df = self.remove_outliers(df, method=outlier_method)

        # Paso 5: Valores faltantes
        df = self.handle_missing(df, method=missing_method, max_gap_hours=max_gap_hours)

        logger.info(
            f"=== Pipeline completado: {n_inicial} → {len(df)} registros "
            f"({n_inicial - len(df)} eliminados) ==="
        )
        return df


# ─────────────────────────────────────────────
# FUNCIÓN DE CONVENIENCIA
# ─────────────────────────────────────────────

def clean_pipeline(
    df: pd.DataFrame,
    outlier_method: str = "iqr",
    missing_method: str = "interpolate",
) -> pd.DataFrame:
    """
    Función de conveniencia: ejecuta el pipeline completo en una línea.

    Ejemplo:
    --------
    df_limpio = clean_pipeline(df_raw)
    """
    cleaner = AirQualityCleaner(df)
    return cleaner.clean_pipeline(
        outlier_method=outlier_method,
        missing_method=missing_method,
    )


# ─────────────────────────────────────────────
# EJEMPLO DE USO
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from ingestion.load_stations import generate_synthetic_data  # noqa

    # Generar datos sintéticos
    df_sint = generate_synthetic_data(n_stations=3, days=7)
    print(f"Datos originales: {len(df_sint)} filas")

    # Limpiar
    cleaner = AirQualityCleaner(df_sint)
    df_limpio = cleaner.clean_pipeline()
    print(f"Datos limpios: {len(df_limpio)} filas")

    # Reporte
    reporte = cleaner.generate_qc_report(df_limpio)
    print("\n=== Reporte de Calidad ===")
    for clave, valor in reporte.items():
        if clave != "gaps_detectados":
            print(f"  {clave}: {valor}")
    print(f"  gaps_detectados (primeros 5): {reporte['gaps_detectados'][:5]}")
