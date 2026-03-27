"""
timeseries.py — Análisis de series de tiempo para datos de calidad del aire
Incluye: descomposición STL, detección de anomalías, pronóstico ARIMA,
detección de cambios estructurales y estadísticas móviles.

Autor: Rodrigo, Ingeniero Ambiental, Universidad Anáhuac Cancún
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

try:
    from statsmodels.tsa.seasonal import STL
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    logging.warning("statsmodels no disponible — instalar: pip install statsmodels")

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    logging.warning("scikit-learn no disponible — instalar: pip install scikit-learn")

try:
    import ruptures as rpt
    HAS_RUPTURES = True
except ImportError:
    HAS_RUPTURES = False
    logging.warning("ruptures no disponible (cambios estructurales) — instalar: pip install ruptures")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import FIGURES_DIR

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
)


def _prepare_timeseries(df: pd.DataFrame, variable: str, freq: str = "h") -> pd.Series:
    """
    Prepara una serie de tiempo con índice temporal y frecuencia definida.
    Rellena huecos con interpolación lineal.
    """
    if "datetime" not in df.columns:
        raise ValueError("Se requiere columna 'datetime'")
    if variable not in df.columns:
        raise KeyError(f"Variable '{variable}' no encontrada")

    serie = (df.set_index(pd.to_datetime(df["datetime"]))[variable]
             .sort_index()
             .resample(freq)
             .mean())
    n_nan = serie.isna().sum()
    if n_nan > 0:
        serie = serie.interpolate(method="time", limit=24)
        logger.debug(f"Interpolados {n_nan} valores en resample")
    return serie


# ─────────────────────────────────────────────
# 1. DESCOMPOSICIÓN STL
# ─────────────────────────────────────────────

def decompose_series(
    df: pd.DataFrame,
    variable: str,
    period: int = 24,       # 24h para datos horarios (ciclo diurno)
    model: str = "additive",
    seasonal_smoother: int = 7,
    freq_resample: str = "h",
) -> dict:
    """
    Descomposición STL (Seasonal and Trend decomposition using Loess).
    Extrae tendencia, componente estacional y residuo.

    Parámetros
    ----------
    df : pd.DataFrame
        Con columnas 'datetime' y variable.
    variable : str
    period : int
        Período de la estacionalidad (24 para diurno, 8760 para anual).
    model : str
        'additive' o 'multiplicative'.
    seasonal_smoother : int
        Suavizador de la componente estacional (debe ser impar).
    freq_resample : str
        Frecuencia del resample ('h' = horario, 'D' = diario).

    Retorna
    -------
    dict con claves: 'trend', 'seasonal', 'residual', 'observed',
                     'strength_seasonal', 'strength_trend'
    """
    if not HAS_STATSMODELS:
        raise ImportError("statsmodels requerido. Instalar: pip install statsmodels")

    serie = _prepare_timeseries(df, variable, freq=freq_resample)
    serie = serie.dropna()

    if len(serie) < 2 * period:
        raise ValueError(
            f"Serie muy corta ({len(serie)} puntos) para descomposición "
            f"con período {period}. Se necesitan al menos {2*period} puntos."
        )

    # Asegurar seasonal_smoother impar
    if seasonal_smoother % 2 == 0:
        seasonal_smoother += 1

    logger.info(
        f"Descomposición STL: '{variable}' | período={period} | "
        f"modelo={model} | n={len(serie)}"
    )

    stl = STL(serie, period=period, seasonal=seasonal_smoother)
    result = stl.fit()

    # Fuerza de la estacionalidad y tendencia (Wang et al., 2006)
    var_residual = np.var(result.resid)
    var_seasonal_residual = np.var(result.resid + result.seasonal)
    var_trend_residual    = np.var(result.resid + result.trend)

    strength_seasonal = max(0, 1 - var_residual / var_seasonal_residual) if var_seasonal_residual > 0 else 0
    strength_trend    = max(0, 1 - var_residual / var_trend_residual) if var_trend_residual > 0 else 0

    return {
        "trend":             pd.Series(result.trend, index=serie.index, name=f"{variable}_trend"),
        "seasonal":          pd.Series(result.seasonal, index=serie.index, name=f"{variable}_seasonal"),
        "residual":          pd.Series(result.resid, index=serie.index, name=f"{variable}_residual"),
        "observed":          serie,
        "strength_seasonal": round(strength_seasonal, 3),
        "strength_trend":    round(strength_trend, 3),
        "period":            period,
        "model":             model,
    }


# ─────────────────────────────────────────────
# 2. DETECCIÓN DE ANOMALÍAS
# ─────────────────────────────────────────────

def detect_anomalies(
    df: pd.DataFrame,
    variable: str,
    method: str = "isolation_forest",
    contamination: float = 0.05,
    zscore_threshold: float = 3.0,
    iqr_factor: float = 3.0,
) -> pd.DataFrame:
    """
    Detecta anomalías en la serie temporal.

    Parámetros
    ----------
    df : pd.DataFrame
    variable : str
    method : str
        'isolation_forest': usa IsolationForest de sklearn.
        'zscore': z-score simple.
        'iqr': rango intercuartílico.
    contamination : float
        Fracción esperada de anomalías (para isolation_forest).
    zscore_threshold : float
        Umbral de z-score.
    iqr_factor : float
        Factor para IQR.

    Retorna
    -------
    pd.DataFrame con columna adicional 'anomalia' (bool) y 'score_anomalia'.
    """
    if variable not in df.columns:
        raise KeyError(f"Variable '{variable}' no encontrada")

    df = df.copy()
    serie = df[variable].values
    validos = ~np.isnan(serie)

    anomalias = np.zeros(len(df), dtype=bool)
    scores    = np.zeros(len(df), dtype=float)

    if method == "isolation_forest":
        if not HAS_SKLEARN:
            logger.warning("sklearn no disponible — usando z-score como fallback")
            method = "zscore"
        else:
            X = serie[validos].reshape(-1, 1)
            iso = IsolationForest(
                contamination=contamination,
                random_state=42,
                n_estimators=100,
            )
            preds = iso.fit_predict(X)
            scores_if = iso.score_samples(X)
            idx_validos = np.where(validos)[0]
            anomalias[idx_validos] = preds == -1
            scores[idx_validos]    = -scores_if  # invertir: mayor = más anómalo
            logger.info(
                f"IsolationForest: {anomalias.sum()} anomalías detectadas "
                f"({anomalias.sum()/validos.sum()*100:.1f}%)"
            )

    if method == "zscore":
        mean = np.nanmean(serie)
        std  = np.nanstd(serie)
        if std > 0:
            z = np.abs((serie - mean) / std)
            anomalias = (z > zscore_threshold) & validos
            scores    = z
        logger.info(f"Z-score (umbral={zscore_threshold}): {anomalias.sum()} anomalías")

    elif method == "iqr":
        q1 = np.nanpercentile(serie, 25)
        q3 = np.nanpercentile(serie, 75)
        iqr = q3 - q1
        lower = q1 - iqr_factor * iqr
        upper = q3 + iqr_factor * iqr
        anomalias = ((serie < lower) | (serie > upper)) & validos
        scores    = np.where(serie > upper, serie - upper, np.where(serie < lower, lower - serie, 0))
        logger.info(f"IQR (factor={iqr_factor}): {anomalias.sum()} anomalías")

    df["anomalia"]      = anomalias
    df["score_anomalia"] = scores
    return df


# ─────────────────────────────────────────────
# 3. PRONÓSTICO ARIMA
# ─────────────────────────────────────────────

def forecast_arima(
    df: pd.DataFrame,
    variable: str,
    steps: int = 30,
    order: tuple | None = None,
    freq_resample: str = "D",
    confidence_level: float = 0.95,
) -> dict:
    """
    Pronóstico ARIMA con auto-selección de parámetros por AIC.

    Parámetros
    ----------
    df : pd.DataFrame
    variable : str
    steps : int
        Pasos hacia adelante a pronosticar.
    order : tuple | None
        (p, d, q) para ARIMA. Si es None, se selecciona automáticamente
        probando una cuadrícula de parámetros.
    freq_resample : str
        Frecuencia del resample ('D'=diario, 'W'=semanal).
    confidence_level : float
        Nivel de confianza para intervalos de predicción.

    Retorna
    -------
    dict con:
        - forecast: pd.Series con el pronóstico
        - ci_lower: intervalo de confianza inferior
        - ci_upper: intervalo de confianza superior
        - order: parámetros ARIMA seleccionados
        - aic: criterio de información de Akaike
        - rmse: error cuadrático medio en datos de entrenamiento
    """
    if not HAS_STATSMODELS:
        raise ImportError("statsmodels requerido. Instalar: pip install statsmodels")

    serie = _prepare_timeseries(df, variable, freq=freq_resample).dropna()
    if len(serie) < 30:
        raise ValueError(f"Serie muy corta ({len(serie)} puntos) para ARIMA")

    # Test de estacionariedad (Dickey-Fuller aumentado)
    adf_stat, adf_p, *_ = adfuller(serie.dropna())
    d = 0 if adf_p < 0.05 else 1
    logger.info(f"ADF test: stat={adf_stat:.3f}, p={adf_p:.3f} → d={d}")

    # Auto-selección de parámetros si no se especifican
    if order is None:
        best_aic = np.inf
        best_order = (1, d, 1)
        for p in range(0, 4):
            for q in range(0, 4):
                try:
                    modelo = ARIMA(serie, order=(p, d, q)).fit()
                    if modelo.aic < best_aic:
                        best_aic = modelo.aic
                        best_order = (p, d, q)
                except Exception:
                    continue
        order = best_order
        logger.info(f"Mejor orden ARIMA por AIC: {order} (AIC={best_aic:.1f})")
    else:
        best_aic = None

    # Ajustar modelo final
    modelo = ARIMA(serie, order=order).fit()
    if best_aic is None:
        best_aic = modelo.aic

    # Pronóstico
    alpha = 1 - confidence_level
    forecast_result = modelo.get_forecast(steps=steps)
    forecast_mean = forecast_result.predicted_mean
    ci = forecast_result.conf_int(alpha=alpha)

    # RMSE en datos de entrenamiento
    fitted = modelo.fittedvalues
    residuales = serie - fitted
    rmse = np.sqrt(np.mean(residuales**2))

    logger.info(
        f"ARIMA{order} | AIC={best_aic:.1f} | RMSE={rmse:.3f} | "
        f"Pronóstico {steps} pasos"
    )

    return {
        "forecast":  forecast_mean,
        "ci_lower":  ci.iloc[:, 0],
        "ci_upper":  ci.iloc[:, 1],
        "order":     order,
        "aic":       round(best_aic, 2),
        "rmse":      round(rmse, 3),
        "serie_original": serie,
    }


# ─────────────────────────────────────────────
# 4. DETECCIÓN DE CAMBIOS ESTRUCTURALES
# ─────────────────────────────────────────────

def change_point_detection(
    df: pd.DataFrame,
    variable: str,
    n_breakpoints: int = 3,
    model: str = "rbf",
    freq_resample: str = "D",
) -> dict:
    """
    Detecta cambios estructurales en la serie temporal usando `ruptures`.

    Parámetros
    ----------
    df : pd.DataFrame
    variable : str
    n_breakpoints : int
        Número de puntos de cambio a detectar.
    model : str
        Modelo de ruptures: 'rbf', 'l2', 'l1', 'normal'.
    freq_resample : str
        Frecuencia del resample.

    Retorna
    -------
    dict con 'breakpoints' (índices) y 'fechas_cambio'.
    """
    if not HAS_RUPTURES:
        logger.warning("ruptures no disponible — instalar: pip install ruptures")
        return {"error": "ruptures no instalado", "breakpoints": []}

    serie = _prepare_timeseries(df, variable, freq=freq_resample).dropna()
    signal = serie.values.reshape(-1, 1)

    algo = rpt.Pelt(model=model).fit(signal)
    breakpoints = algo.predict(pen=10)

    # Convertir índices a fechas
    fechas_cambio = []
    for bp in breakpoints[:-1]:  # el último es el fin de la serie
        if bp < len(serie):
            fechas_cambio.append(str(serie.index[bp]))

    logger.info(f"Cambios estructurales en '{variable}': {len(fechas_cambio)} puntos")
    return {
        "breakpoints":    breakpoints,
        "fechas_cambio":  fechas_cambio,
        "n_segmentos":    len(breakpoints),
        "serie":          serie,
    }


# ─────────────────────────────────────────────
# 5. ESTADÍSTICAS MÓVILES
# ─────────────────────────────────────────────

def rolling_statistics(
    df: pd.DataFrame,
    variable: str,
    windows: list[int] = [7, 30, 90],
    freq_resample: str = "D",
) -> pd.DataFrame:
    """
    Calcula medias y desviaciones estándar móviles para ventanas temporales.

    Parámetros
    ----------
    df : pd.DataFrame
    variable : str
    windows : list[int]
        Tamaños de ventana en días (o períodos según freq_resample).
    freq_resample : str
        Frecuencia del resample.

    Retorna
    -------
    pd.DataFrame con la serie original y columnas de medias/std móviles.
    """
    serie = _prepare_timeseries(df, variable, freq=freq_resample)
    result_df = pd.DataFrame({"observado": serie})

    for w in windows:
        if w >= len(serie):
            logger.warning(f"Ventana {w} mayor que la serie ({len(serie)}) — omitiendo")
            continue
        result_df[f"media_{w}d"]     = serie.rolling(window=w, center=True, min_periods=w//2).mean()
        result_df[f"std_{w}d"]       = serie.rolling(window=w, center=True, min_periods=w//2).std()
        result_df[f"p25_{w}d"]       = serie.rolling(window=w, center=True, min_periods=w//2).quantile(0.25)
        result_df[f"p75_{w}d"]       = serie.rolling(window=w, center=True, min_periods=w//2).quantile(0.75)

    logger.info(f"Estadísticas móviles de '{variable}': ventanas={windows}")
    return result_df


# ─────────────────────────────────────────────
# 6. GRÁFICO COMPLETO DE SERIE TEMPORAL
# ─────────────────────────────────────────────

def plot_timeseries_complete(
    df: pd.DataFrame,
    variable: str,
    output_path: str | Path | None = None,
    show_anomalies: bool = True,
    show_trend: bool = True,
    dpi: int = 150,
) -> Path | None:
    """
    Genera un gráfico completo de la serie temporal con tendencia y anomalías.

    Parámetros
    ----------
    df : pd.DataFrame
    variable : str
    output_path : str | Path | None
        Ruta de guardado. Default: outputs/figures/ts_{variable}.png
    show_anomalies : bool
        Si True, marca las anomalías en el gráfico.
    show_trend : bool
        Si True, superpone la tendencia STL.
    dpi : int

    Retorna
    -------
    Path al archivo guardado, o None si falló.
    """
    if not HAS_MATPLOTLIB:
        logger.error("matplotlib no disponible")
        return None

    if output_path is None:
        var_safe = variable.replace(".", "").replace(" ", "_")
        output_path = FIGURES_DIR / f"ts_completo_{var_safe}.png"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Calcular anomalías
    df_anom = detect_anomalies(df.copy(), variable, method="zscore")
    rolling_df = rolling_statistics(df, variable, windows=[7, 30])

    # Calcular tendencia STL (si hay suficientes datos)
    trend = None
    if show_trend and HAS_STATSMODELS:
        try:
            decomp = decompose_series(df, variable, period=24)
            trend = decomp["trend"].resample("D").mean()
        except Exception as e:
            logger.debug(f"No se pudo calcular tendencia: {e}")

    # ─── Figura con subplots ───
    n_plots = 2 if trend is not None else 1
    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 4 * n_plots), sharex=True)
    if n_plots == 1:
        axes = [axes]

    # Panel 1: Serie + media móvil + anomalías
    ax = axes[0]
    df_sorted = df.sort_values("datetime")
    ax.plot(pd.to_datetime(df_sorted["datetime"]), df_sorted[variable],
            alpha=0.5, lw=0.6, color="#1f77b4", label="Observado")

    if not rolling_df.empty and "media_7d" in rolling_df.columns:
        ax.plot(rolling_df.index, rolling_df["media_7d"],
                color="#d62728", lw=1.5, label="Media móvil 7d")
    if not rolling_df.empty and "media_30d" in rolling_df.columns:
        ax.plot(rolling_df.index, rolling_df["media_30d"],
                color="#2ca02c", lw=2, label="Media móvil 30d")

    if show_anomalies:
        anomalas_df = df_anom[df_anom["anomalia"] == True]
        if len(anomalas_df) > 0:
            ax.scatter(
                pd.to_datetime(anomalas_df["datetime"]),
                anomalas_df[variable],
                color="red", zorder=5, s=20, alpha=0.8, label=f"Anomalías ({len(anomalas_df)})",
            )

    ax.set_ylabel(f"{variable} (µg/m³)")
    ax.set_title(f"Serie temporal de {variable}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: Tendencia (si disponible)
    if trend is not None:
        ax2 = axes[1]
        ax2.plot(trend.index, trend.values, color="#9467bd", lw=2, label="Tendencia STL")
        ax2.set_ylabel(f"Tendencia {variable}")
        ax2.set_xlabel("Fecha")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")
    else:
        axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.setp(axes[0].get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    logger.info(f"Gráfico de serie temporal guardado: {output_path}")
    return output_path


# ─────────────────────────────────────────────
# EJEMPLO DE USO
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from ingestion.load_stations import generate_synthetic_data

    df = generate_synthetic_data(n_stations=1, days=180)
    df["datetime"] = pd.to_datetime(df["fecha"], format="%d/%m/%Y %H:%M")

    print("=== Estadísticas móviles ===")
    rolling = rolling_statistics(df, "PM2.5", windows=[7, 30])
    print(rolling.tail(10))

    print("\n=== Detección de anomalías ===")
    df_anom = detect_anomalies(df, "PM2.5", method="zscore")
    print(f"Anomalías detectadas: {df_anom['anomalia'].sum()}")

    if HAS_STATSMODELS:
        print("\n=== Descomposición STL ===")
        decomp = decompose_series(df, "PM2.5", period=24)
        print(f"Fuerza estacional: {decomp['strength_seasonal']}")
        print(f"Fuerza tendencia: {decomp['strength_trend']}")

    print("\n=== Generando gráfico completo ===")
    path = plot_timeseries_complete(df, "PM2.5")
    print(f"Guardado en: {path}")
