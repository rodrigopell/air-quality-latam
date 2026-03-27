"""
descriptive.py — Estadística descriptiva y visualización para datos de calidad del aire
Calcula métricas, patrones temporales, excedencias y genera gráficos.

Autor: Rodrigo, Ingeniero Ambiental, Universidad Anáhuac Cancún
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

try:
    import matplotlib
    matplotlib.use("Agg")  # backend sin GUI para generación de archivos
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import seaborn as sns
    sns.set_theme(style="whitegrid", context="notebook")
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logging.warning("matplotlib/seaborn no disponibles — gráficos deshabilitados")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import FIGURES_DIR, WHO_GUIDELINES_2021, AQI_CATEGORIES

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
)

# Colores consistentes para estaciones
STATION_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


# ─────────────────────────────────────────────
# 1. ESTADÍSTICA DESCRIPTIVA COMPLETA
# ─────────────────────────────────────────────

def full_descriptive(df: pd.DataFrame, variable: str) -> pd.DataFrame:
    """
    Calcula estadísticas descriptivas completas para una variable.
    Si el DataFrame tiene columna 'station_id', calcula por estación.

    Parámetros
    ----------
    df : pd.DataFrame
    variable : str
        Columna numérica a analizar.

    Retorna
    -------
    pd.DataFrame con estadísticas por estación (o global si no hay station_id).
    Columnas: mean, median, std, cv, p10, p25, p75, p90, p95, p99,
              min, max, skewness, kurtosis, n_valid, n_missing, pct_missing
    """
    if variable not in df.columns:
        raise KeyError(f"Variable '{variable}' no encontrada. Columnas: {list(df.columns)}")

    station_col = next((c for c in ["station_id", "location_id", "station_name"] if c in df.columns), None)

    def _stats(serie: pd.Series) -> dict:
        s = serie.dropna()
        n = len(s)
        n_miss = serie.isna().sum()
        if n < 2:
            return {k: np.nan for k in ["mean", "median", "std", "cv", "p10", "p25",
                                         "p75", "p90", "p95", "p99", "min", "max",
                                         "skewness", "kurtosis"]}
        return {
            "mean":     round(s.mean(), 3),
            "median":   round(s.median(), 3),
            "std":      round(s.std(), 3),
            "cv":       round(s.std() / s.mean() * 100, 1) if s.mean() != 0 else np.nan,
            "p10":      round(s.quantile(0.10), 3),
            "p25":      round(s.quantile(0.25), 3),
            "p75":      round(s.quantile(0.75), 3),
            "p90":      round(s.quantile(0.90), 3),
            "p95":      round(s.quantile(0.95), 3),
            "p99":      round(s.quantile(0.99), 3),
            "min":      round(s.min(), 3),
            "max":      round(s.max(), 3),
            "skewness": round(stats.skew(s), 3),
            "kurtosis": round(stats.kurtosis(s), 3),
            "n_valid":  int(n),
            "n_missing": int(n_miss),
            "pct_missing": round(n_miss / (n + n_miss) * 100, 1) if (n + n_miss) > 0 else 0,
        }

    if station_col:
        filas = []
        for station, grupo in df.groupby(station_col):
            row = {"station": station}
            row.update(_stats(grupo[variable]))
            filas.append(row)
        result = pd.DataFrame(filas).set_index("station")
    else:
        result = pd.DataFrame([_stats(df[variable])], index=["global"])

    logger.info(f"Estadísticas descriptivas de '{variable}': {len(result)} grupos")
    return result


# ─────────────────────────────────────────────
# 2. DESCOMPOSICIÓN TEMPORAL
# ─────────────────────────────────────────────

def temporal_decomposition(
    df: pd.DataFrame,
    variable: str,
    freq: str = "D",
) -> dict:
    """
    Calcula promedios de la variable por hora del día, día de semana,
    mes y año.

    Parámetros
    ----------
    df : pd.DataFrame
        Debe tener columna 'datetime'.
    variable : str
    freq : str
        Frecuencia de agrupación temporal ('H', 'D', 'M', 'Y').

    Retorna
    -------
    dict con claves: 'por_hora', 'por_dia_semana', 'por_mes', 'por_año'
    """
    if "datetime" not in df.columns:
        raise ValueError("Se requiere columna 'datetime'")
    if variable not in df.columns:
        raise KeyError(f"Variable '{variable}' no encontrada")

    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["hora"]      = df["datetime"].dt.hour
    df["dia_sem"]   = df["datetime"].dt.dayofweek  # 0=lunes
    df["mes"]       = df["datetime"].dt.month
    df["anio"]      = df["datetime"].dt.year

    DIAS = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
    MESES = ["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"]

    por_hora = (df.groupby("hora")[variable]
                .agg(["mean", "std", "count"])
                .rename(columns={"mean": "promedio", "std": "desv_std", "count": "n"}))

    por_dia_semana = (df.groupby("dia_sem")[variable]
                      .agg(["mean", "std", "count"])
                      .rename(columns={"mean": "promedio", "std": "desv_std", "count": "n"}))
    por_dia_semana.index = [DIAS[i] for i in por_dia_semana.index]

    por_mes = (df.groupby("mes")[variable]
               .agg(["mean", "std", "count"])
               .rename(columns={"mean": "promedio", "std": "desv_std", "count": "n"}))
    por_mes.index = [MESES[i-1] for i in por_mes.index]

    por_anio = (df.groupby("anio")[variable]
                .agg(["mean", "std", "count"])
                .rename(columns={"mean": "promedio", "std": "desv_std", "count": "n"}))

    logger.info(f"Descomposición temporal de '{variable}' completada")
    return {
        "por_hora":       por_hora,
        "por_dia_semana": por_dia_semana,
        "por_mes":        por_mes,
        "por_anio":       por_anio,
    }


# ─────────────────────────────────────────────
# 3. PATRÓN DIURNO CON INTERVALOS DE CONFIANZA
# ─────────────────────────────────────────────

def diurnal_pattern(
    df: pd.DataFrame,
    variable: str,
    station_col: str | None = None,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """
    Calcula el patrón diurno (promedio por hora) con intervalos de confianza.

    Parámetros
    ----------
    df : pd.DataFrame
    variable : str
    station_col : str | None
        Si se especifica, calcula por estación.
    confidence : float
        Nivel de confianza (default: 0.95 = 95%).

    Retorna
    -------
    pd.DataFrame con columnas: hora, promedio, ci_lower, ci_upper, n
    """
    if "datetime" not in df.columns:
        raise ValueError("Se requiere columna 'datetime'")

    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["hora"] = df["datetime"].dt.hour

    group_cols = ["hora"]
    if station_col and station_col in df.columns:
        group_cols = [station_col, "hora"]

    alpha = 1 - confidence
    z = stats.norm.ppf(1 - alpha / 2)

    filas = []
    for keys, grupo in df.groupby(group_cols):
        s = grupo[variable].dropna()
        n = len(s)
        if n < 2:
            continue
        mean = s.mean()
        sem  = s.sem()
        row = {
            "hora":     keys[-1] if isinstance(keys, tuple) else keys,
            "promedio": round(mean, 3),
            "ci_lower": round(mean - z * sem, 3),
            "ci_upper": round(mean + z * sem, 3),
            "n":        n,
        }
        if station_col and isinstance(keys, tuple):
            row[station_col] = keys[0]
        filas.append(row)

    result = pd.DataFrame(filas)
    logger.info(f"Patrón diurno de '{variable}': {len(result)} filas")
    return result


# ─────────────────────────────────────────────
# 4. TENDENCIA ANUAL (Mann-Kendall)
# ─────────────────────────────────────────────

def annual_trend(df: pd.DataFrame, variable: str) -> dict:
    """
    Calcula la tendencia anual usando regresión lineal y test de Mann-Kendall.

    Parámetros
    ----------
    df : pd.DataFrame
        Con columnas 'datetime' y variable.

    Retorna
    -------
    dict con:
        - slope: pendiente de la regresión (µg/m³/año)
        - intercept: intercepto
        - r2: coeficiente de determinación
        - p_value: significancia estadística
        - mann_kendall_tau: estadístico tau de Kendall
        - mann_kendall_p: p-value del test Mann-Kendall
        - tendencia: 'ascendente', 'descendente' o 'sin tendencia'
        - promedios_anuales: pd.Series de promedios anuales
    """
    if "datetime" not in df.columns:
        raise ValueError("Se requiere columna 'datetime'")

    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["anio"] = df["datetime"].dt.year

    promedios = df.groupby("anio")[variable].mean().dropna()
    if len(promedios) < 3:
        logger.warning(f"Menos de 3 años de datos para tendencia de '{variable}'")
        return {"error": "Datos insuficientes para análisis de tendencia"}

    x = promedios.index.values.astype(float)
    y = promedios.values

    # Regresión lineal
    slope, intercept, r, p_value, se = stats.linregress(x, y)

    # Mann-Kendall
    tau, mk_p = stats.kendalltau(x, y)

    # Determinar dirección de tendencia
    if mk_p < 0.05:
        tendencia = "ascendente" if slope > 0 else "descendente"
    else:
        tendencia = "sin tendencia estadísticamente significativa"

    resultado = {
        "slope":             round(slope, 4),
        "intercept":         round(intercept, 4),
        "r2":                round(r**2, 4),
        "p_value":           round(p_value, 4),
        "mann_kendall_tau":  round(tau, 4),
        "mann_kendall_p":    round(mk_p, 4),
        "tendencia":         tendencia,
        "promedios_anuales": promedios,
        "cambio_por_anio":   f"{slope:+.3f} µg/m³/año",
    }
    logger.info(
        f"Tendencia '{variable}': {tendencia} "
        f"(slope={slope:.4f}, p={p_value:.4f})"
    )
    return resultado


# ─────────────────────────────────────────────
# 5. ESTADÍSTICAS DE EXCEDENCIA
# ─────────────────────────────────────────────

def exceedance_stats(
    df: pd.DataFrame,
    variable: str,
    threshold: float | None = None,
    temporal_unit: str = "day",
) -> dict:
    """
    Calcula estadísticas de excedencia sobre un umbral (OMS o personalizado).

    Parámetros
    ----------
    df : pd.DataFrame
    variable : str
    threshold : float | None
        Umbral en µg/m³. Si es None, usa la guía OMS 2021 para 24h.
    temporal_unit : str
        'hour': excedencias por hora.
        'day': excedencias por día (media 24h).

    Retorna
    -------
    dict con estadísticas de excedencia.
    """
    if variable not in df.columns:
        raise KeyError(f"Variable '{variable}' no encontrada")

    # Usar umbral OMS si no se especifica
    if threshold is None:
        poll_key = variable.replace(".", "")
        if poll_key == "PM25":
            poll_key = "PM2.5"
        guidelines = WHO_GUIDELINES_2021.get(poll_key, {})
        threshold = guidelines.get("24h") or guidelines.get("8h") or guidelines.get("anual")
        if threshold is None:
            logger.warning(f"Sin umbral OMS para '{variable}'. Especificar threshold manualmente.")
            return {"error": f"Sin umbral para {variable}"}
        logger.info(f"Usando umbral OMS para {variable}: {threshold} µg/m³")

    df = df.copy()

    if temporal_unit == "day" and "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df["date"] = df["datetime"].dt.date
        grupo_col = "date"
        serie = df.groupby(grupo_col)[variable].mean()
    else:
        serie = df[variable].dropna()

    n_total = len(serie.dropna())
    excedencias = serie.dropna() > threshold
    n_excede = excedencias.sum()

    # Temporada con más excedencias
    if "datetime" in df.columns and n_excede > 0:
        df["mes"] = pd.to_datetime(df["datetime"]).dt.month
        exc_por_mes = df.groupby("mes")[variable].mean()
        mes_max = int(exc_por_mes.idxmax()) if not exc_por_mes.empty else None
        MESES = {1:"Enero",2:"Febrero",3:"Marzo",4:"Abril",5:"Mayo",6:"Junio",
                 7:"Julio",8:"Agosto",9:"Septiembre",10:"Octubre",11:"Noviembre",12:"Diciembre"}
        mes_peor = MESES.get(mes_max, str(mes_max))
    else:
        mes_peor = None

    resultado = {
        "variable":           variable,
        "umbral":             threshold,
        "temporal_unit":      temporal_unit,
        "n_total":            int(n_total),
        "n_excedencias":      int(n_excede),
        "pct_excedencias":    round(n_excede / n_total * 100, 1) if n_total > 0 else 0,
        "max_concentracion":  round(float(serie.dropna().max()), 2),
        "promedio_excedencia": round(float(serie[excedencias].mean()), 2) if n_excede > 0 else None,
        "mes_mayor_excedencia": mes_peor,
    }
    logger.info(
        f"Excedencias de '{variable}' (>{threshold}): "
        f"{n_excede}/{n_total} ({resultado['pct_excedencias']}%)"
    )
    return resultado


# ─────────────────────────────────────────────
# 6. GENERAR TODOS LOS GRÁFICOS
# ─────────────────────────────────────────────

def plot_all_descriptive(
    df: pd.DataFrame,
    variable: str,
    output_dir: str | Path | None = None,
    station_col: str | None = None,
    dpi: int = 150,
) -> list[Path]:
    """
    Genera y guarda todos los gráficos descriptivos como PNG.

    Gráficos generados:
    - boxplot comparativo por estación
    - violín
    - histograma con curva KDE
    - serie temporal
    - patrón diurno

    Parámetros
    ----------
    df : pd.DataFrame
    variable : str
    output_dir : str | Path | None
        Directorio de salida. Default: outputs/figures/
    station_col : str | None
        Columna de ID de estación.
    dpi : int
        Resolución de las imágenes.

    Retorna
    -------
    list[Path]: Rutas de los archivos generados.
    """
    if not HAS_MATPLOTLIB:
        logger.error("matplotlib no disponible — instalar: pip install matplotlib seaborn")
        return []

    if output_dir is None:
        output_dir = FIGURES_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if station_col is None:
        station_col = next((c for c in ["station_id", "location_id", "station_name"]
                            if c in df.columns), None)

    archivos = []
    var_safe = variable.replace(".", "").replace(" ", "_")

    # ── Boxplot por estación ──
    if station_col and station_col in df.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        df.boxplot(column=variable, by=station_col, ax=ax, flierprops={"marker": "."})
        ax.set_title(f"Distribución de {variable} por estación")
        ax.set_xlabel("Estación")
        ax.set_ylabel(f"{variable} (µg/m³)")
        plt.suptitle("")
        plt.tight_layout()
        p = output_dir / f"boxplot_{var_safe}.png"
        fig.savefig(p, dpi=dpi)
        plt.close(fig)
        archivos.append(p)
        logger.info(f"Guardado: {p.name}")

    # ── Violín ──
    if station_col and station_col in df.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        data_violin = [g[variable].dropna().values
                       for _, g in df.groupby(station_col)
                       if g[variable].notna().sum() > 1]
        labels_violin = [str(s) for s, g in df.groupby(station_col)
                         if g[variable].notna().sum() > 1]
        if data_violin:
            vp = ax.violinplot(data_violin, showmedians=True)
            ax.set_xticks(range(1, len(labels_violin) + 1))
            ax.set_xticklabels(labels_violin, rotation=45, ha="right")
        ax.set_title(f"Distribución (violín) de {variable}")
        ax.set_ylabel(f"{variable} (µg/m³)")
        plt.tight_layout()
        p = output_dir / f"violin_{var_safe}.png"
        fig.savefig(p, dpi=dpi)
        plt.close(fig)
        archivos.append(p)
        logger.info(f"Guardado: {p.name}")

    # ── Histograma + KDE ──
    fig, ax = plt.subplots(figsize=(8, 5))
    serie = df[variable].dropna()
    ax.hist(serie, bins=40, density=True, alpha=0.6, color="#1f77b4", label="Histograma")
    try:
        kde = stats.gaussian_kde(serie)
        x_kde = np.linspace(serie.min(), serie.max(), 200)
        ax.plot(x_kde, kde(x_kde), "r-", lw=2, label="KDE")
    except Exception:
        pass

    # Línea umbral OMS
    guidelines = WHO_GUIDELINES_2021.get(variable.replace(".", ""), {})
    umbral = guidelines.get("24h") or guidelines.get("8h")
    if umbral:
        ax.axvline(umbral, color="orange", ls="--", lw=1.5, label=f"Umbral OMS {umbral}")

    ax.set_title(f"Histograma de {variable}")
    ax.set_xlabel(f"{variable} (µg/m³)")
    ax.set_ylabel("Densidad")
    ax.legend()
    plt.tight_layout()
    p = output_dir / f"histograma_{var_safe}.png"
    fig.savefig(p, dpi=dpi)
    plt.close(fig)
    archivos.append(p)
    logger.info(f"Guardado: {p.name}")

    # ── Serie temporal ──
    if "datetime" in df.columns:
        fig, ax = plt.subplots(figsize=(14, 4))
        df_sorted = df.sort_values("datetime")
        if station_col and station_col in df.columns:
            for i, (station, grupo) in enumerate(df_sorted.groupby(station_col)):
                color = STATION_COLORS[i % len(STATION_COLORS)]
                ax.plot(grupo["datetime"], grupo[variable],
                        alpha=0.7, lw=0.8, color=color, label=str(station))
            ax.legend(fontsize=7, ncol=3)
        else:
            ax.plot(df_sorted["datetime"], df_sorted[variable], alpha=0.8, lw=0.8, color="#1f77b4")

        if umbral:
            ax.axhline(umbral, color="orange", ls="--", lw=1.5, label=f"Umbral OMS {umbral}")
        ax.set_title(f"Serie temporal de {variable}")
        ax.set_xlabel("Fecha")
        ax.set_ylabel(f"{variable} (µg/m³)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.xticks(rotation=45)
        plt.tight_layout()
        p = output_dir / f"serie_temporal_{var_safe}.png"
        fig.savefig(p, dpi=dpi)
        plt.close(fig)
        archivos.append(p)
        logger.info(f"Guardado: {p.name}")

    # ── Patrón diurno ──
    if "datetime" in df.columns:
        try:
            patron = diurnal_pattern(df, variable)
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.fill_between(patron["hora"], patron["ci_lower"], patron["ci_upper"],
                           alpha=0.3, color="#1f77b4")
            ax.plot(patron["hora"], patron["promedio"], "o-", color="#1f77b4",
                   lw=2, ms=4, label="Promedio ± IC 95%")
            if umbral:
                ax.axhline(umbral, color="orange", ls="--", lw=1.5, label=f"Umbral OMS {umbral}")
            ax.set_title(f"Patrón diurno de {variable}")
            ax.set_xlabel("Hora del día (UTC)")
            ax.set_ylabel(f"{variable} (µg/m³)")
            ax.set_xticks(range(0, 24, 3))
            ax.legend()
            plt.tight_layout()
            p = output_dir / f"patron_diurno_{var_safe}.png"
            fig.savefig(p, dpi=dpi)
            plt.close(fig)
            archivos.append(p)
            logger.info(f"Guardado: {p.name}")
        except Exception as e:
            logger.warning(f"No se pudo generar patrón diurno: {e}")

    logger.info(f"Total gráficos generados: {len(archivos)}")
    return archivos


# ─────────────────────────────────────────────
# EJEMPLO DE USO
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from ingestion.load_stations import generate_synthetic_data

    df = generate_synthetic_data(n_stations=4, days=90)
    df["datetime"] = pd.to_datetime(df["fecha"], format="%d/%m/%Y %H:%M")

    print("=== Estadísticas descriptivas de PM2.5 ===")
    stats_df = full_descriptive(df, "PM2.5")
    print(stats_df.to_string())

    print("\n=== Descomposición temporal ===")
    decopm = temporal_decomposition(df, "PM2.5")
    print("Por hora (primeras 6):")
    print(decopm["por_hora"].head(6))

    print("\n=== Tendencia anual ===")
    tendencia = annual_trend(df, "PM2.5")
    print(tendencia)

    print("\n=== Excedencias OMS ===")
    exc = exceedance_stats(df, "PM2.5")
    print(exc)

    print("\n=== Generando gráficos ===")
    archivos = plot_all_descriptive(df, "PM2.5", station_col="station_id")
    print(f"Gráficos guardados: {[str(a) for a in archivos]}")
