"""
correlations.py — Análisis de correlaciones para datos de calidad del aire
Incluye: matrices Pearson/Spearman, correlación cruzada con lag,
correlación parcial, correlación meteorológica y visualización.

Autor: Rodrigo, Ingeniero Ambiental, Universidad Anáhuac Cancún
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

try:
    from statsmodels.stats.stattools import medcouple_1d
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools import add_constant
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
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


# ─────────────────────────────────────────────
# 1. MATRICES DE CORRELACIÓN
# ─────────────────────────────────────────────

def correlation_matrix(
    df: pd.DataFrame,
    variables: list[str],
    method: str = "both",
    min_samples: int = 30,
) -> dict:
    """
    Calcula matrices de correlación Pearson y/o Spearman con p-values.

    Parámetros
    ----------
    df : pd.DataFrame
    variables : list[str]
        Columnas numéricas a correlacionar.
    method : str
        'pearson', 'spearman' o 'both' (calcula ambas).
    min_samples : int
        Mínimo de pares válidos para calcular correlación.

    Retorna
    -------
    dict con claves según method:
        'pearson': {'r': DataFrame, 'p': DataFrame}
        'spearman': {'rho': DataFrame, 'p': DataFrame}
    """
    # Filtrar a columnas disponibles
    cols = [v for v in variables if v in df.columns]
    if len(cols) < 2:
        raise ValueError(f"Se necesitan al menos 2 variables. Disponibles: {cols}")

    df_sub = df[cols].copy()
    n_vars = len(cols)
    results = {}

    def _corr_matrix(method_fn: str):
        """Calcula matriz de correlación y p-values."""
        r_matrix = np.ones((n_vars, n_vars))
        p_matrix = np.zeros((n_vars, n_vars))

        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                x = df_sub[cols[i]].values
                y = df_sub[cols[j]].values
                # Solo usar pares donde ambos tienen valor
                mask = ~(np.isnan(x) | np.isnan(y))
                n = mask.sum()

                if n < min_samples:
                    r_matrix[i, j] = r_matrix[j, i] = np.nan
                    p_matrix[i, j] = p_matrix[j, i] = np.nan
                    continue

                if method_fn == "pearson":
                    r, p = stats.pearsonr(x[mask], y[mask])
                else:
                    r, p = stats.spearmanr(x[mask], y[mask])

                r_matrix[i, j] = r_matrix[j, i] = round(r, 4)
                p_matrix[i, j] = p_matrix[j, i] = round(p, 4)

        r_df = pd.DataFrame(r_matrix, index=cols, columns=cols)
        p_df = pd.DataFrame(p_matrix, index=cols, columns=cols)
        return r_df, p_df

    if method in ("pearson", "both"):
        r_df, p_df = _corr_matrix("pearson")
        results["pearson"] = {"r": r_df, "p": p_df}
        logger.info(f"Correlación Pearson calculada: {n_vars}×{n_vars}")

    if method in ("spearman", "both"):
        rho_df, p_df = _corr_matrix("spearman")
        results["spearman"] = {"rho": rho_df, "p": p_df}
        logger.info(f"Correlación Spearman calculada: {n_vars}×{n_vars}")

    return results


# ─────────────────────────────────────────────
# 2. CORRELACIÓN CRUZADA CON LAG
# ─────────────────────────────────────────────

def lag_correlation(
    df: pd.DataFrame,
    var1: str,
    var2: str,
    max_lag: int = 48,
    freq_resample: str = "h",
) -> pd.DataFrame:
    """
    Calcula la correlación cruzada entre dos variables con diferentes lags.
    Útil para identificar efectos retardados (ej: viento → PM2.5 horas después).

    Parámetros
    ----------
    df : pd.DataFrame
    var1 : str
        Variable "causa" (ej: velocidad del viento).
    var2 : str
        Variable "efecto" (ej: PM2.5).
    max_lag : int
        Máximo lag a evaluar (en períodos de freq_resample).
    freq_resample : str
        Frecuencia de resample.

    Retorna
    -------
    pd.DataFrame con columnas: lag, pearson_r, spearman_rho, p_value,
                               significativo (p < 0.05)
    """
    if var1 not in df.columns or var2 not in df.columns:
        raise KeyError(f"Variables no encontradas: {var1}, {var2}")

    # Preparar series con índice temporal
    if "datetime" in df.columns:
        df_temp = df.set_index(pd.to_datetime(df["datetime"]))
        s1 = df_temp[var1].resample(freq_resample).mean().interpolate()
        s2 = df_temp[var2].resample(freq_resample).mean().interpolate()
    else:
        s1 = df[var1]
        s2 = df[var2]

    resultados = []
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            x = s1.iloc[:lag].values
            y = s2.iloc[-lag:].values
        elif lag == 0:
            x = s1.values
            y = s2.values
        else:
            x = s1.iloc[lag:].values
            y = s2.iloc[:-lag].values

        # Filtrar NaN
        mask = ~(np.isnan(x) | np.isnan(y))
        if mask.sum() < 10:
            continue

        r, p_pearson = stats.pearsonr(x[mask], y[mask])
        rho, _ = stats.spearmanr(x[mask], y[mask])

        resultados.append({
            "lag":          lag,
            "pearson_r":    round(r, 4),
            "spearman_rho": round(rho, 4),
            "p_value":      round(p_pearson, 4),
            "significativo": p_pearson < 0.05,
            "n":            int(mask.sum()),
        })

    result_df = pd.DataFrame(resultados)

    if len(result_df) > 0:
        lag_optimo = result_df.loc[result_df["pearson_r"].abs().idxmax(), "lag"]
        logger.info(
            f"Correlación cruzada {var1}↔{var2}: "
            f"lag óptimo={lag_optimo} ({freq_resample}), "
            f"r={result_df.loc[result_df['lag']==lag_optimo, 'pearson_r'].values[0]:.3f}"
        )

    return result_df


# ─────────────────────────────────────────────
# 3. CORRELACIÓN PARCIAL
# ─────────────────────────────────────────────

def partial_correlation(
    df: pd.DataFrame,
    var1: str,
    var2: str,
    controls: list[str],
) -> dict:
    """
    Calcula la correlación parcial entre var1 y var2 controlando por `controls`.
    Usa residuos de regresión OLS para remover el efecto de las variables de control.

    Parámetros
    ----------
    df : pd.DataFrame
    var1 : str
        Primera variable.
    var2 : str
        Segunda variable.
    controls : list[str]
        Variables de control (confounders).

    Retorna
    -------
    dict con: r_parcial, p_value, n, variables_control
    """
    all_vars = [var1, var2] + controls
    missing = [v for v in all_vars if v not in df.columns]
    if missing:
        raise KeyError(f"Variables no encontradas: {missing}")

    # Eliminar filas con NaN en cualquier variable relevante
    df_sub = df[all_vars].dropna()
    if len(df_sub) < 10:
        return {"error": "Datos insuficientes después de eliminar NaN"}

    def _residuals(y_col: str, x_cols: list[str]) -> np.ndarray:
        """Calcula residuos de regresión de y sobre x."""
        y = df_sub[y_col].values
        X = df_sub[x_cols].values
        if HAS_STATSMODELS:
            X_const = add_constant(X)
            modelo = OLS(y, X_const).fit()
            return modelo.resid
        else:
            # Fallback: regresión con numpy
            X_const = np.column_stack([np.ones(len(X)), X])
            beta = np.linalg.lstsq(X_const, y, rcond=None)[0]
            return y - X_const @ beta

    resid1 = _residuals(var1, controls)
    resid2 = _residuals(var2, controls)

    r_parcial, p_value = stats.pearsonr(resid1, resid2)

    resultado = {
        "var1":             var1,
        "var2":             var2,
        "variables_control": controls,
        "r_parcial":        round(r_parcial, 4),
        "p_value":          round(p_value, 4),
        "significativo":    p_value < 0.05,
        "n":                len(df_sub),
    }
    logger.info(
        f"Correlación parcial {var1}↔{var2} | controles={controls}: "
        f"r={r_parcial:.4f}, p={p_value:.4f}"
    )
    return resultado


# ─────────────────────────────────────────────
# 4. CORRELACIÓN CON VARIABLES METEOROLÓGICAS
# ─────────────────────────────────────────────

def meteorological_correlation(
    df: pd.DataFrame,
    pollutant_col: str,
    meteo_cols: list[str],
    method: str = "both",
) -> pd.DataFrame:
    """
    Calcula la correlación sistemática entre un contaminante y variables
    meteorológicas (temperatura, humedad, velocidad de viento, presión, etc.).

    Parámetros
    ----------
    df : pd.DataFrame
    pollutant_col : str
        Columna del contaminante (ej: 'PM2.5').
    meteo_cols : list[str]
        Lista de variables meteorológicas (ej: ['temperatura', 'humedad', 'viento']).
    method : str
        'pearson', 'spearman' o 'both'.

    Retorna
    -------
    pd.DataFrame con columnas: variable_meteo, pearson_r, spearman_rho,
                               p_pearson, p_spearman, significativo, n
    """
    cols_validas = [c for c in meteo_cols if c in df.columns]
    missing = [c for c in meteo_cols if c not in df.columns]
    if missing:
        logger.warning(f"Variables meteorológicas no encontradas: {missing}")
    if not cols_validas:
        raise ValueError("Ninguna variable meteorológica encontrada en el DataFrame")
    if pollutant_col not in df.columns:
        raise KeyError(f"Contaminante '{pollutant_col}' no encontrado")

    resultados = []
    for meteo in cols_validas:
        sub = df[[pollutant_col, meteo]].dropna()
        n = len(sub)
        if n < 10:
            continue

        x = sub[pollutant_col].values
        y = sub[meteo].values

        r, p_r     = stats.pearsonr(x, y)
        rho, p_rho = stats.spearmanr(x, y)

        resultados.append({
            "variable_meteo":  meteo,
            "pearson_r":       round(r, 4),
            "spearman_rho":    round(rho, 4),
            "p_pearson":       round(p_r, 4),
            "p_spearman":      round(p_rho, 4),
            "significativo":   (p_r < 0.05),
            "n":               n,
            "interpretacion":  _interpret_correlation(r, p_r),
        })

    result_df = pd.DataFrame(resultados).sort_values("pearson_r", key=abs, ascending=False)
    logger.info(
        f"Correlación meteorológica de '{pollutant_col}': "
        f"{len(result_df)} variables analizadas"
    )
    return result_df


def _interpret_correlation(r: float, p: float) -> str:
    """Interpreta la magnitud y significancia de una correlación."""
    sig = "significativa" if p < 0.05 else "no significativa"
    if abs(r) < 0.2:
        fuerza = "muy débil"
    elif abs(r) < 0.4:
        fuerza = "débil"
    elif abs(r) < 0.6:
        fuerza = "moderada"
    elif abs(r) < 0.8:
        fuerza = "fuerte"
    else:
        fuerza = "muy fuerte"
    direccion = "positiva" if r > 0 else "negativa"
    return f"Correlación {fuerza} {direccion}, {sig} (r={r:.2f}, p={p:.3f})"


# ─────────────────────────────────────────────
# 5. HEATMAP DE CORRELACIONES
# ─────────────────────────────────────────────

def plot_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    p_matrix: pd.DataFrame | None = None,
    output_path: str | Path | None = None,
    title: str = "Matriz de Correlaciones",
    method: str = "pearson",
    dpi: int = 150,
) -> Path | None:
    """
    Genera un heatmap de correlaciones con asteriscos para significancia.

    Parámetros
    ----------
    corr_matrix : pd.DataFrame
        Matriz de correlaciones (simétrica, valores en [-1, 1]).
    p_matrix : pd.DataFrame | None
        Matriz de p-values. Si se proporciona, añade asteriscos:
        *** p<0.001, ** p<0.01, * p<0.05
    output_path : str | Path | None
        Ruta de guardado. Default: outputs/figures/heatmap_correlaciones.png
    title : str
        Título del gráfico.
    method : str
        'pearson' o 'spearman' (para la etiqueta).
    dpi : int

    Retorna
    -------
    Path al archivo guardado, o None si falló.
    """
    if not HAS_MATPLOTLIB:
        logger.error("matplotlib/seaborn no disponibles")
        return None

    if output_path is None:
        output_path = FIGURES_DIR / "heatmap_correlaciones.png"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(corr_matrix)
    figsize = max(6, n * 0.9)
    fig, ax = plt.subplots(figsize=(figsize, figsize * 0.85))

    # Generar anotaciones con asteriscos de significancia
    annot = corr_matrix.round(2).astype(str)
    if p_matrix is not None:
        for i in range(n):
            for j in range(n):
                p = p_matrix.iloc[i, j]
                r = corr_matrix.iloc[i, j]
                if i == j:
                    annot.iloc[i, j] = "1.0"
                    continue
                if pd.isna(p):
                    annot.iloc[i, j] = "n/d"
                    continue
                sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
                annot.iloc[i, j] = f"{r:.2f}{sig}"

    # Máscara para triángulo superior (evitar duplicar info)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=annot,
        fmt="",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": f"Correlación {method.capitalize()}"},
        ax=ax,
    )

    ax.set_title(f"{title}\n(*** p<0.001, ** p<0.01, * p<0.05)", fontsize=11)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    logger.info(f"Heatmap guardado: {output_path}")
    return output_path


def plot_lag_correlation(
    lag_df: pd.DataFrame,
    var1: str,
    var2: str,
    output_path: str | Path | None = None,
    dpi: int = 150,
) -> Path | None:
    """
    Gráfico de correlación cruzada con lag (tipo ACF).
    """
    if not HAS_MATPLOTLIB:
        return None

    if output_path is None:
        output_path = FIGURES_DIR / f"lag_corr_{var1}_{var2}.png".replace(".", "")
    output_path = Path(output_path)

    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ["#d62728" if sig else "#1f77b4" for sig in lag_df["significativo"]]
    ax.bar(lag_df["lag"], lag_df["pearson_r"], color=colors, alpha=0.7, width=0.8)
    ax.axhline(0, color="black", lw=0.8)

    # Líneas de significancia (±1.96/√n)
    if "n" in lag_df.columns and lag_df["n"].mean() > 0:
        n_mean = lag_df["n"].mean()
        ci_line = 1.96 / np.sqrt(n_mean)
        ax.axhline(ci_line, color="orange", ls="--", lw=1.2, label=f"IC 95% (±{ci_line:.2f})")
        ax.axhline(-ci_line, color="orange", ls="--", lw=1.2)

    ax.set_xlabel(f"Lag (significativo = rojo)")
    ax.set_ylabel("Correlación de Pearson r")
    ax.set_title(f"Correlación cruzada con lag: {var1} → {var2}")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    logger.info(f"Gráfico lag guardado: {output_path}")
    return output_path


# ─────────────────────────────────────────────
# EJEMPLO DE USO
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from ingestion.load_stations import generate_synthetic_data

    df = generate_synthetic_data(n_stations=3, days=60)
    df["datetime"] = pd.to_datetime(df["fecha"], format="%d/%m/%Y %H:%M")

    # Añadir variables meteorológicas simuladas
    rng = np.random.default_rng(42)
    df["temperatura"] = 25 + rng.normal(0, 5, len(df))
    df["humedad"]     = 70 + rng.normal(0, 15, len(df))
    df["viento"]      = np.abs(rng.normal(3, 2, len(df)))

    print("=== Matriz de correlación ===")
    corr_result = correlation_matrix(df, ["PM2.5", "PM10", "O3", "NO2", "temperatura", "humedad"])
    print("Pearson:")
    print(corr_result["pearson"]["r"].round(3).to_string())

    print("\n=== Correlación cruzada con lag ===")
    lag_df = lag_correlation(df, "viento", "PM2.5", max_lag=24)
    lag_optimo = lag_df.loc[lag_df["pearson_r"].abs().idxmax()]
    print(f"Lag óptimo: {lag_optimo['lag']}h, r={lag_optimo['pearson_r']:.3f}")

    print("\n=== Correlación meteorológica ===")
    meteo = meteorological_correlation(df, "PM2.5", ["temperatura", "humedad", "viento"])
    print(meteo.to_string())

    print("\n=== Generando heatmap ===")
    p_r = corr_result["pearson"]["p"]
    path = plot_correlation_heatmap(
        corr_result["pearson"]["r"],
        p_matrix=p_r,
        title="Correlaciones PM2.5 y Meteorología",
    )
    print(f"Guardado en: {path}")
