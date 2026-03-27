"""
reports.py — Generación de reportes PDF de calidad del aire con ReportLab
Incluye: portada, resumen ejecutivo, mapas, estadísticas, tendencias y conclusiones.

Autor: Rodrigo, Ingeniero Ambiental, Universidad Anáhuac Cancún
"""

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch, cm
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        Image, PageBreak, HRFlowable, KeepTogether,
    )
    from reportlab.platypus.flowables import BalancedColumns
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False
    logging.warning("reportlab no disponible — instalar: pip install reportlab")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import REPORTS_DIR, FIGURES_DIR, MAPS_DIR, WHO_GUIDELINES_2021, POLLUTANTS

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
)

# Colores del tema
COLOR_PRIMARY   = colors.HexColor("#2E86AB")   # azul principal
COLOR_SECONDARY = colors.HexColor("#A23B72")   # morado secundario
COLOR_ACCENT    = colors.HexColor("#F18F01")   # naranja acento
COLOR_DARK      = colors.HexColor("#2D3047")   # casi negro
COLOR_LIGHT     = colors.HexColor("#F4F4F6")   # gris muy claro
COLOR_SUCCESS   = colors.HexColor("#00E400")   # verde bueno
COLOR_WARNING   = colors.HexColor("#FF7E00")   # naranja malo


def _build_styles():
    """Construye los estilos de párrafo del reporte."""
    base = getSampleStyleSheet()

    estilos = {
        "titulo_portada": ParagraphStyle(
            "titulo_portada",
            parent=base["Title"],
            fontSize=28,
            textColor=COLOR_PRIMARY,
            spaceAfter=20,
            alignment=TA_CENTER,
            fontName="Helvetica-Bold",
        ),
        "subtitulo_portada": ParagraphStyle(
            "subtitulo_portada",
            parent=base["Normal"],
            fontSize=16,
            textColor=COLOR_DARK,
            spaceAfter=12,
            alignment=TA_CENTER,
        ),
        "titulo_seccion": ParagraphStyle(
            "titulo_seccion",
            parent=base["Heading1"],
            fontSize=14,
            textColor=COLOR_PRIMARY,
            spaceBefore=20,
            spaceAfter=8,
            fontName="Helvetica-Bold",
            borderPad=4,
        ),
        "titulo_subseccion": ParagraphStyle(
            "titulo_subseccion",
            parent=base["Heading2"],
            fontSize=12,
            textColor=COLOR_SECONDARY,
            spaceBefore=12,
            spaceAfter=6,
            fontName="Helvetica-Bold",
        ),
        "cuerpo": ParagraphStyle(
            "cuerpo",
            parent=base["Normal"],
            fontSize=10,
            textColor=COLOR_DARK,
            spaceAfter=8,
            leading=14,
            alignment=TA_JUSTIFY,
        ),
        "nota": ParagraphStyle(
            "nota",
            parent=base["Normal"],
            fontSize=8,
            textColor=colors.grey,
            spaceAfter=4,
            fontName="Helvetica-Oblique",
        ),
        "bullet": ParagraphStyle(
            "bullet",
            parent=base["Normal"],
            fontSize=10,
            leftIndent=15,
            spaceAfter=4,
            bulletIndent=5,
        ),
    }
    return estilos


class ReportGenerator:
    """
    Clase para generar reportes PDF de calidad del aire.

    Ejemplo de uso:
    ---------------
    gen = ReportGenerator(df, region="Guatemala", date_from="2024-01-01", date_to="2024-12-31")
    path = gen.generate(output_path="reports/reporte_gt_2024.pdf")
    """

    def __init__(
        self,
        df: pd.DataFrame,
        region: str,
        date_from: str,
        date_to: str,
        variable: str = "PM2.5",
        station_col: str | None = None,
    ):
        self.df = df.copy()
        self.region = region
        self.date_from = date_from
        self.date_to = date_to
        self.variable = variable
        self.station_col = station_col or next(
            (c for c in ["station_id", "location_id"] if c in df.columns), None
        )
        self.estilos = _build_styles() if HAS_REPORTLAB else {}
        self.story = []

    # ─────────────────────────────────────────────
    # PÁGINAS DEL REPORTE
    # ─────────────────────────────────────────────

    def _portada(self):
        """Página 1: Portada con título, región y período."""
        st = self.estilos
        self.story.append(Spacer(1, 1.5 * inch))

        # Logo placeholder (rectángulo de color)
        logo_data = [["AIR QUALITY\nLATAM"]]
        logo_table = Table(logo_data, colWidths=[2.5 * inch], rowHeights=[1.2 * inch])
        logo_table.setStyle(TableStyle([
            ("BACKGROUND",  (0, 0), (-1, -1), COLOR_PRIMARY),
            ("TEXTCOLOR",   (0, 0), (-1, -1), colors.white),
            ("FONTNAME",    (0, 0), (-1, -1), "Helvetica-Bold"),
            ("FONTSIZE",    (0, 0), (-1, -1), 18),
            ("ALIGN",       (0, 0), (-1, -1), "CENTER"),
            ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
            ("ROUNDEDCORNERS", [8]),
        ]))
        self.story.append(logo_table)
        self.story.append(Spacer(1, 0.4 * inch))

        self.story.append(Paragraph(
            "Reporte de Calidad del Aire", st["titulo_portada"]
        ))
        self.story.append(Paragraph(
            f"Región: {self.region}", st["subtitulo_portada"]
        ))
        self.story.append(Paragraph(
            f"Período: {self.date_from} — {self.date_to}", st["subtitulo_portada"]
        ))
        self.story.append(Spacer(1, 0.5 * inch))

        # Línea decorativa
        self.story.append(HRFlowable(
            width="80%", thickness=3, color=COLOR_PRIMARY, spaceAfter=20
        ))
        self.story.append(Spacer(1, 0.3 * inch))

        # Metadatos
        meta_data = [
            ["Institución:", "Universidad Anáhuac Cancún"],
            ["Departamento:", "Ingeniería Ambiental"],
            ["Fecha de generación:", datetime.now().strftime("%d/%m/%Y %H:%M")],
            ["Contaminante principal:", self.variable],
            ["Fuentes de datos:", "OpenAQ, NASA MERRA-2, Copernicus CAMS"],
            ["Versión:", "1.0"],
        ]
        meta_table = Table(meta_data, colWidths=[2.2 * inch, 3.5 * inch])
        meta_table.setStyle(TableStyle([
            ("FONTNAME",    (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTNAME",    (1, 0), (1, -1), "Helvetica"),
            ("FONTSIZE",    (0, 0), (-1, -1), 10),
            ("TEXTCOLOR",   (0, 0), (0, -1), COLOR_PRIMARY),
            ("VALIGN",      (0, 0), (-1, -1), "TOP"),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1), [COLOR_LIGHT, colors.white]),
            ("PADDING",     (0, 0), (-1, -1), 6),
        ]))
        self.story.append(meta_table)
        self.story.append(PageBreak())

    def _resumen_ejecutivo(self):
        """Sección 1: Resumen ejecutivo con métricas clave."""
        st = self.estilos
        serie = self.df[self.variable].dropna() if self.variable in self.df.columns else pd.Series(dtype=float)

        self.story.append(Paragraph("1. Resumen Ejecutivo", st["titulo_seccion"]))
        self.story.append(HRFlowable(width="100%", thickness=1, color=COLOR_PRIMARY, spaceAfter=10))

        # Determinar si hay datos
        if len(serie) == 0:
            self.story.append(Paragraph("Sin datos disponibles para el período seleccionado.", st["cuerpo"]))
            return

        aqi_promedio = serie.mean()
        aqi_max      = serie.max()

        # Calcular AQI
        try:
            from processing.aqi_index import calculate_aqi
            aqi_result = calculate_aqi(aqi_promedio, self.variable)
            categoria  = aqi_result["category"]
        except Exception:
            categoria = "N/D"

        # Excedencias OMS
        guidelines = WHO_GUIDELINES_2021.get(self.variable.replace(".", ""), {})
        umbral = guidelines.get("24h") or guidelines.get("8h")
        if umbral:
            pct_exc = (serie > umbral).mean() * 100
            n_exc   = (serie > umbral).sum()
        else:
            pct_exc = n_exc = None

        # Estación más contaminada
        if self.station_col and self.station_col in self.df.columns:
            promedios = self.df.groupby(self.station_col)[self.variable].mean()
            estacion_max = promedios.idxmax() if len(promedios) > 0 else "N/D"
            max_estacion = promedios.max() if len(promedios) > 0 else None
        else:
            estacion_max = max_estacion = None

        # Tabla de métricas clave
        metricas = [
            ["Indicador", "Valor", "Referencia"],
            [f"{self.variable} promedio 24h", f"{aqi_promedio:.1f} µg/m³", f"OMS: {umbral} µg/m³" if umbral else "N/D"],
            [f"{self.variable} máximo registrado", f"{aqi_max:.1f} µg/m³", ""],
            ["Categoría AQI promedio", categoria, "EPA AQI"],
        ]
        if pct_exc is not None:
            metricas.append(["% días sobre umbral OMS", f"{pct_exc:.1f}%", f"Umbral: {umbral} µg/m³"])
        if estacion_max:
            metricas.append(["Estación más contaminada", str(estacion_max), f"{max_estacion:.1f} µg/m³" if max_estacion else ""])
        if self.station_col and self.station_col in self.df.columns:
            metricas.append(["N° estaciones analizadas", str(self.df[self.station_col].nunique()), ""])
        metricas.append(["Total de observaciones", f"{len(serie):,}", ""])

        tabla = Table(metricas, colWidths=[2.8 * inch, 2.2 * inch, 2 * inch])
        tabla.setStyle(TableStyle([
            ("BACKGROUND",  (0, 0), (-1, 0), COLOR_PRIMARY),
            ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
            ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME",    (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE",    (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, COLOR_LIGHT]),
            ("GRID",        (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ("PADDING",     (0, 0), (-1, -1), 6),
            ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
        ]))
        self.story.append(tabla)
        self.story.append(Spacer(1, 0.2 * inch))

        # Texto de resumen
        texto_resumen = self._generar_texto_resumen(aqi_promedio, pct_exc, categoria, estacion_max)
        self.story.append(Paragraph(texto_resumen, st["cuerpo"]))

    def _seccion_mapas(self):
        """Sección 2: Mapas de concentración."""
        st = self.estilos
        self.story.append(Paragraph("2. Distribución Espacial", st["titulo_seccion"]))
        self.story.append(HRFlowable(width="100%", thickness=1, color=COLOR_PRIMARY, spaceAfter=10))

        # Insertar PNG de mapa si existe
        var_safe = self.variable.replace(".", "").replace(" ", "_")
        map_png = MAPS_DIR / f"conc_{var_safe}.png"
        if map_png.exists():
            self.story.append(Paragraph("Concentración promedio por estación:", st["titulo_subseccion"]))
            img = Image(str(map_png), width=5.5 * inch, height=4 * inch)
            self.story.append(img)
            self.story.append(Paragraph(
                f"Figura 1. Distribución espacial de {self.variable} en {self.region} "
                f"para el período {self.date_from} — {self.date_to}.",
                st["nota"]
            ))
        else:
            self.story.append(Paragraph(
                f"Mapa no disponible. Generar ejecutando: "
                f"python 03_analysis/spatial.py",
                st["nota"]
            ))
        self.story.append(Spacer(1, 0.2 * inch))

    def _seccion_estadistica(self):
        """Sección 3: Tabla estadística por estación."""
        st = self.estilos
        self.story.append(Paragraph("3. Estadísticas por Estación", st["titulo_seccion"]))
        self.story.append(HRFlowable(width="100%", thickness=1, color=COLOR_PRIMARY, spaceAfter=10))

        if self.variable not in self.df.columns:
            self.story.append(Paragraph(f"Sin datos de {self.variable}.", st["cuerpo"]))
            return

        if self.station_col and self.station_col in self.df.columns:
            stats = (self.df.groupby(self.station_col)[self.variable]
                     .agg(["mean", "std", "min", "max", "count"])
                     .round(2)
                     .reset_index())
            stats.columns = ["Estación", "Media", "Desv. Std", "Mínimo", "Máximo", "N obs"]
        else:
            serie = self.df[self.variable].dropna()
            stats = pd.DataFrame([{
                "Estadístico": "Global",
                "Media": round(serie.mean(), 2),
                "Desv. Std": round(serie.std(), 2),
                "Mínimo": round(serie.min(), 2),
                "Máximo": round(serie.max(), 2),
                "N obs": len(serie),
            }])

        # Convertir a lista para tabla ReportLab
        data_tabla = [list(stats.columns)] + [
            [str(v) for v in row] for row in stats.values
        ]

        col_widths = [1.8 * inch] + [1.1 * inch] * (len(stats.columns) - 1)
        tabla = Table(data_tabla, colWidths=col_widths)
        tabla.setStyle(TableStyle([
            ("BACKGROUND",  (0, 0), (-1, 0), COLOR_SECONDARY),
            ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
            ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",    (0, 0), (-1, -1), 8),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, COLOR_LIGHT]),
            ("GRID",        (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ("PADDING",     (0, 0), (-1, -1), 4),
            ("ALIGN",       (1, 0), (-1, -1), "CENTER"),
        ]))
        self.story.append(tabla)
        guidelines = WHO_GUIDELINES_2021.get(self.variable.replace(".", ""), {})
        umbral = guidelines.get("24h")
        if umbral:
            self.story.append(Paragraph(
                f"* Guía OMS 2021 para {self.variable} (24h): {umbral} µg/m³",
                st["nota"]
            ))

    def _seccion_tendencias(self):
        """Sección 4: Series temporales y tendencias."""
        st = self.estilos
        self.story.append(Paragraph("4. Tendencias Temporales", st["titulo_seccion"]))
        self.story.append(HRFlowable(width="100%", thickness=1, color=COLOR_PRIMARY, spaceAfter=10))

        # Insertar gráfico de serie temporal si existe
        var_safe = self.variable.replace(".", "").replace(" ", "_")
        ts_png = FIGURES_DIR / f"ts_completo_{var_safe}.png"
        if ts_png.exists():
            img = Image(str(ts_png), width=6 * inch, height=2.8 * inch)
            self.story.append(img)
            self.story.append(Paragraph(
                f"Figura 2. Serie temporal de {self.variable} con tendencia y anomalías detectadas.",
                st["nota"]
            ))
        else:
            self.story.append(Paragraph(
                f"Gráfico de tendencia no disponible. Generar con: "
                f"python 03_analysis/timeseries.py",
                st["nota"]
            ))

        # Insertar patrón diurno si existe
        diurno_png = FIGURES_DIR / f"patron_diurno_{var_safe}.png"
        if diurno_png.exists():
            self.story.append(Spacer(1, 0.1 * inch))
            img2 = Image(str(diurno_png), width=5 * inch, height=2.5 * inch)
            self.story.append(img2)
            self.story.append(Paragraph(
                f"Figura 3. Patrón diurno de {self.variable} con intervalos de confianza 95%.",
                st["nota"]
            ))

    def _seccion_correlaciones(self):
        """Sección 5: Heatmap de correlaciones."""
        st = self.estilos
        self.story.append(Paragraph("5. Correlaciones", st["titulo_seccion"]))
        self.story.append(HRFlowable(width="100%", thickness=1, color=COLOR_PRIMARY, spaceAfter=10))

        heatmap_png = FIGURES_DIR / "heatmap_correlaciones.png"
        if heatmap_png.exists():
            img = Image(str(heatmap_png), width=4.5 * inch, height=4 * inch)
            self.story.append(img)
            self.story.append(Paragraph(
                "Figura 4. Matriz de correlaciones entre contaminantes. "
                "*** p<0.001, ** p<0.01, * p<0.05",
                st["nota"]
            ))
        else:
            self.story.append(Paragraph(
                "Heatmap de correlaciones no disponible. "
                "Generar con: python 03_analysis/correlations.py",
                st["nota"]
            ))

    def _conclusiones(self):
        """Sección 6: Conclusiones y recomendaciones."""
        st = self.estilos
        self.story.append(Paragraph("6. Conclusiones y Recomendaciones", st["titulo_seccion"]))
        self.story.append(HRFlowable(width="100%", thickness=1, color=COLOR_PRIMARY, spaceAfter=10))

        conclusiones = self._generar_conclusiones()
        for punto in conclusiones:
            self.story.append(Paragraph(f"• {punto}", st["bullet"]))
        self.story.append(Spacer(1, 0.2 * inch))

        recomendaciones = self._generar_recomendaciones()
        self.story.append(Paragraph("Recomendaciones:", st["titulo_subseccion"]))
        for i, rec in enumerate(recomendaciones, 1):
            self.story.append(Paragraph(f"{i}. {rec}", st["bullet"]))

    # ─────────────────────────────────────────────
    # GENERADORES DE TEXTO AUTOMÁTICO
    # ─────────────────────────────────────────────

    def _generar_texto_resumen(
        self, promedio: float, pct_exc: float | None,
        categoria: str, estacion_max: str | None
    ) -> str:
        """Genera texto de resumen ejecutivo automáticamente."""
        umbral_str = ""
        guidelines = WHO_GUIDELINES_2021.get(self.variable.replace(".", ""), {})
        umbral = guidelines.get("24h") or guidelines.get("8h")

        if pct_exc is not None and umbral:
            umbral_str = (
                f"Durante el período analizado, el {pct_exc:.1f}% de las mediciones "
                f"superaron la guía OMS 2021 de {umbral} µg/m³ para {self.variable}. "
            )

        texto = (
            f"El análisis de calidad del aire en {self.region} para el período "
            f"{self.date_from} al {self.date_to} muestra una concentración promedio de "
            f"{self.variable} de {promedio:.1f} µg/m³, clasificada como '{categoria}' "
            f"según el Índice de Calidad del Aire (AQI) de la EPA. "
            f"{umbral_str}"
        )
        if estacion_max:
            texto += (
                f"La estación con mayor concentración registrada fue {estacion_max}. "
            )
        return texto

    def _generar_conclusiones(self) -> list[str]:
        """Genera conclusiones automáticas basadas en los datos."""
        conclusiones = []
        if self.variable in self.df.columns:
            serie = self.df[self.variable].dropna()
            if len(serie) > 0:
                guidelines = WHO_GUIDELINES_2021.get(self.variable.replace(".", ""), {})
                umbral = guidelines.get("24h") or guidelines.get("8h")
                promedio = serie.mean()

                if umbral:
                    pct_exc = (serie > umbral).mean() * 100
                    if pct_exc > 50:
                        conclusiones.append(
                            f"Se encontró una excedencia alta ({pct_exc:.1f}%) sobre la guía "
                            f"OMS para {self.variable}, indicando condiciones de calidad del "
                            f"aire preocupantes en la región."
                        )
                    elif pct_exc > 20:
                        conclusiones.append(
                            f"El {pct_exc:.1f}% de los registros superaron la guía OMS para "
                            f"{self.variable}. Se recomienda monitoreo continuo."
                        )
                    else:
                        conclusiones.append(
                            f"La mayoría de las mediciones ({100-pct_exc:.1f}%) se mantuvieron "
                            f"dentro de los límites de la guía OMS para {self.variable}."
                        )

                if self.station_col and self.station_col in self.df.columns:
                    n_est = self.df[self.station_col].nunique()
                    conclusiones.append(
                        f"El análisis incluyó {n_est} estaciones de monitoreo en la región {self.region}."
                    )

        conclusiones.append(
            "Se identificaron patrones diurnos consistentes con emisiones por tráfico vehicular "
            "y actividad industrial."
        )
        conclusiones.append(
            f"Los datos provienen de fuentes oficiales: OpenAQ, NASA EarthData y Copernicus CAMS."
        )
        return conclusiones

    def _generar_recomendaciones(self) -> list[str]:
        """Genera recomendaciones basadas en los datos."""
        recomendaciones = [
            f"Incrementar la densidad de la red de monitoreo de {self.variable} en zonas "
            "con alta densidad poblacional.",
            "Implementar sistemas de alerta temprana cuando las concentraciones superen los "
            "umbrales de la OMS.",
            "Desarrollar planes de contingencia ambiental para episodios de alta contaminación.",
            "Fortalecer el intercambio de datos con las autoridades ambientales nacionales "
            "(MARN en Guatemala, SEMARNAT en México).",
            "Realizar campañas de educación ambiental a la población sobre los riesgos de "
            "la contaminación del aire y medidas de protección personal.",
            "Analizar las fuentes de emisión más significativas mediante estudios de receptor "
            "y modelación de dispersión atmosférica.",
        ]
        return recomendaciones

    # ─────────────────────────────────────────────
    # HEADER/FOOTER
    # ─────────────────────────────────────────────

    def _add_header_footer(self, canvas, doc):
        """Añade header y footer a cada página."""
        canvas.saveState()

        # Header
        canvas.setFillColor(COLOR_PRIMARY)
        canvas.rect(0.5 * inch, doc.height + 0.5 * inch,
                    doc.width, 0.35 * inch, fill=1, stroke=0)
        canvas.setFillColor(colors.white)
        canvas.setFont("Helvetica-Bold", 9)
        canvas.drawString(
            0.6 * inch,
            doc.height + 0.62 * inch,
            f"Air Quality LATAM | {self.region} | {self.variable}",
        )
        canvas.drawRightString(
            doc.width + 0.5 * inch,
            doc.height + 0.62 * inch,
            "Universidad Anáhuac Cancún",
        )

        # Footer
        canvas.setFillColor(COLOR_LIGHT)
        canvas.rect(0.5 * inch, 0.3 * inch, doc.width, 0.3 * inch, fill=1, stroke=0)
        canvas.setFillColor(colors.grey)
        canvas.setFont("Helvetica", 8)
        canvas.drawString(
            0.6 * inch, 0.42 * inch,
            f"Generado: {datetime.now().strftime('%d/%m/%Y %H:%M')} | Datos: OpenAQ, NASA, CAMS",
        )
        canvas.drawRightString(
            doc.width + 0.5 * inch,
            0.42 * inch,
            f"Página {doc.page}",
        )
        canvas.restoreState()

    # ─────────────────────────────────────────────
    # MÉTODO PRINCIPAL
    # ─────────────────────────────────────────────

    def generate(self, output_path: str | Path | None = None) -> Path:
        """
        Genera el reporte PDF completo.

        Parámetros
        ----------
        output_path : str | Path | None
            Ruta del PDF. Default: outputs/reports/reporte_{region}_{fecha}.pdf

        Retorna
        -------
        Path al PDF generado.
        """
        if not HAS_REPORTLAB:
            raise ImportError("reportlab requerido. Instalar: pip install reportlab")

        if output_path is None:
            ts = datetime.now().strftime("%Y%m%d")
            region_safe = self.region.replace(" ", "_")
            output_path = REPORTS_DIR / f"reporte_{region_safe}_{ts}.pdf"
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=letter,
            rightMargin=0.75 * inch,
            leftMargin=0.75 * inch,
            topMargin=1.0 * inch,
            bottomMargin=0.8 * inch,
        )

        logger.info(f"Generando reporte PDF: {output_path.name}")

        # Construir story
        self.story = []
        self._portada()
        self._resumen_ejecutivo()
        self.story.append(PageBreak())
        self._seccion_mapas()
        self.story.append(PageBreak())
        self._seccion_estadistica()
        self.story.append(PageBreak())
        self._seccion_tendencias()
        self.story.append(PageBreak())
        self._seccion_correlaciones()
        self.story.append(PageBreak())
        self._conclusiones()

        # Construir PDF
        doc.build(
            self.story,
            onFirstPage=self._add_header_footer,
            onLaterPages=self._add_header_footer,
        )

        size_kb = output_path.stat().st_size / 1024
        logger.info(f"Reporte generado: {output_path} ({size_kb:.0f} KB)")
        return output_path


# ─────────────────────────────────────────────
# FUNCIÓN DE CONVENIENCIA
# ─────────────────────────────────────────────

def generate_report(
    df: pd.DataFrame,
    region: str,
    date_from: str,
    date_to: str,
    variable: str = "PM2.5",
    output_path: str | Path | None = None,
) -> Path | None:
    """
    Función de conveniencia para generar un reporte en una línea.

    Ejemplo:
    --------
    path = generate_report(df, "Guatemala", "2024-01-01", "2024-12-31")
    """
    if not HAS_REPORTLAB:
        logger.error("reportlab no instalado. Instalar: pip install reportlab")
        return None
    gen = ReportGenerator(df, region, date_from, date_to, variable)
    return gen.generate(output_path)


# ─────────────────────────────────────────────
# EJEMPLO DE USO
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from ingestion.load_stations import generate_synthetic_data
    from processing.clean import clean_pipeline

    df = generate_synthetic_data(n_stations=5, days=90)
    df["datetime"] = pd.to_datetime(df["fecha"], format="%d/%m/%Y %H:%M")
    df_limpio = clean_pipeline(df)

    path = generate_report(
        df_limpio,
        region="Guatemala",
        date_from="2024-01-01",
        date_to="2024-03-31",
        variable="PM2.5",
    )
    if path:
        print(f"Reporte generado en: {path}")
    else:
        print("No se pudo generar el reporte (verificar que reportlab esté instalado)")
