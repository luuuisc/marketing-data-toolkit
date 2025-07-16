# Matplotlib 3.10 — Guía rápida para visualización de datos

**Matplotlib** es la librería estándar para crear visualizaciones estáticas, animadas e interactivas en Python.  
Sirve como base de librerías de más alto nivel (Seaborn, Pandas `.plot`, Plotly Express) y es el “lenguaje gráfico” que permite comunicar *insights* con precisión y control pixel-a-pixel.  
Referencia a: [Matplotlib](https://matplotlib.org/stable/)

---

## ¿Para qué sirve en marketing?

* **KPI dashboards** : evoluciones de gasto, impresiones, clics y ROAS.  
* **Segmentación visual** : comparar audiencias o canales en barras apiladas y diagramas de áreas.  
* **A/B testing** : histogramas y curvas de densidad para efectos de tratamiento.  
* **Storytelling de campañas** : líneas con anotaciones, emojis y *callouts* para hitos.

---

## Instalación mínima

```bash
pip install matplotlib
```

También disponible vía Conda:
conda install -c conda-forge matplotlib  ￼

## Conceptos clave

| Concepto | Papel                       | Ejemplo en marketing                  |
| -------- | -------------------------- | ------------------------------------- |
| `Figure` | Lienzo contenedor          | Dashboard trimestral                  |
| `Axes`   | Sistema de ejes; un gráfico vive aquí | Evolución de CPA por semana     |
| `Artist` | Todo lo que se dibuja (líneas, textos, patches) | Línea de conversiones, logo en la esquina |

Matplotlib 3.9/3.10 incluye subfiguras con z-order controlable y mejoras de layout que facilitan dashboards multi-panel.  ￼

## Métodos esenciales

| Tipo de gráfico | Método rápido              | Insight típico                         |
| --------------- | -------------------------- | -------------------------------------- |
| Línea           | `plt.plot` / `ax.plot`     | Tendencia de ROAS                      |
| Área apilada    | `ax.stackplot`             | Distribución de gasto multicanal       |
| Barras          | `ax.bar`, `ax.barh`        | Top 10 audiencias por CTR              |
| Dispersión      | `ax.scatter`               | Relación CPC vs Conversion Rate        |
| Box/Violin      | `ax.boxplot`, `ax.violinplot` | Distribución de CPA                 |
| Heatmap         | `ax.imshow` + `plt.cm`     | Matriz canal-segmento                  |

Extras valiosos: `ax.twinx()` (ejes gemelos), `fig.autofmt_xdate()` (fechas legibles), `fig.savefig()` (exportar PNG, PDF, SVG).


## Ejemplo

```python
import matplotlib.pyplot as plt
import pandas as pd


df = pd.read_parquet("weekly_kpis.parquet")   # semanales por campaña

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Línea de ROAS
ax1.plot(df["week"], df["roas"], marker="o")
ax1.set_title("ROAS Semanal")
ax1.set_ylabel("ROAS")
ax1.grid(True, alpha=.3)

# Barras de gasto por canal
channels = ["google_ads", "meta_ads", "tiktok_ads"]
ax2.stackplot(df["week"], *(df[c] for c in channels), labels=channels)
ax2.set_title("Gasto semanal por canal")
ax2.set_ylabel("USD")
ax2.legend(loc="upper left")
ax2.grid(True, alpha=.3)

fig.suptitle("Dashboard KPI — Q2 2025")
fig.tight_layout()
fig.savefig("dashboard_kpi_q2.png", dpi=150)
```

Producción de un dashboard listo para incrustar en reportes o apps de BI. Plantillas similares aparecen en compilaciones de data-analytics dashboards con Matplotlib.  ￼

## Buenas prácticas de visualización

- Cuenta una historia : agrega títulos, subtítulos y anotaciones; menos es más.
- Consistencia de estilos : usa `plt.style.use("seaborn-v0_8")` o crea tu estilo corporativo (mpl.rcParams).
- Espaciado automático : `fig.tight_layout()` o `constrained_layout=True` en `plt.subplots`.
- Exporta en alta resolución : `savefig(..., dpi=300, bbox_inches="tight")`.
- Accesibilidad : revisa contraste y usa paletas color-blind friendly (e.g., `mpl.colormaps["tab10"]`).


## Recursos

| Tipo           | Referencia                                                          |
| -------------- | ------------------------------------------------------------------- |
| Documentación  | Matplotlib 3.10.3 — *Quick Start guide*            |



