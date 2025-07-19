# Seaborn 0.12 — Guía rápida para visualización estadística

**Seaborn** es una librería de visualización de datos basada en Matplotlib y optimizada para análisis estadístico.  
Permite crear gráficos informativos, atractivos y bien diseñados con una sintaxis concisa, ideal para análisis exploratorios, storytelling de datos y reportes de marketing.

---

## ¿Para qué sirve en marketing?

* **Análisis exploratorio de datos** con histogramas, violines, pares, boxplots.  
* **Segmentación visual** con mapas de calor, categorización por canal, región o audiencia.  
* **Correlación entre variables clave** como gasto, clics, conversiones y ROAS.  
* **Storytelling en dashboards** con gráficos enriquecidos y contextuales.  
* **Identificación de outliers o patrones de comportamiento**.

---

## Instalación

```bash
pip install seaborn
```

Requiere tener Matplotlib y Pandas instalados.
Se recomienda también: pip install jupyterlab para entornos interactivos.

## Conceptos clave

| Concepto        | Descripción                                 | Ejemplo en marketing                              |
| --------------- | ------------------------------------------- | ------------------------------------------------- |
| `relplot()`     | Gráfico relacional (línea o dispersión)     | CTR vs presupuesto por campaña                    |
| `catplot()`     | Gráfico categórico (barra, box, violín)     | Distribución de CPA por canal                     |
| `pairplot()`    | Matriz de relaciones entre variables        | Gasto vs ROAS vs conversión                       |
| `heatmap()`     | Mapa de calor                               | Correlación entre variables                       |
| `hue`, `col`, `row` | Variables para dividir o colorear subgráficos | Segmentos por audiencia, canal o región     |

## Métodos esenciales 

| Categoría         | Método(s)                            | Uso típico                                                   |
| ----------------- | ------------------------------------ | ------------------------------------------------------------ |
| Dispersión        | `scatterplot()`, `relplot()`         | Visualizar relación entre dos variables (ej. ROAS vs gasto)  |
| Líneas            | `lineplot()`, `relplot(kind="line")` | Evolución temporal de métricas (ej. conversiones semanales)  |
| Barras            | `barplot()`, `countplot()`           | Comparar categorías (ej. campañas por canal o CTR por región)|
| Boxplots          | `boxplot()`, `violinplot()`          | Ver distribución y outliers (ej. CPA por canal)              |
| Mapas de calor    | `heatmap()`                          | Visualizar correlaciones entre variables                     |
| Matriz de pares   | `pairplot()`                         | Ver múltiples relaciones a la vez (scatter, hist, corr.)     |
| Gráficos categóricos | `catplot()`                      | Barras, cajas o violines en múltiples subgráficos            |
| Divisiones visuales | `hue=`, `col=`, `row=`             | Crear subgráficos por segmento, canal, grupo demográfico     |
| Estilo y contexto | `set_theme()`, `set_style()`, `set_context()` | Ajustar estética para informes o dashboards           |
| Guardado          | `plt.savefig()` (via Matplotlib)     | Exportar gráficos a PNG, PDF o SVG                          |

## Ejemplo 

```python
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Datos de ejemplo
df = pd.read_csv("campaign_summary.csv")

# Dispersión ROAS vs Gasto, coloreado por canal
sns.relplot(data=df, x="ad_spend_usd", y="roas", hue="channel", height=6)
plt.title("ROAS vs Gasto Publicitario")
plt.tight_layout()
plt.savefig("roas_vs_spend.png", dpi=150)
```

Un gráfico simple con gran capacidad narrativa para explicar la eficiencia de campañas por canal.

## Buenas prácticas
- Usa `relplot` y `catplot` en vez de `scatterplot` o `barplot` para aprovechar subplots automáticos.
- Siempre agrega `plt.tight_layout()` y títulos claros.
- Usa `palette="deep"` o `"muted"` para presentaciones profesionales.
- Personaliza con `sns.set_context("notebook" | "talk" | "poster")` según el uso.
- Combina ***Seaborn*** con ***matplotlib.pyplot*** para mayor control estético.

## Recursos

| Tipo           | Referencia                                                                       |
| -------------- | --------------------------------------------------------------------------------- |
| Documentación  | [https://seaborn.pydata.org](https://seaborn.pydata.org)                         |
| Guía completa  | [https://seaborn.pydata.org/tutorial.html](https://seaborn.pydata.org/tutorial.html) |                              