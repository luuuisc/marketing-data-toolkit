# Pandas 2.3 — Guía rápida para analítica de marketing

**Pandas** es la librería de referencia en Python para la manipulación y el análisis de datos tabulares.  

Proporciona estructuras de datos de alto rendimiento (Series y DataFrame) y una API inspirada en SQL y en las hojas de cálculo, reduciendo drásticamente el tiempo entre la ingesta de datos crudos y la obtención de *insights*. 

Referencia [oai_citation:0‡pandas.pydata.org](https://pandas.pydata.org/docs/whatsnew/index.html?utm_source=chatgpt.com)

---

## ¿Para qué sirve en marketing?

* **Consolidar fuentes** (CRM, Google Analytics, Meta Ads, Google Ads).  
* **Limpiar y transformar** datos heterogéneos en tablas coherentes.  
* **Construir tablas de hechos** y dimensiones listas para BI o modelado de ROI.  

Referencia: [oai_citation:1‡medium.com](https://medium.com/data-at-the-core/pandas-for-market-analytics-dc91f32e7f05?utm_source=chatgpt.com)

---

## Instalación mínima

```bash
pip install pandas
```

Requiere Python ≥ 3.9. En entornos científicos se recomienda instalar con Conda para dependencias optimizadas.  ￼


### Estructuras de datos principales

| Objeto          | Descripción                      | Ejemplo en marketing                     |
| --------------- | -------------------------------- | ---------------------------------------- |
| `Series`        | Vector 1-D etiquetado            | Historial diario de impresiones          |
| `DataFrame`     | Tabla 2-D etiquetada             | Tabla de campañas (gasto, clics, conv.)  |
| `DatetimeIndex` | Índice temporal especializado    | *Resampling* semanal de KPI              |

Estas estructuras permiten operaciones vectorizadas y el patrón split-apply-combine.  ￼

## Conceptos clave 

| Concepto        | Descripción                                         | Ejemplo en marketing                          |
| --------------- | --------------------------------------------------- | --------------------------------------------- |
| `Series`        | Estructura unidimensional con etiquetas             | Historial diario de impresiones de campaña    |
| `DataFrame`     | Estructura bidimensional tipo tabla                 | Base de datos de leads, campañas o sesiones   |
| `Index`         | Etiquetas o claves para filas y columnas           | Fechas, IDs de usuario, canales de adquisición|
| `DatetimeIndex` | Index temporal especializado                        | Resampleo por semana/mes en análisis de KPI   |
| `groupby()`     | Agrupación de datos por categoría o segmento        | Gasto total por canal o país                  |
| `merge()`       | Unión de múltiples DataFrames                       | Unir datos de CRM con resultados de campañas  |
| `pivot_table()` | Tabla dinámica resumida                             | Promedio de ROAS por canal y semana           |
| `apply()`       | Aplicar funciones personalizadas sobre columnas     | Limpiar o transformar campos como nombres     |
| `read_csv()`    | Carga rápida de datos desde archivos planos         | Ingestar datos de campañas o sesiones         |
| `to_parquet()`  | Exportar DataFrames a formatos eficientes           | Guardar tablas de hechos para dashboards BI   |

### Métodos esenciales


| Categoría       | Método(s)                              | Uso típico                                        |
| --------------- | -------------------------------------- | ------------------------------------------------- |
| I/O             | `read_csv`, `read_parquet`, `read_sql` | Ingestar CSV, Parquet o bases SQL                 |
| Exploración     | `head`, `info`, `describe`             | Vista rápida y estadísticas                       |
| Selección       | `.loc`, `.iloc`, *boolean indexing*    | Subconjuntos por etiqueta o posición              |
| Transformación  | `assign`, `pipe`, `apply`              | Crear / modificar columnas                        |
| Agregación      | `groupby`, `agg`, `pivot_table`        | Resúmenes por segmento o canal                    |
| Fechas          | `resample`, `asfreq`                   | Re-muestreo temporal                              |
| Unión           | `merge`, `join`, `concat`              | Combinar tablas estilo SQL                        |
| Salida          | `to_csv`, `to_parquet`                 | Persistir resultados                              |


## Ejemplo

```python
import pandas as pd
```

### 1. Ingesta multicanal
spend   = pd.read_csv("ad_spend.csv",   parse_dates=["date"])
traffic = pd.read_csv("ga_sessions.csv",parse_dates=["date"])
sales   = pd.read_sql("SELECT * FROM orders", conn, parse_dates=["date"])

### 2. Limpieza
for df in (spend, traffic, sales):
    df["date"] = pd.to_datetime(df["date"])

### 3. Unión
fact = (
    spend.merge(traffic, on=["date", "campaign_id"], how="left")
         .merge(sales,  on=["date", "campaign_id"], how="left")
         .fillna(0)
)

### 4. Agregación semanal
weekly = (
    fact.groupby([pd.Grouper(key="date", freq="W"), "campaign_id"])
        .agg(spend_usd=("cost_usd", "sum"),
             sessions=("sessions", "sum"),
             revenue_usd=("revenue_usd", "sum"))
        .reset_index()
)

weekly.to_parquet("fact_campaign_weekly.parquet")

Genera una tabla de hechos semanal lista para modelos de atribución o dashboards ejecutivos.  ￼

Buenas prácticas de rendimiento
	1.	Vectoriza operaciones y evita loops en Python.
	2.	Optimiza tipos: convierte cadenas repetidas a category y enteros dispersos a Int64.  ￼
	3.	Usa usecols en read_csv para cargar solo las columnas necesarias.  ￼
	4.	Copy-on-Write (CoW) reduce copias accidentales a partir de la versión 2.3.  ￼

## Recursos

| Tipo          | Referencia                                                                                 |
| ------------- | ------------------------------------------------------------------------------------------ |
| Libro         | *Python for Data Analysis*, 3.ª ed., Wes McKinney (2022)                                   |
| Documentación | Pandas 2.3 — *What’s new* y notas de versión                                               |
| Cheat Sheet   | DataCamp — “Pandas Cheat Sheet”                                                            |


#### Fuentes consultadas  

1. Pagina oficial “What’s new in 2.3.0” — Pandas.  [oai_citation:18‡pandas.pydata.org](https://pandas.pydata.org/docs/whatsnew/v2.3.0.html?utm_source=chatgpt.com)  
2. Python for Data Analysis, 3.ª ed. — Wes McKinney.  [oai_citation:19‡wesmckinney.com](https://wesmckinney.com/book/?utm_source=chatgpt.com)  
3. DataCamp “Pandas Cheat Sheet”.  [oai_citation:20‡datacamp.com](https://www.datacamp.com/cheat-sheet/pandas-cheat-sheet-for-data-science-in-python?utm_source=chatgpt.com)  
4. Pandas docs “Copy-on-Write”.  [oai_citation:21‡pandas.pydata.org](https://pandas.pydata.org/docs/dev/user_guide/copy_on_write.html?utm_source=chatgpt.com)  
5. Pandas docs “GroupBy split-apply-combine”.  [oai_citation:22‡pandas.pydata.org](https://pandas.pydata.org/docs/user_guide/groupby.html?utm_source=chatgpt.com)  
6. `read_csv` reference — Pandas docs.  [oai_citation:23‡pandas.pydata.org](https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html?utm_source=chatgpt.com)  
7. W3Resource — Optimización de memoria con `categorical`.  [oai_citation:24‡w3resource.com](https://www.w3resource.com/python-exercises/pandas/optimize-memory-usage-with-categorical-data-type-in-pandas-dataframe.php?utm_source=chatgpt.com)  
8. Medium — Caso de analítica de marketing con Pandas.  [oai_citation:25‡medium.com](https://medium.com/data-at-the-core/pandas-for-market-analytics-dc91f32e7f05?utm_source=chatgpt.com)  
9. Pandas Release Notes index.  [oai_citation:26‡pandas.pydata.org](https://pandas.pydata.org/docs/whatsnew/index.html?utm_source=chatgpt.com)  
10. Amazon listing “Python for Data Analysis”.  [oai_citation:27‡amazon.com](https://www.amazon.com/Python-Data-Analysis-Wrangling-Jupyter/dp/109810403X?utm_source=chatgpt.com)  
11. DataCamp advanced cheat sheet.  [oai_citation:28‡datacamp.com](https://www.datacamp.com/cheat-sheet/pandas-cheat-sheet-data-wrangling-in-python?utm_source=chatgpt.com)  
12. Pandas CoW (2.0 docs) for historical context.  [oai_citation:29‡pandas.pydata.org](https://pandas.pydata.org/pandas-docs/version/2.0/user_guide/copy_on_write.html?utm_source=chatgpt.com)

