# Exploratory Data Analysis & Visualization con Seaborn

## Descripción  
Este proyecto utiliza **seaborn** para realizar un análisis exploratorio de datos (EDA) y generar visualizaciones clave sobre el rendimiento de campañas de email marketing de una marca de cosméticos (por ejemplo, Natura). Con estas gráficas, el equipo de marketing podrá identificar patrones, comparar segmentos y comunicar insights de forma clara y atractiva.

---

## Caso práctico: EDA de campañas  

Partiendo de un histórico de clientes con métricas RFM y de email marketing (recencia, frecuencia, valor monetario, open rate, CTR) junto con datos demográficos (edad, género, región), responderemos preguntas como:  
- ¿Cómo se distribuye la tasa de apertura (`open_rate`)?  
- ¿Varía el CTR por región y género?  
- ¿Qué variables numéricas están más correlacionadas?  
- ¿Existen relaciones lineales entre frecuencia de compra y gasto total?

---

## Estructura de archivos  

```
project/
├── main.py      # Script principal de generación de gráficas
├── data/
│   └── campaign_history.csv      # Dataset de entrada
└── outputs/
    ├── dist_open_rate.png        # Histograma de open_rate
    ├── box_ctr_by_region.png     # Boxplot de CTR por región
    ├── heatmap_correlation.png   # Heatmap de correlaciones
    └── pairplot_features.png     # Pairplot de variables numéricas
```

## Requisitos  
- Python 3.8+  
- pandas  
- seaborn  
- matplotlib  

## Instalación  
```bash
python3 -m venv venv
source venv/bin/activate      # macOS/Linux
# o: venv\Scripts\activate    # Windows

pip install pandas seaborn matplotlib
```

## Uso
1. Coloca tu CSV en data/campaign_history.csv.
2. Crea la carpeta de salida:

```bash
mkdir -p outputs
```

3. Ejecuta el script:

```bash
python seaborn_visualization.py \
  --data_path data/campaign_history.csv \
  --out_dir outputs
```

Al finalizar, encontrarás en outputs/ las cuatro gráficas en formato PNG.

## Formato del CSV de entrada

| Columna       | Tipo         | Descripción                                           |
| ------------- | ------------ | ----------------------------------------------------- |
| `customer_id` | string       | Identificador único de cada cliente                   |
| `recency`     | integer      | Días desde la última compra                           |
| `frequency`   | integer      | Número de compras en el último año                    |
| `monetary`    | float        | Gasto total acumulado                                 |
| `open_rate`   | float (0–1)  | Tasa de apertura en campañas previas                  |
| `ctr`         | float (0–1)  | Click-Through Rate en campañas previas                |
| `age`         | integer      | Edad del cliente                                      |
| `gender`      | categorical  | Género del cliente (M/F/Other)                        |
| `region`      | categorical  | Región geográfica                                     |
| `label`       | integer      | 1 = convirtió tras campaña; 0 = no convirtió          |

## Detalles del script
- --data_path: ruta al CSV de datos.
- --out_dir: carpeta donde se guardan las gráficas.
- Genera cuatro tipos de visualizaciones:
    - Histograma de open_rate
    - Boxplot de ctr por region
    - Heatmap de correlaciones entre variables numéricas
    - Pairplot de variables numéricas coloreado por label

## Explicación de las gráficas de salida

A continuación se describe cada una de las cuatro visualizaciones generadas y cómo interpretarlas para extraer insights de marketing:

### 1. Histograma de distribución de Open Rate (`dist_open_rate.png`)
- **Qué muestra**: la frecuencia con la que aparecen distintos valores de tasa de apertura (`open_rate`) en el conjunto de clientes.
- **Interpretación**:  
  - Un pico en valores altos (por ejemplo 0.6–0.8) indica que muchos clientes suelen abrir la mayoría de los correos.  
  - Una cola larga hacia valores bajos (por ejemplo <0.2) señala un grupo de clientes poco receptivos.  
- **Acción**:  
  - Segmentar clientes con `open_rate` bajo para campañas de re-engagement o pruebas de nuevos asuntos.  
  - Potenciar campañas A/B en los rangos intermedios para identificar asuntos óptimos.

### 2. Boxplot de CTR por región (`box_ctr_by_region.png`)
- **Qué muestra**: la distribución de la tasa de clics (`ctr`) segmentada por cada región geográfica.
- **Interpretación**:  
  - La mediana (línea central) compara qué región convierte mejor.  
  - La amplitud del “caja” y los “bigotes” indican variabilidad: regiones con cajas estrechas tienen comportamientos más homogéneos.  
  - Valores atípicos (“outliers”) señalan usuarios con CTR excepcionalmente alto o bajo.  
- **Acción**:  
  - Focalizar recursos en regiones con mediana y cuartiles superiores (CTR alto).  
  - Investigar causas de baja variabilidad y bajo CTR (p. ej., diferencias culturales, relevancia del contenido).

### 3. Heatmap de correlaciones (`heatmap_correlation.png`)
- **Qué muestra**: coeficientes de correlación de Pearson entre variables numéricas (`recency`, `frequency`, `monetary`, `open_rate`, `ctr`, `age`).
- **Interpretación**:  
  - Valores cercanos a +1 o –1 indican relaciones fuertes.  
  - Por ejemplo, una correlación positiva alta entre `frequency` y `monetary` sugiere que quienes compran más también gastan más.  
  - Una correlación negativa entre `recency` y `frequency` indicaría que clientes recientes suelen comprar con mayor frecuencia.  
- **Acción**:  
  - Priorizar puntuaciones RFM: si `frequency` y `monetary` están fuertemente ligadas, ambos indicadores pueden combinarse para identificar clientes VIP.  
  - Ajustar mensajes: clientes con baja recencia alta frecuencia podrían recibir ofertas de fidelización.

### Conceptos

| Variable    | Significado                                                                                                                                           |
|-------------|--------------------------------------------------------------------------------------------------------------------------------------------------------|
| `recency`   | Días transcurridos desde la última compra del cliente hasta la fecha de análisis. Mide qué tan “reciente” es su actividad de compra.                     |
| `frequency` | Número total de compras realizadas por el cliente en el período de análisis. Indica cuántas veces ha vuelto a comprar.                                   |
| `monetary`  | Gasto total acumulado por el cliente en el período de análisis. Refleja el valor económico que aporta cada cliente.                                      |
| `open_rate` | Tasa de apertura de emails: proporción de correos abiertos sobre el total de correos enviados en campañas previas. Mide el interés inicial en el contenido. |
| `ctr`       | Click-Through Rate (CTR): proporción de clics sobre los correos abiertos en campañas previas. Evalúa la efectividad de la llamada a la acción interna.    |
| `age`       | Edad del cliente en años. Permite segmentar comportamientos de consumo por rango etario.                                                                |

### 4. Pairplot de variables numéricas coloreado por conversión (`pairplot_features.png`)
- **Qué muestra**: diagramas de dispersión y distribuciones conjuntas de todas las variables numéricas, con puntos coloreados según `label` (1 = convirtió, 0 = no convirtió).
- **Interpretación**:  
  - Permite ver patrones bivariados entre pares de variables.  
  - Por ejemplo, en el scatter de `monetary` vs. `frequency`, puedes observar si los clientes que convirtieron (color distinto) tienden a valores altos en ambas.  
  - Las diagonales muestran histogramas separados de cada variable para cada clase.  
- **Acción**:  
  - Detectar umbrales: si los clientes convertidores tienen `open_rate` >0.5 y `ctr` >0.3, esas son buenas señales para la segmentación.  
  - Diseñar reglas de negocio simples (reglas de “si-esto-entonces”) basadas en umbrales visibles en los gráficos.

Estas visualizaciones en conjunto proporcionan una comprensión profunda del comportamiento de los clientes y te permiten tomar decisiones informadas sobre segmentación, personalización de contenido y optimización de campañas de email marketing.  

## Extensiones posibles
- Añadir gráficos de distribución segmentados por género o rango de edad.
- Incorporar líneas de tendencia o regresión en scatterplots.
- Exportar un reporte HTML con todas las figuras integradas.

## Licencia

MIT License — siéntete libre de adaptar este ejemplo para tus necesidades de análisis y visualización.

