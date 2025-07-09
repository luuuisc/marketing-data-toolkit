# Email Marketing Campaign Analysis

## Descripción

Este proyecto proporciona un script en Python para analizar y visualizar el rendimiento semanal de una campaña de email marketing. A partir de un archivo CSV con datos de envíos, aperturas, clics y conversiones, genera un gráfico combinado de barras y líneas que muestra:

- **Volumen de emails enviados** (barras)
- **Tasa de apertura** (línea)
- **CTR (click-through rate)** (línea)
- **Tasa de conversión** (línea)

De esta forma, el equipo de marketing puede identificar semanas con picos o caídas y tomar decisiones basadas en datos (asunto, segmentación, día de envío, etc.).

---

## Estructura de archivos

```
project/
├── main.py     # Script principal
├── data.csv        # Dataset de 52 semanas
└── outputs/
    └── campaign_performance.png  # Gráfico generado
```

## Requisitos

- Python 3.8+
- [pandas](https://pandas.pydata.org/)
- [matplotlib](https://matplotlib.org/)

---

## Instalación

1. Clona este repositorio o descarga los archivos.
2. Crea un entorno virtual (opcional pero recomendado):

   ```bash
   python3 -m venv venv
   source venv/bin/activate   # macOS/Linux
   venv\Scripts\activate      # Windows
   ```

3.	Instala las dependencias:
    ```bash
    pip install pandas matplotlib
    ```

## Uso

1. Coloca tu archivo de datos (CSV) en la raíz del proyecto con el nombre `data.csv`.
2. Ejecuta el script:
    ```bash
    python campaign_analysis.py
    ```
3. Al finalizar, encontrarás el gráfico en la carpeta outputs/ con el nombre `campaign_performance.png`.

## Formato del CSV

Las columnas esperadas en el CSV son:

| Columna       | Tipo    | Descripción                                    |
| ------------- | ------- | ---------------------------------------------- |
| `week`        | string  | Fecha de inicio de la semana (YYYY-MM-DD)      |
| `emails_sent` | integer | Número de emails enviados                      |
| `opens`       | integer | Número de emails abiertos                      |
| `clicks`      | integer | Número de clics registrados                    |
| `conversions` | integer | Número de conversiones (ventas/registro, etc.) |

## Detalles del script
- `load_data(csv_path)`:
    - Carga el CSV, ordena por fecha y calcula:
        - open_rate = opens / emails_sent
        - ctr = clicks / opens
        - conversion_rate = conversions / clicks

- `plot_campaign_performance(df, output_path)`:
    - Genera un gráfico de barras (emails enviados) y líneas (tasas).
    - Guarda la figura en PNG.

## Métricas de la campaña

En la gráfica se muestran cuatro métricas fundamentales que cubren todo el funnel de email marketing: entrega → apertura → clic → conversión.

- **Emails enviados**  

  Número total de correos que se intentaron entregar cada semana. Refleja el tamaño de tu audiencia activa y la capacidad de entrega del sistema.

- **Tasa de apertura (Open Rate)**

    Open Rate = aperturas / emails enviados × 100%
    
    Porcentaje de destinatarios que abrieron el correo. Mide la efectividad del asunto y el “preview text”.

- **Click-to-Open Rate (CTR)**  
    CTR = clics / aperturas × 100%

    Porcentaje de quienes abrieron el correo y luego hicieron clic en al menos un enlace o botón. Evalúa la relevancia del contenido interno y la fuerza de la llamada a la acción (CTA).

- **Tasa de conversión**  
    Conversion Rate = conversiones / clics × 100%
    
    Porcentaje de usuarios que, tras hacer clic, completaron la acción deseada (compra, registro, descarga, etc.). Indica la efectividad de la landing page y la propuesta de valor.



## Extensiones posibles
- Añadir análisis por segmentos (país, edad, género).
- Comparar varios asuntos de email en un mismo gráfico.
- Automatizar la generación semanal mediante CI/CD.
- Incluir métricas adicionales (bounce rate, unsubscribes).

## Licencia

Este proyecto está disponible bajo la licencia MIT. ¡Siéntete libre de adaptarlo y mejorarlo!

