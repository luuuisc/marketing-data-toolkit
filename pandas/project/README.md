# Customer Segmentation with RFM Analysis

## Descripción

Este proyecto utiliza **pandas** para realizar un análisis RFM (Recencia, Frecuencia, Valor Monetario) sobre datos de transacciones de clientes. A partir de un archivo CSV con el histórico de compras, el script:

1. Carga y limpia los datos de transacciones.  
2. Calcula para cada cliente:
   - **Recencia**: días desde su última compra.  
   - **Frecuencia**: número total de compras.  
   - **Monetario**: gasto total.  
3. Asigna a cada cliente un puntaje R, F y M dividido en cuartiles.  
4. Genera un **RFM Score** combinando los puntajes y clasifica clientes en segmentos (“Champions”, “Loyal”, “At Risk”, etc.).  
5. Exporta un CSV con la tabla de RFM y un resumen de cada segmento.

Con este análisis, el equipo de marketing puede diseñar campañas personalizadas, por ejemplo:

- Enviar promociones VIP a “Champions”  
- Reactivar usuarios “At Risk” con cupones  
- Aumentar ventas cruzadas en segmentos “Potential Loyal”

### ¿Qué es el Análisis RFM?

El análisis **RFM** (Recencia, Frecuencia y Valor Monetario) es una técnica de segmentación de clientes ampliamente utilizada en marketing relacional y CRM para cuantificar el comportamiento de compra y el valor de cada cliente. Se basa en tres dimensiones:

1. **Recencia (R)**  
   Mide el tiempo transcurrido (en días) desde la última transacción de un cliente hasta una fecha de referencia (por ejemplo, la fecha de ejecución del análisis). Clientes con un valor de recencia bajo (fecha de compra más reciente) suelen estar más comprometidos.

2. **Frecuencia (F)**  
   Cuenta el número total de transacciones que un cliente ha realizado en el período de análisis. Una frecuencia alta indica clientes leales que compran con asiduidad.

3. **Valor Monetario (M)**  
   Suma el importe total gastado por un cliente durante el período de análisis. Un valor monetario elevado corresponde a clientes de alto valor económico para la empresa.

Cada una de estas métricas se clasifica en cuartiles para asignar un puntaje de 1 a 4 (siendo 4 el mejor desempeño relativo). Los puntajes individuales se combinan en un **RFM Score** (por ejemplo, “4-3-2”), que luego se asocia a segmentos de negocio como “Champions”, “Loyal Customers”, “At Risk” o “Potential Loyalist” para guiar acciones de marketing personalizadas y optimizar la asignación de recursos.

## Estructura de archivos

```
project/
├── main.py            # Script principal
├── data/
│   └── data.csv       # Archivo CSV con transacciones
├── outputs/
│   ├── rfm_table.csv          # Tabla RFM resultante por cliente
│   └── segment_summary.csv    # Resumen estadístico por segmento
└── README.md                  # Este documento
```

## Requisitos

- Python 3.8+
- [pandas](https://pandas.pydata.org/)

---

## Instalación

1. Clona este repositorio o descarga los archivos.  
2. Crea y activa un entorno virtual (opcional):
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # macOS/Linux
   ```
3.	Instala la dependencia:
    ```bash
    pip install pandas
    ```

## Uso

1.	Coloca tu CSV de transacciones en data/data.csv con columnas:

      | Columna            | Tipo       | Descripción                              |
      | ------------------ | ---------- | ---------------------------------------- |
      | `customer_id`      | string     | Identificador único de cada cliente      |
      | `transaction_date` | YYYY-MM-DD | Fecha de la transacción                  |
      | `amount`           | float      | Monto gastado en esa transacción         |

2.	Ejecuta el script:

    ```bash
    python segmentation.py
    ```

3.	Revisa los resultados en outputs/:
- `rfm_table.csv`: cada cliente con sus valores R, F, M y RFM Score
- `segment_summary.csv`: conteo, promedio de M y recencia media por segmento.

## Detalles del script
- `load_data(path)`: carga y formatea fechas.
- `compute_rfm(df, reference_date)`: calcula recencia, frecuencia y monetario usando reference_date (hoy).
- `score_rfm(rfm)`: asigna cuartiles (1–4) y combina en una cadena “RFM Score” (por ej. “4-3-2”).
- `assign_segments(rfm_scored)`: mapea cada “RFM Score” a un segmento de negocio.
- `export_results(rfm_final)`: escribe los CSV de salida.

## Archivos de salida

Al ejecutar `main.py`, el script genera los siguientes archivos en la carpeta `outputs/`:

- **`rfm_table.csv`**  
  - **Qué es**: la tabla detallada de RFM para cada cliente.  
  - **Columnas**:  
    | Columna        | Descripción                                                                                 |
    | -------------- | ------------------------------------------------------------------------------------------- |
    | `customer_id`  | Identificador único de cada cliente                                                         |
    | `recency`      | Días transcurridos desde la última transacción (fecha de referencia − última compra)        |
    | `frequency`    | Número total de transacciones realizadas por el cliente                                     |
    | `monetary`     | Gasto total acumulado por el cliente                                                        |
    | `R_score`      | Puntaje 1–4 de Recencia (4 = clientes más recientes)                                       |
    | `F_score`      | Puntaje 1–4 de Frecuencia (4 = clientes que más compran)                                   |
    | `M_score`      | Puntaje 1–4 de Monetario (4 = clientes con mayor gasto)                                    |
    | `RFM_Score`    | Cadena combinada “R-F-M” (por ejemplo, “4-3-2”)                                            |
    | `segment`      | Segmento asignado según la lógica de negocio (Champions, Loyal Customers, At Risk, etc.)   |
  - **Para qué sirve**:  
    - Permite ver en un solo CSV el comportamiento de cada cliente en las tres dimensiones RFM.  
    - Puedes filtrar clientes específicos o exportar subconjuntos (por ejemplo, todos los “At Risk”).  
    - Base de datos de partida para acciones de marketing personalizadas.

- **`segment_summary.csv`**  
  - **Qué es**: resumen estadístico agrupado por segmento de negocio.  
  - **Columnas**:  
    | Columna         | Descripción                                          |
    | --------------- | ---------------------------------------------------- |
    | `segment`       | Nombre del segmento (Champions, Potential Loyalist…) |
    | `customers`     | Número de clientes en ese segmento                   |
    | `avg_recency`   | Recencia media (días) de los clientes del segmento   |
    | `avg_frequency` | Frecuencia media (número de compras)                 |
    | `avg_monetary`  | Gasto medio por cliente                              |
  - **Para qué sirve**:  
    - Ofrece una visión rápida de cuántos clientes forman cada segmento.  
    - Facilita comparar la lealtad y el valor económico medio entre segmentos.  
    - Base para priorizar campañas (por ejemplo, impulsar un  segmento “At Risk” o premiar a los “Champions”).

## Extensiones posibles
- Añadir visualizaciones (histogramas de R, F y M) con matplotlib.
- Incorporar filtros: solo clientes con valor ≥ X.
- Automatizar generación de reportes semanales mediante un pipeline CI
- Integrar con una dashboard web (Dash o Streamlit) para exploración interactiva.

## Licencia

MIT License – siéntete libre de adaptar y mejorar este análisis para tus necesidades de marketing.

