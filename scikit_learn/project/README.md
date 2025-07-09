# Campaign Response Prediction

## Descripción

Este proyecto utiliza **scikit-learn** para entrenar y evaluar un modelo de clasificación que predice la probabilidad de conversión de los clientes tras una campaña de email marketing. Basado en datos históricos de una marca de cosméticos como Natura, el objetivo es:

- Identificar clientes con alta probabilidad de compra.  
- Optimizar segmentación y personalización de campañas.  
- Mejorar el ROI al enfocar recursos en los perfiles más receptivos.

## Caso práctico: Predicción de conversiones en Natura

Natura lanzó varias campañas de email para su línea de maquillaje. Se recopilaron datos por cliente:

- **Recencia**: días desde la última compra.  
- **Frecuencia**: número de compras en el último año.  
- **Valor Monetario**: gasto total acumulado.  
- **Open Rate** y **CTR** de campañas previas.  
- **Edad**, **Género** y **Región**.  

Con este dataset, entrenamos un **Logistic Regression** que predice la probabilidad de que un cliente realice una compra (label 1) tras la campaña. Marketing puede así:

- Enviar ofertas personalizadas a clientes con alta probabilidad de conversión.  
- Redistribuir presupuesto para retargeting de clientes de baja probabilidad.  
- Medir el impacto real de cambios en asunto y contenido.

## Estructura de archivos

```
projects/
├── train_model.py             # Script de entrenamiento y evaluación
├── data/
│   └── campaign_history.csv   # Dataset con features y label
├── models/
│   └── campaign_model.pkl     # Modelo serializado (output)
├── outputs/
│   └── metrics.csv            # Métricas de evaluación (classification report, ROC AUC)
└── README.md                  # Este documento
```

## Requisitos

- Python 3.8+  
- [pandas](https://pandas.pydata.org/)  
- [numpy](https://numpy.org/)  
- [scikit-learn](https://scikit-learn.org/)  
- [joblib](https://joblib.readthedocs.io/)  

---

## Instalación

```bash
python3 -m venv venv
source venv/bin/activate    # macOS/Linux
venv\Scripts\activate       # Windows

pip install pandas numpy scikit-learn joblib
```

## Uso

```bash
python train_model.py \
  --data_path data/campaign_history.csv \
  --test_size 0.2 \
  --model_out models/campaign_model.pkl \
  --metrics_out outputs/metrics.csv
```

- --data_path: Ruta al CSV de datos (campaign_history.csv).
- --test_size: Proporción del conjunto de prueba (por defecto 0.2).
- --model_out: Archivo donde se serializa el modelo entrenado.
- --metrics_out: CSV con las métricas de evaluación.

Asegúrate de crear las carpetas models/ y outputs/ antes de ejecutar:

```bash
mkdir -p models outputs
```

## Formato del CSV de entrada

| Columna       | Tipo         | Descripción                                           |
| ------------- | ------------ | ----------------------------------------------------- |
| `customer_id` | string       | Identificador único de cada cliente                   |
| `recency`     | integer      | Días desde la última compra                           |
| `frequency`   | integer      | Número de compras en el último año                    |
| `monetary`    | float        | Gasto total acumulado                                 |
| `open_rate`   | float (0-1)  | Porcentaje de apertura en campañas previas            |
| `ctr`         | float (0-1)  | Click-Through Rate en campañas previas                |
| `age`         | integer      | Edad del cliente                                      |
| `gender`      | categorical  | Género (M/F/Otro)                                     |
| `region`      | categorical  | Región geográfica                                     |
| `label`       | integer      | 1 = convirtió tras la campaña; 0 = no convirtió       |


## Detalles del script
- load_data(path): carga el CSV y transforma tipos.
- Preprocesamiento:
    - One-hot encoding de gender y region.
    - Normalización de variables numéricas con StandardScaler.

- train_test_split: divide el dataset en entrenamiento y prueba.
- Modelo:
    - LogisticRegression (se puede ajustar con GridSearchCV).
- Evaluación:
    - Accuracy, Precision, Recall, F1-score y ROC AUC.
    - Se exporta un classification report en outputs/metrics.csv.
- Serialización:
    - Pipeline completo (scaler + model) guardado con joblib.dump() en models/campaign_model.pkl.

## Extensiones posibles
- Incluir GridSearchCV para optimizar hiperparámetros.
- Comparar con otros algoritmos (Random Forest, XGBoost).
- Construir un Pipeline de scikit-learn para un flujo más limpio.
- Añadir visualizaciones de curvas ROC y precision-recall.

## Licencia

MIT License – adapta este ejemplo a tus necesidades de predicción de campañas.

