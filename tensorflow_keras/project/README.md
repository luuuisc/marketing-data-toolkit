# Predicción de Conversión con TensorFlow Keras

## Descripción  
Este proyecto utiliza **TensorFlow Keras** para construir y entrenar una red neuronal profunda que predice la probabilidad de conversión de los clientes tras una campaña de email marketing de una marca de cosméticos. Al aprovechar un modelo no lineal, se capturan interacciones complejas entre variables demográficas, RFM y métricas de email como `open_rate` y `ctr`.

---

## Caso práctico: Predicción de conversiones  
Se han recopilado datos históricos de campañas de email marketing con las siguientes variables por cliente:

- **Recencia** (días desde la última compra)  
- **Frecuencia** (número de compras en el último año)  
- **Valor Monetario** (gasto total acumulado)  
- **Open Rate** y **CTR** de campañas previas  
- **Edad**, **Género** y **Región**  

Vamos a entrenar un modelo de red neuronal con varias capas densas y funciones de activación ReLU, que al final emite una probabilidad de conversión (sigmoide). El área de marketing podrá:

- Priorizar envíos a clientes con mayor probabilidad de conversión.  
- Ajustar contenido y segmentación según la salida del modelo.  
- Incrementar el ROI al focalizar recursos en perfiles de alto potencial.

---

## Estructura de archivos  

```
project/
├── main.py         # Script de entrenamiento con Keras
├── data/
│   └── campaign_history.csv     # Dataset de entrada
├── models/
│   └── keras_campaign_model.h5  # Modelo entrenado (output)
├── outputs/
│   ├── metrics_keras.csv        # Métricas de evaluación (classification report, ROC AUC)
│   └── training_history.png     # Curvas de pérdida y accuracy por época
└── README.md                    # Este documento
```

## Requisitos  
- Python 3.8+  
- tensorflow  
- pandas  
- numpy  
- scikit-learn  

Instalación:

```bash
pip install tensorflow pandas numpy scikit-learn
```

## Uso

1. Clona el repositorio y crea el entorno virtual (opcional):

```bash
python3 -m venv venv
source venv/bin/activate      # macOS/Linux
# o: venv\Scripts\activate     # Windows
pip install tensorflow pandas numpy scikit-learn
```

2. Asegúrate de que tu CSV esté en data/campaign_history.csv.
3. Crea las carpetas de salida:

```bash
mkdir -p models outputs
```

4. Ejecuta el entrenamiento:

```bash
python train_keras_model.py \
  --data_path data/campaign_history.csv \
  --test_size 0.2 \
  --model_out models/keras_campaign_model.h5 \
  --metrics_out outputs/metrics_keras.csv \
  --history_out outputs/training_history.png \
  --epochs 20 \
  --batch_size 32 \
  --lr 1e-3
```

## Parámetros principales:

- --data_path: Ruta al CSV de datos.
- --test_size: Proporción del conjunto de prueba (default=0.2).
- --model_out: Archivo HDF5 para guardar el modelo.
- --metrics_out: CSV con classification report y ROC AUC.
- --history_out: PNG con curvas de pérdida y accuracy.
- --epochs, --batch_size, --lr: Hiperparámetros del entrenamiento.


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
| `gender`      | categorical  | Género (M/F/Other)                                    |
| `region`      | categorical  | Región geográfica                                     |
| `label`       | integer      | 1 = convirtió tras campaña; 0 = no convirtió          |


## Detalles del script

- Carga y preprocesamiento: one-hot encoding de variables categóricas, normalización de numéricas.
- Arquitectura:
    - Capas densas: Input → Dense(ReLU) → Dropout → Dense(ReLU) → Dropout → Dense(sigmoide)
- Entrenamiento: optimizador Adam, pérdida binaria binary_crossentropy.
- Evaluación: accuracy y AUC en conjunto de prueba, + classification report.
- Salida:
    - Modelo Keras (.h5)
    - CSV con métricas
    - Gráfica de evolución de pérdida y accuracy

## Archivos de salida

Al ejecutar `train_keras_model.py` (o el notebook equivalente), se generan estos tres artefactos clave:

- **`models/keras_campaign_model.keras`**  
  - Modelo entrenado guardado en el formato nativo de Keras.  
  - Incluye arquitectura, pesos y configuración de entrenamiento.  
  - Listo para recargar con:
    ```python
    from tensorflow.keras.models import load_model
    model = load_model('models/keras_campaign_model.keras')
    ```
- **`outputs/metrics_keras.csv`**  
  - CSV con el classification report y el ROC AUC para la clase “1” (convertido).  
  - Columnas típicas: `precision`, `recall`, `f1-score`, `support`, `roc_auc`.  
  - Permite comparar numéricamente el desempeño y documentarlo en informes.
- **`outputs/training_history.png`**  
  - Gráfica de la evolución de la pérdida (`loss`) y la precisión (`accuracy`) en entrenamiento y validación por época.  
  - Facilita detectar sobreajuste o underfitting y tomar decisiones sobre hiperparámetros.

---

## Importancia del proyecto

Implementar una solución de Deep Learning con TensorFlow Keras para predecir conversiones tras una campaña de email marketing aporta varios beneficios estratégicos:

1. **Modelado de relaciones no lineales**  
   Redes neuronales profundas capturan interacciones complejas (por ejemplo, entre RFM y métricas de email) que los modelos lineales podrían pasar por alto.

2. **Mejora continua**  
   Con un pipeline reproducible, basta con añadir nuevos datos y reentrenar para mantener el modelo actualizado conforme cambie el comportamiento de los clientes.

3. **Optimización del ROI**  
   Al predecir probabilidades de conversión individuales, Marketing puede:
   - Dirigir ofertas a los clientes con mayor probabilidad de compra.  
   - Ajustar creatividades y asuntos para segmentos de riesgo.  
   - Reducir costes de envío masivo y aumentar la eficacia de cada campaña.

4. **Escalabilidad y despliegue**  
   El modelo guardado (.keras) se integra fácilmente en microservicios (FastAPI, TensorFlow Serving) o workflows de producción, permitiendo inferencias en tiempo real.

5. **Ventaja competitiva basada en datos**  
   Automatizar predicciones con Deep Learning sitúa a la empresa a la vanguardia de la analítica avanzada, transformando datos históricos en decisiones proactivas.

Con este proyecto, Natura (o cualquier marca de cosméticos) pasará de un enfoque reactivo a uno predictivo, alineando esfuerzos de marketing con las necesidades y comportamientos reales de sus clientes.  

## Extensiones posibles
- Experimentar con arquitecturas más profundas o regularización avanzada.
- Fine-tuning de un modelo preentrenado (por ejemplo, embeddings de texto para sentiment, si incluyes texto).
- Integrar callbacks de TensorBoard para monitoreo en tiempo real.
- Desplegar el modelo en un servicio REST con TensorFlow Serving o FastAPI.

## Licencia

MIT License — adapta y mejora este ejemplo según tus requisitos de marketing y negocio.

