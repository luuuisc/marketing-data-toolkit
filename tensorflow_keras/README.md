# TensorFlow 2.16 + Keras — Guía rápida para modelos de Deep Learning

**TensorFlow** es una plataforma de machine learning de extremo a extremo, desarrollada por Google.  
**Keras** es su interfaz de alto nivel para construir y entrenar redes neuronales de manera rápida, modular e intuitiva.  
Ambas juntas permiten crear modelos potentes para visión por computadora, NLP, tabulares, series de tiempo y más.

---

## ¿Para qué sirve en marketing?

* **Predicción de comportamiento de usuario** (click, compra, abandono).  
* **Clasificación de leads o segmentos con modelos neuronales tabulares.**  
* **Análisis de sentimientos con texto de reseñas o encuestas.**  
* **Modelos de series de tiempo para forecasting de ventas o tráfico.**  
* **Personalización con modelos recomendadores.**  
* **Fine-tuning de modelos preentrenados para campañas visuales o análisis de imágenes.**

---

## Instalación

```bash
pip install tensorflow
```

Para usar GPU:
consulta https://www.tensorflow.org/install para ver las versiones compatibles con CUDA/cuDNN.

## Conceptos clave

| Concepto         | Descripción                                     | Ejemplo en marketing                        |
| ---------------- | ----------------------------------------------- | ------------------------------------------- |
| `Model` / `Sequential` | Arquitectura del modelo                | Red neuronal para predicción de churn       |
| `Layer`          | Capa dentro del modelo                         | `Dense`, `Dropout`, `Embedding`, etc.       |
| `compile()`      | Configura la función de pérdida y optimizador  | `loss='binary_crossentropy'`, `optimizer='adam'` |
| `fit()`          | Entrena el modelo con los datos                | Historial de entrenamiento sobre leads      |
| `evaluate()`     | Evalúa el rendimiento sobre test set           | Accuracy o AUC sobre nuevos usuarios        |
| `predict()`      | Genera predicciones                            | Probabilidad de compra o abandono           |

## Métodos esenciales

| Categoría            | Método(s)                                             | Uso típico                                                         |
| -------------------- | ----------------------------------------------------- | ------------------------------------------------------------------ |
| Definición del modelo| `Sequential()`, `Model()`                             | Construcción de redes neuronales personalizadas                   |
| Capas (Layers)       | `Dense()`, `Dropout()`, `Embedding()`                 | Agregar capas al modelo: fully connected, regularización, etc.     |
| Compilación          | `compile()`                                           | Configurar función de pérdida, optimizador y métricas              |
| Entrenamiento        | `fit()`                                               | Entrenar el modelo con datos etiquetados                          |
| Evaluación           | `evaluate()`                                          | Calcular rendimiento sobre conjunto de prueba                      |
| Predicción           | `predict()`                                           | Obtener salidas del modelo para nuevos datos                       |
| Callbacks            | `EarlyStopping()`, `ModelCheckpoint()`, `TensorBoard()`| Controlar entrenamiento y guardar mejores modelos                |
| Preprocesamiento     | `Normalization()`, `TextVectorization()`, `Tokenizer()`| Escalar, vectorizar texto o preparar entradas                     |
| Guardado y carga     | `save()`, `load_model()`                              | Persistir y reutilizar modelos entrenados                         |
| Visualización        | `history.history`, `matplotlib`                      | Graficar curvas de pérdida, accuracy y validación                 |

## Ejemplo (clasificación binaria)

```bash
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Datos simulados
X_train, y_train = get_training_data()  # Suponiendo (n_samples, n_features)
X_test, y_test = get_test_data()

# Modelo
model = keras.Sequential([
    layers.Dense(32, activation="relu", input_shape=(X_train.shape[1],)),
    layers.Dropout(0.3),
    layers.Dense(1, activation="sigmoid")
])

# Compilación
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Entrenamiento
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

# Evaluación
loss, acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {acc:.2f}")
```

Red neuronal densa ideal para clasificar clientes con alto o bajo valor comercial (lead scoring).

## Buenas prácticas
- Usa callbacks como `EarlyStopping` y `ModelCheckpoint`.
- Normaliza los datos antes de entrenar (`StandardScaler o Normalization() layer`).
- Usa `validation_split` o `validation_data` para evitar sobreajuste.
- Visualiza el historial con matplotlib para ver métricas por época.
- Usa `model.save()` para guardar y `keras.models.load_model()` para reutilizar modelos.

## Recursos

| Tipo           | Referencia                                                                         |
| -------------- | ----------------------------------------------------------------------------------- |
| Documentación  | [https://www.tensorflow.org/api_docs](https://www.tensorflow.org/api_docs)         |
| Tutoriales     | [https://www.tensorflow.org/tutorials](https://www.tensorflow.org/tutorials)       |                                 |
| Proyecto real  | Predicción de churn en ecommerce                                                   |
| Modelos listos | `tf.keras.applications`, `TensorFlow Hub`, `Hugging Face Transformers`             |