# Scikit-learn 1.4 — Guía rápida para modelos de machine learning

**Scikit-learn** es la librería más utilizada en Python para aprendizaje automático clásico.  
Ofrece herramientas simples y eficientes para clasificación, regresión, clustering, reducción de dimensionalidad y validación cruzada, con una API unificada y fácil de usar.

Referencia: [Scikit-learn](https://scikit-learn.org/)


---

## ¿Para qué sirve en marketing?

* **Clasificación de leads** como fríos o calientes.  
* **Regresión para estimar ventas, costos o lifetime value (LTV).**  
* **Segmentación de clientes** con clustering no supervisado.  
* **Reducción de variables** para visualizar datos de campañas.  
* **Modelado de churn**, scoring de propensión, análisis de A/B tests.

---

## Instalación

```bash
pip install scikit-learn
```

Requiere Python ≥ 3.8.
También disponible vía conda install -c conda-forge scikit-learn.


## Conceptos clave

| Concepto       | Descripción                                       | Ejemplo en marketing                         |
| -------------- | ------------------------------------------------- | -------------------------------------------- |
| `Estimator`    | Algoritmo de ML (regresión, clasificación, etc.)  | `LogisticRegression`, `RandomForestClassifier` |
| `fit()`        | Ajusta el modelo a los datos                      | Entrenamiento con datos históricos            |
| `predict()`    | Predice nuevos valores                            | Clasificación de leads                        |
| `transform()`  | Transforma datos (p. ej., reducción dimensional)  | PCA para análisis exploratorio                |
| `Pipeline`     | Flujo completo de preprocesamiento + modelo       | Normalización + regresión en un solo objeto   |
| `GridSearchCV` | Optimización de hiperparámetros                   | Búsqueda del mejor modelo                     |

## Métodos esenciales 

| Categoría            | Método(s)                                     | Uso típico                                               |
| -------------------- | --------------------------------------------- | -------------------------------------------------------- |
| Entrenamiento        | `fit()`                                       | Ajustar el modelo a los datos                            |
| Predicción           | `predict()`, `predict_proba()`                | Predecir clases o probabilidades                         |
| Evaluación           | `score()`, `accuracy_score()`, `roc_auc_score()` | Medir el rendimiento del modelo                      |
| Validación cruzada   | `cross_val_score()`, `cross_validate()`       | Validar rendimiento con K-Fold                          |
| Optimización         | `GridSearchCV()`, `RandomizedSearchCV()`      | Buscar los mejores hiperparámetros                      |
| Preprocesamiento     | `StandardScaler()`, `OneHotEncoder()`, `SimpleImputer()` | Escalado, codificación y manejo de valores nulos |
| Pipeline             | `Pipeline()`, `.fit_transform()`              | Encadenar pasos de preprocesamiento y modelado          |
| División de datos    | `train_test_split()`                          | Separar datos en entrenamiento y prueba                 |
| Reducción de dimensión | `PCA()`, `SelectKBest()`                    | Reducir variables o seleccionar las más relevantes       |
| Clustering (sin supervisión) | `KMeans()`, `DBSCAN()`                | Agrupar clientes o comportamientos similares             |

## Ejemplo

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Datos ficticios
X, y = get_marketing_data()  # Función imaginaria

# División
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelo
model = LogisticRegression()
model.fit(X_train, y_train)

# Predicciones
y_pred = model.predict(X_test)

# Evaluación
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.2f}")
```

Caso de clasificación binaria útil para campañas de email marketing, predicción de compra o churn.

## Buenas prácticas
- Usa Pipeline para encadenar preprocesadores y modelos.
- Escala tus datos con StandardScaler, especialmente para SVM o KNN.
- Evalúa con cross_val_score o GridSearchCV.
- No mezcles datos de entrenamiento/test en ningún paso de ingeniería de variables.
- Usa `confusion_matrix`, `classification_report` o `roc_auc_score` para evaluación detallada.

## Recursos

| Tipo           | Referencia                                                                        |
| -------------- | ---------------------------------------------------------------------------------- |
| Documentación  | [https://scikit-learn.org/stable/](https://scikit-learn.org/stable/)              |
| Tutoriales     | [https://scikit-learn.org/stable/tutorial/](https://scikit-learn.org/stable/tutorial/) |
