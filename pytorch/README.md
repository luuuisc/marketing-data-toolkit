# PyTorch 2.2 — Guía rápida para aprendizaje profundo

**PyTorch** es una de las principales librerías de deep learning en Python, desarrollada por Meta AI.  
Permite construir y entrenar modelos neuronales de forma flexible y eficiente, tanto en CPU como GPU, y es ampliamente utilizada en investigación, industria y producción.

Referencia: [PyTorch](https://pytorch.org/)


---

## ¿Para qué sirve en marketing?

* **Modelos de predicción de ventas, churn o LTV** con redes neuronales.  
* **Procesamiento de lenguaje natural (NLP)** para análisis de sentimientos o clasificación de reseñas.  
* **Sistemas de recomendación personalizados** para e-commerce.  
* **Visión por computadora** para detección de productos en imágenes o videos.  
* **Fine-tuning de modelos preentrenados** para tareas específicas del negocio.

---

## Instalación

```bash
pip install torch torchvision torchaudio
```

O bien, usar la guía oficial para instalar con soporte CUDA (GPU):
https://pytorch.org/get-started/locally/


## Conceptos clave

| Concepto    | Descripción                                         | Ejemplo en práctica de marketing                  |
| ----------- | --------------------------------------------------- | ------------------------------------------------- |
| `Tensor`    | Estructura de datos multidimensional (como NumPy)   | Representación de imágenes, texto, series de tiempo |
| `Module`    | Unidad base de red neuronal                         | Capa lineal, convolucional o recurrente           |
| `Model`     | Conjunto de capas y operaciones                     | Red para clasificar leads o segmentar audiencias  |
| `Loss`      | Función objetivo para optimizar                     | MSE, CrossEntropy para clasificación               |
| `Optimizer` | Algoritmo para actualizar los pesos                 | SGD, Adam                                          |
| `Dataloader`| Manejo eficiente de batches de datos                | Entrenamiento por lotes en modelos de predicción  |

## Métodos esenciales 

| Categoría         | Método(s)                                      | Uso típico                                                |
| ----------------- | ---------------------------------------------- | --------------------------------------------------------- |
| Definición de modelo | `nn.Module`, `Sequential`                   | Crear arquitecturas de redes neuronales                   |
| Entrenamiento      | `model.train()`, `loss.backward()`, `optimizer.step()` | Entrenar el modelo ajustando pesos                     |
| Evaluación         | `model.eval()`, `with torch.no_grad()`        | Desactivar gradientes para validación o inferencia       |
| Predicción         | `model(x)`, `torch.argmax()`                  | Obtener salidas y clases predichas                       |
| Optimización       | `torch.optim.Adam`, `SGD`, `.zero_grad()`     | Definir y aplicar algoritmo de optimización              |
| Función de pérdida | `nn.CrossEntropyLoss()`, `nn.MSELoss()`       | Calcular error entre predicción y valor real             |
| Datos              | `Dataset`, `DataLoader`, `batch_size`, `shuffle` | Cargar y alimentar datos al modelo por lotes          |
| Guardado/carga     | `torch.save()`, `torch.load()`                | Persistir y restaurar modelos entrenados                 |
| Tensores           | `torch.tensor()`, `.to(device)`, `.view()`, `.reshape()` | Crear, transformar y mover tensores entre CPU y GPU |

## Ejemplo

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 1. Datos simulados
x = torch.randn(100, 5)
y = torch.randint(0, 2, (100,))

# 2. Modelo simple
class MarketingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(5, 1)
    
    def forward(self, x):
        return torch.sigmoid(self.fc(x))

model = MarketingModel()

# 3. Entrenamiento
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    y_pred = model(x).squeeze()
    loss = criterion(y_pred, y.float())
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

Modelo binario simple, útil para tareas como predicción de compra (0/1) o churn. Ajustable para arquitecturas más complejas.

## Buenas prácticas

- Usa `torch.utils.data.Dataset` y `DataLoader` para datasets grandes.
- Usa GPU si está disponible: `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`
- Guarda y carga modelos con `torch.save()` y `torch.load()`.
- Monitorea métricas personalizadas (accuracy, F1, AUC) con scikit-learn.
- Usa `torch.no_grad()` en inferencia para eficiencia y evitar guardar gradientes.

## Recursos

| Tipo              | Referencia                                                                 |
| ----------------- | -------------------------------------------------------------------------- |
| Documentación     | [https://pytorch.org/docs/stable/index.html](https://pytorch.org/docs/stable/index.html) |
| Tutoriales        | [https://pytorch.org/tutorials/](https://pytorch.org/tutorials/)           |

