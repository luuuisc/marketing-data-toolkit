# Sentiment Analysis for Cosmetic Product Reviews

## Descripción

Este proyecto utiliza **PyTorch** para construir y entrenar un clasificador de sentimiento sobre reseñas de productos de una marca de cosméticos. Con este modelo, el equipo de marketing podrá:

- **Monitorear en tiempo real** si las nuevas reseñas o comentarios en redes sociales son positivos, neutrales o negativos.  
- **Detectar picos de insatisfacción** tras lanzamientos de nuevos productos o campañas publicitarias.  
- **Ajustar mensajes y promociones** de forma dinámica, reaccionando a la percepción del cliente.

---

## Caso práctico: Monitor de sentimiento

Natura lanza una nueva línea de cremas faciales y quiere saber qué opinan sus clientes en sitios de e-commerce y redes sociales. Recopilamos un dataset de reseñas etiquetadas con sentimiento (positivo, neutral, negativo) y construimos un LSTM básico en PyTorch. Tras entrenar y validar el modelo, generamos:

1. **Métricas de desempeño** (precisión, recall, F1) para cada clase.  
2. **Curvas de aprendizaje** (pérdida y accuracy por época).  
3. Un archivo **`sentiment_model.pt`** con los pesos entrenados, listo para servir en producción.

Con ello, Marketing podrá incorporar un pipeline que clasée automáticamente cada nuevo comentario y alerte si > 30 % son negativos en una semana.

## LSTM 
Significa Long Short-Term Memory (“memoria a largo y corto plazo”) y es un tipo de célula de red neuronal recurrente (RNN) diseñada para aprender dependencias en secuencias de datos a diferentes escalas de tiempo. Se introdujo en 1997 por Hochreiter & Schmidhuber para resolver el problema del desvanecimiento del gradiente en RNNs clásicas, que dificulta el aprendizaje de relaciones lejanas en la secuencia.

### Arquitectura del LSTM

Un **LSTM** (Long Short-Term Memory) se compone de dos estados principales y tres puertas que regulan el flujo de información:

1. **Estados**  
   - **Estado de celda** `Cₜ`: transporta información a largo plazo.  
   - **Estado oculto** `hₜ`: salida en cada paso, contiene información de corto plazo.

2. **Puerta de olvido**  

   Decide qué parte del estado previo `Cₜ₋₁` se descarta:  
    fₜ = σ( W_f · [hₜ₋₁, xₜ] + b_f )

3. **Puerta de entrada**  

    Regula la incorporación de nueva información:  
    iₜ = σ( W_i · [hₜ₋₁, xₜ] + b_i )

    Generación del candidato de celda:  
    Ĉₜ = tanh( W_C · [hₜ₋₁, xₜ] + b_C )

4. **Actualización del estado de celda**  

    Combina el viejo estado y el candidato:  
    Cₜ = fₜ ⊙ Cₜ₋₁ + iₜ ⊙ Ĉₜ

5. **Puerta de salida**  

    Controla qué parte de `Cₜ` se emite como estado oculto:  
    oₜ = σ( W_o · [hₜ₋₁, xₜ] + b_o )

    Estado oculto resultante:  
    hₜ = oₜ ⊙ tanh(Cₜ)

- `σ` es la función sigmoide (valores entre 0 y 1).  
- `⊙` indica multiplicación elemento a elemento.  
- `[hₜ₋₁, xₜ]` es la concatenación del estado oculto previo y la entrada actual.  

Esta arquitectura permite al LSTM **mantener** y **actualizar** información relevante a lo largo de secuencias largas, solucionando el problema de desvanecimiento del gradiente de las RNN clásicas.  

### ¿Por qué es útil?
- Mantiene memoria a largo plazo gracias al estado de celda C_t, evitando que la información se “olvide” demasiado rápido
- Control fino de lectura/escritura en la celda a través de las puertas, mejorando la capacidad de modelar secuencias con dependencias lejanas (por ejemplo, texto, series de tiempo).
- Es la base de modelos más avanzados (bidireccionales, apilados, o combinados con atención/transformers).


En el contexto de análisis de sentimiento, un LSTM puede procesar palabra a palabra la reseña de un cliente, reteniendo información relevante de comienzos de la frase (por ejemplo, negaciones o matices) hasta al final, para clasificar correctamente su polaridad.

## Estructura de archivos

```
project/
├── main.py         # Script principal de entrenamiento
├── data/
│   └── reviews.csv            # Dataset: review_text, label
├── models/
│   └── sentiment_model.pt     # Modelo entrenado (output)
├── outputs/
│   ├── metrics.csv            # Matriz de confusión y métricas por clase
│   └── training_log.txt       # Pérdida y accuracy por época
└── README.md                  # Este documento
```

## Requisitos

- Python 3.8+  
- [torch](https://pytorch.org/)  
- [torchtext](https://pytorch.org/text) (o `torchdata`)  
- [scikit-learn](https://scikit-learn.org/)  

## Instalación

```bash
python3 -m venv venv
source venv/bin/activate    # macOS/Linux
venv\Scripts\activate       # Windows


pip install torch torchtext scikit-learn
```

## Formato del CSV de entrada

| Columna       | Tipo    | Descripción                        |
| ------------- | ------- | ---------------------------------- |
| `review_text` | string  | Texto de la reseña o comentario    |
| `label`       | integer | 0 = negativo, 1 = neutral, 2 = positivo |

## Detalles del script
- `load_data(path)`:

    Lee el CSV, tokeniza y construye vocabulario con torchtext.

- `SentimentModel(vocab_size, embed_dim, hidden_dim, output_dim)`:

    Un LSTM bidireccional seguido de una capa lineal.

- `train_loop(model, dataloader, optimizer, criterion)`:

    Entrena por época y registra pérdida y accuracy.

- `evaluate(model, dataloader)`:

    Calcula matriz de confusión y métricas (precision, recall, F1).

- `save_model(model, path)`:

    Guarda `state_dict()` en `sentiment_model.pt`.

## Uso

Para entrenar el modelo y generar los archivos de salida, desde la raíz del proyecto ejecuta:

```bash
python3 main.py \
  --data_path data/reviews.csv \
  --batch_size 32 \
  --epochs 10 \
  --lr 1e-3 \
  --model_out models/sentiment_model.pt \
  --log_out outputs/training_log.txt \
  --metrics_out outputs/metrics.csv
```
- --data_path: Ruta al CSV de reseñas (por ejemplo data/reviews.csv).
- --batch_size: Tamaño de lote para el DataLoader (por defecto 32).
- --epochs: Número de épocas de entrenamiento (por defecto 5).
- --lr: Learning rate del optimizador (por defecto 1e-3).
- --model_out: Ruta donde se guardarán los pesos del modelo entrenado (.pt).
- --log_out: Ruta donde se escribirá el log de entrenamiento (pérdida y precisión por época).
- --metrics_out: Ruta donde se exportará el CSV con métricas de clasificación y matriz de confusión.

Asegúrate de que las carpetas models/ y outputs/ existan antes de ejecutar (puedes crearlas con mkdir -p models outputs).

## Archivos de salida

Al completar la ejecución, el script genera los siguientes archivos:

- `outputs/training_log.txt`

    Contiene un registro línea por línea de cada época, con la pérdida (Loss) y la precisión (Acc):

    Epoch 1: Loss=0.6574, Acc=0.7200

    Epoch 2: Loss=0.5123, Acc=0.8100

    ...

- `outputs/metrics.csv`

    CSV con las métricas de evaluación (precisión, recall, F1-score) para cada clase y la matriz de confusión. Ejemplo de contenido:

|         | precision | recall | f1-score | support |
|---------|-----------|--------|----------|---------|
| 0       | 0.85      | 0.80   | 0.82     | 60      |
| 1       | 0.78      | 0.75   | 0.76     | 60      |
| 2       | 0.89      | 0.92   | 0.90     | 80      |
| accuracy|           |        | 0.82     | 200     |


- `models/sentiment_model.pt`

    Archivo binario con los pesos del modelo entrenado, listo para cargar con PyTorch y servir en producción:

    ```python
    model = SentimentModel(...)
    model.load_state_dict(torch.load("models/sentiment_model.pt"))
    ```

Con estas salidas, podrás:
1. Revisar rápidamente el comportamiento de tu entrenamiento (log).
2. Analizar la calidad de la clasificación en cada clase (metrics.csv).
3. Desplegar el modelo entrenado (.pt) en tu pipeline o servicio de inferencia.

## Importancia del proyecto

El análisis de sentimiento automático aplicado a reseñas de productos cosméticos aporta un valor estratégico muy relevante para el área de marketing de una empresa:

1. **Monitoreo en tiempo real de la percepción de marca**  
   Cada día se publican miles de reseñas, comentarios y opiniones en redes sociales y plataformas de e-commerce. Un clasificador de sentimiento entrenado permite filtrar y agrupar ese feedback de forma inmediata, sin depender de procesos manuales, garantizando una visión actualizada de la satisfacción del cliente.

2. **Detección temprana de problemas y oportunidades**  
   Al establecer alertas cuando un porcentaje significativo de reseñas es negativo, el equipo de producto y calidad puede investigar rápidamente fallos de formulación, fallas en el empaque, o deficiencias en el servicio al cliente. Por otro lado, si las menciones positivas se disparan después de una campaña, se identifica qué mensajes o activos creativos fueron más efectivos.

3. **Optimización de campañas y recursos**  
   Con métricas cuantitativas (precisión, recall, F1) y tendencias temporales, Marketing sabe con mayor certeza en qué segmentos ajustar presupuesto publicitario, qué ingredientes o claims destacar en sus creatividades, y cómo personalizar ofertas según la experiencia real de distintos grupos de usuarios.

4. **Escalabilidad y reproducibilidad**  
   Gracias a PyTorch, el modelo es fácilmente actualizable con nuevos datos y puede integrarse en pipelines de CI/CD o servicios REST. Esto asegura que, a medida que crece el volumen de reseñas, el sistema mantenga su rendimiento y facilite la incorporación de mejoras.

5. **Ventaja competitiva basada en datos**  
   Al automatizar el análisis de sentimiento con técnicas de Deep Learning. Esto fortalece su posicionamiento de marca centrada en el cliente, aprovecha insights que otros competidores pueden pasar por alto y refuerza una cultura organizacional orientada al feedback continuo.

En conjunto, este proyecto transforma datos cualitativos (texto libre) en indicadores cuantificables que alimentan estrategias de marketing, I+D de productos y atención al cliente, contribuyendo a una toma de decisiones más ágil, proactiva y basada en evidencia.  

## Cómo usar el modelo guardado (`.pt`)

Una vez entrenado, el archivo `sentiment_model.pt` contiene únicamente el `state_dict` (los pesos) de la red. Para cargarlo y usarlo en inferencia:

1. **Definir la misma arquitectura de modelo**  
   Debes recrear la clase `SentimentModel` con los mismos parámetros (`vocab_size`, `embed_dim`, `hidden_dim`, `output_dim`) que usaste en entrenamiento.

2. **Construir el vocabulario**  
   El modelo espera índices de tokens según el mismo mapeo `vocab` que construiste en el `SentimentDataset`. Guarda o reconstruye ese diccionario para preprocesar nuevas reseñas.

3. **Ejemplo de script de inferencia**  
   ```python
   import torch
   import torch.nn as nn
   from train_sentiment import SentimentModel  # o donde esté tu clase

   # 1) Parámetros (misma configuración que en training)
   VOCAB_SIZE = len(saved_vocab)      # número de tokens
   EMBED_DIM  = 100
   HIDDEN_DIM = 128
   OUTPUT_DIM = 3                     # negativo, neutral, positivo

   # 2) Instanciar y cargar pesos
   model = SentimentModel(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, OUTPUT_DIM)
   model.load_state_dict(torch.load("models/sentiment_model.pt", map_location="cpu"))
   model.eval()

   # 3) Función de preprocesamiento
   def preprocess(text, vocab, unk_idx=1):
       tokens = text.split()
       indices = [vocab.get(tok.lower(), unk_idx) for tok in tokens]
       return torch.tensor(indices, dtype=torch.long).unsqueeze(0)  # batch_size=1

   # 4) Inferir una nueva reseña
   review = "La crema me deja la piel radiante y suave"
   input_tensor = preprocess(review, saved_vocab)  # shape [1, seq_len]
   lengths = torch.tensor([input_tensor.size(1)])
   with torch.no_grad():
       logits = model(input_tensor, lengths)
       pred = torch.argmax(logits, dim=1).item()

   label_map = {0: "negativo", 1: "neutral", 2: "positivo"}
   print(f"Sentimiento predicho: {label_map[pred]}")
   ```

## Extensiones posibles
- Fine-tuning de un modelo Transformer (BERT, RoBERTa) con transformers.

- Implementar un servicio REST (FastAPI) que reciba texto y devuelva predicción.

- Dashboard interactivo (Streamlit) para visualizar tendencia de sentimiento.

- Entrenamiento continuo (“online learning”) con nuevos comentarios en tiempo real.

## Licencia

MIT License – adapta y mejora este ejemplo para tus flujos de trabajo de marketing.

