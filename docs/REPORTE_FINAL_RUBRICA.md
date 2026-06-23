# Reporte final según rúbrica

## Dataset

El dataset actual tiene 630 imágenes: 315 `yawn` y 315 `no_yawn`, balanceadas de forma exacta. La división es 220/220 para entrenamiento, 47/47 para validación y 48/48 para prueba. El conteo se regenera con:

```bash
python tools/dataset_summary.py
```

La evidencia se presenta en `metrics/dataset_summary.md` y `metrics/dataset_distribution.png`. El proceso, fuentes y criterios de inclusión deben mantenerse en `docs/DATA_COLLECTION.md`.

## Preprocesamiento OpenMP

El pipeline requerido se implementa en `src/openmp_preprocess.c`: escala de grises, Gaussiano 3x3, resize a `64x64`, normalización y flatten a 4096 valores. Se compara la versión serial frente a OpenMP con fotografías reales exportadas desde el dataset.

Evidencias: `metrics/openmp_benchmark.md` y `metrics/openmp_speedup.png`.

### Medicion y speedup OpenMP

El benchmark procesa una pasada completa de todas las imagenes de entrenamiento y prueba ya cargadas en memoria. Mide conversion RGB a escala de grises, filtro Gaussiano, redimensionamiento, normalizacion y flatten. No mide lectura/escritura de archivos, recorte facial, inferencia ni entrenamiento.

La formula es:

```text
speedup_openmp = tiempo_preprocesamiento_serial / tiempo_preprocesamiento_openmp
```

Por tanto, el speedup no corresponde a una sola imagen; corresponde al preprocesamiento completo del conjunto usado en la corrida.

## MLP y CUDA

La arquitectura común es `4096 -> 64 -> 32 -> 1`, con ReLU en las capas ocultas y sigmoid en la salida. TensorFlow puro, sin Keras, proporciona el modelo de referencia e integración con Streamlit.

La implementación CUDA en `src/mlp_cuda.cu` entrena la misma arquitectura con kernels propios de matmul, ReLU, sigmoid, BCE, backpropagation y SGD. El mismo flujo entrena también una versión CPU serial equivalente para comparar tiempos; los pesos CUDA se exportan como el modelo final de Streamlit.

Evidencias: `metrics/cuda_training.md`, `metrics/cuda_training.json`, `metrics/cuda_speedup.png`, `metrics/loss.png` y `metrics/accuracy.png`.

### Medicion y speedup CUDA

CPU y CUDA ejecutan el mismo numero de epocas, los mismos batches y la misma arquitectura. El cronometro cubre el bucle de entrenamiento completo: forward pass, BCE, backpropagation y actualizacion SGD sobre todos los batches de todas las epocas. En CUDA tambien cubre las transferencias de datos de cada batch y la sincronizacion de GPU. Quedan fuera el preprocesamiento OpenMP, la compilacion, la carga inicial de pesos y la evaluacion final.

La formula es:

```text
speedup_cuda = tiempo_entrenamiento_cpu_completo / tiempo_entrenamiento_cuda_completo
```

No es una comparacion de inferencia aislada, de una sola epoca ni de un batch individual.

## Métricas y aplicación

`src.train` genera accuracy, precision, recall, F1-score, matriz de confusión y curvas de pérdida/accuracy. Los resultados se presentan en `metrics/results.md`, `metrics/accuracy.png`, `metrics/loss.png` y `metrics/confusion_matrix.png`.

La aplicación `app_streamlit/streamlit_app.py` permite cargar o capturar una imagen, aplica el pipeline Python idéntico al de entrenamiento y muestra clase más probabilidad.
