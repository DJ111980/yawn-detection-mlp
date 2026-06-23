# Yawn Detection MLP

Clasificador binario de bostezo con imagenes de `64x64`, normalizadas entre `0` y `1` y aplanadas a 4096 caracteristicas. El flujo principal mide el preprocesamiento serial/OpenMP y entrena el mismo MLP en CPU y CUDA con EarlyStopping sobre `val_loss`.

## Ejecutar desde WSL

```bash
conda activate yawn-detection-mlp
cd /mnt/c/Users/USUARIO/Documents/PARALELAS/yawn-detection-mlp
python -m src.train
streamlit run app_streamlit/streamlit_app.py
```

`python -m src.train` es el unico comando de entrenamiento. Prepara el dataset, ejecuta OpenMP, entrena CPU/CUDA, conserva los mejores pesos CUDA segun validacion y genera las metricas. No se requiere ejecutar scripts de entrenamiento separados.

## Resultados

- `metrics/results.md`: accuracy, precision, recall y F1 sobre prueba.
- `metrics/confusion_matrix.png`: matriz de confusion.
- `metrics/openmp_benchmark.md` y `metrics/openmp_speedup.png`: serial vs OpenMP.
- `metrics/cuda_training.md`, `metrics/loss.png`, `metrics/accuracy.png` y `metrics/cuda_speedup.png`: CPU vs CUDA y validacion.

Los archivos JSON de OpenMP y CUDA se conservan solo como fuente de las graficas. Para regenerar visualizaciones y metricas existentes sin entrenar:

```bash
python -m tools.refresh_metrics
```

La aplicacion Streamlit permite subir una imagen, capturar una foto o usar la camara en vivo. Las tres opciones comparten el mismo preprocesamiento y tarjeta de resultado.
