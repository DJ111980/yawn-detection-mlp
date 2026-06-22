# Yawn Detection MLP

Sistema de vision artificial para clasificar imagenes en dos clases:

- `1` = yawn
- `0` = no_yawn

El modelo principal es un MLP implementado con TensorFlow puro, sin Keras. Las imagenes se preprocesan en escala de grises, recorte de boca, suavizado Gaussiano ligero, redimensionamiento a `80x80`, normalizacion de pixeles de `0-255` a `0-1` y vectorizacion a `6400` caracteristicas.

## Estructura

```text
yawn-detection-mlp/
|-- datasets/
|-- notebooks/
|-- src/
|-- models/
|-- metrics/
|-- docs/
|-- parallel/
|-- cuda/
|-- tools/
|-- requirements.txt
|-- environment.yml
`-- README.md
```

## Instalacion

Con Miniconda:

```bash
conda env create -f environment.yml
conda activate yawn-detection-mlp
jupyter lab
```

Tambien se puede instalar con `pip`:

```bash
pip install -r requirements.txt
```

## Dataset esperado

```text
datasets/
|-- train/
|   |-- yawn/
|   `-- no_yawn/
|-- validation/
|   |-- yawn/
|   `-- no_yawn/
`-- test/
    |-- yawn/
    `-- no_yawn/
```

## Entrenamiento

Desde la raiz del proyecto:

```bash
python -m src.train
```

## Validacion cruzada

```bash
python -m src.cross_validation
```

## Probar una imagen nueva

```bash
python -m src.predict_image ruta/a/tu_imagen.jpg
```

La salida muestra la probabilidad de `yawn` y la clase final.

## Probar con camara

```bash
python -m src.webcam_demo
```

Presiona `q` para cerrar la ventana.

## Salidas

- Modelo final exportado con TensorFlow SavedModel: `models/final_model/`
- Mejor modelo exportado con TensorFlow SavedModel: `models/best_model/`
- Graficas: `metrics/accuracy.png`, `metrics/loss.png`
- Matriz de confusion: `metrics/confusion_matrix.png`
- Resultados: `metrics/results.csv`

## Analisis del dataset

```powershell
python tools/dataset_summary.py
```

Genera:

```text
metrics/dataset_summary.csv
metrics/dataset_distribution.png
```

Documento:

```text
docs/DATASET_ANALYSIS.md
```

## OpenMP

Preprocesamiento serial y paralelo en C puro con OpenMP:

```powershell
cd parallel/openmp
mingw32-make
.\benchmark_openmp.exe 160 120 5
cd ..\..
py parallel/scripts/plot_openmp_speedup.py
```

Evidencias:

```text
metrics/openmp_benchmark.csv
metrics/openmp_speedup.png
```

## CUDA

Forward pass del MLP con kernels CUDA propios, estilo CUDA C, y benchmark CPU/GPU:

```powershell
cd cuda
mingw32-make
.\benchmark_cuda.exe
cd ..
py cuda/plot_cuda_speedup.py
```

En Windows, `nvcc` puede requerir Visual Studio Build Tools y `cl.exe`. En este proyecto esa configuracion ya quedo resuelta y el benchmark CUDA se ejecuto correctamente.

Evidencias CUDA generadas:

```text
metrics/cuda_benchmark.csv
metrics/cuda_speedup.png
```

Resultado real registrado:

```text
batch 1  -> speedup 24.585x
batch 8  -> speedup 358.551x
batch 16 -> speedup 398.811x
batch 32 -> speedup 349.470x
```

## Documentacion academica

```text
docs/ANALISIS_METRICAS.md
docs/REPORTE_FINAL_RUBRICA.md
docs/GUIA_EXPOSICION.md
```
