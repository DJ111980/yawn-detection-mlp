# Análisis del dataset

El dataset contiene 630 imágenes, distribuidas de forma balanceada: 315 `no_yawn` y 315 `yawn`.

| Split | No bostezo | Bostezo | Total |
|---|---:|---:|---:|
| Train | 220 | 220 | 440 |
| Validation | 47 | 47 | 94 |
| Test | 48 | 48 | 96 |

Cada clase supera el mínimo de 300 imágenes. Ejecuta `python tools/dataset_summary.py` después de modificar imágenes para regenerar `metrics/dataset_summary.md` y la gráfica de distribución.

La variedad de personas, iluminación y ángulos debe verificarse visualmente y documentarse con sus fuentes en `docs/DATA_COLLECTION.md`.
