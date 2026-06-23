# Entrenamiento

El modelo principal es `4096 -> 64 -> 32 -> 1`, con ReLU en las capas ocultas y sigmoid en la salida. CUDA entrena el modelo final; CPU ejecuta la misma cantidad de trabajo para medir speedup.

La validacion se revisa despues de cada epoca. EarlyStopping monitorea `val_loss`, usa paciencia 8 y restaura los pesos de la mejor epoca antes de evaluar prueba y exportar `models/cuda_weights.bin`.

```bash
python -m src.train
```

No hay un segundo comando de entrenamiento. El reporte de cada corrida queda en `metrics/cuda_training.md`.
