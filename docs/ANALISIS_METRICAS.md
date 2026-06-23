# Analisis de metricas

Use `metrics/results.md` y la matriz de confusion para identificar falsos positivos y falsos negativos. Para el reporte final, explique los errores con ejemplos concretos: recorte de boca, iluminacion, postura, rostro parcialmente visible o expresiones similares a un bostezo.

Las curvas de `loss` y `accuracy` permiten comparar entrenamiento contra validacion. La mejor epoca es la de menor `val_loss`, no necesariamente la ultima ni la de mayor accuracy de entrenamiento.
