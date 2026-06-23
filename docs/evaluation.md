# Evaluacion

El conjunto `validation` decide la mejor epoca con `val_loss`; el conjunto `test` se usa una sola vez al final para reportar accuracy, precision, recall, F1 y matriz de confusion.

Despues de una corrida manual se consultan:

- `metrics/results.md`
- `metrics/confusion_matrix.png`
- `metrics/loss.png`
- `metrics/accuracy.png`

Las curvas diferencian entrenamiento y validacion. Una mejora en entrenamiento que empeora validacion es una senal de sobreajuste.
