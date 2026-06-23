# Metodologia

El objetivo del proyecto es detectar si una persona esta bostezando a partir de imagenes capturadas por camara. La clasificacion es binaria:

- Clase `1`: `yawn`
- Clase `0`: `no_yawn`

La responsabilidad de entrenamiento cubre el preprocesamiento, construccion del MLP, evaluacion, validacion cruzada, seleccion del mejor modelo y exportacion.

## Flujo tecnico

1. Recibir el dataset etiquetado.
2. Revisar distribucion de clases.
3. Aplicar filtros de preprocesamiento.
4. Redimensionar las imagenes a `64x64`.
5. Normalizar pixeles de `0-255` a `0-1`.
6. Convertir cada imagen en un vector de `4096` caracteristicas.
7. Entrenar un MLP con TensorFlow.
8. Evaluar con accuracy, precision, recall, F1 score y matriz de confusion.
9. Aplicar validacion cruzada.
10. Exportar el modelo final para integracion con backend.

## Consideraciones

La clase `no_yawn` debe incluir boca cerrada, sonrisa, conversacion, canto, risa y expresiones neutras. Esto evita que el modelo aprenda una regla incorrecta donde cualquier boca abierta sea interpretada como bostezo.
