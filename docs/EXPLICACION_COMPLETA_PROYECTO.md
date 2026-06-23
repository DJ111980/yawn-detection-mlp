# Explicación del proyecto

El sistema clasifica imágenes faciales en `yawn` y `no_yawn`. El flujo Python usa OpenCV para preprocesar y TensorFlow puro para el MLP de integración, sin usar Keras.

La entrada tiene 4096 valores porque cada imagen final es de 64x64 píxeles en escala de grises. El modelo usa dos capas ReLU de 64 y 32 neuronas, seguidas por una salida sigmoid.

El módulo OpenMP compara serial vs paralelo durante el preprocesamiento de fotos reales. El módulo CUDA implementa y entrena el MLP en CPU y GPU para medir tiempo, pérdida, exactitud y speedup. Los resultados de ambos módulos se muestran como tablas Markdown y gráficas PNG.
