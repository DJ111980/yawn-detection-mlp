# Preprocesamiento

El pipeline Python del modelo realiza: lectura, escala de grises, recorte de región inferior del rostro, Gaussiano 3x3, resize a `64x64`, normalización `0-255 -> 0-1` y flatten a 4096 características.

Durante el entrenamiento TensorFlow se aplican aumentaciones leves solo a entrenamiento: rotación, desplazamiento y cambios de brillo/contraste. Validación, prueba e inferencia no reciben aumentación.

La implementación OpenMP en `src/openmp_preprocess.c` mide las etapas requeridas de gris, Gaussiano, resize, normalización y flatten sobre fotos reales del dataset. Su salida es la entrada del entrenador CUDA.
