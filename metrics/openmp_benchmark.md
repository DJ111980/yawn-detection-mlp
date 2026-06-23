# Preprocesamiento: serial vs OpenMP

Alcance: una pasada completa sobre todas las imagenes de entrenamiento, validacion y prueba en memoria. Incluye escala de grises, Gaussiano, resize, normalizacion y flatten. Excluye lectura/escritura de archivos, recorte facial, inferencia y entrenamiento.

| Modo | Hilos | Imagenes | Tiempo (ms) | Speedup |
|---|---:|---:|---:|---:|
| Serial | 1 | 630 | 95.792 | 1.000x |
| OpenMP | 8 | 630 | 24.347 | 3.934x |
