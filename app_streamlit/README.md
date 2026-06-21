# Aplicacion Streamlit para deteccion de bostezo

Esta carpeta contiene una app sencilla en Streamlit para usar el modelo entrenado del proyecto `yawn-detection-mlp`.

## Que hace

La app permite:

- Subir una imagen.
- Capturar una imagen con la camara.
- Analizar la imagen con el modelo MLP existente.
- Mostrar si hay bostezo o no.
- Mostrar la probabilidad estimada.

La app no entrena modelos nuevos y no modifica datasets, modelos ni metricas.

## Como ejecutar

Desde la raiz del proyecto:

```powershell
py -m streamlit run app_streamlit/streamlit_app.py
```

## Como probar

1. Ejecuta la app.
2. Elige la pestana `Subir imagen` o `Camara`.
3. Carga o captura una imagen facial.
4. Revisa la vista previa.
5. Presiona `Analizar imagen`.
6. Lee el resultado y la probabilidad.

## Archivos que usa internamente

- `models/best_model`
- `models/final_model`
- `metrics/best_threshold.txt`

## Resultado

Si la probabilidad es igual o mayor al umbral interno, la app muestra:

```text
Bostezo detectado
```

Si la probabilidad es menor, muestra:

```text
No se detecta bostezo
```

## Nota

La opcion de camara usa `st.camera_input`, que permite capturar una foto desde el navegador sin agregar dependencias extra.

