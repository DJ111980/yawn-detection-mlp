# Aplicacion Streamlit

La app usa `models/cuda_weights.bin` para predecir `yawn` o `no_yawn`. Ofrece tres modos: imagen subida, foto tomada y camara en vivo.

Los tres modos usan el mismo preprocesamiento y una distribucion de dos columnas: media a la izquierda y resultado a la derecha. El resultado muestra las probabilidades de bostezo y no bostezo con barras horizontales.

## Ejecutar desde WSL

```bash
conda activate yawn-detection-mlp
cd /mnt/c/Users/USUARIO/Documents/PARALELAS/yawn-detection-mlp
streamlit run app_streamlit/streamlit_app.py
```

Abre `http://localhost:8501` y permite el acceso a la camara cuando el navegador lo solicite. La camara en vivo actualiza el panel de probabilidades aproximadamente cada segundo.

## TURN para despliegue publico

La camara en vivo usa las credenciales TURN guardadas en `st.secrets`, nunca en el codigo. En Streamlit Cloud agrega la seccion `[turn]` con `username` y `credential`. Para probar localmente, copia `.streamlit/secrets.toml.example` como `.streamlit/secrets.toml`, completa los valores y no subas ese archivo a Git.
