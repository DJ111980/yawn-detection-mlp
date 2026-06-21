from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app_streamlit.utils import (  # noqa: E402
    bgr_to_rgb,
    check_runtime_dependencies,
    classify_probability,
    decode_uploaded_image,
    load_trained_model,
    predict_probability,
    preprocess_image_array,
    read_default_threshold,
)


st.set_page_config(
    page_title="Detección de Bostezo",
    page_icon="camera",
    layout="centered",
)


@st.cache_resource(show_spinner=False)
def get_cached_model():
    return load_trained_model()


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: #f3f6f8;
            color: #172033;
        }
        .main .block-container {
            max-width: 780px;
            padding-top: 2rem;
            padding-bottom: 2.5rem;
        }
        h1, h2, h3, p, span, label {
            color: #172033;
            letter-spacing: 0;
        }
        .app-header {
            text-align: center;
            margin-bottom: 1.35rem;
        }
        .app-header h1 {
            margin-bottom: 0.35rem;
            font-size: 2.15rem;
            font-weight: 800;
            color: #0f172a;
        }
        .app-header p {
            margin: 0 auto;
            max-width: 560px;
            color: #334155;
            font-size: 1rem;
            line-height: 1.5;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: #ffffff;
            border: 1px solid #dce5ee;
            border-radius: 18px;
            box-shadow: 0 16px 38px rgba(15, 23, 42, 0.08);
        }
        .section-title {
            color: #0f172a;
            font-size: 1.05rem;
            font-weight: 750;
            margin: 0 0 0.7rem 0;
        }
        .hint {
            color: #475569;
            font-size: 0.94rem;
            margin: 0.2rem 0 0.8rem 0;
        }
        .result-card {
            border-radius: 18px;
            padding: 1.25rem;
            margin-top: 1rem;
            text-align: center;
        }
        .result-yawn {
            background: #fff1e8;
            border: 1px solid #fb923c;
        }
        .result-clear {
            background: #eaf8ef;
            border: 1px solid #86d39e;
        }
        .result-title {
            color: #0f172a;
            font-size: 1.75rem;
            line-height: 1.15;
            font-weight: 850;
            margin-bottom: 0.55rem;
        }
        .probability {
            color: #263548;
            font-size: 1.16rem;
            font-weight: 700;
        }
        .stButton > button {
            width: 100%;
            border-radius: 12px;
            border: 1px solid #2563eb;
            background: #2563eb;
            color: #ffffff;
            font-weight: 750;
            padding: 0.75rem 1rem;
        }
        .stButton > button:hover {
            border-color: #1d4ed8;
            background: #1d4ed8;
            color: #ffffff;
        }
        div[data-testid="stAlert"] p {
            color: #172033;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def show_result(label: str, probability: float, threshold: float) -> None:
    result_class = "result-yawn" if probability >= threshold else "result-clear"
    st.markdown(
        f"""
        <div class="result-card {result_class}">
            <div class="result-title">{label}</div>
            <div class="probability">Probabilidad: {probability * 100:.1f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    inject_styles()

    st.markdown(
        """
        <div class="app-header">
            <h1>Detección de Bostezo</h1>
            <p>Sube una imagen o usa la cámara para verificar si hay bostezo.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    missing_dependencies = check_runtime_dependencies()
    threshold = read_default_threshold()
    model = None
    model_error = None

    if not missing_dependencies:
        try:
            model, _ = get_cached_model()
        except Exception as exc:
            model_error = exc

    with st.container(border=True):
        st.markdown('<p class="section-title">Selecciona una imagen</p>', unsafe_allow_html=True)

        uploaded_file = None
        camera_file = None
        upload_tab, camera_tab = st.tabs(["Subir imagen", "Cámara"])

        with upload_tab:
            st.markdown(
                '<p class="hint">Elige una imagen facial desde tu computador.</p>',
                unsafe_allow_html=True,
            )
            uploaded_file = st.file_uploader(
                "Imagen",
                type=["jpg", "jpeg", "png", "bmp", "webp"],
                label_visibility="collapsed",
            )

        with camera_tab:
            st.markdown(
                '<p class="hint">Captura una imagen directamente desde la cámara.</p>',
                unsafe_allow_html=True,
            )
            camera_file = st.camera_input("Capturar imagen", label_visibility="collapsed")

        selected_file = camera_file or uploaded_file

        st.markdown('<p class="section-title">Vista previa</p>', unsafe_allow_html=True)
        if selected_file is None:
            st.info("Sube una imagen o captura una foto para comenzar.")
        else:
            try:
                preview_bgr = decode_uploaded_image(selected_file)
                st.image(bgr_to_rgb(preview_bgr), width="stretch")
            except Exception:
                st.error("No se pudo procesar la imagen, inténtalo de nuevo.")
                selected_file = None

        analyze = st.button("Analizar imagen", type="primary")

        if analyze:
            if missing_dependencies:
                st.error("Faltan dependencias necesarias: " + ", ".join(missing_dependencies))
                return

            if model_error is not None:
                st.error(f"No se pudo cargar el modelo entrenado: {model_error}")
                return

            if model is None:
                st.error("No se encontró un modelo disponible para analizar la imagen.")
                return

            if selected_file is None:
                st.warning("Primero sube una imagen o captura una foto con la cámara.")
                return

            try:
                image_bgr = decode_uploaded_image(selected_file)
                input_vector, _ = preprocess_image_array(image_bgr)
                probability = predict_probability(model, input_vector)
                label = classify_probability(probability, threshold)
                show_result(label, probability, threshold)
            except Exception:
                st.error("No se pudo procesar la imagen, inténtalo de nuevo.")


if __name__ == "__main__":
    main()
