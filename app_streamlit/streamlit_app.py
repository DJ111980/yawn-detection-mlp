"""Interfaz Streamlit para prediccion de bostezo con imagen, foto y camara en vivo."""

from __future__ import annotations

import hashlib
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
    decode_image_bytes,
    load_trained_model,
    predict_probability,
    preprocess_image_array,
)

try:
    import av
    from streamlit_webrtc import webrtc_streamer
except ImportError:  # pragma: no cover - depende del entorno WSL.
    av = None
    webrtc_streamer = None


st.set_page_config(page_title="Deteccion de Bostezo", page_icon="camera", layout="wide")

def build_rtc_configuration() -> dict:
    """Combina STUN publico con TURN privado almacenado fuera del repositorio."""
    configuration = {
        "iceServers": [
            {"urls": ["stun:stun.relay.metered.ca:80"]},
        ],
    }
    try:
        turn = st.secrets["turn"]
        configuration["iceServers"].append(
            {
                "urls": [
                    "turn:standard.relay.metered.ca:80",
                    "turn:standard.relay.metered.ca:80?transport=tcp",
                    "turn:standard.relay.metered.ca:443",
                    "turns:standard.relay.metered.ca:443?transport=tcp",
                ],
                "username": turn["username"],
                "credential": turn["credential"],
            }
        )
    except (KeyError, FileNotFoundError):
        # En WSL local puede no existir secrets.toml; STUN permite conservar la prueba local.
        pass
    return configuration


RTC_CONFIGURATION = build_rtc_configuration()


@st.cache_resource(show_spinner=False)
def get_cached_model():
    """Carga una vez los pesos CUDA para compartirlos entre los tres modos de entrada."""
    return load_trained_model()


def inject_styles() -> None:
    """Define una apariencia clara y consistente para la aplicacion."""
    st.markdown(
        """
        <style>
        .stApp { background: #f4f7f5; color: #17252a; }
        .main .block-container { max-width: 1180px; padding-top: 2rem; padding-bottom: 3rem; }
        h1, h2, h3, p, span, label { color: #17252a; letter-spacing: 0; }
        .app-header { margin-bottom: 1.5rem; }
        .app-header h1 { font-size: 2.1rem; margin: 0; font-weight: 800; color: #0b5d5a; }
        .app-header p { color: #52666a; margin: 0.45rem 0 0; font-size: 1rem; }
        .mode-title { color: #0b5d5a; font-size: 1.05rem; font-weight: 800; margin-bottom: 0.65rem; }
        .mode-hint { color: #52666a; font-size: 0.93rem; margin: 0 0 0.8rem; }
        .result-panel { background: #ffffff; border: 1px solid #d7e2df; border-radius: 8px; padding: 1.25rem; min-height: 280px; }
        .result-state { border-radius: 8px; padding: 0.85rem 1rem; font-weight: 800; margin-bottom: 1.1rem; }
        .state-waiting { background: #eef3f2; color: #52666a; border: 1px solid #d7e2df; }
        .state-yawn { background: #fff0ec; color: #a63a24; border: 1px solid #f4b5a5; }
        .state-clear { background: #e9f6f2; color: #087a65; border: 1px solid #9ed8ca; }
        .result-caption { color: #52666a; font-size: 0.82rem; font-weight: 700; text-transform: uppercase; margin-bottom: 0.22rem; }
        .probability-row { display: flex; justify-content: space-between; gap: 1rem; color: #203338; font-size: 0.96rem; font-weight: 700; margin: 0.9rem 0 0.34rem; }
        .probability-track { height: 12px; background: #e7eeec; border-radius: 6px; overflow: hidden; }
        .probability-fill { height: 100%; border-radius: 6px; }
        .fill-yawn { background: #e76f51; }
        .fill-clear { background: #0f9d8a; }
        .result-note { color: #718287; font-size: 0.88rem; margin-top: 1.2rem; line-height: 1.45; }
        .media-frame { border: 1px solid #d7e2df; border-radius: 8px; overflow: hidden; background: #ffffff; padding: 0.35rem; }
        .media-frame img { max-height: 290px; object-fit: contain; }
        div[data-testid="stCameraInput"] video, div[data-testid="stCameraInput"] img, div[data-testid="stColumn"] video { max-height: 320px; object-fit: cover; }
        .stButton > button { width: 100%; border-radius: 8px; border: 0; background: #0b7a75; color: #ffffff; font-weight: 750; padding: 0.65rem 1rem; }
        .stButton > button:hover { background: #075f5b; color: #ffffff; }
        button[data-baseweb="tab"] p { color: #52666a; font-weight: 700; }
        button[data-baseweb="tab"][aria-selected="true"] p { color: #0b7a75; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_state() -> None:
    for key in (
        "upload_prediction",
        "camera_prediction",
        "upload_signature",
        "camera_signature",
        "upload_prediction_signature",
        "camera_prediction_signature",
        "upload_error",
        "camera_error",
    ):
        st.session_state.setdefault(key, None)


def analyze_image_bytes(model, image_bytes: bytes) -> tuple[str, float]:
    """Aplica el preprocesamiento compartido y devuelve clase junto a probabilidad de bostezo."""
    if not image_bytes:
        raise ValueError("La imagen esta vacia.")
    image_bgr = decode_image_bytes(image_bytes)
    vector, _ = preprocess_image_array(image_bgr)
    probability = predict_probability(model, vector)
    return classify_probability(probability), probability


def prediction_panel(prediction: tuple[str, float] | None, waiting_message: str) -> None:
    """Muestra el resultado y las dos probabilidades en una unica tarjeta reutilizable."""
    if prediction is None:
        st.markdown(
            f"""
            <div class="result-panel">
                <div class="result-state state-waiting">{waiting_message}</div>
                <div class="probability-row"><span>Probabilidad de bostezo</span><span>--</span></div>
                <div class="probability-track"></div>
                <div class="probability-row"><span>Probabilidad de no bostezo</span><span>--</span></div>
                <div class="probability-track"></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    label, yawn_probability = prediction
    no_yawn_probability = 1.0 - yawn_probability
    is_yawn = label == "Bostezo detectado"
    result_class = "state-yawn" if is_yawn else "state-clear"
    st.markdown(
        f"""
        <div class="result-panel">
            <div class="result-caption">Resultado</div>
            <div class="result-state {result_class}">{label}</div>
            <div class="probability-row"><span>Probabilidad de bostezo</span><span>{yawn_probability * 100:.1f}%</span></div>
            <div class="probability-track"><div class="probability-fill fill-yawn" style="width:{yawn_probability * 100:.1f}%"></div></div>
            <div class="probability-row"><span>Probabilidad de no bostezo</span><span>{no_yawn_probability * 100:.1f}%</span></div>
            <div class="probability-track"><div class="probability-fill fill-clear" style="width:{no_yawn_probability * 100:.1f}%"></div></div>
            <div class="result-note">La clase final corresponde a la probabilidad mas alta.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


class LiveYawnProcessor:
    """Procesa frames de camara y comparte la ultima prediccion con la columna de resultados."""

    def __init__(self, model, threshold: float = 0.5, frame_interval: int = 3):
        self.model = model
        self.threshold = threshold
        self.frame_interval = frame_interval
        self.frame_count = 0
        self.prediction: tuple[str, float] | None = None

    def recv(self, frame):
        image_bgr = frame.to_ndarray(format="bgr24")
        self.frame_count += 1
        if self.frame_count % self.frame_interval == 0:
            try:
                # Se reutiliza el mismo preprocesamiento usado por imagen subida y foto.
                vector, _ = preprocess_image_array(image_bgr)
                probability = predict_probability(self.model, vector)
                self.prediction = classify_probability(probability, self.threshold), probability
            except Exception:
                self.prediction = None
        return av.VideoFrame.from_ndarray(image_bgr, format="bgr24")


def live_prediction_panel(context) -> None:
    """Refresca la columna derecha mientras la camara entrega nuevos frames."""
    processor = context.video_processor if context and context.video_processor else None
    prediction_panel(processor.prediction if processor else None, "Esperando que la camara inicie...")


def render_image_mode(title: str, hint: str, source: str, model) -> None:
    """Renderiza el patron comun de dos columnas para archivo subido y foto tomada."""
    left, right = st.columns((1.25, 1), gap="large")
    with left:
        st.markdown(f'<div class="mode-title">{title}</div><p class="mode-hint">{hint}</p>', unsafe_allow_html=True)
        if source == "upload":
            selected = st.file_uploader("Seleccionar imagen", type=["jpg", "jpeg", "png", "bmp", "webp"], label_visibility="collapsed")
        else:
            selected = st.camera_input("Tomar foto", label_visibility="collapsed")

        if selected is not None:
            image_bytes = selected.getvalue()
            if not image_bytes:
                st.session_state[f"{source}_error"] = "La imagen capturada esta vacia. Intenta tomarla de nuevo."
                st.warning("La imagen capturada esta vacia. Intenta tomarla de nuevo.")
                return

            # Una nueva foto o archivo invalida el resultado anterior, nunca reutiliza datos viejos.
            signature = hashlib.sha256(image_bytes).hexdigest()
            if signature != st.session_state[f"{source}_signature"]:
                st.session_state[f"{source}_signature"] = signature
                st.session_state[f"{source}_prediction"] = None
                st.session_state[f"{source}_prediction_signature"] = None
                st.session_state[f"{source}_error"] = None

            if source == "upload":
                try:
                    image_bgr = decode_image_bytes(image_bytes)
                    st.markdown('<div class="media-frame">', unsafe_allow_html=True)
                    st.image(bgr_to_rgb(image_bgr), width=360)
                    st.markdown('</div>', unsafe_allow_html=True)
                except Exception:
                    st.session_state[f"{source}_error"] = "No se pudo leer la imagen subida. Prueba con otro archivo."
                    st.error("No se pudo leer la imagen subida. Prueba con otro archivo.")
                    return
            else:
                st.caption("La foto se muestra directamente en el control de camara.")

            if st.button("Analizar imagen", key=f"analyze_{source}"):
                if st.session_state[f"{source}_prediction_signature"] == signature:
                    # La misma imagen ya fue analizada: se conserva su resultado sin reprocesarla.
                    st.session_state[f"{source}_error"] = None
                else:
                    try:
                        st.session_state[f"{source}_prediction"] = analyze_image_bytes(model, image_bytes)
                        st.session_state[f"{source}_prediction_signature"] = signature
                        st.session_state[f"{source}_error"] = None
                    except Exception as exc:
                        # Un fallo puntual no borra una prediccion correcta obtenida antes.
                        st.session_state[f"{source}_error"] = (
                            "No se pudo procesar este intento. Se conserva el ultimo resultado valido. "
                            f"Detalle: {type(exc).__name__}."
                        )
        else:
            st.session_state[f"{source}_prediction"] = None
            st.session_state[f"{source}_signature"] = None
            st.session_state[f"{source}_prediction_signature"] = None
            st.session_state[f"{source}_error"] = None
            st.info("Selecciona una imagen para comenzar.")

    with right:
        if st.session_state[f"{source}_error"]:
            st.warning(st.session_state[f"{source}_error"])
        prediction_panel(st.session_state[f"{source}_prediction"], "Esperando una imagen para analizar...")


def main() -> None:
    init_state()
    inject_styles()
    st.markdown(
        """
        <div class="app-header">
            <h1>Deteccion de Bostezo</h1>
            <p>Analiza una imagen, toma una foto o activa la camara en vivo.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    missing = check_runtime_dependencies()
    model = None
    model_error = None
    if not missing:
        try:
            model, _ = get_cached_model()
        except Exception as exc:
            model_error = exc

    if missing or model_error is not None or model is None:
        message = ", ".join(missing) if missing else str(model_error or "No se encontro el modelo CUDA.")
        st.error(f"La aplicacion no puede cargar el modelo: {message}")
        return

    upload_tab, camera_tab, live_tab = st.tabs(["Subir imagen", "Capturar foto", "Camara en vivo"])
    with upload_tab:
        render_image_mode("Imagen subida", "Elige una imagen facial desde tu equipo.", "upload", model)
    with camera_tab:
        render_image_mode("Captura de foto", "Toma una foto con la camara del navegador.", "camera", model)
    with live_tab:
        left, right = st.columns((1.25, 1), gap="large")
        with left:
            st.markdown('<div class="mode-title">Camara en vivo</div><p class="mode-hint">Permite el acceso a la camara cuando lo solicite el navegador.</p>', unsafe_allow_html=True)
            if webrtc_streamer is None or av is None:
                st.error("La camara en vivo requiere streamlit-webrtc y av en el entorno WSL.")
                live_context = None
            else:
                live_context = webrtc_streamer(
                    key="yawn-live-camera",
                    video_processor_factory=lambda: LiveYawnProcessor(model),
                    rtc_configuration=RTC_CONFIGURATION,
                    media_stream_constraints={"video": True, "audio": False},
                    async_processing=True,
                    desired_playing_state=True,
                )
        with right:
            if hasattr(st, "fragment"):
                @st.fragment(run_every=1.0)
                def refresh_live_result() -> None:
                    live_prediction_panel(live_context)

                refresh_live_result()
            else:
                live_prediction_panel(live_context)


if __name__ == "__main__":
    main()
