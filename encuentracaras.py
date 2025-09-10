import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageOps
import io
import os
import base64
from datetime import timedelta

try:
    from deepface import DeepFace
    DEEPFACE_OK = True
except Exception as e:
    DEEPFACE_OK = False
    DEEPFACE_ERR = str(e)

st.set_page_config(
    page_title="Buscador de Rostros",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/your-repo/issues",
        "Report a bug": "mailto:you@example.com",
        "About": """
            ## Aplicación Buscador de Rostros  
            Sube una imagen de referencia de un rostro y un video.  
            La aplicación buscará ese rostro en el video y mostrará las coincidencias con marcas de tiempo.
        """
    }
)

st.markdown("<h1 style='text-align: center; color: #FFFFFF;'>Buscador de Rostros</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: #FFFFFF;font-size: 18px;'>Sube una imagen de referencia y un video. El sistema buscará el rostro en el video y reportará los instantes detectados</h2>", unsafe_allow_html=True)

if not DEEPFACE_OK:
    st.error("DeepFace no está disponible. Instala con `pip install deepface`. Error: " + DEEPFACE_ERR)

MODEL_OPTIONS = ["ArcFace", "VGG-Face", "Facenet", "Facenet512", "OpenFace", "SFace", "Dlib"]
DETECTOR_OPTIONS = ["retinaface", "mtcnn", "opencv", "ssd", "dlib", "mediapipe"]

RECO_THRESH = {"ArcFace":0.35, "SFace":0.40, "Facenet":0.80, "Facenet512":0.75, "VGG-Face":0.40, "OpenFace":0.45, "Dlib":0.50}

# --- Inicializar valores en session_state ---
if "model_name" not in st.session_state:
    st.session_state.model_name = MODEL_OPTIONS[0]
if "detector_backend" not in st.session_state:
    st.session_state.detector_backend = DETECTOR_OPTIONS[0]
if "thr" not in st.session_state:
    st.session_state.thr = RECO_THRESH.get(st.session_state.model_name, 0.40)
if "seconds_per_sample" not in st.session_state:
    st.session_state.seconds_per_sample = 1.0
if "max_frames" not in st.session_state:
    st.session_state.max_frames = 5000
if "show_debug" not in st.session_state:
    st.session_state.show_debug = False

with st.sidebar:
    with st.expander("⚙️ Configuración"):
        st.selectbox("Modelo de embeddings", MODEL_OPTIONS, key="model_name")
        st.selectbox("Detector de rostros", DETECTOR_OPTIONS, key="detector_backend")
        st.slider("Umbral de coincidencia (distancia coseno)", 0.20, 0.80, step=0.01, key="thr")
        st.slider("Segundos entre fotogramas analizados", 0.2, 5.0, step=0.1, key="seconds_per_sample")
        st.number_input("Máximo de fotogramas a analizar", 100, 100000, step=100, key="max_frames")
        st.checkbox("Mostrar información de depuración", key="show_debug")

    with st.expander("ℹ️ Información"):
        st.write("Esta aplicación detecta un rostro de referencia dentro de un video.")
        st.markdown("""
        1. Sube una **imagen de referencia** con un rostro claro.
        2. Sube un **archivo de video**.
        3. Ajusta las configuraciones en *Configuración*.
        4. Pulsa **Iniciar Búsqueda** para comenzar el análisis.
        """)

    run_btn = st.button("▶️ Iniciar Búsqueda", type="primary", use_container_width=True)

model_name = st.session_state.model_name
detector_backend = st.session_state.detector_backend
thr = st.session_state.thr
seconds_per_sample = st.session_state.seconds_per_sample
max_frames = st.session_state.max_frames
show_debug = st.session_state.show_debug

col1, col2 = st.columns([1, 1])
with col1:
    ref_file = st.file_uploader("Imagen de referencia (un rostro)", type=["jpg", "jpeg", "png", "webp"])
    if ref_file:
        ref_image_pil = Image.open(ref_file)
        ref_image_pil = ImageOps.exif_transpose(ref_image_pil).convert("RGB")
        st.image(ref_image_pil, caption="Imagen de referencia (auto-orientada)")

with col2:
    video_file = st.file_uploader("Archivo de video", type=["mp4", "mov", "avi", "mkv", "webm"])
    if video_file:
        video_bytes = video_file.getvalue()
        b64_video = base64.b64encode(video_bytes).decode()
        video_html = f'''
            <video id="videoPlayer" width="100%" controls>
                <source src="data:video/mp4;base64,{b64_video}" type="video/mp4">
            </video>
        '''
        st.markdown(video_html, unsafe_allow_html=True)

def l2_normalize(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32)
    n = np.linalg.norm(vec) + eps
    return vec / n

@st.cache_data(show_spinner=False)
def bytes_to_cv(img_bytes: bytes) -> np.ndarray:
    pil = Image.open(io.BytesIO(img_bytes))
    pil = ImageOps.exif_transpose(pil).convert("RGB")
    return np.array(pil)

@st.cache_resource(show_spinner=False)
def load_reference_embedding(img_rgb: np.ndarray, model: str, detector: str):
    detections = DeepFace.extract_faces(img_path=img_rgb, detector_backend=detector, enforce_detection=True)
    if not detections:
        raise RuntimeError("No se encontró ningún rostro en la imagen de referencia.")

    det = sorted(detections, key=lambda d: float(d.get("confidence", 0)), reverse=True)[0]
    face_rgb = det["face"]
    if isinstance(face_rgb, tuple):
        face_rgb = face_rgb[0]
    face_rgb = np.asarray(face_rgb)
    if face_rgb.dtype != np.uint8:
        face_rgb = (face_rgb * 255).astype("uint8")

    rep = DeepFace.represent(face_rgb, model_name=model, detector_backend="skip", enforce_detection=False)
    if not isinstance(rep, list) or "embedding" not in rep[0]:
        raise RuntimeError("No se pudo calcular el embedding del rostro.")

    emb = l2_normalize(np.array(rep[0]["embedding"], dtype=np.float32))
    return emb

if run_btn:
    if not (DEEPFACE_OK and ref_file and video_file):
        st.warning("Por favor, sube una imagen de referencia y un video, y asegúrate de que DeepFace está instalado.")
    else:
        ref_rgb = bytes_to_cv(ref_file.getvalue())
        try:
            ref_emb = load_reference_embedding(ref_rgb, model_name, detector_backend)
        except Exception as exc:
            st.error(f"Error en imagen de referencia: {exc}")
            st.stop()

        tmp_video_path = f"_vid_{np.random.randint(1e9)}.mp4"
        with open(tmp_video_path, "wb") as f:
            f.write(video_file.getvalue())

        cap = cv2.VideoCapture(tmp_video_path)
        if not cap.isOpened():
            st.error("No se pudo abrir el video. Prueba con otro formato.")
            os.remove(tmp_video_path)
            st.stop()

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        sample_every = max(1, int(seconds_per_sample * fps))

        st.info(f"FPS: {fps:.2f} | Fotogramas totales: {total_frames} | Analizando cada {sample_every} fotogramas")
        prog = st.progress(0)
        status = st.empty()


        detections = []
        checked = 0
        frame_idx = 0

        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            if frame_idx % sample_every == 0:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                try:
                    faces = DeepFace.extract_faces(img_path=frame_rgb, detector_backend=detector_backend, enforce_detection=False)
                    for det in faces:
                        face_rgb = det.get("face", None)
                        if face_rgb is None:
                            continue
                        if isinstance(face_rgb, tuple):
                            face_rgb = face_rgb[0]
                        if face_rgb.dtype != np.uint8:
                            face_rgb = (face_rgb * 255).astype("uint8")

                        rep = DeepFace.represent(face_rgb, model_name=model_name, detector_backend="skip", enforce_detection=False)
                        if not isinstance(rep, list) or "embedding" not in rep[0]:
                            continue

                        emb = l2_normalize(np.array(rep[0]["embedding"], dtype=np.float32))
                        cos_sim = float(np.dot(ref_emb, emb))
                        dist = 1.0 - cos_sim

                        if dist <= thr:
                            seconds = frame_idx / max(fps, 1e-6)
                            face_img = Image.fromarray(cv2.cvtColor(face_rgb, cv2.COLOR_BGR2RGB))  # ✅ fixed
                            detections.append({
                                "frame": frame_idx,
                                "tiempo": seconds,
                                "codigo_tiempo": str(timedelta(seconds=seconds)),
                                "distancia": dist,
                                "similitud": cos_sim,
                                "snapshot": face_img
                            })

                except Exception as e:
                    if show_debug:
                        st.write(f"Error en fotograma {frame_idx}: {e}")

                checked += 1
                if max_frames and checked >= int(max_frames // 1):
                    break
                if total_frames > 0:
                    prog.progress(min(1.0, frame_idx / total_frames))
                status.write(f"Fotogramas analizados: {checked} (fotograma {frame_idx}) | Coincidencias: {len(detections)}")

            frame_idx += 1

        cap.release()
        os.remove(tmp_video_path)  # Eliminar el archivo de video temporal

        st.success(f"Búsqueda finalizada. Coincidencias encontradas: {len(detections)}")
        if detections:
            st.write("### Resultados de coincidencias")
            for det in detections:
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.image(det["snapshot"], caption=f"Frame {det['frame']}")
                with col2:
                    st.write(f"⏱ {det['codigo_tiempo']} (Frame {det['frame']})")
                    st.write(f"Similitud: {det['similitud']*100:.1f}%")
