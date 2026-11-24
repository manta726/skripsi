import warnings
import streamlit as st

warnings.filterwarnings("ignore", category=FutureWarning)


@st.cache_resource(show_spinner="Memuat YOLOv8n…")
def get_yolo():
    from ultralytics import YOLO
    return YOLO("yolov8n.pt")


@st.cache_resource(show_spinner="Memuat InsightFace buffalo_l…")
def get_face_app(det_size=(640, 640)):
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l",
                       allowed_modules=["detection", "recognition"],
                       providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=tuple(det_size))
    return app