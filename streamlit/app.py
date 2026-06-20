"""
Aplikasi Web - Deteksi & Klasifikasi Kepadatan Lalu Lintas
=============================================================
Antarmuka Streamlit: upload citra top-view -> deteksi kendaraan ->
klasifikasi kepadatan (Padat / Sedang / Sepi)

Cara menjalankan lokal:
    streamlit run app.py

File pendukung yang HARUS ada di folder yang sama dengan app.py:
    - best.pt               (bobot model YOLOv8 hasil training)
    - config_kepadatan.json (threshold kepadatan hasil tahap sebelumnya)
"""

import streamlit as st
from pathlib import Path
import json
import numpy as np
from PIL import Image
import cv2
from ultralytics import YOLO

# ===================== KONFIGURASI HALAMAN =====================
st.set_page_config(
    page_title="Deteksi Kepadatan Lalu Lintas",
    page_icon="🚗",
    layout="centered",
)


# ===================== MUAT MODEL & KONFIGURASI (DI-CACHE) =====================
@st.cache_resource
def load_model_and_config():
    config_path = Path("config_kepadatan.json")
    if not config_path.exists():
        st.error("File config_kepadatan.json tidak ditemukan di folder aplikasi.")
        st.stop()
    with open(config_path) as f:
        cfg = json.load(f)

    model_path = Path("best.pt")
    if not model_path.exists():
        st.error("File best.pt (bobot model) tidak ditemukan di folder aplikasi.")
        st.stop()
    loaded_model = YOLO(str(model_path))
    return loaded_model, cfg


model, config = load_model_and_config()


# ===================== FUNGSI KLASIFIKASI KEPADATAN =====================
def classify_density(n_vehicles, cfg):
    if n_vehicles < cfg["threshold_sepi_sedang"]:
        return "Sepi"
    elif n_vehicles < cfg["threshold_sedang_padat"]:
        return "Sedang"
    else:
        return "Padat"


DENSITY_COLOR = {
    "Sepi": "#4CAF50",
    "Sedang": "#FFC107",
    "Padat": "#F44336",
}

# ===================== ANTARMUKA UTAMA =====================
st.title("🚗 Deteksi & Klasifikasi Kepadatan Lalu Lintas")
st.write(
    "Unggah citra top-view (CCTV/drone) jalan raya untuk mendeteksi kendaraan "
    "dan mengetahui tingkat kepadatan lalu lintas secara otomatis."
)

with st.sidebar:
    st.header("Pengaturan")
    confidence = st.slider(
        "Confidence threshold deteksi",
        min_value=0.1, max_value=0.9,
        value=float(config.get("confidence", 0.25)), step=0.05,
    )
    st.markdown("---")
    st.subheader("Ambang Batas Kepadatan")
    st.write(f"Sepi   : < {config['threshold_sepi_sedang']:.0f} kendaraan")
    st.write(f"Sedang : {config['threshold_sepi_sedang']:.0f} – {config['threshold_sedang_padat']:.0f} kendaraan")
    st.write(f"Padat  : > {config['threshold_sedang_padat']:.0f} kendaraan")

uploaded_file = st.file_uploader("Pilih citra (.jpg/.png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    with st.spinner("Mendeteksi kendaraan..."):
        results = model.predict(source=img_array, conf=confidence, verbose=False)
        n_vehicles = len(results[0].boxes)
        density_class = classify_density(n_vehicles, config)
        annotated = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Citra Asli", use_container_width=True)
    with col2:
        st.image(annotated, caption="Hasil Deteksi", use_container_width=True)

    st.markdown("---")
    c1, c2 = st.columns(2)
    c1.metric("Jumlah Kendaraan Terdeteksi", n_vehicles)
    c2.markdown(
        f"""
        <div style="background-color:{DENSITY_COLOR[density_class]};
                    padding:14px; border-radius:8px; text-align:center;">
            <span style="color:white; font-size:20px; font-weight:bold;">
                Kepadatan: {density_class}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.info("Silakan unggah citra untuk memulai deteksi.")

st.markdown("---")
st.caption("Model: YOLOv8 (transfer learning) | Dataset: Top-View Vehicle Detection (Kaggle)")
