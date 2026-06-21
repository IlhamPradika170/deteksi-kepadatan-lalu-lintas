"""
Aplikasi Web - Deteksi & Klasifikasi Kepadatan Lalu Lintas
=============================================================
Mendukung: Gambar (.jpg/.png) dan Video (.mp4/.avi) - Deteksi Penuh Tanpa ROI
"""

import streamlit as st
from pathlib import Path
import json
import numpy as np
from PIL import Image
import cv2
import tempfile
from ultralytics import YOLO

# ===================== KONFIGURASI HALAMAN =====================
st.set_page_config(
    page_title="Deteksi Kepadatan Lalu Lintas",
    page_icon="🚗",
    layout="centered",
)

# ===================== MUAT MODEL & KONFIGURASI =====================
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
    "Unggah citra atau video jalan raya untuk mendeteksi kendaraan "
    "dan mengetahui tingkat kepadatan lalu lintas secara otomatis pada keseluruhan layar."
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

uploaded_file = st.file_uploader("Pilih file (.jpg/.png/.mp4/.avi)", type=["jpg", "jpeg", "png", "mp4", "avi"])

if uploaded_file is not None:
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    # ------------------ SEKTOR PROSES GAMBAR STATIS ------------------
    if file_extension in ['jpg', 'jpeg', 'png']:
        image = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(image)

        with st.spinner("Mendeteksi kendaraan pada gambar..."):
            results = model.predict(source=img_array, conf=confidence, verbose=False)
            n_vehicles = len(results[0].boxes)
            density_class = classify_density(n_vehicles, config)
            # Menggunakan plot() bawaan YOLOv8 untuk deteksi penuh satu layar
            annotated = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Citra Asli", use_container_width=True)
        with col2:
            st.image(annotated, caption="Hasil Deteksi Penuh", use_container_width=True)

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

    # ------------------ SEKTOR PROSES VIDEO (REBUILDING METHOD) ------------------
    elif file_extension in ['mp4', 'avi']:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_file.read())
        
        cap = cv2.VideoCapture(tfile.name)
        
        # Mengambil spesifikasi asli dari video input
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Menyiapkan VideoWriter untuk merakit file video baru (.mp4)
        out_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_file.name, fourcc, fps, (width, height))
        
        st.info("Server sedang mendeteksi dan merakit ulang video Anda. Mohon tunggu sampai selesai...")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            progress_bar.progress(min(frame_count / total_frames, 1.0))
            
            # Prediksi YOLOv8 pada frame saat ini
            results = model.predict(source=frame, conf=confidence, verbose=False)
            n_vehicles = len(results[0].boxes)
            density_class = classify_density(n_vehicles, config)
            
            # Ambil hasil gambar kotak deteksi penuh dari YOLOv8 (format BGR)
            annotated_frame = results[0].plot()
            
            # Tulis frame ke dalam video baru
            out.write(annotated_frame)
            
            status_text.markdown(
                f"**Merakit Frame {frame_count} dari {total_frames}** | "
                f"Kendaraan saat ini: **{n_vehicles}** | Kepadatan: **{density_class}**"
            )
            
        cap.release()
        out.release()
        progress_bar.progress(1.0)
        
        st.success("✅ Pemrosesan dan perakitan video selesai!")
        
        # Menampilkan video yang sudah berhasil dijahit ulang secara mulus
        st.video(out_file.name)
        
        # Menyediakan tombol download untuk menyimpan hasil video ke lokal laptop
        with open(out_file.name, 'rb') as v_file:
            st.download_button(
                label="⬇️ Download Video Hasil Deteksi",
                data=v_file,
                file_name="hasil_deteksi_penuh.mp4",
                mime="video/mp4"
            )
else:
    st.info("Silakan unggah citra atau video untuk memulai deteksi.")

st.markdown("---")
st.caption("Model: YOLOv8 (transfer learning) | Dataset: Top-View Vehicle Detection (Kaggle)")
