"""
Aplikasi Web - Deteksi & Klasifikasi Kepadatan Lalu Lintas
=============================================================
Mendukung: Gambar (.jpg/.png) dan Video (.mp4/.avi) dengan Filter ROI
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
st.set_page_config(page_title="Deteksi Kepadatan Lalu Lintas", page_icon="🚗", layout="centered")

# ===================== MUAT MODEL & KONFIGURASI =====================
@st.cache_resource
def load_model_and_config():
    config_path = Path("config_kepadatan.json")
    if not config_path.exists():
        st.error("File config_kepadatan.json tidak ditemukan.")
        st.stop()
    with open(config_path) as f:
        cfg = json.load(f)

    model_path = Path("best.pt")
    if not model_path.exists():
        st.error("File best.pt tidak ditemukan.")
        st.stop()
    loaded_model = YOLO(str(model_path))
    return loaded_model, cfg

model, config = load_model_and_config()

def classify_density(n_vehicles, cfg):
    if n_vehicles < cfg["threshold_sepi_sedang"]: return "Sepi"
    elif n_vehicles < cfg["threshold_sedang_padat"]: return "Sedang"
    else: return "Padat"

DENSITY_COLOR = {"Sepi": "#4CAF50", "Sedang": "#FFC107", "Padat": "#F44336"}

# ===================== ANTARMUKA UTAMA =====================
st.title("🚗 Deteksi Kepadatan Lalu Lintas (Gambar & Video)")
st.write("Unggah citra atau video jalan raya untuk mendeteksi kepadatan kendaraan di area jalur utama.")

with st.sidebar:
    st.header("Pengaturan")
    confidence = st.slider("Confidence threshold", 0.1, 0.9, float(config.get("confidence", 0.25)), 0.05)

# Tipe file ditambah dengan format video
uploaded_file = st.file_uploader("Pilih file (.jpg/.png/.mp4/.avi)", type=["jpg", "jpeg", "png", "mp4", "avi"])

# Koordinat Kerucut Aspal
roi_points = np.array([
    [0, 720], [530, 280], [750, 280], [1280, 720]
], np.int32)

if uploaded_file is not None:
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    # ------------------ JIKA PENGGUNA MENGUNGGAH GAMBAR ------------------
    if file_extension in ['jpg', 'jpeg', 'png']:
        image = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(image)

        with st.spinner("Mendeteksi gambar..."):
            results = model.predict(source=img_array, conf=confidence, verbose=False)
            annotated_img = img_array.copy()
            n_vehicles_valid = 0
            
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                
                if cv2.pointPolygonTest(roi_points, (cx, cy), False) >= 0:
                    cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.circle(annotated_img, (cx, cy), 5, (255, 255, 0), -1)
                    n_vehicles_valid += 1
                    
            cv2.polylines(annotated_img, [roi_points], isClosed=True, color=(0, 255, 0), thickness=2)
            density_class = classify_density(n_vehicles_valid, config)

        st.image(annotated_img, caption="Hasil Deteksi (Terfilter)", use_container_width=True)
        st.success(f"Ditemukan **{n_vehicles_valid} kendaraan**. Kepadatan: **{density_class}**")

  # ------------------ JIKA PENGGUNA MENGUNGGAH VIDEO ------------------
    elif file_extension in ['mp4', 'avi']:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_file.read())
        
        cap = cv2.VideoCapture(tfile.name)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # Menghitung total frame video
        
        st.info("Video sedang diproses di server. Mohon tunggu, ini membutuhkan waktu karena menggunakan CPU (Tanpa GPU).")
        
        # Elemen UI untuk Progress Bar dan Status
        progress_bar = st.progress(0)
        status_text = st.empty()
        stframe = st.empty() # Wadah untuk preview gambar
        
        frame_skip = 10  # Melewati 9 frame agar tidak memberatkan server
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Update Progress Bar setiap kali frame terbaca
            progress = min(frame_count / total_frames, 1.0)
            progress_bar.progress(progress)
            
            # Lewati frame yang tidak kelipatan 10
            if frame_count % frame_skip != 0:
                continue
                
            # Proses deteksi hanya untuk frame yang terpilih
            results = model.predict(source=frame, conf=confidence, verbose=False)
            n_vehicles_valid = 0
            
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                
                if cv2.pointPolygonTest(roi_points, (cx, cy), False) >= 0:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.circle(frame, (cx, cy), 5, (255, 255, 0), -1)
                    n_vehicles_valid += 1
            
            cv2.polylines(frame, [roi_points], isClosed=True, color=(0, 255, 0), thickness=2)
            
            # Tampilkan sebagai "Preview" di web
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame_rgb, caption="Preview Frame Terdeteksi (Mode Cepat)", use_container_width=True)
            
            density_class = classify_density(n_vehicles_valid, config)
            status_text.markdown(f"**Proses Frame {frame_count}/{total_frames}** | Kendaraan: **{n_vehicles_valid}** | Kepadatan: **{density_class}**")
            
        cap.release()
        progress_bar.progress(1.0) # Pastikan bar penuh 100% di akhir
        st.success("Pemrosesan video selesai!")
