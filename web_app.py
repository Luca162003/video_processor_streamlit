import streamlit as st
import cv2
import torch
import numpy as np
import statistics
import math
import yaml
import tempfile
import os
import pandas as pd
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO

if not torch.cuda.is_available():
    st.warning("⚠️ ATTENZIONE: GPU non rilevata! L'elaborazione sarà molto lenta (CPU).")
else:
    st.success(f"✅ GPU Rilevata: {torch.cuda.get_device_name(0)}")

st.set_page_config(page_title="AI-powered Murine Polyp Analysis", layout="wide")

# Parametri di Default (in futuro potrebbero essere resi configurabili)
DEFAULT_CONF = 0.5
DEFAULT_IOU = 0.5
IMG_SIZE = 640
MIN_TRACK_FRAMES = 5
TRACK_BUFFER = 10
MATCH_THRESH = 0.85


def create_compatible_tracker_config():
    """Crea il file di configurazione YAML per il Bytetracker."""
    tracker_config = {
        'tracker_type': 'bytetrack',
        'track_high_thresh': 0.5,
        'track_low_thresh': 0.1,
        'new_track_thresh': 0.6,
        'track_buffer': TRACK_BUFFER,
        'match_thresh': MATCH_THRESH,
        'fuse_score': True,
        'gmc_method': 'sparseOptFlow',
        'proximity_thresh': 0.5,
        'appearance_thresh': 0.25,
        'with_reid': False
    }
    
    # Salviamo in un file temporaneo per evitare conflitti
    tfile = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
    yaml.dump(tracker_config, tfile)
    tfile.close()
    return tfile.name

def get_tumor_score_advanced(mask_poly, img_shape):
    """Calcola lo score tumorale basato sulle dimensioni relative."""
    if len(mask_poly) < 3: return 0, 0.0
    pts = np.array(mask_poly, dtype=np.int32)
    area_pixels = cv2.contourArea(pts)
    if area_pixels == 0: return 0, 0.0

    d_eq = 2 * math.sqrt(area_pixels / math.pi)
    h, w = img_shape[:2]
    D = min(h, w)
    r = d_eq / D

    if r == 0: score = 0
    elif 0 < r <= 0.1: score = 1
    elif 0.1 < r <= 0.125: score = 2
    elif 0.125 < r <= 0.25: score = 3
    elif 0.25 < r <= 0.5: score = 4
    else: score = 5
    return score, r

def draw_inference_overlay(img, detections, valid_ids, display_id_map):
    """Disegna le box e le annotazioni sul frame."""
    frame_scores = {} 
    
    for det in detections:
        original_track_id = det['track_id']
        
        if original_track_id not in valid_ids: continue 

        clean_id = display_id_map[original_track_id]
        box = det['box']
        poly = det['poly']
        x1, y1, x2, y2 = box
        
        score, r_value = get_tumor_score_advanced(poly, img.shape)
        frame_scores[original_track_id] = score

        id_text = f"Polyp {clean_id}"
        label_text = f"{id_text} | S:{score} (r={r_value:.2f})"
        t_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        text_y = int(y1) - 10 if int(y1) - 10 > 20 else int(y1) + 20
        
        cv2.rectangle(img, (int(x1), text_y - t_size[1] - 5), (int(x1) + t_size[0], text_y + 5), (0, 0, 255), -1)
        cv2.putText(img, label_text, (int(x1), text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

    return img, frame_scores

def process_video_pipeline(video_path, model_path, tracker_config_path, progress_bar, status_text):
    """Logica principale di elaborazione (Two-Pass)."""
    
    try:
        model = YOLO(model_path)
    except Exception as e:
        st.error(f"Errore caricamento modello: {e}")
        return None, None, None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return None, None, None

    width, height = int(cap.get(3)), int(cap.get(4))
    fps, total_frames = cap.get(5), int(cap.get(7))

    status_text.text("Fase 1/2: Analisi AI e Tracking in corso...")
    all_frame_detections = [] 
    id_lifespan = defaultdict(int)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret: break 
        
        results = model.track(
            frame, 
            persist=True,
            tracker=tracker_config_path, 
            imgsz=IMG_SIZE, 
            conf=DEFAULT_CONF, 
            iou=DEFAULT_IOU, 
            verbose=False
        )
        
        result = results[0]
        current_frame_dets = []

        if result.boxes is not None and result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            track_ids = result.boxes.id.int().cpu().tolist()
            xy_polys = result.masks.xy if result.masks is not None else [[] for _ in boxes]

            for box, track_id, poly in zip(boxes, track_ids, xy_polys):
                current_frame_dets.append({'box': box, 'track_id': track_id, 'poly': poly})
                id_lifespan[track_id] += 1
        
        all_frame_detections.append(current_frame_dets)
        frame_count += 1
        
        progress = int((frame_count / total_frames) * 50)
        progress_bar.progress(progress)

    cap.release()
    del model
    torch.cuda.empty_cache()

    # --- FILTRAGGIO E RINUMERAZIONE ---
    valid_ids_set = {tid for tid, count in id_lifespan.items() if count >= MIN_TRACK_FRAMES}
    sorted_valid_ids = sorted(list(valid_ids_set))
    display_id_map = {real_id: i+1 for i, real_id in enumerate(sorted_valid_ids)}

    # --- PASS 2: RENDERING ---
    status_text.text("Fase 2/2: Generazione video annotato e calcolo punteggi...")
    
    # Creiamo un file temporaneo per l'output video
    output_temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    output_path = output_temp_file.name
    output_temp_file.close() # Chiudiamo per permettere a CV2 di scriverci

    cap = cv2.VideoCapture(video_path)
    # Usa 'mp4v' per compatibilità generale, o 'avc1' se disponibile per web
    fourcc = cv2.VideoWriter_fourcc(*'avc1') 
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    polyp_scores_dict = defaultdict(list)
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        if frame_idx < len(all_frame_detections):
            detections = all_frame_detections[frame_idx]
            annotated_frame, frame_id_scores = draw_inference_overlay(frame, detections, valid_ids_set, display_id_map)
            
            for tid, score in frame_id_scores.items():
                polyp_scores_dict[tid].append(score)
                
            out.write(annotated_frame)
        else:
            out.write(frame)
        
        frame_idx += 1
        # Aggiorna progress bar (50% -> 100%)
        progress = 50 + int((frame_idx / total_frames) * 50)
        if progress > 100: progress = 100
        progress_bar.progress(progress)

    cap.release()
    out.release()
    status_text.text("Analisi Completata!")

    # --- METRICHE ---
    report_data = []
    final_polyp_modes = []
    
    for real_id in sorted_valid_ids:
        display_id = display_id_map[real_id]
        scores = polyp_scores_dict[real_id]
        if scores:
            try:
                p_mode = statistics.mode(scores)
            except statistics.StatisticsError:
                p_mode = max(set(scores), key=scores.count)
            
            final_polyp_modes.append(p_mode)
            report_data.append({
                "Polyp ID (Display)": display_id,
                "Track ID (Internal)": real_id,
                "Mode Tumor Score": p_mode,
                "Frames Detected": len(scores)
            })

    final_score = max(final_polyp_modes) if final_polyp_modes else 0
    unique_count = len(valid_ids_set)
    
    return output_path, report_data, final_score, unique_count


st.title("AI-powered Murine Polyp Analysis")
st.markdown("""
Analizza video endoscopici murini per rilevare e valutare polipi utilizzando modelli di deep learning avanzati.
""")

with st.sidebar:
    st.header("Impostazioni")
    
    # 1. Caricamento Modello
    st.subheader("1. Modello AI (.pt)")
    model_file = st.file_uploader("Carica i pesi del modello", type=["pt"])
    
    st.info("Se non carichi un modello, verrà utilizzato quello predefinito")

    # 2. Impostazioni Avanzate (opzionali)
    with st.expander("Parametri Avanzati"):
        conf_thresh = st.slider("Confidence Threshold", 0.1, 1.0, DEFAULT_CONF)
        iou_thresh = st.slider("IoU Threshold", 0.1, 1.0, DEFAULT_IOU)

# --- MAIN AREA ---

# 1. Caricamento Video
st.subheader("Carica Video Endoscopico")
uploaded_video = st.file_uploader("Trascina qui il file .mp4", type=["mp4", "avi", "mov"])

start_btn = st.button("Avvia Analisi", type="primary", disabled=(uploaded_video is None))

if start_btn and uploaded_video:
    # Gestione Modello
    if model_file:
        tfile_model = tempfile.NamedTemporaryFile(delete=False, suffix='.pt')
        tfile_model.write(model_file.read())
        model_path = tfile_model.name
    elif os.path.exists("CRC_murine_best.pt"):
        model_path = "CRC_murine_best.pt"
    else:
        st.error("ERRORE: Nessun modello caricato e 'CRC_murine_best.pt' non trovato.")
        st.stop()

    tfile_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile_video.write(uploaded_video.read())
    video_path = tfile_video.name
    tfile_video.close() 

    tracker_config = create_compatible_tracker_config()

    prog_bar = st.progress(0)
    status_txt = st.empty()

    with st.spinner('Elaborazione in corso... Potrebbe richiedere qualche minuto.'):
        out_video_path, report, final_score, unique_count = process_video_pipeline(
            video_path, model_path, tracker_config, prog_bar, status_txt
        )

    if out_video_path:
        st.success("Elaborazione terminata con successo!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Punteggio Video (Worst Score)", f"{final_score}/5")
        with col2:
            st.metric("Polipi Unici Rilevati", unique_count)

        st.divider()

        # Mostra Video Risultante
        st.subheader("Video Annotato")
        # Nota: I browser a volte hanno problemi con mp4v generati da OpenCV. 
        # Leggiamo il file e lo passiamo a streamlit
        with open(out_video_path, 'rb') as v:
            video_bytes = v.read()
            st.video(video_bytes)
        
        # Download Video
        st.download_button(
            label="📥 Scarica Video Annotato",
            data=video_bytes,
            file_name=f"analyzed_{uploaded_video.name}",
            mime="video/mp4"
        )

        st.divider()

        st.subheader("Dettaglio Polipi Rilevati")
        if report:
            df = pd.DataFrame(report)
            st.dataframe(df, use_container_width=True)
            
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Scarica Report Excel (CSV)",
                data=csv,
                file_name=f"report_{uploaded_video.name}.csv",
                mime="text/csv",
            )
        else:
            st.warning("Nessun polipo rilevato in modo stabile (> 5 frame).")

    try:
        os.unlink(video_path)
        os.unlink(tracker_config)
    
    except:
        pass