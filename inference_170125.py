import sys
import cv2
import torch
import numpy as np
import statistics
import math
import csv
import yaml
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO

# ================= CONFIGURATION =================
INPUT_VIDEO_DIR = Path("725 ko.mp4").parent
MODEL_PATH = Path("CRC_murine_best.pt")

CONF_THRESHOLD = 0.5 
IOU_THRESHOLD = 0.5
IMG_SIZE = 640

# --- FILTER SETTINGS ---
MIN_TRACK_FRAMES = 5    # ID must exist for > 5 frames
TRACK_BUFFER = 10       # Forget ID after 10 frames 
MATCH_THRESH = 0.85     

# ================= HELPER FUNCTIONS =================

def create_compatible_tracker_config():
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
    
    config_path = Path("custom_polyp_tracker_fixed.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(tracker_config, f)
    
    return config_path

def validate_paths():
    if not INPUT_VIDEO_DIR.exists():
        raise FileNotFoundError(f"Input directory not found: {INPUT_VIDEO_DIR}")
    
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model weights not found: {MODEL_PATH}")
    if MODEL_PATH.suffix not in ['.pt', '.onnx']:
        raise ValueError(f"Unsupported model format: {MODEL_PATH.suffix}. Supported formats are .pt and .onnx")

def get_tumor_score_advanced(mask_poly, img_shape):
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
    frame_scores = {} 
    
    for det in detections:
        original_track_id = det['track_id']
        
        # Skip if this ID was filtered out
        if original_track_id not in valid_ids: continue 

        # Get the new clean ID (1, 2, 3...)
        clean_id = display_id_map[original_track_id]

        box = det['box']
        poly = det['poly']
        x1, y1, x2, y2 = box
        
        score, r_value = get_tumor_score_advanced(poly, img.shape)
        
        # Store score using the ORIGINAL ID for consistency in tracking logic
        frame_scores[original_track_id] = score

        # Display the CLEAN ID
        id_text = f"Polyp {clean_id}"
        label_text = f"{id_text} | S:{score} (r={r_value:.2f})"
        t_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        text_y = int(y1) - 10 if int(y1) - 10 > 20 else int(y1) + 20
        
        cv2.rectangle(img, (int(x1), text_y - t_size[1] - 5), (int(x1) + t_size[0], text_y + 5), (0, 0, 255), -1)
        cv2.putText(img, label_text, (int(x1), text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

    return img, frame_scores

def process_video(video_path, tracker_config_path):

    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"Errore caricamento modello: {e}")
        return None, None 
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return None, None

    width, height = int(cap.get(3)), int(cap.get(4))
    fps, total_frames = cap.get(5), int(cap.get(7))

    print(f"   Analyzing: {video_path.name} ({total_frames} frames)...")

    # --- PASS 1: INFERENCE ---
    all_frame_detections = [] 
    id_lifespan = defaultdict(int)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret: break 
        
        results = model.track(
            frame, 
            persist=True,
            tracker=str(tracker_config_path), 
            imgsz=IMG_SIZE, 
            conf=CONF_THRESHOLD, 
            iou=IOU_THRESHOLD, 
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
        if frame_count % 50 == 0: print(f"    -> Inference Pass: {frame_count}/{total_frames}", end='\r')

    cap.release()
    
    del model
    torch.cuda.empty_cache()

    valid_ids_set = {tid for tid, count in id_lifespan.items() if count >= MIN_TRACK_FRAMES}
    
    sorted_valid_ids = sorted(list(valid_ids_set))
    display_id_map = {real_id: i+1 for i, real_id in enumerate(sorted_valid_ids)}

    # --- PASS 2: RENDERING & SCORING ---
    cap = cv2.VideoCapture(str(video_path))
    save_path = video_path.parent / f"{video_path.stem}_scored.mp4"
    out = cv2.VideoWriter(str(save_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    polyp_scores_dict = defaultdict(list)
    
    frame_idx = 0
    print(f"\n    -> Rendering Video...")
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        if frame_idx < len(all_frame_detections):
            detections = all_frame_detections[frame_idx]
            
            # Pass the display map so the video shows "Polyp 1", "Polyp 2"...
            annotated_frame, frame_id_scores = draw_inference_overlay(frame, detections, valid_ids_set, display_id_map)
            
            # Accumulate scores
            for tid, score in frame_id_scores.items():
                polyp_scores_dict[tid].append(score)
                
            out.write(annotated_frame)
        else:
            out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()

    # --- METRICS CALCULATION ---
    final_polyp_modes = []

    print(f"\n    📊 Detailed Report for {video_path.name}:")
    
    # Iterate through our CLEAN ID numbers (1, 2, 3...)
    for real_id in sorted_valid_ids:
        display_id = display_id_map[real_id] # Get the clean number (1, 2...)
        scores = polyp_scores_dict[real_id]
        
        if scores:
            try:
                p_mode = statistics.mode(scores)
            except statistics.StatisticsError:
                p_mode = max(set(scores), key=scores.count)
            
            final_polyp_modes.append(p_mode)
            print(f"       - Polyp {display_id} (TrackID {real_id}): Mode Score {p_mode} (seen in {len(scores)} frames)")

    if final_polyp_modes:
        final_score = max(final_polyp_modes)
    else:
        final_score = 0
    
    total_unique_count = len(valid_ids_set)
    
    print(f"    ⭐ Final Video Score (Worst Polyp): {final_score}")
    print(f"    🔢 Unique Polyps Detected: {total_unique_count}")
    
    return final_score, total_unique_count

def main():
    validate_paths()
    
    tracker_config_path = create_compatible_tracker_config()
    print(f"Generated compatible tracker config: {tracker_config_path}")
    
    all_mp4s = sorted(list(INPUT_VIDEO_DIR.glob("*.mp4")))
    video_files = [v for v in all_mp4s if "_scored" not in v.name]
    
    if not video_files: 
        print("No videos found to process.")
        return

    summary_results = []
    
    for vid in video_files:
        try:
            print(f"\n--- Processing: {vid.name} ---")
            # We no longer pass 'model' here, it's loaded INSIDE process_video
            final_score, unique_count = process_video(vid, tracker_config_path)
            
            if final_score is not None:
                summary_results.append({
                    "Video Path": str(vid.resolve()),
                    "Filename": vid.name,
                    "Mode Tumor Score": final_score,
                    "Unique Polyps Detected": unique_count
                })
        except Exception as e:
            print(f"Error processing {vid.name}: {e}")
            import traceback
            traceback.print_exc()

    summary_path = INPUT_VIDEO_DIR / "_results_summary_detailed.csv"
    with open(summary_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Video Path", "Filename", "Mode Tumor Score", "Unique Polyps Detected"])
        writer.writeheader()
        writer.writerows(summary_results)
    
    if tracker_config_path.exists():
        tracker_config_path.unlink() 
    
    print(f"\n{'-'*70}")
    print(f"Processing Complete. Summary saved to: {summary_path}")

if __name__ == "__main__":
    main()