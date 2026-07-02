import os
import cv2
import pandas as pd
import numpy as np
import torch
import shutil
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm
from datetime import datetime
import csv

def append_to_csv(filepath, row):
    with open(filepath, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(row)

def update_chunk_manifest(logs_dir, chunk_id, updates):
    manifest_path = logs_dir / "chunk_manifest.csv"
    df = pd.read_csv(manifest_path, dtype=str)
    idx = df.index[df['chunk_id'] == chunk_id]
    if not idx.empty:
        for k, v in updates.items():
            df.loc[idx, k] = str(v)
        df.to_csv(manifest_path, index=False)

def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    
    x_i1 = max(x1, x3)
    y_i1 = max(y1, y3)
    x_i2 = min(x2, x4)
    y_i2 = min(y2, y4)
    
    inter_area = max(0, x_i2 - x_i1) * max(0, y_i2 - y_i1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x4 - x3) * (y4 - y3)
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0
    return inter_area / union_area

def run_rescue():
    videos_dir = Path("videos").resolve()
    logs_dir = videos_dir / "05_logs"
    multi_face_dir = videos_dir / "03_rejected/phase3_face_spatial/multi_face"
    rescue_out_dir = videos_dir / "04_rescue_pipeline/rescue_crops"
    rescue_log = logs_dir / "rescue_manifest.csv"
    
    df_chunk = pd.read_csv(logs_dir / "chunk_manifest.csv", dtype=str)
    multi_face_chunks = df_chunk[df_chunk['phase3_rejection_reason'] == 'multi_face_detected']
    
    if len(multi_face_chunks) == 0:
        print("No multi_face chunks to rescue.")
        return
        
    print(f"Starting Multi-Face Rescue for {len(multi_face_chunks)} chunks...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO("yolov8n-face.pt")
    
    successful_rescues = 0
    failed_rescues = 0

    for _, row in tqdm(multi_face_chunks.iterrows(), total=len(multi_face_chunks)):
        chunk_id = row['chunk_id']
        parent_id = row['parent_video_id']
        chunk_path = multi_face_dir / f"{chunk_id}.mp4"
        
        if not chunk_path.exists():
            continue
            
        cap = cv2.VideoCapture(str(chunk_path))
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret: break
            frames.append(frame)
        cap.release()
        
        if not frames: continue
        
        frame_h, frame_w = frames[0].shape[:2]
        center_x, center_y = frame_w / 2, frame_h / 2
        
        # Batch predict all 60 frames
        results = model.predict(frames, device=device, verbose=False)
        
        target_track = []
        previous_box = None
        track_valid = True
        
        # Frame 0: Identify Main Face using Center Proximity + Area
        first_res = results[0]
        if len(first_res.boxes) == 0:
            track_valid = False
        else:
            best_score = -1
            best_box = None
            
            for box in first_res.boxes:
                if box.conf.item() < 0.50: continue
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                w, h = x2 - x1, y2 - y1
                area = w * h
                
                # Distance to center
                box_cx, box_cy = x1 + w/2, y1 + h/2
                dist = np.sqrt((box_cx - center_x)**2 + (box_cy - center_y)**2)
                
                # Score formula: Heavily weight size, penalize distance from center
                score = (area * 0.7) - (dist * 0.3)
                if score > best_score:
                    best_score = score
                    best_box = (x1, y1, x2, y2)
                    
            if best_box is None:
                track_valid = False
            else:
                target_track.append(best_box)
                previous_box = best_box
                
        # Frames 1-59: Track Main Face using Intersection over Union (IoU)
        if track_valid:
            for i in range(1, len(results)):
                res = results[i]
                if len(res.boxes) == 0:
                    track_valid = False
                    break
                    
                best_iou = 0
                current_box = None
                
                for box in res.boxes:
                    if box.conf.item() < 0.50: continue
                    b = box.xyxy[0].cpu().numpy()
                    iou = calculate_iou(previous_box, b)
                    if iou > best_iou:
                        best_iou = iou
                        current_box = b
                        
                # If face is lost or tracking swaps to a completely different box (IoU drops)
                if current_box is None or best_iou < 0.3: 
                    track_valid = False
                    break
                    
                target_track.append(current_box)
                previous_box = current_box
                
        # If the track failed mid-chunk, we abandon the rescue
        if not track_valid or len(target_track) < len(frames):
            failed_rescues += 1
            append_to_csv(rescue_log, [chunk_id, parent_id, "", len(target_track), "", 0, "FAILED", "", "", "tracking_lost"])
            continue
            
        # ─── Calculate Stabilized Global Bounding Box ───
        track_arr = np.array(target_track)
        g_x1 = np.min(track_arr[:, 0])
        g_y1 = np.min(track_arr[:, 1])
        g_x2 = np.max(track_arr[:, 2])
        g_y2 = np.max(track_arr[:, 3])
        
        g_w = g_x2 - g_x1
        g_h = g_y2 - g_y1
        
        # Enforce Square Aspect Ratio (1:1) perfectly compatible with ViT (224x224)
        side_len = max(g_w, g_h)
        
        # Add 30% Padding to prevent jitter and provide context
        padded_side = int(side_len * 1.3)
        
        # Find exact center of the global movement
        g_cx = g_x1 + (g_w / 2)
        g_cy = g_y1 + (g_h / 2)
        
        final_x1 = int(g_cx - (padded_side / 2))
        final_y1 = int(g_cy - (padded_side / 2))
        final_x2 = final_x1 + padded_side
        final_y2 = final_y1 + padded_side
        
        # Clamp to actual frame dimensions
        clamped_x1 = max(0, final_x1)
        clamped_y1 = max(0, final_y1)
        clamped_x2 = min(frame_w, final_x2)
        clamped_y2 = min(frame_h, final_y2)
        
        # Write cropped video
        out_path = rescue_out_dir / f"{chunk_id}.mp4"
        fps = 30.0 
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        # Output will ALWAYS be perfectly square (padded_side x padded_side)
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (padded_side, padded_side))
        
        for frame in frames:
            # Extract whatever is safely inside the frame bounds
            crop = frame[clamped_y1:clamped_y2, clamped_x1:clamped_x2]
            ch, cw = crop.shape[:2]
            
            # If the crop hit the edge of the original video frame, we pad it with black
            # to maintain the exact square aspect ratio required by the model
            if ch != padded_side or cw != padded_side:
                padded_crop = np.zeros((padded_side, padded_side, 3), dtype=np.uint8)
                
                # Calculate offsets to place the cropped image into the black padding
                y_offset = clamped_y1 - final_y1
                x_offset = clamped_x1 - final_x1
                
                # Paste the valid pixels into the zeroed array
                padded_crop[y_offset:y_offset+ch, x_offset:x_offset+cw] = crop
                writer.write(padded_crop)
            else:
                writer.write(crop)
                
        writer.release()
        
        # Move original chunk from multi_face back to workspace so Phase 4 can find it
        ws_path = videos_dir / f"01_processing_workspace/chunks/{chunk_id}.mp4"
        shutil.move(str(out_path), str(ws_path))
        
        # Update Manifests so it re-enters the pipeline seamlessly
        update_chunk_manifest(logs_dir, chunk_id, {
            'phase3_status': 'PASSED',
            'phase3_rejection_reason': '',
            'multi_face_detected': 'False',
            'phase4_status': 'PENDING'
        })
        
        append_to_csv(rescue_log, [chunk_id, parent_id, "main_face_track", len(target_track), padded_side, 1.0, "PASSED_RESCUED", "", "", ""])
        successful_rescues += 1

    print(f"\nRescue Complete!")
    print(f"Successfully cropped & squared: {successful_rescues}")
    print(f"Failed (Tracking lost/swapped): {failed_rescues}")

if __name__ == "__main__":
    run_rescue()