import os, sys, subprocess, uuid, csv, time, json, glob, shutil, cv2
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

BASE_DIR = os.path.abspath(".")
LP_VENV = os.path.join(BASE_DIR, "liveportrait_venv/bin/python")
PL_VENV = os.path.join(BASE_DIR, "personalive_venv/bin/python")
LP_DIR = os.path.join(BASE_DIR, "LivePortrait")
PL_DIR = os.path.join(BASE_DIR, "PersonaLive")
MANIFEST = os.path.join(BASE_DIR, "videos/07_generation_logs/bin_B_v2_manifest.csv")
LOG_FILE = os.path.join(BASE_DIR, "videos/07_generation_logs/master_binB_v2.log")

def log(msg):
    with open(LOG_FILE, "a") as f: f.write(f"[{datetime.now()}] {msg}\n")
    print(f"[{datetime.now()}] {msg}")

log("Master Worker Bin B v2 (LP + PersonaLive Blending Fix) Started")

def get_face_bbox(video_path):
    import mediapipe as mp
    mp_face_detection = mp.solutions.face_detection
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 30)
        ret, frame = cap.read()
        cap.release()
        if not ret: return None
        h, w = frame.shape[:2]
        results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not results.detections: return None
        detection = max(results.detections, key=lambda x: x.location_data.relative_bounding_box.width * x.location_data.relative_bounding_box.height)
        bbox = detection.location_data.relative_bounding_box
        cx, cy = int((bbox.xmin + bbox.width/2) * w), int((bbox.ymin + bbox.height/2) * h)
        side = int(max(bbox.width * w, bbox.height * h) * 1.5)
        return (max(0, cx - side//2), max(0, cy - side//2), min(w, cx + side//2), min(h, cy + side//2))

while True:
    if not os.path.exists(MANIFEST):
        log(f"CRITICAL: Manifest missing at {MANIFEST}")
        break
    df = pd.read_csv(MANIFEST)
    pending = df[df['status'] == 'PENDING']
    if pending.empty: break
    
    idx = pending.index[0]
    job = pending.iloc[0]
    df.at[idx, 'status'] = 'RUNNING'
    df.to_csv(MANIFEST, index=False)
    
    log(f"STARTING {job['model_name']} -> {job['chunk_id']}")
    start_t = time.time(); success = False
    
    try:
        if job['model_name'] == 'liveportrait':
            tmp_out = os.path.join(BASE_DIR, f"latentsync_temp/lp_v2_{job['chunk_id']}")
            os.makedirs(tmp_out, exist_ok=True)
            shutil.rmtree(os.path.join(LP_DIR, "animations"), ignore_errors=True)
            cmd = [LP_VENV, "inference.py", "-s", job['source_path'], "-d", job['driver_path'], "-o", tmp_out, "--flag_relative_motion", "--flag_pasteback", "--flag_do_crop"]
            env = os.environ.copy(); env["PYTHONPATH"] = LP_DIR
            subprocess.run(cmd, cwd=LP_DIR, capture_output=True, env=env)
            found = glob.glob(os.path.join(tmp_out, "**", "*.mp4"), recursive=True)
            non_concat = [f for f in found if "_concat.mp4" not in f]
            if non_concat:
                os.makedirs(os.path.dirname(job['output_path']), exist_ok=True)
                shutil.move(non_concat[0], job['output_path']); success = True
            shutil.rmtree(tmp_out, ignore_errors=True)
            
        elif job['model_name'] == 'personalive':
            bbox = get_face_bbox(job['source_path'])
            if not bbox: raise Exception("No face detected for PersonaLive")
            x1, y1, x2, y2 = bbox
            src_square = os.path.join(BASE_DIR, f"latentsync_temp/src_{job['chunk_id']}_square.mp4")
            subprocess.run(["ffmpeg", "-y", "-i", job['source_path'], "-vf", f"crop={x2-x1}:{y2-y1}:{x1}:{y1},scale=512:512", "-c:v", "libx264", src_square], capture_output=True)
            ref_img = os.path.join(BASE_DIR, f"latentsync_temp/pl_{job['chunk_id']}_ref.jpg")
            subprocess.run(["ffmpeg", "-y", "-i", src_square, "-vf", "select=eq(n\\,0)", "-vframes", "1", ref_img], capture_output=True)
            job_name = f"job_{job['chunk_id']}"
            cmd = [PL_VENV, "inference_offline.py", "--config", "configs/prompts/personalive_offline.yaml", "--reference_image", ref_img, "--driving_video", job['driver_path'], "--name", job_name, "-L", "60"]
            env = os.environ.copy(); env["PYTHONPATH"] = PL_DIR
            subprocess.run(cmd, cwd=PL_DIR, capture_output=True, env=env)
            date_str = datetime.now().strftime("%Y%m%d")
            res_pattern = os.path.join(PL_DIR, "results", f"{date_str}--{job_name}", "split_vid", "*.mp4")
            found = glob.glob(res_pattern)
            if found:
                pl_square = found[0]; final_out = job['output_path']
                os.makedirs(os.path.dirname(final_out), exist_ok=True)
                overlay_filter = f"[1:v]scale={x2-x1}:{y2-y1}[face];[0:v][face]overlay={x1}:{y1}"
                subprocess.run(["ffmpeg", "-y", "-i", job['source_path'], "-i", pl_square, "-filter_complex", overlay_filter, "-c:v", "libx264", "-crf", "18", final_out], capture_output=True)
                if os.path.exists(final_out): success = True
            for f in [src_square, ref_img]:
                if os.path.exists(f): os.remove(f)
            shutil.rmtree(os.path.join(PL_DIR, "results", f"{date_str}--{job_name}"), ignore_errors=True)

        df = pd.read_csv(MANIFEST); df.at[idx, 'status'] = 'COMPLETED' if success else 'FAILED'
        df.to_csv(MANIFEST, index=False)
        if success:
            log(f"SUCCESS: {job['chunk_id']}")
            p_dir = os.path.join(BASE_DIR, f"videos/06_generation/bin_B_reenactment/{job['model_name']}/preview_samples/")
            os.makedirs(p_dir, exist_ok=True); shutil.copy2(job['output_path'], p_dir)
    except Exception as e:
        log(f"ERROR for {job['chunk_id']}: {str(e)}")
        df.at[idx, 'status'] = 'FAILED'; df.to_csv(MANIFEST, index=False)
    time.sleep(1)
