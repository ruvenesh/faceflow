import os, sys, subprocess, uuid, csv, time, json, glob, shutil, cv2
import pandas as pd
from datetime import datetime
from pathlib import Path

BASE_DIR = os.path.abspath(".")
LP_VENV = os.path.join(BASE_DIR, "liveportrait_venv/bin/python")
LP_DIR = os.path.join(BASE_DIR, "LivePortrait")
MANIFEST = os.path.join(BASE_DIR, "videos/07_generation_logs/bin_B_bulk_manifest.csv")
LOG_FILE = os.path.join(BASE_DIR, "videos/07_generation_logs/master_binB_bulk.log")

def log(msg):
    with open(LOG_FILE, "a") as f: f.write(f"[{datetime.now()}] {msg}\n")
    print(f"[{datetime.now()}] {msg}")

log("Industrial LivePortrait Engine Started - Bin B Bulk Production")

while True:
    if not os.path.exists(MANIFEST):
        log(f"CRITICAL: Manifest missing at {MANIFEST}")
        break
    df = pd.read_csv(MANIFEST)
    pending = df[df['status'] == 'PENDING']
    if pending.empty:
        log("No more pending jobs. Exiting.")
        break
    
    idx = pending.index[0]
    job = pending.iloc[0]
    df.at[idx, 'status'] = 'RUNNING'
    df.to_csv(MANIFEST, index=False)
    
    log(f"STARTING: {job['chunk_id']} ({job['model_name']})")
    start_t = time.time()
    success = False
    
    try:
        tmp_out = os.path.join(BASE_DIR, f"latentsync_temp/lp_bulk_{job['chunk_id']}")
        os.makedirs(tmp_out, exist_ok=True)
        shutil.rmtree(os.path.join(LP_DIR, "animations"), ignore_errors=True)
        
        # CORRECT HYPHENATED FLAGS: --flag-stitching, --scale 2.6
        cmd = [
            LP_VENV, "inference.py",
            "--source", job['source_path'],
            "--driving", job['driver_path'],
            "--output-dir", tmp_out,
            "--flag-relative-motion",
            "--flag-pasteback",
            "--flag-do-crop",
            "--flag-stitching",
            "--scale", "2.6"
        ]
        env = os.environ.copy(); env["PYTHONPATH"] = LP_DIR
        
        log(f"Running inference for {job['chunk_id']}...")
        res = subprocess.run(cmd, cwd=LP_DIR, capture_output=True, env=env, text=True)
        
        if res.returncode != 0:
            log(f"Error for {job['chunk_id']}: {res.stderr[-500:]}")
        
        found = glob.glob(os.path.join(tmp_out, "**", "*.mp4"), recursive=True)
        non_concat = [f for f in found if "_concat.mp4" not in f]
        
        if non_concat:
            log(f"Found output for {job['chunk_id']}, resizing...")
            cap = cv2.VideoCapture(job['source_path'])
            sw, sh = int(cap.get(3)), int(cap.get(4)); cap.release()
            os.makedirs(os.path.dirname(job['output_path']), exist_ok=True)
            subprocess.run(["ffmpeg", "-y", "-i", non_concat[0], "-vf", f"scale={sw}:{sh}", "-c:v", "libx264", "-crf", "18", job['output_path']], capture_output=True)
            success = True
            
        shutil.rmtree(tmp_out, ignore_errors=True)
        
        df = pd.read_csv(MANIFEST)
        df.at[idx, 'status'] = 'COMPLETED' if success else 'FAILED'
        df.at[idx, 'duration'] = time.time() - start_t
        df.to_csv(MANIFEST, index=False)
        
        if success:
            log(f"SUCCESS: {job['chunk_id']}")
            p_dir = os.path.join(BASE_DIR, "videos/06_generation/bin_B_reenactment/liveportrait/preview_samples/")
            os.makedirs(p_dir, exist_ok=True)
            if len(glob.glob(os.path.join(p_dir, "*.mp4"))) < 10:
                shutil.copy2(job['output_path'], p_dir)
        else:
            log(f"FAILED: {job['chunk_id']}")
            
    except Exception as e:
        log(f"ERROR for {job['chunk_id']}: {str(e)}")
        df.at[idx, 'status'] = 'FAILED'; df.to_csv(MANIFEST, index=False)

    time.sleep(0.5)
