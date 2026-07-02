import os
import sys
import subprocess
import time
import shutil
import pandas as pd
import cv2
from datetime import datetime
from pathlib import Path

# Paths
VENV_PY = os.path.abspath("fomm_venv/bin/python")
FOMM_DIR = os.path.abspath("first-order-model")
MANIFEST = os.path.abspath("videos/07_generation_logs/bin_B_test_manifest.csv")
LOG_FILE = os.path.abspath("videos/07_generation_logs/bin_B_fomm_run_log.txt")
TEMP_DIR = os.path.abspath("latentsync_temp")

def log(msg):
    with open(LOG_FILE, "a") as f:
        f.write(f"[{datetime.now()}] {msg}\n")

log("FOMM Worker Started")
os.makedirs(TEMP_DIR, exist_ok=True)

while True:
    df = pd.read_csv(MANIFEST)
    pending = df[(df['model_name'] == 'fomm') & (df['status'] == 'PENDING')]
    
    if pending.empty:
        log("No more pending FOMM jobs. Exiting.")
        break
        
    idx = pending.index[0]
    job = pending.iloc[0]
    
    df.at[idx, 'status'] = 'RUNNING'
    df.to_csv(MANIFEST, index=False)
    
    log(f"Starting job: {job['chunk_id']}")
    start_t = time.time()
    
    # 1. Extract frame 30
    source_img = os.path.join(TEMP_DIR, f"{job['chunk_id']}_frame30.jpg")
    subprocess.run(["ffmpeg", "-y", "-i", job['source_path'], "-vf", "select=eq(n\\,30)", "-vframes", "1", source_img], capture_output=True)
    
    # 2. Run FOMM (Try GPU first)
    cmd = [
        VENV_PY, "demo.py",
        "--config", "config/vox-256.yaml",
        "--checkpoint", "vox-cpk.pth.tar",
        "--source_image", source_img,
        "--driving_video", job['driver_path'],
        "--result_video", job['output_path'],
        "--relative",
        "--adapt_scale"
    ]
    
    env = os.environ.copy()
    env["PYTHONPATH"] = FOMM_DIR
    
    success = False
    mode = "GPU"
    
    res = subprocess.run(cmd, cwd=FOMM_DIR, capture_output=True, text=True, env=env)
    
    if res.returncode != 0 or not os.path.exists(job['output_path']):
        log(f"GPU Mode FAILED for {job['chunk_id']}, falling back to CPU. Error: {res.stderr[-200:]}")
        mode = "CPU"
        cmd.append("--cpu")
        res = subprocess.run(cmd, cwd=FOMM_DIR, capture_output=True, text=True, env=env)
        
    if os.path.exists(job['output_path']):
        success = True
        log(f"Job {job['chunk_id']} SUCCESS ({mode})")
    else:
        log(f"Job {job['chunk_id']} PERMANENT FAILURE. Stderr: {res.stderr[-500:]}")

    # Final updates
    df = pd.read_csv(MANIFEST)
    df.at[idx, 'status'] = 'COMPLETED' if success else 'FAILED'
    df.at[idx, 'duration'] = time.time() - start_t
    df.at[idx, 'output_size_kb'] = os.path.getsize(job['output_path']) / 1024 if success else 0
    df.to_csv(MANIFEST, index=False)
    
    # Preview Sample Copy
    completed = df[(df['model_name'] == 'fomm') & (df['status'] == 'COMPLETED')]
    if len(completed) <= 3 and success:
        shutil.copy2(job['output_path'], "videos/06_generation/bin_B_reenactment/preview_samples/fomm/")

    # Cleanup
    if os.path.exists(source_img): os.remove(source_img)
    time.sleep(2)
