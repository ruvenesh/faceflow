import os
import sys
import subprocess
import time
import shutil
import pandas as pd
from datetime import datetime
from pathlib import Path

# Paths
VENV_PY = os.path.abspath("liveportrait_venv/bin/python")
LP_DIR = os.path.abspath("LivePortrait")
MANIFEST = os.path.abspath("videos/07_generation_logs/bin_B_test_manifest.csv")
LOG_FILE = os.path.abspath("videos/07_generation_logs/bin_B_liveportrait_run_log.txt")

def log(msg):
    with open(LOG_FILE, "a") as f:
        f.write(f"[{datetime.now()}] {msg}\n")

log("LivePortrait Worker Started")

while True:
    df = pd.read_csv(MANIFEST)
    pending = df[(df['model_name'] == 'liveportrait') & (df['status'] == 'PENDING')]
    
    if pending.empty:
        log("No more pending LivePortrait jobs. Exiting.")
        break
        
    idx = pending.index[0]
    job = pending.iloc[0]
    
    # Update status to RUNNING
    df.at[idx, 'status'] = 'RUNNING'
    df.to_csv(MANIFEST, index=False)
    
    log(f"Starting job: {job['chunk_id']}")
    start_t = time.time()
    
    cmd = [
        VENV_PY, "inference.py",
        "-s", job['source_path'],
        "-d", job['driver_path'],
        "-o", os.path.dirname(job['output_path']),
        "--flag_relative_motion",
        "--flag_pasteback",
        "--flag_do_crop"
    ]
    
    env = os.environ.copy()
    env["PYTHONPATH"] = LP_DIR
    
    res = subprocess.run(cmd, cwd=LP_DIR, capture_output=True, text=True, env=env)
    
    end_t = time.time()
    duration = end_t - start_t
    
    # LivePortrait creates a timestamped subfolder, find the actual result.mp4
    # The output folder we provided was videos/06_generation/bin_B_reenactment/output
    temp_dir = os.path.join(LP_DIR, "animations")
    found_videos = list(Path(temp_dir).rglob("*.mp4"))
    
    success = False
    if found_videos:
        # Sort by creation time to get the newest one
        latest_video = max(found_videos, key=os.path.getmtime)
        shutil.move(str(latest_video), job['output_path'])
        success = True
        log(f"Job {job['chunk_id']} SUCCESS")
    else:
        log(f"Job {job['chunk_id']} FAILED. Stderr: {res.stderr[-500:]}")

    # Final updates
    df = pd.read_csv(MANIFEST)
    df.at[idx, 'status'] = 'COMPLETED' if success else 'FAILED'
    df.at[idx, 'duration'] = duration
    df.at[idx, 'output_size_kb'] = os.path.getsize(job['output_path']) / 1024 if success else 0
    df.to_csv(MANIFEST, index=False)
    
    # Preview Sample Copy
    completed = df[(df['model_name'] == 'liveportrait') & (df['status'] == 'COMPLETED')]
    if len(completed) <= 3 and success:
        shutil.copy2(job['output_path'], "videos/06_generation/bin_B_reenactment/preview_samples/liveportrait/")

    # VRAM Cleanup
    time.sleep(2)
