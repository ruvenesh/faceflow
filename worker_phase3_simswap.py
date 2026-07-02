import os, sys, subprocess, uuid, csv, time, json, shutil
import pandas as pd
from datetime import datetime
from pathlib import Path

BASE_DIR = os.path.abspath(".")
SS_VENV = os.path.join(BASE_DIR, "simswap_venv/bin/python")
SS_DIR = os.path.join(BASE_DIR, "SimSwap")
MANIFEST = os.path.join(BASE_DIR, "videos/07_generation_logs/simswap_test_manifest.csv")
LOG_FILE = os.path.join(BASE_DIR, "videos/07_generation_logs/simswap_test_log.txt")

def log(msg):
    with open(LOG_FILE, "a") as f: f.write(f"[{datetime.now()}] {msg}\n")
    print(f"[{datetime.now()}] {msg}")

df = pd.read_csv(MANIFEST)
log(f"Phase 3 (SimSwap) Started: {len(df)} jobs.")

durations = []

for idx, job in df.iterrows():
    if job['status'] == 'COMPLETED': continue
    
    log(f"STARTING: {job['chunk_id']}")
    start_t = time.time()
    
    # 1. Update manifest to RUNNING
    df.at[idx, 'status'] = 'RUNNING'
    df.to_csv(MANIFEST, index=False)
    
    # 2. Extract frame 30 for target-face (Step 7)
    tmp_face = os.path.join(BASE_DIR, f"latentsync_temp/ss_face_{job['chunk_id']}.jpg")
    subprocess.run(['ffmpeg', '-y', '-i', job['source_path'], '-vf', 'select=eq(n\\,30)', '-vframes', '1', tmp_face], capture_output=True)
    
    # 3. Run Video Swap
    cmd = [
        SS_VENV, "test_video_swapsingle.py",
        "--crop_size", "224",
        "--use_mask",
        "--name", "people",
        "--pic_a_path", job['ref_path'],
        "--video_path", job['source_path'],
        "--output_path", job['output_path'],
        "--temp_path", "./tmp"
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{SS_DIR}:" + env.get("PYTHONPATH", "")
    
    res = subprocess.run(cmd, cwd=SS_DIR, capture_output=True, text=True, env=env)
    
    # 4. Check Integrity (Step 8)
    success = False
    fail_reason = ""
    if os.path.exists(job['output_path']) and os.path.getsize(job['output_path']) > 0:
        import cv2
        cap = cv2.VideoCapture(job['output_path'])
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if 55 <= frames <= 65:
            success = True
        else:
            fail_reason = f"Frame count mismatch: {frames}"
    else:
        fail_reason = "Output file missing or empty"
        if res.returncode != 0:
            fail_reason += f" (Code {res.returncode})"

    # 5. Final update
    duration = time.time() - start_t
    durations.append(duration)
    df.at[idx, 'status'] = 'COMPLETED' if success else 'FAILED'
    df.at[idx, 'fail_reason'] = fail_reason
    df.to_csv(MANIFEST, index=False)
    
    if success:
        log(f"SUCCESS: {job['chunk_id']}")
        # Preview copy (first 3)
        completed = df[df['status'] == 'COMPLETED']
        if len(completed) <= 3:
            shutil.copy2(job['output_path'], "videos/06_generation/bin_CD_swap/preview_samples/simswap/")
    else:
        log(f"FAILED: {job['chunk_id']} ({fail_reason})")
    
    if os.path.exists(tmp_face): os.remove(tmp_face)

avg_time = sum(durations)/len(durations) if durations else 0
log(f"\nSimSwap Test Run Summary\nTotal jobs: 10\nCompleted: {len(df[df['status'] == 'COMPLETED'])}\nFailed: {len(df[df['status'] == 'FAILED'])}\nAvg time/chunk: {avg_time:.1f} seconds")
