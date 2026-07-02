import os, sys, subprocess, time, shutil, glob
import pandas as pd
from datetime import datetime

BASE_DIR = os.path.abspath(".")
FD_VENV = os.path.join(BASE_DIR, "facedancer_venv/Scripts/python.exe")
FD_DIR = os.path.join(BASE_DIR, "FaceDancer")
MANIFEST = os.path.join(BASE_DIR, "videos/07_generation_logs/facedancer_test_manifest.csv")
LOG_FILE = os.path.join(BASE_DIR, "videos/07_generation_logs/facedancer_test_log.txt")

# FaceDancer required paths
MODEL_PATH = os.path.join(FD_DIR, "model_zoo/FaceDancer_config_c_HQ.h5")

def log(msg):
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, "a") as f: f.write(f"[{datetime.now()}] {msg}\n")
    print(f"[{datetime.now()}] {msg}")

if not os.path.exists(MANIFEST):
    log(f"Manifest not found: {MANIFEST}")
    sys.exit(1)

df = pd.read_csv(MANIFEST)
log(f"Phase 4 (FaceDancer) Started: {len(df)} jobs.")

for idx, job in df.iterrows():
    if job.get('status') == 'COMPLETED': continue
    
    log(f"STARTING FaceDancer: {job['chunk_id']}")
    start_t = time.time()
    
    df.at[idx, 'status'] = 'RUNNING'
    df.to_csv(MANIFEST, index=False)
    
    success = True
    final_out = job['output_path']
    def fix_path(p):
        if p.startswith("/mnt/c/"):
            return "C:/" + p[7:]
        if os.path.isabs(p):
            return p
        return os.path.join(BASE_DIR, p)
        
    abs_source = fix_path(job['source_path'])
    abs_ref = fix_path(job['ref_path'])
    abs_out = fix_path(final_out)
    os.makedirs(os.path.dirname(abs_out), exist_ok=True)
    
    cmd = [
        FD_VENV, "test_video_swap_multi.py",
        "--facedancer_path", MODEL_PATH,
        "--vid_path", abs_source,
        "--swap_source", abs_ref,
        "--vid_output", abs_out
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = FD_DIR
    
    res = subprocess.run(cmd, cwd=FD_DIR, capture_output=True, env=env, text=True)
    
    if not os.path.exists(final_out):
        log(f"Inference FAILED for {job['chunk_id']}")
        log(f"Stdout: {res.stdout}")
        log(f"Stderr: {res.stderr}")
        success = False
        
    duration = (time.time() - start_t) / 60
    df.at[idx, 'status'] = 'COMPLETED' if success else 'FAILED'
    df.to_csv(MANIFEST, index=False)
    
    if success:
        log(f"SUCCESS: {job['chunk_id']} ({duration:.1f} mins)")
        preview_dir = "videos/06_generation/bin_CD_swap/preview_samples/facedancer/"
        os.makedirs(preview_dir, exist_ok=True)
        if len(df[df.get('status') == 'COMPLETED']) <= 3:
            shutil.copy2(job['output_path'], preview_dir)

log("Phase 4 Finished.")
