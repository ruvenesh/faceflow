import os, sys, subprocess, uuid, csv, time, json
import pandas as pd
from datetime import datetime
from pathlib import Path

BASE_DIR = os.path.abspath(".")
FF_VENV = os.path.join(BASE_DIR, "facefusion_venv/bin/python")
FF_DIR = os.path.join(BASE_DIR, "facefusion")
MANIFEST = os.path.join(BASE_DIR, "videos/07_generation_logs/inswapper_bulk_manifest.csv")
LOG_FILE = os.path.join(BASE_DIR, "videos/07_generation_logs/inswapper_bulk.log")
ID_POOL = os.path.join(BASE_DIR, "videos/08_identity_pool")

def log(msg):
    with open(LOG_FILE, "a") as f: f.write(f"[{datetime.now()}] {msg}\n")
    print(f"[{datetime.now()}] {msg}")

df = pd.read_csv(MANIFEST)
jobs = df[df['model_name'] == 'inswapper']

log(f"Phase 1 (InSwapper) Started: {len(jobs)} jobs.")

for i, (_, job) in enumerate(jobs.iterrows()):
    target_path = os.path.join(BASE_DIR, "videos/02_passed/chunks_ready/bin_C_gan_swap", f"{job['chunk_id']}.mp4")
    source_path = os.path.join(ID_POOL, f"{job['ident']}.png")
    
    config_name = "raw"
    processors = ["face_swapper"]
    blend = 100
    
    if i >= 3:
        processors.append("face_enhancer")
        config_name = "mild_codeformer" if i < 6 else "full_codeformer"
        blend = 50 if i < 6 else 100
    
    out_name = f"{job['chunk_id']}_fake_swap_inswapper_{config_name}.mp4"
    out_path = os.path.join(BASE_DIR, "videos/06_generation/bin_CD_swap/review", out_name)
    
    log(f"STARTING {job['chunk_id']} -> {config_name}")
    start_t = time.time()
    
    # Convert windows paths to wsl paths for safety
    def wsl_path(p):
        return p.replace("\\", "/").replace("C:", "/mnt/c").replace("c:", "/mnt/c")
        
    bash_script = f'cd /mnt/c/Users/User/Work/deep-fake-detection/download-pipeline/yt-shorts-pipeline/facefusion && ../facefusion_venv/bin/python facefusion.py headless-run --source-paths "{wsl_path(source_path)}" --target-path "{wsl_path(target_path)}" --output-path "{wsl_path(out_path)}" --processors {" ".join(processors)} --face-swapper-model inswapper_128_fp16 --face-swapper-pixel-boost --face-swapper-color-blend {blend}'
    
    if "face_enhancer" in processors:
        bash_script += f' --face-enhancer-model codeformer --face-enhancer-blend {blend}'
        
    bash_script += ' --execution-providers cuda'
    
    cmd = ["wsl", "--", "bash", "-c", bash_script]
    
    # Run
    res = subprocess.run(cmd, cwd=BASE_DIR, capture_output=True, text=True)
    
    duration = time.time() - start_t
    success = os.path.exists(out_path)
    
    if success:
        log(f"SUCCESS: {job['chunk_id']}")
    else:
        log(f"FAILED: {job['chunk_id']}. Code: {res.returncode}")
        # Log stderr for debugging the first few
        if i < 2: log(f"DEBUG Stderr: {res.stderr[-500:]}")
    
    with open("videos/07_generation_logs/phase1_results.csv", "a") as f:
        f.write(f"{job['chunk_id']},inswapper_{config_name},{'COMPLETED' if success else 'FAILED'},{duration}\n")

log("Phase 1 Finished.")
