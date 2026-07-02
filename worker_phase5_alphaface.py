import os, sys, subprocess, time
sys.path.insert(0, '/mnt/c/Users/User/Work/deep-fake-detection/download-pipeline/yt-shorts-pipeline/liveportrait_venv/lib/python3.10/site-packages')
import csv
from datetime import datetime

BASE_DIR = os.path.abspath(".")
ALPHAFACE_VENV = os.path.join(BASE_DIR, "alphaface_venv/bin/python")
ALPHAFACE_DIR = os.path.join(BASE_DIR, "AlphaFace")
MANIFEST = os.path.join(BASE_DIR, "videos/07_generation_logs/bin_CD_phase5_manifest.csv")
LOG_FILE = os.path.join(BASE_DIR, "videos/07_generation_logs/phase5_alphaface.log")
ID_POOL = os.path.join(BASE_DIR, "videos/08_identity_pool")
OUT_DIR = os.path.join(BASE_DIR, "videos/06_generation/bin_CD_swap/review")

os.makedirs(OUT_DIR, exist_ok=True)

def log(msg):
    with open(LOG_FILE, "a") as f: f.write(f"[{datetime.now()}] {msg}\n")
    print(f"[{datetime.now()}] {msg}")

jobs = []
with open(MANIFEST, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['model'] == 'alphaface':
            jobs.append(row)

log(f"Phase 5 (AlphaFace) Started: {len(jobs)} jobs.")

for job in jobs:
    target_path = os.path.join(BASE_DIR, "videos/02_passed/chunks_ready/bin_D_diffusion_swap", f"{job['chunk_id']}.mp4")
    source_path = os.path.join(ID_POOL, f"{job['ident']}.png")
    
    out_name = f"{job['chunk_id']}_fake_swap_alphaface.mp4"
    out_path = os.path.join(OUT_DIR, out_name)
    
    log(f"STARTING {job['chunk_id']}")
    start_t = time.time()
    
    # AlphaFace inference command
    cmd = [
        ALPHAFACE_VENV, "inference.py",
        "--source", source_path,
        "--target", target_path,
        "--output", out_path
    ]
    
    env = os.environ.copy()
    env["PYTHONPATH"] = ALPHAFACE_DIR
    
    res = subprocess.run(cmd, cwd=ALPHAFACE_DIR, capture_output=True, text=True, env=env)
    
    duration = time.time() - start_t
    success = os.path.exists(out_path)
    
    if success:
        log(f"SUCCESS: {job['chunk_id']}")
    else:
        log(f"FAILED: {job['chunk_id']}. Code: {res.returncode}")
        log(f"DEBUG Stderr: {res.stderr[-500:]}")
    
    with open("videos/07_generation_logs/phase5_results.csv", "a") as f:
        f.write(f"{job['chunk_id']},alphaface,{'COMPLETED' if success else 'FAILED'},{duration}\n")

log("Phase 5 (AlphaFace) Finished.")
