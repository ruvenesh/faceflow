import os, sys, subprocess, uuid, csv, time, json
import pandas as pd
from datetime import datetime
from pathlib import Path

BASE_DIR = os.path.abspath(".")
DV_VENV = os.path.join(BASE_DIR, "dreamid_venv/bin/python")
DV_DIR = os.path.join(BASE_DIR, "DreamID-V")
MANIFEST = os.path.join(BASE_DIR, "videos/07_generation_logs/bin_CD_test_manifest.csv")
LOG_FILE = os.path.join(BASE_DIR, "videos/07_generation_logs/phase2_dreamid.log")
ID_POOL = os.path.join(BASE_DIR, "videos/08_identity_pool")

def log(msg):
    with open(LOG_FILE, "a") as f: f.write(f"[{datetime.now()}] {msg}\n")
    print(f"[{datetime.now()}] {msg}")

df = pd.read_csv(MANIFEST)
jobs = df[df['model'] == 'dreamid-v']

log(f"Phase 2 (DreamID-V) Started: {len(jobs)} jobs.")

for _, job in jobs.iterrows():
    target_path = os.path.join(BASE_DIR, "videos/02_passed/chunks_ready/bin_D_diffusion_swap", f"{job['chunk_id']}.mp4")
    source_path = os.path.join(ID_POOL, f"{job['ident']}.png")
    
    out_name = f"{job['chunk_id']}_fake_swap_dreamid.mp4"
    out_path = os.path.join(BASE_DIR, "videos/06_generation/bin_CD_swap/review", out_name)
    
    log(f"STARTING {job['chunk_id']}")
    start_t = time.time()
    
    cmd = [
        DV_VENV, "-u", "generate_dreamidv_faster.py",
        "--task", "swapface",
        "--ref_image", source_path,
        "--ref_video", target_path,
        "--save_file", out_path,
        "--ckpt_dir", os.path.join(DV_DIR, "checkpoints"),
        "--dreamidv_ckpt", os.path.join(DV_DIR, "checkpoints/dreamidv_faster.pth"),
        "--size", "832*480",
        "--frame_num", "61",
        "--offload_model", "True"
    ]
    
    env = os.environ.copy()
    env["PYTHONPATH"] = DV_DIR
    
    # STREAM OUTPUT
    process = subprocess.Popen(cmd, cwd=DV_DIR, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(f"[{job['chunk_id']}] {line.strip()}")
        sys.stdout.flush()
    process.wait()
    
    duration = time.time() - start_t
    success = os.path.exists(out_path)
    
    if success:
        log(f"SUCCESS: {job['chunk_id']}")
    else:
        log(f"FAILED: {job['chunk_id']}. Exit Code: {process.returncode}")
    
    with open("videos/07_generation_logs/phase2_results.csv", "a") as f:
        f.write(f"{job['chunk_id']},dreamid-v,{'COMPLETED' if success else 'FAILED'},{duration}\n")

log("Phase 2 Finished.")
