import os, sys, subprocess, uuid, csv, time, json, glob, shutil, cv2
import pandas as pd
from datetime import datetime
from pathlib import Path

BASE_DIR = os.path.abspath(".")
LP_VENV = os.path.join(BASE_DIR, "liveportrait_venv/bin/python")
FOMM_VENV = os.path.join(BASE_DIR, "fomm_venv/bin/python")
LP_DIR = os.path.join(BASE_DIR, "LivePortrait")
FOMM_DIR = os.path.join(BASE_DIR, "first-order-model")
MANIFEST = os.path.join(BASE_DIR, "videos/07_generation_logs/bin_B_test_manifest.csv")
LOG_FILE = os.path.join(BASE_DIR, "videos/07_generation_logs/master_binB.log")
CKPT_FOMM = os.path.join(FOMM_DIR, "vox-cpk.pth.tar")

def log(msg):
    with open(LOG_FILE, "a") as f: f.write(f"[{datetime.now()}] {msg}\n")
    print(f"[{datetime.now()}] {msg}")

log("Master Worker Bin B Debug FOMM Mode Started")

while True:
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
            tmp_out = os.path.join(BASE_DIR, f"latentsync_temp/lp_{job['chunk_id']}")
            os.makedirs(tmp_out, exist_ok=True)
            cmd = [LP_VENV, "inference.py", "-s", job['source_path'], "-d", job['driver_path'], "-o", tmp_out, "--flag_relative_motion", "--flag_pasteback", "--flag_do_crop"]
            env = os.environ.copy(); env["PYTHONPATH"] = LP_DIR
            subprocess.run(cmd, cwd=LP_DIR, capture_output=True, env=env)
            found = glob.glob(os.path.join(tmp_out, "**", "*.mp4"), recursive=True)
            non_concat = [f for f in found if "_concat.mp4" not in f]
            if non_concat:
                shutil.move(non_concat[0], job['output_path'])
                success = True
            shutil.rmtree(tmp_out, ignore_errors=True)
        else:
            img = os.path.join(BASE_DIR, f"latentsync_temp/{job['chunk_id']}_f30.jpg")
            subprocess.run(["ffmpeg", "-y", "-i", job['source_path'], "-vf", "select=eq(n\\,30)", "-vframes", "1", img], capture_output=True)
            cmd = [FOMM_VENV, "demo.py", "--config", "config/vox-256.yaml", "--checkpoint", CKPT_FOMM, "--source_image", img, "--driving_video", job['driver_path'], "--result_video", job['output_path'], "--relative", "--adapt_scale"]
            env = os.environ.copy(); env["PYTHONPATH"] = FOMM_DIR
            
            res = subprocess.run(cmd, cwd=FOMM_DIR, capture_output=True, text=True, env=env)
            if not os.path.exists(job['output_path']):
                log(f"FOMM GPU Error: {res.stderr[-500:]}")
                log("Trying CPU...")
                res_cpu = subprocess.run(cmd + ["--cpu"], cwd=FOMM_DIR, capture_output=True, text=True, env=env)
                if not os.path.exists(job['output_path']):
                    log(f"FOMM CPU Error: {res_cpu.stderr[-500:]}")
            
            if os.path.exists(job['output_path']): success = True
            if os.path.exists(img): os.remove(img)

        df = pd.read_csv(MANIFEST)
        df.at[idx, 'status'] = 'COMPLETED' if success else 'FAILED'
        df.at[idx, 'duration'] = time.time() - start_t
        df.to_csv(MANIFEST, index=False)
        if success:
            log(f"SUCCESS: {job['chunk_id']}")
    except Exception as e:
        log(f"ERROR: {str(e)}")
        df.at[idx, 'status'] = 'FAILED'; df.to_csv(MANIFEST, index=False)
    time.sleep(1)
