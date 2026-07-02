import os, sys, subprocess, time, shutil
import pandas as pd
from datetime import datetime

BASE_DIR = os.path.abspath(".")
FS_VENV = os.path.join(BASE_DIR, "faceshifter_venv/Scripts/python.exe")
FS_DIR = os.path.join(BASE_DIR, "FaceShifter")
MANIFEST = os.path.join(BASE_DIR, "videos/07_generation_logs/faceshifter_test_manifest.csv")
LOG_FILE = os.path.join(BASE_DIR, "videos/07_generation_logs/faceshifter_test_log.txt")

def log(msg):
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, "a") as f: f.write(f"[{datetime.now()}] {msg}\n")
    print(f"[{datetime.now()}] {msg}")

# Initialize manifest by copying from facedancer manifest if it doesn't exist
if not os.path.exists(MANIFEST):
    src_manifest = os.path.join(BASE_DIR, "videos/07_generation_logs/facedancer_test_manifest.csv")
    if not os.path.exists(src_manifest):
        log(f"Source manifest not found: {src_manifest}")
        sys.exit(1)
    df = pd.read_csv(src_manifest)
    df['status'] = 'PENDING'
    df['output_path'] = df['output_path'].str.replace('facedancer', 'faceshifter')
    df.to_csv(MANIFEST, index=False)

df = pd.read_csv(MANIFEST)
log(f"Phase 4 (FaceShifter) Started: {len(df)} jobs.")

for idx, job in df.iterrows():
    if job.get('status') == 'COMPLETED': continue
    
    log(f"STARTING FaceShifter: {job['chunk_id']}")
    start_t = time.time()
    
    df.at[idx, 'status'] = 'RUNNING'
    df.to_csv(MANIFEST, index=False)
    
    success = True
    final_out = job['output_path']
    def fix_path(p):
        if str(p).startswith("/mnt/c/"):
            return "C:/" + p[7:]
        if os.path.isabs(p):
            return p
        return os.path.join(BASE_DIR, p)
        
    abs_source = fix_path(job['source_path'])
    abs_ref = fix_path(job['ref_path'])
    abs_out = fix_path(final_out)
    os.makedirs(os.path.dirname(abs_out), exist_ok=True)
    
    cmd = [
        FS_VENV, "inference_video.py",
        "--source", abs_ref,
        "--target", abs_source,
        "--output", abs_out
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = FS_DIR
    
    res = subprocess.run(cmd, cwd=FS_DIR, capture_output=True, env=env, text=True)
    
    if not os.path.exists(abs_out):
        log(f"Inference FAILED for {job['chunk_id']}")
        log(f"Stdout: {res.stdout}")
        log(f"Stderr: {res.stderr}")
        success = False
        
    duration = (time.time() - start_t) / 60
    df.at[idx, 'status'] = 'COMPLETED' if success else 'FAILED'
    df.to_csv(MANIFEST, index=False)
    
    if success:
        log(f"SUCCESS: {job['chunk_id']} ({duration:.1f} mins)")
        preview_dir = "videos/06_generation/bin_CD_swap/preview_samples/faceshifter/"
        os.makedirs(preview_dir, exist_ok=True)
        # Copy first few successes to preview
        if len(df[df.get('status') == 'COMPLETED']) <= 3:
            shutil.copy2(abs_out, preview_dir)

log("Phase 4 Finished.")
