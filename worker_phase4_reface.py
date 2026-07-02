import os, sys, subprocess, uuid, csv, time, json, shutil, glob
import pandas as pd
from datetime import datetime
from pathlib import Path

BASE_DIR = os.path.abspath(".")
RF_VENV = os.path.join(BASE_DIR, "reface_venv/bin/python")
RF_DIR = os.path.join(BASE_DIR, "REFace")
MANIFEST = os.path.join(BASE_DIR, "videos/07_generation_logs/reface_test_manifest.csv")
LOG_FILE = os.path.join(BASE_DIR, "videos/07_generation_logs/reface_test_log.txt")
WORKSPACE = os.path.join(BASE_DIR, "01_processing_workspace")

# REFace required paths
CONFIG = os.path.join(RF_DIR, "models/REFace/configs/project_ffhq.yaml")
CKPT = os.path.join(BASE_DIR, "weights/last.ckpt")

def log(msg):
    with open(LOG_FILE, "a") as f: f.write(f"[{datetime.now()}] {msg}\n")
    print(f"[{datetime.now()}] {msg}")

df = pd.read_csv(MANIFEST)
log(f"Phase 4 (REFace) Started: {len(df)} jobs.")

for idx, job in df.iterrows():
    if job['status'] == 'COMPLETED': continue
    
    log(f"STARTING REFace: {job['chunk_id']}")
    start_t = time.time()
    
    df.at[idx, 'status'] = 'RUNNING'
    df.to_csv(MANIFEST, index=False)
    
    out_video_dir = os.path.join(WORKSPACE, f"{job['chunk_id']}_out")
    os.makedirs(out_video_dir, exist_ok=True)
    
    # We clear the hardcoded tmp_frames just in case a previous run crashed
    tmp_frames_dir = os.path.join(RF_DIR, "tmp_frames")
    shutil.rmtree(tmp_frames_dir, ignore_errors=True)
    
    cmd = [
        RF_VENV, "scripts/inference_swap_video.py",
        "--outdir", out_video_dir,
        "--target_video", os.path.abspath(job['source_path']),
        "--config", CONFIG,
        "--ckpt", CKPT,
        "--src_image", os.path.abspath(job['ref_path']),
        "--Base_dir", out_video_dir,
        "--scale", "3",
        "--ddim_steps", str(job['ddim_steps'])
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = RF_DIR
    
    # Run inference_swap_video
    res = subprocess.run(cmd, cwd=RF_DIR, capture_output=True, env=env, text=True)
    
    # Verify output
    success = False
    generated_videos = glob.glob(os.path.join(out_video_dir, "*.mp4"))
    
    if len(generated_videos) > 0:
        success = True
        # Move the generated video to the final output path
        final_out = job['output_path']
        os.makedirs(os.path.dirname(final_out), exist_ok=True)
        shutil.move(generated_videos[0], final_out)
    else:
        log(f"Inference FAILED for chunk {job['chunk_id']}.")
        log(f"Stderr: {res.stderr}")
        
    # Final updates
    duration = (time.time() - start_t) / 60
    df.at[idx, 'status'] = 'COMPLETED' if success else 'FAILED'
    df.to_csv(MANIFEST, index=False)
    
    if success:
        log(f"SUCCESS: {job['chunk_id']} ({duration:.1f} mins)")
        # Copy to preview folder for easy viewing
        preview_dir = "videos/06_generation/bin_CD_swap/preview_samples/reface/"
        os.makedirs(preview_dir, exist_ok=True)
        if len(df[df['status'] == 'COMPLETED']) <= 3:
            shutil.copy2(job['output_path'], preview_dir)
    
    # Cleanup
    shutil.rmtree(out_video_dir, ignore_errors=True)
    shutil.rmtree(tmp_frames_dir, ignore_errors=True)

log("Phase 4 Finished.")
