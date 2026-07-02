import os, sys, subprocess, time, shutil
from datetime import datetime

BASE_DIR = os.path.abspath(".")
VFACE_VENV = os.path.join(BASE_DIR, "vface_venv/bin/python")
VFACE_DIR = os.path.join(BASE_DIR, "VFace")
LOG_FILE = os.path.join(BASE_DIR, "videos/07_generation_logs/vface_preview.log")
ID_POOL = os.path.join(BASE_DIR, "videos/08_identity_pool")
IN_DIR = os.path.join(BASE_DIR, "videos/02_passed/chunks_ready/bin_C_gan_swap")
OUT_DIR = os.path.join(BASE_DIR, "videos/06_generation/bin_CD_swap/preview_samples/vface")

os.makedirs(OUT_DIR, exist_ok=True)

def log(msg):
    with open(LOG_FILE, "a") as f: f.write(f"[{datetime.now()}] {msg}\n")
    print(f"[{datetime.now()}] {msg}")

log("Starting VFace Preview Batch")

# Get next 10 chunks (offset by 10 to not overlap with VividFace)
chunks = [f for f in os.listdir(IN_DIR) if f.endswith('.mp4')][10:20]
identities = [f for f in os.listdir(ID_POOL) if f.endswith('.png')]

if not chunks or not identities:
    log("No chunks or identities found!")
    sys.exit(1)

for idx, chunk in enumerate(chunks):
    chunk_id = chunk.split('.')[0]
    target_path = os.path.join(IN_DIR, chunk)
    source_path = os.path.join(ID_POOL, identities[idx % len(identities)])
    out_path = os.path.join(OUT_DIR, f"{chunk_id}_fake_swap_vface.mp4")
    
    log(f"Processing {chunk_id}")
    
    temp_out_dir = os.path.join(BASE_DIR, f"temp_vface_{chunk_id}")
    os.makedirs(temp_out_dir, exist_ok=True)
    
    wsl_src = source_path.replace("\\", "/").replace("C:", "/mnt/c")
    wsl_tgt = target_path.replace("\\", "/").replace("C:", "/mnt/c")
    wsl_out = temp_out_dir.replace("\\", "/").replace("C:", "/mnt/c")
    
    cmd = [
        "wsl", "--", "bash", "-c",
        f"cd /mnt/c/Users/User/Work/deep-fake-detection/download-pipeline/yt-shorts-pipeline/VFace/REFace && "
        f"source ../../vface_venv/bin/activate && "
        f'export PYTHONPATH="/mnt/c/Users/User/Work/deep-fake-detection/download-pipeline/yt-shorts-pipeline/VFace/REFace:\\$PYTHONPATH" && '
        f"python scripts/VFace_inference_single.py "
        f"--src_image {wsl_src} "
        f"--target_video {wsl_tgt} "
        f"--outdir {wsl_out} "
        f"--Base_dir {wsl_out} "
        f"--config models/REFace/configs/project_ffhq.yaml "
        f"--ckpt models/REFace/checkpoints/last.ckpt"
    ]
    
    start_t = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
    
    # Check output
    # VFace saves to {outdir}/{target_video_name}to{src_name}_swap.mp4
    target_video_name = os.path.basename(target_path).split('.')[0]
    src_name = os.path.basename(source_path).split('.')[0]
    generated_file = os.path.join(temp_out_dir, f"{target_video_name}to{src_name}_swap.mp4")
    
    if os.path.exists(generated_file):
        shutil.move(generated_file, out_path)
        log(f"SUCCESS: {chunk_id} in {time.time()-start_t:.1f}s")
    else:
        log(f"FAILED: Could not find output file {generated_file}")
        log(f"STDOUT: {res.stdout[-1000:]}\nSTDERR: {res.stderr[-1000:]}")
        
    shutil.rmtree(temp_out_dir, ignore_errors=True)

log("VFace Preview Batch Finished")
