import os, sys, subprocess, shutil, time, glob, re
from datetime import datetime

BASE_DIR = os.path.abspath(".")
VF_VENV = os.path.join(BASE_DIR, "vividface_venv/bin/python")
VF_DIR = os.path.join(BASE_DIR, "VividFace")
LOG_FILE = os.path.join(BASE_DIR, "videos/07_generation_logs/vividface_preview.log")
ID_POOL = os.path.join(BASE_DIR, "videos/08_identity_pool")
IN_DIR = os.path.join(BASE_DIR, "videos/02_passed/chunks_ready/bin_C_gan_swap")
OUT_DIR = os.path.join(BASE_DIR, "videos/06_generation/bin_CD_swap/preview_samples/vividface")
os.makedirs(OUT_DIR, exist_ok=True)

def log(msg):
    with open(LOG_FILE, "a") as f: f.write(f"[{datetime.now()}] {msg}\n")
    print(f"[{datetime.now()}] {msg}")

log("Starting VividFace Preview Batch")

# Get 10 chunks
chunks = [f for f in os.listdir(IN_DIR) if f.endswith('.mp4')][:10]
identities = [f for f in os.listdir(ID_POOL) if f.endswith('.png')]

if not chunks or not identities:
    log("No chunks or identities found!")
    sys.exit(1)

for idx, chunk in enumerate(chunks):
    chunk_id = chunk.split('.')[0]
    target_path = os.path.join(IN_DIR, chunk)
    source_path = os.path.join(ID_POOL, identities[idx % len(identities)])
    out_path = os.path.join(OUT_DIR, f"{chunk_id}_fake_swap_vividface.mp4")
    
    log(f"Processing {chunk_id}")
    
    # Setup temp dir
    temp_dir = os.path.join(BASE_DIR, f"temp_vividface_{chunk_id}")
    os.makedirs(os.path.join(temp_dir, "videos"), exist_ok=True)
    os.makedirs(os.path.join(temp_dir, "faces"), exist_ok=True)
    
    temp_video = os.path.join(temp_dir, "videos", "1.mp4")
    temp_txt = os.path.join(temp_dir, "videos", "1.txt")
    temp_face = os.path.join(temp_dir, "faces", "1.png")
    
    shutil.copy(target_path, temp_video)
    shutil.copy(source_path, temp_face)
    
    # Generate keypoints
    log("Generating keypoints...")
    wsl_tgt = temp_video.replace("\\", "/").replace("C:", "/mnt/c").replace("c:", "/mnt/c")
    wsl_txt = temp_txt.replace("\\", "/").replace("C:", "/mnt/c").replace("c:", "/mnt/c")
    gen_cmd = [
        "wsl", "--", "bash", "-c",
        f"cd /mnt/c/Users/User/Work/deep-fake-detection/download-pipeline/yt-shorts-pipeline && "
        f"source vividface_venv/bin/activate && "
        f"python generate_vividface_txt.py '{wsl_tgt}' '{wsl_txt}'"
    ]
    gen_res = subprocess.run(gen_cmd, capture_output=True, text=True)
    
    # Run VividFace (via WSL since venv is WSL-based)
    log("Running VividFace inference...")
    wsl_temp_dir = temp_dir.replace("\\", "/").replace("C:", "/mnt/c").replace("c:", "/mnt/c")
    cmd = [
        "wsl", "--", "bash", "-c",
        f"cd /mnt/c/Users/User/Work/deep-fake-detection/download-pipeline/yt-shorts-pipeline/VividFace && "
        f"source ../vividface_venv/bin/activate && "
        f"export CUDA_HOME=/usr/local/cuda-12.8 && "
        f'export PATH="/mnt/c/Users/User/Work/deep-fake-detection/download-pipeline/yt-shorts-pipeline/gcc13_bin:$PATH" && '
        f"export CC=gcc-13 CXX=g++-13 && "
        f"python infer.py {wsl_temp_dir}"
    ]
    
    start_t = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
    
    # Find saved output
    match = re.search(r"Save in\s*:\s*(.*\.mp4)", res.stdout)
    if match:
        saved_file = match.group(1).strip()
        # Path from stdout is relative to VividFace dir or absolute WSL path
        # Actually it says "Save in : outputs/2026_06_26.../0_0.mp4"
        if not saved_file.startswith("/mnt/c"):
            saved_file = os.path.join(VF_DIR, saved_file)
        else:
            saved_file = saved_file.replace("/mnt/c", "C:").replace("/", "\\")
            
        if os.path.exists(saved_file):
            shutil.move(saved_file, out_path)
            log(f"SUCCESS: {chunk_id} in {time.time()-start_t:.1f}s")
        else:
            log(f"FAILED: Could not find output file {saved_file}")
            log(f"STDOUT: {res.stdout[-500:]}\nSTDERR: {res.stderr[-500:]}")
    else:
        log(f"FAILED: Could not parse output path")
        log(f"STDOUT: {res.stdout[-1000:]}\nSTDERR: {res.stderr[-1000:]}")
        
    shutil.rmtree(temp_dir, ignore_errors=True)

log("VividFace Preview Batch Finished")
