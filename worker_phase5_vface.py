"""
VFace Phase 5 Worker
Runs VFace on video chunks via WSL.
"""
import os, sys, subprocess, time, shutil, glob
import pandas as pd
from datetime import datetime

BASE_DIR = os.path.abspath(".")
WSL_VENV_PYTHON = "vface_venv/bin/python"
VFACE_DIR = "VFace/REFace"
MAX_WORKERS = 1
MANIFEST = os.path.join(BASE_DIR, "videos/07_generation_logs/vface_test_manifest.csv")
LOG_FILE = os.path.join(BASE_DIR, "videos/07_generation_logs/vface_test_log.txt")
TEMP_DIR = os.path.join(BASE_DIR, "videos/07_generation_logs/vface_temp")

def log(msg):
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f: f.write(f"[{datetime.now()}] {msg}\n")
    print(f"[{datetime.now()}] {msg}")

def win_to_wsl(path):
    """Convert Windows path to WSL path."""
    path = os.path.abspath(path).replace("\\", "/")
    if path[1] == ":":
        drive = path[0].lower()
        return f"/mnt/{drive}{path[2:]}"
    return path

# Initialize manifest
if not os.path.exists(MANIFEST):
    src_manifest = os.path.join(BASE_DIR, "videos/07_generation_logs/vividface_test_manifest.csv")
    if not os.path.exists(src_manifest):
        log(f"Source manifest not found: {src_manifest}")
        sys.exit(1)
    df = pd.read_csv(src_manifest)
    df['status'] = 'PENDING'
    df['output_path'] = df['output_path'].str.replace('vividface', 'vface')
    df.to_csv(MANIFEST, index=False)

df = pd.read_csv(MANIFEST)
log(f"Phase 5 (VFace) Started: {len(df)} jobs.")

for idx, job in df.iterrows():
    if job.get('status') == 'COMPLETED': continue

    log(f"STARTING VFace: {job['chunk_id']}")
    start_t = time.time()

    df.at[idx, 'status'] = 'RUNNING'
    df.to_csv(MANIFEST, index=False)

    success = True

    def fix_path(p):
        if str(p).startswith("/mnt/c/"):
            return "C:/" + p[7:]
        if os.path.isabs(p):
            return p
        return os.path.join(BASE_DIR, p)

    abs_source = fix_path(job['source_path'])  # video
    abs_ref = fix_path(job['ref_path'])         # face

    chunk_temp = os.path.join(TEMP_DIR, job['chunk_id'])
    os.makedirs(chunk_temp, exist_ok=True)
    
    # We will copy the source video to temp/videos/video.mp4
    # and the source face to temp/faces/video.jpeg
    wsl_python = win_to_wsl(os.path.join(BASE_DIR, WSL_VENV_PYTHON))
    wsl_cwd = win_to_wsl(os.path.join(BASE_DIR, VFACE_DIR))
    wsl_abs_source = win_to_wsl(abs_source)
    wsl_abs_ref = win_to_wsl(abs_ref)
    wsl_chunk_temp = win_to_wsl(chunk_temp)
    
    log(f"  Running VFace inference...")
    # python scripts/VFace_inference_single.py --target_video <mp4> --src_image <jpeg> --outdir <dir> --Base_dir <dir>
    cmd_infer = [
        "wsl", "--", "bash", "-c",
        f"cd {wsl_cwd} && PYTHONPATH={wsl_cwd}:$PYTHONPATH {wsl_python} scripts/VFace_inference_single.py "
        f"--target_video {wsl_abs_source} --src_image {wsl_abs_ref} --outdir {wsl_chunk_temp}/outputs "
        f"--Base_dir {wsl_chunk_temp}/base_dir --n_frames 3 --ddim_steps 15"
    ]
    res_infer = subprocess.run(cmd_infer, capture_output=True, text=True, encoding='utf-8', errors='replace')
    if res_infer.returncode != 0:
        log(f"  Inference FAILED: {res_infer.stderr}")
        with open(os.path.join(BASE_DIR, "videos", "07_generation_logs", "vface_error.log"), "a", encoding='utf-8') as f:
            f.write(f"\n--- ERROR ---\nSTDOUT:\n{res_infer.stdout}\nSTDERR:\n{res_infer.stderr}\n")
        success = False
    else:
        # VFace outputs to outdir/target_videoto<src_image>_swap.mp4
        try:
            target_name = os.path.splitext(os.path.basename(abs_source))[0]
            src_name = os.path.splitext(os.path.basename(abs_ref))[0]
            output_video = os.path.join(chunk_temp, "outputs", f"{target_name}to{src_name}_swap.mp4")
            
            abs_out = fix_path(job['output_path'])
            os.makedirs(os.path.dirname(abs_out), exist_ok=True)
            shutil.copy2(output_video, abs_out)
            
            # Combine audio using ffmpeg
            abs_out_with_audio = abs_out.replace(".mp4", "_audio.mp4")
            cmd_audio = [
                "ffmpeg", "-y",
                "-i", abs_out,
                "-i", abs_source,
                "-c:v", "copy", "-c:a", "aac",
                "-map", "0:v:0", "-map", "1:a:0?",
                "-shortest",
                abs_out_with_audio
            ]
            subprocess.run(cmd_audio, capture_output=True)
            shutil.move(abs_out_with_audio, abs_out)
            
        except Exception as e:
            log(f"  Failed to copy output video: {e}")
            success = False

    # Cleanup temp
    if os.path.exists(chunk_temp):
        shutil.rmtree(chunk_temp, ignore_errors=True)

    duration = (time.time() - start_t) / 60
    df.at[idx, 'status'] = 'COMPLETED' if success else 'FAILED'
    df.to_csv(MANIFEST, index=False)

    if success:
        log(f"SUCCESS: {job['chunk_id']} ({duration:.1f} mins)")
        preview_dir = os.path.join(BASE_DIR, "videos/06_generation/bin_CD_swap/preview_samples/vface/")
        os.makedirs(preview_dir, exist_ok=True)
        completed_count = len(df[df['status'] == 'COMPLETED'])
        if completed_count <= 3:
            shutil.copy2(abs_out, preview_dir)
    else:
        log(f"FAILED: {job['chunk_id']} ({duration:.1f} mins)")

log("Phase 5 (VFace) Finished.")
