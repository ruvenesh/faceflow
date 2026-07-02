"""
VividFace Phase 4 Worker
Runs VividFace on video chunks via WSL.
"""
import os, sys, subprocess, time, shutil, glob
import pandas as pd
from datetime import datetime

BASE_DIR = os.path.abspath(".")
WSL_VENV_PYTHON = "vividface_venv/bin/python"
VIVIDFACE_DIR = "VividFace"
MANIFEST = os.path.join(BASE_DIR, "videos/07_generation_logs/vividface_bulk_manifest.csv")
LOG_FILE = os.path.join(BASE_DIR, "videos/07_generation_logs/vividface_bulk.log")
TEMP_DIR = os.path.join(BASE_DIR, "videos/07_generation_logs/vividface_temp")

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
    src_manifest = os.path.join(BASE_DIR, "videos/07_generation_logs/diffswap_test_manifest.csv")
    if not os.path.exists(src_manifest):
        log(f"Source manifest not found: {src_manifest}")
        sys.exit(1)
    df = pd.read_csv(src_manifest)
    df['status'] = 'PENDING'
    df['output_path'] = df['output_path'].str.replace('diffswap', 'vividface')
    df.to_csv(MANIFEST, index=False)

df = pd.read_csv(MANIFEST)
log(f"Phase 4 (VividFace) Started: {len(df)} jobs.")

for idx, job in df.iterrows():
    if job.get('status') == 'COMPLETED': continue

    log(f"STARTING VividFace: {job['chunk_id']}")
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

    # VividFace expects a folder with `videos/` and `faces/`
    chunk_temp = os.path.join(TEMP_DIR, job['chunk_id'])
    
    # We will copy the source video to temp/videos/video.mp4
    # and the source face to temp/faces/video.png
    os.makedirs(os.path.join(chunk_temp, "videos"), exist_ok=True)
    os.makedirs(os.path.join(chunk_temp, "faces"), exist_ok=True)
    
    # Needs to match names: short_video_path and short_face_path
    shutil.copy2(abs_source, os.path.join(chunk_temp, "videos", "target.mp4"))
    shutil.copy2(abs_ref, os.path.join(chunk_temp, "faces", "target.png"))
    
    # 1. Generate keypoints using insightface script
    wsl_python = win_to_wsl(os.path.join(BASE_DIR, WSL_VENV_PYTHON))
    wsl_cwd = win_to_wsl(os.path.join(BASE_DIR, VIVIDFACE_DIR))
    wsl_chunk_temp = win_to_wsl(chunk_temp)
    
    abs_target = os.path.join(chunk_temp, "videos", "target.mp4")
    abs_source = os.path.join(chunk_temp, "videos", "source.mp4")
    
    # Resize both videos to 512x512 as VividFace hardcodes 512x512 in infer.py
    for vid in [abs_target, abs_source]:
        if os.path.exists(vid):
            tmp_vid = vid.replace(".mp4", "_tmp.mp4")
            os.rename(vid, tmp_vid)
            subprocess.run([
                "ffmpeg", "-y", "-i", tmp_vid, 
                "-vf", "scale=512:512", 
                "-c:v", "libx264", "-crf", "18", 
                vid
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            os.remove(tmp_vid)

    log(f"  Generating keypoints...")
    cmd_kps = [
        "wsl", "--", "bash", "-c",
        f"cd {wsl_cwd} && PYTHONPATH={wsl_cwd}:$PYTHONPATH {wsl_python} generate_kps.py {wsl_chunk_temp}"
    ]
    res = subprocess.run(cmd_kps, capture_output=True, text=True, encoding='utf-8', errors='replace')
    if res.returncode != 0:
        log(f"  Keypoints generation FAILED: {res.stderr}")
        success = False
    else:
        # 2. Run VividFace inference
        log(f"  Running VividFace inference...")
        cmd_infer = [
            "wsl", "--", "bash", "-c",
            f"cd {wsl_cwd} && PYTHONPATH={wsl_cwd}:$PYTHONPATH {wsl_python} infer.py {wsl_chunk_temp}"
        ]
        res_infer = subprocess.run(cmd_infer, capture_output=True, text=True, encoding='utf-8', errors='replace')
        if res_infer.returncode != 0:
            log(f"  Inference FAILED: {res_infer.stderr}")
            with open(os.path.join(BASE_DIR, "videos", "07_generation_logs", "vividface_error.log"), "a", encoding='utf-8') as f:
                f.write(f"\n--- ERROR ---\nSTDOUT:\n{res_infer.stdout}\nSTDERR:\n{res_infer.stderr}\n")
            success = False
        else:
            # VividFace saves the video in outputs/datetime_checkpoints/videos/target_target.mp4
            # We need to find the latest output directory in VividFace/outputs
            try:
                outputs_dir = os.path.join(BASE_DIR, VIVIDFACE_DIR, "outputs")
                subdirs = sorted([os.path.join(outputs_dir, d) for d in os.listdir(outputs_dir) if os.path.isdir(os.path.join(outputs_dir, d))], key=os.path.getmtime)
                latest_output_dir = subdirs[-1]
                output_video = os.path.join(latest_output_dir, "videos", "target_target.mp4")
                
                abs_out = fix_path(job['output_path'])
                os.makedirs(os.path.dirname(abs_out), exist_ok=True)
                shutil.copy2(output_video, abs_out)
                
                # Combine audio using ffmpeg
                abs_out_with_audio = abs_out.replace(".mp4", "_audio.mp4")
                original_source = fix_path(job['source_path'])
                cmd_audio = [
                    "ffmpeg", "-y",
                    "-i", abs_out,
                    "-i", original_source,
                    "-c:v", "copy", "-c:a", "aac",
                    "-map", "0:v:0", "-map", "1:a:0?",
                    "-shortest",
                    abs_out_with_audio
                ]
                subprocess.run(cmd_audio, capture_output=True)
                if os.path.exists(abs_out_with_audio):
                    shutil.move(abs_out_with_audio, abs_out)
                
            except Exception as e:
                log(f"  Failed to copy output video: {e}")
                with open(os.path.join(BASE_DIR, "videos", "07_generation_logs", "vividface_error.log"), "a", encoding='utf-8') as f:
                    f.write(f"\n--- ERROR (Copy failed) ---\nSTDOUT:\n{res_infer.stdout}\nSTDERR:\n{res_infer.stderr}\n")
                success = False

    # Cleanup temp
    if os.path.exists(chunk_temp):
        shutil.rmtree(chunk_temp, ignore_errors=True)

    duration = (time.time() - start_t) / 60
    df.at[idx, 'status'] = 'COMPLETED' if success else 'FAILED'
    df.to_csv(MANIFEST, index=False)

    if success:
        log(f"SUCCESS: {job['chunk_id']} ({duration:.1f} mins)")
        preview_dir = os.path.join(BASE_DIR, "videos/06_generation/bin_CD_swap/preview_samples/vividface/")
        os.makedirs(preview_dir, exist_ok=True)
        completed_count = len(df[df['status'] == 'COMPLETED'])
        if completed_count <= 3:
            shutil.copy2(abs_out, preview_dir)
    else:
        log(f"FAILED: {job['chunk_id']} ({duration:.1f} mins)")

log("Phase 4 (VividFace) Finished.")
